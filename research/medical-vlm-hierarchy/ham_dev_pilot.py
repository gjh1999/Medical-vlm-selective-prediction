"""Paired zero-shot and error-ranking checks on a fixed HAM development sample."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from extract_features import encode_images, load_encoder, read_inputs
from prepare_biomedclip import REVISION
from selective_core import Taxonomy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_rows(records, cfg):
    if cfg["role"] != "dev":
        raise ValueError("This pilot is restricted to development groups")
    representatives = {}
    for row in sorted(records, key=lambda r: r["image_id"]):
        if row["split"] == "dev" and row["representative"] == "1":
            representatives.setdefault(row["split_group"], row)
    def selection_key(group):
        return hashlib.sha256(f'{cfg["selection_salt"]}|{group}'.encode()).digest()
    groups = sorted(representatives, key=selection_key)
    if not 0 < cfg["groups"] <= len(groups):
        raise ValueError("Insufficient distinct dev groups")
    return [representatives[g] for g in groups[:cfg["groups"]]]


def softmax(scores):
    values = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def js(left, right):
    middle = (left + right) / 2
    return ((left * (np.log(left) - np.log(middle))).sum(axis=-1)
            + (right * (np.log(right) - np.log(middle))).sum(axis=-1)) / 2


def error_auc(wrong, score):
    positives, negatives = score[wrong], score[~wrong]
    if min(len(positives), len(negatives)) == 0:
        return None
    differences = positives[:, None] - negatives
    return float(((differences > 0) + .5 * (differences == 0)).mean())


def evaluate(truth, leaf_probs, signals, tree, thresholds):
    leaves = tree.leaves
    top = np.asarray(leaves)[leaf_probs.argmax(axis=1)]
    wrong = top != truth
    classes = {}
    for leaf in leaves:
        actual, predicted = truth == leaf, top == leaf
        tp = int((actual & predicted).sum())
        den = int(actual.sum() + predicted.sum())
        classes[leaf] = {"support": int(actual.sum()), "predicted": int(predicted.sum()),
                         "correct": tp, "f1": 2 * tp / den if den else None}
    policies = {}
    for threshold in thresholds:
        choices = [tree.climbing(dict(zip(leaves, row)), threshold) for row in leaf_probs]
        policies[str(threshold)] = tree.metrics(list(truth), choices)
    return {"n": len(truth), "correct": int((~wrong).sum()), "accuracy": float((~wrong).mean()),
            "macro_f1_supported": float(np.mean([r["f1"] for r in classes.values() if r["support"]])),
            "classes": classes, "error_auroc": {k: error_auc(wrong, v) for k, v in signals.items()},
            "climbing": policies}


def run_model(name, args, cfg, rows, keep, tree, out, info):
    start = time.perf_counter()
    weights = args.openai_weights if name == "clip" else args.biomed_dir / "open_clip_pytorch_model.bin"
    path = None if name == "clip" else args.biomed_dir / "open_clip_config.json"
    model, preprocess, tokenize = load_encoder(name, weights, args.device, path, info, with_tokenizer=True)
    info["revision"] = "OpenAI-ViT-B-16" if name == "clip" else REVISION
    nodes = sorted(cfg["names"])
    prompts = [template.format(label=cfg["names"][node]) for template in cfg["templates"] for node in nodes]
    if len(cfg["templates"]) != 2:
        raise ValueError("Paired template divergence requires two templates")
    with torch.inference_mode():
        tokens = tokenize(prompts).to(args.device)
        vectors = model.encode_text(tokens).float()
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        scale = float(model.logit_scale.exp().item()) / cfg["temperature"]
    text = vectors.cpu().numpy().reshape(2, len(nodes), -1)
    if not np.allclose(np.linalg.norm(text, axis=-1), 1., atol=1e-5):
        raise ValueError("Invalid text prototypes")
    info["context_length"] = tokens.shape[1]
    info["logit_scale"] = scale
    t_img = time.perf_counter()
    image = np.concatenate([encode_images(model, preprocess, rows[i:i + args.batch_size], args.device)
                            for i in range(0, len(rows), args.batch_size)])
    info["image_seconds"] = round(time.perf_counter() - t_img, 3)
    # Average unit text embeddings, renormalize each prototype, then use the checkpoint scale.
    prototypes = text.mean(axis=0)
    prototypes /= np.linalg.norm(prototypes, axis=-1, keepdims=True)
    logits = (image @ prototypes.T).astype(np.float64) * scale
    leaf_idx = [nodes.index(k) for k in tree.leaves]
    leaf_probs = softmax(logits[:, leaf_idx])
    parents = sorted(k for k, parent in tree.parent.items() if parent == tree.root)
    direct = softmax(logits[:, [nodes.index(k) for k in parents]])
    pooled = np.column_stack([leaf_probs[:, [tree.leaves.index(k) for k in sorted(tree.desc[p])]].sum(axis=1)
                              for p in parents])
    individual = softmax((image @ text[:, leaf_idx, :].transpose(0, 2, 1)).astype(np.float64) * scale)
    ordered = np.sort(leaf_probs, axis=1)
    signals = {"negative_margin": ordered[:, -2] - ordered[:, -1],
               "entropy": -(leaf_probs * np.log(leaf_probs)).sum(axis=1),
               "prompt_js": js(individual[0], individual[1]), "parent_js": js(pooled, direct)}
    if set(signals) != set(cfg["error_ranking"]) or not np.isfinite(list(signals.values())).all():
        raise ValueError("Invalid ranking signal")
    truth = np.asarray([r["label"] for r in rows])
    for label, mask in {"primary": np.ones(len(rows), dtype=bool), "sensitivity": keep}.items():
        if not mask.any():
            raise ValueError("Empty evaluation subset")
        info[label] = evaluate(truth[mask], leaf_probs[mask], {k: s[mask] for k, s in signals.items()},
                               tree, cfg["climbing_thresholds"])
    cache = out / f"{name}.npz"
    np.savez(cache, image_features=image, text_features=text, leaf_probs=leaf_probs,
             direct_parent_probs=direct, pooled_parent_probs=pooled, **signals)
    info.update(cache_sha256=digest(cache), elapsed_seconds=round(time.perf_counter() - start, 3),
                leaf_order=tree.leaves, parent_order=parents, text_node_order=nodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("audit", "sensitivity", "openai-weights", "biomed-dir", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/ham-dev-pilot-v1.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    cfg = json.loads(args.config.read_bytes())
    audit, records = read_inputs(args.audit)
    sensitivity = json.loads((args.sensitivity / "summary.json").read_bytes())
    groups_csv = args.sensitivity / "sensitivity_groups.csv"
    if (cfg["split_sha256"] != audit["split_sha256"]
            or sensitivity["split_sha256"][str(cfg["split_seed"])] != audit["split_sha256"]
            or sensitivity["protocol"] != cfg["protocol"]
            or sensitivity["manifest_sha256"] != audit["manifest_sha256"]
            or digest(groups_csv) != sensitivity["sensitivity_groups_sha256"]):
        raise ValueError("Audit, split protocol and sensitivity membership must agree")
    rows = select_rows(records, cfg)
    with groups_csv.open(encoding="utf-8", newline="") as stream:
        membership = {r["split_group"]: r["excluded"] for r in csv.DictReader(stream)}
    keep = np.asarray([membership[r["split_group"]] == "0" for r in rows])
    tree = Taxonomy(cfg["parent"])
    args.output.mkdir(exist_ok=False, parents=True)
    selected = args.output / "selection.csv"
    with selected.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "split_group", "role", "label", "sensitivity_retained"])
        writer.writerows((r["image_id"], r["split_group"], r["split"], r["label"], int(k)) for r, k in zip(rows, keep))
    info = {"scope": "development pilot", "role": cfg["role"], "groups": len(rows), "seed": cfg["split_seed"],
            "selection_sha256": digest(selected), "config_sha256": digest(args.config),
            "manifest_sha256": audit["manifest_sha256"], "split_sha256": audit["split_sha256"],
            "sensitivity_groups_sha256": digest(groups_csv), "device": args.device, "batch_size": args.batch_size,
            "python": platform.python_version(), "versions": {p: importlib.metadata.version(p)
            for p in ["torch","torchvision","numpy","pillow","clip","open_clip_torch","timm","transformers"]},
            "models": {}}
    start = time.perf_counter()
    try:
        for name in ["clip", "biomedclip"]:
            info["models"][name] = {}
            run_model(name, args, cfg, rows, keep, tree, args.output, info["models"][name])
        info["status"] = "complete"
    except Exception as exc:
        info.update(status="failed", error=str(exc))
        raise
    finally:
        info["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
