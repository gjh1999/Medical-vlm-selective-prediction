"""Recheck pilot scores from direct model calls, error ranks, and paired group resampling."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from extract_features import load_encoder, read_inputs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_auc_bootstrap(wrong, hierarchy, entropy):
    # All pairwise error/correct comparisons are fixed; bootstrap weights count repeated groups.
    def comparisons(values):
        d = values[wrong, None] - values[None, ~wrong]
        return (d > 0).astype(float) + .5 * (d == 0)
    delta = comparisons(hierarchy) - comparisons(entropy)
    rng = np.random.default_rng(17)
    values = []
    for _ in range(2000):
        cnt = np.bincount(rng.integers(0, len(wrong), len(wrong)), minlength=len(wrong))
        pos, neg = cnt[wrong], cnt[~wrong]
        if pos.sum() and neg.sum():
            values.append(float((pos @ delta @ neg) / (pos.sum() * neg.sum())))
    return {"paired_delta_parent_minus_entropy": float(delta.mean()),
            "percentiles_2_5_97_5": list(np.percentile(values, [2.5, 97.5])),
            "valid_resamples": len(values)}


def rank_auc(target, score):
    # Average ranks for ties, then Mann–Whitney U for errors as the positive class.
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    rank = (counts.cumsum() - (counts - 1) / 2)[inverse]
    n = target.sum()
    return float((rank[target].sum() - n * (n+1) / 2) / (n * (len(score) - n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ["audit", "pilot", "openai-weights", "biomed-dir", "output"]:
        ap.add_argument("--"+arg, type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/ham-dev-pilot-v1.json")
    args = ap.parse_args()
    t = time.perf_counter()
    audit, rows = read_inputs(args.audit)
    orig = json.loads((args.pilot / "summary.json").read_bytes())
    if (orig["status"] != "complete" or sha(args.config) != orig["config_sha256"]
            or audit["manifest_sha256"] != orig["manifest_sha256"]
            or sha(args.pilot / "selection.csv") != orig["selection_sha256"]):
        raise ValueError("Pilot provenance mismatch")
    with (args.pilot / "selection.csv").open(newline="", encoding="utf-8") as stream:
        sample = list(csv.DictReader(stream))
    by_id = {r["image_id"]: r for r in rows}
    groups = [r["split_group"] for r in sample]
    if len(set(groups)) != len(groups) or any(
            by_id[r["image_id"]]["split"] != "dev" or any(r[k] != by_id[r["image_id"]][k] for k in ("split_group","label"))
            for r in sample):
        raise ValueError("Development sample isolation mismatch")
    cfg = json.loads(args.config.read_bytes())
    train_counts = Counter(r["label"] for r in rows if r["representative"] == "1" and r["split"] == "train")
    majority = max(sorted(train_counts), key=train_counts.get)
    truth = np.asarray([r["label"] for r in sample])
    keep = np.array([r["sensitivity_retained"] == "1" for r in sample])
    report = {"source_config_sha256": orig["config_sha256"], "source_selection_sha256": orig["selection_sha256"],
              "majority_class_from_training": majority, "training_counts": dict(train_counts),
              "majority_correct": int((truth == majority).sum()), "n": len(truth), "models": {},
              "bootstrap": "2000 resamples of selected development groups, seed 17; descriptive only"}
    for name, result in orig["models"].items():
        path = args.pilot / f"{name}.npz"
        if sha(path) != result["cache_sha256"]:
            raise ValueError("Changed model output")
        with np.load(path, allow_pickle=False) as cache:
            weights = args.openai_weights if name == "clip" else args.biomed_dir / "open_clip_pytorch_model.bin"
            config = None if name == "clip" else args.biomed_dir / "open_clip_config.json"
            model, preprocess, tokenize = load_encoder(name, weights, "cpu", config, {}, with_tokenizer=True)
            nodes = result["text_node_order"]
            prompts = [t.format(label=cfg["names"][k]) for t in cfg["templates"] for k in nodes]
            ids = [0, len(sample) - 1]
            text = cache["text_features"].reshape(len(prompts), -1)
            expected = result["logit_scale"] * (cache["image_features"][ids] @ text.T)
            direct = []
            with torch.inference_mode():
                for index in ids:
                    row = by_id[sample[index]["image_id"]]
                    img = Path(row["image_path"])
                    if sha(img) != row["sha256"]:
                        raise ValueError("Changed reference image")
                    with Image.open(img) as im:
                        tensor = preprocess(ImageOps.exif_transpose(im).convert("RGB"))[None]
                    output = model(tensor, tokenize(prompts))
                    if name == "clip":
                        logits = output[0]
                    else:
                        image_f, text_f, scale = output
                        logits = scale * (image_f @ text_f.T)
                    direct.append(logits[0].numpy())
            difference = float(np.abs(np.asarray(direct) - expected).max())
            if difference > 1e-3:
                raise ValueError(f"Single-image reference logits differ for {name}: {difference}")
            leaves = result["leaf_order"]
            wrong = np.asarray(leaves)[cache["leaf_probs"].argmax(axis=1)] != truth
            checks = {"single_image_max_logit_abs_difference": difference, "reference_images": len(ids)}
            for key, mask in {"primary": np.ones(len(truth), dtype=bool), "sensitivity": keep}.items():
                summary = result[key]
                y = wrong[mask]
                if int((~y).sum()) != summary["correct"] or any(
                        abs(rank_auc(y, cache[k][mask]) - value) > 1e-12 for k, value in summary["error_auroc"].items()):
                    raise ValueError(f"Accuracy/error ranking mismatch: {name} {key}")
                checks[key] = paired_auc_bootstrap(y, cache["parent_js"][mask], cache["entropy"][mask])
            report["models"][name] = checks
    report.update(status="verified", seconds=round(time.perf_counter()-t, 3))
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
