"""Matched frozen-text, image-centroid and ridge baselines on disjoint balanced HAM dev groups."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from extract_features import encode_images, load_encoder, read_inputs
from ham_dev_pilot import digest


def select_balanced(records, cfg, leaves, previous):
    candidates = {"train": {}, "dev": {}}
    for r in sorted(records, key=lambda x: x["image_id"]):
        role = r["split"]
        if (role in candidates and r["representative"] == "1" and r["split_group"] not in previous):
            candidates[role].setdefault(r["split_group"], r)
    selected = {}
    for role, inventory in candidates.items():
        rows = []
        count = cfg[f"{role}_per_class"]
        for leaf in leaves:
            options = [r for r in inventory.values() if r["label"] == leaf]
            options.sort(key=lambda r: hashlib.sha256(
                f'{cfg["salt"]}|{role}|{r["split_group"]}'.encode()).digest())
            if len(options) < count:
                raise ValueError(f"Too few {role} groups for {leaf}")
            rows.extend(options[:count])
        selected[role] = rows
    if {r["split_group"] for r in selected["train"]} & {r["split_group"] for r in selected["dev"]}:
        raise ValueError("Train and dev groups overlap")
    return selected


def probe_scores(train_x, dev_x, y, classes, penalty):
    train_x, dev_x = train_x.astype(np.float64), dev_x.astype(np.float64)
    centers = np.stack([train_x[y == k].mean(0) for k in range(classes)])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    # Dual ridge with an unpenalized intercept; all means and weights use training rows only.
    x_mean = train_x.mean(0)
    xc = train_x - x_mean
    target = np.eye(classes)[y]
    y_mean = target.mean(0)
    dual = np.linalg.solve(xc @ xc.T + penalty * np.eye(len(xc)), target - y_mean)
    return {"cosine_nearest_centroid": dev_x @ centers.T,
            "centered_ridge": (dev_x-x_mean) @ xc.T @ dual + y_mean}


def compare(outputs, truth, labels, bootstrap):
    rng = np.random.default_rng(bootstrap["seed"])
    draws = np.column_stack([rng.choice(np.flatnonzero(truth == k),
                              size=(bootstrap["resamples"], int((truth == k).sum())), replace=True)
                             for k in range(len(labels))])
    baseline = outputs["frozen_text"].argmax(axis=1) == truth
    metrics = {}
    for name, scores in outputs.items():
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite probe score")
        pred = scores.argmax(axis=1)
        correct = pred == truth
        delta = correct.astype(int) - baseline.astype(int)
        confusion = np.zeros((len(labels), len(labels)), dtype=int)
        np.add.at(confusion, (truth, pred), 1)
        metrics[name] = {"n": len(truth), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
                         "corrected_vs_text": int((delta == 1).sum()), "lost_vs_text": int((delta == -1).sum()),
                         "paired_difference": float(delta.mean()),
                         "paired_percentiles_2_5_97_5": np.percentile(delta[draws].mean(axis=1), [2.5,97.5]).tolist(),
                         "confusion_true_by_predicted": confusion.tolist()}
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ["audit", "pilot", "openai-weights", "biomed-dir", "output"]:
        ap.add_argument("--"+arg, type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/ham-balanced-probe-v1.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cpu")
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("Positive batch size required")
    start = time.perf_counter()
    cfg = json.loads(args.config.read_bytes())
    pilot = json.loads((args.pilot/"summary.json").read_bytes())
    audit, records = read_inputs(args.audit)
    selection = args.pilot / "selection.csv"
    if (audit["split_sha256"] != cfg["split_sha256"] or audit["manifest_sha256"] != pilot["manifest_sha256"]
            or digest(selection) != cfg["pilot_selection_sha256"] or pilot["selection_sha256"] != digest(selection)
            or pilot["status"] != "complete" or cfg["ridge_penalty"] <= 0):
        raise ValueError("Probe configuration or pilot provenance mismatch")
    with selection.open(encoding="utf-8", newline="") as f:
        previous = {r["split_group"] for r in csv.DictReader(f)}
    labels = pilot["models"]["clip"]["leaf_order"]
    picked = select_balanced(records, cfg, labels, previous)
    rows = picked["train"] + picked["dev"]
    n_train = len(picked["train"])
    y = np.array([labels.index(r["label"]) for r in rows])
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = args.output / "selection.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id","split_group","role","label"])
        w.writerows((r["image_id"],r["split_group"],r["split"],r["label"]) for r in rows)
    report = {"scope": "balanced disjoint development diagnostic", "config_sha256": digest(args.config),
              "selection_sha256": digest(manifest), "split_sha256": audit["split_sha256"],
              "prior_selection_sha256": digest(selection), "prior_config_sha256": pilot["config_sha256"],
              "manifest_sha256": audit["manifest_sha256"], "n_train": n_train, "n_dev": len(rows) - n_train,
              "overlap_with_prior_dev": 0, "leaf_order": labels, "device": args.device, "batch_size": args.batch_size,
              "models": {}, "bootstrap": cfg["bootstrap"]}
    try:
        for name, baseline in pilot["models"].items():
            t = time.perf_counter()
            if baseline["leaf_order"] != labels:
                raise ValueError("Leaf ordering differs between models")
            file = args.pilot/f"{name}.npz"
            if digest(file) != baseline["cache_sha256"]:
                raise ValueError("Invalid source text cache")
            with np.load(file, allow_pickle=False) as cache:
                indices = [baseline["text_node_order"].index(k) for k in labels]
                text = cache["text_features"][:,indices,:].mean(0)
                text /= np.linalg.norm(text, axis=1, keepdims=True)
            weights = args.openai_weights if name == "clip" else args.biomed_dir/"open_clip_pytorch_model.bin"
            config = None if name == "clip" else args.biomed_dir/"open_clip_config.json"
            info = {"source_cache_sha256": baseline["cache_sha256"]}
            model, preprocess, _ = load_encoder(name, weights, args.device, config, info)
            image = np.concatenate([encode_images(model, preprocess, rows[i:i+args.batch_size], args.device)
                                    for i in range(0,len(rows),args.batch_size)])
            scores = {"frozen_text": image[n_train:] @ text.T}
            scores.update(probe_scores(image[:n_train], image[n_train:], y[:n_train], len(labels), cfg["ridge_penalty"]))
            if set(scores) != set(cfg["methods"]):
                raise ValueError("Methods differ from prespecified configuration")
            info["metrics"] = compare(scores, y[n_train:], labels, cfg["bootstrap"])
            out_cache = args.output/f"{name}.npz"
            np.savez(out_cache, image=image, **scores)
            info.update(cache_sha256=digest(out_cache), seconds=round(time.perf_counter()-t, 3))
            report["models"][name] = info
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start, 3)
        (args.output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
