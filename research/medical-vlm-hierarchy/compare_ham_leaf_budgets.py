"""Compare two previously fixed leaf classifiers under equal, nested support budgets on existing HAM dev cohorts."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ham_balanced_probe import probe_scores
from ham_dev_pilot import digest
from reallocate_ham_development_training import ROOT, read_cohorts
from selective_core import Taxonomy


def distribution(numbers):
    return {"mean": float(np.mean(numbers)), "p2_5_median_p97_5": np.percentile(numbers, [2.5, 50, 97.5]).tolist()}


def fit_and_check(x, y, evaluation, penalty, classes):
    main = probe_scores(x, evaluation, y, classes, penalty)
    x_mean = x.mean(axis=0)
    target = np.eye(classes)[y]
    mean = target.mean(axis=0)
    u, sig, vt = np.linalg.svd(x - x_mean, full_matrices=False)
    ridge = (evaluation - x_mean) @ (vt.T * (sig/(sig*sig+penalty))) @ (u.T @ (target - mean)) + mean
    gap = float(np.abs(ridge - main["centered_ridge"]).max())
    if gap > 1e-10 or not np.array_equal(ridge.argmax(1), main["centered_ridge"].argmax(1)):
        raise ArithmeticError("Primal SVD and dual ridge disagree")
    return main, gap


def summarize(predictions, truth, leaves):
    means = {}
    n = len(truth)
    counts = np.bincount(truth, minlength=len(leaves))
    masks = [truth == i for i in range(len(leaves))]
    correct_arrays = {}
    for name, guesses in predictions.items():
        correct = guesses == truth
        correct_arrays[name] = correct
        by_leaf = np.column_stack([correct[:, mask].sum(1) for mask in masks])
        macro = (by_leaf[:, counts > 0] / counts[counts > 0]).mean(1)
        means[name] = {"correct": distribution(correct.sum(1)), "macro_recall_present_labels": distribution(macro),
                       "mean_correct_by_label": dict(zip(leaves, by_leaf.mean(0).tolist()))}
    centroid, ridge = correct_arrays["cosine_nearest_centroid"], correct_arrays["centered_ridge"]
    gained = (ridge & ~centroid).sum(1)
    lost = (centroid & ~ridge).sum(1)
    return {"n": n, "by_label": dict(zip(leaves, counts.tolist())), "heads": means,
            "ridge_minus_centroid": {"corrected": distribution(gained), "lost": distribution(lost),
            "net_correct": distribution(gained - lost), "more": int((gained > lost).sum()),
            "equal": int((gained == lost).sum()), "fewer": int((gained < lost).sum())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "balanced", "increment", "confirmation", "crossed", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    out = {}
    try:
        cfg_path = ROOT / "configs/ham-leaf-budget-v1.json"
        cfg = json.loads(cfg_path.read_bytes())
        tree = Taxonomy(json.loads((ROOT / "configs/ham-dev-pilot-v1.json").read_bytes())["parent"])
        paths = [a.pilot, a.balanced, a.increment, a.confirmation, a.crossed]
        infos, ys, (old_tr, new_tr), fingerprint = read_cohorts(paths, tree)
        sy = np.r_[ys[1][old_tr], ys[4][new_tr]]
        ey = np.r_[ys[2], ys[3], ys[1][~old_tr]]
        if not np.all(np.bincount(sy) == 16) or len(ey) != sum(cfg["evaluation"].values()):
            raise ValueError("Support or development inventory differs")
        rng = np.random.default_rng(cfg["seed"])
        by_class = [np.flatnonzero(sy == k) for k in range(len(tree.leaves))]
        draws = [[rng.permutation(indices) for indices in by_class] for _ in range(cfg["partitions"])]
        natural = cfg["evaluation"]["natural"]
        out.update(scope=cfg["interpretation"], membership_sha256=fingerprint, config_sha256=digest(cfg_path),
                   partitions=cfg["partitions"], numpy=np.__version__, models={})
        for model in ("clip", "biomedclip"):
            files = [p / f"{model}.npz" for p in paths[1:]]
            src = infos[2]["models"][model]
            expected = [src["support_cache"], src["cache_sha256"], infos[3]["models"][model]["cache_sha256"],
                        infos[4]["models"][model]["cache_sha256"]]
            if list(map(digest, files)) != expected:
                raise ValueError("Feature provenance differs")
            with (np.load(files[0]) as b, np.load(files[1]) as f, np.load(files[2]) as s, np.load(files[3]) as z):
                x = np.vstack([b["image"][old_tr], z["image_features"][new_tr]]).astype(np.float64)
                ev = np.vstack([f["image_features"], s["image_features"], b["image"][~old_tr]]).astype(np.float64)
                control, _ = fit_and_check(x[:old_tr.sum()], sy[:old_tr.sum()], ev[natural:],
                                           cfg["ridge_penalty"], len(tree.leaves))
                if any(not np.array_equal(control[k].argmax(1), b[k].argmax(1)) for k in cfg["readouts"]):
                    raise ValueError("Original balanced E01 predictions differ")
            results = {}
            max_gap = 0.
            for shots in cfg["shots_per_leaf"]:
                guesses = {k: [] for k in cfg["readouts"]}
                for order in draws:
                    indices = np.concatenate([part[:shots] for part in order])
                    scores, gap = fit_and_check(x[indices], sy[indices], ev, cfg["ridge_penalty"], len(tree.leaves))
                    max_gap = max(max_gap, gap)
                    for k in guesses:
                        guesses[k].append(scores[k].argmax(1))
                results[shots] = {}
                for cohort, part in (("natural", slice(0, natural)), ("balanced", slice(natural, None))):
                    by_method = {k: np.stack(v)[:, part] for k, v in guesses.items()}
                    results[shots][cohort] = summarize(by_method, ey[part], tree.leaves)
            out["models"][model] = {"budgets": results, "max_independent_ridge_score_difference": max_gap,
                                    "ridge_scores_checked": len(draws)*len(cfg["shots_per_leaf"])*len(ey)*len(tree.leaves)}
        out["status"] = "complete"
    except Exception as exc:
        out.update(status="failed", error=str(exc))
        raise
    finally:
        out["seconds"] = round(time.perf_counter() - start, 3)
        (a.output / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
