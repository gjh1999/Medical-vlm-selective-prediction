"""Estimate fixed-pool training-set sensitivity using complementary allocations of the existing HAM groups."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ham_dev_pilot import digest, error_auc
from ham_error_increment import logistic_predict, read_csv, signals
from selective_core import Taxonomy


ROOT = Path(__file__).resolve().parent
COHORTS = ("pooled_natural", "balanced")
CELLS = ("original_support", "new_support"), ("original_fit", "new_fit")


def read_cohorts(paths, tree):
    info = [json.loads((p / "summary.json").read_bytes()) for p in paths]
    rows = [read_csv(p / "selection.csv") for p in paths]
    if any(value["status"] != "complete" or digest(p / "selection.csv") != value["selection_sha256"]
           for p, value in zip(paths, info)):
        raise ValueError("Documented source cohort changed")
    roles = ({"dev": "original_fit"}, {"train": "original_support", "dev": "balanced_evaluation"},
             {"dev": "first_natural"}, {"dev": "second_natural"}, {"train": "new_support", "dev": "new_fit"})
    tags = [(names[r["role"]], r["image_id"], r["role"], r["label"])
            for names, batch in zip(roles, rows) for r in batch]
    fingerprint = hashlib.sha256("".join(",".join(t) + "\n" for t in sorted(tags)).encode()).hexdigest()
    known = json.loads((ROOT / "configs/ham-development-exposure-v1.json").read_bytes())
    if fingerprint != known["membership_sha256"] or len(tags) != known["n_images"]:
        raise ValueError("Used rows do not match the audited 616 supervised groups")
    labels = [np.array([tree.leaves.index(r["label"]) for r in batch]) for batch in rows]
    indices = [np.array([r["role"] == "train" for r in rows[i]]) for i in (1, 4)]
    return info, labels, indices, fingerprint


def allocations(labels, n_fit, cfg):
    size = cfg["support_per_leaf"]
    classes = [np.flatnonzero(labels == y) for y in np.unique(labels)]
    if len(classes) != 7 or any(len(ids) != size * 2 for ids in classes) or n_fit != 2 * cfg["fit_size"]:
        raise ValueError("Source pools differ from the registered quotas")
    rng = np.random.default_rng(cfg["seed"])
    for _ in range(cfg["partitions"]):
        permuted = [rng.permutation(ids) for ids in classes]
        support = [np.sort(np.concatenate([ids[i*size:(i+1)*size] for ids in permuted])) for i in range(2)]
        fits = [np.sort(chunk) for chunk in np.split(rng.permutation(n_fit), 2)]
        yield support, fits


def evaluate(support_images, support_labels, support_sets, fits, images, truth, text, meta, tree, rule, bounds):
    cells = np.full((2, 2, len(bounds)), np.nan)
    fit_count = sum(map(len, fits))
    for i, subset in enumerate(support_sets):
        centers = np.stack([support_images[subset[support_labels[subset] == k]].mean(0) for k in range(len(tree.leaves))])
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        pred, fields = signals(images, text, meta["text_node_order"], tree.leaves, tree, meta["logit_scale"],
                               centers)["cosine_centroid"]
        errors = pred != truth
        matrix = np.column_stack([fields[k] for k in [*rule["base"], rule["extra"]]])
        for j, fit in enumerate(fits):
            if np.unique(errors[fit]).size < 2:
                continue
            scores = [logistic_predict(matrix[fit, :width], errors[fit], matrix[fit_count:, :width],
                                       rule["l2_penalty"])[0] for width in (matrix.shape[1]-1, matrix.shape[1])]
            for c, cohort in enumerate(bounds):
                bad = errors[fit_count:][cohort]
                a, b = [error_auc(bad, values[cohort]) for values in scores]
                if a is not None and b is not None:
                    cells[i, j, c] = b - a
    return cells


def summarize(cells):
    flat = cells.ravel()
    valid = flat[np.isfinite(flat)]
    if not len(valid):
        raise ValueError("No estimable cells")
    complete = cells[np.isfinite(cells).all(axis=(1, 2))]
    def contrast_variance(axis):
        return np.square(complete.mean(axis=axis).std(axis=1)).mean()
    support = float(contrast_variance(2))
    fit = float(contrast_variance(1))
    interaction_term = (complete[:, 0, 0] - complete[:, 0, 1] - complete[:, 1, 0] + complete[:, 1, 1]) / 4
    interaction = float(np.square(interaction_term).mean())
    total = float(complete.var(axis=(1, 2)).mean())
    if not np.isclose(total, support + fit + interaction, rtol=1e-10, atol=1e-15):
        raise ArithmeticError("Within-partition variance decomposition differs")
    return {"valid_cells": int(len(valid)), "undefined_cells": int(len(flat) - len(valid)),
            "positive": int((valid > 0).sum()), "zero": int((valid == 0).sum()), "negative": int((valid < 0).sum()),
            "median": float(np.median(valid)), "empirical_2_5_97_5": np.percentile(valid, [2.5, 97.5]).tolist(),
            "complete_partitions_for_variance": len(complete), "within_partition_mean_square":
            {"support": support, "risk_fit": fit, "interaction": interaction, "total": total}}


def sign_counts(a, b):
    finite = np.isfinite(a) & np.isfinite(b)
    return {"both_positive": int(((a > 0) & (b > 0)).sum()), "both_negative": int(((a < 0) & (b < 0)).sum()),
            "first_positive_second_negative": int(((a > 0) & (b < 0)).sum()),
            "first_negative_second_positive": int(((a < 0) & (b > 0)).sum()),
            "any_zero": int((finite & ((a == 0) | (b == 0))).sum()), "undefined": int((~finite).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "balanced", "increment", "confirmation", "crossed", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    start = time.perf_counter()
    result = {}
    try:
        cfg_path = ROOT / "configs/ham-g1-reallocation-v1.json"
        rule_path = ROOT / "configs/ham-error-increment-v1.json"
        taxonomy_path = ROOT / "configs/ham-dev-pilot-v1.json"
        cfg, rule, taxonomy = [json.loads(p.read_bytes()) for p in (cfg_path, rule_path, taxonomy_path)]
        tree = Taxonomy(taxonomy["parent"])
        paths = [args.pilot, args.balanced, args.increment, args.confirmation, args.crossed]
        infos, labels, (s_old, s_new), membership = read_cohorts(paths, tree)
        if digest(rule_path) != infos[2]["config_sha256"] or digest(taxonomy_path) != infos[0]["config_sha256"]:
            raise ValueError("Frozen rule or taxonomy differs")
        sy = np.r_[labels[1][s_old], labels[4][s_new]]
        fy = np.r_[labels[0], labels[4][~s_new]]
        ey = np.r_[labels[2], labels[3], labels[1][~s_old]]
        n = cfg["evaluation"]["pooled_natural"]
        if n != len(labels[2]) + len(labels[3]) or len(ey) - n != cfg["evaluation"]["balanced"]:
            raise ValueError("Evaluation cohorts differ")
        cohorts = [slice(0, n), slice(n, None)]
        fixed = [np.split(np.arange(len(sy)), 2), np.split(np.arange(len(fy)), 2)]
        draws = list(allocations(sy, len(fy), cfg))
        results = {}
        result.update(scope=cfg["scope_limit"], config_sha256=digest(cfg_path), membership_sha256=membership,
                      numpy=np.__version__, partitions=len(draws), evaluation=cfg["evaluation"], models={})
        for model in ("clip", "biomedclip"):
            files = [p / f"{model}.npz" for p in paths]
            original = infos[2]["models"][model]
            expected = [original["fit_cache"], original["support_cache"], original["cache_sha256"],
                        infos[3]["models"][model]["cache_sha256"], infos[4]["models"][model]["cache_sha256"]]
            if [digest(p) for p in files] != expected:
                raise ValueError("Frozen feature cache changed")
            with (np.load(files[0]) as pilot, np.load(files[1]) as balanced, np.load(files[2]) as first,
                  np.load(files[3]) as second, np.load(files[4]) as crossed):
                support = np.vstack([balanced["image"][s_old], crossed["image_features"][s_new]]).astype(np.float64)
                x = np.vstack([pilot["image_features"], crossed["image_features"][~s_new], first["image_features"],
                               second["image_features"], balanced["image"][~s_old]])
                text = pilot["text_features"]
            def score(draw):
                return evaluate(support, sy, *draw, x, np.r_[fy, ey], text, infos[0]["models"][model], tree, rule, cohorts)
            control = score(fixed)
            previous = infos[4]["models"][model]["cells"]
            target = np.array([[[previous[f"{sa}/{fa}"]["cosine_centroid"][c]["difference"]
                                 for c in ("pooled", "balanced")] for fa in CELLS[1]] for sa in CELLS[0]])
            diff = float(abs(control - target).max())
            if diff > 1e-12:
                raise ValueError("The documented four-cell control differs")
            values = np.stack([score(draw) for draw in draws])
            results[model] = values
            result["models"][model] = {"control_max_absolute_difference": diff,
                                        **{c: summarize(values[:, :, :, i]) for i, c in enumerate(COHORTS)},
                                        "natural_and_balanced": sign_counts(values[..., 0], values[..., 1])}
        result["backbone_direction"] = {c: sign_counts(results["clip"][..., i], results["biomedclip"][..., i])
                                        for i, c in enumerate(COHORTS)}
        comparisons = np.stack([results["clip"], results["biomedclip"]])
        result["both_backbones_and_both_cohorts"] = {
            "all_positive": int((comparisons > 0).all(axis=(0, 4)).sum()),
            "all_negative": int((comparisons < 0).all(axis=(0, 4)).sum()),
            "estimable": int(np.isfinite(comparisons).all(axis=(0, 4)).sum())}
        local = args.output / "cell_differences.npz"
        np.savez(local, **results)
        result.update(status="complete", cell_differences_sha256=digest(local))
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        result["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
