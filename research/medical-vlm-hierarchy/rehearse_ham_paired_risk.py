"""Rehearse baseline and supervised error-ranking controllers at matched calibration membership and budgets."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ham_dev_pilot import digest, error_auc, softmax
from ham_error_increment import logistic_predict, signals
from reallocate_ham_development_training import ROOT, read_cohorts
from rehearse_ham_baseline_controller import exact_pass, policy_actions
from selective_core import Taxonomy, select_policy


def risk_actions(tree, probabilities, scores, cfg):
    leaves = [max(tree.leaves, key=p.get) for p in probabilities]
    parents = []
    for leaf, p in zip(leaves, probabilities):
        parent = tree.parent[leaf]
        eligible = parent != tree.root and sum(p[y] for y in tree.desc[parent]) >= cfg["parent_sum_threshold"]
        parents.append(parent if eligible else tree.root)
    result = {}
    for tau in cfg["risk_cutoffs"]:
        for fallback in cfg["fallbacks"]:
            alternatives = parents if fallback == "same_path_parent" else [tree.root] * len(scores)
            result[f"risk_{tau:.3f}_{fallback}"] = [leaf if err <= tau else alt for leaf, err, alt in zip(leaves, scores, alternatives)]
    return result


def rehearse(tree, truth, predictions, calibration, evaluation, cfg):
    if len(predictions) != cfg["per_family_candidates"]:
        raise ValueError("Unmatched policy family budget")
    labels = [truth[i] for i in calibration]
    options = {name: [choices[i] for i in calibration] for name, choices in predictions.items()}
    chosen = select_policy(tree, labels, options, cfg["family_failure_probability"], cfg["leaf_target"], cfg["nonroot_target"])
    out = {"selected": chosen["selected"], "status": chosen["status"], "calibration_groups": len(labels),
           "per_check_delta": chosen["per_check_delta"], "candidates": {}, "evaluation": {}}
    lowest = None
    for r in chosen["audit"]:
        m = r["metrics"]
        independent = [exact_pass(m["leaf_errors"], m["leaf_n"], cfg["leaf_target"], chosen["per_check_delta"]),
                       exact_pass(m["node_errors"], m["node_n"], cfg["nonroot_target"], chosen["per_check_delta"])]
        if independent != [r["leaf_upper"] is not None and r["leaf_upper"] <= cfg["leaf_target"],
                           r["node_upper"] is not None and r["node_upper"] <= cfg["nonroot_target"]]:
            raise ArithmeticError("Risk bound and binomial test disagree")
        out["candidates"][r["name"]] = {k: m[k] for k in ("leaf_n", "leaf_errors", "node_n", "node_errors", "correct_information_utility")}
        out["candidates"][r["name"]].update(leaf_upper=r["leaf_upper"], nonroot_upper=r["node_upper"], passed=independent)
        if None not in (r["leaf_upper"], r["node_upper"]):
            ratio = max(r["leaf_upper"]/cfg["leaf_target"], r["node_upper"]/cfg["nonroot_target"])
            if lowest is None or ratio < lowest[0]:
                lowest = (ratio, r["name"])
    out["closest_joint_ratio_and_candidate"] = lowest
    for cohort, idx in evaluation.items():
        actions = predictions[chosen["selected"]] if chosen["selected"] else [tree.root] * len(truth)
        out["evaluation"][cohort] = tree.metrics([truth[i] for i in idx], [actions[i] for i in idx])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "balanced", "increment", "confirmation", "crossed", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = {}
    try:
        cfg_path = ROOT / "configs/ham-paired-risk-rehearsal-v1.json"
        rule_path = ROOT / "configs/ham-error-increment-v1.json"
        tree_path = ROOT / "configs/ham-dev-pilot-v1.json"
        cfg, rule, taxonomy = [json.loads(p.read_bytes()) for p in (cfg_path, rule_path, tree_path)]
        baseline = json.loads((ROOT / "configs/ham-controller-rehearsal-v1.json").read_bytes())
        tree = Taxonomy(taxonomy["parent"])
        paths = [a.pilot, a.balanced, a.increment, a.confirmation, a.crossed]
        infos, ys, (old_tr, new_tr), signature = read_cohorts(paths, tree)
        if digest(tree_path) != infos[0]["config_sha256"] or digest(rule_path) != infos[2]["config_sha256"]:
            raise ValueError("Unexpected hierarchy or risk fitting rule")
        y = np.r_[ys[0], ys[4][~new_tr], ys[2], ys[3], ys[1][~old_tr]]
        counts = list(map(len, (ys[0], ys[4][~new_tr], ys[2], ys[3], ys[1][~old_tr])))
        if counts != [128,128,96,96,56]:
            raise ValueError("Unexpected fit or evaluation cohorts")
        old_fit, new_fit, first, second, bal = np.split(np.arange(len(y)), np.cumsum(counts)[:-1])
        truth = [tree.leaves[k] for k in y]
        result.update(scope=cfg["interpretation"], config_sha256=digest(cfg_path), membership_sha256=signature, models={})
        for model in ("clip", "biomedclip"):
            files = [p / f"{model}.npz" for p in paths]
            src = infos[2]["models"][model]
            gold = infos[4]["models"][model]
            expected = [src["fit_cache"], src["support_cache"], src["cache_sha256"],
                        infos[3]["models"][model]["cache_sha256"], gold["cache_sha256"]]
            if list(map(digest, files)) != expected:
                raise ValueError("Source cache differs")
            with (np.load(files[0]) as u, np.load(files[1]) as b, np.load(files[2]) as f,
                  np.load(files[3]) as s, np.load(files[4]) as z):
                text = u["text_features"]
                xx = np.vstack([u["image_features"], z["image_features"][~new_tr], f["image_features"],
                                s["image_features"], b["image"][~old_tr]]).astype(np.float64)
                supports = {"original_support": (b["image"][old_tr], ys[1][old_tr]),
                            "new_support": (z["image_features"][new_tr], ys[4][new_tr])}
            info = infos[0]["models"][model]
            result["models"][model] = {}
            for support, (sx, sy) in supports.items():
                centers = np.stack([sx[sy == k].astype(np.float64).mean(0) for k in range(len(tree.leaves))])
                centers /= np.linalg.norm(centers, axis=1, keepdims=True)
                pred, features = signals(xx, text, info["text_node_order"], tree.leaves, tree,
                                         info["logit_scale"], centers)["cosine_centroid"]
                p = softmax((xx @ centers.T) * info["logit_scale"])
                if not np.array_equal(pred, p.argmax(1)):
                    raise ValueError("Probability and risk readouts disagree")
                prob = [dict(zip(tree.leaves, row)) for row in p]
                flat = policy_actions(tree, prob, baseline)
                wrong = pred != y
                matrix = np.column_stack([features[k] for k in [*rule["base"], rule["extra"]]])
                for fit_name, train, other in (("original_fit",old_fit,new_fit),("new_fit",new_fit,old_fit)):
                    key = support + "/" + fit_name
                    cal = np.r_[other, first]
                    eval_sets = {"second_natural": second, "balanced": bal}
                    if len(np.unique(np.r_[train, cal, second, bal])) != len(y):
                        raise ValueError("This cell reuses a fitting, calibration or evaluation group")
                    families = {"fixed_flat_and_climbing": flat}
                    for width, name, prior in ((3, "base_risk", "base_auroc"), (4, "augmented_risk", "augmented_auroc")):
                        scores, _ = logistic_predict(matrix[train, :width], wrong[train], matrix[:, :width], rule["l2_penalty"])
                        for cohort, ids in (("first",first), ("second",second), ("balanced",bal)):
                            reference = gold["cells"][key]["cosine_centroid"][cohort][prior]
                            if abs(error_auc(wrong[ids], scores[ids]) - reference) > 1e-12:
                                raise ArithmeticError("Original development risk ranking differs")
                        families[name] = risk_actions(tree, prob, scores, cfg)
                    result["models"][model][key] = {name: rehearse(tree, truth, predictions, cal, eval_sets, cfg)
                                                      for name, predictions in families.items()}
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
        raise
    finally:
        result["seconds"] = round(time.perf_counter() - start, 3)
        (a.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
