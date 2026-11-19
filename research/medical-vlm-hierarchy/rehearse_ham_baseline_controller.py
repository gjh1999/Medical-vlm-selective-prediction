"""Exercise the fixed baseline calibration controller on previously used HAM development groups."""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ham_dev_pilot import digest, softmax
from ham_error_increment import read_csv
from reallocate_ham_development_training import ROOT, read_cohorts
from selective_core import Taxonomy, policy_actions, select_policy, upper_bound


def exact_pass(errors, accepted, risk, delta):
    if not accepted:
        return False
    probability = math.fsum(math.comb(accepted, j) * risk**j * (1-risk)**(accepted-j) for j in range(errors+1))
    return probability <= delta


def best_possible(tree, truth, prob, delta, cfg):
    actions = []
    for y, row in zip(truth, prob):
        leaf = max(tree.leaves, key=row.get)
        node = leaf
        while node != tree.root and y not in tree.desc[node]:
            node = tree.parent[node]
        actions.append(node)
    metrics = tree.metrics(truth, actions)
    if metrics["node_errors"] or metrics["leaf_errors"]:
        raise ArithmeticError("A label-aware oracle made an incorrect nonroot prediction")
    leaf = upper_bound(0, metrics["leaf_n"], delta)
    node = upper_bound(0, metrics["node_n"], delta)
    feasible = leaf is not None and leaf <= cfg["leaf_target"] and node is not None and node <= cfg["nonroot_target"]
    return {"correct_leaf_available": metrics["leaf_n"], "correct_nonroot_available": metrics["node_n"],
            "minimum_leaf_upper": leaf, "minimum_nonroot_upper": node, "oracle_could_pass": feasible}


def analyze(tree, probabilities, labels, fold, bounds, cfg):
    stream = [dict(zip(tree.leaves, row)) for row in probabilities]
    candidates = policy_actions(tree, stream, cfg)
    by_budget = {}
    for budget in cfg["nested_budgets"]:
        ix = fold[:budget]
        p = [stream[i] for i in ix]
        y = [labels[i] for i in ix]
        proxy = {name: [all_actions[i] for i in ix] for name, all_actions in candidates.items()}
        trial = select_policy(tree, y, proxy, cfg["family_failure_probability"], cfg["leaf_target"], cfg["nonroot_target"])
        delta = trial["per_check_delta"]
        if len(candidates) != 16 or delta != .05/32:
            raise ValueError("Multiplicity budget differs from the frozen controller")
        oracle = best_possible(tree, y, p, delta, cfg)
        rows = {}
        for row in trial["audit"]:
            m = row["metrics"]
            checks = [exact_pass(m["leaf_errors"], m["leaf_n"], cfg["leaf_target"], delta),
                      exact_pass(m["node_errors"], m["node_n"], cfg["nonroot_target"], delta)]
            if all(checks) != row["feasible"] or (row["feasible"] and not oracle["oracle_could_pass"]):
                raise ArithmeticError("Independent binomial inversion or oracle limit differs")
            rows[row["name"]] = {k: m[k] for k in ("leaf_n", "leaf_errors", "node_n", "node_errors", "correct_information_utility")}
            rows[row["name"]].update(leaf_upper=row["leaf_upper"], nonroot_upper=row["node_upper"], passed=checks)
        name = trial["selected"]
        evaluations = {}
        for population, subset in bounds.items():
            actions = candidates[name][subset] if name else [tree.root] * len(labels[subset])
            evaluations[population] = tree.metrics(labels[subset], actions)
        by_budget[budget] = {"groups": len(y), "label_counts": dict(sorted(Counter(y).items())),
                "per_check_delta": delta, "selected": name, "status": trial["status"],
                "oracle": oracle, "passing": {"leaf_only_constraint": sum(v["passed"][0] for v in rows.values()),
                "nonroot_only_constraint": sum(v["passed"][1] for v in rows.values()),
                "both": sum(all(v["passed"]) for v in rows.values())}, "candidates": rows, "evaluation": evaluations}
    return by_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "balanced", "increment", "confirmation", "crossed", "splits", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threshold-envelope", action="store_true",
                        help="Descriptively exhaust all flat/Climbing cutoffs on the complete proxy-calibration cohort")
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {}
    try:
        cfg_path = ROOT / "configs/ham-controller-rehearsal-v1.json"
        tree_path = ROOT / "configs/ham-dev-pilot-v1.json"
        cfg = json.loads(cfg_path.read_bytes())
        tree = Taxonomy(json.loads(tree_path.read_bytes())["parent"])
        paths = [a.pilot, a.balanced, a.increment, a.confirmation, a.crossed]
        infos, labels, (old_tr, new_tr), signature = read_cohorts(paths, tree)
        split = a.splits / "split_seed17.csv"
        if digest(split) != infos[4]["split_sha256"] or digest(tree_path) != infos[0]["config_sha256"]:
            raise ValueError("Unexpected split or hierarchy")
        active = {r["image_id"]: r for r in read_csv(split)}
        history = [read_csv(p / "selection.csv") for p in paths]
        seen = set()
        for cohort in history:
            for row in cohort:
                now = active[row["image_id"]]
                g = now["split_group"]
                if (g in seen or now["representative"] != "1" or now["split"] != row["role"]
                        or now["label"] != row["label"] or now["split"] not in ("dev", "train")):
                    raise ValueError("Supervised cohorts overlap or development role differs")
                seen.add(g)
        fit_ids = [r["image_id"] for r in [*history[0], *[v for v in history[4] if v["role"] == "dev"]]]
        order = sorted(range(len(fit_ids)), key=lambda i: hashlib.sha256(
            (cfg["order_salt"] + "|" + active[fit_ids[i]]["split_group"]).encode()).digest())
        n_fit, n_natural = len(order), len(labels[2]) + len(labels[3])
        targets = np.r_[labels[0], labels[4][~new_tr], labels[2], labels[3], labels[1][~old_tr]]
        truth = [tree.leaves[i] for i in targets]
        bounds = {"pooled_natural": slice(n_fit, n_fit+n_natural), "balanced": slice(n_fit+n_natural, None)}
        if (n_fit, n_natural, len(truth[bounds["balanced"]])) != (256, 192, 56):
            raise ValueError("Rehearsal cohort sizes differ")
        report.update(scope=cfg["interpretation"], config_sha256=digest(cfg_path), membership_sha256=signature,
                      split_sha256=digest(split), models={})
        for model in ("clip", "biomedclip"):
            source = infos[2]["models"][model]
            files = [p / f"{model}.npz" for p in paths]
            expected = [source["fit_cache"], source["support_cache"], source["cache_sha256"],
                        infos[3]["models"][model]["cache_sha256"], infos[4]["models"][model]["cache_sha256"]]
            if list(map(digest, files)) != expected:
                raise ValueError("Frozen feature provenance differs")
            with (np.load(files[0]) as p, np.load(files[1]) as b, np.load(files[2]) as first,
                  np.load(files[3]) as second, np.load(files[4]) as crossed):
                x = np.concatenate([p["image_features"], crossed["image_features"][~new_tr], first["image_features"],
                                    second["image_features"], b["image"][~old_tr]]).astype(np.float64)
                info = infos[0]["models"][model]
                templates = p["text_features"].astype(np.float64).mean(0)
                heads = {"frozen_text": templates[[info["text_node_order"].index(y) for y in tree.leaves]]}
                for key, vectors, y in (("original_centroid", b["image"][old_tr], labels[1][old_tr]),
                                        ("new_centroid", crossed["image_features"][new_tr], labels[4][new_tr])):
                    heads[key] = np.stack([vectors[y == k].astype(np.float64).mean(0) for k in range(len(tree.leaves))])
            report["models"][model] = {}
            for name, head in heads.items():
                prob = softmax(info["logit_scale"] * (x @ (head / np.linalg.norm(head, axis=1, keepdims=True)).T))
                support = "new_support" if name == "new_centroid" else "original_support"
                source_name = "frozen_text" if name == "frozen_text" else "cosine_centroid"
                ref = infos[4]["models"][model]["cells"][support + "/original_fit"][source_name]
                for subset, key in (("pooled_natural", "pooled"), ("balanced", "balanced")):
                    sl = bounds[subset]
                    if np.sum(prob[sl].argmax(1) != targets[sl]) != ref[key]["errors"]:
                        raise ValueError("Evaluation predictions differ from the crossed reference")
                if a.threshold_envelope:
                    from ham_threshold_envelope import threshold_envelope
                    report["models"][model][name] = threshold_envelope(tree, prob[order], [truth[k] for k in order], cfg)
                else:
                    report["models"][model][name] = analyze(tree, prob, truth, order, bounds, cfg)
        if a.threshold_envelope:
            report["scope"] = ("Exhaustive, label-aware development diagnostic at the fixed 16-candidate numerical cutoffs. "
                               "No selected policy or risk certificate from this search.")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - start, 3)
        (a.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
