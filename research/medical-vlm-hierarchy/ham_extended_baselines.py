"""Matched fixed-candidate HAM baselines and separate split-conformal set evaluation."""
import math
from collections import Counter
from pathlib import Path

from ham_paired_inference import paired_intervals
from selective_core import conformal_quantile, lac_set, select_policy


RECIPE = Path(__file__).resolve().parent / "configs/ham-extended-baselines-v1.json"
MAX_PROBABILITY = "flat_max_probability"


def top_level_branches(tree):
    branches = {}
    for node in tree.parent:
        ancestor = node
        while tree.parent[ancestor] != tree.root:
            ancestor = tree.parent[ancestor]
        branches[node] = ancestor
    return branches


def test_metrics(tree, labels, predictions, branch, unrestricted=False):
    report = tree.metrics(labels, predictions)
    errors = Counter(y for y, pred in zip(labels, predictions) if pred != tree.root and branch[y] != branch[pred])
    for label, metrics in report["per_class"].items():
        metrics["cross_branch_errors"] = errors[label]
        metrics["cross_branch_risk"] = errors[label] / metrics["node_n"] if metrics["node_n"] else None
    report["cross_branch_errors"] = sum(errors.values())
    report["cross_branch_risk"] = report["cross_branch_errors"] / report["node_n"] if report["node_n"] else None
    if unrestricted:
        confusion = Counter(zip(labels, predictions))
        leaf_f1 = {}
        for leaf in tree.leaves:
            denominator = labels.count(leaf) + predictions.count(leaf)
            leaf_f1[leaf] = 2 * confusion[leaf, leaf] / denominator if denominator else None
        defined = [score for score in leaf_f1.values() if score is not None]
        report["leaf_f1"] = leaf_f1
        report["macro_f1"] = sum(defined) / len(defined)
        report["accuracy"] = 1 - report["leaf_risk"]
    return report


def cutoff_grid(dev_scores, slots):
    ordered = sorted(map(float, dev_scores))
    if not ordered or not all(map(math.isfinite, ordered)) or not 1 <= slots <= 16:
        raise ValueError("Expected finite development scores and 1..16 candidate slots")
    n = len(ordered)
    return {f"slot_{j:02d}": ordered[(j * n + slots) // (slots + 1) - 1] for j in range(1, slots + 1)}


def actions(tree, mode, probabilities, confidence, cutoff):
    if mode == "climbing":
        return [tree.climbing(row, cutoff) for row in probabilities]
    if mode == "two_stage":
        return [tree.two_stage(row, cutoff, cutoff) for row in probabilities]
    return [tree.flat(row) if score >= cutoff else tree.root for row, score in zip(probabilities, confidence)]


def evaluate_lac(tree, probabilities, labels, quantile, branches):
    sets = [lac_set(row, quantile) for row in probabilities]
    predictions = [tree.lca(names) for names in sets]
    contained = [y in names for y, names in zip(labels, sets)]
    histogram = Counter(map(len, sets))
    return {"n": len(labels), "set_contains_truth": sum(contained), "set_coverage": sum(contained) / len(labels),
            "mean_set_size": sum(map(len, sets)) / len(labels), "set_size_histogram": dict(sorted(histogram.items())),
            "empty_set_to_root": histogram[0],
            "nonempty_sets_expanded": sum(bool(names) and len(names) < len(tree.desc[pred])
                                         for names, pred in zip(sets, predictions)),
            "class_set_coverage": {leaf: {"n": labels.count(leaf),
                                          "covered": sum(ok for y, ok in zip(labels, contained) if y == leaf)}
                                   for leaf in tree.leaves},
            "projection": test_metrics(tree, labels, predictions, branches)}


def compare_at_temperature(tree, probabilities, signals, development, populations, config, controller,
                           bootstrap=None, evaluation_groups=None):
    """Split indices use only development rows to set candidate thresholds."""
    branches = top_level_branches(tree)
    grids = {mode: cutoff_grid(scores[development], config["grid_slots_per_family"]) for mode, scores in signals.items()}
    for mode in ("climbing", "two_stage"):
        grids[mode] = grids[MAX_PROBABILITY]
    output = {}
    for population, (cal_indices, cal_labels, views) in populations.items():
        calibration = [probabilities[i] for i in cal_indices]
        test = {name: ([probabilities[i] for i in idx], idx, labels) for name, (idx, labels) in views.items()}
        families = {}
        paired_predictions = {name: {} for name in evaluation_groups[population]} if bootstrap is not None else {}
        for mode, grid in grids.items():
            scores = signals.get(mode, signals[MAX_PROBABILITY])
            candidates = {slot: actions(tree, mode, calibration, scores[cal_indices], cutoff)
                          for slot, cutoff in grid.items()}
            trial = select_policy(tree, cal_labels, candidates, controller["family_failure_probability"],
                                  controller["leaf_target"], controller["nonroot_target"])
            picked = trial["selected"]
            test_results = {}
            for name, (rows, indices, labels) in test.items():
                predictions = (actions(tree, mode, rows, scores[indices], grid[picked]) if picked
                               else [tree.root] * len(rows))
                test_results[name] = test_metrics(tree, labels, predictions, branches)
                if name in paired_predictions:
                    paired_predictions[name][mode] = predictions
            audit = {}
            for row in trial["audit"]:
                metrics = row["metrics"]
                audit[row["name"]] = {key: metrics[key] for key in
                                      ("leaf_n", "leaf_errors", "node_n", "node_errors",
                                       "correct_information_utility", "leaf_coverage")}
                audit[row["name"]].update(leaf_upper=row["leaf_upper"], node_upper=row["node_upper"],
                                          feasible=row["feasible"])
            families[mode] = {"cutoffs": grid, "distinct_cutoffs": len(set(grid.values())), "selected": picked,
                              "per_check_delta": trial["per_check_delta"], "calibration": audit, "test": test_results}

        nonconformity = [1 - row[y] for row, y in zip(calibration, cal_labels)]
        lac = {}
        for alpha in config["lac_alphas"]:
            quantile = conformal_quantile(nonconformity, alpha)
            lac[str(alpha)] = {"quantile": quantile if math.isfinite(quantile) else None,
                               "unbounded_quantile": math.isinf(quantile),
                               "order_statistic": math.ceil((len(cal_labels) + 1) * (1 - alpha)),
                               "test": {name: evaluate_lac(tree, rows, labels, quantile, branches)
                                        for name, (rows, _, labels) in test.items()}}
        output[population] = {"calibration_groups": len(cal_labels), "families": families, "lac": lac}
        if bootstrap is not None:
            contrasts = [(mode, MAX_PROBABILITY) for mode in families if mode != MAX_PROBABILITY]
            output[population]["paired_bootstrap"] = {
                name: paired_intervals(tree, test[name][2], evaluation_groups[population][name], predictions, contrasts, bootstrap)
                for name, predictions in paired_predictions.items()}
    return output


def evaluate_readout(tree, logits, negative_prompt_js, development, populations, config, controller,
                     bootstrap=None, evaluation_groups=None):
    """Evaluate native and development-fitted temperatures on the same cohorts."""
    import numpy as np

    from audit_ham_temperature import fit_inverse_temperature, loss
    from ham_dev_pilot import softmax

    dev_indices, dev_labels = development
    targets = np.asarray([tree.leaves.index(label) for label in dev_labels])
    dev_logits = logits[dev_indices]
    beta = fit_inverse_temperature(dev_logits, targets, config["inverse_temperature_bounds"])
    result = {"development_groups": len(targets), "native_dev_nll": loss(dev_logits, targets, 1),
              "fitted_dev_nll": loss(dev_logits, targets, beta), "temperature_at_bound": beta in
              config["inverse_temperature_bounds"], "temperatures": {}}
    for name, inverse_temperature in (("native", 1.), ("dev_fitted", beta)):
        probs = softmax(logits * inverse_temperature)
        order = np.sort(probs, axis=1)
        log_probs = np.zeros_like(probs)
        np.log(probs, out=log_probs, where=probs > 0)
        signals = {MAX_PROBABILITY: order[:, -1], "flat_margin": order[:, -1] - order[:, -2],
                   "flat_negative_entropy": np.sum(probs * log_probs, axis=1),
                   "flat_negative_prompt_js": negative_prompt_js}
        populations_result = compare_at_temperature(tree, [dict(zip(tree.leaves, row)) for row in probs], signals,
                                                     dev_indices, populations, config, controller, bootstrap, evaluation_groups)
        for population, (_, _, views) in populations.items():
            reliability = {}
            for view, (indices, labels) in views.items():
                target = np.asarray([tree.leaves.index(label) for label in labels])
                residual = probs[indices].copy()
                residual[np.arange(len(labels)), target] -= 1
                brier = np.sum(residual ** 2, axis=1)
                reliability[view] = {"negative_log_likelihood": loss(logits[indices], target, inverse_temperature),
                                     "multiclass_brier_sum": float(np.mean(brier))}
            populations_result[population]["leaf_reliability"] = reliability
        result["temperatures"][name] = {"inverse_temperature": inverse_temperature, "populations": populations_result}
    return result


def compare_prompt_templates(tree, prompt_probabilities, ensemble_logits, populations):
    """Score each fixed template and the embedding ensemble on identical test views."""
    import numpy as np

    leaves = np.asarray(tree.leaves)
    branches = top_level_branches(tree)
    outputs = {f"template_{i}": leaves[probs.argmax(axis=1)] for i, probs in enumerate(prompt_probabilities, 1)}
    outputs["ensemble"] = leaves[ensemble_logits.argmax(axis=1)]
    return {name: {view: {head: test_metrics(tree, labels, mode[indices].tolist(), branches, unrestricted=True)
                          for head, mode in outputs.items()}
                   for view, (indices, labels) in views.items()}
            for name, (_, _, views) in populations.items()}
