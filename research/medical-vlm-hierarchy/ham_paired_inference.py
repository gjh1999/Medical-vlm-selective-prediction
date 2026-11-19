"""Paired split-group bootstrap for frozen HAM prediction policies."""
import math
from pathlib import Path


RECIPE = Path(__file__).resolve().parent / "configs/ham-paired-bootstrap-v1.json"
METRICS = ("correct_information_utility", "leaf_coverage")
CLASSIFICATION_CONTRAST = ("cosine_centroid", "frozen_text")


def paired_intervals(tree, truth, group_ids, predictions, contrasts, config):
    import numpy as np

    n = len(truth)
    if n == 0 or len(group_ids) != n or not all(group_ids) or not set(truth) <= set(tree.leaves):
        raise ValueError("Expected a nonempty aligned cohort with a valid split group and leaf label per row")
    if not contrasts or any(len(actions) != n or not set(actions) <= tree.nodes for actions in predictions.values()):
        raise ValueError("Expected aligned predictions and at least one paired contrast")
    groups, group_index = np.unique(group_ids, return_inverse=True)
    sizes = np.bincount(group_index)
    specificity = {node: 1 - math.log(len(leaves)) / math.log(len(tree.leaves)) for node, leaves in tree.desc.items()}
    scores = {name: np.asarray([(specificity[node] if y in tree.desc[node] else 0., node in tree.leaves)
                                for y, node in zip(truth, actions)]) for name, actions in predictions.items()}
    differences = np.stack([scores[candidate] - scores[reference] for candidate, reference in contrasts], axis=1)
    group_sums = np.zeros((len(groups), len(contrasts), len(METRICS)))
    np.add.at(group_sums, group_index, differences)
    point = group_sums.sum(axis=0) / n

    reps = config["replicates"]
    samples = np.zeros((reps, len(contrasts), len(METRICS)))
    if np.any(group_sums):
        rng = np.random.Generator(np.random.PCG64(config["seed"]))
        for offset in range(0, reps, 64):
            batch = samples[offset:offset + 64]
            picked = rng.integers(len(groups), size=(len(batch), len(groups)))
            batch[:] = group_sums[picked].sum(axis=1) / sizes[picked].sum(axis=1)[:, None, None]
    lower, upper = np.percentile(samples, config["interval_percentiles"], axis=0, method="linear")
    comparisons = {}
    for i, (candidate, reference) in enumerate(contrasts):
        comparisons[f"{candidate}_minus_{reference}"] = {
            "candidate": candidate, "reference": reference,
            "different_nodes": int(sum(a != b for a, b in zip(predictions[candidate], predictions[reference]))),
            "metrics": {metric: {"difference": float(point[i, j]), "interval": [float(lower[i, j]), float(upper[i, j])]}
                        for j, metric in enumerate(METRICS)}}
    return {"analysis_units": n, "split_groups": len(groups), "replicates": reps, "seed": config["seed"],
            "interval_percentiles": config["interval_percentiles"], "comparisons": comparisons}


def unrestricted_readout_interval(tree, truth, group_ids, text_logits, centroid_logits, config):
    import numpy as np

    leaves = np.asarray(tree.leaves)
    predictions = {"frozen_text": leaves[text_logits.argmax(axis=1)].tolist(),
                   "cosine_centroid": leaves[centroid_logits.argmax(axis=1)].tolist()}
    panel = paired_intervals(tree, truth, group_ids, predictions, [CLASSIFICATION_CONTRAST], config)
    panel["correct_leaf_counts"] = {name: sum(y == node for y, node in zip(truth, actions))
                                    for name, actions in predictions.items()}
    return panel


def mean_split_differences(panels):
    return {name: {metric: math.fsum(panel["comparisons"][name]["metrics"][metric]["difference"] for panel in panels) / len(panels)
                   for metric in METRICS} for name in panels[0]["comparisons"]}
