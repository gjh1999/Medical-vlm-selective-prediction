"""Find attainable count/upper-bound pairs as a single Flat or Climbing threshold ranges from 0 to 1."""
import math
from collections import defaultdict
from functools import lru_cache

from selective_core import upper_bound


@lru_cache(maxsize=8000)
def bound(errors, accepted, delta):
    return upper_bound(errors, accepted, delta)


def threshold_envelope(tree, probabilities, labels, cfg):
    """Scan every action-changing threshold and verify the emitted counts against the scalar policies."""
    inputs = [dict(zip(tree.leaves, map(float, row))) for row in probabilities]
    delta = cfg["family_failure_probability"] / (2*len(cfg["baselines"])*len(cfg["thresholds"]))
    output = {}
    for mode in cfg["baselines"]:
        scalar = {"flat": tree.flat, "climbing": tree.climbing}[mode]
        paths = []
        events = defaultdict(set)
        for i, p in enumerate(inputs):
            tree.validate(p)
            node = max(tree.leaves, key=p.get)
            path = []
            while node != tree.root:
                score = sum(p[y] for y in tree.desc[node])
                path.append((node, score))
                if score < 1.:
                    events[score].add(i)
                node = tree.parent[node] if mode == "climbing" else tree.root
            paths.append(path)
        current = [path[0][0] for path in paths]

        def contributions(i, node):
            leaf = node in tree.leaves
            accepted = node != tree.root
            return (int(leaf), int(leaf and node != labels[i]), int(accepted), int(accepted and labels[i] not in tree.desc[node]))

        totals = [sum(v) for v in zip(*(contributions(i, node) for i, node in enumerate(current)))]
        stats = {"states": 0, "scalar_agreement": 0, "leaf_constraint_passes": 0, "nonroot_constraint_passes": 0,
                 "joint_passes": 0, "max_zero_error_leaf_outputs": 0, "minimum_joint_ratio": None}
        def record(threshold):
            direct = [scalar(p, threshold) for p in inputs]
            if (current != direct or totals != [sum(v) for v in zip(*(contributions(i, n) for i, n in enumerate(direct)))]):
                raise ArithmeticError("Event sweep and direct scalar policy differ")
            n, k, m, e = totals
            leaf, nonroot = bound(k, n, delta), bound(e, m, delta)
            ok_leaf = leaf is not None and leaf <= cfg["leaf_target"]
            ok_nonroot = nonroot is not None and nonroot <= cfg["nonroot_target"]
            stats["states"] += 1
            stats["scalar_agreement"] += 1
            stats["leaf_constraint_passes"] += int(ok_leaf)
            stats["nonroot_constraint_passes"] += int(ok_nonroot)
            stats["joint_passes"] += int(ok_leaf and ok_nonroot)
            if k == 0:
                stats["max_zero_error_leaf_outputs"] = max(stats["max_zero_error_leaf_outputs"], n)
            if leaf is None or nonroot is None:
                return
            ratio = max(leaf/cfg["leaf_target"], nonroot/cfg["nonroot_target"])
            if stats["minimum_joint_ratio"] is None or ratio < stats["minimum_joint_ratio"]:
                stats["minimum_joint_ratio"] = ratio
                stats["best_joint_state"] = {"threshold": threshold, "leaf_n": n, "leaf_errors": k,
                                            "nonroot_n": m, "nonroot_errors": e, "leaf_upper": leaf, "nonroot_upper": nonroot}

        record(0.)
        for score, members in sorted(events.items()):
            change = False
            for i in members:
                old = current[i]
                if scalar(inputs[i], score) != old:
                    raise ArithmeticError("Inclusive threshold rule differs at an exact event")
                new = next((n for n, mass in paths[i] if mass > score), tree.root)
                if new == old:
                    continue
                previous, following = contributions(i, old), contributions(i, new)
                totals = [count - lost + got for count, lost, got in zip(totals, previous, following)]
                current[i] = new
                change = True
            if change:
                record(math.nextafter(score, 1.))
        if current != [scalar(p, 1.) for p in inputs]:
            raise ArithmeticError("Threshold one is missing from the final policy state")
        output[mode] = stats
    return {"development_groups": len(labels), "per_check_delta_for_comparison_only": delta, "families": output}
