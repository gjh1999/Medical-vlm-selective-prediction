"""Audit the best-case calibration capacity using only PAD group metadata."""
import argparse
import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from selective_core import upper_bound


def read_calibration_groups(path):
    label_groups = defaultdict(set)
    groups = set()
    representatives = 0
    with path.open(encoding="utf-8-sig", newline="") as src:
        for row in csv.DictReader(src):
            if row["split"] == "calibration" and row["representative"] == "1":
                group = row["split_group"]
                label = row["label"]
                if not group.strip() or not label.strip():
                    raise ValueError(f"Missing calibration group or label in {path}")
                representatives += 1
                groups.add(group)
                label_groups[label].add(group)
    if not groups:
        raise ValueError(f"No calibration representatives in {path}")
    counts = {
        "representatives": representatives,
        "groups": len(groups),
        "groups_containing_each_label": {k: len(v) for k, v in sorted(label_groups.items())},
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return counts, groups, label_groups


def overlap_counts(groups_by_seed):
    sets = list(groups_by_seed.values())
    return {
        "distinct_across_seeds": len(set.union(*sets)),
        "common_to_all_seeds": len(set.intersection(*sets)),
        "pairwise_intersections": {
            f"{a}/{b}": len(groups_by_seed[a] & groups_by_seed[b])
            for a, b in combinations(groups_by_seed, 2)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8-sig"))
    delta = protocol["family_failure_probability"] / (
        protocol["max_candidate_policies"] * protocol["simultaneous_constraints"]
    )
    splits = {}
    groups_by_seed = {}
    labels_by_seed = {}
    for seed in protocol["split_seeds"]:
        counts, groups, label_groups = read_calibration_groups(args.split_dir / f"split_seed{seed}.csv")
        splits[seed] = counts
        groups_by_seed[seed] = groups
        labels_by_seed[seed] = label_groups
    labels = sorted(set().union(*(by_label.keys() for by_label in labels_by_seed.values())))
    limits = {}
    for seed, counts in splits.items():
        available = counts["groups"]
        budgets = [n for n in protocol["calibration_size_grid_independent_groups"]
                   if isinstance(n, int) and n <= available]
        limits[seed] = {}
        for n in sorted(set(budgets + [available])):
            # Assume every group is accepted; any rejection tightens this ceiling.
            allowed = {}
            for name, target in protocol["risk_targets"].items():
                k = 0
                while k <= n and upper_bound(k, n, delta) <= target:
                    k += 1
                allowed[name] = k - 1 if k else None
            limits[seed][n] = allowed
    result = {
        "protocol": protocol["version"], "python": platform.python_version(),
        "per_check_delta": delta, "splits": splits,
        "max_errors_if_all_groups_accepted": limits,
        "calibration_group_overlap": {
            "overall": overlap_counts(groups_by_seed),
            "by_label": {
                label: overlap_counts({seed: by_label.get(label, set()) for seed, by_label in labels_by_seed.items()})
                for label in labels
            },
        },
        "scope": "Count ceiling only. No model outputs or iid claim.",
        "elapsed_compute_seconds": time.perf_counter() - start,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "capacity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
