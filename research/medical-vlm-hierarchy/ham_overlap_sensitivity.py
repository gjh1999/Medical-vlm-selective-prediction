"""Measure the cost of withholding groups touched by cross-split dHash candidates."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from extract_features import read_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "split-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    audit, records = read_inputs(args.audit)
    base = {r["image_id"]:r for r in records}
    prepared = json.loads((args.split_dir / "audit.json").read_bytes())
    if prepared.get("split_protocol") != "v1.3":
        raise ValueError("Expected reviewed HAM v1.3 splits")
    assignments = {}
    for seed, entry in prepared["splits"].items():
        path = args.split_dir / f"split_seed{seed}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Changed split for seed {seed}")
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(base) or {r["image_id"] for r in rows} != base.keys():
            raise ValueError("Split inventory differs from audit")
        roles = {}
        for r in rows:
            key = r["split_group"]
            reference = base[r["image_id"]]
            if any(r[k] != reference[k] for k in ("split_group", "analysis_group", "representative", "label")):
                raise ValueError("Reviewed group or representative differs across seeds")
            if roles.setdefault(key, r["split"]) != r["split"]:
                raise ValueError("Group crosses partitions")
        assignments[seed] = roles
    pairs_path = args.audit / "duplicate_candidates.json"
    pairs = json.loads(pairs_path.read_bytes())
    if len(pairs) != audit['near_candidate_pairs']:
        raise ValueError("Candidate list count differs from audit")
    excluded = set()
    crossing = 0
    for pair in pairs:
        a = base[pair["first"]]["split_group"]
        b = base[pair["second"]]["split_group"]
        if any(roles[a] != roles[b] for roles in assignments.values()):
            excluded.update((a, b))
            crossing += 1
    args.output.mkdir(exist_ok=False, parents=True)
    group_file = args.output / "sensitivity_groups.csv"
    with group_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split_group", "excluded"))
        writer.writerows((g, int(g in excluded)) for g in sorted(next(iter(assignments.values()))))
    reps = [r for r in records if r["representative"] == "1"]
    report = {"protocol":prepared["split_protocol"], "scope":"all image views, union over all three splits",
              "manifest_sha256":audit["manifest_sha256"],
              "candidate_list_sha256":hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
              "sensitivity_groups_sha256":hashlib.sha256(group_file.read_bytes()).hexdigest(),
              "split_sha256":{s:v["sha256"] for s,v in prepared["splits"].items()},
              "crossing_pairs_in_union":crossing, "groups":prepared["split_groups"],
              "excluded_groups":len(excluded), "analysis_units":len(reps),
              "excluded_analysis_units":sum(r["split_group"] in excluded for r in reps), "splits":{}}
    for seed, roles in assignments.items():
        by_role = {}
        for role in ("train", "dev", "calibration", "test"):
            subset = [r for r in reps if roles[r["split_group"]] == role]
            kept = [r for r in subset if r["split_group"] not in excluded]
            by_role[role] = {"original":len(subset), "retained":len(kept),
                             "original_labels":dict(Counter(r["label"] for r in subset)),
                             "retained_labels":dict(Counter(r["label"] for r in kept))}
        report["splits"][seed] = by_role
    report["seconds"] = round(time.perf_counter() - start, 3)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
