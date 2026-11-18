"""Count the population eligible for each source-to-target label comparison, before model evaluation."""
import argparse
import csv
import hashlib
import io
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

from selective_core import upper_bound


def counts(rows, shared, coarse):
    labels = Counter(r["label"] for r in rows)
    by_group = defaultdict(set)
    for row in rows:
        by_group[row["split_group"]].add(row["label"])
    matched = [ys for ys in by_group.values() if ys & shared]
    groups = {key: sum(bool(ys & set(leaves)) for ys in by_group.values()) for key, leaves in coarse.items()}
    return {"representatives": len(rows), "groups": len(by_group), "leaf_counts": dict(sorted(labels.items())),
            "shared_leaf_representatives": sum(labels[k] for k in shared),
            "groups_containing_shared_leaves": len(matched),
            "groups_with_only_shared_leaves": sum(ys <= shared for ys in matched),
            "groups_with_shared_and_other_leaves": sum(not ys <= shared for ys in matched),
            "groups_with_any_shared_each": {k: sum(k in ys for ys in matched) for k in sorted(shared)},
            "coarse_representatives": {key: sum(labels[k] for k in leaves) for key, leaves in coarse.items()},
            "groups_containing_coarse_node": groups,
            "groups_spanning_coarse_nodes": sum(all(ys.intersection(leaves) for leaves in coarse.values())
                                                for ys in by_group.values())}


def dataset_summary(name, folder, cfg):
    audits = json.loads((folder / "audit.json").read_bytes())["splits"]
    shared = set(cfg["shared_nominal_leaves"] if name == "ham" else cfg["shared_nominal_leaves"].values())
    coarse = cfg["coarse_partition"][name]
    flat = [leaf for leaves in coarse.values() for leaf in leaves] + cfg["unmatched"][name]
    if len(set(flat)) != len(flat) or not shared <= set(flat):
        raise ValueError("Crosswalk repeats or omits a leaf")
    result = {"seeds": {}}
    inventory = None
    for seed, info in sorted(audits.items()):
        path = folder / f"split_seed{seed}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Changed split {seed}")
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = [r for r in csv.DictReader(stream) if r["representative"] == "1"]
        if {r["label"] for r in rows} != set(flat):
            raise ValueError("Label inventory and crosswalk disagree")
        roles = defaultdict(list)
        assignments = {}
        for r in rows:
            group = r["split_group"]
            if assignments.setdefault(group, r["split"]) != r["split"]:
                raise ValueError("A group crosses partitions")
            roles[r["split"]].append(r)
        current = {(r["image_id"], r["split_group"], r["label"]) for r in rows}
        if inventory is None:
            inventory = current
            result["all"] = counts(rows, shared, coarse)
        if len(current) != len(rows) or current != inventory:
            raise ValueError("Representative inventory differs across seeds")
        result["seeds"][seed] = {role: counts(part, shared, coarse) for role, part in sorted(roles.items())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("ham-splits", "pad-splits", "paper-source", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/ham-pad-crosswalk-v1.json")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).parent / "protocol.json")
    args = parser.parse_args()
    start = time.perf_counter()
    args.output.mkdir(exist_ok=False, parents=True)
    cfg = json.loads(args.config.read_bytes())
    report = {"scope": "metadata only, no test predictions or clinical tree validation",
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()}
    try:
        with ZipFile(args.paper_source / "PMC7479321.zip") as archive:
            source = archive.read("mmc3.csv")
        if hashlib.sha256(source).hexdigest() != cfg["pad_supplement_sha256"]:
            raise ValueError("Published PAD terminology table changed")
        terms = list(csv.DictReader(io.StringIO(source.decode("utf-8-sig"))))
        ids = {r["CONCEPT ID"].split(":")[-1].strip(): r for r in terms}
        matched = {}
        for key, codes in cfg["pad_authors_sctids"].items():
            matched[key] = []
            for code in codes:
                row = ids[code]
                if parse_qs(urlsplit(row["CONCEPT LINK"]).query).get("conceptId1") != [code]:
                    raise ValueError(f"Author concept ID and its URL differ for {key}")
                matched[key].append(row["SNOMED CT CONCEPT"].strip().rsplit("(", 1)[-1].removesuffix(")"))
        report.update(pad_supplement_rows=len(terms), pad_author_concept_types=matched)
        report["ham"] = dataset_summary("ham", args.ham_splits, cfg)
        report["pad"] = dataset_summary("pad", args.pad_splits, cfg)
        policy = json.loads(args.protocol.read_bytes())
        delta = policy["family_failure_probability"] / (policy["max_candidate_policies"] * policy["simultaneous_constraints"])
        target = policy["risk_targets"]["nonroot_set_conditional_error"]
        minima = {}
        for errors in (0, 1, 2):
            n = max(1, errors)
            while upper_bound(errors, n, delta) > target:
                n += 1
            minima[str(errors)] = n
        capacity = {}
        for seed, splits in report["pad"]["seeds"].items():
            s = splits["calibration"]
            capacity[seed] = {}
            for field in ("groups_containing_shared_leaves", "groups_with_only_shared_leaves"):
                n = s[field]
                capacity[seed][field] = {"groups": n, "zero_errors_upper": upper_bound(0,n,delta),
                                        "one_error_upper": upper_bound(1,n,delta)}
        report["nominal_group_capacity"] = {
            "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(), "per_check_delta": delta,
            "count_assumption": "one independent Bernoulli observation per accepted group; fixed policy; prospective best case",
            "nonroot_risk_target": target, "minimum_accepted_groups_by_error_count": minima, "pad_calibration": capacity}
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
