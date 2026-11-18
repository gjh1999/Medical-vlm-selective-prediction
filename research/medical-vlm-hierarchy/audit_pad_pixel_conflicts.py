"""Quantify deterministic image-only ambiguity within PAD, with and without duplicated pixel groups."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from extract_features import read_inputs


def pixel_summary(rows):
    buckets=defaultdict(list)
    for r in rows:
        buckets[r["pixel_sha256"]].append(r)
    duplicate=[group for group in buckets.values() if len(group)>1]
    conflicts=[g for g in duplicate if len({r["label"] for r in g})>1]
    minimum=sum(len(g)-max(Counter(r["label"] for r in g).values()) for g in conflicts)
    label_pairs=Counter()
    for g in conflicts:
        for a,b in combinations(g,2):
            if a["label"]!=b["label"]:
                label_pairs["/".join(sorted((a["label"],b["label"])))] += 1
    return {"rows":len(rows),"distinct_pixels":len(buckets),"duplicate_pixel_clusters":len(duplicate),
            "conflicting_pixel_clusters":len(conflicts),"conflicting_rows":sum(map(len,conflicts)),
            "conflicting_pairs":sum(label_pairs.values()),"different_label_pairs":dict(sorted(label_pairs.items())),
            "minimum_image_only_errors":minimum,
            "deterministic_image_only_accuracy_ceiling":1-minimum/len(rows) if rows else None,
            "affected_conservative_groups":len({r["split_group"] for g in conflicts for r in g})}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ["audit","split-dir","output"]:
        ap.add_argument("--"+k,type=Path,required=True)
    args=ap.parse_args()
    start=time.perf_counter()
    audit,rows=read_inputs(args.audit)
    buckets=defaultdict(list)
    for r in rows:
        buckets[r["pixel_sha256"]].append(r)
    duplicates=[g for g in buckets.values() if len(g)>1]
    conflicts=[g for g in duplicates if len({r["label"] for r in g})>1]
    affected={r["split_group"] for g in conflicts for r in g}
    reps=[r for r in rows if r["representative"]=="1"]
    core={"NEV","BCC","MEL"}
    report={"manifest_sha256":audit["manifest_sha256"],"unit":"deterministic function of decoded RGB pixels",
            "all_views":pixel_summary(rows),"representatives":pixel_summary(reps),
            "shared_leaf_candidates_representatives":pixel_summary([r for r in reps if r["label"] in core]),
            "all_duplicate_clusters_with_multiple_patient_ids":sum(len({r["patient_id"] for r in g})>1 for g in duplicates),
            "all_duplicate_clusters_with_multiple_lesion_keys":sum(len({r["lesion_id"] for r in g})>1 for g in duplicates),
            "affected_groups":len(affected),"splits":{}}
    details=json.loads((args.split_dir/"audit.json").read_bytes())
    for seed,entry in details["splits"].items():
        path=args.split_dir/f"split_seed{seed}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError("Changed PAD split")
        with path.open(newline="",encoding="utf-8") as f:
            roles={r["image_id"]:r for r in csv.DictReader(f)}
        if roles.keys()!={r["image_id"] for r in rows} or any(
                any(roles[r["image_id"]][k]!=r[k] for k in ("split_group","representative","label")) for r in rows):
            raise ValueError("PAD identity inventory mismatch")
        report["splits"][seed]={}
        for role in ("train","dev","calibration","test"):
            subset=[r for r in reps if roles[r["image_id"]]["split"]==role]
            keep=[r for r in subset if r["split_group"] not in affected]
            report["splits"][seed][role]={"representatives":len(subset),"retained":len(keep),
                                         "groups":len({r["split_group"] for r in subset}),
                                         "retained_groups":len({r["split_group"] for r in keep}),
                                         "original_class_counts":dict(Counter(r["label"] for r in subset)),
                                         "retained_class_counts":dict(Counter(r["label"] for r in keep))}
    args.output.mkdir(exist_ok=False,parents=True)
    file=args.output/"sensitivity_groups.csv"
    with file.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["split_group","excluded"])
        writer.writerows((g,int(g in affected)) for g in sorted({r["split_group"] for r in rows}))
    report["groups_sha256"]=hashlib.sha256(file.read_bytes()).hexdigest()
    report["seconds"]=round(time.perf_counter()-start,3)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
