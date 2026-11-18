"""Measure how screened HAM image pairs touch released splits and previously used pilot cohorts."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from prepare_ham_scene_splits import read_csv, regroup, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("splits","geometry","review","sensitivity","pilot","balanced","increment","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    args = ap.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    report = {"scope":"sensitivity scenarios on v1.4; no released partition changes"}
    try:
        geometry = json.loads((args.geometry/"summary.json").read_bytes())
        review = json.loads((args.review/"summary.json").read_bytes())
        masking = json.loads((args.sensitivity/"summary.json").read_bytes())
        split_info = json.loads((args.splits/"audit.json").read_bytes())
        if (geometry["status"]!="complete" or review["status"]!="complete" or split_info["split_protocol"]!="v1.4"
                or review["geometry_sha256"]!=geometry["scores_sha256"]
                or sha(args.geometry/"scores.csv")!=geometry["scores_sha256"]
                or sha(args.review/"scores.csv")!=review["scores_sha256"]
                or masking["sensitivity_groups_sha256"]!=sha(args.sensitivity/"sensitivity_groups.csv")):
            raise ValueError("Input evidence chain changed")
        old_mask = {r["split_group"] for r in read_csv(args.sensitivity/"sensitivity_groups.csv") if r["excluded"]=="1"}
        retrieval = defaultdict(Counter)
        with (args.geometry/"scores.csv").open(newline="",encoding="utf-8") as f:
            low,high = set(),set()
            for r in csv.DictReader(f):
                if r["in_pool"]!="1":
                    continue
                rank = int(r["best_word_rank"])
                band = next((name for maximum,name in ((1,"1"),(5,"2-5"),(10,"6-10"),(20,"11-20"))
                             if 0 < rank <= maximum),"appearance_only")
                summary = retrieval[band]
                summary["unique_pairs"] += 1
                summary["in_appearance_retrieval"] += r["old_appearance"]=="1"
                if r["same_v14_group"]=="0" and int(r["inliers"])>=8:
                    edge = r["first"],r["second"]
                    low.add(edge)
                    summary["loose_distinct_group_pairs"] += 1
                    if int(r["inliers"])>=16 and int(r["cells"])>=4:
                        high.add(edge)
                        summary["strong_distinct_group_pairs"] += 1
                        summary["strong_also_in_appearance"] += r["old_appearance"]=="1"
                        summary["strong_crossing_any_seed"] += bool(r["v14_cross_seeds"])
        if sum(v["unique_pairs"] for v in retrieval.values())!=geometry["candidate_pairs"]:
            raise ValueError("Rank-band counts differ from the complete scored pool")
        checked = [r for r in read_csv(args.review/"scores.csv") if r["source"]=="new_screen_hit"]
        if {(r["first"],r["second"]) for r in checked}!=high:
            raise ValueError("Independent review does not cover every strong distinct-group match")
        cohorts = defaultdict(set)
        for path,batch in ((args.pilot,"first"),(args.balanced,"balanced"),(args.increment,"fresh")):
            info = json.loads((path/"summary.json").read_bytes())
            table = path/"selection.csv"
            if info["status"]!="complete" or sha(table)!=info["selection_sha256"]:
                raise ValueError("Changed pilot selection")
            for row in read_csv(table):
                cohorts[f"{batch}_{row['role']}"].add(row["image_id"])
        seed_rows = {}
        for seed,info in split_info["splits"].items():
            path = args.splits/f"split_seed{seed}.csv"
            if sha(path)!=info["sha256"] or sha(path)!=masking["split_sha256"][seed]:
                raise ValueError("Split differs from sensitivity calculation")
            seed_rows[seed] = read_csv(path)
        examples = {r["image_id"]:r for r in seed_rows["17"]}
        tiers = {"root8":low,"root16_cells4":high,
                 "root16_and_orb12_cells4":{(r["first"],r["second"]) for r in checked
                                            if int(r["orb_inliers"])>=12 and int(r["orb_cells"])>=4},
                 "also_strict_registered_pixels":{(r["first"],r["second"]) for r in checked if r["passes"]=="1"}}
        report.update(source_geometry_sha256=geometry["scores_sha256"],source_review_sha256=review["scores_sha256"],
                      cohort_counts={k:len(v) for k,v in cohorts.items()},tiers={},
                      retrieval_by_best_word_rank={k:dict(v) for k,v in retrieval.items()})
        for tier,edges in tiers.items():
            touched = {examples[i]["split_group"] for e in edges for i in e}
            tier_report = {"pairs":len(edges),"groups":len(touched),"also_in_old_dhash_mask":len(touched&old_mask),
                           "pairs_sha256":hashlib.sha256("".join(f"{a},{b}\n" for a,b in sorted(edges)).encode()).hexdigest(),
                           "missing_from_old_dhash_mask":len(touched-old_mask),"seeds":{}}
            for seed,rows in seed_rows.items():
                old = {r["image_id"]:r for r in rows}
                new = {r["image_id"]:r for r in regroup(rows,sorted(edges))}
                roles = {r["split_group"]:r["split"] for r in rows}
                tier_report["seeds"][seed] = {
                    "cross_pairs_now":sum(old[a]["split"]!=old[b]["split"] for a,b in edges),
                    "cross_pairs_after_hypothetical_binding":sum(new[a]["split"]!=new[b]["split"] for a,b in edges),
                    "groups_touched_by_role":dict(Counter(roles[g] for g in touched)),
                    "representatives_touched_by_role":dict(Counter(r["split"] for r in rows
                                                                  if r["representative"]=="1" and r["split_group"] in touched)),
                    "groups_after_hypothetical_binding":len({r["split_group"] for r in new.values()}),
                    "images_moved_if_bound":sum(r["split"]!=old[i]["split"] for i,r in new.items())}
                if seed=="17":
                    histories = {}
                    membership = defaultdict(list)
                    for cohort,ids in cohorts.items():
                        histories[cohort] = {"touched":sum(old[i]["split_group"] in touched for i in ids),
                                             "would_change_role":sum(old[i]["split"]!=new[i]["split"] for i in ids)}
                        for i in ids:
                            membership[new[i]["split_group"]].append(cohort)
                    tier_report["previous_cohorts"] = histories
                    tier_report["hypothetical_groups_with_multiple_previous_rows"] = sum(len(v)>1 for v in membership.values())
                    tier_report["hypothetical_groups_crossing_previous_cohorts"] = sum(len(set(v))>1 for v in membership.values())
            report["tiers"][tier] = tier_report
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
