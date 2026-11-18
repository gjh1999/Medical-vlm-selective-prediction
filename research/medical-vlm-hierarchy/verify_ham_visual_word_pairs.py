"""Check visual-word candidates with local geometry and count crossings of the existing HAM split."""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2

from audit_ham_texture_controls import file_hash, score
from prepare_ham_texture_cache import load_index, load_pack


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("cache","words","appearance","controls","splits","output"):
        ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--max-rank",type=int,choices=(1,5,10,20),required=True)
    ap.add_argument("--include-appearance",action="store_true",help="Also visit every prior thumbnail pair")
    ap.add_argument("--reuse",type=Path,help="Reuse completed identical geometry on a subset of these candidates")
    ap.add_argument("--workers",type=int,choices=range(1,9),default=3)
    args = ap.parse_args()
    start = time.perf_counter()
    cache,rows = load_index(args.cache)
    words = json.loads((args.words/"summary.json").read_bytes())
    controls = json.loads((args.controls/"summary.json").read_bytes())
    appearance = json.loads((args.appearance/"summary.json").read_bytes())
    if (words["status"]!="complete" or cache["index_sha256"]!=words["cache_index_sha256"]
            or file_hash(args.words/"candidates.csv")!=words["candidate_sha256"]
            or file_hash(args.controls/"pairs.csv")!=controls["pairs_sha256"]):
        raise ValueError("Changed retrieval or control data")
    if words["old_candidate_sha256"]!=appearance["candidate_sha256"] or file_hash(args.appearance/"candidates.csv")!=appearance["candidate_sha256"]:
        raise ValueError("Previous appearance rankings changed")
    index = {r["image_id"]:r for r in rows}
    pairs = {}
    with (args.words/"candidates.csv").open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rank = int(row["best_rank"])
            if rank <= args.max_rank:
                pairs[(row["first"],row["second"])] = [rank,int(row["in_old_appearance"])]
    if args.include_appearance:
        with (args.appearance/"candidates.csv").open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pair = tuple(sorted((row["first"],row["second"])))
                pairs.setdefault(pair,[0,1])
    expected = {}
    with (args.controls/"pairs.csv").open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pair = tuple(sorted((row["first"],row["second"])))
            expected[pair] = (int(row["new_inliers"]),int(row["new_cells"]),row["source"])
    assignments = {}
    split_audit = json.loads((args.splits/"audit.json").read_bytes())
    if split_audit["split_protocol"]!="v1.4":
        raise ValueError("Expected released v1.4 split")
    groups = {}
    for seed,entry in split_audit["splits"].items():
        path = args.splits/f"split_seed{seed}.csv"
        if file_hash(path)!=entry["sha256"]:
            raise ValueError("Changed split")
        with path.open(newline="",encoding="utf-8") as f:
            members = list(csv.DictReader(f))
        roles = {r["image_id"]:r["split"] for r in members}
        current = {r["image_id"]:r["split_group"] for r in members}
        if set(roles)!=set(index) or (groups and current!=groups):
            raise ValueError("Benchmark inventories or group definitions differ")
        assignments[seed] = roles
        groups = current
    # Add out-of-pool controls to check the packed cache without inflating candidate/retrieval counts.
    candidates = set(pairs)
    for pair in expected:
        pairs.setdefault(pair,[0,0])
    previous = {}
    if args.reuse:
        prior = json.loads((args.reuse/"summary.json").read_bytes())
        if (prior["status"]!="complete" or prior["cache_sha256"]!=cache["index_sha256"]
                or file_hash(args.reuse/"scores.csv")!=prior["scores_sha256"]):
            raise ValueError("Previous geometry cache differs")
        with (args.reuse/"scores.csv").open(newline="",encoding="utf-8") as f:
            for item in csv.DictReader(f):
                pair = item["first"],item["second"]
                if pair in pairs:
                    previous[pair] = int(item["inliers"]),int(item["cells"])
    by_left = defaultdict(list)
    for a,b in sorted(pairs):
        by_left[a].append(b)
    args.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    report = {"cache_sha256":cache["index_sha256"],"word_candidates_sha256":words["candidate_sha256"],
              "max_rank":args.max_rank,"include_appearance":args.include_appearance,"candidate_pairs":len(candidates),
              "extra_control_pairs":len(pairs)-len(candidates),"reused":len(previous),"workers":args.workers}
    try:
        @lru_cache(maxsize=128)
        def load(key):
            return load_pack(args.cache,index[key])
        def compare(left):
            items = []
            for right in by_left[left]:
                pair = left,right
                if pair in previous:
                    n,cells = previous[pair]
                else:
                    n,cells = score(load(left),load(right))
                reference = expected.get(pair)
                if reference and (n,cells)!=reference[:2]:
                    raise ValueError("Packed feature score differs from uncached control")
                crossing = ",".join(seed for seed,roles in assignments.items() if roles[left]!=roles[right])
                items.append((left,right,*pairs[pair],int(pair in candidates),reference[2] if reference else "",
                              n,cells,int(groups[left]==groups[right]),crossing))
            return items
        profile = defaultdict(Counter)
        total = 0
        with ThreadPoolExecutor(args.workers) as pool, (args.output/"scores.csv").open("w",encoding="utf-8",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["first","second","best_word_rank","old_appearance","in_pool","control","inliers","cells","same_v14_group","v14_cross_seeds"])
            for chunk in pool.map(compare,by_left):
                writer.writerows(chunk)
                for a,b,rank,old,in_pool,kind,n,cells,same,cross in chunk:
                    if not in_pool:
                        continue
                    section = "within_group" if same else ("cross_partition" if cross else "same_partitions_distinct_group")
                    item = profile[section]
                    item["pairs"] += 1
                    item["at_least_8"] += n>=8
                    item["at_least_16_and_4_cells"] += (n>=16 and cells>=4)
                    item["at_least_32_and_4_cells"] += (n>=32 and cells>=4)
                total += len(chunk)
                if total//5000 != (total-len(chunk))//5000:
                    f.flush()
                    print(f"Scored {total}/{len(pairs)}",flush=True)
        report.update(status="complete",summary=dict(profile),control_scores_verified=len(expected),
                      scores_sha256=file_hash(args.output/"scores.csv"))
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
