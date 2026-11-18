"""Score every retrieved cross-split pair and check equivalence with the first 512+controls."""
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from extract_features import read_inputs
from review_ham_candidates import describe, geometry


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["audit","appearance","shortlist","output"]:
        ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--within-only",action="store_true",help="Complementary pairs that shared a partition in all v1.3 seeds")
    args = ap.parse_args()
    if not 1 <= args.workers <= 8:
        ap.error("workers must be 1..8")
    start = time.perf_counter()
    audit, records = read_inputs(args.audit)
    rows = {r["image_id"]:r for r in records}
    appearance = json.loads((args.appearance/"summary.json").read_bytes())
    small = json.loads((args.shortlist/"summary.json").read_bytes())
    source = args.appearance/"candidates.csv"
    if (checksum(source) != appearance["candidate_sha256"] or small["candidate_sha256"] != checksum(source)
            or small["candidate_source"] != "top_512_cross_split_appearance_pairs"
            or any(r["manifest_sha256"] != audit["manifest_sha256"] for r in (small,appearance))):
        raise ValueError("Candidate manifests differ")
    pairs = {}
    category = "same_roles_candidate" if args.within_only else "cross_split_candidate"
    cross_flag = "0" if args.within_only else "1"
    with source.open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["cross_any_split"] == cross_flag and not r["published"]:
                pairs[r["first"],r["second"]] = category
    previous = {}
    with (args.shortlist/"ranked.csv").open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["first"],r["second"])
            if args.within_only and r["source"] == "cross_split_candidate":
                continue
            if r["source"] == "cross_split_candidate" and key not in pairs:
                raise ValueError("Unexpected previous candidate")
            pairs[key] = r["source"]
            previous[key] = (int(r["inliers"]),int(r["occupied_cells"]))

    args.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    info = {"workers":args.workers,"opencv":cv2.__version__,"numpy":np.__version__,
            "scope":category,
            "manifest_sha256":audit["manifest_sha256"],"candidate_sha256":appearance["candidate_sha256"],
            "shortlist_sha256":checksum(args.shortlist/"ranked.csv"),"pairs":len(pairs)}
    try:
        keys = sorted({i for pair in pairs for i in pair})
        def feature(key):
            row = rows[key]
            variants = describe(Path(row["image_path"]),row["sha256"],cv2.SIFT_create(nfeatures=256))
            # OpenCV's default SIFT returns integer-valued float32 descriptors. Store bytes between calls.
            packed = []
            for xy, descriptors, shape in variants:
                if descriptors is not None:
                    compact = descriptors.astype(np.uint8)
                    if not np.array_equal(compact,descriptors):
                        raise ValueError("Unexpected SIFT descriptor precision")
                    descriptors = compact
                packed.append((xy,descriptors,shape))
            return key,packed
        with ThreadPoolExecutor(args.workers) as pool:
            features = dict(pool.map(feature,keys))
        info.update(unique_images=len(keys),feature_seconds=round(time.perf_counter()-start,3))
        def floating(variant):
            xy,desc,size = variant
            return xy, (desc.astype(np.float32) if desc is not None else None), size
        def score(pair):
            a,b = pair
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            ref = floating(features[a][0])
            n,spread = max(geometry(ref,floating(v),matcher) for v in features[b])
            return [a,b,pairs[pair],n,spread]
        scores = []
        with ThreadPoolExecutor(args.workers) as pool, (args.output/"scores.csv").open("w",encoding="utf-8",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["first","second","source","inliers","occupied_cells"])
            for row in pool.map(score,pairs):
                if (row[0],row[1]) in previous and tuple(row[3:]) != previous[row[0],row[1]]:
                    raise ValueError("Score differs from prior float32 shortlist run")
                writer.writerow(row)
                scores.append(row)
                if len(scores)%5000 == 0:
                    f.flush()
                    print(f"Completed {len(scores)}/{len(pairs)} pairs",flush=True)
        info["shortlist_identical"] = len(previous)
        info["by_source"] = {}
        for key in sorted(set(pairs.values())):
            subset = [r for r in scores if r[2]==key]
            info["by_source"][key] = {"pairs":len(subset),"at_least_8":sum(r[3]>=8 for r in subset),
                                     "at_least_16":sum(r[3]>=16 for r in subset),
                                     "at_least_16_and_4_cells":sum(r[3]>=16 and r[4]>=4 for r in subset),
                                     "max_inliers":max(r[3] for r in subset)}
        info.update(status="complete",scores_sha256=checksum(args.output/"scores.csv"))
    except Exception as exc:
        info.update(status="failed",error=str(exc))
        raise
    finally:
        info["seconds"] = round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(info,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
