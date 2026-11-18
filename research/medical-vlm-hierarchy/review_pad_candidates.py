"""Geometric triage of all PAD dHash pairs and HAM-to-PAD candidates, using PAD exact pairs as controls."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from extract_features import read_inputs
from review_ham_candidates import describe, geometry


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("pad-audit","ham-audit","cross","output"):
        ap.add_argument("--"+key,type=Path,required=True)
    args=ap.parse_args()
    start=time.perf_counter()
    pad_info,pad=read_inputs(args.pad_audit)
    ham_info,ham=read_inputs(args.ham_audit)
    cross=json.loads((args.cross/"summary.json").read_bytes())
    if (cross["ham_manifest"]!=ham_info["manifest_sha256"] or cross["pad_manifest"]!=pad_info["manifest_sha256"]):
        raise ValueError("Cross-dataset input changed")
    # ID namespace is disjoint in these two releases; reject ambiguities explicitly.
    rows={r["image_id"]:r for r in pad+ham}
    if len(rows)!=len(pad)+len(ham):
        raise ValueError("Ambiguous case ID between datasets")
    cases=[]
    for r in json.loads((args.pad_audit/"duplicate_candidates.json").read_bytes()):
        a,b=r["first"],r["second"]
        if r["same_decoded_pixels"]:
            category="pad_exact_control"
        else:
            category="pad_cross_group" if rows[a]["split_group"]!=rows[b]["split_group"] else "pad_within_group"
        cases.append((category,a,b))
    with (args.cross/"pairs.csv").open(encoding="utf-8",newline="") as f:
        held=[(r["ham_image"],r["pad_image"]) for r in csv.DictReader(f)]
    if len(held)!=cross["candidate_pairs"] or any(
            (int(rows[a]["dhash"],16)^int(rows[b]["dhash"],16)).bit_count()>3 for a,b in held):
        raise ValueError("Invalid cross-dataset pairs")
    cases.extend(("ham_pad",a,b) for a,b in held)
    cv2.setNumThreads(1)
    sift=cv2.SIFT_create(nfeatures=256)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    keys={k for _,a,b in cases for k in (a,b)}
    desc={k:describe(Path(rows[k]["image_path"]),rows[k]["sha256"],sift) for k in sorted(keys)}
    result=[]
    for cat,a,b in cases:
        n,spread=max(geometry(desc[a][0],v,matcher) for v in desc[b])
        result.append((cat,n,spread,a,b))
    args.output.mkdir(exist_ok=False,parents=True)
    path=args.output/"ranked.csv"
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["source","inliers","occupied_cells","first","second"])
        writer.writerows(sorted(result,key=lambda v:(v[0],-v[1],-v[2],v[3],v[4])))
    report={"opencv":cv2.__version__,"loaded_images":len(keys),"pad_manifest":pad_info["manifest_sha256"],
            "ham_manifest":ham_info["manifest_sha256"],"scores_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "by_source":{}}
    for name in sorted({v[0] for v in result}):
        scores=[v[1] for v in result if v[0]==name]
        report["by_source"][name]={"pairs":len(scores),"max_inliers":max(scores),"median_inliers":float(np.median(scores)),
                                   "at_least_8":sum(n>=8 for n in scores),"at_least_16":sum(n>=16 for n in scores)}
    report["seconds"]=round(time.perf_counter()-start,3)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
