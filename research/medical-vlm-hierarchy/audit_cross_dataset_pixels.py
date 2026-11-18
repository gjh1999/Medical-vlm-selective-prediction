"""Check exact decoded images and dHash candidates between HAM and PAD without using predictions."""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from extract_features import read_inputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("ham-audit","pad-audit","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    args=ap.parse_args()
    start=time.perf_counter()
    ham_info,ham=read_inputs(args.ham_audit)
    pad_info,pad=read_inputs(args.pad_audit)
    if not ham or not pad or "analysis_group" not in ham[0] or "patient_id" not in pad[0]:
        raise ValueError("Expected HAM then PAD manifests")
    refs=[int(r["dhash"],16) for r in ham]
    pairs=[]
    for src in pad:
        value=int(src["dhash"],16)
        for i,x in enumerate(refs):
            distance=(value^x).bit_count()
            if distance<=3:
                pairs.append((ham[i],src,distance))
    exact_bytes={r["sha256"] for r in ham}&{r["sha256"] for r in pad}
    exact_pixels={r["pixel_sha256"] for r in ham}&{r["pixel_sha256"] for r in pad}
    if exact_pixels!={a["pixel_sha256"] for a,b,_ in pairs if a["pixel_sha256"]==b["pixel_sha256"]}:
        raise ValueError("Exact pixel overlap disagrees with the pair scan")
    args.output.mkdir(exist_ok=False,parents=True)
    with (args.output/"pairs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["ham_image","pad_image","distance"])
        w.writerows((a["image_id"],b["image_id"],d) for a,b,d in pairs)
    report={"ham_manifest":ham_info["manifest_sha256"],"pad_manifest":pad_info["manifest_sha256"],
            "ham_images":len(ham),"pad_images":len(pad),"comparisons":len(ham)*len(pad),
            "shared_file_digests":len(exact_bytes),"shared_pixel_digests":len(exact_pixels),
            "candidate_pairs":len(pairs),"distance_histogram":dict(sorted(Counter(d for _,_,d in pairs).items())),
            "ham_images_in_candidates":len({a["image_id"] for a,_,_ in pairs}),
            "pad_images_in_candidates":len({b["image_id"] for _,b,_ in pairs}),
            "candidate_pairs_both_representatives":sum(a["representative"]==b["representative"]=="1" for a,b,_ in pairs),
            "seconds":round(time.perf_counter()-start,3)}
    (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
