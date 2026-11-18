"""Check whether higher-detail local texture improves scene correspondence on published HAM controls."""
import argparse
import csv
import hashlib
import io
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from extract_features import read_inputs
from review_ham_candidates import geometry


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def root_sift(raw):
    values = raw.astype(np.float32,copy=False)
    return np.sqrt(values/np.maximum(values.sum(axis=1,keepdims=True),1))


def texture(row, packed=False):
    data = Path(row["image_path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Benchmark image changed")
    with Image.open(io.BytesIO(data)) as im:
        gray = ImageOps.exif_transpose(im).convert("L")
        gray.thumbnail((640,640),Image.Resampling.LANCZOS)
        pixels = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(np.asarray(gray))
    sift = cv2.SIFT_create(nfeatures=1200,contrastThreshold=.02)
    h,w = pixels.shape
    mask = np.zeros_like(pixels)
    cv2.ellipse(mask,(w//2,h//2),(int(.46*w),int(.46*h)),0,0,360,255,-1)
    variants = []
    for image in (pixels,cv2.flip(pixels,1)):
        points, desc = sift.detectAndCompute(image, mask)
        xy = np.float32([p.pt for p in points]).reshape(-1,2)
        if desc is None:
            desc = np.empty((0,128),np.uint8 if packed else np.float32)
        elif packed:
            compact = desc.astype(np.uint8)
            if not np.array_equal(compact,desc):
                raise ValueError("Unexpected non-integral SIFT descriptor")
            desc = compact
        else:
            desc = root_sift(desc)
        variants.append((xy,desc,(w,h)))
    return variants


def score(left, right):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    return max(geometry(left[0],view,matcher) for view in right)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("audit","baseline","output"):
        ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--workers",type=int,choices=range(1,9),default=3)
    args = ap.parse_args()
    start = time.perf_counter()
    audit, rows = read_inputs(args.audit)
    old = json.loads((args.baseline/"summary.json").read_bytes())
    scores_path = args.baseline/"scores.csv"
    if (old["status"] != "complete" or old["scores_sha256"] != file_hash(scores_path)
            or old["manifest_sha256"] != audit["manifest_sha256"]):
        raise ValueError("Baseline geometry and manifest differ")
    indexed = {r["image_id"]:r for r in rows}
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope":"method development on known pairs and a sample of unadjudicated candidates",
              "baseline_scores_sha256":old["scores_sha256"],"manifest_sha256":audit["manifest_sha256"],
              "opencv":cv2.__version__,"numpy":np.__version__,"workers":args.workers}
    try:
        chosen, decoys = [], []
        with scores_path.open(newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                pair = (r["first"],r["second"],int(r["inliers"]),int(r["occupied_cells"]))
                if r["source"].startswith("published"):
                    chosen.append((r["source"],*pair))
                elif r["source"] == "cross_split_candidate":
                    if pair[2] >= 8:
                        chosen.append(("previous_scene_candidate",*pair))
                    elif pair[2] <= 3:
                        decoys.append(pair)
        sampled = random.Random(17).sample(sorted(decoys),200)
        chosen.extend(("low_baseline_candidate",*p) for p in sampled)
        keys = sorted({k for _, a,b,_,_ in chosen for k in (a,b)})
        with ThreadPoolExecutor(args.workers) as pool:
            features = dict(zip(keys,pool.map(texture,[indexed[k] for k in keys])))
            def compare(item):
                kind,a,b,n,c = item
                return (kind,a,b,n,c,*score(features[a],features[b]))
            records = list(pool.map(compare,chosen))
        with (args.output/"pairs.csv").open("w",newline="",encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source","first","second","old_inliers","old_cells","new_inliers","new_cells"])
            writer.writerows(records)
        categories = {}
        for kind in sorted({r[0] for r in records}):
            selected = [r for r in records if r[0]==kind]
            categories[kind] = {"pairs":len(selected)}
            for label,i in (("old",3),("new",5)):
                categories[kind][label] = {
                    "at_least_8":sum(r[i]>=8 for r in selected),
                    "at_least_8_and_4_cells":sum(r[i]>=8 and r[i+1]>=4 for r in selected),
                    "at_least_16_and_4_cells":sum(r[i]>=16 and r[i+1]>=4 for r in selected),
                    "max_inliers":max(r[i] for r in selected)}
        report.update(status="complete",unique_images=len(keys),categories=categories,
                      pairs_sha256=file_hash(args.output/"pairs.csv"))
    except Exception as error:
        report.update(status="failed",error=str(error))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
