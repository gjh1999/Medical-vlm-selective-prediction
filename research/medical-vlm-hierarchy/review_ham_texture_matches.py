"""Cross-check RootSIFT-screened image pairs with ORB and registered pixel detail."""
import argparse
import csv
import hashlib
import io
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from audit_ham_texture_controls import file_hash
from extract_features import read_inputs
from review_ham_candidates import panels


def extract(row):
    data = Path(row["image_path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError("Image changed after audit")
    with Image.open(io.BytesIO(data)) as im:
        gray = ImageOps.exif_transpose(im).convert("L")
        gray.thumbnail((640,640),Image.Resampling.LANCZOS)
        initial = np.asarray(gray)
    result = []
    orb = cv2.ORB_create(nfeatures=2200,edgeThreshold=15,fastThreshold=7)
    for frame in (initial,cv2.flip(initial,1)):
        h,w = frame.shape
        mask = np.zeros((h,w),dtype=np.uint8)
        cv2.ellipse(mask,(w//2,h//2),(int(.44*w),int(.44*h)),0,0,360,255,-1)
        points,desc = orb.detectAndCompute(frame,mask)
        xy = np.float32([p.pt for p in points]).reshape(-1,2)
        result.append((frame,mask,xy,desc))
    return result


def agreement(left,right):
    gray,lmask,xy,desc = left
    other,rmask,uv,desc2 = right
    empty = (0,0,0.,0.,0.)
    if min(len(xy),len(uv))<4:
        return empty
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    def ratio(a,b):
        return {m.queryIdx:m.trainIdx for m,n in bf.knnMatch(a,b,k=2) if m.distance<.8*n.distance}
    forward = ratio(desc,desc2)
    reverse = ratio(desc2,desc)
    pairs = [(i,j) for i,j in forward.items() if reverse.get(j)==i]
    if len(pairs)<4:
        return empty
    a,b = np.asarray(pairs).T
    src,dst = xy[a],uv[b]
    cv2.setRNGSeed(17)
    transform,mask = cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3)
    if transform is None:
        return empty
    keep = mask.ravel().astype(bool)
    count = int(keep.sum())
    def covered(points,shape):
        wh = np.array(shape[::-1])
        bins = np.clip((4*points[keep]/wh).astype(int),0,3)
        return len(np.unique(bins,axis=0))
    cells = min(covered(src,gray.shape),covered(dst,other.shape))
    h,w = other.shape
    warped_mask = cv2.warpAffine(lmask,transform,(w,h),flags=cv2.INTER_NEAREST)
    use = cv2.erode(np.minimum(warped_mask,rmask),np.ones((7,7),np.uint8)).astype(bool)
    overlap = use.sum()/max(np.count_nonzero(rmask),np.count_nonzero(warped_mask),1)
    if use.sum()<256:
        return count,cells,float(overlap),0.,0.
    warped = cv2.warpAffine(gray.astype(np.float32),transform,(w,h))
    target = other.astype(np.float32)
    def correlate(a,b):
        x,y = a[use].astype(np.float64), b[use].astype(np.float64)
        x-=x.mean()
        y-=y.mean()
        return float(x @ y / max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))
    intensity = correlate(warped,target)
    detail = correlate(warped-cv2.GaussianBlur(warped,(0,0),3),target-cv2.GaussianBlur(target,(0,0),3))
    return count,cells,float(overlap),intensity,detail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("audit","controls","output"):
        ap.add_argument("--"+arg,type=Path,required=True)
    ap.add_argument("--screen",type=Path,help="Completed visual-word geometry output; add distinct-group screen hits")
    ap.add_argument("--workers",type=int,choices=range(1,9),default=3)
    args = ap.parse_args()
    audit,rows = read_inputs(args.audit)
    indexed = {r["image_id"]:r for r in rows}
    ctrl = json.loads((args.controls/"summary.json").read_bytes())
    if ctrl["status"]!="complete" or file_hash(args.controls/"pairs.csv")!=ctrl["pairs_sha256"]:
        raise ValueError("Changed controls")
    candidates = {}
    with (args.controls/"pairs.csv").open(encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            candidates[r["first"],r["second"]] = (r["source"],int(r["new_inliers"]),int(r["new_cells"]),"")
    if args.screen:
        info = json.loads((args.screen/"summary.json").read_bytes())
        if info["status"]!="complete" or file_hash(args.screen/"scores.csv")!=info["scores_sha256"]:
            raise ValueError("Changed candidate screen")
        with (args.screen/"scores.csv").open(encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                n,c = int(r["inliers"]),int(r["cells"])
                if r["in_pool"]=="1" and r["same_v14_group"]=="0" and n>=16 and c>=4:
                    candidates[r["first"],r["second"]] = ("new_screen_hit",n,c,r["v14_cross_seeds"])
    args.output.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    cv2.setNumThreads(1)
    report = {"scope":"technical scene verification, no patient identity or diagnosis inferred",
              "audit_manifest_sha256":audit["manifest_sha256"],"control_sha256":ctrl["pairs_sha256"],
              "criteria":"mutual ORB ratio 0.8, >=12 affine inliers and >=4 cells, masked overlap >=0.20, registered detail correlation >=0.30"}
    if args.screen:
        report["geometry_sha256"]=info["scores_sha256"]
    try:
        @lru_cache(maxsize=48)
        def views(key):
            return extract(indexed[key])
        def evaluate(pair):
            a,b = pair
            src = views(a)[0]
            scores = [agreement(src,v) for v in views(b)]
            def passes(result):
                n,c,area,_,detail=result
                return n>=12 and c>=4 and area>=.20 and detail>=.30
            best = max(scores,key=lambda s:(passes(s),s[0],s[1],s[4]))
            return [*pair,*candidates[pair],*best,int(passes(best))]
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(evaluate,sorted(candidates)))
        summary = defaultdict(Counter)
        for r in results:
            item = summary[r[2]]
            item["pairs"] += 1
            item["orb12_cells4"] += (r[6]>=12 and r[7]>=4)
            item["all_pixel_and_geometry_criteria"] += r[-1]
        with (args.output/"scores.csv").open("w",encoding="utf-8",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["first","second","source","root_inliers","root_cells","v14_cross_seeds","orb_inliers",
                             "orb_cells","overlap","intensity_correlation","detail_correlation","passes"])
            writer.writerows(results)
        hits = [r for r in results if r[2]=="new_screen_hit"]
        if hits:
            hits.sort(key=lambda r:(-r[-1],-r[6],-r[3],r[0],r[1]))
            panels([("screen",(r[0],r[1])) for r in hits[:60]],indexed,args.output)
        report.update(status="complete",categories=dict(summary),review_panels_pairs=min(len(hits),60),
                      scores_sha256=file_hash(args.output/"scores.csv"))
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
