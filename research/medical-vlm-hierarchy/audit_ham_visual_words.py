"""Retrieve whole-library HAM neighbours from local visual words and score published controls."""
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from audit_ham_texture_controls import file_hash
from ham_reviewed_pairs import CONFIRMED, UNCERTAIN, load_tables
from prepare_ham_texture_cache import load_index, load_pack


BRANCH = 64
SAMPLES_PER_VIEW = 4
DEPTH = 20


def centroids(rows):
    n = min(BRANCH,len(rows))
    if n <= 1:
        return rows.copy()
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 1e-4)
    cv2.setRNGSeed(17)
    _,_,means = cv2.kmeans(rows,n,None,criteria,1,cv2.KMEANS_PP_CENTERS)
    return means


def closest(rows, means):
    if not len(rows):
        return np.empty(0,dtype=np.int32)
    _,assignment = cv2.batchDistance(np.ascontiguousarray(rows),means,cv2.CV_32F,normType=cv2.NORM_L2,K=1)
    return assignment[:,0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("cache","pair-dir","appearance","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--workers",type=int,choices=range(1,9),default=3)
    args = ap.parse_args()
    cache,rows = load_index(args.cache)
    old = json.loads((args.appearance/"summary.json").read_bytes())
    old_path = args.appearance/"candidates.csv"
    if old["candidate_sha256"] != file_hash(old_path) or len(rows) != old["images"]:
        raise ValueError("Previous retrieval inventory unavailable")
    ids = [r["image_id"] for r in rows]
    ix = {key:i for i,key in enumerate(ids)}
    raw = np.asarray([r["lesion_id"] for r in rows])
    args.output.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    report = {"scope":"image-provenance retrieval only, no disease labels",
              "seed":17,"branching":BRANCH,"training_samples_per_view":SAMPLES_PER_VIEW,
              "top_k":DEPTH,"workers":args.workers,"cache_index_sha256":cache["index_sha256"],
              "old_candidate_sha256":old["candidate_sha256"],"stages":{}}
    try:
        cv2.setNumThreads(1)
        old_pairs = set()
        with old_path.open(newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                old_pairs.add(tuple(sorted((ix[r["first"]],ix[r["second"]]))))
        tables = load_tables(args.output,args.pair_dir)
        controls = {key:{tuple(sorted((ix[a],ix[b]))) for a,b in pairs} for key,pairs in tables.items()}
        def sample(r):
            digest = hashlib.sha256(("seed17|"+r["image_id"]).encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8],"little"))
            views = load_pack(args.cache,r)
            return np.concatenate([d[rng.choice(len(d),min(len(d),SAMPLES_PER_VIEW),replace=False)]
                                   for _,d,_ in views])
        with ThreadPoolExecutor(args.workers) as pool:
            samples = np.concatenate(list(pool.map(sample,rows)))
        report["stages"]["sample_seconds"] = round(time.perf_counter()-start,3)
        now = time.perf_counter()
        coarse = centroids(samples)
        assignments = closest(samples,coarse)
        sub = [centroids(samples[assignments==i]) for i in range(len(coarse))]
        report["actual_visual_words"] = sum(map(len,sub))
        np.savez(args.output/"vocabulary.npz",coarse=coarse,**{f"sub{i}":v for i,v in enumerate(sub)})
        report["stages"]["vocabulary_seconds"] = round(time.perf_counter()-now,3)
        now = time.perf_counter()
        def histogram(r):
            descriptors = np.concatenate([d for _,d,_ in load_pack(args.cache,r)])
            parents = closest(descriptors,coarse)
            counts = np.zeros(BRANCH**2,dtype=np.int32)
            for parent in np.unique(parents):
                word = parent*BRANCH+closest(descriptors[parents==parent],sub[parent])
                word_ids,word_counts = np.unique(word,return_counts=True)
                counts[word_ids] = word_counts
            if int(counts.sum()) != len(descriptors):
                raise ValueError("Visual word counts differ from descriptor count")
            return counts
        hist = np.empty((len(rows),BRANCH**2),dtype=np.float32)
        with ThreadPoolExecutor(args.workers) as pool:
            for i,values in enumerate(pool.map(histogram,rows)):
                hist[i] = values
                if (i+1)%2000 == 0:
                    print(f"Quantized {i+1}/{len(rows)} images",flush=True)
        df = np.count_nonzero(hist,axis=0)
        np.sqrt(hist,out=hist)
        hist *= np.log((len(rows)+1)/(df+1))
        hist /= np.maximum(np.linalg.norm(hist,axis=1,keepdims=True),1e-12)
        report["stages"]["histogram_seconds"] = round(time.perf_counter()-now,3)
        now = time.perf_counter()
        ranks = np.empty((len(rows),DEPTH),dtype=np.int32)
        values = np.empty_like(ranks,dtype=np.float32)
        for begin in range(0,len(rows),96):
            end = min(begin+96,len(rows))
            similarity = hist[begin:end] @ hist.T
            similarity[raw[begin:end,None]==raw[None,:]] = -np.inf
            top = np.argsort(-similarity,axis=1,kind="stable")[:,:DEPTH]
            ranks[begin:end] = top
            values[begin:end] = np.take_along_axis(similarity,top,axis=1)
        pairs = {}
        for a,neighbours in enumerate(ranks):
            for rank,b in enumerate(neighbours,1):
                key = tuple(sorted((a,int(b))))
                prev = pairs.get(key)
                if prev is None or rank < prev[0]:
                    pairs[key] = (rank,float(values[a,rank-1]))
        output = args.output/"candidates.csv"
        with output.open("w",encoding="utf-8",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["first","second","best_rank","cosine","in_old_appearance"])
            for (a,b),(rank,val) in sorted(pairs.items()):
                writer.writerow((ids[a],ids[b],rank,val,int((a,b) in old_pairs)))
        report["stages"]["retrieval_and_write_seconds"] = round(time.perf_counter()-now,3)
        report["controls_by_depth"] = {}
        for depth in (1,5,10,20):
            chosen = {k for k,(r,_) in pairs.items() if r<=depth}
            report["controls_by_depth"][str(depth)] = {
                "unique_pairs":len(chosen),"overlap_old":len(chosen & old_pairs),
                "confirmed":len(chosen & controls[CONFIRMED]),"uncertain":len(chosen & controls[UNCERTAIN]),
                "confirmed_union_with_old":len((chosen|old_pairs)&controls[CONFIRMED])}
        report.update(status="complete",images=len(rows),training_descriptors=len(samples),
                      candidate_sha256=file_hash(output),vocabulary_sha256=file_hash(args.output/"vocabulary.npz"))
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
