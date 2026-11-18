"""Measure grayscale nearest-neighbour retrieval of known HAM duplicates under crops and rotations."""
import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from extract_features import read_inputs
from ham_reviewed_pairs import CONFIRMED, UNCERTAIN, load_tables

SIZE = 16
SCALES = (1., .75)
K = 5


def thumbnails(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Image changed since audit")
    with Image.open(io.BytesIO(raw)) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        w, h = gray.size
        out = []
        for scale in SCALES:
            dx, dy = round((1-scale)*w/2), round((1-scale)*h/2)
            crop = gray.crop((dx, dy, w-dx, h-dy)).resize((SIZE,SIZE), Image.Resampling.LANCZOS)
            out.append(np.asarray(crop, dtype=np.float32))
    return out


def describe(images, mask):
    vec = images[...,mask].copy()
    vec -= vec.mean(axis=-1, keepdims=True)
    vec /= np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-6)
    return np.ascontiguousarray(vec)


def best_matches(thumbs, raw_ids):
    axis = np.arange(SIZE) - (SIZE-1)/2
    mask = axis[:,None]**2 + axis[None,:]**2 <= (SIZE/2)**2
    targets = describe(thumbs, mask).reshape(len(thumbs)*len(SCALES), -1)
    matches = {}
    for i in range(0, len(thumbs), 96):
        block = thumbs[i:i+96]
        best = np.full((len(block),len(thumbs)), -np.inf, dtype=np.float32)
        for rotation in range(4):
            rotated = np.rot90(block, rotation, axes=(-2,-1))
            for candidate in (rotated, rotated[...,::-1]):
                queries = describe(candidate, mask).reshape(len(block)*len(SCALES), -1)
                scores = (queries @ targets.T).reshape(len(block),len(SCALES),len(thumbs),len(SCALES))
                np.maximum(best, scores.max(axis=(1,3)), out=best)
        best[raw_ids[i:i+len(block),None] == raw_ids[None,:]] = -np.inf
        top = np.argsort(-best, axis=1, kind="stable")[:,:K]
        for ix, neighbours in enumerate(top):
            for j in neighbours:
                key = tuple(sorted((i+ix, int(j))))
                matches[key] = max(matches.get(key, -1.), float(best[ix,j]))
        if i % (96*16) == 0:
            print(f"Ranked {min(i+96,len(thumbs))}/{len(thumbs)}", flush=True)
    return matches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ["audit", "split-dir", "output"]:
        ap.add_argument("--"+name, type=Path, required=True)
    args = ap.parse_args()
    t = time.perf_counter()
    audit, records = read_inputs(args.audit)
    records.sort(key=lambda r:r["image_id"])
    idx = {r["image_id"]: i for i,r in enumerate(records)}
    split_report = json.loads((args.split_dir/"audit.json").read_bytes())
    if split_report["split_protocol"] != "v1.3":
        raise ValueError("Expected reviewed HAM splits")
    roles = {}
    for seed, detail in split_report["splits"].items():
        file = args.split_dir / f"split_seed{seed}.csv"
        if hashlib.sha256(file.read_bytes()).hexdigest() != detail["sha256"]:
            raise ValueError("Changed split")
        with file.open(encoding="utf-8",newline="") as f:
            split = {r["image_id"]: r["split"] for r in csv.DictReader(f)}
        if split.keys() != idx.keys():
            raise ValueError("Different image inventories")
        roles[seed] = [split[r["image_id"]] for r in records]
    args.output.mkdir(exist_ok=False,parents=True)
    tables = load_tables(args.output,args.split_dir)
    known = {name:{tuple(sorted((idx[a],idx[b]))) for a,b in pairs} for name,pairs in tables.items()}
    old = {tuple(sorted((idx[p["first"]],idx[p["second"]]))) for p in
           json.loads((args.audit/"duplicate_candidates.json").read_bytes())}
    thumbs = np.asarray([thumbnails(Path(r["image_path"]),r["sha256"]) for r in records])
    encode_seconds = time.perf_counter()-t
    matches = best_matches(thumbs, np.asarray([r["lesion_id"] for r in records]))
    output = args.output/"candidates.csv"
    with output.open("w",encoding="utf-8",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["first","second","cosine","in_old_dhash","published","cross_any_split"])
        for (a,b),score in sorted(matches.items(),key=lambda x:(-x[1],x[0])):
            pair = (a,b)
            pub = next((name for name, edges in known.items() if pair in edges),"")
            writer.writerow((records[a]["image_id"], records[b]["image_id"],score,int(pair in old),pub,
                             int(any(v[a] != v[b] for v in roles.values()))))
    report = {"protocol":split_report["split_protocol"],"images":len(records),"size":SIZE,"central_scales":SCALES,
              "orientations":8,"top_k_per_image_distinct_raw_id":K,"pairs":len(matches),
              "overlap_with_dhash":len(matches.keys() & old),"manifest_sha256":audit["manifest_sha256"],
              "candidate_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
              "known_retrieval":{},"cross_by_seed":{},"thumbnail_seconds":round(encode_seconds,3)}
    for name in (CONFIRMED, UNCERTAIN):
        edges = known[name]
        report["known_retrieval"][name] = {"pairs":len(edges),"dhash":len(edges & old),
                                          "appearance":len(edges & matches.keys()),
                                          "either":len(edges & (matches.keys() | old))}
    for seed, split in roles.items():
        pairs = [(a,b) for a,b in matches if split[a] != split[b]]
        report["cross_by_seed"][seed] = {"all":len(pairs),"both_representatives":sum(
            records[a]["representative"] == records[b]["representative"] == "1" for a,b in pairs)}
    report["seconds"] = round(time.perf_counter()-t,3)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
