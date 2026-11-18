"""Prepare reusable, byte-exact local descriptors for a full HAM image-provenance audit."""
import argparse
import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from audit_ham_texture_controls import file_hash, root_sift, texture
from extract_features import read_inputs


def load_pack(folder, item):
    content = (folder / "features" / item["file"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != item["file_sha256"]:
        raise ValueError("Local descriptor cache changed")
    with np.load(io.BytesIO(content),allow_pickle=False) as pack:
        size = tuple(int(v) for v in pack["size"])
        return [(pack[f"xy{i}"],root_sift(pack[f"d{i}"]),size) for i in range(2)]


def load_index(folder):
    info = json.loads((folder/"summary.json").read_bytes())
    path = folder/"index.csv"
    if info["status"] != "complete" or info["index_sha256"] != file_hash(path):
        raise ValueError("Incomplete or changed texture index")
    with path.open(encoding="utf-8",newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != info["images"] or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Cached features and inventory differ")
    return info,rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--workers",type=int,choices=range(1,9),default=3)
    args = ap.parse_args()
    audit, rows = read_inputs(args.audit)
    rows.sort(key=lambda r:r["image_id"])
    args.output.mkdir(parents=True,exist_ok=False)
    features = args.output/"features"
    features.mkdir()
    cv2.setNumThreads(1)
    start = time.perf_counter()
    report = {"status":"running","images":len(rows),"source_manifest_sha256":audit["manifest_sha256"],
              "opencv":cv2.__version__,"numpy":np.__version__,"workers":args.workers,
              "representation":"unmodified uint8 SIFT for native and mirrored frames with float32 coordinates"}
    try:
        def save(entry):
            i,row = entry
            normal,flipped = texture(row,packed=True)
            name = f"{i:05d}.npz"
            dest = features/name
            np.savez(dest,xy0=normal[0],d0=normal[1],xy1=flipped[0],d1=flipped[1],size=normal[2])
            return [row["image_id"],row["lesion_id"],row["sha256"],name,file_hash(dest),len(normal[0]),len(flipped[0])]
        path = args.output/"index.csv"
        size = 0
        with ThreadPoolExecutor(args.workers) as pool, path.open("w",encoding="utf-8",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id","lesion_id","image_sha256","file","file_sha256","native_points","mirrored_points"])
            for count, result in enumerate(pool.map(save,enumerate(rows)),1):
                writer.writerow(result)
                size += sum(result[-2:])
                if count % 1000 == 0:
                    print(f"Cached {count}/{len(rows)}",flush=True)
                    f.flush()
        report.update(status="complete",total_descriptors=size,index_sha256=file_hash(path))
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
