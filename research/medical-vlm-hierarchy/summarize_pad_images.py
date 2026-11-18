"""Check PAD decoded hashes across the three patient-group splits and against a brute-force search."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from extract_features import read_inputs
from prepare_pad_images import HASHES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["audit","split-dir","mirror","output"]:
        ap.add_argument("--"+key,type=Path,required=True)
    args = ap.parse_args()
    start=time.perf_counter()
    audit, rows=read_inputs(args.audit)
    provenance=json.loads((args.mirror/"agreement.json").read_bytes())
    if (provenance["status"]!="mirror_agreement_verified" or provenance["byte_identical"]!=len(rows)
            or provenance["metadata_sha256"]!=HASHES["metadata.csv"]):
        raise ValueError("Missing verified mirror inventory")
    if not args.mirror.joinpath("images").resolve().samefile(audit["images_root"]):
        raise ValueError("Image audit refers to another directory")
    original=args.split_dir/"metadata.csv"
    if hashlib.sha256(original.read_bytes()).hexdigest()!=HASHES["metadata.csv"]:
        raise ValueError("Unexpected official metadata")
    def read(path):
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    meta={r["img_id"]:r for r in read(original)}
    by_id={r["image_id"]:r for r in rows}
    if by_id.keys()!=meta.keys():
        raise ValueError("Image IDs differ from official metadata")
    pairs_path=args.audit/"duplicate_candidates.json"
    pairs=json.loads(pairs_path.read_bytes())
    pairs_by_id={tuple(sorted([r["first"],r["second"]])):r for r in pairs}
    hashes=[int(r["dhash"],16) for r in rows]
    expected={(rows[j]["image_id"],row["image_id"]):(h^hashes[j]).bit_count()
              for i,(row,h) in enumerate(zip(rows,hashes)) for j in range(i) if (h^hashes[j]).bit_count()<=3}
    if {tuple(sorted(k)):v for k,v in expected.items()}!={k:r["hamming"] for k,r in pairs_by_id.items()}:
        raise ValueError("Indexed candidate pairs differ from exhaustive radius-3 search")
    splits=json.loads((args.split_dir/"audit.json").read_bytes())
    report={"source":"agreement between two distribution mirrors; official ZIP content unverified",
            "decoded":len(rows),"patient_groups":len({r["split_group"] for r in rows}),
            "manifest_sha256":audit["manifest_sha256"],"candidate_sha256":hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
            "all_unordered_pairs_checked":len(rows)*(len(rows)-1)//2,"dhash_pairs":len(pairs),
            "same_file_bytes":sum(p["same_bytes"] for p in pairs),"same_pixels":sum(p["same_decoded_pixels"] for p in pairs),
            "different_patients":0,"different_split_groups":0,"different_diagnoses":0,
            "width_range":[min(int(r["width"]) for r in rows),max(int(r["width"]) for r in rows)],
            "height_range":[min(int(r["height"]) for r in rows),max(int(r["height"]) for r in rows)],
            "class_images":dict(Counter(r["label"] for r in rows)),
            "pillow":audit["pillow_version"],"decode_seconds":audit["seconds"],"splits":{}}
    for a,b in pairs_by_id:
        report["different_patients"]+=int(meta[a]["patient_id"]!=meta[b]["patient_id"])
        report["different_split_groups"]+=int(by_id[a]["split_group"]!=by_id[b]["split_group"])
        report["different_diagnoses"]+=int(meta[a]["diagnostic"]!=meta[b]["diagnostic"])
    for seed,entry in splits["splits"].items():
        path=args.split_dir/f"split_seed{seed}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError("PAD split checksum mismatch")
        by={r["image_id"]:r for r in read(path)}
        if by.keys()!=by_id.keys() or any(by[k]["split_group"]!=by_id[k]["split_group"]
                                          or by[k]["label"]!=meta[k]["diagnostic"] for k in by):
            raise ValueError("PAD split case inventory mismatch")
        cross=[(a,b) for a,b in pairs_by_id if by[a]["split"]!=by[b]["split"]]
        report["splits"][seed]={"split_sha256":entry["sha256"],"cross_candidates":len(cross),
                               "cross_both_representatives":sum(by[a]["representative"]==by[b]["representative"]=="1" for a,b in cross),
                               "cross_exact_pixels":sum(pairs_by_id[a,b]["same_decoded_pixels"] for a,b in cross)}
    report["summary_seconds"]=round(time.perf_counter()-start,3)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
