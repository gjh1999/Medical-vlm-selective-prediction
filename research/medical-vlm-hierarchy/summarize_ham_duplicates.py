"""Count potential HAM split leakage across seeds using one image content audit."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from extract_features import read_inputs


def summarize_split(audited, pairs, split_path):
    with split_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["image_id"]: row for row in rows}
    if len(rows) != len(audited) or by_id.keys() != audited.keys():
        raise ValueError(f"Image inventory differs in {split_path}")
    for image_id, row in by_id.items():
        if any(row[k] != audited[image_id][k] for k in ("representative", "lesion_id", "label")):
            raise ValueError(f"Image metadata differs in {split_path}")
    cross_split = exact = representative_pairs = exact_reps = 0
    for pair in pairs:
        first, second = by_id[pair["first"]], by_id[pair["second"]]
        if first["split"] == second["split"]:
            continue
        if first["lesion_id"] == second["lesion_id"]:
            raise ValueError("A lesion spans splits")
        cross_split += 1
        identical = pair["same_bytes"] or pair["same_decoded_pixels"]
        exact += identical
        if first["representative"] == second["representative"] == "1":
            representative_pairs += 1
            exact_reps += identical
    return {"split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "cross_split_candidates": cross_split, "exact_cross_split_pairs": exact,
            "cross_split_representative_pairs": representative_pairs,
            "exact_cross_split_representative_pairs": exact_reps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    audit, rows = read_inputs(args.audit)
    audited = {row["image_id"]: row for row in rows}
    payload = (args.audit / "duplicate_candidates.json").read_bytes()
    pairs = json.loads(payload)
    if len(pairs) != audit["near_candidate_pairs"]:
        raise ValueError("Candidate count does not match image audit")
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text(encoding="utf-8-sig"))
    splits = {seed: summarize_split(audited, pairs, args.split_dir / f"split_seed{seed}.csv")
              for seed in protocol["split_seeds"]}
    source = next(s for s in splits.values() if s["split_sha256"] == audit["split_sha256"])
    for field in ("cross_split_candidates", "exact_cross_split_pairs"):
        if source[field] != audit[field]:
            raise ValueError(f"Split projection disagrees with image audit on {field}")
    report = {"protocol": protocol["version"], "decoded": audit["decoded"],
              "missing": len(audit["missing"]), "corrupt": len(audit["corrupt"]),
              "pillow": audit["pillow_version"], "radius": 3,
              "manifest_sha256": audit["manifest_sha256"],
              "candidate_list_sha256": hashlib.sha256(payload).hexdigest(),
              "candidate_pairs": len(pairs),
              "different_lesion_id_pairs": sum(audited[p["first"]]["lesion_id"] !=
                                               audited[p["second"]]["lesion_id"] for p in pairs),
              "identical_pixel_pairs": sum(p["same_decoded_pixels"] for p in pairs),
              "splits": splits, "decode_and_audit_seconds": audit["seconds"],
              "projection_seconds": round(time.perf_counter() - start, 3)}
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
