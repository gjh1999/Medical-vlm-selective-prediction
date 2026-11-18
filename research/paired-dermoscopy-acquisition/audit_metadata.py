"""Audit public MILK10k training CSVs without reading images or benchmark labels."""

import argparse
import csv
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run(source, ham_split):
    started = time.perf_counter()
    paths = {name: source / name for name in
             ("training_input.csv", "milk10k.csv", "training_supp.csv")}
    tables = {name: read_rows(path) for name, path in paths.items()}
    inputs = tables["training_input.csv"]
    pairs = defaultdict(Counter)
    for row in inputs:
        if not row["isic_id"] or not row["lesion_id"]:
            raise ValueError("A model input row lacks the image or lesion identifier.")
        pairs[row["lesion_id"]][row["image_type"]] += 1
    ids = {name: {row["isic_id"] for row in rows} for name, rows in tables.items()}
    input_ids = ids["training_input.csv"]
    input_keys = {(row["isic_id"], row["lesion_id"], row["image_type"]) for row in inputs}
    archive_keys = {(row["isic_id"], row["lesion_id"], row["image_type"])
                    for row in tables["milk10k.csv"]}
    ham_ids = {row["image_id"] for row in read_rows(ham_split)}
    paths["ham_split_seed17.csv"] = ham_split
    return {
        "source_doi": "10.34970/648456",
        "python": platform.python_version(),
        "sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "tables": {name: {"rows": len(rows), "columns": list(rows[0]),
                          "distinct_image_ids": len(ids[name]),
                          "id_symmetric_difference_from_input": len(ids[name] ^ input_ids)}
                   for name, rows in tables.items()},
        "distinct_lesions": len(pairs),
        "one_clinical_one_dermoscopic": sum(
            counts == {"clinical: close-up": 1, "dermoscopic": 1} for counts in pairs.values()),
        "image_lesion_type_symmetric_difference": len(input_keys ^ archive_keys),
        "licenses": dict(Counter(row["copyright_license"] for row in inputs)),
        "ham_distinct_image_ids": len(ham_ids),
        "ham_milk_equal_image_ids": len(ham_ids & input_ids),
        "elapsed_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ham-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.source, args.ham_split)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
