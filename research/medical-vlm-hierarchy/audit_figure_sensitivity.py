"""Independently screen public figure panels by pixel detail and count affected groups in all splits."""
import argparse
import csv
import hashlib
import io
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps

from audit_public_figure_panels import FIGURES
from extract_features import read_inputs


def fingerprints(image):
    gray = np.asarray(image.convert("L").resize((48, 48), Image.Resampling.LANCZOS), dtype=np.float64)
    # Discard the outer border, where gutters, labels or minor figure cropping have disproportionate influence.
    result = []
    for channel in (gray, gray - cv2.GaussianBlur(gray, (0, 0), 2)):
        inner = channel[3:-3, 3:-3].ravel()
        inner -= inner.mean()
        result.append(inner / max(np.linalg.norm(inner), 1e-12))
    return result


def run(dataset, source, audit_dir, splits, geometry_dir, geo, output, workers):
    audit, rows = read_inputs(audit_dir)
    if audit["manifest_sha256"] != geo["manifest_sha256"]:
        raise ValueError("Input geometry audit and manifest differ")
    def load(row):
        data = Path(row["image_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Benchmark image changed")
        with Image.open(io.BytesIO(data)) as original:
            return fingerprints(ImageOps.exif_transpose(original))
    with ThreadPoolExecutor(workers) as executor:
        features = np.stack(list(executor.map(load, rows)))
    groups = np.asarray([r["split_group"] for r in rows])
    with (geometry_dir / f"{dataset}-matches.csv").open(encoding="utf-8", newline="") as stream:
        geometry_hits = list(csv.DictReader(stream))
    if len(geometry_hits) != sum(p["above_threshold"] for p in geo["panels"].values()):
        raise ValueError("Unexpected geometry match count")
    by_id = {r["image_id"]:r for r in rows}
    if any(by_id[h["image"]]["split_group"] != h["group"] for h in geometry_hits):
        raise ValueError("Geometry membership changed")
    detected = {h["group"] for h in geometry_hits}
    panel_summary = {}
    pmcid, figures = FIGURES[dataset]
    with ZipFile(source / f"{pmcid}.zip") as archive:
        for name, size, boxes in figures:
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != geo["figure_sha256"][name]:
                raise ValueError("Figure changed")
            with Image.open(io.BytesIO(data)) as figure:
                if figure.size != size:
                    raise ValueError("Panel coordinates need the original figure size")
                for i, box in enumerate(boxes, 1):
                    query = fingerprints(figure.crop(box))
                    low = features[:, 0] @ query[0]
                    detail = features[:, 1] @ query[1]
                    best = int(np.argmax(detail))
                    group = rows[best]["split_group"]
                    gap = float(detail[best] - detail[groups != group].max())
                    passes = bool(low[best] >= .97 and detail[best] >= .75 and gap >= .15)
                    if passes:
                        detected.add(group)
                    panel = f"{name}/{i}"
                    geom_groups = {h["group"] for h in geometry_hits if h["panel"] == panel}
                    panel_summary[panel] = {"pixel_match": passes, "intensity": float(low[best]),
                                            "detail": float(detail[best]), "next_other_group_gap": gap,
                                            "matches_geometry_group": bool(passes and group in geom_groups)}

    with (output / f"{dataset}-groups.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split_group"])
        writer.writerows((k,) for k in sorted(detected))
    split_info = json.loads((splits / "audit.json").read_bytes())
    strata = {}
    for seed, original in split_info["splits"].items():
        path = splits / f"split_seed{seed}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != original["sha256"]:
            raise ValueError(f"Changed split {seed}")
        with path.open(newline="", encoding="utf-8") as f:
            members = list(csv.DictReader(f))
        assignments = {}
        for row in members:
            group = row["split_group"]
            if group in detected and assignments.setdefault(group, row["split"]) != row["split"]:
                raise ValueError("Public figure group spans partitions")
        if set(assignments) != detected or {r["image_id"] for r in members} != set(by_id):
            raise ValueError("Split inventory mismatch")
        strata[seed] = dict(Counter(assignments.values()))
    return {"panels": panel_summary, "geometry_groups": len({h["group"] for h in geometry_hits}),
            "combined_groups": len(detected), "groups_by_seed": strata,
            "mask_sha256": hashlib.sha256((output / f"{dataset}-groups.csv").read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "ham-audit", "pad-audit", "ham-splits", "pad-splits", "geometry", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1,9), default=3)
    args = parser.parse_args()
    geo = json.loads((args.geometry / "summary.json").read_bytes())
    if geo["status"] != "complete":
        raise ValueError("Incomplete geometry audit")
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    report = {"scope": "posthoc publication exposure sensitivity; membership in actual pretraining unknown",
              "pixel_criterion": {"intensity_min": .97, "detail_min": .75, "different_group_detail_gap_min": .15},
              "datasets": {}, "workers": args.workers}
    try:
        for name, audit, splits in (("ham", args.ham_audit, args.ham_splits), ("pad", args.pad_audit, args.pad_splits)):
            report["datasets"][name] = run(name, args.source, audit, splits, args.geometry,
                                           geo["datasets"][name], args.output, args.workers)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
