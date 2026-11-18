"""Find benchmark images depicted in specific public dataset-paper figures."""
import argparse
import csv
import hashlib
import io
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageOps

from extract_features import read_inputs
from review_ham_candidates import geometry


# Pixel coordinates in the Europe PMC JPEGs. Each box excludes panel labels and gutters.
FIGURES = {
    "ham": ("PMC6091241", [
        ("sdata2018161-f2.jpg", (697, 244), [(2, 2, 359, 240), (377, 2, 693, 240)]),
        ("sdata2018161-f4.jpg", (700, 174), [(2, 2, 229, 172), (237, 2, 463, 172), (472, 2, 697, 172)]),
    ]),
    "pad": ("PMC7479321", [
        ("gr3.jpg", (800, 634), [(7, 2, 239, 234), (287, 2, 518, 234), (565, 2, 796, 234),
                                 (2, 336, 234, 568), (282, 336, 512, 568), (560, 336, 792, 568)]),
    ]),
}


def descriptor(image, sift):
    gray = image.convert("L")
    gray.thumbnail((400, 400), Image.Resampling.LANCZOS)
    points, desc = sift.detectAndCompute(np.asarray(gray), None)
    coords = np.float32([point.pt for point in points]).reshape(-1, 2)
    return coords, desc, gray.size


def thumbnail(image):
    grey = image.convert("L").resize((40, 40), Image.Resampling.LANCZOS)
    values = np.asarray(grey, dtype=np.float64).ravel()
    values -= values.mean()
    return values / max(np.linalg.norm(values), 1e-12)


def evaluate(dataset, source, audit_dir, output, workers):
    start = time.perf_counter()
    pmcid, figures = FIGURES[dataset]
    article = ElementTree.parse(source / f"{pmcid}.xml")
    graphics = {g.get("{http://www.w3.org/1999/xlink}href") for g in article.findall(".//fig//graphic")}
    queries = []
    hashes = {}
    with ZipFile(source / f"{pmcid}.zip") as archive:
        for filename, expected_size, boxes in figures:
            if filename not in graphics:
                raise ValueError("Figure absent from source article")
            data = archive.read(filename)
            hashes[filename] = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as figure:
                if figure.size != expected_size:
                    raise ValueError("Figure version does not match panel coordinates")
                for i, box in enumerate(boxes, 1):
                    panel = figure.crop(box)
                    queries.append((f"{filename}/{i}", descriptor(panel, cv2.SIFT_create(nfeatures=512)),
                                    thumbnail(panel)))

    info, rows = read_inputs(audit_dir)
    def compare(row):
        data = Path(row["image_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("Image content changed since audit")
        with Image.open(io.BytesIO(data)) as original:
            image = ImageOps.exif_transpose(original)
            feat = descriptor(image, cv2.SIFT_create(nfeatures=512))
            small = thumbnail(image)
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        results = []
        for name, ref, reference_thumb in queries:
            n, coverage = geometry(ref, feat, matcher)
            correlation = float(small @ reference_thumb)
            results.append((name, row["image_id"], row["split_group"], row["split"], n, coverage, correlation))
        return results

    all_scores = {name: [] for name, _, _ in queries}
    with ThreadPoolExecutor(workers) as pool:
        for count, scores in enumerate(pool.map(compare, rows), 1):
            for score in scores:
                all_scores[score[0]].append(score)
            if count % 2000 == 0:
                print(f"{dataset} {count}/{len(rows)}", flush=True)

    fields = ["panel", "image", "group", "seed17_split", "inliers", "occupied_cells", "correlation"]
    matched = {}
    panel_results = {}
    with (output / f"{dataset}-matches.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for name, scores in all_scores.items():
            ranked = sorted(scores, key=lambda r: (-r[4], -r[5], -r[6], r[1]))
            hits = [r for r in ranked if r[4] >= 16 and r[5] >= 4]
            writer.writerows(hits)
            for r in hits:
                matched[r[1]] = (r[2], r[3])
            panel_results[name] = {
                "above_threshold": len(hits), "strongest_inliers": ranked[0][4],
                "strongest_cells": ranked[0][5], "strongest_full_frame_correlation": ranked[0][6],
                "runner_up_inliers": ranked[1][4],
                "max_full_frame_correlation": max(r[6] for r in ranked),
            }
    groups = dict(matched.values())
    return {"pmcid": pmcid, "figure_sha256": hashes, "manifest_sha256": info["manifest_sha256"],
            "benchmark_images": len(rows), "panel_image_comparisons": len(rows) * len(queries),
            "panels": panel_results, "distinct_matched_images": len(matched),
            "distinct_matched_groups": len(groups), "seed17_group_counts": dict(Counter(groups.values())),
            "seconds": round(time.perf_counter() - start, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "ham-audit", "pad-audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 9))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    report = {"scope": "published figure correspondence; checkpoint membership unverified",
              "workers": args.workers, "opencv": cv2.__version__, "numpy": np.__version__,
              "criterion": "at least 16 mutual-ratio SIFT affine inliers across at least four grid cells",
              "datasets": {}}
    start = time.perf_counter()
    try:
        for name, audit_dir in (("ham", args.ham_audit), ("pad", args.pad_audit)):
            report["datasets"][name] = evaluate(name, args.source, audit_dir, args.output, args.workers)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
