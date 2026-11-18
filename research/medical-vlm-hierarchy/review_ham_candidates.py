"""Prioritize HAM representative-image pairs for visual review without reading diagnoses."""
import argparse
import csv
import hashlib
import io
import json
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from extract_features import read_inputs
from ham_reviewed_pairs import CONFIRMED, load_tables


def describe(path, expected_hash, sift):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError(f"Image changed since audit: {path.name}")
    with Image.open(io.BytesIO(content)) as image:
        view = ImageOps.exif_transpose(image).convert("L")
        view.thumbnail((400, 400), Image.Resampling.LANCZOS)
        gray = np.asarray(view)
    features = []
    for pixels in [gray, cv2.flip(gray, 1)]:
        h, w = pixels.shape
        mask = np.zeros_like(pixels)
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * .43), int(h * .43)), 0, 0, 360, 255, -1)
        points, desc = sift.detectAndCompute(pixels, mask)
        xy = np.float32([p.pt for p in points]).reshape(-1, 2)
        features.append((xy, desc, (w, h)))
    return features


def geometry(first, second, matcher):
    xy, desc, size_a = first
    uv, other, size_b = second
    if min(len(xy), len(uv)) < 4:
        return 0, 0
    forward = matcher.knnMatch(desc, other, k=2)
    backward = matcher.knnMatch(other, desc, k=2)
    reverse = {m.queryIdx: m.trainIdx for m, n in backward if m.distance < .75 * n.distance}
    good = [m for m, n in forward if m.distance < .75 * n.distance and reverse.get(m.trainIdx) == m.queryIdx]
    if len(good) < 4:
        return 0, 0
    src = xy[[m.queryIdx for m in good]]
    dst = uv[[m.trainIdx for m in good]]
    cv2.setRNGSeed(17)
    transform, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    if transform is None:
        return 0, 0
    keep = mask.ravel().astype(bool)
    def cells(points, size):
        bins = np.clip((4 * points[keep] / size).astype(int), 0, 3)
        return len(np.unique(bins, axis=0))
    return int(keep.sum()), min(cells(src, size_a), cells(dst, size_b))


def panels(items, rows, out):
    index = []
    for offset in range(0, len(items), 6):
        sheet = Image.new("RGB", (800, 3 * 167), "white")
        draw = ImageDraw.Draw(sheet)
        for tile, (source, (a, b)) in enumerate(items[offset:offset + 6]):
            number = offset + tile + 1
            x, y = (tile % 2) * 400, (tile // 2) * 167
            draw.text((x + 6, y + 2), f"{number:03d}", fill="black")
            for col, key in enumerate((a, b)):
                with Image.open(rows[key]["image_path"]) as original:
                    thumb = ImageOps.exif_transpose(original).convert("RGB")
                    thumb.thumbnail((197, 145), Image.Resampling.LANCZOS)
                    sheet.paste(thumb, (x + col * 200, y + 18))
            index.append((number, source, a, b))
        sheet.save(out / f"panel-{offset // 6 + 1:02d}.jpg", quality=94)
    with (out / "panel_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("number", "source", "first", "second"))
        writer.writerows(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "split-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    audit, records = read_inputs(args.audit)
    rows = {row["image_id"]: row for row in records}
    split_report = json.loads((args.split_dir / "audit.json").read_bytes())
    if split_report.get("split_protocol") != "v1.3":
        raise ValueError("Use the published-pair corrected HAM splits")
    assignments = {}
    for seed, entry in split_report["splits"].items():
        path = args.split_dir / f"split_seed{seed}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Changed split")
        with path.open(newline="", encoding="utf-8") as stream:
            split_rows = list(csv.DictReader(stream))
        assignments[seed] = {r["image_id"]: r["split"] for r in split_rows}
        if assignments[seed].keys() != rows.keys() or any(
                r["representative"] != rows[r["image_id"]]["representative"] for r in split_rows):
            raise ValueError("Audited inventory/representative mismatch")
    tables = load_tables(args.output, args.split_dir)
    kind = {}
    for table, pairs in tables.items():
        for pair in pairs:
            kind[tuple(sorted(pair))] = "published_confirmed" if table == CONFIRMED else "published_uncertain"
    candidates_path = args.audit / "duplicate_candidates.json"
    for pair in json.loads(candidates_path.read_bytes()):
        a, b = sorted((pair["first"], pair["second"]))
        if rows[a]["representative"] == rows[b]["representative"] == "1" and any(
                split[a] != split[b] for split in assignments.values()):
            kind.setdefault((a, b), "cross_split_candidate")
    sift = cv2.SIFT_create(nfeatures=256)
    keys = sorted({key for pair in kind for key in pair})
    desc = {k: describe(Path(rows[k]["image_path"]), rows[k]["sha256"], sift) for k in keys}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ranked = []
    for (a, b), category in sorted(kind.items()):
        score, spread = max(geometry(desc[a][0], variant, matcher) for variant in desc[b])
        ranked.append((score, spread, a, b, category))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    with (args.output / "ranked.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("inliers", "occupied_cells", "first", "second", "source"))
        writer.writerows(ranked)
    unknown = [(a, b) for _, _, a, b, category in ranked if category == "cross_split_candidate"]
    rng = random.Random(17)
    items = [("ranked", pair) for pair in unknown[:18]]
    items.extend(("random_remainder", pair) for pair in rng.sample(unknown[18:], min(18, len(unknown[18:]))))
    items.extend((category, pair) for pair, category in sorted(kind.items()) if category != "cross_split_candidate")
    rng.shuffle(items)
    panels(items, rows, args.output)
    stats = {}
    for category in sorted(set(kind.values())):
        scores = [s for s, _, _, _, k in ranked if k == category]
        stats[category] = {"n":len(scores), "median_inliers":float(np.median(scores)), "max_inliers":max(scores),
                           "at_least_8":sum(i >= 8 for i in scores), "at_least_16":sum(i >= 16 for i in scores)}
    report = {"opencv":cv2.__version__, "numpy":np.__version__, "radius":3,
              "manifest_sha256":audit["manifest_sha256"], "unique_images":len(keys), "scores":stats,
              "visual_sample":dict(Counter(tag for tag, pair in items)),
              "panel_index_sha256":hashlib.sha256((args.output / "panel_index.csv").read_bytes()).hexdigest(),
              "elapsed_seconds":round(time.perf_counter() - start, 3)}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
