"""Compare a fresh v1.5 build against a release assembled from earlier verified evidence."""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path


def fingerprint(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load(directory, name="summary.json"):
    return json.loads((directory / name).read_bytes())


def check_review(new, old):
    fields = ("intensity_correlation", "detail_correlation")
    with ((new/"scores.csv").open(encoding="utf-8", newline="") as a,
          (old/"scores.csv").open(encoding="utf-8", newline="") as b):
        first, second = csv.DictReader(a), csv.DictReader(b)
        if first.fieldnames != second.fieldnames or not set(fields) <= set(first.fieldnames):
            raise ValueError("Review score schema differs")
        maxima = dict.fromkeys(fields, 0.)
        differing = dict.fromkeys(fields, 0)
        n = 0
        for one, two in zip(first, second, strict=True):
            for key in fields:
                x, y = float(one.pop(key)), float(two.pop(key))
                if not math.isfinite(x) or not math.isfinite(y) or abs(x-y) > 1e-12:
                    raise ValueError("Nonfinite or materially different pixel correlation")
                if key == "detail_correlation" and (x >= .3) != (y >= .3):
                    raise ValueError("Pixel eligibility threshold changed")
                maxima[key] = max(maxima[key], abs(x-y))
                differing[key] += int(x != y)
            if one != two:
                raise ValueError("Pair identity, geometric evidence or scientific decision changed")
            n += 1
    current, prior = load(new), load(old)
    for info, directory in ((current, new), (prior, old)):
        if info.pop("scores_sha256") != fingerprint(directory/"scores.csv"):
            raise ValueError("Review source hash changed")
        del info["seconds"]
    if current != prior:
        raise ValueError("Review categories, thresholds or counts changed")
    return {"rows": n, "correlation_absolute_tolerance": 1e-12, "max_absolute_difference": maxima,
            "nonidentical_correlations": differing, "fresh_scores_sha256": fingerprint(new/"scores.csv"),
            "prior_scores_sha256": fingerprint(old/"scores.csv"), "all_other_fields_and_decisions_exact": True}


def verify(args):
    fresh, released = args.fresh, args.released
    run, baseline = load(fresh, "run.json"), load(released, "run.json")
    stages = ["controls", "local-features", "words", "geometry", "review", "splits", "audit", "summary", "sensitivity"]
    if (run["status"] != "complete" or run["reused_verified_screen"] or baseline["status"] != "complete"
            or not baseline["reused_verified_screen"] or [s["stage"] for s in run["stages"]] != stages):
        raise ValueError("Expected a fresh nine-stage build and a completed cached reference build")
    compared = {}
    evidence = {
        "controls": (args.controls, ["pairs.csv"]),
        "local-features": (args.features, ["index.csv"]),
        "words": (args.words, ["candidates.csv", "vocabulary.npz"]),
        "geometry": (args.geometry, ["scores.csv"]),
        "review": (args.review, ["panel_index.csv"]),
        "splits": (released / "splits", ["split_seed17.csv", "split_seed29.csv", "split_seed43.csv", "texture_candidates.csv"]),
        "audit": (released / "audit", ["manifest.csv", "duplicate_candidates.json"]),
        "sensitivity": (released / "sensitivity", ["sensitivity_groups.csv"]),
    }
    panels = sorted(p.name for p in args.review.glob("panel-*.jpg"))
    if not panels or panels != sorted(p.name for p in (fresh / "review").glob("panel-*.jpg")):
        raise ValueError("Unexpected panel inventory")
    evidence["review"][1].extend(panels)
    for stage, (prior, artifacts) in evidence.items():
        for file in artifacts:
            relative = stage + "/" + file
            value = fingerprint(fresh / relative)
            if value != fingerprint(prior / file):
                raise ValueError(f"Byte mismatch in {relative}")
            compared[relative] = value
    review = check_review(fresh / "review", args.review)
    feature_count = feature_bytes = 0
    with (fresh / "local-features/index.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            name = row["file"]
            if Path(name).name != name:
                raise ValueError("Unexpected descriptor path")
            packed = fresh / "local-features/features" / name
            if fingerprint(packed) != row["file_sha256"] or fingerprint(args.features / "features" / name) != row["file_sha256"]:
                raise ValueError("Descriptor pack does not match the shared index")
            feature_count += 1
            feature_bytes += packed.stat().st_size
    if feature_count != load(args.features)["images"]:
        raise ValueError("Descriptor pack count differs")
    # The split builder and duplicate/sensitivity summaries contain runtime bookkeeping.
    for relative, keys in [
        ("splits/audit.json", ("elapsed_seconds",)),
        ("audit/audit.json", ("split", "seconds")),
        ("summary/summary.json", ("decode_and_audit_seconds", "projection_seconds", "split_build_seconds")),
        ("sensitivity/summary.json", ("seconds",)),
    ]:
        a, b = load(fresh, relative), load(released, relative)
        for k in keys:
            del a[k], b[k]
        if a != b:
            raise ValueError(f"Scientific fields differ in {relative}")
    geometry = load(fresh / "geometry")
    old_geometry = load(args.geometry)
    if geometry["reused"] != 0 or geometry["status"] != "complete" or old_geometry["status"] != "complete":
        raise ValueError("Expected complete geometry without reuse in the fresh run")
    for key in ("summary", "candidate_pairs", "extra_control_pairs", "control_scores_verified"):
        if geometry[key] != old_geometry[key]:
            raise ValueError(f"Geometry differs on {key}")
    with (fresh / "geometry/scores.csv").open(encoding="utf-8", newline="") as source:
        scores = sum(1 for _ in csv.DictReader(source))
    if scores != geometry["candidate_pairs"] + geometry["extra_control_pairs"]:
        raise ValueError("Wrong score-table length")
    release = load(fresh / "splits", "audit.json")
    sensitivity = load(fresh / "sensitivity")
    return {"status": "complete", "review_comparison": review, "matched_files": compared, "matched_file_count": len(compared),
            "feature_packs_verified_in_each_run": feature_count, "feature_pack_bytes_per_run": feature_bytes,
            "semantic_json_comparisons": 4, "geometry_score_rows": scores,
            "candidate_pairs": geometry["candidate_pairs"], "extra_control_pairs": geometry["extra_control_pairs"],
            "control_scores_verified": geometry["control_scores_verified"], "fresh_reused_pairs": geometry["reused"],
            "original_reused_pairs": old_geometry["reused"], "fresh_seconds": run["seconds"],
            "release_with_existing_screen_seconds": baseline["seconds"], "image_count": release["image_count"],
            "groups": sensitivity["groups"], "analysis_units": release["analysis_units"],
            "sensitivity_groups": sensitivity["excluded_groups"], "sensitivity_analysis_units": sensitivity["excluded_analysis_units"],
            "strong_edges": release["eligible_geometry_pairs"], "loose_edges": release["loose_geometry_pairs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fresh", "released", "controls", "features", "words", "geometry", "review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    info = {}
    try:
        info = verify(args)
    except Exception as exc:
        info.update(status="failed", error=str(exc))
        raise
    finally:
        info["verification_seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
