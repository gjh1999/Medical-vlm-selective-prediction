"""Compare a batched CLIP smoke cache with single-image runs of the official loader."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from extract_features import read_inputs
from prepare_openai_clip import WEIGHT_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "smoke", "weights", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"absolute_tolerance": 1e-4, "status": "in_progress"}
    try:
        import clip
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        audit, rows = read_inputs(args.audit)
        cache = json.loads((args.smoke / "cache.json").read_text(encoding="utf-8"))
        if cache["model"] != "clip" or cache["status"] != "smoke_complete":
            raise ValueError("Expected a completed OpenAI CLIP smoke cache")
        with args.weights.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHT_SHA256:
                raise ValueError("Unexpected reference checkpoint")
        with (args.smoke / "manifest.csv").open(newline="", encoding="utf-8") as stream:
            samples = list(csv.DictReader(stream))
        eligible = [r for r in rows if r["representative"] == "1" and r["split"] == "train"]
        expected = sorted(eligible, key=lambda row: row["image_id"])[:cache["batch_size"]]
        if not expected or samples != expected or audit["manifest_sha256"] != cache["manifest_sha256"]:
            raise ValueError("Smoke rows differ from the audited train representatives")
        features_path = args.smoke / "image_features.npy"
        if hashlib.sha256(features_path.read_bytes()).hexdigest() != cache["features_sha256"]:
            raise ValueError("Cached features changed after extraction")
        saved = np.load(features_path, allow_pickle=False)
        if saved.shape != (len(samples), 512) or saved.dtype != np.float32:
            raise ValueError("Unexpected ViT-B/16 smoke cache shape or dtype")
        device = cache["device"]
        reference, transform = clip.load(str(args.weights.resolve()), device=device, jit=False)
        reference.float().eval()
        single = []
        with torch.inference_mode():
            for row in samples:
                path = Path(row["image_path"])
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError("Image changed after audit")
                with Image.open(path) as image:
                    tensor = transform(ImageOps.exif_transpose(image).convert("RGB"))
                embedding = reference.encode_image(tensor.unsqueeze(0).to(device))
                single.append(torch.nn.functional.normalize(embedding).cpu().numpy()[0])
        norm_error = float(np.abs(np.linalg.norm(saved, axis=1) - 1).max())
        feature_error = float(np.max(np.abs(saved - np.stack(single))))
        if not np.isfinite([feature_error, norm_error]).all():
            raise ValueError("Non-finite reference comparison")
        report.update(device=device, images=len(samples), dimensions=saved.shape[1],
                      features_sha256=cache["features_sha256"], max_unit_norm_error=norm_error,
                      max_absolute_feature_error=feature_error)
        if max(norm_error, feature_error) > report["absolute_tolerance"]:
            raise ValueError("Reference comparison exceeds the FP32 smoke tolerance")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
