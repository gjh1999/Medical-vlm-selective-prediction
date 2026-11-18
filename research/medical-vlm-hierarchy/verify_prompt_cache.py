"""Check a train-only image/text smoke cache using independent single-item encoder calls."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from extract_features import load_encoder, read_inputs


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(args):
    cache = json.loads((args.cache / "cache.json").read_bytes())
    audit, rows = read_inputs(args.audit)
    config = json.loads(args.prompt_config.read_bytes())
    if (cache["status"] != "smoke_complete" or cache["manifest_sha256"] != audit["manifest_sha256"]
            or digest(args.audit / "audit.json") != cache["audit_sha256"]
            or digest(args.prompt_config) != cache["text"]["config_sha256"]):
        raise ValueError("Smoke cache provenance differs")
    expected = sorted((r for r in rows if r["split"] == "train" and r["representative"] == "1"),
                      key=lambda row: row["image_id"])[:cache["batch_size"]]
    with (args.cache / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        if not expected or list(csv.DictReader(stream)) != expected:
            raise ValueError("Cache rows differ from the audited training representatives")
    image_path = args.cache / "image_features.npy"
    text_path = args.cache / "text_features.npy"
    if digest(image_path) != cache["features_sha256"] or digest(text_path) != cache["text"]["features_sha256"]:
        raise ValueError("Feature file changed")
    nodes = sorted(config["names"])
    templates = config["templates"]
    stored_text = cache["text"]
    if nodes != stored_text["node_order"] or templates != stored_text["templates"]:
        raise ValueError("Unexpected saved prompt order")
    saved_image = np.load(image_path, allow_pickle=False)
    saved_text = np.load(text_path, allow_pickle=False)
    if (saved_image.shape != (len(expected), 512) or saved_text.shape != (len(templates), len(nodes), 512)
            or saved_text.dtype != np.float32 or saved_image.dtype != np.float32):
        raise ValueError("Unexpected cache dimensions or precision")
    device = cache["device"]
    metadata = {}
    model, transform, tokenize = load_encoder(cache["model"], args.weights, device, args.model_config,
                                               metadata, with_tokenizer=True)
    if metadata["weights_sha256"] != cache["weights_sha256"]:
        raise ValueError("Checkpoint mismatch")
    prompts = [t.format(label=config["names"][n]) for t in templates for n in nodes]
    text, images = [], []
    with torch.inference_mode():
        for prompt in prompts:
            tokens = tokenize([prompt]).to(device)
            if tokens.shape[1] != stored_text["context_length"]:
                raise ValueError("Tokenizer context changed")
            feature = model.encode_text(tokens).float()
            text.append(torch.nn.functional.normalize(feature).cpu().numpy()[0])
        for row in expected:
            path = Path(row["image_path"])
            if digest(path) != row["sha256"]:
                raise ValueError("Training image changed")
            with Image.open(path) as image:
                tensor = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(device)
            feature = model.encode_image(tensor).float()
            images.append(torch.nn.functional.normalize(feature).cpu().numpy()[0])
        scale = float(model.logit_scale.exp().item())
    if scale != stored_text["checkpoint_logit_scale"]:
        raise ValueError("Raw logit scale changed")
    cached = saved_text.reshape(len(prompts), -1)
    text_error = float(np.abs(cached - text).max())
    image_error = float(np.abs(saved_image - images).max())
    cached_logits = scale * saved_image @ cached.T
    reference_logits = scale * np.asarray(images) @ np.asarray(text).T
    logit_error = float(np.abs(cached_logits - reference_logits).max())
    if not (max(text_error, image_error) <= 1e-4 and logit_error <= 1e-3):
        raise ValueError(f"Reference errors exceed tolerance: {text_error}, {image_error}, {logit_error}")
    return {"model": cache["model"], "device": device, "train_images": len(expected), "prompts": len(prompts),
            "verified_logits": cached_logits.size, "scale": scale, "feature_tolerance": 1e-4, "logit_tolerance": 1e-3,
            "max_text_feature_error": text_error, "max_image_feature_error": image_error,
            "max_logit_error": logit_error, "config_sha256": stored_text["config_sha256"],
            "cache_total_seconds": cache["total_seconds"], "text_seconds": stored_text["seconds"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("cache", "audit", "prompt-config", "weights", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--model-config", type=Path)
    args = ap.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    start = time.perf_counter()
    report = {}
    try:
        report.update(check(args), status="passed")
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["reference_seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
