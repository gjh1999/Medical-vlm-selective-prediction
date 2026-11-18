"""Fetch the pinned model and local tokenizer/configuration for offline BiomedCLIP."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from prepare_ham import download


MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
REVISION = "9f341de24bfb00180f1b847274256e9b65a3a32e"
WEIGHT_SHA256 = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
BERT = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
BERT_REVISION = "d673b8835373c6fa116d6d8006b33d48734e305d"
CONFIG_BLOBS = {
    "open_clip_config.json": "127246c8e2bdbec4fff72874d2814f92ac77a403",
    "special_tokens_map.json": "a8b3208c2884c4efb86e49300fdd3dc877220cdf",
    "tokenizer.json": "12dad549bdbaebc16053beef12821e667968ec20",
    "tokenizer_config.json": "42046bd361a033a24264dc41de1fe6cafd68072f",
    "vocab.txt": "9d65c8495e044c70ce1a30e2ae8e2f0b3738dbae",
    "config.json": "f8a8cd37533eee7ab2f72c09890b360824352953",
}


def validate_config_file(path, filename):
    data = path.read_bytes()
    git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if git_blob != CONFIG_BLOBS[filename]:
        raise ValueError(f"Unexpected published blob for {filename}")
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    text_dir = args.output / "text"
    text_dir.mkdir()
    start = time.perf_counter()
    model_url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
    bert_url = f"https://huggingface.co/{BERT}/resolve/{BERT_REVISION}"
    report = {"model": MODEL, "revision": REVISION, "text_config_revision": BERT_REVISION, "files": {}}
    try:
        for filename in CONFIG_BLOBS:
            url = f"{bert_url if filename == 'config.json' else model_url}/{filename}"
            path = (args.output if filename == "open_clip_config.json" else text_dir) / filename
            with urllib.request.urlopen(url, timeout=45) as stream:
                path.write_bytes(stream.read())
            report["files"][filename] = {"url": url, "sha256": validate_config_file(path, filename)}
        name = "open_clip_pytorch_model.bin"
        report["files"][name] = download(f"{model_url}/{name}", args.output / name, expected_sha256=WEIGHT_SHA256)
        report["status"] = "verified"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "weights.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
