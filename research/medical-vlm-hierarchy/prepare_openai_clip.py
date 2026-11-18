"""Download and verify the original OpenAI ViT-B/16 checkpoint."""
import argparse
import json
import time
from pathlib import Path

from prepare_ham import download


WEIGHT_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
WEIGHT_URL = f"https://openaipublic.azureedge.net/clip/models/{WEIGHT_SHA256}/ViT-B-16.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    start = time.perf_counter()
    report = {"url": WEIGHT_URL, "expected_sha256": WEIGHT_SHA256}
    try:
        report.update(download(WEIGHT_URL, args.output / "ViT-B-16.pt", expected_sha256=WEIGHT_SHA256))
        report["status"] = "verified"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "weights.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
