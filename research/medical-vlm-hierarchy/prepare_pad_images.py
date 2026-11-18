"""Verify the PAD-UFES-20 v1 archives and extract only metadata-listed PNGs."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

from prepare_ham import download

HASHES = {
    "imgs_part_1.zip": "0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca",
    "imgs_part_2.zip": "e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b",
    "imgs_part_3.zip": "ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c",
    "metadata.csv": "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--archives", type=Path, help="Reuse any already verified archives from this directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    source = json.loads((args.split_dir / "dataset_api.json").read_bytes())
    entries = {f["filename"]: f["content_details"] for f in source["files"]}
    if (source["doi"]["id"] != "10.17632/zr7vgbcyr2.1"
            or {name: entries[name]["sha256_hash"] for name in HASHES} != HASHES):
        raise ValueError("Unexpected PAD version or archive digests")
    meta = args.split_dir / "metadata.csv"
    if hashlib.sha256(meta.read_bytes()).hexdigest() != HASHES["metadata.csv"]:
        raise ValueError("Unexpected metadata content")
    with meta.open(encoding="utf-8-sig", newline="") as stream:
        ids = {r["img_id"] for r in csv.DictReader(stream)}
    if any(not re.fullmatch(r"PAT_[0-9]+_[0-9]+_[0-9]+\.png", name) for name in ids):
        raise ValueError("Unexpected metadata image filename")
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / "images"
    images.mkdir()
    report = {"doi": source["doi"]["id"], "expected": len(ids), "archives": {}}
    seen = set()
    try:
        for name, expected in HASHES.items():
            if name == "metadata.csv":
                continue
            t = time.perf_counter()
            details = entries[name]
            reuse = args.archives / name if args.archives else None
            if reuse and reuse.is_file():
                archive = reuse
                with archive.open("rb") as stream:
                    checksum = hashlib.file_digest(stream, "sha256").hexdigest()
                if checksum != expected:
                    raise ValueError(f"Checksum mismatch for cached {name}")
                entry = {"sha256": checksum, "bytes": archive.stat().st_size, "reused": True}
            else:
                archive = args.output / name
                if not details["download_url"].startswith("https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/"):
                    raise ValueError("Unexpected source URL")
                entry = download(details["download_url"], archive, expected_sha256=expected)
                entry["reused"] = False
            report["archives"][name] = entry
            if entry["bytes"] != details["size"]:
                raise ValueError(f"Archive size mismatch for {name}")
            extracted = 0
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    path = PurePosixPath(member.filename)
                    if path.suffix.lower() != ".png" or "__MACOSX" in path.parts:
                        continue
                    key = path.name
                    if key not in ids or key in seen:
                        raise ValueError(f"Unlisted or repeated image in {name}: {key}")
                    with zf.open(member) as src, (images / key).open("xb") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
                    seen.add(key)
                    extracted += 1
            entry.update(images=extracted, seconds=round(time.perf_counter() - t, 3))
            print(f"{name}: verified and extracted {extracted} images", flush=True)
        if seen != ids:
            raise ValueError(f"Missing {len(ids-seen)} listed images")
        report["status"] = "verified"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report.update(extracted=len(seen), seconds=round(time.perf_counter() - start, 3))
        (args.output / "archives.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
