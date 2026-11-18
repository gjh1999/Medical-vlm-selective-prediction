"""Audit public PAD mirror agreement against the original case inventory, offline after download."""
import argparse
import csv
import hashlib
import json
import time
import urllib.request
from pathlib import Path, PurePosixPath

from prepare_ham import download
from prepare_pad_images import HASHES

PARQUET = "chehablab/PAD-UFES-20"
PARQUET_REV = "db6a195e84dfbb7b8c9d7506882e42946bbcaa7d"
PNG = "SalmaneExploring/pad-ufes-20"
PNG_REV = "086dd94836d7b2094132a21406a75dcb6bf7ec2c"


def tree(repo, revision, folder):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{folder}?limit=1000"
    with urllib.request.urlopen(url, timeout=45) as response:
        if 'rel="next"' in response.headers.get("Link", ""):
            raise ValueError(f"Incomplete inventory for {folder}")
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, help="Directory containing complete, checksum-verifiable shards")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow
    import pyarrow.parquet as pq
    meta = args.split_dir / "metadata.csv"
    if hashlib.sha256(meta.read_bytes()).hexdigest() != HASHES["metadata.csv"]:
        raise ValueError("Unexpected official metadata")
    with meta.open(encoding="utf-8-sig", newline="") as f:
        truth = {r["img_id"]: r for r in csv.DictReader(f)}
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / "images"
    images.mkdir()
    report = {"source_type": "two public mirrors; original publisher ZIP checksums not verified",
              "parquet": [PARQUET, PARQUET_REV], "png_inventory": [PNG, PNG_REV],
              "metadata_sha256": HASHES["metadata.csv"], "pyarrow": pyarrow.__version__, "shards": {},
              "byte_identical": 0, "byte_disagreement": 0, "metadata_rows_agree": 0}
    t = time.perf_counter()
    try:
        digests = {}
        for part in range(1, 4):
            for item in tree(PNG, PNG_REV, f"all_images/imgs_part_{part}"):
                name = PurePosixPath(item["path"]).name
                if name in digests or name not in truth:
                    raise ValueError("PNG inventory repeats a case or differs from original metadata")
                digests[name] = item["lfs"]["oid"]
        if set(digests) != truth.keys():
            raise ValueError("Incomplete PNG mirror")
        report["png_inventory_sha256"] = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
        shards = tree(PARQUET, PARQUET_REV, "data")
        seen = set()
        for item in shards:
            filename = PurePosixPath(item["path"]).name
            if not filename.endswith(".parquet"):
                raise ValueError("Unexpected file in parquet data directory")
            expected = item["lfs"]["oid"]
            path = args.reuse / filename if args.reuse else None
            if path is not None and path.is_file():
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError("Cached shard checksum mismatch")
                report["shards"][filename] = {"sha256": expected, "reused": True, "bytes": path.stat().st_size}
            else:
                path = args.output / filename
                url = f'https://huggingface.co/datasets/{PARQUET}/resolve/{PARQUET_REV}/{item["path"]}'
                report["shards"][filename] = download(url, path, expected_sha256=expected)
            columns = ["image_id", "image", "patient_id", "lesion_id", "diagnostic"]
            for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4):
                for row in batch.to_pylist():
                    name = row["image_id"]
                    if not name.endswith(".png"):
                        name += ".png"
                    if name not in truth or name in seen:
                        raise ValueError("Repeated or unexpected case in parquet mirror")
                    seen.add(name)
                    if any(str(row[k]) != truth[name][k] for k in ("patient_id", "lesion_id", "diagnostic")):
                        raise ValueError("Mirror case metadata differs from publisher")
                    report["metadata_rows_agree"] += 1
                    content = row["image"]["bytes"]
                    if hashlib.sha256(content).hexdigest() != digests[name]:
                        report["byte_disagreement"] += 1
                        continue
                    with (images / name).open("xb") as stream:
                        stream.write(content)
                    report["byte_identical"] += 1
            print(f"{filename}: {len(seen)} cases, {report['byte_identical']} agree", flush=True)
        if seen != truth.keys() or report["byte_disagreement"]:
            raise ValueError("Incomplete agreement between mirrors")
        report["status"] = "mirror_agreement_verified"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-t, 3)
        (args.output / "agreement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
