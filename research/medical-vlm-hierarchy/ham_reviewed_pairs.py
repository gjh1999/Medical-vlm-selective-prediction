"""Apply published HAM10000 identity reviews while preserving all original identifiers."""
import csv
import hashlib
import io
import urllib.request
from pathlib import Path


SOURCE_COMMIT = "ac655ac5a5b93264c57d69d6d520778d915bfc89"
TABLE_URL = (f"https://raw.githubusercontent.com/kakumarabhishek/Corrected-Skin-Image-Datasets/"
             f"{SOURCE_COMMIT}/DermaMNIST/HAM10000_DuplicateConfirmation/AnalysesOutputFiles")
CONFIRMED = "newly_discovered_duplicates.csv"
UNCERTAIN = "near_duplicates_different_times.csv"
TABLE_HASHES = {
    CONFIRMED: "fafe916c6a97c4d6c6de3c7c93eae1f7eb601680692c25ea82c8ef595fca73c7",
    UNCERTAIN: "6c4aff9e4b138c25765045783de8a47273051448da13670bad9087f87530492a",
}


def load_tables(out, source_dir=None):
    result = {}
    for filename, expected in TABLE_HASHES.items():
        if source_dir is None:
            with urllib.request.urlopen(f"{TABLE_URL}/{filename}", timeout=45) as stream:
                data = stream.read()
        else:
            data = (source_dir / filename).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Published pair table checksum mismatch: {filename}")
        (out / filename).write_bytes(data)
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        result[filename] = [(Path(row["from_img"]).stem, Path(row["to_img"]).stem) for row in reader]
    return result


def components(ids, edges):
    parent = {key: key for key in ids}

    def root(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for left, right in edges:
        a, b = root(left), root(right)
        parent[max(a, b)] = min(a, b)
    return {key: root(key) for key in parent}


def reviewed_groups(rows, tables):
    by_image = {row["image_id"]: row for row in rows}
    edges = {}
    for table, pairs in tables.items():
        edges[table] = []
        for left, right in pairs:
            a, b = by_image[left], by_image[right]
            if table == CONFIRMED and a["dx"] != b["dx"]:
                raise ValueError(f"Conflicting diagnoses in reviewed pair: {left}, {right}")
            edges[table].append((a["lesion_id"], b["lesion_id"]))
    ids = {r["lesion_id"] for r in rows}
    analysis = components(ids, edges[CONFIRMED])
    split_groups = components(ids, edges[CONFIRMED] + edges[UNCERTAIN])
    return analysis, split_groups
