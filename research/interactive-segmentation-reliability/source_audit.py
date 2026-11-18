"""Archive official source evidence without installing a model or executing upstream code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import time
import urllib.request


def fetch(out, name, url, limit=8 * 1024 * 1024):
    start = time.perf_counter()
    result = {"name": name, "url": url}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "MCCAI-Research/0.2"})
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError("Source exceeds size limit")
            result.update(status=response.status, final_url=response.url,
                          bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        (out / name).write_bytes(data)
    except (OSError, ValueError) as exc:
        result["error"] = str(exc)
    result["seconds"] = round(time.perf_counter() - start, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "platform": platform.platform(), "sources": [], "model_inference_run": False}
    commit = fetch(args.output, "commit.json", "https://api.github.com/repos/halleewong/ScribblePrompt/commits/main")
    report["sources"].append(commit)
    jobs = [("kvasir.html", "https://datasets.simula.no/kvasir-seg/"),
            ("paper.html", "https://arxiv.org/html/2312.07381v3")]
    if "error" not in commit:
        revision = json.loads((args.output / "commit.json").read_text(encoding="utf-8"))["sha"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Invalid Git revision")
        report["revision"] = revision
        for path in ("README.md", "LICENSE", "requirements.txt", "setup.py", "scribbleprompt/models/unet.py",
                     "scribbleprompt/models/network.py", "MedScribble/README.md", "checkpoints/README.md"):
            jobs.append((path.replace("/", "_"), f"https://raw.githubusercontent.com/halleewong/ScribblePrompt/{revision}/{path}"))
        jobs.append(("tree.json", f"https://api.github.com/repos/halleewong/ScribblePrompt/git/trees/{revision}?recursive=1"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        report["sources"].extend(pool.map(lambda item: fetch(args.output, *item), jobs))
    report["seconds"] = round(time.perf_counter() - start, 3)
    report["status"] = "partial_failure" if any("error" in x for x in report["sources"]) else "sources_archived"
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 2 if report["status"] == "partial_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
