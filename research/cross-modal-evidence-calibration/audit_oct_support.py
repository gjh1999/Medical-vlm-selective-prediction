"""Count randomized patient identifiers in the released MVFA OCT2017 episode lists."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


REVISION = "a30bf7bb2c51772a36e7be58d3c072be480620fe"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = []
    for shots in (2, 4, 8, 16):
        path = f"dataset/fewshot_seed/Retina_OCT2017/{shots}-shot.txt"
        data = subprocess.check_output(
            ["git", "-C", args.repo, "show", f"{REVISION}:{path}"], encoding="utf-8"
        )
        for line in data.splitlines():
            label, *images = line.split()
            if len(images) != shots or len(set(images)) != shots:
                raise ValueError(f"Unexpected image count in {path}, {label}")
            identifiers = set()
            for name in images:
                match = re.fullmatch(r"(NORMAL|CNV|DRUSEN|DME)-(\d+)-\d+\.jpeg", name)
                if not match or (match[1] == "NORMAL") != label.startswith("n-"):
                    raise ValueError(f"Unexpected label or filename in {path}")
                identifiers.add(match[2])
            rows.append(dict(shots=shots, episode=label.rstrip(":"), images=len(images),
                             patient_ids=len(identifiers)))
    result = dict(revision=REVISION, rows=rows, elapsed_seconds=time.perf_counter() - started)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
