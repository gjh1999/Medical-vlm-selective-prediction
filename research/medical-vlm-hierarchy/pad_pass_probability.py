"""Compute a fixed policy's pass probability from the audited PAD error ceilings."""
import argparse
import hashlib
import json
import math
import platform
import time
from fractions import Fraction
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    source = args.capacity.read_bytes()
    capacity = json.loads(source)
    # Seed 17 is the first prespecified split. Do not pool overlapping groups.
    rows = []
    for budget, limits in capacity["max_errors_if_all_groups_accepted"]["17"].items():
        n = int(budget)
        ceiling = limits["nonroot_set_conditional_error"]
        probabilities = {}
        for percent in (1, 2, 3, 5):
            p = Fraction(percent, 100)
            probability = Fraction(0)
            if ceiling is not None:
                probability = sum(math.comb(n, k) * p**k * (1-p)**(n-k)
                                  for k in range(ceiling + 1))
            probabilities[str(percent)] = float(probability)
        rows.append({"accepted_groups": n, "max_errors": ceiling,
                     "pass_probability_by_true_error_percent": probabilities})
    result = {
        "protocol_version": capacity["protocol"], "seed": 17,
        "capacity_sha256": hashlib.sha256(source).hexdigest(),
        "per_check_delta": capacity["per_check_delta"], "rows": rows,
        "scope": "One fixed policy, iid accepted groups, nonroot risk only.",
        "python": platform.python_version(),
        "elapsed_compute_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "probability.json").write_text(json.dumps(result, indent=2) + "\n",
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
