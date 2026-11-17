"""Zero-failure binomial feasibility only; no model/data experiment or general CP solver."""
import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


def zero_failure_upper(n, delta):
    if n < 0 or not 0 < delta < 1:
        raise ValueError("n >= 0 and 0 < delta < 1 required")
    return None if n == 0 else -math.expm1(math.log(delta) / n)


def minimum_zero_failure_n(alpha, delta):
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must be in (0, 1)")
    return math.ceil(math.log(delta) / math.log1p(-alpha))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = []
    for policies in (1, 5, 20, 100):
        for alpha in (0.1, 0.05, 0.01):
            delta = 0.05 / policies
            n = minimum_zero_failure_n(alpha, delta)
            upper = zero_failure_upper(n, delta)
            previous = zero_failure_upper(n - 1, delta)
            # Check minimality, not just successful rounding.
            if upper > alpha + 1e-14 or (previous is not None and previous <= alpha):
                raise RuntimeError("Sample-size minimality check failed")
            rows.append({"policies": policies, "alpha": alpha,
                         "family_delta": 0.05, "per_policy_delta": delta,
                         "minimum_accepted_independent_units": n,
                         "zero_failures_upper_bound": upper})
    if zero_failure_upper(0, 0.05) is not None:
        raise RuntimeError("Empty acceptance must have undefined risk")
    result = {
        "kind": "analytic_zero_failure_sample_size_not_model_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assumptions": ["fixed finite policy family", "iid independent calibration units",
                        "zero observed failures among accepted units", "Bonferroni correction"],
        "rows": rows,
        "example_400_calibration_50_percent_acceptance_20_policies_upper":
            zero_failure_upper(200, 0.05 / 20),
        "checks": "12 minimality checks plus empty-acceptance check passed",
        "compute_seconds_excluding_output_io": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
