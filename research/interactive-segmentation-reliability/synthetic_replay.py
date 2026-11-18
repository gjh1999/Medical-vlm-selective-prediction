"""Synthetic score/quality trajectories for implementation checks, NOT segmentation evidence."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

from stopping import Policy, calibrate, evaluate


def make_rows(n, seed, prefix, shifted=False):
    rng = random.Random(seed)
    rows = []
    clamp = lambda x: min(1.0, max(0.0, x))
    for i in range(n):
        hard = rng.random() < .2
        start = rng.uniform(.35, .65)
        observations, qualities = [], []
        for t in range(1, 6):
            quality = clamp(start + (.04 if hard else .14) * t + rng.gauss(0, .015))
            # Artificially informative probes on iid data, misleading under shift.
            confidence = clamp((.96 if hard else quality) + rng.gauss(0, .025))
            probe = clamp((.97 if shifted and hard else quality) + rng.gauss(0, .02))
            observations.append({"confidence": confidence, "probe_consistency": probe})
            qualities.append(quality)
        rows.append({"group_id": f"{prefix}-{i}", "observations": observations, "dice": qualities,
                     "submitted_clicks": [1] * 5, "base_calls": [1] * 5, "probe_calls": [2] * 5})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    policies = [Policy(f"{signal}-{threshold}-{patience}",
                       ("confidence",) if signal == "confidence" else ("confidence", "probe_consistency"),
                       threshold, patience)
                for signal in ("confidence", "combined") for threshold in (.7, .8, .85, .9, .95)
                for patience in (1, 2)]
    calibration = make_rows(400, 17, "cal")
    test = make_rows(2000, 29, "test")
    shift = make_rows(2000, 43, "shift", shifted=True)
    result = calibrate(calibration, policies)
    selected = next((p for p in policies if p.name == result["selected"]), None)
    baseline = Policy("uncalibrated-confidence", ("confidence",), .9)
    report = {"kind": "synthetic_implementation_check_not_medical_experiment",
              "utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "seed_map": {"cal": 17, "test": 29, "shift": 43},
              "counts": {"cal": 400, "test": 2000, "shift": 2000},
              "policies": [asdict(p) for p in policies], "calibration": result,
              "iid_test": {"raw_confidence": evaluate(test, baseline), "calibrated": evaluate(test, selected)},
              "shift_test": {"raw_confidence": evaluate(shift, baseline), "calibrated": evaluate(shift, selected)},
              "test_bounds_note": "Descriptive one-sided 95% bounds per report, not a simultaneous test guarantee",
              "limitations": "Probe quality correlation and its shift are constructed; cannot establish H1/H2/H3 or clinical utility",
              "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("stopping.py", "synthetic_replay.py")},
              "seconds": round(time.perf_counter() - start, 6)}
    # Full synthetic traces retained separately so score/label isolation can be inspected.
    (args.output / "trajectories.json").write_text(json.dumps({"cal": calibration, "test": test, "shift": shift}), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"policies", "calibration"}}, indent=2))
    print("selected:", result["selected"])


if __name__ == "__main__":
    main()
