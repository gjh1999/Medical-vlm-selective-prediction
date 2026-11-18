"""Compute finite-sample risk-check operating probabilities under explicit within-group dependence."""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from selective_core import upper_bound


def binomial_mass(n, p, k):
    return math.comb(n, k) * p**k * (1-p)**(n-k)


def error_ceiling(n, target, delta):
    k = 0
    while k <= n and upper_bound(k, n, delta) <= target:
        k += 1
    return k - 1


def cluster_mass(m, p, rho):
    if rho == 0:
        mass = [binomial_mass(m, p, k) for k in range(m+1)]
    elif rho == 1:
        mass = [1-p] + [0.0] * (m-1) + [p]
    else:
        concentration = (1-rho) / rho
        alpha, beta = p * concentration, (1-p) * concentration
        # Rising-factorial form of the beta-binomial distribution, stable for the observed group sizes.
        zero = math.prod((beta+j)/(alpha+beta+j) for j in range(m))
        mass = [zero]
        for k in range(m):
            ratio = (m-k) * (k+alpha) / ((k+1) * (m-k-1+beta))
            mass.append(mass[-1] * ratio)
    mean = math.fsum(k*v for k,v in enumerate(mass))
    variance = math.fsum((k-mean)**2 * v for k,v in enumerate(mass))
    expected_variance = m*p*(1-p)*(1+(m-1)*rho)
    if not all(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-13) for a,b in (
            (math.fsum(mass),1), (mean,m*p), (variance,expected_variance))):
        raise ArithmeticError("Group PMF failed normalization or moment checks")
    return mass


def pass_probability(hist, p, rho, cutoff):
    if cutoff < 0:
        return 0.0
    current = [1.0] + [0.0] * cutoff
    for size, count in sorted(hist.items()):
        mass = cluster_mass(size, p, rho)
        for _ in range(count):
            current = [math.fsum(mass[j]*current[k-j] for j in range(min(k,size)+1))
                       for k in range(cutoff+1)]
    return math.fsum(current)


def characterize(sizes, target, delta):
    hist = Counter(sizes)
    n = sum(sizes)
    groups = len(sizes)
    ceiling = error_ceiling(n, target, delta)
    group_ceiling = error_ceiling(groups, target, delta)
    result = {"representatives": n, "groups": groups, "size_histogram": dict(sorted(hist.items())),
              "image_error_ceiling_if_treated_as_iid": ceiling, "independent_group_error_ceiling": group_ceiling,
              "scenarios": {}}
    for p in (.03, .05, .06):
        independent = math.fsum(binomial_mass(n,p,k) for k in range(ceiling+1))
        row = {"one_preselected_representative_per_group":
               math.fsum(binomial_mass(groups,p,k) for k in range(group_ceiling+1)),
               "counting_all_representatives_as_iid": {}}
        for rho in (0, .25, .5, 1):
            value = pass_probability(hist, p, rho, ceiling)
            if rho == 0 and not math.isclose(value, independent, abs_tol=1e-13, rel_tol=1e-11):
                raise ArithmeticError("Independent endpoint disagrees with the direct binomial sum")
            row["counting_all_representatives_as_iid"][str(rho)] = value
        result["scenarios"][str(p)] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    here = Path(__file__).parent
    parser.add_argument("--crosswalk", type=Path, default=here / "configs/ham-pad-crosswalk-v1.json")
    parser.add_argument("--protocol", type=Path, default=here / "protocol.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    policy = json.loads(args.protocol.read_bytes())
    shared = set(json.loads(args.crosswalk.read_bytes())["shared_nominal_leaves"].values())
    target = policy["risk_targets"]["nonroot_set_conditional_error"]
    delta = policy["family_failure_probability"] / (policy["max_candidate_policies"] * policy["simultaneous_constraints"])
    report = {"scope":"analytic sensitivity using observed calibration group sizes; error probabilities assumed",
              "assumptions":"one fixed nonroot policy accepts every row; independent groups, constant marginal error p; within each group beta-binomial ICC rho with limiting endpoints",
              "per_check_delta":delta, "target":target, "seeds":{},
              "protocol_sha256":hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              "crosswalk_sha256":hashlib.sha256(args.crosswalk.read_bytes()).hexdigest()}
    try:
        audits = json.loads((args.split_dir / "audit.json").read_bytes())["splits"]
        for seed in policy["split_seeds"]:
            path = args.split_dir / f"split_seed{seed}.csv"
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha != audits[str(seed)]["sha256"]:
                raise ValueError("Split changed")
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows = [r for r in csv.DictReader(f) if r["representative"] == "1" and r["split"] == "calibration"]
            scopes = {"all":rows, "shared_leaves":[r for r in rows if r["label"] in shared]}
            summary = {"split_sha256":sha}
            for name, members in scopes.items():
                sizes = list(Counter(r["split_group"] for r in members).values())
                summary[name] = characterize(sizes, target, delta)
            report["seeds"][str(seed)] = summary
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start, 3)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
