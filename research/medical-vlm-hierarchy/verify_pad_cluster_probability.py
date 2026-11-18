"""Verify the PAD sensitivity calculation by exact rational generating functions."""
import argparse
import json
import math
import time
from fractions import Fraction
from pathlib import Path


def multiply(left, right, degree):
    result = [Fraction(0)] * min(degree+1, len(left)+len(right)-1)
    for i, a in enumerate(left):
        for j, b in enumerate(right[:len(result)-i]):
            result[i+j] += a*b
    return result


def rising(value, count):
    return math.prod(value+j for j in range(count))


def distribution(n, p, rho):
    if rho == 0:
        return [math.comb(n,k) * p**k * (1-p)**(n-k) for k in range(n+1)]
    if rho == 1:
        return [1-p]+[Fraction(0)]*(n-1)+[p]
    concentration = (1-rho)/rho
    return [math.comb(n,k)*rising(p*concentration,k)*rising((1-p)*concentration,n-k)
            / rising(concentration,n) for k in range(n+1)]


def compute(hist, p, rho, cutoff):
    total = [Fraction(1)]
    for size, multiplicity in hist.items():
        factor = distribution(int(size),p,rho)
        if sum(factor) != 1:
            raise ArithmeticError("Exact group mass differs from one")
        factor = factor[:cutoff+1]
        while multiplicity:
            if multiplicity % 2:
                total = multiply(total,factor,cutoff)
            multiplicity //= 2
            if multiplicity:
                factor = multiply(factor,factor,cutoff)
    return sum(total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {}
    try:
        data = json.loads((args.source/"summary.json").read_bytes())
        if data["status"] != "complete":
            raise ValueError("Incomplete source")
        memo = {}
        comparisons = []
        for row in data["seeds"].values():
            for name in ("all", "shared_leaves"):
                cohort = row[name]
                hist = tuple(sorted(cohort["size_histogram"].items()))
                cutoff = cohort["image_error_ceiling_if_treated_as_iid"]
                if cutoff < 0:
                    raise ValueError("No certifiable outcomes for this verification input")
                for p, values in cohort["scenarios"].items():
                    for rho, observed in values["counting_all_representatives_as_iid"].items():
                        key = (hist,p,rho,cutoff)
                        if key not in memo:
                            memo[key] = float(compute(dict(hist),Fraction(p),Fraction(rho),cutoff))
                        comparisons.append(abs(memo[key]-observed))
        error = max(comparisons)
        if error > 1e-12:
            raise ArithmeticError("Exact result differs from floating-point recurrence")
        report.update(status="passed", distinct_exact_calculations=len(memo), verified_entries=len(comparisons),
                      max_absolute_probability_error=error)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start, 3)
        (args.output/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
