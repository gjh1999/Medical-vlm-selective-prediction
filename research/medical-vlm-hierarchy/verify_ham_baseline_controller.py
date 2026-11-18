"""Check both numerical risk bounds in each rehearsal cell against high-precision binomial probabilities."""
import argparse
import json
import math
import time
from decimal import Decimal, localcontext
from pathlib import Path


def verify_bound(k, n, upper, target, delta):
    if n == 0:
        if k != 0 or upper is not None:
            raise ValueError("Invalid empty acceptance bound")
        return None, False
    if not 0 <= k <= n or upper is None:
        raise ValueError("Nonempty constraint lacks a bound")
    def cdf(p):
        if p == 1:
            return Decimal(int(k == n))
        return sum(Decimal(math.comb(n, i)) * p**i * (1-p)**(n-i) for i in range(k+1))
    if k == n:
        if upper != 1.0:
            raise ValueError("All-error upper bound differs")
        residual = Decimal(0)
    else:
        residual = abs(cdf(Decimal(str(upper))) - delta)
    passed = cdf(Decimal(str(target))) <= delta
    if residual > Decimal("1e-12") or passed != (upper <= target):
        raise ArithmeticError("High-precision binomial check differs")
    return residual, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    out = {"status": "running", "method": "50-digit decimal binomial sums at the returned bound and original target",
           "candidate_constraints": 0, "oracle_constraints": 0, "empty": 0, "nonempty": 0, "selected_replayed": 0}
    try:
        config = json.loads((Path(__file__).parent / "configs/ham-controller-rehearsal-v1.json").read_bytes())
        source = json.loads((args.results / "summary.json").read_bytes())
        if source["status"] != "complete":
            raise ValueError("Incomplete rehearsal")
        worst = Decimal(0)
        with localcontext() as ctx:
            ctx.prec = 50
            for readouts in source["models"].values():
                for run in readouts.values():
                    for budget, result in run.items():
                        delta = Decimal(str(config["family_failure_probability"])) / (2 * len(result["candidates"]))
                        eligible = []
                        for name, row in result["candidates"].items():
                            passed = []
                            for prefix, target in (("leaf", config["leaf_target"]), ("node", config["nonroot_target"])):
                                key = "leaf_upper" if prefix == "leaf" else "nonroot_upper"
                                res, ok = verify_bound(row[prefix+"_errors"], row[prefix+"_n"], row[key], target, delta)
                                out["candidate_constraints"] += 1
                                out["empty" if res is None else "nonempty"] += 1
                                if res is not None:
                                    worst = max(worst, res)
                                passed.append(ok)
                            if passed != row["passed"]:
                                raise ArithmeticError("Individual constraint comparison differs")
                            if all(passed):
                                eligible.append((row["correct_information_utility"], row["leaf_n"]/int(budget), name))
                        chosen = max(eligible)[2] if eligible else None
                        if chosen != result["selected"]:
                            raise ArithmeticError("Selected policy differs")
                        out["selected_replayed"] += 1
                        oracle = result["oracle"]
                        attain = []
                        for prefix, target in (("leaf", config["leaf_target"]), ("nonroot", config["nonroot_target"])):
                            res, ok = verify_bound(0, oracle["correct_"+prefix+"_available"],
                                                   oracle["minimum_"+prefix+"_upper"], target, delta)
                            out["oracle_constraints"] += 1
                            if res is not None:
                                worst = max(worst, res)
                            attain.append(ok)
                        if all(attain) != oracle["oracle_could_pass"]:
                            raise ArithmeticError("Oracle feasibility differs")
        out.update(status="complete", maximum_absolute_probability_residual=float(worst))
    except Exception as exc:
        out.update(status="failed", error=str(exc))
        raise
    finally:
        out["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
