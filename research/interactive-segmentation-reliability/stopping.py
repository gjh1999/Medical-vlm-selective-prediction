"""Fixed-policy trajectory replay and finite-family calibration; no segmentation model.

Scores must be produced without ground truth. Independence and pre-registration
are scientific requirements: ID checks cannot establish either property.
"""
from dataclasses import dataclass
import math


def upper_bound(errors, accepted, delta):
    """One-sided Clopper-Pearson upper bound, numerically inverted in log space.

    Formula follows the reference in ../medical-vlm-hierarchy/selective_core.py;
    kept local so this direction can run independently.
    """
    if (type(errors) is not int or type(accepted) is not int
            or not 0 <= errors <= accepted or not 0 < delta < 1):
        raise ValueError("Invalid integer counts or delta")
    if accepted == 0:
        return None
    if errors == accepted:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(delta) / accepted)
    coefficients = [math.lgamma(accepted + 1) - math.lgamma(i + 1)
                    - math.lgamma(accepted - i + 1) for i in range(errors + 1)]
    low, high = 0.0, 1.0
    for _ in range(64):
        p = (low + high) / 2
        if p == low or p == high:
            break
        terms = [c + i * math.log(p) + (accepted - i) * math.log1p(-p)
                 for i, c in enumerate(coefficients)]
        maximum = max(terms)
        log_cdf = maximum + math.log(sum(math.exp(x - maximum) for x in terms))
        if log_cdf > math.log(delta):
            low = p
        else:
            high = p
    return high


@dataclass(frozen=True)
class Policy:
    name: str
    signals: tuple[str, ...]
    threshold: float
    patience: int = 1

    def __post_init__(self):
        if (not self.name or not self.signals or len(set(self.signals)) != len(self.signals)
                or any(s not in {"confidence", "stability", "probe_consistency"} for s in self.signals)
                or not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1
                or type(self.patience) is not int or self.patience < 1):
            raise ValueError("Invalid policy")


def stop_step(observations, policy):
    """Return first qualifying step (1-based); no GT/quality argument exists."""
    run = 0
    for step, row in enumerate(observations, 1):
        scores = [row.get(key) for key in policy.signals]
        for score in scores:
            if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Scores must be finite and in [0,1]")
        qualifies = all(s is not None and s >= policy.threshold for s in scores)
        run = run + 1 if qualifies else 0
        if run >= policy.patience:
            return step
    return None


def validate(trajectories, budget):
    if type(budget) is not int or budget < 1 or not trajectories:
        raise ValueError("Positive budget and nonempty trajectories required")
    groups = set()
    for trajectory in trajectories:
        group = trajectory["group_id"]
        if not isinstance(group, str) or not group or group in groups:
            raise ValueError("One trajectory per unique nonempty independent group ID required")
        groups.add(group)
        if len(trajectory["observations"]) != budget or len(trajectory["dice"]) != budget:
            raise ValueError("Full trajectory must match budget")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in trajectory["dice"]):
            raise ValueError("Invalid quality values")
        for row in trajectory["observations"]:
            if not set(row) <= {"confidence", "stability", "probe_consistency"}:
                raise ValueError("Unknown observation field: keep quality out of observations")
            if any(x is not None and (not math.isfinite(x) or not 0 <= x <= 1) for x in row.values()):
                raise ValueError("Invalid observation score")
        for key in ("submitted_clicks", "base_calls", "probe_calls"):
            values = trajectory[key]
            if len(values) != budget or any(type(v) is not int or v < 0 for v in values):
                raise ValueError("Per-step nonnegative integer costs required")
        if any(v > 1 for v in trajectory["submitted_clicks"]):
            raise ValueError("Protocol permits at most one submitted click per step")


def evaluate(trajectories, policy, budget=5, target=0.85, delta=0.05):
    validate(trajectories, budget)
    if not 0 < target <= 1 or not 0 < delta < 1:
        raise ValueError("Invalid target or delta")
    accepted = failures = attempts = clicks = base_calls = probe_calls = 0
    uses_probe = policy is not None and "probe_consistency" in policy.signals
    for trajectory in trajectories:
        step = stop_step(trajectory["observations"], policy) if policy else None
        end = step if step is not None else budget
        attempts += end
        clicks += sum(trajectory["submitted_clicks"][:end])
        base_calls += sum(trajectory["base_calls"][:end])
        if uses_probe:
            probe_calls += sum(trajectory["probe_calls"][:end])
        if step is not None:
            accepted += 1
            failures += trajectory["dice"][step - 1] < target
    n = len(trajectories)
    return {"units": n, "accepted": accepted, "failures": failures,
            "coverage": accepted / n, "referral_rate": 1 - accepted / n,
            "selective_risk": failures / accepted if accepted else None,
            "overall_wrong_stop_rate": failures / n,
            "upper_bound": upper_bound(failures, accepted, delta),
            "mean_interaction_attempts": attempts / n, "mean_submitted_clicks": clicks / n,
            "mean_model_calls": (base_calls + probe_calls) / n,
            "manual_completion_time": None}


def calibrate(trajectories, policies, budget=5, target=0.85, alpha=0.05, delta=0.05):
    if (not policies or len({p.name for p in policies}) != len(policies)
            or not 0 < alpha < 1 or not 0 < delta < 1):
        raise ValueError("Unique nonempty fixed policy family and valid alpha/delta required")
    rows = []
    for policy in policies:
        metrics = evaluate(trajectories, policy, budget, target, delta / len(policies))
        bound = metrics["upper_bound"]
        rows.append({"name": policy.name, "metrics": metrics,
                     "eligible": bound is not None and bound <= alpha})
    eligible = [r for r in rows if r["eligible"]]
    # Fixed ordering: coverage, attempts, calls, then name. No test-set access.
    best = min(eligible, key=lambda r: (-r["metrics"]["coverage"],
               r["metrics"]["mean_interaction_attempts"], r["metrics"]["mean_model_calls"], r["name"])) if eligible else None
    return {"selected": best["name"] if best else None,
            "status": "selected" if best else "all_manual_referral",
            "per_policy_delta": delta / len(policies), "policies": rows,
            "conditions": "fixed family and iid independent calibration units; no distribution-shift guarantee"}
