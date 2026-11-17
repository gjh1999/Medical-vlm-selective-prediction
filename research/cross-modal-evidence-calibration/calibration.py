"""Small score-level reference baselines; no model inference or new method claim."""
from bisect import bisect_left
import math
import statistics


def finite(values):
    values = tuple(float(x) for x in values)
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected nonempty finite scores")
    return values


class Normalizer:
    """Fit on a reference set separate from conformal calibration."""
    def __init__(self, rows):
        rows = [finite(row) for row in rows]
        if len(rows) < 2 or any(len(r) != 2 for r in rows):
            raise ValueError("Need >=2 rows of two evidence channels")
        columns = list(zip(*rows))
        self.mean = tuple(statistics.mean(c) for c in columns)
        self.scale = tuple(max(statistics.stdev(c), 1e-8) for c in columns)

    def score(self, row):
        row = finite(row)
        if len(row) != 2:
            raise ValueError("Expected two evidence channels")
        return sum((v - m) / s for v, m, s in zip(row, self.mean, self.scale)) / 2


class NormalConformal:
    """Upper-tail p-value. Valid marginally only under null exchangeability."""
    def __init__(self, scores):
        self.scores = sorted(finite(scores))

    def pvalue(self, score):
        score = finite([score])[0]
        return (1 + len(self.scores) - bisect_left(self.scores, score)) / (len(self.scores) + 1)


def auroc(normal, abnormal):
    normal, abnormal = sorted(finite(normal)), sorted(finite(abnormal))
    from bisect import bisect_right
    return sum(bisect_left(normal, a) + (bisect_right(normal, a) - bisect_left(normal, a)) / 2
               for a in abnormal) / (len(normal) * len(abnormal))


def average_precision(normal, abnormal):
    """Non-interpolated AP with tied scores processed together."""
    normal, abnormal = finite(normal), finite(abnormal)
    groups = {}
    for label, values in [(0, normal), (1, abnormal)]:
        for value in values:
            counts = groups.setdefault(value, [0, 0])
            counts[label] += 1
    tp = total = 0
    result = 0.0
    for score in sorted(groups, reverse=True):
        neg, pos = groups[score]
        tp += pos
        total += neg + pos
        result += pos / len(abnormal) * tp / total
    return result


def evaluate(normal, abnormal, calibration, alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    normal, abnormal = finite(normal), finite(abnormal)
    cp = NormalConformal(calibration)
    fp = sum(cp.pvalue(s) <= alpha for s in normal)
    tp = sum(cp.pvalue(s) <= alpha for s in abnormal)
    return dict(auroc=auroc(normal, abnormal), average_precision=average_precision(normal, abnormal),
                fpr=fp / len(normal), sensitivity=tp / len(abnormal),
                false_positives=fp, true_positives=tp, n_normal=len(normal), n_abnormal=len(abnormal))
