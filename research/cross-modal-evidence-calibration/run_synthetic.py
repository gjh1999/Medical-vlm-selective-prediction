"""CPU mechanism experiment with deliberately constructed score shifts, not medical data."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from calibration import Normalizer, evaluate


def sample(rng, n, scenario, abnormal=False):
    rows = []
    for _ in range(n):
        shared, first, second = (rng.gauss(0, 1) for _ in range(3))
        x, y = 0.5 * shared + first, 0.5 * shared + second
        if abnormal:
            x, y = x + 0.5, y + 2.0
        if scenario == "one_channel_shift":
            x = 3 * x + 3
        elif scenario == "shared_shift":
            x, y = x + 2, y + 2
        rows.append((x, y))
    return rows


def run(seed, scenario, config):
    rng = random.Random(seed)
    # Disjoint generated observations; no test observation fits any transformation.
    source_fit = sample(rng, config['n_fit'], 'iid')
    source_cal = sample(rng, config['n_cal'], 'iid')
    target_fit = sample(rng, config['n_fit'], scenario)
    target_cal = sample(rng, config['n_cal'], scenario)
    normal = sample(rng, config['n_test_per_class'], scenario)
    abnormal = sample(rng, config['n_test_per_class'], scenario, True)
    raw = lambda row: sum(row) / 2
    source_z, target_z = Normalizer(source_fit), Normalizer(target_fit)
    methods = [('source_raw', raw, source_cal), ('source_z', source_z.score, source_cal),
               ('target_raw', raw, target_cal), ('target_z', target_z.score, target_cal)]
    result = []
    for name, score, cal in methods:
        metrics = evaluate(list(map(score, normal)), list(map(score, abnormal)),
                           list(map(score, cal)), config['alpha'])
        result.append(dict(seed=seed, scenario=scenario, method=name, **metrics))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = dict(protocol='synthetic-v1', seeds=[17, 29, 43], n_fit=199, n_cal=199,
                  n_test_per_class=2000, alpha=0.05,
                  scenarios=['iid', 'one_channel_shift', 'shared_shift'])
    rows = [row for seed in config['seeds'] for scenario in config['scenarios']
            for row in run(seed, scenario, config)]
    summary = []
    for scenario in config['scenarios']:
        for method in ['source_raw', 'source_z', 'target_raw', 'target_z']:
            selected = [r for r in rows if r['scenario'] == scenario and r['method'] == method]
            summary.append(dict(scenario=scenario, method=method,
                **{metric: dict(mean=statistics.mean(r[metric] for r in selected),
                                sd=statistics.stdev(r[metric] for r in selected))
                   for metric in ['auroc', 'average_precision', 'fpr', 'sensitivity']}))
    result = dict(config=config, results=rows, summary=summary,
                  elapsed_seconds=time.perf_counter()-started,
                  python=sys.version, platform=platform.platform(),
                  created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), Path(__file__).with_name('calibration.py')]},
                  limitations='Synthetic scores only; no clinical claim; 3 seeds are not confidence intervals.')
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(elapsed_seconds=result['elapsed_seconds'], summary=summary), indent=2))


if __name__ == '__main__':
    main()
