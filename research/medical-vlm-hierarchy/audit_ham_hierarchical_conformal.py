"""Compare the authors' randomized hierarchical conformal predictors with a seven-leaf reference."""
import argparse
import ast
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from audit_ham_set_complexity import nested, output_space
from ham_cached_readouts import development_readouts
from ham_dev_pilot import digest
from selective_core import conformal_quantile


SOURCE_REVISION = "886ef7ba072b8ba54bbc1b5072e034e50cc7b6a6"


def leaf_path(tree, leaf):
    path = [leaf]
    while path[-1] != tree.root:
        path.append(tree.parent[path[-1]])
    return ";".join(reversed(path))


def reference_intervals(tree, space, probs, rule):
    """Return the mass before and after each leaf first enters the nested set."""
    intervals = np.empty((len(probs), len(tree.leaves), 2))
    for i, row in enumerate(probs):
        if rule == "ancestor":
            node = tree.leaves[int(np.argmax(row))]
            masks = []
            while True:
                masks.append(sum(1 << tree.leaves.index(y) for y in tree.desc[node]))
                if node == tree.root:
                    break
                node = tree.parent[node]
        else:
            masks = nested(row, int(rule), space)
        previous, lower = 0, 0.0
        for mask in masks:
            if mask == previous:
                continue
            upper = sum(row[j] for j in range(len(row)) if mask & (1 << j))
            for j in range(len(row)):
                if (mask ^ previous) & (1 << j):
                    intervals[i, j] = lower, upper
            previous, lower = mask, upper
        if previous != (1 << len(row))-1:
            raise ValueError("Reference prediction path did not reach the root")
    return intervals


def measure(masks, truth, space, budget, leaf_count):
    size = [m.bit_count() for m in masks]
    nodes = [space[m] for m in masks]
    correct = [bool(m & (1 << int(y))) for m, y in zip(masks, truth)]
    if max(nodes) > budget:
        raise ValueError("Prediction exceeds the representation budget")
    return {"n": len(masks), "covers_truth": sum(correct), "mean_leaves": float(np.mean(size)),
            "mean_minimum_nodes": float(np.mean(nodes)), "empty": size.count(0),
            "all_leaves": size.count(leaf_count), "single_leaf": size.count(1),
            "correct_proper_nonempty": sum(ok and 0 < n < leaf_count for ok, n in zip(correct, size)),
            "multi_node": sum(n > 1 for n in nodes)}


def compare(tree, space, probs, truth, author, rule, draws, alpha):
    paths = [leaf_path(tree, leaf) for leaf in tree.leaves]
    author_to_local = [paths.index(s) for s in author.classes_]
    if {frozenset(tree.leaves[author_to_local[j]] for j in ast.literal_eval(k)) for k in author.tree_} != {
            frozenset(s) for s in tree.desc.values()}:
        raise ValueError("Author tree leaf memberships differ from the local taxonomy")
    args = {} if rule == "ancestor" else {"r": int(rule)}
    budget = 1 if rule == "ancestor" else int(rule)
    reference = {k: reference_intervals(tree, space, p, rule) for k,p in probs.items()}
    prepared = {k:p[:, author_to_local] for k,p in probs.items()}
    results = {}
    for seed, randomness in draws.items():
        scores = {}
        for name, bounds in reference.items():
            lo, hi = bounds[:,:,0], bounds[:,:,1]
            scores[name] = hi - randomness[name][:,None] * (hi-lo)
        n = len(truth["fit"])
        reference_fit = scores["fit"][np.arange(n), truth["fit"]]
        fit_labels = np.array([author_to_local.index(y) for y in truth["fit"]])
        source_fit = author.calibrate(prepared["fit"], fit_labels, u=randomness["fit"], **args)
        ref_q = conformal_quantile(reference_fit.tolist(), alpha)
        q = conformal_quantile(source_fit.tolist(), alpha)
        entry = {"reference_threshold":ref_q, "author_threshold":q,
                 "fit_score_differences_gt_1e-10":int(np.count_nonzero(abs(reference_fit-source_fit)>1e-10)),
                 "max_abs_fit_difference":float(np.max(abs(reference_fit-source_fit)))}
        for name in ("natural","balanced"):
            predicted = author.predict_set(prepared[name], q, u=randomness[name], **args)
            masks = [sum(1 << author_to_local[j] for j in set(s)) for s in predicted]
            # Eval draws are disjoint from fit draws. The equality convention matters only at ties.
            ref_masks = [sum(1 << int(j) for j in np.flatnonzero(row<=ref_q)) for row in scores[name]]
            same_q = [sum(1 << int(j) for j in np.flatnonzero(row<=q)) for row in scores[name]]
            entry[name] = {"author":measure(masks,truth[name],space,budget,len(tree.leaves)),
                           "reference":measure(ref_masks,truth[name],space,budget,len(tree.leaves)),
                           "different_outputs_own_thresholds":sum(a!=b for a,b in zip(masks,ref_masks)),
                           "different_outputs_at_author_threshold":sum(a!=b for a,b in zip(masks,same_q))}
        results[str(seed)] = entry
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot","balanced","increment","author-source","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    args = ap.parse_args()
    source = args.author_source.resolve()
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()!=SOURCE_REVISION:
        raise ValueError("Unexpected author source revision")
    subprocess.run(["git","diff","--quiet","HEAD","--","hcp"],cwd=source,check=True)
    sys.path.insert(0,str(source))
    from hcp.base import CRSVP, CRSVPR
    import sklearn
    args.output.mkdir(exist_ok=False,parents=True)
    start = time.perf_counter()
    report = {"author_revision":SOURCE_REVISION,"alpha":.10,"randomization_seeds":[17,29,43],
              "scope":"reused development cohorts only; prediction-set membership and cardinality",
              "quantile":"ascending order statistic ceil((n+1)*(1-alpha)), equation 5",
              "taxonomy_sha256":digest(args.taxonomy),"numpy":np.__version__,"sklearn":sklearn.__version__,
              "models":{}}
    try:
        _,tree,labels,models = development_readouts(args.pilot,args.balanced,args.increment,args.taxonomy)
        space = output_space(tree)
        truth = {k:np.array([tree.leaves.index(n) for n in v]) for k,v in labels.items()}
        draws = {}
        for seed in report["randomization_seeds"]:
            rng = np.random.default_rng(seed)
            draws[seed] = {k:rng.random(len(truth[k])) for k in ("fit","natural","balanced")}
        report["fit_n"] = len(truth["fit"])
        report["ascending_rank"] = math.ceil((report["fit_n"]+1)*(1-report["alpha"]))
        path_names = [leaf_path(tree,n) for n in tree.leaves]
        rules = {"ancestor":CRSVP().fit(path_names), "1":CRSVPR().fit(path_names), "2":CRSVPR().fit(path_names)}
        for model,methods in models.items():
            report["models"][model] = {}
            for method,probs in methods.items():
                report["models"][model][method] = {
                    rule:compare(tree,space,probs,truth,authors,rule,draws,report["alpha"])
                    for rule,authors in rules.items()}
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
