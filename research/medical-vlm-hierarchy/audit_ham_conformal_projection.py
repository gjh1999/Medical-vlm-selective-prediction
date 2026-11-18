"""Evaluate label-set coverage and the specificity lost when projecting sets to a single tree node."""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from audit_ham_set_complexity import output_space
from ham_cached_readouts import development_readouts
from ham_dev_pilot import digest
from selective_core import conformal_quantile, lac_set


def evaluate(tree,space,probs,truth,q):
    predictions=[]
    sizes=[]
    complexity=[]
    inside=[]
    added_true=empty_to_root=inflated=0
    for row,y in zip(probs,truth):
        names=lac_set(dict(zip(tree.leaves,row)),q)
        expected=np.flatnonzero((1-row)<=q)
        if names!=[tree.leaves[i] for i in expected]:
            raise ValueError("Set construction differs from the score threshold")
        action=tree.lca(names)
        label_set=set(names)
        if not label_set<=tree.desc[action]:
            raise ValueError("Projection dropped a candidate label")
        # Independently verify minimality among single tree nodes for a nonempty set.
        if names and len(tree.desc[action])!=min(len(d) for d in tree.desc.values() if label_set<=d):
            raise ValueError("Projection is not the smallest encompassing node")
        predictions.append(action)
        inside.append(y in label_set)
        sizes.append(len(names))
        complexity.append(space[sum(1<<int(i) for i in expected)])
        empty_to_root+=not names
        inflated+=bool(names) and label_set!=tree.desc[action]
        added_true+=bool(names) and y not in label_set and y in tree.desc[action]
    stats=tree.metrics(truth,predictions)
    counts=Counter(predictions)
    return {"n":len(truth),"set_contains_truth":sum(inside),"mean_set_size":float(np.mean(sizes)),
            "set_size_histogram":dict(sorted(Counter(sizes).items())),
            "minimum_exact_tree_nodes_histogram":dict(sorted(Counter(complexity).items())),
            "projection_node_counts":dict(sorted(counts.items())),"empty_set_to_root":empty_to_root,
            "nonempty_sets_expanded":inflated,"true_label_added_to_nonempty_set_by_projection":added_true,
            "correct_nonroot_projection":stats["node_n"]-stats["node_errors"],
            "projection_metrics":{k:v for k,v in stats.items() if k!="per_class"},
            "set_coverage_by_class":{n:{"n":truth.count(n),"covered":sum(ok for y,ok in zip(truth,inside) if y==n)}
                                     for n in tree.leaves}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("pilot","balanced","increment","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    report={"scope":"development-only surrogate calibration; natural and balanced evaluations reuse frozen cohorts",
            "score":"1 - frozen probability of the true leaf","alphas":[.05,.1,.2],"models":{}}
    try:
        _,tree,truths,models=development_readouts(args.pilot,args.balanced,args.increment,args.taxonomy)
        fit=[tree.leaves.index(n) for n in truths["fit"]]
        space=output_space(tree)
        report.update(fit_groups=len(fit),taxonomy_sha256=digest(args.taxonomy))
        for model,methods in models.items():
            results={}
            for method,cohorts in methods.items():
                calibration=(1-cohorts["fit"][np.arange(len(fit)),fit]).tolist()
                results[method]={}
                for alpha in report["alphas"]:
                    rank=math.ceil((len(fit)+1)*(1-alpha))
                    q=conformal_quantile(calibration,alpha)
                    other=float(np.partition(np.asarray(calibration),rank-1)[rank-1])
                    if q!=other:
                        raise ValueError("Order statistic differs")
                    entry={"quantile":q,"order_statistic":rank}
                    for name in ("natural","balanced"):
                        entry[name]=evaluate(tree,space,cohorts[name],truths[name],q)
                    results[method][str(alpha)]=entry
            report["models"][model]=results
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
