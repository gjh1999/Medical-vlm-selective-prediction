"""Fit scalar temperature on archived development logits and evaluate it on disjoint development groups."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ham_cached_readouts import development_readouts
from ham_dev_pilot import digest, error_auc, softmax


def loss(logits, targets, beta):
    shifted = (logits-logits.max(1,keepdims=True))*beta
    return float(np.mean(np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(targets)),targets]))


def fit_inverse_temperature(logits, targets):
    """Minimize the convex validation NLL on beta >= 0 using its monotone derivative."""
    true = logits[np.arange(len(targets)),targets]
    def derivative(beta):
        return float(np.mean((softmax(beta*logits)*logits).sum(1)-true))
    if derivative(0) >= 0:
        return 0.
    if np.all(true==logits.max(1)):
        raise ValueError("Unbounded inverse-temperature optimum for perfectly ranked fitting labels")
    lo,hi = 0.,1.
    while derivative(hi)<0:
        hi*=2
    for _ in range(70):
        mid=(lo+hi)/2
        if derivative(mid)<0:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2


def metrics(tree, labels, logits, beta):
    targets=np.asarray([tree.leaves.index(k) for k in labels])
    probs=softmax(logits*beta)
    # Preserve the beta->0+ modal leaf at the uniform boundary for reliability diagnostics.
    pred=logits.argmax(1)
    wrong=pred!=targets
    conf=probs.max(1)
    bins=np.minimum((conf*15).astype(int),14)
    reliability=[]
    total=0.
    for b in sorted(set(bins.tolist())):
        use=bins==b
        acc=float(np.mean(~wrong[use]))
        mean=float(np.mean(conf[use]))
        total += use.sum()*abs(acc-mean)
        reliability.append({"bin":b,"n":int(use.sum()),"mean_max_probability":mean,"accuracy":acc})
    brier=float(np.mean(((probs-np.eye(len(tree.leaves))[targets])**2).sum(1)))
    out={"n":len(labels),"correct_leaf_top1":int((~wrong).sum()),"mean_max_probability":float(conf.mean()),
         "negative_log_likelihood":loss(logits,targets,beta),"multiclass_brier_sum":brier,
         "ece_15_equal_width":float(total/len(labels)),"reliability":reliability,
         "negative_confidence_error_auroc":error_auc(wrong,-conf),"climbing":{}}
    for threshold in (.5,.7,.9):
        predictions=[]
        for index,row in zip(pred,probs):
            node=tree.leaves[index]
            while node!=tree.root and sum(row[tree.leaves.index(y)] for y in sorted(tree.desc[node]))<threshold:
                node=tree.parent[node]
            if beta>0 and node!=tree.climbing(dict(zip(tree.leaves,row)),threshold):
                raise ValueError("Reference climbing output differs")
            predictions.append(node)
        summary=tree.metrics(labels,predictions)
        out["climbing"][str(threshold)]={k:summary[k] for k in ("leaf_n","leaf_errors","node_n","node_errors",
                                                                 "correct_information_utility")}
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("pilot","balanced","increment","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    ap.add_argument("--prior-baseline",action="store_true",help="Also evaluate a Laplace-smoothed fitting-cohort label prior")
    a=ap.parse_args()
    start=time.perf_counter()
    a.output.mkdir(parents=True,exist_ok=False)
    report={"scope":"development calibration probe on fixed cohorts; no formal calibration or test predictions",
            "fit":"one nonnegative inverse temperature per frozen backbone/readout, minimizing NLL",
            "taxonomy_sha256":digest(a.taxonomy),"models":{}}
    try:
        _,tree,labels,models=development_readouts(a.pilot,a.balanced,a.increment,a.taxonomy)
        targets=np.asarray([tree.leaves.index(k) for k in labels["fit"]])
        if a.prior_baseline:
            counts=np.bincount(targets,minlength=len(tree.leaves))
            prior=(counts+1)/(len(targets)+len(tree.leaves))
            report["prior_only"]={"fit_counts":dict(zip(tree.leaves,counts.tolist())),"additive_count":1,
                                  "by_cohort":{k:metrics(tree,v,np.tile(np.log(prior),(len(v),1)),1)
                                               for k,v in labels.items()}}
        for model,methods in models.items():
            report["models"][model]={}
            for name,probs in methods.items():
                if any(np.any(p<=0) for p in probs.values()):
                    raise ValueError("Log probabilities unavailable for an underflowed probability")
                logits={k:np.log(p) for k,p in probs.items()}
                beta=fit_inverse_temperature(logits["fit"],targets)
                # Independent dense, non-adaptive search across the uniform limit and broad log grid.
                grid=np.r_[0.,np.geomspace(1e-4,max(100.,2*beta),1001)]
                grid_loss=min(loss(logits["fit"],targets,b) for b in grid)
                optimum=loss(logits["fit"],targets,beta)
                if optimum>grid_loss+1e-10 or optimum>loss(logits["fit"],targets,1)+1e-10:
                    raise ValueError("Selected temperature failed the independent objective check")
                report["models"][model][name]={"inverse_temperature":beta,"temperature":1/beta if beta else None,
                    "uniform_boundary":beta==0,"fit_groups":len(targets),"best_grid_minus_optimal_nll":grid_loss-optimum,
                    "fixed":{k:metrics(tree,labels[k],val,1) for k,val in logits.items()},
                    "fitted":{k:metrics(tree,labels[k],val,beta) for k,val in logits.items()}}
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (a.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
