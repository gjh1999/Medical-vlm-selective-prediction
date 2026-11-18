"""Re-fit the penalty using L-BFGS and replay frozen risk scores on a class-balanced dev cohort."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ham_dev_pilot import digest
from ham_error_increment import read_csv, signals
from selective_core import Taxonomy
from verify_ham_dev_pilot import rank_auc


def lbfgs(x,y,penalty):
    x=torch.tensor(x,dtype=torch.float64)
    y=torch.tensor(y,dtype=torch.float64)
    coef=torch.zeros(x.shape[1]+1,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([coef],line_search_fn="strong_wolfe",max_iter=160,
                          tolerance_grad=1e-11,tolerance_change=1e-15)
    def closure():
        opt.zero_grad()
        logits=coef[0]+x@coef[1:]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction="sum")
        loss=loss+.5*penalty*coef[1:].square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return coef.detach().numpy()


def evaluate(target,base,augmented):
    if target.all() or not target.any():
        return {"n":len(target),"errors":int(target.sum()),"auroc_difference":None}
    auc0=rank_auc(target,base)
    auc1=rank_auc(target,augmented)
    return {"n":len(target),"errors":int(target.sum()),"base_auroc":auc0,"augmented_auroc":auc1,
            "auroc_difference":auc1-auc0,"base_brier":float(np.mean((base-target)**2)),
            "augmented_brier":float(np.mean((augmented-target)**2))}


def standardized_difference(labels,errors,base,augmented,target_counts):
    weights=np.zeros(len(labels))
    for k,count in target_counts.items():
        keep=labels==k
        weights[keep]=count/keep.sum()
    both=weights[errors,None]*weights[None,~errors]
    def auc(values):
        d=values[errors,None]-values[None,~errors]
        return float((both*((d>0)+.5*(d==0))).sum()/both.sum()) if both.sum() else None
    original,extra=auc(base),auc(augmented)
    return None if original is None else extra-original


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("increment","pilot","balanced","output"):
        ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/ham-error-increment-v1.json")
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    a=ap.parse_args()
    start=time.perf_counter()
    source=json.loads((a.increment/"summary.json").read_bytes())
    first=json.loads((a.pilot/"summary.json").read_bytes())
    balanced=json.loads((a.balanced/"summary.json").read_bytes())
    cfg=json.loads(a.config.read_bytes())
    if (source["status"]!="complete" or source["config_sha256"]!=digest(a.config)
            or first["config_sha256"]!=digest(a.taxonomy)):
        raise ValueError("Changed experiment inputs")
    tree=Taxonomy(json.loads(a.taxonomy.read_bytes())["parent"])
    leaves=tree.leaves
    selections=[]
    for path,info in [(a.pilot,first),(a.increment,source),(a.balanced,balanced)]:
        if digest(path/"selection.csv")!=info["selection_sha256"]:
            raise ValueError("Changed selection")
        selections.append(read_csv(path/"selection.csv"))
    fit,new,old=selections
    if len({r["split_group"] for rows in selections for r in rows})!=sum(map(len,selections)):
        raise ValueError("Development or training groups overlap")
    tr=[i for i,r in enumerate(old) if r["role"]=="train"]
    ev=[i for i,r in enumerate(old) if r["role"]=="dev"]
    old_y=np.array([leaves.index(r["label"]) for r in old])
    y=np.array([leaves.index(r["label"]) for r in fit+new]+list(old_y[ev]))
    n=len(fit)
    stop=n+len(new)
    report={"source_config_sha256":source["config_sha256"],"new_groups":len(new),
            "posthoc_balanced_dev":len(ev),"method":"independent torch LBFGS and rank-sum AUROC",
            "scope":"balanced cohort previously examined for leaf prediction, excluded from risk-detector fitting",
            "models":{}}
    for name,info in source["models"].items():
        paths=[p/f"{name}.npz" for p in (a.pilot,a.balanced,a.increment)]
        if list(map(digest,paths))!=[info["fit_cache"],info["support_cache"],info["cache_sha256"]]:
            raise ValueError("Changed embedding/prediction cache")
        with np.load(paths[0],allow_pickle=False) as u, np.load(paths[1],allow_pickle=False) as v, np.load(paths[2],allow_pickle=False) as w:
            support=v["image"][tr].astype(np.float64)
            centers=np.stack([support[old_y[tr]==i].mean(0) for i in range(len(leaves))])
            centers/=np.linalg.norm(centers,axis=1,keepdims=True)
            rows=np.concatenate([u["image_features"],w["image_features"],v["image"][ev]])
            by=signals(rows,u["text_features"],first["models"][name]["text_node_order"],leaves,tree,
                       first["models"][name]["logit_scale"],centers)
            checks={}
            for readout,(pred,features) in by.items():
                wrong=pred!=y
                ranks={}
                deviations=[]
                for method in ("base","augmented"):
                    fitted=info["readouts"][readout][method]
                    matrix=np.column_stack([features[k] for k in fitted["features"]])
                    mu=matrix[:n].mean(0)
                    sd=matrix[:n].std(0)
                    sd[sd<1e-12]=1.
                    if (not np.allclose(mu,fitted["fit_mean"],atol=1e-10)
                            or not np.allclose(sd,fitted["fit_scale"],atol=1e-10)):
                        raise ValueError("Deployed standardization differs from fit")
                    x=(matrix-mu)/sd
                    b=lbfgs(x[:n],wrong[:n],cfg["l2_penalty"])
                    saved=np.asarray(fitted["coefficients"])
                    p=1/(1+np.exp(-(saved[0]+x[n:]@saved[1:])))
                    check=1/(1+np.exp(-(b[0]+x[n:stop]@b[1:])))
                    diff=float(np.max(np.abs(check-w[f"{readout}_{method}"])))
                    if diff>1e-5 or not np.allclose(p[:len(new)],w[f"{readout}_{method}"],rtol=0,atol=1e-10):
                        raise ValueError("Independent optimizer or saved score mismatch")
                    deviations.append(diff)
                    ranks[method]=p
                natural=evaluate(wrong[n:stop],ranks["base"][:len(new)],ranks["augmented"][:len(new)])
                if abs(natural["auroc_difference"]-info["readouts"][readout]["auroc_difference"])>1e-12:
                    raise ValueError("Rank-sum and paired-comparison AUROC differ")
                nv=y[n:stop]==leaves.index("nv")
                checks[readout]={"maximum_optimizer_probability_difference":max(deviations),"natural":natural,
                                 "balanced":evaluate(wrong[stop:],ranks["base"][len(new):],ranks["augmented"][len(new):]),
                                 "natural_nv_only":evaluate(wrong[n:stop][nv],ranks["base"][:len(new)][nv],ranks["augmented"][:len(new)][nv])}
                common=set(y[n:stop]) & set(y[stop:])
                mix={"equal_common_classes":{k:1 for k in common},
                     "natural_sample_proportions":{k:int((y[n:stop]==k).sum()) for k in common}}
                standardized={}
                for cohort,sub in [("natural",slice(0,len(new))),("balanced",slice(len(new),None))]:
                    standardized[cohort]={label:standardized_difference(y[n:][sub],wrong[n:][sub],
                        ranks["base"][sub],ranks["augmented"][sub],counts) for label,counts in mix.items()}
                checks[readout]["posthoc_standardized_auc_differences"]=standardized
            report["models"][name]=checks
    report.update(status="verified",seconds=round(time.perf_counter()-start,3))
    a.output.mkdir(exist_ok=False,parents=True)
    (a.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
