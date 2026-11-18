"""Resample leaf support, risk fitting and fresh dev evaluation groups to measure E02 variability."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ham_dev_pilot import digest, error_auc
from ham_error_increment import logistic_predict, read_csv, signals
from selective_core import Taxonomy


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ["increment","pilot","balanced","output"]:
        ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--confirmation",type=Path,help="Pool both natural development batches and preserve each batch size in resampling")
    ap.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/ham-error-increment-v1.json")
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    a=ap.parse_args()
    start=time.perf_counter()
    sources=[(p,json.loads((p/"summary.json").read_bytes())) for p in (a.pilot,a.balanced,a.increment)]
    cfg=json.loads(a.config.read_bytes())
    if sources[2][1]["config_sha256"]!=digest(a.config) or sources[0][1]["config_sha256"]!=digest(a.taxonomy):
        raise ValueError("Input configuration mismatch")
    tree=Taxonomy(json.loads(a.taxonomy.read_bytes())["parent"])
    selections=[]
    for folder,info in sources:
        if info["status"]!="complete" or info["selection_sha256"]!=digest(folder/"selection.csv"):
            raise ValueError("Changed selection or incomplete source")
        selections.append(read_csv(folder/"selection.csv"))
    fit,old,new=selections
    if len({r["split_group"] for rows in selections for r in rows})!=sum(map(len,selections)):
        raise ValueError("Resampling strata overlap")
    tr=np.asarray([i for i,r in enumerate(old) if r["role"]=="train"])
    y_train=np.asarray([tree.leaves.index(old[i]["label"]) for i in tr])
    by_class=[np.flatnonzero(y_train==k) for k in range(len(tree.leaves))]
    confirmation=None
    extra=[]
    if a.confirmation:
        confirmation=json.loads((a.confirmation/"summary.json").read_bytes())
        table=a.confirmation/"selection.csv"
        extra=read_csv(table)
        if (confirmation["status"]!="complete" or digest(table)!=confirmation["selection_sha256"] or
                len(extra)!=confirmation["selected"] or any(r["role"]!="dev" for r in extra) or
                {r["image_id"] for r in extra} & {r["image_id"] for cohort in selections for r in cohort}):
            raise ValueError("Additional development batch changed or overlaps the original pilots")
        if any(r["pooled_two_natural"]["n"]!=len(new)+len(extra)
               for m in confirmation["models"].values() for r in m["readouts"].values()):
            raise ValueError("Pooled resampling requires distinct eligible groups from the original batch")
    truth=np.asarray([tree.leaves.index(r["label"]) for r in fit+new+extra])
    n=len(fit)
    reps=cfg["bootstrap_replicates"]
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    draws=[(np.concatenate([rng.choice(g,len(g),replace=True) for g in by_class]),
            rng.integers(0,len(fit),len(fit)),len(fit)+rng.integers(0,len(new),len(new))) for _ in range(reps)]
    if confirmation:
        draws=[(train,train_risk,np.r_[ev,n+len(new)+rng.integers(0,len(extra),len(extra))])
               for train,train_risk,ev in draws]
    args_out=a.output
    args_out.mkdir(exist_ok=False,parents=True)
    result={"scope":"development bootstrap under empirical cohort distributions",
            "config_sha256":digest(a.config),"replicates":reps,"seed":cfg["bootstrap_seed"],
            "resampling":"paired across detectors/encoders, stratified replacement within each support class, independent replacement of fit and dev groups",
            "models":{}}
    if confirmation:
        result.update(evaluation_batches=[len(new),len(extra)],confirmation_sha256=confirmation["selection_sha256"],
                      evaluation_resampling="resample separately within each of the two natural development batches")
    for name,ref in sources[2][1]["models"].items():
        paths=[p/f"{name}.npz" for p,_ in sources]
        if list(map(digest,paths))!=[ref["fit_cache"],ref["support_cache"],ref["cache_sha256"]]:
            raise ValueError("Feature cache changed")
        with np.load(paths[0],allow_pickle=False) as u, np.load(paths[1],allow_pickle=False) as v, np.load(paths[2],allow_pickle=False) as w:
            support=v["image"][tr].astype(np.float64)
            x=np.vstack([u["image_features"],w["image_features"]])
            text=u["text_features"]
        if confirmation:
            new_cache=a.confirmation/f"{name}.npz"
            if digest(new_cache)!=confirmation["models"][name]["cache_sha256"]:
                raise ValueError("Additional cached features changed")
            with np.load(new_cache,allow_pickle=False) as added:
                x=np.vstack([x,added["image_features"]])
        model=sources[0][1]["models"][name]
        values={k:[] for k in cfg["readouts"]}
        attempts={k:{"single_class_fit":0,"single_class_evaluation":0} for k in cfg["readouts"]}
        for i,(support_ids,fit_ids,eval_ids) in enumerate(draws):
            centers=np.stack([support[support_ids[y_train[support_ids]==k]].mean(0) for k in range(len(tree.leaves))])
            centers/=np.linalg.norm(centers,axis=1,keepdims=True)
            by=signals(x,text,model["text_node_order"],tree.leaves,tree,model["logit_scale"],centers)
            for readout,(pred,features) in by.items():
                wrong=pred!=truth
                bad_train=wrong[fit_ids]
                bad_eval=wrong[eval_ids]
                if np.unique(bad_train).size<2:
                    attempts[readout]["single_class_fit"]+=1
                    continue
                if np.unique(bad_eval).size<2:
                    attempts[readout]["single_class_evaluation"]+=1
                    continue
                scores=[]
                for keys in (cfg["base"],cfg["base"]+[cfg["extra"]]):
                    mat=np.column_stack([features[k] for k in keys])
                    score,_=logistic_predict(mat[fit_ids],bad_train,mat[eval_ids],cfg["l2_penalty"])
                    scores.append(error_auc(bad_eval,score))
                values[readout].append(scores[1]-scores[0])
            if (i+1)%500==0:
                print(f"{name} {i+1}/{reps}",flush=True)
        result["models"][name]={}
        for method,arr in values.items():
            if not arr:
                raise ValueError("No estimable resamples")
            observed=(confirmation["models"][name]["readouts"][method]["pooled_two_natural"]["difference"]
                      if confirmation else ref["readouts"][method]["auroc_difference"])
            result["models"][name][method]={"observed_delta":observed,
                "valid_replicates":len(arr),"invalid":attempts[method],"median":float(np.median(arr)),
                "percentiles_2_5_97_5":np.percentile(arr,[2.5,97.5]).tolist()}
    result.update(status="complete",seconds=round(time.perf_counter()-start,3))
    (args_out/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
