"""Evaluate the original frozen HAM error detectors once on a separately prespecified development batch."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from extract_features import encode_images, load_encoder, read_inputs
from ham_dev_pilot import digest, error_auc
from ham_error_increment import read_csv, signals
from selective_core import Taxonomy
from verify_ham_dev_pilot import rank_auc


def parameters(info):
    return {name:{method:{variant:{field:model[variant][field] for field in
                                   ("features","coefficients","fit_mean","fit_scale")}
                          for variant in ("base","augmented")} for method,model in entry["readouts"].items()}
            for name,entry in info["models"].items()}


def predict(features, fitted):
    x=np.column_stack([features[k] for k in fitted["features"]])
    z=(x-np.asarray(fitted["fit_mean"]))/np.asarray(fitted["fit_scale"])
    coef=np.asarray(fitted["coefficients"])
    return 1/(1+np.exp(-np.clip(coef[0]+z@coef[1:],-700,700)))


def measure(wrong, base, augmented):
    a,b=(error_auc(wrong,s) for s in (base,augmented))
    if a is not None and any(abs(value-rank_auc(wrong,score))>1e-12
                             for value,score in ((a,base),(b,augmented))):
        raise ValueError("Pair-count and rank-sum AUROCs differ")
    return {"n":len(wrong),"errors":int(wrong.sum()),"base_auroc":a,"augmented_auroc":b,
            "difference":None if a is None else b-a,
            "base_brier":float(np.mean((base-wrong)**2)) if len(base) else None,
            "augmented_brier":float(np.mean((augmented-wrong)**2)) if len(base) else None}


def choose(rows, history, cfg):
    lookup={r["image_id"]:r for r in rows}
    used=Counter(lookup[r["image_id"]]["split_group"] for cohort in history for r in cohort)
    eligible={}
    for r in sorted(rows,key=lambda item:item["image_id"]):
        g=r["split_group"]
        if r["split"]==cfg["cohort_role"] and r["representative"]=="1" and g not in used:
            eligible.setdefault(g,r)
    def order(g):
        return hashlib.sha256((cfg["salt"]+"|"+g).encode()).digest(),g
    selected=[eligible[g] for g in sorted(eligible,key=order)[:cfg["n"]]]
    if len(selected)!=cfg["n"]:
        raise ValueError("Too few previously unused development groups")
    # Historical pooling is descriptive and omits first-batch groups that collide with any past row.
    pool_old=np.asarray([lookup[r["image_id"]]["split"]=="dev" and used[lookup[r["image_id"]]["split_group"]]==1
                         for r in history[2]])
    return selected,used,pool_old,len(eligible)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ("audit","splits","sensitivity","pilot","balanced","increment","openai-weights","biomed-dir","output"):
        ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/ham-g1-confirmation-v1.json")
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    ap.add_argument("--batch-size",type=int,default=8)
    a=ap.parse_args()
    cfg=json.loads(a.config.read_bytes())
    tree=Taxonomy(json.loads(a.taxonomy.read_bytes())["parent"])
    paths=[a.pilot,a.balanced,a.increment]
    infos=[json.loads((p/"summary.json").read_bytes()) for p in paths]
    if (any(p["status"]!="complete" for p in infos) or infos[0]["config_sha256"]!=digest(a.taxonomy)
            or [p["selection_sha256"] for p in infos]!=cfg["previous_selections"]
            or [digest(p/"selection.csv") for p in paths]!=cfg["previous_selections"] or a.batch_size<1):
        raise ValueError("Historical selection, taxonomy or batch size differs")
    fitted=parameters(infos[2])
    param_sha=hashlib.sha256(json.dumps(fitted,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if param_sha!=cfg["risk_parameters_sha256"]:
        raise ValueError("Historical fitted risk models differ from registration")
    audit,rows=read_inputs(a.audit)
    seed=str(cfg["split_seed"])
    split=json.loads((a.splits/"audit.json").read_bytes())
    sensitivity=json.loads((a.sensitivity/"summary.json").read_bytes())
    mask_file=a.sensitivity/"sensitivity_groups.csv"
    if (split["split_protocol"]!=cfg["protocol"] or audit["split_sha256"]!=split["splits"][seed]["sha256"]
            or audit["split_sha256"]!=digest(a.splits/f"split_seed{seed}.csv")
            or sensitivity["split_sha256"][seed]!=audit["split_sha256"] or digest(mask_file)!=sensitivity["sensitivity_groups_sha256"]):
        raise ValueError("Development selection needs the released split and matching sensitivity mask")
    history=[read_csv(p/"selection.csv") for p in paths]
    evaluation,reserved,pool_old,n_eligible=choose(rows,history,cfg)
    blacklist={r["split_group"] for r in read_csv(mask_file) if r["excluded"]=="1"}
    keep=np.array([r["split_group"] not in blacklist for r in evaluation])
    support_ids=[i for i,r in enumerate(history[1]) if r["role"]=="train"]
    support_y=np.array([tree.leaves.index(history[1][i]["label"]) for i in support_ids])
    old_y=np.array([tree.leaves.index(r["label"]) for r in history[2]])
    y=np.array([tree.leaves.index(r["label"]) for r in evaluation])
    rng=np.random.default_rng(cfg["bootstrap_seed"])
    draws=rng.integers(0,len(y),(cfg["replicates"],len(y)))
    start=time.perf_counter()
    a.output.mkdir(exist_ok=False,parents=True)
    selection=a.output/"selection.csv"
    with selection.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(("image_id","split_group","role","label"))
        writer.writerows((r["image_id"],r["split_group"],r["split"],r["label"]) for r in evaluation)
    report={"scope":"additional label-blind development batch, no formal test or calibration scores",
            "config_sha256":digest(a.config),"risk_parameters_sha256":param_sha,"split_sha256":audit["split_sha256"],
            "sensitivity_sha256":digest(mask_file),"selection_sha256":digest(selection),"historical_groups_reserved":len(reserved),
            "eligible_groups":n_eligible,"selected":len(y),"retained_sensitivity":int(keep.sum()),
            "new_class_counts":dict(Counter(r["label"] for r in evaluation)),"models":{}}
    try:
        for model,ref in infos[2]["models"].items():
            now=time.perf_counter()
            files=[p/f"{model}.npz" for p in paths]
            if list(map(digest,files))!=[ref["fit_cache"],ref["support_cache"],ref["cache_sha256"]]:
                raise ValueError("Historical model features changed")
            info={"device":a.device,"batch_size":a.batch_size,"readouts":{}}
            weight=a.openai_weights if model=="clip" else a.biomed_dir/"open_clip_pytorch_model.bin"
            model_cfg=None if model=="clip" else a.biomed_dir/"open_clip_config.json"
            encoder,prep,_=load_encoder(model,weight,a.device,model_cfg,info)
            images=np.concatenate([encode_images(encoder,prep,evaluation[i:i+a.batch_size],a.device)
                                   for i in range(0,len(y),a.batch_size)])
            stored={"image_features":images}
            with np.load(files[0],allow_pickle=False) as first, np.load(files[1],allow_pickle=False) as bal, np.load(files[2],allow_pickle=False) as old:
                support=bal["image"][support_ids].astype(np.float64)
                centers=np.stack([support[support_y==i].mean(0) for i in range(len(tree.leaves))])
                centers/=np.linalg.norm(centers,axis=1,keepdims=True)
                meta=infos[0]["models"][model]
                rows_by=signals(np.vstack([old["image_features"],images]),first["text_features"],meta["text_node_order"],
                                tree.leaves,tree,meta["logit_scale"],centers)
                n=len(old_y)
                for name,(pred,features) in rows_by.items():
                    wrong=pred!=np.r_[old_y,y]
                    base,aug=(predict(features,fitted[model][name][k]) for k in ("base","augmented"))
                    if (not np.array_equal(wrong[:n],old[name+"_error"]) or
                            any(not np.allclose(s[:n],old[name+"_"+v],rtol=0,atol=1e-12)
                                for s,v in ((base,"base"),(aug,"augmented")))):
                        raise ValueError("Deployed detector did not reproduce the first independent batch")
                    new=wrong[n:]
                    variants={"primary":np.r_[np.zeros(n,dtype=bool),np.ones(len(y),dtype=bool)],
                              "new_sensitivity":np.r_[np.zeros(n,dtype=bool),keep],
                              "pooled_two_natural":np.r_[pool_old,np.ones(len(y),dtype=bool)]}
                    result={k:measure(wrong[idx],base[idx],aug[idx]) for k,idx in variants.items()}
                    diffs=[error_auc(new[s],aug[n:][s])-error_auc(new[s],base[n:][s]) for s in draws
                           if np.any(new[s]) and not np.all(new[s])]
                    result["conditional_bootstrap"]={"valid":len(diffs),"replicates":len(draws),
                        "percentiles_2_5_97_5":np.percentile(diffs,[2.5,97.5]).tolist() if diffs else None}
                    info["readouts"][name]=result
                    stored.update({name+"_error":new,name+"_base":base[n:],name+"_augmented":aug[n:]})
            dest=a.output/f"{model}.npz"
            np.savez(dest,**stored)
            info.update(cache_sha256=digest(dest),seconds=round(time.perf_counter()-now,3))
            report["models"][model]=info
            del encoder
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (a.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
