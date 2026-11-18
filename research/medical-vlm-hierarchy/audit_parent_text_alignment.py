"""Compare the genuine parent-text alignment to every reassignment under the same risk-fitting protocol."""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from ham_dev_pilot import digest, error_auc, js
from ham_error_increment import logistic_predict, read_csv, signals
from selective_core import Taxonomy


def image_controls(base, error, pooled, direct, branch_truth, fit_n, batches, penalty):
    """Remove or shuffle the image-specific direct branch signal while retaining one extra predictor."""
    from audit_ham_temperature import fit_inverse_temperature
    from ham_dev_pilot import softmax

    def evaluate(extra):
        design=np.column_stack((base,extra)) if extra is not None else base
        score,_=logistic_predict(design[:fit_n],error[:fit_n],design[fit_n:],penalty)
        return {name:error_auc(error[fit_n:][s],score[s]) for name,s in batches.items()}
    baseline=evaluate(None)
    def increment(extra):
        auc=evaluate(extra)
        return {name:auc[name]-baseline[name] for name in batches}
    uniform=np.full_like(direct,1/direct.shape[1])
    average=np.broadcast_to(direct[:fit_n].mean(0),direct.shape)
    beta_direct=fit_inverse_temperature(np.log(direct[:fit_n]),branch_truth[:fit_n])
    beta_pooled=fit_inverse_temperature(np.log(pooled[:fit_n]),branch_truth[:fit_n])
    normalized=js(softmax(beta_pooled*np.log(pooled)),softmax(beta_direct*np.log(direct)))
    genuine=increment(js(pooled,direct))
    report={"genuine":genuine,"uniform":increment(js(pooled,uniform)),"fit_mean":increment(js(pooled,average)),
            "fit_branch_inverse_temperatures":{"direct":beta_direct,"pooled":beta_pooled},
            "branch_temperatures_fitted":increment(normalized),"shuffle_seed":1911,"shuffle_cycles":200}
    # Each batch uses an independent random cycle. This keeps the direct vectors and their
    # marginal distribution fixed and makes every row receive another image's vector.
    groups=[np.arange(fit_n)]+[np.arange(fit_n,len(error))[s] for s in batches.values()]
    if any(len(g)<2 for g in groups):
        raise ValueError("Each batch needs two distinct images for this control")
    rng=np.random.default_rng(report["shuffle_seed"])
    values={k:[] for k in batches}
    for _ in range(report["shuffle_cycles"]):
        order=np.arange(len(error))
        for group in groups:
            cycle=rng.permutation(group)
            order[cycle]=np.roll(cycle,1)
        if np.any(order==np.arange(len(error))):
            raise ValueError("A shuffled vector stayed paired with its original image")
        delta=increment(js(pooled,direct[order]))
        for k,v in delta.items():
            values[k].append(v)
    report["image_shuffle"]={k:{"median":float(np.median(v)),"central_95_spread":np.percentile(v,[2.5,97.5]).tolist(),
                                 "cycles_at_least_genuine":sum(s>=genuine[k]-1e-12 for s in v)} for k,v in values.items()}
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("increment","pilot","balanced","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--confirmation",type=Path,help="Add the separately prespecified second natural development batch")
    ap.add_argument("--image-controls",action="store_true",help="Add constant, shuffled-image and branch-temperature controls")
    ap.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/ham-error-increment-v1.json")
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    a=ap.parse_args()
    start=time.perf_counter()
    src=json.loads((a.increment/"summary.json").read_bytes())
    prev=json.loads((a.pilot/"summary.json").read_bytes())
    bal=json.loads((a.balanced/"summary.json").read_bytes())
    if (src["status"]!="complete" or src["config_sha256"]!=digest(a.config)
            or prev["config_sha256"]!=digest(a.taxonomy)):
        raise ValueError("Source policy configuration changed")
    cfg=json.loads(a.config.read_bytes())
    tree=Taxonomy(json.loads(a.taxonomy.read_bytes())["parent"])
    leaves=tree.leaves
    sels=[]
    for path,info in [(a.pilot,prev),(a.increment,src),(a.balanced,bal)]:
        if digest(path/"selection.csv")!=info["selection_sha256"]:
            raise ValueError("Source group selection changed")
        sels.append(read_csv(path/"selection.csv"))
    fit,new,old=sels
    if len({r["split_group"] for rows in sels for r in rows})!=sum(map(len,sels)):
        raise ValueError("Overlapping sample groups")
    tr=[i for i,r in enumerate(old) if r["role"]=="train"]
    ev=[i for i,r in enumerate(old) if r["role"]=="dev"]
    extra_rows=[]
    confirmation=None
    if a.confirmation:
        confirmation=json.loads((a.confirmation/"summary.json").read_bytes())
        extra_path=a.confirmation/"selection.csv"
        extra_rows=read_csv(extra_path)
        if (confirmation["status"]!="complete" or digest(extra_path)!=confirmation["selection_sha256"]
                or len(extra_rows)!=confirmation["selected"] or any(r["role"]!="dev" for r in extra_rows)
                or {r["image_id"] for r in extra_rows} & {r["image_id"] for cohort in sels for r in cohort}):
            raise ValueError("Invalid additional development batch")
    truth=np.asarray([leaves.index(r["label"]) for r in fit+new+[old[i] for i in ev]+extra_rows])
    support_y=np.asarray([leaves.index(old[i]["label"]) for i in tr])
    n=len(fit)
    out={"scope":"posthoc parent-text semantic correspondence control; no permutation selection",
         "fit_groups":n,"new_dev_groups":len(new),"balanced_dev_groups":len(ev),
         "config_sha256":src["config_sha256"],"selection_sha256":src["selection_sha256"],"models":{}}
    if confirmation:
        out.update(confirmation_groups=len(extra_rows),confirmation_sha256=confirmation["selection_sha256"])
    for name,report in src["models"].items():
        paths=[d/f"{name}.npz" for d in (a.pilot,a.balanced,a.increment)]
        if list(map(digest,paths))!=[report["fit_cache"],report["support_cache"],report["cache_sha256"]]:
            raise ValueError("Changed embedding cache")
        with np.load(paths[0],allow_pickle=False) as u, np.load(paths[1],allow_pickle=False) as v, np.load(paths[2],allow_pickle=False) as w:
            train=v["image"][tr].astype(np.float64)
            centers=np.stack([train[support_y==i].mean(0) for i in range(len(leaves))])
            centers/=np.linalg.norm(centers,axis=1,keepdims=True)
            batches={"natural":slice(0,len(new)),"balanced":slice(len(new),len(new)+len(ev))}
            parts=[u["image_features"],w["image_features"],v["image"][ev]]
            expected_predictions={}
            if confirmation:
                extra_cache=a.confirmation/f"{name}.npz"
                if digest(extra_cache)!=confirmation["models"][name]["cache_sha256"]:
                    raise ValueError("Additional development cache changed")
                with np.load(extra_cache,allow_pickle=False) as fresh:
                    parts.append(fresh["image_features"])
                    expected_predictions={k:fresh[k+"_augmented"] for k in cfg["readouts"]}
                batches["confirmation"]=slice(len(new)+len(ev),None)
            x=np.vstack(parts)
            inputs,direct,pooled,roots=signals(x,u["text_features"],prev["models"][name]["text_node_order"],leaves,tree,
                prev["models"][name]["logit_scale"],centers,return_parent_distributions=True)
            identity=tuple(range(len(roots)))
            coarse=[i for i,k in enumerate(roots) if len(tree.desc[k])>1]
            if len(coarse)!=2:
                raise ValueError("Coarse-only swap requires exactly two multi-leaf parents")
            swap=list(identity)
            i,j=coarse
            swap[i],swap[j]=swap[j],swap[i]
            result={}
            for readout,(pred,scores) in inputs.items():
                error=pred!=truth
                base=np.column_stack([scores[k] for k in cfg["base"]])
                prob,_=logistic_predict(base[:n],error[:n],base[n:],cfg["l2_penalty"])
                ref={label:error_auc(error[n:][s],prob[s]) for label,s in batches.items()}
                deltas={label:{} for label in batches}
                for perm in itertools.permutations(identity):
                    extra=js(pooled[readout],direct[:,perm])
                    design=np.column_stack((base,extra))
                    risk,_=logistic_predict(design[:n],error[:n],design[n:],cfg["l2_penalty"])
                    if perm==identity and not np.allclose(risk[:len(new)],w[f"{readout}_augmented"],atol=1e-9,rtol=0):
                        raise ValueError("Genuine correspondence differs from original risk prediction")
                    if (perm==identity and confirmation and not np.allclose(risk[batches["confirmation"]],
                            expected_predictions[readout],atol=1e-9,rtol=0)):
                        raise ValueError("Genuine correspondence differs from the frozen detector on the second batch")
                    for label,s in batches.items():
                        deltas[label][perm]=error_auc(error[n:][s],risk[s])-ref[label]
                result[readout]={}
                for label,values in deltas.items():
                    correct=values[identity]
                    controls=[v for perm,v in values.items() if perm!=identity]
                    result[readout][label]={"base_auroc":ref[label],"genuine_delta":correct,
                        "coarse_labels_swapped_delta":values[tuple(swap)],"other_assignments":len(controls),
                        "other_delta_min":min(controls),"other_delta_median":float(np.median(controls)),
                        "other_delta_max":max(controls),"others_at_least_genuine":sum(v>=correct-1e-12 for v in controls)}
                if a.image_controls:
                    membership=np.array([next(j for j,r in enumerate(roots) if y in tree.desc[r]) for y in leaves])
                    mechanism=image_controls(base,error,pooled[readout],direct,membership[truth],n,batches,cfg["l2_penalty"])
                    if any(abs(mechanism["genuine"][k]-result[readout][k]["genuine_delta"])>1e-12 for k in batches):
                        raise ValueError("Image-specific control differs from the genuine semantic baseline")
                    result[readout]["image_controls"]=mechanism
            out["models"][name]=result
    out.update(status="complete",seconds=round(time.perf_counter()-start,3))
    a.output.mkdir(exist_ok=False,parents=True)
    (a.output/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
