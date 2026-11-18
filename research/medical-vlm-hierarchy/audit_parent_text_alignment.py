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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("increment","pilot","balanced","output"):
        ap.add_argument("--"+k,type=Path,required=True)
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
    truth=np.asarray([leaves.index(r["label"]) for r in fit+new+[old[i] for i in ev]])
    support_y=np.asarray([leaves.index(old[i]["label"]) for i in tr])
    n=len(fit)
    out={"scope":"posthoc parent-text semantic correspondence control; no permutation selection",
         "fit_groups":n,"new_dev_groups":len(new),"balanced_dev_groups":len(ev),
         "config_sha256":src["config_sha256"],"selection_sha256":src["selection_sha256"],"models":{}}
    for name,report in src["models"].items():
        paths=[d/f"{name}.npz" for d in (a.pilot,a.balanced,a.increment)]
        if list(map(digest,paths))!=[report["fit_cache"],report["support_cache"],report["cache_sha256"]]:
            raise ValueError("Changed embedding cache")
        with np.load(paths[0],allow_pickle=False) as u, np.load(paths[1],allow_pickle=False) as v, np.load(paths[2],allow_pickle=False) as w:
            train=v["image"][tr].astype(np.float64)
            centers=np.stack([train[support_y==i].mean(0) for i in range(len(leaves))])
            centers/=np.linalg.norm(centers,axis=1,keepdims=True)
            x=np.vstack([u["image_features"],w["image_features"],v["image"][ev]])
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
                batches={"natural":slice(0,len(new)),"balanced":slice(len(new),None)}
                ref={label:error_auc(error[n:][s],prob[s]) for label,s in batches.items()}
                deltas={label:{} for label in batches}
                for perm in itertools.permutations(identity):
                    extra=js(pooled[readout],direct[:,perm])
                    design=np.column_stack((base,extra))
                    risk,_=logistic_predict(design[:n],error[:n],design[n:],cfg["l2_penalty"])
                    if perm==identity and not np.allclose(risk[:len(new)],w[f"{readout}_augmented"],atol=1e-9,rtol=0):
                        raise ValueError("Genuine correspondence differs from original risk prediction")
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
            out["models"][name]=result
    out.update(status="complete",seconds=round(time.perf_counter()-start,3))
    a.output.mkdir(exist_ok=False,parents=True)
    (a.output/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
