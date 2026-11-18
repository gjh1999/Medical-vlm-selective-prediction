"""Independently check the crossed design's allocation, branch signals, optimizer and resulting metrics."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from extract_features import read_inputs
from ham_dev_pilot import digest
from ham_error_increment import read_csv
from selective_core import Taxonomy
from verify_error_increment import lbfgs
from verify_ham_dev_pilot import rank_auc


def verify_allocation(rows, histories, actual, cfg, leaves):
    by_id={r["image_id"]:r for r in rows}
    reserved={by_id[r["image_id"]]["split_group"] for cohort in histories for r in cohort}
    groups=itertools.groupby(sorted(rows,key=lambda r:(r["split_group"],r["image_id"])),key=lambda r:r["split_group"])
    eligible=[next((r for r in members if r["representative"]=="1"),None) for g,members in groups if g not in reserved]
    winners=[]
    for name,quota,role,leaf in [("support",cfg["new_support_per_leaf"],"train",y) for y in leaves]+[
            ("fit",cfg["new_risk_fit"],"dev",None)]:
        data=[r for r in eligible if r and r["split"]==role and (leaf is None or r["label"]==leaf)]
        salt=cfg[name+"_salt"].encode()
        scores=[int(hashlib.sha256(salt+b"|"+r["split_group"].encode()).hexdigest(),16) for r in data]
        selected=np.argsort(np.array(scores,dtype=object),kind="stable")[:quota]
        winners.extend(data[i]["image_id"] for i in selected)
    if winners!=[r["image_id"] for r in actual] or len({by_id[k]["split_group"] for k in winners})!=len(winners):
        raise ValueError("Independent hash ranking or group inclusion differs")


def features(images, text, centers, nodes, tree, scale):
    x=torch.tensor(images,dtype=torch.float64)
    t=torch.tensor(text,dtype=torch.float64)
    leaves=[nodes.index(k) for k in tree.leaves]
    roots=sorted(k for k,v in tree.parent.items() if v==tree.root)
    parents=[nodes.index(k) for k in roots]
    proto=normalize(t.mean(0),dim=1)
    direct=(scale*x@proto[parents].T).softmax(1)
    templates=torch.stack([(scale*x@one[leaves].T).softmax(1) for one in t])
    incidence=torch.tensor([[y in tree.desc[k] for k in roots] for y in tree.leaves],dtype=torch.float64)
    def divergence(a,b):
        mid=(a+b)*.5
        return .5*(((a/mid).log()*a).sum(1)+((b/mid).log()*b).sum(1))
    result={}
    for name,vectors in (("frozen_text",proto[leaves]),("cosine_centroid",centers)):
        p=(scale*x@vectors.T).softmax(1)
        top=p.topk(2,dim=1).values
        matrix=torch.stack((top[:,1]-top[:,0],-torch.sum(p*p.log(),1),
                            divergence(*templates),divergence(p@incidence,direct)),1)
        result[name]=(p.argmax(1).numpy(),matrix.numpy())
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("audit","crossed","pilot","balanced","increment","confirmation","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/ham-g1-crossed-fit-v1.json")
    ap.add_argument("--rule",type=Path,default=Path(__file__).parent/"configs/ham-error-increment-v1.json")
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    a=ap.parse_args()
    start=time.perf_counter()
    a.output.mkdir(parents=True,exist_ok=False)
    out={"method":"independent group ranking; torch float64 signals, L-BFGS fit and Mann-Whitney ranks",
         "checks":0,"max_abs_auroc_difference":0.,"max_abs_brier_difference":0.}
    try:
        torch.set_num_threads(1)
        source=json.loads((a.crossed/"summary.json").read_bytes())
        cfg=json.loads(a.config.read_bytes())
        rule=json.loads(a.rule.read_bytes())
        tree=Taxonomy(json.loads(a.taxonomy.read_bytes())["parent"])
        paths=(a.pilot,a.balanced,a.increment,a.confirmation)
        infos=[json.loads((p/"summary.json").read_bytes()) for p in paths]
        if (digest(a.config)!=source["config_sha256"] or source["status"]!="complete"
                or digest(a.taxonomy)!=infos[0]["config_sha256"] or digest(a.rule)!=infos[2]["config_sha256"]
                or [digest(p/"selection.csv") for p in paths]!=cfg["previous_selections"]
                or digest(a.crossed/"selection.csv")!=source["selection_sha256"]):
            raise ValueError("Changed experiment provenance")
        audit,rows=read_inputs(a.audit)
        if audit["split_sha256"]!=cfg["split_sha256"] or rule["base"]+ [rule["extra"]]!=[
                "negative_margin","entropy","prompt_js","parent_js"]:
            raise ValueError("Unexpected split or feature definition")
        histories=[read_csv(p/"selection.csv") for p in paths]
        fresh=read_csv(a.crossed/"selection.csv")
        verify_allocation(rows,histories,fresh,cfg,tree.leaves)
        oi=[i for i,r in enumerate(histories[1]) if r["role"]=="train"]
        bi=[i for i,r in enumerate(histories[1]) if r["role"]=="dev"]
        ni=[i for i,r in enumerate(fresh) if r["role"]=="train"]
        fi=[i for i,r in enumerate(fresh) if r["role"]=="dev"]
        def truth(part):
            return np.array([tree.leaves.index(r["label"]) for r in part])
        evaluation=histories[2]+histories[3]+[histories[1][i] for i in bi]
        first_end=len(histories[2])
        second_end=first_end+len(histories[3])
        groups={"first":slice(0,first_end),"second":slice(first_end,second_end),"pooled":slice(0,second_end),
                "balanced":slice(second_end,None)}
        out["selected_rows_verified"]=len(fresh)
        for model,reference in source["models"].items():
            files=[p/f"{model}.npz" for p in (*paths,a.crossed)]
            if list(map(digest, files))!=[infos[2]["models"][model]["fit_cache"],infos[2]["models"][model]["support_cache"],
                    infos[2]["models"][model]["cache_sha256"],infos[3]["models"][model]["cache_sha256"],reference["cache_sha256"]]:
                raise ValueError("Cached embeddings changed")
            with (np.load(files[0],allow_pickle=False) as u,np.load(files[1],allow_pickle=False) as v,
                  np.load(files[2],allow_pickle=False) as w,np.load(files[3],allow_pickle=False) as z,
                  np.load(files[4],allow_pickle=False) as f):
                eval_x=np.concatenate([w["image_features"],z["image_features"],v["image"][bi]])
                for sup,sx,sy in (("original_support",v["image"][oi],truth([histories[1][i] for i in oi])),
                                  ("new_support",f["image_features"][ni],truth([fresh[i] for i in ni]))):
                    tensor=torch.tensor(sx,dtype=torch.float64)
                    centers=normalize(torch.stack([tensor[sy==i].mean(0) for i in range(len(tree.leaves))]),dim=1)
                    for fit,fx,fy in (("original_fit",u["image_features"],truth(histories[0])),
                                      ("new_fit",f["image_features"][fi],truth([fresh[i] for i in fi]))):
                        estimates=features(np.concatenate([fx,eval_x]),u["text_features"],centers,
                                           infos[0]["models"][model]["text_node_order"],tree,infos[0]["models"][model]["logit_scale"])
                        for method,(prediction,matrix) in estimates.items():
                            errors=prediction!=np.r_[fy,truth(evaluation)]
                            train=matrix[:len(fy)]
                            sd=train.std(0)
                            sd[sd<1e-12]=1.
                            scaled=(matrix-train.mean(0))/sd
                            results=reference["cells"][sup+"/"+fit][method]
                            for variant,dim in (("base",3),("augmented",4)):
                                beta=lbfgs(scaled[:len(fy),:dim],errors[:len(fy)],rule["l2_penalty"])
                                score=1/(1+np.exp(-(beta[0]+scaled[len(fy):,:dim]@beta[1:])))
                                for name,sl in groups.items():
                                    y=errors[len(fy):][sl]
                                    res=results[name]
                                    if len(y)!=res["n"] or int(y.sum())!=res["errors"]:
                                        raise ValueError("Independent readout error counts differ")
                                    au=abs(rank_auc(y,score[sl])-res[variant+"_auroc"])
                                    bs=abs(float(np.mean((score[sl]-y)**2))-res[variant+"_brier"])
                                    out["max_abs_auroc_difference"]=max(out["max_abs_auroc_difference"],au)
                                    out["max_abs_brier_difference"]=max(out["max_abs_brier_difference"],bs)
                                    out["checks"]+=1
        if out["max_abs_auroc_difference"]>1e-12 or out["max_abs_brier_difference"]>1e-6:
            raise ValueError("Independent estimates differ from the main four-cell results")
        out["status"]="complete"
    except Exception as exc:
        out.update(status="failed",error=str(exc))
        raise
    finally:
        out["seconds"]=round(time.perf_counter()-start,3)
        (a.output/"summary.json").write_text(json.dumps(out,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
