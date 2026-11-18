"""Evaluate whether parent disagreement adds error-ranking information on fresh HAM development groups."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from extract_features import encode_images, load_encoder, read_inputs
from ham_dev_pilot import digest, error_auc, js, softmax
from selective_core import Taxonomy


def read_csv(file):
    with file.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def logistic_predict(fit, target, evaluation, penalty):
    if len(np.unique(target))!=2:
        raise ValueError("Fitting sample needs errors and correct classifications")
    mean=fit.mean(0)
    sd=fit.std(0)
    sd[sd<1e-12]=1.
    x=np.column_stack([np.ones(len(fit)),(fit-mean)/sd])
    x_eval=np.column_stack([np.ones(len(evaluation)),(evaluation-mean)/sd])
    beta=np.zeros(x.shape[1])
    regularizer=np.eye(len(beta))*penalty
    regularizer[0,0]=0.
    def objective(coeff):
        z=x@coeff
        return (np.logaddexp(0,z)-target*z).sum()+.5*coeff@regularizer@coeff
    for _ in range(100):
        p=1/(1+np.exp(-np.clip(x@beta,-700,700)))
        grad=x.T@(p-target)+regularizer@beta
        if np.max(np.abs(grad))<1e-8:
            break
        hessian=x.T@((p*(1-p))[:,None]*x)+regularizer
        step=np.linalg.solve(hessian,grad)
        rate=1.
        while objective(beta-rate*step)>objective(beta)+1e-12:
            rate*=.5
            if rate<1e-10:
                raise ValueError("Logistic fit failed to descend")
        beta-=rate*step
    else:
        raise ValueError("Logistic fit did not converge")
    fitted={"coefficients":beta.tolist(),"fit_mean":mean.tolist(),"fit_scale":sd.tolist()}
    return 1/(1+np.exp(-np.clip(x_eval@beta,-700,700))),fitted


def signals(image,text,nodes,leaves,tree,scale,centers,return_parent_distributions=False):
    text=text.astype(np.float64)
    x=image.astype(np.float64)
    avg=text.mean(0)
    avg/=np.linalg.norm(avg,axis=1,keepdims=True)
    leaf_idx=[nodes.index(k) for k in leaves]
    roots=sorted(k for k,parent in tree.parent.items() if parent==tree.root)
    direct=softmax(x@avg[[nodes.index(k) for k in roots]].T*scale)
    template=softmax((x@text[:,leaf_idx,:].transpose(0,2,1))*scale)
    prompt=js(*template)
    out={}
    pooled_by_readout={}
    for name,matrix in [("frozen_text",avg[leaf_idx]),("cosine_centroid",centers)]:
        prob=softmax(x@matrix.T*scale)
        pooled=np.column_stack([prob[:,[leaves.index(k) for k in sorted(tree.desc[parent])]].sum(1) for parent in roots])
        pooled_by_readout[name]=pooled
        sorted_p=np.sort(prob,axis=1)
        out[name]=(prob.argmax(1),{"negative_margin":sorted_p[:,-2]-sorted_p[:,-1],
                   "entropy":-(prob*np.log(prob)).sum(1),"prompt_js":prompt,"parent_js":js(pooled,direct)})
    if return_parent_distributions:
        return out,direct,pooled_by_readout,roots
    return out


def metrics(target,score):
    return {"error_auroc":error_auc(target,score),"brier":float(np.mean((score-target)**2))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ["audit","pilot","balanced","openai-weights","biomed-dir","output"]:
        ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/ham-error-increment-v1.json")
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    ap.add_argument("--batch-size",type=int,default=8)
    args=ap.parse_args()
    start=time.perf_counter()
    conf=json.loads(args.config.read_bytes())
    pilot=json.loads((args.pilot/"summary.json").read_bytes())
    balanced=json.loads((args.balanced/"summary.json").read_bytes())
    tree=Taxonomy(json.loads(args.taxonomy.read_bytes())["parent"])
    leaves=tree.leaves
    audit,rows=read_inputs(args.audit)
    if (audit["split_sha256"]!=conf["split_sha256"] or digest(args.taxonomy)!=pilot["config_sha256"]
            or any(r["status"]!="complete" for r in [pilot,balanced]) or args.batch_size<1):
        raise ValueError("Invalid input provenance or batch size")
    lookup={r["image_id"]:r for r in rows}
    fit=read_csv(args.pilot/"selection.csv")
    old=read_csv(args.balanced/"selection.csv")
    for source,previous,summary in [(args.pilot,fit,pilot),(args.balanced,old,balanced)]:
        if digest(source/"selection.csv")!=summary["selection_sha256"]:
            raise ValueError("Previous selection changed")
        if any(lookup[r["image_id"]]["split"]!=r["role"] or any(
                r[k]!=lookup[r["image_id"]][k] for k in ("label","split_group")) for r in previous):
            raise ValueError("Previous groups or labels changed in the new protocol")
    if any(r["role"]!="dev" for r in fit):
        raise ValueError("Risk fitting must use development groups")
    reserved={r["split_group"] for r in fit+old}
    eligible={}
    for r in sorted(rows,key=lambda i:i["image_id"]):
        if r["split"]=="dev" and r["representative"]=="1" and r["split_group"] not in reserved:
            eligible.setdefault(r["split_group"],r)
    selected=sorted(eligible,key=lambda k:hashlib.sha256(f'{conf["salt"]}|{k}'.encode()).digest())[:conf["eval_size"]]
    if len(selected)!=conf["eval_size"]:
        raise ValueError("Too few new development groups")
    evaluation=[eligible[g] for g in selected]
    y_fit=np.array([leaves.index(r["label"]) for r in fit])
    y_eval=np.array([leaves.index(r["label"]) for r in evaluation])
    train_idx=[i for i,r in enumerate(old) if r["role"]=="train"]
    y_train=np.array([leaves.index(old[i]["label"]) for i in train_idx])
    args.output.mkdir(exist_ok=False,parents=True)
    sel=args.output/"selection.csv"
    with sel.open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["image_id","split_group","role","label"])
        w.writerows((r["image_id"],r["split_group"],r["split"],r["label"]) for r in evaluation)
    report={"scope":"frozen error-detector comparison on new development groups","config_sha256":digest(args.config),
            "manifest_sha256":audit["manifest_sha256"],"split_sha256":audit["split_sha256"],
            "selection_sha256":digest(sel),"n_fit":len(fit),"n_eval":len(evaluation),
            "n_leaf_support":len(train_idx),"reserved_groups":len(reserved),"overlap_with_reserved":0,
            "eval_class_counts":{k:int((y_eval==i).sum()) for i,k in enumerate(leaves)},
            "device":args.device,"batch_size":args.batch_size,"models":{}}
    try:
        for name,base in pilot["models"].items():
            t=time.perf_counter()
            source_file=args.pilot/f"{name}.npz"
            bal_file=args.balanced/f"{name}.npz"
            if (digest(source_file)!=base["cache_sha256"] or digest(bal_file)!=balanced["models"][name]["cache_sha256"]
                    or base["leaf_order"]!=leaves):
                raise ValueError("Prior feature cache changed")
            info={"fit_cache":base["cache_sha256"],"support_cache":balanced["models"][name]["cache_sha256"],"readouts":{}}
            weights=args.openai_weights if name=="clip" else args.biomed_dir/"open_clip_pytorch_model.bin"
            cfg=None if name=="clip" else args.biomed_dir/"open_clip_config.json"
            model,prep,_=load_encoder(name,weights,args.device,cfg,info)
            x_eval=np.concatenate([encode_images(model,prep,evaluation[i:i+args.batch_size],args.device)
                                   for i in range(0,len(evaluation),args.batch_size)])
            with np.load(source_file,allow_pickle=False) as first, np.load(bal_file,allow_pickle=False) as second:
                support=second["image"][train_idx].astype(np.float64)
                centers=np.stack([support[y_train==i].mean(0) for i in range(len(leaves))])
                centers/=np.linalg.norm(centers,axis=1,keepdims=True)
                all_signals=signals(np.vstack([first["image_features"],x_eval]),first["text_features"],
                                    base["text_node_order"],leaves,tree,base["logit_scale"],centers)
                if not np.array_equal(first["leaf_probs"].argmax(1),all_signals["frozen_text"][0][:len(fit)]):
                    raise ValueError("Shared text readout differs from the source pilot")
            local={"image_features":x_eval}
            for readout in conf["readouts"]:
                pred,feature=all_signals[readout]
                bad_fit=pred[:len(fit)]!=y_fit
                bad_eval=pred[len(fit):]!=y_eval
                if len(np.unique(bad_eval))!=2:
                    raise ValueError("Evaluation has no estimable error AUROC")
                stats={"fit_errors":int(bad_fit.sum()),"eval_errors":int(bad_eval.sum())}
                scores={}
                for variant,keys in [("base",conf["base"]),("augmented",conf["base"]+[conf["extra"]])]:
                    matrix=np.column_stack([feature[k] for k in keys])
                    score,fitted=logistic_predict(matrix[:len(fit)],bad_fit,matrix[len(fit):],conf["l2_penalty"])
                    scores[variant]=score
                    local[f"{readout}_{variant}"]=score
                    stats[variant]={**metrics(bad_eval,score),"features":keys,**fitted}
                local[f"{readout}_error"]=bad_eval
                diff=[]
                rng=np.random.default_rng(conf["bootstrap_seed"])
                for _ in range(conf["bootstrap_replicates"]):
                    draws=rng.integers(0,len(bad_eval),len(bad_eval))
                    y=bad_eval[draws]
                    if y.any() and not y.all():
                        diff.append(error_auc(y,scores["augmented"][draws])-error_auc(y,scores["base"][draws]))
                stats["auroc_difference"]=stats["augmented"]["error_auroc"]-stats["base"]["error_auroc"]
                stats["percentiles_2_5_97_5"]=np.percentile(diff,[2.5,97.5]).tolist()
                stats["valid_bootstraps"]=len(diff)
                info["readouts"][readout]=stats
            cache=args.output/f"{name}.npz"
            np.savez(cache,**local)
            info.update(cache_sha256=digest(cache),seconds=round(time.perf_counter()-t,3))
            report["models"][name]=info
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
