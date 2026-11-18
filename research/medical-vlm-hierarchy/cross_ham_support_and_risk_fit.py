"""Cross two independent equal-budget training supports with two independent development risk fits."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from confirm_ham_error_increment import measure, parameters
from extract_features import encode_images, load_encoder, read_inputs
from ham_dev_pilot import digest
from ham_error_increment import logistic_predict, read_csv, signals
from selective_core import Taxonomy


def select(rows, history, cfg, leaves):
    index={r["image_id"]:r for r in rows}
    used=set()
    for batch in history:
        for old in batch:
            current=index[old["image_id"]]
            if current["split"]!=old["role"] or current["label"]!=old["label"] or current["representative"]!="1":
                raise ValueError("Previously used representative, class or role changed")
            used.add(current["split_group"])
    if len(used)!=cfg["groups_reserved"] or len(used)!=sum(len(h) for h in history):
        raise ValueError("Historical groups collide or do not match registration")
    representatives={}
    for row in sorted(rows,key=lambda r:r["image_id"]):
        if row["representative"]=="1" and row["split_group"] not in used and row["split"] in ("train","dev"):
            representatives.setdefault(row["split_group"],row)
    def ordered(role,salt):
        return sorted((r for r in representatives.values() if r["split"]==role),
                      key=lambda r:(hashlib.sha256((salt+"|"+r["split_group"]).encode()).digest(),r["split_group"]))
    train=ordered("train",cfg["support_salt"])
    per_class=cfg["new_support_per_leaf"]
    support=[r for y in leaves for r in [v for v in train if v["label"]==y][:per_class]]
    fit=ordered("dev",cfg["fit_salt"])[:cfg["new_risk_fit"]]
    ids={r["split_group"] for r in support+fit}
    if (len(support)!=per_class*len(leaves) or len(fit)!=cfg["new_risk_fit"]
            or len(ids)!=len(support)+len(fit) or ids&used):
        raise ValueError("Insufficient or overlapping alternative groups")
    return support,fit


def run_cells(tree, rule, meta, supports, support_truth, images, truth, text, baseline):
    cells={}
    for support_name,x_support in supports.items():
        centers=np.stack([x_support[support_truth[support_name]==i].mean(0) for i in range(len(tree.leaves))])
        centers/=np.linalg.norm(centers,axis=1,keepdims=True)
        batches=list(images)
        counts=[len(images[k]) for k in batches]
        stops=np.cumsum([0]+counts)
        slices={name:slice(a,b) for name,a,b in zip(batches,stops[:-1],stops[1:])}
        by=signals(np.concatenate([images[b] for b in batches]),text,meta["text_node_order"],tree.leaves,
                   tree,meta["logit_scale"],centers)
        y=np.concatenate([truth[b] for b in batches])
        ev=np.r_[np.arange(len(y))[slices["first"]],np.arange(len(y))[slices["second"]],np.arange(len(y))[slices["balanced"]]]
        a=len(images["first"])
        b=a+len(images["second"])
        for fit_name in ("original_fit","new_fit"):
            key=f"{support_name}/{fit_name}"
            outcomes={}
            for name,(pred,features) in by.items():
                wrong=pred!=y
                risk=[]
                for cols in (rule["base"],rule["base"]+[rule["extra"]]):
                    matrix=np.column_stack([features[k] for k in cols])
                    fit=slices[fit_name]
                    predicted,_=logistic_predict(matrix[fit],wrong[fit],matrix[ev],rule["l2_penalty"])
                    risk.append(predicted)
                target=wrong[ev]
                out={}
                for group,batch in (("first",slice(0,a)),("second",slice(a,b)),
                                    ("pooled",slice(0,b)),("balanced",slice(b,None))):
                    out[group]=measure(target[batch],risk[0][batch],risk[1][batch])
                if key=="original_support/original_fit":
                    for i,k in enumerate(("base","augmented")):
                        if not np.allclose(risk[i][:b],baseline[name+"_"+k],rtol=0,atol=1e-10):
                            raise ValueError("Original risk predictions did not reproduce on both natural batches")
                    if not np.array_equal(target[:b],baseline[name+"_error"]):
                        raise ValueError("Original leaf errors changed")
                outcomes[name]=out
            cells[key]=outcomes
    for f in ("original_fit","new_fit"):
        first=cells["original_support/"+f]["frozen_text"]
        if first!=cells["new_support/"+f]["frozen_text"]:
            raise ValueError("Text predictions depend on image support")
    differences={}
    for method in rule["readouts"]:
        differences[method]={}
        for cohort in ("first","second","pooled","balanced"):
            aa,ab,ba,bb=[cells[k][method][cohort]["difference"] for k in
                         ("original_support/original_fit","original_support/new_fit","new_support/original_fit","new_support/new_fit")]
            differences[method][cohort]=None if any(v is None for v in (aa,ab,ba,bb)) else {
                "change_at_new_support_average_over_risk_fits":((ba-aa)+(bb-ab))/2,
                "change_at_new_risk_fit_average_over_supports":((ab-aa)+(bb-ba))/2,
                "interaction_difference_of_differences":bb-ba-ab+aa}
    return {"cells":cells,"descriptive_contrasts":differences}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("audit","pilot","balanced","increment","confirmation","openai-weights","biomed-dir","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/ham-g1-crossed-fit-v1.json")
    ap.add_argument("--rule",type=Path,default=Path(__file__).parent/"configs/ham-error-increment-v1.json")
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    ap.add_argument("--batch-size",type=int,default=8)
    ap.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    a=ap.parse_args()
    cfg=json.loads(a.config.read_bytes())
    rule=json.loads(a.rule.read_bytes())
    tree=Taxonomy(json.loads(a.taxonomy.read_bytes())["parent"])
    paths=(a.pilot,a.balanced,a.increment,a.confirmation)
    infos=[json.loads((p/"summary.json").read_bytes()) for p in paths]
    audit,rows=read_inputs(a.audit)
    param_sha=hashlib.sha256(json.dumps(parameters(infos[2]),sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if (audit["split_sha256"]!=cfg["split_sha256"] or param_sha!=cfg["original_risk_parameters_sha256"]
            or digest(a.taxonomy)!=infos[0]["config_sha256"] or digest(a.rule)!=infos[2]["config_sha256"]
            or any(v["status"]!="complete" for v in infos) or a.batch_size<1
            or [v["selection_sha256"] for v in infos]!=cfg["previous_selections"]
            or [digest(p/"selection.csv") for p in paths]!=cfg["previous_selections"]):
        raise ValueError("Registered cohort, taxonomy, split or baseline changed")
    history=[read_csv(p/"selection.csv") for p in paths]
    alternative,risk_fit=select(rows,history,cfg,tree.leaves)
    old_support=[i for i,r in enumerate(history[1]) if r["role"]=="train"]
    old_balanced=[i for i,r in enumerate(history[1]) if r["role"]=="dev"]
    def labels(rows):
        return np.array([tree.leaves.index(r["label"]) for r in rows])
    support_y={"original_support":labels([history[1][i] for i in old_support]),"new_support":labels(alternative)}
    truth={"original_fit":labels(history[0]),"new_fit":labels(risk_fit),"first":labels(history[2]),
           "second":labels(history[3]),"balanced":labels([history[1][i] for i in old_balanced])}
    start=time.perf_counter()
    a.output.mkdir(exist_ok=False,parents=True)
    table=a.output/"selection.csv"
    with table.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(("image_id","split_group","role","label"))
        w.writerows((r["image_id"],r["split_group"],r["split"],r["label"]) for r in alternative+risk_fit)
    report={"scope":cfg["scope"],"config_sha256":digest(a.config),"split_sha256":audit["split_sha256"],
            "selection_sha256":digest(table),"reserved_groups":cfg["groups_reserved"],
            "new_support_counts":dict(Counter(r["label"] for r in alternative)),
            "new_fit_counts":dict(Counter(r["label"] for r in risk_fit)),"models":{}}
    try:
        for model,ref in infos[2]["models"].items():
            now=time.perf_counter()
            cache=[p/f"{model}.npz" for p in paths]
            expected=[ref["fit_cache"],ref["support_cache"],ref["cache_sha256"],infos[3]["models"][model]["cache_sha256"]]
            if list(map(digest,cache))!=expected:
                raise ValueError("Original feature cache changed")
            loader={"batch_size":a.batch_size,"device":a.device}
            w=a.openai_weights if model=="clip" else a.biomed_dir/"open_clip_pytorch_model.bin"
            model_cfg=None if model=="clip" else a.biomed_dir/"open_clip_config.json"
            encoder,prep,_=load_encoder(model,w,a.device,model_cfg,loader)
            new_rows=alternative+risk_fit
            new_img=np.concatenate([encode_images(encoder,prep,new_rows[i:i+a.batch_size],a.device)
                                    for i in range(0,len(new_rows),a.batch_size)])
            del encoder
            local=a.output/f"{model}.npz"
            np.savez(local,image_features=new_img)
            with (np.load(cache[0],allow_pickle=False) as u, np.load(cache[1],allow_pickle=False) as v,
                  np.load(cache[2],allow_pickle=False) as first, np.load(cache[3],allow_pickle=False) as second):
                images={"original_fit":u["image_features"],"new_fit":new_img[len(alternative):],
                        "first":first["image_features"],"second":second["image_features"],"balanced":v["image"][old_balanced]}
                supports={"original_support":v["image"][old_support].astype(np.float64),"new_support":new_img[:len(alternative)].astype(np.float64)}
                baseline={r+"_"+k:np.r_[first[r+"_"+k],second[r+"_"+k]] for r in rule["readouts"] for k in ("error","base","augmented")}
                result=run_cells(tree,rule,infos[0]["models"][model],supports,support_y,images,truth,u["text_features"],baseline)
            report["models"][model]={**result,"loader":loader,"cache_sha256":digest(local),"seconds":round(time.perf_counter()-now,3)}
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (a.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
