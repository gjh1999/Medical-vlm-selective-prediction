"""Replay frozen E02 comparisons after projecting previous pilot cohorts to a verified split and mask."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ham_dev_pilot import digest, error_auc
from ham_error_increment import logistic_predict, read_csv, signals
from selective_core import Taxonomy


def verify_selections(paths, source_infos, split_file, mask_file):
    selections = []
    for path, info in zip(paths, source_infos):
        if info["status"] != "complete" or digest(path/"selection.csv") != info["selection_sha256"]:
            raise ValueError("Historical pilot selection changed")
        selections.append(read_csv(path/"selection.csv"))
    fit, old, fresh = selections
    train = [i for i,r in enumerate(old) if r["role"] == "train"]
    balanced = [i for i,r in enumerate(old) if r["role"] == "dev"]
    if len(train)+len(balanced) != len(old) or any(r["role"] != "dev" for r in fit+fresh):
        raise ValueError("Unexpected original cohort roles")
    cohorts = {"support":[old[i] for i in train],"fit":fit,"natural":fresh,"balanced":[old[i] for i in balanced]}
    members = [r for batch in cohorts.values() for r in batch]
    if len({r["split_group"] for r in members}) != len(members):
        raise ValueError("Original cohorts overlap")
    split = {r["image_id"]:r for r in read_csv(split_file)}
    excluded = {r["split_group"] for r in read_csv(mask_file) if r["excluded"] == "1"}
    used = Counter(split[r["image_id"]]["split_group"] for r in members)
    scenarios = {"original":{}, "current_roles_unique":{}, "current_roles_unique_and_sensitivity":{}}
    changes = {}
    for name, batch in cohorts.items():
        current = [split[r["image_id"]] for r in batch]
        if any(s["label"]!=r["label"] or s["representative"]!="1" for s,r in zip(current,batch)):
            raise ValueError("Historical label or analysis representative changed")
        correct_role = np.array([s["split"]==r["role"] for s,r in zip(current,batch)])
        unique = np.array([used[s["split_group"]]==1 for s in current])
        masked = np.array([s["split_group"] in excluded for s in current])
        scenarios["original"][name] = np.ones(len(batch),dtype=bool)
        scenarios["current_roles_unique"][name] = correct_role & unique
        scenarios["current_roles_unique_and_sensitivity"][name] = correct_role & unique & ~masked
        changes[name] = {"original":len(batch),"role_changed":int((~correct_role).sum()),
                         "rows_in_repeated_current_group":int((~unique).sum()),"in_sensitivity_mask":int(masked.sum())}
    return cohorts, train, balanced, scenarios, changes


def evaluate(tree, cfg, labels, keep, images, support, text, meta):
    selected = np.flatnonzero(keep["support"])
    centers = []
    for i in range(len(tree.leaves)):
        ids = selected[labels["support"][selected]==i]
        if not len(ids):
            return {"status":"unavailable_support_class","leaf":tree.leaves[i]}
        centers.append(support[ids].astype(np.float64).mean(0))
    centers = np.stack(centers)
    centers /= np.linalg.norm(centers,axis=1,keepdims=True)
    names = ("fit","natural","balanced")
    truth = np.concatenate([labels[k][keep[k]] for k in names])
    rows = np.concatenate([images[k][keep[k]] for k in names])
    fit_n = int(keep["fit"].sum())
    bounds = np.cumsum([int(keep[k].sum()) for k in names])
    signals_by_readout = signals(rows,text,meta["text_node_order"],tree.leaves,tree,meta["logit_scale"],centers)
    methods = {}
    for readout,(pred,features) in signals_by_readout.items():
        wrong = pred != truth
        if len(np.unique(wrong[:fit_n])) != 2:
            methods[readout] = {"status":"single_class_fit"}
            continue
        scores = {}
        for name,keys in (("base",cfg["base"]),("augmented",cfg["base"]+[cfg["extra"]])):
            matrix = np.column_stack([features[k] for k in keys])
            scores[name],_ = logistic_predict(matrix[:fit_n],wrong[:fit_n],matrix[fit_n:],cfg["l2_penalty"])
        methods[readout] = {}
        for name,start,end in zip(names[1:],bounds[:-1],bounds[1:]):
            y = wrong[start:end]
            s = slice(start-fit_n,end-fit_n)
            a,b = (error_auc(y,scores[k][s]) for k in ("base","augmented"))
            methods[readout][name] = {"n":len(y),"errors":int(y.sum()),"base_auroc":a,"augmented_auroc":b,
                                      "difference":None if a is None else b-a}
    return {"status":"complete","readouts":methods}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot","balanced","increment","splits","sensitivity","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    ap.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/ham-error-increment-v1.json")
    a = ap.parse_args()
    paths = (a.pilot,a.balanced,a.increment)
    infos = [json.loads((p/"summary.json").read_bytes()) for p in paths]
    cfg = json.loads(a.config.read_bytes())
    if infos[0]["config_sha256"]!=digest(a.taxonomy) or infos[2]["config_sha256"]!=digest(a.config):
        raise ValueError("Source experiment config changed")
    tree = Taxonomy(json.loads(a.taxonomy.read_bytes())["parent"])
    seed = str(json.loads(a.taxonomy.read_bytes())["split_seed"])
    split_file, mask_file = a.splits/f"split_seed{seed}.csv", a.sensitivity/"sensitivity_groups.csv"
    split = json.loads((a.splits/"audit.json").read_bytes())
    mask = json.loads((a.sensitivity/"summary.json").read_bytes())
    if (split["splits"][seed]["sha256"]!=digest(split_file) or mask["split_sha256"][seed]!=digest(split_file)
            or digest(mask_file)!=mask["sensitivity_groups_sha256"] or mask["protocol"]!=split["split_protocol"]):
        raise ValueError("Split or sensitivity mask mismatch")
    a.output.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    report = {"scope":"posthoc development-only refit; original caches, no replacement groups",
              "protocol":mask["protocol"],"seed":seed,"split_sha256":digest(split_file),
              "sensitivity_sha256":digest(mask_file),"models":{}}
    try:
        cohorts,tr,dev,scenarios,changes = verify_selections(paths,infos,split_file,mask_file)
        labels = {n:np.array([tree.leaves.index(r["label"]) for r in rows]) for n,rows in cohorts.items()}
        report["historical_cohorts"] = changes
        report["retained_by_label"] = {s:{n:dict(Counter(r["label"] for r,ok in zip(cohorts[n],keep) if ok))
                                               for n,keep in parts.items()} for s,parts in scenarios.items()}
        for model,info in infos[2]["models"].items():
            files = [p/f"{model}.npz" for p in paths]
            if list(map(digest,files)) != [info["fit_cache"],info["support_cache"],info["cache_sha256"]]:
                raise ValueError("Feature cache changed")
            with np.load(files[0],allow_pickle=False) as first, np.load(files[1],allow_pickle=False) as bal, np.load(files[2],allow_pickle=False) as fresh:
                images = {"fit":first["image_features"],"natural":fresh["image_features"],"balanced":bal["image"][dev]}
                model_runs = {k:evaluate(tree,cfg,labels,keep,images,bal["image"][tr],first["text_features"],
                                         infos[0]["models"][model]) for k,keep in scenarios.items()}
            for method,ref in info["readouts"].items():
                replay = model_runs["original"]["readouts"][method]["natural"]
                if replay["errors"]!=ref["eval_errors"] or any(abs(replay[k+"_auroc"]-ref[k]["error_auroc"])>1e-12
                                                               for k in ("base","augmented")):
                    raise ValueError("Original cached E02 comparison did not reproduce")
            report["models"][model] = model_runs
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start,3)
        (a.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__ == "__main__":
    main()
