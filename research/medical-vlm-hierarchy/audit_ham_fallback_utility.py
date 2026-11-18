"""Explain how much information an ancestor fallback can retain on the existing HAM dev cohorts."""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ham_dev_pilot import digest, softmax
from ham_error_increment import read_csv
from selective_core import Taxonomy


def summary(tree,truth,choices):
    measured = tree.metrics(truth, choices)
    return {k:v for k,v in measured.items() if k!="per_class"}


def analyze(tree,truth,prob,thresholds):
    guesses = [tree.leaves[i] for i in prob.argmax(axis=1)]
    information = {n:1-math.log(len(tree.desc[n]))/math.log(len(tree.leaves)) for n in tree.nodes}
    best_actions = []
    types = Counter()
    for y,leaf in zip(truth,guesses):
        parent = tree.parent[leaf]
        if y==leaf:
            node,kind=leaf,"correct_leaf"
        elif parent!=tree.root and y in tree.desc[parent]:
            node,kind=parent,"wrong_leaf_same_nontrivial_branch"
        else:
            node,kind=tree.root,"wrong_leaf_across_branches"
        best_actions.append(node)
        types[kind] += 1
    oracle = summary(tree,truth,best_actions)
    # Check the oracle against all legal ancestor-path actions for each sample.
    total = 0.
    for y,leaf in zip(truth,guesses):
        path = [leaf]
        while path[-1]!=tree.root:
            path.append(tree.parent[path[-1]])
        total += max(information[n] for n in path if y in tree.desc[n])
    if not math.isclose(total/len(truth),oracle["correct_information_utility"],abs_tol=1e-12):
        raise ValueError("Oracle does not maximize correct information on the predicted path")
    result={"leaf_outcomes":dict(types),"all_leaf":summary(tree,truth,guesses),"oracle":oracle,"thresholds":{}}
    for threshold in thresholds:
        actions=[tree.climbing(dict(zip(tree.leaves,p)),threshold) for p in prob]
        same_acceptance=[leaf if node!=tree.root else tree.root for leaf,node in zip(guesses,actions)]
        up=summary(tree,truth,actions)
        flat=summary(tree,truth,same_acceptance)
        rescued=lost=0
        increments=[]
        for y,leaf,action in zip(truth,guesses,actions):
            if action not in (leaf,tree.root):
                if y==leaf:
                    lost+=1
                    increments.append(information[action]-1)
                elif y in tree.desc[action]:
                    rescued+=1
                    increments.append(information[action])
        change = up["correct_information_utility"]-flat["correct_information_utility"]
        if (up["node_n"]!=flat["node_n"] or flat["node_errors"]-up["node_errors"]!=rescued
                or not math.isclose(math.fsum(increments)/len(truth),change,abs_tol=1e-12)):
            raise ValueError("Matched acceptance or information decomposition differs")
        result["thresholds"][str(threshold)] = {"climbing":up,"flat_on_identical_accepted_cases":flat,
                   "utility_difference":change,"wrong_leaves_rescued":rescued,"correct_leaves_coarsened":lost}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("increment","balanced","pilot","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    args=ap.parse_args()
    start=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    report={"scope":"posthoc development diagnosis; oracle uses reference labels; no threshold selection or risk certification"}
    try:
        cfg=json.loads(args.taxonomy.read_bytes())
        tree=Taxonomy(cfg["parent"])
        paths=[args.pilot,args.balanced,args.increment]
        infos=[json.loads((p/"summary.json").read_bytes()) for p in paths]
        if (infos[0]["config_sha256"]!=digest(args.taxonomy)
                or any(s["status"]!="complete" or s["selection_sha256"]!=digest(p/"selection.csv")
                       for s,p in zip(infos,paths))):
            raise ValueError("Input taxonomy or group selection changed")
        past,balanced,fresh=[read_csv(p/"selection.csv") for p in paths]
        selected=past+balanced+fresh
        if len(selected)!=len({r["split_group"] for r in selected}):
            raise ValueError("Development cohorts overlap")
        train=[i for i,r in enumerate(balanced) if r["role"]=="train"]
        dev=[i for i,r in enumerate(balanced) if r["role"]=="dev"]
        if any(r["role"]!="dev" for r in fresh):
            raise ValueError("New evaluation rows must be development only")
        train_labels=np.asarray([balanced[i]["label"] for i in train])
        truths={"natural":[r["label"] for r in fresh],"balanced":[balanced[i]["label"] for i in dev]}
        report["fixed_constant_branch_controls"]={cohort:{
            n:summary(tree,ys,[n]*len(ys)) for n in sorted(tree.nodes) if n!=tree.root and len(tree.desc[n])>1
        } for cohort,ys in truths.items()}
        report["models"]={}
        for model,ref in infos[2]["models"].items():
            files=[p/f"{model}.npz" for p in paths]
            if [digest(p) for p in files]!=[ref["fit_cache"],ref["support_cache"],ref["cache_sha256"]]:
                raise ValueError("Feature cache changed")
            meta=infos[0]["models"][model]
            with np.load(files[0],allow_pickle=False) as first, np.load(files[1],allow_pickle=False) as bal, np.load(files[2],allow_pickle=False) as new:
                text=first["text_features"].astype(np.float64).mean(0)
                text/=np.linalg.norm(text,axis=1,keepdims=True)
                frozen=text[[meta["text_node_order"].index(n) for n in tree.leaves]]
                support=bal["image"][train].astype(np.float64)
                centers=np.stack([support[train_labels==n].mean(0) for n in tree.leaves])
                centers/=np.linalg.norm(centers,axis=1,keepdims=True)
                readouts={}
                for method,proto,bal_key in (("frozen_text",frozen,"frozen_text"),("cosine_centroid",centers,"cosine_nearest_centroid")):
                    cohorts={}
                    for label,x in (("natural",new["image_features"]),("balanced",bal["image"][dev])):
                        prob=softmax((x.astype(np.float64)@proto.T)*meta["logit_scale"])
                        guess=prob.argmax(1)
                        y=np.asarray([tree.leaves.index(n) for n in truths[label]])
                        if label=="natural" and not np.array_equal(guess!=y,new[f"{method}_error"]):
                            raise ValueError("Leaf errors differ from E02 cache")
                        if label=="balanced" and not np.array_equal(guess,bal[bal_key].argmax(1)):
                            raise ValueError("Balanced predictions differ from the original probe")
                        cohorts[label]=analyze(tree,truths[label],prob,cfg["climbing_thresholds"])
                    readouts[method]=cohorts
            report["models"][model]=readouts
        report.update(status="complete",taxonomy_sha256=digest(args.taxonomy),thresholds=cfg["climbing_thresholds"])
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
