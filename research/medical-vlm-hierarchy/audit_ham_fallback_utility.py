"""Explain how much information an ancestor fallback can retain on the existing HAM dev cohorts."""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from ham_cached_readouts import development_readouts
from ham_dev_pilot import digest


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
        cfg,tree,truths,scores=development_readouts(args.pilot,args.balanced,args.increment,args.taxonomy)
        cohorts=("natural","balanced")
        report["fixed_constant_branch_controls"]={cohort:{
            n:summary(tree,truths[cohort],[n]*len(truths[cohort])) for n in sorted(tree.nodes)
            if n!=tree.root and len(tree.desc[n])>1} for cohort in cohorts}
        report["models"]={model:{method:{c:analyze(tree,truths[c],probs[c],cfg["climbing_thresholds"])
                                         for c in cohorts} for method,probs in methods.items()}
                          for model,methods in scores.items()}
        report.update(status="complete",taxonomy_sha256=digest(args.taxonomy),thresholds=cfg["climbing_thresholds"])
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
