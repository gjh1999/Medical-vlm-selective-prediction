"""Compare a parent-mass two-stage rule with mode-path fallback under the same thresholds."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from audit_ham_temperature import fit_inverse_temperature
from ham_cached_readouts import development_readouts
from ham_dev_pilot import digest, softmax


THRESHOLDS = ((.5,.5),(.7,.7),(.9,.9),(.7,.5),(.9,.7))


def evaluate(tree, labels, probabilities, leaf_tau, branch_tau):
    choices, reference = [], []
    examples = Counter()
    branch_names = sorted(n for n,par in tree.parent.items() if par==tree.root)
    assignment = np.array([[y in tree.desc[n] for y in tree.leaves] for n in branch_names])
    for y,row in zip(labels,probabilities):
        p = dict(zip(tree.leaves,row))
        leaf = tree.leaves[int(row.argmax())]
        branch = tree.parent[leaf]
        path = leaf if p[leaf]>=leaf_tau else tree.root
        if path==tree.root and branch!=tree.root and sum(p[k] for k in sorted(tree.desc[branch]))>=branch_tau:
            path=branch
        if leaf_tau==branch_tau and path!=tree.climbing(p,leaf_tau):
            raise ValueError("Mode-path comparator differs from Climbing on the shallow tree")
        choice = tree.two_stage(p,leaf_tau,branch_tau)
        masses = assignment @ row
        idx = int(np.argmax(masses))
        # At a tied maximum the scalar rule resolves branches in lexical order.
        if np.count_nonzero(np.isclose(masses,masses[idx],atol=1e-13,rtol=0))==1:
            independently = leaf if p[leaf]>=leaf_tau else (
                branch_names[idx] if masses[idx]>=branch_tau and branch_names[idx] not in tree.leaves else tree.root)
            if choice!=independently:
                raise ValueError("Direct incidence-matrix calculation differs")
        choices.append(choice)
        reference.append(path)
        if path!=choice:
            examples["changed"]+=1
            examples["root_to_nonroot"]+=path==tree.root and choice!=tree.root
            examples["previous_correct_nonroot"]+=path!=tree.root and y in tree.desc[path]
            examples["new_correct_nonroot"]+=choice!=tree.root and y in tree.desc[choice]
    result={"paired_changes":dict(examples)}
    for name,actions in (("two_stage",choices),("mode_path",reference)):
        summary=tree.metrics(labels,actions)
        result[name]={k:summary[k] for k in ("n","node_n","node_errors","leaf_n","leaf_errors","correct_information_utility")}
    if result["two_stage"]["leaf_n"]!=result["mode_path"]["leaf_n"]:
        raise ValueError("Fallback should preserve the common leaf-acceptance gate")
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("pilot","balanced","increment","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--taxonomy",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    report={"scope":"fixed development comparisons; no threshold selection",
            "taxonomy_sha256":digest(a.taxonomy),"thresholds":THRESHOLDS,"models":{}}
    try:
        _,tree,labels,models=development_readouts(a.pilot,a.balanced,a.increment,a.taxonomy)
        if any(tree.parent[y]!=tree.root and tree.parent[tree.parent[y]]!=tree.root for y in tree.leaves):
            raise ValueError("This paired path comparison requires a two-level tree")
        targets=np.array([tree.leaves.index(y) for y in labels["fit"]])
        for model,methods in models.items():
            report["models"][model]={}
            for method,probs in methods.items():
                calibrated=fit_inverse_temperature(np.log(probs["fit"]),targets)
                result={}
                for temp,beta in (("fixed",1.),("fitted",calibrated)):
                    result[temp]={}
                    for cohort in ("natural","balanced"):
                        rows=softmax(beta*np.log(probs[cohort]))
                        result[temp][cohort]={f"{x}/{y}":evaluate(tree,labels[cohort],rows,x,y) for x,y in THRESHOLDS}
                report["models"][model][method]=result
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (a.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
