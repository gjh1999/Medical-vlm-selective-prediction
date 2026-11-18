"""Enumerate the seven-leaf output space for the nested covering problem in Mortier et al. (Eq. 8)."""
import argparse
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from selective_core import Taxonomy


def output_space(tree):
    leaves = tree.leaves
    masks = {sum(1<<leaves.index(x) for x in subtree) for subtree in tree.desc.values()}
    minimum = {0:0}
    for node in sorted(masks):
        for existing,count in list(minimum.items()):
            if not existing & node:
                full = existing | node
                minimum[full] = min(count+1,minimum.get(full,len(leaves)+1))
    if len(minimum) != 2**len(leaves):
        raise ValueError("Missing leaf subsets")
    return minimum


def nested(prob, budget, space):
    order = np.argsort(-prob,kind="stable")
    # Eq. 8 first minimizes leaf cardinality, then maximizes enclosed probability mass.
    options = [mask for mask, count in space.items() if 0<count<=budget]
    mass = {m:sum(float(p) for j,p in enumerate(prob) if m & (1<<j)) for m in options}
    options.sort(key=lambda m:(m.bit_count(),-mass[m],m))
    prev = 0
    result = []
    for idx in order:
        target = prev | 1<<int(idx)
        prev = next(mask for mask in options if mask & target == target)
        result.append(prev)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/ham-dev-pilot-v1.json")
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    t=time.perf_counter()
    tree=Taxonomy(json.loads(args.config.read_bytes())["parent"])
    if len(tree.leaves)!=7:
        raise ValueError("Exhaustive reference is for the small HAM tree")
    pilot=json.loads((args.pilot/"summary.json").read_bytes())
    if (pilot["status"]!="complete" or pilot["role"]!="dev"
            or hashlib.sha256(args.config.read_bytes()).hexdigest()!=pilot["config_sha256"]):
        raise ValueError("Unexpected pilot or tree")
    space=output_space(tree)
    # Independent characterization: the maximal complete subtrees contained in a set are disjoint.
    for k in range(8):
        for subset in itertools.combinations(tree.leaves,k):
            target=set(subset)
            nodes=[n for n in tree.nodes if tree.desc[n]<=target and
                   (n==tree.root or not tree.desc[tree.parent[n]]<=target)]
            mask=sum(1<<tree.leaves.index(x) for x in target)
            if len(nodes)!=space[mask]:
                raise ValueError("Inconsistent representation complexity")
    report={"source":"Mortier et al., AISTATS 2026, equation 8; exhaustive finite-tree reference",
            "scope":"development probabilities only, before calibration, no evaluation labels",
            "minimum_nodes_histogram":dict(sorted(Counter(space.values()).items())),
            "prior_config_sha256":pilot["config_sha256"],"selection_sha256":pilot["selection_sha256"],"models":{}}
    full=(1<<len(tree.leaves))-1
    for name,model in pilot["models"].items():
        path=args.pilot/f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=model["cache_sha256"] or model["leaf_order"]!=tree.leaves:
            raise ValueError("Changed probability cache")
        with np.load(path,allow_pickle=False) as cache:
            probs=cache["leaf_probs"]
        hist={}
        for r in [1,2,7]:
            paths=[nested(row,r,space) for row in probs]
            for p,row in zip(paths,probs):
                if p[-1]!=full or any(a & b != a for a,b in zip(p,p[1:])):
                    raise ValueError("Non-nested path")
                if r==7 and [m.bit_count() for m in p]!=list(range(1,8)):
                    raise ValueError("Unconstrained case should equal sorted leaf prefixes")
            hist[str(r)]=[{"rank":i+1,"mean_leaf_count":float(np.mean([p[i].bit_count() for p in paths])),
                           "root_count":sum(p[i]==full for p in paths)} for i in range(7)]
        report["models"][name]={"n":len(probs),"budgets":hist}
    report["seconds"]=round(time.perf_counter()-t,3)
    args.output.mkdir(exist_ok=False,parents=True)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
