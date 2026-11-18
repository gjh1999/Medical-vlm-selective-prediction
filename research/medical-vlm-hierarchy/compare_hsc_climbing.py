"""Replay authors' pinned HSC Climbing rule on the same HAM dev probabilities."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from selective_core import Taxonomy

REVISION = "87f59e214d1521827118b1a22e9b5af8deaa57b8"
SOURCE_SHA = "f029c2e02edfbde3a7d5e10359d7be8196fa436bef1d81c7dfbde9492aba268d"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ["pilot","official-source","output"]:
        ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/ham-dev-pilot-v1.json")
    args = ap.parse_args()
    t = time.perf_counter()
    if hashlib.sha256(args.official_source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Expected pinned authors' source file")
    spec = importlib.util.spec_from_file_location("authors_hsc_rules", args.official_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pilot = json.loads((args.pilot/"summary.json").read_bytes())
    if (pilot["status"] != "complete" or pilot["role"] != "dev" or
            hashlib.sha256(args.config.read_bytes()).hexdigest() != pilot["config_sha256"]):
        raise ValueError("Unexpected source of development probabilities")
    cfg = json.loads(args.config.read_bytes())
    tree = Taxonomy(cfg["parent"])
    nodes = tree.leaves + sorted(tree.nodes-set(tree.leaves)-{tree.root}) + [tree.root]
    ancestors = torch.tensor([[leaf in tree.desc[node] for leaf in tree.leaves] for node in nodes])
    hierarchy = SimpleNamespace(anc_matrix=ancestors, root_index=nodes.index(tree.root), num_leaves=len(tree.leaves))
    official = module.ClimbingInferenceRule(hierarchy)

    def compare(p, thresholds):
        all_probs = ancestors.double() @ torch.from_numpy(p.T.copy()).double()
        # The paper treats the root as containing all probability mass.
        all_probs[hierarchy.root_index] = 1.
        top = torch.tensor(p.argmax(axis=1))
        dicts = [dict(zip(tree.leaves,row)) for row in p]
        different = {}
        for theta in thresholds:
            _, idx = official.predict(all_probs,top,theta)
            ours = [nodes.index(tree.climbing(row,theta)) for row in dicts]
            different[str(theta)] = int((np.asarray(ours) != idx.numpy()).sum())
        return {"rows":len(p), "thresholds":list(thresholds), "comparisons":len(p)*len(thresholds),
                "node_mismatches":different}

    report = {"source_commit":REVISION,"source_sha256":SOURCE_SHA, "config_sha256":pilot["config_sha256"],
              "selection_sha256":pilot["selection_sha256"],"models":{}}
    for name, model in pilot["models"].items():
        path = args.pilot/f"{name}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=model["cache_sha256"] or model["leaf_order"]!=tree.leaves:
            raise ValueError("Changed probabilities or class order")
        with np.load(path, allow_pickle=False) as cache:
            report["models"][name] = compare(cache["leaf_probs"], [0., *cfg["climbing_thresholds"],1.])
    # Exact binary fractions, tied top leaves and a zero-sibling branch exercise equality handling.
    special = np.zeros((4,len(tree.leaves)), dtype=float)
    for index, probs in enumerate([{"mel":.5,"nv":.5},{"bcc":1.},{"akiec":.25,"bcc":.25,"nv":.5},
                                    {"df":.25,"vasc":.25,"nv":.25,"mel":.25}]):
        for k,value in probs.items():
            special[index,tree.leaves.index(k)] = value
    report["constructed_cases"] = compare(special,[0.,.25,.5,.75,1.])
    total = sum(sum(v["node_mismatches"].values()) for v in [*report["models"].values(),report["constructed_cases"]])
    report.update(total_mismatches=total,seconds=round(time.perf_counter()-t,3))
    args.output.mkdir(exist_ok=False,parents=True)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    if total:
        raise ValueError(f"Found {total} differing node predictions")


if __name__=="__main__":
    main()
