"""Bind HAM views with geometric correspondence to the same partition, preserving all analysis units."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from ham_reviewed_pairs import CONFIRMED, UNCERTAIN, components, load_tables

ROLE_ORDER = ("train","dev","calibration","test")
V14_EDGES_SHA256 = "4dcd84dffab0a854097a6979989ca0025972336f66c37f8c76cd084e55d3335d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def guard_edges(appearance, cross, within):
    report = json.loads((appearance/"summary.json").read_bytes())
    ranking = appearance/"candidates.csv"
    if sha(ranking)!=report["candidate_sha256"]:
        raise ValueError("Changed appearance ranking")
    items = read_csv(ranking)
    found = set()
    for tag, directory, cross_flag in [("cross_split_candidate",cross,"1"),("same_roles_candidate",within,"0")]:
        info=json.loads((directory/"summary.json").read_bytes())
        path=directory/"scores.csv"
        if (info["status"]!="complete" or info["candidate_sha256"]!=sha(ranking)
                or info["manifest_sha256"]!=report["manifest_sha256"] or sha(path)!=info["scores_sha256"]):
            raise ValueError("Geometry evidence does not match the appearance ranking")
        expected={(r["first"],r["second"]) for r in items if r["cross_any_split"]==cross_flag and not r["published"]}
        rows=[r for r in read_csv(path) if r["source"]==tag]
        keys={(r["first"],r["second"]) for r in rows}
        if len(rows)!=len(keys) or keys!=expected:
            raise ValueError(f"Incomplete geometric scoring of {tag}")
        found.update((r["first"],r["second"]) for r in rows if int(r["inliers"])>=8)
    return sorted(found)


def regroup(rows, edges):
    by={r["image_id"]:r for r in rows}
    groups={r["split_group"] for r in rows}
    grouped=components(groups,[(by[a]["split_group"],by[b]["split_group"]) for a,b in edges])
    roles=defaultdict(set)
    for r in rows:
        roles[grouped[r["split_group"]]].add(r["split"])
    # A previously exposed train/dev group cannot be moved into a held-out partition.
    chosen={k:min(s,key=ROLE_ORDER.index) for k,s in roles.items()}
    output=[]
    for r in rows:
        group=grouped[r["split_group"]]
        output.append({**r,"split_group":group,"split":chosen[group]})
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ["split-dir","appearance","cross-geometry","within-geometry","output"]:
        ap.add_argument("--"+key,type=Path,required=True)
    args=ap.parse_args()
    start=time.perf_counter()
    src=json.loads((args.split_dir/"audit.json").read_bytes())
    if src["split_protocol"]!="v1.3":
        raise ValueError("Expected stage-one v1.3 group splits")
    edges=guard_edges(args.appearance,args.cross_geometry,args.within_geometry)
    # Canonicalized identifiers are private; the portable digest detects platform or input drift.
    edge_hash=hashlib.sha256("".join(f"{a},{b}\n" for a,b in edges).encode()).hexdigest()
    if edge_hash != V14_EDGES_SHA256:
        raise ValueError("Geometric links differ from the released v1.4 set; inspect the evidence before versioning a new split")
    args.output.mkdir(parents=True,exist_ok=False)
    tables=load_tables(args.output,args.split_dir)
    evidence={"confirmed":tables[CONFIRMED],"uncertain":tables[UNCERTAIN],"geometry":edges}
    report={k:src[k] for k in ["review_source_commit","review_source_hashes","lesion_count","analysis_units","image_count"]}
    report.update(split_protocol="v1.4",parent_protocol="v1.3",eligible_geometry_pairs=len(edges),
                  geometry_edges_sha256=edge_hash,source_split_groups=src["split_groups"],splits={})
    for seed,entry in src["splits"].items():
        old_path=args.split_dir/f"split_seed{seed}.csv"
        if sha(old_path)!=entry["sha256"]:
            raise ValueError("Previous split changed")
        rows=read_csv(old_path)
        revised=regroup(rows,edges)
        before={r["image_id"]:r for r in rows}
        after={r["image_id"]:r for r in revised}
        crossings={name:{phase:sum(d[a]["split"]!=d[b]["split"] for a,b in pairs)
                         for phase,d in [("before",before),("after",after)]} for name,pairs in evidence.items()}
        if any(n["after"] for n in crossings.values()):
            raise ValueError("Evidence pair crosses a revised split")
        moved=[r for r in revised if r["split"]!=before[r["image_id"]]["split"]]
        if any(ROLE_ORDER.index(r["split"])>ROLE_ORDER.index(before[r["image_id"]]["split"]) for r in moved):
            raise ValueError("A group moved to a later-stage partition")
        groups={r["split_group"]:r["split"] for r in revised}
        reps=[r for r in revised if r["representative"]=="1"]
        path=args.output/old_path.name
        with path.open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(revised)
        report["split_groups"]=len(groups)
        report["splits"][seed]={"sha256":sha(path),"v1_3_sha256":sha(old_path),
                                "group_counts":dict(Counter(groups.values())),
                                "analysis_unit_counts":dict(Counter(r["split"] for r in reps)),
                                "moved_images":len(moved),"moved_raw_lesions":len({r["lesion_id"] for r in moved}),
                                "reviewed_pair_crossings":crossings,"group_leakage_count":0,
                                "representative_labels":{s:dict(Counter(r["label"] for r in reps if r["split"]==s))
                                                         for s in ROLE_ORDER}}
    report.update(status="geometry_split_binding_complete",elapsed_seconds=round(time.perf_counter()-start,3))
    (args.output/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
