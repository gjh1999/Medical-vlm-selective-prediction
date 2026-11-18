"""Apply a released set of HAM local-texture co-assignment constraints to the complete v1.4 split."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from ham_reviewed_pairs import CONFIRMED, UNCERTAIN, load_tables
from prepare_ham_scene_splits import ROLE_ORDER, V14_EDGES_SHA256, read_csv, regroup, sha


def checksum_edges(edges):
    return hashlib.sha256("".join(f"{a},{b}\n" for a,b in sorted(edges)).encode()).hexdigest()


def check_components(before, after, edges):
    adjacency = defaultdict(set)
    old_roles = {}
    for row in before.values():
        g = row["split_group"]
        if old_roles.setdefault(g,row["split"])!=row["split"]:
            raise ValueError("A source group crosses partitions")
    for a,b in edges:
        ga,gb = before[a]["split_group"],before[b]["split_group"]
        adjacency[ga].add(gb)
        adjacency[gb].add(ga)
    # Compare union-find output to an independent traversal, including all untouched singleton groups.
    expected = {}
    for g in old_roles:
        if g in expected:
            continue
        component = {g}
        pending = [g]
        while pending:
            unseen = adjacency.get(pending.pop(),set())-component
            component.update(unseen)
            pending.extend(unseen)
        identity = min(component)
        role = min((old_roles[k] for k in component),key=ROLE_ORDER.index)
        expected.update((k,(identity,role)) for k in component)
    if set(after)!=set(before) or any((after[i]["split_group"],after[i]["split"])!=expected[r["split_group"]]
                                      for i,r in before.items()):
        raise ValueError("Independent component membership or exposure precedence differs")


def verify_screen(words, appearance, geometry, review, expected):
    folders = (words, appearance, geometry, review)
    infos = [json.loads((d/"summary.json").read_bytes()) for d in folders]
    paths = [d/n for d,n in zip(folders,("candidates.csv","candidates.csv","scores.csv","scores.csv"))]
    for path,info,key in zip(paths,infos,("candidate_sha256","candidate_sha256","scores_sha256","scores_sha256")):
        if sha(path)!=info[key]:
            raise ValueError("Retrieval or scoring table differs from the audit")
    word,old,scored,checked = infos
    if (scored["max_rank"]!=20 or not scored["include_appearance"] or scored["status"]!="complete"
            or word["status"]!="complete" or checked["status"]!="complete"
            or scored["word_candidates_sha256"]!=word["candidate_sha256"]
            or checked["geometry_sha256"]!=scored["scores_sha256"]
            or word["old_candidate_sha256"]!=old["candidate_sha256"]):
        raise ValueError("The released split requires the completed combined candidate pool and independent review")
    pairs = set()
    for path in paths[:2]:
        with path.open(newline="",encoding="utf-8") as f:
            for row in csv.DictReader(f):
                a,b = sorted((row["first"],row["second"]))
                if a==b:
                    raise ValueError("A retrieval candidate refers to itself")
                pairs.add((a,b))
    if len(pairs)!=expected["candidate_pairs"] or len(pairs)!=scored["candidate_pairs"]:
        raise ValueError("Combined candidate count differs from release")
    weak,strong = set(),set()
    with paths[2].open(encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f):
            if row["in_pool"]=="1":
                edge = row["first"],row["second"]
                if edge not in pairs:
                    raise ValueError("Duplicate or unexpected scored candidate")
                pairs.remove(edge)
                if row["same_v14_group"]=="0" and int(row["inliers"])>=8:
                    weak.add(edge)
                    if int(row["inliers"])>=16 and int(row["cells"])>=4:
                        strong.add(edge)
    if pairs:
        raise ValueError("Some combined candidates have no geometry score")
    checked_edges = {(r["first"],r["second"]) for r in read_csv(paths[3]) if r["source"]=="new_screen_hit"}
    if (checked_edges!=strong or checksum_edges(weak)!=expected["weak_edges_sha256"]
            or checksum_edges(strong)!=expected["strong_edges_sha256"]):
        raise ValueError("Canonical scene evidence differs from the independently verified release")
    return weak,strong


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("splits","words","appearance","geometry","review","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--release",type=Path,default=Path(__file__).parent/"configs/ham-texture-release-v1.json")
    args = ap.parse_args()
    start = time.perf_counter()
    src = json.loads((args.splits/"audit.json").read_bytes())
    expected = json.loads(args.release.read_bytes())
    old_hashes = {s:v["sha256"] for s,v in src["splits"].items()}
    if (src["split_protocol"]!="v1.4" or src["geometry_edges_sha256"]!=V14_EDGES_SHA256
            or old_hashes!=expected["source_split_sha256"]):
        raise ValueError("The input splits differ from the released v1.4 baseline")
    weak,strong = verify_screen(args.words,args.appearance,args.geometry,args.review,expected)
    args.output.mkdir(parents=True,exist_ok=False)
    tables=load_tables(args.output,args.splits)
    result={k:src[k] for k in ("review_source_commit","review_source_hashes","lesion_count","analysis_units","image_count")}
    result.update(split_protocol="v1.5",parent_protocol="v1.4",source_split_groups=src["split_groups"],
                  prior_geometry_edges_sha256=V14_EDGES_SHA256,release_config_sha256=sha(args.release),
                  geometry_edges_sha256=checksum_edges(strong),loose_edges_sha256=checksum_edges(weak),
                  eligible_geometry_pairs=len(strong),loose_geometry_pairs=len(weak),splits={})
    try:
        texture_path=args.output/"texture_candidates.csv"
        with texture_path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f)
            writer.writerow(["first","second","bound"])
            writer.writerows((a,b,int((a,b) in strong)) for a,b in sorted(weak))
        result["texture_candidates_sha256"]=sha(texture_path)
        for seed,entry in src["splits"].items():
            path=args.splits/f"split_seed{seed}.csv"
            if sha(path)!=entry["sha256"]:
                raise ValueError("Source split file changed")
            rows=read_csv(path)
            old={r["image_id"]:r for r in rows}
            if any(old[a]["split_group"]==old[b]["split_group"] for a,b in weak):
                raise ValueError("Distinct-group evidence contains an already-bound pair")
            revised=regroup(rows,sorted(strong))
            new={r["image_id"]:r for r in revised}
            check_components(old,new,strong)
            moved=[r for r in revised if r["split"]!=old[r["image_id"]]["split"]]
            if any(ROLE_ORDER.index(r["split"])>ROLE_ORDER.index(old[r["image_id"]]["split"]) for r in moved):
                raise ValueError("An exposed image moves to a later partition")
            descendants=defaultdict(set)
            roles=defaultdict(set)
            for r in revised:
                previous=old[r["image_id"]]
                if any(previous[k]!=r[k] for k in previous if k not in ("split","split_group")):
                    raise ValueError("Non-partition metadata changed")
                descendants[previous["split_group"]].add(r["split_group"])
                roles[r["split_group"]].add(r["split"])
            if any(len(v)!=1 for v in (*roles.values(),*descendants.values())):
                raise ValueError("An earlier group was divided or a revised group crosses partitions")
            evidence={"confirmed":tables[CONFIRMED],"uncertain":tables[UNCERTAIN],"new_texture":strong}
            crossings={k:{phase:sum(by[a]["split"]!=by[b]["split"] for a,b in edges)
                          for phase,by in (("before",old),("after",new))} for k,edges in evidence.items()}
            if any(v["after"] for v in crossings.values()):
                raise ValueError("A binding pair still crosses partitions")
            dest=args.output/path.name
            with dest.open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(revised)
            reps=[r for r in revised if r["representative"]=="1"]
            result["split_groups"]=len(roles)
            if seed=="17":
                result["strong_edges_with_different_dataset_labels"]=sum(old[a]["label"]!=old[b]["label"] for a,b in strong)
                result["strong_edges_with_two_representatives"]=sum(old[a]["representative"]==old[b]["representative"]=="1" for a,b in strong)
                units=Counter(r["split_group"] for r in reps)
                result["analysis_units_per_group_histogram"]=dict(sorted(Counter(units.values()).items()))
            result["splits"][seed]={"sha256":sha(dest),"v1_4_sha256":entry["sha256"],
                "group_counts":dict(Counter(next(iter(v)) for v in roles.values())),
                "analysis_unit_counts":dict(Counter(r["split"] for r in reps)),
                "moved_images":len(moved),"moved_raw_lesions":len({r["lesion_id"] for r in moved}),
                "reviewed_pair_crossings":crossings,"group_leakage_count":0,
                "representative_labels":{s:dict(Counter(r["label"] for r in reps if r["split"]==s)) for s in ROLE_ORDER}}
        result["status"]="texture_split_binding_complete"
    except Exception as exc:
        result.update(status="failed",error=str(exc))
        raise
    finally:
        result["elapsed_seconds"]=round(time.perf_counter()-start,3)
        (args.output/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
