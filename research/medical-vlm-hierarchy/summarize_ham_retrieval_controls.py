"""Measure the full retrieval-plus-verification cascade on the same published positive controls."""
import argparse
import csv
import json
import time
from pathlib import Path

from audit_ham_texture_controls import file_hash


def read_rows(path):
    with path.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def pair(r):
    return tuple(sorted((r["first"],r["second"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("words","appearance","texture","orb","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    result = {"scope":"published positive and unresolved pairs only; not population-level sensitivity"}
    try:
        paths = (args.words,args.appearance,args.texture,args.orb)
        infos = [json.loads((p/"summary.json").read_bytes()) for p in paths]
        files = ("candidates.csv","candidates.csv","pairs.csv","scores.csv")
        keys = ("candidate_sha256","candidate_sha256","pairs_sha256","scores_sha256")
        if (any(file_hash(p/n)!=info[k] for p,info,n,k in zip(paths,infos,files,keys))
                or infos[0]["old_candidate_sha256"]!=infos[1]["candidate_sha256"]
                or infos[3]["control_sha256"]!=infos[2]["pairs_sha256"]):
            raise ValueError("Changed upstream tables")
        result["source_sha256"] = [info[k] for info,k in zip(infos,keys)]
        confirmations = [r for r in read_rows(args.texture/"pairs.csv") if r["source"].startswith("published_")]
        by_pair = {pair(r): r for r in confirmations}
        ranks = {}
        with (args.words/"candidates.csv").open(newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = pair(r)
                if key in by_pair:
                    ranks[key] = int(r["best_rank"])
        appearance = {pair(r) for r in read_rows(args.appearance/"candidates.csv") if pair(r) in by_pair}
        orb = {pair(r):r for r in read_rows(args.orb/"scores.csv") if pair(r) in by_pair}
        if set(orb)!=set(by_pair):
            raise ValueError("ORB control coverage differs")
        result["tables"] = {}
        for kind in sorted({r["source"] for r in confirmations}):
            controls = {p for p,r in by_pair.items() if r["source"]==kind}
            root8 = {p for p in controls if int(by_pair[p]["new_inliers"])>=8}
            root16 = {p for p in root8 if int(by_pair[p]["new_inliers"])>=16 and int(by_pair[p]["new_cells"])>=4}
            pixel = {p for p in controls if orb[p]["passes"]=="1"}
            table = {}
            for k in (1,5,10,20):
                words = {p for p in controls if ranks.get(p,21)<=k}
                combined = words | (controls & appearance)
                table[str(k)] = {"known_total":len(controls),"words_retrieved":len(words),
                    "words_then_root8":len(words & root8),"combined_retrieved":len(combined),
                    "combined_then_root8":len(combined & root8),"combined_then_root16_cells4":len(combined & root16),
                    "combined_then_root16_cells4_and_strict_pixel":len(combined & root16 & pixel)}
            result["tables"][kind] = table
        result["status"]="complete"
    except Exception as exc:
        result.update(status="failed",error=str(exc))
        raise
    finally:
        result["seconds"]=round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
