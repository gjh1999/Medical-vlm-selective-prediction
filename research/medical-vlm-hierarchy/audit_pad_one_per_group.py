"""Measure PAD calibration coverage after drawing one already-defined representative per group."""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path


SALT = "pad-independent-group-v1"


def support(groups, chosen, labels):
    fractions = [sum(r["label"] in labels for r in rows)/len(rows) for rows in groups.values()]
    possible = sum(p>0 for p in fractions)
    guaranteed = sum(p==1 for p in fractions)
    pmf = [1.0]
    for p in fractions:
        if 0 < p < 1:
            pmf = [a*(1-p)+b*p for a,b in zip(pmf+[0.0],[0.0]+pmf)]
    mean = math.fsum(fractions)
    if (not math.isclose(math.fsum(pmf),1.0,abs_tol=1e-12)
            or not math.isclose(math.fsum((guaranteed+i)*v for i,v in enumerate(pmf)),mean,abs_tol=1e-11)):
        raise ArithmeticError("Poisson-binomial mass or mean mismatch")
    def quantile(q):
        cdf = 0.
        for i,probability in enumerate(pmf):
            cdf += probability
            if cdf>=q:
                return guaranteed+i
        return possible
    return {"containing":possible,"guaranteed":guaranteed,"expected_under_uniform_draw":mean,
            "uniform_draw_central_95_count_interval":[quantile(.025),quantile(.975)],
            "selected_with_fixed_salt":sum(row["label"] in labels for row in chosen)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir",type=Path,required=True)
    ap.add_argument("--crosswalk",type=Path,default=Path(__file__).parent/"configs/ham-pad-crosswalk-v1.json")
    ap.add_argument("--output",type=Path,required=True)
    args = ap.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    shared = set(json.loads(args.crosswalk.read_bytes())["shared_nominal_leaves"].values())
    audits = json.loads((args.split_dir/"audit.json").read_bytes())["splits"]
    report = {"scope":"supplemental calibration-unit selection and exact conditional randomization distribution",
              "salt":SALT,"conditioning":"eligible representatives fixed; within-group uniform draw independent across groups",
              "crosswalk_sha256":hashlib.sha256(args.crosswalk.read_bytes()).hexdigest(),"seeds":{}}
    try:
        for seed,info in audits.items():
            path = args.split_dir/f"split_seed{seed}.csv"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info["sha256"]:
                raise ValueError("Source split changed")
            with path.open(encoding="utf-8-sig",newline="") as f:
                rows = [r for r in csv.DictReader(f) if r["representative"]=="1" and r["split"]=="calibration"]
            result = {}
            for scope,eligible in (("all",rows),("condition_on_shared_labels_first",[r for r in rows if r["label"] in shared])):
                groups = defaultdict(list)
                for r in eligible:
                    groups[r["split_group"]].append(r)
                def rank(r):
                    return (hashlib.sha256(f"{SALT}|{r['split_group']}|{r['image_id']}".encode()).digest(),r["image_id"])
                chosen = [min(groups[g],key=rank) for g in sorted(groups)]
                record = args.output/f"seed{seed}-{scope}.csv"
                with record.open("w",encoding="utf-8",newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["split_group","image_id"])
                    writer.writerows((r["split_group"],r["image_id"]) for r in chosen)
                labels = sorted({r["label"] for r in eligible})
                result[scope] = {"source_representatives":len(eligible),"group_draws":len(chosen),
                                 "selection_sha256":hashlib.sha256(record.read_bytes()).hexdigest(),
                                 "by_label":{label:support(groups,chosen,{label}) for label in labels},
                                 "any_shared":support(groups,chosen,shared)}
            report["seeds"][seed] = result
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"] = round(time.perf_counter()-start,3)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
