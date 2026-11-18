"""Audit label verification and acquisition provenance, and attribute development AUC by source strata."""
import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ham_dev_pilot import digest


def read(path):
    with path.open(newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def profile(rows, cols):
    result={"n":len(rows)}
    for col in cols:
        by=defaultdict(Counter)
        for r in rows:
            by[r[col]][r["label"]]+=1
        result[col]={"counts":{val:sum(n.values()) for val,n in sorted(by.items())},
                     "by_diagnosis":{val:dict(sorted(n.items())) for val,n in sorted(by.items())}}
    return result


def prepare(name,metadata,splits):
    ham=name=="ham"
    source=metadata/("HAM10000_metadata.original" if ham else "metadata.csv")
    source_info=json.loads((metadata/"audit.json").read_bytes())
    expected=source_info["metadata_download"]["sha256"] if ham else source_info["metadata_sha256"]
    if digest(source)!=expected:
        raise ValueError("Published metadata differs from the verified download")
    cols=("dx_type","dataset") if ham else ("biopsed",)
    raw=read(source)
    indexed={r["image_id" if ham else "img_id"]:r for r in raw}
    manifest=json.loads((splits/"audit.json").read_bytes())
    report={"metadata_sha256":expected,"seeds":{}}
    lookup={}
    for seed,info in manifest["splits"].items():
        p=splits/f"split_seed{seed}.csv"
        if digest(p)!=info["sha256"]:
            raise ValueError("Split changed")
        records=read(p)
        if len(raw)!=len(indexed) or len(records)!=len(raw) or set(indexed)!={r["image_id"] for r in records}:
            raise ValueError("Raw metadata and prepared image inventory differ")
        for r in records:
            original=indexed[r["image_id"]]
            if original["dx" if ham else "diagnostic"]!=r["label"]:
                raise ValueError("Original diagnostic label differs from prepared split")
            r.update({k:original[k].strip() for k in cols})
            if any(not r[k] for k in cols):
                raise ValueError("Missing verification or source field")
        reps=[r for r in records if r["representative"]=="1"]
        report["seeds"][seed]={role:profile([r for r in reps if r["split"]==role],cols)
                               for role in ("train","dev","calibration","test")}
        if seed=="17":
            lookup={r["image_id"]:r for r in records}
            report.update(images=profile(records,cols),representatives=profile(reps,cols),mixed={})
            for col in cols:
                levels={}
                for key,values in ((("analysis_group" if ham else "lesion_id"),records),("split_group",reps)):
                    by=defaultdict(set)
                    for r in values:
                        by[r[key]].add(r[col])
                    levels[key]={"groups":len(by),"multiple_field_values":sum(len(v)>1 for v in by.values())}
                report["mixed"][col]=levels
    return report,lookup


def decompose(wrong, base, augmented, strata):
    """Sum AUROC concordance changes over disjoint error/correct pairs."""
    pos,neg=np.flatnonzero(wrong),np.flatnonzero(~wrong)
    n=len(pos)*len(neg)
    def wins(scores):
        diff=scores[pos,None]-scores[None,neg]
        return (diff>0).astype(float)+.5*(diff==0)
    a,b=wins(base),wins(augmented)
    def summary(mask):
        count=int(mask.sum())
        return {"pairs":count,"base":float(a[mask].mean()) if count else None,
                "augmented":float(b[mask].mean()) if count else None,
                "contribution_to_overall_delta":float((b[mask]-a[mask]).sum()/n)}
    left,right=strata[pos,None],strata[None,neg]
    same=left==right
    within,between=summary(same),summary(~same)
    by={val:{"n":int((strata==val).sum()),"errors":int((wrong&(strata==val)).sum()),
             **summary((left==val)&(right==val))} for val in sorted(set(strata))}
    total=float(b.mean()-a.mean())
    if (sum(v["pairs"] for v in by.values())!=within["pairs"] or within["pairs"]+between["pairs"]!=n
            or not math.isclose(sum(v["contribution_to_overall_delta"] for v in by.values()),within["contribution_to_overall_delta"],abs_tol=1e-12)
            or not math.isclose(within["contribution_to_overall_delta"]+between["contribution_to_overall_delta"],total,abs_tol=1e-12)):
        raise ValueError("Pairwise AUC decomposition failed")
    return {"overall_delta":total,"pairs":n,"within":within,"between":between,"by_stratum":by}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("ham-metadata","ham-splits","pad-splits","pilot","balanced","increment","confirmation","crossed","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    a=ap.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    start=time.perf_counter()
    report={"scope":"verified dataset metadata and posthoc descriptive development comparisons",
            "auc_groups":"pooled original frozen detectors, two prior natural development batches only"}
    try:
        report["ham"],lookup=prepare("ham",a.ham_metadata,a.ham_splits)
        report["pad"],_=prepare("pad",a.pad_splits,a.pad_splits)
        cohorts={}
        infos={}
        for name,folder in (("original_fit",a.pilot),("first_natural",a.increment),("second_natural",a.confirmation),
                            ("balanced",a.balanced),("alternative",a.crossed)):
            info=json.loads((folder/"summary.json").read_bytes())
            if info["status"]!="complete" or digest(folder/"selection.csv")!=info["selection_sha256"]:
                raise ValueError("Previous cohort changed")
            selection=read(folder/"selection.csv")
            for r in selection:
                current=lookup[r["image_id"]]
                if current["label"]!=r["label"] or current["split"]!=r["role"] or current["representative"]!="1":
                    raise ValueError("Development cohort member or role differs from audited metadata")
            for role in sorted({r["role"] for r in selection}):
                cohorts[name+"_"+role]=[lookup[r["image_id"]] for r in selection if r["role"]==role]
            infos[name]=info
        if sum(map(len,cohorts.values()))!=len({r["split_group"] for batch in cohorts.values() for r in batch}):
            raise ValueError("Historical cohorts overlap under the current split")
        report["development"]={k:profile(rows,("dataset","dx_type")) for k,rows in cohorts.items()}
        people=cohorts["first_natural_dev"]+cohorts["second_natural_dev"]
        report["natural_pooled"]=profile(people,("dataset","dx_type"))
        report["risk_by_source"]={}
        for m,source in infos["second_natural"]["models"].items():
            first,second=a.increment/f"{m}.npz",a.confirmation/f"{m}.npz"
            if digest(first)!=infos["first_natural"]["models"][m]["cache_sha256"] or digest(second)!=source["cache_sha256"]:
                raise ValueError("Previous risk arrays changed")
            report["risk_by_source"][m]={}
            with np.load(first,allow_pickle=False) as u,np.load(second,allow_pickle=False) as v:
                for method,ref in source["readouts"].items():
                    wrong,base,aug=[np.r_[u[method+"_"+k],v[method+"_"+k]] for k in ("error","base","augmented")]
                    result={k:decompose(wrong,base,aug,np.array([r[k] for r in people])) for k in ("dataset","dx_type","label")}
                    if any(abs(row["overall_delta"]-ref["pooled_two_natural"]["difference"])>1e-12 for row in result.values()):
                        raise ValueError("Metadata linkage changed the original pooled metric")
                    report["risk_by_source"][m][method]=result
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (a.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
