"""Rebuild HAM v1.4 splits, content audit and sensitivity masks from the original images."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images",type=Path,required=True,help="Extracted HAM v4 image directory")
    ap.add_argument("--source-dir",type=Path,required=True,help="Directory with dataset_api.json and HAM10000_metadata.original")
    ap.add_argument("--pair-dir",type=Path,help="Directory containing both fixed published CSV pair tables")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if not 1<=a.workers<=8:
        ap.error("workers must be 1..8")
    out=a.output.resolve()
    images=a.images.resolve()
    if not images.is_dir():
        ap.error("Original images directory missing")
    out.mkdir(exist_ok=False,parents=True)
    script_dir=Path(__file__).resolve().parent
    old=out/"v1.3-splits"
    audit=out/"v1.3-audit"
    ranking=out/"appearance"
    shortlist=out/"geometry-shortlist"
    cross=out/"geometry-cross"
    within=out/"geometry-within"
    new=out/"splits"
    current=out/"audit"
    pairs=["--pair-dir",a.pair_dir.resolve()] if a.pair_dir else []
    geom=["--audit",audit,"--appearance",ranking,"--shortlist",shortlist,"--workers",str(a.workers)]
    steps=[
        ("prepare_ham.py",["--source-dir",a.source_dir.resolve(),*pairs,"--output",old]),
        ("audit_images.py",["--images",images,"--split",old/"split_seed17.csv","--output",audit]),
        ("audit_ham_appearance.py",["--audit",audit,"--split-dir",old,"--output",ranking]),
        ("review_ham_candidates.py",["--audit",audit,"--split-dir",old,"--appearance",ranking,"--output",shortlist]),
        ("verify_ham_appearance_geometry.py",[*geom,"--output",cross]),
        ("verify_ham_appearance_geometry.py",[*geom,"--within-only","--output",within]),
        ("prepare_ham_scene_splits.py",["--split-dir",old,"--appearance",ranking,
             "--cross-geometry",cross,"--within-geometry",within,"--output",new]),
        ("audit_images.py",["--images",images,"--split",new/"split_seed17.csv","--output",current]),
        ("summarize_ham_duplicates.py",["--audit",current,"--split-dir",new,"--output",out/"summary"]),
        ("ham_overlap_sensitivity.py",["--audit",current,"--split-dir",new,"--output",out/"sensitivity"]),
    ]
    env=os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS",str(a.workers))
    report={"protocol":"v1.4","status":"running","stages":[]}
    log=out/"run.json"
    start=time.perf_counter()
    try:
        for script,arguments in steps:
            report["active_stage"]=str(arguments[-1].name)
            log.write_text(json.dumps(report,indent=2),encoding="utf-8")
            now=time.perf_counter()
            print(f"Starting {report['active_stage']}",flush=True)
            subprocess.run([sys.executable,str(script_dir/script),*map(str,arguments)],env=env,check=True)
            report["stages"].append({"stage":report["active_stage"],"script":script,
                                     "seconds":round(time.perf_counter()-now,3)})
        report["status"]="complete"
        del report["active_stage"]
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        log.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
