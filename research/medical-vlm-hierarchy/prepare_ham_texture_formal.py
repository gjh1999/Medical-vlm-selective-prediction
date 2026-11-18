"""Rebuild HAM texture-based splits and sensitivity masks starting from a verified v1.4 run."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v14",type=Path,required=True,help="Completed prepare_ham_formal.py output on this host")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,choices=range(1,9),default=3)
    parser.add_argument("--existing-evidence",type=Path,nargs=4,metavar=("APPEARANCE","WORDS","GEOMETRY","REVIEW"),
                        help="Reuse both retrieval tables, all-candidate geometry and independent review outputs")
    args=parser.parse_args()
    base=args.v14.resolve()
    output=args.output.resolve()
    if json.loads((base/"run.json").read_bytes())["status"]!="complete":
        raise ValueError("Complete the v1.4 rebuild first")
    audit=base/"audit"
    splits=base/"splits"
    old=base/"appearance"
    original_images=Path(json.loads((audit/"audit.json").read_bytes())["images_root"])
    if not original_images.is_dir():
        raise ValueError("Original images are unavailable at the local v1.4 audit path")
    stages=[]
    if args.existing_evidence:
        old,words,geom,review=[p.resolve() for p in args.existing_evidence]
    else:
        controls=output/"controls"
        cache=output/"local-features"
        words=output/"words"
        geom=output/"geometry"
        review=output/"review"
        workers=["--workers",str(args.workers)]
        stages.extend([
            ("audit_ham_texture_controls.py",["--audit",base/"v1.3-audit","--baseline",base/"geometry-cross",*workers,"--output",controls]),
            ("prepare_ham_texture_cache.py",["--audit",audit,*workers,"--output",cache]),
            ("audit_ham_visual_words.py",["--cache",cache,"--pair-dir",splits,"--appearance",old,*workers,"--output",words]),
            ("verify_ham_visual_word_pairs.py",["--cache",cache,"--words",words,"--appearance",old,"--controls",controls,
                                               "--splits",splits,"--max-rank","20","--include-appearance",*workers,"--output",geom]),
            ("review_ham_texture_matches.py",["--audit",audit,"--controls",controls,"--screen",geom,*workers,"--output",review])])
    new=output/"splits"
    current=output/"audit"
    stages.extend([
        ("prepare_ham_texture_splits.py",["--splits",splits,"--words",words,"--appearance",old,
                                          "--geometry",geom,"--review",review,"--output",new]),
        ("audit_images.py",["--images",original_images,"--split",new/"split_seed17.csv","--output",current]),
        ("summarize_ham_duplicates.py",["--audit",current,"--split-dir",new,"--output",output/"summary"]),
        ("ham_overlap_sensitivity.py",["--audit",current,"--split-dir",new,"--output",output/"sensitivity"])])
    output.mkdir(exist_ok=False,parents=True)
    script_dir=Path(__file__).resolve().parent
    env=os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS",str(args.workers))
    env.setdefault("OMP_NUM_THREADS",str(args.workers))
    log=output/"run.json"
    report={"protocol":"v1.5","reused_verified_screen":bool(args.existing_evidence),"stages":[]}
    start=time.perf_counter()
    try:
        for script,arguments in stages:
            now=time.perf_counter()
            report.update(status="running",active_stage=arguments[-1].name)
            log.write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(f"Starting {arguments[-1].name}",flush=True)
            subprocess.run([sys.executable,str(script_dir/script),*map(str,arguments)],env=env,check=True)
            report["stages"].append({"stage":arguments[-1].name,"script":script,"seconds":round(time.perf_counter()-now,3)})
        del report["active_stage"]
        report["status"]="complete"
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        log.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
