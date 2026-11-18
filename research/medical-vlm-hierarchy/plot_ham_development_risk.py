"""Aggregate empirical leaf risk at matched coverage for the two natural HAM development batches."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ham_dev_pilot import digest, error_auc


COVERAGE = (10,20,25,30,40,50,60,70,75,80,90,100)


def curve(errors, scores):
    order=np.argsort(scores,kind="stable")
    sorted_scores=scores[order]
    sorted_errors=errors[order].astype(int)
    values,starts,counts=np.unique(sorted_scores,return_index=True,return_counts=True)
    block_errors=np.add.reduceat(sorted_errors,starts)
    cumulative=block_errors.cumsum()
    risks=[]
    previous=0
    for start,count,bad in zip(starts,counts,block_errors):
        # Average the cumulative error count over all tie permutations within a score block.
        k=np.arange(1,count+1)
        risks.extend((previous+k*(bad/count))/(start+k))
        previous+=bad
    points={}
    for percent in COVERAGE:
        take=math.ceil(len(errors)*percent/100)
        block=np.searchsorted(values,sorted_scores[take-1])
        n=int(starts[block]+counts[block])
        accepted=scores<=values[block]
        wrong=int(cumulative[block])
        if int(accepted.sum())!=n or int(errors[accepted].sum())!=wrong:
            raise ValueError("Ranked and thresholded acceptance disagree")
        points[str(percent)]={"accepted":n,"errors":wrong,"risk":wrong/n,"coverage":n/len(errors)}
    if not np.isclose(risks[-1],errors.mean()):
        raise ValueError("Full coverage must recover the original leaf error rate")
    return {"empirical_aurc":float(np.mean(risks)),"distinct_scores":len(values),"coverage_points":points}


def plot(report,directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig,axes=plt.subplots(1,2,figsize=(8.2,4.25),sharex=True,sharey=True)
    n=report["models"]["clip"]["cosine_centroid"]["pooled"]["n"]
    fig.text(.10,.96,"Leaf-error risk at matched coverage",fontsize=15,weight="bold")
    fig.text(.10,.905,f"HAM10000  |  {n} development groups  |  Image nearest-centroid readouts",fontsize=9.5)
    for ax,(model,title) in zip(axes,(("clip","OpenAI CLIP"),("biomedclip","BiomedCLIP"))):
        data=report["models"][model]["cosine_centroid"]["pooled"]
        for key,label,style,color in (("base","Confidence features","--","#536578"),
                                     ("augmented","With parent disagreement","-","#006d83")):
            pts=list(data[key]["coverage_points"].values())
            ax.plot([v["coverage"] for v in pts],[v["risk"] for v in pts],style,
                    color=color,label=label,marker="o",markersize=3,linewidth=1.7)
        ax.set_title(title,loc="left",fontsize=12,weight="bold")
        ax.set_xlabel("Groups accepted",fontsize=10)
        ax.set_xlim(.07,1.03)
        ax.set_xticks([.1,.25,.5,.75,1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(axis="y",color="#d9dfe5",lw=.7)
        ax.spines[["right","top"]].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Leaf errors among accepted groups",fontsize=10)
    axes[0].set_ylim(bottom=0)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,0))
    fig.subplots_adjust(left=.10,right=.98,top=.78,bottom=.24,wspace=.15)
    fig.savefig(directory/"risk-coverage.png",dpi=210,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return matplotlib.__version__


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("increment","confirmation","output"):
        ap.add_argument("--"+k,type=Path,required=True)
    a=ap.parse_args()
    start=time.perf_counter()
    a.output.mkdir(parents=True,exist_ok=False)
    report={"scope":"descriptive development risk at matched coverage; no operating threshold selected",
            "tie_rule":"include all scores tied at requested coverage; AURC averages over tie order",
            "paired_evaluation_bootstrap":{"per_batch":True,"repetitions":2000,"seed":17043},"models":{}}
    try:
        infos=[json.loads((p/"summary.json").read_bytes()) for p in (a.increment,a.confirmation)]
        if any(r["status"]!="complete" or digest(p/"selection.csv")!=r["selection_sha256"]
               for p,r in zip((a.increment,a.confirmation),infos)):
            raise ValueError("Changed source cohorts")
        report["selection_sha256"]=[r["selection_sha256"] for r in infos]
        first_n,second_n=infos[0]["n_eval"],infos[1]["selected"]
        rng=np.random.default_rng(17043)
        draws=np.hstack([rng.integers(0,first_n,(2000,first_n)),first_n+rng.integers(0,second_n,(2000,second_n))])
        for name in infos[0]["models"]:
            paths=[p/f"{name}.npz" for p in (a.increment,a.confirmation)]
            if list(map(digest,paths))!=[r["models"][name]["cache_sha256"] for r in infos]:
                raise ValueError("Risk score cache changed")
            report["models"][name]={}
            with np.load(paths[0],allow_pickle=False) as old, np.load(paths[1],allow_pickle=False) as new:
                for method in infos[0]["models"][name]["readouts"]:
                    def both(key):
                        return np.r_[old[method+"_"+key],new[method+"_"+key]]
                    wrong,base,augmented=both("error"),both("base"),both("augmented")
                    ref=infos[1]["models"][name]["readouts"][method]["pooled_two_natural"]
                    if ref["n"]!=len(wrong) or ref["errors"]!=int(wrong.sum()):
                        raise ValueError("Pooled cohorts are not the verified population")
                    delta=error_auc(wrong,augmented)-error_auc(wrong,base)
                    if abs(delta-ref["difference"])>1e-12:
                        raise ValueError("Pooled detector comparison differs")
                    result={}
                    for batch,sl in (("first",slice(0,first_n)),("second",slice(first_n,None)),("pooled",slice(None))):
                        result[batch]={"n":len(wrong[sl]),"base":curve(wrong[sl],base[sl]),
                                       "augmented":curve(wrong[sl],augmented[sl])}
                    diff=[error_auc(wrong[s],augmented[s])-error_auc(wrong[s],base[s]) for s in draws
                          if np.any(wrong[s]) and not np.all(wrong[s])]
                    result["pooled"]["fixed_detector_auroc_difference_interval"]=np.percentile(diff,[2.5,97.5]).tolist()
                    result["pooled"]["valid_resamples"]=len(diff)
                    report["models"][name][method]=result
        report.update(matplotlib=plot(report,a.output),status="complete")
    except Exception as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report["seconds"]=round(time.perf_counter()-start,3)
        (a.output/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
