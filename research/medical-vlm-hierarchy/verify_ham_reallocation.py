"""Check the individual reallocated HAM contrasts with torch, then draw the conditional distributions."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import normalize

from ham_dev_pilot import digest
from reallocate_ham_development_training import ROOT, allocations, read_cohorts
from selective_core import Taxonomy
from verify_error_increment import lbfgs
from verify_ham_crossed_fit import features
from verify_ham_dev_pilot import rank_auc


def draw_figure(values, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8.6, 4.65))
    fig.subplots_adjust(left=.09, right=.98, bottom=.24, top=.72, wspace=.13)
    fig.text(.09, .953, "How much does the training sample change error ranking?", fontsize=14.3, weight="bold")
    fig.text(.09, .895, "Same 616 previously used groups   |   128 complementary partitions, four training combinations each",
             fontsize=9, color="#374151")
    colors = {"clip": "#005a82", "biomedclip": "#ab4600"}
    labels = {"clip": "OpenAI CLIP", "biomedclip": "BiomedCLIP"}
    for idx, (axis, title) in enumerate(zip(axes, ["Natural development mix  ·  192 groups", "Class-balanced  ·  56 groups"])):
        axis.set_title(title, loc="left", fontsize=10.5, weight="bold", pad=8)
        axis.axvline(0, color="#374151", lw=1, ls="--")
        for model in colors:
            sorted_values = np.sort(values[model][..., idx].ravel())
            mass = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            axis.step(np.r_[sorted_values[0], sorted_values], np.r_[0, mass], where="post", lw=2,
                      color=colors[model], label=labels[model])
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Change in error AUROC", fontsize=9)
        axis.tick_params(labelsize=8)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[0].set_ylabel("Cumulative share of training combinations", fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[0].set_ylim(0, 1.02)
    fig.text(.09, .09, "Positive values mean that adding parent disagreement improved ranking. Each curve uses 512 dependent combinations.",
             fontsize=8.1, color="#374151")
    fig.text(.09, .052, "Training budget per combination: 8 support groups per diagnosis and 128 groups to fit the risk model.",
             fontsize=8.1, color="#374151")
    fig.savefig(destination, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "balanced", "increment", "confirmation", "crossed", "results", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    out = {"method": "torch float64 independent signals, L-BFGS, Mann-Whitney ranks on the same design"}
    try:
        torch.set_num_threads(1)
        cfg_path = ROOT / "configs/ham-g1-reallocation-v1.json"
        cfg = json.loads(cfg_path.read_bytes())
        rule = json.loads((ROOT / "configs/ham-error-increment-v1.json").read_bytes())
        tree = Taxonomy(json.loads((ROOT / "configs/ham-dev-pilot-v1.json").read_bytes())["parent"])
        results = json.loads((args.results / "summary.json").read_bytes())
        ref_path = args.results / "cell_differences.npz"
        if results["status"] != "complete" or digest(ref_path) != results["cell_differences_sha256"] or digest(cfg_path) != results["config_sha256"]:
            raise ValueError("Reallocation output differs from the declared design")
        paths = [args.pilot, args.balanced, args.increment, args.confirmation, args.crossed]
        infos, labels, (old_tr, new_tr), signature = read_cohorts(paths, tree)
        sy = np.r_[labels[1][old_tr], labels[4][new_tr]]
        fit_y = np.r_[labels[0], labels[4][~new_tr]]
        ey = np.r_[labels[2], labels[3], labels[1][~old_tr]]
        draws = list(allocations(sy, len(fit_y), cfg))
        for ss, ff in draws:
            if (not np.array_equal(np.sort(np.concatenate(ss)), np.arange(len(sy))) or
                    not np.array_equal(np.sort(np.concatenate(ff)), np.arange(len(fit_y))) or
                    any(np.any(np.bincount(sy[s], minlength=7) != cfg["support_per_leaf"]) for s in ss) or
                    any(len(f) != cfg["fit_size"] for f in ff)):
                raise ValueError("Noncomplementary or unequal-budget training sets")
        checked = {}
        values = {}
        for name in ("clip", "biomedclip"):
            with np.load(ref_path) as ref:
                reference = ref[name]
            with (np.load(paths[0]/f"{name}.npz") as p, np.load(paths[1]/f"{name}.npz") as b,
                  np.load(paths[2]/f"{name}.npz") as i, np.load(paths[3]/f"{name}.npz") as c,
                  np.load(paths[4]/f"{name}.npz") as r):
                sup = torch.tensor(np.concatenate([b["image"][old_tr], r["image_features"][new_tr]]), dtype=torch.float64)
                x = np.concatenate([p["image_features"], r["image_features"][~new_tr],
                                    i["image_features"], c["image_features"], b["image"][~old_tr]])
                t = p["text_features"]
            meta = infos[0]["models"][name]
            actual = np.zeros(reference.shape)
            k = cfg["evaluation"]["pooled_natural"]
            for step, (supports, fits) in enumerate(draws):
                for a, indices in enumerate(supports):
                    centers = normalize(torch.stack([sup[indices[sy[indices] == y]].mean(0) for y in range(7)]), dim=1)
                    pred, mat = features(x, t, centers, meta["text_node_order"], tree, meta["logit_scale"])["cosine_centroid"]
                    target = pred != np.r_[fit_y, ey]
                    for b, f in enumerate(fits):
                        scale = mat[f].std(0)
                        scale[scale < 1e-12] = 1.
                        z = (mat - mat[f].mean(0)) / scale
                        ranks = []
                        for width in (3, 4):
                            coef = lbfgs(z[f, :width], target[f], rule["l2_penalty"])
                            score = coef[0] + z[len(fit_y):, :width] @ coef[1:]
                            ranks.append(np.array([rank_auc(target[len(fit_y):][part], score[part])
                                                   for part in (slice(0, k), slice(k, None))]))
                        actual[step, a, b] = ranks[1] - ranks[0]
            gap = np.abs(actual - reference)
            checked[name] = {"increments": int(actual.size), "within_1e_minus_12": int((gap <= 1e-12).sum()),
                             "max_absolute_difference": float(gap.max()), "sign_disagreements": int(np.sum(np.sign(actual) != np.sign(reference)))}
            values[name] = reference
        out.update(status="complete", models=checked, allocations_checked=len(draws), membership_sha256=signature)
        if args.plot:
            path = args.output / "training-allocations.png"
            draw_figure(values, path)
            out["figure_sha256"] = digest(path)
    except Exception as exc:
        out.update(status="failed", error=str(exc))
        raise
    finally:
        out["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
