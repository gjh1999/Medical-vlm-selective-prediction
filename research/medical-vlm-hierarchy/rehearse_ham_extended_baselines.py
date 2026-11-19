"""Rehearse the extended evaluator on audited, disjoint, previously used development cohorts."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from audit_ham_temperature import loss
from evaluate_ham_frozen_baselines import CONTROLLER, PROMPTS, RECIPE as BASE_RECIPE, config_digest, digest
from ham_dev_pilot import js, softmax
from ham_extended_baselines import RECIPE, compare_prompt_templates, evaluate_readout
from reallocate_ham_development_training import read_cohorts
from rehearse_ham_baseline_controller import exact_pass
from selective_core import Taxonomy


def verify_families(families, controller):
    checked = 0
    for family in families.values():
        delta = family["per_check_delta"]
        if delta != .05 / 32 or len(family["calibration"]) != 16:
            raise ValueError("Candidate budget differs between methods")
        passing = {}
        for slot, counts in family["calibration"].items():
            leaf = exact_pass(counts["leaf_errors"], counts["leaf_n"], controller["leaf_target"], delta)
            node = exact_pass(counts["node_errors"], counts["node_n"], controller["nonroot_target"], delta)
            if counts["feasible"] != (leaf and node):
                raise ArithmeticError("Independent binomial probability gives a different feasibility result")
            if leaf and node:
                passing[slot] = (counts["correct_information_utility"], counts["leaf_coverage"], slot)
            checked += 2
        if max(passing, key=passing.get, default=None) != family["selected"]:
            raise ArithmeticError("Independent risk-policy selection differs")
    return checked


def summarize(readout, controller, reference):
    summary = {k: readout[k] for k in ("development_groups", "native_dev_nll", "fitted_dev_nll", "temperature_at_bound")}
    summary["temperatures"] = {}
    checks = matched = 0
    for name, values in readout["temperatures"].items():
        population = values["populations"]["development_rehearsal"]
        checks += verify_families(population["families"], controller)
        methods = {}
        for method, family in population["families"].items():
            audits = [r for r in family["calibration"].values() if r["node_upper"] is not None]
            methods[method] = {"selected": family["selected"], "distinct_cutoffs": family["distinct_cutoffs"],
                               "minimum_node_upper": min((r["node_upper"] for r in audits), default=None),
                               "passing_slots": sum(r["feasible"] for r in family["calibration"].values())}
        sets = {}
        for alpha, value in population["lac"].items():
            sets[alpha] = {view: {"n": test["n"], "set_contains_truth": test["set_contains_truth"],
                                  "mean_set_size": test["mean_set_size"], "root_outputs": test["n"] - test["projection"]["node_n"],
                                  "correct_nonroot": test["projection"]["node_n"] - test["projection"]["node_errors"]}
                           for view, test in value["test"].items()}
            if name == "native":
                saved = reference[alpha]
                if value["order_statistic"] != saved["order_statistic"] or abs(value["quantile"] - saved["quantile"]) > 1e-12:
                    raise ArithmeticError("Native conformal quantile differs from prior development result")
                for view, old_name in (("first_natural", "natural"), ("balanced", "balanced")):
                    test = value["test"][view]
                    old = saved[old_name]
                    if (test["set_contains_truth"], sets[alpha][view]["correct_nonroot"],
                        test["empty_set_to_root"], test["projection"]["node_n"]) != (
                        old["set_contains_truth"], old["correct_nonroot_projection"],
                        old["empty_set_to_root"], old["projection_metrics"]["node_n"]
                    ):
                        raise ArithmeticError("Native conformal output counts differ from prior development result")
                    matched += 1
        summary["temperatures"][name] = {"inverse_temperature": values["inverse_temperature"], "families": methods,
                                          "lac": sets, "leaf_reliability": population["leaf_reliability"]}
    return summary, checks, matched


def run(args, config, controller, tree):
    paths = [args.pilot, args.balanced, args.increment, args.confirmation, args.crossed]
    info, labels, (old_train, new_train), membership = read_cohorts(paths, tree)
    old_labels = labels[1][old_train]
    names = np.asarray(tree.leaves)
    truth_blocks = (labels[4][~new_train], labels[0], labels[2], labels[3], labels[1][~old_train])
    sizes = [len(block) for block in truth_blocks]
    if sizes != [128, 128, 96, 96, 56] or len(old_labels) != 56:
        raise ValueError("Unexpected development-rehearsal cohort sizes")
    rows = np.split(np.arange(sum(sizes)), np.cumsum(sizes)[:-1])
    d, cal, first, second, balanced = [(idx, names[block].tolist()) for idx, block in zip(rows, truth_blocks)]
    populations = {"development_rehearsal": (*cal, {"first_natural": first, "second_natural": second, "balanced": balanced})}
    reference = json.loads(args.lac_reference.read_bytes())["models"]
    results = {}
    compact = {}
    checks = matched = temperature_checks = 0
    for model in ("clip", "biomedclip"):
        source = info[2]["models"][model]
        hashes = [source["fit_cache"], source["support_cache"], source["cache_sha256"],
                  info[3]["models"][model]["cache_sha256"], info[4]["models"][model]["cache_sha256"]]
        files = [p / f"{model}.npz" for p in paths]
        if list(map(digest, files)) != hashes:
            raise ValueError("Archived feature cache fingerprints differ")
        with (np.load(files[0]) as p, np.load(files[1]) as b, np.load(files[2]) as i,
              np.load(files[3]) as c, np.load(files[4]) as x):
            images = np.concatenate([x["image_features"][~new_train], p["image_features"], i["image_features"],
                                     c["image_features"], b["image"][~old_train]]).astype(np.float64)
            meta = info[0]["models"][model]
            leaf_text = p["text_features"][:, [meta["text_node_order"].index(y) for y in tree.leaves]].astype(np.float64)
            support = b["image"][old_train].astype(np.float64)
        scale = meta["logit_scale"]
        prompt_probabilities = softmax(scale * (images @ leaf_text.transpose(0, 2, 1)))
        prompt_confidence = -js(*prompt_probabilities)
        results[model] = {}
        compact[model] = {}
        for name, head in (("frozen_text", leaf_text.mean(axis=0)),
                           ("cosine_centroid", np.stack([support[old_labels == k].mean(0) for k in range(len(names))]))):
            logits = scale * (images @ (head / np.linalg.norm(head, axis=1, keepdims=True)).T)
            report = evaluate_readout(tree, logits, prompt_confidence, d, populations, config, controller)
            grid = np.geomspace(*config["inverse_temperature_bounds"], 1001)
            grid_nll = min(loss(logits[d[0]], truth_blocks[0], beta) for beta in grid)
            if report["fitted_dev_nll"] > min(grid_nll, report["native_dev_nll"]) + 1e-12:
                raise ArithmeticError("Development temperature failed an independent NLL grid check")
            temperature_checks += 1
            compact[model][name], n_checks, n_matched = summarize(report, controller, reference[model][name])
            results[model][name] = report
            checks += n_checks
            matched += n_matched
            if name == "frozen_text":
                prompt_results = compare_prompt_templates(tree, prompt_probabilities, logits, populations)
                results[model]["prompt_template_comparison"] = prompt_results
                compact[model]["prompt_template_comparison"] = {
                    view: {name: {"n": m["n"], "correct": m["leaf_n"] - m["leaf_errors"], "macro_f1": m["macro_f1"]}
                           for name, m in heads.items()}
                    for view, heads in prompt_results["development_rehearsal"].items()}
    return results, compact, {"membership_sha256": membership, "calibration_constraint_checks": checks,
                              "historical_conformal_cohort_matches": matched, "temperature_grid_checks": temperature_checks,
                              "development_groups": sizes[0], "proxy_calibration_groups": sizes[1],
                              "evaluation_groups": dict(zip(populations["development_rehearsal"][2], sizes[2:]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("pilot", "balanced", "increment", "confirmation", "crossed", "lac-reference", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    summary = {"scope": "Previously used, disjoint HAM development cohorts; no formal calibration or test predictions",
               "recipe_sha256": config_digest(RECIPE), "numpy": np.__version__, "python": platform.python_version()}
    try:
        config = json.loads(RECIPE.read_bytes())
        base = json.loads(BASE_RECIPE.read_bytes())
        if (config["base_recipe_sha256"] != config_digest(BASE_RECIPE) or
                base["prompt_config_sha256"] != config_digest(PROMPTS) or
                base["controller_rehearsal_sha256"] != config_digest(CONTROLLER)):
            raise ValueError("Frozen taxonomy, policy grid or baseline recipe differs")
        controller = json.loads(CONTROLLER.read_bytes())
        tree = Taxonomy(json.loads(PROMPTS.read_bytes())["parent"])
        results, compact, verified = run(args, config, controller, tree)
        summary.update(status="complete", models=compact, **verified)
        (args.output / "details.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
