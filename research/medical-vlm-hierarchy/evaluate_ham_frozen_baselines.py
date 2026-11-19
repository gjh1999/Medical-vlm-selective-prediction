"""Validate HAM v1.5 inputs and evaluate fixed leaf, Flat and Climbing baselines."""
import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

from audit_ham_development_exposure import SEEDS, digest, read_csv, read_split
from extract_features import BIOMED_SHA256, REVISION, WEIGHT_SHA256, read_inputs
from selective_core import Taxonomy, policy_actions, select_policy


ROOT = Path(__file__).resolve().parent
RECIPE = ROOT / "configs/ham-frozen-baseline-v1.json"
PROMPTS = ROOT / "configs/ham-dev-pilot-v1.json"
CONTROLLER = ROOT / "configs/ham-controller-rehearsal-v1.json"
RELEASE = ROOT / "reports/E00-ham-texture-splits-01.json"
EXPOSURE = ROOT / "reports/E00-ham-development-exposure-01.json"
MODELS = {"clip": ("OpenAI-ViT-B-16", WEIGHT_SHA256), "biomedclip": (REVISION, BIOMED_SHA256)}
POPULATIONS = {"all": (), "without_supervised": ("supervised",),
               "without_image_overlap": ("image_overlap",), "without_either": ("supervised", "image_overlap")}


def config_digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def group_representatives(rows):
    representatives = {}
    for row in sorted(rows, key=lambda row: row["image_id"]):
        representatives.setdefault(row["split_group"], row)
    return list(representatives.values())


def prepare_cohorts(splits, mask_path, verified_rows, recipe, tree):
    release = json.loads(RELEASE.read_bytes())
    exposure = json.loads(EXPOSURE.read_bytes())
    if digest(mask_path) != exposure["mask_sha256"]:
        raise ValueError("Development exposure and image-overlap mask fingerprint differs")
    mask = {row["split_group"]: row for row in read_csv(mask_path)}
    cohort = {}
    for seed in SEEDS:
        expected = release["splits"][str(seed)]
        split = read_split(splits, seed, expected["sha256"])
        same = ("analysis_group", "split_group", "representative", "label")
        if seed == 17:
            same += ("split",)
        if split.keys() != verified_rows.keys() or any(
            tuple(row[f] for f in same) != tuple(verified_rows[key][f] for f in same)
            or row["split"] != mask[row["split_group"]][f"seed{seed}"]
            for key, row in split.items()
        ):
            raise ValueError(f"Audit, split or mask disagree for seed {seed}")
        reps = [row for row in split.values() if row["representative"] == "1"]
        train = group_representatives(row for row in reps if row["split"] == "train")
        ranked = sorted(train, key=lambda row: hashlib.sha256(
            f'{recipe["selection_salt"]}|{seed}|{row["split_group"]}'.encode()).digest())
        n = recipe["centroid_train_groups_per_leaf"]
        support = {leaf: [row for row in ranked if row["label"] == leaf][:n] for leaf in tree.leaves}
        if any(len(rows) != n for rows in support.values()):
            raise ValueError(f"Insufficient distinct training groups for seed {seed}")
        evaluation = {}
        for name, flags in POPULATIONS.items():
            roles = {role: [row for row in reps if row["split"] == role and
                            all(mask[row["split_group"]][flag] == "0" for flag in flags)]
                     for role in ("calibration", "test")}
            evaluation[name] = (group_representatives(roles["calibration"]), roles["calibration"],
                                group_representatives(roles["test"]), roles["test"])
        cohort[str(seed)] = (support, evaluation)
    return cohort


def describe_cohorts(cohorts):
    def counts(rows):
        return {"n": len(rows), "labels": dict(sorted(Counter(row["label"] for row in rows).items()))}

    return {seed: {"train_per_leaf": {leaf: len(rows) for leaf, rows in support.items()},
                   "populations": {name: dict(zip(("calibration_groups", "calibration_analysis_units",
                                                   "test_groups", "test_analysis_units"), map(counts, roles)))
                                   for name, roles in evaluation.items()}}
            for seed, (support, evaluation) in cohorts.items()}


def load_cache(directory, model, audit, manifest, prompts):
    import numpy as np

    meta = json.loads((directory / "cache.json").read_bytes())
    revision, weights = MODELS[model]
    required = {"status": "complete", "scope": "all_audited_images", "device": "cuda", "model": model,
                "revision": revision, "weights_sha256": weights, "model_training": False,
                "audit_sha256": digest(audit / "audit.json"), "manifest_sha256": digest(audit / "manifest.csv"),
                "precision": "float32", "normalization": "L2", "input_count": len(manifest)}
    if any(meta.get(key) != value for key, value in required.items()):
        raise ValueError(f"Expected complete pinned CUDA feature cache for {model}")
    if model == "clip" and (meta.get("loader_commit"), meta.get("visual_activations")) != (
        "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6", ["QuickGELU"]
    ):
        raise ValueError("OpenAI CLIP loader or activation differs")
    if read_csv(directory / "manifest.csv") != manifest:
        raise ValueError(f"Cache row order differs from image audit for {model}")
    text = meta["text"]
    nodes = sorted(prompts["names"])
    scale = text["checkpoint_logit_scale"]
    if (text["config_sha256"] != digest(PROMPTS) or text["templates"] != prompts["templates"]
            or text["node_order"] != nodes or not math.isfinite(scale) or scale <= 0):
        raise ValueError(f"Cache prompt configuration or logit scale differs for {model}")
    arrays = []
    for filename, fingerprint, shape in (
        ("image_features.npy", meta["features_sha256"], (len(manifest), 512)),
        ("text_features.npy", text["features_sha256"], (len(prompts["templates"]), len(nodes), 512)),
    ):
        path = directory / filename
        if digest(path) != fingerprint:
            raise ValueError(f"Feature fingerprint differs for {model} {filename}")
        values = np.load(path, allow_pickle=False)
        if (values.dtype != np.float32 or values.shape != shape or
                not np.allclose(np.linalg.norm(values, axis=-1), 1, atol=1e-5, rtol=0)):
            raise ValueError(f"Unexpected features for {model} {filename}")
        arrays.append(values.astype(np.float64))
    fields = ("revision", "weights_sha256", "features_sha256", "versions", "gpu", "batch_size", "cuda_build",
              "encoding_seconds", "peak_gpu_memory_bytes")
    provenance = {key: meta[key] for key in fields}
    provenance["text_features_sha256"] = text["features_sha256"]
    images, text_embeddings = arrays
    return images, text_embeddings, scale, provenance


def test_metrics(tree, labels, predictions, branch, unrestricted=False):
    report = tree.metrics(labels, predictions)
    errors = Counter(y for y, pred in zip(labels, predictions) if pred != tree.root and branch[y] != branch[pred])
    for label, metrics in report["per_class"].items():
        metrics["cross_branch_errors"] = errors[label]
        metrics["cross_branch_risk"] = errors[label] / metrics["node_n"] if metrics["node_n"] else None
    report["cross_branch_errors"] = sum(errors.values())
    report["cross_branch_risk"] = report["cross_branch_errors"] / report["node_n"] if report["node_n"] else None
    if unrestricted:
        confusion = Counter(zip(labels, predictions))
        leaf_f1 = {}
        for leaf in tree.leaves:
            denominator = labels.count(leaf) + predictions.count(leaf)
            leaf_f1[leaf] = 2 * confusion[leaf, leaf] / denominator if denominator else None
        defined = [score for score in leaf_f1.values() if score is not None]
        report["leaf_f1"] = leaf_f1
        report["macro_f1"] = sum(defined) / len(defined)
        report["accuracy"] = 1 - report["leaf_risk"]
    return report


def fixed_policy_results(tree, calibration, test_groups, test_units, probabilities, controller):
    def inputs(rows):
        return [row["label"] for row in rows], [probabilities[row["image_id"]] for row in rows]

    labels, proba = inputs(calibration)
    candidates = policy_actions(tree, proba, controller)
    trial = select_policy(tree, labels, candidates, controller["family_failure_probability"],
                          controller["leaf_target"], controller["nonroot_target"])
    best = {"joint": trial["selected"]}
    for mode in controller["baselines"]:
        eligible = [row for row in trial["audit"] if row["feasible"] and row["name"].startswith(mode + "_")]
        winner = max(eligible, key=lambda row: (row["metrics"]["correct_information_utility"],
                     row["metrics"]["leaf_coverage"], row["name"]), default=None)
        best[mode] = winner["name"] if winner else None
    branch = {}
    for node in tree.parent:
        ancestor = node
        while tree.parent[ancestor] != tree.root:
            ancestor = tree.parent[ancestor]
        branch[node] = ancestor
    test_results = {}
    for name, rows in (("analysis_units", test_units), ("group_representatives", test_groups)):
        labels, proba = inputs(rows)
        chosen = {"unrestricted_leaf": [tree.flat(row) for row in proba]}
        for mode, policy in best.items():
            if policy is None:
                chosen[mode] = [tree.root] * len(rows)
            else:
                kind, cutoff = policy.rsplit("_", 1)
                action = {"flat": tree.flat, "climbing": tree.climbing}[kind]
                chosen[mode] = [action(row, float(cutoff)) for row in proba]
        test_results[name] = {mode: test_metrics(tree, labels, predictions, branch, mode == "unrestricted_leaf")
                              for mode, predictions in chosen.items()}
    return {"calibration": trial, "selected": best, "test": test_results}


def evaluate(model, directory, audit, manifest, prompts, tree, cohorts, controller):
    import numpy as np
    from ham_dev_pilot import softmax

    start = time.perf_counter()
    images, text, scale, provenance = load_cache(directory, model, audit, manifest, prompts)
    image_index = {row["image_id"]: i for i, row in enumerate(manifest)}
    nodes = sorted(prompts["names"])

    def normalize(vectors):
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Cannot normalize zero-length class embeddings")
        return vectors / norms

    def probabilities(centers):
        scores = softmax(scale * (images @ centers.T))
        return {key: dict(zip(tree.leaves, scores[i])) for key, i in image_index.items()}

    frozen_text = probabilities(normalize(text.mean(axis=0)[[nodes.index(leaf) for leaf in tree.leaves]]))
    results = {}
    for seed, (support, populations) in cohorts.items():
        centers = np.stack([images[[image_index[row["image_id"]] for row in support[leaf]]].mean(axis=0)
                            for leaf in tree.leaves])
        results[seed] = {readout: {name: fixed_policy_results(tree, calibration, test_groups, test_units, scores, controller)
                                   for name, (calibration, _, test_groups, test_units) in populations.items()}
                         for readout, scores in (("frozen_text", frozen_text),
                                                 ("cosine_centroid", probabilities(normalize(centers))))}
    return {"provenance": provenance, "seeds": results, "evaluation_seconds": round(time.perf_counter() - start, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("audit", "splits", "exposure", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--inputs-only", action="store_true")
    for model in MODELS:
        parser.add_argument(f"--{model}-cache", type=Path)
    args = parser.parse_args()
    caches = {model: getattr(args, f"{model}_cache") for model in MODELS}
    if (args.inputs_only and any(caches.values())) or (not args.inputs_only and not all(caches.values())):
        parser.error("Choose --inputs-only or provide both full CUDA cache directories")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = {"python": platform.python_version()}
    try:
        recipe = json.loads(RECIPE.read_bytes())
        prompts = json.loads(PROMPTS.read_bytes())
        controller = json.loads(CONTROLLER.read_bytes())
        if (config_digest(PROMPTS) != recipe["prompt_config_sha256"] or
                config_digest(CONTROLLER) != recipe["controller_rehearsal_sha256"]):
            raise ValueError("Frozen taxonomy, prompts or policy grid differs")
        tree = Taxonomy(prompts["parent"])
        audit, manifest = read_inputs(args.audit)
        released = json.loads(RELEASE.read_bytes())
        if (audit["split_sha256"] != released["splits"]["17"]["sha256"] or
                len(manifest) != released["image_count"] or
                len({row["image_id"] for row in manifest}) != len(manifest)):
            raise ValueError("Expected the complete v1.5 seed-17 image audit")
        mask_path = args.exposure / "development_exposure.csv"
        cohorts = prepare_cohorts(args.splits, mask_path, {row["image_id"]: row for row in manifest}, recipe, tree)
        report.update(protocol=recipe["protocol"], recipe_sha256=config_digest(RECIPE), prompt_sha256=config_digest(PROMPTS),
                      controller_sha256=config_digest(CONTROLLER), split_sha256={s: v["sha256"] for s, v in released["splits"].items()},
                      mask_sha256=digest(mask_path), audit_images=len(manifest), cohorts=describe_cohorts(cohorts),
                      risk_scope="Conditional on fixed candidates and iid calibration representatives; HAM patient independence unknown")
        if not args.inputs_only:
            report["models"] = {model: evaluate(model, directory, args.audit, manifest, prompts, tree, cohorts, controller)
                                for model, directory in caches.items()}
        report["status"] = "inputs_validated" if args.inputs_only else "complete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
