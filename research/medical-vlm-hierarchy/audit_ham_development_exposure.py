"""Replay the logged HAM supervised cohorts and audit their roles across all released splits."""
import argparse
import csv
import hashlib
import itertools
import json
import time
from collections import Counter
from pathlib import Path


CONFIG = Path(__file__).resolve().parent / "configs"
SEEDS = (17, 29, 43)
ROLES = ("train", "dev", "calibration", "test")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def config(name):
    return json.loads((CONFIG / f"ham-{name}-v1.json").read_bytes())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_split(directory, seed, expected):
    path = directory / f"split_seed{seed}.csv"
    if digest(path) != expected:
        raise ValueError(f"Split fingerprint differs for seed {seed}")
    rows = read_csv(path)
    by_image = {r["image_id"]: r for r in rows}
    if len(by_image) != len(rows):
        raise ValueError("Duplicate image in split inventory")
    return by_image


def replay(v13, v14, v15):
    pilot, balanced, first, second, crossed = [config(name) for name in
        ("dev-pilot", "balanced-probe", "error-increment", "g1-confirmation", "g1-crossed-fit")]
    history = {}
    leaves = sorted({r["label"] for r in v13.values()})

    def sample(name, inventory, role, salt, count, per_label=False):
        used = {inventory[k]["split_group"] for k in history}
        representatives = {}
        for key in sorted(inventory):
            row = inventory[key]
            group = row["split_group"]
            if row["split"] == role and row["representative"] == "1" and group not in used:
                representatives.setdefault(group, row)
        ranked = sorted(representatives, key=lambda g: hashlib.sha256(f"{salt}|{g}".encode()).digest())
        for label in leaves if per_label else [None]:
            eligible = [representatives[g] for g in ranked if label is None or representatives[g]["label"] == label]
            if len(eligible) < count:
                raise ValueError(f"Insufficient eligible groups for {name}")
            for r in eligible[:count]:
                history[r["image_id"]] = (name, role, r["label"])

    sample("original_fit", v13, "dev", pilot["selection_salt"], pilot["groups"])
    for role, name in (("train", "original_support"), ("dev", "balanced_evaluation")):
        sample(name, v13, role, balanced["salt"] + "|" + role, balanced[f"{role}_per_class"], True)
    sample("first_natural", v14, "dev", first["salt"], first["eval_size"])
    sample("second_natural", v15, "dev", second["salt"], second["n"])
    sample("new_support", v15, "train", crossed["support_salt"], crossed["new_support_per_leaf"], True)
    sample("new_fit", v15, "dev", crossed["fit_salt"], crossed["new_risk_fit"])
    return history


def summarize(role_rows, selected):
    reps = [r for r in role_rows if r["split_group"] in selected and r["representative"] == "1"]
    return {"groups": len({r["split_group"] for r in reps}), "analysis_units": len(reps),
            "representative_labels": dict(sorted(Counter(r["label"] for r in reps).items()))}


def run(v14, v15, output):
    for path in (v14, v15):
        if json.loads((path / "run.json").read_bytes())["status"] != "complete":
            raise ValueError("Both protocol rebuilds must be complete")
    pilot = config("dev-pilot")
    if pilot["split_sha256"] != config("balanced-probe")["split_sha256"]:
        raise ValueError("Original supervised recipes require different seed-17 splits")
    old = read_split(v14 / "v1.3-splits", 17, pilot["split_sha256"])
    middle = read_split(v14 / "splits", 17, config("error-increment")["split_sha256"])
    mask_info = json.loads((v15 / "sensitivity/summary.json").read_bytes())
    release = json.loads((v15 / "splits/audit.json").read_bytes())
    split_hashes = {str(s): release["splits"][str(s)]["sha256"] for s in SEEDS}
    if (release["split_protocol"] != "v1.5" or split_hashes != mask_info["split_sha256"]
            or split_hashes["17"] != config("g1-crossed-fit")["split_sha256"]):
        raise ValueError("Current split provenance differs")
    splits = {s: read_split(v15 / "splits", s, split_hashes[str(s)]) for s in SEEDS}
    base = splits[17]
    history = replay(old, middle, base)
    frozen = config("development-exposure")
    encoded = "".join(sorted(f"{cohort},{k},{role},{label}\n" for k, (cohort, role, label) in history.items()))
    fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
    if len(history) != frozen["n_images"] or fingerprint != frozen["membership_sha256"]:
        raise ValueError("Historical replay differs from the fingerprint of the logged selections")
    cohorts = {}
    for k, (cohort, role, label) in history.items():
        r = base[k]
        if (r["split"], r["label"], r["representative"]) != (role, label, "1") or r["split_group"] in cohorts:
            raise ValueError("Changes in the mapping of documented supervised groups")
        cohorts[r["split_group"]] = cohort
    supervised = set(cohorts)
    smoke_images = sorted(k for k, r in base.items() if r["split"] == "train" and r["representative"] == "1")[:8]
    smoke = {base[k]["split_group"] for k in smoke_images}
    mask = v15 / "sensitivity/sensitivity_groups.csv"
    if digest(mask) != mask_info["sensitivity_groups_sha256"]:
        raise ValueError("Image-overlap sensitivity mask provenance differs")
    flags = read_csv(mask)
    groups = {r["split_group"] for r in base.values()}
    if len(flags) != len(groups) or {r["split_group"] for r in flags} != groups or any(
            r["excluded"] not in ("0", "1") for r in flags):
        raise ValueError("Undefined or duplicated group in sensitivity inventory")
    texture = {r["split_group"] for r in flags if r["excluded"] == "1"}
    role_maps = {}
    public = {}
    tests = {}
    held_out = {}
    for seed, inventory in splits.items():
        roles = {}
        if inventory.keys() != base.keys():
            raise ValueError("Image inventories differ")
        for k, r in inventory.items():
            group = r["split_group"]
            if any(r[f] != base[k][f] for f in ("split_group", "analysis_group", "representative", "label")):
                raise ValueError("Group mapping differs across seeds")
            if roles.setdefault(group, r["split"]) != r["split"]:
                raise ValueError("Group assigned to multiple roles")
        role_maps[seed] = roles
        if set(roles.values()) != set(ROLES):
            raise ValueError("Unexpected split roles")
        tests[seed] = {g for g in groups if roles[g] == "test"}
        held_out[seed] = {g for g in groups if roles[g] in ("calibration", "test")}
        public[seed] = {}
        for role in ROLES:
            rows = [r for r in inventory.values() if r["split"] == role]
            members = {r["split_group"] for r in rows}
            used = members & supervised
            public[seed][role] = {
                "total": summarize(rows, members), "supervised": summarize(rows, used),
                "by_supervised_cohort": dict(sorted(Counter(cohorts[g] for g in used).items())),
                "smoke_groups": len(members & smoke), "smoke_or_supervised_groups": len(members & (smoke | used)),
                "without_supervised": summarize(rows, members - supervised),
                "without_supervised_or_image_overlap": summarize(rows, members - supervised - texture)}
    if held_out[17] & (supervised | smoke):
        raise ValueError("Logged development exposure in seed-17 calibration or test")
    path = output / "development_exposure.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["split_group", "supervised", "cohort", "technical_smoke", "image_overlap", *[f"seed{s}" for s in SEEDS]]
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        for g in sorted(groups):
            writer.writerow([g, int(g in supervised), cohorts.get(g, ""), int(g in smoke), int(g in texture),
                             *[role_maps[s][g] for s in SEEDS]])
    combinations = {f"{s1}_{s2}": len(tests[s1] & tests[s2]) for s1, s2 in itertools.combinations(SEEDS, 2)}
    return {"status": "complete", "protocol": "v1.5", "unit": "current HAM split_group",
            "groups": len(groups), "analysis_units": release["analysis_units"], "split_sha256": split_hashes,
            "membership_sha256": fingerprint, "mask_sha256": digest(path),
            "image_sensitivity_sha256": digest(mask), "supervised_groups": len(supervised),
            "supervised_cohort_counts": dict(sorted(Counter(cohorts.values()).items())), "technical_smoke_groups": len(smoke),
            "technical_smoke_and_supervised": len(smoke & supervised), "roles": public,
            "cross_seed_overlap": {"pairwise_test_groups": combinations, "test_union": len(set.union(*tests.values())),
                                   "test_all_seeds": len(set.intersection(*tests.values())),
                                   "calibration_or_test_all_seeds": len(set.intersection(*held_out.values()))},
            "scope": "Documented supervised pilot participation. Technical smoke listed separately; patient independence unknown."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("v14", "v15", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = {}
    try:
        result = run(args.v14, args.v15, args.output)
    except Exception as exc:
        result.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        result["seconds"] = round(time.perf_counter() - start, 3)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
