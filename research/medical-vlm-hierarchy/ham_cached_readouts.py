"""Reconstruct verified frozen HAM leaf probabilities for the existing development cohorts."""
import json

import numpy as np

from ham_dev_pilot import digest, softmax
from ham_error_increment import read_csv
from selective_core import Taxonomy


def development_readouts(pilot_dir, balanced_dir, increment_dir, taxonomy_path):
    config=json.loads(taxonomy_path.read_bytes())
    tree=Taxonomy(config["parent"])
    paths=[pilot_dir,balanced_dir,increment_dir]
    infos=[json.loads((p/"summary.json").read_bytes()) for p in paths]
    if (infos[0]["config_sha256"]!=digest(taxonomy_path)
            or any(info["status"]!="complete" or info["selection_sha256"]!=digest(p/"selection.csv")
                   for info,p in zip(infos,paths))):
        raise ValueError("Source taxonomy or cohort changed")
    fit,support,fresh=[read_csv(p/"selection.csv") for p in paths]
    selected=fit+support+fresh
    if len(selected)!=len({r["split_group"] for r in selected}):
        raise ValueError("Development or support cohorts overlap")
    train=[i for i,r in enumerate(support) if r["role"]=="train"]
    dev=[i for i,r in enumerate(support) if r["role"]=="dev"]
    if len(train)+len(dev)!=len(support) or any(r["role"]!="dev" for r in fresh+fit):
        raise ValueError("Unexpected split role")
    support_labels=np.asarray([support[i]["label"] for i in train])
    truths={"fit":[r["label"] for r in fit],"natural":[r["label"] for r in fresh],
            "balanced":[support[i]["label"] for i in dev]}
    outputs={}
    for model,ref in infos[2]["models"].items():
        files=[p/f"{model}.npz" for p in paths]
        if [digest(p) for p in files]!=[ref["fit_cache"],ref["support_cache"],ref["cache_sha256"]]:
            raise ValueError("Source feature cache changed")
        meta=infos[0]["models"][model]
        with np.load(files[0],allow_pickle=False) as first, np.load(files[1],allow_pickle=False) as bal, np.load(files[2],allow_pickle=False) as new:
            text=first["text_features"].astype(np.float64).mean(0)
            text/=np.linalg.norm(text,axis=1,keepdims=True)
            frozen=text[[meta["text_node_order"].index(n) for n in tree.leaves]]
            train_images=bal["image"][train].astype(np.float64)
            centers=np.stack([train_images[support_labels==n].mean(0) for n in tree.leaves])
            centers/=np.linalg.norm(centers,axis=1,keepdims=True)
            images={"fit":first["image_features"],"natural":new["image_features"],"balanced":bal["image"][dev]}
            outputs[model]={}
            for method,proto,bal_key in (("frozen_text",frozen,"frozen_text"),("cosine_centroid",centers,"cosine_nearest_centroid")):
                scores={k:softmax((x.astype(np.float64)@proto.T)*meta["logit_scale"]) for k,x in images.items()}
                target=np.asarray([tree.leaves.index(y) for y in truths["natural"]])
                if not np.array_equal(scores["natural"].argmax(1)!=target,new[f"{method}_error"]):
                    raise ValueError("Natural cohort leaf errors differ from E02")
                if not np.array_equal(scores["balanced"].argmax(1),bal[bal_key].argmax(1)):
                    raise ValueError("Balanced cohort predictions differ from E01")
                if method=="frozen_text" and not np.array_equal(scores["fit"].argmax(1),first["leaf_probs"].argmax(1)):
                    raise ValueError("Initial cohort text predictions differ from E01")
                outputs[model][method]=scores
    return config,tree,truths,outputs
