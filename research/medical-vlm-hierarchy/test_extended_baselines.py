import importlib.util
import json
import unittest
from unittest.mock import patch

from evaluate_ham_frozen_baselines import CONTROLLER, evaluate
from ham_extended_baselines import RECIPE, actions, cutoff_grid, evaluate_lac, top_level_branches
from selective_core import Taxonomy, conformal_quantile


class ExtendedBaselineTest(unittest.TestCase):
    def setUp(self):
        self.tree = Taxonomy({"a": "root", "b": "branch", "c": "branch", "branch": "root"})

    def test_grid_keeps_tied_slots_and_uses_nearest_rank(self):
        self.assertEqual(cutoff_grid([.1, .2, .3, .4, .5], 2), {"slot_01": .2, "slot_02": .4})
        repeated = cutoff_grid([.1, .2], 16)
        self.assertEqual(len(repeated), 16)
        self.assertEqual(set(repeated.values()), {.1, .2})

    def test_flat_climbing_and_two_stage_use_the_same_leaf_gate(self):
        rows = [{"a": .4, "b": .35, "c": .25}, {"a": .5, "b": .3, "c": .2}]
        confidence = [.4, .5]
        self.assertEqual(actions(self.tree, "flat_max_probability", rows, confidence, .5), ["root", "a"])
        self.assertEqual(actions(self.tree, "climbing", rows, confidence, .5), ["root", "a"])
        self.assertEqual(actions(self.tree, "two_stage", rows, confidence, .5), ["branch", "a"])

    def test_lac_keeps_original_coverage_when_projection_adds_labels(self):
        rows = [{"a": .5, "b": .5, "c": 0.}, {"a": .34, "b": .33, "c": .33}]
        labels = ["c", "a"]
        branches = top_level_branches(self.tree)
        finite = evaluate_lac(self.tree, rows, labels, .5, branches)
        self.assertEqual(finite["set_contains_truth"], 0)
        self.assertEqual(finite["projection"]["node_n"], 0)
        self.assertEqual(finite["nonempty_sets_expanded"], 1)
        self.assertEqual(finite["empty_set_to_root"], 1)

        infinite = evaluate_lac(self.tree, rows, labels, conformal_quantile([.5], .05), branches)
        self.assertEqual(infinite["set_contains_truth"], 2)
        self.assertEqual(infinite["projection"]["node_n"], 0)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is used by the feature-cache evaluator")
    def test_extended_cache_path_preserves_initial_report_and_scores_both_readouts(self):
        import numpy as np

        prompts = {"names": {node: node for node in self.tree.parent}}
        nodes = sorted(prompts["names"])

        def rows(prefix, labels):
            return [{"image_id": f"{prefix}-{i}", "label": label} for i, label in enumerate(labels)]

        support = {leaf: rows(leaf, [leaf]) for leaf in self.tree.leaves}
        dev = rows("dev", ["a", "b", "c"] * 6)
        cal = rows("cal", ["a", "b", "c"] * 44)
        test = rows("test", ["a", "b"])
        training = [row for group in support.values() for row in group]
        manifest = training + dev + cal + test
        cohorts = {"synthetic": (support, {"all": (cal, cal, test[:1], test)}, dev)}
        basis = np.eye(len(nodes))
        images = basis[[nodes.index(row["label"]) for row in manifest]]
        controller = json.loads(CONTROLLER.read_bytes())
        extension = json.loads(RECIPE.read_bytes())
        with patch("evaluate_ham_frozen_baselines.load_cache", return_value=(images, np.stack([basis, basis]), 20., {})):
            base = evaluate("clip", None, None, manifest, prompts, self.tree, cohorts, controller)
            new = evaluate("clip", None, None, manifest, prompts, self.tree, cohorts, controller, extension)

        self.assertEqual(base["seeds"], new["seeds"])
        result = new["extended_baselines"]["synthetic"]
        for head in ("frozen_text", "cosine_centroid"):
            self.assertTrue(result[head]["temperature_at_bound"])
            self.assertEqual(result[head]["temperatures"]["dev_fitted"]["inverse_temperature"], 100.)
            for values in result[head]["temperatures"].values():
                matched = values["populations"]["all"]
                self.assertEqual(len(matched["families"]), 6)
                self.assertTrue(all(method["selected"] for method in matched["families"].values()))
                self.assertEqual(matched["lac"]["0.1"]["test"]["analysis_units"]["set_coverage"], 1.)
        self.assertEqual(result["prompt_template_comparison"]["all"]["analysis_units"]["template_1"]["accuracy"], 1.)


if __name__ == "__main__":
    unittest.main()
