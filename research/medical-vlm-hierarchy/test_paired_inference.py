import importlib.util
import statistics
import unittest

from ham_paired_inference import METRICS, mean_split_differences, paired_intervals
from selective_core import Taxonomy


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is used by the clustered bootstrap")
class PairedInferenceTest(unittest.TestCase):
    def setUp(self):
        self.tree = Taxonomy({"a": "root", "b": "branch", "c": "branch", "branch": "root"})
        self.config = {"replicates": 257, "seed": 17043, "interval_percentiles": [2.5, 97.5]}

    def test_unequal_groups_match_literal_paired_resampling(self):
        import numpy as np

        labels = ["b", "c", "a", "b", "b", "c"]
        ids = ["large", "large", "small", "large", "middle", "middle"]
        predictions = {"candidate": ["branch", "branch", "root", "b", "branch", "a"],
                       "reference": ["root", "b", "a", "root", "branch", "c"]}
        contrast = ("candidate", "reference")
        panel = paired_intervals(self.tree, labels, ids, predictions, [contrast], self.config)
        self.assertEqual((panel["split_groups"], panel["analysis_units"]), (3, 6))
        result = panel["comparisons"]["candidate_minus_reference"]
        self.assertEqual(result["different_nodes"], 5)

        groups = [[i for i, group in enumerate(ids) if group == name] for name in sorted(set(ids))]
        rng = np.random.Generator(np.random.PCG64(self.config["seed"]))

        def differences(indices):
            truth = [labels[i] for i in indices]
            candidate, reference = [self.tree.metrics(truth, [predictions[name][i] for i in indices]) for name in contrast]
            return [candidate[metric] - reference[metric] for metric in METRICS]

        sampled = [differences([i for j in picked for i in groups[j]])
                   for picked in rng.integers(len(groups), size=(self.config["replicates"], len(groups)))]
        point = differences(range(len(ids)))
        for index, metric in enumerate(METRICS):
            unweighted_groups = statistics.mean(differences(group)[index] for group in groups)
            self.assertNotAlmostEqual(point[index], unweighted_groups)
            quantiles = statistics.quantiles([row[index] for row in sampled], n=40, method="inclusive")
            self.assertAlmostEqual(result["metrics"][metric]["difference"], point[index])
            self.assertAlmostEqual(result["metrics"][metric]["interval"][0], quantiles[0])
            self.assertAlmostEqual(result["metrics"][metric]["interval"][1], quantiles[-1])

    def test_identical_rejects_and_cross_split_mean(self):
        data = ["a", "b"]
        select = {"candidate": ["root", "root"], "reference": ["root", "root"]}
        names = [("candidate", "reference")]
        zero = paired_intervals(self.tree, data, ["one", "two"], select, names, self.config)
        self.assertEqual(zero["comparisons"]["candidate_minus_reference"]["different_nodes"], 0)
        for stats in zero["comparisons"]["candidate_minus_reference"]["metrics"].values():
            self.assertEqual(stats, {"difference": 0., "interval": [0., 0.]})

        select["candidate"] = data
        perfect = paired_intervals(self.tree, data, ["one", "two"], select, names, self.config)
        mean = mean_split_differences([zero, perfect])
        self.assertEqual(mean, {"candidate_minus_reference": dict.fromkeys(METRICS, .5)})


if __name__ == "__main__":
    unittest.main()
