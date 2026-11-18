import unittest

from ham_threshold_envelope import threshold_envelope
from selective_core import Taxonomy


class EnvelopeTest(unittest.TestCase):
    def test_tied_scores_singletons_and_exact_one(self):
        tree = Taxonomy({"a":"g", "b":"g", "c":"root", "g":"root"})
        cfg = {"baselines":["flat", "climbing"], "thresholds":[.5, .8], "family_failure_probability":.05,
               "leaf_target":.1, "nonroot_target":.05}
        rows = [[.5,.5,0.], [.25,.25,.5], [0.,0.,1.]]
        result = threshold_envelope(tree, rows, ["b","c","c"], cfg)["families"]
        self.assertEqual(result["flat"]["states"], 2)
        self.assertEqual(result["climbing"]["states"], 2)
        self.assertEqual(result["flat"]["max_zero_error_leaf_outputs"], 1)
        self.assertEqual(result["climbing"]["max_zero_error_leaf_outputs"], 1)
        self.assertEqual(result["climbing"]["scalar_agreement"], 2)
