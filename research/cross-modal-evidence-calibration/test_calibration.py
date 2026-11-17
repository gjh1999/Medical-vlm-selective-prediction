import unittest
from calibration import NormalConformal, Normalizer, auroc, average_precision, evaluate


class CalibrationTests(unittest.TestCase):
    def test_conservative_ties_and_small_sample(self):
        self.assertEqual(NormalConformal([1, 1, 1]).pvalue(1), 1)
        self.assertEqual(NormalConformal([1] * 9).pvalue(2), 0.1)
        self.assertEqual(evaluate([2], [3], [1] * 9, .05)['sensitivity'], 0)

    def test_exact_rank_null_bound(self):
        # Enumerate exchangeable held-out ranks, including all edge ranks.
        pvalues = [NormalConformal([x for x in range(20) if x != held]).pvalue(held)
                   for held in range(20)]
        self.assertEqual(sum(p <= .05 for p in pvalues), 1)

    def test_ranking_and_ties(self):
        self.assertEqual(auroc([0, 1], [2, 3]), 1)
        self.assertEqual(auroc([2, 3], [0, 1]), 0)
        self.assertEqual(auroc([1, 1], [1, 1]), .5)
        self.assertEqual(average_precision([1, 1], [1, 1]), .5)
        self.assertEqual(average_precision([0, 1], [2, 3]), 1)

    def test_invalid_inputs(self):
        for scores in [[], [float('nan')], [float('inf')]]:
            with self.assertRaises(ValueError):
                NormalConformal(scores)
        with self.assertRaises(ValueError):
            Normalizer([(1, 2), (1, 2, 3)])
        with self.assertRaises(ValueError):
            evaluate([1], [2], [0], 0)

    def test_positive_affine_channel_invariance(self):
        base = [(0, 1), (2, 3), (4, 5)]
        shifted = [(3*x+7, 2*y-4) for x, y in base]
        self.assertAlmostEqual(Normalizer(base).score((3, 4)),
                               Normalizer(shifted).score((16, 4)))


if __name__ == '__main__':
    unittest.main()
