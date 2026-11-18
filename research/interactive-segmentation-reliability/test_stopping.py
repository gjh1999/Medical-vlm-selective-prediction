import copy
import math
import unittest

from stopping import Policy, calibrate, evaluate, stop_step, upper_bound


def trajectory(group="a"):
    return {"group_id": group, "observations": [{"confidence": x} for x in (0.9, 0.5, 0.9)],
            "dice": [0.5, 0.95, 0.95], "submitted_clicks": [1, 0, 1],
            "base_calls": [1, 0, 1], "probe_calls": [2, 0, 2]}


class StoppingTests(unittest.TestCase):
    def test_nonzero_failure_exact_bound(self):
        # For n=2,k=1 the CDF is 1-p^2; independent closed form.
        self.assertAlmostEqual(upper_bound(1, 2, 0.05), math.sqrt(0.95), places=13)

    def test_zero_all_and_invalid(self):
        self.assertIsNone(upper_bound(0, 0, 0.05))
        self.assertEqual(upper_bound(2, 2, 0.05), 1)
        self.assertAlmostEqual(upper_bound(0, 59, 0.05), 1 - 0.05 ** (1 / 59))
        for args in ((1, 0, .05), (0.5, 2, .05), (0, 2, 0)):
            with self.assertRaises(ValueError):
                upper_bound(*args)

    def test_first_crossing_not_final_quality(self):
        metrics = evaluate([trajectory()], Policy("p", ("confidence",), .8), budget=3)
        self.assertEqual((metrics["accepted"], metrics["failures"]), (1, 1))
        self.assertEqual(metrics["mean_model_calls"], 1)

    def test_quality_does_not_change_decision(self):
        first = trajectory()
        second = copy.deepcopy(first)
        second["dice"] = [1, 1, 1]
        policy = Policy("p", ("confidence",), .8)
        self.assertEqual(stop_step(first["observations"], policy), stop_step(second["observations"], policy))

    def test_patience_requires_consecutive_passes(self):
        self.assertIsNone(stop_step(trajectory()["observations"], Policy("p", ("confidence",), .8, 2)))

    def test_missing_stability_cannot_stop(self):
        self.assertIsNone(stop_step([{"stability": None}], Policy("p", ("stability",), .8)))

    def test_referral_and_omitted_click_cost(self):
        metrics = evaluate([trajectory()], None, budget=3)
        self.assertIsNone(metrics["selective_risk"])
        self.assertEqual(metrics["mean_interaction_attempts"], 3)
        self.assertEqual(metrics["mean_submitted_clicks"], 2)
        self.assertEqual(metrics["coverage"], 0)

    def test_probe_cost_is_counted(self):
        row = trajectory()
        for obs in row["observations"]:
            obs["probe_consistency"] = 1.0
        metrics = evaluate([row], Policy("p", ("probe_consistency",), .8), budget=3)
        self.assertEqual(metrics["mean_model_calls"], 3)

    def test_duplicates_and_quality_leakage_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([trajectory(), trajectory()], None, budget=3)
        row = trajectory()
        row["observations"][0]["dice"] = 1
        with self.assertRaises(ValueError):
            evaluate([row], None, budget=3)

    def test_family_correction_and_empty_fallback(self):
        rows = [trajectory(str(i)) for i in range(59)]
        for row in rows:
            row["dice"] = [1, 1, 1]
        p = Policy("p", ("confidence",), .8)
        self.assertEqual(calibrate(rows, [p], budget=3)["selected"], "p")
        policies = [Policy(str(i), ("confidence",), .8) for i in range(20)]
        result = calibrate(rows, policies, budget=3)
        self.assertIsNone(result["selected"])
        self.assertEqual(result["per_policy_delta"], .0025)

    def test_nan_and_truncated_trajectory_rejected(self):
        row = trajectory()
        row["observations"][0]["confidence"] = float("nan")
        with self.assertRaises(ValueError):
            evaluate([row], None, budget=3)
        with self.assertRaises(ValueError):
            evaluate([trajectory()], None, budget=5)


if __name__ == "__main__":
    unittest.main()
