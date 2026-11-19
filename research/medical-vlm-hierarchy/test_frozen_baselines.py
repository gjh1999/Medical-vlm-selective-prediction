import json
import tempfile
import unittest
from pathlib import Path

from evaluate_ham_frozen_baselines import CONTROLLER, config_digest, fixed_policy_results, group_representatives
from selective_core import Taxonomy


class FrozenBaselineTest(unittest.TestCase):
    def test_checkout_line_endings_keep_frozen_config_hash(self):
        with tempfile.TemporaryDirectory() as output:
            lf, crlf = (Path(output) / name for name in ("lf.json", "crlf.json"))
            lf.write_bytes(b'{\n  "test": true\n}\n')
            crlf.write_bytes(lf.read_bytes().replace(b"\n", b"\r\n"))
            self.assertEqual(config_digest(lf), config_digest(crlf))

    def test_test_labels_cannot_change_calibration_and_all_units_are_scored(self):
        def row(image, group, label):
            return {"image_id": image, "split_group": group, "label": label}

        tree = Taxonomy({"a": "root", "b": "branch", "c": "branch", "branch": "root"})
        controller = json.loads(CONTROLLER.read_bytes())
        calibration = [row(f"cal-{i}", f"g-{i}", "a") for i in range(130)]
        test = [row("test-2", "same-group", "c"), row("test-1", "same-group", "b")]
        representatives = group_representatives(test)
        self.assertEqual([r["image_id"] for r in representatives], ["test-1"])
        probabilities = {r["image_id"]: {"a": 1., "b": 0., "c": 0.} for r in calibration}
        probabilities.update({"test-1": {"a": 0., "b": 1., "c": 0.}, "test-2": {"a": 1., "b": 0., "c": 0.}})

        report = fixed_policy_results(tree, calibration, representatives, test, probabilities, controller)
        self.assertEqual(report["calibration"]["per_check_delta"], .05 / 32)
        self.assertEqual(len(report["calibration"]["audit"]), 16)
        self.assertTrue(all(report["selected"].values()))
        grouped = report["test"]["group_representatives"]["joint"]
        all_units = report["test"]["analysis_units"]["joint"]
        self.assertEqual((grouped["n"], grouped["leaf_errors"]), (1, 0))
        self.assertEqual((all_units["n"], all_units["leaf_errors"], all_units["cross_branch_errors"]), (2, 1, 1))

        changed = [dict(r, label="a") for r in test]
        other = fixed_policy_results(tree, calibration, group_representatives(changed), changed, probabilities, controller)
        self.assertEqual(other["calibration"], report["calibration"])
        self.assertEqual(other["selected"], report["selected"])

        smaller = fixed_policy_results(tree, calibration[:100], representatives, test, probabilities, controller)
        self.assertIsNone(smaller["selected"]["joint"])
        self.assertEqual(smaller["test"]["analysis_units"]["joint"]["node_n"], 0)
        self.assertIsNone(smaller["test"]["analysis_units"]["joint"]["node_risk"])


if __name__ == "__main__":
    unittest.main()
