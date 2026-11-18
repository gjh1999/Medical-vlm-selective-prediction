import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from audit_pad_capacity import read_calibration_groups


class PadCapacityTests(unittest.TestCase):
    def test_cli_counts_group_membership_across_seeds(self):
        script = Path(__file__).with_name("audit_pad_capacity.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (17, 29, 43):
                with (root / f"split_seed{seed}.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["split", "representative", "split_group", "label"])
                    writer.writerows([
                        ["calibration", 1, "shared", "BCC"],
                        ["calibration", 1, "shared", "BCC"],
                        ["calibration", 1, f"group{seed}", "BCC"],
                        ["calibration", 0, "ignored", "MEL"],
                        ["test", 1, "ignored", "MEL"],
                    ])
                    if seed != 43:
                        writer.writerow(["calibration", 1, "shared", "MEL"])
            out = root / "run"
            subprocess.run([sys.executable, str(script), "--split-dir", str(root),
                            "--output", str(out)], check=True)
            report = json.loads((out / "capacity.json").read_text())
            self.assertEqual(report["splits"]["17"]["representatives"], 4)
            self.assertEqual(report["splits"]["17"]["groups"], 2)
            self.assertEqual(report["splits"]["17"]["groups_containing_each_label"], {"BCC": 2, "MEL": 1})
            overlap = report["calibration_group_overlap"]
            self.assertEqual(overlap["overall"], {
                "distinct_across_seeds": 4, "common_to_all_seeds": 1,
                "pairwise_intersections": {"17/29": 1, "17/43": 1, "29/43": 1},
            })
            self.assertEqual(overlap["by_label"]["MEL"], {
                "distinct_across_seeds": 1, "common_to_all_seeds": 0,
                "pairwise_intersections": {"17/29": 1, "17/43": 0, "29/43": 0},
            })

    def test_representative_with_missing_group_cannot_count_as_a_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv"
            path.write_text("split,representative,split_group,label\ncalibration,1,,MEL\n")
            with self.assertRaisesRegex(ValueError, "Missing calibration group"):
                read_calibration_groups(path)
