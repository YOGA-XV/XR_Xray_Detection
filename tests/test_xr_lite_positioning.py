import csv
import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_xr_lite_positioning import build_positioning


class XRLitePositioningTest(unittest.TestCase):
    def test_marks_xr_lite_as_xr_family_leader_without_overstating_overall_accuracy(self):
        workspace_tmp = Path("test_artifacts") / f"xr_lite_positioning_{uuid.uuid4().hex}"
        baseline_csv = workspace_tmp / "baseline.csv"
        speed_csv = workspace_tmp / "speed.csv"

        try:
            workspace_tmp.mkdir(parents=True)
            with baseline_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model",
                        "best_epoch",
                        "precision",
                        "recall",
                        "f1",
                        "map50",
                        "map75",
                        "map50_95",
                        "ap_small",
                        "recall_small",
                        "ap_thin",
                        "recall_thin",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "model": "yolov8n",
                            "best_epoch": "1",
                            "precision": "0.90",
                            "recall": "0.90",
                            "f1": "0.90",
                            "map50": "0.90",
                            "map75": "0.85",
                            "map50_95": "0.79",
                            "ap_small": "0.67",
                            "recall_small": "0.80",
                            "ap_thin": "0.84",
                            "recall_thin": "0.90",
                        },
                        {
                            "model": "yolov8s",
                            "best_epoch": "1",
                            "precision": "0.92",
                            "recall": "0.91",
                            "f1": "0.91",
                            "map50": "0.94",
                            "map75": "0.87",
                            "map50_95": "0.81",
                            "ap_small": "0.73",
                            "recall_small": "0.86",
                            "ap_thin": "0.87",
                            "recall_thin": "0.95",
                        },
                    ]
                )
            with speed_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model",
                        "params_m",
                        "gflops",
                        "fps_end_to_end",
                        "precision",
                        "recall",
                        "map50",
                        "map50_95",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "model": "A3_pcn_egi",
                            "params_m": "3.01",
                            "gflops": "8.23",
                            "fps_end_to_end": "480",
                            "precision": "0.94",
                            "recall": "0.88",
                            "map50": "0.93",
                            "map50_95": "0.792",
                        },
                        {
                            "model": "XR-Lite",
                            "params_m": "3.44",
                            "gflops": "8.65",
                            "fps_end_to_end": "450",
                            "precision": "0.95",
                            "recall": "0.87",
                            "map50": "0.92",
                            "map50_95": "0.785",
                        },
                        {
                            "model": "P2-Lite",
                            "params_m": "2.88",
                            "gflops": "11.50",
                            "fps_end_to_end": "400",
                            "precision": "0.94",
                            "recall": "0.86",
                            "map50": "0.92",
                            "map50_95": "0.776",
                        },
                    ]
                )

            report = build_positioning(baseline_csv=baseline_csv, speed_csv=speed_csv)

            self.assertEqual(report["claims"]["xr_family_map50_95_leader"], "XR-Lite")
            self.assertEqual(report["claims"]["xr_family_fps_leader"], "XR-Lite")
            self.assertEqual(report["claims"]["n_family_map50_95_leader"], "XR-Nano")
            self.assertEqual(report["claims"]["overall_map50_95_leader"], "yolov8s")
            self.assertIn("not the overall mAP50:95 leader", report["warnings"])
        finally:
            shutil.rmtree(workspace_tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
