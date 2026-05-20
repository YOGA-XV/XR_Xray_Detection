import csv
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_stage5_xr_variants import VariantSpec, compare_variants


class Stage5VariantComparisonTest(unittest.TestCase):
    def test_compares_variant_metrics_and_writes_csv_and_json(self):
        workspace_tmp = Path("test_artifacts") / f"stage5_compare_{uuid.uuid4().hex}"
        output_csv = workspace_tmp / "summary.csv"
        output_json = workspace_tmp / "summary.json"
        try:
            specs = []
            for name, map_value, ap_small, ap_thin, params in [
                ("XR-Lite", 0.78, 0.68, 0.84, 3_400_000),
                ("XR-Plus", 0.79, 0.70, 0.85, 3_300_000),
            ]:
                root = workspace_tmp / name
                root.mkdir(parents=True, exist_ok=True)
                metrics = root / "metrics.json"
                small = root / "small.json"
                thin = root / "thin.json"
                metrics.write_text(
                    json.dumps(
                        {
                            "best": {
                                "epoch": 10,
                                "metrics/precision(B)": 0.91,
                                "metrics/recall(B)": 0.82,
                                "metrics/F1(B)": 0.86,
                                "metrics/mAP50(B)": 0.90,
                                "metrics/mAP75(B)": 0.81,
                                "metrics/mAP50-95(B)": map_value,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                small.write_text(json.dumps({"ap": ap_small, "recall": 0.88}), encoding="utf-8")
                thin.write_text(json.dumps({"ap": ap_thin, "recall": 0.92}), encoding="utf-8")
                specs.append(
                    VariantSpec(
                        name=name,
                        yaml_path=None,
                        final_metrics=metrics,
                        ap_small=small,
                        ap_thin=thin,
                        params=params,
                    )
                )

            rows = compare_variants(
                specs=specs,
                output_csv=output_csv,
                output_json=output_json,
                reference_name="XR-Lite",
                allow_missing=False,
            )

            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(rows[1]["delta_map50_95_vs_ref"], 0.01)
            self.assertAlmostEqual(rows[1]["delta_ap_small_vs_ref"], 0.02)
            self.assertTrue(output_csv.exists())
            self.assertTrue(output_json.exists())
            with output_csv.open(encoding="utf-8", newline="") as f:
                csv_rows = list(csv.DictReader(f))
            self.assertEqual(csv_rows[1]["variant"], "XR-Plus")
        finally:
            shutil.rmtree(workspace_tmp, ignore_errors=True)

    def test_allow_missing_keeps_row_with_empty_metrics(self):
        workspace_tmp = Path("test_artifacts") / f"stage5_missing_{uuid.uuid4().hex}"
        output_csv = workspace_tmp / "summary.csv"
        output_json = workspace_tmp / "summary.json"
        try:
            rows = compare_variants(
                specs=[
                    VariantSpec(
                        name="XR-Plus-v2",
                        yaml_path=None,
                        final_metrics=workspace_tmp / "missing_metrics.json",
                        ap_small=workspace_tmp / "missing_small.json",
                        ap_thin=workspace_tmp / "missing_thin.json",
                        params=3_600_000,
                    )
                ],
                output_csv=output_csv,
                output_json=output_json,
                reference_name=None,
                allow_missing=True,
            )

            self.assertEqual(rows[0]["variant"], "XR-Plus-v2")
            self.assertEqual(rows[0]["params"], 3_600_000)
            self.assertEqual(rows[0]["map50_95"], "")
        finally:
            shutil.rmtree(workspace_tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
