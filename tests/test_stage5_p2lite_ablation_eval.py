import csv
import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_stage5_p2lite_ablation import evaluate_existing_outputs, load_reference_row


class Stage5P2LiteAblationEvalTest(unittest.TestCase):
    def test_evaluates_p2lite_ablation_outputs_against_a3_reference(self):
        workspace_tmp = Path("test_artifacts") / f"stage5_p2lite_{uuid.uuid4().hex}"
        results_csv = workspace_tmp / "train" / "results.csv"
        labels = workspace_tmp / "labels"
        predictions = workspace_tmp / "predictions"
        baseline_summary = workspace_tmp / "stage2_summary.csv"
        output_dir = workspace_tmp / "out"
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        predictions.mkdir(parents=True, exist_ok=True)

        try:
            results_csv.write_text(
                "\n".join(
                    [
                        "epoch,metrics/precision(B),metrics/recall(B),metrics/F1(B),metrics/mAP50(B),metrics/mAP75(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss",
                        "0,0.1,0.2,0.13,0.3,0.25,0.20,1.0,2.0,3.0",
                        "1,0.4,0.5,0.44,0.6,0.55,0.50,0.9,1.9,2.9",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            baseline_summary.write_text(
                "\n".join(
                    [
                        "experiment,best_epoch,precision,recall,f1,map50,map75,map50_95,ap_small,recall_small,ap_thin,recall_thin",
                        "A3_pcn_egi,195,0.93,0.88,0.90,0.93,0.85,0.49,0.50,0.80,0.60,0.90",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            labels.joinpath("img1.txt").write_text("0 0.5 0.5 0.08 0.08\n1 0.5 0.5 0.40 0.10\n", encoding="utf-8")
            predictions.joinpath("img1.txt").write_text(
                "0 0.5 0.5 0.08 0.08 0.90\n1 0.5 0.5 0.40 0.10 0.80\n",
                encoding="utf-8",
            )

            summary = evaluate_existing_outputs(
                results_csv=results_csv,
                labels_dir=labels,
                predictions_dir=predictions,
                output_dir=output_dir,
                baseline_summary=baseline_summary,
                experiment="S5_p2lite",
            )

            self.assertEqual(summary["experiment"], "S5_p2lite")
            self.assertEqual(summary["best_epoch"], 1)
            self.assertAlmostEqual(summary["map50_95"], 0.50)
            self.assertAlmostEqual(summary["ap_small"], 1.0)
            self.assertAlmostEqual(summary["ap_thin"], 1.0)
            self.assertAlmostEqual(summary["delta_map50_95_vs_A3"], 0.01)
            self.assertAlmostEqual(summary["delta_ap_small_vs_A3"], 0.5)
            self.assertTrue(output_dir.joinpath("yolov8n_xr_p2lite_final_metrics.json").exists())
            self.assertTrue(output_dir.joinpath("yolov8n_xr_p2lite_ap_small.json").exists())
            self.assertTrue(output_dir.joinpath("yolov8n_xr_p2lite_ap_thin.json").exists())
            self.assertTrue(output_dir.joinpath("yolov8n_xr_p2lite_analysis.json").exists())

            with output_dir.joinpath("stage5_p2lite_ablation_summary.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["experiment"], "S5_p2lite")
        finally:
            shutil.rmtree(workspace_tmp, ignore_errors=True)

    def test_load_reference_row_fails_when_reference_experiment_is_missing(self):
        workspace_tmp = Path("test_artifacts") / f"stage5_ref_{uuid.uuid4().hex}"
        summary_csv = workspace_tmp / "summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            summary_csv.write_text("experiment,map50_95\nother,0.1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "A3_pcn_egi"):
                load_reference_row(summary_csv, "A3_pcn_egi")
        finally:
            shutil.rmtree(workspace_tmp, ignore_errors=True)

    def test_load_reference_row_fails_with_clear_error_for_wrong_summary_schema(self):
        workspace_tmp = Path("test_artifacts") / f"stage5_ref_schema_{uuid.uuid4().hex}"
        summary_csv = workspace_tmp / "summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        try:
            summary_csv.write_text("model,map50_95\nyolov8n,0.1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "experiment.*model"):
                load_reference_row(summary_csv, "A3_pcn_egi")
        finally:
            shutil.rmtree(workspace_tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
