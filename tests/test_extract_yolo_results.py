from pathlib import Path
import shutil
import uuid

from scripts.extract_yolo_results import load_best_result, load_last_result, numeric_subset


def test_extracts_last_and_best_rows_from_results_csv():
    workspace_tmp = Path("test_artifacts") / f"results_{uuid.uuid4().hex}"
    results = workspace_tmp / "results.csv"
    results.parent.mkdir(parents=True, exist_ok=True)
    try:
        results.write_text(
            "\n".join(
                [
                    "epoch,metrics/precision(B),metrics/recall(B),metrics/F1(B),metrics/mAP50(B),metrics/mAP75(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss",
                    "0,0.1,0.2,0.13,0.3,0.25,0.20,1.0,2.0,3.0",
                    "1,0.4,0.5,0.44,0.6,0.55,0.50,0.9,1.9,2.9",
                    "2,0.3,0.4,0.34,0.5,0.45,0.40,0.8,1.8,2.8",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        last = numeric_subset(load_last_result(results))
        best = numeric_subset(load_best_result(results, "metrics/mAP50-95(B)"))

        assert last["epoch"] == 2
        assert last["metrics/mAP50-95(B)"] == 0.40
        assert best["epoch"] == 1
        assert best["metrics/mAP50-95(B)"] == 0.50
    finally:
        shutil.rmtree(workspace_tmp, ignore_errors=True)
