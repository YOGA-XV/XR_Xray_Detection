from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel


class Stage5P2LiteTest(unittest.TestCase):
    def test_stage5_p2lite_model_builds_with_four_detection_scales(self):
        model = DetectionModel(ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-p2lite.yaml", nc=12, ch=4, verbose=False)

        out = model(torch.zeros(1, 4, 64, 64))

        self.assertIsInstance(out, dict)
        self.assertEqual(len(out["feats"]), 4)
        self.assertEqual([tuple(feat.shape[-2:]) for feat in out["feats"]], [(16, 16), (8, 8), (4, 4), (2, 2)])
        self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

    def test_stage5_plus_combines_lite_modules_with_p2lite_head(self):
        model = DetectionModel(ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-plus.yaml", nc=12, ch=4, verbose=False)

        out = model(torch.zeros(1, 4, 64, 64))
        layer_types = [module.__class__.__name__ for module in model.model]

        self.assertIn("SPPFAIFILite", layer_types)
        self.assertIn("C2fEMALite", layer_types)
        self.assertEqual(len(out["feats"]), 4)
        self.assertEqual([tuple(feat.shape[-2:]) for feat in out["feats"]], [(16, 16), (8, 8), (4, 4), (2, 2)])
        self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])


if __name__ == "__main__":
    unittest.main()
