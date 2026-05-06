from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import AIFILite, EMALite
from ultralytics.nn.tasks import DetectionModel


class Stage4XRModulesTest(unittest.TestCase):
    def test_ema_lite_preserves_feature_shape(self):
        module = EMALite(64, groups=8)
        x = torch.randn(2, 64, 20, 20)

        y = module(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_aifi_lite_preserves_feature_shape(self):
        module = AIFILite(64, reduction=4, heads=4, ffn_expansion=2)
        x = torch.randn(2, 64, 10, 10)

        y = module(x)

        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_stage4_ema_model_yaml_builds_and_forwards_with_edge_channel(self):
        model = DetectionModel(ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-ema.yaml", nc=12, ch=4, verbose=False)

        out = model(torch.zeros(1, 4, 64, 64))

        self.assertIsInstance(out, dict)
        self.assertEqual(len(out["feats"]), 3)

    def test_stage4_lite_model_yaml_builds_and_forwards_with_edge_channel(self):
        model = DetectionModel(ROOT / "ultralytics/cfg/models/v8/yolov8n-xr-lite.yaml", nc=12, ch=4, verbose=False)

        out = model(torch.zeros(1, 4, 64, 64))

        self.assertIsInstance(out, dict)
        self.assertEqual(len(out["feats"]), 3)


if __name__ == "__main__":
    unittest.main()
