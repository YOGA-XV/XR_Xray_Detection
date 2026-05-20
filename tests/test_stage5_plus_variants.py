from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel


def build_model(name: str) -> DetectionModel:
    return DetectionModel(ROOT / f"ultralytics/cfg/models/v8/{name}", nc=12, ch=4, verbose=False)


def module_names(model: DetectionModel) -> list[str]:
    return [module.__class__.__name__ for module in model.model]


class Stage5PlusVariantsTest(unittest.TestCase):
    def assert_four_scale_detector(self, model: DetectionModel) -> None:
        out = model(torch.zeros(1, 4, 64, 64))
        self.assertEqual(len(out["feats"]), 4)
        self.assertEqual([tuple(feat.shape[-2:]) for feat in out["feats"]], [(16, 16), (8, 8), (4, 4), (2, 2)])
        self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

    def test_v2_strengthens_p2_branch_with_c2f_fusion(self):
        base = build_model("yolov8n-xr-plus.yaml")
        model = build_model("yolov8n-xr-plus-v2.yaml")

        names = module_names(model)
        self.assertGreater(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in base.parameters()))
        self.assertLessEqual(sum(p.numel() for p in model.parameters()), 4_000_000)
        self.assertIn("C2fEMALite", names)
        self.assertIn("SPPFAIFILite", names)
        self.assertGreaterEqual(names.count("C2f"), module_names(base).count("C2f") + 1)
        self.assert_four_scale_detector(model)

    def test_v3_adds_ema_to_p3_branch(self):
        base = build_model("yolov8n-xr-plus.yaml")
        model = build_model("yolov8n-xr-plus-v3.yaml")

        names = module_names(model)
        self.assertGreater(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in base.parameters()))
        self.assertLessEqual(sum(p.numel() for p in model.parameters()), 4_000_000)
        self.assertGreaterEqual(names.count("C2fEMALite"), module_names(base).count("C2fEMALite") + 1)
        self.assertIn("SPPFAIFILite", names)
        self.assert_four_scale_detector(model)

    def test_v4_combines_stronger_p2_and_p3_ema(self):
        base = build_model("yolov8n-xr-plus.yaml")
        model = build_model("yolov8n-xr-plus-v4.yaml")

        names = module_names(model)
        self.assertGreater(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in base.parameters()))
        self.assertLessEqual(sum(p.numel() for p in model.parameters()), 4_000_000)
        self.assertGreaterEqual(names.count("C2fEMALite"), module_names(base).count("C2fEMALite") + 2)
        self.assertIn("SPPFAIFILite", names)
        self.assert_four_scale_detector(model)


if __name__ == "__main__":
    unittest.main()
