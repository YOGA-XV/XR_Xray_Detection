from scripts import evaluate_per_class_speed as speed


def test_resolve_gflops_keeps_nonzero_model_info_value(monkeypatch):
    def fail_if_called(_model, _imgsz):
        raise AssertionError("profiler fallback should not be called")

    monkeypatch.setattr(speed, "get_flops_with_torch_profiler", fail_if_called)

    assert speed.resolve_gflops(model=object(), imgsz=640, info_flops=8.1252) == 8.1252


def test_resolve_gflops_uses_profiler_when_model_info_is_zero(monkeypatch):
    monkeypatch.setattr(speed, "get_flops_with_torch_profiler", lambda _model, imgsz: 28.458619 if imgsz == 640 else 0.0)

    assert speed.resolve_gflops(model=object(), imgsz=640, info_flops=0.0) == 28.45862
