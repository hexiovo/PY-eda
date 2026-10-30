"""UI acceptance checks for structured extended parameters and environment repair."""

from __future__ import annotations

import copy
import time


def _wait_for(app, condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(condition())


def test_extended_parameter_forms_round_trip_schema_fields(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from eda_toolkit.config import get_preset, resolve_config
    from eda_toolkit.ui.parameter_form import ParameterForm

    app = QApplication.instance() or QApplication([])
    extended = get_preset()["features"]["extended"]
    for family in ("trial_series", "nonspecific", "spectral", "autocorrelation", "complexity", "group_temporal"):
        form = ParameterForm(family, copy.deepcopy(extended[family]))
        assert form.value() == extended[family]
        assert all(not hasattr(widget, "setPlainText") for widget in form.field_widgets.values())

    trial = ParameterForm("trial_series", copy.deepcopy(extended["trial_series"]))
    series = trial.field_widgets["series_by"]
    series.set_value(["stimulus_id", "condition"])
    assert trial.value()["series_by"] == ["condition", "stimulus_id"]
    assert trial.value()["min_slope_trials"] == 5

    nonspecific = ParameterForm("nonspecific", copy.deepcopy(extended["nonspecific"]))
    coverage = nonspecific.field_widgets["annotation_coverage"]
    coverage.add_row({"recording_id": "rec_1", "channel": "eda", "start_s": 0.25,
                      "end_s": 12.5, "clock_domain": "local", "complete": True})
    assert nonspecific.value()["annotation_coverage"] == [{
        "recording_id": "rec_1", "channel": "eda", "start_s": 0.25,
        "end_s": 12.5, "clock_domain": "local", "complete": True,
    }]

    spectral = ParameterForm("spectral", copy.deepcopy(extended["spectral"]))
    bands = spectral.field_widgets["bands"]
    bands.add_row({"id": "slow_band", "low_hz": 0.01, "high_hz": 0.04})
    spectral.field_widgets["normalization_band_hz"].set_value([0.02, 0.4])
    candidate = get_preset()
    candidate["features"]["extended"]["spectral"] = spectral.value()
    assert [row["id"] for row in resolve_config(candidate)["features"]["extended"]["spectral"]["bands"]] == [
        "eda_045_025", "slow_band"]
    assert spectral.value()["normalization_band_hz"] == [0.02, 0.4]

    autocorrelation = ParameterForm("autocorrelation", copy.deepcopy(extended["autocorrelation"]))
    autocorrelation.field_widgets["lags_s"].add_value(20.0)
    assert autocorrelation.value()["lags_s"] == [1.0, 5.0, 10.0, 20.0]

    complexity = ParameterForm("complexity", copy.deepcopy(extended["complexity"]))
    complexity.field_widgets["sampen.m"].setValue(3)
    complexity.field_widgets["permen.normalize"].setChecked(False)
    complexity.field_widgets["dfa.min_scales"].setValue(5)
    nested = complexity.value()
    assert nested["sampen"]["m"] == 3
    assert nested["permen"]["normalize"] is False
    assert nested["dfa"]["min_scales"] == 5


def test_environment_dialog_installs_in_background_and_rechecks(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from eda_toolkit.ui.environment_dialog import EnvironmentDialog

    calls: list[tuple[str, object]] = []
    inspections = {"environment": 0, "gpu": 0}

    class FakeEnvironmentService:
        @staticmethod
        def inspect_environment():
            inspections["environment"] += 1
            missing = [] if inspections["environment"] > 1 else ["numpy"]
            return {"status": "ready" if not missing else "missing", "missing": missing,
                    "installed": ["pandas"]}

        @staticmethod
        def inspect_gpu_environment(refresh: bool = False):
            inspections["gpu"] += 1
            missing = [] if inspections["gpu"] > 1 else ["cupy"]
            return {"status": "ready" if not missing else "missing", "missing": missing,
                    "backend_enabled": not bool(missing)}

        @staticmethod
        def install_missing_dependencies(*, include_gpu, progress_callback, cancel_callback):
            calls.append(("install", include_gpu))
            progress_callback({"phase": "download", "message": "正在下载测试依赖", "progress": None})
            assert not cancel_callback()
            return {"status": "succeeded", "installed": ["numpy", "cupy" if include_gpu else ""],
                    "missing": [], "log_path": "install.log"}

    app = QApplication.instance() or QApplication([])
    changed: list[dict[str, object]] = []
    dialog = EnvironmentDialog(service=FakeEnvironmentService, on_status_changed=changed.append)
    dialog.show()
    assert _wait_for(app, lambda: dialog._operation is None and dialog._snapshot is not None)
    assert dialog.install_button.isEnabled()
    assert "本机能力" in dialog.details.toPlainText()
    dialog.gpu_checkbox.setChecked(True)
    dialog.install_button.click()
    assert _wait_for(app, lambda: dialog._operation == "inspect" and dialog._future is not None)
    assert _wait_for(app, lambda: dialog._operation is None and len(changed) >= 2)
    assert calls == [("install", True)]
    assert inspections["environment"] >= 2 and inspections["gpu"] >= 2
    assert "安装结果" in dialog.details.toPlainText()
    assert "完成" in dialog.details.toPlainText()
    assert not dialog.install_button.isEnabled()
    dialog.close()
    dialog.shutdown()
