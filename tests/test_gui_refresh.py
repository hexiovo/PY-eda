from __future__ import annotations

import copy
import json
from pathlib import Path


def test_bootstrap_shows_starting_state_before_runtime_imports(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QDialog, QMainWindow
    from eda_toolkit.ui import bootstrap

    observed: dict[str, bool] = {}
    original_show = QDialog.show

    def track_splash_show(dialog):
        original_show(dialog)
        if dialog.windowTitle().endswith("正在启动"):
            observed["splash_visible"] = dialog.isVisible()

    monkeypatch.setattr(QDialog, "show", track_splash_show)

    def configure_identity():
        observed["identity_after_application"] = QApplication.instance() is not None
        return {"configured": True}

    monkeypatch.setattr(bootstrap, "configure_app_identity", configure_identity)

    class FakeUi:
        def warm_runtime_imports(self, progress_callback):
            observed["splash_visible_before_imports"] = observed.get("splash_visible", False)
            progress_callback("正在准备测试界面…")

        def create_window(self, _config):
            return QMainWindow(), {"shutdown": lambda: None}

    assert bootstrap.main(["--smoke"], ui_module=FakeUi()) == 0
    assert observed["identity_after_application"] is True
    assert observed["splash_visible_before_imports"] is True


def test_workbench_default_layout_is_compact_and_logs_and_json_are_not_in_footer(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QApplication
    from eda_toolkit.ui.app import create_window

    # Current 1920x1080 desktop at 150% scale leaves this logical work area.
    available = QRect(0, 0, 1280, 672)
    class Screen:
        def availableGeometry(self):
            return available
    monkeypatch.setattr(QApplication, "primaryScreen", staticmethod(lambda: Screen()))
    window, state = create_window()
    try:
        assert window.width() <= available.width()
        assert window.height() <= available.height()
        assert window.minimumWidth() <= available.width()
        assert window.minimumHeight() <= available.height()
        window.show()
        state["app"].processEvents()
        assert available.contains(window.frameGeometry())
        window.resize(1120, 760)
        window.show()
        state["app"].processEvents()

        navigation = state["widgets"]["navigation"]
        assert navigation.count() == 10
        last_row = navigation.visualItemRect(navigation.item(navigation.count() - 1))
        assert last_row.bottom() <= navigation.viewport().rect().bottom()
        assert state["app"].style().objectName().lower() == "fusion"
        assert window.windowTitle() == "EDA Toolkit · 皮肤电研究工作台"

        widgets = state["widgets"]
        assert widgets["import_config"].text() == "导入设置"
        assert widgets["save_config"].text() == "导出设置"
        assert "JSON" in widgets["save_config"].toolTip()
        assert widgets["status_bar"].isVisible()
        assert widgets["cancel"].isHidden()
        assert widgets["add_task_button"].text().startswith("添加任务…")
        assert widgets["start_queue"].text() == "开始队列"
        assert "本次会话" in widgets["log_header"].text()
        assert widgets["log"].parentWidget() is widgets["session_log_group"]
        assert "json_group" not in widgets or widgets["json_group"].isHidden()
        assert "json_toggle" not in widgets or widgets["json_toggle"].isHidden()
        assert widgets["config_json"].isHidden()
        assert widgets["apply_json"].isHidden()
        assert {action.text() for action in window.menuBar().actions()} >= {"设置", "任务"}

        navigation.setCurrentRow(9)
        state["app"].processEvents()
        assert widgets["log"].isVisible()
    finally:
        window.close()
        state["shutdown"]()


def test_export_always_prompts_and_uses_current_applied_controls(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QFileDialog
    from eda_toolkit.ui.app import create_window

    targets = [tmp_path / "settings-a.json", tmp_path / "settings-b.json"]
    calls: list[tuple] = []

    def choose_target(*args, **kwargs):
        calls.append(args)
        return str(targets[len(calls) - 1]), "JSON (*.json)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(choose_target))
    window, state = create_window()
    try:
        widgets = state["widgets"]
        preset = widgets["preset"]
        assert preset.currentText() == "单人基础分析"
        assert preset.currentData() == state["config"]["pipeline_id"] == "single_standard"
        assert {preset.itemData(index) for index in range(preset.count())} == {
            "single_standard", "single_event", "single_advanced", "single_driver", "group_sync", "custom"
        }
        interpolation = widgets["preprocess.interpolation"]
        smoothing = widgets["preprocess.smoothing"]
        assert interpolation.currentText() == "线性插值"
        assert interpolation.currentData() == "linear"
        assert smoothing.currentText() == "不平滑"
        assert smoothing.currentData() == "none"
        original_workers = state["config"]["compute"]["workers"]
        gap_key = "preprocess.interpolate_max_gap_s"
        gap_control = widgets[gap_key]
        original_gap = float(state["config"]["preprocess"]["interpolate_max_gap_s"])
        applied_gap = original_gap + 0.25 if original_gap <= 3599.5 else original_gap - 0.25
        gap_control.setValue(applied_gap)
        assert state["config"]["preprocess"]["interpolate_max_gap_s"] == applied_gap

        draft = copy.deepcopy(state["config"])
        draft_workers = 2 if original_workers != 2 else 3
        draft["compute"]["workers"] = draft_workers
        widgets["config_json"].setPlainText(json.dumps(draft, ensure_ascii=False, indent=2))

        widgets["save_config"].click()
        widgets["save_config"].click()

        assert len(calls) == 2
        assert "导出设置为 JSON" in calls[0][1]
        assert all(target.is_file() for target in targets)
        for target in targets:
            exported = json.loads(target.read_text(encoding="utf-8"))
            assert exported["preprocess"]["interpolate_max_gap_s"] == applied_gap
            assert exported["compute"]["workers"] == original_workers
            assert exported["compute"]["workers"] != draft_workers
    finally:
        window.close()
        state["shutdown"]()
