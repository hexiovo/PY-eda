"""真实 PySide6 工作台与 pipeline 后台通路 smoke。"""

from __future__ import annotations

import json
import hashlib
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


def test_gui_config_path_resolution_and_duplicate_json_rejection(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QMessageBox
    from eda_toolkit.config import get_preset
    from eda_toolkit.core_io import load_recordings
    from eda_toolkit.ui.app import create_window

    project = tmp_path / "relative-project"
    project.mkdir()
    pd.DataFrame({"time_s": [0.0, 0.25, 0.5, 0.75], "eda_uS": [1.0, 1.1, 1.2, 1.3]}).to_csv(
        project / "signals.csv", index=False)
    config = get_preset()
    config["input"].update(signals=["signals.csv"], manifest=None, base_directory=None)
    config["mapping"].update(columns={"time_s": "time_s", "eda_uS": "eda_uS"},
                              sampling_rate_hz=4.0, eda_unit="uS")
    config_path = project / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    window, state = create_window(str(config_path))
    try:
        assert Path(state["config"]["input"]["base_directory"]) == project.resolve()
        assert state["capabilities"]["output_directory"]["writable"] is True
        assert state["widgets"]["environment_check_button"].text() == "环境监测"
        assert state["widgets"]["capabilities_label"].isHidden()
        assert state["widgets"]["compute_backend"].currentData() == "cpu"
        assert state["widgets"]["compute_workers"].value() == 1
        records, _tables, _report = load_recordings(options=state["config"])
        assert len(records) == 1
        assert records[0]["time_s"].tolist() == [0.0, 0.25, 0.5, 0.75]
        assert Path(records[0]["metadata"]["source_path"]) == (project / "signals.csv").resolve()
    finally:
        window.close()
        state["shutdown"]()

    window, state = create_window()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args: warnings.append(str(args[-1]))))
    try:
        text = state["widgets"]["config_json"].toPlainText()
        assert '"workers": 1' in text
        state["widgets"]["config_json"].setPlainText(text.replace('"workers": 1', '"workers": 1,\n    "workers": 2', 1))
        state["widgets"]["apply_json"].click()
        assert state["config"]["compute"]["workers"] == 1
        assert warnings and "duplicate_json_key" in warnings[-1]
    finally:
        window.close()
        state["shutdown"]()

def test_workbench_config_templates_and_real_background_pipeline(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QMessageBox
    warning_messages: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args: warning_messages.append(str(args[-1]))))
    uncaught_qt_errors: list[str] = []
    monkeypatch.setattr(sys, "excepthook", lambda exc_type, value, _traceback:
                        uncaught_qt_errors.append(f"{exc_type.__name__}: {value}"))
    monkeypatch.setattr(threading, "excepthook", lambda args:
                        uncaught_qt_errors.append(f"{args.exc_type.__name__}: {args.exc_value}"))
    from eda_toolkit.demo import create_demo
    from eda_toolkit.ui.app import create_window

    demo = create_demo(tmp_path / "demo-empty", duration_s=120, sampling_rate_hz=4, seed=2026)
    config = json.loads(Path(demo["single_config"]).read_text(encoding="utf-8"))
    config["output"]["directory"] = str(tmp_path / "real-pipeline-output")
    config.setdefault("workspace", {})["project_directory"] = str(tmp_path / "persisted-workbench")
    window, state = create_window(config)
    window.show()
    state["app"].processEvents()

    widgets = state["widgets"]
    assert len(widgets["pages"]) == 10
    assert len(widgets["template_buttons"]) == 9
    assert all(set(buttons) == {"view", "download", "import"} for buttons in widgets["template_buttons"].values())

    screenshot_root = os.environ.get("HEXIPLAN_UI_EVIDENCE_DIR")

    def capture_page(filename: str, index: int) -> None:
        if not screenshot_root:
            return
        target_dir = Path(screenshot_root)
        target_dir.mkdir(parents=True, exist_ok=True)
        widgets["navigation"].setCurrentRow(index)
        state["app"].processEvents()
        assert window.grab().save(str(target_dir / filename), "PNG")

    widgets["preset"].setCurrentIndex(widgets["preset"].findData("single_event"))
    assert state["config"]["pipeline_id"] == "single_event"
    widgets["preset"].setCurrentIndex(widgets["preset"].findData("single_standard"))
    widgets["config_json"].setPlainText(json.dumps(state["config"], ensure_ascii=False, indent=2))
    widgets["apply_json"].click()
    assert state["config"]["pipeline_id"] == "single_standard"

    # QC 页载入真实 synthetic records，并测试 GUI 到 core_qc 的追加/撤销/重做。
    widgets["qc_load"].click()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and state.get("qc_future") is not None:
        state["app"].processEvents()
        time.sleep(0.02)
    state["app"].processEvents()
    assert len(state["records"]) == 4
    widgets["qc_start"].setValue(10.0)
    widgets["qc_end"].setValue(12.0)
    widgets["qc_operator"].setText("smoke")
    widgets["qc_reason"].setText("GUI 审计通路验证")
    widgets["qc_exclude"].click()
    assert state["config"]["qc"]["annotations"][-1]["action"] == "exclude"
    widgets["qc_undo"].click()
    assert state["config"]["qc"]["annotations"][-1]["action"] == "undo"
    widgets["qc_redo"].click()
    assert state["config"]["qc"]["annotations"][-1]["action"] == "redo"
    assert state["project"]["qc_revision"] >= 3

    # Marker edits preserve the stable internal ID and are saved into the real
    # project snapshot consumed by the next preflighted task.
    metadata = state["records"][0]["metadata"]
    marker_fields = widgets["marker_fields"]
    for key, value in {"recording_id": metadata["recording_id"],
                       "channel": metadata.get("channel", "*"), "source_event_id": "001",
                       "onset_s": "10", "duration_s": "0.1", "event_type": "stimulus",
                       "condition": "demo", "clock_domain": metadata.get("clock_domain", "recording_local"),
                       "operator": "smoke", "reason": "GUI marker persistence"}.items():
        marker_fields[key].setText(str(value))
    widgets["marker_add"].click()
    marker_id = widgets["marker_table"].item(0, 0).data(256)  # Qt.UserRole
    assert marker_id
    widgets["marker_table"].setCurrentCell(0, 0)
    state["app"].processEvents()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and state.get("marker_waveform_canvas") is None:
        state["app"].processEvents()
        time.sleep(0.02)
    canvas = state.get("marker_waveform_canvas")
    axes = state.get("marker_waveform_axes")
    assert canvas is not None and axes is not None, (
        state["widgets"]["marker_waveform_status"].text() +
        f" · ids={state.get('qc_records_preview_id')}/{state.get('preview_id')} · "
        f"hashes={state.get('qc_records_config_sha256')}/{state.get('qc_load_config_sha256')}"
    )
    widgets["navigation"].setCurrentRow(8)
    state["app"].processEvents()
    marker_scroll = widgets["pages"]["Marker 编辑"]["scroll"]
    marker_scroll.ensureWidgetVisible(canvas)
    state["app"].processEvents()
    canvas.draw()
    assert canvas.isVisible() and canvas.height() >= 220 and canvas.width() >= 480, (
        f"waveform canvas is not visibly laid out: visible={canvas.isVisible()} "
        f"size={canvas.width()}x{canvas.height()} host={widgets['marker_waveform_host'].size()}"
    )
    assert len(axes.lines[0].get_xdata()) >= 2 and axes.get_xlabel().startswith("recording-local time")
    pixmap = canvas.grab()
    assert pixmap.width() >= 480 and pixmap.height() >= 220
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    signal_values = state["records"][0]["eda_uS"]
    y_value = float(signal_values[len(signal_values) // 2])
    x_pixel, y_pixel = axes.transData.transform((11.5, y_value))
    qt_point = QPoint(round(x_pixel), round(canvas.height() - y_pixel))
    assert canvas.rect().contains(qt_point), f"test click falls outside visible plot: {qt_point} {canvas.rect()}"
    clicked_time = axes.transData.inverted().transform(
        (qt_point.x(), canvas.height() - qt_point.y()))[0]
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=qt_point)
    state["app"].processEvents()
    actual_onset = float(state["widgets"]["marker_fields"]["onset_s"].text())
    assert abs(actual_onset - clicked_time) < 0.02
    assert abs(actual_onset - 11.5) < 0.08
    marker_fields["source_event_id"].setText("002")
    widgets["marker_save"].click()
    from eda_toolkit.marker_store import get_markers
    saved_marker = get_markers(state["project"])[0]
    assert saved_marker["marker_id"] == marker_id
    assert saved_marker["source_event_id"] == "002"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and "已校验当前输入文件 SHA256" not in widgets["marker_waveform_status"].text():
        state["app"].processEvents()
        time.sleep(0.02)
    assert "已校验当前输入文件 SHA256" in widgets["marker_waveform_status"].text(), (
        widgets["marker_waveform_status"].text())
    canvas = state.get("marker_waveform_canvas")
    axes = state.get("marker_waveform_axes")
    assert canvas is not None and axes is not None
    marker_scroll.ensureWidgetVisible(canvas)
    state["app"].processEvents()
    canvas.draw()
    assert canvas.isVisible() and canvas.height() >= 220 and canvas.width() >= 480
    assert axes.lines and len(axes.lines[0].get_xdata()) >= 2
    capture_page("marker-editor.png", 8)

    # Run button performs preflight and queues only; explicit Start Queue launches.
    feature_table = widgets["feature_registry_table"]
    mean_row = next(row for row in range(feature_table.rowCount())
                    if feature_table.item(row, 2).text() == "mean")
    feature_table.item(mean_row, 0).setCheckState(Qt.CheckState.Unchecked)
    frozen_feature_ids = list(state["config"]["features"]["ids"])
    assert frozen_feature_ids and "mean" not in frozen_feature_ids
    widgets["pages"]["分解与 SCR"]["scroll"].ensureWidgetVisible(feature_table)
    state["app"].processEvents()
    capture_page("metric-selectors.png", 4)
    widgets["extended_parameter_tabs"].setCurrentIndex(2)
    widgets["pages"]["分解与 SCR"]["scroll"].ensureWidgetVisible(widgets["extended_parameter_tabs"])
    state["app"].processEvents()
    capture_page("metric-parameters.png", 4)
    batch_clicks: list[bool] = []
    widgets["batch"].clicked.connect(lambda checked=False: batch_clicks.append(checked))
    widgets["batch"].click()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (state.get("preflight_future") is not None or
                                            state.get("pending_enqueue") is not None):
        state["app"].processEvents()
        time.sleep(0.02)
    state["app"].processEvents()
    assert state["preflight_report"] is not None, (
        f"preflight was not produced; enabled={widgets['batch'].isEnabled()} "
        f"warning={warning_messages} status={widgets['status'].text()} "
        f"task_status={widgets['task_status'].text()} busy={state.get('busy')} "
        f"preflight_future={state.get('preflight_future')} pending={state.get('pending_enqueue')} "
        f"run_future={state.get('run_future')} project={state.get('project')} "
        f"uncaught={uncaught_qt_errors} clicks={batch_clicks}"
    )
    assert state["preflight_report"]["eligible_to_enqueue"] is True, json.dumps(
        state["preflight_report"], ensure_ascii=False, indent=2)
    assert state["run_future"] is None
    from eda_toolkit.task_store import list_tasks
    queued = list_tasks(state["project"])
    assert len(queued) == 1 and queued[0]["state"] == "queued"
    assert queued[0]["config"]["features"]["ids"] == frozen_feature_ids
    assert widgets["progress"].value() == 0, "preflight completion is not analysis progress; queued work has not started"
    from eda_toolkit.marker_store import read_marker_csv
    frozen_events, _frozen_report = read_marker_csv(queued[0]["config"]["input"]["events"])
    frozen_marker = next(row for row in frozen_events if row.get("source_event_id") == "002")
    assert frozen_marker["event_id"] == saved_marker["event_id"]
    widgets["select_visible_features"].click()
    assert state["config"]["features"]["ids"] == []
    assert list_tasks(state["project"])[0]["config"]["features"]["ids"] == frozen_feature_ids
    capture_page("preflight-queued-task.png", 9)
    widgets["start_queue"].click()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and state["run_future"] is not None:
        state["app"].processEvents()
        time.sleep(0.025)
    state["app"].processEvents()
    assert state["run_future"] is None, "后台 pipeline 超时未完成"
    summary = state["last_summary"]
    assert summary and summary["status"] == "succeeded", summary
    assert summary["completed_records"] == 4
    assert (Path(summary["directory"]) / "manifest.json").is_file()
    assert any(item["path"].endswith("report.html") for item in summary["artifacts"])
    assert Path(summary["directory"], "runtime", "runtime_log.jsonl").is_file()
    work_events = [event for event in state["runtime_events"]
                   if event.get("kind") == "completed" and
                   isinstance(event.get("details"), dict) and event["details"].get("work_id")]
    assert len(work_events) >= 16, "pipeline stages must reach the GUI through the shared runtime event stream"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and state["widgets"]["progress"].value() < 100:
        state["app"].processEvents()
        time.sleep(0.02)
    assert state["widgets"]["progress"].value() == 100
    capture_page("completed-task-history.png", 9)

    window.close()
    state["shutdown"]()

    # Reopening the project restores durable task/marker/config state and logs.
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Yes))
    reopened_window, reopened = create_window(config)
    reopened_window.show()
    reopened["app"].processEvents()
    try:
        assert len(get_markers(reopened["project"])) == 1
        assert list_tasks(reopened["project"])[0]["state"] == "succeeded"
        from eda_toolkit.task_store import list_task_attempts
        reopened["widgets"]["task_table"].selectRow(0)
        reopened["app"].processEvents()
        assert reopened["widgets"]["task_open_result"].isEnabled()
        attempts = list_task_attempts(reopened["project"], list_tasks(reopened["project"])[0]["task_id"])
        expected_run_directory = Path(attempts[-1]["run_directory"]).resolve()
        reopened["widgets"]["task_open_result"].click()
        reopened["app"].processEvents()
        assert reopened["widgets"]["navigation"].currentRow() == 7
        assert Path(reopened["widgets"]["result_dir"].text()).resolve() == expected_run_directory
        assert reopened["last_existing_verification"]["valid"] is True
        assert "与当前 marker/QC 修订一致" in reopened["widgets"]["result_revision_status"].text()
        assert any(str(reopened["widgets"]["artifact_combo"].itemData(index)).endswith(".html")
                   for index in range(reopened["widgets"]["artifact_combo"].count()))
        from PySide6.QtWidgets import QTextBrowser
        assert isinstance(reopened["widgets"]["artifact_preview"].widget(), QTextBrowser)
        assert "EDA 分析报告" in reopened["widgets"]["artifact_preview"].widget().toPlainText()
        if screenshot_root:
            target_dir = Path(screenshot_root)
            target_dir.mkdir(parents=True, exist_ok=True)
            assert reopened_window.grab().save(str(target_dir / "selected-task-result.png"), "PNG")
        old_manifest_path = expected_run_directory / "manifest.json"
        old_manifest_sha256 = hashlib.sha256(old_manifest_path.read_bytes()).hexdigest()
        fields = reopened["widgets"]["marker_fields"]
        existing_marker = get_markers(reopened["project"])[0]
        fields["recording_id"].setText(existing_marker["recording_id"])
        fields["channel"].setText(existing_marker["channel_scope"])
        fields["source_event_id"].setText("marker-stale-r5")
        fields["onset_s"].setText("40.0")
        fields["duration_s"].setText("0.1")
        fields["event_type"].setText("stimulus")
        fields["condition"].setText("stale-check")
        fields["clock_domain"].setText(existing_marker["clock_domain"])
        fields["operator"].setText("smoke")
        fields["reason"].setText("verify historical results become stale")
        reopened["widgets"]["marker_add"].click()
        stale_text = reopened["widgets"]["result_revision_status"].text()
        assert "已过期" in stale_text and "marker r" in stale_text
        assert hashlib.sha256(old_manifest_path.read_bytes()).hexdigest() == old_manifest_sha256
        reopened["widgets"]["qc_load"].click()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and reopened.get("qc_future") is not None:
            reopened["app"].processEvents()
            time.sleep(0.02)
        reopened["app"].processEvents()
        reopened["widgets"]["qc_start"].setValue(24.0)
        reopened["widgets"]["qc_end"].setValue(26.0)
        fields_qc = reopened["widgets"]
        fields_qc["qc_operator"].setText("smoke")
        fields_qc["qc_reason"].setText("verify QC result staleness")
        fields_qc["qc_exclude"].click()
        stale_text = reopened["widgets"]["result_revision_status"].text()
        assert "已过期" in stale_text and "QC r" in stale_text
        assert hashlib.sha256(old_manifest_path.read_bytes()).hexdigest() == old_manifest_sha256
        reopened["widgets"]["navigation"].setCurrentRow(9)
        reopened["app"].processEvents()
        reopened["widgets"]["log_search"].click()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and reopened.get("log_search_future") is not None:
            reopened["app"].processEvents()
            time.sleep(0.02)
        reopened["app"].processEvents()
        log_next = reopened["widgets"]["log_next"]
        first_log_page = reopened["widgets"]["log_results"].toPlainText()
        assert "#500" in first_log_page and log_next.isEnabled(), (
            "the first 500 durable log events should expose the next page")
        log_next.click()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and reopened.get("log_search_future") is not None:
            reopened["app"].processEvents()
            time.sleep(0.02)
        reopened["app"].processEvents()
        assert "任务结束" in reopened["widgets"]["log_results"].toPlainText()
        assert not log_next.isEnabled(), "the final log page should disable pagination"
        if screenshot_root:
            target_dir = Path(screenshot_root)
            target_dir.mkdir(parents=True, exist_ok=True)
            reopened["widgets"]["navigation"].setCurrentRow(9)
            reopened["app"].processEvents()
            task_scroll = reopened["widgets"]["pages"]["任务与日志"]["scroll"].verticalScrollBar()
            assert task_scroll.maximum() > 0, "long pages must remain reachable through their scroll area"
            task_scroll.setValue(task_scroll.maximum())
            reopened["app"].processEvents()
            assert reopened_window.grab().save(str(target_dir / "persistent-log-center.png"), "PNG")
        assert not uncaught_qt_errors, "uncaught Qt/thread callback errors: " + "; ".join(uncaught_qt_errors)
    finally:
        reopened_window.close()
        reopened["shutdown"]()


def test_metric_registry_selector_and_extended_parameter_controls(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox
    from eda_toolkit.config import get_preset
    from eda_toolkit.core_features import LEGACY_FEATURE_IDS, list_feature_definitions
    from eda_toolkit.project_store import read_project_snapshot
    from eda_toolkit.ui.app import create_window

    config = get_preset()
    config["workspace"]["project_directory"] = str(tmp_path / "metric-selector-project")
    window, state = create_window(config)
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args: warnings.append(str(args[-1]))))
    try:
        window.show()
        state["app"].processEvents()
        widgets = state["widgets"]
        table = widgets["feature_registry_table"]
        active_ids = set(list_feature_definitions(state["config"]))
        catalog_config = get_preset("group_sync")
        catalog_config["decomposition"]["method"] = "bateman_sparse"
        catalog_ids = set(list_feature_definitions(catalog_config))
        assert len(active_ids) == 124
        assert len(catalog_ids) == 144
        assert table.rowCount() == len(catalog_ids)
        ids = {str(table.item(row, 2).text()) for row in range(table.rowCount())}
        assert ids == catalog_ids
        assert "mean" in ids and "scr_count" in ids
        group_id = "sync_positive_time_proportion"
        group_row = next(row for row in range(table.rowCount()) if table.item(row, 2).text() == group_id)
        assert not (table.item(group_row, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable)
        widgets["synchrony.enabled"].setChecked(True)
        state["app"].processEvents()
        registry_ids = set(list_feature_definitions(state["config"]))
        ids = {str(table.item(row, 2).text()) for row in range(table.rowCount())}
        assert table.rowCount() == 144 and len(registry_ids) == 135
        assert ids == catalog_ids and "sync_positive_time_proportion" in registry_ids
        if os.environ.get("HEXIPLAN_UI_EVIDENCE_DIR"):
            selector_root = Path(os.environ["HEXIPLAN_UI_EVIDENCE_DIR"])
            selector_root.mkdir(parents=True, exist_ok=True)
            widgets["navigation"].setCurrentRow(4)
            widgets["pages"]["分解与 SCR"]["scroll"].ensureWidgetVisible(table)
            state["app"].processEvents()
            assert window.grab().save(str(selector_root / "metric-selectors.png"), "PNG")
            widgets["extended_parameter_tabs"].setCurrentIndex(2)
            widgets["pages"]["分解与 SCR"]["scroll"].ensureWidgetVisible(widgets["extended_parameter_tabs"])
            state["app"].processEvents()
            assert window.grab().save(str(selector_root / "metric-parameters.png"), "PNG")
        initially_checked = {str(table.item(row, 0).data(Qt.ItemDataRole.UserRole))
                             for row in range(table.rowCount())
                             if table.item(row, 0).checkState() == Qt.CheckState.Checked}
        assert initially_checked == registry_ids  # [] means all registry entries.
        mean_row = next(row for row in range(table.rowCount()) if table.item(row, 2).text() == "mean")
        assert "单位：" in table.item(mean_row, 1).toolTip()
        table.scrollToItem(table.item(mean_row, 0))
        state["app"].processEvents()
        checkbox_rect = table.visualItemRect(table.item(mean_row, 0))
        QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                         pos=checkbox_rect.topLeft() + QPoint(8, checkbox_rect.height() // 2))
        state["app"].processEvents()
        assert state["config"]["features"]["ids"]
        assert "mean" not in state["config"]["features"]["ids"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and state.get("autosave_dirty"):
            state["app"].processEvents()
            time.sleep(0.02)
        saved_config = read_project_snapshot(state["project"])["config"]
        assert "mean" not in saved_config["features"]["ids"]
        advanced_id = next(item for item in sorted(ids) if item.startswith("clean_autocorr_lag_"))
        advanced_row = next(row for row in range(table.rowCount()) if table.item(row, 2).text() == advanced_id)
        table.setCurrentCell(advanced_row, 0)
        widgets["select_current_feature_only"].click()
        assert state["config"]["features"]["ids"] == [advanced_id]
        assert not set(state["config"]["features"]["ids"]).intersection(LEGACY_FEATURE_IDS)
        autocorrelation_editor = widgets["extended_parameter_editors"]["autocorrelation"]
        autocorrelation_editor.field_widgets["enabled"].setChecked(False)
        widgets["apply_extended_autocorrelation"].click()
        assert state["config"]["features"]["extended"]["autocorrelation"]["enabled"] is True
        assert any("先取消选择" in warning for warning in warnings)
        widgets["clear_visible_features"].click()
        assert state["config"]["features"]["ids"] == [advanced_id]
        checked_ids = {str(table.item(row, 0).data(Qt.ItemDataRole.UserRole))
                       for row in range(table.rowCount())
                       if table.item(row, 0).checkState() == Qt.CheckState.Checked}
        assert checked_ids == {advanced_id}
        widgets["select_visible_features"].click()
        state["app"].processEvents()
        assert state["config"]["features"]["ids"] == []
        checked_after_select_all = {str(table.item(row, 0).data(Qt.ItemDataRole.UserRole))
                                     for row in range(table.rowCount())
                                     if table.item(row, 0).checkState() == Qt.CheckState.Checked}
        assert checked_after_select_all == registry_ids
        assert "全部可用指标" in widgets["feature_selection_status"].text()

        spectral_editor = widgets["extended_parameter_editors"]["spectral"]
        spectral_editor.field_widgets["min_cycles"].setValue(2.5)
        widgets["apply_extended_spectral"].click()
        state["app"].processEvents()
        assert state["config"]["features"]["extended"]["spectral"]["min_cycles"] == 2.5
        spectral_bands = spectral_editor.field_widgets["bands"].table
        spectral_bands.cellWidget(0, 2).setValue(0.03)
        widgets["apply_extended_spectral"].click()
        assert warnings and state["config"]["features"]["extended"]["spectral"]["min_cycles"] == 2.5
        assert "未应用" in widgets["extended_parameter_status_spectral"].text()
        assert not hasattr(spectral_editor, "setPlainText")

        widgets["feature_baseline_min"].setValue(2e-6)
        state["app"].processEvents()
        assert state["config"]["features"]["extended"]["baseline_min_uS"] == 2e-6
        assert widgets["feature_family_filter"].count() >= 2

        group_row = next(row for row in range(table.rowCount()) if table.item(row, 2).text() == group_id)
        table.setCurrentCell(group_row, 0)
        widgets["select_current_feature_only"].click()
        assert state["config"]["features"]["ids"] == [group_id]
        widgets["synchrony.enabled"].setChecked(False)
        assert widgets["synchrony.enabled"].isChecked()
        assert state["config"]["features"]["ids"] == [group_id]
        assert any("群体时序指标" in warning for warning in warnings)
        table.setCurrentCell(mean_row, 0)
        widgets["select_current_feature_only"].click()
        widgets["synchrony.enabled"].setChecked(False)
        state["app"].processEvents()
        assert not state["config"]["synchrony"]["enabled"]
        assert state["config"]["features"]["ids"] == ["mean"]
        assert table.rowCount() == 144
        assert not (table.item(group_row, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable)
        widgets["synchrony.enabled"].setChecked(True)
        state["app"].processEvents()
        assert state["config"]["synchrony"]["enabled"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and state.get("autosave_dirty"):
            state["app"].processEvents()
            time.sleep(0.02)
        saved_config = read_project_snapshot(state["project"])["config"]
        assert saved_config["features"]["extended"]["baseline_min_uS"] == 2e-6
        assert saved_config["features"]["extended"]["spectral"]["min_cycles"] == 2.5

        workers = 3 if state["config"]["compute"]["workers"] != 3 else 4
        widgets["compute_workers"].setValue(workers)
        assert state["config"]["compute"]["workers"] == workers
        if state["config"]["decomposition"]["method"] != "bateman_sparse":
            widgets["compute_backend"].setCurrentIndex(widgets["compute_backend"].findData("gpu"))
            assert state["config"]["compute"]["backend"] == "cpu"
            assert any("GPU 仅支持 bateman_sparse" in warning for warning in warnings)
    finally:
        window.close()
        state["shutdown"]()


def test_run_metadata_help_uses_run_frozen_dynamic_definitions(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QDialog, QMessageBox, QTextBrowser
    from eda_toolkit.config import get_preset
    from eda_toolkit.ui.app import create_window

    run_dir = tmp_path / "historical-run"
    table_dir = run_dir / "tables"
    table_dir.mkdir(parents=True)
    run_config = get_preset()
    run_config["features"]["ids"] = ["clean_bandpower_custom_020_080_uS2"]
    run_config["features"]["extended"]["spectral"]["bands"] = [
        {"id": "custom_020_080", "low_hz": 0.2, "high_hz": 0.8}
    ]
    definition = {
        "clean_bandpower_custom_020_080_uS2": {
            "id": "clean_bandpower_custom_020_080_uS2",
            "label_zh": "本次冻结谱带绝对功率",
            "unit": "uS2",
            "basis": "clean_uS",
            "formula": "run-frozen 0.2–0.8 Hz band integral",
            "validity": "连续且均匀的有效窗口",
            "qualification": "experimental_engineering_checked",
            "parameters": {"low_hz": 0.2, "high_hz": 0.8},
        }
    }
    files = {
        "resolved_config.json": json.dumps(run_config, ensure_ascii=False),
        "provenance.json": json.dumps({"run_id": "historical-dynamic-run",
                                         "feature_definitions": definition}, ensure_ascii=False),
        "feature_definitions.json": json.dumps(definition, ensure_ascii=False),
        "tables/features_demo1234.csv": "feature_id,value,unit,status,reason\n"
                                              "clean_bandpower_custom_020_080_uS2,1.25,uS2,valid,\n",
    }
    artifacts = []
    for relative, content in files.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")
        path.write_bytes(encoded)
        artifacts.append({"path": relative.replace("\\", "/"), "size": len(encoded),
                          "sha256": hashlib.sha256(encoded).hexdigest()})
    (run_dir / "manifest.json").write_text(json.dumps({"schema_version": "1.0",
                                                        "run_id": "historical-dynamic-run",
                                                        "status": "succeeded",
                                                        "artifacts": artifacts}, ensure_ascii=False),
                                           encoding="utf-8")

    captured: dict[str, str] = {}
    info_messages: list[str] = []
    def capture_dialog(dialog):
        captured["text"] = dialog.findChild(QTextBrowser).toPlainText()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(QDialog, "exec", capture_dialog)
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *args: info_messages.append(str(args[-1]))))

    config = get_preset()
    config["workspace"]["project_directory"] = str(tmp_path / "metadata-project")
    window, state = create_window(config)
    try:
        state["widgets"]["result_dir"].setText(str(run_dir))
        state["widgets"]["run_metadata_help"].click()
        assert "historical-dynamic-run" in captured["text"]
        assert "clean_bandpower_custom_020_080_uS2" in captured["text"]
        assert "本次冻结谱带绝对功率" in captured["text"]
        assert "0.2–0.8 Hz band integral" in captured["text"]
        assert '"high_hz":0.8' in captured["text"] and '"low_hz":0.2' in captured["text"]
        assert "feature_id, value, unit, status, reason" in captured["text"]
        assert "clean_bandpower_eda_045_025_uS2" not in captured["text"]
        state["last_summary"] = {"directory": str(run_dir)}
        state["widgets"]["result_dir"].setText(str(tmp_path / "missing-run"))
        state["widgets"]["run_metadata_help"].click()
        assert info_messages and "已有运行目录" in info_messages[-1]
    finally:
        window.close()
        state["shutdown"]()


def test_marker_add_refuses_to_replace_unmanaged_configured_source_event(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QMessageBox
    from eda_toolkit.config import get_preset
    from eda_toolkit.marker_store import get_markers
    from eda_toolkit.ui.app import create_window

    source = tmp_path / "legacy-events.csv"
    source.write_text(
        "recording_id,event_id,onset_s,duration_s,event_type,channel,clock_domain\n"
        "r1,001,1.25,0.5,stimulus,c1,device\n", encoding="utf-8",
    )
    config = get_preset()
    config["input"]["events"] = str(source)
    config["workspace"]["project_directory"] = str(tmp_path / "project")
    window, state = create_window(config)
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *args: warnings.append(str(args[-1]))))
    try:
        fields = state["widgets"]["marker_fields"]
        for key, value in {"recording_id": "r1", "channel": "c1", "source_event_id": "001",
                           "onset_s": "3.0", "duration_s": "0.2", "event_type": "stimulus",
                           "condition": "A", "clock_domain": "device", "operator": "smoke",
                           "reason": "source conflict check"}.items():
            fields[key].setText(value)
        state["widgets"]["marker_add"].click()
        assert get_markers(state["project"]) == []
        assert warnings and "只读 events CSV" in warnings[-1]
    finally:
        window.close()
        state["shutdown"]()


def test_switch_project_flushes_pending_autosave_before_leaving_old_project(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    from eda_toolkit.config import get_preset
    from eda_toolkit.project_store import open_project, read_project_snapshot
    from eda_toolkit.ui.app import create_window

    old_project = tmp_path / "project-a"
    new_project = tmp_path / "project-b"
    config = get_preset()
    config["workspace"]["project_directory"] = str(old_project)
    config["output"]["directory"] = str(tmp_path / "old-results")
    window, state = create_window(config)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(
        lambda *args, **kwargs: str(new_project)))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *args, **kwargs: QMessageBox.StandardButton.No))
    try:
        expected_output = str(tmp_path / "new-results")
        state["widgets"]["output_directory"].setText(expected_output)
        state["widgets"]["output_directory"].editingFinished.emit()
        assert state["autosave_dirty"] is True
        assert state["autosave_timer"].isActive()
        state["widgets"]["project_open"].click()
        assert Path(state["project"]["path"]).resolve() == new_project.resolve()
        old_handle = open_project(old_project)
        saved = read_project_snapshot(old_handle)
        assert saved["config"]["output"]["directory"] == expected_output
        assert state["autosave_dirty"] is False
    finally:
        window.close()
        state["shutdown"]()


def test_recovery_prompt_lists_config_marker_and_qc_differences(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QMessageBox
    from eda_toolkit.config import get_preset
    from eda_toolkit.project_store import open_project, save_project_revision
    from eda_toolkit.ui.app import create_window

    project_path = tmp_path / "recovery-project"
    baseline = get_preset()
    baseline["workspace"]["project_directory"] = str(project_path)
    project = open_project(project_path)
    save_project_revision(project, {"config": baseline, "markers": []}, 0, kind="project")
    recovered_config = json.loads(json.dumps(baseline))
    recovered_config["qc"]["annotations"] = [{"action": "exclude", "start_s": 2.0,
                                                "end_s": 3.0, "reason": "recovery diff"}]
    marker = {"marker_id": "b70d4096-7d84-4aee-ae08-ec8dc8f4a9e4", "recording_id": "r1",
              "channel_scope": "c1", "source_event_id": "001", "onset_s": 2.5,
              "duration_s": 0.2, "event_type": "stimulus"}
    save_project_revision(project, {"config": recovered_config, "markers": [marker]}, 1,
                          kind="marker")

    prompts: list[str] = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *args, **kwargs: prompts.append(str(args[2])) or QMessageBox.StandardButton.No))
    window, state = create_window(baseline)
    try:
        assert prompts
        prompt = prompts[0]
        assert "qc.annotations" in prompt
        assert "marker：0 → 1 条" in prompt
        assert "QC：内嵌审计标注 0 → 1 条" in prompt
        assert "当前工作配置与恢复草稿字段差异" in prompt
    finally:
        window.close()
        state["shutdown"]()


def test_log_export_runs_in_background_worker(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QFileDialog
    import eda_toolkit.runtime_logging as runtime_logging
    from eda_toolkit.config import get_preset
    from eda_toolkit.ui.app import create_window

    project_path = tmp_path / "log-project"
    config = get_preset()
    config["workspace"]["project_directory"] = str(project_path)
    target = tmp_path / "exported.jsonl"
    original_export = runtime_logging.export_logs
    started = threading.Event()
    release = threading.Event()

    def slow_export(*args, **kwargs):
        started.set()
        release.wait(5)
        return original_export(*args, **kwargs)

    monkeypatch.setattr(runtime_logging, "export_logs", slow_export)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(
        lambda *args, **kwargs: (str(target), "JSONL (*.jsonl)")))
    window, state = create_window(config)
    try:
        state["widgets"]["log_export"].click()
        assert started.wait(2)
        assert state.get("log_export_future") is not None
        state["widgets"]["navigation"].setCurrentRow(0)
        state["app"].processEvents()
        assert state["widgets"]["navigation"].currentRow() == 0
        release.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and state.get("log_export_future") is not None:
            state["app"].processEvents()
            time.sleep(0.02)
        assert target.is_file()
        assert state.get("log_export_future") is None
    finally:
        release.set()
        window.close()
        state["shutdown"]()




def test_preflight_cancel_and_generation_discard_for_applied_config_and_project_changes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QFileDialog, QMessageBox
    import eda_toolkit.runtime_preflight as runtime_preflight
    from eda_toolkit.config import get_preset
    from eda_toolkit.task_store import list_tasks
    from eda_toolkit.ui.app import create_window

    starts: queue.Queue[Any] = queue.Queue()

    def cooperative_preflight(config, input_manifest, project_state, cancel_callback=None, event_sink=None):
        starts.put(cancel_callback)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (cancel_callback and cancel_callback()):
            time.sleep(0.01)
        return {"eligible_to_enqueue": True, "errors": [], "warnings": [],
                "work_plan": {"total_units": 1}, "report_id": "discarded-test-report"}

    monkeypatch.setattr(runtime_preflight, "preflight_run", cooperative_preflight)
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    config = get_preset()
    config["workspace"]["project_directory"] = str(project_a)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(
        lambda *args, **kwargs: str(project_b)))
    information: list[str] = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(
        lambda *args: information.append(str(args[2]))))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(
        lambda *args, **kwargs: QMessageBox.StandardButton.No))
    window, state = create_window(config)

    def wait_for_discard() -> None:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and state.get("preflight_future") is not None:
            state["app"].processEvents()
            time.sleep(0.01)
        state["app"].processEvents()
        assert state.get("preflight_future") is None
        assert state.get("preflight_report") is None
        assert state.get("pending_enqueue") is None
        assert list_tasks(state["project"]) == []

    def start_and_wait_for_worker() -> Any:
        state["widgets"]["batch"].click()
        return starts.get(timeout=10)

    try:
        cancel_callback = start_and_wait_for_worker()
        assert callable(cancel_callback)
        state["widgets"]["cancel"].click()
        wait_for_discard()
        assert cancel_callback() is True

        cancel_callback = start_and_wait_for_worker()
        gap_control = state["widgets"]["preprocess.interpolate_max_gap_s"]
        original_gap = gap_control.value()
        changed_gap = original_gap + 0.25 if original_gap <= 3599.5 else original_gap - 0.25
        gap_control.setValue(changed_gap)
        wait_for_discard()
        assert cancel_callback() is True

        gap_control.setValue(original_gap)
        cancel_callback = start_and_wait_for_worker()
        state["widgets"]["project_open"].click()
        assert Path(state["project"]["path"]).resolve() == project_a.resolve()
        assert information and "再次切换项目" in information[-1]
        wait_for_discard()
        assert cancel_callback() is True
        state["widgets"]["project_open"].click()
        assert Path(state["project"]["path"]).resolve() == project_b.resolve()
        assert state.get("preflight_report") is None
    finally:
        window.close()
        state["shutdown"]()




def test_gui_reconciliation_marks_confirmed_dead_owner_interrupted(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import sqlite3
    from eda_toolkit.config import get_preset
    from eda_toolkit.project_store import open_project
    from eda_toolkit.task_queue import claim_next_task
    from eda_toolkit.task_store import enqueue_task, get_task
    from eda_toolkit.ui.app import create_window
    import eda_toolkit.runtime_preflight as runtime_preflight

    project_path = tmp_path / "dead-owner-project"
    project = open_project(project_path)
    config = get_preset()
    config["workspace"]["project_directory"] = str(project_path)
    config["output"]["directory"] = str(tmp_path / "results")
    monkeypatch.setattr(runtime_preflight, "validate_preflight_snapshot",
                        lambda *_args, **_kwargs: {"valid": True, "stale": False, "errors": []})
    preflight = {"report_id": "dead-owner-fixture", "report_sha256": "a" * 64,
                 "eligible_to_enqueue": True, "errors": [], "project_id": project["project_id"],
                 "project_revision": project["project_revision"], "draft_revision": project["draft_revision"],
                 "marker_revision": project["marker_revision"], "qc_revision": project["qc_revision"],
                 "input_fingerprints": {}}
    frozen = {"config": config, "input_manifest": {"input_fingerprints": []},
              "project_revision": project["project_revision"], "draft_revision": project["draft_revision"],
              "marker_revision": project["marker_revision"], "qc_revision": project["qc_revision"]}
    task_id = enqueue_task(project, frozen, preflight)
    claimed = claim_next_task(project, "test-owner-that-will-exit")
    assert claimed and claimed["task_id"] == task_id
    with sqlite3.connect(project["db_path"]) as connection:
        connection.execute("UPDATE task_attempts SET pid=?,process_start_identity=? WHERE attempt_id=?",
                           (2_000_000_000, "known-dead-test-process", claimed["attempt_id"]))

    window, state = create_window(config)
    try:
        assert state["task_reconciliation"]["interrupted"] == [task_id]
        assert get_task(state["project"], task_id)["state"] == "interrupted"
        assert state["run_future"] is None
    finally:
        window.close()
        state["shutdown"]()
