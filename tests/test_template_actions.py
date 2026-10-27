"""实际点击九类 core_io 模板的查看、下载、导入 GUI 动作；不启动分析 pipeline。"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_all_template_view_download_import_actions_use_real_core_io(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QDialog, QFileDialog, QLabel, QMessageBox, QTableWidget
    from eda_toolkit.core_io import export_template, get_template_path, list_templates, preview_csv, read_template
    from eda_toolkit.ui.app import create_window

    templates = list_templates()
    assert len(templates) == 9
    window, state = create_window()
    viewed: dict[str, dict] = {}
    dialog_errors: list[str] = []
    current = {"name": None, "save_path": None, "open_path": None}

    def inspect_view_dialog(dialog):
        name = current["name"]
        assert name is not None
        example = read_template(name, "example")
        schema = read_template(name, "schema")
        table = dialog.findChild(QTableWidget)
        assert table is not None
        displayed_headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
        assert displayed_headers == [str(column) for column in example.columns]
        assert table.rowCount() == min(len(example), 200)
        if len(example) and len(example.columns):
            first = table.item(0, 0)
            expected = example.iloc[0, 0]
            assert (first.text() if first else "") == ("" if expected is None else str(expected))
        labels = [label.text() for label in dialog.findChildren(QLabel)]
        assert any(f"字段 schema 有 {len(schema.get('fields', []))} 项" in text for text in labels)
        assert "SYNTHETIC_ONLY" in dialog.windowTitle()
        viewed[name] = {"headers": displayed_headers, "row_count": table.rowCount(),
                        "schema_fields": len(schema.get("fields", []))}
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(QDialog, "exec", inspect_view_dialog)
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *_args, **kwargs: dialog_errors.append(str(_args[-1]) if _args else str(kwargs))))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *_args, **kwargs: dialog_errors.append(str(_args[-1]) if _args else str(kwargs))))
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        lambda *_args, **_kwargs: (str(current["save_path"]), "CSV (*.csv)"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        lambda *_args, **_kwargs: (str(current["open_path"]), "CSV (*.csv *.tsv)"))

    try:
        state["widgets"]["navigation"].setCurrentRow(1)
        for metadata in templates:
            name = metadata["name"]
            current["name"] = name
            buttons = state["template_buttons"][name]

            # 真实点击“查看”，拦截 Qt 模态循环并检查 handler 构造的真实表格/schema。
            buttons["view"].click()
            assert not dialog_errors, f"{name} 查看 handler 报错：{dialog_errors}"
            assert name in viewed

            # 真实点击“下载”，再逐文件核对 core_io 的 CSV 模板字节和表头。
            download_path = tmp_path / "downloads" / f"{name}.template.csv"
            current["save_path"] = download_path
            buttons["download"].click()
            assert not dialog_errors, f"{name} 下载 handler 报错：{dialog_errors}"
            assert download_path.is_file()
            assert download_path.read_bytes() == get_template_path(name, "template").read_bytes()
            expected_template = read_template(name, "template")
            exported = preview_csv(download_path, max_rows=3)
            assert list(exported["preview"].columns) == [str(column) for column in expected_template.columns]

            # 真实 example 通过 core_io 导出后交给 UI 导入 handler，检查绑定路径和预览控件。
            example_path = export_template(name, tmp_path / "imports" / f"{name}.example.csv", "example")
            current["open_path"] = example_path
            buttons["import"].click()
            assert not dialog_errors, f"{name} 导入 handler 报错：{dialog_errors}"
            imported = preview_csv(example_path, max_rows=60, options=state["config"])
            assert state["config"]["input"]["base_directory"] == str(example_path.parent.resolve())
            if name == "signals":
                assert state["config"]["input"]["signals"] == [str(example_path)]
                assert state["config"]["input"]["manifest"] is None
            elif name == "input_manifest":
                assert state["config"]["input"]["manifest"] == str(example_path)
            else:
                assert state["config"]["input"][name] == str(example_path)
            preview_title = state["widgets"]["preview_title"].text()
            assert f"{imported['total_rows']} 行" in preview_title
            preview_table = state["widgets"]["preview_host"].widget()
            assert isinstance(preview_table, QTableWidget)
            assert preview_table.rowCount() == min(imported["total_rows"], 60)
            assert [preview_table.horizontalHeaderItem(i).text() for i in range(preview_table.columnCount())] == list(imported["preview"].columns)

        assert set(viewed) == {item["name"] for item in templates}
    finally:
        window.close()
        state["shutdown"]()
