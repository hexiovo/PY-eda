"""Unified Chinese Excel workbook templates and validated workbook imports."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from ..core_io import list_templates, read_template, validate_table


_DEFAULT_FILENAME = "EDA研究工作簿.xlsx"
_STATUS_SHEET = "完整性状态"
_DICTIONARY_SHEET = "字段字典"
_INSTRUCTIONS_SHEET = "填写说明"
_SUPPORT_SHEETS = {_STATUS_SHEET, _DICTIONARY_SHEET, _INSTRUCTIONS_SHEET}
_INPUT_KEYS = {
    "signals": "signals",
    "input_manifest": "manifest",
}
_STATISTIC_DESIGN_TYPES = {
    "independent_correlation": "correlation",
    "random_intercept_lmm": "mixed_effects",
}
_STATISTIC_CONTRAST_KEYS = {
    "conditions", "groups", "condition_column", "group_column", "reference", "predictor",
}
_FIELD_LABELS = {
    "recording_id": "记录标识",
    "subject_id": "被试标识",
    "session_id": "会话标识",
    "channel": "通道",
    "time_s": "时间（秒）",
    "eda_uS": "皮肤电导（微西门子）",
    "clock_domain": "时钟域",
    "experimental_group": "实验组别",
    "age_years": "年龄（岁）",
    "sex": "性别/研究声明类别",
    "event_id": "事件标识",
    "onset_s": "起始时间（秒）",
    "duration_s": "持续时间（秒）",
    "event_type": "事件类型",
    "condition": "条件名称",
    "interval_id": "区间标识",
    "start_s": "区间起始（秒）",
    "end_s": "区间结束（秒）",
    "file_path": "信号文件路径",
    "sampling_rate_hz": "采样率（Hz）",
    "eda_unit": "来源 EDA 单位",
    "mapping_profile": "映射方案名称",
    "time_origin": "原始时间原点",
    "time_unit": "原始时间单位",
    "group_id": "互动/配对组标识",
    "role": "互动角色",
    "anchor_id": "同步锚点标识",
    "local_time_s": "本地锚点时间（秒）",
    "reference_time_s": "参考锚点时间（秒）",
    "reference_clock_domain": "参考时钟域",
    "uncertainty_s": "时间不确定度（秒）",
    "quality": "锚点质量或来源",
    "annotation_id": "QC 标注标识",
    "action": "QC 操作",
    "reason": "操作原因",
    "operator": "操作人员",
    "timestamp": "操作时间",
    "revision": "修订号",
    "parent_revision": "父修订号",
    "target_annotation_id": "目标标注标识",
    "design_id": "统计设计标识",
    "outcome": "结果变量/特征编号",
    "design_type": "统计方法",
    "independent_unit": "独立分析单位",
    "subject_id_column": "被试标识列名",
    "group_column": "实验组别列名",
    "session_id_column": "会话标识列名",
    "condition_column": "条件列名",
    "covariates": "协变量 JSON 数组",
    "contrast": "结构化对比 JSON",
    "correction_family": "多重检验家族",
}

_NAVY = "17365D"
_BLUE = "DDEBF7"
_ORANGE = "FCE4D6"
_LIGHT = "F4F7FA"
_GRAY = "666666"
_RED = "9C0006"
_GREEN = "E2F0D9"


def _catalog() -> list[tuple[dict[str, Any], dict[str, Any], list[str]]]:
    result = []
    for item in list_templates():
        schema = read_template(item["name"], "schema")
        template = read_template(item["name"], "template")
        result.append((item, schema, list(template.columns)))
    return result


def _input_key(item: dict[str, Any]) -> str:
    """Use the catalog binding when present, with compatibility for current entries."""
    declared = item.get("input_key")
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    return _INPUT_KEYS.get(item["name"], item["name"])


def _field_type(field: dict[str, Any]) -> str:
    value_type = field.get("type", "string")
    return "JSON" if value_type == "json" else value_type


def _conditional_fields(table_name: str, config: dict[str, Any] | None) -> dict[str, str]:
    """Return conditional required-field notes driven by the current config."""
    if table_name != "signals":
        return {}
    mapping = config.get("mapping", {}) if isinstance(config, dict) else {}
    rate = mapping.get("sampling_rate_hz") if isinstance(mapping, dict) else None
    if rate is None:
        return {"time_s": "信号表没有 time_s 时，必须在 mapping.sampling_rate_hz 中声明采样率。"}
    return {}


def _required_tables(config: dict[str, Any] | None) -> set[str]:
    """Table sheets made necessary by enabled analyses in the supplied config."""
    if not isinstance(config, dict):
        return set()
    required: set[str] = set()
    statistics = config.get("statistics", {})
    if isinstance(statistics, dict) and statistics.get("enabled"):
        required.add("statistical_design")
    synchrony = config.get("synchrony", {})
    if isinstance(synchrony, dict) and synchrony.get("enabled"):
        required.add("group_membership")
        if not synchrony.get("shared_clock_verified"):
            required.add("sync_anchors")
    features = config.get("features", {})
    feature_ids = features.get("ids", []) if isinstance(features, dict) else []
    if isinstance(feature_ids, (list, tuple)) and any(str(value).startswith("event_") for value in feature_ids):
        required.add("events")
    if config.get("pipeline_id") == "single_event":
        required.add("events")
    return required


def table_requirements(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Describe schema and config-dependent requirements for all workbook tables.

    This helper intentionally does not treat a configured path as a validated
    import. Use ``import_workbook`` statuses for complete/missing/error state.
    """
    required = _required_tables(config)
    result = {}
    for item, schema, _ in _catalog():
        name = item["name"]
        result[name] = {
            "table": name,
            "sheet_name": item["display_name"],
            "input_key": _input_key(item),
            "required_fields": [field["name"] for field in schema["fields"] if field.get("required")],
            "conditional_fields": _conditional_fields(name, config),
            "required_by_config": name in required,
        }
    return result


def _header_comment(field: dict[str, Any]) -> str:
    name = field["name"]
    required_column = bool(field.get("required"))
    required_value = field.get("missing", "allowed") == "forbidden"
    description = field.get("description") or _FIELD_LABELS.get(name, "按字段 ID 填写。")
    lines = [
        f"字段说明：{description}",
        f"列必需：{'是' if required_column else '否'}",
        f"每行必填：{'是' if required_value else '否'}",
        f"数据类型：{_field_type(field)}",
    ]
    unit = field.get("unit")
    if unit and unit != "none":
        lines.append(f"单位：{unit}")
    allowed = field.get("allowed_values")
    if allowed:
        lines.append("允许值：" + "、".join(str(value) for value in allowed))
    foreign_key = field.get("foreign_key")
    if foreign_key:
        lines.append(f"参考：{foreign_key}")
    clock = field.get("clock")
    if clock:
        lines.append(f"时间所属字段：{clock}")
    if name == "file_path":
        lines.append("相对路径以本工作簿所在目录为基准。")
    if name == "covariates":
        lines.append("填写 JSON 字符串数组，例如 [\"age_years\"]。")
    if name == "contrast":
        lines.append("填写结构化 JSON 对象；不接受公式或可执行代码。")
    if name in ("start_s", "end_s") and field.get("description", "").startswith(("exclude/restore",)):
        lines.append("仅 action=exclude 或 restore 时必填。")
    if name == "target_annotation_id":
        lines.append("仅 action=undo 或 redo 时必填。")
    return "\n".join(lines)


def _style_header(cell, required: bool) -> None:
    cell.fill = PatternFill("solid", fgColor=_ORANGE if required else _BLUE)
    cell.font = Font(bold=True, color=_NAVY)
    cell.alignment = Alignment(vertical="center", wrap_text=True)
    cell.border = Border(bottom=Side(style="medium", color=_NAVY))


def _write_instructions(workbook: Workbook) -> None:
    sheet = workbook.create_sheet(_INSTRUCTIONS_SHEET)
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 27
    sheet.column_dimensions["B"].width = 106
    rows = [
        ("EDA 研究数据工作簿", "统一填写九类 EDA 输入数据。模板数据区为空，不含示例记录。"),
        ("填写顺序", "按需要填写中文数据工作表；未使用的可选表保持空白即可。"),
        ("表头", "第一行字段 ID 是导入列名，请勿修改或翻译。橙色表头为必需列，蓝色表头为可选列；悬停字段 ID 可查看说明。"),
        ("信号来源（二选一）", "可直接填写“信号”长表，或填写“输入清单”并在 file_path 中引用原始信号文件。不要同时填写这两张表。"),
        ("信号长表", "每行是一条采样值。保留原始时间顺序；同一 recording_id/channel 内 time_s 必须有限且严格递增。没有 time_s 时，请在配置中声明 mapping.sampling_rate_hz。"),
        ("输入清单路径", "file_path 支持绝对路径或相对路径；相对路径按本工作簿所在目录解析。导入时会把可用相对路径解析到绝对路径快照。"),
        ("标识与单位", "recording_id、subject_id、session_id、channel 等标识按文本填写；标准信号幅值列 eda_uS 的单位固定为微西门子。"),
        ("可选字段", "可选字段或可选表可留空。必需列标记不一定表示每个单元格都必填；请查看“字段字典”中的“每行必填”。"),
        ("统计设计", "covariates 与 contrast 使用 JSON。contrast 只保存结构化对比字段，不执行公式。独立相关请在 contrast 中声明 predictor。"),
        ("人工 QC", "exclude/restore 必须填写 start_s 和 end_s；undo/redo 必须填写 target_annotation_id。时间区间采用 [start_s,end_s)。"),
        ("状态", "导入时会逐表显示完整、缺失或无效状态、数据行数和错误字段。“完整性状态”页列出每张表的必需字段和状态判定方式。"),
        ("状态含义", "完整：有数据且通过字段/关系检查。缺少必需：启用的分析依赖此表但表为空，或缺少必需字段。无效：字段值、引用或格式未通过检查。未填写的可选表不会阻止导入。"),
    ]
    for row_number, (label, message) in enumerate(rows, start=1):
        sheet.cell(row_number, 1, label)
        sheet.cell(row_number, 2, message)
        sheet.cell(row_number, 1).font = Font(bold=True, color=_NAVY)
        sheet.cell(row_number, 1).fill = PatternFill("solid", fgColor=_LIGHT)
        for cell in sheet[row_number]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        sheet.row_dimensions[row_number].height = 34 if row_number in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12) else 28
    sheet.freeze_panes = "A2"


def _write_status_sheet(workbook: Workbook, catalog) -> None:
    sheet = workbook.create_sheet(_STATUS_SHEET)
    sheet.sheet_view.showGridLines = False
    headers = ["工作表", "数据表 ID", "表用途", "必需字段（列）", "完整性状态", "必需性说明"]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor=_NAVY)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for item, schema, _ in catalog:
        name = item["name"]
        required_fields = [field["name"] for field in schema["fields"] if field.get("required")]
        requirement = {
            "signals": "与输入清单二选一作为信号来源",
            "input_manifest": "与信号长表二选一作为信号来源",
        }.get(name, "启用对应分析时必需；否则为可选表")
        sheet.append([
            item["display_name"], name, item.get("description", ""),
            ", ".join(required_fields), "导入时检查", requirement,
        ])
    widths = [22, 24, 56, 52, 19, 41]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        row[4].fill = PatternFill("solid", fgColor=_GREEN)
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def _write_dictionary_sheet(workbook: Workbook, catalog) -> None:
    sheet = workbook.create_sheet(_DICTIONARY_SHEET)
    sheet.sheet_view.showGridLines = False
    headers = [
        "工作表", "数据表 ID", "字段 ID（导入列名）", "中文说明", "类型", "单位",
        "列必需", "每行必填", "允许值", "参考字段", "时钟字段",
    ]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.fill = PatternFill("solid", fgColor=_NAVY)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for item, schema, _ in catalog:
        for field in schema["fields"]:
            name = field["name"]
            sheet.append([
                item["display_name"], item["name"], name,
                field.get("description") or _FIELD_LABELS.get(name, "按字段 ID 填写。"),
                _field_type(field), field.get("unit", "none"),
                "是" if field.get("required") else "否",
                "是" if field.get("missing", "allowed") == "forbidden" else "否",
                ", ".join(str(value) for value in field.get("allowed_values", [])),
                field.get("foreign_key") or "", field.get("clock") or "",
            ])
            color = _ORANGE if field.get("required") else _BLUE
            for cell in sheet[sheet.max_row]:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            sheet.cell(sheet.max_row, 3).fill = PatternFill("solid", fgColor=color)
    for index, width in enumerate([22, 24, 28, 62, 15, 14, 12, 12, 54, 35, 25], start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions


def _write_data_sheet(workbook: Workbook, item: dict[str, Any], schema: dict[str, Any], headers: list[str]) -> None:
    sheet = workbook.create_sheet(item["display_name"])
    sheet.sheet_view.showGridLines = False
    sheet.freeze_panes = "A2"
    fields = {field["name"]: field for field in schema["fields"]}
    for index, name in enumerate(headers, start=1):
        cell = sheet.cell(1, index, name)
        field = fields[name]
        _style_header(cell, bool(field.get("required")))
        cell.comment = Comment(_header_comment(field), "EDA Toolkit")
        width = max(14, min(30, len(name) + 4))
        sheet.column_dimensions[get_column_letter(index)].width = width
        if field.get("type") == "string":
            sheet.column_dimensions[get_column_letter(index)].number_format = "@"
    sheet.row_dimensions[1].height = 33
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"
    for field in schema["fields"]:
        allowed = field.get("allowed_values")
        if not allowed or field["name"] not in headers:
            continue
        values = ",".join(str(value) for value in allowed)
        if len(values) <= 255:
            column = get_column_letter(headers.index(field["name"]) + 1)
            rule = DataValidation(type="list", formula1=f'"{values}"', allow_blank=True)
            rule.error = "请选择允许值中的一项。"
            rule.errorTitle = "字段值不符合 schema"
            sheet.add_data_validation(rule)
            rule.add(f"{column}2:{column}1048576")


def export_workbook(path: str | os.PathLike[str]) -> Path:
    """Create the empty, schema-driven Chinese EDA input workbook."""
    target = Path(path).expanduser()
    if target.exists() and target.is_dir():
        target = target / _DEFAULT_FILENAME
    elif not target.suffix:
        target = target.with_suffix(".xlsx")
    if target.suffix.casefold() != ".xlsx":
        raise ValueError("统一研究工作簿必须使用 .xlsx 格式。")
    target.parent.mkdir(parents=True, exist_ok=True)

    catalog = _catalog()
    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.title = "EDA 研究数据工作簿"
    workbook.properties.subject = "九类 EDA 输入表的统一空白模板"
    workbook.properties.creator = "EDA Toolkit"
    _write_instructions(workbook)
    _write_status_sheet(workbook, catalog)
    _write_dictionary_sheet(workbook, catalog)
    for item, schema, headers in catalog:
        _write_data_sheet(workbook, item, schema, headers)

    temporary = target.with_name(f".{target.stem}.writing{target.suffix}")
    try:
        workbook.save(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target.resolve()


def _is_empty(value: Any) -> bool:
    return value is None or value == ""


def _cell_value(value: Any, field: dict[str, Any] | None, number_format: str | None) -> Any:
    if _is_empty(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, datetime) else value.isoformat()
    if field and field.get("type") == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        # Text-formatted columns preserve identifiers such as 001. If Excel stored a
        # formatted numeric identifier, retain the visible leading-zero width.
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(number_format, str) and number_format and set(number_format) == {"0"} and float(value).is_integer():
            return f"{int(value):0{len(number_format)}d}"
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".15g")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _error(code: str, table: str, message: str, *, row: int | None = None,
           column: str | None = None) -> dict[str, Any]:
    return {"code": code, "table": table, "row": row, "column": column, "message": message}


def _read_sheet(sheet, table_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Read one data sheet without pandas' default NaN/identifier coercions."""
    max_column = max(sheet.max_column, 1)
    header_cells = list(sheet.iter_rows(min_row=1, max_row=1, min_col=1, max_col=max_column))[0]
    raw_headers = [cell.value for cell in header_cells]
    while raw_headers and _is_empty(raw_headers[-1]):
        raw_headers.pop()
        header_cells.pop()
    has_data = False
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, min_col=1, max_col=max_column):
        if any(not _is_empty(cell.value) for cell in row):
            has_data = True
            break

    if not raw_headers:
        if not has_data:
            return {"frame": None, "errors": [], "rows": 0, "empty": True, "structure_valid": True}
        return {
            "frame": None,
            "errors": [_error("missing_header", table_name, "工作表有数据，但第一行缺少字段 ID 表头。")],
            "rows": 0,
            "empty": False,
            "structure_valid": False,
        }

    headers = [str(value) if value is not None else "" for value in raw_headers]
    errors: list[dict[str, Any]] = []
    blank_headers = [index + 1 for index, value in enumerate(headers) if not value]
    if blank_headers:
        errors.append(_error("empty_header", table_name, f"字段 ID 表头为空：列 {blank_headers}。"))
    duplicates = sorted({name for name in headers if name and headers.count(name) > 1})
    if duplicates:
        errors.append(_error("duplicate_header", table_name, f"字段 ID 表头重复：{duplicates}。"))

    fields = {field["name"]: field for field in schema["fields"]}
    unknown = [name for name in headers if name and name not in fields]
    for name in unknown:
        errors.append(_error("unknown_column", table_name, f"不认识字段 ID {name!r}；请恢复 schema 中的原始字段 ID。", column=name))

    records: list[dict[str, Any]] = []
    row_indexes: list[int] = []
    formula_errors = []
    for excel_row, cells in enumerate(
        sheet.iter_rows(min_row=2, max_row=sheet.max_row, min_col=1, max_col=max_column), start=2
    ):
        if not any(not _is_empty(cell.value) for cell in cells):
            continue
        record: dict[str, Any] = {}
        for index, cell in enumerate(cells):
            if index >= len(headers):
                if not _is_empty(cell.value):
                    errors.append(_error("value_without_header", table_name,
                                         "表头右侧存在未命名的数据列。", row=excel_row))
                continue
            name = headers[index]
            if cell.data_type == "f":
                formula_errors.append((excel_row, name))
            record[name] = _cell_value(cell.value, fields.get(name), cell.number_format)
        records.append(record)
        row_indexes.append(excel_row - 2)
    for row, column in formula_errors:
        errors.append(_error("formula_not_allowed", table_name,
                             "输入数据不能是 Excel 公式；请粘贴为值。", row=row, column=column or None))
    if records:
        frame = pd.DataFrame(records, columns=headers, index=row_indexes, dtype=object)
    else:
        frame = pd.DataFrame(columns=headers, dtype=object)
    return {
        "frame": frame,
        "errors": errors,
        "rows": len(records),
        "empty": not records,
        "structure_valid": not any(item["code"] in {"empty_header", "duplicate_header", "value_without_header", "formula_not_allowed"} for item in errors),
    }


def _conditional_errors(table_name: str, frame: pd.DataFrame, config: dict[str, Any] | None) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if table_name == "signals" and "time_s" not in frame.columns:
        mapping = config.get("mapping", {}) if isinstance(config, dict) else {}
        rate = mapping.get("sampling_rate_hz") if isinstance(mapping, dict) else None
        if rate is None:
            errors.append(_error(
                "conditional_required", table_name,
                "缺少 time_s；请添加时间列，或在 mapping.sampling_rate_hz 中声明采样率。",
                column="time_s",
            ))
    if table_name == "manual_qc" and "action" in frame:
        for index, row in frame.iterrows():
            action = "" if pd.isna(row.get("action")) else str(row.get("action")).strip().casefold()
            required = ("start_s", "end_s") if action in ("exclude", "restore") else (
                ("target_annotation_id",) if action in ("undo", "redo") else ()
            )
            for field_name in required:
                value = row.get(field_name) if field_name in frame else None
                if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
                    errors.append(_error(
                        "conditional_required", table_name,
                        f"action={action} 时字段 {field_name} 必填。",
                        row=int(index) + 2, column=field_name,
                    ))
    return errors


def _resolve_manifest_paths(frame: pd.DataFrame, workbook_directory: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    snapshot = frame.copy(deep=True)
    errors: list[dict[str, Any]] = []
    if "file_path" not in snapshot:
        return snapshot, errors
    for index, value in snapshot["file_path"].items():
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
            continue
        source = Path(str(value).strip()).expanduser()
        if not source.is_absolute():
            source = workbook_directory / source
        resolved = source.resolve()
        snapshot.at[index, "file_path"] = str(resolved)
        if not resolved.is_file():
            errors.append(_error(
                "source_file_missing", "input_manifest",
                f"输入清单文件不存在：{resolved}",
                row=int(index) + 2, column="file_path",
            ))
    return snapshot, errors


def _translate_statistics(frame: pd.DataFrame) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    designs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        item: dict[str, Any] = {}
        for name, value in row.items():
            if value is None or pd.isna(value) or isinstance(value, str) and not value.strip():
                continue
            if name in ("covariates", "contrast"):
                try:
                    item[name] = json.loads(value) if isinstance(value, str) else value
                except (TypeError, ValueError, json.JSONDecodeError):
                    # validate_table will already report malformed JSON; retain an
                    # additional conversion diagnostic rather than binding a partial design.
                    errors.append(_error("statistics_design_json", "statistical_design",
                                         f"{name} 必须是 JSON。", row=int(index) + 2, column=name))
                    continue
            else:
                item[name] = value

        if not isinstance(item.get("covariates", []), list) or not all(
            isinstance(value, str) for value in item.get("covariates", [])
        ):
            errors.append(_error("statistics_design_covariates", "statistical_design",
                                 "covariates 必须是列名字符串数组。", row=int(index) + 2, column="covariates"))
        contrast = item.get("contrast")
        if contrast is not None and (not isinstance(contrast, dict) or set(contrast) - _STATISTIC_CONTRAST_KEYS):
            errors.append(_error("statistics_design_contrast", "statistical_design",
                                 "contrast 必须是受支持字段组成的 JSON 对象。",
                                 row=int(index) + 2, column="contrast"))

        design_type = item.get("design_type")
        if design_type in _STATISTIC_DESIGN_TYPES:
            item["design_type"] = _STATISTIC_DESIGN_TYPES[design_type]
        if design_type == "independent_correlation" and not (isinstance(contrast, dict) and contrast.get("predictor")):
            errors.append(_error("statistics_design_predictor", "statistical_design",
                                 "independent_correlation 必须在 contrast 中声明 predictor。",
                                 row=int(index) + 2, column="contrast"))
        if design_type == "random_intercept_lmm":
            item["random_intercept"] = True

        for source, target in (("subject_id_column", "subject_column"),
                               ("session_id_column", "session_column")):
            if source in item:
                if target in item and item[target] != item[source]:
                    errors.append(_error("statistics_design_alias_conflict", "statistical_design",
                                         f"{source} 与 {target} 冲突。", row=int(index) + 2, column=source))
                item[target] = item.pop(source)
        designs.append(item)
    return (None if errors else designs), errors


def _missing_fields(errors: list[dict[str, Any]], required_fields: list[str]) -> list[str]:
    missing = []
    for item in errors:
        if item.get("code") in {"required_column", "missing_value", "conditional_required"} and item.get("column"):
            if item["column"] not in missing:
                missing.append(item["column"])
    if any(item.get("code") in {"required_sheet_empty", "required_sheet_missing"} for item in errors):
        return list(required_fields)
    return missing


def _write_snapshot(frame: pd.DataFrame, target: Path) -> None:
    frame.to_csv(
        target,
        index=False,
        encoding="utf-8-sig",
        na_rep="",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )


def import_workbook(
    path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and snapshot populated data sheets into a unique import directory.

    The returned ``bindings`` mapping can replace ``config['input']``. Invalid or
    empty tables never point at a snapshot. Statistics designs are returned in
    the pipeline's accepted shape and an empty statistics sheet returns ``[]``
    so a prior design cannot remain active by accident.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_root = Path(destination).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    workbook = load_workbook(source, data_only=False, read_only=False)
    catalog = _catalog()
    required_tables = _required_tables(config)
    catalog_by_name = {item["name"]: (item, schema, headers) for item, schema, headers in catalog}
    sheet_to_table = {item["display_name"]: item["name"] for item, _, _ in catalog}
    unrecognized = [sheet.title for sheet in workbook.worksheets
                    if sheet.title not in sheet_to_table and sheet.title not in _SUPPORT_SHEETS]

    parsed: dict[str, dict[str, Any]] = {}
    statuses: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for item, schema, headers in catalog:
        name = item["name"]
        required_fields = [field["name"] for field in schema["fields"] if field.get("required")]
        required_by_config = name in required_tables
        if item["display_name"] not in workbook.sheetnames:
            errors = []
            status = "not_provided"
            if required_by_config:
                errors.append(_error("required_sheet_missing", name,
                                     "当前配置启用的分析需要此表，但工作簿中没有对应工作表。"))
                status = "missing_required"
            parsed[name] = {"frame": None, "errors": errors, "rows": 0, "empty": True,
                            "structure_valid": True, "present": False}
            statuses[name] = {
                "table": name, "sheet_name": item["display_name"], "input_key": _input_key(item),
                "required_by_config": required_by_config, "status": status, "rows": 0,
                "required_fields": required_fields, "conditional_fields": _conditional_fields(name, config),
                "missing_fields": list(required_fields) if required_by_config else [], "errors": errors,
            }
            continue
        result = _read_sheet(workbook[item["display_name"]], name, schema)
        result["present"] = True
        parsed[name] = result
        frame = result["frame"]
        errors = list(result["errors"])
        if frame is not None and not result["empty"] and result["structure_valid"]:
            frames[name] = frame
            valid_fields = {field["name"] for field in schema["fields"]}
            unknown = [column for column in frame.columns if column not in valid_fields]
            # Unknown fields were recorded by _read_sheet. Avoid duplicate diagnostics.
            if not unknown:
                pass
        elif frame is not None and not result["empty"]:
            # Structurally invalid data is still captured as a snapshot for review,
            # but it is excluded from cross-table reference targets.
            frames[name] = frame

        if result["empty"]:
            status = "optional_empty"
            if required_by_config:
                errors.append(_error("required_sheet_empty", name,
                                     "当前配置启用的分析需要此表，但工作表没有数据行。"))
                status = "missing_required"
        elif not result["structure_valid"]:
            status = "invalid"
        else:
            status = "pending_validation"
        statuses[name] = {
            "table": name, "sheet_name": item["display_name"], "input_key": _input_key(item),
            "required_by_config": required_by_config, "status": status, "rows": result["rows"],
            "required_fields": required_fields, "conditional_fields": _conditional_fields(name, config),
            "missing_fields": _missing_fields(errors, required_fields), "errors": errors,
        }

    # All value-based cross-table checks receive the workbook's populated tables.
    # statistical_design.*_column cells name analysis columns, not referenced data
    # values, so those two catalog references are deliberately omitted for that
    # table while its schema/type checks still run through validate_table.
    for name, result in parsed.items():
        frame = result["frame"]
        status = statuses[name]
        if frame is None or result["empty"] or not result["structure_valid"]:
            continue
        reference_tables = dict(frames)
        if name == "statistical_design":
            reference_tables.pop("participants", None)
            reference_tables.pop("conditions", None)
        validation = validate_table(name, frame, tables=reference_tables)
        errors = list(status["errors"]) + list(validation.get("errors", []))
        errors.extend(_conditional_errors(name, frame, config))
        if name == "input_manifest":
            _, path_errors = _resolve_manifest_paths(frame, source.parent)
            errors.extend(path_errors)
        statistics_designs = None
        if name == "statistical_design":
            statistics_designs, translation_errors = _translate_statistics(frame)
            errors.extend(translation_errors)
        status["errors"] = errors
        status["missing_fields"] = _missing_fields(errors, status["required_fields"])
        if errors:
            status["status"] = "invalid"
        else:
            status["status"] = "complete"
        result["statistics_designs"] = statistics_designs
        result["validated"] = not errors

    # The two signal source formats are alternatives; silently preferring the
    # manifest would otherwise make a populated signals sheet appear imported.
    signal_data = parsed.get("signals", {})
    manifest_data = parsed.get("input_manifest", {})
    both_signal_sources = (
        not signal_data.get("empty", True) and not manifest_data.get("empty", True)
        and statuses["signals"]["status"] == "complete"
        and statuses["input_manifest"]["status"] == "complete"
    )
    if both_signal_sources:
        for name in ("signals", "input_manifest"):
            statuses[name]["status"] = "invalid"
            statuses[name]["errors"].append(_error(
                "signal_source_conflict", name,
                "请只填写“信号”或“输入清单”其中一张表，避免配置忽略另一种信号来源。",
            ))
            parsed[name]["validated"] = False

    has_valid_signal_source = any(
        statuses[name]["status"] == "complete" for name in ("signals", "input_manifest")
    )
    requirements = [{
        "id": "signal_source", "label": "信号来源", "kind": "one_of",
        "tables": ["signals", "input_manifest"],
        "status": "complete" if has_valid_signal_source else "missing_required",
        "satisfied_by": next((name for name in ("signals", "input_manifest")
                              if statuses[name]["status"] == "complete"), None),
        "message": "信号表与输入清单二选一。" if not has_valid_signal_source else "已提供有效信号来源。",
    }]
    if not has_valid_signal_source:
        requirements[0]["errors"] = [_error(
            "signal_source_missing", "signals",
            "请填写“信号”长表或“输入清单”其中一张表。",
        )]
        if statuses["signals"]["status"] in {"optional_empty", "not_provided"}:
            statuses["signals"]["status"] = "missing_required"
            statuses["signals"]["required_by_config"] = True
            statuses["signals"]["errors"].append(requirements[0]["errors"][0])

    # Allocate snapshots only after parsing/validation so no existing files can be
    # overwritten. Invalid nonempty sheets remain available for review but do not
    # produce an active binding.
    nonempty_frames = {
        name: result["frame"] for name, result in parsed.items()
        if result.get("frame") is not None and not result.get("empty")
    }
    snapshot_directory: Path | None = None
    snapshot_paths: dict[str, Path] = {}
    if nonempty_frames:
        snapshot_directory = Path(tempfile.mkdtemp(prefix="eda-workbook-import-", dir=destination_root)).resolve()
        for name, frame in nonempty_frames.items():
            snapshot = frame
            if name == "input_manifest":
                snapshot, _ = _resolve_manifest_paths(frame, source.parent)
            snapshot_path = snapshot_directory / f"{name}.csv"
            _write_snapshot(snapshot, snapshot_path)
            snapshot_paths[name] = snapshot_path

    bindings: dict[str, Any] = {
        "manifest": None,
        "signals": [],
        "participants": None,
        "events": None,
        "conditions": None,
        "group_membership": None,
        "sync_anchors": None,
        "manual_qc": None,
        "statistical_design": None,
        "base_directory": str(source.parent),
    }
    for item, _, _ in catalog:
        name = item["name"]
        snapshot_path = snapshot_paths.get(name)
        if snapshot_path is None or statuses[name]["status"] != "complete":
            continue
        key = _input_key(item)
        if name == "signals":
            bindings[key] = [str(snapshot_path)]
        else:
            bindings[key] = str(snapshot_path)

    stats_status = statuses["statistical_design"]["status"]
    if not parsed["statistical_design"].get("present") or parsed["statistical_design"].get("empty"):
        statistics_designs: list[dict[str, Any]] | None = []
    elif stats_status == "complete":
        statistics_designs = parsed["statistical_design"].get("statistics_designs", [])
    else:
        statistics_designs = None

    for status in statuses.values():
        status["missing_fields"] = _missing_fields(status["errors"], status["required_fields"])

    return {
        "source_workbook": str(source),
        "snapshot_directory": str(snapshot_directory) if snapshot_directory else None,
        "snapshots": {name: str(path) for name, path in snapshot_paths.items()},
        "bindings": bindings,
        "statuses": statuses,
        "requirements": requirements,
        "statistics_designs": statistics_designs,
        "unrecognized_sheets": unrecognized,
    }

