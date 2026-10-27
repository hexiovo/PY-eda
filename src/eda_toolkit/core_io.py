"""CSV 输入、模板和关系校验；时间统一为秒，电导统一为微西门子。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import make_record
from .errors import fail


_TEMPLATE_PACKAGE = "eda_toolkit.resources.templates"
_TEMPLATE_NAMES = (
    "signals", "participants", "events", "conditions", "input_manifest",
    "group_membership", "sync_anchors", "manual_qc", "statistical_design",
)
_ASSOCIATED_INPUTS = (
    "participants", "events", "conditions", "group_membership", "sync_anchors",
    "manual_qc", "statistical_design",
)
_TIME_FACTORS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
_UNIT_ALIASES = {
    "us": "uS", "µs": "uS", "μs": "uS", "microsiemens": "uS",
    "microsiemen": "uS", "ms": "mS", "s": "S", "siemens": "S",
    "ohm": "ohm", "ω": "ohm", "Ω": "ohm", "kohm": "kohm",
    "kΩ": "kohm", "kohms": "kohm", "mohm": "Mohm", "mΩ": "Mohm",
    "megohm": "Mohm", "megohms": "Mohm",
}
_CONDUCTANCE_TO_US = {"uS": 1.0, "mS": 1e3, "S": 1e6}


def _mapping(value: Any, options: Any) -> dict[str, Any]:
    """合并调用方映射和整份配置中的 mapping 小节，不做模糊列名猜测。"""
    option_mapping = options.get("mapping", {}) if isinstance(options, dict) else {}
    result = dict(option_mapping) if isinstance(option_mapping, dict) else {}
    if value is None:
        return result
    if isinstance(value, str):
        if value.casefold() not in ("default", "signals", "standard"):
            fail("mapping_profile_missing", f"未提供映射配置：{value}", field="mapping_profile")
        result["name"] = value
        return result
    if not isinstance(value, dict):
        fail("mapping_type", "mapping 必须是字典、已知标准名称或 None。", field="mapping")
    if "columns" in value or "constants" in value or any(
        key in value for key in ("eda_unit", "sampling_rate_hz", "time_unit", "clock_domain")
    ):
        for key, item in value.items():
            if key in ("columns", "constants"):
                if not isinstance(item, dict):
                    fail("mapping_type", f"mapping.{key} 必须是字典。", field=f"mapping.{key}")
                merged = dict(result.get(key, {}))
                merged.update(item)
                result[key] = merged
            else:
                result[key] = item
        return result
    # 允许 GUI 直接传“标准字段 -> 来源列”的简写字典。
    columns = dict(result.get("columns", {}))
    columns.update(value)
    result["columns"] = columns
    return result


def _sections(options: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(options, dict):
        return {}, {}
    input_options = options.get("input", {})
    mapping_options = options.get("mapping", {})
    return (
        dict(input_options) if isinstance(input_options, dict) else {},
        dict(mapping_options) if isinstance(mapping_options, dict) else {},
    )


def _read_bytes_once(path: Path) -> tuple[bytes, str]:
    """读取单份原始字节并以前后 stat 检查避免解析时源文件变动。"""
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            fail("not_regular_file", f"输入不是普通文件：{path}", field="file_path")
        with path.open("rb") as stream:
            payload = stream.read()
        after = path.stat()
    except OSError as error:
        fail("file_read", f"无法读取输入文件 {path}：{error}", field="file_path")
    signature_before = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, getattr(before, "st_ino", None))
    signature_after = (after.st_size, after.st_mtime_ns, after.st_ctime_ns, getattr(after, "st_ino", None))
    if signature_before != signature_after or len(payload) != after.st_size:
        fail("source_changed", f"读取期间源文件发生变化：{path}", field="file_path")
    return payload, hashlib.sha256(payload).hexdigest()


def _choose_encoding(payload: bytes, preferred: Any = None) -> str:
    if preferred:
        candidates = [str(preferred)]
        preferred_name = str(preferred).casefold().replace("_", "-")
        if preferred_name in ("utf-8", "utf8", "utf-8-sig", "utf8-sig"):
            candidates.extend(["utf-8-sig", "utf-8", "gb18030"])
    elif payload.startswith(b"\xef\xbb\xbf"):
        candidates = ["utf-8-sig"]
    else:
        candidates = ["utf-8-sig", "utf-8", "gb18030"]
    for candidate in dict.fromkeys(candidates):
        try:
            payload.decode(candidate, errors="strict")
            return candidate
        except (LookupError, UnicodeDecodeError):
            continue
    fail("encoding", "文件编码无法识别；请在 input.encoding 中明确指定。", field="encoding")


def _sniff_delimiter(text: str, preferred: Any = None) -> str:
    if preferred not in (None, "", "auto"):
        delimiter = str(preferred)
        if len(delimiter) != 1:
            fail("delimiter", "CSV分隔符必须是一个字符。", field="delimiter")
        return delimiter
    sample = "\n".join(text.splitlines()[:40])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _csv_rows(payload: bytes, *, encoding: Any = None, delimiter: Any = None) -> tuple[list[str], list[list[str]], str, str]:
    selected_encoding = _choose_encoding(payload, encoding)
    text = payload.decode(selected_encoding, errors="strict")
    selected_delimiter = _sniff_delimiter(text, delimiter)
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=selected_delimiter, strict=True)
    try:
        header = next(reader)
    except StopIteration:
        fail("empty_csv", "CSV文件为空，没有表头。", field="file_path")
    header = [name.strip() for name in header]
    if not header or any(not name for name in header):
        fail("empty_header", "CSV表头存在空字段名。", field="header")
    if len(header) != len(set(header)):
        duplicates = sorted({name for name in header if header.count(name) > 1})
        fail("duplicate_header", f"CSV表头字段重复：{duplicates}", field="header")
    rows: list[list[str]] = []
    for row_number, row in enumerate(reader, start=2):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) != len(header):
            fail(
                "csv_row_width",
                f"第 {row_number} 行有 {len(row)} 列，表头有 {len(header)} 列。",
                field="row",
            )
        rows.append(row)
    return header, rows, selected_encoding, selected_delimiter


def _table_from_bytes(payload: bytes, *, encoding: Any = None, delimiter: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    header, rows, selected_encoding, selected_delimiter = _csv_rows(
        payload, encoding=encoding, delimiter=delimiter
    )
    frame = pd.DataFrame(rows, columns=header, dtype=object)
    return frame, {"encoding": selected_encoding, "delimiter": selected_delimiter, "rows": len(frame)}


def preview_csv(path: str | os.PathLike[str], max_rows: int = 50, options: Any = None) -> dict[str, Any]:
    """读取原始字节一次，返回总数据行数与有上限的预览，不加载完整表到 pandas。"""
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 0:
        fail("preview_limit", "max_rows 必须是非负整数。", field="max_rows")
    max_rows = min(max_rows, 500)
    input_options, _ = _sections(options)
    csv_path = Path(path).expanduser()
    payload, digest = _read_bytes_once(csv_path)
    encoding = _choose_encoding(payload, input_options.get("encoding"))
    text = payload.decode(encoding, errors="strict")
    delimiter = _sniff_delimiter(text, input_options.get("delimiter"))
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    try:
        header = [name.strip() for name in next(reader)]
    except StopIteration:
        fail("empty_csv", "CSV文件为空，没有表头。", field="file_path")
    if not header or any(not name for name in header):
        fail("empty_header", "CSV表头存在空字段名。", field="header")
    if len(header) != len(set(header)):
        fail("duplicate_header", "CSV表头字段重复。", field="header")
    preview_rows: list[list[str]] = []
    total_rows = 0
    for row_number, row in enumerate(reader, start=2):
        if not row or (len(row) == 1 and not row[0].strip()):
            continue
        if len(row) != len(header):
            fail("csv_row_width", f"第 {row_number} 行列数与表头不一致。", field="row")
        total_rows += 1
        if len(preview_rows) < max_rows:
            preview_rows.append(row)
    preview = pd.DataFrame(preview_rows, columns=header, dtype=object)
    return {
        "total_rows": total_rows,
        "preview": preview,
        "source_sha256": digest,
        "encoding": encoding,
        "delimiter": delimiter,
    }


def _text_values(series: pd.Series, missing_tokens: set[str] | None = None) -> pd.Series:
    result = series.astype("string").str.strip()
    tokens = {""} if missing_tokens is None else missing_tokens
    return result.mask(result.isna() | result.isin(tokens), pd.NA)


def _decimal_time(values: pd.Series, *, missing_tokens: set[str], origin: Any, time_unit: Any) -> tuple[np.ndarray, str, bool]:
    if time_unit not in _TIME_FACTORS:
        fail("time_unit", "time_unit 必须明确为 s、ms 或 us。", field="time_unit")
    tokens = values.astype("string").str.strip()
    missing = tokens.isna() | tokens.isin(missing_tokens)
    if missing.any():
        bad_rows = (np.flatnonzero(missing.to_numpy())[:5] + 2).tolist()
        fail("missing_time", f"时间列存在缺失值（CSV数据行）：{bad_rows}。不自动排序或删除时间。", field="time_s")
    parsed: list[Decimal] = []
    for row_number, token in enumerate(tokens, start=2):
        try:
            value = Decimal(str(token))
        except (InvalidOperation, ValueError):
            fail("invalid_time", f"第 {row_number} 行时间不是数值。", field="time_s")
        if not value.is_finite():
            fail("invalid_time", f"第 {row_number} 行时间不是有限值。", field="time_s")
        parsed.append(value)
    origin_was_declared = origin is not None and str(origin).strip() != ""
    try:
        origin_value = Decimal(str(origin)) if origin_was_declared else parsed[0]
    except (InvalidOperation, ValueError):
        fail("time_origin", "time_origin 必须是与来源时间同单位的数值。", field="time_origin")
    factor = Decimal(str(_TIME_FACTORS[time_unit]))
    seconds = np.asarray([float((value - origin_value) * factor) for value in parsed], dtype=np.float64)
    if not np.all(np.isfinite(seconds)):
        fail("invalid_time", "时间换算后超出有限浮点范围。", field="time_s")
    return seconds, str(origin_value), not origin_was_declared


def _parse_number(series: pd.Series, *, missing_tokens: set[str], field: str) -> np.ndarray:
    tokens = series.astype("string").str.strip()
    missing = tokens.isna() | tokens.isin(missing_tokens)
    parsed = pd.to_numeric(tokens.mask(missing, pd.NA), errors="coerce")
    invalid = parsed.isna() & ~missing
    if invalid.any():
        rows = (np.flatnonzero(invalid.to_numpy())[:5] + 2).tolist()
        fail("numeric_value", f"字段 {field} 含非数值内容（CSV数据行）：{rows}。", field=field)
    values = parsed.to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(values).any():
        fail("nonfinite_signal", f"字段 {field} 包含正无穷或负无穷。", field=field)
    return values


def _canonical_unit(value: Any) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    raw = str(value).strip()
    key = raw.casefold().replace("μ", "μ").replace("µ", "µ")
    # Preserve case-sensitive SI prefixes before case-folding aliases.
    exact = {"uS": "uS", "µS": "uS", "μS": "uS", "mS": "mS", "S": "S",
             "ohm": "ohm", "Ω": "ohm", "kohm": "kohm", "kΩ": "kohm",
             "Mohm": "Mohm", "MΩ": "Mohm"}
    if raw in exact:
        return exact[raw]
    return _UNIT_ALIASES.get(key)


def _convert_conductance(values: np.ndarray, unit: str, *, field: str = "eda_uS") -> np.ndarray:
    if unit in _CONDUCTANCE_TO_US:
        return values * _CONDUCTANCE_TO_US[unit]
    resistance_factor = {"ohm": 1.0, "kohm": 1e3, "Mohm": 1e6}.get(unit)
    if resistance_factor is None:
        fail("eda_unit", f"不支持的 EDA 来源单位：{unit}。", field=field)
    valid = np.isfinite(values)
    if np.any(values[valid] <= 0):
        fail("resistance_nonpositive", "电阻换算只接受正值；零和负电阻不是有效缺失值。", field=field)
    converted = np.full(values.shape, np.nan, dtype=np.float64)
    converted[valid] = 1e6 / (resistance_factor * values[valid])
    return converted


def _stable_mapping_revision(mapping: dict[str, Any]) -> str:
    serializable = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serializable.encode("utf-8")).hexdigest()


def _path_from(value: Any, *, base_directory: Path | None = None) -> Path:
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute() and base_directory is not None:
        candidate = base_directory / candidate
    return candidate.resolve()


def _file_metadata_values(
    frame: pd.DataFrame,
    *,
    path: Path,
    digest: str,
    mapping: dict[str, Any],
    mapping_revision: str,
    manifest_defaults: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    columns = dict(mapping.get("columns", {}))
    constants = dict(mapping.get("constants", {}))
    defaults = dict(manifest_defaults or {})
    missing_tokens = set(str(x) for x in mapping.get("missing_tokens", ("", "NA", "NaN")))
    input_options = mapping.get("_input_options", {})

    canonical_fields = (
        "recording_id", "subject_id", "session_id", "channel", "time_s", "eda_uS",
        "eda_unit", "clock_domain", "time_origin", "timezone",
    )
    resolved_columns: dict[str, str] = {}
    for canonical in canonical_fields:
        source = columns.get(canonical, canonical if canonical in frame.columns else None)
        if source is not None:
            if source not in frame.columns:
                fail("mapped_column_missing", f"映射字段 {canonical} 指向不存在的来源列 {source!r}。", field=canonical)
            resolved_columns[canonical] = str(source)
    sources = list(resolved_columns.values())
    if len(sources) != len(set(sources)):
        fail("mapping_collision", "同一个来源列被映射到多个标准字段。", field="mapping.columns")
    if "eda_uS" not in resolved_columns:
        fail("signal_column_missing", "必须映射电导幅值列；默认列名为 eda_uS。", field="eda_uS")

    rows = frame.copy(deep=True)
    for canonical in ("recording_id", "subject_id", "session_id", "channel"):
        source = resolved_columns.get(canonical)
        if source is None:
            rows[canonical] = pd.NA
        else:
            # “NA” 可能是合法字符串 ID；只把空字符串作为身份字段缺失。
            rows[canonical] = _text_values(rows[source], {""})

    # Manifest、映射常量可以补 ID，但与文件内已存在的身份字段冲突时拒绝覆盖。
    id_generated: dict[str, bool] = {}
    for field in ("recording_id", "subject_id", "session_id", "channel"):
        default = defaults.get(field, constants.get(field, pd.NA))
        supplied = not (default is None or pd.isna(default) or not str(default).strip())
        if supplied and field == "channel":
            # manifest.channel 是多通道筛选条件；原始通道列保持逐行值，稍后再筛选。
            if rows[field].isna().all():
                rows[field] = str(default).strip()
                id_generated[field] = False
            else:
                id_generated[field] = False
        elif supplied:
            value = str(default).strip()
            current = rows[field]
            conflict = current.notna() & (current.astype("string") != value)
            if conflict.any():
                fail("identity_conflict", f"{field} 来源列与 manifest/映射常量不一致。", field=field)
            rows[field] = current.fillna(value).astype("string")
            id_generated[field] = False
        else:
            id_generated[field] = False
    for field, fallback, is_generated in (
        ("recording_id", path.stem, True),
        ("subject_id", "unknown", True),
        ("session_id", "unknown", True),
        ("channel", "unknown", True),
    ):
        if rows[field].isna().any():
            id_generated[field] = True
            rows[field] = rows[field].fillna(str(fallback)).astype("string")

    unit_column = resolved_columns.get("eda_unit")
    declared_unit = defaults.get("eda_unit", mapping.get("eda_unit", constants.get("eda_unit")))
    amplitude_source = resolved_columns["eda_uS"]
    if declared_unit is None and unit_column is None and amplitude_source != "eda_uS":
        fail("eda_unit_required", "自定义幅值来源列必须明确声明 eda_unit。", field="eda_unit")
    if amplitude_source == "eda_uS" and declared_unit is not None and _canonical_unit(declared_unit) != "uS":
        fail("unit_conflict", "标准来源列 eda_uS 的单位固定为 uS，不可声明为其它单位。", field="eda_unit")
    values = _parse_number(rows[amplitude_source], missing_tokens=missing_tokens, field="eda_uS")
    if unit_column is not None:
        source_units = _text_values(rows[unit_column], missing_tokens)
    elif declared_unit is not None:
        source_units = pd.Series([str(declared_unit)] * len(rows), dtype="string")
    else:
        source_units = pd.Series(["uS"] * len(rows), dtype="string")
    if declared_unit is not None:
        declared_canonical = _canonical_unit(declared_unit)
        if declared_canonical is None:
            fail("eda_unit", f"不支持的 EDA 来源单位：{declared_unit!r}。", field="eda_unit")
        unit_conflict = source_units.notna() & source_units.map(_canonical_unit).ne(declared_canonical)
        if unit_conflict.any():
            fail("unit_conflict", "逐行单位列与 manifest/mapping 声明的单位冲突。", field="eda_unit")
        source_units = source_units.fillna(str(declared_unit))
    if source_units.isna().any():
        fail("eda_unit_required", "幅值来源单位缺失，必须在 mapping 或 manifest 中声明。", field="eda_unit")
    canonical_units = source_units.map(_canonical_unit)
    if canonical_units.isna().any():
        bad = sorted(set(source_units[canonical_units.isna()].astype(str)))
        fail("eda_unit", f"不支持的 EDA 来源单位：{bad}。", field="eda_unit")
    converted = np.full(values.shape, np.nan, dtype=np.float64)
    for unit in sorted(set(canonical_units.astype(str))):
        selected = canonical_units.to_numpy() == unit
        converted[selected] = _convert_conductance(values[selected], unit)
    rows["eda_uS"] = converted
    rows["source_unit"] = source_units.astype("string")

    time_column = resolved_columns.get("time_s")
    time_unit = defaults.get("time_unit", mapping.get("time_unit"))
    if time_unit is None:
        if time_column is not None:
            time_unit = "s"
        else:
            fail("time_unit_required", "缺少时间列时必须明确声明 time_unit（s、ms 或 us）。", field="time_unit")
    time_unit = str(time_unit)
    if time_unit not in _TIME_FACTORS:
        fail("time_unit", "time_unit 必须明确为 s、ms 或 us。", field="time_unit")
    declared_rate = defaults.get("sampling_rate_hz", mapping.get("sampling_rate_hz", constants.get("sampling_rate_hz")))
    time_origin_default = defaults.get("time_origin", constants.get("time_origin"))
    rows["_time_origin"] = ""
    rows["_time_origin_generated"] = False
    if time_column is None:
        if declared_rate is None or pd.isna(declared_rate):
            fail("sampling_rate_required", "缺少时间列时必须声明 sampling_rate_hz。", field="sampling_rate_hz")
        try:
            declared_rate = float(declared_rate)
        except (TypeError, ValueError):
            fail("sampling_rate", "sampling_rate_hz 必须是正数。", field="sampling_rate_hz")
        if not math.isfinite(declared_rate) or declared_rate <= 0:
            fail("sampling_rate", "sampling_rate_hz 必须是有限正数。", field="sampling_rate_hz")
        rows["source_time"] = pd.NA
        # 每条记录独立生成时间轴，避免同一 CSV 中各 recording 的采样序列互相偏移。
        rows["time_s"] = np.nan
        for _, indices in rows.groupby(["recording_id", "channel"], sort=False, dropna=False).groups.items():
            indices = list(indices)
            rows.loc[indices, "time_s"] = np.arange(len(indices), dtype=np.float64) / declared_rate
            rows.loc[indices, "_time_origin"] = str(time_origin_default if time_origin_default is not None else "0")
            rows.loc[indices, "_time_origin_generated"] = time_origin_default is None
    else:
        rows["source_time"] = rows[time_column].astype("string")
        rows["time_s"] = np.nan
        for _, indices in rows.groupby(["recording_id", "channel"], sort=False, dropna=False).groups.items():
            indices = list(indices)
            seconds, origin_value, origin_generated = _decimal_time(
                rows.loc[indices, "source_time"], missing_tokens=missing_tokens,
                origin=time_origin_default, time_unit=time_unit,
            )
            rows.loc[indices, "time_s"] = seconds
            rows.loc[indices, "_time_origin"] = origin_value
            rows.loc[indices, "_time_origin_generated"] = origin_generated

    clock_column = resolved_columns.get("clock_domain")
    clock_default = defaults.get("clock_domain", mapping.get("clock_domain", constants.get("clock_domain", "recording_local")))
    if clock_column is None:
        rows["clock_domain"] = str(clock_default)
    else:
        rows["clock_domain"] = _text_values(rows[clock_column], missing_tokens)
        missing_clock = rows["clock_domain"].isna()
        if missing_clock.any():
            rows.loc[missing_clock, "clock_domain"] = str(clock_default)
        expected_clock = defaults.get("clock_domain", mapping.get("clock_domain", constants.get("clock_domain")))
        if expected_clock is not None:
            conflicts = rows["clock_domain"].astype("string") != str(expected_clock)
            if conflicts.any():
                fail("clock_conflict", "文件内 clock_domain 与 manifest 声明冲突。", field="clock_domain")

    timezone_column = resolved_columns.get("timezone")
    if timezone_column is not None:
        rows["timezone"] = _text_values(rows[timezone_column], missing_tokens)

    mapped_sources = set(resolved_columns.values())
    extras = [column for column in frame.columns if column not in mapped_sources]
    # 留存所有未解释来源列，不将其猜测性地纳入分析字段。
    rows["sample_index"] = np.arange(len(rows), dtype=np.int64)
    rows["extras"] = [dict(zip(extras, values)) for values in frame[extras].itertuples(index=False, name=None)]

    group_records: list[dict[str, Any]] = []
    group_columns = ["recording_id", "channel"]
    for (recording_id, channel), positions in rows.groupby(group_columns, sort=False, dropna=False).groups.items():
        positions = np.asarray(list(positions), dtype=np.int64)
        group = rows.iloc[positions]
        if not len(group):
            continue
        for field in ("subject_id", "session_id", "clock_domain", "source_unit"):
            unique = group[field].dropna().astype(str).unique().tolist()
            if len(unique) != 1:
                fail("record_metadata_conflict", f"同一 recording_id/channel 内的 {field} 不唯一。", field=field)
        times = group["time_s"].to_numpy(dtype=np.float64)
        differences = np.diff(times)
        if np.any(~np.isfinite(times)) or np.any(differences <= 0):
            bad_index = int(np.flatnonzero(differences <= 0)[0] + 2) if np.any(differences <= 0) else None
            fail(
                "invalid_timebase",
                f"recording_id={recording_id!r}, channel={channel!r} 的时间必须有限且严格递增；重复、倒序或缺失值需先纠正，不能静默排序/删除。第一个异常约在数据行 {bad_index}。",
                record_id=str(recording_id), field="time_s",
            )
        if len(differences):
            median_dt = float(np.median(differences))
            fs_estimated = 1.0 / median_dt if median_dt > 0 else None
            jitter = float(np.median(np.abs(differences - median_dt)) / median_dt) if median_dt > 0 else None
        else:
            fs_estimated = None
            jitter = None
        group_declared_rate_values = []
        if declared_rate is not None:
            group_declared_rate_values.append(float(declared_rate))
        if group_declared_rate_values and fs_estimated is not None:
            group_declared = group_declared_rate_values[0]
            relative_rate_difference = abs(group_declared - fs_estimated) / fs_estimated
        else:
            group_declared = None
            relative_rate_difference = None
        fs_effective = fs_estimated if fs_estimated is not None else group_declared
        units = group["source_unit"].astype(str).unique().tolist()
        timezone_values = group["timezone"].dropna().astype(str).unique().tolist() if "timezone" in group else []
        if len(timezone_values) > 1:
            fail("timezone_conflict", "同一 recording/channel 的 timezone 不唯一。", field="timezone")
        record_generated = {
            "recording_id": bool(id_generated["recording_id"]),
            "subject_id": bool(id_generated["subject_id"]),
            "session_id": bool(id_generated["session_id"]),
            "channel": bool(id_generated["channel"]),
        }
        group_origins = group["_time_origin"].astype(str).unique().tolist()
        group_origin_flags = group["_time_origin_generated"].astype(bool).unique().tolist()
        if len(group_origins) != 1 or len(group_origin_flags) != 1:
            fail("time_origin_conflict", "同一 recording/channel 的 time_origin 不唯一。", field="time_origin")
        record_metadata = {
            "recording_id": str(recording_id),
            "subject_id": str(group["subject_id"].iloc[0]),
            "session_id": str(group["session_id"].iloc[0]),
            "channel": str(channel),
            "source_path": str(path),
            "source_sha256": digest,
            "source_unit": units[0] if len(units) == 1 else units,
            "unit": "uS",
            "fs_hz": fs_effective,
            "fs_declared": group_declared,
            "fs_estimated": fs_estimated,
            "time_origin": group_origins[0],
            "time_origin_generated": group_origin_flags[0],
            "time_unit": time_unit,
            "timezone": timezone_values[0] if timezone_values else None,
            "clock_domain": str(group["clock_domain"].iloc[0]),
            "mapping_revision": mapping_revision,
            "generated": record_generated,
            "time_jitter_mad_fraction": jitter,
        }
        if relative_rate_difference is not None and relative_rate_difference > 1e-6:
            record_metadata["rate_mismatch_relative"] = relative_rate_difference
        group_records.append({"positions": positions, "metadata": record_metadata})

    normalized_columns = [
        "recording_id", "subject_id", "session_id", "channel", "time_s", "eda_uS", "sample_index"
    ]
    normalized = rows[normalized_columns].copy()
    reserved_columns = set(normalized_columns) | {"extras", "source_unit", "source_time", "clock_domain", "_time_origin", "_time_origin_generated"}
    for extra_column in extras:
        if extra_column not in reserved_columns:
            normalized[extra_column] = frame[extra_column].to_numpy(copy=True)
    # 留下逐行原始坐标字符串；time_s 是相对秒，不能替代高精度来源时间。
    normalized["source_time"] = rows["source_time"].copy()
    if "timezone" in rows:
        normalized["timezone"] = rows["timezone"].copy()
    normalized["extras"] = rows["extras"].copy()
    file_info = {
        "source_path": str(path),
        "source_sha256": digest,
        "encoding": input_options.get("_actual_encoding"),
        "delimiter": input_options.get("_actual_delimiter"),
        "rows": int(len(normalized)),
        "mapping_revision": mapping_revision,
    }
    return normalized, group_records, file_info


def _read_signal_payload(
    payload: bytes,
    digest: str,
    path: Path,
    mapping: dict[str, Any],
    options: Any,
    manifest_defaults: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    input_options, _ = _sections(options)
    encoding = input_options.get("encoding")
    delimiter = input_options.get("delimiter")
    frame, parse_info = _table_from_bytes(payload, encoding=encoding, delimiter=delimiter)
    mapping_for_file = dict(mapping)
    mapping_for_file["_input_options"] = {**input_options, "_actual_encoding": parse_info["encoding"], "_actual_delimiter": parse_info["delimiter"]}
    mapping_for_file["missing_tokens"] = input_options.get("missing_tokens", mapping.get("missing_tokens", ["", "NA", "NaN"]))
    samples, groups, file_info = _file_metadata_values(
        frame,
        path=path,
        digest=digest,
        mapping=mapping_for_file,
        mapping_revision=_stable_mapping_revision(mapping),
        manifest_defaults=manifest_defaults,
    )
    file_info.update(parse_info)
    metadata = {
        "source_path": str(path),
        "source_sha256": digest,
        "unit": "uS",
        "mapping_revision": file_info["mapping_revision"],
        "groups": [group["metadata"] for group in groups],
    }
    if len(groups) == 1:
        metadata.update(groups[0]["metadata"])
    report = {
        "valid": True,
        "source_path": str(path),
        "source_sha256": digest,
        "rows": int(len(samples)),
        "records": len(groups),
        "encoding": parse_info["encoding"],
        "delimiter": parse_info["delimiter"],
        "diagnostics": [],
        "warnings": [],
    }
    for group in groups:
        item = group["metadata"]
        if item.get("rate_mismatch_relative") is not None:
            report["warnings"].append({
                "code": "sampling_rate_disagreement",
                "recording_id": item["recording_id"],
                "channel": item["channel"],
                "fs_declared": item["fs_declared"],
                "fs_estimated": item["fs_estimated"],
                "relative_difference": item["rate_mismatch_relative"],
                "message": "声明采样率与时间戳稳健估计不一致；保留两者供后续核对。",
            })
    return samples, metadata, report


def read_signal_csv(
    path: str | os.PathLike[str],
    mapping: Any = None,
    options: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """读取并标准化信号 CSV；返回样本表、来源元数据与可展示的导入诊断。"""
    csv_path = Path(path).expanduser().resolve()
    payload, digest = _read_bytes_once(csv_path)
    resolved_mapping = _mapping(mapping, options)
    input_options, _ = _sections(options)
    resolved_mapping["missing_tokens"] = input_options.get(
        "missing_tokens", resolved_mapping.get("missing_tokens", ["", "NA", "NaN"])
    )
    samples, metadata, report = _read_signal_payload(
        payload, digest, csv_path, resolved_mapping, options
    )
    return samples, metadata, report


def _read_table_source(value: Any, *, key: str, options: Any, base_directory: Path | None) -> pd.DataFrame | None:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, dict):
        return pd.DataFrame([value])
    if not isinstance(value, (str, os.PathLike)):
        fail("table_source", f"{key} 必须是路径、表格或行记录列表。", field=key)
    input_options, _ = _sections(options)
    table_path = _path_from(value, base_directory=base_directory)
    payload, digest = _read_bytes_once(table_path)
    frame, _ = _table_from_bytes(
        payload,
        encoding=input_options.get("encoding"),
        delimiter=input_options.get("delimiter"),
    )
    frame.attrs["source_path"] = str(table_path)
    frame.attrs["source_sha256"] = digest
    return frame


def _manifest_rows(manifest: Any, input_options: dict[str, Any], options: Any, base_directory: Path | None) -> tuple[list[dict[str, Any]], Path | None, pd.DataFrame]:
    source_path: Path | None = None
    if isinstance(manifest, (str, os.PathLike)):
        source_path = _path_from(manifest, base_directory=base_directory)
        table = _read_table_source(source_path, key="manifest", options=options, base_directory=None)
        rows = table.to_dict(orient="records") if table is not None else []
        return rows, source_path.parent, table if table is not None else pd.DataFrame()
    if isinstance(manifest, pd.DataFrame):
        table = manifest.copy(deep=True)
        return table.to_dict(orient="records"), base_directory, table
    if isinstance(manifest, dict):
        table = pd.DataFrame([dict(manifest)])
        return [dict(manifest)], base_directory, table
    if isinstance(manifest, list):
        if not manifest:
            return [], base_directory, pd.DataFrame(columns=["file_path"])
        if all(isinstance(item, str) for item in manifest):
            rows = [{"file_path": item} for item in manifest]
            return rows, base_directory, pd.DataFrame(rows)
        if all(isinstance(item, dict) for item in manifest):
            rows = [dict(item) for item in manifest]
            return rows, base_directory, pd.DataFrame(rows)
    fail("manifest_type", "manifest 必须是 CSV 路径、DataFrame、字典行列表或 None。", field="manifest")


def _constant_conflict(table: pd.DataFrame, key: str) -> dict[str, Any] | None:
    if table is None or key not in table.columns:
        return None
    values = table[key].dropna().astype(str).unique().tolist()
    if len(values) > 1:
        return {"code": "duplicate_subject_conflict", "subject_id": values[0], "field": key,
                "message": f"participant 表中的 {key} 有相互冲突的值。"}
    return None


def _identity_index(records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]], set[str]]:
    by_recording: dict[str, dict[str, Any]] = {}
    pairs: set[tuple[str, str]] = set()
    subjects: set[str] = set()
    for record in records:
        meta = record["metadata"]
        rid = str(meta["recording_id"])
        channel = str(meta["channel"])
        if rid in by_recording:
            old = by_recording[rid]
            if old["subject_id"] != meta["subject_id"] or old["session_id"] != meta["session_id"]:
                fail("recording_identity_conflict", f"recording_id={rid!r} 对应多个 subject/session。", record_id=rid)
        else:
            by_recording[rid] = meta
        pairs.add((rid, channel))
        subjects.add(str(meta["subject_id"]))
    return by_recording, pairs, subjects


def _cross_validate_tables(records: list[dict[str, Any]], tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """检验跨表外键、唯一身份和时间坐标；缺失的可选伴随表不推断。"""
    errors: list[dict[str, Any]] = []
    by_recording, recording_channels, signal_subjects = _identity_index(records)
    participants = tables.get("participants")
    participant_subjects = set()
    if participants is not None and "subject_id" in participants:
        if participants["subject_id"].duplicated(keep=False).any():
            duplicate_ids = participants.loc[participants["subject_id"].duplicated(keep=False), "subject_id"].astype(str).unique().tolist()
            for subject_id in duplicate_ids:
                group = participants[participants["subject_id"].astype(str) == subject_id]
                value_columns = [column for column in group.columns if column != "subject_id"]
                if len(group[value_columns].drop_duplicates()) > 1:
                    errors.append({"code": "duplicate_subject_conflict", "table": "participants", "subject_id": subject_id,
                                   "message": "同一 subject_id 的参与者属性互相冲突。"})
                else:
                    errors.append({"code": "duplicate_subject_id", "table": "participants", "subject_id": subject_id,
                                   "message": "participants.subject_id 必须唯一。"})
        participant_subjects = set(participants["subject_id"].dropna().astype(str))
        for subject_id in signal_subjects - {"unknown"}:
            if participant_subjects and subject_id not in participant_subjects:
                errors.append({"code": "participant_foreign_key", "table": "participants", "subject_id": subject_id,
                               "message": "信号记录引用了 participants 表中不存在的 subject_id。"})

    def validate_scope(table_name: str, *, allow_broadcast: bool = True) -> None:
        table = tables.get(table_name)
        if table is None:
            return
        for row_index, row in table.iterrows():
            rid_value = row.get("recording_id")
            if pd.isna(rid_value) or str(rid_value).strip() not in by_recording:
                errors.append({"code": "recording_foreign_key", "table": table_name, "row": int(row_index) + 2,
                               "recording_id": None if pd.isna(rid_value) else str(rid_value),
                               "message": "表中 recording_id 不存在于已导入信号。"})
                continue
            rid = str(rid_value).strip()
            channels = {channel for recording, channel in recording_channels if recording == rid}
            channel_value = row.get("channel")
            channel_missing = _is_missing(channel_value) or not str(channel_value).strip()
            channel = "*" if channel_missing else str(channel_value).strip()
            if channel_missing and len(channels) > 1:
                errors.append({"code": "channel_scope_required", "table": table_name, "row": int(row_index) + 2,
                               "message": "多通道记录必须指定 channel 或显式使用 * 表示广播。"})
            elif channel == "*":
                if len(channels) > 1 and not allow_broadcast:
                    errors.append({"code": "channel_scope_required", "table": table_name, "row": int(row_index) + 2,
                                   "message": "多通道记录必须指定 channel 或显式使用 * 表示广播。"})
            elif (rid, channel) not in recording_channels:
                errors.append({"code": "channel_foreign_key", "table": table_name, "row": int(row_index) + 2,
                               "channel": channel, "message": "channel 不存在于相应 recording_id。"})
            record_meta = by_recording[rid]
            if "clock_domain" in row.index and not pd.isna(row.get("clock_domain")):
                if str(row.get("clock_domain")).strip() != str(record_meta.get("clock_domain")):
                    errors.append({"code": "clock_domain_mismatch", "table": table_name, "row": int(row_index) + 2,
                                   "recording_id": rid, "message": "时间型伴随表 clock_domain 与信号记录不一致。"})

    for name in ("events", "conditions", "manual_qc", "sync_anchors"):
        validate_scope(name)

    conditions = tables.get("conditions")
    if conditions is not None and "condition" in conditions.columns:
        conditions_by_record: dict[str, set[str]] = defaultdict(set)
        for _, row in conditions.iterrows():
            if not pd.isna(row.get("recording_id")) and not pd.isna(row.get("condition")):
                conditions_by_record[str(row["recording_id"])].add(str(row["condition"]))
        events = tables.get("events")
        if events is not None and "condition" in events.columns:
            for row_index, row in events.iterrows():
                rid, condition = row.get("recording_id"), row.get("condition")
                if not pd.isna(rid) and not pd.isna(condition) and str(condition).strip():
                    if str(condition) not in conditions_by_record.get(str(rid), set()):
                        errors.append({"code": "condition_foreign_key", "table": "events", "row": int(row_index) + 2,
                                       "recording_id": str(rid), "condition": str(condition),
                                       "message": "event 引用的 condition 不存在于同一 recording 的 conditions 表。"})

    membership = tables.get("group_membership")
    if membership is not None:
        for row_index, row in membership.iterrows():
            subject = row.get("subject_id")
            session = row.get("session_id")
            rid = row.get("recording_id")
            if not pd.isna(rid):
                meta = by_recording.get(str(rid))
                if meta is None:
                    errors.append({"code": "recording_foreign_key", "table": "group_membership", "row": int(row_index) + 2,
                                   "message": "group_membership 引用了未知 recording_id。"})
                    continue
                if not pd.isna(subject) and str(subject) != str(meta.get("subject_id")):
                    errors.append({"code": "membership_subject_conflict", "table": "group_membership", "row": int(row_index) + 2,
                                   "message": "group_membership.subject_id 与 recording 元数据不一致。"})
                if not pd.isna(session) and str(session) != str(meta.get("session_id")):
                    errors.append({"code": "membership_session_conflict", "table": "group_membership", "row": int(row_index) + 2,
                                   "message": "group_membership.session_id 与 recording 元数据不一致。"})
            if not pd.isna(subject) and participant_subjects and str(subject) not in participant_subjects:
                errors.append({"code": "participant_foreign_key", "table": "group_membership", "row": int(row_index) + 2,
                               "message": "group_membership 引用了未知 subject_id。"})

    anchors = tables.get("sync_anchors")
    if anchors is not None:
        for row_index, row in anchors.iterrows():
            if "reference_clock_domain" not in row.index or pd.isna(row.get("reference_clock_domain")) or not str(row.get("reference_clock_domain")).strip():
                errors.append({"code": "reference_clock_required", "table": "sync_anchors", "row": int(row_index) + 2,
                               "message": "同步锚点必须声明 reference_clock_domain；不能从秒单位推断时钟一致。"})

    manifest = tables.get("input_manifest")
    if manifest is not None:
        known_sources = {str(record["metadata"].get("source_path", "")) for record in records}
        for row_index, row in manifest.iterrows():
            rid = row.get("recording_id")
            if not _is_missing(rid) and str(rid) not in by_recording:
                errors.append({"code": "manifest_recording_foreign_key", "table": "input_manifest", "row": int(row_index) + 2,
                               "recording_id": str(rid), "message": "input_manifest.recording_id 未导入到信号记录。"})
            file_path = row.get("file_path")
            if not _is_missing(file_path) and str(file_path).strip():
                resolved = str(Path(str(file_path)).expanduser().resolve())
                if len(known_sources) == 1 and resolved not in known_sources:
                    # 相对路径可能相对清单目录解析；最终路径由实际 records 元数据校验。
                    source_names = {Path(source).name for source in known_sources}
                    if Path(str(file_path)).name not in source_names:
                        errors.append({"code": "manifest_path_mismatch", "table": "input_manifest", "row": int(row_index) + 2,
                                       "message": "清单路径与实际读取的信号文件不匹配。"})

    manual = tables.get("manual_qc")
    if manual is not None and "action" in manual:
        valid_actions = {"exclude", "restore", "undo", "redo"}
        for row_index, row in manual.iterrows():
            action = str(row.get("action", "")).strip().casefold()
            if action not in valid_actions:
                errors.append({"code": "manual_qc_action", "table": "manual_qc", "row": int(row_index) + 2,
                               "message": f"不支持的 manual_qc action：{action!r}。"})
    return errors


def load_recordings(
    manifest: Any = None,
    mapping: Any = None,
    options: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame], dict[str, Any]]:
    """读取清单中的信号及伴随表，构造 record 并执行跨表引用和时钟校验。"""
    input_options, mapping_options = _sections(options)
    resolved_mapping = _mapping(mapping, options)
    base_value = input_options.get("base_directory")
    base_directory = _path_from(base_value) if base_value else None
    selected_manifest = manifest
    manifest_table_path: Path | None = None
    if selected_manifest is None:
        if input_options.get("manifest") is not None:
            selected_manifest = input_options.get("manifest")
        else:
            selected_manifest = input_options.get("signals", [])
    rows, manifest_base, manifest_frame = _manifest_rows(selected_manifest, input_options, options, base_directory)
    if isinstance(selected_manifest, (str, os.PathLike)):
        manifest_table_path = _path_from(selected_manifest, base_directory=base_directory)
    if base_directory is None:
        base_directory = manifest_base

    table_base = base_directory
    tables: dict[str, pd.DataFrame] = {}
    table_reports: dict[str, dict[str, Any]] = {}
    if len(manifest_frame) or ("file_path" in manifest_frame.columns and isinstance(selected_manifest, (str, os.PathLike))):
        manifest_validation = validate_table("input_manifest", manifest_frame)
        table_reports["input_manifest"] = manifest_validation
        if not manifest_validation["valid"]:
            fail("table_validation", f"input_manifest 表未通过字段校验：{manifest_validation['errors'][:3]}", field="input_manifest")
        tables["input_manifest"] = manifest_frame.copy(deep=True)
    for name in _ASSOCIATED_INPUTS:
        source = input_options.get(name)
        if source is None and name == "manual_qc":
            qc_options = options.get("qc", {}) if isinstance(options, dict) else {}
            source = qc_options.get("annotations") if isinstance(qc_options, dict) else None
        frame = _read_table_source(source, key=name, options=options, base_directory=table_base)
        if frame is None:
            continue
        if frame.empty:
            # 空的可选输入表示“没有该类记录”，但返回带 schema 列的空表供控制器稳定消费。
            field_names = [item["name"] for item in read_template(name, "schema")["fields"]]
            for field_name in field_names:
                if field_name not in frame.columns:
                    frame[field_name] = pd.Series(dtype="object")
        if name == "participants" and "subject_id" in frame and frame["subject_id"].duplicated(keep=False).any():
            repeated = frame.loc[frame["subject_id"].duplicated(keep=False), "subject_id"].astype(str).unique()
            for subject_id in repeated:
                group = frame[frame["subject_id"].astype(str) == subject_id]
                value_columns = [column for column in group.columns if column != "subject_id"]
                if len(group[value_columns].drop_duplicates()) > 1:
                    fail("duplicate_subject_conflict", f"participants.subject_id={subject_id!r} 的属性记录冲突。", field="subject_id")
        validation = validate_table(name, frame)
        table_reports[name] = validation
        if not validation["valid"]:
            fail("table_validation", f"{name} 表未通过字段校验：{validation['errors'][:3]}", field=name)
        tables[name] = frame

    cache: dict[Path, tuple[bytes, str]] = {}
    records: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    seen_manifest_keys: set[tuple[str, str, str]] = set()
    for row_number, item in enumerate(rows, start=2):
        if isinstance(item, str):
            item = {"file_path": item}
        if not isinstance(item, dict):
            fail("manifest_row", f"清单第 {row_number} 行不是对象。", field="manifest")
        file_value = item.get("file_path", item.get("path"))
        if file_value is None or not str(file_value).strip():
            fail("manifest_path", f"清单第 {row_number} 行缺少 file_path。", field="file_path")
        signal_path = _path_from(file_value, base_directory=base_directory)
        if signal_path not in cache:
            cache[signal_path] = _read_bytes_once(signal_path)
        payload, digest = cache[signal_path]

        item_mapping = resolved_mapping
        profile = item.get("mapping_profile")
        if profile is not None:
            profile_name = str(profile).strip()
            if profile_name.casefold() not in ("default", "signals", "standard"):
                configured_name = mapping.get("name") if isinstance(mapping, dict) else None
                if configured_name != profile_name:
                    fail("mapping_profile_missing", f"未提供 mapping_profile={profile_name!r} 的映射内容。", field="mapping_profile")

        defaults = {key: item[key] for key in (
            "recording_id", "subject_id", "session_id", "sampling_rate_hz", "eda_unit",
            "channel", "clock_domain", "time_origin", "time_unit",
        ) if key in item and item[key] is not None and str(item[key]).strip()}
        samples, metadata, report = _read_signal_payload(
            payload, digest, signal_path, item_mapping, options, defaults
        )
        requested_channel = defaults.get("channel")
        if requested_channel is not None and samples["channel"].nunique(dropna=False) > 1:
            samples = samples[samples["channel"].astype(str) == str(requested_channel)].copy()
            if samples.empty:
                fail("manifest_channel", f"manifest 选择的 channel 不存在：{requested_channel!r}。", field="channel")
        file_reports.append(report)
        for (recording_id, channel), group in samples.groupby(["recording_id", "channel"], sort=False, dropna=False):
            manifest_key = (str(signal_path), str(recording_id), str(channel))
            if manifest_key in seen_manifest_keys:
                fail("manifest_duplicate", f"清单重复导入同一文件/recording/channel：{manifest_key}", field="manifest")
            seen_manifest_keys.add(manifest_key)
            group = group.reset_index(drop=True)
            group_metadata_rows = metadata.get("groups", [])
            group_metadata = next((entry for entry in group_metadata_rows
                                   if entry["recording_id"] == str(recording_id) and entry["channel"] == str(channel)), None)
            if group_metadata is None and len(group_metadata_rows) == 1:
                group_metadata = group_metadata_rows[0]
            if group_metadata is None:
                # Channel filtering is applied after parsing; use a single-record parse result as fallback.
                group_metadata = {key: value for key, value in metadata.items() if key != "groups"}
            record_extras = pd.DataFrame(list(group["extras"]))
            record_extras["source_time"] = group["source_time"].to_numpy(copy=True)
            if "timezone" in group:
                record_extras["timezone"] = group["timezone"].to_numpy(copy=True)
            record = make_record(
                group["time_s"].to_numpy(dtype=np.float64),
                group["eda_uS"].to_numpy(dtype=np.float64),
                group_metadata,
                sample_index=group["sample_index"].to_numpy(dtype=np.int64),
                extras=record_extras,
            )
            records.append(record)

    if not records:
        fail("empty_manifest", "清单没有可导入的信号记录。", field="manifest")

    # 表中的空字符串只作为表字段缺失处理，不让 pandas 的默认 NA 规则误改 ID。
    for name, frame in list(tables.items()):
        tables[name] = frame.copy(deep=True)
    relationship_errors = _cross_validate_tables(records, tables)
    if relationship_errors:
        fail("foreign_key", f"伴随表关系校验失败：{relationship_errors[:5]}", field="tables")

    samples_rows = sum(report["rows"] for report in file_reports)
    consumed_inputs: dict[str, str] = {}
    for entry in file_reports:
        consumed_inputs[entry["source_path"]] = entry["source_sha256"]
    for frame in [manifest_frame, *tables.values()]:
        source_path = frame.attrs.get("source_path") if isinstance(frame, pd.DataFrame) else None
        source_sha256 = frame.attrs.get("source_sha256") if isinstance(frame, pd.DataFrame) else None
        if source_path and source_sha256:
            consumed_inputs[str(source_path)] = str(source_sha256)
    report = {
        "valid": True,
        "files": file_reports,
        "recordings": len(records),
        "rows": samples_rows,
        "tables": {name: {"rows": len(frame), **table_reports.get(name, {})} for name, frame in tables.items()},
        "diagnostics": [],
        "warnings": [warning for file_report in file_reports for warning in file_report.get("warnings", [])],
        "source_hashes": {entry["source_path"]: entry["source_sha256"] for entry in file_reports},
        "consumed_inputs": consumed_inputs,
    }
    return records, tables, report


def list_templates() -> list[dict[str, Any]]:
    """列出九类可供 GUI 查看、下载和导入的 CSV 模板。"""
    root = resources.files(_TEMPLATE_PACKAGE)
    index = json.loads(root.joinpath("index.json").read_text(encoding="utf-8"))
    return [dict(item) for item in index["templates"]]


def _template_file(name: str, kind: str) -> Any:
    if name not in _TEMPLATE_NAMES:
        fail("template_name", f"未知模板名称：{name!r}。", field="template")
    suffixes = {"template": f"{name}.csv", "example": f"{name}.example.csv", "schema": f"{name}.schema.json"}
    if kind not in suffixes:
        fail("template_kind", "kind 只能是 template、example 或 schema。", field="kind")
    return resources.files(_TEMPLATE_PACKAGE).joinpath(suffixes[kind])


def get_template_path(name: str, kind: str = "template") -> Any:
    """返回包内模板资源的 Traversable 路径（冻结包也通过 importlib.resources 解析）。"""
    return _template_file(name, kind)


def read_template(name: str, kind: str = "template") -> pd.DataFrame | dict[str, Any]:
    """读取空表头、SYNTHETIC_ONLY 示例或字段 schema。"""
    resource = _template_file(name, kind)
    if kind == "schema":
        return json.loads(resource.read_text(encoding="utf-8"))
    payload = resource.read_bytes()
    frame, _ = _table_from_bytes(payload, encoding="utf-8-sig", delimiter=",")
    return frame


def export_template(name: str, destination: str | os.PathLike[str], kind: str = "template") -> Path:
    """把模板、示例或 schema 复制到用户指定文件，原资源只读。"""
    resource = _template_file(name, kind)
    target = Path(destination).expanduser()
    if target.exists() and target.is_dir():
        target = target / resource.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resource.read_bytes())
    return target.resolve()


def _diagnostic(code: str, table: str, message: str, *, row: int | None = None, column: str | None = None) -> dict[str, Any]:
    return {"code": code, "table": table, "row": row, "column": column, "message": message}


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else False
    except (TypeError, ValueError):
        return False


def validate_table(template_name: str, table: Any, tables: dict[str, Any] | None = None) -> dict[str, Any]:
    """按包内 schema 检查字段、基本类型、唯一约束和可选外键。"""
    schema = read_template(template_name, "schema")
    if isinstance(table, pd.DataFrame):
        frame = table.copy(deep=False)
    elif isinstance(table, list):
        frame = pd.DataFrame(table)
    elif isinstance(table, dict):
        frame = pd.DataFrame([table])
    else:
        return {"valid": False, "errors": [_diagnostic("table_type", template_name, "table 必须是 DataFrame、字典行或字典列表。")], "warnings": []}
    errors: list[dict[str, Any]] = []
    field_schema = {field["name"]: field for field in schema["fields"]}
    for field_name, field in field_schema.items():
        if field.get("required", False) and field_name not in frame.columns:
            errors.append(_diagnostic("required_column", template_name, f"缺少必需字段 {field_name}。", column=field_name))
            continue
        if field_name not in frame.columns:
            continue
        for index, value in frame[field_name].items():
            missing = _is_missing(value) or (isinstance(value, str) and not value.strip())
            if missing and field.get("missing", "allowed") == "forbidden":
                errors.append(_diagnostic("missing_value", template_name, f"必填字段 {field_name} 缺失。", row=int(index) + 2, column=field_name))
                continue
            if missing:
                continue
            value_type = field.get("type")
            if value_type in ("number", "integer"):
                try:
                    numeric = float(value)
                    valid = math.isfinite(numeric) and (value_type != "integer" or numeric.is_integer())
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    errors.append(_diagnostic("value_type", template_name, f"字段 {field_name} 需要有限的 {value_type} 值。", row=int(index) + 2, column=field_name))
            elif value_type == "boolean" and not isinstance(value, (bool, np.bool_)):
                if str(value).strip().casefold() not in ("true", "false", "0", "1"):
                    errors.append(_diagnostic("value_type", template_name, f"字段 {field_name} 需要布尔值。", row=int(index) + 2, column=field_name))
            allowed = field.get("allowed_values")
            if allowed is not None and str(value) not in {str(item) for item in allowed}:
                errors.append(_diagnostic("allowed_value", template_name, f"字段 {field_name} 值 {value!r} 不在允许值中。", row=int(index) + 2, column=field_name))

    for unique_fields in schema.get("unique", []):
        existing = [field for field in unique_fields if field in frame.columns]
        if len(existing) == len(unique_fields) and frame.duplicated(existing, keep=False).any():
            for index in frame.index[frame.duplicated(existing, keep=False)]:
                errors.append(_diagnostic("unique_constraint", template_name,
                                          f"字段组合 {unique_fields} 必须唯一。", row=int(index) + 2))

    reference_tables = tables or {}
    for field in schema["fields"]:
        foreign_key = field.get("foreign_key")
        name = field["name"]
        if not foreign_key or name not in frame.columns:
            continue
        target_name, target_field = foreign_key.split(".", 1)
        target = reference_tables.get(target_name)
        if target is None:
            continue
        if not isinstance(target, pd.DataFrame):
            target = pd.DataFrame(target)
        if target_field not in target:
            errors.append(_diagnostic("foreign_key_schema", template_name, f"外键目标 {foreign_key} 不存在。", column=name))
            continue
        values = set(target[target_field].dropna().astype(str))
        for index, value in frame[name].items():
            if not _is_missing(value) and str(value) not in values:
                errors.append(_diagnostic("foreign_key", template_name, f"{name}={value!r} 不存在于 {foreign_key}。", row=int(index) + 2, column=name))
    if template_name == "events" and "duration_s" in frame:
        for index, value in frame["duration_s"].items():
            if not _is_missing(value):
                try:
                    if float(value) < 0:
                        errors.append(_diagnostic("duration_range", template_name, "duration_s 必须非负。", row=int(index) + 2, column="duration_s"))
                except (TypeError, ValueError):
                    pass  # 基本类型错误已经在字段检查中报告。
    if template_name in ("conditions", "manual_qc") and {"start_s", "end_s"}.issubset(frame.columns):
        for index, row in frame.iterrows():
            if template_name == "manual_qc" and str(row.get("action", "")).casefold() in ("undo", "redo"):
                continue
            if _is_missing(row.get("start_s")) or _is_missing(row.get("end_s")):
                continue
            try:
                if float(row["start_s"]) >= float(row["end_s"]):
                    errors.append(_diagnostic("interval_range", template_name, "时间区间必须满足 start_s < end_s。", row=int(index) + 2))
            except (TypeError, ValueError):
                pass
    if template_name == "statistical_design":
        for field_name in ("covariates", "contrast"):
            if field_name not in frame:
                continue
            for index, value in frame[field_name].items():
                if _is_missing(value) or not str(value).strip():
                    continue
                try:
                    decoded = json.loads(value) if isinstance(value, str) else value
                    if not isinstance(decoded, (dict, list)):
                        raise ValueError("JSON值必须是对象或数组")
                except (TypeError, ValueError, json.JSONDecodeError):
                    errors.append(_diagnostic("json_value", template_name, f"字段 {field_name} 必须是 JSON 对象或数组。", row=int(index) + 2, column=field_name))
    return {"valid": not errors, "errors": errors, "warnings": [], "table": template_name, "rows": int(len(frame))}
