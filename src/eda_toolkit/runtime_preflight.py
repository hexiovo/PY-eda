"""有界运行前检查与快照新鲜度复核；不执行EDA分析。"""
from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
import shutil
import stat
import time
from datetime import datetime, timezone
import uuid

import pandas as pd

from .capabilities import get_capabilities, inspect_output_directory
from .config import canonical_json, config_hash, feature_work_method, resolve_config, synchrony_work_method
from .core_export import json_safe
from .errors import error_to_dict
from .runtime_events import emit_run_event


MAX_PREFLIGHT_TABLE_BYTES = 64 * 1024 * 1024
MAX_PREFLIGHT_TABLE_ROWS = 500
MAX_PREFLIGHT_HEADER_BYTES = 64 * 1024
MAX_PREFLIGHT_WORK_UNITS = 10_000
_ASSOCIATED_TABLES = ("participants", "events", "conditions", "group_membership", "sync_anchors",
                      "manual_qc", "statistical_design")


def _issue(code, stage, message, *, path="", remediation="", blocking=True):
    return {"code": str(code), "stage": str(stage), "path": str(path or ""),
            "message": str(message), "remediation": str(remediation), "blocking": bool(blocking)}


def _blank(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"", "na", "nan", "null"}
    return False


def _emit(event_sink, kind, message, *, units_done=None, units_total=None, details=None, level="INFO"):
    if event_sink is None:
        return
    try:
        emit_run_event(event_sink, {"stage": "preflight", "kind": kind, "level": level, "message": message,
                                   "units_done": units_done, "units_total": units_total,
                                   "unit_type": "input_file", "details": details or {}})
    except Exception:
        # 进度订阅故障不能把preflight变成分析失败。
        return


def _cancelled(cancel_callback):
    try:
        return bool(cancel_callback and cancel_callback())
    except Exception:
        return False


def _project_snapshot(project_state, config):
    state = project_state if isinstance(project_state, dict) else {}
    project_directory = (state.get("project_directory") or config.get("workspace", {}).get("project_directory")
                         or config.get("input", {}).get("base_directory") or str(Path.cwd()))
    project_path = Path(project_directory).expanduser().resolve()
    project_id = state.get("project_id")
    generated = False
    if not project_id:
        project_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "eda-project:" + str(project_path)))
        generated = True
    try:
        project_id = str(uuid.UUID(str(project_id)))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("project_state.project_id必须是UUID字符串。")
    revisions = {}
    for field in ("project_revision", "draft_revision", "marker_revision", "qc_revision"):
        value = state.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"project_state.{field}必须是非负整数且不能是bool。")
        revisions[field] = value
    return {"project_id": project_id, **revisions}, project_path, generated


def _input_base(config, manifest_value, project_path):
    configured = config.get("input", {}).get("base_directory")
    if configured:
        base = Path(configured).expanduser()
        return base.resolve() if base.is_absolute() else (project_path / base).resolve()
    if isinstance(manifest_value, (str, os.PathLike)):
        candidate = Path(manifest_value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve().parent
        return (project_path / candidate).resolve().parent
    return project_path


def _memory_manifest_rows(value):
    safe_value = json_safe(value)
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        rows = safe_value if isinstance(safe_value, list) else []
        return rows, safe_value
    if isinstance(value, dict):
        if isinstance(value.get("signals"), list):
            return value["signals"], safe_value
        if isinstance(value.get("files"), list):
            return value["files"], safe_value
        return [value], safe_value
    if isinstance(value, list):
        return value, safe_value
    return [], safe_value


def _read_csv_preview(path, config, *, max_rows=MAX_PREFLIGHT_TABLE_ROWS):
    from .core_io import preview_csv
    candidate = Path(path).expanduser().resolve()
    size = candidate.stat().st_size
    if size > MAX_PREFLIGHT_TABLE_BYTES:
        raise ValueError(f"preflight_table_size_limit:{size}")
    result = preview_csv(candidate, max_rows=max_rows, options=config)
    if result["total_rows"] > max_rows:
        raise ValueError(f"preflight_table_row_limit:{result['total_rows']}")
    return result


def _hash_file_stable(path, cancel_callback=None):
    candidate = Path(path).expanduser().resolve()
    before = candidate.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("input_not_regular_file")
    digest = hashlib.sha256()
    total = 0
    with candidate.open("rb") as stream:
        while True:
            if _cancelled(cancel_callback):
                raise InterruptedError("preflight_cancelled")
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
    after = candidate.stat()
    if (before.st_size, before.st_mtime_ns, getattr(before, "st_ino", None)) != (
            after.st_size, after.st_mtime_ns, getattr(after, "st_ino", None)) or total != after.st_size:
        raise ValueError("input_changed_during_preflight")
    return digest.hexdigest(), total


def _read_header(path, config):
    from .core_io import _choose_encoding, _sniff_delimiter
    candidate = Path(path).expanduser().resolve()
    with candidate.open("rb") as stream:
        payload = stream.read(MAX_PREFLIGHT_HEADER_BYTES)
    encoding = _choose_encoding(payload, config.get("input", {}).get("encoding"))
    text = payload.decode(encoding, errors="strict")
    delimiter = _sniff_delimiter(text, config.get("input", {}).get("delimiter"))
    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    try:
        header = [item.strip() for item in next(reader)]
    except StopIteration:
        raise ValueError("empty_csv")
    if not header or any(not item for item in header) or len(header) != len(set(header)):
        raise ValueError("invalid_csv_header")
    return header, encoding, delimiter


def _resolve_path(value, base):
    candidate = Path(str(value)).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _record_sources(manifest_value, config, base, issues, fingerprints, file_sizes, cancel_callback, event_sink):
    rows = []
    manifest_path = None
    manifest_table = None
    if manifest_value is not None:
        if isinstance(manifest_value, (str, os.PathLike)):
            manifest_path = _resolve_path(manifest_value, base)
            try:
                preview = _read_csv_preview(manifest_path, config)
                manifest_table = preview["preview"]
                rows = manifest_table.to_dict(orient="records")
                fingerprints[str(manifest_path)] = preview["source_sha256"]
                file_sizes[str(manifest_path)] = manifest_path.stat().st_size
            except InterruptedError:
                raise
            except Exception as error:
                code = "manifest_preflight_limit" if str(error).startswith("preflight_table_") else "manifest_unreadable"
                issues.append(_issue(code, "input", f"输入清单无法在预检上限内解析：{error}。", path=str(manifest_path),
                                     remediation="检查清单格式；若超过预检上限，拆成较小的清单后重试。"))
        else:
            rows, safe_value = _memory_manifest_rows(manifest_value)
            manifest_table = pd.DataFrame(rows)
            try:
                fingerprints["memory:input_manifest"] = hashlib.sha256(canonical_json(safe_value).encode("utf-8")).hexdigest()
            except (TypeError, ValueError):
                issues.append(_issue("manifest_not_serializable", "input", "内存输入清单不能稳定序列化。",
                                     remediation="使用字符串字段/数值/null组成的记录列表。"))
    else:
        signals = config.get("input", {}).get("signals", [])
        rows = [{"file_path": item} if isinstance(item, str) else dict(item) for item in signals if isinstance(item, (str, dict))]
        if rows:
            fingerprints["memory:signals"] = hashlib.sha256(canonical_json(json_safe(rows)).encode("utf-8")).hexdigest()

    if not rows:
        issues.append(_issue("no_signal_inputs", "input", "没有可检查的信号输入。", remediation="选择信号CSV或输入清单。"))
    signal_sources = []
    seen = set()
    for index, row in enumerate(rows):
        if _cancelled(cancel_callback):
            raise InterruptedError("preflight_cancelled")
        if isinstance(row, str):
            row = {"file_path": row}
        if not isinstance(row, dict):
            issues.append(_issue("signal_manifest_row", "input", f"第{index + 1}条信号清单必须是对象。"))
            continue
        file_value = row.get("file_path", row.get("path"))
        if _blank(file_value):
            issues.append(_issue("signal_path_missing", "input", f"第{index + 1}条信号缺少file_path。",
                                 remediation="为每条信号指定可读CSV路径。"))
            continue
        path = _resolve_path(file_value, base)
        key = str(path)
        source = {"path": path, "row": row, "index": index}
        signal_sources.append(source)
        if key in seen:
            continue
        seen.add(key)
        try:
            if not path.is_file():
                issues.append(_issue("signal_unreadable", "input", "信号路径不存在或不是普通文件。", path=key,
                                     remediation="确认路径与项目相对目录，并检查文件权限。"))
                continue
            header, _encoding, _delimiter = _read_header(path, config)
            digest, size = _hash_file_stable(path, cancel_callback)
            fingerprints[key] = digest
            file_sizes[key] = size
            _emit(event_sink, "progress", "已完成信号文件快照指纹", units_done=1,
                  details={"work_id": f"signal-file:{index}", "path": key, "size_bytes": size})
            mapping = config.get("mapping", {})
            columns = mapping.get("columns", {}) if isinstance(mapping.get("columns"), dict) else {}
            eda_column = columns.get("eda_uS", "eda_uS")
            time_column = columns.get("time_s", "time_s")
            if eda_column not in header:
                issues.append(_issue("signal_column_missing", "input", f"信号文件缺少映射的幅值列{eda_column!r}。", path=key,
                                     remediation="修正mapping.columns.eda_uS或CSV表头。"))
            row_unit = source["row"].get("eda_unit")
            if _blank(row_unit):
                row_unit = None
            declared_unit = row_unit or mapping.get("eda_unit")
            if eda_column != "eda_uS" and not declared_unit and columns.get("eda_unit", "eda_unit") not in header:
                issues.append(_issue("eda_unit_required", "input", "自定义幅值列必须有明确的eda_unit。", path=key,
                                     remediation="在清单/映射中声明源单位，或映射eda_unit列。"))
            if time_column not in header:
                sampling_rate = source["row"].get("sampling_rate_hz")
                if _blank(sampling_rate):
                    sampling_rate = mapping.get("sampling_rate_hz")
                if sampling_rate is None:
                    issues.append(_issue("sampling_rate_required", "input", "信号缺少时间列且没有声明采样率。", path=key,
                                         remediation="映射time_s列，或提供正的sampling_rate_hz。"))
            clock = source["row"].get("clock_domain")
            if _blank(clock):
                clock = mapping.get("clock_domain")
            if not isinstance(clock, str) or not clock.strip():
                issues.append(_issue("clock_domain_required", "input", "信号时钟域必须明确。", path=key,
                                     remediation="在输入清单或映射中声明clock_domain。"))
            source["header"] = header
        except InterruptedError:
            raise
        except Exception as error:
            details = error_to_dict(error, stage="preflight")
            issues.append(_issue(details.get("code", "signal_preflight"), "input", details.get("message", str(error)), path=key,
                                 remediation=details.get("remediation", "检查信号文件编码、表头和访问权限。")))
    return signal_sources, rows, manifest_table


def _associated_tables(config, base, issues, fingerprints, file_sizes, cancel_callback):
    frames = {}
    for name in _ASSOCIATED_TABLES:
        value = config.get("input", {}).get(name)
        if value is None:
            continue
        path = _resolve_path(value, base)
        key = str(path)
        try:
            preview = _read_csv_preview(path, config)
            frame = preview["preview"]
            frames[name] = frame
            fingerprints[key] = preview["source_sha256"]
            file_sizes[key] = path.stat().st_size
        except InterruptedError:
            raise
        except Exception as error:
            code = "table_preflight_limit" if str(error).startswith("preflight_table_") else f"{name}_unreadable"
            issues.append(_issue(code, name, f"{name}表无法在预检上限内读取：{error}。", path=key,
                                 remediation="确认CSV可读且不超过预检大小/行数上限。"))
    if "input_manifest" not in frames and config.get("input", {}).get("manifest"):
        # 清单同时作为信号来源时，由_load_sources执行schema验证；不重复读取。
        pass
    for name, frame in frames.items():
        if "clock_domain" not in frame.columns or frame.empty:
            continue
        declared = {str(value).strip() for value in frame["clock_domain"].dropna().tolist() if str(value).strip()}
        if len(declared) > 1 and name in {"events", "conditions", "manual_qc"}:
            issues.append(_issue("associated_clock_mixed", name,
                                 f"{name}表包含多个时钟域：{sorted(declared)}。", blocking=True,
                                 remediation="为事件/条件/QC声明可验证的clock映射或拆分表。"))
    return frames


def _check_cross_references(signal_sources, rows, tables, issues, config):
    known_recordings = set()
    known_subjects = set()
    recording_ids_complete = True
    subject_ids_complete = True
    for source in signal_sources:
        row = source["row"]
        recording_id = row.get("recording_id")
        if not _blank(recording_id):
            known_recordings.add(str(recording_id).strip())
        else:
            mapping_columns = config.get("mapping", {}).get("columns", {})
            if source.get("header") and mapping_columns.get("recording_id", "recording_id") in source["header"]:
                recording_ids_complete = False
            else:
                known_recordings.add(source["path"].stem)
        subject = row.get("subject_id")
        if not _blank(subject):
            known_subjects.add(str(subject).strip())
        elif source.get("header") and config.get("mapping", {}).get("columns", {}).get("subject_id", "subject_id") in source["header"]:
            subject_ids_complete = False
    participants = tables.get("participants")
    if participants is not None and "subject_id" in participants:
        known_subjects.update(str(value).strip() for value in participants["subject_id"].dropna().tolist())
    for table_name in ("events", "conditions", "manual_qc", "sync_anchors", "group_membership"):
        frame = tables.get(table_name)
        if frame is None:
            continue
        if "recording_id" in frame and recording_ids_complete:
            for index, value in frame["recording_id"].items():
                if not _blank(value) and str(value).strip() not in known_recordings:
                    issues.append(_issue(f"{table_name}_recording_fk", table_name,
                                         f"第{int(index) + 2}行引用未知recording_id。", blocking=True,
                                         remediation="修正recording_id，或将记录加入输入清单。"))
                    break
        if "subject_id" in frame and known_subjects and subject_ids_complete:
            for index, value in frame["subject_id"].items():
                if not _blank(value) and str(value).strip() not in known_subjects:
                    issues.append(_issue(f"{table_name}_subject_fk", table_name,
                                         f"第{int(index) + 2}行引用未知subject_id。", blocking=True,
                                         remediation="修正subject_id或补齐participants清单。"))
                    break
        if "clock_domain" in frame and not frame.empty:
            domains = {str(value).strip() for value in frame["clock_domain"].dropna().tolist() if str(value).strip()}
            signal_domains = {str(source["row"].get("clock_domain") or config.get("mapping", {}).get("clock_domain", "recording_local"))
                              for source in signal_sources}
            has_anchors = bool(config.get("input", {}).get("sync_anchors"))
            if domains and not domains <= signal_domains and (not config.get("synchrony", {}).get("enabled", False) or not has_anchors):
                issues.append(_issue("associated_clock_unresolved", table_name,
                                     f"{table_name}时钟域{sorted(domains)}与信号时钟域{sorted(signal_domains)}不一致。",
                                     remediation="声明有效同步锚点并启用同步校准，或修正表中的clock_domain。"))


def _work_plan(signal_sources, config, *, total_known):
    stage_definitions = [
        ("qc", "record_stage", "quality_mask"),
        ("preprocess", "record_stage", "preprocess"),
        ("decomposition", "record_stage", config.get("decomposition", {}).get("method")),
        ("features", "record_stage", feature_work_method(config)),
    ]
    units = []
    for source in signal_sources:
        row = source["row"]
        record = str(row.get("recording_id") or source["path"].stem)
        channel = str(row.get("channel") or "unknown")
        fs = row.get("sampling_rate_hz")
        if _blank(fs):
            fs = config.get("mapping", {}).get("sampling_rate_hz")
        rate = float(fs) if isinstance(fs, (int, float)) and not isinstance(fs, bool) else None
        for stage, unit_type, method in stage_definitions:
            units.append({"work_id": f"{record}::{channel}:{stage}", "stage": stage, "unit_type": unit_type,
                          "recording_id": record, "method": method, "rate_hz": rate})
    if config.get("synchrony", {}).get("enabled"):
        units.append({"work_id": "synchrony:group", "stage": "synchrony", "unit_type": "analysis_stage",
                      "method": synchrony_work_method(config), "rate_hz": config.get("synchrony", {}).get("analysis_rate_hz")})
    if config.get("statistics", {}).get("enabled"):
        units.append({"work_id": "statistics:planned", "stage": "statistics", "unit_type": "analysis_stage",
                      "method": "structured_statistics", "rate_hz": None})
    export_formats = list(config.get("output", {}).get("formats", []))
    for output_format in export_formats:
        units.append({"work_id": f"export:{output_format}", "stage": "export", "unit_type": "export_file",
                      "method": output_format, "rate_hz": None})
    units.append({"work_id": "run:manifest_finalization", "stage": "finalize", "unit_type": "manifest",
                  "method": "manifest_sha256", "rate_hz": None})
    compact = len(units) <= MAX_PREFLIGHT_WORK_UNITS
    return {"revision": 0, "total_units": len(units) if total_known else None,
            "units": units if compact else [], "unit_groups": [],
            "eta_min_completed_units": config.get("runtime", {}).get("eta_min_completed_units", 3),
            "estimated_units": len(units), "enumeration_truncated": not compact,
            "unknown_total_reason": "channel_count_not_declared" if not total_known else ""}


def _report_hash(report):
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    return hashlib.sha256(canonical_json(json_safe(payload)).encode("utf-8")).hexdigest()


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value.lower()))


def _capability_signature(capabilities):
    """Hash stable runtime semantics while excluding live resource probes."""
    if not isinstance(capabilities, dict):
        raise ValueError("capabilities must be an object")
    fields = ("python", "platform", "packages", "methods", "science_qualification")
    stable = {field: capabilities.get(field) for field in fields}
    if (not isinstance(stable["python"], str) or not isinstance(stable["platform"], str) or
            not isinstance(stable["packages"], dict) or not isinstance(stable["methods"], dict) or
            not isinstance(stable["science_qualification"], str)):
        raise ValueError("stable capability fields are missing")
    return hashlib.sha256(canonical_json(json_safe(stable)).encode("utf-8")).hexdigest()


def _snapshot_inputs(input_manifest, resolved, project_path):
    # Reuses bounded manifest/table parsing so the enqueue-time fingerprint rules are identical.
    issues = []
    fingerprints = {}
    file_sizes = {}
    manifest_value = input_manifest if input_manifest is not None else resolved.get("input", {}).get("manifest")
    base = _input_base(resolved, manifest_value, project_path)
    signal_sources, rows, manifest_table = _record_sources(manifest_value, resolved, base, issues, fingerprints,
                                                            file_sizes, None, None)
    tables = _associated_tables(resolved, base, issues, fingerprints, file_sizes, None)
    if manifest_table is not None:
        tables["input_manifest"] = manifest_table
    from .core_io import validate_table
    for name, frame in tables.items():
        validation = validate_table(name, frame, tables=tables)
        path_value = resolved.get("input", {}).get(name) or (manifest_value if name == "input_manifest" else "")
        path = str(_resolve_path(path_value, base)) if isinstance(path_value, (str, os.PathLike)) else ""
        for error in validation.get("errors", [])[:50]:
            issues.append(_issue(error.get("code", f"{name}_invalid"), name,
                                 error.get("message", f"{name}表字段、唯一键或引用无效。"),
                                 path=path, remediation=f"按{name}模板修复字段、类型、唯一键和引用。"))
    _check_cross_references(signal_sources, rows, tables, issues, resolved)
    return fingerprints, file_sizes, issues, signal_sources, rows, tables, base


def preflight_run(config, input_manifest, project_state, cancel_callback=None, event_sink=None):
    """检查配置/输入/资源与项目revision，返回可复核报告，不执行完整分析。"""
    started = time.monotonic()
    errors = []
    warnings = []
    fingerprints = {}
    file_sizes = {}
    resolved = None
    report_config_hash = None
    try:
        resolved = resolve_config(config)
        report_config_hash = config_hash(resolved)
    except Exception as error:
        details = error_to_dict(error, stage="preflight")
        errors.append(_issue(details.get("code", "invalid_config"), "configuration",
                             details.get("message", str(error)), path=details.get("field", ""),
                             remediation=details.get("remediation", "修正配置schema与数值范围后重新预检。")))
    if resolved is None:
        try:
            resolved = resolve_config(None)
        except Exception:
            resolved = None
    try:
        project_info, project_path, project_id_generated = _project_snapshot(project_state, resolved or {})
    except Exception as error:
        project_info = {"project_id": str(uuid.uuid4()), "project_revision": 0, "draft_revision": 0,
                        "marker_revision": 0, "qc_revision": 0}
        project_path = Path.cwd().resolve()
        project_id_generated = True
        errors.append(_issue("project_state_invalid", "project", str(error),
                             remediation="由项目store提供UUID与非负整数revision。"))
    if project_id_generated:
        warnings.append(_issue("project_id_generated", "project", "预检未收到稳定project_id，使用项目路径派生的临时UUID。",
                               blocking=False, remediation="项目store打开后请用持久project_id重新执行预检。"))
    _emit(event_sink, "stage", "检查配置、输入来源与资源预算")
    if _cancelled(cancel_callback):
        errors.append(_issue("preflight_cancelled", "preflight", "运行前检查已取消。", remediation="重新执行预检。"))

    signal_sources = []
    rows = []
    tables = {}
    base = project_path
    if resolved is not None and not any(issue["code"] == "preflight_cancelled" for issue in errors):
        try:
            fingerprints, file_sizes, input_issues, signal_sources, rows, tables, base = _snapshot_inputs(
                input_manifest, resolved, project_path)
            errors.extend(issue for issue in input_issues if issue["blocking"])
            warnings.extend(issue for issue in input_issues if not issue["blocking"])
        except InterruptedError:
            errors.append(_issue("preflight_cancelled", "preflight", "运行前检查已取消。", remediation="重新执行预检。"))
        except Exception as error:
            details = error_to_dict(error, stage="preflight")
            errors.append(_issue(details.get("code", "input_preflight"), "input",
                                 details.get("message", str(error)), path=details.get("field", ""),
                                 remediation=details.get("remediation", "检查输入清单、文件路径和单位/时间映射。")))

    output_config = resolved or resolve_config(None)
    output_directory = Path(output_config["output"]["directory"]).expanduser()
    if not output_directory.is_absolute():
        output_directory = (base / output_directory).resolve()
    output_status = inspect_output_directory(output_directory)
    if not output_status.get("writable"):
        errors.append(_issue("output_not_writable", "output", "输出目录不可写。", path=str(output_directory),
                             remediation=str(output_status.get("reason", "选择可写输出目录。"))))
    existing_parent = output_directory
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    try:
        disk_free = shutil.disk_usage(existing_parent).free
    except OSError:
        disk_free = None
    input_bytes = sum(file_sizes.values())
    # 这只是有界字节比例估算，不代替完整样本数/算法缓冲检查。
    estimated_memory_bytes = input_bytes * 12
    estimated_output_bytes = max(1_048_576, input_bytes * 2)
    if disk_free is not None and estimated_output_bytes > disk_free:
        errors.append(_issue("output_disk_space", "output", "估算的输出空间超过当前可用磁盘空间。", path=str(output_directory),
                             remediation="选择空间充足的输出目录或减少批次规模。"))
    capabilities = {}
    source_fp = None
    environment_fp = None
    capability_signature = None
    try:
        capabilities = get_capabilities(output_directory)
        source_fp = capabilities.get("source_fingerprint") if isinstance(capabilities, dict) else None
        environment_fp = (capabilities.get("numerical_environment_fingerprint")
                          if isinstance(capabilities, dict) else None)
        if not _is_sha256(source_fp) or not _is_sha256(environment_fp):
            errors.append(_issue("capability_fingerprint_unavailable", "capabilities",
                                 "源码或数值环境指纹不可用，不能生成可入队预检。",
                                 remediation="检查发行构建指纹及数值依赖后重新预检。"))
        else:
            capability_signature = _capability_signature(capabilities)
        if resolved is not None:
            from .capabilities import require_capabilities
            try:
                require_capabilities(resolved)
            except Exception as error:
                details = error_to_dict(error, stage="preflight")
                errors.append(_issue(details.get("code", "capability_unavailable"), "capabilities",
                                     details.get("message", str(error)), remediation=details.get("remediation", "选择可用方法或安装可选依赖。")))
    except Exception as error:
        errors.append(_issue("capability_probe_unavailable", "capabilities", str(error),
                             remediation="检查运行环境和发行构建指纹后重新预检。"))
    if resolved is not None and resolved["synchrony"]["enabled"]:
        if not resolved["input"].get("sync_anchors") and not resolved["synchrony"].get("shared_clock_verified"):
            errors.append(_issue("sync_clock_evidence_missing", "synchrony", "同步分析未声明可验证的共享时钟或sync_anchors。",
                                 remediation="提供同步锚点，或按实际采集证据声明共享时钟。"))
        record_count = len(signal_sources)
        pair_count = record_count * max(0, record_count - 1) // 2
        if pair_count > resolved["compute"]["max_pairs"]:
            errors.append(_issue("pair_budget", "synchrony", "按输入记录数估算的配对数超过max_pairs。",
                                 remediation="显式调整max_pairs或拆分输入批次。"))
        sync = resolved["synchrony"]
        lag_count = int(2 * sync["max_lag_s"] / sync["lag_step_s"]) + 1
        estimated_null = pair_count * resolved["compute"]["max_windows"] * lag_count * sync["null_replicates"]
        if estimated_null > resolved["compute"]["max_null_evaluations"]:
            warnings.append(_issue("null_budget_upper_bound", "synchrony",
                                   "最坏情况null计算上界超过max_null_evaluations；实际窗口数仍需导入后精确估计。",
                                   blocking=False, remediation="查看计划报告；若实际上界仍超限，显式缩小窗口/配对/重复数。"))
    work_plan = _work_plan(signal_sources, output_config,
                           total_known=bool(signal_sources) and not any(
                               _blank(source["row"].get("channel")) and source.get("header") and
                               resolved and resolved["mapping"].get("columns", {}).get("channel", "channel") in source["header"]
                               for source in signal_sources))
    resource_estimate = {
        "input_file_count": len(file_sizes), "input_bytes_total": input_bytes,
        "estimated_memory_bytes": estimated_memory_bytes,
        "estimated_memory_basis": "input_csv_bytes_x12_heuristic_not_full_sample_count",
        "configured_memory_limit_mb": output_config["compute"]["memory_limit_mb"],
        "available_memory_mb": capabilities.get("memory", {}).get("available_mb"),
        "estimated_output_bytes": estimated_output_bytes, "disk_free_bytes": disk_free,
        "workers": output_config["compute"]["workers"],
        "work_unit_count": work_plan.get("total_units"),
        "bounded_preflight": {"table_max_bytes": MAX_PREFLIGHT_TABLE_BYTES,
                              "table_max_rows": MAX_PREFLIGHT_TABLE_ROWS,
                              "header_max_bytes": MAX_PREFLIGHT_HEADER_BYTES},
        "output_probe": output_status,
    }
    report = {
        "report_id": str(uuid.uuid4()), "report_sha256": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": report_config_hash, "input_fingerprints": dict(sorted(fingerprints.items())),
        "source_fingerprint": source_fp,
        "numerical_environment_fingerprint": environment_fp,
        "capability_signature": capability_signature,
        **project_info, "errors": errors, "warnings": warnings, "capabilities": capabilities,
        "resource_estimate": resource_estimate, "work_plan": work_plan,
        "eligible_to_enqueue": not any(issue.get("blocking") for issue in errors),
    }
    report["report_sha256"] = _report_hash(report)
    _emit(event_sink, "completed" if report["eligible_to_enqueue"] else "failed",
          "运行前检查完成" if report["eligible_to_enqueue"] else "运行前检查发现阻断项",
          units_done=1, units_total=1, details={"report_id": report["report_id"],
                                                  "eligible_to_enqueue": report["eligible_to_enqueue"],
                                                  "elapsed_s": time.monotonic() - started},
          level="INFO" if report["eligible_to_enqueue"] else "WARNING")
    return report


def validate_preflight_snapshot(report, config, input_manifest, project_state):
    """入队前复算配置、输入、实现/环境指纹、资源预算与项目revision。"""
    errors = []
    stale = False
    if not isinstance(report, dict):
        return {"valid": False, "stale": False, "errors": [_issue("preflight_report_missing", "preflight",
                                                                       "预检报告缺失或类型错误。")]}
    report_id = report.get("report_id")
    try:
        uuid.UUID(str(report_id))
    except (ValueError, TypeError, AttributeError):
        errors.append(_issue("preflight_report_id", "preflight", "report_id不是UUID，报告不可复核。"))
    if report.get("report_sha256") != _report_hash(report):
        errors.append(_issue("preflight_report_hash", "preflight", "预检报告内容哈希不匹配。"))
        stale = True
    try:
        resolved = resolve_config(config)
        actual_config_hash = config_hash(resolved)
    except Exception as error:
        details = error_to_dict(error, stage="preflight")
        errors.append(_issue(details.get("code", "invalid_config"), "configuration",
                             details.get("message", str(error)), remediation=details.get("remediation", "修正配置后重新预检。")))
        resolved = None
        actual_config_hash = None
    if actual_config_hash != report.get("config_hash"):
        errors.append(_issue("preflight_config_stale", "configuration", "配置与预检报告时不同。",
                             remediation="配置发生变化，请重新预检并重新入队。"))
        stale = True
    project_path = Path.cwd().resolve()
    try:
        project_info, project_path, _ = _project_snapshot(project_state, resolved or {})
        for field, actual in project_info.items():
            if report.get(field) != actual:
                errors.append(_issue("preflight_revision_stale", "project", f"{field}已变化。",
                                     remediation="项目/marker/QC草稿revision发生变化，请重新预检。"))
                stale = True
    except Exception as error:
        errors.append(_issue("project_state_invalid", "project", str(error),
                             remediation="由project store提供稳定UUID与当前revision。"))
    input_base = project_path
    if resolved is not None:
        try:
            current_inputs, _sizes, input_issues, _signals, _rows, _tables, input_base = _snapshot_inputs(
                input_manifest, resolved, project_path)
            if any(issue.get("blocking") for issue in input_issues):
                errors.extend(issue for issue in input_issues if issue.get("blocking"))
                stale = True
            if current_inputs != report.get("input_fingerprints"):
                errors.append(_issue("preflight_inputs_stale", "input", "输入文件指纹与预检报告时不同。",
                                     remediation="输入内容变化，请重新预检并重新入队。"))
                stale = True
        except Exception as error:
            errors.append(_issue("preflight_inputs_unavailable", "input", f"无法复核输入指纹：{error}。",
                                 remediation="检查输入文件可读性和稳定性后重新预检。"))
            stale = True
        try:
            current_capabilities = get_capabilities()
            current_source = current_capabilities.get("source_fingerprint")
            current_environment = current_capabilities.get("numerical_environment_fingerprint")
            current_signature = _capability_signature(current_capabilities)
            if not _is_sha256(current_source) or not _is_sha256(current_environment):
                raise ValueError("runtime fingerprint is missing or malformed")
            for field, actual, code, label in (
                ("source_fingerprint", current_source, "preflight_source_stale", "源码指纹"),
                ("numerical_environment_fingerprint", current_environment,
                 "preflight_environment_stale", "数值环境指纹"),
                ("capability_signature", current_signature, "preflight_capabilities_stale", "稳定能力签名"),
            ):
                if not _is_sha256(report.get(field)):
                    errors.append(_issue("preflight_runtime_fingerprint_missing", "capabilities",
                                         f"预检报告缺少有效的{label}。",
                                         remediation="重新生成当前版本的预检报告。"))
                    stale = True
                elif report.get(field) != actual:
                    errors.append(_issue(code, "capabilities", f"{label}与预检报告时不同。",
                                         remediation="实现、依赖或方法能力已变化，请重新预检并重新入队。"))
                    stale = True
        except Exception as error:
            errors.append(_issue("preflight_runtime_fingerprint_unavailable", "capabilities",
                                 f"无法复核当前运行环境指纹：{error}。",
                                 remediation="检查源码构建指纹和依赖后重新预检。"))
            stale = True

        # Recheck resource sufficiency, but do not compare volatile memory/disk
        # readings byte-for-byte: ordinary machine activity changes those values.
        try:
            output_directory = Path(resolved["output"]["directory"]).expanduser()
            if not output_directory.is_absolute():
                output_directory = (input_base / output_directory).resolve()
            current_output = inspect_output_directory(output_directory)
            if not current_output.get("writable"):
                errors.append(_issue("preflight_output_not_writable", "output", "当前输出目录不可写。",
                                     path=str(output_directory), remediation=str(current_output.get("reason") or
                                                                                 "选择可写目录后重新预检。")))
                stale = True
            estimated_bytes = (report.get("resource_estimate", {}).get("estimated_output_bytes")
                               if isinstance(report.get("resource_estimate"), dict) else None)
            if isinstance(estimated_bytes, bool) or not isinstance(estimated_bytes, int) or estimated_bytes < 0:
                errors.append(_issue("preflight_resource_estimate_invalid", "output",
                                     "预检报告缺少有效的输出空间估算。",
                                     remediation="重新生成预检报告。"))
            else:
                existing_parent = output_directory
                while not existing_parent.exists() and existing_parent != existing_parent.parent:
                    existing_parent = existing_parent.parent
                disk_free = shutil.disk_usage(existing_parent).free
                if estimated_bytes > disk_free:
                    errors.append(_issue("preflight_output_disk_space", "output",
                                         "当前可用磁盘空间低于预检输出预算。", path=str(output_directory),
                                         remediation="释放磁盘空间或选择空间充足的输出目录后重新预检。"))
                    stale = True
        except Exception as error:
            errors.append(_issue("preflight_resource_recheck_failed", "output",
                                 f"无法复核当前输出资源：{error}。",
                                 remediation="检查输出路径和磁盘状态后重新预检。"))
            stale = True
    report_blocking = [issue for issue in report.get("errors", []) if isinstance(issue, dict) and issue.get("blocking")]
    if report.get("eligible_to_enqueue") is not True or report_blocking:
        errors.append(_issue("preflight_blocked", "preflight", "预检报告含阻断项，不能入队。",
                             remediation="逐项修复errors后重新预检。"))
    valid = not stale and not errors
    return {"valid": valid, "stale": stale, "errors": errors}


__all__ = ["preflight_run", "validate_preflight_snapshot", "MAX_PREFLIGHT_TABLE_BYTES",
           "MAX_PREFLIGHT_TABLE_ROWS", "MAX_PREFLIGHT_HEADER_BYTES"]
