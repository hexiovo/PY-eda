"""GUI/CLI共用运行控制器；计算函数与文件提交、任务生命周期分开。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import pandas as pd

from .capabilities import get_capabilities, require_capabilities, source_fingerprint, numerical_environment_fingerprint, inspect_output_directory
from .config import config_hash, feature_work_method, resolve_config, synchrony_work_method
from .core_export import atomic_csv, atomic_json, export_run, json_safe, refresh_manifest, safe_name, sha256_file, verify_manifest
from .errors import error_to_dict, fail
from .runtime_events import emit_run_event


def _cancel(callback):
    if callback is not None and callback():
        raise InterruptedError("用户请求取消；已提交的记录会保留。")


def _notify(callback, stage, message, *, completed=0, total=0, record_id=""):
    if callback is not None:
        try:
            callback({"stage": stage, "message": message, "completed": completed,
                      "total": total, "record_id": record_id})
        except Exception:
            # 界面进度回调错误不能损坏计算结果或将其变为假成功。
            pass


def _runtime_event(event_sink, *, stage, kind, message, run_id=None, project_id=None,
                   task_id=None, attempt_id=None, record_id=None, units_done=None,
                   units_total=None, unit_type="", details=None, level="INFO"):
    """送入真实生命周期/工作单位事件；事件sink故障不改变科学计算结果。"""
    if event_sink is None:
        return
    try:
        emit_run_event(event_sink, {
            "project_id": project_id, "task_id": task_id, "attempt_id": attempt_id,
            "run_id": run_id, "stage": stage, "record_id": record_id,
            "kind": kind, "level": level, "message": message,
            "units_done": units_done, "units_total": units_total,
            "unit_type": unit_type, "details": details or {},
        })
    except Exception:
        # UI订阅者/日志服务故障不应让一个已开始的分析被伪装成算法失败。
        return


def _run_work(event_sink, *, stage, message, work_id, unit_type, run_id, project_id,
              task_id, attempt_id, record_id=None, units_total=None, method=None,
              rate_hz=None, function):
    """围住真实执行的一个阶段；只有成功返回后才报告完成单位。"""
    started = time.monotonic()
    details = {"work_id": work_id, "method": method, "rate_hz": rate_hz}
    _runtime_event(event_sink, stage=stage, kind="work_started", message=message,
                   run_id=run_id, project_id=project_id, task_id=task_id,
                   attempt_id=attempt_id, record_id=record_id, units_done=0,
                   units_total=units_total, unit_type=unit_type, details=details)
    try:
        value = function()
    except InterruptedError as error:
        _runtime_event(event_sink, stage=stage, kind="cancelled", message=str(error) or "阶段已取消",
                       run_id=run_id, project_id=project_id, task_id=task_id,
                       attempt_id=attempt_id, record_id=record_id, units_done=0,
                       units_total=units_total, unit_type=unit_type,
                       details={**details, "duration_s": time.monotonic() - started})
        raise
    except Exception as error:
        _runtime_event(event_sink, stage=stage, kind="failed", message=str(error) or type(error).__name__,
                       run_id=run_id, project_id=project_id, task_id=task_id,
                       attempt_id=attempt_id, record_id=record_id, units_done=0,
                       units_total=units_total, unit_type=unit_type,
                       details={**details, "duration_s": time.monotonic() - started,
                                "error_type": type(error).__name__}, level="ERROR")
        raise
    duration = time.monotonic() - started
    _runtime_event(event_sink, stage=stage, kind="completed", message=message + "完成",
                   run_id=run_id, project_id=project_id, task_id=task_id,
                   attempt_id=attempt_id, record_id=record_id, units_done=1,
                   units_total=units_total, unit_type=unit_type,
                   details={**details, "duration_s": duration, "record_status": "succeeded"})
    return value


def _resolve_for_run(config):
    resolved = resolve_config(config)
    origin = Path(config).resolve().parent if isinstance(config, (str, Path)) else Path.cwd()
    base = Path(resolved["input"].get("base_directory") or origin)
    if not base.is_absolute():
        base = origin / base
    resolved["input"]["base_directory"] = str(base.resolve())
    return resolved


resolve_run_config = _resolve_for_run


def preview_plan(config):
    resolved = _resolve_for_run(config)
    capabilities = require_capabilities(resolved)
    from .core_features import list_feature_definitions
    registry = list_feature_definitions(resolved)
    selected = set(resolved.get("features", {}).get("ids", []))
    effective_ids = selected or set(registry)
    feature_definitions = {feature_id: registry[feature_id]
                           for feature_id in sorted(effective_ids) if feature_id in registry}
    stages = ["导入与校验", "时间基准", "质控", "分段预处理", "滤波分解", "SCR", "窗口与特征"]
    if resolved["synchrony"]["enabled"]:
        stages += ["共同网格与多人同步", "零模型"]
    if resolved["statistics"]["enabled"]:
        stages.append("按设计统计")
    stages += ["绘图", "导出与完整性清单"]
    output = Path(resolved["output"]["directory"])
    if not output.is_absolute():
        output = Path(resolved["input"]["base_directory"]) / output
    return {"config_sha256": config_hash(resolved), "stages": stages, "resolved_config": resolved,
            "output_directory": inspect_output_directory(output),
            "capabilities": capabilities, "parameter_qualification": "research_examples_not_device_qualified",
            "workers": resolved["compute"]["workers"], "backend": "cpu",
            "feature_definitions": feature_definitions}


def _key(record):
    meta = record["metadata"]
    return f"{meta.get('recording_id','')}::{meta.get('channel','eda')}"


def _save_cache(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundle = result["bundle"]
    arrays = {}
    for name in ("time_s", "clean_uS", "tonic_uS", "phasic_uS", "residual_uS", "driver_uS_s", "driver", "detector_eligible", "onset_eligible", "analysis_signal", "sample_index"):
        if isinstance(bundle.get(name), np.ndarray):
            arrays[name] = bundle[name]
    for name, value in bundle.get("quality", {}).items():
        if isinstance(value, np.ndarray):
            arrays["quality__" + name] = value
    for name, value in bundle.get("quality", {}).get("facts", {}).items():
        if isinstance(value, np.ndarray):
            arrays["fact__" + name] = value
    raw = bundle.get("record", {})
    for name in ("time_s", "eda_uS", "sample_index", "local_time_s"):
        if isinstance(raw.get(name), np.ndarray):
            arrays["raw__" + name] = raw[name]
    np.savez_compressed(directory / ".writing-cache.npz", **arrays)
    (directory / ".writing-cache.npz").replace(directory / "signals.npz")
    meta = {name: bundle.get(name) for name in ("metadata", "method", "basis", "diagnostics", "sampling_rate_hz", "max_gap_s", "operations", "segments", "scaling", "timebase_metadata", "baseline_offset_uS", "analysis_unit")}
    meta["record_metadata"] = raw.get("metadata", bundle.get("metadata", {}))
    atomic_json(directory / "bundle.json", meta)
    for name, table in result["tables"].items():
        atomic_json(directory / f"{name}.json", {"columns": list(table.columns), "rows": table})
    atomic_json(directory / "record_summary.json", result["summary"])
    artifacts = [{"path": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
                 for path in directory.iterdir() if path.is_file() and not path.name.startswith(".") and path.name != "manifest.json"]
    atomic_json(directory / "manifest.json", {"status": result["summary"]["status"], "artifacts": artifacts})


def _read_cache(directory):
    directory = Path(directory)
    if not verify_manifest(directory)["valid"]:
        fail("cache_checksum", "已完成记录的清单/校验不完整，不能复用。", category="input", stage="resume")
    bundle = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    bundle["quality"] = {"facts": {}}
    bundle["record"] = {"metadata": bundle.pop("record_metadata", copy.deepcopy(bundle.get("metadata", {})))}
    with np.load(directory / "signals.npz", allow_pickle=False) as arrays:
        for name in arrays.files:
            if name.startswith("quality__"):
                bundle["quality"][name.split("__", 1)[1]] = arrays[name].copy()
            elif name.startswith("fact__"):
                bundle["quality"]["facts"][name.split("__", 1)[1]] = arrays[name].copy()
            elif name.startswith("raw__"):
                bundle["record"][name.split("__", 1)[1]] = arrays[name].copy()
            else:
                bundle[name] = arrays[name].copy()
    tables = {}
    for path in directory.glob("*.json"):
        if path.stem in ("bundle", "record_summary", "manifest"):
            continue
        content = json.loads(path.read_text(encoding="utf-8"))
        tables[path.stem] = pd.DataFrame(content["rows"], columns=content["columns"])
    return {"bundle": bundle, "tables": tables,
            "summary": json.loads((directory / "record_summary.json").read_text(encoding="utf-8"))}


def _record_tables(result, associated):
    windows = result["tables"]["windows"]
    features = result["tables"]["features"].copy()
    extra = [name for name in ("window_id", "kind", "condition", "event_id", "start_s", "end_s") if name in windows]
    if "window_id" in features and extra:
        features = features.merge(windows[extra], on="window_id", how="left", validate="many_to_one")
    metadata = result["bundle"]["metadata"]
    features["channel"] = metadata.get("channel", "eda")
    for table_name, join_key in (("participants", "subject_id"), ("group_membership", "recording_id")):
        table = associated.get(table_name)
        if isinstance(table, pd.DataFrame) and not table.empty and join_key in table:
            selected = table.loc[table[join_key].astype(str) == str(metadata.get(join_key, ""))]
            if "session_id" in selected:
                selected = selected.loc[selected["session_id"].astype(str) == str(metadata.get("session_id", ""))]
            if len(selected) == 1:
                for column, value in selected.iloc[0].items():
                    if column not in features.columns:
                        features[column] = value
    result["tables"]["features"] = features
    return result


def _channel_scoped_associated(associated, channel):
    """将带channel的附属标注限定到当前通道；缺列/空值/*保持旧广播语义。"""
    scoped = dict(associated)
    for name in ("events", "conditions", "manual_qc"):
        frame = associated.get(name)
        if not isinstance(frame, pd.DataFrame) or frame.empty or "channel" not in frame.columns:
            continue

        def applies(value):
            if value is None or pd.isna(value):
                return True
            text = str(value).strip()
            if text.casefold() in {"", "na", "nan", "null"} or text == "*":
                return True
            return text == str(channel)

        scoped[name] = frame.loc[frame["channel"].map(applies)].copy()
    return scoped


def _feature_progress_adapter(runtime_args, record_id, progress_callback=None):
    """Forward feature helper work events to the runtime sink and UI progress callback."""
    def callback(update):
        if not isinstance(update, dict):
            return
        event_kind = update.get("kind")
        if event_kind in {"work_started", "completed", "failed", "cancelled", "warning", "progress"}:
            details = dict(update.get("details") or {})
            for key in ("work_id", "method", "rate_hz", "window_id", "family", "duration_s", "record_status"):
                if key in update:
                    details.setdefault(key, update[key])
            _runtime_event(runtime_args.get("event_sink"), stage=str(update.get("stage") or "features"),
                           kind=str(event_kind), message=str(update.get("message") or ""),
                           run_id=runtime_args.get("run_id"), project_id=runtime_args.get("project_id"),
                           task_id=runtime_args.get("task_id"), attempt_id=runtime_args.get("attempt_id"),
                           record_id=str(update.get("record_id") or record_id),
                           units_done=update.get("units_done", update.get("completed", 0)),
                           units_total=update.get("units_total", update.get("total")),
                           unit_type=str(update.get("unit_type") or "feature_family"),
                           details=details,
                           level=str(update.get("level") or ("ERROR" if event_kind == "failed" else "INFO")))
        if progress_callback is not None:
            try:
                progress_callback(update)
            except Exception:
                pass
    return callback


def _process_one(record, associated, config, cancel_callback, progress_callback, runtime=None):
    from .core_qc import build_quality_mask
    from .core_preprocess import preprocess_record
    from .core_decompose import separate_tonic_phasic
    from .core_features import build_windows, detect_scr, extract_window_features, scr_onset_eligible_mask
    identifier = str(record["metadata"]["recording_id"])
    metadata = record["metadata"]
    channel = str(metadata.get("channel") or "unknown")
    runtime = runtime or {}
    runtime_args = {key: runtime.get(key) for key in ("event_sink", "run_id", "project_id", "task_id", "attempt_id")}
    stage_totals = runtime.get("stage_totals", {})
    rate = metadata.get("fs_hz", metadata.get("fs_estimated"))
    stage_order = ("qc", "preprocess", "decomposition", "features")

    def execute(stage, label, method, function, position):
        try:
            return _run_work(runtime_args["event_sink"], stage=stage, message=label,
                             work_id=f"{identifier}::{channel}:{stage}", unit_type="record_stage",
                             run_id=runtime_args["run_id"], project_id=runtime_args["project_id"],
                             task_id=runtime_args["task_id"], attempt_id=runtime_args["attempt_id"],
                             record_id=identifier, units_total=stage_totals.get(stage),
                             method=method, rate_hz=rate, function=function)
        except Exception:
            for skipped_stage in stage_order[position + 1:]:
                _runtime_event(runtime_args["event_sink"], stage=skipped_stage, kind="skipped",
                               message="前序阶段失败或取消，未执行该阶段。", run_id=runtime_args["run_id"],
                               project_id=runtime_args["project_id"], task_id=runtime_args["task_id"],
                               attempt_id=runtime_args["attempt_id"], record_id=identifier,
                               units_done=0, units_total=stage_totals.get(skipped_stage),
                               unit_type="record_stage",
                               details={"work_id": f"{identifier}::{channel}:{skipped_stage}",
                                        "method": method, "rate_hz": rate})
            raise

    _cancel(cancel_callback)
    _notify(progress_callback, "qc", "质量控制", record_id=identifier)
    qc = execute("qc", "质量控制", "quality_mask",
                 lambda: build_quality_mask(record, config, associated.get("manual_qc")), 0)
    _cancel(cancel_callback)
    processed = execute("preprocess", "分段预处理", "preprocess",
                        lambda: preprocess_record(record, qc, config), 1)
    _cancel(cancel_callback)
    _notify(progress_callback, "decomposition", "滤波分解与 SCR", record_id=identifier)

    def decompose_and_detect():
        bundle = separate_tonic_phasic(processed, config, cancel_callback=cancel_callback)
        _cancel(cancel_callback)
        scr, detection = detect_scr(bundle, config)
        onset_mask, _onset_report = scr_onset_eligible_mask(bundle, config)
        bundle["onset_eligible"] = onset_mask
        bundle["quality"]["onset_eligible"] = onset_mask
        return bundle, scr, detection

    bundle, scr, detection = execute("decomposition", "滤波分解与 SCR", config["decomposition"]["method"],
                                     decompose_and_detect, 2)

    feature_progress = _feature_progress_adapter(runtime_args, identifier, progress_callback)

    def extract_features():
        windows = build_windows(bundle, associated.get("events"), associated.get("conditions"), config)
        if len(windows) > config["compute"]["max_windows"]:
            fail("window_budget", "窗口数量超过预先配置的资源上限。", category="resource", stage="features")
        features, validity = extract_window_features(bundle, scr, windows, config,
                                                     cancel_callback=cancel_callback,
                                                     progress_callback=feature_progress)
        return windows, features, validity

    windows, features, validity = execute("features", "窗口与特征", feature_work_method(config), extract_features, 3)
    links = validity.attrs.get("event_scr_links", pd.DataFrame())
    tables = {"scr": scr, "windows": windows, "features": features, "validity": validity,
              "event_scr_links": links, "qc_intervals": qc.get("intervals", pd.DataFrame())}
    extended_tables = validity.attrs.get("extended_tables", {}) if isinstance(validity, pd.DataFrame) else {}
    if isinstance(extended_tables, dict):
        for name, table in extended_tables.items():
            if isinstance(name, str) and isinstance(table, pd.DataFrame):
                tables[name] = table.copy(deep=True)
    # The linked tables have their own cached/exported result names. Remove the
    # DataFrame-valued attrs before multi-record concatenation (pandas compares
    # attrs and can otherwise ask for an ambiguous DataFrame truth value).
    validity.attrs = {}
    tables = {name: table if isinstance(table, pd.DataFrame) else pd.DataFrame(table) for name, table in tables.items()}
    time_array = bundle["time_s"]
    signals = {"time_s": time_array}
    for name in ("clean_uS", "tonic_uS", "phasic_uS", "residual_uS", "driver_uS_s", "analysis_signal"):
        values = bundle.get(name)
        if values is not None and len(values) == len(time_array):
            signals[name] = values
    for name in ("eligible", "analysis_eligible", "detector_eligible", "onset_eligible", "driver_eligible", "interpolated", "edge_affected"):
        values = bundle.get("quality", {}).get(name)
        if isinstance(values, np.ndarray) and len(values) == len(time_array):
            signals[name] = values
    tables["signals"] = pd.DataFrame(signals)
    raw = bundle.get("record", record)
    raw_signals = {"time_s": raw["time_s"], "eda_uS": raw["eda_uS"], "sample_index": raw["sample_index"]}
    if "local_time_s" in raw:
        raw_signals["local_time_s"] = raw["local_time_s"]
    tables["raw_signals"] = pd.DataFrame(raw_signals)
    metadata = bundle["metadata"]
    for table in tables.values():
        for name in ("recording_id", "subject_id", "session_id", "channel"):
            if name not in table.columns:
                table[name] = metadata.get(name, "")
    usable_detection = detection.get("detector_eligible_duration_s", 0) > 0
    record_status = "succeeded" if usable_detection and bundle.get("diagnostics", {}).get("status") != "unavailable" else "partial"
    result = {"bundle": bundle, "tables": tables,
              "summary": {"record_key": _key(record), "recording_id": identifier, "status": record_status,
                          "reason": "" if record_status == "succeeded" else "no_qualified_detector_support",
                          "sample_count": len(time_array), "scr_count": len(scr), "qc": qc.get("summary", {}),
                          "decomposition": bundle.get("diagnostics", {}), "detection": detection,
                          "unavailable_features": int((features.get("status", pd.Series(dtype=str)) != "valid").sum())}}
    return _record_tables(result, associated)


def _pipeline_work_plan(records, config):
    """创建导入后准确的计划；多通道数量未知时由实际记录列表修订。"""
    units = []
    stage_totals = {name: 0 for name in ("qc", "preprocess", "decomposition", "features")}
    stage_methods = {"qc": "quality_mask", "preprocess": "preprocess",
                     "decomposition": config.get("decomposition", {}).get("method"),
                     "features": feature_work_method(config)}
    for record in records:
        metadata = record.get("metadata", {})
        recording_id = str(metadata.get("recording_id", ""))
        channel = str(metadata.get("channel") or "unknown")
        rate = metadata.get("fs_hz", metadata.get("fs_estimated"))
        for stage in stage_totals:
            units.append({"work_id": f"{recording_id}::{channel}:{stage}", "stage": stage,
                          "unit_type": "record_stage", "recording_id": recording_id,
                          "channel": channel, "method": stage_methods[stage], "rate_hz": rate})
            stage_totals[stage] += 1
    if config.get("synchrony", {}).get("enabled"):
        units.append({"work_id": "synchrony:group", "stage": "synchrony", "unit_type": "analysis_stage",
                      "method": synchrony_work_method(config),
                      "rate_hz": config["synchrony"].get("analysis_rate_hz")})
        stage_totals["synchrony"] = 1
    if config.get("statistics", {}).get("enabled"):
        units.append({"work_id": "statistics:planned", "stage": "statistics", "unit_type": "analysis_stage",
                      "method": "structured_statistics", "rate_hz": None})
        stage_totals["statistics"] = 1
    for output_format in config.get("output", {}).get("formats", []):
        units.append({"work_id": f"export:{output_format}", "stage": "export", "unit_type": "export_file",
                      "method": output_format, "rate_hz": None})
        stage_totals["export"] = stage_totals.get("export", 0) + 1
    units.append({"work_id": "run:manifest_finalization", "stage": "finalize", "unit_type": "manifest",
                  "method": "manifest_sha256", "rate_hz": None})
    stage_totals["finalize"] = 1
    return {"revision": 1, "total_units": len(units), "units": units,
            "eta_min_completed_units": config.get("runtime", {}).get("eta_min_completed_units", 3),
            "stage_totals": stage_totals}


def _combine(records):
    collected = {}
    for result in records:
        for name, table in result["tables"].items():
            # 附属事件链接已作为独立表提交；pandas concat会比较attrs，
            # DataFrame形式attrs的相等比较不是标量，因此在汇总边界移除attrs。
            frame = table.copy(deep=False)
            frame.attrs = {}
            collected.setdefault(name, []).append(frame)
    return {name: pd.concat(frames, ignore_index=True) for name, frames in collected.items()}


def _merge_group_temporal_results(combined, group_features, episodes, node_strength, diagnostics, config):
    """Preserve group audit tables and expose selected group metrics in the shared feature table."""
    selected = config.get("features", {}).get("ids", []) if isinstance(config, dict) else []
    feature_table = group_features.copy(deep=True) if isinstance(group_features, pd.DataFrame) else pd.DataFrame()
    if selected:
        feature_table = feature_table.loc[feature_table.get("feature_id", pd.Series(dtype=str)).isin(selected)].copy()
    combined["group_temporal_features"] = feature_table
    combined["sync_episodes"] = episodes.copy(deep=True) if isinstance(episodes, pd.DataFrame) else pd.DataFrame()
    combined["node_strength"] = node_strength.copy(deep=True) if isinstance(node_strength, pd.DataFrame) else pd.DataFrame()
    combined["group_temporal_diagnostics"] = diagnostics.copy(deep=True) if isinstance(diagnostics, pd.DataFrame) else pd.DataFrame()
    if not feature_table.empty:
        existing = combined.get("features", pd.DataFrame())
        if not isinstance(existing, pd.DataFrame):
            existing = pd.DataFrame(existing)
        combined["features"] = pd.concat([existing, feature_table], ignore_index=True, sort=False)
    return combined


def _input_hashes(records, tables, report):
    hashes = {}
    for record in records:
        meta = record["metadata"]
        if meta.get("source_path") and meta.get("source_sha256"):
            hashes[str(Path(meta["source_path"]).resolve())] = meta["source_sha256"]
    for table in tables.values():
        if isinstance(table, pd.DataFrame):
            path, digest = table.attrs.get("source_path"), table.attrs.get("source_sha256")
            if path and digest:
                hashes[str(Path(path).resolve())] = digest
    consumed = report.get("consumed_inputs", {}) if isinstance(report, dict) else {}
    if isinstance(consumed, dict):
        for path, digest in consumed.items():
            hashes[str(Path(path).resolve())] = digest if isinstance(digest, str) else digest.get("sha256")
    return {path: digest for path, digest in hashes.items() if digest}


def _normalized_input_fingerprints(values, *, source):
    if not isinstance(values, dict):
        fail("input_snapshot_invalid", f"{source}输入指纹必须是路径到SHA-256的对象。",
             category="input", stage="import", remediation="重新执行预检并用冻结输入快照入队。")
    normalized = {}
    for path, digest in values.items():
        if not isinstance(path, str) or not path.strip() or not isinstance(digest, str) or len(digest) != 64:
            fail("input_snapshot_invalid", f"{source}包含无效路径或SHA-256指纹。",
                 category="input", stage="import", field=str(path),
                 remediation="重新执行预检并用冻结输入快照入队。")
        if path.startswith("memory:"):
            # These keys bind in-memory manifest/signal rows through the frozen
            # config and report hash; they have no file bytes to compare here.
            continue
        try:
            canonical_path = os.path.normcase(str(Path(path).expanduser().resolve()))
        except (OSError, RuntimeError, ValueError) as error:
            fail("input_snapshot_invalid", f"{source}输入路径无法规范化：{error}。",
                 category="input", stage="import", field=path,
                 remediation="检查输入路径后重新预检并入队。")
        normalized[canonical_path] = digest.lower()
    return normalized


def _verify_consumed_input_fingerprints(expected, import_report):
    """Bind queued analysis to the exact bytes parsed by the importer."""
    if expected is None:
        return
    actual = import_report.get("consumed_inputs") if isinstance(import_report, dict) else None
    expected_values = _normalized_input_fingerprints(expected, source="冻结预检")
    actual_values = _normalized_input_fingerprints(actual, source="实际导入")
    changed = sorted(path for path in set(expected_values) | set(actual_values)
                     if expected_values.get(path) != actual_values.get(path))
    if changed:
        shown = changed[:5]
        fail("input_snapshot_mismatch", f"实际导入的输入字节与入队时冻结的预检指纹不同，分析已停止：{shown}。",
             category="input", stage="import", field=changed[0],
             remediation="检查字段中列出的输入是否在预检后变更；修复后重新预检并创建新的队列任务。")


def _summary_artifact_rows(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if (path.is_file() and not path.name.startswith(".") and
                path.name not in {"summary.json", "manifest.json", "pipeline_state.json"}):
            rows.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                         "sha256": sha256_file(path)})
    return rows


def run_pipeline(config, input_manifest=None, progress_callback=None, cancel_callback=None, *, _resume_from=None,
                 event_sink=None, runtime_session=None, project_id=None, task_id=None, attempt_id=None,
                 runtime_context=None):
    """运行同一条真实CPU流程，保留部分/失败/取消状态和可校验的记录缓存。"""
    from .core_io import load_recordings
    started = time.monotonic()
    resolved = _resolve_for_run(config)
    if input_manifest is not None and isinstance(input_manifest, (str, Path)):
        resolved["input"]["manifest"] = str(Path(input_manifest).resolve())
    require_capabilities(resolved)
    base = Path(resolved["input"].get("base_directory") or Path.cwd())
    output = Path(resolved["output"]["directory"])
    if not output.is_absolute():
        output = base / output
    writable = inspect_output_directory(output)
    if not writable["writable"]:
        fail("output_not_writable", "输出目录不可写，请选择有写入权限的新目录。", category="io", stage="preflight", field=str(output), remediation=writable["reason"])
    run_id = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    if runtime_session is not None and event_sink is None:
        event_sink = runtime_session.get("event_sink")
    runtime_project_id = project_id or (runtime_session.get("project_id") if isinstance(runtime_session, dict) else None)
    if runtime_project_id is None:
        runtime_project_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "eda-project:" + str(base.resolve())))
    work_states = {}
    work_state_lock = threading.Lock()
    base_event_sink = event_sink

    def tracking_event_sink(event):
        details = event.get("details") if isinstance(event, dict) else None
        work_id = details.get("work_id") if isinstance(details, dict) else None
        if work_id and event.get("kind") in {"completed", "failed", "skipped", "cancelled"}:
            with work_state_lock:
                work_states[str(work_id)] = event["kind"]
        if base_event_sink is not None:
            base_event_sink(event)

    event_sink = tracking_event_sink if base_event_sink is not None else None
    runtime_args = {"event_sink": event_sink, "run_id": run_id, "project_id": runtime_project_id,
                    "task_id": task_id, "attempt_id": attempt_id}
    root = output / run_id
    root.mkdir(parents=True, exist_ok=False)
    lock = root / ".running.lock"
    lock.write_text(run_id, encoding="ascii")
    summary = {"run_id": run_id, "status": "running", "directory": str(root.resolve()), "record_statuses": [],
               "errors": [], "warnings": [], "science_qualification": "pending_representative_real_data"}
    provenance = {"schema_version": "1.0", "config_sha256": config_hash(resolved), "resolved_config": resolved,
                  "source_fingerprint": source_fingerprint(),
                  "environment_fingerprint": numerical_environment_fingerprint(),
                  "environment": get_capabilities(), "input_hashes": {}, "method_versions": {},
                  "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}
    if isinstance(runtime_session, dict):
        provenance["runtime_session_id"] = runtime_session.get("session_id")
    completed, committed, cache_paths = [], {}, {}
    combined = {}
    work_plan = None
    export_completed = set()
    export_failed = set()
    _runtime_event(event_sink, stage="run", kind="run", message="正在冻结配置、读取输入",
                   run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                   attempt_id=attempt_id, details={"config_sha256": provenance["config_sha256"]})
    _notify(progress_callback, "start", "正在冻结配置、读取输入")
    atomic_json(root / "resolved_config.json", resolved)
    atomic_json(root / "pipeline_state.json", {**summary, "config_sha256": provenance["config_sha256"], "source_fingerprint": provenance["source_fingerprint"], "environment_fingerprint": provenance["environment_fingerprint"], "committed_records": {}})
    try:
        _cancel(cancel_callback)
        records, associated, import_report = load_recordings(input_manifest or resolved["input"].get("manifest"), resolved["mapping"], resolved)
        if isinstance(runtime_context, dict):
            _verify_consumed_input_fingerprints(runtime_context.get("expected_input_fingerprints"), import_report)
        if not records:
            fail("no_records", "没有可分析的信号记录。", field="input")
        if len({_key(record) for record in records}) != len(records):
            fail("duplicate_record", "recording_id 与 channel 联合标识重复。")
        work_plan = _pipeline_work_plan(records, resolved)
        runtime_args["units_total"] = work_plan["total_units"]
        runtime_args["stage_totals"] = work_plan["stage_totals"]
        plan_payload = {key: value for key, value in work_plan.items() if key != "stage_totals"}
        if len(plan_payload["units"]) > 10_000:
            plan_payload["units"] = []
            plan_payload["enumeration_truncated"] = True
        _runtime_event(event_sink, stage="plan", kind="plan_revised", message="输入读取完成，已按实际recording/channel冻结工作计划",
                       run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                       attempt_id=attempt_id, units_done=0, units_total=work_plan["total_units"],
                       unit_type="planned_work", details={"work_plan": plan_payload})
        _runtime_event(event_sink, stage="import", kind="completed", message="输入文件已读取并完成关系检查",
                       run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                       attempt_id=attempt_id, units_done=0, unit_type="input_file",
                       details={"record_count": len(records), "table_names": sorted(associated)})
        provenance["input_hashes"] = _input_hashes(records, associated, import_report)
        manifest_path = resolved["input"].get("manifest")
        if manifest_path:
            candidate = Path(manifest_path)
            if not candidate.is_absolute():
                candidate = base / candidate
            if candidate.is_file():
                source_key = str(candidate.resolve())
                consumed_hash = provenance["input_hashes"].get(source_key)
                if consumed_hash is None:
                    fail("manifest_hash_missing", "导入未提供已消费清单字节的哈希，无法确认输入快照。", stage="import")
                if sha256_file(candidate) != consumed_hash:
                    fail("input_mutated", "清单在读取后发生变化，已停止该运行；保留已消费字节的哈希。", stage="import")
        atomic_json(root / "import_report.json", import_report)
        memory_estimate = sum(len(record["time_s"]) for record in records) * 8 * 30
        if memory_estimate > resolved["compute"]["memory_limit_mb"] * 1024 ** 2:
            fail("memory_budget", "输入与处理缓冲估计超过配置内存预算；请显式调整预算或拆分批次。", category="resource")
        if resolved["synchrony"]["enabled"]:
            from .core_sync import apply_clock_alignment
            records, associated, alignment = apply_clock_alignment(records, associated, resolved, cancel_callback=cancel_callback)
            atomic_json(root / "clock_alignment.json", alignment)
            provenance["clock_alignment"] = alignment
        previous = None
        if _resume_from is not None:
            previous = json.loads((Path(_resume_from) / "pipeline_state.json").read_text(encoding="utf-8"))
            cache_paths = previous.get("committed_records", {})
        total = len(records)

        def consume(record, result=None, error=None):
            identifier = _key(record)
            if error is not None:
                if isinstance(error, InterruptedError):
                    raise error
                details = error_to_dict(error, stage="record", record_id=record["metadata"].get("recording_id", ""))
                summary["errors"].append(details)
                summary["record_statuses"].append({"record_key": identifier, "status": "failed", "error": details})
                _runtime_event(event_sink, stage=details.get("stage", "record"), kind="failed",
                               message=details.get("message", str(error)), run_id=run_id,
                               project_id=runtime_project_id, task_id=task_id, attempt_id=attempt_id,
                               record_id=str(record["metadata"].get("recording_id", "")),
                               unit_type="record", level="ERROR",
                               details={"record_key": identifier, "error": details})
            else:
                cache = root / "records" / safe_name(identifier)
                _save_cache(result, cache)
                completed.append(result)
                summary["record_statuses"].append(result["summary"])
                _runtime_event(event_sink, stage="record", kind="stage", message="记录处理结果已提交",
                               run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                               attempt_id=attempt_id,
                               record_id=str(record["metadata"].get("recording_id", "")),
                               details={"record_key": identifier, "record_status": result["summary"]["status"]})
                committed[identifier] = cache.relative_to(root).as_posix()
                atomic_json(root / "pipeline_state.json", {**summary, "config_sha256": provenance["config_sha256"],
                                                          "source_fingerprint": provenance["source_fingerprint"],
                                                          "environment_fingerprint": provenance["environment_fingerprint"],
                                                          "input_hashes": provenance["input_hashes"], "committed_records": committed})
            _notify(progress_callback, "record_done", "记录已处理", completed=len(summary["record_statuses"]), total=total, record_id=identifier)

        todo = []
        for record in records:
            identifier = _key(record)
            if identifier in cache_paths:
                result = _read_cache(Path(_resume_from) / cache_paths[identifier])
                result["summary"]["reused_from"] = str(_resume_from)
                metadata = record["metadata"]
                channel = str(metadata.get("channel") or "unknown")
                methods = {"qc": "quality_mask", "preprocess": "preprocess",
                           "decomposition": resolved["decomposition"]["method"],
                           "features": feature_work_method(resolved)}
                for stage, method in methods.items():
                    _runtime_event(event_sink, stage=stage, kind="completed", message="复用校验通过的已提交记录缓存",
                                   run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                                   attempt_id=attempt_id, record_id=str(metadata.get("recording_id", "")),
                                   units_done=1, units_total=work_plan["stage_totals"].get(stage),
                                   unit_type="record_stage",
                                   details={"work_id": f"{metadata.get('recording_id','')}::{channel}:{stage}",
                                            "method": method,
                                            "rate_hz": metadata.get("fs_hz", metadata.get("fs_estimated")),
                                            "record_status": result["summary"].get("status", "partial"),
                                            "reused_cache": True})
                consume(record, result=result)
            else:
                todo.append(record)
        workers = min(resolved["compute"]["workers"], len(todo) or 1)
        provenance["compute"] = {"backend": "cpu", "workers_requested": resolved["compute"]["workers"], "workers_actual": workers,
                                 "executor": "threads", "estimated_buffer_bytes": memory_estimate}
        if workers == 1:
            for record in todo:
                _cancel(cancel_callback)
                try:
                    scoped = _channel_scoped_associated(associated, record["metadata"].get("channel", "unknown"))
                    consume(record, result=_process_one(record, scoped, resolved, cancel_callback,
                                                        progress_callback, runtime_args))
                except InterruptedError:
                    raise
                except Exception as error:
                    consume(record, error=error)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eda-record") as executor:
                futures = {executor.submit(_process_one, record,
                                          _channel_scoped_associated(associated, record["metadata"].get("channel", "unknown")),
                                          resolved, cancel_callback, None, runtime_args): record for record in todo}
                for future in as_completed(futures):
                    _cancel(cancel_callback)
                    try:
                        consume(futures[future], result=future.result())
                    except InterruptedError:
                        for pending in futures:
                            pending.cancel()
                        raise
                    except Exception as error:
                        consume(futures[future], error=error)
        completed.sort(key=lambda result: result["summary"]["record_key"])
        combined = _combine(completed)
        _cancel(cancel_callback)
        incomplete = len(completed) != len(records) or any(item["summary"]["status"] != "succeeded" for item in completed)
        if resolved["synchrony"]["enabled"]:
            if incomplete:
                summary["warnings"].append("有必需成员失败，同步推断不可用；未自动改成幸存成员队列。")
                combined["synchrony"] = pd.DataFrame([{"status": "unavailable", "reason": "required_member_failed"}])
                from .core_sync import extract_group_temporal_features
                group_features, episodes, node_strength, group_diagnostics = extract_group_temporal_features(
                    pd.DataFrame(), pd.DataFrame(), {}, resolved)
                _merge_group_temporal_results(combined, group_features, episodes,
                                              node_strength, group_diagnostics, resolved)
                _runtime_event(event_sink, stage="synchrony", kind="skipped", message="必需成员不完整，同步阶段未执行",
                               run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                               attempt_id=attempt_id, units_done=0,
                               units_total=work_plan["stage_totals"].get("synchrony"),
                               unit_type="analysis_stage",
                               details={"work_id": "synchrony:group", "reason": "required_member_failed"})
            else:
                from .core_sync import (align_recordings, compute_pair_synchrony, estimate_sync_null,
                                        extract_group_temporal_features, summarize_group_synchrony)
                _notify(progress_callback, "synchrony", "计算共同网格上的有符号同步")

                def perform_synchrony():
                    bundles = [item["bundle"] for item in completed]
                    for bundle in bundles:
                        bundle["membership"] = associated.get("group_membership", pd.DataFrame())
                    aligned, alignment_report = align_recordings(bundles, associated.get("sync_anchors"), resolved,
                                                                 cancel_callback=cancel_callback)
                    atomic_json(root / "alignment_report.json", alignment_report)
                    aligned["membership"] = associated.get("group_membership", aligned.get("membership", pd.DataFrame()))
                    values, coverage = compute_pair_synchrony(aligned, None, resolved, cancel_callback=cancel_callback)
                    combined["synchrony"], combined["sync_coverage"] = values, coverage
                    combined["group_synchrony"] = summarize_group_synchrony(values, aligned, resolved,
                                                                             cancel_callback=cancel_callback)
                    group_features, episodes, node_strength, group_diagnostics = extract_group_temporal_features(
                        values, combined["group_synchrony"], aligned, resolved)
                    _merge_group_temporal_results(combined, group_features, episodes,
                                                  node_strength, group_diagnostics, resolved)
                    _cancel(cancel_callback)
                    if resolved["synchrony"]["null_method"] != "none":
                        null_summary, tails = estimate_sync_null(aligned, None, resolved,
                                                                 resolved["compute"]["seed"],
                                                                 cancel_callback=cancel_callback)
                        combined["sync_null_summary"], combined["sync_tail_probabilities"] = null_summary, tails
                    return None

                _run_work(event_sink, stage="synchrony", message="共同网格与多人同步",
                          work_id="synchrony:group", unit_type="analysis_stage", run_id=run_id,
                          project_id=runtime_project_id, task_id=task_id, attempt_id=attempt_id,
                          units_total=work_plan["stage_totals"].get("synchrony"),
                          method=synchrony_work_method(resolved),
                          rate_hz=resolved["synchrony"].get("analysis_rate_hz"), function=perform_synchrony)
        if resolved["statistics"]["enabled"]:
            if incomplete:
                combined["statistics"] = pd.DataFrame([{"status": "unavailable", "reason": "planned_cohort_incomplete"}])
                _runtime_event(event_sink, stage="statistics", kind="skipped", message="计划成员不完整，统计阶段未运行",
                               run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                               attempt_id=attempt_id, units_done=0,
                               units_total=work_plan["stage_totals"].get("statistics"),
                               unit_type="analysis_stage",
                               details={"work_id": "statistics:planned", "reason": "planned_cohort_incomplete"})
            else:
                from .core_statistics import run_statistics

                def perform_statistics():
                    estimates, diagnostics = run_statistics(combined.get("features", pd.DataFrame()),
                                                            associated.get("statistical_design"), resolved)
                    combined["statistics"], combined["statistics_diagnostics"] = estimates, diagnostics
                    return diagnostics

                diagnostics = _run_work(event_sink, stage="statistics", message="按预先声明的统计设计计算",
                                        work_id="statistics:planned", unit_type="analysis_stage", run_id=run_id,
                                        project_id=runtime_project_id, task_id=task_id, attempt_id=attempt_id,
                                        units_total=work_plan["stage_totals"].get("statistics"),
                                        method="structured_statistics", rate_hz=None,
                                        function=perform_statistics)
                if "status" in diagnostics and diagnostics["status"].eq("failed").any():
                    incomplete = True
                    summary["warnings"].append("部分计划统计不可估计；已保留失败假设与原多重比较家族。")
        summary["status"] = "partial" if incomplete and completed else "failed" if incomplete else "succeeded"
    except (InterruptedError, KeyboardInterrupt) as error:
        summary["status"] = "cancelled"
        summary["warnings"].append(str(error))
        combined = _combine(completed)
    except Exception as error:
        summary["status"] = "partial" if completed else "failed"
        summary["errors"].append(error_to_dict(error, stage="pipeline"))
        combined = _combine(completed)
    finally:
        summary["elapsed_s"] = time.monotonic() - started
        summary["completed_records"] = len(completed)
        summary["artifacts"] = []
        summary["source_integrity"] = "exact_consumed_bytes_hashed" if provenance["input_hashes"] else "not_consumed_or_import_failed"
        provenance["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        result = {"summary": summary, "tables": combined, "records": completed}
        manifest_finalized = False
        manifest = None
        try:
            if completed and "png" in resolved["output"]["formats"]:
                try:
                    stage_total = (work_plan or {}).get("stage_totals", {}).get("export", 1)

                    def make_plot_files():
                        from .core_plot import make_figures
                        return make_figures(result, resolved, root / "figures")

                    _run_work(event_sink, stage="export", message="PNG图表导出",
                              work_id="export:png", unit_type="export_file", run_id=run_id,
                              project_id=runtime_project_id, task_id=task_id, attempt_id=attempt_id,
                              units_total=stage_total, method="png", rate_hz=None, function=make_plot_files)
                    export_completed.add("png")
                except Exception as error:
                    export_failed.add("png")
                    summary["errors"].append(error_to_dict(error, stage="plot"))
                    if summary["status"] == "succeeded":
                        summary["status"] = "partial"
            elif "png" in resolved["output"]["formats"]:
                _runtime_event(event_sink, stage="export", kind="skipped", message="没有成功处理的记录，未生成PNG图表",
                               run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                               attempt_id=attempt_id, units_done=0,
                               units_total=(work_plan or {}).get("stage_totals", {}).get("export"),
                               unit_type="export_file", details={"work_id": "export:png", "method": "png",
                                                                  "reason": "no_completed_records"})

            def export_progress(update):
                kind = update.get("kind", "completed")
                work_id = update.get("work_id")
                output_format = update.get("format")
                if kind == "completed" and output_format:
                    export_completed.add(output_format)
                elif kind == "failed" and output_format:
                    export_failed.add(output_format)
                stage = update.get("stage", "export")
                unit_type = update.get("unit_type", "export_file")
                total_units = 1
                details = {"work_id": work_id, "method": update.get("method", output_format),
                           "rate_hz": None, "duration_s": update.get("duration_s")}
                done = 1 if kind == "completed" else 0
                _runtime_event(event_sink, stage=stage, kind=kind, message=update.get("message", ""),
                               run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                               attempt_id=attempt_id, units_done=done, units_total=total_units,
                               unit_type=unit_type, details=details,
                               level="ERROR" if kind == "failed" else "INFO")

            manifest = export_run(result, provenance, root, resolved, progress_callback=export_progress)
            if not verify_manifest(root).get("valid"):
                raise ValueError("初始输出清单未通过完整性检查。")
        except Exception as error:
            summary["errors"].append(error_to_dict(error, stage="export"))
            summary["status"] = "partial" if completed else "failed"
            _runtime_event(event_sink, stage="export", kind="failed", message=str(error) or "运行产物导出失败",
                           run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                           attempt_id=attempt_id, units_done=0, units_total=None,
                           unit_type="artifact_bundle", level="ERROR",
                           details={"error_type": type(error).__name__})

        if work_plan is not None:
            for output_format in resolved["output"]["formats"]:
                if output_format in export_completed or output_format in export_failed:
                    continue
                _runtime_event(event_sink, stage="export", kind="skipped",
                               message="运行未到达该格式导出阶段。", run_id=run_id,
                               project_id=runtime_project_id, task_id=task_id, attempt_id=attempt_id,
                               units_done=0, units_total=(work_plan or {}).get("stage_totals", {}).get("export"),
                               unit_type="export_file",
                               details={"work_id": f"export:{output_format}", "method": output_format,
                                        "reason": "export_not_reached"})

        if event_sink is not None and work_plan is not None:
            planned_units = work_plan.get("units", [])
            if isinstance(planned_units, list) and len(planned_units) <= 10_000:
                for unit in planned_units:
                    if not isinstance(unit, dict):
                        continue
                    work_id = str(unit.get("work_id", ""))
                    if work_id == "run:manifest_finalization":
                        continue
                    with work_state_lock:
                        state = work_states.get(work_id)
                    if not work_id or state is not None:
                        continue
                    stage = str(unit.get("stage", "pipeline"))
                    _runtime_event(event_sink, stage=stage, kind="skipped",
                                   message="运行结束前未执行该计划工作单位。", run_id=run_id,
                                   project_id=runtime_project_id, task_id=task_id,
                                   attempt_id=attempt_id, record_id=unit.get("recording_id"),
                                   units_done=0, units_total=work_plan["stage_totals"].get(stage),
                                   unit_type=str(unit.get("unit_type", "")),
                                   details={"work_id": work_id, "method": unit.get("method"),
                                            "rate_hz": unit.get("rate_hz"),
                                            "reason": "run_ended_before_unit"})
            elif isinstance(planned_units, list):
                _runtime_event(event_sink, stage="plan", kind="skipped",
                               message="大型工作计划中未执行单位的逐项状态未枚举。", run_id=run_id,
                               project_id=runtime_project_id, task_id=task_id,
                               attempt_id=attempt_id, units_done=0,
                               units_total=work_plan.get("total_units"), unit_type="planned_work",
                               details={"unreported_planned_units": max(0, len(planned_units) - len(work_states)),
                                        "reason": "plan_enumeration_truncated"})

        finishing_level = runtime_session.get("log_level", "INFO") if isinstance(runtime_session, dict) else "INFO"
        _runtime_event(event_sink, stage="finalize", kind="stage",
                       message="正在封存运行日志并提交最终摘要与完整性清单",
                       run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                       attempt_id=attempt_id, level=finishing_level,
                       details={"phase": "finishing", "terminal_status_pending": summary["status"]})
        if isinstance(runtime_session, dict) and callable(runtime_session.get("flush")):
            try:
                runtime_session["flush"]()
                from .runtime_logging import export_run_logs
                export_run_logs(runtime_session, run_id, root / "runtime")
            except Exception as error:
                summary["warnings"].append("运行日志无法复制到结果清单；项目持久日志仍保留。")
                summary["errors"].append(error_to_dict(error, stage="runtime_log_export"))
                summary["status"] = "partial" if completed else "failed"
                manifest_finalized = False

        def mark_finalization_error(error, stage):
            nonlocal manifest_finalized
            summary["status"] = "partial" if completed else "failed"
            summary["errors"].append(error_to_dict(error, stage=stage))
            manifest_finalized = False

        def final_pipeline_state():
            return {**summary, "config_sha256": provenance["config_sha256"],
                    "source_fingerprint": provenance["source_fingerprint"],
                    "environment_fingerprint": provenance["environment_fingerprint"],
                    "input_hashes": provenance["input_hashes"], "committed_records": committed}

        def persist_final_snapshot():
            nonlocal manifest, manifest_finalized
            # This timestamp includes analysis, plots, exports, and the copied run log.
            # The small final summary/manifest commit follows it and is intentionally
            # excluded to avoid rewriting the summary after its manifest hash is bound.
            summary["elapsed_s"] = time.monotonic() - started
            summary["elapsed_scope"] = "through_analysis_exports_and_runtime_log_snapshot_before_final_manifest_commit"
            summary["artifacts"] = _summary_artifact_rows(root)
            atomic_json(root / "summary.json", summary)
            atomic_json(root / "pipeline_state.json", final_pipeline_state())
            if (root / "manifest.json").is_file():
                manifest = refresh_manifest(root, run_id=run_id, status=summary["status"],
                                            config_sha256=provenance["config_sha256"],
                                            input_hashes=provenance["input_hashes"])
                verification = verify_manifest(root)
                manifest_finalized = bool(verification.get("valid") and
                                          manifest.get("status") == summary["status"])
                if not manifest_finalized:
                    raise RuntimeError("final_manifest_invalid_or_status_mismatch")
            else:
                manifest_finalized = False

        manifest_started = time.monotonic()
        try:
            persist_final_snapshot()
            if not manifest_finalized:
                raise RuntimeError("final_manifest_missing")
        except Exception as error:
            stage = "manifest_refresh" if (root / "manifest.json").is_file() else "finalization"
            mark_finalization_error(error, stage)
            try:
                # Commit an accurate partial/failed result after a one-shot late I/O
                # failure. A persistent filesystem failure leaves success unclaimed.
                persist_final_snapshot()
            except Exception as recovery_error:
                summary["errors"].append(error_to_dict(recovery_error, stage="finalization_recovery"))
                manifest_finalized = False
                try:
                    summary["elapsed_s"] = time.monotonic() - started
                    summary["artifacts"] = _summary_artifact_rows(root)
                    atomic_json(root / "summary.json", summary)
                    atomic_json(root / "pipeline_state.json", final_pipeline_state())
                except Exception as last_error:
                    summary["errors"].append(error_to_dict(last_error, stage="finalization_recovery"))

        try:
            lock.unlink(missing_ok=True)
        except Exception as error:
            mark_finalization_error(error, "run_lock_cleanup")
            summary["warnings"].append("运行锁无法在本次提交中删除。")
            try:
                persist_final_snapshot()
            except Exception as recovery_error:
                summary["errors"].append(error_to_dict(recovery_error, stage="finalization_recovery"))
                manifest_finalized = False
            try:
                lock.unlink(missing_ok=True)
            except Exception:
                pass

        if event_sink is not None and work_plan is not None:
            planned_units = work_plan.get("units", [])
            if isinstance(planned_units, list) and len(planned_units) <= 10_000:
                for unit in planned_units:
                    if not isinstance(unit, dict):
                        continue
                    work_id = str(unit.get("work_id", ""))
                    if work_id == "run:manifest_finalization":
                        continue
                    with work_state_lock:
                        state = work_states.get(work_id)
                    if not work_id or state is not None:
                        continue
                    stage = str(unit.get("stage", "pipeline"))
                    _runtime_event(event_sink, stage=stage, kind="skipped",
                                   message="运行结束前未执行该计划工作单位。", run_id=run_id,
                                   project_id=runtime_project_id, task_id=task_id,
                                   attempt_id=attempt_id, record_id=unit.get("recording_id"),
                                   units_done=0, units_total=work_plan["stage_totals"].get(stage),
                                   unit_type=str(unit.get("unit_type", "")),
                                   details={"work_id": work_id, "method": unit.get("method"),
                                            "rate_hz": unit.get("rate_hz"),
                                            "reason": "run_ended_before_unit"})
            elif isinstance(planned_units, list):
                _runtime_event(event_sink, stage="plan", kind="skipped",
                               message="大型工作计划中未执行单位的逐项状态未枚举。", run_id=run_id,
                               project_id=runtime_project_id, task_id=task_id,
                               attempt_id=attempt_id, units_done=0,
                               units_total=work_plan.get("total_units"), unit_type="planned_work",
                               details={"unreported_planned_units": max(0, len(planned_units) - len(work_states)),
                                        "reason": "plan_enumeration_truncated"})

        if manifest_finalized:
            _runtime_event(event_sink, stage="finalize", kind="completed",
                           message="运行产物完整性清单已最终写入并校验",
                           run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                           attempt_id=attempt_id, units_done=1,
                           units_total=(work_plan or {}).get("stage_totals", {}).get("finalize", 1),
                           unit_type="manifest",
                           details={"work_id": "run:manifest_finalization", "method": "manifest_sha256",
                                    "duration_s": time.monotonic() - manifest_started,
                                    "record_status": summary["status"]})
        else:
            _runtime_event(event_sink, stage="finalize", kind="failed",
                           message="最终运行清单未通过完整性提交",
                           run_id=run_id, project_id=runtime_project_id, task_id=task_id,
                           attempt_id=attempt_id, units_done=0, units_total=1,
                           unit_type="manifest", level="ERROR",
                           details={"work_id": "run:manifest_finalization", "method": "manifest_sha256",
                                    "record_status": summary["status"]})

        terminal_kind = {"succeeded": "completed", "partial": "completed",
                         "failed": "failed", "cancelled": "cancelled"}.get(summary["status"], "failed")
        _runtime_event(event_sink, stage="finished", kind=terminal_kind,
                       message="运行结束：" + summary["status"], run_id=run_id,
                       project_id=runtime_project_id, task_id=task_id, attempt_id=attempt_id,
                       details={"run_status": summary["status"], "manifest_finalized": manifest_finalized,
                                "elapsed_s": summary.get("elapsed_s", time.monotonic() - started)})
        if isinstance(runtime_session, dict) and callable(runtime_session.get("flush")):
            try:
                runtime_session["flush"]()
            except Exception:
                # The run snapshot ends at `finishing`; the terminal event remains
                # in the project session log and is not part of its own manifest.
                pass
        _notify(progress_callback, "finished", "运行结束：" + summary["status"], completed=len(completed), total=len(summary["record_statuses"]))
    return json_safe(summary)


def resume_run(directory, progress_callback=None, cancel_callback=None, *, event_sink=None,
               runtime_session=None, project_id=None, task_id=None, attempt_id=None, runtime_context=None):
    """校验已提交缓存与原始输入；恢复写入新运行，保留先前的取消/失败事实。"""
    root = Path(directory).resolve()
    if (root / ".running.lock").exists():
        fail("run_locked", "运行锁仍存在；不能同时写入或猜测锁已失效。", stage="resume",
             remediation="确认原进程退出，保留锁与日志并显式解除过期锁后重试。")
    state = json.loads((root / "pipeline_state.json").read_text(encoding="utf-8"))
    if state.get("source_fingerprint") != source_fingerprint():
        fail("resume_implementation_changed", "算法实现/版本已改变；请保留旧结果并以新运行重新计算。", stage="resume")
    if state.get("environment_fingerprint") != numerical_environment_fingerprint():
        fail("resume_environment_changed", "数值依赖/解释器/线程环境已变化，不能复用旧环境的缓存。", stage="resume")
    resolved = resolve_config(root / "resolved_config.json")
    if config_hash(resolved) != state["config_sha256"]:
        fail("resume_config_changed", "已保存的参数发生改变，不能作为同一任务恢复。", stage="resume")
    for path, expected in state.get("input_hashes", {}).items():
        if not Path(path).is_file() or sha256_file(path) != expected:
            fail("resume_input_changed", "已消费输入发生变化或缺失，必须作为新运行重新分析。", stage="resume", field=path)
    for relative in state.get("committed_records", {}).values():
        if not verify_manifest(root / relative)["valid"]:
            fail("resume_checksum", "记录缓存不完整，不能复用。", stage="resume")
    if state.get("status") == "succeeded":
        if not verify_manifest(root)["valid"]:
            fail("resume_manifest", "运行导出清单不完整。", stage="resume")
        if runtime_session is not None and event_sink is None:
            event_sink = runtime_session.get("event_sink")
        _runtime_event(event_sink, stage="finished", kind="completed", message="已有成功运行通过清单复核，无需重复计算",
                       run_id=state.get("run_id"), project_id=project_id or (runtime_session.get("project_id") if isinstance(runtime_session, dict) else None),
                       task_id=task_id, attempt_id=attempt_id,
                       details={"run_status": "succeeded", "manifest_finalized": True,
                                "resume_action": "already_complete_verified"})
        return {**state, "resume_action": "already_complete_verified"}
    return run_pipeline(resolved, progress_callback=progress_callback, cancel_callback=cancel_callback,
                        _resume_from=root, event_sink=event_sink, runtime_session=runtime_session,
                        project_id=project_id, task_id=task_id, attempt_id=attempt_id,
                        runtime_context=runtime_context)
