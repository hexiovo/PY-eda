"""共享的无GUI运行事件、进度适配与ETA估计。"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import time
import uuid

from .contracts import RUNTIME_EVENT_FIELDS, RUNTIME_EVENT_KINDS, RUNTIME_EVENT_SCHEMA_VERSION
from .errors import fail


_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_NULL_IDS = ("project_id", "task_id", "attempt_id", "run_id", "record_id")


def make_run_event(event):
    """补全事件固定字段；seq留给唯一controller写入。"""
    if not isinstance(event, dict):
        fail("runtime_event_type", "运行事件必须是普通字典。", category="runtime")
    extra = set(event) - set(RUNTIME_EVENT_FIELDS)
    if extra:
        fail("runtime_event_fields", f"运行事件包含未知字段：{sorted(extra)}。", category="runtime")
    result = {field: event.get(field) for field in RUNTIME_EVENT_FIELDS}
    result["schema_version"] = result["schema_version"] or RUNTIME_EVENT_SCHEMA_VERSION
    result["event_id"] = result["event_id"] or str(uuid.uuid4())
    result["seq"] = None if result["seq"] is None else result["seq"]
    result["timestamp_utc"] = result["timestamp_utc"] or datetime.now(timezone.utc).isoformat()
    result["monotonic_s"] = time.monotonic() if result["monotonic_s"] is None else result["monotonic_s"]
    result["level"] = str(result["level"] or "INFO").upper()
    for name in _NULL_IDS:
        value = result[name]
        if value is not None and not isinstance(value, str):
            fail("runtime_event_id", f"运行事件字段 {name} 必须是字符串或 null。", category="runtime", field=name)
    for name in ("stage", "kind", "message", "unit_type"):
        if result[name] is None:
            result[name] = ""
        if not isinstance(result[name], str):
            fail("runtime_event_field", f"运行事件字段 {name} 必须是字符串。", category="runtime", field=name)
    if result["schema_version"] != RUNTIME_EVENT_SCHEMA_VERSION:
        fail("runtime_event_version", "不支持该运行事件schema版本。", category="runtime")
    try:
        uuid.UUID(result["event_id"])
    except (ValueError, TypeError, AttributeError):
        fail("runtime_event_uuid", "event_id 必须是UUID字符串。", category="runtime", field="event_id")
    if result["seq"] is not None and (isinstance(result["seq"], bool) or not isinstance(result["seq"], int) or result["seq"] < 0):
        fail("runtime_event_seq", "seq 必须是非负整数或由controller暂留为null。", category="runtime", field="seq")
    if result["level"] not in _LEVELS:
        fail("runtime_event_level", f"level 必须为 {sorted(_LEVELS)} 之一。", category="runtime", field="level")
    if result["kind"] not in RUNTIME_EVENT_KINDS:
        fail("runtime_event_kind", f"kind 必须为 {list(RUNTIME_EVENT_KINDS)} 之一。", category="runtime", field="kind")
    try:
        timestamp = datetime.fromisoformat(result["timestamp_utc"].replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        fail("runtime_event_timestamp", "timestamp_utc 必须是ISO 8601时间。", category="runtime", field="timestamp_utc")
    if timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
        fail("runtime_event_timestamp", "timestamp_utc 必须显式使用UTC时区。", category="runtime", field="timestamp_utc")
    if isinstance(result["monotonic_s"], bool) or not isinstance(result["monotonic_s"], (int, float)) or not math.isfinite(result["monotonic_s"]):
        fail("runtime_event_monotonic", "monotonic_s 必须是有限秒数。", category="runtime", field="monotonic_s")
    for name in ("units_done", "units_total"):
        value = result[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            fail("runtime_event_units", f"{name} 必须是非负整数或null。", category="runtime", field=name)
    if not isinstance(result["details"], dict):
        fail("runtime_event_details", "details 必须是普通字典。", category="runtime", field="details")
    return result


def emit_run_event(event_sink, event):
    """把规范化事件送入sink；本函数不分配seq，也不写日志。"""
    if event_sink is None:
        return None
    if not callable(event_sink):
        fail("runtime_event_sink", "event_sink 必须是可调用对象。", category="runtime")
    event_sink(make_run_event(event))
    return None


def make_progress_callback(event_sink, *, project_id=None, task_id=None, attempt_id=None, run_id=None):
    """适配现有pipeline callback字典；只有明确完成的工作单位增加units_done。"""
    def callback(update):
        if not isinstance(update, dict):
            return
        stage = str(update.get("stage", "stage"))
        message = str(update.get("message", ""))
        kind = update.get("kind")
        if kind is None:
            kind = {
                "start": "run", "finished": "completed", "record_done": "completed",
                "cancel_requested": "cancel_requested", "cancelled": "cancelled",
                "warning": "warning", "resource_snapshot": "resource_snapshot",
            }.get(stage, "stage" if stage in {"qc", "preprocess", "decomposition", "features", "synchrony", "statistics", "export", "import", "finalize"} else "progress")
        completed = update.get("units_done")
        unit_type = update.get("unit_type") or ""
        total = update.get("units_total")
        record_id = update.get("record_id")
        details = dict(update.get("details") or {})
        if completed is None and stage == "record_done":
            completed = 1
            unit_type = "record"
            details.setdefault("work_id", f"record:{record_id or update.get('completed', '')}")
            details.setdefault("record_status", update.get("status", "succeeded"))
        elif completed is None:
            completed = 0
        if total is None:
            raw_total = update.get("total")
            total = raw_total if isinstance(raw_total, int) and not isinstance(raw_total, bool) and raw_total > 0 else None
        if stage == "finished":
            details.setdefault("run_status", update.get("run_status", "unknown"))
            details.setdefault("manifest_finalized", bool(update.get("manifest_finalized", False)))
        emit_run_event(event_sink, {
            "project_id": update.get("project_id", project_id),
            "task_id": update.get("task_id", task_id),
            "attempt_id": update.get("attempt_id", attempt_id),
            "run_id": update.get("run_id", run_id),
            "stage": stage,
            "record_id": str(record_id) if record_id is not None else None,
            "kind": kind,
            "level": update.get("level", "INFO"),
            "message": message,
            "units_done": completed,
            "units_total": total,
            "unit_type": unit_type,
            "details": details,
        })
    return callback


def _work_units(work_plan):
    if isinstance(work_plan, list):
        return work_plan
    if not isinstance(work_plan, dict):
        return None
    for key in ("units", "work_units", "items"):
        value = work_plan.get(key)
        if isinstance(value, list):
            return value
    return None


def _rate_key(row):
    return (row.get("method"), row.get("rate_hz", row.get("sampling_rate_hz")), row.get("unit_type"))


def _has_timing_key(key):
    # 非采样工作（例如统计/导出）可明确以rate_hz=null参与同类匹配。
    return bool(key[0]) and bool(key[2])


def _duration_samples(rows):
    values = []
    for row in rows:
        duration = row.get("duration_s", row.get("elapsed_s"))
        if isinstance(duration, (int, float)) and not isinstance(duration, bool) and math.isfinite(duration) and duration >= 0:
            values.append(float(duration))
    return values


def estimate_progress(events, work_plan, history):
    """基于实际完成事件与匹配历史耗时估计进度；不生成定时虚假百分比。"""
    event_rows = [row for row in (events or []) if isinstance(row, dict)]
    ordered_events = sorted(enumerate(event_rows), key=lambda item: (item[1].get("seq") is not None,
                                                                     item[1].get("seq") if item[1].get("seq") is not None else item[0]))
    latest = ordered_events[-1][1] if ordered_events else {}
    units = _work_units(work_plan)
    unit_by_id = {}
    if units is not None:
        for index, unit in enumerate(units):
            if isinstance(unit, dict):
                unit_by_id[str(unit.get("work_id", unit.get("id", index)))] = unit
    total = (work_plan.get("total_units") if isinstance(work_plan, dict) else len(unit_by_id))
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        total = None
    completed_ids = set()
    completed_by_type = {}
    completed_by_stage = {}
    completed_records = []
    failed = 0
    skipped = 0
    for _, event in ordered_events:
        kind = event.get("kind")
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        if kind in ("failed",):
            failed += 1
        elif kind == "skipped":
            skipped += 1
        elif kind == "completed":
            if details.get("record_status") == "failed":
                failed += 1
                continue
            work_id = details.get("work_id")
            if unit_by_id:
                if str(work_id) in unit_by_id:
                    completed_ids.add(str(work_id))
                    completed_records.append(event)
            else:
                amount = event.get("units_done")
                if isinstance(amount, int) and not isinstance(amount, bool) and amount > 0:
                    completed_records.append(event)
                    unit_type = event.get("unit_type") or "unknown"
                    completed_by_type[unit_type] = completed_by_type.get(unit_type, 0) + amount
                    stage_key = (event.get("stage") or "", unit_type)
                    completed_by_stage[stage_key] = completed_by_stage.get(stage_key, 0) + amount
    done = len(completed_ids) if unit_by_id else sum(completed_by_type.values())
    run_status = None
    manifest_finalized = False
    for _, event in reversed(ordered_events):
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        kind = event.get("kind")
        if kind in ("failed", "cancelled"):
            run_status = details.get("run_status", kind)
            manifest_finalized = details.get("manifest_finalized") is True
            break
        if (event.get("stage") == "finished" and kind == "completed" and
                details.get("run_status") in ("succeeded", "partial", "failed", "cancelled")):
            run_status = details["run_status"]
            manifest_finalized = details.get("manifest_finalized") is True
            break
    successful_manifest = run_status == "succeeded" and manifest_finalized
    percent = 100.0 if successful_manifest else (min(99.0, 100.0 * done / total) if total and total > 0 else None)

    stage = latest.get("stage") or None
    stage_units = [unit for unit in unit_by_id.values() if unit.get("stage") == stage]
    stage_total = len(stage_units) if units is not None else None
    stage_done = sum(1 for key in completed_ids if key in unit_by_id and unit_by_id[key].get("stage") == stage)
    if stage_total is None:
        stage_done = completed_by_stage.get((stage or "", latest.get("unit_type") or "unknown"), 0)
        if latest.get("units_total") is not None:
            stage_total = latest.get("units_total")
    if stage_total is None and latest.get("units_total") is not None:
        stage_total = latest.get("units_total")
    if latest.get("units_done") is not None and not unit_by_id:
        stage_done = max(stage_done, latest.get("units_done", 0))

    remaining_units = []
    if unit_by_id:
        remaining_units = [unit for key, unit in unit_by_id.items() if key not in completed_ids]
    current_samples = []
    if unit_by_id:
        for event in completed_records:
            details = event.get("details") if isinstance(event.get("details"), dict) else {}
            unit = unit_by_id.get(str(details.get("work_id")), {})
            current_samples.append({**unit, "duration_s": details.get("duration_s", event.get("duration_s")),
                                    "method": unit.get("method", details.get("method")),
                                    "rate_hz": unit.get("rate_hz", details.get("rate_hz")),
                                    "unit_type": event.get("unit_type", unit.get("unit_type"))})
    history_rows = [row for row in (history or []) if isinstance(row, dict)]
    by_key = {}
    for row in current_samples + history_rows:
        key = _rate_key(row)
        if _has_timing_key(key):
            by_key.setdefault(key, []).append(row)
    min_samples = work_plan.get("eta_min_completed_units", 3) if isinstance(work_plan, dict) else 3
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 3:
        min_samples = 3
    eta_seconds = 0.0
    eta_available = bool(remaining_units) and (total is not None or units is None)
    confidence = "low"
    reason = "work_plan_unknown" if units is None else "insufficient_matching_timing_history"
    if units is not None and not remaining_units:
        eta_available = False
        reason = "all_planned_units_complete"
    else:
        for unit in remaining_units:
            key = _rate_key(unit)
            samples = _duration_samples(by_key.get(key, [])) if _has_timing_key(key) else []
            if len(samples) < min_samples:
                eta_available = False
                break
            if len(samples) >= 2:
                alpha = 0.4
                estimate = samples[0]
                for sample in samples[1:]:
                    estimate = alpha * sample + (1 - alpha) * estimate
            else:
                estimate = samples[0]
            eta_seconds += estimate
        if eta_available:
            confidence = "moderate" if len(history_rows) >= min_samples else "low"
            reason = "matched_method_rate_unit_history_ewma"
    return {
        "stage": stage, "done": stage_done, "total": stage_total, "percent": percent,
        "eta_s": eta_seconds if eta_available else None,
        "eta_low_s": eta_seconds * 0.75 if eta_available else None,
        "eta_high_s": eta_seconds * 1.35 if eta_available else None,
        "confidence": confidence if eta_available else "none", "reason": reason,
        "global_done": done, "global_total": total,
        "failed_units": failed, "skipped_units": skipped,
        "plan_revision": work_plan.get("revision", 0) if isinstance(work_plan, dict) else None,
    }


__all__ = ["make_run_event", "emit_run_event", "make_progress_callback", "estimate_progress"]
