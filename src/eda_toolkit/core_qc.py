"""Quality facts, candidate intervals and append-only manual QC history."""

from __future__ import annotations

import copy
import math
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .contracts import QUALITY_FACTS
from .errors import fail


def _config_sections(config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(config, dict):
        return {}, {}
    qc = config.get("qc", config)
    mapping = config.get("mapping", {})
    return (dict(qc) if isinstance(qc, dict) else {}, dict(mapping) if isinstance(mapping, dict) else {})


def _as_annotations(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, (list, tuple)) and all(isinstance(item, dict) for item in value):
        return [dict(item) for item in value]
    fail("annotation_type", "annotations 必须是 DataFrame 或字典行列表。", field="annotations")


def _revision_number(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        fail("annotation_revision", f"{field} 必须是非负整数。", field=field)
    try:
        revision = int(value)
    except (TypeError, ValueError):
        fail("annotation_revision", f"{field} 必须是非负整数。", field=field)
    if revision < 0 or str(revision) != str(value).strip():
        fail("annotation_revision", f"{field} 必须是非负整数。", field=field)
    return revision


def _validated_history(history: Any) -> tuple[list[dict[str, Any]], int, set[str]]:
    entries = _as_annotations(history)
    ids: set[str] = set()
    active_targets: set[str] = set()
    targets: dict[str, dict[str, Any]] = {}
    previous_revision = 0
    normalized: list[dict[str, Any]] = []
    for position, raw in enumerate(entries, start=1):
        entry = dict(raw)
        annotation_id = entry.get("annotation_id")
        if annotation_id is None or not str(annotation_id).strip():
            fail("annotation_id", f"QC 历史第 {position} 条缺少 annotation_id。", field="annotation_id")
        annotation_id = str(annotation_id).strip()
        if annotation_id in ids:
            fail("annotation_duplicate", f"annotation_id 重复：{annotation_id!r}。", field="annotation_id")
        ids.add(annotation_id)
        revision = _revision_number(entry.get("revision"), field="revision")
        parent_revision = _revision_number(entry.get("parent_revision"), field="parent_revision")
        if revision != previous_revision + 1 or parent_revision != previous_revision:
            fail("annotation_revision_conflict",
                 f"QC 历史不连续：revision={revision}, parent_revision={parent_revision}, 期望接续 {previous_revision}。",
                 field="parent_revision")
        action = str(entry.get("action", "")).strip().casefold()
        if action not in ("exclude", "restore", "undo", "redo"):
            fail("annotation_action", f"不支持的 QC action：{action!r}。", field="action")
        if action in ("exclude", "restore"):
            for field in ("recording_id", "start_s", "end_s", "reason", "operator", "timestamp", "clock_domain"):
                if entry.get(field) is None or not str(entry.get(field)).strip():
                    fail("annotation_field", f"{action} 历史缺少必填字段 {field}。", field=field)
            try:
                start = float(entry["start_s"])
                end = float(entry["end_s"])
            except (TypeError, ValueError):
                fail("annotation_interval", "QC 时间区间端点必须是秒数。", field="start_s/end_s")
            if not math.isfinite(start) or not math.isfinite(end) or start >= end:
                fail("annotation_interval", "QC 区间必须是有限且非空的 [start_s,end_s)。", field="start_s/end_s")
            targets[annotation_id] = entry
            active_targets.add(annotation_id)
        else:
            target_id = entry.get("target_annotation_id")
            if target_id is None or str(target_id) not in targets:
                fail("annotation_target", f"{action} 引用未知 QC 标注。", field="target_annotation_id")
            target_id = str(target_id)
            if action == "undo":
                if target_id not in active_targets:
                    fail("annotation_undo_state", f"標注 {target_id!r} 已撤销，不能重复撤销。", field="target_annotation_id")
                active_targets.remove(target_id)
            else:
                if target_id in active_targets:
                    fail("annotation_redo_state", f"標注 {target_id!r} 当前未撤销，不能重做。", field="target_annotation_id")
                active_targets.add(target_id)
        entry["annotation_id"] = annotation_id
        entry["action"] = action
        entry["revision"] = revision
        entry["parent_revision"] = parent_revision
        normalized.append(entry)
        previous_revision = revision
    return normalized, previous_revision, active_targets


def append_annotation(
    history: Any,
    annotation: dict[str, Any],
    *,
    expected_parent_revision: Any,
) -> list[dict[str, Any]]:
    """以乐观并发检查追加 exclude/restore；不修改或覆盖传入历史。"""
    entries, current_revision, _ = _validated_history(history)
    expected = _revision_number(expected_parent_revision, field="expected_parent_revision")
    if expected != current_revision:
        fail("annotation_parent_conflict", f"标注父版本冲突：期望 {expected}，当前历史为 {current_revision}。",
             field="expected_parent_revision")
    if not isinstance(annotation, dict):
        fail("annotation_type", "annotation 必须是字典。", field="annotation")
    entry = copy.deepcopy(annotation)
    entry["revision"] = current_revision + 1
    entry["parent_revision"] = current_revision
    _validated_history([*entries, entry])
    return [*entries, entry]


def _append_history_command(
    history: Any,
    action: str,
    target_annotation_id: str,
    *,
    expected_parent_revision: Any,
    operator: str,
    reason: str,
    timestamp: str | None,
) -> list[dict[str, Any]]:
    entries, current_revision, active = _validated_history(history)
    expected = _revision_number(expected_parent_revision, field="expected_parent_revision")
    if expected != current_revision:
        fail("annotation_parent_conflict", f"标注父版本冲突：期望 {expected}，当前历史为 {current_revision}。",
             field="expected_parent_revision")
    target = str(target_annotation_id)
    if not any(entry["annotation_id"] == target for entry in entries):
        fail("annotation_target", f"未知 annotation_id：{target!r}。", field="target_annotation_id")
    if action == "undo" and target not in active:
        fail("annotation_undo_state", f"标注 {target!r} 已撤销，不能重复撤销。", field="target_annotation_id")
    if action == "redo" and target in active:
        fail("annotation_redo_state", f"标注 {target!r} 当前未撤销，不能重做。", field="target_annotation_id")
    if not str(operator).strip() or not str(reason).strip():
        fail("annotation_audit", "撤销/重做必须记录 operator 和 reason。", field="operator/reason")
    now = timestamp or datetime.now(timezone.utc).isoformat()
    target_entry = next(item for item in entries if item["annotation_id"] == target)
    entry = {
        "annotation_id": f"qc-{action}-{uuid.uuid4().hex}",
        "recording_id": target_entry.get("recording_id", ""),
        "channel": target_entry.get("channel"),
        "start_s": target_entry.get("start_s"),
        "end_s": target_entry.get("end_s"),
        "clock_domain": target_entry.get("clock_domain", ""),
        "action": action,
        "target_annotation_id": target,
        "reason": str(reason),
        "operator": str(operator),
        "timestamp": str(now),
        "revision": current_revision + 1,
        "parent_revision": current_revision,
    }
    return [*entries, entry]


def undo_annotation(
    history: Any,
    target_annotation_id: str,
    *,
    expected_parent_revision: Any,
    operator: str,
    reason: str,
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    """追加 undo 记录；原决定仍保留在审计历史中。"""
    return _append_history_command(history, "undo", target_annotation_id,
                                   expected_parent_revision=expected_parent_revision,
                                   operator=operator, reason=reason, timestamp=timestamp)


def redo_annotation(
    history: Any,
    target_annotation_id: str,
    *,
    expected_parent_revision: Any,
    operator: str,
    reason: str,
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    """追加 redo 记录；重做生成新 revision，不重写原决定。"""
    return _append_history_command(history, "redo", target_annotation_id,
                                   expected_parent_revision=expected_parent_revision,
                                   operator=operator, reason=reason, timestamp=timestamp)


def _boolean_intervals(mask: np.ndarray, time_s: np.ndarray, gap_limit_s: float | None) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        gap_split = index > 0 and gap_limit_s is not None and (
            not np.isfinite(time_s[index] - time_s[index - 1]) or time_s[index] - time_s[index - 1] > gap_limit_s
        )
        if start is not None and (not active or gap_split):
            intervals.append((start, index))
            start = None
        if active and start is None:
            start = index
    if start is not None:
        intervals.append((start, len(mask)))
    return intervals


def _flatline_mask(signal: np.ndarray, time_s: np.ndarray, valid_time: np.ndarray,
                   duration_s: float, tolerance_uS: float, gap_limit_s: float | None) -> np.ndarray:
    result = np.zeros(len(signal), dtype=bool)
    if duration_s <= 0:
        return result
    finite = np.isfinite(signal) & valid_time
    start = 0
    while start < len(signal):
        if not finite[start]:
            start += 1
            continue
        end = start + 1
        while end < len(signal):
            dt = time_s[end] - time_s[end - 1]
            same_run = finite[end] and np.isfinite(dt) and dt > 0 and (gap_limit_s is None or dt <= gap_limit_s)
            same_level = same_run and abs(signal[end] - signal[end - 1]) <= tolerance_uS
            if not same_level:
                break
            end += 1
        if end - start >= 2 and time_s[end - 1] - time_s[start] >= duration_s:
            result[start:end] = True
        start = max(end, start + 1)
    return result


def _active_annotation_facts(
    history: list[dict[str, Any]], active: set[str], record: dict[str, Any], time_s: np.ndarray,
    facts: dict[str, np.ndarray],
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = record.get("metadata", {})
    recording_id = str(metadata.get("recording_id", ""))
    channel = str(metadata.get("channel", "unknown"))
    clock_domain = str(metadata.get("clock_domain", ""))
    decisions = [entry for entry in history if entry["annotation_id"] in active and entry["action"] in ("exclude", "restore")]
    intervals: list[dict[str, Any]] = []
    manual_counts = {"exclude": 0, "restore": 0, "undo": 0, "redo": 0}
    for entry in history:
        if entry["action"] in manual_counts:
            manual_counts[entry["action"]] += 1
    for entry in decisions:
        if str(entry.get("recording_id")) != recording_id:
            continue
        annotation_channel = entry.get("channel")
        if annotation_channel is not None and not pd.isna(annotation_channel) and str(annotation_channel).strip():
            if str(annotation_channel).strip() not in (channel, "*"):
                continue
        annotation_clock = str(entry.get("clock_domain", "")).strip()
        if annotation_clock != clock_domain:
            fail("annotation_clock", f"manual QC clock_domain={annotation_clock!r} 与记录时钟 {clock_domain!r} 不同。",
                 record_id=recording_id, field="clock_domain")
        start, end = float(entry["start_s"]), float(entry["end_s"])
        selected = np.isfinite(time_s) & (time_s >= start) & (time_s < end)
        action = entry["action"]
        if action == "exclude":
            facts["manual_exclude"] |= selected
        else:
            facts["manual_restore"] |= selected
        intervals.append({
            "annotation_id": entry["annotation_id"], "action": action,
            "start_s": start, "end_s": end, "channel": annotation_channel or "*",
            "reason": str(entry.get("reason", "")), "operator": str(entry.get("operator", "")),
            "timestamp": str(entry.get("timestamp", "")), "revision": entry["revision"],
        })
    return (history[-1]["revision"] if history else 0), intervals, [
        {"action": action, "count": count} for action, count in manual_counts.items()
    ]


def build_quality_mask(record: dict[str, Any], config: Any, annotations: Any = None) -> dict[str, Any]:
    """基于原始样本生成 QC facts、自动候选、手工决定和连续有效段。"""
    if not isinstance(record, dict) or not isinstance(record.get("metadata"), dict):
        fail("record_contract", "QC 输入必须是包含 metadata 的 record 字典。", stage="qc")
    time_s = np.asarray(record.get("time_s", []), dtype=np.float64)
    signal = np.asarray(record.get("eda_uS", []), dtype=np.float64)
    if time_s.ndim != 1 or signal.ndim != 1 or time_s.shape != signal.shape or len(time_s) == 0:
        fail("record_shape", "QC 的 time_s 与 eda_uS 必须等长且为非空一维数组。", stage="qc")
    qc_config, _ = _config_sections(config)
    n = len(signal)
    inherited = record.get("qc_facts", record.get("facts", {}))
    if not isinstance(inherited, dict):
        inherited = {}
    facts = {name: np.zeros(n, dtype=bool) for name in QUALITY_FACTS}
    for name in QUALITY_FACTS:
        source = inherited.get(name)
        if source is not None:
            values = np.asarray(source, dtype=bool)
            if values.shape != (n,):
                fail("qc_fact_shape", f"继承的 QC fact {name} 长度不匹配。", field=name)
            facts[name] |= values
    facts["original_missing"] |= np.isnan(signal)
    facts["nonfinite"] |= np.isinf(signal)
    invalid_time = ~np.isfinite(time_s)
    facts["time_untrusted"] |= invalid_time
    dt = np.diff(time_s)
    invalid_dt = ~np.isfinite(dt) | (dt <= 0)
    if len(dt):
        facts["time_untrusted"][:-1] |= invalid_dt
        facts["time_untrusted"][1:] |= invalid_dt
    valid_time = np.isfinite(time_s) & ~facts["time_untrusted"]
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(positive_dt)) if len(positive_dt) else None
    try:
        gap_factor = float(qc_config.get("gap_factor", 1.5))
    except (TypeError, ValueError):
        fail("gap_factor", "gap_factor 必须是大于等于 1 的有限数。", field="gap_factor")
    if not math.isfinite(gap_factor) or gap_factor < 1:
        fail("gap_factor", "gap_factor 必须是大于等于 1 的有限数。", field="gap_factor")
    gap_limit_s = median_dt * gap_factor if median_dt is not None else None

    reasons: dict[str, np.ndarray] = {}
    finite_signal = np.isfinite(signal)
    reasons["original_missing"] = facts["original_missing"].copy()
    reasons["nonfinite"] = facts["nonfinite"].copy()

    for key, relation in (("min_uS", "below_min_uS"), ("max_uS", "above_max_uS")):
        threshold = qc_config.get(key)
        if threshold is not None:
            try:
                threshold_value = float(threshold)
            except (TypeError, ValueError):
                fail("qc_threshold", f"{key} 必须是数值。", field=key)
            mask = finite_signal & (signal < threshold_value if key == "min_uS" else signal > threshold_value)
            reasons[relation] = mask
    flatline_duration = qc_config.get("flatline_duration_s", 5.0)
    flatline_tolerance = qc_config.get("flatline_tolerance_uS", 0.0)
    try:
        flatline_duration = float(flatline_duration)
        flatline_tolerance = float(flatline_tolerance)
    except (TypeError, ValueError):
        fail("flatline_config", "平线时长与容差必须是数值。", field="flatline_duration_s/flatline_tolerance_uS")
    if not math.isfinite(flatline_duration) or flatline_duration < 0 or not math.isfinite(flatline_tolerance) or flatline_tolerance < 0:
        fail("flatline_config", "平线时长和噪声容差必须是有限非负数。", field="flatline_duration_s/flatline_tolerance_uS")
    reasons["flatline"] = _flatline_mask(signal, time_s, valid_time, flatline_duration, flatline_tolerance, gap_limit_s)

    for key, relation in (("max_slope_uS_s", "slope_limit"), ("max_jump_uS", "jump_limit")):
        threshold = qc_config.get(key)
        if threshold is None:
            continue
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            fail("qc_threshold", f"{key} 必须是数值。", field=key)
        if not math.isfinite(threshold_value) or threshold_value <= 0:
            fail("qc_threshold", f"{key} 必须是有限正数。", field=key)
        mask = np.zeros(n, dtype=bool)
        if len(dt):
            amplitude_delta = np.diff(signal)
            valid_pair = np.isfinite(dt) & (dt > 0) & (dt <= (gap_limit_s if gap_limit_s is not None else np.inf))
            measurement = np.abs(amplitude_delta) / dt if key == "max_slope_uS_s" else np.abs(amplitude_delta)
            candidates = valid_pair & np.isfinite(measurement) & (measurement > threshold_value)
            mask[:-1] |= candidates
            mask[1:] |= candidates
        reasons[relation] = mask

    # Acquisition gaps 是分段边界，不自动删除两侧样本，也不跨越该间隔统计变化率。
    gap_boundaries = []
    if len(dt) and gap_limit_s is not None:
        for index in np.flatnonzero(np.isfinite(dt) & (dt > gap_limit_s)):
            gap_boundaries.append({"left_index": int(index), "right_index": int(index + 1),
                                   "start_s": float(time_s[index]), "end_s": float(time_s[index + 1]),
                                   "gap_s": float(dt[index]), "reason": "gap_factor"})
    facts["auto_candidate"] |= np.logical_or.reduce(list(reasons.values())) if reasons else False

    annotation_source = annotations
    if annotation_source is None:
        annotation_source = record.get("annotations", qc_config.get("annotations", []))
    history, annotation_revision, active = _validated_history(annotation_source)
    manual_intervals_revision, manual_intervals, manual_counts = _active_annotation_facts(
        history, active, record, time_s, facts
    ) if history else (0, [], [{"action": action, "count": 0} for action in ("exclude", "restore", "undo", "redo")])
    if history and annotation_revision != manual_intervals_revision:
        fail("annotation_revision", "manual QC 历史修订号不一致。", field="revision")

    # 保留所有重叠原因；手工 restore 只覆盖自动候选，不复活原始缺失/非有限/不可信时间。
    facts["auto_candidate"] |= np.logical_or.reduce(list(reasons.values())) if reasons else False
    exclude_auto = bool(qc_config.get("exclude_auto", True))
    eligible = finite_signal & valid_time & ~facts["manual_exclude"] & ~facts["edge_affected"]
    if exclude_auto:
        eligible &= (~facts["auto_candidate"] | facts["manual_restore"])
    eligible &= ~facts["original_missing"] & ~facts["nonfinite"] & ~facts["time_untrusted"]

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, good in enumerate(eligible):
        gap_split = index > 0 and gap_limit_s is not None and (
            not np.isfinite(dt[index - 1]) or dt[index - 1] > gap_limit_s
        )
        if start is not None and (not good or gap_split):
            segments.append((start, index))
            start = None
        if good and start is None:
            start = index
    if start is not None:
        segments.append((start, n))

    intervals: list[dict[str, Any]] = []
    for reason_name, mask in reasons.items():
        for first, stop in _boolean_intervals(mask, time_s, gap_limit_s):
            end_point = float(time_s[stop - 1])
            # 用下一个可表示浮点数构造包住最后一个样本的半开区间，不外推采集尾部时长。
            end_exclusive = float(np.nextafter(end_point, np.inf)) if np.isfinite(end_point) else end_point
            intervals.append({"kind": "automatic_candidate", "reason": reason_name,
                              "start_s": float(time_s[first]), "end_s": end_exclusive,
                              "start_index": first, "end_index": stop})
    intervals.extend({"kind": "manual_decision", **item} for item in manual_intervals)

    finite_differences = dt[np.isfinite(dt) & (dt > 0)]
    raw_duration_s = float(np.sum(finite_differences[finite_differences <= gap_limit_s])) if gap_limit_s is not None else float(np.sum(finite_differences))
    usable_duration_s = 0.0
    observed_duration_s = 0.0
    if n > 1:
        valid_intervals = np.isfinite(dt) & (dt > 0)
        if gap_limit_s is not None:
            valid_intervals &= dt <= gap_limit_s
        usable_duration_s = float(np.sum(dt[valid_intervals & eligible[:-1] & eligible[1:]]))
        observed_mask = ~facts["original_missing"] & ~facts["nonfinite"] & valid_time
        observed_duration_s = float(np.sum(dt[valid_intervals & observed_mask[:-1] & observed_mask[1:]]))
        interpolated_mask = facts["interpolated"] & finite_signal & valid_time
        interpolated_duration_s = float(np.sum(dt[valid_intervals & interpolated_mask[:-1] & interpolated_mask[1:]]))
    else:
        interpolated_duration_s = 0.0
    missing_duration_s = max(0.0, raw_duration_s - observed_duration_s)
    reason_counts = {reason: int(np.count_nonzero(mask)) for reason, mask in reasons.items()}
    reason_counts.update({
        "manual_exclude": int(np.count_nonzero(facts["manual_exclude"])),
        "manual_restore": int(np.count_nonzero(facts["manual_restore"])),
        "edge_affected": int(np.count_nonzero(facts["edge_affected"])),
        "time_untrusted": int(np.count_nonzero(facts["time_untrusted"])),
    })
    summary = {
        "sample_count": int(n),
        "eligible_sample_count": int(np.count_nonzero(eligible)),
        "raw_duration_s": raw_duration_s,
        "usable_duration_s": usable_duration_s,
        "observed_duration_s": observed_duration_s,
        "interpolated_duration_s": interpolated_duration_s,
        "missing_duration_s": missing_duration_s,
        "missing_proportion_duration": missing_duration_s / raw_duration_s if raw_duration_s > 0 else None,
        "excluded_duration_s": max(0.0, raw_duration_s - usable_duration_s),
        "longest_gap_s": float(np.max(dt[np.isfinite(dt)])) if np.isfinite(dt).any() else None,
        "median_dt_s": median_dt,
        "gap_limit_s": gap_limit_s,
        "candidate_counts_by_reason": reason_counts,
        "manual_decisions": manual_counts,
    }
    return {
        "facts": facts,
        "eligible": eligible,
        "segments": segments,
        "intervals": intervals,
        "gap_boundaries": gap_boundaries,
        "candidate_reasons": {key: value.copy() for key, value in reasons.items()},
        "summary": summary,
        "annotation_revision": annotation_revision,
        "annotation_history": history,
        "report": {"valid": True, "diagnostics": [], "warnings": []},
    }
