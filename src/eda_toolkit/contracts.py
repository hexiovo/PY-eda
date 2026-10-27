"""数组/表格/字典数据契约；本模块不依赖任何 GUI 组件。"""
import copy
import numpy as np
import pandas as pd

from .errors import fail

RUN_STATUSES = ("planned", "running", "succeeded", "partial", "failed", "cancelled", "skipped", "not_applicable")
QUALITY_FACTS = ("original_missing", "nonfinite", "auto_candidate", "manual_exclude",
                 "manual_restore", "interpolated", "edge_affected", "time_untrusted")
SCR_COLUMNS = ["scr_id", "recording_id", "onset_s", "peak_s", "amplitude_uS", "rise_time_s",
               "recovery_s", "recovery_time_s", "area_uS_s", "status", "reason"]
WINDOW_COLUMNS = ["window_id", "recording_id", "kind", "start_s", "end_s", "event_id", "condition"]
FEATURE_COLUMNS = ["recording_id", "subject_id", "session_id", "window_id", "feature_id", "value",
                   "unit", "basis", "method", "status", "reason", "valid_n", "valid_duration_s"]
RUNTIME_EVENT_SCHEMA_VERSION = "1.0"
RUNTIME_EVENT_FIELDS = (
    "schema_version", "event_id", "seq", "timestamp_utc", "monotonic_s", "level",
    "project_id", "task_id", "attempt_id", "run_id", "stage", "record_id", "kind",
    "message", "units_done", "units_total", "unit_type", "details",
)
RUNTIME_EVENT_KINDS = (
    "run", "stage", "work_started", "progress", "completed", "failed", "skipped",
    "cancel_requested", "cancelled", "warning", "resource_snapshot", "plan_revised",
)
PREFLIGHT_ISSUE_FIELDS = ("code", "stage", "path", "message", "remediation", "blocking")
PREFLIGHT_REPORT_FIELDS = (
    "report_id", "report_sha256", "created_at", "config_hash", "input_fingerprints",
    "source_fingerprint", "numerical_environment_fingerprint", "capability_signature",
    "project_id", "project_revision", "draft_revision", "marker_revision", "qc_revision",
    "errors", "warnings", "capabilities", "resource_estimate", "work_plan", "eligible_to_enqueue",
)
PROJECT_STATE_REVISION_FIELDS = (
    "project_revision", "draft_revision", "marker_revision", "qc_revision",
)


def empty_table(columns):
    """返回有固定列名的空表，零响应与缺失结果均可被下游明确识别。"""
    return pd.DataFrame(columns=list(columns))


def validate_record(record):
    """校验物理电导记录；时间单位秒，幅值单位微西门子，NaN保留。"""
    if not isinstance(record, dict) or "metadata" not in record:
        fail("record_contract", "记录必须包含 metadata 普通字典。")
    time = np.asarray(record.get("time_s", []), dtype=np.float64)
    signal = np.asarray(record.get("eda_uS", []), dtype=np.float64)
    if time.ndim != 1 or signal.shape != time.shape or not len(time):
        fail("record_shape", "时间与电导必须是等长且非空的一维数组。")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        fail("invalid_timebase", "时间必须有限且严格递增；请先处理重复、倒序或缺失时间。", field="time_s")
    if np.any(np.isinf(signal)):
        fail("infinite_signal", "电导包含无穷值；请更正输入，缺失数据使用空值。", field="eda_uS")
    if record["metadata"].get("unit", "uS") != "uS":
        fail("physical_unit", "标准记录的电导必须已显式换算为 uS。")
    return record


def make_record(time_s, eda_uS, metadata, *, sample_index=None, extras=None):
    """复制输入构造记录，防止处理阶段原地修改原始电导或时间数组。"""
    time = np.array(time_s, dtype=np.float64, copy=True)
    signal = np.array(eda_uS, dtype=np.float64, copy=True)
    result = {"metadata": copy.deepcopy(metadata), "time_s": time, "eda_uS": signal,
              "sample_index": np.arange(len(time), dtype=np.int64) if sample_index is None else np.array(sample_index, dtype=np.int64, copy=True),
              "extras": pd.DataFrame(index=range(len(time))) if extras is None else extras.copy(deep=True)}
    validate_record(result)
    return result


def continuous_segments(mask, time_s=None, max_gap_s=None):
    """计算半开索引区间 [start,end)，可按真实时间间隔进一步断开。"""
    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 1:
        fail("mask_shape", "有效性标记必须是一维数组。")
    segments = []
    start = None
    time = None if time_s is None else np.asarray(time_s, dtype=float)
    for index, good in enumerate(valid):
        split = index > 0 and time is not None and max_gap_s is not None and time[index] - time[index - 1] > max_gap_s
        if start is not None and (not good or split):
            segments.append((start, index))
            start = None
        if good and start is None:
            start = index
    if start is not None:
        segments.append((start, len(valid)))
    return segments


def valid_duration(time_s, eligible, *, max_gap_s=None):
    """点采样暴露时长：只累计相邻有效点，不外推尾点、不跨缺口。"""
    time = np.asarray(time_s, dtype=float)
    mask = np.asarray(eligible, dtype=bool)
    if len(time) < 2:
        return 0.0
    dt = np.diff(time)
    valid = mask[:-1] & mask[1:] & np.isfinite(dt) & (dt > 0)
    if max_gap_s is not None:
        valid &= dt <= max_gap_s
    return float(np.sum(dt[valid]))
