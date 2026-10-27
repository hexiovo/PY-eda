"""分段式 EDA 预处理；物理电导始终与独立变换分支分开保存。"""

from __future__ import annotations

from fractions import Fraction
import copy
import math

import numpy as np
from scipy import interpolate, ndimage, signal

from .config import DEFAULT_CONFIG
from .contracts import continuous_segments, validate_record
from .errors import fail


_FACT_NAMES = ("original_missing", "nonfinite", "auto_candidate", "manual_exclude",
               "manual_restore", "interpolated", "edge_affected", "time_untrusted")


def _section(config, name):
    """读取完整配置或单个 section，复制默认值以便旧调用方仍可传子配置。"""
    if config is None:
        incoming = {}
    elif not isinstance(config, dict):
        fail("numeric_config", "数值接口配置必须是字典。", category="configuration")
    else:
        candidate = config.get(name, config) if name in config else config
        incoming = candidate if isinstance(candidate, dict) else {}
    merged = dict(DEFAULT_CONFIG[name])
    merged.update(incoming)
    return merged


def _full_config(config):
    """返回逐节合并的只读配置视图，不调用会拒绝可选研究键的全局校验器。"""
    result = {}
    for name, value in DEFAULT_CONFIG.items():
        if isinstance(value, dict):
            incoming = config.get(name, {}) if isinstance(config, dict) else {}
            result[name] = {**value, **(incoming if isinstance(incoming, dict) else {})}
        else:
            result[name] = value
    return result


def _bool_array(values, n, name, default=False):
    if values is None:
        return np.full(n, default, dtype=bool)
    array = np.asarray(values, dtype=bool)
    if array.ndim != 1 or len(array) != n:
        fail("qc_shape", f"QC 标记 {name} 必须与记录等长的一维布尔数组。", field=name)
    return np.array(array, copy=True)


def _qc_arrays(qc, n, finite):
    """接受 qc['facts'] 或平铺标记；绝不就地改写 QC 输入。"""
    qc = qc if isinstance(qc, dict) else {}
    facts = qc.get("facts", {})
    facts = facts if isinstance(facts, dict) else {}
    arrays = {name: _bool_array(facts.get(name, qc.get(name)), n, name) for name in _FACT_NAMES}
    arrays["nonfinite"] |= ~finite
    arrays["original_missing"] |= ~finite
    supplied = qc.get("eligible")
    if supplied is not None:
        eligible = _bool_array(supplied, n, "eligible")
    else:
        eligible = finite & ~arrays["manual_exclude"] & ~(arrays["auto_candidate"] & ~arrays["manual_restore"])
    eligible &= finite & ~arrays["manual_exclude"] & ~arrays["time_untrusted"]
    eligible &= ~(arrays["auto_candidate"] & ~arrays["manual_restore"])
    arrays["eligible"] = eligible
    return arrays


def _nominal_dt(time, mask=None, gap_factor=1.5):
    """以有效相邻间隔的中位数估计采样周期；较大的缺口不参与估计。"""
    time = np.asarray(time, dtype=float)
    if len(time) < 2:
        return None
    dt = np.diff(time)
    valid = np.isfinite(dt) & (dt > 0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        valid &= mask[:-1] & mask[1:]
    candidates = dt[valid]
    if not len(candidates):
        candidates = dt[np.isfinite(dt) & (dt > 0)]
    if not len(candidates):
        return None
    median = float(np.median(candidates))
    local = candidates[candidates <= median * max(float(gap_factor), 1.000001)]
    return float(np.median(local)) if len(local) else median


def _gap_limit(time, mask, qc_cfg, mapping_cfg):
    explicit = None
    if isinstance(qc_cfg, dict):
        explicit = qc_cfg.get("max_gap_s")
    if explicit is not None:
        if not np.isfinite(explicit) or explicit <= 0:
            fail("gap_limit", "max_gap_s 必须为正有限秒数。", field="max_gap_s")
        return float(explicit)
    dt = _nominal_dt(time, mask, qc_cfg.get("gap_factor", 1.5))
    if dt is None:
        return None
    return dt * float(qc_cfg.get("gap_factor", 1.5))


def _segments(mask, time, max_gap_s):
    return continuous_segments(mask, time_s=time, max_gap_s=max_gap_s)


def _sampling_rate(time, mask, config, *, metadata=None, require_regular=True):
    """验证连续段内 jitter；只在显式容差内按均匀采样解释时间。"""
    full = _full_config(config)
    preprocess = full["preprocess"]
    qc_cfg = full["qc"]
    mapping = full["mapping"]
    metadata = metadata if isinstance(metadata, dict) else {}
    max_gap = _gap_limit(time, mask, qc_cfg, mapping)
    segments = _segments(mask, time, max_gap)
    deltas = []
    for start, end in segments:
        if end - start > 1:
            deltas.extend(np.diff(np.asarray(time)[start:end]).tolist())
    tolerance = preprocess.get("jitter_tolerance", mapping.get("jitter_tolerance", 0.01))
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
        fail("jitter_tolerance", "jitter_tolerance 必须为非负有限比例。", category="configuration")

    local_declared = mapping.get("sampling_rate_hz")
    clock_domain = metadata.get("clock_domain", "recording_local")
    time_mapping = metadata.get("time_mapping", {})
    scale = time_mapping.get("b", time_mapping.get("scale")) if isinstance(time_mapping, dict) else None
    if scale is None:
        scale = metadata.get("time_scale_b")
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float)) or
                              not np.isfinite(scale) or scale <= 0):
        fail("time_mapping_scale", "time mapping scale b 必须为正有限值。", field="metadata.time_mapping.b")

    if metadata.get("processed_grid") is True:
        declared = metadata.get("fs_hz", metadata.get("sampling_rate_hz"))
        if isinstance(declared, bool) or not isinstance(declared, (int, float)) or not np.isfinite(declared) or declared <= 0:
            fail("metadata_sampling_rate", "processed-grid metadata 需要正有限 fs_hz。", field="metadata.fs_hz")
        reference = 1.0 / float(declared)
    elif clock_domain == "reference":
        reference_rates = [metadata.get(key) for key in ("fs_reference_hz", "fs_hz") if metadata.get(key) is not None]
        local_rates = [metadata.get(key) for key in ("fs_local_hz", "source_fs_hz") if metadata.get(key) is not None]
        for value in reference_rates + local_rates:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                fail("metadata_sampling_rate", "reference/local sampling rate metadata 必须为正有限 Hz。", field="metadata")
        if local_declared is not None and local_rates and abs(float(local_rates[0]) - float(local_declared)) / float(local_declared) > tolerance:
            fail("sampling_rate_mismatch", "mapping.sampling_rate_hz 与 reference metadata 的 fs_local_hz 不一致。",
                 field="mapping.sampling_rate_hz")
        reference_rate = float(reference_rates[0]) if reference_rates else None
        if len(reference_rates) > 1 and abs(float(reference_rates[0]) - float(reference_rates[1])) / float(reference_rates[0]) > tolerance:
            fail("sampling_rate_mismatch", "metadata.fs_reference_hz 与 metadata.fs_hz 不一致。", field="metadata")
        local_rate = float(local_rates[0]) if local_rates else float(local_declared) if local_declared is not None else None
        if reference_rate is not None:
            if scale is not None and local_rate is not None:
                mapped_rate = local_rate / float(scale)
                if abs(reference_rate - mapped_rate) / reference_rate > tolerance:
                    fail("sampling_rate_mismatch", "metadata reference fs 与 fs_local/b 不一致。", field="metadata.fs_reference_hz")
            reference = 1.0 / reference_rate
        elif scale is not None and local_rate is not None:
            reference = 1.0 / (local_rate / float(scale))
        else:
            # 已在 reference 时钟而缺少变换采样率时，不能把本地 mapping fs 套到新时间轴。
            reference = None
    else:
        declared = local_declared
        if declared is None:
            declared = metadata.get("fs_hz", metadata.get("sampling_rate_hz"))
        if declared is not None and (isinstance(declared, bool) or not isinstance(declared, (int, float)) or
                                     not np.isfinite(declared) or declared <= 0):
            fail("metadata_sampling_rate", "声明采样率必须为正有限 Hz。", field="sampling_rate_hz")
        reference = 1.0 / float(declared) if declared is not None else None

    if not deltas:
        if reference is not None:
            return 1.0 / reference, segments, max_gap
        fail("sampling_rate_unknown", "有效连续段不足两个样本，无法估计采样率。", field="time_s")
    deltas = np.asarray(deltas, dtype=float)
    median = float(np.median(deltas))
    if reference is None:
        reference = median
    relative_deviation = np.abs(deltas - reference) / reference
    if require_regular and np.any(relative_deviation > tolerance + 1e-12):
        worst = float(np.max(relative_deviation))
        fail("irregular_timebase", f"连续段内最大采样间隔偏差 {worst:.6g} 超过显式容差 {tolerance:.6g}；不会按均匀采样滤波或重采样。",
             field="time_s", remediation="检查时钟/采样率，或在经批准配置中设定 jitter_tolerance。")
    return 1.0 / reference, segments, max_gap


def _sos_padlen(sos):
    """复现 SciPy sosfiltfilt 的计算性最小填充长度规则。"""
    zeros_at_origin = min(int((sos[:, 2] == 0).sum()), int((sos[:, 5] == 0).sum()))
    return 3 * (2 * len(sos) + 1 - zeros_at_origin)


def _settling_radius(sos, tolerance):
    """用最慢极点的几何衰减上界估计 IIR 保守影响半径（样本）。"""
    _, poles, _ = signal.sos2zpk(sos)
    radius = float(np.max(np.abs(poles))) if len(poles) else 0.0
    if not 0.0 < radius < 1.0:
        return 0
    budget = max(float(tolerance) / max(2 * len(poles), 1), np.finfo(float).tiny)
    return max(1, int(math.ceil(math.log(budget) / math.log(radius))))


def _mark_radius(target, center_mask, radius, segments):
    if radius <= 0:
        return
    for start, end in segments:
        local = np.flatnonzero(center_mask[start:end]) + start
        for index in local:
            target[max(start, index - radius):min(end, index + radius + 1)] = True


def _resample_arrays(time, values, quality, process_mask, target_hz, old_hz, segments):
    """分段 polyphase 抗混叠重采样，并按 FIR 有限支撑传播坏值/边缘影响。"""
    ratio = Fraction(float(target_hz) / float(old_hz)).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    if up < 1 or down < 1:
        fail("resample_ratio", "无法建立正有理重采样倍率。")
    max_rate = max(up, down)
    half_len = 0 if max_rate == 1 else 10 * max_rate
    taps = np.array([1.0]) if half_len == 0 else signal.firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0))
    support_samples = half_len / up
    out_t, out_y = [], []
    out_q = {name: [] for name in quality}
    out_q["edge_affected"] = []
    out_q["eligible"] = []
    out_q["processing_eligible"] = []
    out_q["analysis_eligible"] = []
    out_q["detector_eligible"] = []
    out_q["interpolated"] = []
    out_q["artifact"] = []
    out_q["source_eligible"] = []
    for start, end in segments:
        if end <= start:
            continue
        x = np.asarray(values[start:end], dtype=float)
        if len(x) == 1 or max_rate == 1:
            y = x.copy() if len(x) == 1 else signal.resample_poly(x, up, down, window=taps)
        else:
            y = signal.resample_poly(x, up, down, window=taps, padtype="reflect")
        local_t = np.asarray(time[start:end], dtype=float)
        new_t = local_t[0] + np.arange(len(y), dtype=float) / float(target_hz)
        keep = new_t <= local_t[-1] + 1e-12
        y, new_t = y[keep], new_t[keep]
        if not len(y):
            continue
        out_t.append(new_t)
        out_y.append(y)
        radius = int(math.ceil(support_samples))
        for j in range(len(y)):
            source_position = j * down / up
            lo = max(0, int(math.floor(source_position - support_samples)))
            hi = min(end - start, int(math.ceil(source_position + support_samples)) + 1)
            source_indices = np.arange(start + lo, start + hi, dtype=int)
            support_inside = source_position - support_samples >= 0 and source_position + support_samples <= end - start - 1
            edge = not support_inside
            for fact in _FACT_NAMES:
                if fact == "edge_affected":
                    edge |= bool(np.any(quality["edge_affected"][source_indices]))
                elif fact == "interpolated":
                    continue
                elif fact in quality:
                    out_q[fact].append(bool(np.any(quality[fact][source_indices])))
            interpolated = bool(np.any(quality["interpolated"][source_indices]))
            eligible = bool(np.all(quality["eligible"][source_indices])) and not edge
            process_ok = bool(np.all(quality["processing_eligible"][source_indices]))
            detector = bool(np.all(quality["detector_eligible"][source_indices])) and not edge
            out_q["interpolated"].append(interpolated)
            out_q["edge_affected"].append(edge)
            out_q["eligible"].append(eligible)
            out_q["processing_eligible"].append(process_ok)
            out_q["analysis_eligible"].append(eligible)
            out_q["detector_eligible"].append(detector)
            out_q["artifact"].append(bool(np.any(quality["artifact"][source_indices])))
            out_q["source_eligible"].append(bool(np.all(quality["source_eligible"][source_indices])))
    if not out_t:
        return np.array([], float), np.array([], float), {k: np.array([], dtype=bool) for k in out_q}, {
            "up": up, "down": down, "fir_taps": len(taps), "support_half_width_input_samples": support_samples,
            "status": "unavailable", "reason": "no_process_eligible_segment"}
    result_q = {key: np.asarray(value, dtype=bool) for key, value in out_q.items()}
    return np.concatenate(out_t), np.concatenate(out_y), result_q, {
        "up": up, "down": down, "fir_taps": len(taps), "support_half_width_input_samples": support_samples,
        "status": "succeeded", "reason": ""}


def preprocess_record(record, qc, config):
    """按固定顺序对记录作分段预处理并返回新字典，不改写输入数组。

    插值只填两侧有效且全因缺失的短缺口，最大跨度由 `interpolate_max_gap_s` 秒限制；
    平滑和低通只在有效连续段内处理。低通为单程 Butterworth SOS 再前后向运行，
    设计截止单位 Hz，正反向总响应为单程幅频响应平方；padding 仅供数值计算，
    另以最慢极点衰减上界和 `edge_tolerance` 排除边缘。resample_poly 用有理倍率，
    抗混叠 FIR 支撑穿过边缘/不可用输入的输出会标记为无效。基线区间按记录秒数
    `[start,end)` 减均值，保留 offset；zscore/range/log 只写入 `analysis_signal`，
    其物理 `clean_uS` 不被缩放。返回含原 `record` 快照、处理时间/信号、布尔质量数组
    和逐操作报告；不外推、不跨缺口，时间抖动超过显式比例容差时拒绝均匀滤波。
    """
    validate_record(record)
    if not isinstance(qc, dict):
        fail("qc_contract", "预处理 QC 必须为事实/有效性字典。")
    full = _full_config(config)
    p, qc_cfg, mapping = full["preprocess"], full["qc"], full["mapping"]
    source_time = np.asarray(record["time_s"], dtype=float)
    source = np.asarray(record["eda_uS"], dtype=float)
    n = len(source)
    finite = np.isfinite(source)
    quality = _qc_arrays(qc, n, finite)
    quality["source_eligible"] = quality["eligible"].copy()
    quality["interpolated"] |= _bool_array(qc.get("facts", {}).get("interpolated") if isinstance(qc.get("facts"), dict) else None,
                                           n, "interpolated")
    operations = []
    clean = np.array(source, dtype=float, copy=True)
    was_missing = quality["original_missing"] | ~finite
    other_bad = quality["manual_exclude"] | quality["time_untrusted"] | (quality["auto_candidate"] & ~quality["manual_restore"])
    # 仅来源明确为缺失且没有其他坏因的 NaN 允许插值；flatline/异常值不会进入该分支。
    fillable = was_missing & ~other_bad
    interp_max = float(p.get("interpolate_max_gap_s", 0.0) or 0.0)
    interpolation = str(p.get("interpolation", "linear"))
    filled = np.zeros(n, dtype=bool)
    if interp_max > 0:
        missing_runs = continuous_segments(fillable)
        for start, end in missing_runs:
            left, right = start - 1, end
            if left < 0 or right >= n or not quality["eligible"][left] or not quality["eligible"][right]:
                continue
            if not finite[left] or not finite[right]:
                continue
            span = float(source_time[right] - source_time[left])
            if span > interp_max + 1e-12:
                continue
            if interpolation == "pchip":
                lefts = np.arange(max(0, left - 2), left + 1)
                rights = np.arange(right, min(n, right + 3))
                support = np.concatenate((lefts, rights))
                if len(lefts) < 2 or len(rights) < 2 or not np.all(quality["eligible"][support]) or not np.all(finite[support]):
                    continue
                x = source_time[support]
                y = source[support]
                clean[start:end] = interpolate.PchipInterpolator(x, y, extrapolate=False)(source_time[start:end])
            else:
                clean[start:end] = np.interp(source_time[start:end], [source_time[left], source_time[right]],
                                             [source[left], source[right]])
            filled[start:end] = True
    quality["interpolated"] |= filled
    process_eligible = (quality["eligible"] | filled |
                        (quality["auto_candidate"] & quality["manual_restore"])) & (finite | filled)
    process_eligible &= ~quality["manual_exclude"] & ~quality["time_untrusted"]
    process_eligible &= ~(quality["auto_candidate"] & ~quality["manual_restore"])
    quality["processing_eligible"] = process_eligible.copy()
    quality["edge_affected"] |= _bool_array(qc.get("facts", {}).get("edge_affected") if isinstance(qc.get("facts"), dict) else None,
                                              n, "edge_affected")
    quality["artifact"] = (quality["auto_candidate"] & ~quality["manual_restore"]) | quality["manual_exclude"]
    gap_limit = _gap_limit(source_time, process_eligible, qc_cfg, mapping)
    segments = _segments(process_eligible, source_time, gap_limit)

    uniform_ops = p.get("smoothing", "none") != "none" or p.get("lowpass_hz") is not None or p.get("resample_hz") is not None
    fs = None
    n_process = int(np.count_nonzero(process_eligible))
    if uniform_ops or n_process >= 2:
        fs, segments, gap_limit = _sampling_rate(source_time, process_eligible, config,
                                                 metadata=record.get("metadata"), require_regular=True)
    else:
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        reference_fs = metadata.get("fs_reference_hz") if metadata.get("clock_domain") == "reference" else None
        declared = reference_fs or metadata.get("fs_hz") or mapping.get("sampling_rate_hz") or metadata.get("sampling_rate_hz")
        dt = _nominal_dt(source_time, process_eligible, qc_cfg.get("gap_factor", 1.5))
        fs = float(declared) if declared is not None else 1.0 / dt if dt else None

    # 缺失填补受后续滤波核影响；只传播标记，不把短插值区伪装成原始观测。
    if p.get("smoothing") != "none":
        if not fs or fs <= 0:
            fail("sampling_rate_unknown", "平滑需要可验证的正采样率。", field="time_s")
        width = max(1, int(round(float(p.get("smoothing_window_s", 0.5)) * fs)))
        edge_radius = width // 2
        for start, end in segments:
            if p["smoothing"] == "moving":
                clean[start:end] = ndimage.uniform_filter1d(clean[start:end], size=width, mode="nearest")
            elif p["smoothing"] == "median":
                clean[start:end] = ndimage.median_filter(clean[start:end], size=width, mode="nearest")
            else:
                fail("smoothing_method", f"不支持平滑方法：{p['smoothing']}", category="configuration")
            if edge_radius:
                quality["edge_affected"][start:min(end, start + edge_radius)] = True
                quality["edge_affected"][max(start, end - edge_radius):end] = True
        _mark_radius(quality["edge_affected"], filled, edge_radius, segments)
        operations.append({"name": p["smoothing"], "status": "succeeded", "window_samples": width,
                           "window_s": width / fs, "edge_radius_samples": edge_radius})
    else:
        operations.append({"name": "smoothing", "status": "not_requested"})

    lowpass = p.get("lowpass_hz")
    filter_report = None
    if lowpass is not None:
        if not fs or fs <= 0:
            fail("sampling_rate_unknown", "低通滤波需要可验证的正采样率。", field="time_s")
        if not np.isfinite(lowpass) or lowpass <= 0 or lowpass >= fs / 2:
            fail("invalid_cutoff", f"低通截止频率必须满足 0 < cutoff < Nyquist ({fs / 2:g} Hz)。", field="preprocess.lowpass_hz")
        order = int(p.get("filter_order", 2))
        sos = signal.butter(order, float(lowpass), btype="lowpass", fs=fs, output="sos")
        padlen = _sos_padlen(sos)
        settle = _settling_radius(sos, float(p.get("edge_tolerance", 0.001)))
        min_duration = float(p.get("min_segment_duration_s", 0.0))
        outcomes = []
        for start, end in segments:
            duration = float(source_time[end - 1] - source_time[start]) if end - start > 1 else 0.0
            if end - start <= padlen or duration < min_duration or end - start <= 1:
                quality["processing_eligible"][start:end] = False
                outcomes.append({"start": start, "end": end, "status": "skipped", "reason": "segment_too_short",
                                 "samples": end - start, "duration_s": duration, "required_samples_gt": padlen,
                                 "min_segment_duration_s": min_duration})
                continue
            clean[start:end] = signal.sosfiltfilt(sos, clean[start:end], padtype="odd", padlen=padlen)
            quality["edge_affected"][start:min(end, start + settle)] = True
            quality["edge_affected"][max(start, end - settle):end] = True
            _mark_radius(quality["edge_affected"], filled, settle, [(start, end)])
            outcomes.append({"start": start, "end": end, "status": "succeeded", "duration_s": duration,
                             "samples": end - start, "edge_radius_samples": settle,
                             "edge_radius_s": settle / fs})
        filter_report = {"cutoff_hz": float(lowpass), "single_pass_order": order, "execution": "sosfiltfilt_forward_backward",
                         "padlen_samples": padlen, "padtype": "odd", "edge_tolerance": float(p.get("edge_tolerance", 0.001)),
                         "edge_radius_samples": settle, "edge_radius_s": settle / fs, "segments": outcomes,
                         "response_note": "前后向运行的幅频响应为单程幅频响应平方；cutoff_hz 是单程 Butterworth 设计截止。"}
        operations.append({"name": "butterworth_lowpass", **filter_report})
    else:
        operations.append({"name": "lowpass", "status": "not_requested"})

    analysis_eligible = quality["processing_eligible"] & ~quality["edge_affected"]
    quality["analysis_eligible"] = analysis_eligible.copy()
    quality["eligible"] = analysis_eligible.copy()
    quality["detector_eligible"] = analysis_eligible & ~quality["interpolated"] & ~quality["artifact"]

    # 明确区间中心化只对可用的原采样物理分支生效，绝对差值与单位不变。
    baseline_offset = None
    baseline = p.get("baseline")
    if baseline is not None:
        lo, hi = map(float, baseline)
        in_baseline = (source_time >= lo) & (source_time < hi) & quality["analysis_eligible"]
        if not np.any(in_baseline):
            fail("baseline_unavailable", "指定基线区间没有可用样本。", field="preprocess.baseline")
        baseline_offset = float(np.mean(clean[in_baseline]))
        clean[quality["processing_eligible"]] -= baseline_offset
        operations.append({"name": "baseline_center", "status": "succeeded", "interval_s": [lo, hi],
                           "support": "[start,end)", "offset_uS": baseline_offset, "reversible": True})
    else:
        operations.append({"name": "baseline_center", "status": "not_requested"})

    transformed = np.full(n, np.nan, dtype=float)
    processed_time = np.array(source_time, copy=True)
    processed_index = np.array(record["sample_index"], copy=True)
    if p.get("resample_hz") is not None:
        target_hz = float(p["resample_hz"])
        if not fs or target_hz <= 0 or not np.isfinite(target_hz):
            fail("resample_rate", "目标采样率必须为正且输入采样率已知。", field="preprocess.resample_hz")
        resample_segments = _segments(quality["processing_eligible"], source_time, gap_limit)
        processed_time, clean, resampled_q, resample_report = _resample_arrays(
            source_time, clean, quality, quality["processing_eligible"], target_hz, fs, resample_segments)
        quality = resampled_q
        fs = target_hz
        processed_index = np.arange(len(clean), dtype=np.int64)
        operations.append({"name": "polyphase_resample", "target_hz": target_hz, "source_hz": fs,
                           "origin_policy": "per_segment_first_timestamp", "grid_rounding": "retain outputs <= source last timestamp",
                           "gap_policy": "resample segments independently; never fill long gaps", **resample_report})
    else:
        operations.append({"name": "resample", "status": "not_requested"})

    # 变换分支在最终处理网格上拟合；物理 clean_uS 和 SCR 输入不受缩放影响。
    scaling = p.get("scaling", "none")
    transform_mask = quality.get("analysis_eligible", quality.get("eligible", np.zeros(len(clean), bool)))
    transformed = np.full(len(clean), np.nan, dtype=float)
    values = clean[transform_mask]
    scaling_info = {"method": scaling, "unit": "uS" if scaling == "none" else "dimensionless"}
    if scaling == "none":
        transformed[transform_mask] = values
    elif scaling == "zscore":
        mean = float(np.mean(values)) if len(values) else math.nan
        std = float(np.std(values)) if len(values) else math.nan
        if not len(values) or not np.isfinite(std) or std == 0:
            fail("zero_variance_scaling", "zscore 需要非空且非零方差的可用样本。", field="preprocess.scaling")
        transformed[transform_mask] = (values - mean) / std
        scaling_info.update({"mean_uS": mean, "scale_uS": std, "ddof": 0})
    elif scaling == "range":
        minimum = float(np.min(values)) if len(values) else math.nan
        maximum = float(np.max(values)) if len(values) else math.nan
        if not len(values) or maximum <= minimum:
            fail("zero_range_scaling", "range scaling 需要非空且有变化的可用样本。", field="preprocess.scaling")
        transformed[transform_mask] = (values - minimum) / (maximum - minimum)
        scaling_info.update({"minimum_uS": minimum, "maximum_uS": maximum, "scale_uS": maximum - minimum})
    elif scaling == "log":
        if not len(values) or np.any(values <= 0):
            fail("log_nonpositive", "log 分支要求所有纳入样本为正；使用 log(clean_uS / 1 uS)，不更改物理分支。",
                 field="preprocess.scaling")
        transformed[transform_mask] = np.log(values / 1.0)
        scaling_info["reference_uS"] = 1.0
    else:
        fail("scaling_method", f"不支持 scaling：{scaling}", category="configuration")
    operations.append({"name": "scaling_branch", "status": "succeeded", **scaling_info})

    # 保留原始记录快照；顶层 time_s/sample_index 描述清理后的处理网格。
    # 深复制完整来源记录，以保留 clock alignment 添入的 local_time_s 与
    # 其他逐样本来源字段；处理和重采样只替换 processed 顶层网格，不改 raw snapshot。
    raw_record = copy.deepcopy(record)
    processed = {**raw_record, "record": raw_record, "time_s": processed_time,
                 "sample_index": processed_index, "clean_uS": np.array(clean, dtype=float, copy=True),
                 "analysis_signal": np.array(transformed, dtype=float, copy=True), "analysis_unit": scaling_info["unit"],
                 "quality": quality, "operations": operations, "sampling_rate_hz": float(p["resample_hz"])
                 if p.get("resample_hz") is not None else fs, "baseline_offset_uS": baseline_offset,
                 "scaling": scaling_info, "segments": _segments(quality.get("processing_eligible", quality.get("eligible", [])),
                                                                   processed_time,
                                                                   _gap_limit(processed_time, quality.get("processing_eligible", quality.get("eligible", [])), qc_cfg, mapping)
                                                                   if len(processed_time) else None)}
    timebase_metadata = copy.deepcopy(record.get("metadata", {}))
    if p.get("resample_hz") is not None:
        # local_time_s 属于源采样网格；重采样后仅在 record 原始快照中保留，
        # 避免处理网格顶层出现长度不同的时间列。
        processed.pop("local_time_s", None)
        timebase_metadata["processed_grid"] = True
        timebase_metadata["fs_hz"] = fs
        timebase_metadata.pop("time_mapping", None)
        timebase_metadata.pop("fs_local_hz", None)
        timebase_metadata.pop("source_fs_hz", None)
        if timebase_metadata.get("clock_domain") == "reference":
            timebase_metadata["fs_reference_hz"] = fs
    processed["timebase_metadata"] = timebase_metadata
    quality_segments = processed["segments"]
    quality["facts"] = {name: np.array(quality.get(name, np.zeros(len(processed_time), dtype=bool)), copy=True)
                         for name in _FACT_NAMES}
    quality["segments"] = [tuple(segment) for segment in quality_segments]
    quality["intervals"] = copy.deepcopy(qc.get("intervals", []))
    quality["summary"] = copy.deepcopy(qc.get("summary", {}))
    quality["annotation_revision"] = qc.get("annotation_revision")
    processed["quality"] = quality
    processed["sample_index_space"] = "processed_grid" if p.get("resample_hz") is not None else "source_grid"
    return processed


__all__ = ["preprocess_record"]
