"""可审计的 tonic/phasic 分离研究基线；不冒称外部算法兼容。"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .errors import fail
from .core_preprocess import (_full_config, _gap_limit, _sampling_rate, _segments,
                              _settling_radius, _sos_padlen, _bool_array)


def separate_tonic_phasic(processed, config, *, cancel_callback=None):
    """用 SciPy Butterworth 二阶 SOS 低/高通分别计算 tonic 与 phasic。

    默认研究 cutoff 为 0.05 Hz，单位 Hz，物理输入输出为 μS；低通为 tonic、
    独立高通为 phasic，前后向零相位运行，两路不保证相加还原输入。残差按
    `residual_uS = clean_uS - tonic_uS - phasic_uS` 逐点定义。每个有效连续段独立
    处理，要求 `0 < cutoff < fs/2`、大于 SciPy SOS padding 长度且满足
    `min_segment_duration_s`；不能处理的段保留 NaN 并报告 skipped。边缘按两路
    最慢极点衰减到 `edge_tolerance` 的保守半径标记，且不跨越插值区传播有效性。
    本方法名为 `filter_separation`，不是 NeuroKit、cvxEDA 或 Ledalab 的兼容实现；
    不提供 driver，solver/boundary diagnostics 随 bundle 返回。
    """
    if not isinstance(processed, dict) or "clean_uS" not in processed or "time_s" not in processed:
        fail("processed_contract", "分解需要 preprocess_record 返回的 processed 字典。", stage="decomposition")
    full = _full_config(config)
    d = full["decomposition"]
    if d.get("method") == "bateman_sparse":
        from .driver_decompose import separate_driver
        return separate_driver(processed, config, cancel_callback=cancel_callback)
    if d.get("method") != "filter_separation":
        fail("decomposition_unavailable", f"当前实现只支持显式 filter_separation；不会把 {d.get('method')} 静默替换成高通。",
             stage="decomposition", field="decomposition.method",
             remediation="改用 filter_separation，或接入经过授权与参考验证的算法适配器。")
    time = np.asarray(processed["time_s"], dtype=float)
    source = np.asarray(processed["clean_uS"], dtype=float)
    if time.ndim != 1 or source.shape != time.shape:
        fail("processed_shape", "processed time_s 与 clean_uS 必须等长一维数组。", stage="decomposition")
    quality_in = processed.get("quality", {})
    quality_in = quality_in if isinstance(quality_in, dict) else {}
    process_mask = _bool_array(quality_in.get("processing_eligible", quality_in.get("eligible")),
                               len(source), "processing_eligible") & np.isfinite(source)
    analysis_mask = _bool_array(quality_in.get("analysis_eligible", quality_in.get("eligible")),
                                len(source), "analysis_eligible") & process_mask
    detector_input = _bool_array(quality_in.get("detector_eligible", quality_in.get("eligible")),
                                 len(source), "detector_eligible") & analysis_mask
    interpolated = _bool_array(quality_in.get("interpolated"), len(source), "interpolated")
    edge = _bool_array(quality_in.get("edge_affected"), len(source), "edge_affected")
    artifact = _bool_array(quality_in.get("artifact"), len(source), "artifact")

    if np.any(process_mask):
        fs, segments, max_gap = _sampling_rate(time, process_mask, config,
                                               metadata=processed.get("timebase_metadata", processed.get("metadata")),
                                               require_regular=True)
    else:
        fs = processed.get("sampling_rate_hz") or full["mapping"].get("sampling_rate_hz")
        if not fs:
            fail("sampling_rate_unknown", "没有可用样本且未声明采样率，无法解释分解时间尺度。", stage="decomposition")
        fs = float(fs)
        max_gap = _gap_limit(time, process_mask, full["qc"], full["mapping"])
        segments = []
    cutoff = float(d.get("cutoff_hz", 0.05))
    if not np.isfinite(cutoff) or cutoff <= 0 or cutoff >= fs / 2:
        fail("invalid_cutoff", f"分解 cutoff 必须满足 0 < cutoff < Nyquist ({fs / 2:g} Hz)。",
             stage="decomposition", field="decomposition.cutoff_hz")
    order = int(d.get("order", 2))
    tolerance = float(d.get("edge_tolerance", 0.001))
    low_sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    high_sos = signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    low_pad, high_pad = _sos_padlen(low_sos), _sos_padlen(high_sos)
    padlen = max(low_pad, high_pad)
    low_radius = _settling_radius(low_sos, tolerance)
    high_radius = _settling_radius(high_sos, tolerance)
    radius = max(low_radius, high_radius)
    minimum_duration = float(full["preprocess"].get("min_segment_duration_s", 0.0))
    tonic = np.full(len(source), np.nan, dtype=float)
    phasic = np.full(len(source), np.nan, dtype=float)
    residual = np.full(len(source), np.nan, dtype=float)
    detector = np.zeros(len(source), dtype=bool)
    diagnostics = []
    edge_result = edge.copy()
    for start, end in segments:
        duration = float(time[end - 1] - time[start]) if end - start > 1 else 0.0
        if end - start <= padlen or end - start <= 1 or duration < minimum_duration:
            diagnostics.append({"start": int(start), "end": int(end), "status": "skipped",
                                "reason": "segment_too_short", "samples": int(end - start),
                                "duration_s": duration, "required_samples_gt": int(padlen),
                                "min_segment_duration_s": minimum_duration})
            continue
        tonic[start:end] = signal.sosfiltfilt(low_sos, source[start:end], padtype="odd", padlen=low_pad)
        phasic[start:end] = signal.sosfiltfilt(high_sos, source[start:end], padtype="odd", padlen=high_pad)
        residual[start:end] = source[start:end] - tonic[start:end] - phasic[start:end]
        edge_result[start:min(end, start + radius)] = True
        edge_result[max(start, end - radius):end] = True
        for index in np.flatnonzero(interpolated[start:end]) + start:
            edge_result[max(start, index - radius):min(end, index + radius + 1)] = True
        detector[start:end] = detector_input[start:end] & ~edge_result[start:end] & ~interpolated[start:end] & ~artifact[start:end]
        diagnostics.append({"start": int(start), "end": int(end), "status": "succeeded",
                            "samples": int(end - start), "duration_s": duration,
                            "edge_radius_samples": int(radius), "edge_radius_s": radius / fs,
                            "lowpass_padlen_samples": int(low_pad), "highpass_padlen_samples": int(high_pad)})

    quality = {key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else copy_value(value))
               for key, value in quality_in.items()}
    quality["edge_affected"] = edge_result
    quality["analysis_eligible"] = analysis_mask & ~edge_result
    quality["eligible"] = quality["analysis_eligible"].copy()
    quality["detector_eligible"] = detector
    quality["segments"] = [tuple(segment) for segment in segments]
    succeeded_n = sum(row["status"] == "succeeded" for row in diagnostics)
    report_status = "unavailable" if not succeeded_n else "succeeded" if succeeded_n == len(diagnostics) else "partial"
    report = {"method": "filter_separation", "version": "scipy-butterworth-sos-local-implementation",
              "basis": "clean_uS", "unit": "uS", "cutoff_hz": cutoff, "order": order,
              "sampling_rate_hz": fs, "execution": "independent_sosfiltfilt_lowpass_and_highpass",
              "residual_definition": "clean_uS - tonic_uS - phasic_uS", "driver": None,
              "edge_tolerance": tolerance, "edge_radius_samples": int(radius),
              "edge_radius_s": radius / fs, "padlen_samples": int(padlen),
              "segments": diagnostics,
              "status": report_status,
              "notes": ["两路独立滤波，components 不承诺精确重构输入。",
                        "不是 NeuroKit/cvxEDA/Ledalab 兼容实现。", "未处理段输出 NaN 并在 diagnostics 标记。"]}
    bundle = {key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else copy_value(value))
              for key, value in processed.items() if key not in {"quality", "operations", "segments"}}
    bundle.update({"quality": quality, "operations": list(processed.get("operations", [])),
                   "segments": [tuple(x) for x in segments], "tonic_uS": tonic, "phasic_uS": phasic,
                   "residual_uS": residual, "driver": None, "method": "filter_separation",
                   "basis": "clean_uS", "diagnostics": report, "detector_eligible": detector,
                   "sampling_rate_hz": fs, "max_gap_s": max_gap})
    return bundle


def copy_value(value):
    """复制普通容器，不复制只读标量；该 helper 避免把输入 numpy 数组共享给输出。"""
    if isinstance(value, dict):
        return {key: copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_value(item) for item in value)
    return value


__all__ = ["separate_tonic_phasic"]
