"""Clean-basis spectral and complexity metrics for a single supported window.

This module contains no GUI or feature-registry dependency.  It deliberately
refuses to turn gaps, invalid samples, or irregular grids into a new signal by
deleting or joining samples.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from .metrics_ids import autocorrelation_feature_id


_STATIC_COMPLEXITY_IDS = (
    "clean_sample_entropy",
    "clean_permutation_entropy",
    "clean_dfa_alpha",
)
_SPECTRA_COLUMNS = (
    "recording_id", "window_id", "frequency_hz", "psd_uS2_per_hz", "actual_fs_hz",
    "nperseg", "noverlap", "nfft", "window", "detrend", "status", "reason",
)
_DIAGNOSTIC_COLUMNS = (
    "recording_id", "window_id", "family", "feature_id", "status", "reason",
    "parameters_json", "diagnostics_json",
)


def _json_value(value: Any) -> Any:
    """Convert numerical scalars to strict-JSON values without hiding invalidity."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _json_text(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _result(value: float | None, *, unit: str, method: str, reason: str = "",
            valid_n: int = 0, valid_duration_s: float = 0.0,
            parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    status = "valid" if value is not None and math.isfinite(float(value)) else "unavailable"
    if status == "unavailable":
        value = None
        if not reason:
            reason = "nonfinite_result"
    return {
        "value": None if value is None else float(value),
        "unit": str(unit),
        "basis": "clean_uS",
        "method": str(method),
        "status": status,
        "reason": str(reason or ""),
        "valid_n": max(0, int(valid_n)),
        "valid_duration_s": max(0.0, float(valid_duration_s)),
        "parameters": dict(parameters or {}),
    }


def _identity_fields(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        return {"recording_id": None, "window_id": None}
    recording_id = identity.get("recording_id", identity.get("record_id"))
    window_id = identity.get("window_id")
    return {
        "recording_id": None if recording_id is None else str(recording_id),
        "window_id": None if window_id is None else str(window_id),
    }


def _extended_section(extended_config: Any) -> dict[str, Any]:
    if not isinstance(extended_config, dict):
        return {}
    if isinstance(extended_config.get("features"), dict):
        feature_section = extended_config["features"]
        if isinstance(feature_section.get("extended"), dict):
            return feature_section["extended"]
    if isinstance(extended_config.get("extended"), dict):
        return extended_config["extended"]
    return extended_config


def _family_config(extended: dict[str, Any], family: str) -> dict[str, Any]:
    value = extended.get(family, {})
    return value if isinstance(value, dict) else {}


def _requested_set(selected_ids: Any) -> set[str] | None:
    if selected_ids is None:
        return None
    if isinstance(selected_ids, str):
        return {selected_ids}
    try:
        return {str(item) for item in selected_ids}
    except TypeError as error:
        raise ValueError("selected_ids 必须为 ID 字符串序列或 None。") from error


def _is_requested(feature_id: str, selected: set[str] | None) -> bool:
    return selected is None or feature_id in selected


def _check_cancel(cancel_callback: Any) -> None:
    if not callable(cancel_callback):
        return
    if cancel_callback():
        raise InterruptedError("clean 高级指标计算已取消。")


def _emit(progress_callback: Any, event: dict[str, Any]) -> None:
    if not callable(progress_callback):
        return
    try:
        progress_callback(event)
    except Exception:
        # Progress consumers are observers; their failure cannot change a metric.
        return


def _family_work(progress_callback: Any, identity: dict[str, Any], family: str,
                 method: str, callback: Any) -> Any:
    started = time.monotonic()
    record_id = identity.get("recording_id")
    window_id = identity.get("window_id")
    work_id = f"{record_id or 'record'}::{window_id or 'window'}:clean_{family}"
    common = {"stage": "features", "record_id": record_id, "window_id": window_id,
              "family": family, "method": method, "rate_hz": None,
              "unit_type": "feature_family", "work_id": work_id,
              "details": {"work_id": work_id, "family": family, "method": method,
                          "window_id": window_id, "rate_hz": None}}
    _emit(progress_callback, {**common, "kind": "work_started", "message": f"开始计算 clean {family} 指标",
                              "completed": 0, "total": 1, "units_done": 0, "units_total": 1})
    try:
        result = callback()
    except InterruptedError as error:
        _emit(progress_callback, {**common, "kind": "cancelled", "message": str(error) or "指标计算已取消",
                                  "completed": 0, "total": 1, "units_done": 0, "units_total": 1,
                                  "duration_s": time.monotonic() - started,
                                  "details": {**common["details"], "duration_s": time.monotonic() - started}})
        raise
    except Exception as error:
        elapsed = time.monotonic() - started
        _emit(progress_callback, {**common, "kind": "failed", "message": f"clean {family} 指标计算失败",
                                  "completed": 0, "total": 1, "units_done": 0, "units_total": 1,
                                  "duration_s": elapsed,
                                  "details": {**common["details"], "duration_s": elapsed,
                                              "error_type": type(error).__name__}})
        raise
    elapsed = time.monotonic() - started
    _emit(progress_callback, {**common, "kind": "completed", "message": f"clean {family} 指标计算完成",
                              "completed": 1, "total": 1, "units_done": 1, "units_total": 1,
                              "duration_s": elapsed,
                              "details": {**common["details"], "duration_s": elapsed,
                                          "record_status": "succeeded"}})
    return result


def _empty_tables() -> dict[str, pd.DataFrame]:
    return {
        "spectra": pd.DataFrame(columns=list(_SPECTRA_COLUMNS)),
        "complexity_diagnostics": pd.DataFrame(columns=list(_DIAGNOSTIC_COLUMNS)),
    }


def _support(time_s: Any, clean_uS: Any, eligible: Any, *, window_start_s: float,
             window_end_s: float, declared_fs_hz: float | None, max_gap_s: float | None,
             jitter_tolerance: float) -> dict[str, Any]:
    try:
        time = np.asarray(time_s, dtype=np.float64)
        values = np.asarray(clean_uS, dtype=np.float64)
        mask = np.asarray(eligible, dtype=bool)
    except (TypeError, ValueError, OverflowError):
        return {"status": "unavailable", "reason": "invalid_array_values", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": None, "duration_s": 0.0,
                "parameters": {}}
    if time.ndim != 1 or values.ndim != 1 or mask.ndim != 1 or time.shape != values.shape or time.shape != mask.shape:
        return {"status": "unavailable", "reason": "array_shape_mismatch", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": None, "duration_s": 0.0,
                "parameters": {}}
    if not len(time):
        return {"status": "unavailable", "reason": "no_window_samples", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": None, "duration_s": 0.0,
                "parameters": {}}
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        return {"status": "unavailable", "reason": "invalid_timebase", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": None, "duration_s": 0.0,
                "parameters": {}}
    if (isinstance(window_start_s, bool) or isinstance(window_end_s, bool) or
            not isinstance(window_start_s, (int, float)) or not isinstance(window_end_s, (int, float)) or
            not math.isfinite(window_start_s) or not math.isfinite(window_end_s) or window_end_s <= window_start_s):
        return {"status": "unavailable", "reason": "invalid_window_bounds", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": None, "duration_s": 0.0,
                "parameters": {}}
    window_mask = (time >= float(window_start_s)) & (time < float(window_end_s))
    indices = np.flatnonzero(window_mask)
    if not len(indices):
        return {"status": "unavailable", "reason": "no_window_samples", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": None, "duration_s": 0.0,
                "parameters": {}}
    window_time = time[indices]
    window_values = values[indices]
    window_eligible = mask[indices]
    if np.any(window_eligible & ~np.isfinite(window_values)):
        return {"status": "unavailable", "reason": "nonfinite_clean_signal", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": None, "duration_s": 0.0,
                "parameters": {"window_n": int(len(indices))}}
    if not np.all(window_eligible):
        return {"status": "unavailable", "reason": "ineligible_samples_in_window", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": None, "duration_s": 0.0,
                "parameters": {"window_n": int(len(indices)), "eligible_n": int(np.count_nonzero(window_eligible))}}
    if len(window_time) < 2:
        return {"status": "unavailable", "reason": "insufficient_samples_for_grid", "time": np.array([]),
                "values": np.array([]), "n": int(len(window_time)), "fs_hz": None,
                "duration_s": 0.0, "parameters": {"window_n": int(len(indices))}}

    dt = np.diff(window_time)
    measured_dt = float(np.median(dt))
    measured_fs = 1.0 / measured_dt
    grid_parameters = {"declared_fs_hz": declared_fs_hz, "actual_fs_hz": measured_fs,
                       "median_dt_s": measured_dt, "max_observed_gap_s": float(np.max(dt)),
                       "jitter_tolerance": jitter_tolerance, "max_gap_s": max_gap_s,
                       "window_start_s": float(window_start_s), "window_end_s": float(window_end_s)}
    if isinstance(max_gap_s, (int, float)) and not isinstance(max_gap_s, bool):
        if not math.isfinite(float(max_gap_s)) or float(max_gap_s) <= 0:
            return {"status": "unavailable", "reason": "invalid_max_gap_s", "time": np.array([]),
                    "values": np.array([]), "n": 0, "fs_hz": measured_fs,
                    "duration_s": 0.0, "parameters": grid_parameters}
        if np.any(dt > float(max_gap_s) + 1e-12):
            return {"status": "unavailable", "reason": "gap_exceeds_max_gap_s", "time": np.array([]),
                    "values": np.array([]), "n": 0, "fs_hz": measured_fs,
                    "duration_s": 0.0, "parameters": grid_parameters}
    if isinstance(declared_fs_hz, bool) or not isinstance(declared_fs_hz, (int, float)) or not math.isfinite(declared_fs_hz) or declared_fs_hz <= 0:
        return {"status": "unavailable", "reason": "declared_sampling_rate_missing_or_invalid", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": measured_fs,
                "duration_s": 0.0, "parameters": grid_parameters}
    if isinstance(jitter_tolerance, bool) or not isinstance(jitter_tolerance, (int, float)) or not math.isfinite(jitter_tolerance) or jitter_tolerance < 0:
        return {"status": "unavailable", "reason": "invalid_jitter_tolerance", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": measured_fs,
                "duration_s": 0.0, "parameters": grid_parameters}
    expected_dt = 1.0 / float(declared_fs_hz)
    deviations = np.abs(dt - expected_dt) / expected_dt
    grid_parameters["max_relative_jitter"] = float(np.max(deviations))
    grid_parameters["declared_rate_relative_error"] = abs(measured_fs - float(declared_fs_hz)) / float(declared_fs_hz)
    if np.any(deviations > float(jitter_tolerance) + 1e-12):
        return {"status": "unavailable", "reason": "irregular_sampling_grid", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": measured_fs,
                "duration_s": 0.0, "parameters": grid_parameters}
    if grid_parameters["declared_rate_relative_error"] > float(jitter_tolerance) + 1e-12:
        return {"status": "unavailable", "reason": "sampling_rate_mismatch", "time": np.array([]),
                "values": np.array([]), "n": 0, "fs_hz": measured_fs,
                "duration_s": 0.0, "parameters": grid_parameters}
    return {"status": "valid", "reason": "", "time": window_time, "values": window_values,
            "n": int(len(window_time)), "fs_hz": measured_fs,
            "duration_s": float(window_time[-1] - window_time[0]), "parameters": grid_parameters}


def _band_integral(frequency: np.ndarray, psd: np.ndarray, low_hz: float,
                   high_hz: float) -> tuple[float | None, str]:
    if not math.isfinite(low_hz) or not math.isfinite(high_hz) or low_hz <= 0 or high_hz <= low_hz:
        return None, "invalid_frequency_band"
    tolerance = 1e-12 * max(1.0, abs(float(frequency[-1])))
    if low_hz < float(frequency[0]) - tolerance or high_hz > float(frequency[-1]) + tolerance:
        return None, "band_above_nyquist_or_outside_frequency_support"
    low = min(max(low_hz, float(frequency[0])), float(frequency[-1]))
    high = min(max(high_hz, float(frequency[0])), float(frequency[-1]))
    inner = frequency[(frequency > low) & (frequency < high)]
    x = np.concatenate(([low], inner, [high]))
    y = np.concatenate(([float(np.interp(low, frequency, psd))],
                        psd[(frequency > low) & (frequency < high)],
                        [float(np.interp(high, frequency, psd))]))
    if len(x) < 2 or not np.all(np.isfinite(y)):
        return None, "band_frequency_support_missing"
    integral = float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))
    if not math.isfinite(integral):
        return None, "nonfinite_band_integral"
    return integral, ""


def _metric_for_band(feature_id: str, unit: str, value: float | None, reason: str,
                     *, parameters: dict[str, Any], valid_n: int, valid_duration_s: float) -> dict[str, Any]:
    return _result(value, unit=unit, method="scipy.signal.welch_one_sided_density_linear_band_integral",
                   reason=reason, valid_n=valid_n, valid_duration_s=valid_duration_s,
                   parameters={**parameters, "feature_id": feature_id})


def _metric_unit(feature_id: str, complexity_config: dict[str, Any] | None = None) -> str:
    """Return the configured unit for an ID, including unnormalized PermEn."""
    if feature_id.endswith("_uS2"):
        return "uS2"
    if feature_id.endswith("_normalized"):
        return "fraction"
    if feature_id.startswith("clean_autocorr_lag_"):
        return "correlation"
    if feature_id == "clean_sample_entropy":
        return "nats"
    if feature_id == "clean_permutation_entropy":
        cfg = complexity_config.get("permen", {}) if isinstance(complexity_config, dict) else {}
        if isinstance(cfg, dict) and cfg.get("normalize", True) is False:
            return "nats"
    return "dimensionless"


def _frequency_band(value: Any) -> tuple[float, float] | None:
    """Parse a configured [low, high] pair without accepting booleans."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = value
    if (isinstance(low, bool) or isinstance(high, bool) or
            not isinstance(low, (int, float)) or not isinstance(high, (int, float))):
        return None
    low = float(low)
    high = float(high)
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high <= low:
        return None
    return low, high


def _compute_spectral(time: np.ndarray, values: np.ndarray, fs_hz: float, duration_s: float,
                      spectral_config: dict[str, Any], requested: set[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    spectra_rows: list[dict[str, Any]] = []
    bands = spectral_config.get("bands", [])
    if not isinstance(bands, list):
        bands = []
    selected_bands = []
    for band in bands:
        if not isinstance(band, dict):
            continue
        band_id = str(band.get("id", ""))
        if not band_id:
            continue
        absolute_id = f"clean_bandpower_{band_id}_uS2"
        normalized_id = f"clean_bandpower_{band_id}_normalized"
        if absolute_id in requested or normalized_id in requested:
            selected_bands.append((band, absolute_id, normalized_id))
    if not selected_bands:
        return metrics, spectra_rows, diagnostics

    overlap = spectral_config.get("overlap", 0.5)
    segment_s = spectral_config.get("segment_s", 256.0)
    window = str(spectral_config.get("window", "hann"))
    detrend = str(spectral_config.get("detrend", "constant"))
    min_segments = spectral_config.get("min_segments", 2)
    min_cycles = spectral_config.get("min_cycles", 2.0)
    nperseg = int(round(float(segment_s) * fs_hz)) if isinstance(segment_s, (int, float)) and math.isfinite(segment_s) else 0
    noverlap = int(math.floor(float(overlap) * nperseg)) if nperseg > 0 and isinstance(overlap, (int, float)) and math.isfinite(overlap) else -1
    nfft = nperseg
    actual_segments = 0 if nperseg <= 0 or noverlap < 0 or noverlap >= nperseg or len(values) < nperseg else 1 + (len(values) - nperseg) // (nperseg - noverlap)
    normalization_band = _frequency_band(spectral_config.get("normalization_band_hz", [0.01, 0.5]))
    spectral_parameters = {
        "actual_fs_hz": fs_hz, "declared_segment_s": segment_s,
        "nperseg": nperseg, "noverlap": noverlap, "nfft": nfft,
        "window": window, "detrend": detrend, "one_sided": True,
        "scaling": "density", "average": "mean", "segment_count": actual_segments,
        "min_segments": min_segments, "min_cycles": min_cycles,
        "overlap": overlap, "normalization_band_hz": list(normalization_band) if normalization_band else None,
        "frequency_edge_policy": "linear_interpolation_at_band_edges; Nyquist endpoint allowed",
    }
    common_reason = ""
    normalization_reason = ""
    if nperseg <= 0 or noverlap < 0 or noverlap >= nperseg:
        common_reason = "invalid_welch_parameters"
    elif isinstance(min_segments, bool) or not isinstance(min_segments, int) or min_segments < 1:
        common_reason = "invalid_min_segments"
    elif actual_segments < min_segments:
        common_reason = "insufficient_welch_segments"
    elif not math.isfinite(float(min_cycles)) or float(min_cycles) <= 0:
        common_reason = "invalid_min_cycles"

    frequency = np.array([], dtype=np.float64)
    psd = np.array([], dtype=np.float64)
    if not common_reason:
        if normalization_band is None:
            normalization_reason = "invalid_normalization_band"
        else:
            normalization_low = normalization_band[0]
            cycles = normalization_low * nperseg / fs_hz
            if cycles < float(min_cycles):
                normalization_reason = "insufficient_cycles_in_normalization_band"
    if not common_reason:
        try:
            frequency, psd = scipy_signal.welch(
                values, fs=fs_hz, window=window, nperseg=nperseg, noverlap=noverlap,
                nfft=nfft, detrend=detrend, return_onesided=True, scaling="density", average="mean",
            )
            if not np.all(np.isfinite(psd)) or not np.all(np.isfinite(frequency)):
                common_reason = "nonfinite_welch_spectrum"
        except (TypeError, ValueError) as error:
            common_reason = "welch_parameter_error:" + type(error).__name__
    if not common_reason and len(frequency):
        for index, (freq, density) in enumerate(zip(frequency, psd)):
            spectra_rows.append({"frequency_hz": float(freq), "psd_uS2_per_hz": float(density),
                                 "actual_fs_hz": fs_hz, "nperseg": nperseg,
                                 "noverlap": noverlap, "nfft": nfft,
                                 "window": window, "detrend": detrend,
                                 "status": "valid", "reason": ""})

    norm_power = None
    norm_reason = common_reason or normalization_reason
    if not common_reason and normalization_band is not None and not normalization_reason:
        norm_power, band_reason = _band_integral(frequency, psd,
                                                 normalization_band[0],
                                                 normalization_band[1])
        if band_reason:
            norm_reason = band_reason
    if not common_reason and not norm_reason and norm_power == 0:
        norm_reason = "zero_normalization_power"

    for band, absolute_id, normalized_id in selected_bands:
        low = band.get("low_hz")
        high = band.get("high_hz")
        band_parameters = {**spectral_parameters, "band_id": str(band.get("id")),
                           "band_hz": [low, high]}
        absolute_value = None
        absolute_reason = common_reason
        if not absolute_reason:
            try:
                if (isinstance(low, bool) or isinstance(high, bool) or
                        not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or
                        not math.isfinite(low) or not math.isfinite(high)):
                    absolute_reason = "invalid_frequency_band"
                elif float(low) * nperseg / fs_hz < float(min_cycles):
                    absolute_reason = "insufficient_cycles_in_band"
                else:
                    absolute_value, absolute_reason = _band_integral(frequency, psd, float(low), float(high))
            except (TypeError, ValueError, OverflowError):
                absolute_reason = "invalid_frequency_band"
        if absolute_id in requested:
            metrics[absolute_id] = _metric_for_band(
                absolute_id, "uS2", absolute_value, absolute_reason,
                parameters=band_parameters, valid_n=len(values), valid_duration_s=duration_s)
            diagnostics.append({"feature_id": absolute_id, "status": metrics[absolute_id]["status"],
                                "reason": metrics[absolute_id]["reason"],
                                "parameters": band_parameters,
                                "diagnostics": {"absolute_power_uS2": absolute_value,
                                                "frequency_bin_count": int(len(frequency))}})
        normalized_value = None
        normalized_reason = common_reason or norm_reason
        if not normalized_reason:
            if normalization_band is None:
                normalized_reason = "invalid_normalization_band"
            elif absolute_reason:
                normalized_reason = absolute_reason
            elif float(low) < normalization_band[0] - 1e-12 or float(high) > normalization_band[1] + 1e-12:
                normalized_reason = "band_outside_normalization_range"
            elif absolute_value is None:
                normalized_reason = absolute_reason or "band_power_unavailable"
            elif norm_power is None or norm_power == 0:
                normalized_reason = "zero_normalization_power"
            else:
                normalized_value = float(absolute_value / norm_power)
                normalized_reason = ""
        if normalized_id in requested:
            normalized_parameters = {**band_parameters, "normalization_band_hz": list(normalization_band) if normalization_band else None,
                                     "normalization_power_uS2": norm_power}
            metrics[normalized_id] = _metric_for_band(
                normalized_id, "fraction", normalized_value, normalized_reason,
                parameters=normalized_parameters, valid_n=len(values), valid_duration_s=duration_s)
            diagnostics.append({"feature_id": normalized_id, "status": metrics[normalized_id]["status"],
                                "reason": metrics[normalized_id]["reason"],
                                "parameters": normalized_parameters,
                                "diagnostics": {"band_power_uS2": absolute_value,
                                                "normalization_power_uS2": norm_power,
                                                "normalized_power": normalized_value}})
    return metrics, spectra_rows, diagnostics


def _compute_autocorrelation(time: np.ndarray, values: np.ndarray, fs_hz: float,
                             duration_s: float, autocorrelation_config: dict[str, Any],
                             requested_lags: list[tuple[float, str]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    min_pairs = autocorrelation_config.get("min_pairs", 20)
    for lag_s, feature_id in requested_lags:
        parameters = {"requested_lag_s": lag_s, "actual_fs_hz": fs_hz,
                      "min_pairs": min_pairs, "coefficient": "Pearson(x[:-k], x[k:])",
                      "grid_tolerance_s": 1e-6}
        reason = ""
        value = None
        pair_n = 0
        actual_lag = None
        grid_error = None
        if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or min_pairs < 1:
            reason = "invalid_min_pairs"
        else:
            k = int(round(float(lag_s) * fs_hz))
            if k <= 0:
                reason = "lag_grid_mismatch"
            else:
                actual_lag = k / fs_hz
                grid_error = abs(actual_lag - float(lag_s))
                parameters.update({"sample_lag": k, "actual_lag_s": actual_lag,
                                   "lag_grid_error_s": grid_error})
                if grid_error > 1e-6:
                    reason = "lag_grid_mismatch"
                elif k >= len(values):
                    reason = "lag_outside_window"
                else:
                    left = values[:-k]
                    right = values[k:]
                    pair_n = int(len(left))
                    if pair_n < min_pairs:
                        reason = "insufficient_pairs"
                    else:
                        left_centered = left - float(np.mean(left))
                        right_centered = right - float(np.mean(right))
                        left_ss = float(np.dot(left_centered, left_centered))
                        right_ss = float(np.dot(right_centered, right_centered))
                        if left_ss <= 0 or right_ss <= 0:
                            reason = "zero_variance"
                        else:
                            value = float(np.dot(left_centered, right_centered) / math.sqrt(left_ss * right_ss))
                            if not math.isfinite(value):
                                value = None
                                reason = "nonfinite_pearson_result"
        metrics[feature_id] = _result(value, unit="correlation", method="pearson_lag_correlation",
                                       reason=reason, valid_n=pair_n,
                                       valid_duration_s=duration_s,
                                       parameters=parameters)
        parameters["valid_pairs"] = pair_n
        metrics[feature_id]["parameters"]["valid_pairs"] = pair_n
        diagnostics.append({"feature_id": feature_id, "status": metrics[feature_id]["status"],
                            "reason": metrics[feature_id]["reason"], "parameters": parameters,
                            "diagnostics": {"actual_lag_s": actual_lag, "lag_grid_error_s": grid_error,
                                            "valid_pairs": pair_n, "correlation": value}})
    return metrics, diagnostics


def _sample_entropy(values: np.ndarray, complexity_config: dict[str, Any],
                    *, cancel_callback: Any = None) -> tuple[float | None, str, dict[str, Any]]:
    cfg = complexity_config.get("sampen", {})
    cfg = cfg if isinstance(cfg, dict) else {}
    n = int(len(values))
    min_samples = complexity_config.get("min_samples", 200)
    max_samples = complexity_config.get("max_samples", 5000)
    m = cfg.get("m", 2)
    delay = cfg.get("delay", 1)
    r_mode = cfg.get("r_mode", "sd_fraction")
    r_parameter = cfg.get("r", 0.2)
    parameters = {"N": n, "m": m, "delay": delay, "r_mode": r_mode,
                  "r_parameter": r_parameter, "match_metric": "Chebyshev",
                  "pair_rule": "unordered_i_lt_j_no_self", "common_start_count_M": None,
                  "comparison_estimate": None, "B_m_matches": None,
                  "A_m_plus_1_matches": None, "r_uS": None}
    if n < min_samples:
        return None, "below_min_samples", parameters
    if n > max_samples:
        return None, "above_max_samples", parameters
    if isinstance(m, bool) or not isinstance(m, int) or m < 1 or isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        return None, "invalid_template_parameters", parameters
    if r_mode not in {"sd_fraction", "absolute"} or isinstance(r_parameter, bool) or not isinstance(r_parameter, (int, float)) or not math.isfinite(r_parameter) or r_parameter <= 0:
        return None, "invalid_tolerance_parameters", parameters
    common_starts = n - m * delay
    parameters["common_start_count_M"] = int(max(0, common_starts))
    if common_starts < 2:
        return None, "insufficient_common_templates", parameters
    pair_count = common_starts * (common_starts - 1) // 2
    comparison_estimate = pair_count * (m + 1)
    parameters["comparison_estimate"] = int(comparison_estimate)
    comparison_cap = complexity_config.get("max_pair_comparisons", 25_000_000)
    parameters["comparison_cap"] = comparison_cap
    if isinstance(comparison_cap, bool) or not isinstance(comparison_cap, int) or comparison_cap < 1:
        return None, "invalid_comparison_budget", parameters
    if comparison_estimate > comparison_cap:
        return None, "resource_budget", parameters
    sample_sd = float(np.std(values, ddof=1))
    if not math.isfinite(sample_sd) or sample_sd == 0:
        return None, "zero_variance", parameters
    radius = float(r_parameter) * sample_sd if r_mode == "sd_fraction" else float(r_parameter)
    if not math.isfinite(radius) or radius <= 0:
        return None, "invalid_tolerance_parameters", parameters
    parameters["r_uS"] = radius
    starts = np.arange(common_starts, dtype=np.int64)
    offsets = np.arange(m + 1, dtype=np.int64) * delay
    templates = values[starts[:, None] + offsets[None, :]]
    block_size = max(1, min(64, 4_000_000 // max(common_starts * (m + 1), 1)))
    matches_m = 0
    matches_m_plus_1 = 0
    for block_start in range(0, common_starts, block_size):
        _check_cancel(cancel_callback)
        i_indices = np.arange(block_start, min(block_start + block_size, common_starts), dtype=np.int64)
        differences = np.abs(templates[i_indices, None, :] - templates[None, :, :])
        upper_triangle = np.arange(common_starts)[None, :] > i_indices[:, None]
        matches_first_m = np.all(differences[:, :, :m] <= radius, axis=2)
        matches_all = np.all(differences <= radius, axis=2)
        matches_m += int(np.count_nonzero(matches_first_m & upper_triangle))
        matches_m_plus_1 += int(np.count_nonzero(matches_all & upper_triangle))
    parameters["B_m_matches"] = matches_m
    parameters["A_m_plus_1_matches"] = matches_m_plus_1
    if matches_m == 0:
        return None, "zero_m_matches", parameters
    if matches_m_plus_1 == 0:
        return None, "zero_m_plus_1_matches", parameters
    value = -math.log(matches_m_plus_1 / matches_m)
    parameters["sampen_nats"] = value
    return value, "", parameters


def _permutation_entropy(values: np.ndarray, complexity_config: dict[str, Any],
                         *, cancel_callback: Any = None) -> tuple[float | None, str, dict[str, Any]]:
    cfg = complexity_config.get("permen", {})
    cfg = cfg if isinstance(cfg, dict) else {}
    n = int(len(values))
    min_samples = complexity_config.get("min_samples", 200)
    max_samples = complexity_config.get("max_samples", 5000)
    dimension = cfg.get("dimension", 3)
    delay = cfg.get("delay", 1)
    normalize = cfg.get("normalize", True)
    tie_policy = cfg.get("ties", "stable")
    parameters = {"N": n, "dimension": dimension, "delay": delay, "normalize": normalize,
                  "ties": tie_policy, "stable_tie_order": "sample_index", "pattern_count": 0,
                  "tie_vector_count": 0, "tie_fraction": None, "H_nats": None,
                  "normalized_H": None}
    if n < min_samples:
        return None, "below_min_samples", parameters
    if n > max_samples:
        return None, "above_max_samples", parameters
    if (isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2 or
            isinstance(delay, bool) or not isinstance(delay, int) or delay < 1 or
            not isinstance(normalize, bool) or tie_policy not in {"stable", "reject"}):
        return None, "invalid_permutation_parameters", parameters
    if float(np.max(values) - np.min(values)) == 0.0:
        return None, "zero_variance", parameters
    starts = n - (dimension - 1) * delay
    if starts <= 0:
        return None, "insufficient_ordinal_patterns", parameters
    counts: dict[tuple[int, ...], int] = {}
    tie_count = 0
    offsets = np.arange(dimension, dtype=np.int64) * delay
    for start in range(starts):
        if start % 128 == 0:
            _check_cancel(cancel_callback)
        pattern_values = values[start + offsets]
        if len(np.unique(pattern_values)) < dimension:
            tie_count += 1
        order = np.argsort(pattern_values, kind="stable")
        pattern = tuple(int(index) for index in order)
        counts[pattern] = counts.get(pattern, 0) + 1
    parameters["pattern_count"] = int(starts)
    parameters["tie_vector_count"] = int(tie_count)
    parameters["tie_fraction"] = float(tie_count / starts)
    if tie_policy == "reject" and tie_count:
        return None, "ties_rejected", parameters
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / float(starts)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalization = math.log(math.factorial(dimension))
    normalized_entropy = float(entropy / normalization) if normalization > 0 else None
    parameters["H_nats"] = entropy
    parameters["normalized_H"] = normalized_entropy
    parameters["pattern_frequencies"] = {"-".join(map(str, key)): count for key, count in sorted(counts.items())}
    value = normalized_entropy if normalize else entropy
    if value is None or not math.isfinite(value):
        return None, "nonfinite_entropy", parameters
    return value, "", parameters


def _dfa(values: np.ndarray, complexity_config: dict[str, Any],
         *, cancel_callback: Any = None) -> tuple[float | None, str, dict[str, Any]]:
    cfg = complexity_config.get("dfa", {})
    cfg = cfg if isinstance(cfg, dict) else {}
    n = int(len(values))
    min_samples = cfg.get("min_samples", 256)
    min_scale = cfg.get("min_scale", 4)
    max_scale_fraction = cfg.get("max_scale_fraction", 0.25)
    scale_count = cfg.get("scale_count", 10)
    min_scales = cfg.get("min_scales", 6)
    order = cfg.get("order", 1)
    parameters = {"N": n, "order": order, "min_samples": min_samples,
                  "min_scale": min_scale, "max_scale_fraction": max_scale_fraction,
                  "scale_count": scale_count, "min_scales": min_scales,
                  "scales": [], "F_s": [], "box_counts": [],
                  "discarded_tail_samples_by_scale": {}, "valid_scales": [],
                  "valid_F_s": [], "alpha": None, "r_squared": None}
    if n < min_samples:
        return None, "below_min_samples", parameters
    if n > complexity_config.get("max_samples", 5000):
        return None, "above_max_samples", parameters
    if order != 1:
        return None, "unsupported_detrend_order", parameters
    if (isinstance(min_scale, bool) or not isinstance(min_scale, int) or min_scale < 2 or
            isinstance(scale_count, bool) or not isinstance(scale_count, int) or scale_count < 1 or
            isinstance(min_scales, bool) or not isinstance(min_scales, int) or min_scales < 2 or
            isinstance(max_scale_fraction, bool) or not isinstance(max_scale_fraction, (int, float)) or
            not math.isfinite(max_scale_fraction) or max_scale_fraction <= 0):
        return None, "invalid_dfa_parameters", parameters
    if float(np.max(values) - np.min(values)) == 0.0:
        return None, "zero_variance", parameters
    max_scale = int(math.floor(n * float(max_scale_fraction)))
    if max_scale < min_scale:
        return None, "insufficient_scale_range", parameters
    scales = np.unique(np.rint(np.logspace(math.log10(min_scale), math.log10(max_scale), num=scale_count)).astype(np.int64))
    scales = scales[(scales >= min_scale) & (scales <= max_scale)]
    parameters["scales"] = scales.astype(int).tolist()
    if len(scales) < min_scales:
        return None, "insufficient_valid_scales", parameters
    profile = np.cumsum(values - float(np.mean(values)))
    candidate_scales: list[int] = []
    candidate_f: list[float] = []
    scale_box_counts: list[int] = []
    for scale_value in scales:
        _check_cancel(cancel_callback)
        scale = int(scale_value)
        box_count = n // scale
        tail = n - box_count * scale
        scale_box_counts.append(box_count)
        parameters["discarded_tail_samples_by_scale"][str(scale)] = int(tail)
        if box_count < 4:
            candidate_f.append(float("nan"))
            continue
        used = profile[:box_count * scale].reshape(box_count, scale)
        centered_t = np.arange(scale, dtype=np.float64) - (scale - 1) / 2.0
        centered_y = used - np.mean(used, axis=1, keepdims=True)
        denom = float(np.dot(centered_t, centered_t))
        if denom <= 0:
            candidate_f.append(float("nan"))
            continue
        slopes = np.sum(centered_y * centered_t[None, :], axis=1) / denom
        residuals = centered_y - slopes[:, None] * centered_t[None, :]
        fluctuation = float(math.sqrt(float(np.mean(residuals * residuals))))
        candidate_f.append(fluctuation)
        if math.isfinite(fluctuation) and fluctuation > 0:
            candidate_scales.append(scale)
            candidate_f_value = fluctuation
            candidate_f[-1] = candidate_f_value
    parameters["box_counts"] = scale_box_counts
    parameters["F_s"] = [None if not math.isfinite(value) else float(value) for value in candidate_f]
    parameters["valid_scales"] = candidate_scales
    parameters["valid_F_s"] = [float(value) for value in candidate_f if math.isfinite(value) and value > 0]
    if len(candidate_scales) < min_scales:
        if any(count < 4 for count in scale_box_counts):
            reason = "insufficient_boxes"
        elif any(value == 0 for value in candidate_f if math.isfinite(value)):
            reason = "zero_F"
        else:
            reason = "insufficient_valid_scales"
        return None, reason, parameters
    log_scale = np.log(np.asarray(candidate_scales, dtype=np.float64))
    log_f = np.log(np.asarray(parameters["valid_F_s"], dtype=np.float64))
    design = np.column_stack((np.ones(len(log_scale)), log_scale))
    if np.linalg.matrix_rank(design) < 2:
        return None, "rank_deficient", parameters
    coefficients, _, rank, _ = np.linalg.lstsq(design, log_f, rcond=None)
    if rank < 2:
        return None, "rank_deficient", parameters
    alpha = float(coefficients[1])
    predicted = design @ coefficients
    residual_ss = float(np.sum((log_f - predicted) ** 2))
    total_ss = float(np.sum((log_f - float(np.mean(log_f))) ** 2))
    r_squared = None if total_ss <= 0 else float(1.0 - residual_ss / total_ss)
    parameters["alpha"] = alpha
    parameters["r_squared"] = r_squared
    parameters["fit_intercept_logF"] = float(coefficients[0])
    parameters["fit_rank"] = int(rank)
    if not math.isfinite(alpha):
        return None, "nonfinite_fit", parameters
    return alpha, "", parameters


def _compute_complexity(values: np.ndarray, duration_s: float,
                        complexity_config: dict[str, Any], requested: set[str],
                        *, cancel_callback: Any = None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    if not requested.intersection(_STATIC_COMPLEXITY_IDS):
        return metrics, diagnostics
    if complexity_config.get("enabled", True) is not True:
        for feature_id in requested.intersection(_STATIC_COMPLEXITY_IDS):
            permutation = complexity_config.get("permen", {})
            permutation_normalized = permutation.get("normalize", True) if isinstance(permutation, dict) else True
            unit = ("nats" if feature_id == "clean_sample_entropy" or
                    feature_id == "clean_permutation_entropy" and not permutation_normalized
                    else "dimensionless")
            metric = _result(None, unit=unit,
                             method=feature_id, reason="family_disabled")
            metrics[feature_id] = metric
            diagnostics.append({"feature_id": feature_id, "status": metric["status"],
                                "reason": metric["reason"], "parameters": {}, "diagnostics": {}})
        return metrics, diagnostics

    def add(feature_id: str, unit: str, method: str, computation: Any) -> None:
        _check_cancel(cancel_callback)
        value, reason, parameters = computation
        metric = _result(value, unit=unit, method=method, reason=reason,
                         valid_n=len(values), valid_duration_s=duration_s,
                         parameters=parameters)
        metrics[feature_id] = metric
        diagnostics.append({"feature_id": feature_id, "status": metric["status"],
                            "reason": metric["reason"], "parameters": parameters,
                            "diagnostics": parameters})

    if "clean_sample_entropy" in requested:
        add("clean_sample_entropy", "nats", "sample_entropy_chebyshev",
            _sample_entropy(values, complexity_config, cancel_callback=cancel_callback))
    if "clean_permutation_entropy" in requested:
        cfg = complexity_config.get("permen", {})
        normalize = cfg.get("normalize", True) if isinstance(cfg, dict) else True
        unit = "dimensionless" if normalize else "nats"
        add("clean_permutation_entropy", unit, "permutation_entropy_ordinal_patterns",
            _permutation_entropy(values, complexity_config, cancel_callback=cancel_callback))
    if "clean_dfa_alpha" in requested:
        add("clean_dfa_alpha", "dimensionless", "detrended_fluctuation_analysis_order_1",
            _dfa(values, complexity_config, cancel_callback=cancel_callback))
    return metrics, diagnostics


def compute_clean_advanced_window(
    time_s: Any,
    clean_uS: Any,
    eligible: Any,
    *,
    window_start_s: float,
    window_end_s: float,
    declared_fs_hz: float | None,
    max_gap_s: float | None,
    jitter_tolerance: float,
    extended_config: dict[str, Any],
    selected_ids: Any,
    identity: dict[str, Any],
    cancel_callback: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Compute the selected experimental clean-basis metrics for one window.

    ``eligible`` must already combine the analysis mask, finite clean values,
    and the configured interpolation policy.  Within the half-open window this
    helper requires every sample to be eligible and on one contiguous uniform
    grid; it never drops an invalid sample or joins support across a gap.
    """
    extended = _extended_section(extended_config)
    selected = _requested_set(selected_ids)
    identity_fields = _identity_fields(identity)
    metrics: dict[str, dict[str, Any]] = {}
    tables = _empty_tables()
    diagnostic_rows: list[dict[str, Any]] = []
    spectra_rows: list[dict[str, Any]] = []

    spectral_config = _family_config(extended, "spectral")
    autocorrelation_config = _family_config(extended, "autocorrelation")
    complexity_config = _family_config(extended, "complexity")
    spectral_ids: set[str] = set()
    for band in spectral_config.get("bands", []) if isinstance(spectral_config.get("bands", []), list) else []:
        if not isinstance(band, dict) or not band.get("id"):
            continue
        band_id = str(band["id"])
        spectral_ids.update({f"clean_bandpower_{band_id}_uS2", f"clean_bandpower_{band_id}_normalized"})
    lags = autocorrelation_config.get("lags_s", [])
    lag_pairs: list[tuple[float, str]] = []
    if isinstance(lags, list):
        for lag in lags:
            if isinstance(lag, bool) or not isinstance(lag, (int, float)) or not math.isfinite(lag) or lag <= 0:
                continue
            try:
                feature_id = autocorrelation_feature_id(float(lag))
            except (ValueError, InvalidOperation):
                continue
            lag_pairs.append((float(lag), feature_id))
    known_ids = spectral_ids | {feature_id for _, feature_id in lag_pairs} | set(_STATIC_COMPLEXITY_IDS)
    requested = known_ids if selected is None else known_ids.intersection(selected)
    if not requested:
        return {"metrics": metrics, "tables": tables}

    support = _support(time_s, clean_uS, eligible,
                       window_start_s=window_start_s, window_end_s=window_end_s,
                       declared_fs_hz=declared_fs_hz, max_gap_s=max_gap_s,
                       jitter_tolerance=jitter_tolerance)
    if support["status"] != "valid":
        for feature_id in sorted(requested):
            family = ("spectral" if feature_id in spectral_ids else
                      "autocorrelation" if feature_id in {item[1] for item in lag_pairs} else "complexity")
            metric = _result(None, unit=_metric_unit(feature_id, complexity_config),
                             method="clean_advanced_window", reason=support["reason"],
                             valid_n=support["n"], valid_duration_s=support["duration_s"],
                             parameters=support["parameters"])
            metrics[feature_id] = metric
            diagnostic_rows.append({"feature_id": feature_id, "status": metric["status"],
                                    "reason": metric["reason"], "parameters": metric["parameters"],
                                    "diagnostics": {"support_status": support["status"]}})
        tables["complexity_diagnostics"] = pd.DataFrame(
            [{**identity_fields, "family": ("spectral" if row["feature_id"] in spectral_ids else
                                               "autocorrelation" if row["feature_id"] in {item[1] for item in lag_pairs} else "complexity"),
              "feature_id": row["feature_id"], "status": row["status"], "reason": row["reason"],
              "parameters_json": _json_text(row["parameters"]), "diagnostics_json": _json_text(row["diagnostics"])}
             for row in diagnostic_rows], columns=list(_DIAGNOSTIC_COLUMNS))
        return {"metrics": metrics, "tables": tables}

    _check_cancel(cancel_callback)
    if requested.intersection(spectral_ids) and spectral_config.get("enabled", True) is True:
        def run_spectral():
            values, spectra, diagnostics = _compute_spectral(
                support["time"], support["values"], support["fs_hz"], support["duration_s"],
                spectral_config, requested.intersection(spectral_ids))
            metrics.update(values)
            spectra_rows.extend(spectra)
            diagnostic_rows.extend({**row, "family": "spectral"} for row in diagnostics)
        _family_work(progress_callback, identity_fields, "spectral", "welch_bandpower", run_spectral)
    if requested.intersection({item[1] for item in lag_pairs}) and autocorrelation_config.get("enabled", True) is True:
        def run_autocorrelation():
            values, diagnostics = _compute_autocorrelation(
                support["time"], support["values"], support["fs_hz"], support["duration_s"],
                autocorrelation_config, [pair for pair in lag_pairs if pair[1] in requested])
            metrics.update(values)
            diagnostic_rows.extend({**row, "family": "autocorrelation"} for row in diagnostics)
        _family_work(progress_callback, identity_fields, "autocorrelation", "pearson_lag_correlation", run_autocorrelation)
    if requested.intersection(_STATIC_COMPLEXITY_IDS) and complexity_config.get("enabled", True) is True:
        def run_complexity():
            values, diagnostics = _compute_complexity(
                support["values"], support["duration_s"], complexity_config,
                requested.intersection(_STATIC_COMPLEXITY_IDS), cancel_callback=cancel_callback)
            metrics.update(values)
            diagnostic_rows.extend({**row, "family": "complexity"} for row in diagnostics)
        _family_work(progress_callback, identity_fields, "complexity", "clean_complexity", run_complexity)

    # If a configured metric's family is disabled, retain an explicit unavailable
    # result instead of silently dropping a selected identifier.
    for feature_id in sorted(requested - set(metrics)):
        family = "spectral" if feature_id in spectral_ids else "autocorrelation" if feature_id in {item[1] for item in lag_pairs} else "complexity"
        family_config = {"spectral": spectral_config, "autocorrelation": autocorrelation_config,
                         "complexity": complexity_config}[family]
        reason = "family_disabled" if family_config.get("enabled", True) is not True else "metric_not_computed"
        unit = _metric_unit(feature_id, complexity_config)
        metrics[feature_id] = _result(None, unit=unit, method=feature_id, reason=reason,
                                      valid_n=support["n"], valid_duration_s=support["duration_s"],
                                      parameters=support["parameters"])
        diagnostic_rows.append({"feature_id": feature_id, "family": family,
                                "status": "unavailable", "reason": reason,
                                "parameters": support["parameters"], "diagnostics": {}})

    tables["spectra"] = pd.DataFrame(
        [{**identity_fields, **row} for row in spectra_rows], columns=list(_SPECTRA_COLUMNS))
    tables["complexity_diagnostics"] = pd.DataFrame(
        [{**identity_fields, "family": row["family"], "feature_id": row["feature_id"],
          "status": row["status"], "reason": row["reason"],
          "parameters_json": _json_text(row["parameters"]),
          "diagnostics_json": _json_text(row.get("diagnostics", {}))}
         for row in diagnostic_rows], columns=list(_DIAGNOSTIC_COLUMNS))
    return {"metrics": metrics, "tables": tables}


__all__ = ["compute_clean_advanced_window"]
