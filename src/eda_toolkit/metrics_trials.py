"""事件试次表、逐试次摘要及描述性习惯化拟合。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .contracts import FEATURE_COLUMNS
from .metrics_common import TRANSFORMS, annotate_amplitudes


TRIAL_COLUMNS = ["trial_id", "recording_id", "channel", "subject_id", "session_id", "event_id",
                 "trial_ordinal", "condition", "event_type", "event_onset_s", "baseline_start_s",
                 "baseline_end_s", "response_start_s", "response_end_s", "trial_status", "reason",
                 "response_present", "amplitude_zero_uS", "amplitude_conditional_uS",
                 "peak_latency_s", "baseline_mean_uS", "baseline_sd_uS", "response_mean_uS",
                 "response_change_uS", "linked_scr_ids", "series_id"]
TRIAL_SUMMARY_COLUMNS = ["series_id", "recording_id", "channel", "subject_id", "session_id", "condition",
                         "event_type", "planned_n", "valid_n", "excluded_n", "exclusion_reasons_json",
                         "response_count", "nonresponse_count", "method"]
FIT_DIAGNOSTIC_COLUMNS = ["series_id", "feature_id", "status", "reason", "valid_n", "parameters_json",
                          "rmse", "jacobian_rank", "converged", "initialization_json", "method"]


def _value(row, key, default=None):
    value = row.get(key, default)
    if value is None or (np.isscalar(value) and pd.isna(value)):
        return default
    return value


def _eligible_duration(time, mask, start, end, max_gap):
    total = 0.0
    for idx in range(max(0, len(time) - 1)):
        dt = float(time[idx + 1] - time[idx])
        if mask[idx] and mask[idx + 1] and dt > 0 and (max_gap is None or dt <= max_gap):
            total += max(0.0, min(end, float(time[idx + 1])) - max(start, float(time[idx])))
    return total


def build_trial_table(bundle, scr, windows, event_links, config):
    """从事件窗、唯一 event-SCR 链接及 clean baseline/response 信号建立试次。"""
    if not isinstance(windows, pd.DataFrame):
        windows = pd.DataFrame(windows)
    if not isinstance(event_links, pd.DataFrame):
        event_links = pd.DataFrame(event_links)
    time = np.asarray(bundle.get("time_s", []), float)
    clean = np.asarray(bundle.get("clean_uS", []), float)
    quality = bundle.get("quality", {}) or {}
    eligible = np.asarray(quality.get("analysis_eligible", quality.get("eligible", np.ones(len(time), bool))), bool)
    if eligible.shape != time.shape:
        eligible = np.zeros(len(time), bool)
    interpolated = np.asarray(quality.get("interpolated", np.zeros(len(time), bool)), bool)
    if interpolated.shape == time.shape and config.get("features", {}).get("exclude_interpolated", True):
        eligible = eligible & ~interpolated
    eligible = eligible & np.isfinite(clean)
    max_gap = bundle.get("max_gap_s")
    windows_rows = windows.to_dict("records")
    responses = [r for r in windows_rows if r.get("kind") == "event_response" and _value(r, "event_id") is not None]
    baselines = {str(r.get("event_id")): r for r in windows_rows if r.get("kind") == "event_baseline"}
    scr_rows = scr.to_dict("records") if isinstance(scr, pd.DataFrame) else []
    rid = str(bundle.get("metadata", {}).get("recording_id", bundle.get("recording_id", "")))
    channel = str(bundle.get("metadata", {}).get("channel", bundle.get("channel", "eda")))
    subject = bundle.get("metadata", {}).get("subject_id")
    session = bundle.get("metadata", {}).get("session_id")
    links_by_event = {}
    for link in event_links.to_dict("records"):
        if _value(link, "event_id") is not None:
            links_by_event.setdefault(str(link["event_id"]), []).append(link)
    rows = []
    for fallback_ordinal, response in enumerate(responses, 1):
        event_id = str(response["event_id"])
        baseline = baselines.get(event_id)
        event_onset = float(_value(response, "event_onset_s", np.nan))
        ordinal = int(_value(response, "event_ordinal", fallback_ordinal))
        start_r, end_r = float(response["start_s"]), float(response["end_s"])
        start_b = float(baseline["start_s"]) if baseline is not None else np.nan
        end_b = float(baseline["end_s"]) if baseline is not None else np.nan
        response_mask = (time >= start_r) & (time < end_r) & eligible
        baseline_mask = (time >= start_b) & (time < end_b) & eligible if baseline is not None else np.zeros(len(time), bool)
        response_samples = clean[response_mask]
        baseline_samples = clean[baseline_mask]
        response_duration = _eligible_duration(time, eligible, start_r, end_r, max_gap)
        baseline_duration = _eligible_duration(time, eligible, start_b, end_b, max_gap) if baseline is not None else 0.0
        resp_complete = (response_duration + 1e-9 >= max(0.0, end_r - start_r) and
                         len(response_samples) > 0 and not bool(_value(response, "condition_crossing", False)))
        base_complete = (baseline is not None and baseline_duration + 1e-9 >= max(0.0, end_b - start_b) and
                         len(baseline_samples) > 0)
        baseline_contamination = any(start_b <= float(item.get("onset_s", np.nan)) < end_b
                                     for item in scr_rows if baseline is not None and
                                     np.isfinite(float(_value(item, "onset_s", np.nan))))
        link_rows = links_by_event.get(event_id, [])
        link_statuses = {str(item.get("link_status", "")) for item in link_rows}
        linked_ids = [str(item["scr_id"]) for item in link_rows
                      if item.get("link_status") == "unique" and _value(item, "scr_id") is not None]
        linked = [item for item in scr_rows if str(_value(item, "scr_id", "")) in set(linked_ids)]
        # Every event must have exactly one unique/no-response resolution; ambiguity or partial context excludes it.
        if not resp_complete:
            status, reason = "excluded", "response_window_invalid_or_incomplete"
        elif not base_complete:
            status, reason = "excluded", "baseline_window_invalid_or_incomplete"
        elif baseline_contamination:
            status, reason = "excluded", "baseline_contaminated_by_scr"
        elif "ambiguous" in link_statuses:
            status, reason = "excluded", "ambiguous_event_scr_assignment"
        elif "unavailable_onset_context" in link_statuses or "invalid_window" in link_statuses:
            status, reason = "excluded", "event_onset_context_unavailable"
        elif not ("unique" in link_statuses or "no_response" in link_statuses):
            status, reason = "excluded", "event_scr_assignment_missing"
        else:
            status, reason = "valid", ""
        baseline_mean = float(np.mean(baseline_samples)) if len(baseline_samples) else math.nan
        baseline_sd = float(np.std(baseline_samples, ddof=1)) if len(baseline_samples) >= 3 else math.nan
        response_mean = float(np.mean(response_samples)) if len(response_samples) else math.nan
        response_change = response_mean - baseline_mean if np.isfinite(response_mean) and np.isfinite(baseline_mean) else math.nan
        linked.sort(key=lambda item: (float(_value(item, "peak_s", np.inf)), -float(_value(item, "amplitude_uS", 0.0))))
        strongest = sorted(linked, key=lambda item: (-float(_value(item, "amplitude_uS", -np.inf)),
                                                     float(_value(item, "peak_s", np.inf))))
        responder = bool(linked)
        amplitude = float(strongest[0]["amplitude_uS"]) if responder else 0.0
        peak_latency = float(strongest[0]["peak_s"]) - event_onset if responder and np.isfinite(event_onset) else math.nan
        series_by = config.get("features", {}).get("extended", {}).get("trial_series", {}).get(
            "series_by", ["condition", "event_type"])
        condition = _value(response, "condition")
        event_type = _value(response, "event_type")
        series_parts = [str(subject or ""), str(session or "")]
        for field in series_by:
            series_parts.append(str(condition if field == "condition" else event_type if field == "event_type"
                                    else _value(response, field, "")))
        series_id = "trial-series:" + "|".join(series_parts)
        rows.append({"trial_id": f"{rid}/{channel}/{event_id}", "recording_id": rid, "channel": channel,
                     "subject_id": subject, "session_id": session, "event_id": event_id,
                     "trial_ordinal": ordinal, "condition": condition, "event_type": event_type,
                     "event_onset_s": event_onset, "baseline_start_s": start_b, "baseline_end_s": end_b,
                     "response_start_s": start_r, "response_end_s": end_r, "trial_status": status,
                     "reason": reason, "response_present": int(responder) if status == "valid" else np.nan,
                     "amplitude_zero_uS": amplitude if status == "valid" else np.nan,
                     "amplitude_conditional_uS": amplitude if status == "valid" and responder else np.nan,
                     "peak_latency_s": peak_latency if status == "valid" and responder else np.nan,
                     "baseline_mean_uS": baseline_mean, "baseline_sd_uS": baseline_sd,
                     "response_mean_uS": response_mean, "response_change_uS": response_change,
                     "linked_scr_ids": linked_ids if status == "valid" else [], "series_id": series_id})
    table = pd.DataFrame(rows, columns=TRIAL_COLUMNS)
    return annotate_amplitudes(table, "amplitude_zero_uS", table["trial_status"].eq("valid"))


def summarize_trial_series(trial_table, config):
    """按 subject/session/series_by 输出试次数、零包含/条件幅值与描述趋势。

    有界衰减模型固定为 ``c + a*exp(-k*(ordinal-xmin))`` 且 ``a,c,k>=0``。
    只有相对嵌套常量模型的残差平方和改善超过浮点求和误差，且自然参数尺度化
    Jacobian 未出现数值病态、参数未贴在 ``a/k`` 边界时才报告 ``k`` 和 ``1/k``。
    因此优化器收敛到近常量拟合本身不代表衰减率可识别；拟合诊断保留常量对照、
    参数尺度、尺度化奇异值/条件数和边界信息。
    """
    if not isinstance(trial_table, pd.DataFrame):
        trial_table = pd.DataFrame(trial_table)
    feature_rows, summary_rows, fit_rows = [], [], []
    if trial_table.empty:
        return (pd.DataFrame(columns=FEATURE_COLUMNS), pd.DataFrame(columns=TRIAL_SUMMARY_COLUMNS),
                pd.DataFrame(columns=FIT_DIAGNOSTIC_COLUMNS))
    trial_table = annotate_amplitudes(trial_table, "amplitude_zero_uS", trial_table["trial_status"].eq("valid"))
    settings = config.get("features", {}).get("extended", {}).get("trial_series", {})
    metadata_cols = [c for c in ("recording_id", "subject_id", "session_id") if c in trial_table]
    group_cols = [c for c in ("series_id", *metadata_cols, "channel", "condition", "event_type") if c in trial_table]
    grouped = trial_table.groupby(group_cols, dropna=False, sort=False)

    def emit(series_id, meta, feature_id, value, unit, basis, method, status, reason, n, duration=0.0):
        feature_rows.append({"recording_id": meta.get("recording_id", ""), "subject_id": meta.get("subject_id"),
                             "session_id": meta.get("session_id"), "window_id": series_id,
                             "feature_id": feature_id, "value": value if status == "valid" else np.nan,
                             "unit": unit, "basis": basis, "method": method, "status": status,
                             "reason": reason or "", "valid_n": int(n), "valid_duration_s": float(duration)})

    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(group_cols, key))
        series_id = str(meta.get("series_id", "trial-series"))
        valid = group.loc[group["trial_status"].eq("valid")].sort_values("trial_ordinal", kind="stable")
        excluded = group.loc[~group["trial_status"].eq("valid")]
        reasons = excluded["reason"].value_counts().to_dict()
        responses = valid["response_present"].to_numpy(dtype=float)
        amplitudes = valid["amplitude_zero_uS"].to_numpy(dtype=float)
        responders = valid.loc[valid["response_present"].eq(1), "amplitude_conditional_uS"].to_numpy(dtype=float)
        summary_rows.append({"series_id": series_id, "recording_id": meta.get("recording_id", ""),
                             "channel": str(group.iloc[0].get("channel", "")), "subject_id": meta.get("subject_id"),
                             "session_id": meta.get("session_id"), "condition": meta.get("condition"),
                             "event_type": meta.get("event_type"), "planned_n": int(len(group)),
                             "valid_n": int(len(valid)), "excluded_n": int(len(excluded)),
                             "exclusion_reasons_json": __import__("json").dumps(reasons, sort_keys=True),
                             "response_count": int(np.sum(responses)) if len(valid) else 0,
                             "nonresponse_count": int(np.sum(responses == 0)) if len(valid) else 0,
                             "method": "trial_counts; original_trial_ordinal_preserved"})
        planned, n = len(group), len(valid)
        for feature_id, value, unit, method in (
                ("event_response_probability", float(np.mean(responses)) if n else None, "probability", "sum_response_present/valid_trial_n"),
                ("event_nonresponse_rate", float(1 - np.mean(responses)) if n else None, "probability", "one_minus_response_probability"),
                ("event_amplitude_zero_mean_uS", float(np.mean(amplitudes)) if n else None, "uS", "mean_amplitude_zero_including_nonresponse_zero"),
                ("event_amplitude_conditional_mean_uS", float(np.mean(responders)) if len(responders) else None, "uS", "mean_responder_trial_maximum_only")):
            status = "valid" if n and (feature_id != "event_amplitude_conditional_mean_uS" or len(responders)) else "unavailable"
            reason = "" if status == "valid" else "no_valid_trials" if not n else "no_responding_trials"
            emit(series_id, meta, feature_id, value, unit, "valid_event_trials", method, status, reason,
                 n if feature_id != "event_amplitude_conditional_mean_uS" else len(responders))

        x = valid["trial_ordinal"].to_numpy(dtype=float)
        for transform, (_, unit, method) in TRANSFORMS.items():
            column = f"amplitude_zero_{transform}"
            statuses = valid[f"{column}_status"]
            okay = bool(n and statuses.eq("valid").all())
            reason = ("no_valid_trials" if not n else "" if okay else
                      str(valid.loc[statuses.ne("valid"), f"{column}_reason"].iloc[0]))
            emit(series_id, meta, f"event_amplitude_zero_{transform}_mean",
                 float(valid[column].mean()) if okay else None, unit, "valid_event_trials_zero_inclusive",
                 method, "valid" if okay else "unavailable", reason, n)
        if n >= int(settings.get("min_slope_trials", 5)) and len(np.unique(x)) >= 2:
            slope = float(np.polyfit(x, amplitudes, 1)[0])
            amp_status, amp_reason = "valid", ""
        else:
            slope, amp_status, amp_reason = None, "unavailable", "requires_minimum_valid_trials_and_ordinal_span"
        emit(series_id, meta, "habituation_amplitude_slope_uS_per_trial", slope, "uS/trial",
             "valid_event_trials_zero_inclusive", "OLS(amplitude_zero_uS~original_trial_ordinal)", amp_status,
             amp_reason, n)
        min_logit = int(settings.get("min_logit_trials", 8))
        logit_slope = None
        probability_change = None
        logit_status, logit_reason = "unavailable", "requires_minimum_valid_trials_both_outcomes_and_full_rank"
        fit_params = {}
        if n >= min_logit and len(np.unique(x)) >= 2 and len(np.unique(responses)) == 2:
            try:
                import statsmodels.api as sm
                design = sm.add_constant(x, has_constant="add")
                rank = int(np.linalg.matrix_rank(design))
                if rank < 2:
                    logit_reason = "rank_deficient_design"
                else:
                    fit = sm.GLM(responses, design, family=sm.families.Binomial()).fit()
                    converged = bool(getattr(fit, "converged", False))
                    predicted = np.asarray(fit.predict(design), float)
                    separated = bool(np.any(predicted < 1e-10) or np.any(predicted > 1 - 1e-10) or
                                     np.any(np.abs(np.asarray(fit.params, float)) > 1e6))
                    if converged and not separated and np.all(np.isfinite(fit.params)):
                        logit_slope = float(fit.params[1])
                        prediction = 1 / (1 + np.exp(-np.clip(np.asarray(fit.params[0]) +
                                                              np.asarray(fit.params[1]) * np.array([np.min(x), np.max(x)]),
                                                              -700, 700)))
                        probability_change = float(prediction[1] - prediction[0])
                        logit_status, logit_reason = "valid", ""
                        fit_params = {"intercept": float(fit.params[0]), "slope": logit_slope,
                                      "rank": rank, "converged": converged}
                    else:
                        logit_reason = "separation_or_nonconverged_fit" if separated or not converged else "nonfinite_fit"
            except Exception as exc:  # statsmodels signals perfect separation/version-specific failures
                logit_reason = "separation_or_fit_error:" + type(exc).__name__
        emit(series_id, meta, "habituation_response_logit_slope_per_trial", logit_slope, "log-odds/trial",
             "valid_event_trials", "binomial_GLM_response_present~ordinal", logit_status, logit_reason, n)
        emit(series_id, meta, "habituation_response_probability_change", probability_change, "probability",
             "valid_event_trials", "GLM_fitted_probability_last_minus_first_ordinal", logit_status,
             logit_reason, n)
        fit_rows.append({"series_id": series_id, "feature_id": "habituation_response_logit_slope_per_trial",
                         "status": logit_status, "reason": logit_reason, "valid_n": n,
                         "parameters_json": __import__("json").dumps(fit_params, sort_keys=True),
                         "rmse": np.nan, "jacobian_rank": fit_params.get("rank", np.nan),
                         "converged": fit_params.get("converged", False),
                         "initialization_json": "{}", "method": "statsmodels.GLM Binomial"})

        decay_settings = settings
        minimum_decay = int(decay_settings.get("min_decay_trials", 8))
        distinct = np.unique(x)
        decay_status, decay_reason = "unavailable", "requires_minimum_trials_four_ordinals_and_amplitude_variation"
        decay_values = {"habituation_decay_rate_per_trial": None, "habituation_decay_tau_trials": None,
                        "habituation_decay_asymptote_uS": None}
        diag = {"a": np.nan, "c": np.nan, "k": np.nan, "rmse": np.nan, "rank": np.nan,
                "condition": np.nan, "constant_rss": np.nan, "decay_rss": np.nan,
                "rss_improvement": np.nan, "rss_tolerance": np.nan,
                "exponential_change_uS": np.nan, "relative_exponential_change": np.nan,
                "active_bounds": [], "scaled_singular_values": [], "parameter_scales": [],
                "converged": False, "init": {}}
        if n >= minimum_decay and len(distinct) >= 4 and np.ptp(amplitudes) > 0:
            xmin = float(np.min(x))
            amplitude_scale = max(float(np.max(np.abs(amplitudes))), float(np.ptp(amplitudes)),
                                  np.finfo(float).tiny)
            normalized_amplitudes = amplitudes / amplitude_scale
            c0, a0, k0 = 0.0, max(float(np.max(normalized_amplitudes)), 1e-12), 0.1
            init = {"c_normalized": c0, "a_normalized": a0, "k": k0,
                    "amplitude_scale_uS": amplitude_scale,
                    "scheme": "fixed_normalized_c0_0_a0_max_scaled_amplitude_k0_0.1"}
            try:
                result = least_squares(lambda p: p[0] + p[1] * np.exp(-p[2] * (x - xmin)) - normalized_amplitudes,
                                       x0=np.array([c0, a0, k0]), bounds=([0.0, 0.0, 0.0],
                                                                            [np.inf, np.inf, float(decay_settings.get("decay_max_rate", 10.0))]),
                                       max_nfev=int(decay_settings.get("decay_max_nfev", 1000)))
                c_normalized, a_normalized, k = map(float, result.x)
                c, a = c_normalized * amplitude_scale, a_normalized * amplitude_scale
                residual_normalized = np.asarray(result.fun, dtype=float)
                residual = residual_normalized * amplitude_scale
                rmse = float(np.sqrt(np.mean(residual * residual)))
                decay_rss_normalized = float(np.dot(residual_normalized, residual_normalized))
                constant_normalized = max(0.0, float(np.mean(normalized_amplitudes)))
                constant_residual_normalized = normalized_amplitudes - constant_normalized
                constant_rss_normalized = float(np.dot(constant_residual_normalized,
                                                       constant_residual_normalized))
                decay_rss_uS2 = decay_rss_normalized * amplitude_scale * amplitude_scale
                constant_rss_uS2 = constant_rss_normalized * amplitude_scale * amplitude_scale
                ordinal_span = max(float(np.ptp(x)), 1.0)
                # The fit's amplitude parameters are normalized, so [1,1,1/span]
                # is the natural dimensionless scale for its Jacobian columns.
                parameter_scales = np.array([1.0, 1.0, 1.0 / ordinal_span])
                scaled_jacobian = np.asarray(result.jac, dtype=float) * parameter_scales[None, :]
                singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
                rank_tolerance = (np.finfo(float).eps * max(scaled_jacobian.shape) * singular_values[0]
                                  if len(singular_values) and singular_values[0] > 0 else 0.0)
                jac_rank = int(np.count_nonzero(singular_values > rank_tolerance))
                condition = (float(singular_values[0] / singular_values[-1])
                             if len(singular_values) and singular_values[-1] > 0 else math.inf)
                rss_scale_normalized = max(float(np.dot(normalized_amplitudes, normalized_amplitudes)),
                                constant_rss_normalized, decay_rss_normalized,
                                np.finfo(float).tiny)
                rss_tolerance_normalized = (np.finfo(float).eps * max(len(amplitudes), 3) *
                                            rss_scale_normalized)
                rss_improvement_normalized = constant_rss_normalized - decay_rss_normalized
                exponential_change_normalized = abs(a_normalized) * abs(float(-np.expm1(-k * ordinal_span)))
                exponential_change = exponential_change_normalized * amplitude_scale
                relative_change = exponential_change_normalized / max(
                    float(np.max(np.abs(normalized_amplitudes))), np.finfo(float).tiny)
                rss_tolerance_uS2 = rss_tolerance_normalized * amplitude_scale * amplitude_scale
                rss_improvement_uS2 = rss_improvement_normalized * amplitude_scale * amplitude_scale
                active_bounds = ["c:lower" if result.active_mask[0] < 0 else
                                 "c:upper" if result.active_mask[0] > 0 else None,
                                 "a:lower" if result.active_mask[1] < 0 else
                                 "a:upper" if result.active_mask[1] > 0 else None,
                                 "k:lower" if result.active_mask[2] < 0 else
                                 "k:upper" if result.active_mask[2] > 0 else None]
                active_bounds = [item for item in active_bounds if item is not None]
                diag = {"a": a, "c": c, "k": k, "rmse": rmse, "rank": jac_rank,
                        "condition": condition, "constant_rss_uS2": constant_rss_uS2,
                        "decay_rss_uS2": decay_rss_uS2,
                        "rss_improvement_uS2": rss_improvement_uS2,
                        "rss_tolerance_uS2": rss_tolerance_uS2,
                        "constant_rss_normalized": constant_rss_normalized,
                        "decay_rss_normalized": decay_rss_normalized,
                        "rss_improvement_normalized": rss_improvement_normalized,
                        "rss_tolerance_normalized": rss_tolerance_normalized,
                        "model_amplitude_scale_uS": amplitude_scale,
                        "exponential_change_uS": exponential_change,
                        "exponential_change_normalized": exponential_change_normalized,
                        "relative_exponential_change": relative_change,
                        "active_bounds": active_bounds,
                        "scaled_singular_values": singular_values.tolist(),
                        "parameter_scales": parameter_scales.tolist(),
                        "converged": bool(result.success), "init": init}
                # First compare against the nested constant model. A converged
                # solver can otherwise manufacture tiny positive a/k values at
                # the constant boundary and imply an enormous, unsupported tau.
                if not result.success:
                    decay_reason = "decay_fit_not_converged"
                elif rss_improvement_normalized <= rss_tolerance_normalized:
                    decay_reason = "no_identifiable_decay_beyond_constant_boundary"
                elif result.active_mask[1] != 0:
                    decay_reason = "exponential_amplitude_at_parameter_boundary"
                elif result.active_mask[2] != 0:
                    decay_reason = "decay_rate_at_parameter_boundary"
                elif exponential_change_normalized <= math.sqrt(np.finfo(float).eps):
                    decay_reason = "exponential_change_below_numerical_resolution"
                elif jac_rank < 3:
                    decay_reason = "unidentifiable_scaled_jacobian_rank"
                elif condition >= 1.0 / math.sqrt(np.finfo(float).eps):
                    decay_reason = "unidentifiable_scaled_jacobian_condition"
                else:
                    decay_status, decay_reason = "valid", ""
                    decay_values = {"habituation_decay_rate_per_trial": k,
                                    "habituation_decay_tau_trials": 1.0 / k,
                                    "habituation_decay_asymptote_uS": c}
            except Exception as exc:
                decay_reason = "decay_fit_error:" + type(exc).__name__
        for fid, unit in (("habituation_decay_rate_per_trial", "1/trial"),
                          ("habituation_decay_tau_trials", "trials"),
                          ("habituation_decay_asymptote_uS", "uS")):
            emit(series_id, meta, fid, decay_values[fid], unit, "valid_event_trials_zero_inclusive",
                 "amplitude_normalized_bounded_least_squares_c_plus_a_exp_minus_k_times_trial_delta",
                 decay_status, decay_reason, n)
        fit_rows.append({"series_id": series_id, "feature_id": "habituation_decay_rate_per_trial",
                         "status": decay_status, "reason": decay_reason, "valid_n": n,
                         "parameters_json": __import__("json").dumps(
                             {key: (value if isinstance(value, list) else
                                    float(value) if np.isscalar(value) and np.isfinite(value) else None)
                              for key, value in diag.items()
                              if key not in {"rmse", "rank", "converged", "init"}}, sort_keys=True),
                         "rmse": diag["rmse"], "jacobian_rank": diag["rank"], "converged": diag["converged"],
                         "initialization_json": __import__("json").dumps(diag["init"], sort_keys=True),
                         "method": "bounded scipy.optimize.least_squares_normalized_amplitude"})
    return (pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS),
            pd.DataFrame(summary_rows, columns=TRIAL_SUMMARY_COLUMNS),
            pd.DataFrame(fit_rows, columns=FIT_DIAGNOSTIC_COLUMNS))
