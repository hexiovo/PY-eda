"""Trough-to-peak SCR、半开窗口及单记录样本/暴露特征。"""

from __future__ import annotations

import math
import copy

import numpy as np
import pandas as pd
from scipy import signal

from .contracts import FEATURE_COLUMNS, SCR_COLUMNS, WINDOW_COLUMNS
from .core_preprocess import _full_config, _nominal_dt, _segments
from .core_preprocess import _bool_array
from .errors import fail
from .metrics_ids import autocorrelation_feature_id
from .metrics_common import (COMMON_DEFINITIONS, COMMON_IDS, DISTRIBUTIONS, TRANSFORMS,
                             amplitude_reference, annotate_amplitudes, common_parameters,
                             distribution_summary, transform_amplitudes)
from .metrics_nonspecific import classify_nonspecific_scr
from .metrics_trials import build_trial_table, summarize_trial_series
from .metrics_driver import DRIVER_SPECS, DRIVER_IDS, driver_bursts, driver_window_metrics


SCR_EXTRA_COLUMNS = ["onset_level_uS", "peak_level_uS", "recovery_fraction_returned", "threshold_type",
                     "threshold_value_uS", "signal_basis", "unit", "method", "onset_sample_index",
                     "peak_sample_index", "recovery_censor_reason", "area_baseline", "area_bounds"]
SCR_EXTRA_COLUMNS += ["onset_eligible", "qualification_reason"]
SCR_EXTRA_COLUMNS += ["segment_id", "rise_slope_uS_s", "rise_max_slope_uS_s",
                      "recovery_50_s", "recovery_50_time_s", "recovery_50_status",
                      "recovery_50_reason", "recovery_63_s", "recovery_63_time_s",
                      "recovery_63_status", "recovery_63_reason", "half_width_50_s",
                      "half_width_50_status", "half_width_50_reason"]
WINDOW_EXTRA_COLUMNS = ["window_status", "window_reason", "event_onset_s", "event_type", "interval_id",
                        "source_start_s", "source_end_s", "condition_crossing", "condition_labels",
                        "event_ordinal", "channel"]
VALIDITY_COLUMNS = ["recording_id", "window_id", "kind", "start_s", "end_s", "status", "reason",
                    "coverage", "expected_duration_s", "valid_duration_s", "detector_eligible_duration_s",
                    "search_eligible_duration_s", "onset_eligible_duration_s", "onset_left_context_s",
                    "onset_right_context_s", "onset_context_complete", "valid_n", "interpolated_n",
                    "artifact_n", "gap_crossing", "policy_id"]
EVENT_LINK_COLUMNS = ["recording_id", "event_id", "scr_id", "link_status", "link_basis", "reason"]
LEGACY_FEATURE_IDS = (
    "valid_sample_proportion", "interpolated_sample_proportion", "artifact_sample_proportion",
    "mean", "median", "range", "sd", "iqr", "slope", "tonic_mean", "tonic_mean_change",
    "phasic_auc", "scr_count", "scr_rate", "scr_amplitude_mean", "scr_amplitude_median",
    "scr_amplitude_sum", "scr_rise_time_mean", "scr_recovery_time_mean", "scr_area_sum_uS_s",
    "scr_area_mean_uS_s", "event_baseline_mean_uS", "event_scr_count", "event_response_present",
    "event_first_amplitude_uS", "event_max_amplitude_uS", "event_amp_sum_uS", "event_latency_s",
    "event_phasic_auc_uS_s", "event_response_mean_change_uS", "event_peak_max_amplitude_uS",
    "event_recovery_mean_s", "event_scr_area_sum_uS_s",
)

_ENGINEERING_QUALIFICATION = "experimental_engineering_checked_not_real_data_qualified"


def _definition(feature_id, label, family, formula, basis, unit, min_n, *, duration=0.0,
                parameters=None, validity="finite eligible samples/support", qualification=None):
    return {"id": feature_id, "label_zh": label, "family": family, "formula": formula,
            "basis": basis, "unit": unit, "min_n": int(min_n), "min_duration_s": float(duration),
            "parameters": dict(parameters or {}), "validity": validity,
            "qualification": qualification or _ENGINEERING_QUALIFICATION}


def _definition_parameters(feature_id, extended):
    """生成registry中与具体指标相关的冻结统计参数快照。"""
    parameters = {}
    if feature_id == "sd" or feature_id.endswith("_sd") or "_sd_" in feature_id:
        parameters["ddof"] = 1
    if "iqr" in feature_id or "p90" in feature_id:
        parameters["percentile_method"] = "linear"
    if feature_id.endswith("_p90"):
        parameters["percentile"] = 90
    if "iqr" in feature_id:
        parameters["quartiles"] = [25, 75]
    if feature_id.startswith("phasic_") and feature_id.endswith("auc_uS_s"):
        parameters.update({"integration": "piecewise_linear_trapezoid",
                           "zero_crossing": "exact_linear_split",
                           "gap_policy": "do_not_bridge"})
    if feature_id.startswith("scr_recovery_"):
        parameters["return_fraction"] = 0.50 if "_50_" in feature_id else 0.63
        parameters["threshold"] = "onset_level + (1-return_fraction)*amplitude"
        parameters.update({"crossing": "first_downward_linear_interpolation",
                           "search_stop": "next_response_onset_or_segment_end"})
    if feature_id.startswith("scr_half_width_"):
        parameters.update({"height_fraction": 0.50,
                           "crossings": "first_upward_then_first_downward_linear_interpolation"})
    if feature_id.startswith("scr_isi_"):
        parameters.update({"scope": "consecutive_onsets_within_window_and_same_segment",
                           "never_bridge_gaps": True})
    if feature_id.startswith("event_") and "baseline_" in feature_id:
        parameters["baseline_min_uS"] = float(extended.get("baseline_min_uS", 1e-6))
        if feature_id.endswith("_z"):
            parameters.update({"baseline_sd_ddof": 1, "minimum_baseline_n": 3})
    if feature_id.startswith("habituation_"):
        parameters.update(copy.deepcopy(extended.get("trial_series", {})))
    if feature_id.startswith("sync_") or feature_id.startswith("node_strength_"):
        parameters.update(copy.deepcopy(extended.get("group_temporal", {})))
    if feature_id == "sync_lag_sd_s":
        parameters.update({"ddof": 0,
                           "weighting": "nonoverlapping_midpoint_cell_intersect_pair_eligible_support",
                           "estimator": "duration_weighted_population_standard_deviation"})
    elif feature_id == "sync_lag_stability_proportion":
        parameters.update({"weighting": "nonoverlapping_midpoint_cell_intersect_pair_eligible_support",
                           "median": "duration_weighted_median"})
    if feature_id.startswith("ns_scr_"):
        parameters.update({"annotation_coverage": "explicit_complete_only",
                           "event_response_s": copy.deepcopy(_full_config({}).get("windows", {}).get("event_response_s"))})
    return parameters


_LEGACY_LABELS = {
    "valid_sample_proportion": ("有效样本比例", "fraction", "valid sample count / recorded window sample count", "quality"),
    "interpolated_sample_proportion": ("插值样本比例", "fraction", "interpolated sample count / recorded window sample count", "quality"),
    "artifact_sample_proportion": ("伪迹样本比例", "fraction", "artifact sample count / recorded window sample count", "quality"),
    "mean": ("均值", "uS", "eligible clean sample mean", "clean_uS"),
    "median": ("中位数", "uS", "eligible clean sample median", "clean_uS"),
    "range": ("极差", "uS", "sample max - sample min", "clean_uS"),
    "sd": ("标准差", "uS", "sample SD, ddof=1", "clean_uS"),
    "iqr": ("四分位距", "uS", "Q75-Q25", "clean_uS"),
    "slope": ("线性斜率", "uS/s", "OLS(clean_uS ~ time_s)", "clean_uS"),
    "tonic_mean": ("tonic均值", "uS", "eligible tonic sample mean", "tonic_uS"),
    "tonic_mean_change": ("tonic均值变化", "uS", "tonic sample mean - first eligible tonic sample", "tonic_uS"),
    "phasic_auc": ("phasic带符号面积", "uS_s", "clipped signed trapezoidal integral", "phasic_uS"),
    "scr_count": ("SCR计数", "count", "qualified SCR onsets in [start,end)", "qualified_global_scr_catalogue"),
    "scr_rate": ("SCR速率", "responses/min", "60 * qualified count / onset-eligible exposure", "qualified_global_scr_catalogue"),
    "scr_amplitude_mean": ("SCR幅值均值", "uS", "mean qualified SCR amplitude", "qualified_global_scr_catalogue"),
    "scr_amplitude_median": ("SCR幅值中位数", "uS", "median qualified SCR amplitude", "qualified_global_scr_catalogue"),
    "scr_amplitude_sum": ("SCR幅值和", "uS", "sum qualified SCR amplitude; zero-inclusive", "qualified_global_scr_catalogue"),
    "scr_rise_time_mean": ("SCR上升时间均值", "s", "mean SCR rise duration", "qualified_global_scr_catalogue"),
    "scr_recovery_time_mean": ("SCR恢复时间均值", "s", "mean completed recovery duration", "qualified_global_scr_catalogue"),
    "scr_area_sum_uS_s": ("SCR上升段基线差面积和", "uS_s", "sum signed onset-to-peak area above onset level", "qualified_global_scr_catalogue.area_uS_s"),
    "scr_area_mean_uS_s": ("SCR上升段基线差面积均值", "uS_s", "mean signed onset-to-peak area above onset level", "qualified_global_scr_catalogue.area_uS_s"),
    "event_baseline_mean_uS": ("事件基线均值", "uS", "clean sample mean in configured baseline window", "clean_uS"),
    "event_scr_count": ("事件相关SCR计数", "count", "unique event-SCR links", "unique_event_scr_links"),
    "event_response_present": ("事件响应是否出现", "boolean", "unique event-linked SCR count > 0", "unique_event_scr_links"),
    "event_first_amplitude_uS": ("事件首个SCR幅值", "uS", "earliest linked SCR amplitude", "unique_event_scr_links"),
    "event_max_amplitude_uS": ("事件最大SCR幅值", "uS", "maximum linked SCR amplitude", "unique_event_scr_links"),
    "event_amp_sum_uS": ("事件SCR幅值和", "uS", "sum linked SCR amplitudes; zero-inclusive", "unique_event_scr_links"),
    "event_latency_s": ("事件首个SCR潜伏期", "s", "first linked SCR onset - event onset", "unique_event_scr_links"),
    "event_phasic_auc_uS_s": ("事件phasic带符号面积", "uS_s", "clipped signed phasic integral in response window", "phasic_uS"),
    "event_response_mean_change_uS": ("事件响应均值变化", "uS", "clean response sample mean - clean baseline sample mean", "clean_uS"),
    "event_peak_max_amplitude_uS": ("事件峰值窗最大SCR幅值", "uS", "maximum linked SCR amplitude by peak time in configured peak window", "unique_event_scr_links"),
    "event_recovery_mean_s": ("事件SCR完成恢复时间均值", "s", "mean completed linked SCR recovery duration", "unique_event_scr_links"),
    "event_scr_area_sum_uS_s": ("事件SCR上升段基线差面积和", "uS_s", "sum unique event-linked onset-to-peak area", "unique_event_scr_links.area_uS_s"),
}

_EXTENDED_FIXED = [
    ("tonic_sd", "tonic标准差", "tonic", "sample SD(ddof=1)", "tonic_uS", "uS", 2),
    ("tonic_median", "tonic中位数", "tonic", "sample median", "tonic_uS", "uS", 1),
    ("tonic_iqr", "tonic四分位距", "tonic", "linear Q75-Q25", "tonic_uS", "uS", 2),
    ("tonic_slope", "tonic线性斜率", "tonic", "OLS(tonic_uS ~ time_s)", "tonic_uS", "uS/s", 2),
    ("tonic_first_last_change_uS", "tonic首末变化", "tonic", "last eligible tonic sample - first eligible tonic sample", "tonic_uS", "uS", 2),
    ("scr_amplitude_sd", "SCR幅值标准差", "scr_distribution", "sample SD(ddof=1)", "qualified SCR amplitude_uS", "uS", 2),
    ("scr_amplitude_max", "SCR最大幅值", "scr_distribution", "max(amplitude_uS)", "qualified SCR amplitude_uS", "uS", 1),
    ("scr_amplitude_p90", "SCR幅值第90百分位", "scr_distribution", "linear percentile 90", "qualified SCR amplitude_uS", "uS", 1),
    ("scr_amplitude_iqr", "SCR幅值四分位距", "scr_distribution", "linear Q75-Q25", "qualified SCR amplitude_uS", "uS", 2),
    ("scr_rise_slope_mean_uS_s", "SCR平均上升斜率均值", "scr_rise", "mean(A/(peak_s-onset_s))", "qualified SCR", "uS/s", 1),
    ("scr_rise_max_slope_mean_uS_s", "SCR局部最大上升斜率均值", "scr_rise", "mean(per-SCR max adjacent diff(signal)/dt)", "qualified detector signal", "uS/s", 1),
    ("scr_rise_max_slope_max_uS_s", "SCR局部最大上升斜率最大值", "scr_rise", "max(per-SCR max adjacent diff(signal)/dt)", "qualified detector signal", "uS/s", 1),
    ("scr_recovery_50_mean_s", "SCR恢复50%时间均值", "scr_recovery", "mean observed peak-to-first-crossing duration", "qualified SCR", "s", 1),
    ("scr_recovery_63_mean_s", "SCR恢复63%时间均值", "scr_recovery", "mean observed peak-to-first-crossing duration", "qualified SCR", "s", 1),
    ("scr_unrecovered_50_proportion", "未观察到恢复比例（含截尾）50%", "scr_recovery", "right-censored responses / primary responses", "qualified SCR", "fraction", 1),
    ("scr_unrecovered_63_proportion", "未观察到恢复比例（含截尾）63%", "scr_recovery", "right-censored responses / primary responses", "qualified SCR", "fraction", 1),
    ("scr_half_width_50_mean_s", "SCR半高宽均值", "scr_width", "mean(half-height fall crossing - rise crossing)", "qualified SCR", "s", 1),
    ("scr_isi_mean_s", "SCR起始间隔均值", "scr_isi", "mean within-window consecutive same-segment onset intervals", "qualified SCR", "s", 1),
    ("scr_isi_median_s", "SCR起始间隔中位数", "scr_isi", "median within-window consecutive same-segment onset intervals", "qualified SCR", "s", 1),
    ("scr_isi_sd_s", "SCR起始间隔标准差", "scr_isi", "sample SD(ddof=1)", "qualified SCR", "s", 2),
    ("scr_isi_cv", "SCR起始间隔变异系数", "scr_isi", "sample SD / mean", "qualified SCR", "dimensionless", 2),
    ("phasic_positive_auc_uS_s", "phasic正向积分", "phasic_integral", "integral(max(y,0)) with exact linear zero-crossing split", "phasic_uS", "uS_s", 2),
    ("phasic_absolute_auc_uS_s", "phasic绝对积分", "phasic_integral", "integral(abs(y)) with exact linear zero-crossing split", "phasic_uS", "uS_s", 2),
    ("phasic_auc_per_valid_s", "phasic带符号时间均值", "phasic_integral", "signed AUC / valid phasic exposure", "phasic_uS", "uS", 2),
    ("phasic_positive_auc_per_valid_s", "phasic正向时间均值", "phasic_integral", "positive AUC / valid phasic exposure", "phasic_uS", "uS", 2),
    ("phasic_absolute_auc_per_valid_s", "phasic绝对时间均值", "phasic_integral", "absolute AUC / valid phasic exposure", "phasic_uS", "uS", 2),
    ("event_response_probability", "事件响应概率", "trial_series", "sum(response_present)/valid trial count", "unique valid event trials", "probability", 1),
    ("event_nonresponse_rate", "事件无响应率", "trial_series", "1-response_probability", "unique valid event trials", "probability", 1),
    ("event_amplitude_zero_mean_uS", "事件零包含最大SCR幅值均值", "trial_series", "mean per-trial amplitude_zero_uS including nonresponse zeros", "unique valid event trials", "uS", 1),
    ("event_amplitude_conditional_mean_uS", "事件响应条件幅值均值", "trial_series", "mean responder-only trial maximum amplitude", "responding valid event trials", "uS", 1),
    ("event_peak_latency_s", "事件最大SCR峰潜伏期", "event_trial", "peak_s of maximum linked SCR - event onset; earlier tie", "unique event-SCR links", "s", 1),
    ("event_max_amplitude_baseline_percent", "事件最大幅值/基线百分比", "event_trial", "100*amplitude_zero_uS/B", "clean baseline", "%", 1),
    ("event_response_change_baseline_percent", "事件响应变化/基线百分比", "event_trial", "100*(mean clean response-B)/B", "clean baseline and clean response", "%", 1),
    ("event_max_amplitude_baseline_z", "事件最大幅值/基线SD", "event_trial", "amplitude_zero_uS/SD_baseline(ddof=1)", "clean baseline", "z", 1),
    ("event_response_change_baseline_z", "事件响应变化/基线SD", "event_trial", "(mean clean response-B)/SD_baseline(ddof=1)", "clean baseline and clean response", "z", 1),
    ("habituation_amplitude_slope_uS_per_trial", "试次幅值趋势斜率", "habituation", "OLS(amplitude_zero_uS ~ 1 + original_trial_ordinal)", "valid event trials", "uS/trial", 5),
    ("habituation_response_logit_slope_per_trial", "试次响应logit趋势斜率", "habituation", "binomial GLM(response_present ~ 1 + original_trial_ordinal)", "valid event trials", "log-odds/trial", 8),
    ("habituation_response_probability_change", "试次响应概率变化", "habituation", "fitted probability(last ordinal)-fitted probability(first ordinal)", "valid event trials", "probability", 8),
    ("habituation_decay_rate_per_trial", "试次幅值衰减率", "habituation", "bounded fit c+a*exp(-k*(x-xmin)); k", "valid event trials incl. zero responses", "1/trial", 8),
    ("habituation_decay_tau_trials", "试次幅值衰减时间常数", "habituation", "1/k from identifiable bounded exponential fit", "valid event trials incl. zero responses", "trials", 8),
    ("habituation_decay_asymptote_uS", "试次幅值衰减渐近值", "habituation", "bounded fit c+a*exp(-k*(x-xmin)); c", "valid event trials incl. zero responses", "uS", 8),
    ("ns_scr_count", "非事件相关SCR计数", "nonspecific", "qualified non-event-associated onset count", "complete annotation-covered onset exposure", "count", 0),
    ("ns_scr_rate", "非事件相关SCR速率", "nonspecific", "60*count/onset exposure seconds", "complete annotation-covered onset exposure", "responses/min", 0),
    ("ns_scr_amplitude_mean", "非事件相关SCR幅值均值", "nonspecific", "mean qualifying amplitude", "complete annotation-covered SCRs", "uS", 1),
    ("ns_scr_amplitude_median", "非事件相关SCR幅值中位数", "nonspecific", "median qualifying amplitude", "complete annotation-covered SCRs", "uS", 1),
    ("ns_scr_amplitude_sd", "非事件相关SCR幅值标准差", "nonspecific", "sample SD(ddof=1)", "complete annotation-covered SCRs", "uS", 2),
    ("ns_scr_amplitude_max", "非事件相关SCR最大幅值", "nonspecific", "max qualifying amplitude", "complete annotation-covered SCRs", "uS", 1),
    ("ns_scr_amplitude_p90", "非事件相关SCR幅值第90百分位", "nonspecific", "linear percentile 90", "complete annotation-covered SCRs", "uS", 1),
    ("ns_scr_amplitude_iqr", "非事件相关SCR幅值四分位距", "nonspecific", "linear Q75-Q25", "complete annotation-covered SCRs", "uS", 2),
    ("sync_positive_time_proportion", "正同步有效时间比例", "group_temporal", "positive signed-r duration / valid class duration", "lag-zero group r and common eligible support", "fraction", 1),
    ("sync_negative_time_proportion", "负同步有效时间比例", "group_temporal", "negative signed-r duration / valid class duration", "lag-zero group r and common eligible support", "fraction", 1),
    ("sync_positive_episode_duration_mean_s", "完整正同步片段均值", "group_temporal", "mean duration of uncensored positive episodes", "lag-zero group r", "s", 1),
    ("sync_negative_episode_duration_mean_s", "完整负同步片段均值", "group_temporal", "mean duration of uncensored negative episodes", "lag-zero group r", "s", 1),
    ("sync_lag_sd_s", "同步滞后时间加权标准差", "group_temporal", "duration-weighted population SD(ddof=0) of configured selected lag", "pair/window selected lag; equal-pair group summary", "s", 3),
    ("sync_lag_stability_proportion", "同步滞后稳定比例", "group_temporal", "duration within tolerance of duration-weighted median / valid duration", "pair/window selected lag; equal-pair group summary", "fraction", 3),
    ("node_strength_signed", "节点有符号强度", "group_temporal", "sum of lag-zero signed r over fixed incident edges", "fixed graph", "dimensionless", 1),
    ("node_strength_positive", "节点正强度", "group_temporal", "sum max(r,0) over fixed incident edges", "fixed graph", "dimensionless", 1),
    ("node_strength_negative", "节点负强度", "group_temporal", "sum min(r,0) over fixed incident edges", "fixed graph", "dimensionless", 1),
    ("node_strength_absolute", "节点绝对强度", "group_temporal", "sum abs(r) over fixed incident edges", "fixed graph", "dimensionless", 1),
    ("node_strength_normalized_signed", "节点标准化有符号强度", "group_temporal", "signed strength / expected degree", "fixed graph", "dimensionless", 1),
]


def list_feature_definitions(config=None):
    """返回静态与动态特征定义，供配置校验、GUI选择器和导出使用。"""
    full = _full_config(config or {})
    definitions = {}
    for feature_id in LEGACY_FEATURE_IDS:
        label, unit, formula, basis = _LEGACY_LABELS[feature_id]
        min_n = 2 if feature_id in {"sd", "slope", "phasic_auc"} else 1
        if feature_id in {"scr_count", "event_scr_count", "event_response_present"}:
            min_n = 0
        min_duration = 1e-12 if feature_id in {"phasic_auc", "scr_rate"} else 0.0
        definitions[feature_id] = _definition(feature_id, label, "legacy", formula, basis, unit, min_n,
                                               duration=min_duration,
                                               qualification="legacy_v0.1_method_preserved_not_real_data_qualified")
    extended = full.get("features", {}).get("extended", {})
    if extended.get("enabled", True) is not True:
        return definitions
    disabled_families = set()
    if extended.get("nonspecific", {}).get("enabled", True) is not True:
        disabled_families.add("nonspecific")
    if extended.get("group_temporal", {}).get("enabled", True) is not True:
        disabled_families.add("group_temporal")
    if full.get("synchrony", {}).get("enabled", False) is not True:
        disabled_families.add("group_temporal")
    for feature_id, label, family, formula, basis, unit, min_n in [*_EXTENDED_FIXED, *COMMON_DEFINITIONS]:
        if family in disabled_families:
            continue
        validity = "finite eligible samples/support; sample-based unless formula explicitly integrates time"
        if family == "phasic_integral":
            validity = "at least two adjacent eligible phasic samples and positive clipped duration; gaps are not bridged"
        elif family in ("scr_distribution", "scr_rise", "scr_recovery", "scr_width", "scr_isi"):
            validity = "primary onset-eligible SCRs with onset in the half-open window; censored values retain catalogue status"
        elif family == "trial_series":
            validity = "unique valid unpolluted event trials; subject/session/condition/event_type series"
        elif family == "event_trial":
            validity = "one valid unique event trial; clean baseline and response on the same physical basis"
        elif family == "habituation":
            validity = "valid event trials in original ordinal order; descriptive trend only"
        elif family == "nonspecific":
            validity = "explicit matching complete annotation coverage and positive onset exposure"
        elif family == "group_temporal":
            validity = "fixed member/edge set, signed lag-zero group summary, aligned common eligible support"
        definitions[feature_id] = _definition(feature_id, label, family, formula, basis, unit, min_n,
                                               duration=1e-12 if family in ("phasic_integral", "nonspecific") else 0.0,
                                               parameters=_definition_parameters(feature_id, extended), validity=validity)
        if feature_id in COMMON_IDS:
            definitions[feature_id]["parameters"].update(common_parameters(
                feature_id, full["scr"]["recovery_fraction_returned"]))
    spectral = extended.get("spectral", {})
    for band in (spectral.get("bands", []) if spectral.get("enabled", True) is True else []):
        band_id = str(band["id"])
        for suffix, label, unit, formula in (
                ("uS2", f"clean频带功率：{band_id}", "uS2", "Welch PSD at exact band edges; linear interpolation then trapezoid"),
                ("normalized", f"clean频带归一化功率：{band_id}", "fraction", "band power / configured normalization-band power")):
            feature_id = f"clean_bandpower_{band_id}_{suffix}"
            min_segments_duration = float(spectral.get("segment_s", 256.0)) * (
                1.0 + max(0, int(spectral.get("min_segments", 2)) - 1) *
                (1.0 - float(spectral.get("overlap", 0.5))))
            lowest_required_hz = float(band["low_hz"])
            normalization_range = spectral.get("normalization_band_hz")
            if suffix == "normalized" and isinstance(normalization_range, (list, tuple)) and normalization_range:
                lowest_required_hz = min(lowest_required_hz, float(normalization_range[0]))
            spectral_min = max(min_segments_duration,
                               float(spectral.get("min_cycles", 2.0)) / lowest_required_hz)
            definitions[feature_id] = _definition(feature_id, label, "spectral", formula, "clean_uS", unit,
                                                   max(2, int(spectral.get("min_segments", 2))), duration=spectral_min,
                                                   parameters={"band": copy.deepcopy(band),
                                                               "segment_s": spectral.get("segment_s"),
                                                               "overlap": spectral.get("overlap"),
                                                               "window": spectral.get("window"),
                                                               "detrend": spectral.get("detrend"),
                                                               "min_segments": spectral.get("min_segments"),
                                                               "min_cycles": spectral.get("min_cycles"),
                                                               "normalization_band_hz": copy.deepcopy(spectral.get("normalization_band_hz"))},
                                                   validity="single contiguous eligible uniform clean window")
    autocorrelation = extended.get("autocorrelation", {})
    for lag in (autocorrelation.get("lags_s", []) if autocorrelation.get("enabled", True) is True else []):
        feature_id = autocorrelation_feature_id(lag)
        definitions[feature_id] = _definition(feature_id, f"clean自相关：{lag:g}s", "autocorrelation",
                                               "Pearson(clean[:-k],clean[k:])", "clean_uS", "correlation",
                                               int(autocorrelation.get("min_pairs", 20)),
                                               parameters={"lag_s": float(lag), "min_pairs": autocorrelation.get("min_pairs", 20),
                                                           "grid_tolerance_s": 1e-6},
                                               validity="single contiguous eligible uniform clean window; exact grid lag")
    complexity = extended.get("complexity", {})
    for feature_id, label, formula in ((
            ("clean_sample_entropy", "clean样本熵", "-ln(A/B), unordered non-self Chebyshev template matches, nats"),
            ("clean_permutation_entropy", "clean排列熵", "Shannon ordinal-pattern entropy, optionally /ln(m!)"),
            ("clean_dfa_alpha", "clean DFA指数", "OLS slope(log F(s) ~ log s), order-1 detrended profile"))
                          if complexity.get("enabled", True) is True else ()):
        normalized_permutation = complexity.get("permen", {}).get("normalize", True)
        unit = ("nats" if feature_id == "clean_sample_entropy" or
                feature_id == "clean_permutation_entropy" and not normalized_permutation else "dimensionless")
        complexity_min_n = (int(complexity.get("dfa", {}).get("min_samples", 256))
                            if feature_id == "clean_dfa_alpha" else int(complexity.get("min_samples", 200)))
        definitions[feature_id] = _definition(feature_id, label, "complexity", formula, "clean_uS", unit, complexity_min_n,
                                               parameters=copy.deepcopy(complexity),
                                               validity="single contiguous eligible uniform clean window")
    if full["decomposition"]["method"] == "bateman_sparse":
        from .driver_decompose import driver_settings
        for fid, label, unit, formula in DRIVER_SPECS:
            definitions[fid] = _definition(fid, label, "driver", formula, "driver_uS_s", unit, 1,
                duration=1e-12, parameters={**driver_settings(full), "method": "bateman_sparse",
                    "kernel_normalization": "continuous_peak_1", "driver_definition": "impulse_mass_uS/dt",
                    "integration": "adjacent_piecewise_linear_no_gap_no_extrapolation"},
                validity="converged driver segments; configured coverage; burst counts exclude boundary-censored episodes")
    return definitions


_REGISTRY_SUPERSET_CONFIG = {"synchrony": {"enabled": True}, "decomposition": {"method": "bateman_sparse"}}
SUPPORTED_FEATURE_IDS = tuple(list_feature_definitions(_REGISTRY_SUPERSET_CONFIG))


def _rows(value):
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, dict):
        return [dict(value)]
    try:
        return [dict(item) for item in value]
    except (TypeError, ValueError):
        fail("table_contract", "事件和条件必须是 DataFrame、字典列表或 None。")


def _recording_id(record):
    return record.get("recording_id") or record.get("metadata", {}).get("recording_id", "")


def _metadata_value(bundle, key):
    value = bundle.get(key)
    if value is not None:
        return value
    metadata = bundle.get("metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if value is not None:
            return value
    record = bundle.get("record")
    if isinstance(record, dict) and isinstance(record.get("metadata"), dict):
        return record["metadata"].get(key)
    return None


def _as_time(record):
    time = np.asarray(record.get("time_s", []), dtype=float)
    if time.ndim != 1 or not len(time) or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        fail("window_timebase", "窗口构建需要有限且严格递增的 time_s。", field="time_s")
    return time


def build_windows(record, events, conditions, config):
    """生成 whole/fixed/event/condition 四类窗口，所有边界均为半开 `[start,end)`。

    whole-record 使用首时间戳到末时间戳的 `nextafter` 右边界，只为将末点纳入
    样本选择，不创建额外时间暴露；固定窗从首时间锚定，length/step 单位秒，
    `tail_policy=drop` 丢弃不足整窗尾部，`include` 保留实际裁短尾窗。事件字段按
    planning/02 的 `event_id,onset_s,event_type,condition` 解释，baseline/response/
    peak/recovery 的配置区间均相对 onset 平移，不剪裁到记录边界；跨界和跨坏段
    由特征有效性标记。条件区间按 `recording_id,interval_id,start_s,end_s,condition`
    逐行保留，重叠不合并。返回窗口列包含固定 contracts 列及必要来源元数据。
    """
    full = _full_config(config)
    cfg = full["windows"]
    time = _as_time(record)
    rid = _recording_id(record)
    record_start = float(time[0])
    record_end = float(np.nextafter(time[-1], np.inf))
    windows = []
    channel = str(record.get("metadata", {}).get("channel", record.get("channel", "eda")))

    def add(window_id, kind, start, end, *, event_id=None, condition=None, event_onset=None,
            event_type=None, interval_id=None, source_start=None, source_end=None,
            event_ordinal=None, source_channel=None):
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            fail("window_bounds", f"窗口 {window_id} 必须为有限且 start_s < end_s。")
        windows.append({"window_id": str(window_id), "recording_id": rid, "kind": kind,
                        "start_s": float(start), "end_s": float(end), "event_id": event_id,
                        "condition": condition, "window_status": "planned", "window_reason": "",
                        "event_onset_s": event_onset, "event_type": event_type,
                        "interval_id": interval_id, "source_start_s": source_start,
                        "source_end_s": source_end, "condition_crossing": False,
                        "condition_labels": [], "event_ordinal": event_ordinal,
                        "channel": source_channel or channel})

    if cfg.get("whole_record", True):
        add(f"{rid}:whole", "whole_record", record_start, record_end)

    length, step = cfg.get("fixed_length_s"), cfg.get("fixed_step_s")
    if length is not None:
        length, step = float(length), float(step if step is not None else length)
        if length <= 0 or step <= 0:
            fail("fixed_window_config", "fixed_length_s 和 fixed_step_s 必须为正秒数。", field="windows")
        start = record_start
        ordinal = 0
        while start < record_end:
            nominal_end = start + length
            if nominal_end <= record_end + 1e-12:
                add(f"{rid}:fixed:{ordinal:05d}", "fixed", start, min(nominal_end, record_end))
            elif cfg.get("tail_policy") == "include" and start < record_end:
                add(f"{rid}:fixed:{ordinal:05d}", "fixed", start, record_end,
                    source_start=start, source_end=nominal_end)
            else:
                break
            ordinal += 1
            start = record_start + ordinal * step

    event_specs = (("event_baseline", "event_baseline_s"), ("event_response", "event_response_s"),
                   ("event_peak", "event_peak_s"), ("event_recovery", "event_recovery_s"))
    for i, event in enumerate(_rows(events)):
        event_rid = event.get("recording_id")
        if event_rid not in (None, "", rid):
            continue
        event_channel = event.get("channel")
        if event_channel not in (None, "", "*") and str(event_channel) != channel:
            continue
        onset = event.get("onset_s")
        if onset is None or not np.isfinite(float(onset)):
            fail("event_onset", f"events[{i}] 缺少有限 onset_s。", field="onset_s")
        onset = float(onset)
        event_id = event.get("event_id") or f"event-{i + 1:05d}"
        for kind, key in event_specs:
            relative_start, relative_end = map(float, cfg[key])
            add(f"{rid}:event:{event_id}:{kind}", kind, onset + relative_start, onset + relative_end,
                event_id=event_id, condition=event.get("condition"), event_onset=onset,
                event_type=event.get("event_type"), event_ordinal=i + 1,
                source_channel=event_channel)

    condition_rows = []
    for i, condition_row in enumerate(_rows(conditions)):
        condition_rid = condition_row.get("recording_id")
        if condition_rid not in (None, "", rid):
            continue
        condition_channel = condition_row.get("channel")
        if condition_channel not in (None, "", "*") and str(condition_channel) != channel:
            continue
        start, end = condition_row.get("start_s"), condition_row.get("end_s")
        if start is None or end is None or not np.isfinite(float(start)) or not np.isfinite(float(end)) or float(end) <= float(start):
            fail("condition_interval", f"conditions[{i}] 需要有限且 start_s < end_s。", field="conditions")
        interval_id = condition_row.get("interval_id") or f"condition-{i + 1:05d}"
        condition_rows.append({**condition_row, "interval_id": interval_id,
                               "start_s": float(start), "end_s": float(end)})
        add(f"{rid}:condition:{interval_id}", "condition", float(start), float(end),
            condition=condition_row.get("condition"), interval_id=interval_id,
            source_start=float(start), source_end=float(end), source_channel=condition_channel)

    # 固定/整体窗保留条件穿越信息，不将一个有交叠的窗静默归到单个 condition。
    for window in windows:
        if window["kind"] not in ("fixed", "whole_record"):
            continue
        overlapping = [item for item in condition_rows
                       if float(item["start_s"]) < window["end_s"] and float(item["end_s"]) > window["start_s"]]
        labels = list(dict.fromkeys(item.get("condition") for item in overlapping if item.get("condition") is not None))
        boundaries_inside = any(window["start_s"] < float(item["start_s"]) < window["end_s"] or
                                window["start_s"] < float(item["end_s"]) < window["end_s"]
                                for item in overlapping)
        window["condition_labels"] = labels
        window["condition_crossing"] = len(labels) > 1 or boundaries_inside
        fully_contained = [item for item in overlapping
                           if float(item["start_s"]) <= window["start_s"] and float(item["end_s"]) >= window["end_s"]]
        if len(fully_contained) == 1 and len(overlapping) == 1:
            window["condition"] = fully_contained[0].get("condition")

    if len(windows) > int(full["compute"]["max_windows"]):
        fail("window_limit", f"生成窗口数 {len(windows)} 超过 compute.max_windows。", category="resource")
    return pd.DataFrame(windows, columns=WINDOW_COLUMNS + WINDOW_EXTRA_COLUMNS)


def _candidate_records(bundle, cfg, gap_factor=1.5):
    time = np.asarray(bundle["time_s"], dtype=float)
    basis = cfg.get("signal_basis", "phasic")
    key = "phasic_uS" if basis == "phasic" else "clean_uS"
    if key not in bundle:
        fail("scr_basis", f"SCR basis {basis} 要求 bundle 包含 {key}。", stage="scr")
    y = np.asarray(bundle[key], dtype=float)
    if y.shape != time.shape:
        fail("scr_shape", "SCR 信号与时间网格必须等长。", stage="scr")
    quality = bundle.get("quality", {})
    mask = _bool_array(bundle.get("detector_eligible", quality.get("detector_eligible", quality.get("eligible"))),
                       len(time), "detector_eligible") & np.isfinite(y)
    max_gap = bundle.get("max_gap_s")
    if max_gap is None:
        dt = _nominal_dt(time, mask, gap_factor)
        max_gap = dt * gap_factor if dt else None
    segments = _segments(mask, time, max_gap)
    min_rise, max_rise = float(cfg.get("min_rise_s", 0.0)), float(cfg.get("max_rise_s", 10.0))
    result, rejected = [], {"no_preceding_trough": 0, "rise_out_of_range": 0, "nonpositive_amplitude": 0}
    for segment_id, (seg_start, seg_end) in enumerate(segments):
        if seg_end - seg_start < 3:
            continue
        local = y[seg_start:seg_end]
        peaks, peak_props = signal.find_peaks(local, plateau_size=1)
        if not len(peaks):
            continue
        # 平台峰用 SciPy 返回的左边缘，遵循 first maximum 规则，而非平台中点。
        peaks = peak_props["left_edges"]
        troughs, trough_props = signal.find_peaks(-local, plateau_size=1)
        troughs = trough_props["left_edges"]
        previous_peak = -1
        for local_peak in peaks:
            local_peak = int(local_peak)
            valleys = troughs[(troughs > previous_peak) & (troughs < local_peak)]
            if not len(valleys):
                rejected["no_preceding_trough"] += 1
                previous_peak = local_peak
                continue
            local_onset = int(valleys[-1])
            onset, peak = seg_start + local_onset, seg_start + local_peak
            amp = float(y[peak] - y[onset])
            rise = float(time[peak] - time[onset])
            previous_peak = local_peak
            if amp <= 0:
                rejected["nonpositive_amplitude"] += 1
                continue
            if rise < min_rise - 1e-12 or rise > max_rise + 1e-12:
                rejected["rise_out_of_range"] += 1
                continue
            result.append({"onset_i": onset, "peak_i": peak, "onset_s": float(time[onset]),
                           "peak_s": float(time[peak]), "onset_level": float(y[onset]),
                           "peak_level": float(y[peak]), "amplitude": amp, "rise": rise,
                           "segment_start": int(seg_start), "segment_end": int(seg_end),
                           "segment_id": int(segment_id),
                           "rise_slope": amp / rise,
                           "rise_max_slope": float(np.max(np.diff(y[onset:peak + 1]) /
                                                           np.diff(time[onset:peak + 1])))})
    return time, y, mask, segments, result, rejected, max_gap, basis


def scr_onset_eligible_mask(bundle, config):
    """计算 SCR primary onset 曝光样本掩码，供 pipeline 写出逐样本信号表。

    返回 ``(bool_mask, report)``；mask 与处理网格等长，不修改 bundle。report 同时
    给出 search/onset 可用秒数、连续段及解析后的真实秒 context。掩码只要求左右
    context 位于同一 detector 连续段内；不运行 peak detector，也不构成设备科学资格。
    """
    if not isinstance(bundle, dict) or "time_s" not in bundle:
        fail("signal_bundle", "onset eligibility 需要 SignalBundle 字典。", stage="scr")
    full = _full_config(config)
    cfg = full["scr"]
    time = np.asarray(bundle["time_s"], dtype=float)
    if time.ndim != 1 or not len(time) or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        fail("scr_timebase", "onset eligibility 需要有限且严格递增的时间轴。", stage="scr")
    basis = cfg.get("signal_basis", "phasic")
    key = "clean_uS" if basis == "clean" else "phasic_uS"
    values = np.asarray(bundle.get(key, []), dtype=float)
    if values.shape != time.shape:
        fail("scr_shape", f"onset eligibility 的 {key} 与 time_s 必须等长。", stage="scr")
    quality = bundle.get("quality", {})
    search = _bool_array(bundle.get("detector_eligible", quality.get("detector_eligible", quality.get("eligible"))),
                         len(time), "detector_eligible") & np.isfinite(values)
    max_gap = bundle.get("max_gap_s")
    if max_gap is None:
        dt = _nominal_dt(time, search, full["qc"].get("gap_factor", 1.5))
        max_gap = dt * float(full["qc"].get("gap_factor", 1.5)) if dt else None
    segments = _segments(search, time, max_gap)
    left = float(cfg.get("onset_left_context_s", 0.0))
    right = float(cfg.get("onset_right_context_s") if cfg.get("onset_right_context_s") is not None
                  else cfg.get("max_rise_s", 10.0))
    mask, intervals = _onset_support_intervals(time, segments, left, right)
    search_seconds = _eligible_duration(time, search, 0.0, float(time[-1]), max_gap)
    onset_seconds = _support_duration(intervals, float(time[0]), float(time[-1]))
    return mask, {"search_eligible_duration_s": search_seconds,
                  "onset_eligible_duration_s": onset_seconds,
                  "detector_eligible_duration_s": onset_seconds,
                  "search_eligible_sample_n": int(np.count_nonzero(search)),
                  "onset_eligible_sample_n": int(np.count_nonzero(mask)),
                  "onset_left_context_s": left, "onset_right_context_s": right,
                  "segments": [tuple(item) for item in segments],
                  "support_intervals_s": intervals,
                  "definition": "continuous_search_segment_duration_minus_left_and_right_context; no_gap_crossing"}


def _trapezoid_until(time, y, start, stop, baseline):
    if stop <= start:
        return 0.0
    t = time[start:stop + 1]
    values = y[start:stop + 1] - baseline
    if len(t) < 2:
        return 0.0
    return float(np.trapezoid(values, t))


def _recovery(candidate, next_candidate, time, y, segment_end, fraction, max_gap):
    onset_level, amplitude = candidate["onset_level"], candidate["amplitude"]
    threshold = onset_level + (1.0 - fraction) * amplitude
    peak = candidate["peak_i"]
    next_onset = next_candidate["onset_i"] if next_candidate is not None else segment_end
    stop = min(int(next_onset), int(segment_end - 1))
    for i in range(peak + 1, stop + 1):
        if y[i - 1] > threshold and y[i] <= threshold:
            dy = y[i] - y[i - 1]
            ratio = 0.0 if dy == 0 else (threshold - y[i - 1]) / dy
            crossing = float(time[i - 1] + ratio * (time[i] - time[i - 1]))
            return crossing, crossing - float(time[peak]), "", threshold
    if next_candidate is not None and next_candidate["onset_i"] < segment_end:
        reason = "next_response"
    elif segment_end >= len(time):
        reason = "recording_end"
    elif max_gap is not None and float(time[segment_end] - time[segment_end - 1]) > max_gap:
        reason = "data_gap"
    else:
        reason = "ineligible_segment_boundary"
    return math.nan, math.nan, reason, threshold


def _linear_crossing(time, y, left, right, threshold, *, direction):
    """在有限索引范围内寻找首次线性穿越；不外推相邻样本区间。"""
    for i in range(int(left) + 1, int(right) + 1):
        crossed = (y[i - 1] < threshold <= y[i]) if direction == "up" else (y[i - 1] > threshold >= y[i])
        if crossed:
            dy = float(y[i] - y[i - 1])
            if dy == 0:
                return float(time[i])
            fraction = (float(threshold) - float(y[i - 1])) / dy
            return float(time[i - 1] + fraction * (time[i] - time[i - 1]))
    return math.nan


def _extended_recovery(candidate, next_candidate, time, y, fraction):
    """逐响应计算返回比例 f 的首次下降穿越，保留右截尾状态。"""
    peak = int(candidate["peak_i"])
    end = int(candidate["segment_end"] - 1)
    if next_candidate is not None and int(next_candidate["segment_id"]) == int(candidate["segment_id"]):
        end = min(end, int(next_candidate["onset_i"]))
    threshold = float(candidate["onset_level"] + (1.0 - fraction) * candidate["amplitude"])
    crossing = _linear_crossing(time, y, peak, end, threshold, direction="down")
    if np.isfinite(crossing):
        return crossing, crossing - float(time[peak]), "valid", "", threshold
    if end < int(candidate["segment_end"] - 1):
        reason = "next_response"
    else:
        reason = "right_censored_segment_end"
    return math.nan, math.nan, "censored", reason, threshold


def _extended_half_width(candidate, next_candidate, time, y):
    """计算首次上/下穿越半高阈值的宽度，右侧不可见时保留 censored。"""
    onset, peak = int(candidate["onset_i"]), int(candidate["peak_i"])
    threshold = float(candidate["onset_level"] + 0.5 * candidate["amplitude"])
    rising = _linear_crossing(time, y, onset, peak, threshold, direction="up")
    if not np.isfinite(rising):
        return math.nan, "unavailable", "half_height_rise_crossing_not_observed"
    end = int(candidate["segment_end"] - 1)
    if next_candidate is not None and int(next_candidate["segment_id"]) == int(candidate["segment_id"]):
        end = min(end, int(next_candidate["onset_i"]))
    falling = _linear_crossing(time, y, peak, end, threshold, direction="down")
    if np.isfinite(falling):
        return float(falling - rising), "valid", ""
    return math.nan, "censored", "half_height_fall_not_observed_before_support_end"


def _onset_support_intervals(time, segments, left_context_s, right_context_s):
    """按合格连续段的实际时间端点构造 onset 可判定区间与样本资格掩码。

    每个检测段至少需要三个样本以支持局部 trough/peak 关系。支持区间为
    ``[t_first+L, t_last-R]``；区间外不补点、不跨 gap。返回区间秒数和每个
    真实样本点的资格 bool，供 catalogue 判定及窗口暴露分别复用。
    """
    sample_eligible = np.zeros(len(time), dtype=bool)
    intervals = []
    left, right = float(left_context_s), float(right_context_s)
    for start, stop in segments:
        if stop - start < 3:
            continue
        lo = float(time[start]) + left
        hi = float(time[stop - 1]) - right
        if hi <= lo + 1e-12:
            continue
        intervals.append((lo, hi))
        sample_eligible[start:stop] = (time[start:stop] >= lo - 1e-12) & (time[start:stop] <= hi + 1e-12)
    return sample_eligible, intervals


def _support_duration(intervals, start, end):
    """求多个 onset 支持区间与查询半开窗的交叠秒数，不外推或跨缺口。"""
    if end <= start:
        return 0.0
    return float(sum(max(0.0, min(float(end), hi) - max(float(start), lo)) for lo, hi in intervals))


def _support_covers(intervals, start, end):
    """判断连续 onset 支持是否覆盖完整半开窗；仅 exposure 部分覆盖不作零响应结论。"""
    start, end = float(start), float(end)
    if end <= start:
        return False
    cursor = start
    for lo, hi in sorted(intervals):
        if hi <= cursor + 1e-12:
            continue
        if lo > cursor + 1e-12:
            return False
        cursor = max(cursor, hi)
        if cursor >= end - 1e-12:
            return True
    return cursor >= end - 1e-12


def detect_scr(bundle, config):
    """在整段 detector-eligible 数据上先收集 trough-to-peak 候选，再解阈值与间隔冲突。

    `phasic` 使用分解后物理 μS phasic，`clean` 使用清理物理 μS；每个有效连续段
    内以局部最大值作 peak、以此前上一峰之后最近局部谷作 onset，平台峰取最左
    最大点。rise=`t_peak-t_onset` 秒。absolute 阈值为 μS，relative 阈值为
    `relative_fraction × 全 recording onset-eligible rise 候选的最大幅值` μS，阈值只在整段
    候选收集后计算，不随分析窗改变；缺少完整 onset context 的 review 候选不会抬高主
    分析阈值。`min_interval_s` 按 onset 间隔；冲突时留幅值
    较大者，等幅留较早峰。恢复首次下降穿越 `L+(1-f)A`（f 是 returned fraction）
    并按相邻点线性插值；下一个响应、缺口/坏段或记录末端前未穿越则 recovery NA
    且标 censor reason。`area_uS_s` 为从 trough onset 到 peak、以 onset level 为零
    基线的带符号梯形面积，不延伸到未观察到的恢复段。搜索掩码保留整段支持以定位
    trough、peak 与 recovery；primary onset eligibility 要求候选时刻前 L 秒、后 R
    秒均在同一合格连续段内，R 默认 max_rise_s 且不能更短。边缘候选仍写入 catalogue，
    以 onset_eligible=False / qualification_reason 标记，不参与主 count/amplitude/rate，
    也不允许边缘较大候选压掉已合格候选。报告分开写 search 与 onset exposure；兼容字段
    detector_eligible_duration_s 代表 onset exposure。无完整 onset exposure 报 unavailable，
    不能把“不可检测”写成 no_responses。候选、阈值和拒绝数写入报告。
    """
    if not isinstance(bundle, dict) or "time_s" not in bundle:
        fail("signal_bundle", "SCR 检测需要 SignalBundle 字典。", stage="scr")
    full = _full_config(config)
    cfg = full["scr"]
    if cfg.get("method") != "trough_to_peak":
        fail("scr_method", "当前只实现已命名的 trough_to_peak，不做隐式替换。", stage="scr")
    time, y, detector_mask, segments, candidates, rejected, max_gap, basis = _candidate_records(
        bundle, cfg, full["qc"].get("gap_factor", 1.5))
    recording_id = _metadata_value(bundle, "recording_id") or ""
    left_context = float(cfg.get("onset_left_context_s", 0.0))
    right_context = float(cfg.get("onset_right_context_s") if cfg.get("onset_right_context_s") is not None
                          else cfg.get("max_rise_s", 10.0))
    onset_mask, onset_intervals = _onset_support_intervals(time, segments, left_context, right_context)
    onset_seconds = _support_duration(onset_intervals, float(time[0]), float(time[-1]))
    for candidate in candidates:
        candidate["onset_eligible"] = bool(onset_mask[candidate["onset_i"]])
        candidate["qualification_reason"] = "" if candidate["onset_eligible"] else "unavailable_onset_context"
    primary_reference_candidates = [item for item in candidates if item["onset_eligible"]]
    eligible_max = max((item["amplitude"] for item in primary_reference_candidates), default=0.0)
    if cfg.get("threshold_type") == "absolute":
        threshold = float(cfg.get("amplitude_min_uS", 0.05))
    elif cfg.get("threshold_type") == "relative":
        threshold = float(cfg.get("relative_fraction", 0.1)) * eligible_max
    else:
        fail("scr_threshold_type", "threshold_type 仅支持 absolute/relative。", stage="scr")
    thresholded = [item for item in candidates if item["amplitude"] >= threshold]
    primary_thresholded = [item for item in thresholded if item["onset_eligible"]]
    review_thresholded = [item for item in thresholded if not item["onset_eligible"]]

    # 只有 onset context 足够的候选参与主分析的最小间隔冲突；右边界候选保留
    # 在 review catalogue，不能反过来压掉已经具备完整上下文的较早主分析响应。
    interval = float(cfg.get("min_interval_s", 1.0))
    ranked = sorted(primary_thresholded, key=lambda item: (-item["amplitude"], item["peak_s"]))
    kept = []
    conflict_count = 0
    for candidate in ranked:
        if any(int(candidate["segment_id"]) == int(existing["segment_id"]) and
               abs(candidate["onset_s"] - existing["onset_s"]) < interval - 1e-12 for existing in kept):
            conflict_count += 1
        else:
            kept.append(candidate)
    kept.extend(review_thresholded)
    kept.sort(key=lambda item: item["onset_s"])
    all_sorted = sorted(candidates, key=lambda item: item["onset_s"])
    # recovery 的下一个响应采用已接受 catalogue 的 onset，不因窗口裁剪而重算。
    rows = []
    recovery_censored = 0
    for ordinal, candidate in enumerate(kept, 1):
        next_candidate = next((item for item in kept[ordinal:]
                               if int(item["segment_id"]) == int(candidate["segment_id"])), None)
        fraction = float(cfg.get("recovery_fraction_returned", 0.5))
        recovery_s, recovery_duration, censor_reason, recovery_threshold = _recovery(
            candidate, next_candidate, time, y, candidate["segment_end"], fraction, max_gap)
        if censor_reason:
            recovery_censored += 1
        area = _trapezoid_until(time, y, candidate["onset_i"], candidate["peak_i"], candidate["onset_level"])
        crossing_50, duration_50, recovery_50_status, recovery_50_reason, _ = _extended_recovery(
            candidate, next_candidate, time, y, 0.50)
        crossing_63, duration_63, recovery_63_status, recovery_63_reason, _ = _extended_recovery(
            candidate, next_candidate, time, y, 0.63)
        half_width, half_width_status, half_width_reason = _extended_half_width(
            candidate, next_candidate, time, y)
        rows.append({"scr_id": f"{recording_id}:SCR:{ordinal:06d}", "recording_id": recording_id,
                     "onset_s": candidate["onset_s"], "peak_s": candidate["peak_s"],
                     "amplitude_uS": candidate["amplitude"], "rise_time_s": candidate["rise"],
                     "recovery_s": recovery_s, "recovery_time_s": recovery_duration,
                     "area_uS_s": area, "status": "censored" if censor_reason else "valid",
                     "reason": f"recovery_censored:{censor_reason}" if censor_reason else "",
                     "onset_level_uS": candidate["onset_level"], "peak_level_uS": candidate["peak_level"],
                     "recovery_fraction_returned": fraction, "threshold_type": cfg["threshold_type"],
                     "threshold_value_uS": threshold, "signal_basis": basis, "unit": "uS",
                     "method": "trough_to_peak", "onset_sample_index": int(candidate["onset_i"]),
                     "peak_sample_index": int(candidate["peak_i"]), "recovery_censor_reason": censor_reason,
                     "area_baseline": "onset_level_uS", "area_bounds": "[onset_s,peak_s]",
                     "onset_eligible": bool(candidate["onset_eligible"]),
                     "qualification_reason": candidate["qualification_reason"],
                     "segment_id": int(candidate["segment_id"]),
                     "rise_slope_uS_s": float(candidate["rise_slope"]),
                     "rise_max_slope_uS_s": float(candidate["rise_max_slope"]),
                     "recovery_50_s": crossing_50, "recovery_50_time_s": duration_50,
                     "recovery_50_status": recovery_50_status, "recovery_50_reason": recovery_50_reason,
                     "recovery_63_s": crossing_63, "recovery_63_time_s": duration_63,
                     "recovery_63_status": recovery_63_status, "recovery_63_reason": recovery_63_reason,
                     "half_width_50_s": half_width, "half_width_50_status": half_width_status,
                     "half_width_50_reason": half_width_reason})
    columns = SCR_COLUMNS + SCR_EXTRA_COLUMNS
    table = pd.DataFrame(rows, columns=columns)
    search_seconds = _eligible_duration(time, detector_mask, 0.0, float(time[-1]), max_gap)
    detector_supported = onset_seconds > 0
    if rows:
        detector_status, detector_reason = ("succeeded", "") if detector_supported else (
            "unavailable", "no_detector_eligible_exposure")
    elif not detector_supported:
        # “没有观察到响应”只适用于本来具备可检测时长与局部峰支持的记录。
        detector_status, detector_reason = "unavailable", "no_detector_eligible_exposure"
    else:
        detector_status, detector_reason = "no_responses", ""
    report = {"method": "trough_to_peak", "signal_basis": basis, "unit": "uS",
              "threshold_type": cfg["threshold_type"], "threshold_value_uS": threshold,
              "relative_reference_scope": cfg.get("reference_scope"), "relative_reference_basis": "max_amplitude_of_onset_eligible_rise_candidates",
              "relative_reference_max_uS": eligible_max, "candidate_count_before_threshold": len(candidates),
              "primary_relative_reference_candidate_count": len(primary_reference_candidates),
              "candidate_count_after_threshold": len(thresholded), "detected_count": len(rows),
              "primary_eligible_count": int(sum(bool(row["onset_eligible"]) for row in rows)),
              "unavailable_onset_context_count": int(sum(not bool(row["onset_eligible"]) for row in rows)),
              "recovery_censored_count": recovery_censored, "recovery_fraction_definition": "returned fraction f; threshold=L+(1-f)A",
              "area_definition": "signed trapezoid above onset level on [onset_s,peak_s]",
              "min_interval_policy": "greedy_keep_larger_amplitude_then_earlier_peak; conflict uses onset separation",
              "min_interval_conflicts": conflict_count, "rejected": rejected,
              "search_eligible_duration_s": search_seconds,
              "search_eligible_sample_n": int(np.count_nonzero(detector_mask)),
              # 兼容既有rate/report消费者：detector exposure现按primary onset支持定义。
              "detector_eligible_duration_s": onset_seconds,
              "detector_eligible_sample_n": int(np.count_nonzero(onset_mask)),
              "onset_eligible_duration_s": onset_seconds,
              "onset_eligible_sample_n": int(np.count_nonzero(onset_mask)),
              "onset_left_context_s": left_context, "onset_right_context_s": right_context,
              "onset_exposure_definition": "continuous_search_segment_duration_minus_left_and_right_context; clipped_window_intersection; no_gap_crossing",
              "segment_count": len(segments), "status": detector_status, "reason": detector_reason,
              "candidate_ordering_note": "all eligible segments collected before recording-level threshold; no bridging"}
    return table, report


def _eligible_duration(time, mask, start, end, max_gap):
    """点采样 exposure：裁剪相邻合格区间，不跨 gap、不外推尾点。"""
    total = 0.0
    for i in range(max(0, len(time) - 1)):
        dt = float(time[i + 1] - time[i])
        if not (mask[i] and mask[i + 1] and dt > 0):
            continue
        if max_gap is not None and dt > max_gap:
            continue
        total += max(0.0, min(float(end), float(time[i + 1])) - max(float(start), float(time[i])))
    return float(total)


def _clip_integral(time, values, mask, start, end, max_gap):
    """对全部支撑合格的相邻区间作裁剪梯形积分，不连接缺口。"""
    area = 0.0
    n_intervals = 0
    for i in range(max(0, len(time) - 1)):
        t0, t1 = float(time[i]), float(time[i + 1])
        dt = t1 - t0
        if not (mask[i] and mask[i + 1] and dt > 0) or (max_gap is not None and dt > max_gap):
            continue
        a, b = max(float(start), t0), min(float(end), t1)
        if b <= a:
            continue
        y0, y1 = float(values[i]), float(values[i + 1])
        ya = y0 + (y1 - y0) * ((a - t0) / dt)
        yb = y0 + (y1 - y0) * ((b - t0) / dt)
        area += (ya + yb) * 0.5 * (b - a)
        n_intervals += 1
    return float(area), n_intervals


def _clip_signed_positive_absolute(time, values, mask, start, end, max_gap):
    """精确积分分段线性phasic曲线的带符号、正值和绝对面积。"""
    signed = positive = absolute = duration = 0.0
    interval_count = 0
    for i in range(max(0, len(time) - 1)):
        t0, t1 = float(time[i]), float(time[i + 1])
        dt = t1 - t0
        if not (mask[i] and mask[i + 1] and dt > 0) or (max_gap is not None and dt > max_gap):
            continue
        a, b = max(float(start), t0), min(float(end), t1)
        if b <= a:
            continue
        y0, y1 = float(values[i]), float(values[i + 1])
        ya = y0 + (y1 - y0) * ((a - t0) / dt)
        yb = y0 + (y1 - y0) * ((b - t0) / dt)
        span = b - a
        trapezoid = 0.5 * (ya + yb) * span
        signed += trapezoid
        duration += span
        interval_count += 1
        if ya >= 0 and yb >= 0:
            positive += trapezoid
            absolute += trapezoid
        elif ya <= 0 and yb <= 0:
            absolute -= trapezoid
        else:
            crossing_fraction = -ya / (yb - ya)
            crossing_fraction = float(np.clip(crossing_fraction, 0.0, 1.0))
            left_duration = span * crossing_fraction
            right_duration = span - left_duration
            left_triangle = 0.5 * abs(ya) * left_duration
            right_triangle = 0.5 * abs(yb) * right_duration
            positive += right_triangle if ya < 0 < yb else left_triangle
            absolute += left_triangle + right_triangle
    return {"signed": float(signed), "positive": float(positive), "absolute": float(absolute),
            "duration": float(duration), "interval_n": interval_count}


def _window_context(bundle, window, config):
    time = np.asarray(bundle["time_s"], dtype=float)
    n = len(time)
    start, end = float(window["start_s"]), float(window["end_s"])
    quality = bundle.get("quality", {})
    cfg = _full_config(config)
    mask = _bool_array(quality.get("analysis_eligible", quality.get("eligible")), n, "analysis_eligible")
    clean = np.asarray(bundle.get("clean_uS", np.full(n, np.nan)), dtype=float)
    mask &= np.isfinite(clean)
    interpolated = _bool_array(quality.get("interpolated"), n, "interpolated")
    artifact = _bool_array(quality.get("artifact"), n, "artifact")
    if cfg["features"].get("exclude_interpolated", True):
        mask &= ~interpolated
    max_gap = bundle.get("max_gap_s")
    if max_gap is None:
        dt = _nominal_dt(time, None, cfg["qc"].get("gap_factor", 1.5))
        max_gap = dt * float(cfg["qc"].get("gap_factor", 1.5)) if dt else None
    in_samples = (time >= start) & (time < end)
    sample_mask = in_samples & mask
    expected = max(0.0, end - start)
    valid_seconds = _eligible_duration(time, mask, start, end, max_gap)
    scr_basis = cfg["scr"].get("signal_basis", "phasic")
    detector_signal = clean if scr_basis == "clean" else np.asarray(bundle.get("phasic_uS", clean), dtype=float)
    detector = _bool_array(bundle.get("detector_eligible", quality.get("detector_eligible", quality.get("eligible"))),
                           n, "detector_eligible") & np.isfinite(detector_signal)
    search_seconds = _eligible_duration(time, detector, start, end, max_gap)
    detector_segments = _segments(detector, time, max_gap)
    left_context = float(cfg["scr"].get("onset_left_context_s", 0.0))
    right_context = float(cfg["scr"].get("onset_right_context_s") if cfg["scr"].get("onset_right_context_s") is not None
                          else cfg["scr"].get("max_rise_s", 10.0))
    _onset_mask, onset_intervals = _onset_support_intervals(time, detector_segments, left_context, right_context)
    detector_seconds = _support_duration(onset_intervals, start, end)
    onset_window_supported = _support_covers(onset_intervals, start, end)
    coverage = valid_seconds / expected if expected > 0 else 0.0
    gap_crossing = False
    if len(time) > 1 and max_gap is not None:
        dt = np.diff(time)
        gap_crossing = bool(np.any((dt > max_gap) & (time[:-1] < end) & (time[1:] > start)))
    clipped_start, clipped_end = max(start, float(time[0])), min(end, float(time[-1]))
    extends_record = start < time[0] or end > np.nextafter(time[-1], np.inf)
    all_n = int(np.count_nonzero(in_samples))
    valid_n = int(np.count_nonzero(sample_mask))
    interpolated_n = int(np.count_nonzero(in_samples & interpolated))
    artifact_n = int(np.count_nonzero(in_samples & artifact))
    excluded_n = all_n - valid_n
    min_coverage = float(cfg["windows"].get("min_coverage", 0.8))
    if gap_crossing:
        status, reason = "invalid", "window_crosses_gap"
    elif coverage + 1e-12 < min_coverage or extends_record:
        status, reason = "invalid", "insufficient_coverage_or_outside_recording"
    elif excluded_n:
        status, reason = "partial", "excluded_edge_or_interpolated_samples"
    else:
        status, reason = "succeeded", ""
    return {"time": time, "start": start, "end": end, "max_gap": max_gap,
            "sample_mask": sample_mask, "in_samples": in_samples, "interpolated": interpolated,
            "artifact": artifact, "detector": detector, "valid_duration": valid_seconds,
            "detector_duration": detector_seconds, "search_duration": search_seconds,
            "onset_duration": detector_seconds, "onset_left_context_s": left_context,
            "onset_right_context_s": right_context, "onset_intervals": onset_intervals,
            "onset_window_supported": onset_window_supported, "expected_duration": expected,
            "coverage": coverage, "gap_crossing": gap_crossing, "extends_record": extends_record,
            "valid_n": valid_n, "interpolated_n": interpolated_n, "artifact_n": artifact_n,
            "all_n": all_n, "excluded_n": excluded_n, "status": status, "reason": reason,
            "primary_valid": status == "succeeded", "strict_valid": status != "invalid"}


def _stat_status(ctx, values, *, requires_duration=False, scr_metric=False):
    if ctx["status"] == "invalid" or (scr_metric and ctx["gap_crossing"]):
        return "invalid", ctx["reason"] or "invalid_window"
    if not len(values):
        return "unavailable", "no_eligible_samples"
    if requires_duration and ctx["valid_duration"] <= 0:
        return "unavailable", "zero_valid_exposure"
    return ctx["status"], ctx["reason"]


def _append_feature(rows, metadata, window, feature_id, value, unit, basis, method, status, reason,
                    valid_n, duration):
    # FeatureTable 的 measurement status 与阶段执行状态分开；未估计/歧义不携带伪数值。
    status_map = {"succeeded": "valid", "partial": "valid", "invalid": "unavailable"}
    status = status_map.get(status, status)
    if status in ("unavailable", "ambiguous", "censored"):
        value = None
    rows.append({**metadata, "window_id": window["window_id"], "feature_id": feature_id,
                 "value": value, "unit": unit, "basis": basis, "method": method,
                 "status": status, "reason": reason or "", "valid_n": int(valid_n),
                 "valid_duration_s": float(duration)})


def _accepted_scrs(scr, *, include_unqualified=False):
    if scr is None or not isinstance(scr, pd.DataFrame) or scr.empty:
        return []
    required = {"onset_s", "amplitude_uS"}
    if not required <= set(scr.columns):
        fail("scr_table", f"SCR catalogue 缺字段 {sorted(required - set(scr.columns))}。")
    rows = []
    for item in scr.to_dict(orient="records"):
        status = item.get("status", "detected")
        if status not in ("valid", "detected", "accepted", "censored") or not np.isfinite(item.get("onset_s", np.nan)):
            continue
        if not np.isfinite(item.get("amplitude_uS", np.nan)):
            continue
        if not include_unqualified and not bool(item.get("onset_eligible", True)):
            continue
        rows.append(item)
    return rows


def _cancel_requested(cancel_callback):
    if cancel_callback is not None and cancel_callback():
        raise InterruptedError("用户请求取消特征计算。")


def _feature_progress(progress_callback, *, family, completed, total, window_id=None):
    if progress_callback is not None:
        progress_callback({"stage": "features", "family": family, "completed": int(completed),
                           "total": int(total), "window_id": window_id})


def _event_scr_links(recording_id, response_specs, scr_rows, contexts):
    candidates_by_scr = {}
    rows = []
    event_candidate_ids: set[str] = set()
    for event_id, window in response_specs.items():
        ctx = contexts[window["window_id"]]
        for item in scr_rows:
            onset = float(item["onset_s"])
            if not (float(window["start_s"]) <= onset < float(window["end_s"])):
                continue
            key = str(item.get("scr_id", ""))
            event_candidate_ids.add(str(event_id))
            if not bool(item.get("onset_eligible", True)):
                rows.append({"recording_id": recording_id, "event_id": event_id, "scr_id": key,
                             "link_status": "unavailable_onset_context", "link_basis": "scr_onset_in_event_response_window",
                             "reason": item.get("qualification_reason") or "unavailable_onset_context"})
                continue
            candidates_by_scr.setdefault(key, []).append((event_id, item, ctx))
    for scr_id, candidates in candidates_by_scr.items():
        if len(candidates) > 1:
            for event_id, item, _ in candidates:
                rows.append({"recording_id": recording_id, "event_id": event_id, "scr_id": scr_id,
                             "link_status": "ambiguous", "link_basis": "scr_onset_in_event_response_window",
                             "reason": "onset_in_multiple_event_response_windows"})
        else:
            event_id, item, ctx = candidates[0]
            status = ("invalid_window" if ctx["status"] == "invalid" else
                      "unavailable_onset_context" if not ctx["onset_window_supported"] else "unique")
            rows.append({"recording_id": recording_id, "event_id": event_id, "scr_id": scr_id,
                         "link_status": status, "link_basis": "scr_onset_in_event_response_window",
                         "reason": (ctx["reason"] if status == "invalid_window" else
                                    "insufficient_onset_context" if status == "unavailable_onset_context" else "")})
    linked_ids = set(candidates_by_scr)
    for item in scr_rows:
        scr_id = str(item.get("scr_id", ""))
        if scr_id not in linked_ids:
            rows.append({"recording_id": recording_id, "event_id": None, "scr_id": scr_id,
                         "link_status": "unlinked", "link_basis": "scr_onset_in_event_response_window",
                     "reason": item.get("qualification_reason") or "no_event_response_membership"})
    events_with_scr = {event_id for values in candidates_by_scr.values() for event_id, _, _ in values}
    for event_id in response_specs:
        if event_id not in events_with_scr and event_id not in event_candidate_ids:
            ctx = contexts[response_specs[event_id]["window_id"]]
            if ctx["status"] == "invalid":
                link_status, reason = "invalid_window", ctx["reason"]
            elif not ctx["onset_window_supported"]:
                link_status, reason = "unavailable_onset_context", "insufficient_onset_context"
            else:
                link_status, reason = "no_response", "no_scr_onset_in_response_window"
            rows.append({"recording_id": recording_id, "event_id": event_id, "scr_id": None,
                         "link_status": link_status, "link_basis": "scr_onset_in_event_response_window",
                         "reason": reason})
    return pd.DataFrame(rows, columns=EVENT_LINK_COLUMNS)


def extract_window_features(bundle, scr, windows, config, *, cancel_callback=None, progress_callback=None):
    """按样本/时间支撑分别计算单记录特征与逐窗有效性，不把样本均值称作时间加权。

    mean/median/SD(ddof=1)/IQR/range/slope 明确按窗口内有效采样点计算；slope 为
    OLS μS/s。tonic mean/change 使用 tonic 样本均值及相对窗内首个有效值；phasic
    AUC 为物理 μS 的带符号梯形积分。AUC 和 valid_duration 只累加相邻有效点区间，
    与 `[start,end)` 求交，不跨 gap、不外推尾点。SCR count 按 onset 落入半开窗，
    rate=`60 × qualified_onset_count / onset_eligible_duration_s`；零 exposure 时 NA，
    有效无响应时 count=0 且 amplitude mean/median NA。边缘 SCR 保留在 catalogue 但不
    参加主计数。事件 response 的无响应 count=0 仅在 onset 支持覆盖完整半开窗时给出，
    否则 status=unavailable。振幅 sum 是零包含特征。
    SCR area sum/mean 复用 catalogue 的 `area_uS_s`：以 onset level 为基线的 signed
    trapezoid，范围 `[onset_s,peak_s]`，只表示上升段基线差面积，不是完整恢复面积；
    recovery 被截尾不影响已观测的 upstroke area。event SCR area sum 只汇总唯一 event link。
    event windows 含 event baseline/response/peak/recovery 四种；baseline 有 SCR 污染、
    跨 gap/边界或 coverage 不足時，baseline-relative event features invalid。事件
    Measurement status 使用 valid/unavailable/censored/ambiguous；阶段执行 status
    单独报告。SCR 按 onset 落入 response window 唯一归属；多个事件竞争时 link 标
    ambiguous，不用于唯一事件特征。返回 validity.attrs['event_scr_links'] 独立链接表。每个 feature
    行携带单位、signal basis、算法、status/reason、有效样本数和有效暴露秒数。
    `features.ids=[]` 返回全部 registry 特征；非空列表只输出指定 ID，未知 ID 是配置错误。
    """
    if not isinstance(bundle, dict) or "time_s" not in bundle:
        fail("signal_bundle", "特征提取需要 SignalBundle 字典。")
    full = _full_config(config)
    requested_ids = full["features"].get("ids", [])
    if not isinstance(requested_ids, list) or any(not isinstance(item, str) for item in requested_ids):
        fail("feature_ids", "features.ids 必须是由 feature ID 字符串组成的列表。",
             category="configuration", stage="features", field="features.ids")
    registry = list_feature_definitions(full)
    supported_ids = set(registry)
    unknown_ids = sorted(set(requested_ids) - supported_ids)
    if unknown_ids:
        fail("unknown_feature_id", f"未知 features.ids：{unknown_ids}。",
             category="configuration", stage="features", field="features.ids",
             remediation=f"使用已支持 ID：{', '.join(sorted(supported_ids))}。")
    requested_set = set(requested_ids) if requested_ids else set(supported_ids)
    selected_ids = requested_set
    if not isinstance(windows, pd.DataFrame):
        windows = pd.DataFrame(windows, columns=WINDOW_COLUMNS + WINDOW_EXTRA_COLUMNS)
    missing = set(WINDOW_COLUMNS) - set(windows.columns)
    if missing:
        fail("window_table", f"窗口表缺少固定契约列 {sorted(missing)}。")
    time = np.asarray(bundle["time_s"], dtype=float)
    clean = np.asarray(bundle.get("clean_uS", np.full(len(time), np.nan)), dtype=float)
    tonic = np.asarray(bundle.get("tonic_uS", np.full(len(time), np.nan)), dtype=float)
    phasic = np.asarray(bundle.get("phasic_uS", np.full(len(time), np.nan)), dtype=float)
    if any(array.shape != time.shape for array in (clean, tonic, phasic)):
        fail("feature_shape", "bundle 的 time/clean/tonic/phasic 数组必须等长。")
    rid = _metadata_value(bundle, "recording_id") or (str(windows.iloc[0]["recording_id"]) if len(windows) else "")
    metadata = {"recording_id": rid, "subject_id": _metadata_value(bundle, "subject_id"),
                "session_id": _metadata_value(bundle, "session_id")}
    catalogue_rows = _accepted_scrs(scr, include_unqualified=True)
    scr_rows = [item for item in catalogue_rows if bool(item.get("onset_eligible", True))]
    recording_amplitude_reference = amplitude_reference([float(item["amplitude_uS"]) for item in scr_rows
                                                         if float(item["amplitude_uS"]) >= 0])
    extended_scr = scr.copy(deep=True) if isinstance(scr, pd.DataFrame) else pd.DataFrame(columns=SCR_COLUMNS + SCR_EXTRA_COLUMNS)
    extended_scr = annotate_amplitudes(extended_scr, "amplitude_uS", [
        item.get("status", "detected") in ("valid", "detected", "accepted", "censored") and
        np.isfinite(item.get("onset_s", np.nan)) and bool(item.get("onset_eligible", True))
        for item in extended_scr.to_dict("records")])
    contexts = {str(row["window_id"]): _window_context(bundle, row, config)
                for row in windows.to_dict(orient="records")}
    response_specs = {str(row["event_id"]): row for row in windows.to_dict(orient="records")
                      if row["kind"] == "event_response" and row.get("event_id") is not None}
    links = _event_scr_links(rid, response_specs, catalogue_rows, contexts)
    trial_table = build_trial_table(bundle, scr, windows, links, full)
    trial_features, trial_summaries, trial_fit_diagnostics = summarize_trial_series(trial_table, full)
    trial_by_event = ({str(row["event_id"]): row for row in trial_table.to_dict("records")}
                      if not trial_table.empty else {})
    extended_tables = {"trial_table": trial_table,
                       "trial_summaries": trial_summaries,
                       "trial_fit_diagnostics": trial_fit_diagnostics,
                       "spectra": pd.DataFrame(),
                       "complexity_diagnostics": pd.DataFrame(),
                       "ns_classification": pd.DataFrame(),
                       "scr_events_extended": (scr.copy(deep=True) if isinstance(scr, pd.DataFrame)
                                               else pd.DataFrame(columns=SCR_COLUMNS + SCR_EXTRA_COLUMNS))}
    unique_by_event = {}
    ambiguous_by_event = set()
    unavailable_by_event = set()
    invalid_by_event = set()
    if not links.empty:
        for row in links.to_dict(orient="records"):
            if row["link_status"] == "unique":
                unique_by_event.setdefault(str(row["event_id"]), set()).add(str(row["scr_id"]))
            elif row["link_status"] == "ambiguous":
                ambiguous_by_event.add(str(row["event_id"]))
            elif row["link_status"] == "unavailable_onset_context":
                unavailable_by_event.add(str(row["event_id"]))
            elif row["link_status"] == "invalid_window":
                invalid_by_event.add(str(row["event_id"]))
    scr_by_id = {str(item.get("scr_id", "")): item for item in scr_rows}
    detector_signal = clean if full["scr"].get("signal_basis") == "clean" else phasic
    detector_mask = _bool_array(bundle.get("detector_eligible", bundle.get("quality", {}).get("detector_eligible",
                                                                                           bundle.get("quality", {}).get("eligible"))),
                                len(time), "detector_eligible") & np.isfinite(detector_signal)
    detector_segments = _segments(detector_mask, time, contexts[next(iter(contexts))]["max_gap"] if contexts else bundle.get("max_gap_s"))
    for item in catalogue_rows:
        sid = item.get("segment_id")
        if sid is None or pd.isna(sid):
            onset = float(item["onset_s"])
            sample_i = int(np.argmin(np.abs(time - onset))) if len(time) else -1
            sid = next((index for index, (start_i, stop_i) in enumerate(detector_segments)
                        if start_i <= sample_i < stop_i), None)
        item["analysis_segment_id"] = sid
    onset_intervals = next(iter(contexts.values()))["onset_intervals"] if contexts else []
    ns_table, ns_intervals, ns_policy_reason = classify_nonspecific_scr(
        bundle, scr if isinstance(scr, pd.DataFrame) else pd.DataFrame(), windows,
        onset_intervals, links, full)
    extended_tables["ns_classification"] = ns_table
    feature_rows = []
    validity_rows = []
    advanced_chunks = {"spectra": [], "complexity_diagnostics": []}
    extended_ids = {feature_id for feature_id, definition in registry.items()
                    if definition["family"] in ("spectral", "autocorrelation", "complexity")}
    advanced_selected = selected_ids.intersection(extended_ids)
    driver_catalogue = driver_bursts(bundle, full)
    for window_index, window in enumerate(windows.to_dict(orient="records"), 1):
        _cancel_requested(cancel_callback)
        _feature_progress(progress_callback, family="time_domain", completed=window_index - 1,
                          total=len(windows), window_id=str(window.get("window_id", "")))
        wid = str(window["window_id"])
        ctx = contexts[wid]
        in_window = ctx["sample_mask"]
        sample_idx = np.flatnonzero(in_window)
        values = clean[sample_idx]
        selected_tonic = tonic[sample_idx]
        selected_phasic = phasic[sample_idx]
        feature_status, feature_reason = _stat_status(ctx, values)
        valid_duration = ctx["valid_duration"]
        validity_rows.append({"recording_id": rid, "window_id": wid, "kind": window["kind"],
                              "start_s": float(window["start_s"]), "end_s": float(window["end_s"]),
                              "status": {"succeeded": "valid", "partial": "valid", "invalid": "unavailable"}.get(ctx["status"], ctx["status"]),
                              "reason": ctx["reason"], "coverage": ctx["coverage"],
                              "expected_duration_s": ctx["expected_duration"], "valid_duration_s": valid_duration,
                              "detector_eligible_duration_s": ctx["detector_duration"], "valid_n": ctx["valid_n"],
                              "search_eligible_duration_s": ctx["search_duration"],
                              "onset_eligible_duration_s": ctx["onset_duration"],
                              "onset_left_context_s": ctx["onset_left_context_s"],
                              "onset_right_context_s": ctx["onset_right_context_s"],
                              "onset_context_complete": ctx["onset_window_supported"],
                              "interpolated_n": ctx["interpolated_n"], "artifact_n": ctx["artifact_n"],
                              "gap_crossing": ctx["gap_crossing"], "policy_id": "point_sample_adjacent_interval_clip"})

        def add(fid, val, unit, basis, method, status=None, reason=None, count=None, duration=None):
            _append_feature(feature_rows, metadata, window, fid, val, unit, basis, method,
                            status or feature_status, feature_reason if reason is None else reason,
                            ctx["valid_n"] if count is None else count,
                            valid_duration if duration is None else duration)

        if selected_ids.intersection(DRIVER_IDS):
            for fid, result in driver_window_metrics(bundle, window, full, driver_catalogue).items():
                if fid in selected_ids:
                    definition = registry[fid]
                    add(fid, result["value"], definition["unit"], "driver_uS_s", definition["formula"],
                        status=result["status"], reason=result["reason"], count=result["n"], duration=result["duration"])

        # Validity fractions are deliberately sample-count ratios, not exposure-weighted proportions.
        if ctx["all_n"]:
            add("valid_sample_proportion", ctx["valid_n"] / ctx["all_n"], "fraction", "quality",
                "valid_sample_count/recorded_window_sample_count", status="succeeded", reason="",
                count=ctx["all_n"])
            add("interpolated_sample_proportion", ctx["interpolated_n"] / ctx["all_n"], "fraction", "quality",
                "interpolated_sample_count/recorded_window_sample_count", status="succeeded", reason="",
                count=ctx["all_n"])
            add("artifact_sample_proportion", ctx["artifact_n"] / ctx["all_n"], "fraction", "quality",
                "artifact_sample_count/recorded_window_sample_count", status="succeeded", reason="",
                count=ctx["all_n"])
        else:
            for fid in ("valid_sample_proportion", "interpolated_sample_proportion", "artifact_sample_proportion"):
                add(fid, None, "fraction", "quality", "sample_count_fraction", status="unavailable",
                    reason="no_recorded_samples", count=0)

        if len(values):
            add("mean", float(np.mean(values)), "uS", "clean_uS", "sample_mean")
            add("median", float(np.median(values)), "uS", "clean_uS", "sample_median")
            add("range", float(np.max(values) - np.min(values)), "uS", "clean_uS", "sample_max_minus_min")
            add("sd", float(np.std(values, ddof=1)) if len(values) >= 2 else None, "uS", "clean_uS",
                "sample_standard_deviation_ddof1", status=feature_status if len(values) >= 2 else "unavailable",
                reason=feature_reason if len(values) >= 2 else "requires_at_least_two_samples")
            q25, q75 = np.percentile(values, [25, 75])
            add("iqr", float(q75 - q25), "uS", "clean_uS", "sample_percentile_75_minus_25")
            if len(values) >= 2 and np.ptp(time[sample_idx]) > 0:
                slope = float(np.polyfit(time[sample_idx], values, 1)[0])
                add("slope", slope, "uS/s", "clean_uS", "ordinary_least_squares_vs_time")
            else:
                add("slope", None, "uS/s", "clean_uS", "ordinary_least_squares_vs_time",
                    status="unavailable", reason="requires_two_distinct_sample_times")
        else:
            for fid, unit, method in (("mean", "uS", "sample_mean"), ("median", "uS", "sample_median"),
                                      ("range", "uS", "sample_max_minus_min"), ("sd", "uS", "sample_standard_deviation_ddof1"),
                                      ("iqr", "uS", "sample_percentile_75_minus_25"), ("slope", "uS/s", "ordinary_least_squares_vs_time")):
                add(fid, None, unit, "clean_uS", method, status="unavailable", reason="no_eligible_samples", count=0)

        for fid, component, label in (("tonic_mean", tonic, "tonic"), ("tonic_mean_change", tonic, "tonic")):
            select = in_window & np.isfinite(component)
            component_values = component[select]
            if len(component_values):
                value = float(np.mean(component_values)) if fid == "tonic_mean" else float(np.mean(component_values) - component_values[0])
                method = "sample_mean" if fid == "tonic_mean" else "sample_mean_minus_first_valid_tonic_sample"
                add(fid, value, "uS", label, method, count=len(component_values))
            else:
                add(fid, None, "uS", label, "sample_mean", status="unavailable", reason="no_decomposed_tonic_samples", count=0)

        tonic_mask = in_window & np.isfinite(tonic)
        tonic_indices = np.flatnonzero(tonic_mask)
        tonic_values = tonic[tonic_indices]
        for basis, distribution_values in (("clean", values), ("tonic", tonic_values)):
            for statistic in DISTRIBUTIONS:
                fid = f"{basis}_{statistic}"
                if fid not in selected_ids:
                    continue
                value, reason = distribution_summary(distribution_values, statistic)
                definition = registry[fid]
                add(fid, value, definition["unit"], definition["basis"], definition["formula"],
                    status=feature_status if not reason else "unavailable",
                    reason=feature_reason if ctx["status"] == "invalid" else reason or feature_reason,
                    count=len(distribution_values))
        ext_status = feature_status if len(tonic_values) else "unavailable"
        ext_reason = feature_reason if len(tonic_values) else "no_eligible_tonic_samples"
        for fid, value, unit, method, minimum in (
                ("tonic_median", float(np.median(tonic_values)) if len(tonic_values) else None, "uS", "sample_median", 1),
                ("tonic_sd", float(np.std(tonic_values, ddof=1)) if len(tonic_values) >= 2 else None, "uS", "sample_sd_ddof1", 2),
                ("tonic_iqr", float(np.percentile(tonic_values, 75, method="linear") -
                                     np.percentile(tonic_values, 25, method="linear")) if len(tonic_values) >= 2 else None,
                 "uS", "linear_percentile_75_minus_25", 2)):
            valid = len(tonic_values) >= minimum
            add(fid, value if valid else None, unit, "tonic_uS", method,
                status=ext_status if valid else "unavailable",
                reason=ext_reason if valid else "requires_at_least_two_samples" if len(tonic_values) else ext_reason,
                count=len(tonic_values))
        if len(tonic_values) >= 2 and np.ptp(time[tonic_indices]) > 0:
            tonic_slope = float(np.polyfit(time[tonic_indices], tonic_values, 1)[0])
            tonic_slope_status, tonic_slope_reason = ext_status, ext_reason
        else:
            tonic_slope, tonic_slope_status, tonic_slope_reason = None, "unavailable", "requires_two_distinct_sample_times"
        add("tonic_slope", tonic_slope, "uS/s", "tonic_uS", "ordinary_least_squares_vs_time",
            status=tonic_slope_status, reason=tonic_slope_reason, count=len(tonic_values))
        if len(tonic_indices) >= 2:
            first_i, last_i = int(tonic_indices[0]), int(tonic_indices[-1])
            span_valid = bool(np.all(tonic_mask[first_i:last_i + 1]))
            if span_valid and last_i > first_i:
                deltas = np.diff(time[first_i:last_i + 1])
                span_valid = bool(np.all(deltas > 0) and np.all(deltas <= ctx["max_gap"] + 1e-12))
            first_last = float(tonic[last_i] - tonic[first_i]) if span_valid else None
            first_last_status = ext_status if span_valid else "unavailable"
            first_last_reason = ext_reason if span_valid else "NA_gap"
        else:
            first_last, first_last_status, first_last_reason = None, "unavailable", "requires_at_least_two_samples"
        add("tonic_first_last_change_uS", first_last, "uS", "tonic_uS", "last_minus_first_eligible_sample_no_gap",
            status=first_last_status, reason=first_last_reason, count=len(tonic_values))

        # 边界插值使用相邻有效点作为线性支撑；样本点本身可在半开窗外，积分区间仍先与窗裁剪。
        phasic_mask = _bool_array(bundle.get("quality", {}).get("analysis_eligible",
                                                bundle.get("quality", {}).get("eligible")),
                                  len(time), "analysis_eligible") & np.isfinite(phasic)
        if full["features"].get("exclude_interpolated", True):
            phasic_mask &= ~_bool_array(bundle.get("quality", {}).get("interpolated"), len(time), "interpolated")
        phasic_area, phasic_intervals = _clip_integral(time, phasic, phasic_mask, ctx["start"], ctx["end"], ctx["max_gap"])
        if phasic_intervals and not ctx["gap_crossing"] and ctx["status"] != "invalid":
            add("phasic_auc", phasic_area, "uS_s", "phasic_uS", "signed_trapezoid_clipped_to_window",
                count=int(np.count_nonzero(phasic_mask)))
        else:
            status = "invalid" if ctx["gap_crossing"] or ctx["status"] == "invalid" else "unavailable"
            add("phasic_auc", None, "uS_s", "phasic_uS", "signed_trapezoid_clipped_to_window",
                status=status, reason=ctx["reason"] if status == "invalid" else "no_valid_adjacent_intervals")
        phasic_integrals = _clip_signed_positive_absolute(time, phasic, phasic_mask,
                                                           ctx["start"], ctx["end"], ctx["max_gap"])
        integral_status = feature_status if phasic_integrals["duration"] > 0 and not ctx["gap_crossing"] else "unavailable"
        integral_reason = feature_reason if integral_status == feature_status else (
            ctx["reason"] if ctx["status"] == "invalid" else "no_valid_phasic_duration")
        for fid, val, unit, method in (
                ("phasic_positive_auc_uS_s", phasic_integrals["positive"], "uS_s", "piecewise_linear_positive_integral_exact_zero_crossing"),
                ("phasic_absolute_auc_uS_s", phasic_integrals["absolute"], "uS_s", "piecewise_linear_absolute_integral_exact_zero_crossing"),
                ("phasic_auc_per_valid_s", phasic_integrals["signed"] / phasic_integrals["duration"] if phasic_integrals["duration"] > 0 else None,
                 "uS", "signed_auc_divided_by_valid_phasic_duration"),
                ("phasic_positive_auc_per_valid_s", phasic_integrals["positive"] / phasic_integrals["duration"] if phasic_integrals["duration"] > 0 else None,
                 "uS", "positive_auc_divided_by_valid_phasic_duration"),
                ("phasic_absolute_auc_per_valid_s", phasic_integrals["absolute"] / phasic_integrals["duration"] if phasic_integrals["duration"] > 0 else None,
                 "uS", "absolute_auc_divided_by_valid_phasic_duration")):
            add(fid, val if integral_status != "unavailable" else None, unit, "phasic_uS", method,
                status=integral_status, reason=integral_reason,
                count=int(np.count_nonzero(phasic_mask)), duration=phasic_integrals["duration"])

        in_scr = [item for item in scr_rows if float(window["start_s"]) <= float(item["onset_s"]) < float(window["end_s"])]
        strict_scr_ok = ctx["status"] != "invalid"
        onset_support = ctx["onset_duration"] > 0
        scr_usable = strict_scr_ok and onset_support
        scr_status = feature_status if scr_usable else "invalid" if not strict_scr_ok else "unavailable"
        scr_reason = (feature_reason if scr_usable else ctx["reason"] if not strict_scr_ok else
                      "no_detector_eligible_exposure")
        amps = np.asarray([float(item["amplitude_uS"]) for item in in_scr], dtype=float)
        add("scr_count", int(len(in_scr)) if scr_usable else None, "count", "qualified_global_scr_catalogue",
            "onset_membership_half_open", status=scr_status, reason=scr_reason, count=len(in_scr))
        if ctx["detector_duration"] > 0 and strict_scr_ok:
            add("scr_rate", 60.0 * len(in_scr) / ctx["detector_duration"], "responses/min", "qualified_global_scr_catalogue",
                "60_times_onset_count_over_onset_eligible_seconds", status=feature_status, reason=feature_reason,
                count=len(in_scr), duration=ctx["detector_duration"])
        else:
            add("scr_rate", None, "responses/min", "qualified_global_scr_catalogue",
                "60_times_onset_count_over_onset_eligible_seconds", status="invalid" if not strict_scr_ok else "unavailable",
                reason=scr_reason if not strict_scr_ok else "no_detector_eligible_exposure", count=len(in_scr),
                duration=ctx["detector_duration"])
        amp_mean = float(np.mean(amps)) if len(amps) and scr_usable else None
        amp_median = float(np.median(amps)) if len(amps) and scr_usable else None
        amp_sum = float(np.sum(amps)) if scr_usable else None
        amp_empty_status = "invalid" if not strict_scr_ok else "unavailable" if not onset_support or not len(amps) else feature_status
        amp_empty_reason = scr_reason if not scr_usable else "no_responses" if not len(amps) else feature_reason
        add("scr_amplitude_mean", amp_mean, "uS", "qualified_global_scr_catalogue", "mean_of_qualified_onset_member_amplitudes",
            status=amp_empty_status, reason=amp_empty_reason, count=len(amps))
        add("scr_amplitude_median", amp_median, "uS", "qualified_global_scr_catalogue", "median_of_qualified_onset_member_amplitudes",
            status=amp_empty_status, reason=amp_empty_reason, count=len(amps))
        add("scr_amplitude_sum", amp_sum, "uS", "qualified_global_scr_catalogue", "zero_inclusive_sum_of_qualified_onset_member_amplitudes",
            status=scr_status if scr_usable else amp_empty_status, reason=scr_reason if not scr_usable else feature_reason,
            count=len(amps))
        scr_areas = []
        for item in in_scr:
            try:
                area_value = float(item.get("area_uS_s", np.nan))
            except (TypeError, ValueError):
                area_value = np.nan
            if np.isfinite(area_value):
                scr_areas.append(area_value)
        areas_complete = len(scr_areas) == len(in_scr)
        if not scr_usable:
            area_sum_value, area_sum_status, area_sum_reason = None, scr_status, scr_reason
        elif in_scr and not areas_complete:
            area_sum_value, area_sum_status, area_sum_reason = None, "unavailable", "scr_area_missing_from_catalogue"
        else:
            area_sum_value, area_sum_status, area_sum_reason = float(np.sum(scr_areas)), feature_status, feature_reason
        if not scr_usable:
            area_mean_value, area_mean_status, area_mean_reason = None, scr_status, scr_reason
        elif not in_scr:
            area_mean_value, area_mean_status, area_mean_reason = None, "unavailable", "no_responses"
        elif not areas_complete:
            area_mean_value, area_mean_status, area_mean_reason = None, "unavailable", "scr_area_missing_from_catalogue"
        else:
            area_mean_value, area_mean_status, area_mean_reason = float(np.mean(scr_areas)), feature_status, feature_reason
        area_basis = "qualified_global_scr_catalogue.area_uS_s"
        add("scr_area_sum_uS_s", area_sum_value, "uS_s", area_basis,
            "sum_of_qualified_scr_signed_trapezoid_above_onset_level_on_[onset_s,peak_s]",
            status=area_sum_status, reason=area_sum_reason, count=len(scr_areas))
        add("scr_area_mean_uS_s", area_mean_value, "uS_s", area_basis,
            "mean_of_qualified_scr_signed_trapezoid_above_onset_level_on_[onset_s,peak_s]",
            status=area_mean_status, reason=area_mean_reason, count=len(scr_areas))
        rises = [float(item["rise_time_s"]) for item in in_scr if np.isfinite(item.get("rise_time_s", np.nan))]
        censored_recoveries = [item for item in in_scr if item.get("recovery_censor_reason")]
        recoveries = [float(item["recovery_time_s"]) for item in in_scr
                      if np.isfinite(item.get("recovery_time_s", np.nan)) and not item.get("recovery_censor_reason")]
        for transform, (_, unit, method) in TRANSFORMS.items():
            fid = f"scr_amplitude_{transform}_mean"
            if fid not in selected_ids:
                continue
            transformed, reason = transform_amplitudes(amps, transform, recording_amplitude_reference)
            okay = scr_usable and not reason
            add(fid, float(np.mean(transformed)) if okay else None, unit, "qualified_global_scr_catalogue", method,
                status=feature_status if okay else "unavailable",
                reason=scr_reason if not scr_usable else reason or feature_reason, count=len(amps))

        def add_timing_dispersion(prefix, durations, *, suffix="", censored=0, unknown=False):
            for statistic, minimum in (("median", 1), ("sd", 2)):
                fid = f"{prefix}_{statistic}{suffix}"
                if fid not in selected_ids:
                    continue
                reason = (scr_reason if not scr_usable else "recovery_status_unavailable" if unknown else
                          "all_recoveries_censored" if not durations and censored else
                          f"requires_at_least_{minimum}_completed_durations" if len(durations) < minimum else "")
                okay = scr_usable and not reason
                value = (float(np.median(durations)) if statistic == "median" else
                         float(np.std(durations, ddof=1))) if okay else None
                status = feature_status if okay else "censored" if scr_usable and not unknown and not durations and censored else "unavailable"
                add(fid, value, "s", "qualified_global_scr_catalogue", registry[fid]["formula"],
                    status=status, reason=reason or ("some_censored_recoveries_excluded" if censored else feature_reason),
                    count=len(durations))

        add_timing_dispersion("scr_rise_time", rises)
        add_timing_dispersion("scr_recovery_time", recoveries, censored=len(censored_recoveries),
                             unknown=any(not np.isfinite(item.get("recovery_time_s", np.nan)) and
                                         not item.get("recovery_censor_reason") for item in in_scr))
        for fid, vals, unit, method, unavailable_reason in (
                ("scr_rise_time_mean", rises, "s", "mean_of_valid_rise_durations", "no_responses" if not in_scr else "no_valid_rise_times"),
                ("scr_recovery_time_mean", recoveries, "s", "mean_of_completed_recovery_durations", "no_responses" if not in_scr else "no_completed_recoveries")):
            is_recovery = fid == "scr_recovery_time_mean"
            status = (feature_status if vals and scr_usable else "invalid" if not strict_scr_ok else
                      "unavailable" if not onset_support else
                      "censored" if is_recovery and censored_recoveries else "unavailable")
            reason = (feature_reason if vals and scr_usable else scr_reason if not strict_scr_ok or not onset_support else
                      "some_censored_recoveries_excluded" if is_recovery and vals and censored_recoveries else
                      "all_recoveries_censored" if is_recovery and censored_recoveries else unavailable_reason)
            add(fid, float(np.mean(vals)) if vals and strict_scr_ok else None, unit, "global_scr_catalogue", method,
                status=status, reason=reason, count=len(vals))

        amp_summary_specs = (
            ("scr_amplitude_sd", float(np.std(amps, ddof=1)) if len(amps) >= 2 else None, "uS", "sample_sd_ddof1", 2),
            ("scr_amplitude_max", float(np.max(amps)) if len(amps) else None, "uS", "sample_max", 1),
            ("scr_amplitude_p90", float(np.percentile(amps, 90, method="linear")) if len(amps) else None,
             "uS", "linear_percentile_90", 1),
            ("scr_amplitude_iqr", float(np.percentile(amps, 75, method="linear") -
                                         np.percentile(amps, 25, method="linear")) if len(amps) >= 2 else None,
             "uS", "linear_percentile_75_minus_25", 2))
        for fid, value, unit, method, minimum in amp_summary_specs:
            okay = scr_usable and len(amps) >= minimum
            reason = scr_reason if not scr_usable else "no_responses" if not len(amps) else "requires_at_least_two_responses"
            add(fid, value if okay else None, unit, "qualified_global_scr_catalogue", method,
                status=feature_status if okay else "invalid" if not strict_scr_ok else "unavailable",
                reason=feature_reason if okay else reason, count=len(amps))

        rise_mean = float(np.mean([float(item["amplitude_uS"]) / float(item["rise_time_s"])
                                   for item in in_scr if np.isfinite(item.get("rise_time_s", np.nan)) and
                                   float(item["rise_time_s"]) > 0])) if any(
                                       np.isfinite(item.get("rise_time_s", np.nan)) and float(item["rise_time_s"]) > 0
                                       for item in in_scr) else None
        max_rise_slopes = [float(item["rise_max_slope_uS_s"]) for item in in_scr
                           if np.isfinite(item.get("rise_max_slope_uS_s", np.nan))]
        valid_rise_n = sum(np.isfinite(item.get("rise_time_s", np.nan)) and float(item["rise_time_s"]) > 0 for item in in_scr)
        for fid, value, unit, method, count in (
                ("scr_rise_slope_mean_uS_s", rise_mean, "uS/s", "mean_per_scr_amplitude_divided_by_rise_time", valid_rise_n),
                ("scr_rise_max_slope_mean_uS_s", float(np.mean(max_rise_slopes)) if max_rise_slopes else None,
                 "uS/s", "mean_of_per_scr_max_adjacent_signed_rise_slope", len(max_rise_slopes)),
                ("scr_rise_max_slope_max_uS_s", float(np.max(max_rise_slopes)) if max_rise_slopes else None,
                 "uS/s", "maximum_of_per_scr_max_adjacent_signed_rise_slope", len(max_rise_slopes))):
            enough = scr_usable and count > 0
            add(fid, value if enough else None, unit, "qualified_global_scr_catalogue", method,
                status=feature_status if enough else "invalid" if not strict_scr_ok else "unavailable",
                reason=feature_reason if enough else scr_reason if not scr_usable else "no_valid_scr_rise_slope", count=count)

        for fraction_id, label in (("50", "50"), ("63", "63")):
            status_key = f"recovery_{fraction_id}_status"
            duration_key = f"recovery_{fraction_id}_time_s"
            valid_recoveries = [float(item[duration_key]) for item in in_scr
                                if item.get(status_key) == "valid" and np.isfinite(item.get(duration_key, np.nan))]
            censored_n = sum(item.get(status_key) == "censored" for item in in_scr)
            unknown_recovery = any(item.get(status_key) not in ("valid", "censored") for item in in_scr)
            add_timing_dispersion(f"scr_recovery_{fraction_id}", valid_recoveries, suffix="_s",
                                 censored=censored_n, unknown=unknown_recovery)
            recovery_mean = float(np.mean(valid_recoveries)) if valid_recoveries else None
            mean_status = ("unavailable" if unknown_recovery else
                           feature_status if valid_recoveries and scr_usable else
                           "censored" if censored_n else "unavailable")
            mean_reason = ("recovery_status_unavailable" if unknown_recovery else
                           feature_reason if valid_recoveries and scr_usable else
                           "all_recoveries_right_censored" if censored_n and not unknown_recovery else
                           "no_responses" if not in_scr else "recovery_status_unavailable")
            add(f"scr_recovery_{fraction_id}_mean_s", recovery_mean, "s", "qualified_global_scr_catalogue",
                f"mean_observed_recovery_duration_f_returned_{label}", status=mean_status,
                reason=mean_reason, count=len(valid_recoveries))
            proportion = censored_n / len(in_scr) if in_scr and not unknown_recovery else None
            p_status = feature_status if proportion is not None and scr_usable else "unavailable"
            p_reason = feature_reason if p_status == feature_status else "no_responses" if not in_scr else "recovery_status_unavailable"
            add(f"scr_unrecovered_{fraction_id}_proportion", proportion, "fraction", "qualified_global_scr_catalogue",
                "right_censored_response_count_divided_by_primary_response_count_including_next_response_cutoff",
                status=p_status, reason=p_reason, count=len(in_scr))

        widths = [float(item["half_width_50_s"]) for item in in_scr
                  if item.get("half_width_50_status") == "valid" and np.isfinite(item.get("half_width_50_s", np.nan))]
        width_censored = any(item.get("half_width_50_status") == "censored" for item in in_scr)
        width_status = feature_status if widths and scr_usable else "censored" if width_censored else "unavailable"
        width_reason = feature_reason if widths and scr_usable else "half_width_right_censored" if width_censored else "no_observed_complete_half_width"
        add("scr_half_width_50_mean_s", float(np.mean(widths)) if widths and scr_usable else None, "s",
            "qualified_global_scr_catalogue", "mean_of_complete_50_percent_half_widths_only",
            status=width_status, reason=width_reason, count=len(widths))

        isi_items = sorted(in_scr, key=lambda item: float(item["onset_s"]))
        isi_values = []
        for previous, current in zip(isi_items[:-1], isi_items[1:]):
            previous_segment = previous.get("segment_id", previous.get("analysis_segment_id"))
            current_segment = current.get("segment_id", current.get("analysis_segment_id"))
            if previous_segment is not None and current_segment is not None and str(previous_segment) != str(current_segment):
                continue
            if previous_segment is None and current_segment is None:
                continue
            isi_values.append(float(current["onset_s"]) - float(previous["onset_s"]))
        isi = np.asarray(isi_values, dtype=float)
        isi_specs = (
            ("scr_isi_mean_s", float(np.mean(isi)) if len(isi) else None, "s", "sample_mean_of_within_segment_onset_intervals", 1),
            ("scr_isi_median_s", float(np.median(isi)) if len(isi) else None, "s", "sample_median_of_within_segment_onset_intervals", 1),
            ("scr_isi_sd_s", float(np.std(isi, ddof=1)) if len(isi) >= 2 else None, "s", "sample_sd_ddof1", 2),
            ("scr_isi_cv", float(np.std(isi, ddof=1) / np.mean(isi)) if len(isi) >= 2 and np.mean(isi) > 0 else None,
             "dimensionless", "sample_sd_divided_by_mean", 2))
        for fid, value, unit, method, minimum in isi_specs:
            okay = scr_usable and len(isi) >= minimum and (fid != "scr_isi_cv" or np.mean(isi) > 0)
            reason = scr_reason if not scr_usable else "no_within_segment_isi" if len(isi) == 0 else (
                "requires_at_least_two_isi_values" if len(isi) < minimum else "nonpositive_mean_isi")
            add(fid, value if okay else None, unit, "qualified_global_scr_catalogue", method,
                status=feature_status if okay else "invalid" if not strict_scr_ok else "unavailable",
                reason=feature_reason if okay else reason, count=len(isi))

        ns_exposure = _support_duration(ns_intervals, float(window["start_s"]), float(window["end_s"]))
        ns_scr_rows = []
        if not ns_table.empty and not ns_policy_reason:
            ns_ids = set(ns_table.loc[ns_table["class"].eq("non_event_associated"), "scr_id"].astype(str))
            ns_scr_rows = [item for item in scr_rows
                           if str(item.get("scr_id", "")) in ns_ids and
                           float(window["start_s"]) <= float(item["onset_s"]) < float(window["end_s"])]
        ns_usable = ns_exposure > 0 and not ns_policy_reason
        ns_reason = ns_policy_reason or "zero_non_event_onset_exposure"
        ns_amps = np.asarray([float(item["amplitude_uS"]) for item in ns_scr_rows], dtype=float)
        add("ns_scr_count", len(ns_scr_rows) if ns_usable else None, "count", "non_event_associated_scr",
            "qualified_non_event_onsets_in_half_open_window", status="valid" if ns_usable else "unavailable",
            reason="" if ns_usable else ns_reason, count=len(ns_scr_rows), duration=ns_exposure)
        add("ns_scr_rate", 60.0 * len(ns_scr_rows) / ns_exposure if ns_usable else None, "responses/min",
            "non_event_associated_scr", "60_times_count_over_complete_annotation_onset_exposure",
            status="valid" if ns_usable else "unavailable", reason="" if ns_usable else ns_reason,
            count=len(ns_scr_rows), duration=ns_exposure)
        ns_summaries = (
            ("ns_scr_amplitude_mean", float(np.mean(ns_amps)) if len(ns_amps) else None, "uS", "sample_mean", 1),
            ("ns_scr_amplitude_median", float(np.median(ns_amps)) if len(ns_amps) else None, "uS", "sample_median", 1),
            ("ns_scr_amplitude_sd", float(np.std(ns_amps, ddof=1)) if len(ns_amps) >= 2 else None, "uS", "sample_sd_ddof1", 2),
            ("ns_scr_amplitude_max", float(np.max(ns_amps)) if len(ns_amps) else None, "uS", "sample_max", 1),
            ("ns_scr_amplitude_p90", float(np.percentile(ns_amps, 90, method="linear")) if len(ns_amps) else None,
             "uS", "linear_percentile_90", 1),
            ("ns_scr_amplitude_iqr", float(np.percentile(ns_amps, 75, method="linear") -
                                             np.percentile(ns_amps, 25, method="linear")) if len(ns_amps) >= 2 else None,
             "uS", "linear_percentile_75_minus_25", 2))
        for fid, value, unit, method, minimum in ns_summaries:
            ns_value_valid = ns_usable and len(ns_amps) >= minimum
            reason = "" if ns_value_valid else ns_reason if not ns_usable else "no_non_event_associated_responses" if not len(ns_amps) else "requires_at_least_two_responses"
            add(fid, value if ns_value_valid else None, unit, "non_event_associated_scr_amplitude",
                method, status="valid" if ns_value_valid else "unavailable", reason=reason,
                count=len(ns_amps), duration=ns_exposure)

        # Event metrics are attached to the matching event window; unique links בלבד enter summaries.
        kind = str(window["kind"])
        event_id = window.get("event_id")
        event_id_key = str(event_id) if event_id is not None else ""
        if kind.startswith("event_") and event_id is not None:
            event_baseline = next((row for row in windows.to_dict(orient="records")
                                   if str(row.get("event_id")) == event_id_key and row["kind"] == "event_baseline"), None)
            baseline_value, baseline_valid = None, False
            if event_baseline is not None:
                bctx = contexts[str(event_baseline["window_id"])]
                bmask = bctx["sample_mask"]
                baseline_samples = clean[bmask]
                base_contamination = any(float(event_baseline["start_s"]) <= float(item["onset_s"]) < float(event_baseline["end_s"])
                                         for item in catalogue_rows)
                baseline_valid = bctx["status"] == "succeeded" and not base_contamination and len(baseline_samples) > 0
                if baseline_valid:
                    baseline_value = float(np.mean(baseline_samples))
            event_scrs = [scr_by_id[sid] for sid in unique_by_event.get(event_id_key, set()) if sid in scr_by_id]
            event_scrs.sort(key=lambda item: float(item["onset_s"]))
            event_ambiguous = event_id_key in ambiguous_by_event
            event_unavailable = event_id_key in unavailable_by_event or not ctx["onset_window_supported"]
            event_invalid = ctx["status"] == "invalid" or event_id_key in invalid_by_event
            event_signal_valid = ctx["status"] == "succeeded"
            event_window_valid = event_signal_valid and ctx["onset_window_supported"] and not event_ambiguous and not event_unavailable and not event_invalid
            if event_ambiguous:
                invalid_event_status, invalid_event_reason = "ambiguous", "onset_in_multiple_event_response_windows"
            elif event_invalid:
                invalid_event_status, invalid_event_reason = "invalid", ctx["reason"] or "invalid_event_window"
            elif event_unavailable:
                invalid_event_status, invalid_event_reason = "unavailable", "insufficient_onset_context"
            else:
                invalid_event_status, invalid_event_reason = "invalid", ctx["reason"] or "invalid_event_window"
            if kind == "event_baseline":
                reason = "baseline_contaminated_by_scr" if base_contamination else ctx["reason"]
                status = "invalid" if base_contamination else "succeeded" if baseline_valid else "invalid" if ctx["status"] == "invalid" else "unavailable"
                add("event_baseline_mean_uS", baseline_value, "uS", "clean_uS", "sample_mean_in_configured_baseline_window",
                    status=status, reason=reason, count=len(baseline_samples))
            elif kind == "event_response":
                response_count = len(event_scrs) if event_window_valid else None
                first = event_scrs[0] if event_scrs else None
                amps_event = [float(item["amplitude_uS"]) for item in event_scrs]
                for fid, value, unit, basis, method, status, reason in (
                    ("event_scr_count", response_count, "count", "unique_event_scr_links", "unique_onset_membership_count",
                     "succeeded" if event_window_valid else invalid_event_status, "" if event_window_valid else invalid_event_reason),
                    ("event_response_present", int(bool(event_scrs)) if event_window_valid else None, "boolean", "unique_event_scr_links", "count_gt_zero",
                     "succeeded" if event_window_valid else invalid_event_status, "" if event_window_valid else invalid_event_reason),
                    ("event_first_amplitude_uS", float(first["amplitude_uS"]) if first and event_window_valid else None, "uS", "unique_event_scr_links", "earliest_scr_onset_amplitude",
                     "succeeded" if first and event_window_valid else "unavailable" if event_window_valid else invalid_event_status, "no_unique_response" if event_window_valid and not first else invalid_event_reason),
                    ("event_max_amplitude_uS", float(max(amps_event)) if amps_event and event_window_valid else None, "uS", "unique_event_scr_links", "max_unique_response_amplitude",
                     "succeeded" if amps_event and event_window_valid else "unavailable" if event_window_valid else invalid_event_status, "no_unique_response" if event_window_valid and not amps_event else invalid_event_reason),
                    ("event_amp_sum_uS", float(sum(amps_event)) if event_window_valid else None, "uS", "unique_event_scr_links", "zero_inclusive_unique_response_sum",
                     "succeeded" if event_window_valid else invalid_event_status, "" if event_window_valid else invalid_event_reason),
                    ("event_latency_s", float(first["onset_s"] - float(window.get("event_onset_s", np.nan))) if first and event_window_valid else None,
                     "s", "unique_event_scr_links", "first_scr_onset_minus_event_onset",
                     "succeeded" if first and event_window_valid else "unavailable" if event_window_valid else invalid_event_status,
                     "no_unique_response" if event_window_valid and not first else invalid_event_reason),
                ):
                    add(fid, value, unit, basis, method, status=status, reason=reason, count=len(event_scrs))
                event_areas = []
                for item in event_scrs:
                    try:
                        area_value = float(item.get("area_uS_s", np.nan))
                    except (TypeError, ValueError):
                        area_value = np.nan
                    if np.isfinite(area_value):
                        event_areas.append(area_value)
                if not event_window_valid:
                    event_area_sum, event_area_status, event_area_reason = None, invalid_event_status, invalid_event_reason
                elif event_scrs and len(event_areas) != len(event_scrs):
                    event_area_sum, event_area_status, event_area_reason = None, "unavailable", "scr_area_missing_from_catalogue"
                else:
                    event_area_sum, event_area_status, event_area_reason = float(np.sum(event_areas)), "succeeded", ""
                add("event_scr_area_sum_uS_s", event_area_sum, "uS_s", "unique_event_scr_links.area_uS_s",
                    "sum_of_unique_event_scr_signed_trapezoid_above_onset_level_on_[onset_s,peak_s]",
                    status=event_area_status, reason=event_area_reason, count=len(event_areas))
                response_phasic, response_intervals = _clip_integral(time, phasic, phasic_mask,
                                                                      ctx["start"], ctx["end"], ctx["max_gap"])
                add("event_phasic_auc_uS_s", response_phasic if response_intervals and event_signal_valid else None,
                    "uS_s", "phasic_uS", "signed_trapezoid_in_event_response_window",
                    status="succeeded" if response_intervals and event_signal_valid else "invalid" if ctx["status"] == "invalid" else "unavailable",
                    reason="" if response_intervals and event_signal_valid else ctx["reason"] if ctx["status"] == "invalid" else "no_valid_adjacent_intervals")
                response_clean = clean[in_window & np.isfinite(clean)]
                response_mean = float(np.mean(response_clean)) if len(response_clean) else None
                delta = response_mean - baseline_value if response_mean is not None and baseline_valid and event_signal_valid else None
                add("event_response_mean_change_uS", delta, "uS", "clean_uS",
                    "sample_clean_mean_minus_valid_clean_baseline_mean",
                    status="succeeded" if delta is not None else "invalid" if ctx["status"] == "invalid" or not baseline_valid else "unavailable",
                    reason="" if delta is not None else "baseline_invalid_or_contaminated_or_incomplete_signal_window")
                trial = trial_by_event.get(event_id_key)
                trial_valid = trial is not None and trial.get("trial_status") == "valid"
                trial_resp = bool(trial.get("response_present")) if trial_valid else False
                trial_amplitude = float(trial["amplitude_zero_uS"]) if trial_valid else None
                trial_latency = float(trial["peak_latency_s"]) if trial_valid and trial_resp and np.isfinite(trial.get("peak_latency_s", np.nan)) else None
                event_row_status = ("valid" if trial_valid else "ambiguous" if event_ambiguous else
                                    "unavailable" if not trial_valid else invalid_event_status)
                add("event_peak_latency_s", trial_latency, "s", "unique_event_scr_links",
                    "maximum_linked_scr_peak_minus_event_onset_earlier_peak_tie",
                    status="valid" if trial_valid and trial_resp and trial_latency is not None else
                    "unavailable" if trial_valid else event_row_status,
                    reason="" if trial_valid and trial_resp and trial_latency is not None else
                    "no_unique_response" if trial_valid else str(trial.get("reason", invalid_event_reason) if trial is not None else invalid_event_reason),
                    count=len(event_scrs))
                bmean = float(trial["baseline_mean_uS"]) if trial_valid and np.isfinite(trial.get("baseline_mean_uS", np.nan)) else None
                bsd = float(trial["baseline_sd_uS"]) if trial_valid and np.isfinite(trial.get("baseline_sd_uS", np.nan)) else None
                response_change = float(trial["response_change_uS"]) if trial_valid and np.isfinite(trial.get("response_change_uS", np.nan)) else None
                baseline_min = float(full["features"].get("extended", {}).get("baseline_min_uS", 1e-6))
                if not trial_valid:
                    scaled_status, scaled_reason = event_row_status, str(trial.get("reason", invalid_event_reason) if trial is not None else invalid_event_reason)
                elif bmean is None or bmean <= baseline_min:
                    scaled_status, scaled_reason = "unavailable", "baseline_mean_not_above_baseline_min_uS"
                else:
                    scaled_status, scaled_reason = "valid", ""
                baseline_percent = (100.0 * trial_amplitude / bmean) if scaled_status == "valid" and trial_amplitude is not None else None
                response_percent = (100.0 * response_change / bmean) if scaled_status == "valid" and response_change is not None else None
                z_status, z_reason = scaled_status, scaled_reason
                if z_status == "valid" and (bsd is None or bsd <= 0):
                    z_status, z_reason = "unavailable", "baseline_sd_requires_three_samples_and_positive_variance"
                baseline_z = (trial_amplitude / bsd) if z_status == "valid" and trial_amplitude is not None else None
                response_z = (response_change / bsd) if z_status == "valid" and response_change is not None else None
                for fid, value, unit, method, mstatus, mreason in (
                        ("event_max_amplitude_baseline_percent", baseline_percent, "%", "100_times_zero_inclusive_trial_max_amplitude_divided_by_clean_baseline_mean", scaled_status, scaled_reason),
                        ("event_response_change_baseline_percent", response_percent, "%", "100_times_clean_response_change_divided_by_clean_baseline_mean", scaled_status, scaled_reason),
                        ("event_max_amplitude_baseline_z", baseline_z, "z", "zero_inclusive_trial_max_amplitude_divided_by_clean_baseline_sd_ddof1", z_status, z_reason),
                        ("event_response_change_baseline_z", response_z, "z", "clean_response_change_divided_by_clean_baseline_sd_ddof1", z_status, z_reason)):
                    add(fid, value, unit, "clean_baseline", method, status=mstatus, reason=mreason,
                        count=int(np.count_nonzero(contexts[str(event_baseline["window_id"])]["sample_mask"]))
                        if event_baseline is not None and baseline_valid else 0)
            elif kind == "event_peak":
                peak_start, peak_end = float(window["start_s"]), float(window["end_s"])
                peak_scrs = [item for item in event_scrs
                             if np.isfinite(item.get("peak_s", np.nan)) and peak_start <= float(item["peak_s"]) < peak_end]
                peak_value = max((float(item["amplitude_uS"]) for item in peak_scrs), default=None)
                add("event_peak_max_amplitude_uS", peak_value if event_window_valid else None, "uS", "unique_event_scr_links",
                    "max_amplitude_by_scr_peak_in_peak_search_window",
                    status="succeeded" if peak_value is not None and event_window_valid else "unavailable" if event_window_valid else invalid_event_status,
                    reason="no_unique_response_peak_in_peak_window" if event_window_valid and peak_value is None else invalid_event_reason)
            elif kind == "event_recovery":
                recovery_lo, recovery_hi = float(window["start_s"]), float(window["end_s"])
                recovery_items = [item for item in event_scrs
                                  if np.isfinite(item.get("recovery_s", np.nan)) and
                                  recovery_lo <= float(item["recovery_s"]) < recovery_hi]
                durations = [float(item["recovery_time_s"]) for item in recovery_items
                             if np.isfinite(item.get("recovery_time_s", np.nan)) and not item.get("recovery_censor_reason")]
                has_censored = any(item.get("recovery_censor_reason") for item in event_scrs)
                if not event_window_valid:
                    recovery_status, recovery_reason = invalid_event_status, invalid_event_reason
                elif durations:
                    recovery_status = "succeeded"
                    recovery_reason = "completed_only_censored_recoveries_excluded" if has_censored else ""
                elif has_censored:
                    recovery_status, recovery_reason = "censored", "all_unique_recoveries_censored"
                else:
                    recovery_status, recovery_reason = "unavailable", "no_completed_unique_recoveries"
                add("event_recovery_mean_s", float(np.mean(durations)) if durations and event_window_valid else None,
                    "s", "unique_event_scr_links", "mean_of_completed_unique_link_recoveries",
                    status=recovery_status, reason=recovery_reason,
                    count=len(durations))

        # Advanced clean-only families require one contiguous complete uniform window.
        if advanced_selected:
            _cancel_requested(cancel_callback)
            if full["features"].get("extended", {}).get("enabled", True) is not True:
                advanced_result = {"metrics": {fid: {"value": None, "unit": registry[fid]["unit"],
                                                      "basis": "clean_uS", "method": "clean_advanced_window",
                                                      "status": "unavailable", "reason": "extended_metrics_disabled",
                                                      "valid_n": 0, "valid_duration_s": 0.0, "parameters": {}}
                                                  for fid in advanced_selected}, "tables": {}}
            else:
                from .metrics_spectral_complexity import compute_clean_advanced_window
                advanced_mask = _bool_array(bundle.get("quality", {}).get("analysis_eligible",
                                                                            bundle.get("quality", {}).get("eligible")),
                                            len(time), "analysis_eligible") & np.isfinite(clean)
                if full["features"].get("exclude_interpolated", True):
                    advanced_mask &= ~_bool_array(bundle.get("quality", {}).get("interpolated"),
                                                   len(time), "interpolated")
                meta = bundle.get("metadata", {}) if isinstance(bundle.get("metadata"), dict) else {}
                declared_fs = bundle.get("sampling_rate_hz", meta.get("fs_reference_hz", meta.get("fs_hz")))
                jitter = float(full.get("preprocess", {}).get("jitter_tolerance", 0.01))
                advanced_result = compute_clean_advanced_window(
                    time, clean, advanced_mask, window_start_s=float(window["start_s"]),
                    window_end_s=float(window["end_s"]), declared_fs_hz=declared_fs,
                    max_gap_s=ctx["max_gap"], jitter_tolerance=jitter,
                    extended_config=full["features"].get("extended", {}),
                    selected_ids=advanced_selected,
                    identity={"recording_id": rid, "window_id": wid},
                    cancel_callback=cancel_callback, progress_callback=progress_callback)
            for feature_id, result in advanced_result.get("metrics", {}).items():
                add(feature_id, result.get("value"), result.get("unit", registry[feature_id]["unit"]),
                    result.get("basis", registry[feature_id]["basis"]), result.get("method", "clean_advanced_window"),
                    status=result.get("status", "unavailable"), reason=result.get("reason", ""),
                    count=result.get("valid_n", 0), duration=result.get("valid_duration_s", 0.0))
            for table_name in advanced_chunks:
                table = advanced_result.get("tables", {}).get(table_name)
                if isinstance(table, pd.DataFrame):
                    advanced_chunks[table_name].append(table)
        _feature_progress(progress_callback, family="time_domain", completed=window_index,
                          total=len(windows), window_id=wid)

    feature_rows.extend(trial_features.to_dict(orient="records"))
    features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    features = features.loc[features["feature_id"].isin(requested_set)].reset_index(drop=True)
    validity = pd.DataFrame(validity_rows, columns=VALIDITY_COLUMNS)
    validity.attrs["event_scr_links"] = pd.DataFrame(links, columns=EVENT_LINK_COLUMNS)
    validity.attrs["extended_tables"] = {
        "trial_table": trial_table,
        "trial_summaries": trial_summaries,
        "trial_fit_diagnostics": trial_fit_diagnostics,
        "spectra": pd.concat(advanced_chunks["spectra"], ignore_index=True) if advanced_chunks["spectra"] else pd.DataFrame(),
        "complexity_diagnostics": pd.concat(advanced_chunks["complexity_diagnostics"], ignore_index=True)
        if advanced_chunks["complexity_diagnostics"] else pd.DataFrame(),
        "ns_classification": ns_table,
        "scr_events_extended": extended_scr,
    }
    validity.attrs["window_policy_id"] = "point_sample_adjacent_interval_clip"
    if bundle.get("method") == "bateman_sparse":
        from .driver_decompose import driver_diagnostics_table
        validity.attrs["extended_tables"]["driver_bursts"] = driver_catalogue
        validity.attrs["extended_tables"]["driver_diagnostics"] = driver_diagnostics_table(bundle)
    features.attrs["feature_policy"] = "sample_based_unless_method_names_time_integration"
    return features, validity


__all__ = ["detect_scr", "build_windows", "extract_window_features", "scr_onset_eligible_mask",
           "list_feature_definitions", "build_trial_table", "summarize_trial_series",
           "SUPPORTED_FEATURE_IDS", "LEGACY_FEATURE_IDS"]
