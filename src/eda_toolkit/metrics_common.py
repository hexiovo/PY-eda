"""Common sample summaries and explicit recording-scoped amplitude transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


TRANSFORMS = {
    "sqrt": ("平方根", "sqrt_uS", "mean(sqrt(A_uS))"),
    "log1p": ("对数ln(1+A/1uS)", "dimensionless", "mean(ln(1+A_uS/1uS))"),
    "recording_z": ("当前记录Z值", "z", "mean((A-reference_mean)/reference_sample_SD)"),
    "recording_max_fraction": ("当前记录最大值校正", "fraction", "mean(A/reference_max)"),
}
REFERENCE_KEYS = ("recording_id", "subject_id", "session_id", "channel")
REFERENCE_SCOPE = "same_recording_subject_session_channel_across_conditions"
DISTRIBUTIONS = {
    "min": ("最小值", "min(x)", 1),
    "max": ("最大值", "max(x)", 1),
    "p05": ("第5百分位", "linear percentile 5", 1),
    "p25": ("第25百分位", "linear percentile 25", 1),
    "p75": ("第75百分位", "linear percentile 75", 1),
    "p95": ("第95百分位", "linear percentile 95", 1),
    "skewness": ("偏度", "adjusted Fisher-Pearson skewness; bias=False", 3),
    "kurtosis": ("超额峰度", "unbiased Fisher excess kurtosis; bias=False", 4),
    "mad": ("中位绝对偏差", "median(abs(x-median(x))); unscaled", 1),
}


def common_definitions():
    definitions = []
    for basis, label, family in (("clean", "clean", "sample"), ("tonic", "tonic", "tonic")):
        for suffix, (title, formula, minimum) in DISTRIBUTIONS.items():
            unit = "dimensionless" if suffix in ("skewness", "kurtosis") else "uS"
            definitions.append((f"{basis}_{suffix}", label + title, family, formula,
                                f"{basis}_uS", unit, minimum))
    for prefix, label, family, basis in (
            ("scr_amplitude", "SCR幅值", "scr_distribution", "qualified SCR amplitude_uS"),
            ("event_amplitude_zero", "事件零包含幅值", "trial_series", "valid zero-inclusive event trials")):
        for transform, (title, unit, formula) in TRANSFORMS.items():
            definitions.append((f"{prefix}_{transform}_mean", label + title + "均值", family,
                                formula, basis, unit, 1))
    for prefix, label, family, minimum_suffix in (
            ("scr_rise_time", "SCR上升时间", "scr_rise", ""),
            ("scr_recovery_time", "SCR配置恢复时间", "scr_recovery", ""),
            ("scr_recovery_50", "SCR恢复50%时间", "scr_recovery", "_s"),
            ("scr_recovery_63", "SCR恢复63%时间", "scr_recovery", "_s")):
        for statistic, title, minimum in (("median", "中位数", 1), ("sd", "标准差", 2)):
            formula = "median(observed durations)" if statistic == "median" else "sample SD(observed durations), ddof=1"
            definitions.append((f"{prefix}_{statistic}{minimum_suffix}", label + title, family,
                                formula, "qualified SCR completed durations", "s", minimum))
    return definitions


COMMON_DEFINITIONS = common_definitions()
COMMON_IDS = frozenset(row[0] for row in COMMON_DEFINITIONS)


def common_parameters(feature_id, recovery_fraction=0.5):
    parameters = {}
    suffix = feature_id.rsplit("_", 1)[-1]
    if suffix in DISTRIBUTIONS:
        if suffix.startswith("p"):
            parameters.update(percentile=int(suffix[1:]), percentile_method="linear")
        elif suffix in ("skewness", "kurtosis"):
            parameters.update(bias=False, constant_policy="unavailable")
            if suffix == "kurtosis":
                parameters["fisher"] = True
        elif suffix == "mad":
            parameters["scale"] = 1.0
    if feature_id.startswith(("scr_amplitude_", "event_amplitude_zero_")):
        parameters.update(transform_before_aggregation=True, amplitude_input_unit="uS",
                          reference_scope=REFERENCE_SCOPE,
                          reference_population=("qualified_onset_eligible_responses" if feature_id.startswith("scr_")
                                                else "all_valid_trials_including_nonresponse_zeros"))
        if "log1p" in feature_id:
            parameters.update(log_base="e", reference_amplitude_uS=1.0)
        if "recording_z" in feature_id:
            parameters.update(reference_ddof=1, minimum_reference_n=2, zero_sd_policy="unavailable")
        if "recording_max_fraction" in feature_id:
            parameters.update(correction="A/max(A); no minimum subtraction", zero_max_policy="unavailable")
    if feature_id.startswith("scr_recovery_"):
        parameters.update(return_fraction=(0.50 if "_50_" in feature_id else 0.63 if "_63_" in feature_id
                                           else float(recovery_fraction)),
                          censored_policy="completed_only; retain_exclusion_reason")
    if "_sd" in feature_id:
        parameters["ddof"] = 1
    return parameters


def distribution_summary(values, statistic):
    """Return value/reason using finite eligible observations supplied by caller."""
    values = np.asarray(values, dtype=float)
    minimum = DISTRIBUTIONS[statistic][2]
    if len(values) < minimum:
        return None, f"requires_at_least_{minimum}_samples"
    if not np.all(np.isfinite(values)):
        return None, "nonfinite_samples"
    if statistic in ("skewness", "kurtosis"):
        # Work on a bounded centered vector to avoid powers of physical amplitudes.
        scale = float(np.max(np.abs(values)))
        normalized = values / scale if scale else values
        centered = normalized - np.mean(normalized)
        spread = float(np.max(np.abs(centered)))
        if spread <= np.finfo(float).eps * 4:
            return None, "constant_or_numerically_constant_samples"
        centered = centered / spread
        value = (stats.skew(centered, bias=False) if statistic == "skewness" else
                 stats.kurtosis(centered, fisher=True, bias=False))
    elif statistic == "mad":
        value = np.median(np.abs(values - np.median(values)))
    elif statistic.startswith("p"):
        value = np.percentile(values, int(statistic[1:]), method="linear")
    else:
        value = np.min(values) if statistic == "min" else np.max(values)
    return (float(value), "") if np.isfinite(value) else (None, "nonfinite_summary")


def amplitude_reference(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values) & (values >= 0)):
        return {"n": 0, "mean_uS": np.nan, "sd_uS": np.nan, "max_uS": np.nan}
    maximum = float(np.max(values))
    scaled = values / maximum if maximum else values
    return {"n": len(values), "mean_uS": float(np.mean(scaled) * maximum),
            "sd_uS": float(np.std(scaled, ddof=1) * maximum) if len(values) >= 2 else np.nan,
            "max_uS": maximum}


def transform_amplitudes(values, transform, reference):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.full(len(values), np.nan), "no_valid_amplitudes"
    if not np.all(np.isfinite(values) & (values >= 0)):
        return np.full(len(values), np.nan), "invalid_or_negative_amplitude"
    if transform == "sqrt":
        return np.sqrt(values), ""
    if transform == "log1p":
        return np.log1p(values), ""
    if transform == "recording_z":
        if reference["n"] < 2:
            return np.full(len(values), np.nan), "requires_two_reference_amplitudes"
        sd = reference["sd_uS"]
        if not np.isfinite(sd) or sd <= 4 * np.finfo(float).eps * reference["max_uS"]:
            return np.full(len(values), np.nan), "constant_reference_amplitudes"
        return (values - reference["mean_uS"]) / sd, ""
    if transform == "recording_max_fraction":
        maximum = reference["max_uS"]
        if not np.isfinite(maximum) or maximum <= 0:
            return np.full(len(values), np.nan), "no_positive_reference_amplitude"
        return values / maximum, ""
    raise ValueError(f"Unknown amplitude transform: {transform}")


def annotate_amplitudes(table, value_column, eligible):
    """Copy a table and annotate per-row transforms and auditable reference values.

    Reference populations are isolated by recording, subject, session and channel.
    Conditions are deliberately not grouping keys. Excluded rows stay missing.
    """
    table = table.copy(deep=True)
    mask = np.asarray(eligible, dtype=bool)
    if mask.shape != (len(table),):
        raise ValueError("amplitude_eligibility_shape")
    values = pd.to_numeric(table.get(value_column, pd.Series(index=table.index, dtype=float)), errors="coerce").to_numpy(float)
    mask = mask & np.isfinite(values) & (values >= 0)
    prefix = value_column.removesuffix("_uS")
    for transform in TRANSFORMS:
        table[f"{prefix}_{transform}"] = np.nan
        table[f"{prefix}_{transform}_status"] = "unavailable"
        table[f"{prefix}_{transform}_reason"] = "excluded_or_invalid_amplitude"
    table["amplitude_reference_scope"] = REFERENCE_SCOPE
    for key in ("n", "mean_uS", "sd_uS", "max_uS"):
        table[f"amplitude_reference_{key}"] = np.nan
    keys = [key for key in REFERENCE_KEYS if key in table]
    groups = table.groupby(keys, dropna=False, sort=False).indices.values() if keys else [np.arange(len(table))]
    for positions in groups:
        positions = np.asarray(positions, dtype=int)
        selected = positions[mask[positions]]
        reference = amplitude_reference(values[selected])
        for key, value in reference.items():
            table.iloc[positions, table.columns.get_loc(f"amplitude_reference_{key}")] = value
        for transform in TRANSFORMS:
            transformed, reason = transform_amplitudes(values[selected], transform, reference)
            table.iloc[selected, table.columns.get_loc(f"{prefix}_{transform}")] = transformed
            table.iloc[selected, table.columns.get_loc(f"{prefix}_{transform}_status")] = "unavailable" if reason else "valid"
            table.iloc[selected, table.columns.get_loc(f"{prefix}_{transform}_reason")] = reason
    return table
