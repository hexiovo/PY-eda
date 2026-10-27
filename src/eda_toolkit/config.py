"""版本化研究配置。默认参数仅作研究示例，尚未按设备/任务科学资格验证。"""
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from .errors import fail

DEFAULT_CONFIG = {
    "schema_version": "1.1", "pipeline_id": "single_standard", "preset_version": "1.1",
    "input": {"manifest": None, "signals": [], "participants": None, "events": None,
              "conditions": None, "group_membership": None, "sync_anchors": None,
              "manual_qc": None, "statistical_design": None, "encoding": "utf-8-sig",
              "delimiter": ",", "missing_tokens": ["", "NA", "NaN"], "base_directory": None},
    "mapping": {"columns": {}, "constants": {}, "eda_unit": None, "sampling_rate_hz": None,
                "time_unit": "s", "clock_domain": "recording_local"},
    "qc": {"min_uS": None, "max_uS": None, "max_slope_uS_s": None, "max_jump_uS": None,
           "flatline_duration_s": 5.0, "flatline_tolerance_uS": 0.0, "gap_factor": 1.5,
           "exclude_auto": True, "annotations": []},
    "preprocess": {"interpolate_max_gap_s": 0.0, "interpolation": "linear", "smoothing": "none",
                   "smoothing_window_s": 0.5, "lowpass_hz": None, "filter_order": 2,
                   "resample_hz": None, "baseline": None, "scaling": "none",
                   "edge_tolerance": 0.001, "min_segment_duration_s": 0.0, "jitter_tolerance": 0.01},
    "decomposition": {"method": "filter_separation", "cutoff_hz": 0.05, "order": 2, "edge_tolerance": 0.001,
                      "driver": {"tau_rise_s": 0.7, "tau_decay_s": 2.0, "tonic_knot_s": 10.0,
                                 "sparsity": 0.01, "ridge": 1e-6, "tonic_smoothness": 10.0,
                                 "kernel_tail_tolerance": 0.001, "max_iterations": 2000,
                                 "gradient_tolerance": 1e-7, "function_tolerance": 1e-12,
                                 "kkt_tolerance": 1e-4, "max_segment_samples": 100000,
                                 "burst_threshold_uS_s": 0.01}},
    "scr": {"method": "trough_to_peak", "signal_basis": "phasic", "threshold_type": "absolute",
            "amplitude_min_uS": 0.05, "relative_fraction": 0.1, "reference_scope": "recording",
            "min_rise_s": 0.0, "max_rise_s": 10.0, "min_interval_s": 1.0, "recovery_fraction_returned": 0.5,
            "onset_left_context_s": 0.0, "onset_right_context_s": None},
    "windows": {"whole_record": True, "fixed_length_s": 30.0, "fixed_step_s": 30.0, "tail_policy": "drop",
                "event_baseline_s": [-2.0, 0.0], "event_response_s": [1.0, 4.0],
                "event_peak_s": [1.0, 10.0], "event_recovery_s": [1.0, 20.0], "min_coverage": 0.8},
    "features": {"ids": [], "exclude_interpolated": True,
                  "extended": {"enabled": True, "baseline_min_uS": 1e-6,
                               "trial_series": {"series_by": ["condition", "event_type"],
                                                "min_slope_trials": 5, "min_logit_trials": 8,
                                                "min_decay_trials": 8, "decay_max_rate": 10.0,
                                                "decay_max_nfev": 1000},
                               "nonspecific": {"enabled": True, "annotation_coverage": []},
                               "spectral": {"enabled": True, "basis": "clean", "segment_s": 256.0,
                                            "overlap": 0.5, "window": "hann", "detrend": "constant",
                                            "min_segments": 2, "min_cycles": 2.0,
                                            "bands": [{"id": "eda_045_025", "low_hz": 0.045, "high_hz": 0.25}],
                                            "normalization_band_hz": [0.01, 0.5]},
                               "autocorrelation": {"enabled": True, "lags_s": [1.0, 5.0, 10.0],
                                                   "min_pairs": 20},
                               "complexity": {"enabled": True, "basis": "clean", "min_samples": 200,
                                              "max_samples": 5000, "max_pair_comparisons": 25000000,
                                              "sampen": {"m": 2, "delay": 1, "r_mode": "sd_fraction", "r": 0.2},
                                              "permen": {"dimension": 3, "delay": 1, "normalize": True,
                                                         "ties": "stable"},
                                              "dfa": {"order": 1, "min_samples": 256, "min_scale": 4,
                                                      "max_scale_fraction": 0.25, "scale_count": 10,
                                                      "min_scales": 6}},
                               "group_temporal": {"enabled": True, "correlation_threshold": 0.1,
                                                  "lag_stability_tolerance_s": 1.0,
                                                  "min_valid_windows": 3}}},
    "synchrony": {"enabled": False, "basis": "phasic", "metric": "pearson", "window_s": 30.0,
                  "step_s": 30.0, "max_lag_s": 5.0, "lag_step_s": 1.0, "min_coverage": 0.8,
                  "min_samples": 8, "analysis_rate_hz": 4.0, "timing_tolerance_s": 0.1,
                  "allow_provisional": False, "shared_clock_verified": False, "standardize": True, "null_method": "circular_shift",
                  "null_replicates": 99, "min_shift_s": 10.0, "max_attempts": 10000,
                  "tail": "two-sided", "seed": 2026, "pseudodyad_strata": [], "matching_fields": [],
                  "analysis_intervals_s": None, "analysis_context_s": 0.0, "pseudodyad_exchangeability": False},
    "statistics": {"enabled": False, "designs": [], "alpha": 0.05, "correction": "holm",
                   "missing_policy": "complete_case", "planned_hypotheses": []},
    "compute": {"backend": "cpu", "workers": 1, "seed": 2026, "memory_limit_mb": 2048,
                "max_pairs": 10000, "max_windows": 10000, "max_null_evaluations": 10000000},
    "output": {"directory": "results", "overwrite": False, "formats": ["csv", "json", "html", "png", "xlsx"],
               "spreadsheet_safe": True, "redact_paths": True, "save_processed_signals": True},
    "workspace": {"project_directory": None, "autosave_enabled": True,
                  "autosave_debounce_ms": 750, "restore_prompt": True,
                  "queue_autostart": False, "queue_parallel_tasks": 1},
    "runtime": {"log_level": "INFO", "log_max_bytes": 10485760,
                "log_backup_count": 5, "eta_min_completed_units": 3,
                "event_schema_version": "1.0"},
}

# Schema 1.0 remains readable through an explicit in-memory migration. These defaults
# describe the released v0.1 configuration and must not inherit later additions.
LEGACY_DEFAULT_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
LEGACY_DEFAULT_CONFIG["schema_version"] = "1.0"
LEGACY_DEFAULT_CONFIG["preset_version"] = "1.0"
LEGACY_DEFAULT_CONFIG.pop("workspace")
LEGACY_DEFAULT_CONFIG.pop("runtime")
LEGACY_DEFAULT_CONFIG["features"].pop("extended")
LEGACY_DEFAULT_CONFIG["decomposition"].pop("driver")
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

PRESETS = ("single_standard", "single_event", "single_advanced", "single_driver", "group_sync", "custom")
DESIGN_KEYS = {"design_id", "outcome", "design_type", "independent_unit", "condition_column", "conditions",
               "group_column", "groups", "subject_column", "session_column", "group_id_column", "reference",
               "covariates", "random_intercept", "correction_family", "alternative", "aggregate", "predictor", "contrast"}
ANNOTATION_KEYS = {"annotation_id", "recording_id", "channel", "start_s", "end_s", "action", "reason",
                   "operator", "timestamp", "revision", "parent_revision", "clock_domain", "target_annotation_id"}
SIGNAL_KEYS = {"file_path", "recording_id", "subject_id", "session_id", "sampling_rate_hz", "eda_unit",
               "mapping_profile", "channel", "clock_domain"}


def _reject_constant(token):
    fail("nonstandard_json", f"JSON 不允许 {token}。", category="configuration")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("duplicate_json_key", f"JSON 字段重复：{key}", category="configuration", field=key)
        result[key] = value
    return result


def load_json(path):
    """读取严格JSON；拒绝重复键与Python默认允许的非标准数值。"""
    return parse_json_text(Path(path).read_text(encoding="utf-8-sig"))


def parse_json_text(text):
    """文件导入、界面编辑和命令行共用的严格JSON文本解析器。"""
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def canonical_json(value):
    """稳定序列化用于缓存哈希；数值结果中的缺失值由导出边界转换。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(config):
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def feature_work_method(config):
    """Return an ETA history key bound to feature selection and its parameters."""
    if not isinstance(config, dict):
        config = {}
    payload = {name: copy.deepcopy(config.get(name, {}))
               for name in ("features", "windows", "scr", "qc", "preprocess", "decomposition")}
    selected = payload.get("features", {}).get("ids")
    if isinstance(selected, list):
        payload["features"]["ids"] = sorted(selected)
    return "feature_extraction:v1:" + config_hash(payload)


def synchrony_work_method(config):
    """Return an ETA history key bound to group-sync and temporal feature settings."""
    if not isinstance(config, dict):
        config = {}
    selected = config.get("features", {}).get("ids", [])
    if isinstance(selected, list):
        selected = sorted(selected)
    payload = {
        "synchrony": copy.deepcopy(config.get("synchrony", {})),
        "group_temporal": copy.deepcopy(config.get("features", {}).get("extended", {}).get("group_temporal", {})),
        "feature_ids": selected,
    }
    return "group_synchrony:v1:" + config_hash(payload)


def _merge(target, incoming, location=""):
    if not isinstance(incoming, dict):
        fail("config_type", f"{location or 'config'} 必须是 JSON 对象。", category="configuration")
    for key, value in incoming.items():
        field = f"{location}.{key}".strip(".")
        if key not in target:
            fail("unknown_field", f"未知配置字段：{field}", category="configuration", field=field)
        if field in ("mapping.columns", "mapping.constants"):
            if not isinstance(value, dict):
                fail("config_type", f"{field} 必须是对象。", category="configuration", field=field)
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict):
            _merge(target[key], value, field)
        else:
            target[key] = copy.deepcopy(value)


def _number(value, field, *, minimum=None, maximum=None, integer=False, nullable=False):
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail("config_number", f"{field} 必须是有限数值。", category="configuration", field=field)
    if integer and not isinstance(value, int):
        fail("config_integer", f"{field} 必须是整数。", category="configuration", field=field)
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        fail("config_range", f"{field} 超出允许范围 [{minimum}, {maximum}]。", category="configuration", field=field)


def _choice(value, allowed, field):
    if value not in allowed:
        fail("config_choice", f"{field} 必须为 {list(allowed)} 之一。", category="configuration", field=field)


def _object_list(items, allowed, field):
    if not isinstance(items, list):
        fail("config_list", f"{field} 必须是数组。", category="configuration", field=field)
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) - allowed:
            fail("unknown_field", f"{field}[{index}] 必须是符合契约的对象，不能含 formula 或未知字段。", category="configuration", field=field)


def _check_boolean(value, field):
    if not isinstance(value, bool):
        fail("config_boolean", f"{field} 必须是布尔值。", category="configuration", field=field)


def _check_positive_number(value, field, *, minimum=None, maximum=None, integer=False):
    _number(value, field, minimum=minimum, maximum=maximum, integer=integer)


def _check_feature_identifier(value, field):
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
        fail("feature_identifier", f"{field} 必须匹配 [a-z][a-z0-9_]*。", category="configuration", field=field)


def _validate_extended_features(config):
    extended = config["features"]["extended"]
    _check_boolean(extended["enabled"], "features.extended.enabled")
    _number(extended["baseline_min_uS"], "features.extended.baseline_min_uS", minimum=1e-12)

    trial = extended["trial_series"]
    allowed_series = {"condition", "event_type", "stimulus_id"}
    if not isinstance(trial["series_by"], list) or not trial["series_by"] or not all(
            isinstance(item, str) and item in allowed_series for item in trial["series_by"]):
        fail("feature_series_by", "trial_series.series_by 只能含 condition、event_type、stimulus_id 且不能为空。",
             category="configuration", field="features.extended.trial_series.series_by")
    if len(set(trial["series_by"])) != len(trial["series_by"]):
        fail("feature_series_by", "trial_series.series_by 不能重复。", category="configuration",
             field="features.extended.trial_series.series_by")
    for key in ("min_slope_trials", "min_logit_trials", "min_decay_trials"):
        _number(trial[key], f"features.extended.trial_series.{key}", minimum=1, maximum=1_000_000, integer=True)
    _number(trial["decay_max_rate"], "features.extended.trial_series.decay_max_rate", minimum=1e-12)
    _number(trial["decay_max_nfev"], "features.extended.trial_series.decay_max_nfev", minimum=1,
            maximum=1_000_000, integer=True)

    nonspecific = extended["nonspecific"]
    _check_boolean(nonspecific["enabled"], "features.extended.nonspecific.enabled")
    coverage = nonspecific["annotation_coverage"]
    if not isinstance(coverage, list):
        fail("annotation_coverage", "annotation_coverage 必须是数组。", category="configuration",
             field="features.extended.nonspecific.annotation_coverage")
    seen_coverage = set()
    coverage_fields = {"recording_id", "channel", "start_s", "end_s", "clock_domain", "complete"}
    for index, row in enumerate(coverage):
        field = f"features.extended.nonspecific.annotation_coverage[{index}]"
        if not isinstance(row, dict) or set(row) != coverage_fields:
            fail("annotation_coverage", f"{field} 必须包含且仅包含 {sorted(coverage_fields)}。",
                 category="configuration", field=field)
        for key in ("recording_id", "channel", "clock_domain"):
            if not isinstance(row[key], str) or not row[key].strip():
                fail("annotation_coverage", f"{field}.{key} 必须是非空字符串。", category="configuration",
                     field=f"{field}.{key}")
        _number(row["start_s"], f"{field}.start_s")
        _number(row["end_s"], f"{field}.end_s")
        if row["start_s"] >= row["end_s"]:
            fail("annotation_coverage", f"{field} 必须是非空半开时间区间。", category="configuration",
                 field=field)
        if row["complete"] is not True:
            fail("annotation_coverage", f"{field}.complete 必须为 true。", category="configuration",
                 field=f"{field}.complete")
        signature = (row["recording_id"], row["channel"], row["start_s"], row["end_s"], row["clock_domain"])
        if signature in seen_coverage:
            fail("annotation_coverage", f"{field} 与已有声明重复。", category="configuration", field=field)
        seen_coverage.add(signature)

    spectral = extended["spectral"]
    _check_boolean(spectral["enabled"], "features.extended.spectral.enabled")
    _choice(spectral["basis"], ("clean",), "features.extended.spectral.basis")
    _number(spectral["segment_s"], "features.extended.spectral.segment_s", minimum=1e-12)
    _number(spectral["overlap"], "features.extended.spectral.overlap", minimum=0, maximum=0.999999999)
    _choice(spectral["window"], ("hann", "boxcar"), "features.extended.spectral.window")
    _choice(spectral["detrend"], ("constant", "linear"), "features.extended.spectral.detrend")
    for key in ("min_segments",):
        _number(spectral[key], f"features.extended.spectral.{key}", minimum=1, maximum=100_000, integer=True)
    _number(spectral["min_cycles"], "features.extended.spectral.min_cycles", minimum=1e-12)
    bands = spectral["bands"]
    if not isinstance(bands, list) or not bands:
        fail("spectral_bands", "spectral.bands 必须是非空数组。", category="configuration",
             field="features.extended.spectral.bands")
    seen_bands = set()
    for index, band in enumerate(bands):
        field = f"features.extended.spectral.bands[{index}]"
        if not isinstance(band, dict) or set(band) != {"id", "low_hz", "high_hz"}:
            fail("spectral_band", f"{field} 必须仅含 id、low_hz、high_hz。", category="configuration", field=field)
        _check_feature_identifier(band["id"], f"{field}.id")
        if band["id"] in seen_bands:
            fail("spectral_band", f"{field}.id 不能重复。", category="configuration", field=f"{field}.id")
        seen_bands.add(band["id"])
        _number(band["low_hz"], f"{field}.low_hz", minimum=1e-12)
        _number(band["high_hz"], f"{field}.high_hz", minimum=1e-12)
        if band["high_hz"] <= band["low_hz"]:
            fail("spectral_band", f"{field}.high_hz 必须大于 low_hz。", category="configuration", field=field)
    normalization_band = spectral["normalization_band_hz"]
    if not isinstance(normalization_band, list) or len(normalization_band) != 2:
        fail("normalization_band", "normalization_band_hz 必须是两个端点。", category="configuration",
             field="features.extended.spectral.normalization_band_hz")
    for index, endpoint in enumerate(normalization_band):
        _number(endpoint, f"features.extended.spectral.normalization_band_hz[{index}]", minimum=1e-12)
    if normalization_band[1] <= normalization_band[0]:
        fail("normalization_band", "normalization_band_hz 终点必须大于起点。", category="configuration",
             field="features.extended.spectral.normalization_band_hz")

    autocorrelation = extended["autocorrelation"]
    _check_boolean(autocorrelation["enabled"], "features.extended.autocorrelation.enabled")
    lags = autocorrelation["lags_s"]
    if not isinstance(lags, list) or not lags:
        fail("autocorrelation_lags", "autocorrelation.lags_s 必须是非空数组。", category="configuration",
             field="features.extended.autocorrelation.lags_s")
    for index, lag in enumerate(lags):
        _number(lag, f"features.extended.autocorrelation.lags_s[{index}]", minimum=1e-12)
        if index and lag <= lags[index - 1]:
            fail("autocorrelation_lags", "lags_s 必须严格递增且不重复。", category="configuration",
                 field="features.extended.autocorrelation.lags_s")
    _number(autocorrelation["min_pairs"], "features.extended.autocorrelation.min_pairs", minimum=1,
            maximum=1_000_000, integer=True)

    complexity = extended["complexity"]
    _check_boolean(complexity["enabled"], "features.extended.complexity.enabled")
    _choice(complexity["basis"], ("clean",), "features.extended.complexity.basis")
    for key in ("min_samples", "max_samples", "max_pair_comparisons"):
        _number(complexity[key], f"features.extended.complexity.{key}", minimum=1,
                maximum=1_000_000_000, integer=True)
    if complexity["max_samples"] < complexity["min_samples"]:
        fail("complexity_range", "complexity.max_samples 不能小于 min_samples。", category="configuration",
             field="features.extended.complexity.max_samples")
    sampen = complexity["sampen"]
    _number(sampen["m"], "features.extended.complexity.sampen.m", minimum=2, maximum=5, integer=True)
    _number(sampen["delay"], "features.extended.complexity.sampen.delay", minimum=1, maximum=1000, integer=True)
    _choice(sampen["r_mode"], ("sd_fraction", "absolute"), "features.extended.complexity.sampen.r_mode")
    _number(sampen["r"], "features.extended.complexity.sampen.r", minimum=1e-12)
    permen = complexity["permen"]
    _number(permen["dimension"], "features.extended.complexity.permen.dimension", minimum=2, maximum=7, integer=True)
    _number(permen["delay"], "features.extended.complexity.permen.delay", minimum=1, maximum=1000, integer=True)
    _check_boolean(permen["normalize"], "features.extended.complexity.permen.normalize")
    _choice(permen["ties"], ("stable", "reject"), "features.extended.complexity.permen.ties")
    dfa = complexity["dfa"]
    _number(dfa["order"], "features.extended.complexity.dfa.order", minimum=1, maximum=1, integer=True)
    _number(dfa["min_samples"], "features.extended.complexity.dfa.min_samples", minimum=1,
            maximum=1_000_000, integer=True)
    _number(dfa["min_scale"], "features.extended.complexity.dfa.min_scale", minimum=2,
            maximum=1_000_000, integer=True)
    _number(dfa["max_scale_fraction"], "features.extended.complexity.dfa.max_scale_fraction", minimum=1e-12, maximum=1)
    _number(dfa["scale_count"], "features.extended.complexity.dfa.scale_count", minimum=2,
            maximum=1_000_000, integer=True)
    _number(dfa["min_scales"], "features.extended.complexity.dfa.min_scales", minimum=2,
            maximum=1_000_000, integer=True)
    if dfa["min_scales"] > dfa["scale_count"]:
        fail("dfa_scale_count", "dfa.min_scales 不能大于 scale_count。", category="configuration",
             field="features.extended.complexity.dfa.min_scales")

    group_temporal = extended["group_temporal"]
    _check_boolean(group_temporal["enabled"], "features.extended.group_temporal.enabled")
    _number(group_temporal["correlation_threshold"], "features.extended.group_temporal.correlation_threshold",
            minimum=-1, maximum=1)
    _number(group_temporal["lag_stability_tolerance_s"],
            "features.extended.group_temporal.lag_stability_tolerance_s", minimum=0)
    _number(group_temporal["min_valid_windows"], "features.extended.group_temporal.min_valid_windows",
            minimum=1, maximum=1_000_000, integer=True)


def _validate_runtime_workspace(config):
    workspace = config["workspace"]
    runtime = config["runtime"]
    if workspace["project_directory"] is not None and not isinstance(workspace["project_directory"], str):
        fail("workspace_project_directory", "workspace.project_directory 必须是路径字符串或 null。",
             category="configuration", field="workspace.project_directory")
    for key in ("autosave_enabled", "restore_prompt"):
        _check_boolean(workspace[key], f"workspace.{key}")
    _number(workspace["autosave_debounce_ms"], "workspace.autosave_debounce_ms", minimum=100,
            maximum=10000, integer=True)
    if workspace["queue_autostart"] is not False:
        fail("queue_autostart", "workspace.queue_autostart 固定为 false。", category="configuration",
             field="workspace.queue_autostart")
    _number(workspace["queue_parallel_tasks"], "workspace.queue_parallel_tasks", minimum=1,
            maximum=1, integer=True)
    _choice(runtime["log_level"], ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), "runtime.log_level")
    _number(runtime["log_max_bytes"], "runtime.log_max_bytes", minimum=1024, integer=True)
    _number(runtime["log_backup_count"], "runtime.log_backup_count", minimum=1, maximum=100, integer=True)
    _number(runtime["eta_min_completed_units"], "runtime.eta_min_completed_units", minimum=3,
            maximum=1000000, integer=True)
    if runtime["event_schema_version"] != "1.0":
        fail("event_schema_version", "runtime.event_schema_version 固定为 1.0。", category="configuration",
             field="runtime.event_schema_version")


def validate_config(config):
    """校验完整配置的类型、单位和资源边界，返回同一配置。"""
    if not isinstance(config, dict):
        fail("config_type", "config 必须是对象。", category="configuration")
    if "schema_version" not in config or "preset_version" not in config:
        fail("config_version", "配置必须声明 schema_version 和 preset_version。", category="configuration")
    schema_version = config["schema_version"]
    _choice(schema_version, ("1.0", "1.1"), "schema_version")
    _choice(config["pipeline_id"], PRESETS, "pipeline_id")
    expected_preset_version = "1.0" if schema_version == "1.0" else "1.1"
    if config["preset_version"] != expected_preset_version:
        fail("preset_version", f"schema {schema_version} 仅支持 preset_version {expected_preset_version}。",
             category="configuration")
    schema_defaults = LEGACY_DEFAULT_CONFIG if schema_version == "1.0" else DEFAULT_CONFIG
    # 同时拒绝用 true 冒充整数，或把字符串当作布尔开关。
    for section, defaults in schema_defaults.items():
        if not isinstance(defaults, dict):
            continue
        if section not in config or not isinstance(config[section], dict):
            fail("config_type", f"{section} 必须是对象。", category="configuration", field=section)
        for key, default in defaults.items():
            if key not in config[section]:
                fail("config_missing_field", f"缺少必需配置字段：{section}.{key}", category="configuration",
                     field=f"{section}.{key}")
            value = config[section][key]
            field = f"{section}.{key}"
            if isinstance(default, bool) and not isinstance(value, bool):
                fail("config_boolean", f"{field} 必须是布尔值。", category="configuration", field=field)
            if isinstance(default, str) and not isinstance(value, str):
                fail("config_string", f"{field} 必须是字符串。", category="configuration", field=field)
            if isinstance(default, list) and not isinstance(value, list):
                fail("config_list", f"{field} 必须是数组。", category="configuration", field=field)
            if isinstance(default, (int, float)) and not isinstance(default, bool):
                _number(value, field, integer=isinstance(default, int), nullable=field in ("windows.fixed_length_s", "windows.fixed_step_s"))
    for key in ("manifest", "participants", "events", "conditions", "group_membership", "sync_anchors", "manual_qc", "statistical_design", "base_directory"):
        value = config["input"][key]
        if value is not None and not isinstance(value, str):
            fail("input_path", f"input.{key} 必须是路径字符串或 null。", category="configuration")
    if len(config["input"]["delimiter"]) != 1:
        fail("delimiter", "CSV分隔符必须恰好一个字符。", category="configuration")
    for item in config["input"]["signals"]:
        if not isinstance(item, (str, dict)) or isinstance(item, dict) and set(item) - SIGNAL_KEYS:
            fail("signal_input", "signals 只能包含路径字符串或规定的清单对象。", category="configuration")
    if not all(isinstance(x, str) for x in config["input"]["missing_tokens"]):
        fail("missing_tokens", "missing_tokens 只能含字符串。", category="configuration")
    _choice(config["mapping"]["time_unit"], ("s", "ms", "us"), "mapping.time_unit")
    _choice(config["mapping"]["eda_unit"], (None, "uS", "µS", "μS", "mS", "S", "ohm", "kohm", "Mohm"), "mapping.eda_unit")
    _number(config["mapping"]["sampling_rate_hz"], "mapping.sampling_rate_hz", minimum=1e-12, nullable=True)
    for key in ("min_uS", "max_uS", "max_slope_uS_s", "max_jump_uS"):
        _number(config["qc"][key], f"qc.{key}", nullable=True)
    q = config["qc"]
    if q["min_uS"] is not None and q["max_uS"] is not None and q["min_uS"] >= q["max_uS"]:
        fail("qc_range", "min_uS 必须小于 max_uS。", category="configuration")
    for key in ("flatline_duration_s", "flatline_tolerance_uS"):
        _number(q[key], f"qc.{key}", minimum=0)
    _number(q["gap_factor"], "qc.gap_factor", minimum=1.000001)
    _object_list(q["annotations"], ANNOTATION_KEYS, "qc.annotations")
    p = config["preprocess"]
    _choice(p["interpolation"], ("linear", "pchip"), "preprocess.interpolation")
    _choice(p["smoothing"], ("none", "moving", "median"), "preprocess.smoothing")
    _choice(p["scaling"], ("none", "zscore", "range", "log"), "preprocess.scaling")
    for key in ("interpolate_max_gap_s", "smoothing_window_s", "min_segment_duration_s"):
        _number(p[key], f"preprocess.{key}", minimum=0)
    _number(p["jitter_tolerance"], "preprocess.jitter_tolerance", minimum=0, maximum=0.5)
    for key in ("lowpass_hz", "resample_hz"):
        _number(p[key], f"preprocess.{key}", minimum=1e-12, nullable=True)
    if p["baseline"] is not None:
        if not isinstance(p["baseline"], list) or len(p["baseline"]) != 2:
            fail("baseline_window", "baseline 必须是 [start_s,end_s] 或 null。", category="configuration")
        for value in p["baseline"]:
            _number(value, "preprocess.baseline")
        if p["baseline"][0] >= p["baseline"][1]:
            fail("baseline_window", "基线区间终点必须大于起点。", category="configuration")
    for section, key in (("preprocess", "filter_order"), ("decomposition", "order")):
        _number(config[section][key], f"{section}.{key}", minimum=1, maximum=12, integer=True)
    for section in ("preprocess", "decomposition"):
        _number(config[section]["edge_tolerance"], f"{section}.edge_tolerance", minimum=1e-12, maximum=0.1)
    _choice(config["decomposition"]["method"], ("filter_separation", "bateman_sparse", "cvxeda", "ledalab"), "decomposition.method")
    _number(config["decomposition"]["cutoff_hz"], "decomposition.cutoff_hz", minimum=1e-12)
    driver = config["decomposition"].get("driver", DEFAULT_CONFIG["decomposition"]["driver"])
    for key in ("tau_rise_s", "tau_decay_s", "tonic_knot_s", "ridge", "gradient_tolerance",
                "function_tolerance", "kkt_tolerance", "burst_threshold_uS_s"):
        _number(driver[key], f"decomposition.driver.{key}", minimum=1e-12)
    for key in ("sparsity", "tonic_smoothness"):
        _number(driver[key], f"decomposition.driver.{key}", minimum=0)
    _number(driver["kernel_tail_tolerance"], "decomposition.driver.kernel_tail_tolerance", minimum=1e-8, maximum=0.01)
    _number(driver["max_iterations"], "decomposition.driver.max_iterations", minimum=1, maximum=100000, integer=True)
    _number(driver["max_segment_samples"], "decomposition.driver.max_segment_samples", minimum=10, maximum=1000000, integer=True)
    if driver["tau_decay_s"] <= driver["tau_rise_s"] * 1.01:
        fail("driver_time_constants", "driver 的衰减时间必须大于上升时间的 1.01 倍。", category="configuration")
    if driver["tonic_knot_s"] < driver["tau_decay_s"]:
        fail("driver_tonic_timescale", "driver 的 tonic 节点间隔不能短于衰减时间。", category="configuration")
    s = config["scr"]
    _choice(s["method"], ("trough_to_peak",), "scr.method")
    _choice(s["signal_basis"], ("phasic", "clean"), "scr.signal_basis")
    _choice(s["threshold_type"], ("absolute", "relative"), "scr.threshold_type")
    _choice(s["reference_scope"], ("recording",), "scr.reference_scope")
    for key in ("relative_fraction", "recovery_fraction_returned"):
        _number(s[key], f"scr.{key}", minimum=1e-12, maximum=1)
    for key in ("amplitude_min_uS", "min_rise_s", "max_rise_s", "min_interval_s"):
        _number(s[key], f"scr.{key}", minimum=0)
    if s["max_rise_s"] < s["min_rise_s"]:
        fail("rise_range", "SCR 最大上升时间不能小于最小时间。", category="configuration")
    _number(s["onset_left_context_s"], "scr.onset_left_context_s", minimum=0)
    _number(s["onset_right_context_s"], "scr.onset_right_context_s", minimum=0, nullable=True)
    if s["onset_right_context_s"] is None:
        s["onset_right_context_s"] = s["max_rise_s"]
    if s["onset_right_context_s"] < s["max_rise_s"]:
        fail("onset_context", "SCR起点右上下文不能短于最大允许上升时间。", category="configuration")
    w = config["windows"]
    for key in ("fixed_length_s", "fixed_step_s"):
        _number(w[key], f"windows.{key}", minimum=1e-12, nullable=True)
    _choice(w["tail_policy"], ("drop", "include"), "windows.tail_policy")
    for key in ("event_baseline_s", "event_response_s", "event_peak_s", "event_recovery_s"):
        value = w[key]
        if len(value) != 2:
            fail("window_range", f"windows.{key} 必须有两个端点。", category="configuration")
        for endpoint in value:
            _number(endpoint, f"windows.{key}")
        if value[0] >= value[1]:
            fail("window_range", f"windows.{key} 必须起点小于终点。", category="configuration")
    for section in ("windows", "synchrony"):
        _number(config[section]["min_coverage"], f"{section}.min_coverage", minimum=0, maximum=1)
    if schema_version == "1.1":
        _validate_runtime_workspace(config)
        _validate_extended_features(config)
    selected_features = config["features"]["ids"]
    if not all(isinstance(item, str) for item in selected_features):
        fail("feature_ids", "features.ids 只能包含特征编号字符串。", category="configuration")
    if len(set(selected_features)) != len(selected_features):
        fail("feature_ids", "features.ids 不能包含重复编号。", category="configuration", field="features.ids")
    if selected_features:
        from . import core_features
        from .core_features import SUPPORTED_FEATURE_IDS
        if schema_version == "1.0":
            supported_feature_ids = set(LEGACY_FEATURE_IDS)
        else:
            supported_feature_ids = set(SUPPORTED_FEATURE_IDS)
            registry = getattr(core_features, "list_feature_definitions", None)
            if callable(registry):
                resolved_registry = registry(config)
                if isinstance(resolved_registry, dict):
                    supported_feature_ids = set(resolved_registry)
                elif isinstance(resolved_registry, (list, tuple, set)):
                    supported_feature_ids = set(resolved_registry)
        unknown = sorted(set(selected_features) - supported_feature_ids)
        if unknown:
            fail("unknown_features", f"未知特征编号：{unknown}", category="configuration", field="features.ids")
    sync = config["synchrony"]
    _choice(sync["metric"], ("pearson", "spearman"), "synchrony.metric")
    _choice(sync["basis"], ("clean", "tonic", "phasic"), "synchrony.basis")
    _choice(sync["null_method"], ("circular_shift", "pseudodyad", "none"), "synchrony.null_method")
    _choice(sync["tail"], ("two-sided", "greater", "less"), "synchrony.tail")
    for key in ("window_s", "step_s", "lag_step_s", "analysis_rate_hz"):
        _number(sync[key], f"synchrony.{key}", minimum=1e-12)
    for key in ("max_lag_s", "timing_tolerance_s", "min_shift_s"):
        _number(sync[key], f"synchrony.{key}", minimum=0)
    _number(sync["analysis_context_s"], "synchrony.analysis_context_s", minimum=0)
    if sync["analysis_intervals_s"] is not None:
        if not isinstance(sync["analysis_intervals_s"], list) or not sync["analysis_intervals_s"]:
            fail("analysis_intervals", "分析区间必须是非空的 [start,end] 数组或null。", category="configuration")
        previous_end = None
        for interval in sync["analysis_intervals_s"]:
            if not isinstance(interval, list) or len(interval) != 2:
                fail("analysis_intervals", "每个分析区间必须有两个有限数值端点。", category="configuration")
            for endpoint in interval:
                _number(endpoint, "synchrony.analysis_intervals_s")
            if interval[0] >= interval[1] or previous_end is not None and interval[0] < previous_end:
                fail("analysis_intervals", "分析区间必须按时间排序、互不重叠且起点小于终点。", category="configuration")
            previous_end = interval[1]
    for key, minimum in (("min_samples", 3), ("null_replicates", 0), ("max_attempts", 1)):
        _number(sync[key], f"synchrony.{key}", minimum=minimum, integer=True)
    stat = config["statistics"]
    _choice(stat["correction"], ("holm", "fdr_bh", "fdr_by", "none"), "statistics.correction")
    _choice(stat["missing_policy"], ("complete_case", "fail"), "statistics.missing_policy")
    _number(stat["alpha"], "statistics.alpha", minimum=1e-12, maximum=0.5)
    _object_list(stat["designs"], DESIGN_KEYS, "statistics.designs")
    for design in stat["designs"]:
        _choice(design.get("design_type", "descriptive"), ("descriptive", "one_sample", "paired", "welch", "correlation", "mixed_effects", "wilcoxon", "mannwhitney"), "design.design_type")
        for key in ("conditions", "groups", "covariates"):
            if key in design and (not isinstance(design[key], list) or not all(isinstance(item, str) for item in design[key])):
                fail("design_list", f"design.{key} 必须是字符串数组。", category="configuration")
        for key in ("design_id", "outcome", "independent_unit", "condition_column", "group_column", "subject_column", "session_column", "group_id_column", "correction_family", "alternative", "aggregate", "predictor"):
            if key in design and not isinstance(design[key], str):
                fail("design_string", f"design.{key} 必须是字符串。", category="configuration")
        if "reference" in design:
            _number(design["reference"], "design.reference")
        if "contrast" in design:
            contrast = design["contrast"]
            if not isinstance(contrast, dict) or set(contrast) - {"conditions", "groups", "condition_column", "group_column", "reference", "predictor"}:
                fail("structured_contrast", "contrast 必须是声明条件/组/参照的结构化对象，不能是公式。", category="configuration")
        if "random_intercept" in design and not isinstance(design["random_intercept"], bool):
            fail("design_boolean", "design.random_intercept 必须是布尔值。", category="configuration")
    for hypothesis in stat["planned_hypotheses"]:
        if not isinstance(hypothesis, str) and (not isinstance(hypothesis, dict) or set(hypothesis) - {"hypothesis_id", "correction_family"} or not isinstance(hypothesis.get("hypothesis_id"), str)):
            fail("planned_hypotheses", "计划假设必须是编号字符串或含 hypothesis_id/correction_family 的对象。", category="configuration")
    _choice(config["compute"]["backend"], ("cpu", "gpu"), "compute.backend")
    for key in ("workers", "memory_limit_mb", "max_pairs", "max_windows", "max_null_evaluations"):
        _number(config["compute"][key], f"compute.{key}", minimum=1, integer=True)
    if config["compute"]["workers"] > 32:
        fail("worker_limit", "workers 最多 32；需同时考虑内存与本机核心数。", category="configuration")
    if config["output"]["overwrite"]:
        fail("unsafe_overwrite", "运行结果使用独立目录，不支持覆盖已有运行。", category="configuration")
    if not config["output"]["directory"].strip():
        fail("output_path", "输出目录不能为空。", category="configuration")
    if not set(config["output"]["formats"]) <= {"csv", "json", "html", "png", "xlsx", "pdf"}:
        fail("output_format", "未知输出格式。", category="configuration")
    return config


def _preset_from_defaults(name, defaults):
    _choice(name, PRESETS, "pipeline_id")
    config = copy.deepcopy(defaults)
    config["pipeline_id"] = name
    if name == "single_event":
        config["windows"]["fixed_length_s"] = None
    if name == "single_driver":
        config["decomposition"]["method"] = "bateman_sparse"
    if name == "single_advanced":
        config["qc"]["exclude_auto"] = True
        config["windows"]["min_coverage"] = 0.95
    if name == "group_sync":
        config["synchrony"]["enabled"] = True
        config["synchrony"]["analysis_context_s"] = 60.0
    return config


def get_preset(name="single_standard"):
    """返回schema 1.1预设；扩展指标开启但不改动旧数值方法。"""
    return _preset_from_defaults(name, DEFAULT_CONFIG)


def migrate_config(config):
    """把schema 1.0配置迁移到内存中的1.1对象，不写回来源文件。"""
    if isinstance(config, (str, Path)):
        config = load_json(config)
    if not isinstance(config, dict):
        fail("config_type", "config 必须是 JSON 对象。", category="configuration")
    version = config.get("schema_version", "1.0" if config.get("preset_version") == "1.0" else "1.1")
    if version == "1.1":
        resolved = _preset_from_defaults(config.get("pipeline_id", "single_standard"), DEFAULT_CONFIG)
        _merge(resolved, config)
        validate_config(resolved)
        canonical_json(resolved)
        return resolved, []
    if version != "1.0":
        fail("schema_version", "仅支持从schema 1.0迁移或读取schema 1.1。", category="configuration")

    legacy = _preset_from_defaults(config.get("pipeline_id", "single_standard"), LEGACY_DEFAULT_CONFIG)
    _merge(legacy, config)
    validate_config(legacy)
    canonical_json(legacy)
    migrated = copy.deepcopy(legacy)
    migrated["schema_version"] = "1.1"
    migrated["preset_version"] = "1.1"
    migrated["workspace"] = copy.deepcopy(DEFAULT_CONFIG["workspace"])
    migrated["runtime"] = copy.deepcopy(DEFAULT_CONFIG["runtime"])
    migrated["features"]["extended"] = copy.deepcopy(DEFAULT_CONFIG["features"]["extended"])
    migrated["features"]["extended"]["enabled"] = False
    migrated["decomposition"]["driver"] = copy.deepcopy(DEFAULT_CONFIG["decomposition"]["driver"])
    if not migrated["features"]["ids"]:
        migrated["features"]["ids"] = list(LEGACY_FEATURE_IDS)
    validate_config(migrated)
    canonical_json(migrated)
    warnings = [
        "schema 1.0配置已在内存中迁移为1.1；请另存新文件，源文件未修改。",
        "迁移保留原预设与数值参数；扩展指标保持关闭，空特征选择固定为原33项。",
    ]
    return migrated, warnings


def resolve_config(config=None):
    if config is None:
        config = {}
    if isinstance(config, (str, Path)):
        config = load_json(config)
    if not isinstance(config, dict):
        fail("config_type", "config 必须是 JSON 对象。", category="configuration")
    version = config.get("schema_version", "1.0" if config.get("preset_version") == "1.0" else "1.1")
    if version == "1.0":
        return migrate_config(config)[0]
    if version != "1.1":
        fail("schema_version", "不支持该schema版本。", category="configuration")
    result = get_preset(config.get("pipeline_id", "single_standard"))
    _merge(result, config)
    validate_config(result)
    canonical_json(result)  # 也拒绝任意映射值中的非标准数值。
    return result


load_config = resolve_config


def save_config(config, path):
    resolved = resolve_config(config)
    Path(path).write_text(json.dumps(resolved, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return str(path)
