"""数值核心最小回归夹具：手算 exposure、边界、缺口和物理数组不变性。"""

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.config import get_preset
from eda_toolkit.core_features import (SCR_EXTRA_COLUMNS, SUPPORTED_FEATURE_IDS, build_windows, detect_scr,
                                       extract_window_features, scr_onset_eligible_mask)
from eda_toolkit.core_decompose import separate_tonic_phasic
from eda_toolkit.core_preprocess import preprocess_record
from eda_toolkit.contracts import make_record


def _record(signal, fs=1.0, recording_id="r1"):
    values = np.asarray(signal, dtype=float)
    time = np.arange(len(values), dtype=float) / fs
    return make_record(time, values, {"recording_id": recording_id, "unit": "uS", "sampling_rate_hz": fs})


def _qc(n, *, eligible=None, facts=None):
    return {"eligible": np.ones(n, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool),
            "facts": {} if facts is None else facts, "segments": [], "intervals": [],
            "summary": {}, "annotation_revision": 0}


def _bundle(time, clean, *, phasic=None, max_gap=1.5, eligible=None, detector=None, quality=None):
    time = np.asarray(time, dtype=float)
    clean = np.asarray(clean, dtype=float)
    valid = np.ones(len(time), bool) if eligible is None else np.asarray(eligible, dtype=bool)
    detection = valid.copy() if detector is None else np.asarray(detector, dtype=bool)
    q = {"processing_eligible": valid.copy(), "analysis_eligible": valid.copy(), "eligible": valid.copy(),
         "detector_eligible": detection.copy(), "interpolated": np.zeros(len(time), bool),
         "edge_affected": np.zeros(len(time), bool), "artifact": np.zeros(len(time), bool)}
    if quality:
        q.update(quality)
    return {"metadata": {"recording_id": "r1", "subject_id": "s1", "session_id": "ss1"},
            "time_s": time, "clean_uS": clean, "tonic_uS": clean.copy(),
            "phasic_uS": clean.copy() if phasic is None else np.asarray(phasic, dtype=float),
            "residual_uS": np.zeros(len(time)), "quality": q, "detector_eligible": detection,
            "max_gap_s": max_gap, "sampling_rate_hz": 1.0}


def test_interpolates_only_short_interior_missing_run_and_preserves_input_arrays():
    raw = _record([1.0, 2.0, np.nan, 4.0, 5.0])
    original_time = raw["time_s"].copy()
    original_signal = raw["eda_uS"].copy()
    config = get_preset()
    config["preprocess"]["interpolate_max_gap_s"] = 2.0
    qc = _qc(5, eligible=[True, True, False, True, True], facts={"original_missing": [False, False, True, False, False]})

    result = preprocess_record(raw, qc, config)

    assert result["clean_uS"][2] == pytest.approx(3.0)
    assert result["quality"]["interpolated"].tolist() == [False, False, True, False, False]
    assert result["quality"]["processing_eligible"].all()
    np.testing.assert_array_equal(raw["time_s"], original_time)
    np.testing.assert_equal(raw["eda_uS"], original_signal)


def test_exposure_and_phasic_auc_clip_window_without_tail_extrapolation():
    bundle = _bundle([0, 1, 2, 3], [0, 1, 2, 3], phasic=[0, 1, 2, 3])
    windows = pd.DataFrame([{"window_id": "w", "recording_id": "r1", "kind": "fixed",
                             "start_s": 0.5, "end_s": 2.5, "event_id": None, "condition": None}])

    features, validity = extract_window_features(bundle, pd.DataFrame(), windows, get_preset())

    row = validity.iloc[0]
    assert row["valid_duration_s"] == pytest.approx(2.0)
    assert row["search_eligible_duration_s"] == pytest.approx(2.0)
    assert row["onset_eligible_duration_s"] == pytest.approx(0.0)
    assert row["detector_eligible_duration_s"] == pytest.approx(0.0)
    assert features.loc[features["feature_id"] == "mean", "status"].iloc[0] == "valid"
    rate = features.loc[features["feature_id"] == "scr_rate"].iloc[0]
    assert rate["status"] == "unavailable"
    assert rate["reason"] == "no_detector_eligible_exposure"
    auc = features.loc[features["feature_id"] == "phasic_auc"].iloc[0]
    assert auc["value"] == pytest.approx(3.0)
    assert auc["method"] == "signed_trapezoid_clipped_to_window"


def test_feature_registry_selection_and_scr_upstroke_area_summaries():
    time = np.arange(11.0)
    bundle = _bundle(time, np.ones(len(time)), phasic=np.ones(len(time)))
    scr = pd.DataFrame([
        {"scr_id": "a", "recording_id": "r1", "onset_s": 2.0, "peak_s": 2.5,
         "amplitude_uS": 0.2, "rise_time_s": 0.5, "area_uS_s": 0.3,
         "status": "censored", "onset_eligible": True, "recovery_censor_reason": "recording_end"},
        {"scr_id": "b", "recording_id": "r1", "onset_s": 4.0, "peak_s": 4.5,
         "amplitude_uS": 0.4, "rise_time_s": 0.5, "area_uS_s": 0.7,
         "status": "censored", "onset_eligible": True, "recovery_censor_reason": "recording_end"},
    ])
    windows = pd.DataFrame([
        {"window_id": "whole", "recording_id": "r1", "kind": "whole_record",
         "start_s": 0.0, "end_s": 10.0, "event_id": None, "condition": None},
        {"window_id": "response", "recording_id": "r1", "kind": "event_response",
         "start_s": 1.0, "end_s": 7.0, "event_id": "e1", "condition": None, "event_onset_s": 0.0},
    ])
    config = get_preset()
    config["scr"].update(max_rise_s=1.0, onset_right_context_s=1.0)
    features, _ = extract_window_features(bundle, scr, windows, config)
    area_rows = features.loc[features.feature_id.eq("scr_area_sum_uS_s")].set_index("window_id")
    assert area_rows.loc["whole", "value"] == pytest.approx(scr.area_uS_s.sum())
    assert area_rows.loc["whole", "unit"] == "uS_s"
    assert area_rows.loc["whole", "basis"] == "qualified_global_scr_catalogue.area_uS_s"
    event_area = features.loc[features.feature_id.eq("event_scr_area_sum_uS_s")].iloc[0]
    assert event_area.value == pytest.approx(scr.area_uS_s.sum())
    assert event_area.method.startswith("sum_of_unique_event_scr_signed_trapezoid")
    assert {"scr_area_sum_uS_s", "scr_area_mean_uS_s", "event_scr_area_sum_uS_s"} <= set(SUPPORTED_FEATURE_IDS)

    mean_config = get_preset()
    mean_config["features"]["ids"] = ["mean"]
    selected, _ = extract_window_features(bundle, pd.DataFrame(), windows.iloc[[0]], mean_config)
    assert selected.feature_id.tolist() == ["mean"]

    invalid_config = get_preset()
    invalid_config["features"]["ids"] = ["mean", "unknown_feature"]
    with pytest.raises(ValueError) as caught:
        extract_window_features(bundle, pd.DataFrame(), windows.iloc[[0]], invalid_config)
    assert caught.value.eda_details["category"] == "configuration"
    assert caught.value.eda_details["field"] == "features.ids"


def test_gap_is_not_bridged_and_window_crossing_gap_is_invalid():
    bundle = _bundle([0, 1, 2, 10, 11], [0, 1, 2, 3, 4], max_gap=1.5)
    windows = pd.DataFrame([{"window_id": "gap", "recording_id": "r1", "kind": "whole_record",
                             "start_s": 0.0, "end_s": 11.1, "event_id": None, "condition": None}])

    _, validity = extract_window_features(bundle, pd.DataFrame(), windows, get_preset())

    row = validity.iloc[0]
    assert row["valid_duration_s"] == pytest.approx(3.0)
    assert bool(row["gap_crossing"])
    assert row["status"] == "unavailable"
    assert row["reason"] == "window_crosses_gap"


def test_flat_signal_returns_empty_scr_table_with_contract_columns():
    bundle = _bundle(np.arange(41.0), np.ones(41), phasic=np.zeros(41))

    scr, report = detect_scr(bundle, get_preset())

    assert scr.empty
    assert list(scr.columns) == [
        "scr_id", "recording_id", "onset_s", "peak_s", "amplitude_uS", "rise_time_s",
        "recovery_s", "recovery_time_s", "area_uS_s", "status", "reason", *SCR_EXTRA_COLUMNS,
    ]
    assert report["status"] == "no_responses"
    assert report["onset_eligible_duration_s"] == pytest.approx(30.0)
    assert report["detected_count"] == 0

    short_bundle = _bundle(np.arange(10.0), np.ones(10), phasic=np.zeros(10))
    short_scr, short_report = detect_scr(short_bundle, get_preset())
    assert short_scr.empty
    assert short_report["status"] == "unavailable"
    assert short_report["reason"] == "no_detector_eligible_exposure"


def test_lowpass_rejects_cutoff_at_nyquist():
    raw = _record([0, 1, 0, 1, 0, 1, 0, 1], fs=4.0)
    config = get_preset()
    config["mapping"]["sampling_rate_hz"] = 4.0
    config["preprocess"]["lowpass_hz"] = 2.0

    with pytest.raises(ValueError) as exc:
        preprocess_record(raw, _qc(8), config)

    assert exc.value.eda_details["code"] == "invalid_cutoff"


def test_uniform_filter_path_rejects_jitter_beyond_explicit_tolerance():
    raw = make_record([0.0, 0.25, 0.5, 0.76, 1.0], [1, 2, 1, 2, 1],
                      {"recording_id": "r1", "unit": "uS"})
    config = get_preset()
    config["mapping"]["sampling_rate_hz"] = 4.0
    config["preprocess"]["lowpass_hz"] = 1.0
    config["preprocess"]["jitter_tolerance"] = 0.01

    with pytest.raises(ValueError) as exc:
        preprocess_record(raw, _qc(5), config)

    assert exc.value.eda_details["code"] == "irregular_timebase"


@pytest.mark.parametrize("scale", [0.99, 1.01])
def test_reference_clock_fs_is_used_after_clock_drift_mapping(scale):
    fs_local = 4.0
    time_reference = np.arange(400, dtype=float) / fs_local * scale
    raw = make_record(time_reference, np.sin(2 * np.pi * 0.1 * time_reference),
                      {"recording_id": "r1", "unit": "uS", "clock_domain": "reference",
                       "fs_hz": fs_local / scale, "fs_reference_hz": fs_local / scale,
                       "fs_local_hz": fs_local, "source_fs_hz": fs_local,
                       "time_mapping": {"a": 3.0, "b": scale}})
    config = get_preset()
    config["mapping"]["sampling_rate_hz"] = fs_local
    config["preprocess"]["lowpass_hz"] = 0.5

    processed = preprocess_record(raw, _qc(400), config)

    assert processed["sampling_rate_hz"] == pytest.approx(fs_local / scale)
    filter_report = next(item for item in processed["operations"] if item["name"] == "butterworth_lowpass")
    assert filter_report["edge_radius_s"] > 0


def test_polyphase_resampling_keeps_long_qc_gap_separate_and_propagates_mask():
    values = np.sin(np.arange(100) * 0.2)
    raw = _record(values, fs=4.0)
    eligible = np.ones(100, dtype=bool)
    eligible[25:30] = False
    config = get_preset()
    config["mapping"]["sampling_rate_hz"] = 4.0
    config["preprocess"]["resample_hz"] = 2.0

    result = preprocess_record(raw, _qc(100, eligible=eligible), config)

    assert len(result["time_s"]) == len(result["clean_uS"]) == len(result["quality"]["eligible"])
    assert not np.any((result["time_s"] > 6.0) & (result["time_s"] < 7.5))
    assert np.any(result["quality"]["analysis_eligible"])
    assert result["quality"]["annotation_revision"] == 0
    assert "facts" in result["quality"] and len(result["quality"]["facts"]["edge_affected"]) == len(result["time_s"])


def test_resampling_preserves_original_local_clock_only_in_raw_snapshot():
    raw = _record(np.sin(np.arange(100) * 0.2), fs=4.0)
    raw["metadata"].update({"clock_domain": "reference", "fs_hz": 4.0, "fs_reference_hz": 4.0,
                            "clock_map_a_s": 0.1, "clock_map_b": 0.99,
                            "anchor_sha256": "fixture", "time_mapping_version": "1.0"})
    raw["local_time_s"] = np.arange(100, dtype=float) / 4.0
    local_time = raw["local_time_s"].copy()
    config = get_preset()
    config["preprocess"]["resample_hz"] = 2.0

    result = preprocess_record(raw, _qc(100), config)

    assert "local_time_s" not in result
    np.testing.assert_array_equal(result["record"]["local_time_s"], local_time)
    assert result["timebase_metadata"]["clock_map_b"] == 0.99
    assert result["timebase_metadata"]["anchor_sha256"] == "fixture"
    assert len(result["time_s"]) != len(result["record"]["local_time_s"])


def test_filter_separation_residual_definition_and_edge_mask():
    fs = 10.0
    time = np.arange(1000) / fs
    raw = _record(2.0 + 0.3 * np.sin(2 * np.pi * 0.1 * time) + 0.1 * np.sin(2 * np.pi * time), fs=fs)
    config = get_preset()
    config["decomposition"]["cutoff_hz"] = 0.5
    processed = preprocess_record(raw, _qc(1000), config)

    bundle = separate_tonic_phasic(processed, config)

    np.testing.assert_allclose(bundle["residual_uS"], bundle["clean_uS"] - bundle["tonic_uS"] - bundle["phasic_uS"])
    assert bundle["driver"] is None
    assert np.any(bundle["detector_eligible"])
    assert np.any(bundle["quality"]["edge_affected"])


def test_recovery_at_record_end_is_censored_instead_of_fabricated():
    bundle = _bundle(np.arange(7.0), [0] * 7, phasic=[0, -1, 0, 2, 1.8, 1.6, 1.5], max_gap=1.5)
    config = get_preset()
    config["scr"]["amplitude_min_uS"] = 0.1
    config["scr"]["min_interval_s"] = 0.0

    scr, report = detect_scr(bundle, config)

    assert len(scr) == 1
    assert np.isnan(scr.iloc[0]["recovery_s"])
    assert scr.iloc[0]["recovery_censor_reason"] == "recording_end"
    assert scr.iloc[0]["status"] == "censored"
    assert report["recovery_censored_count"] == 1


def _rise_trace(onset, peak, *, end=40):
    values = np.zeros(end + 1, dtype=float)
    values[onset] = -1.0
    if peak > onset + 1:
        values[onset + 1:peak + 1] = np.linspace(0.0, 3.0, peak - onset)
    values[peak + 1:] = np.linspace(2.9, 2.4, end - peak)
    return _bundle(np.arange(end + 1, dtype=float), values, phasic=values)


def _onset_context_config():
    config = get_preset()
    config["scr"].update({"max_rise_s": 10.0, "onset_left_context_s": 0.0,
                          "onset_right_context_s": 10.0, "amplitude_min_uS": 0.05,
                          "min_interval_s": 0.0})
    return config


def test_scr_onset_context_clips_exposure_preserves_boundary_catalogue_and_event_zero_status():
    config = _onset_context_config()

    interior_bundle = _rise_trace(29, 35)
    sample_mask, mask_report = scr_onset_eligible_mask(interior_bundle, config)
    assert sample_mask[29] and not sample_mask[35]
    assert mask_report["onset_eligible_duration_s"] == pytest.approx(30.0)
    interior, report = detect_scr(interior_bundle, config)
    assert report["search_eligible_duration_s"] == pytest.approx(40.0)
    assert report["onset_eligible_duration_s"] == pytest.approx(30.0)
    assert report["detector_eligible_duration_s"] == pytest.approx(30.0)
    assert interior.iloc[0]["onset_s"] == pytest.approx(29.0)
    assert bool(interior.iloc[0]["onset_eligible"])
    assert interior.iloc[0]["recovery_censor_reason"] == "recording_end"

    full_window = pd.DataFrame([{"window_id": "full", "recording_id": "r1", "kind": "fixed",
                                 "start_s": 0.0, "end_s": 40.0, "event_id": None, "condition": None}])
    features, validity = extract_window_features(_rise_trace(29, 35), interior, full_window, config)
    rate = features.loc[features["feature_id"] == "scr_rate"].iloc[0]
    assert rate["value"] == pytest.approx(2.0)
    assert validity.iloc[0]["onset_eligible_duration_s"] == pytest.approx(30.0)

    boundary, boundary_report = detect_scr(_rise_trace(35, 38), config)
    assert boundary_report["onset_eligible_duration_s"] == pytest.approx(30.0)
    assert len(boundary) == 1
    assert not bool(boundary.iloc[0]["onset_eligible"])
    assert boundary.iloc[0]["qualification_reason"] == "unavailable_onset_context"

    event_window = pd.DataFrame([{"window_id": "event-edge", "recording_id": "r1", "kind": "event_response",
                                  "start_s": 35.0, "end_s": 40.0, "event_id": "edge",
                                  "condition": None, "event_onset_s": 34.0}])
    boundary_features, boundary_validity = extract_window_features(_rise_trace(35, 38), boundary,
                                                                  event_window, config)
    edge_count = boundary_features.loc[boundary_features["feature_id"] == "event_scr_count"].iloc[0]
    assert edge_count["status"] == "unavailable"
    assert pd.isna(edge_count["value"])
    edge_link = boundary_validity.attrs["event_scr_links"].iloc[0]
    assert edge_link["link_status"] == "unavailable_onset_context"

    no_peak = _bundle(np.arange(41, dtype=float), np.zeros(41), phasic=np.zeros(41))
    supported_event = pd.DataFrame([{"window_id": "event-interior", "recording_id": "r1", "kind": "event_response",
                                     "start_s": 10.0, "end_s": 20.0, "event_id": "interior",
                                     "condition": None, "event_onset_s": 9.0}])
    no_scr, _ = detect_scr(no_peak, config)
    zero_features, zero_validity = extract_window_features(no_peak, no_scr, supported_event, config)
    zero_count = zero_features.loc[zero_features["feature_id"] == "event_scr_count"].iloc[0]
    assert zero_count["status"] == "valid"
    assert zero_count["value"] == 0
    assert zero_validity.attrs["event_scr_links"].iloc[0]["link_status"] == "no_response"


def test_scr_onset_membership_uses_half_open_window_boundary():
    bundle = _bundle(np.arange(11.0), np.ones(11))
    scr = pd.DataFrame([{"scr_id": "x", "recording_id": "r1", "onset_s": 2.0,
                         "peak_s": 2.5, "amplitude_uS": 0.2, "rise_time_s": 0.5,
                         "recovery_s": np.nan, "recovery_time_s": np.nan, "area_uS_s": 0.1,
                         "status": "detected", "recovery_censor_reason": "recording_end"}])
    windows = pd.DataFrame([
        {"window_id": "left", "recording_id": "r1", "kind": "fixed", "start_s": 0, "end_s": 2,
         "event_id": None, "condition": None},
        {"window_id": "right", "recording_id": "r1", "kind": "fixed", "start_s": 2, "end_s": 4,
         "event_id": None, "condition": None},
    ])

    config = get_preset()
    config["scr"]["max_rise_s"] = 1.0
    config["scr"]["onset_right_context_s"] = 1.0
    features, _ = extract_window_features(bundle, scr, windows, config)

    counts = features.loc[features["feature_id"] == "scr_count"].set_index("window_id")["value"]
    assert counts["left"] == 0
    assert counts["right"] == 1


def test_build_windows_emits_empty_contract_columns_and_event_relative_ranges():
    record = _record([0, 0, 0, 0, 0], fs=1.0)
    config = get_preset()
    config["windows"]["fixed_length_s"] = None
    windows = build_windows(record, [{"recording_id": "r1", "event_id": "e1", "onset_s": 2.0,
                                      "event_type": "stim", "condition": "A"}], None, config)

    baseline = windows.loc[windows["kind"] == "event_baseline"].iloc[0]
    response = windows.loc[windows["kind"] == "event_response"].iloc[0]
    assert set(["window_id", "recording_id", "kind", "start_s", "end_s", "event_id", "condition"]) <= set(windows.columns)
    assert (baseline["start_s"], baseline["end_s"]) == pytest.approx((0.0, 2.0))
    assert (response["start_s"], response["end_s"]) == pytest.approx((3.0, 6.0))


def test_event_response_change_uses_clean_baseline_on_same_basis_and_peak_time_window():
    time = np.arange(15.0)
    clean = np.where(time < 5.0, 1.0, 3.0)
    phasic = np.where(time < 5.0, -100.0, 0.5)
    bundle = _bundle(time, clean, phasic=phasic, max_gap=1.5)
    config = get_preset()
    config["windows"]["fixed_length_s"] = None
    config["windows"]["event_response_s"] = [0.0, 2.0]
    config["windows"]["event_peak_s"] = [1.0, 5.0]
    config["windows"]["event_recovery_s"] = [1.0, 12.0]
    config["scr"]["max_rise_s"] = 2.0
    config["scr"]["onset_right_context_s"] = 2.0
    windows = build_windows(bundle, [{"recording_id": "r1", "event_id": "e1", "onset_s": 5.0}], None, config)
    scr = pd.DataFrame([{"scr_id": "scr1", "recording_id": "r1", "onset_s": 5.2,
                         "peak_s": 5.8, "amplitude_uS": 0.4, "rise_time_s": 0.6,
                         "recovery_s": np.nan, "recovery_time_s": np.nan, "area_uS_s": 0.1,
                         "status": "detected", "recovery_censor_reason": "recording_end"}])

    features, validity = extract_window_features(bundle, scr, windows, config)

    change = features.loc[features["feature_id"] == "event_response_mean_change_uS"].iloc[0]
    peak = features.loc[features["feature_id"] == "event_peak_max_amplitude_uS"].iloc[0]
    assert change["value"] == pytest.approx(2.0)
    assert change["basis"] == "clean_uS"
    # Peak 5.8 s precedes this peak-search window [6,10), although onset 5.2 is inside response [5,7).
    assert pd.isna(peak["value"])
    assert peak["status"] == "unavailable"
    assert validity.attrs["event_scr_links"].iloc[0]["link_status"] == "unique"


def test_competing_event_links_are_ambiguous_and_not_summarized_twice():
    bundle = _bundle(np.arange(10.0), np.ones(10))
    windows = pd.DataFrame([
        {"window_id": "e1r", "recording_id": "r1", "kind": "event_response", "start_s": 1, "end_s": 4,
         "event_id": "e1", "condition": None, "event_onset_s": 0},
        {"window_id": "e2r", "recording_id": "r1", "kind": "event_response", "start_s": 2, "end_s": 5,
         "event_id": "e2", "condition": None, "event_onset_s": 1},
    ])
    scr = pd.DataFrame([{"scr_id": "scr1", "recording_id": "r1", "onset_s": 2.5,
                         "peak_s": 3.0, "amplitude_uS": 0.2, "rise_time_s": 0.5,
                         "status": "detected"}])

    features, validity = extract_window_features(bundle, scr, windows, get_preset())

    links = validity.attrs["event_scr_links"]
    assert set(links["link_status"]) == {"ambiguous"}
    counts = features.loc[features["feature_id"] == "event_scr_count"]
    assert len(counts) == 2
    assert counts["status"].tolist() == ["ambiguous", "ambiguous"]
    assert counts["value"].isna().all()

