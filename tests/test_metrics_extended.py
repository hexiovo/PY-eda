"""WB02 hand-computed checks for the C-owned time-domain/trial/NS methods."""

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.config import get_preset
from eda_toolkit.core_features import (SUPPORTED_FEATURE_IDS, detect_scr, extract_window_features,
                                      list_feature_definitions)
from eda_toolkit.metrics_nonspecific import classify_nonspecific_scr
from eda_toolkit.metrics_trials import build_trial_table, summarize_trial_series


def _bundle(time, clean, *, phasic=None, eligible=None, max_gap=1.5):
    time = np.asarray(time, dtype=float)
    clean = np.asarray(clean, dtype=float)
    mask = np.ones(len(time), bool) if eligible is None else np.asarray(eligible, bool)
    return {"metadata": {"recording_id": "r1", "subject_id": "s1", "session_id": "ss1",
                         "channel": "eda", "clock_domain": "recording_local"},
            "time_s": time, "clean_uS": clean, "tonic_uS": clean.copy(),
            "phasic_uS": clean.copy() if phasic is None else np.asarray(phasic, float),
            "quality": {"analysis_eligible": mask.copy(), "detector_eligible": mask.copy(),
                        "interpolated": np.zeros(len(time), bool), "artifact": np.zeros(len(time), bool)},
            "detector_eligible": mask.copy(), "max_gap_s": max_gap, "sampling_rate_hz": 1.0}


def _one_window(start, end, *, window_id="w", kind="fixed", event_id=None, onset=None):
    return pd.DataFrame([{"recording_id": "r1", "window_id": window_id, "kind": kind,
                          "start_s": start, "end_s": end, "event_id": event_id, "condition": "A",
                          "event_onset_s": onset}])


def test_registry_contains_legacy_and_dynamic_ids_with_stable_metadata():
    config = get_preset("group_sync")
    config["decomposition"]["method"] = "bateman_sparse"
    config["features"]["extended"]["spectral"]["bands"].append(
        {"id": "low_test", "low_hz": 0.02, "high_hz": 0.04})
    config["features"]["extended"]["autocorrelation"]["lags_s"].append(0.5)
    definitions = list_feature_definitions(config)
    assert set(SUPPORTED_FEATURE_IDS) <= set(definitions) | {
        "clean_bandpower_low_test_uS2", "clean_bandpower_low_test_normalized", "clean_autocorr_lag_0_5s"}
    assert len(SUPPORTED_FEATURE_IDS) == 144
    assert len(list_feature_definitions(get_preset("group_sync"))) == 135
    assert "clean_bandpower_low_test_uS2" in definitions
    assert "clean_bandpower_low_test_normalized" in definitions
    assert "clean_autocorr_lag_0_5s" in definitions
    assert definitions["scr_amplitude_sd"]["min_n"] == 2
    assert definitions["iqr"]["min_n"] == 1
    assert definitions["sync_lag_sd_s"]["parameters"]["ddof"] == 0
    assert definitions["sync_lag_sd_s"]["parameters"]["weighting"] == \
        "nonoverlapping_midpoint_cell_intersect_pair_eligible_support"
    assert definitions["clean_sample_entropy"]["basis"] == "clean_uS"
    assert set(definitions["tonic_sd"]) == {
        "id", "label_zh", "family", "formula", "basis", "unit", "min_n",
        "min_duration_s", "parameters", "validity", "qualification"}
    single_definitions = list_feature_definitions(get_preset())
    assert "sync_positive_time_proportion" not in single_definitions
    disabled = get_preset("group_sync")
    disabled["features"]["extended"]["enabled"] = False
    assert set(list_feature_definitions(disabled)) == set(SUPPORTED_FEATURE_IDS[:33])


def test_piecewise_linear_phasic_integrals_split_at_zero_and_do_not_bridge_gap():
    bundle = _bundle([0.0, 1.0], [1.0, 1.0], phasic=[-1.0, 1.0])
    config = get_preset()
    config["features"]["ids"] = ["phasic_auc_per_valid_s", "phasic_positive_auc_uS_s",
                                  "phasic_absolute_auc_uS_s", "phasic_positive_auc_per_valid_s",
                                  "phasic_absolute_auc_per_valid_s"]
    features, _ = extract_window_features(bundle, pd.DataFrame(), _one_window(0.0, 1.0), config)
    values = features.set_index("feature_id")["value"]
    assert values["phasic_auc_per_valid_s"] == pytest.approx(0.0)
    assert values["phasic_positive_auc_uS_s"] == pytest.approx(0.25)
    assert values["phasic_absolute_auc_uS_s"] == pytest.approx(0.5)
    assert values["phasic_positive_auc_per_valid_s"] == pytest.approx(0.25)
    assert values["phasic_absolute_auc_per_valid_s"] == pytest.approx(0.5)

    gap_bundle = _bundle([0.0, 1.0, 5.0, 6.0], [1.0] * 4, phasic=[1.0] * 4)
    gap_window = _one_window(0.0, 6.0)
    gap_features, validity = extract_window_features(gap_bundle, pd.DataFrame(), gap_window, get_preset())
    assert validity.iloc[0]["valid_duration_s"] == pytest.approx(2.0)
    assert gap_features.loc[gap_features.feature_id.eq("phasic_positive_auc_uS_s"), "status"].iloc[0] == "unavailable"
    assert gap_features.loc[gap_features.feature_id.eq("phasic_positive_auc_uS_s"), "reason"].iloc[0] == "window_crosses_gap"


def test_single_selected_advanced_metric_hides_internal_dependencies():
    rng = np.random.default_rng(41)
    time = np.arange(301.0)
    values = np.cumsum(rng.normal(0.0, 0.02, len(time))) + 2.0
    bundle = _bundle(time, values, max_gap=1.5)
    window = _one_window(0.0, 300.0)
    config = get_preset()
    config["features"]["ids"] = ["clean_sample_entropy"]
    features, _ = extract_window_features(bundle, pd.DataFrame(), window, config)
    assert features.feature_id.tolist() == ["clean_sample_entropy"]
    assert features.iloc[0]["basis"] == "clean_uS"
    assert features.iloc[0]["unit"] == "nats"


def test_scr_rise_recovery_halfwidth_and_minimum_interval_are_segment_scoped():
    values = np.zeros(21)
    values[:9] = [1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0]
    bundle = _bundle(np.arange(21.0), np.ones(21), phasic=values)
    config = get_preset()
    config["scr"].update({"max_rise_s": 4.0, "onset_left_context_s": 0.0,
                          "onset_right_context_s": 2.0, "min_interval_s": 30.0,
                          "amplitude_min_uS": 0.1})
    scr, report = detect_scr(bundle, config)
    assert len(scr) == 1
    row = scr.iloc[0]
    assert row["rise_slope_uS_s"] == pytest.approx(1.0)
    assert row["rise_max_slope_uS_s"] == pytest.approx(1.0)
    assert row["recovery_50_time_s"] == pytest.approx(1.5)
    assert row["recovery_63_time_s"] == pytest.approx(1.89, abs=0.02)
    assert row["half_width_50_s"] == pytest.approx(3.0)
    assert row["segment_id"] == 0

    # Same onset distance would violate min_interval if the two detector segments
    # were compared globally; the real data gap separates them.
    gap_time = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], float)
    gap_signal = np.array([2, 0, 1, 2, 1, 2, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0], float)
    gap_bundle = _bundle(gap_time, np.ones(len(gap_time)), phasic=gap_signal)
    gap_scr, _ = detect_scr(gap_bundle, config)
    assert len(gap_scr) == 2
    assert gap_scr.segment_id.tolist() == [0, 1]

    truncated = _bundle(np.arange(6.0), np.ones(6), phasic=[1, 0, 1, 2, 3, 2.8])
    censor_config = get_preset()
    censor_config["scr"].update({"max_rise_s": 4.0, "onset_right_context_s": 0.0,
                                 "amplitude_min_uS": 0.1, "min_interval_s": 0.0})
    censored, _ = detect_scr(truncated, censor_config)
    assert censored.iloc[0]["recovery_50_status"] == "censored"
    assert censored.iloc[0]["half_width_50_status"] == "censored"


def test_trial_denominator_excludes_invalid_and_keeps_zero_inclusive_amplitudes():
    bundle = _bundle(np.arange(20.0), np.linspace(2.0, 4.0, 20))
    rows = []
    for i, (baseline, response) in enumerate(((0, 3), (7, 10), (14, 17)), 1):
        event_id = f"e{i}"
        onset = baseline + 3
        rows.extend([
            {"window_id": f"{event_id}:b", "recording_id": "r1", "kind": "event_baseline",
             "start_s": float(baseline), "end_s": float(baseline + 2), "event_id": event_id,
             "condition": "A", "event_onset_s": onset, "event_ordinal": i},
            {"window_id": f"{event_id}:r", "recording_id": "r1", "kind": "event_response",
             "start_s": float(response), "end_s": float(response + 3), "event_id": event_id,
             "condition": "A", "event_onset_s": onset, "event_ordinal": i},
        ])
    windows = pd.DataFrame(rows)
    scr = pd.DataFrame([{"scr_id": "s1", "recording_id": "r1", "onset_s": 4.0,
                         "peak_s": 5.0, "amplitude_uS": 0.2, "status": "valid",
                         "onset_eligible": True}])
    links = pd.DataFrame([
        {"recording_id": "r1", "event_id": "e1", "scr_id": "s1", "link_status": "unique"},
        {"recording_id": "r1", "event_id": "e2", "scr_id": None, "link_status": "no_response"},
        {"recording_id": "r1", "event_id": "e3", "scr_id": None, "link_status": "invalid_window"},
    ])
    table = build_trial_table(bundle, scr, windows, links, get_preset())
    assert table.trial_ordinal.tolist() == [1, 2, 3]
    assert table.trial_status.tolist() == ["valid", "valid", "excluded"]
    assert table.amplitude_zero_uS.iloc[0] == pytest.approx(0.2)
    assert table.amplitude_zero_uS.iloc[1] == pytest.approx(0.0)
    assert pd.isna(table.amplitude_zero_uS.iloc[2])
    features, summaries, _ = summarize_trial_series(table, get_preset())
    summary = summaries.iloc[0]
    assert summary.planned_n == 3 and summary.valid_n == 2 and summary.excluded_n == 1
    feature_values = features.set_index("feature_id")["value"]
    assert feature_values["event_response_probability"] == pytest.approx(0.5)
    assert feature_values["event_nonresponse_rate"] == pytest.approx(0.5)
    assert feature_values["event_amplitude_zero_mean_uS"] == pytest.approx(0.1)
    assert feature_values["event_amplitude_conditional_mean_uS"] == pytest.approx(0.2)


def test_event_peak_latency_and_baseline_percent_z_use_one_clean_signal_basis():
    time = np.arange(51.0)
    clean = np.full(len(time), 2.0)
    clean[:3] = [2.0, 3.0, 4.0]
    clean[3:6] = [4.0, 5.0, 6.0]
    bundle = _bundle(time, clean)
    windows = pd.DataFrame([
        {"window_id": "e:baseline", "recording_id": "r1", "kind": "event_baseline",
         "start_s": 0.0, "end_s": 3.0, "event_id": "e", "condition": "A", "event_onset_s": 3.0,
         "event_ordinal": 1},
        {"window_id": "e:response", "recording_id": "r1", "kind": "event_response",
         "start_s": 3.0, "end_s": 6.0, "event_id": "e", "condition": "A", "event_onset_s": 3.0,
         "event_ordinal": 1},
    ])
    scr = pd.DataFrame([{"scr_id": "scr-e", "recording_id": "r1", "onset_s": 4.0, "peak_s": 5.0,
                         "amplitude_uS": 0.3, "rise_time_s": 1.0, "area_uS_s": 0.15,
                         "recovery_s": 7.0, "recovery_time_s": 2.0, "status": "valid",
                         "onset_eligible": True, "segment_id": 0, "recovery_censor_reason": ""}])
    config = get_preset()
    config["features"]["ids"] = ["event_peak_latency_s", "event_max_amplitude_baseline_percent",
                                  "event_response_change_baseline_percent", "event_max_amplitude_baseline_z",
                                  "event_response_change_baseline_z"]
    features, validity = extract_window_features(bundle, scr, windows, config)
    response = features.loc[features.window_id.eq("e:response")].set_index("feature_id")
    assert response.loc["event_peak_latency_s", "value"] == pytest.approx(2.0)
    assert response.loc["event_max_amplitude_baseline_percent", "value"] == pytest.approx(10.0)
    assert response.loc["event_response_change_baseline_percent", "value"] == pytest.approx(200.0 / 3.0)
    assert response.loc["event_max_amplitude_baseline_z", "value"] == pytest.approx(0.3)
    assert response.loc["event_response_change_baseline_z", "value"] == pytest.approx(2.0)
    assert response.loc["event_response_change_baseline_percent", "basis"] == "clean_baseline"
    assert validity.attrs["event_scr_links"].iloc[0]["link_status"] == "unique"


def test_non_specific_requires_matching_complete_annotation_coverage_and_subtracts_events():
    bundle = _bundle(np.arange(21.0), np.ones(21))
    scr = pd.DataFrame([
        {"scr_id": "ns1", "onset_s": 5.0, "amplitude_uS": 0.1, "onset_eligible": True},
        {"scr_id": "ev1", "onset_s": 12.0, "amplitude_uS": 0.2, "onset_eligible": True},
        {"scr_id": "ns2", "onset_s": 16.0, "amplitude_uS": 0.3, "onset_eligible": True},
    ])
    windows = pd.DataFrame([{"kind": "event_response", "event_id": "e1", "event_onset_s": 10.0,
                             "start_s": 11.0, "end_s": 14.0}])
    config = get_preset()
    config["features"]["extended"]["nonspecific"]["annotation_coverage"] = [{
        "recording_id": "r1", "channel": "eda", "start_s": 0.0, "end_s": 30.0,
        "clock_domain": "recording_local", "complete": True}]
    table, intervals, reason = classify_nonspecific_scr(bundle, scr, windows, [(0.0, 20.0)],
                                                        pd.DataFrame(), config)
    assert reason == ""
    assert intervals == [(4.0, 11.0), (14.0, 20.0)]
    classes = table.set_index("scr_id")["class"]
    assert classes["ns1"] == "non_event_associated"
    assert classes["ev1"] == "event_associated_or_outside"
    assert classes["ns2"] == "non_event_associated"

    config["features"]["extended"]["nonspecific"]["annotation_coverage"] = []
    table, intervals, reason = classify_nonspecific_scr(bundle, scr, windows, [(0.0, 20.0)],
                                                        pd.DataFrame(), config)
    assert reason == "NA_annotation_incomplete"
    assert intervals == []
    assert set(table.status) == {"unavailable"}


def test_non_specific_complete_coverage_is_partition_invariant_before_onset_transform():
    from eda_toolkit.metrics_nonspecific import build_nonspecific_support

    config = get_preset()
    bundle = {"metadata": {"recording_id": "r1", "channel": "eda", "clock_domain": "recording_local"}}
    row = {"recording_id": "r1", "channel": "eda", "clock_domain": "recording_local", "complete": True}
    config["features"]["extended"]["nonspecific"]["annotation_coverage"] = [
        {**row, "start_s": 0.0, "end_s": 20.0}]
    whole, _, _ = build_nonspecific_support(bundle, pd.DataFrame(), [(0.0, 30.0)], config)
    config["features"]["extended"]["nonspecific"]["annotation_coverage"] = [
        {**row, "start_s": 0.0, "end_s": 10.0},
        {**row, "start_s": 10.0, "end_s": 20.0}]
    partitioned, _, _ = build_nonspecific_support(bundle, pd.DataFrame(), [(0.0, 30.0)], config)
    assert whole == [(4.0, 21.0)]
    assert partitioned == whole
    assert sum(end - start for start, end in partitioned) == pytest.approx(17.0)

    # Any positive source-coverage gap remains real; it cannot be swallowed by
    # a blanket epsilon when constructing the union.
    config["features"]["extended"]["nonspecific"]["annotation_coverage"] = [
        {**row, "start_s": 0.0, "end_s": 10.0},
        {**row, "start_s": 10.0 + 5e-13, "end_s": 20.0}]
    gapped, _, _ = build_nonspecific_support(bundle, pd.DataFrame(), [(0.0, 30.0)], config)
    assert gapped == [(4.0, 11.0), (14.0 + 5e-13, 21.0)]


def test_increasing_valid_trial_amplitudes_do_not_report_identifiable_decay_tau():
    rows = [{"series_id": "series", "recording_id": "r1", "channel": "eda",
             "subject_id": "s1", "session_id": "ss1", "condition": "A", "event_type": "stim",
             "trial_status": "valid", "reason": "", "trial_ordinal": i,
             "amplitude_zero_uS": float(i), "response_present": 1,
             "amplitude_conditional_uS": float(i)} for i in range(1, 9)]
    features, summaries, diagnostics = summarize_trial_series(pd.DataFrame(rows), get_preset())
    tau = features.loc[features.feature_id.eq("habituation_decay_tau_trials")].iloc[0]
    fit = diagnostics.loc[diagnostics.feature_id.eq("habituation_decay_rate_per_trial")].iloc[0]
    assert summaries.iloc[0].valid_n == 8
    assert tau.status == "unavailable"
    assert pd.isna(tau.value)
    assert fit.status == "unavailable"
    assert fit.reason == "no_identifiable_decay_beyond_constant_boundary"


def test_identifiable_exponential_decay_retains_rate_and_tau_estimates():
    ordinal = np.arange(1, 13, dtype=float)
    amplitudes = 0.5 + 3.0 * np.exp(-0.2 * (ordinal - ordinal[0]))
    rows = [{"series_id": "series", "recording_id": "r1", "channel": "eda",
             "subject_id": "s1", "session_id": "ss1", "condition": "A", "event_type": "stim",
             "trial_status": "valid", "reason": "", "trial_ordinal": int(i),
             "amplitude_zero_uS": float(a), "response_present": 1,
             "amplitude_conditional_uS": float(a)} for i, a in zip(ordinal, amplitudes)]
    features, _, diagnostics = summarize_trial_series(pd.DataFrame(rows), get_preset())
    selected = features.set_index("feature_id")
    fit = diagnostics.loc[diagnostics.feature_id.eq("habituation_decay_rate_per_trial")].iloc[0]
    assert fit.status == "valid"
    assert selected.loc["habituation_decay_rate_per_trial", "value"] == pytest.approx(0.2, abs=1e-6)
    assert selected.loc["habituation_decay_tau_trials", "value"] == pytest.approx(5.0, abs=1e-4)
    assert selected.loc["habituation_decay_asymptote_uS", "value"] == pytest.approx(0.5, abs=1e-6)


def test_exponential_decay_parameters_are_invariant_to_amplitude_unit_scale():
    ordinal = np.arange(8, dtype=float)
    base = 0.5 + 2.0 * np.exp(-0.2 * ordinal)
    rows = []
    for series_id, scale in (("native", 1.0), ("micro", 1e-6)):
        for index, value in enumerate(base):
            amplitude = float(value * scale)
            rows.append({"series_id": series_id, "recording_id": "r1", "channel": "eda",
                         "subject_id": "s1", "session_id": "ss1", "condition": "A",
                         "event_type": "stim", "trial_status": "valid", "reason": "",
                         "trial_ordinal": index, "amplitude_zero_uS": amplitude,
                         "response_present": 1, "amplitude_conditional_uS": amplitude})
    features, _, _ = summarize_trial_series(pd.DataFrame(rows), get_preset())
    rates = features.loc[features.feature_id.eq("habituation_decay_rate_per_trial")].set_index("window_id")
    taus = features.loc[features.feature_id.eq("habituation_decay_tau_trials")].set_index("window_id")
    for series_id in ("native", "micro"):
        assert rates.loc[series_id, "status"] == "valid"
        assert rates.loc[series_id, "value"] == pytest.approx(0.2, abs=1e-6)
        assert taus.loc[series_id, "value"] == pytest.approx(5.0, abs=1e-4)
