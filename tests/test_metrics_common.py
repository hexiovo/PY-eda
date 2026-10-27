"""Analytical examples and invariants for the common EDA metrics."""

import json

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.config import get_preset, resolve_config
from eda_toolkit.core_features import extract_window_features, list_feature_definitions
from eda_toolkit.metrics_common import (COMMON_IDS, amplitude_reference, annotate_amplitudes,
                                        distribution_summary, transform_amplitudes)
from eda_toolkit.metrics_trials import summarize_trial_series


def bundle(values):
    values = np.asarray(values, float)
    mask = np.ones(len(values), bool)
    return {"metadata": {"recording_id": "r", "subject_id": "s", "session_id": "v", "channel": "eda"},
            "time_s": np.arange(len(values), dtype=float), "clean_uS": values,
            "tonic_uS": values.copy(), "phasic_uS": values.copy(),
            "quality": {"analysis_eligible": mask, "detector_eligible": mask,
                        "interpolated": ~mask, "artifact": ~mask},
            "detector_eligible": mask, "sampling_rate_hz": 1., "max_gap_s": 1.5}


def windows(end, start=0.):
    return pd.DataFrame([{"recording_id": "r", "window_id": "w", "kind": "fixed",
                          "start_s": start, "end_s": end, "event_id": None, "condition": "A"}])


def scr_table():
    return pd.DataFrame([dict(scr_id=str(i), onset_s=onset, peak_s=onset+rise,
                             amplitude_uS=amplitude, rise_time_s=rise, recovery_time_s=recovery,
                             recovery_censor_reason="", recovery_50_time_s=recovery,
                             recovery_50_status="valid", recovery_63_time_s=recovery*1.26,
                             recovery_63_status="valid", status="valid", onset_eligible=True,
                             segment_id=0) for i, (onset, amplitude, rise, recovery) in enumerate(
                                 [(10., 1., 1., 2.), (30., 4., 3., 4.), (50., 9., 5., 6.)])])


def test_hand_computed_sample_summaries_and_half_open_window():
    config = get_preset()
    config["features"]["ids"] = sorted(fid for fid in COMMON_IDS if fid.startswith(("clean_", "tonic_")))
    # Last outlier is outside [0,5); it must not enter any sample summary.
    features, _ = extract_window_features(bundle([1, 2, 3, 4, 5, 1000]), pd.DataFrame(), windows(5.), config)
    expected = {"min": 1., "max": 5., "p05": 1.2, "p25": 2., "p75": 4., "p95": 4.8,
                "skewness": 0., "kurtosis": -1.2, "mad": 1.}
    results = features.set_index("feature_id")
    assert len(results) == 18
    for prefix in ("clean", "tonic"):
        for suffix, value in expected.items():
            row = results.loc[f"{prefix}_{suffix}"]
            assert row.value == pytest.approx(value, rel=2e-14, abs=2e-14)
            assert row.status == "valid" and row.valid_n == 5


@pytest.mark.parametrize("values,statistic,reason", [
    ([], "min", "requires_at_least_1_samples"),
    ([1., 2.], "skewness", "requires_at_least_3_samples"),
    ([1., 2., 3.], "kurtosis", "requires_at_least_4_samples"),
    ([2.] * 4, "skewness", "constant_or_numerically_constant_samples"),
    ([2.] * 4, "kurtosis", "constant_or_numerically_constant_samples"),
])
def test_undefined_distribution_moments_remain_missing(values, statistic, reason):
    assert distribution_summary(values, statistic) == (None, reason)


def test_invalid_window_and_interpolation_policy():
    data = bundle([1, 2, 100, 4, 5, 6])
    data["quality"]["interpolated"][2] = True
    config = get_preset()
    config["features"]["ids"] = ["clean_max", "tonic_mad"]
    config["windows"]["min_coverage"] = 0.5
    features, _ = extract_window_features(data, pd.DataFrame(), windows(5.), config)
    assert features.status.eq("valid").all()  # 3/5 seconds meets the explicit 0.5 threshold.
    assert features.set_index("feature_id").loc["clean_max", "value"] == 5.
    assert features.valid_n.eq(4).all()
    config["windows"]["min_coverage"] = 0.8
    features, _ = extract_window_features(data, pd.DataFrame(), windows(5.), config)
    assert features.status.eq("unavailable").all()
    assert features.value.isna().all()
    config["features"]["exclude_interpolated"] = False
    features, _ = extract_window_features(data, pd.DataFrame(), windows(5.), config)
    assert features.set_index("feature_id").loc["clean_max", "value"] == 100.


def test_transform_before_aggregation_and_whole_record_reference():
    config = get_preset()
    config["features"]["ids"] = sorted(fid for fid in COMMON_IDS if fid.startswith("scr_amplitude_"))
    features, validity = extract_window_features(bundle(np.ones(81)), scr_table(), windows(40.), config)
    rows = features.set_index("feature_id")
    # Window includes A=[1,4]; the independent reference includes [1,4,9].
    expected = {"sqrt": 1.5, "log1p": np.log(10)/2,
                "recording_z": (2.5-14/3)/(7/np.sqrt(3)), "recording_max_fraction": 2.5/9}
    for kind, value in expected.items():
        assert rows.loc[f"scr_amplitude_{kind}_mean", "value"] == pytest.approx(value, abs=1e-14)
    table = validity.attrs["extended_tables"]["scr_events_extended"]
    np.testing.assert_array_equal(table.amplitude_uS, [1., 4., 9.])
    np.testing.assert_allclose(table.amplitude_sqrt, [1., 2., 3.])
    assert table.amplitude_reference_n.eq(3).all()
    assert table.amplitude_reference_max_uS.eq(9).all()
    assert table.amplitude_reference_scope.str.contains("same_recording").all()


def test_timing_dispersion_uses_completed_recoveries_and_records_censoring():
    config = get_preset()
    config["features"]["ids"] = sorted(fid for fid in COMMON_IDS if fid.startswith(("scr_rise", "scr_recovery")))
    catalogue = scr_table()
    catalogue.loc[2, ["recovery_50_status", "recovery_63_status"]] = "censored"
    catalogue.loc[2, "recovery_censor_reason"] = "segment_end"
    features, _ = extract_window_features(bundle(np.ones(81)), catalogue, windows(70.), config)
    rows = features.set_index("feature_id")
    assert rows.loc["scr_rise_time_median", "value"] == 3.
    assert rows.loc["scr_rise_time_sd", "value"] == 2.
    for prefix, suffix, scale in (("scr_recovery_time", "", 1.), ("scr_recovery_50", "_s", 1.),
                                  ("scr_recovery_63", "_s", 1.26)):
        assert rows.loc[f"{prefix}_median{suffix}", "value"] == pytest.approx(3*scale)
        assert rows.loc[f"{prefix}_sd{suffix}", "value"] == pytest.approx(np.sqrt(2)*scale)
        assert rows.loc[f"{prefix}_sd{suffix}", "reason"] == "some_censored_recoveries_excluded"
        assert rows.loc[f"{prefix}_sd{suffix}", "valid_n"] == 2
    catalogue["recovery_50_status"] = "censored"
    features, _ = extract_window_features(bundle(np.ones(81)), catalogue, windows(70.), config)
    result = features.set_index("feature_id").loc["scr_recovery_50_median_s"]
    assert result.status == "censored" and pd.isna(result.value)


def trials():
    return pd.DataFrame([dict(recording_id="r", subject_id="s", session_id="v", channel="eda",
                             series_id=condition, condition=condition, event_type="event", trial_ordinal=i+1,
                             trial_status="valid" if i<4 else "excluded", reason="" if i<4 else "artifact",
                             response_present=int(amplitude>0), amplitude_zero_uS=amplitude,
                             amplitude_conditional_uS=amplitude if amplitude else np.nan)
                         for i, (condition, amplitude) in enumerate([("A", 0.), ("A", 1.), ("B", 4.), ("B", 9.), ("B", 999.)])])


def test_trial_reference_across_conditions_keeps_zero_and_excludes_invalid_trials():
    table = trials()
    annotated = annotate_amplitudes(table, "amplitude_zero_uS", table.trial_status.eq("valid"))
    assert annotated.amplitude_reference_n.eq(4).all()
    assert annotated.amplitude_reference_max_uS.eq(9).all()
    assert annotated.iloc[0].amplitude_zero_log1p == 0.
    assert pd.isna(annotated.iloc[4].amplitude_zero_log1p)
    features, _, _ = summarize_trial_series(table, get_preset())
    rows = features.loc[features.window_id.eq("A")].set_index("feature_id")
    assert rows.loc["event_amplitude_zero_sqrt_mean", "value"] == .5
    assert rows.loc["event_amplitude_zero_log1p_mean", "value"] == pytest.approx(np.log(2)/2)
    assert rows.loc["event_amplitude_zero_recording_max_fraction_mean", "value"] == pytest.approx(.5/9)
    assert rows.loc["event_amplitude_zero_recording_z_mean", "value"] == pytest.approx(-3/(7/np.sqrt(3)))


@pytest.mark.parametrize("key", ["subject_id", "session_id", "recording_id", "channel"])
def test_reference_isolation(key):
    first = trials().iloc[:4].copy()
    second = first.copy()
    second[key] = "different"
    second["amplitude_zero_uS"] *= 100
    table = pd.concat([first, second], ignore_index=True)
    result = annotate_amplitudes(table, "amplitude_zero_uS", np.ones(len(table), bool))
    np.testing.assert_array_equal(result.amplitude_reference_max_uS, [9]*4+[900]*4)
    np.testing.assert_allclose(result.amplitude_zero_recording_z[:4], result.amplitude_zero_recording_z[4:])


@pytest.mark.parametrize("values,kind,reason", [
    ([0., 0.], "recording_z", "constant_reference_amplitudes"),
    ([2., 2.], "recording_z", "constant_reference_amplitudes"),
    ([0., 0.], "recording_max_fraction", "no_positive_reference_amplitude"),
    ([1.], "recording_z", "requires_two_reference_amplitudes"),
    ([-1.], "sqrt", "invalid_or_negative_amplitude"),
    ([], "log1p", "no_valid_amplitudes"),
])
def test_degenerate_amplitude_references(values, kind, reason):
    transformed, actual = transform_amplitudes(values, kind, amplitude_reference(values))
    assert actual == reason
    assert np.isnan(transformed).all()


def test_all_nonresponse_trials_keep_transforms_zero_and_normalizations_missing():
    table = trials().iloc[:4].copy()
    table["response_present"] = 0
    table["amplitude_zero_uS"] = 0.
    table["amplitude_conditional_uS"] = np.nan
    features, _, _ = summarize_trial_series(table, get_preset())
    selected = features.loc[features.feature_id.str.startswith("event_amplitude_zero_")]
    transformed = selected.loc[selected.feature_id.str.contains("sqrt|log1p")]
    assert transformed.value.eq(0).all() and transformed.status.eq("valid").all()
    normalized = selected.loc[selected.feature_id.str.contains("recording_")]
    assert normalized.value.isna().all() and normalized.status.eq("unavailable").all()


def test_seeded_scale_and_shift_invariants_and_small_log_amplitudes():
    rng = np.random.default_rng(271026)
    for _ in range(20):
        values = rng.uniform(.01, 5., 30)
        for statistic in ("skewness", "kurtosis"):
            original = distribution_summary(values, statistic)[0]
            moved = distribution_summary(values*7+20, statistic)[0]
            assert moved == pytest.approx(original, abs=5e-14)
        for kind in ("recording_z", "recording_max_fraction"):
            original, _ = transform_amplitudes(values, kind, amplitude_reference(values))
            scaled, _ = transform_amplitudes(values*1000, kind, amplitude_reference(values*1000))
            np.testing.assert_allclose(original, scaled, rtol=2e-13, atol=2e-14)
    small, _ = transform_amplitudes([1e-20], "log1p", amplitude_reference([1e-20]))
    assert small[0] == pytest.approx(1e-20, rel=1e-15, abs=0.)


def test_registry_freezes_definitions_and_preserves_legacy_disable():
    config = get_preset()
    config["scr"]["recovery_fraction_returned"] = .8
    registry = list_feature_definitions(config)
    assert COMMON_IDS <= registry.keys()
    assert len(COMMON_IDS) == 34
    assert registry["scr_recovery_time_sd"]["parameters"]["return_fraction"] == .8
    assert registry["scr_recovery_50_sd_s"]["parameters"]["return_fraction"] == .5
    assert registry["clean_kurtosis"]["parameters"]["fisher"] is True
    config["features"]["extended"]["enabled"] = False
    assert not COMMON_IDS.intersection(list_feature_definitions(config))


def test_pipeline_exports_new_selected_metrics_and_reference_columns(tmp_path):
    from eda_toolkit.demo import create_demo
    from eda_toolkit.pipeline import run_pipeline, resume_run
    create_demo(tmp_path / "demo", duration_s=120, sampling_rate_hz=4, seed=271026)
    config = resolve_config(tmp_path / "demo" / "single.json")
    selected = ["clean_mad", "tonic_p95", "scr_amplitude_log1p_mean", "scr_rise_time_sd",
                "event_amplitude_zero_sqrt_mean"]
    config["features"]["ids"] = selected
    config["output"].update(directory=str(tmp_path / "results"), formats=["csv", "json", "xlsx"])
    summary = run_pipeline(config)
    assert summary["status"] == "succeeded", summary
    from pathlib import Path
    root = Path(summary["directory"])
    definitions = json.loads((root / "feature_definitions.json").read_text(encoding="utf-8"))
    assert set(definitions) == set(selected)
    from eda_toolkit.core_export import verify_manifest, safe_name
    assert verify_manifest(root)["valid"]
    features = pd.read_csv(root / "tables" / f"{safe_name('features')}.csv")
    assert set(features.feature_id) == set(selected)
    extended = pd.read_csv(root / "tables" / f"{safe_name('scr_events_extended')}.csv")
    assert {"amplitude_uS", "amplitude_log1p", "amplitude_reference_max_uS"} <= set(extended)
    trial = pd.read_csv(root / "tables" / f"{safe_name('trial_table')}.csv")
    assert {"amplitude_zero_uS", "amplitude_zero_sqrt", "amplitude_reference_n"} <= set(trial)
    resumed = resume_run(root)
    assert resumed["status"] == "succeeded", resumed
    restored = pd.read_csv(Path(resumed["directory"]) / "tables" / f"{safe_name('features')}.csv")
    pd.testing.assert_frame_equal(features, restored)
