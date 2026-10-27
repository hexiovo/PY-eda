import math

import numpy as np
import pytest

from eda_toolkit.metrics_ids import autocorrelation_feature_id
from eda_toolkit.metrics_spectral_complexity import compute_clean_advanced_window


def _window(values, fs_hz=16.0, *, time_s=None, eligible=None, max_gap_s=None,
            jitter_tolerance=0.01, extended=None, selected_ids=None, progress=None,
            cancel_callback=None):
    values = np.asarray(values, dtype=np.float64)
    time_s = np.arange(len(values), dtype=np.float64) / fs_hz if time_s is None else np.asarray(time_s, dtype=np.float64)
    eligible = np.ones(len(values), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if max_gap_s is None:
        max_gap_s = 1.5 / fs_hz
    return compute_clean_advanced_window(
        time_s, values, eligible,
        window_start_s=float(time_s[0]) if len(time_s) else 0.0,
        window_end_s=(float(time_s[-1]) + 1 / fs_hz) if len(time_s) else 1.0,
        declared_fs_hz=fs_hz, max_gap_s=max_gap_s, jitter_tolerance=jitter_tolerance,
        extended_config=extended or {}, selected_ids=selected_ids or [],
        identity={"recording_id": "R001", "window_id": "R001:whole"},
        cancel_callback=cancel_callback, progress_callback=progress,
    )


def _spectral_config(*, band=(0.5, 1.5), normalization=(0.25, 1.75),
                     segment_s=8.0, overlap=0.0, window="boxcar"):
    return {"spectral": {
        "enabled": True, "basis": "clean", "segment_s": segment_s,
        "overlap": overlap, "window": window, "detrend": "constant",
        "min_segments": 2, "min_cycles": 2.0,
        "bands": [{"id": "known_tone", "low_hz": band[0], "high_hz": band[1]}],
        "normalization_band_hz": list(normalization),
    }}


def _complexity_config(**overrides):
    config = {
        "enabled": True, "min_samples": 200, "max_samples": 5000,
        "max_pair_comparisons": 25_000_000,
        "sampen": {"m": 2, "delay": 1, "r_mode": "sd_fraction", "r": 0.2},
        "permen": {"dimension": 3, "delay": 1, "normalize": True, "ties": "stable"},
        "dfa": {"order": 1, "min_samples": 256, "min_scale": 4,
                "max_scale_fraction": 0.25, "scale_count": 10, "min_scales": 6},
    }
    for key, value in overrides.items():
        config[key] = value
    return {"complexity": config}


def test_welch_sinusoid_matches_hand_integrated_band_and_normalized_power():
    fs = 16.0
    n = 256
    time = np.arange(n) / fs
    values = 2.0 * np.cos(2 * np.pi * 1.0 * time)
    result = _window(
        values, fs, extended=_spectral_config(),
        selected_ids=["clean_bandpower_known_tone_uS2", "clean_bandpower_known_tone_normalized"],
    )
    metrics = result["metrics"]
    assert metrics["clean_bandpower_known_tone_uS2"]["status"] == "valid"
    assert metrics["clean_bandpower_known_tone_uS2"]["value"] == pytest.approx(2.0, abs=1e-12)
    assert metrics["clean_bandpower_known_tone_normalized"]["status"] == "valid"
    assert metrics["clean_bandpower_known_tone_normalized"]["value"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["clean_bandpower_known_tone_normalized"]["unit"] == "fraction"
    assert metrics["clean_bandpower_known_tone_uS2"]["basis"] == "clean_uS"
    spectra = result["tables"]["spectra"]
    assert len(spectra) == 65
    peak = spectra.loc[np.isclose(spectra["frequency_hz"], 1.0)].iloc[0]
    assert peak["psd_uS2_per_hz"] == pytest.approx(16.0, abs=1e-12)
    assert peak["nperseg"] == 128 and peak["noverlap"] == 0 and peak["nfft"] == 128
    assert peak["recording_id"] == "R001" and peak["window_id"] == "R001:whole"


def test_welch_requires_two_segments_and_exact_nyquist_endpoint_is_supported():
    fs = 16.0
    short = np.cos(2 * np.pi * np.arange(255) / fs)
    short_result = _window(short, fs, extended=_spectral_config(),
                           selected_ids=["clean_bandpower_known_tone_uS2"])
    short_metric = short_result["metrics"]["clean_bandpower_known_tone_uS2"]
    assert short_metric["status"] == "unavailable"
    assert short_metric["reason"] == "insufficient_welch_segments"
    assert short_result["tables"]["spectra"].empty

    nyquist_config = _spectral_config(band=(7.0, 8.0), normalization=(6.0, 8.0))
    values = np.cos(2 * np.pi * np.arange(256) / 16.0)
    nyquist_result = _window(values, 16.0, extended=nyquist_config,
                             selected_ids=["clean_bandpower_known_tone_uS2"])
    nyquist_metric = nyquist_result["metrics"]["clean_bandpower_known_tone_uS2"]
    assert nyquist_metric["status"] == "valid"
    assert np.isfinite(nyquist_metric["value"])


def test_normalization_conditions_do_not_block_absolute_band_power():
    config = _spectral_config()
    config["spectral"]["normalization_band_hz"] = [0.5, 0.5]
    result = _window(
        2.0 * np.cos(2 * np.pi * np.arange(256) / 16.0),
        16.0, extended=config,
        selected_ids=["clean_bandpower_known_tone_uS2", "clean_bandpower_known_tone_normalized"],
    )
    metrics = result["metrics"]
    assert metrics["clean_bandpower_known_tone_uS2"]["status"] == "valid"
    assert metrics["clean_bandpower_known_tone_uS2"]["value"] == pytest.approx(2.0, abs=1e-12)
    assert metrics["clean_bandpower_known_tone_normalized"]["status"] == "unavailable"
    assert metrics["clean_bandpower_known_tone_normalized"]["reason"] == "invalid_normalization_band"

    insufficient_cycles = _spectral_config(normalization=(0.01, 0.5))
    result = _window(
        2.0 * np.cos(2 * np.pi * np.arange(256) / 16.0),
        16.0, extended=insufficient_cycles,
        selected_ids=["clean_bandpower_known_tone_uS2", "clean_bandpower_known_tone_normalized"],
    )
    metrics = result["metrics"]
    assert metrics["clean_bandpower_known_tone_uS2"]["status"] == "valid"
    assert metrics["clean_bandpower_known_tone_normalized"]["status"] == "unavailable"
    assert metrics["clean_bandpower_known_tone_normalized"]["reason"] == "insufficient_cycles_in_normalization_band"


def test_autocorrelation_is_pearson_on_exact_sample_lag_and_rejects_off_grid():
    fs = 4.0
    index = np.arange(100, dtype=np.float64)
    values = np.sin(index * 0.33) + 0.1 * np.cos(index * 0.13)
    lag_id = autocorrelation_feature_id(1.0)
    result = _window(values, fs, extended={"autocorrelation": {"enabled": True, "lags_s": [1.0], "min_pairs": 20}},
                     selected_ids=[lag_id])
    expected = np.corrcoef(values[:-4], values[4:])[0, 1]
    metric = result["metrics"][lag_id]
    assert metric["status"] == "valid"
    assert metric["unit"] == "correlation"
    assert metric["value"] == pytest.approx(expected, abs=1e-14)
    diag = result["tables"]["complexity_diagnostics"].iloc[0]
    assert diag["family"] == "autocorrelation"
    params = __import__("json").loads(diag["parameters_json"])
    assert params["actual_lag_s"] == 1.0 and params["valid_pairs"] == 96

    off_grid_id = autocorrelation_feature_id(0.3)
    off_grid = _window(values, fs,
                       extended={"autocorrelation": {"enabled": True, "lags_s": [0.3], "min_pairs": 20}},
                       selected_ids=[off_grid_id])["metrics"][off_grid_id]
    assert off_grid["status"] == "unavailable"
    assert off_grid["reason"] == "lag_grid_mismatch"


def test_autocorrelation_constant_and_short_pair_support_are_unavailable():
    fs = 4.0
    feature_id = autocorrelation_feature_id(1.0)
    constant = _window(np.ones(100), fs,
                       extended={"autocorrelation": {"enabled": True, "lags_s": [1.0], "min_pairs": 20}},
                       selected_ids=[feature_id])["metrics"][feature_id]
    assert constant["status"] == "unavailable" and constant["reason"] == "zero_variance"
    too_short = _window(np.arange(10, dtype=float), fs,
                        extended={"autocorrelation": {"enabled": True, "lags_s": [1.0], "min_pairs": 20}},
                        selected_ids=[feature_id])["metrics"][feature_id]
    assert too_short["reason"] == "insufficient_pairs"


def test_sampen_exact_common_start_counts_match_independent_oracle():
    values = np.tile(np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64), 25)
    result = _window(values, 4.0, extended=_complexity_config(), selected_ids=["clean_sample_entropy"])
    metric = result["metrics"]["clean_sample_entropy"]
    assert metric["status"] == "valid"
    assert metric["value"] == pytest.approx(0.5188995491723452, abs=1e-14)
    diagnostics = __import__("json").loads(result["tables"]["complexity_diagnostics"].iloc[0]["diagnostics_json"])
    assert diagnostics["common_start_count_M"] == 198
    assert diagnostics["B_m_matches"] == 6052
    assert diagnostics["A_m_plus_1_matches"] == 3602
    assert diagnostics["comparison_estimate"] == 58509


def _de_bruijn(alphabet_size, order):
    a = [0] * (alphabet_size * order)
    sequence = []

    def visit(t, period):
        if t > order:
            if order % period == 0:
                sequence.extend(a[1:period + 1])
            return
        a[t] = a[t - period]
        visit(t + 1, period)
        for value in range(a[t - period] + 1, alphabet_size):
            a[t] = value
            visit(t + 1, t)

    visit(1, 1)
    return sequence


def test_sampen_zero_extension_match_and_comparison_budget_are_structured_na():
    values = np.asarray(_de_bruijn(10, 3)[:200], dtype=np.float64)
    metric = _window(values, 4.0, extended=_complexity_config(),
                     selected_ids=["clean_sample_entropy"])["metrics"]["clean_sample_entropy"]
    assert metric["status"] == "unavailable"
    assert metric["reason"] == "zero_m_plus_1_matches"

    large = np.tile(np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64), 625)
    budget = _window(large, 4.0, extended=_complexity_config(),
                     selected_ids=["clean_sample_entropy"])["metrics"]["clean_sample_entropy"]
    assert budget["status"] == "unavailable"
    assert budget["reason"] == "resource_budget"
    assert budget["parameters"]["comparison_estimate"] == 37_462_509


def test_permutation_entropy_normalizes_and_audits_stable_and_rejected_ties():
    values = np.arange(200, dtype=np.float64) % 3
    config = _complexity_config()
    no_ties = _window(values, 4.0, extended=config, selected_ids=["clean_permutation_entropy"])
    metric = no_ties["metrics"]["clean_permutation_entropy"]
    assert metric["status"] == "valid"
    assert metric["value"] == pytest.approx(0.6131471927654585, abs=2e-15)
    params = __import__("json").loads(no_ties["tables"]["complexity_diagnostics"].iloc[0]["diagnostics_json"])
    assert params["tie_vector_count"] == 0
    assert params["pattern_count"] == 198

    tied_values = np.tile(np.asarray([0.0, 0.0, 1.0]), 67)[:200]
    stable = _window(tied_values, 4.0, extended=config, selected_ids=["clean_permutation_entropy"])
    stable_metric = stable["metrics"]["clean_permutation_entropy"]
    assert stable_metric["status"] == "valid"
    assert stable_metric["value"] == pytest.approx(0.6131471927654585, abs=2e-15)
    stable_params = __import__("json").loads(stable["tables"]["complexity_diagnostics"].iloc[0]["diagnostics_json"])
    assert stable_params["tie_vector_count"] == 198
    rejected_config = _complexity_config()
    rejected_config["complexity"]["permen"]["ties"] = "reject"
    rejected = _window(tied_values, 4.0, extended=rejected_config,
                       selected_ids=["clean_permutation_entropy"])["metrics"]["clean_permutation_entropy"]
    assert rejected["status"] == "unavailable" and rejected["reason"] == "ties_rejected"


def test_dfa_scales_fluctuations_and_known_alpha_match_oracle():
    values = (-1.0) ** np.arange(256)
    result = _window(values, 4.0, extended=_complexity_config(), selected_ids=["clean_dfa_alpha"])
    metric = result["metrics"]["clean_dfa_alpha"]
    assert metric["status"] == "valid"
    assert metric["value"] == pytest.approx(0.023595950948563836, abs=2e-14)
    diagnostics = __import__("json").loads(result["tables"]["complexity_diagnostics"].iloc[0]["diagnostics_json"])
    assert diagnostics["scales"] == [4, 5, 7, 10, 14, 19, 25, 35, 47, 64]
    assert diagnostics["box_counts"] == [64, 51, 36, 25, 18, 13, 10, 7, 5, 4]
    assert diagnostics["F_s"][0] == pytest.approx(0.4472135954999579, abs=3e-15)

    sinusoid_mix = np.sin(2 * np.pi * np.arange(256) / 17) + 0.1 * np.sin(2 * np.pi * np.arange(256) / 7)
    mixed = _window(sinusoid_mix, 4.0, extended=_complexity_config(), selected_ids=["clean_dfa_alpha"])
    assert mixed["metrics"]["clean_dfa_alpha"]["value"] == pytest.approx(0.9664610656592191, abs=2e-12)
    short = _window(np.arange(255, dtype=np.float64), 4.0,
                    extended=_complexity_config(), selected_ids=["clean_dfa_alpha"])
    assert short["metrics"]["clean_dfa_alpha"]["reason"] == "below_min_samples"


def test_constant_complexity_metrics_are_unavailable_not_zero_or_infinity():
    values = np.full(256, 3.0)
    result = _window(values, 4.0, extended=_complexity_config(),
                     selected_ids=["clean_sample_entropy", "clean_permutation_entropy", "clean_dfa_alpha"])
    assert {metric["status"] for metric in result["metrics"].values()} == {"unavailable"}
    assert {metric["reason"] for metric in result["metrics"].values()} == {"zero_variance"}
    assert all(metric["value"] is None for metric in result["metrics"].values())


def test_unavailable_unnormalized_permutation_entropy_retains_nats_unit():
    config = _complexity_config()
    config["complexity"]["enabled"] = False
    config["complexity"]["permen"]["normalize"] = False
    metric = _window(np.arange(200, dtype=np.float64), 4.0,
                     extended=config, selected_ids=["clean_permutation_entropy"])["metrics"]["clean_permutation_entropy"]
    assert metric["status"] == "unavailable"
    assert metric["reason"] == "family_disabled"
    assert metric["unit"] == "nats"


def test_gapped_ineligible_or_irregular_support_is_never_joined_or_dropped():
    fs = 16.0
    values = np.sin(2 * np.pi * np.arange(256) / 16)
    time = np.arange(256) / fs
    time[100:] += 0.1
    gap = _window(values, fs, time_s=time, max_gap_s=0.09,
                  extended=_spectral_config(), selected_ids=["clean_bandpower_known_tone_uS2"])
    assert gap["metrics"]["clean_bandpower_known_tone_uS2"]["reason"] == "gap_exceeds_max_gap_s"
    assert gap["tables"]["spectra"].empty

    time = np.arange(256) / fs
    eligible = np.ones(256, dtype=bool)
    eligible[90] = False
    invalid = _window(values, fs, time_s=time, eligible=eligible,
                      extended=_spectral_config(), selected_ids=["clean_bandpower_known_tone_uS2"])
    assert invalid["metrics"]["clean_bandpower_known_tone_uS2"]["reason"] == "ineligible_samples_in_window"
    assert invalid["tables"]["spectra"].empty

    irregular_time = np.arange(256) / fs
    irregular_time[100:] += 0.002
    irregular = _window(values, fs, time_s=irregular_time, max_gap_s=0.1,
                        extended=_spectral_config(), selected_ids=["clean_bandpower_known_tone_uS2"])
    assert irregular["metrics"]["clean_bandpower_known_tone_uS2"]["reason"] == "irregular_sampling_grid"


def test_cancel_is_observed_inside_sampen_blocks_and_progress_only_completes_real_family():
    values = np.tile(np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64), 25)
    events = []
    result = _window(values, 4.0, extended=_complexity_config(),
                     selected_ids=["clean_sample_entropy"], progress=events.append)
    assert result["metrics"]["clean_sample_entropy"]["status"] == "valid"
    assert [event["kind"] for event in events] == ["work_started", "completed"]
    assert events[-1]["work_id"].endswith(":clean_complexity")
    assert events[-1]["duration_s"] >= 0
    assert events[-1]["details"]["duration_s"] == events[-1]["duration_s"]

    with pytest.raises(InterruptedError):
        _window(values, 4.0, extended=_complexity_config(),
                selected_ids=["clean_sample_entropy"], cancel_callback=lambda: True)
