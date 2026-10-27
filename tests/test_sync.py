"""Small deterministic P07 checks for clock evidence and synchrony contracts."""

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.config import get_preset
from eda_toolkit.core_sync import (align_recordings, apply_clock_alignment,
                                   compute_pair_synchrony, estimate_sync_null,
                                   matched_pseudodyads, summarize_group_synchrony)
from eda_toolkit.contracts import make_record


def _config(**sync_changes):
    config = get_preset("group_sync")
    config["synchrony"].update({"analysis_rate_hz": 4.0, "analysis_context_s": 0.0, "window_s": 100.0,
                                "step_s": 100.0, "max_lag_s": 0.0,
                                "lag_step_s": 1.0, "min_samples": 8,
                                "min_coverage": 0.5, "null_replicates": 2,
                                "max_attempts": 4, "min_shift_s": 2.0})
    config["synchrony"].update(sync_changes)
    return config


def _raw_record(recording_id="r1", *, channel="eda", domain="device-local", fs=10.0, end_s=10.0):
    time = np.arange(0.0, end_s, 1.0 / fs)
    signal = np.sin(time)
    return make_record(time, signal, {"recording_id": recording_id, "channel": channel,
                                      "subject_id": "s1", "session_id": "ss1",
                                      "clock_domain": domain, "sampling_rate_hz": fs})


def _bundle(recording_id, subject_id, signal, *, session="ss1", group="g1", time=None,
            eligible=None, extra_metadata=None):
    time = np.arange(len(signal), dtype=float) / 4.0 if time is None else np.asarray(time, float)
    mask = np.ones(len(time), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    metadata = {"recording_id": recording_id, "channel": "eda", "subject_id": subject_id,
                "session_id": session, "clock_domain": "reference", "fs_reference_hz": 4.0}
    metadata.update(extra_metadata or {})
    membership = pd.DataFrame([{"group_id": group, "session_id": session, "subject_id": subject_id,
                                "recording_id": recording_id, "channel": "eda"}])
    return {"metadata": metadata, "time_s": time, "clean_uS": np.asarray(signal, float),
            "tonic_uS": np.asarray(signal, float), "phasic_uS": np.asarray(signal, float),
            "sampling_rate_hz": 4.0, "max_gap_s": 0.375,
            "quality": {"analysis_eligible": mask.copy(), "detector_eligible": mask.copy(),
                        "interpolated": np.zeros(len(time), bool), "artifact": np.zeros(len(time), bool),
                        "edge_affected": np.zeros(len(time), bool)},
            "detector_eligible": mask.copy(), "membership": membership}


def _aligned_pair(a, b, *, subjects=("s1", "s2"), group="g1", session="ss1", time=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    time = np.arange(len(a), dtype=float) / 4.0 if time is None else np.asarray(time, float)
    keys = ("r1::eda", "r2::eda")
    members = pd.DataFrame([
        {"group_id": group, "session_id": session, "subject_id": subjects[0], "recording_id": "r1", "channel": "eda"},
        {"group_id": group, "session_id": session, "subject_id": subjects[1], "recording_id": "r2", "channel": "eda"},
    ])
    return {"time_s": time, "signals": {keys[0]: a, keys[1]: b},
            "eligible": {keys[0]: np.ones(len(time), bool), keys[1]: np.ones(len(time), bool)},
            "metadata": {keys[0]: {"recording_id": "r1", "channel": "eda", "subject_id": subjects[0],
                                   "session_id": session, "clock_domain": "reference"},
                         keys[1]: {"recording_id": "r2", "channel": "eda", "subject_id": subjects[1],
                                   "session_id": session, "clock_domain": "reference"}},
            "membership": members, "basis": "phasic_uS", "unit": "uS",
            "alignment_report": {"formal_synchrony_allowed": True, "status": "aligned"}}


def test_clock_map_two_anchors_preserves_local_time_converts_tables_and_rate():
    record = _raw_record()
    original_time = record["time_s"].copy()
    events = pd.DataFrame([{"recording_id": "r1", "event_id": "e1", "onset_s": 2.0, "duration_s": 0.5}])
    conditions = pd.DataFrame([{"recording_id": "r1", "start_s": 1.0, "end_s": 3.0}])
    manual_qc = pd.DataFrame([{"recording_id": "r1", "start_s": 4.0, "end_s": 5.0}])
    anchors = pd.DataFrame([
        {"recording_id": "r1", "anchor_id": "a", "local_time_s": 0.0, "reference_time_s": 12.0, "reference_clock_domain": "study-clock"},
        {"recording_id": "r1", "anchor_id": "b", "local_time_s": 5.0, "reference_time_s": 22.0, "reference_clock_domain": "study-clock"},
    ])

    mapped, tables, report = apply_clock_alignment([record], {
        "sync_anchors": anchors, "events": events, "conditions": conditions,
        "manual_qc": manual_qc,
    }, _config())

    result = mapped[0]
    assert report["status"] == "aligned"
    assert report["records"][0]["fit_status"] == "two_anchors_no_empirical_uncertainty"
    assert report["records"][0]["uncertainty_status"] == "not_estimable_two_anchors"
    assert report["records"][0]["outliers_removed"] == 0
    assert result["metadata"]["clock_domain"] == "reference"
    assert result["metadata"]["clock_map_a"] == pytest.approx(12.0)
    assert result["metadata"]["clock_map_b"] == pytest.approx(2.0)
    assert result["metadata"]["fs_reference_hz"] == pytest.approx(5.0)
    assert result["metadata"]["fs_local_hz"] == pytest.approx(10.0)
    assert result["metadata"]["time_mapping_version"]
    assert len(result["metadata"]["anchor_sha256"]) == 64
    np.testing.assert_array_equal(result["local_time_s"], original_time)
    np.testing.assert_allclose(result["time_s"], 12 + 2 * original_time)
    np.testing.assert_array_equal(record["time_s"], original_time)
    assert tables["events"].loc[0, "onset_s"] == pytest.approx(16.0)
    assert tables["events"].loc[0, "local_onset_s"] == pytest.approx(2.0)
    assert tables["events"].loc[0, "duration_s"] == pytest.approx(1.0)
    assert tables["events"].loc[0, "clock_domain"] == "reference"
    assert tables["events"].loc[0, "local_clock_domain"] == "device-local"
    assert tables["conditions"].loc[0, "start_s"] == pytest.approx(14.0)
    assert tables["conditions"].loc[0, "end_s"] == pytest.approx(18.0)
    assert tables["manual_qc"].loc[0, "start_s"] == pytest.approx(20.0)
    assert events.loc[0, "onset_s"] == pytest.approx(2.0)
    mapped_again, tables_again, second_report = apply_clock_alignment(mapped, tables, _config())
    np.testing.assert_array_equal(mapped_again[0]["time_s"], result["time_s"])
    assert mapped_again[0]["metadata"]["clock_map_b"] == pytest.approx(2.0)
    assert mapped_again[0]["metadata"]["anchor_sha256"] == result["metadata"]["anchor_sha256"]
    assert mapped_again[0]["metadata"]["fs_reference_hz"] == pytest.approx(5.0)
    assert tables_again["events"].loc[0, "onset_s"] == pytest.approx(16.0)
    assert second_report["records"][0]["status"] == "already_reference"


def test_one_anchor_is_offset_only_and_no_anchors_is_unverified():
    record = _raw_record(end_s=2.0)
    anchors = pd.DataFrame([{"recording_id": "r1", "anchor_id": "a",
                             "local_time_s": 1.0, "reference_time_s": 5.0,
                             "reference_clock_domain": "study-clock"}])
    mapped, _, report = apply_clock_alignment([record], {"sync_anchors": anchors}, _config())
    assert report["records"][0]["fit_status"] == "one_anchor_offset_only"
    assert report["records"][0]["b"] == pytest.approx(1.0)
    np.testing.assert_allclose(mapped[0]["time_s"], record["time_s"] + 4.0)

    untouched, _, missing_report = apply_clock_alignment([record], {}, _config())
    assert missing_report["status"] == "unverified"
    assert missing_report["formal_synchrony_allowed"] is False
    assert untouched[0]["metadata"]["clock_alignment_status"] == "unverified"
    np.testing.assert_array_equal(untouched[0]["time_s"], record["time_s"])


def test_three_anchor_ols_keeps_outlier_and_reports_residuals():
    record = _raw_record(end_s=10.0)
    anchors = pd.DataFrame([
        {"recording_id": "r1", "anchor_id": "a", "local_time_s": 0.0, "reference_time_s": 1.0, "reference_clock_domain": "study-clock"},
        {"recording_id": "r1", "anchor_id": "b", "local_time_s": 4.0, "reference_time_s": 9.0, "reference_clock_domain": "study-clock"},
        {"recording_id": "r1", "anchor_id": "c", "local_time_s": 8.0, "reference_time_s": 17.5, "reference_clock_domain": "study-clock"},
    ])
    _, _, report = apply_clock_alignment([record], {"sync_anchors": anchors}, _config())
    result = report["records"][0]
    assert result["fit_status"] == "ols_all_anchors"
    assert result["anchor_count"] == 3
    assert result["outliers_removed"] == 0
    assert result["residual_rms_s"] > 0
    assert result["anchor_span_s"] == pytest.approx(8.0)
    assert result["extrapolated_sample_fraction"] > 0


def test_mismatched_associated_table_clock_raises_instead_of_using_local_seconds_as_reference():
    record = _raw_record(domain="device", fs=1.0, end_s=10.0)
    anchors = pd.DataFrame([
        {"recording_id": "r1", "anchor_id": "a", "local_time_s": 0.0,
         "reference_time_s": 100.0, "reference_clock_domain": "session-reference"},
        {"recording_id": "r1", "anchor_id": "b", "local_time_s": 10.0,
         "reference_time_s": 110.0, "reference_clock_domain": "session-reference"},
    ])
    events = pd.DataFrame([{"recording_id": "r1", "event_id": "e1", "onset_s": 5.0,
                            "clock_domain": "other"}])

    with pytest.raises(ValueError) as exc:
        apply_clock_alignment([record], {"sync_anchors": anchors, "events": events}, _config())

    assert exc.value.eda_details["code"] == "sync_table_clock_unresolved"
    assert events.loc[0, "onset_s"] == pytest.approx(5.0)


def test_shared_clock_requires_explicit_flag_and_reference_metadata_is_not_remapped():
    record = _raw_record(domain="reference")
    record["time_s"] += 100.0
    already, _, report = apply_clock_alignment([record], {}, _config())
    np.testing.assert_array_equal(already[0]["time_s"], record["time_s"])
    assert report["records"][0]["status"] == "already_reference"

    local = _raw_record()
    not_shared, _, report = apply_clock_alignment([local], {}, _config())
    assert report["status"] == "unverified"
    verified_config = _config(shared_clock_verified=True)
    shared, _, report = apply_clock_alignment([local], {}, verified_config)
    assert report["status"] == "aligned"
    assert report["records"][0]["status"] == "shared_clock_verified"
    np.testing.assert_array_equal(shared[0]["time_s"], local["time_s"])


def test_direct_alignment_maps_local_bundle_with_anchors_and_reference_rate():
    local_time = np.arange(0.0, 10.0, 0.1)
    local_signal = np.sin(2 * np.pi * 0.2 * local_time)
    local = _bundle("r1", "s1", local_signal, time=local_time)
    local["metadata"].update({"clock_domain": "device-local", "sampling_rate_hz": 10.0})
    local["metadata"].pop("fs_reference_hz")
    local["metadata"].pop("fs_hz", None)
    ref_time = np.arange(12.0, 32.0, 0.2)
    reference = _bundle("r2", "s2", np.sin(2 * np.pi * 0.2 * ((ref_time - 12) / 2)),
                        time=ref_time, extra_metadata={"fs_reference_hz": 5.0})
    reference["sampling_rate_hz"] = 5.0
    anchors = pd.DataFrame([
        {"recording_id": "r1", "anchor_id": "a", "local_time_s": 0.0, "reference_time_s": 12.0, "reference_clock_domain": "study-clock"},
        {"recording_id": "r1", "anchor_id": "b", "local_time_s": 10.0, "reference_time_s": 32.0, "reference_clock_domain": "study-clock"},
    ])
    config = _config(analysis_rate_hz=4.0, window_s=10.0)
    aligned, report = align_recordings([local, reference], anchors=anchors, config=config)
    assert report["status"] == "aligned"
    assert report["maps"][0]["b"] == pytest.approx(2.0)
    assert aligned["metadata"]["r1::eda"]["fs_reference_hz"] == pytest.approx(5.0)
    assert aligned["time_s"][0] >= 12.0
    assert aligned["time_s"][-1] <= 32.0


def test_alignment_and_pair_lag_signs_include_both_signed_directions():
    rng = np.random.default_rng(31)
    a = rng.normal(size=400)
    delay = 8  # two seconds at 4 Hz
    delayed = np.r_[rng.normal(size=delay), a[:-delay]]
    leading = np.r_[a[delay:], rng.normal(size=delay)]
    config = _config(window_s=100.0, max_lag_s=3.0, lag_step_s=1.0,
                     min_samples=20, min_coverage=0.5, metric="pearson")
    aligned = _aligned_pair(a, delayed)
    values, coverage = compute_pair_synchrony(aligned, None, config)
    best = values.loc[values["status"] == "valid"].iloc[values.loc[values["status"] == "valid", "r"].abs().argmax()]
    assert best["lag_s"] == pytest.approx(2.0)
    assert best["r"] > 0.99
    assert (values["lag_s"] == 0).any()
    assert coverage["zero_lag_present"].all()

    aligned_negative = _aligned_pair(a, leading)
    values_negative, _ = compute_pair_synchrony(aligned_negative, None, config)
    best_negative = values_negative.loc[values_negative["status"] == "valid"].iloc[
        values_negative.loc[values_negative["status"] == "valid", "r"].abs().argmax()]
    assert best_negative["lag_s"] == pytest.approx(-2.0)
    assert best_negative["r"] > 0.99


def test_constant_series_is_unavailable_and_group_summary_requires_fixed_edges():
    config = _config(max_lag_s=0.0, min_samples=8)
    aligned = _aligned_pair(np.ones(400), np.arange(400.0))
    values, _ = compute_pair_synchrony(aligned, None, config)
    assert values.iloc[0]["status"] == "unavailable"
    assert values.iloc[0]["reason"] == "constant_series"
    summary = summarize_group_synchrony(values, aligned, config)
    assert summary.iloc[0]["status"] == "unavailable"
    assert summary.iloc[0]["aggregation"] == "fisher_z_mean"


def test_insufficient_accepted_surrogates_do_not_emit_a_tail_probability():
    rng = np.random.default_rng(17)
    aligned = _aligned_pair(rng.normal(size=1200), rng.normal(size=1200))
    config = _config(window_s=100.0, max_lag_s=0.0, min_shift_s=5.0,
                     analysis_intervals_s=[[100.0, 200.0]], analysis_context_s=100.0,
                     null_replicates=2, max_attempts=1, min_samples=20)
    summary, tails = estimate_sync_null(aligned, None, config, seed=9)
    assert not summary.empty
    assert summary["accepted_replicates"].lt(2).all()
    assert summary["status"].eq("insufficient_accepted_replicates").all()
    assert tails["surrogate_tail_probability"].isna().all()
    assert tails["status"].eq("insufficient_accepted_replicates").all()


def test_missing_fixed_group_member_makes_group_result_unavailable():
    aligned = _aligned_pair(np.arange(80.0), np.arange(80.0) ** 2)
    aligned["membership"] = pd.concat([aligned["membership"], pd.DataFrame([{
        "group_id": "g1", "session_id": "ss1", "subject_id": "s3",
        "recording_id": "missing", "channel": "eda",
    }])], ignore_index=True)
    config = _config(window_s=20.0, max_lag_s=0.0, min_samples=8)
    values, _ = compute_pair_synchrony(aligned, None, config)
    assert not values.empty
    assert values["status"].eq("unavailable").all()
    assert values["reason"].eq("membership_record_missing").all()
    group = summarize_group_synchrony(values, aligned, config)
    assert group["status"].eq("unavailable").all()


def test_null_reports_joint_shift_infeasible_without_claiming_a_probability():
    n = 100
    aligned = _aligned_pair(np.arange(n, dtype=float), np.arange(n, dtype=float)[::-1],
                            time=np.arange(n, dtype=float))
    config = _config(analysis_rate_hz=1.0, window_s=100.0, step_s=100.0,
                     max_lag_s=0.0, min_shift_s=50.0, null_replicates=5,
                     max_attempts=20, min_samples=8)
    summary, tails = estimate_sync_null(aligned, None, config, seed=11)
    assert summary.iloc[0]["status"] == "unavailable"
    assert summary.iloc[0]["reason"] == "joint_shift_infeasible"
    assert summary.iloc[0]["attempted_replicates"] == 0
    assert summary.iloc[0]["preflight_status"] == "joint_shift_infeasible"
    assert summary.iloc[0]["preflight_proposals"] == 1
    assert tails.empty
    assert "surrogate_tail_probability" in tails.columns


def test_buffered_analysis_interval_accepts_requested_surrogate_count():
    rng = np.random.default_rng(110)
    aligned = _aligned_pair(rng.normal(size=800), rng.normal(size=800),
                            time=np.arange(800, dtype=float))
    config = _config(analysis_rate_hz=1.0, analysis_intervals_s=[[100.0, 200.0]],
                     analysis_context_s=100.0, window_s=50.0, step_s=50.0,
                     max_lag_s=0.0, min_shift_s=10.0, null_replicates=4,
                     max_attempts=100, min_samples=20, min_coverage=0.5)
    summary, tails = estimate_sync_null(aligned, None, config, seed=13)
    assert summary["accepted_replicates"].eq(4).all()
    assert summary["attempted_replicates"].le(100).all()
    assert summary["status"].eq("experimental_uncalibrated").all()
    assert summary["analysis_intervals_s"].iloc[0] == [[100.0, 200.0]]
    assert summary["analysis_context_s"].iloc[0] == pytest.approx(100.0)
    assert summary["sampling_with_replacement"].all()
    assert summary["proposal_count"].ge(summary["attempted_replicates"]).all()
    assert summary["support_policy_version"].eq("participant-required-mask-seam-v1").all()
    assert tails["surrogate_tail_probability"].notna().all()
    short_config = _config(analysis_rate_hz=1.0, analysis_intervals_s=[[100.0, 120.0]],
                           analysis_context_s=100.0, window_s=50.0, step_s=50.0,
                           max_lag_s=0.0, min_samples=8)
    short_values, _ = compute_pair_synchrony(aligned, None, short_config)
    assert short_values["reason"].eq("analysis_interval_too_short").all()


def test_negative_coupling_uses_distinct_two_sided_greater_and_less_searches():
    rng = np.random.default_rng(710)
    signal = rng.normal(size=200)
    aligned = _aligned_pair(signal, -signal, time=np.arange(200, dtype=float))
    common = {"analysis_rate_hz": 1.0, "analysis_intervals_s": [[50.0, 150.0]],
              "analysis_context_s": 50.0, "window_s": 100.0, "step_s": 100.0,
              "max_lag_s": 0.0, "min_shift_s": 10.0, "null_replicates": 8,
              "max_attempts": 200, "min_samples": 20, "min_coverage": 0.5}
    results = {}
    for tail in ("two-sided", "greater", "less"):
        summary, tails = estimate_sync_null(aligned, None, _config(**common, tail=tail), seed=5)
        assert summary["accepted_replicates"].eq(8).all()
        assert summary["selected_r"].iloc[0] == pytest.approx(-1.0)
        results[tail] = (summary.iloc[0], tails.iloc[0])
    assert results["two-sided"][0]["observed_statistic"] == pytest.approx(1.0)
    assert results["greater"][0]["observed_statistic"] == pytest.approx(-1.0)
    assert results["less"][0]["observed_statistic"] == pytest.approx(-1.0)
    assert np.isnan(results["greater"][0]["observed_max_abs_r"])
    assert results["greater"][1]["surrogate_tail_probability"] == pytest.approx(1.0)
    assert results["less"][1]["surrogate_tail_probability"] < results["greater"][1]["surrogate_tail_probability"]


def test_pseudodyad_missing_metadata_never_falls_back_to_circular_shift(monkeypatch):
    import eda_toolkit.core_sync as core_sync

    aligned = _aligned_pair(np.arange(100.0), np.arange(100.0) ** 2, time=np.arange(100, dtype=float))
    config = _config(null_method="pseudodyad", pseudodyad_exchangeability=True,
                     analysis_intervals_s=[[10.0, 80.0]], window_s=30.0,
                     null_replicates=3, max_attempts=10)
    monkeypatch.setattr(core_sync, "_circular_shift_bundle",
                        lambda *args, **kwargs: pytest.fail("pseudodyad must not run circular shifts"))
    summary, tails = estimate_sync_null(aligned, None, config, seed=3)
    assert summary.iloc[0]["status"] == "unavailable"
    assert summary.iloc[0]["reason"] == "pseudodyad_matching_metadata_missing"
    assert tails.empty


def test_complete_matched_pseudodyad_world_reassigns_independent_group_donors():
    rng = np.random.default_rng(99)
    time = np.arange(80.0)
    signals, eligible, metadata, members = {}, {}, {}, []
    for group_index in range(4):
        group = f"g{group_index}"
        for slot_index, role in enumerate(("lead", "follow")):
            record = f"{group}-r{slot_index}"
            subject = f"{group}-s{slot_index}"
            key = f"{record}::eda"
            signals[key] = rng.normal(size=len(time))
            eligible[key] = np.ones(len(time), dtype=bool)
            metadata[key] = {"recording_id": record, "channel": "eda", "subject_id": subject,
                             "session_id": "V01", "clock_domain": "reference",
                             "time_basis": "task_relative"}
            members.append({"group_id": group, "session_id": "V01", "subject_id": subject,
                            "recording_id": record, "channel": "eda", "role": role,
                            "task": "joint-task", "stimulus": "stim-A", "timeline_id": "timeline-1",
                            "time_basis": "task_relative"})
    aligned = {"time_s": time, "signals": signals, "eligible": eligible, "metadata": metadata,
               "membership": pd.DataFrame(members), "basis": "phasic_uS", "unit": "uS",
               "alignment_report": {"formal_synchrony_allowed": True, "status": "aligned"}}
    config = _config(null_method="pseudodyad", pseudodyad_exchangeability=True,
                     analysis_intervals_s=[[10.0, 70.0]], analysis_context_s=0.0,
                     analysis_rate_hz=1.0, window_s=20.0, step_s=20.0, max_lag_s=0.0,
                     min_samples=8, min_coverage=0.5, null_replicates=3, max_attempts=20)
    candidates, diagnostics = matched_pseudodyads(aligned, config)
    assert diagnostics["status"] == "available"
    assert not candidates.empty
    assert candidates["session_id_a"].eq("V01").all()
    assert candidates["session_id_b"].eq("V01").all()
    summary, tails = estimate_sync_null(aligned, None, config, seed=12)
    assert summary["status"].eq("experimental_uncalibrated").all()
    assert summary["accepted_replicates"].eq(3).all()
    assert summary["finite_support_size"].gt(0).all()
    assert summary["unique_worlds"].ge(1).all()
    assert summary["proposal_count"].ge(summary["attempted_replicates"]).all()
    assert summary["support_policy_version"].eq("complete-block-matched-donor-v1").all()
    assert '"donor_record_key"' in tails["provenance_json"].iloc[0]


def test_align_recordings_uses_reference_grid_and_preserves_masks():
    time = np.arange(0.0, 30.0, 0.25)
    signal = np.sin(2 * np.pi * 0.2 * time)
    left = _bundle("r1", "s1", signal, time=time)
    right_mask = np.ones(len(time), bool)
    right_mask[40:44] = False
    right = _bundle("r2", "s2", signal, time=time, eligible=right_mask)
    config = _config(window_s=20.0, analysis_rate_hz=4.0)
    aligned, report = align_recordings([left, right], config=config)
    assert report["status"] == "aligned"
    assert len(aligned["time_s"]) > 0
    assert aligned["signals"]["r1::eda"].shape == aligned["time_s"].shape
    assert not aligned["eligible"]["r2::eda"][40:44].any()
    assert aligned["membership"].shape[0] == 2


def test_alignment_downsampling_antialiases_above_target_nyquist():
    time = np.arange(0.0, 20.0, 0.05)
    signal = np.sin(2 * np.pi * 8.0 * time)  # Above the 2 Hz target Nyquist.
    bundle = _bundle("r1", "s1", signal, time=time)
    bundle["metadata"].update({"fs_reference_hz": 20.0, "fs_hz": 20.0})
    bundle["sampling_rate_hz"] = 20.0
    bundle["max_gap_s"] = 0.075
    aligned, report = align_recordings([bundle], config=_config(analysis_rate_hz=4.0))
    values = aligned["signals"]["r1::eda"][aligned["eligible"]["r1::eda"]]
    assert report["status"] == "aligned"
    assert len(values) > 20
    assert float(np.std(values)) < 0.03


def test_matched_pseudodyads_require_task_stimulus_role_and_timeline_metadata():
    aligned = _aligned_pair(np.arange(20.0), np.arange(20.0), session="ss1")
    table, diagnostics = matched_pseudodyads(aligned, _config())
    assert table.empty
    assert diagnostics["status"] == "unavailable"
    assert "role_or_member_slot" in diagnostics["missing_fields"]
    assert any("timeline_id" in item for item in diagnostics["missing_fields"])

