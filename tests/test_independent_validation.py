"""Independent contract counterexamples; hand/model oracles, no scientific qualification."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.config import get_preset, resolve_config
from eda_toolkit.contracts import make_record
from eda_toolkit.core_statistics import run_statistics
from eda_toolkit.core_sync import apply_clock_alignment


def _mixed_frame():
    rng = np.random.default_rng(2718)
    rows = []
    for subject in range(16):
        intercept = rng.normal(0, 1.2)
        for condition in ("A", "B"):
            rows.append({"subject_id": str(subject), "condition": condition,
                         "good": intercept + (condition == "B") + rng.normal(0, .2)})
    data = pd.DataFrame(rows)
    data["bad"] = data["good"].where(data.condition.eq("A"))
    return data


def test_mixed_model_does_not_pool_undeclared_condition_with_reference():
    data = _mixed_frame()
    extra = data.loc[data.condition.eq("A")].copy()
    extra["condition"] = "C"
    extra["good"] += .3
    data = pd.concat([data, extra], ignore_index=True)
    cfg = resolve_config({"statistics": {"enabled": True, "designs": [
        {"design_id": "h", "outcome": "good", "design_type": "mixed_effects", "conditions": ["A", "B"]}]}})
    estimates, diagnostics = run_statistics(data, None, cfg)
    assert estimates.status.eq("unavailable").all(), estimates.to_dict("records")
    assert diagnostics.status.eq("failed").all()


def test_failed_mixed_model_retains_each_planned_coefficient_in_family():
    cfg = resolve_config({"statistics": {"enabled": True, "designs": [
        {"design_id": name, "outcome": outcome, "design_type": "mixed_effects", "conditions": ["A", "B"]}
        for name, outcome in [("h1", "good"), ("h2", "bad")]]}})
    estimates, _ = run_statistics(_mixed_frame(), None, cfg)
    expected = {"h1:Intercept", "h1:condition:B", "h2:Intercept", "h2:condition:B"}
    assert set(estimates.hypothesis_id) == expected, estimates.to_dict("records")


def test_unresolved_associated_clock_is_not_declared_aligned():
    cfg = get_preset("group_sync")
    record = make_record(np.arange(101) / 10, np.sin(np.arange(101) / 10),
                         {"recording_id": "r", "subject_id": "s", "session_id": "ss",
                          "channel": "eda", "unit": "uS", "clock_domain": "device", "fs_hz": 10.0})
    tables = {
        "sync_anchors": pd.DataFrame([
            {"recording_id": "r", "local_time_s": 0., "reference_time_s": 100.},
            {"recording_id": "r", "local_time_s": 10., "reference_time_s": 110.}]),
        "events": pd.DataFrame([{"recording_id": "r", "event_id": "e", "clock_domain": "other", "onset_s": 5.}]),
    }
    try:
        _, _, report = apply_clock_alignment([record], tables, cfg)
    except ValueError:
        return  # Explicit refusal is also a valid contract implementation.
    assert report["status"] != "aligned", report
    assert report["formal_synchrony_allowed"] is False


def _short_config(tmp_path):
    signal = tmp_path / "signal.csv"
    pd.DataFrame({"time_s": np.arange(20) / 4, "eda_uS": 2 + .1 * np.sin(np.arange(20))}).to_csv(signal, index=False)
    return {"input": {"signals": [str(signal)], "base_directory": str(tmp_path)},
            "output": {"directory": str(tmp_path / "results"), "formats": ["json"]}}


def test_zero_eligible_analysis_is_not_a_successful_record(tmp_path):
    from eda_toolkit.pipeline import run_pipeline
    summary = run_pipeline(_short_config(tmp_path))
    assert summary["completed_records"] == 1, summary
    assert summary["status"] != "succeeded", summary
    record = summary["record_statuses"][0]
    assert record["detection"]["status"] != "no_responses", record


def test_manifest_hash_refers_to_consumed_bytes_during_change(tmp_path, monkeypatch):
    import eda_toolkit.core_io as io
    from eda_toolkit.pipeline import run_pipeline
    cfg = _short_config(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("file_path\nsignal.csv\n", encoding="utf-8")
    consumed = hashlib.sha256(manifest.read_bytes()).hexdigest()
    cfg["input"]["manifest"] = str(manifest)
    cfg["input"]["signals"] = []
    original = io.load_recordings
    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        manifest.write_text("file_path\nsignal.csv\n\n", encoding="utf-8")
        return result
    monkeypatch.setattr(io, "load_recordings", change_after_read)
    summary = run_pipeline(cfg)
    provenance = json.loads((__import__("pathlib").Path(summary["directory"]) / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["input_hashes"][str(manifest.resolve())] == consumed


def test_resume_refuses_consumed_source_change_after_cancel(tmp_path):
    from eda_toolkit.pipeline import run_pipeline, resume_run
    cfg = _short_config(tmp_path)
    cancel = {"value": False}
    def progress(event):
        if event["stage"] == "record_done":
            cancel["value"] = True
    result = run_pipeline(cfg, progress_callback=progress, cancel_callback=lambda: cancel["value"])
    assert result["status"] == "cancelled", result
    signal = tmp_path / "signal.csv"
    signal.write_text(signal.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        resume_run(result["directory"])
    assert caught.value.eda_details["code"] == "resume_input_changed"


def _all_valid_aligned(n=16):
    rng = np.random.default_rng(314159)
    keys = ("r1::eda", "r2::eda")
    return {"time_s": np.arange(n) / 4,
            "signals": {key: rng.normal(size=n) for key in keys},
            "eligible": {key: np.ones(n, bool) for key in keys},
            "metadata": {key: {"recording_id": f"r{i + 1}", "channel": "eda", "subject_id": f"s{i + 1}", "session_id": "ss"}
                         for i, key in enumerate(keys)},
            "membership": pd.DataFrame([{"recording_id": f"r{i + 1}", "subject_id": f"s{i + 1}", "group_id": "g", "session_id": "ss", "channel": "eda"} for i in range(2)]),
            "basis": "phasic_uS", "alignment_report": {"formal_synchrony_allowed": True}}


def test_exhaustive_null_support_counterexample_when_windows_cover_grid():
    """Every feasible shift world loses seam support against full valid observed support."""
    from eda_toolkit.core_sync import _circular_shift_bundle, _support_signature, compute_pair_synchrony
    aligned = _all_valid_aligned()
    cfg = get_preset("group_sync")
    cfg["synchrony"].update(analysis_context_s=0., window_s=4., step_s=4., max_lag_s=0., min_samples=3, min_coverage=.5)
    observed, _ = compute_pair_synchrony(aligned, None, cfg)
    original = _support_signature(observed)
    feasible = accepted = 0
    for a in range(2, 14):
        for b in range(2, 14):
            separation = abs(a - b)
            if min(separation, 16 - separation) < 2:
                continue
            feasible += 1
            shifted = _circular_shift_bundle(aligned, {("ss", "s1"): a, ("ss", "s2"): b}, 0)
            values, _ = compute_pair_synchrony(shifted, None, cfg)
            accepted += _support_signature(values) == original
    assert feasible > 0
    assert accepted == 0


def test_null_is_reachable_when_preplanned_windows_leave_large_context_gaps():
    from eda_toolkit.core_sync import estimate_sync_null
    aligned = _all_valid_aligned(32)
    cfg = get_preset("group_sync")
    cfg["synchrony"].update(analysis_context_s=0., window_s=1., step_s=8., max_lag_s=0., min_samples=3,
                            min_coverage=.5, null_replicates=3, max_attempts=100, min_shift_s=.5)
    summary, tails = estimate_sync_null(aligned, None, cfg, seed=2718)
    assert summary.accepted_replicates.eq(3).all(), summary.to_dict("records")
    assert tails.status.eq("experimental_uncalibrated").all()
    assert tails.surrogate_tail_probability.notna().all()


def _onset_hand_bundle():
    time_s = np.arange(41, dtype=float)
    phasic = np.zeros(41)
    phasic[28] = .5
    phasic[30:36] = np.arange(1, 7) / 10
    phasic[37:39] = [5., 10.]
    valid = np.ones(41, bool)
    return {"metadata": {"recording_id": "hand", "subject_id": "hand", "session_id": "ss"},
            "time_s": time_s, "clean_uS": phasic + 2, "phasic_uS": phasic,
            "tonic_uS": np.full(41, 2.), "detector_eligible": valid,
            "quality": {"analysis_eligible": valid, "eligible": valid}, "max_gap_s": 1.5}


def test_independent_onset_hand_exposure_keeps_peak_search_context():
    from eda_toolkit.core_features import scr_onset_eligible_mask, detect_scr, extract_window_features
    bundle = _onset_hand_bundle()
    cfg = get_preset()
    mask, support = scr_onset_eligible_mask(bundle, cfg)
    assert support["search_eligible_duration_s"] == pytest.approx(40.)
    assert support["onset_eligible_duration_s"] == pytest.approx(30.)
    assert mask[29] and not mask[35]
    catalogue, report = detect_scr(bundle, cfg)
    primary = catalogue.loc[catalogue.onset_eligible]
    assert primary.onset_s.tolist() == [29.]
    assert primary.peak_s.tolist() == [35.]  # Outside onset support but inside preserved search support.
    windows = pd.DataFrame([{"window_id": "w", "recording_id": "hand", "kind": "fixed",
                             "start_s": 0., "end_s": 30., "event_id": None, "condition": None}])
    features, validity = extract_window_features(bundle, catalogue, windows, cfg)
    assert validity.iloc[0].detector_eligible_duration_s == pytest.approx(30.)
    assert features.loc[features.feature_id.eq("scr_count"), "value"].iloc[0] == 1
    assert features.loc[features.feature_id.eq("scr_rate"), "value"].iloc[0] == pytest.approx(2.)


def test_unqualified_edge_peak_does_not_define_primary_relative_threshold():
    from eda_toolkit.core_features import detect_scr
    cfg = get_preset()
    cfg["scr"].update(threshold_type="relative", relative_fraction=.1)
    catalogue, report = detect_scr(_onset_hand_bundle(), cfg)
    assert catalogue.loc[catalogue.onset_eligible, "onset_s"].tolist() == [29.], report


def _two_record_config(tmp_path):
    files = []
    t = np.arange(401) / 4
    for index in range(2):
        path = tmp_path / f"r{index}.csv"
        y = 2 + .2 * np.sin(2 * np.pi * .1 * t + .2 * index) + .04 * np.sin(2 * np.pi * .8 * t)
        pd.DataFrame({"time_s": t, "eda_uS": y, "recording_id": f"r{index}",
                      "subject_id": f"s{index}", "session_id": "ss", "channel": "eda"}).to_csv(path, index=False)
        files.append(path.name)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"file_path": files}).to_csv(manifest, index=False)
    return {"input": {"manifest": str(manifest), "base_directory": str(tmp_path)},
            "mapping": {"sampling_rate_hz": 4.}, "decomposition": {"cutoff_hz": .5},
            "windows": {"min_coverage": .5, "fixed_length_s": 20., "fixed_step_s": 20.},
            "output": {"directory": str(tmp_path / "results"), "formats": ["json"]}}


def test_cancel_resume_reuses_committed_cache_and_refuses_incompatible_changes(tmp_path, monkeypatch):
    import eda_toolkit.pipeline as pipeline
    cfg = _two_record_config(tmp_path)
    fingerprint = {"value": "independent-test-source-v1"}
    monkeypatch.setattr(pipeline, "source_fingerprint", lambda: fingerprint["value"])
    calls = []
    process = pipeline._process_one
    def counted(record, *args, **kwargs):
        calls.append(record["metadata"]["recording_id"])
        return process(record, *args, **kwargs)
    monkeypatch.setattr(pipeline, "_process_one", counted)
    cancel = {"value": False}
    def progress(event):
        if event["stage"] == "record_done":
            cancel["value"] = True
    cancelled = pipeline.run_pipeline(cfg, progress_callback=progress, cancel_callback=lambda: cancel["value"])
    assert cancelled["status"] == "cancelled" and calls == ["r0"], cancelled
    root = Path(cancelled["directory"])
    state = json.loads((root / "pipeline_state.json").read_text(encoding="utf-8"))
    fingerprint["value"] = "independent-test-source-v2"
    with pytest.raises(ValueError) as caught:
        pipeline.resume_run(root)
    assert caught.value.eda_details["code"] == "resume_implementation_changed"
    fingerprint["value"] = "independent-test-source-v1"
    cache = root / next(iter(state["committed_records"].values())) / "signals.npz"
    saved = cache.read_bytes()
    cache.write_bytes(saved + b"tamper")
    with pytest.raises(ValueError) as caught:
        pipeline.resume_run(root)
    assert caught.value.eda_details["code"] == "resume_checksum"
    cache.write_bytes(saved)
    manifest = Path(cfg["input"]["manifest"])
    original = manifest.read_bytes()
    manifest.write_bytes(original + b"\n")
    with pytest.raises(ValueError) as caught:
        pipeline.resume_run(root)
    assert caught.value.eda_details["code"] == "resume_input_changed"
    manifest.write_bytes(original)
    resumed = pipeline.resume_run(root)
    assert resumed["status"] == "succeeded", resumed
    assert calls == ["r0", "r1"], "The committed r0 stage must be read from cache rather than recomputed."
    assert resumed["record_statuses"][0]["reused_from"] == str(root)


def test_serial_parallel_features_and_scr_are_identical(tmp_path):
    from eda_toolkit.pipeline import run_pipeline
    cfg = _two_record_config(tmp_path)
    runs = []
    for workers in (1, 2):
        cfg["compute"] = {"workers": workers}
        started = time.perf_counter()
        result = run_pipeline(cfg)
        elapsed = time.perf_counter() - started
        assert result["status"] == "succeeded", result
        runs.append((result, elapsed))
    for table in ("features", "scr"):
        outputs = []
        for run, _ in runs:
            path = next((Path(run["directory"]) / "tables").glob(f"{table}_*.json"))
            outputs.append(json.loads(path.read_text(encoding="utf-8")))
        assert outputs[0] == outputs[1], table
    measurement = {"kind": "small_synthetic_end_to_end", "records": 2, "samples_per_record": 401,
                   "sampling_rate_hz": 4., "formats": ["json"], "plotting": False,
                   "serial_elapsed_s": runs[0][1], "workers2_elapsed_s": runs[1][1],
                   "peak_memory": "not_measured", "claim": "numerical agreement only; no general speedup claim"}
    print("INDEPENDENT_PARALLEL_MEASUREMENT=" + json.dumps(measurement))


def test_gui_discards_mapping_preview_after_file_selection_changes(tmp_path, monkeypatch):
    from concurrent.futures import Future
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QLineEdit
    from eda_toolkit.core_io import read_signal_csv
    from eda_toolkit.ui.app import create_window
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({"time_s": np.arange(7), "eda_uS": np.arange(7) + 1}).to_csv(first, index=False)
    pd.DataFrame({"time_s": np.arange(5), "eda_uS": np.arange(5) + 2}).to_csv(second, index=False)
    window, state = create_window()
    try:
        editor = next(item for item in window.findChildren(QLineEdit) if item.placeholderText() == "选择一条信号 CSV 做映射预览")
        completed = Future()
        completed.set_result(read_signal_csv(first, {"columns": {"time_s": "time_s", "eda_uS": "eda_uS"}}, state["config"]))
        state["mapping_future"] = (state["preview_id"], completed)
        editor.setText(str(second))
        editor.editingFinished.emit()
        current_title = state["widgets"]["preview_title"].text()
        state["timer"].timeout.emit()
        assert state["widgets"]["preview_title"].text() == current_title, "An older file's background preview replaced the newly selected file."
    finally:
        window.close()
        state["shutdown"]()


def test_null_full_coverage_preflight_refuses_before_any_correlation(monkeypatch):
    import eda_toolkit.core_sync as sync
    aligned = _all_valid_aligned(16)
    cfg = get_preset("group_sync")
    cfg["synchrony"].update(analysis_context_s=0., window_s=4., step_s=4., max_lag_s=0.,
                            min_shift_s=.5, min_samples=3, min_coverage=.5,
                            null_replicates=9, max_attempts=100)
    monkeypatch.setattr(sync, "compute_pair_synchrony", lambda *a, **kw: pytest.fail("infeasible geometry must not evaluate correlations"))
    summary, tails = sync.estimate_sync_null(aligned, None, cfg, seed=1)
    assert summary.iloc[0].reason == "joint_shift_infeasible"
    assert summary.iloc[0].attempted_replicates == 0
    assert summary.iloc[0].accepted_replicates == 0
    assert summary.iloc[0].proposal_count == 0
    assert tails.empty


def test_buffered_group_preset_three_tails_match_independent_lag_search_oracle(monkeypatch):
    import eda_toolkit.core_sync as sync
    aligned = _all_valid_aligned(640)
    signal = np.random.default_rng(523).normal(size=640)
    aligned["signals"]["r1::eda"] = signal
    aligned["signals"]["r2::eda"] = -np.roll(signal, 2)
    actual = sync.compute_pair_synchrony
    captured = []
    def capture(world, *args, **kwargs):
        result = actual(world, *args, **kwargs)
        captured.append(result[0].copy())
        return result
    monkeypatch.setattr(sync, "compute_pair_synchrony", capture)
    results = {}
    for tail in ("two-sided", "greater", "less"):
        captured.clear()
        cfg = get_preset("group_sync")  # Its actual 60-second context is used, not overwritten.
        cfg["synchrony"].update(window_s=40., step_s=40., max_lag_s=.5, lag_step_s=.25,
                                tail=tail, min_shift_s=10., null_replicates=4, max_attempts=200)
        summary, tails = sync.estimate_sync_null(aligned, None, cfg, seed=13)
        assert summary.accepted_replicates.eq(4).all(), summary.to_dict("records")
        assert summary.analysis_context_s.eq(60.).all()
        assert len(captured) == 5  # Once observed and once for each accepted world.
        assert tails.status.eq("experimental_uncalibrated").all()
        for _, row in tails.iterrows():
            def oracle(frame):
                local = frame.loc[frame.window_id.eq(row.window_id) & frame.status.eq("valid")].sort_values("lag_s")
                values = local.r.to_numpy()
                scores = abs(values) if tail == "two-sided" else -values if tail == "less" else values
                index = int(np.argmax(scores))
                return (float(abs(values[index]) if tail == "two-sided" else values[index]),
                        float(local.iloc[index].lag_s), float(values[index]))
            observed, lag, signed = oracle(captured[0])
            nulls = np.array([oracle(frame)[0] for frame in captured[1:]])
            extreme = np.count_nonzero(nulls <= observed + 1e-12) if tail == "less" else np.count_nonzero(nulls >= observed - 1e-12)
            assert row.observed_statistic == pytest.approx(observed)
            assert row.selected_lag_s == pytest.approx(lag)
            assert row.selected_r == pytest.approx(signed)
            assert row.surrogate_tail_probability == pytest.approx((1 + extreme) / 5)
        results[tail] = tails
    assert results["less"].selected_r.eq(-1.).all()
    assert results["two-sided"].selected_lag_s.eq(.5).all()
    assert results["greater"].selected_r.gt(-.9).all()


def _origin_declared_pseudodyad_fixture():
    aligned = {"time_s": np.arange(80.), "signals": {}, "eligible": {}, "metadata": {},
               "basis": "phasic_uS", "alignment_report": {"formal_synchrony_allowed": True}}
    rng = np.random.default_rng(8001)
    members = []
    for group in range(3):
        for slot in range(2):
            record, subject = f"g{group}r{slot}", f"g{group}s{slot}"
            key = record + "::eda"
            aligned["signals"][key] = rng.normal(size=80)
            aligned["eligible"][key] = np.ones(80, bool)
            aligned["metadata"][key] = {"recording_id": record, "subject_id": subject,
                                       "session_id": "V01", "channel": "eda", "clock_domain": "reference",
                                       "task_time_origin_s": group * 2.}
            members.append({"group_id": f"g{group}", "session_id": "V01", "subject_id": subject,
                            "recording_id": record, "channel": "eda", "role": str(slot),
                            "task": "task", "stimulus": "stim", "timeline_id": "timeline"})
    aligned["membership"] = pd.DataFrame(members)
    cfg = get_preset("group_sync")
    cfg["synchrony"].update(null_method="pseudodyad", pseudodyad_exchangeability=True,
                            analysis_context_s=0., analysis_intervals_s=[[20., 40.]],
                            analysis_rate_hz=1., window_s=20., step_s=20., max_lag_s=1., lag_step_s=1.,
                            min_samples=8, min_coverage=.5, null_replicates=3, max_attempts=20)
    return aligned, cfg


def test_pseudodyad_donor_world_values_follow_task_origin_without_wrap_or_circle(monkeypatch):
    import eda_toolkit.core_sync as sync
    aligned, cfg = _origin_declared_pseudodyad_fixture()
    actual = sync.compute_pair_synchrony
    captured = []
    def capture(world, *args, **kwargs):
        captured.append({key: value.copy() for key, value in world["signals"].items()})
        return actual(world, *args, **kwargs)
    monkeypatch.setattr(sync, "compute_pair_synchrony", capture)
    monkeypatch.setattr(sync, "_circular_shift_bundle", lambda *a, **kw: pytest.fail("pseudodyad cannot fall back to circular shift"))
    summary, tails = sync.estimate_sync_null(aligned, None, cfg, seed=3)
    assert summary.accepted_replicates.eq(3).all(), summary.to_dict("records")
    assert tails.method.eq("pseudodyad").all()
    assert tails.qualification.eq("experimental_uncalibrated").all()
    assert len(captured) == 4
    for replicate in range(3):
        for _, row in tails.iterrows():
            pair = json.loads(row.provenance_json)[replicate]
            assert len(pair) == 2
            assert pair[0]["donor_group_id"] != pair[1]["donor_group_id"]
            assert all(endpoint["target_session_id"] == endpoint["donor_session_id"] == "V01" for endpoint in pair)
            for endpoint in pair:
                target, donor = endpoint["target_record_key"], endpoint["donor_record_key"]
                assert endpoint["target_group_id"] != endpoint["donor_group_id"]
                difference = int(aligned["metadata"][donor]["task_time_origin_s"] - aligned["metadata"][target]["task_time_origin_s"])
                source_index = np.arange(80) + difference
                inside = (source_index >= 0) & (source_index < 80)
                expected = np.full(80, np.nan)
                expected[inside] = aligned["signals"][donor][source_index[inside]]
                np.testing.assert_array_equal(captured[replicate + 1][target], expected)


def test_pseudodyad_missing_origin_or_fractional_unsupported_origin_is_unavailable(monkeypatch):
    import copy
    import eda_toolkit.core_sync as sync
    aligned, cfg = _origin_declared_pseudodyad_fixture()
    monkeypatch.setattr(sync, "_circular_shift_bundle", lambda *a, **kw: pytest.fail("pseudodyad cannot fall back to circular shift"))
    missing = copy.deepcopy(aligned)
    for metadata in missing["metadata"].values():
        metadata.pop("task_time_origin_s")
    summary, tails = sync.estimate_sync_null(missing, None, cfg, seed=3)
    assert summary.iloc[0].reason == "pseudodyad_task_time_basis_missing_or_invalid"
    assert tails.empty
    fractional = copy.deepcopy(aligned)
    for key, metadata in fractional["metadata"].items():
        if key.startswith("g0"):
            metadata["task_time_origin_s"] = .25
    summary, tails = sync.estimate_sync_null(fractional, None, cfg, seed=3)
    assert summary.iloc[0].status == "unavailable"
    assert summary.iloc[0].reason == "pseudodyad_assignment_infeasible"
    assert tails.empty
