import json

import numpy as np
import pandas as pd

from eda_toolkit.config import (feature_work_method, resolve_config, synchrony_work_method)
from eda_toolkit.contracts import make_record
from eda_toolkit.metrics_ids import autocorrelation_feature_id
from eda_toolkit.metrics_spectral_complexity import compute_clean_advanced_window
from eda_toolkit.core_export import export_run, verify_manifest
from eda_toolkit.pipeline import (_feature_progress_adapter, _merge_group_temporal_results,
                                  _pipeline_work_plan, _process_one)
from eda_toolkit.runtime_preflight import _work_plan


def test_preflight_and_pipeline_eta_plan_share_feature_parameter_signature(tmp_path):
    config = resolve_config()
    source = {"row": {"recording_id": "R001", "channel": "eda", "sampling_rate_hz": 4.0},
              "path": tmp_path / "R001.csv"}
    record = {"metadata": {"recording_id": "R001", "channel": "eda", "fs_hz": 4.0}}

    preflight = _work_plan([source], config, total_known=True)
    actual = _pipeline_work_plan([record], config)
    preflight_feature = next(unit for unit in preflight["units"] if unit["stage"] == "features")
    actual_feature = next(unit for unit in actual["units"] if unit["stage"] == "features")
    assert preflight_feature["method"] == actual_feature["method"] == feature_work_method(config)

    changed = resolve_config(config)
    changed["features"]["ids"] = ["mean"]
    assert feature_work_method(changed) != feature_work_method(config)
    changed["features"]["extended"]["complexity"]["sampen"]["m"] = 3
    assert feature_work_method(changed) != feature_work_method(resolve_config(config))

    sync_config = resolve_config(config)
    sync_config["synchrony"]["enabled"] = True
    preflight_sync = _work_plan([source], sync_config, total_known=True)
    actual_sync = _pipeline_work_plan([record], sync_config)
    assert next(unit for unit in preflight_sync["units"] if unit["stage"] == "synchrony")["method"] == \
        next(unit for unit in actual_sync["units"] if unit["stage"] == "synchrony")["method"] == \
        synchrony_work_method(sync_config)


def test_group_temporal_results_are_filtered_and_join_shared_feature_output():
    combined = {"features": pd.DataFrame([{"feature_id": "mean", "value": 2.0}])}
    group_features = pd.DataFrame([
        {"recording_id": "G1", "session_id": "S1", "feature_id": "sync_positive_time_proportion", "value": 0.5},
        {"recording_id": "G1", "session_id": "S1", "feature_id": "sync_negative_time_proportion", "value": 0.2},
    ])
    episodes = pd.DataFrame([{"sign": "positive", "observed_duration_s": 3.0}])
    nodes = pd.DataFrame([{"subject_id": "S001", "signed": 0.4}])
    diagnostics = pd.DataFrame([{"group_id": "G1", "status": "valid"}])
    config = {"features": {"ids": ["mean", "sync_positive_time_proportion"]}}

    result = _merge_group_temporal_results(combined, group_features, episodes, nodes, diagnostics, config)
    assert result["group_temporal_features"]["feature_id"].tolist() == ["sync_positive_time_proportion"]
    assert result["features"]["feature_id"].tolist() == ["mean", "sync_positive_time_proportion"]
    assert result["sync_episodes"].equals(episodes)
    assert result["node_strength"].equals(nodes)
    assert result["group_temporal_diagnostics"].equals(diagnostics)


def test_process_one_forwards_metric_progress_and_flattens_extended_tables():
    config = resolve_config()
    config["features"]["ids"] = ["clean_sample_entropy"]
    config["windows"].update({"whole_record": False, "fixed_length_s": 60.0, "fixed_step_s": 30.0})
    time_s = np.arange(481, dtype=np.float64) / 4.0
    signal = 2.0 + 0.12 * np.sin(2 * np.pi * time_s / 17.0) + 0.005 * np.cos(time_s * 1.7)
    record = make_record(time_s, signal, {
        "recording_id": "R001", "subject_id": "S001", "session_id": "session-1",
        "channel": "eda", "unit": "uS", "fs_hz": 4.0, "clock_domain": "recording_local",
    })
    runtime_events = []
    legacy_progress = []
    runtime = {"event_sink": runtime_events.append, "run_id": "run-test", "project_id": "project-test",
               "task_id": "task-test", "attempt_id": "attempt-test",
               "stage_totals": {"qc": 1, "preprocess": 1, "decomposition": 1, "features": 1}}

    result = _process_one(record, {}, config, lambda: False, legacy_progress.append, runtime)

    expected_tables = {
        "trial_table", "trial_summaries", "trial_fit_diagnostics", "spectra",
        "complexity_diagnostics", "ns_classification", "scr_events_extended",
    }
    assert expected_tables <= set(result["tables"])
    assert result["tables"]["validity"].attrs == {}
    metrics = result["tables"]["features"].loc[
        result["tables"]["features"]["feature_id"].eq("clean_sample_entropy")]
    assert not metrics.empty
    assert any(item.get("family") == "time_domain" for item in legacy_progress)


def test_feature_progress_adapter_forwards_real_helper_family_events():
    fs_hz = 4.0
    time_s = np.arange(256, dtype=np.float64) / fs_hz
    values = np.sin(2 * np.pi * time_s / 5.0) + 0.05 * np.cos(time_s * 1.3)
    feature_id = autocorrelation_feature_id(1.0)
    runtime_events = []
    progress_events = []
    callback = _feature_progress_adapter({"event_sink": runtime_events.append, "run_id": "run-test",
                                          "project_id": "project-test", "task_id": "task-test",
                                          "attempt_id": "attempt-test"}, "R001", progress_events.append)

    result = compute_clean_advanced_window(
        time_s, values, np.ones(len(values), dtype=bool),
        window_start_s=0.0, window_end_s=float(time_s[-1] + 1 / fs_hz),
        declared_fs_hz=fs_hz, max_gap_s=1.5 / fs_hz, jitter_tolerance=0.01,
        extended_config={"autocorrelation": {"enabled": True, "lags_s": [1.0], "min_pairs": 20}},
        selected_ids=[feature_id], identity={"recording_id": "R001", "window_id": "W001"},
        progress_callback=callback,
    )

    assert result["metrics"][feature_id]["status"] == "valid"
    family_events = [event for event in runtime_events if event["unit_type"] == "feature_family"]
    assert [event["kind"] for event in family_events] == ["work_started", "completed"]
    assert family_events[-1]["details"]["work_id"] == "R001::W001:clean_autocorrelation"
    assert family_events[-1]["details"]["duration_s"] >= 0
    assert [event["kind"] for event in progress_events] == ["work_started", "completed"]


def test_export_freezes_selected_feature_definitions_in_provenance(tmp_path):
    config = resolve_config()
    config["features"]["ids"] = ["clean_sample_entropy"]
    config["output"]["formats"] = ["json"]
    provenance = {"resolved_config": config, "environment": {"python": "synthetic-test"}}
    result = {"summary": {"status": "succeeded"},
              "tables": {"features": pd.DataFrame([{"feature_id": "clean_sample_entropy",
                                                        "value": 0.2, "unit": "nats"}])}}

    manifest = export_run(result, provenance, tmp_path, config)

    definitions = json.loads((tmp_path / "feature_definitions.json").read_text(encoding="utf-8"))
    saved_provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert list(definitions) == ["clean_sample_entropy"]
    assert definitions["clean_sample_entropy"]["basis"] == "clean_uS"
    assert saved_provenance["feature_definitions"] == definitions
    assert verify_manifest(tmp_path)["valid"]
    assert any(item["path"] == "feature_definitions.json" for item in manifest["artifacts"])
