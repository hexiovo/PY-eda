import pandas as pd
from eda_toolkit.pipeline import _combine


def test_combining_measurements_does_not_compare_dataframe_attributes():
    frames = []
    for value in (1, 2):
        table = pd.DataFrame({"value": [value]})
        table.attrs["event_scr_links"] = pd.DataFrame({"link": [value]})
        frames.append({"tables": {"validity": table}})
    combined = _combine(frames)
    assert combined["validity"].value.tolist() == [1, 2]
    assert "event_scr_links" in frames[0]["tables"]["validity"].attrs


def test_resume_refuses_changed_numerical_environment(tmp_path, monkeypatch):
    import pytest
    import eda_toolkit.pipeline as pipeline
    monkeypatch.setattr(pipeline, "numerical_environment_fingerprint", lambda backend="cpu": "environment-one")
    summary = pipeline.run_pipeline({"output": {"directory": str(tmp_path), "formats": ["json"]}}, cancel_callback=lambda: True)
    assert summary["status"] == "cancelled"
    monkeypatch.setattr(pipeline, "numerical_environment_fingerprint", lambda backend="cpu": "environment-two")
    with pytest.raises(ValueError) as caught:
        pipeline.resume_run(summary["directory"])
    assert caught.value.eda_details["code"] == "resume_environment_changed"


def test_frozen_input_fingerprints_reject_changed_bytes_before_analysis(tmp_path):
    import uuid
    from pathlib import Path
    from eda_toolkit.config import resolve_config
    from eda_toolkit.demo import create_demo
    from eda_toolkit.runtime_preflight import preflight_run
    from eda_toolkit.pipeline import run_pipeline

    demo = tmp_path / "synthetic"
    create_demo(demo, duration_s=120, sampling_rate_hz=4, seed=2602)
    config = resolve_config(demo / "single.json")
    config["output"]["directory"] = str((tmp_path / "results").resolve())
    config["workspace"]["project_directory"] = str((tmp_path / "project").resolve())
    state = {"project_id": str(uuid.uuid4()), "project_revision": 0,
             "draft_revision": 0, "marker_revision": 0, "qc_revision": 0}
    input_manifest = config["input"].get("manifest")
    report = preflight_run(config, input_manifest, state)
    assert report["eligible_to_enqueue"], report["errors"]

    signal_path = demo / "synthetic_001.csv"
    signal_path.write_bytes(signal_path.read_bytes() + b"\n")
    summary = run_pipeline(config, input_manifest=input_manifest,
                           runtime_context={"expected_input_fingerprints": report["input_fingerprints"]})
    assert summary["status"] == "failed"
    assert any(error.get("code") == "input_snapshot_mismatch" for error in summary["errors"])
    assert not summary["record_statuses"]


def test_memory_manifest_fingerprint_is_bound_without_fake_file_lookup(tmp_path):
    import pytest
    from eda_toolkit.pipeline import _verify_consumed_input_fingerprints

    signal = tmp_path / "signal.csv"
    signal.write_text("time_s,eda_uS\n0,1.0\n", encoding="utf-8")
    expected = {"memory:signals": "a" * 64, str(signal): "b" * 64}
    _verify_consumed_input_fingerprints(expected, {"consumed_inputs": {str(signal.resolve()): "b" * 64}})
    with pytest.raises(ValueError) as caught:
        _verify_consumed_input_fingerprints(expected, {"consumed_inputs": {str(signal.resolve()): "c" * 64}})
    assert caught.value.eda_details["code"] == "input_snapshot_mismatch"


def test_cached_resampled_record_retains_timebase_and_original_grid(tmp_path):
    import numpy as np
    from eda_toolkit.pipeline import _read_cache, _save_cache
    raw_time = np.arange(41) / 8
    processed_time = np.arange(21) / 4
    grid = {"processed_grid": True, "fs_hz": 4., "clock_domain": "reference"}
    bundle = {"metadata": {"recording_id": "r", "fs_hz": 8., "clock_domain": "reference"},
              "time_s": processed_time, "clean_uS": np.sin(processed_time), "sampling_rate_hz": 4.,
              "timebase_metadata": grid, "quality": {},
              "record": {"metadata": {"recording_id": "r"}, "time_s": raw_time,
                         "eda_uS": np.sin(raw_time), "sample_index": np.arange(41)}}
    _save_cache({"bundle": bundle, "tables": {}, "summary": {"status": "succeeded"}}, tmp_path)
    restored = _read_cache(tmp_path)["bundle"]
    assert restored["timebase_metadata"] == grid
    assert restored["sampling_rate_hz"] == 4
    np.testing.assert_array_equal(restored["record"]["time_s"], raw_time)
    np.testing.assert_array_equal(restored["time_s"], processed_time)
