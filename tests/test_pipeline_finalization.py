import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _config(tmp_path):
    from eda_toolkit.config import get_preset

    source = tmp_path / "signal.csv"
    if not source.exists():
        time_s = np.arange(0.0, 120.0, 0.25)
        signal = 2.0 + 0.12 * np.sin(2 * np.pi * time_s / 37.0)
        for onset in (20.0, 50.0, 80.0):
            elapsed = np.maximum(0.0, time_s - onset)
            signal += 0.35 * (np.exp(-elapsed / 3.0) - np.exp(-elapsed / 0.6)) * (time_s >= onset)
        pd.DataFrame({"time_s": time_s, "eda_uS": signal}).to_csv(source, index=False)

    config = get_preset("single_standard")
    config["input"].update({
        "base_directory": str(tmp_path),
        "signals": [{"file_path": source.name, "recording_id": "r001", "subject_id": "s001",
                     "session_id": "s001", "sampling_rate_hz": 4.0,
                     "eda_unit": "uS", "clock_domain": "recording_local"}],
        "manifest": None,
        "participants": None,
        "events": None,
        "conditions": None,
        "group_membership": None,
        "sync_anchors": None,
        "manual_qc": None,
        "statistical_design": None,
    })
    config["mapping"].update({"eda_unit": "uS", "sampling_rate_hz": 4.0})
    config["features"]["ids"] = ["mean"]
    config["features"]["extended"]["enabled"] = False
    config["output"].update({"directory": str((tmp_path / "results").resolve()), "formats": ["json"],
                             "save_processed_signals": False})
    config["workspace"]["project_directory"] = str((tmp_path / "project").resolve())
    return config


def _run(config, project, monkeypatch=None):
    import eda_toolkit.pipeline as pipeline
    from eda_toolkit.runtime_logging import create_runtime_session

    events = []
    session = create_runtime_session(project, subscribers=events.append, install_hooks=False)
    try:
        summary = pipeline.run_pipeline(config, event_sink=session["event_sink"], runtime_session=session,
                                        project_id=session["project_id"])
        session["flush"]()
    finally:
        session["close"]()
    return summary, events


def test_final_terminal_follows_committed_manifest_and_eta_history_is_reachable(tmp_path, monkeypatch):
    from eda_toolkit.core_export import verify_manifest
    import eda_toolkit.pipeline as pipeline
    from eda_toolkit.runtime_events import estimate_progress
    from eda_toolkit.runtime_logging import create_runtime_session, load_eta_history

    config = _config(tmp_path)
    project = Path(config["workspace"]["project_directory"])
    project.mkdir(parents=True)
    successful_runs = []
    for _ in range(3):
        summary, events = _run(config, project)
        assert summary["status"] == "succeeded", summary["errors"]
        assert verify_manifest(summary["directory"])["valid"]
        terminal = [event for event in events if event.get("stage") == "finished"]
        assert len(terminal) == 1
        assert terminal[0]["details"]["run_status"] == "succeeded"
        assert terminal[0]["details"]["manifest_finalized"] is True
        successful_runs.append(summary["run_id"])

    # Measure ETA from the three prior runs' actual persisted completion events.
    current_events = []
    current_work_plan = None
    eta_at_first_completed_unit = {}
    config_for_fault = _config(tmp_path)
    session = create_runtime_session(project, install_hooks=False)
    session["subscribe"](current_events.append)
    original_refresh = pipeline.refresh_manifest
    refresh_calls = 0

    def fail_once(*args, **kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        if refresh_calls == 1:
            raise OSError("injected late manifest write failure")
        return original_refresh(*args, **kwargs)

    original_sink = session["event_sink"]

    def inspect_event(event):
        nonlocal current_work_plan
        current_events.append(event)
        details = event.get("details", {})
        if event.get("kind") == "plan_revised":
            current_work_plan = details.get("work_plan")
        if (event.get("kind") == "completed" and isinstance(details, dict) and
                str(details.get("work_id", "")).endswith(":qc") and not eta_at_first_completed_unit):
            history = load_eta_history(project, exclude_run_id=event.get("run_id"))
            eta_at_first_completed_unit.update(estimate_progress(current_events, current_work_plan, history))

    session["subscribers"].remove(current_events.append)
    session["subscribe"](inspect_event)
    monkeypatch.setattr(pipeline, "refresh_manifest", fail_once)
    try:
        failed_summary = pipeline.run_pipeline(
            config_for_fault, event_sink=original_sink, runtime_session=session,
            project_id=session["project_id"])
        session["flush"]()
    finally:
        session["close"]()
        monkeypatch.setattr(pipeline, "refresh_manifest", original_refresh)

    assert refresh_calls == 2
    assert failed_summary["status"] == "partial"
    assert any(error.get("stage") == "manifest_refresh" for error in failed_summary["errors"])
    assert verify_manifest(failed_summary["directory"])["valid"]
    manifest = json.loads((Path(failed_summary["directory"]) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    terminal = [event for event in current_events if event.get("stage") == "finished"]
    assert len(terminal) == 1
    assert terminal[0]["details"]["run_status"] == "partial"
    assert terminal[0]["details"]["manifest_finalized"] is True
    assert estimate_progress(current_events, current_work_plan, [])["percent"] < 100
    assert eta_at_first_completed_unit["eta_s"] is not None
    assert eta_at_first_completed_unit["eta_low_s"] <= eta_at_first_completed_unit["eta_s"] <= eta_at_first_completed_unit["eta_high_s"]
    assert failed_summary["elapsed_s"] > 0
    assert failed_summary["elapsed_scope"].endswith("before_final_manifest_commit")

    local_run_events = [json.loads(line) for line in
                        (Path(failed_summary["directory"]) / "runtime" / "runtime_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert local_run_events[-1]["stage"] == "finalize"
    assert local_run_events[-1]["details"]["phase"] == "finishing"
    assert all(event.get("stage") != "finished" for event in local_run_events)
    finalization_completions = [event for event in current_events
                                if event.get("kind") == "completed" and
                                event.get("details", {}).get("work_id") == "run:manifest_finalization"]
    assert len(finalization_completions) == 1
    assert isinstance(finalization_completions[0]["details"].get("duration_s"), (int, float))
    assert finalization_completions[0]["details"]["duration_s"] >= 0

    evidence_path = os.environ.get("EDA_WB01_FINALIZE_EVIDENCE")
    if evidence_path:
        evidence = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "test": "test_final_terminal_follows_committed_manifest_and_eta_history_is_reachable",
            "prior_successful_run_ids": successful_runs,
            "late_failure_run_id": failed_summary["run_id"],
            "late_failure_status": failed_summary["status"],
            "late_failure_manifest_valid": True,
            "late_failure_manifest_status": manifest["status"],
            "terminal_event": terminal[0],
            "progress_before_late_failure": {key: eta_at_first_completed_unit.get(key) for key in
                                              ("done", "total", "percent", "eta_s", "eta_low_s",
                                               "eta_high_s", "confidence", "reason")},
            "manifest_work_duration_s": finalization_completions[0]["details"]["duration_s"],
            "elapsed_s": failed_summary["elapsed_s"],
            "elapsed_scope": failed_summary["elapsed_scope"],
            "runtime_snapshot_ends_at_finishing": local_run_events[-1]["details"]["phase"] == "finishing",
            "refresh_manifest_calls": refresh_calls,
        }
        path = Path(evidence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
