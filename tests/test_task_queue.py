"""Durable queue dispatch and restart reconciliation boundaries."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from eda_toolkit.project_store import open_project
from eda_toolkit.task_queue import claim_next_task, execute_claimed_task, reconcile_interrupted_tasks
from eda_toolkit.task_store import enqueue_task, get_task


def _install_preflight(monkeypatch):
    module = types.ModuleType("eda_toolkit.runtime_preflight")
    module.validate_preflight_snapshot = lambda report, config, manifest, state: {
        "valid": True, "stale": False, "errors": []}
    monkeypatch.setitem(sys.modules, "eda_toolkit.runtime_preflight", module)


def _queued_task(project, monkeypatch):
    _install_preflight(monkeypatch)
    config = {"pipeline_id": "single_standard", "input": {"base_directory": str(Path(project["path"]))},
              "output": {"directory": str(Path(project["path"]) / "results")}}
    input_manifest = {"input_fingerprints": []}
    report = {"report_id": "preflight-1", "report_sha256": "b" * 64,
              "eligible_to_enqueue": True, "errors": [], "project_id": project["project_id"],
              "project_revision": project["project_revision"], "draft_revision": project["draft_revision"],
              "marker_revision": project["marker_revision"], "qc_revision": project["qc_revision"],
              "input_fingerprints": {}}
    spec = {"config": config, "input_manifest": input_manifest,
            "project_revision": 0, "draft_revision": 0, "marker_revision": 0, "qc_revision": 0}
    return enqueue_task(project, spec, report)


def test_claim_execute_persists_verified_manifest_and_monotone_events(tmp_path: Path, monkeypatch):
    project = open_project(tmp_path / "project")
    task_id = _queued_task(project, monkeypatch)
    claimed = claim_next_task(project, "test-owner")
    assert claimed["task_id"] == task_id
    events = []
    from eda_toolkit.runtime_logging import create_runtime_session
    runtime_session = create_runtime_session(project["path"], subscribers=events.append, install_hooks=False)

    def fake_run_pipeline(config, input_manifest=None, progress_callback=None, cancel_callback=None, **runtime_kwargs):
        run_directory = Path(config["output"]["directory"]) / "run-1"
        run_directory.mkdir(parents=True)
        (run_directory / ".running.lock").write_text("run-1", encoding="ascii")
        (run_directory / "artifact.txt").write_text("result", encoding="utf-8")
        state = {"run_id": "run-1", "status": "running", "directory": str(run_directory),
                 "config_sha256": "c" * 64}
        (run_directory / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
        progress_callback({"stage": "start", "message": "start"})
        task_binding = run_directory / "task_binding.json"
        artifacts = []
        for path in (run_directory / "artifact.txt", task_binding):
            artifacts.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (run_directory / "manifest.json").write_text(json.dumps({"status": "succeeded", "artifacts": artifacts}),
                                                    encoding="utf-8")
        progress_callback({"stage": "finished", "message": "done", "completed": 1, "total": 1})
        return {"run_id": "run-1", "status": "succeeded", "directory": str(run_directory),
                "completed_records": 1, "record_statuses": [{"status": "succeeded"}], "errors": []}

    monkeypatch.setattr("eda_toolkit.pipeline.run_pipeline", fake_run_pipeline)
    try:
        summary = execute_claimed_task(task_id, runtime_session["event_sink"], project=project)
    finally:
        runtime_session["close"]()
    assert summary["status"] == "succeeded"
    task = get_task(project, task_id)
    assert task["state"] == "succeeded"
    assert task["run_id"] == "run-1"
    assert events and [event["seq"] for event in events] == sorted(event["seq"] for event in events)
    assert any(event["kind"] == "completed" for event in events)


def test_reconcile_refuses_restart_when_process_liveness_is_unknown(tmp_path: Path, monkeypatch):
    project = open_project(tmp_path / "project")
    task_id = _queued_task(project, monkeypatch)
    claim_next_task(project, "test-owner")
    report = reconcile_interrupted_tasks(project, lambda task: None)
    assert report["attention"] == [{"task_id": task_id, "reason": "owner_liveness_unproven"}]
    assert get_task(project, task_id)["state"] == "attention"


def test_reconcile_marks_dead_owner_interrupted_without_auto_retry(tmp_path: Path, monkeypatch):
    project = open_project(tmp_path / "project")
    task_id = _queued_task(project, monkeypatch)
    claim_next_task(project, "test-owner")
    report = reconcile_interrupted_tasks(project, lambda task: False)
    assert report["interrupted"] == [task_id]
    assert get_task(project, task_id)["state"] == "interrupted"
    assert claim_next_task(project, "other-owner") is None


def test_attention_reconciles_verified_failed_manifest_after_liveness_is_proven_dead(tmp_path: Path, monkeypatch):
    project = open_project(tmp_path / "project")
    task_id = _queued_task(project, monkeypatch)
    claimed = claim_next_task(project, "possibly-live-owner")
    uncertain = reconcile_interrupted_tasks(project, lambda task: None)
    assert uncertain["attention"] == [{"task_id": task_id, "reason": "owner_liveness_unproven"}]

    run_directory = tmp_path / "failed-run"
    run_directory.mkdir()
    artifact = run_directory / "evidence.txt"
    artifact.write_text("verified terminal failure", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (run_directory / "manifest.json").write_text(
        json.dumps({"status": "failed", "artifacts": [{"path": artifact.name, "sha256": digest}]}),
        encoding="utf-8")
    (run_directory / "task_binding.json").write_text(
        json.dumps({"task_id": task_id, "attempt_id": claimed["attempt_id"],
                    "project_id": project["project_id"]}), encoding="utf-8")
    with sqlite3.connect(project["db_path"]) as connection:
        connection.execute("UPDATE task_attempts SET run_directory=? WHERE attempt_id=?",
                           (str(run_directory), claimed["attempt_id"]))

    recovered = reconcile_interrupted_tasks(project, lambda task: False)
    assert recovered["recovered"] == [{"task_id": task_id, "state": "failed"}]
    assert get_task(project, task_id)["state"] == "failed"


def test_execution_rejects_signal_bytes_changed_after_enqueue(tmp_path: Path, monkeypatch):
    from eda_toolkit.config import get_preset
    from eda_toolkit.runtime_preflight import preflight_run

    project = open_project(tmp_path / "project")
    signal_path = tmp_path / "signal.csv"
    signal_path.write_text("time_s,eda_uS\n0,1.2\n0.25,1.3\n0.5,1.4\n", encoding="utf-8")
    config = get_preset("single_standard")
    config["input"].update({"signals": [str(signal_path)], "manifest": None,
                            "base_directory": str(tmp_path)})
    config["mapping"].update({"columns": {"time_s": "time_s", "eda_uS": "eda_uS"},
                              "sampling_rate_hz": 4.0, "eda_unit": "uS"})
    config["output"]["directory"] = str(tmp_path / "results")
    config["workspace"]["project_directory"] = project["path"]
    project_state = {key: project[key] for key in (
        "project_id", "project_revision", "draft_revision", "marker_revision", "qc_revision")}
    project_state["project_directory"] = project["path"]
    report = preflight_run(config, None, project_state)
    assert report["eligible_to_enqueue"] is True, report["errors"]
    frozen_spec = {"config": config, "input_manifest": None,
                   "project_revision": project["project_revision"],
                   "draft_revision": project["draft_revision"],
                   "marker_revision": project["marker_revision"],
                   "qc_revision": project["qc_revision"]}
    task_id = enqueue_task(project, frozen_spec, report)

    signal_path.write_text("time_s,eda_uS\n0,1.2\n0.25,9.9\n0.5,1.4\n", encoding="utf-8")
    claimed = claim_next_task(project, "changed-input-owner")
    assert claimed and claimed["task_id"] == task_id
    pipeline_calls = []
    monkeypatch.setattr("eda_toolkit.pipeline.run_pipeline",
                        lambda *args, **kwargs: pipeline_calls.append((args, kwargs)))
    with pytest.raises(RuntimeError, match="queued_task_preflight_stale"):
        execute_claimed_task(task_id, lambda _event: None, project=project)
    assert pipeline_calls == []
    failed = get_task(project, task_id)
    assert failed["state"] == "failed"
    from eda_toolkit.task_store import list_task_attempts
    attempt = list_task_attempts(project, task_id)[-1]
    assert "preflight_inputs_stale" in attempt["error"]["message"]

