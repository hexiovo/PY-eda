"""Queue state is durable and does not start work when tasks are enqueued."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from eda_toolkit.project_store import open_project
from eda_toolkit.task_queue import claim_next_task
from eda_toolkit.task_store import (
    enqueue_task,
    get_task,
    list_tasks,
    reorder_queued_task,
    request_task_cancel,
    retry_task,
)


def _install_preflight(monkeypatch, *, valid=True):
    module = types.ModuleType("eda_toolkit.runtime_preflight")
    module.validate_preflight_snapshot = lambda report, config, manifest, state: {
        "valid": valid, "stale": not valid, "errors": [] if valid else [{"code": "stale"}],
    }
    monkeypatch.setitem(sys.modules, "eda_toolkit.runtime_preflight", module)


def _enqueue(project, monkeypatch, *, suffix="a", validator_valid=True):
    _install_preflight(monkeypatch, valid=validator_valid)
    report = {"report_id": f"report-{suffix}", "report_sha256": "a" * 64,
              "eligible_to_enqueue": True, "errors": []}
    config = {"pipeline_id": "single_standard", "input": {"base_directory": ""},
              "output": {"directory": "results"}}
    manifest = {"input_fingerprints": []}
    spec = {"config": config, "input_manifest": manifest,
            "project_revision": project["project_revision"],
            "draft_revision": project["draft_revision"],
            "marker_revision": project["marker_revision"], "qc_revision": project["qc_revision"]}
    return enqueue_task(project, spec, report)


def test_enqueue_freezes_task_without_starting_it(tmp_path: Path, monkeypatch):
    project = open_project(tmp_path / "project")
    task_id = _enqueue(project, monkeypatch)
    task = get_task(project, task_id)
    assert task["state"] == "queued"
    assert task["active_attempt_id"] is None
    assert task["config"]["pipeline_id"] == "single_standard"
    assert task["preflight_report"]["report_id"] == "report-a"
    assert claim_next_task(project, "test-owner")["task_id"] == task_id


def test_stale_preflight_and_second_active_task_are_rejected(tmp_path: Path, monkeypatch):
    project = open_project(tmp_path / "project")
    task_id = _enqueue(project, monkeypatch)
    claimed = claim_next_task(project, "owner")
    assert claimed["task_id"] == task_id
    with pytest.raises(RuntimeError, match="project_task_already_active"):
        _enqueue(project, monkeypatch, suffix="b")

    other = open_project(tmp_path / "other")
    with pytest.raises(RuntimeError, match="preflight_snapshot_stale_or_invalid"):
        _enqueue(other, monkeypatch, validator_valid=False)


def test_queue_order_cancel_and_explicit_retry(tmp_path: Path, monkeypatch):
    project = open_project(tmp_path / "project")
    first = _enqueue(project, monkeypatch, suffix="first")
    second = _enqueue(project, monkeypatch, suffix="second")
    reorder_queued_task(project, second, 1)
    assert [row["task_id"] for row in list_tasks(project)] == [second, first]

    cancelled = request_task_cancel(project, second)
    assert cancelled["state"] == "cancelled"
    attempt_id = retry_task(project, second, "retry")
    retried = get_task(project, second)
    assert retried["state"] == "queued"
    assert retried["active_attempt_id"] == attempt_id

    claimed = claim_next_task(project, "owner")
    assert claimed["task_id"] == second

