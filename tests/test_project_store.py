"""Durable local project identity, snapshots, and audit history."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from eda_toolkit.project_store import (
    append_audit_action,
    list_audit_actions,
    load_recovery_candidates,
    load_last_good_draft_export,
    open_project,
    read_project_snapshot,
    save_project_revision,
)


def test_project_identity_and_revision_roundtrip(tmp_path: Path):
    project = open_project(tmp_path / "workspace")
    assert uuid.UUID(project["project_id"])
    assert project["project_revision"] == 0
    assert json.loads((Path(project["path"]) / "project.json").read_text(encoding="utf-8"))["project_id"] == project["project_id"]

    revision = save_project_revision(project, {"config": {"pipeline_id": "single_standard"}, "ui_state": {"page": 2}}, 0)
    assert revision == 1
    assert project["draft_revision"] == 1
    assert read_project_snapshot(project) == {"config": {"pipeline_id": "single_standard"}, "ui_state": {"page": 2}}
    assert open_project(project["path"])["project_id"] == project["project_id"]


def test_project_revision_compare_and_swap_rejects_stale_draft(tmp_path: Path):
    project = open_project(tmp_path / "workspace")
    save_project_revision(project, {"config": {"value": 1}}, 0)
    with pytest.raises(RuntimeError, match="project_revision_conflict"):
        save_project_revision(project, {"config": {"value": 2}}, 0)
    assert read_project_snapshot(project)["config"]["value"] == 1


def test_recovery_candidates_and_append_only_audit(tmp_path: Path):
    project = open_project(tmp_path / "workspace")
    save_project_revision(project, {"markers": []}, 0)
    save_project_revision(project, {"markers": [{"marker_id": "m1"}]}, 1, kind="marker")
    candidates = load_recovery_candidates(project)
    assert [item["revision"] for item in candidates] == [2, 1]
    assert candidates[0]["payload"]["markers"][0]["marker_id"] == "m1"
    action = {
        "action_id": str(uuid.uuid4()), "revision": 1, "parent_revision": 0,
        "operator": "test", "time": "2026-10-26T00:00:00Z", "operation": "import",
        "before": [], "after": [{"marker_id": "m1"}], "reason": "fixture",
    }
    append_audit_action(project, "marker_actions", action)
    saved = list_audit_actions(project)
    assert len(saved) == 1
    assert saved[0]["operation"] == "import"
    assert saved[0]["after"] == [{"marker_id": "m1"}]


def test_corrupt_database_is_not_overwritten(tmp_path: Path):
    path = tmp_path / "broken"
    (path / ".eda").mkdir(parents=True)
    database = path / ".eda" / "state.sqlite"
    database.write_bytes(b"not a sqlite database")
    original = database.read_bytes()
    with pytest.raises(ValueError, match="project_database_corrupt_not_overwritten"):
        open_project(path)
    assert database.read_bytes() == original


def test_corrupt_newest_revision_is_not_blessed_and_older_snapshot_remains_recoverable(tmp_path: Path):
    project = open_project(tmp_path / "corrupt-revision")
    save_project_revision(project, {"config": {"value": "safe-revision-one"}}, 0)
    save_project_revision(project, {"config": {"value": "newer"}}, 1)
    with sqlite3.connect(project["db_path"]) as connection:
        connection.execute("UPDATE revisions SET payload_json=? WHERE revision=2",
                           (json.dumps({"config": {"value": "tampered"}}),))

    candidates = load_recovery_candidates(project)
    assert candidates[0]["revision"] == 2
    assert candidates[0]["valid"] is False
    assert candidates[0]["error"] == "project_snapshot_hash_mismatch"
    assert candidates[1]["revision"] == 1
    assert candidates[1]["valid"] is True
    assert candidates[1]["payload"]["config"]["value"] == "safe-revision-one"
    with pytest.raises(ValueError, match="project_snapshot_hash_mismatch"):
        save_project_revision(project, {"ui": {"page": 3}}, 2)
    with sqlite3.connect(project["db_path"]) as connection:
        row = connection.execute("SELECT payload_json,payload_sha256 FROM revisions WHERE revision=2").fetchone()
    assert row[0] == json.dumps({"config": {"value": "tampered"}})
    assert row[1] != __import__("hashlib").sha256(row[0].encode("utf-8")).hexdigest()


def test_last_good_draft_export_survives_corrupt_database_and_can_restore_elsewhere(tmp_path: Path):
    project = open_project(tmp_path / "source")
    save_project_revision(project, {"config": {"value": "recoverable"}, "markers": []}, 0)
    database = Path(project["db_path"])
    original = database.read_bytes()
    database.write_bytes(b"corrupt primary database")
    with pytest.raises(ValueError, match="project_database_corrupt_not_overwritten"):
        open_project(project["path"])
    assert database.read_bytes() == b"corrupt primary database"

    backup = load_last_good_draft_export(project["path"])
    assert backup["snapshot"]["config"]["value"] == "recoverable"
    restored = tmp_path / "restored"
    from eda_toolkit.project_store import restore_last_good_draft_export
    new_project = restore_last_good_draft_export(project["path"], restored)
    assert read_project_snapshot(new_project)["config"]["value"] == "recoverable"
    assert database.read_bytes() != original
    assert database.read_bytes() == b"corrupt primary database"

