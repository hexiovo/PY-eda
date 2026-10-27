"""Transactional task and attempt persistence for a single local project."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .project_store import _canonical_json, _connect, _project_path


_ACTIVE_STATES = {"claimed", "running", "cancellation_requested"}
_TERMINAL_STATES = {"succeeded", "partial", "failed", "cancelled"}
_TASK_STATES = {"queued", "claimed", "running", "cancellation_requested", "succeeded",
                "partial", "failed", "cancelled", "interrupted", "attention"}
_TRANSITIONS = {
    "queued": {"claimed", "cancelled"},
    "claimed": {"running", "cancellation_requested", "succeeded", "partial", "failed", "interrupted", "attention"},
    "running": {"cancellation_requested", "succeeded", "partial", "failed", "cancelled", "interrupted", "attention"},
    "cancellation_requested": {"succeeded", "partial", "cancelled", "failed", "interrupted", "attention"},
    "interrupted": {"queued", "attention", "succeeded", "partial"},
    # Terminal manifest-backed reconciliation is allowed after process death is proven.
    "attention": {"queued", "interrupted", "succeeded", "partial", "failed", "cancelled"},
    "succeeded": set(), "partial": {"queued"}, "failed": {"queued"}, "cancelled": {"queued"},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _project_state_row(connection: sqlite3.Connection, project_id: str) -> dict[str, Any]:
    row = connection.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
    if row is None:
        raise ValueError("project_not_open")
    return {key: row[key] for key in (
        "project_id", "project_revision", "draft_revision", "marker_revision", "qc_revision",
        "current_snapshot_revision", "project_path",
    )}


def _freshness_check(report: dict[str, Any], config: dict[str, Any], input_manifest: Any,
                     project_state: dict[str, Any]) -> dict[str, Any]:
    try:
        from .runtime_preflight import validate_preflight_snapshot
    except ImportError as error:
        raise RuntimeError("runtime_preflight_not_available") from error
    result = validate_preflight_snapshot(report, config, input_manifest, project_state)
    if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
        raise ValueError("preflight_freshness_result_invalid")
    return result


def _decode_task(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    for key in ("config_json", "input_manifest_json", "preflight_report_json"):
        try:
            item[key.removesuffix("_json")] = json.loads(item[key])
        except (TypeError, json.JSONDecodeError):
            item[key.removesuffix("_json")] = None
    item["cancel_requested"] = bool(item.get("cancel_requested"))
    return item


def enqueue_task(project: dict[str, Any], frozen_spec: dict[str, Any], preflight_report: dict[str, Any]) -> str:
    """Freeze a validated preflight snapshot in one transaction; never starts work."""
    if not isinstance(project, dict) or not isinstance(frozen_spec, dict) or not isinstance(preflight_report, dict):
        raise TypeError("project_frozen_spec_and_preflight_must_be_objects")
    config = frozen_spec.get("config")
    input_manifest = frozen_spec.get("input_manifest")
    if not isinstance(config, dict) or "input_manifest" not in frozen_spec:
        raise ValueError("frozen_spec_requires_config_and_input_manifest_field")
    if preflight_report.get("eligible_to_enqueue") is not True:
        raise ValueError("preflight_not_eligible_to_enqueue")
    blocking = [item for item in preflight_report.get("errors", [])
                if isinstance(item, dict) and item.get("blocking") is True]
    if blocking:
        raise ValueError("preflight_has_blocking_errors")
    if not isinstance(preflight_report.get("report_id"), str) or not preflight_report.get("report_id"):
        raise ValueError("preflight_report_id_required")
    report_sha = preflight_report.get("report_sha256")
    if not isinstance(report_sha, str) or len(report_sha) != 64:
        raise ValueError("preflight_report_sha256_required")
    project_path = _project_path(project)
    project_id = str(project.get("project_id", ""))
    uuid.UUID(project_id)
    expected_revision = frozen_spec.get("project_revision", project.get("project_revision"))
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
        raise ValueError("frozen_project_revision_invalid")
    task_id = str(frozen_spec.get("task_id") or uuid.uuid4())
    run_id = str(frozen_spec.get("run_id") or uuid.uuid4())
    uuid.UUID(task_id)
    uuid.UUID(run_id)
    config_json = _canonical_json(config)
    config_sha = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    input_json = _canonical_json(input_manifest)
    input_hash = hashlib.sha256(input_json.encode("utf-8")).hexdigest()
    report_json = _canonical_json(preflight_report)
    now = _utc_now()

    connection = _connect(project_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        project_state = _project_state_row(connection, project_id)
        project_state["project_directory"] = project_state["project_path"]
        if int(project_state["project_revision"]) != expected_revision:
            raise RuntimeError("preflight_stale_project_revision")
        if int(project_state["draft_revision"]) != int(frozen_spec.get("draft_revision", project_state["draft_revision"])):
            raise RuntimeError("preflight_stale_draft_revision")
        if int(project_state["marker_revision"]) != int(frozen_spec.get("marker_revision", project_state["marker_revision"])):
            raise RuntimeError("preflight_stale_marker_revision")
        if int(project_state["qc_revision"]) != int(frozen_spec.get("qc_revision", project_state["qc_revision"])):
            raise RuntimeError("preflight_stale_qc_revision")
        freshness = _freshness_check(preflight_report, config, input_manifest, project_state)
        if freshness.get("valid") is not True or freshness.get("stale") is True:
            raise RuntimeError("preflight_snapshot_stale_or_invalid")
        active = connection.execute(
            "SELECT task_id FROM tasks WHERE project_id=? AND state IN ('claimed','running','cancellation_requested') LIMIT 1",
            (project_id,),
        ).fetchone()
        if active:
            raise RuntimeError("project_task_already_active")
        row = connection.execute("SELECT COALESCE(MAX(ordinal),0)+1 AS next_ordinal FROM tasks WHERE project_id=?",
                                 (project_id,)).fetchone()
        ordinal = int(row["next_ordinal"])
        connection.execute(
            "INSERT INTO tasks(task_id,project_id,ordinal,created_at,state,draft_revision,config_json,config_sha256,"
            "input_manifest_json,inputs_hash,marker_revision,qc_revision,preflight_report_json,run_id,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (task_id, project_id, ordinal, now, "queued", int(project_state["draft_revision"]), config_json,
             config_sha, input_json, input_hash, int(project_state["marker_revision"]),
             int(project_state["qc_revision"]), report_json, run_id, now),
        )
        # Revalidate immediately before commit while the project revision is locked.
        project_state = _project_state_row(connection, project_id)
        project_state["project_directory"] = project_state["project_path"]
        freshness = _freshness_check(preflight_report, config, input_manifest, project_state)
        if freshness.get("valid") is not True or freshness.get("stale") is True:
            raise RuntimeError("preflight_snapshot_changed_during_enqueue")
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return task_id


def get_task(project: Any, task_id: str) -> dict[str, Any] | None:
    path = _project_path(project)
    project_id = str(project.get("project_id")) if isinstance(project, dict) else None
    connection = _connect(path)
    try:
        if project_id is None:
            row = connection.execute("SELECT project_id FROM projects WHERE project_path=?", (str(path),)).fetchone()
            if row is None:
                return None
            project_id = str(row["project_id"])
        return _decode_task(connection.execute("SELECT * FROM tasks WHERE project_id=? AND task_id=?",
                                               (project_id, str(task_id))).fetchone())
    finally:
        connection.close()


def list_task_attempts(project: Any, task_id: str) -> list[dict[str, Any]]:
    path = _project_path(project)
    connection = _connect(path)
    try:
        rows = connection.execute(
            "SELECT * FROM task_attempts WHERE task_id=? ORDER BY started_at,attempt_id", (str(task_id),)
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            if item.get("error_json"):
                try:
                    item["error"] = json.loads(item["error_json"])
                except json.JSONDecodeError:
                    item["error"] = {"message": "stored_error_json_invalid"}
            result.append(item)
        return result
    finally:
        connection.close()


def list_tasks(project: Any, *, states: list[str] | None = None, limit: int = 500,
               newest_first: bool = False) -> list[dict[str, Any]]:
    if isinstance(limit, bool) or int(limit) < 1 or int(limit) > 10000:
        raise ValueError("task_limit_out_of_range")
    if states is not None and any(state not in _TASK_STATES for state in states):
        raise ValueError("task_state_invalid")
    path = _project_path(project)
    project_id = str(project.get("project_id")) if isinstance(project, dict) else None
    connection = _connect(path)
    try:
        if project_id is None:
            row = connection.execute("SELECT project_id FROM projects WHERE project_path=?", (str(path),)).fetchone()
            if row is None:
                return []
            project_id = str(row["project_id"])
        order = "DESC" if newest_first else "ASC"
        if states:
            placeholders = ",".join("?" for _ in states)
            rows = connection.execute(
                f"SELECT * FROM tasks WHERE project_id=? AND state IN ({placeholders}) ORDER BY ordinal {order} LIMIT ?",
                (project_id, *states, int(limit)),
            ).fetchall()
        else:
            rows = connection.execute(
                f"SELECT * FROM tasks WHERE project_id=? ORDER BY ordinal {order} LIMIT ?",
                (project_id, int(limit)),
            ).fetchall()
        return [item for item in (_decode_task(row) for row in rows) if item is not None]
    finally:
        connection.close()


def query_tasks(project: Any, query: str = "", *, states: list[str] | None = None,
                limit: int = 500) -> list[dict[str, Any]]:
    rows = list_tasks(project, states=states, limit=max(int(limit), 1), newest_first=True)
    token = str(query).strip().casefold()
    if not token:
        return rows
    return [item for item in rows if token in " ".join(str(item.get(key, "")) for key in
            ("task_id", "run_id", "state", "config", "input_manifest")).casefold()]


def request_task_cancel(project: Any, task_id: str) -> dict[str, Any]:
    path = _project_path(project)
    project_id = str(project.get("project_id")) if isinstance(project, dict) else None
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        if project_id is None:
            found = connection.execute("SELECT project_id FROM projects WHERE project_path=?", (str(path),)).fetchone()
            if found is None:
                raise ValueError("project_not_open")
            project_id = str(found["project_id"])
        row = connection.execute("SELECT * FROM tasks WHERE project_id=? AND task_id=?",
                                 (project_id, str(task_id))).fetchone()
        if row is None:
            raise KeyError("task_not_found")
        state = str(row["state"])
        if state == "queued":
            next_state = "cancelled"
        elif state in {"claimed", "running"}:
            next_state = "cancellation_requested"
        elif state == "cancellation_requested":
            next_state = state
        else:
            next_state = state
        if next_state != state:
            if next_state not in _TRANSITIONS[state]:
                raise RuntimeError(f"invalid_task_transition:{state}->{next_state}")
            now = _utc_now()
            connection.execute("UPDATE tasks SET state=?,cancel_requested=1,updated_at=? WHERE task_id=?",
                               (next_state, now, str(task_id)))
            attempt_id = row["active_attempt_id"]
            if next_state == "cancelled" and attempt_id:
                connection.execute("UPDATE task_attempts SET state='cancelled',ended_at=? WHERE attempt_id=?",
                                   (now, attempt_id))
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    result = get_task({"path": str(path), "project_id": project_id}, task_id)
    assert result is not None
    return result


def retry_task(project: Any, task_id: str, mode: str = "resume") -> str:
    """Append an explicit resume/retry attempt and queue the same task for user start."""
    if mode not in {"resume", "retry"}:
        raise ValueError("retry_mode_must_be_resume_or_retry")
    path = _project_path(project)
    project_id = str(project.get("project_id")) if isinstance(project, dict) else None
    connection = _connect(path)
    attempt_id = str(uuid.uuid4())
    now = _utc_now()
    try:
        connection.execute("BEGIN IMMEDIATE")
        if project_id is None:
            found = connection.execute("SELECT project_id FROM projects WHERE project_path=?", (str(path),)).fetchone()
            if found is None:
                raise ValueError("project_not_open")
            project_id = str(found["project_id"])
        row = connection.execute("SELECT * FROM tasks WHERE project_id=? AND task_id=?",
                                 (project_id, str(task_id))).fetchone()
        if row is None:
            raise KeyError("task_not_found")
        state = str(row["state"])
        if state not in {"partial", "failed", "cancelled", "interrupted", "attention"}:
            raise RuntimeError("task_state_not_retryable")
        if connection.execute("SELECT 1 FROM tasks WHERE project_id=? AND state IN ('claimed','running','cancellation_requested') LIMIT 1",
                              (project_id,)).fetchone():
            raise RuntimeError("project_task_already_active")
        parent_id = row["active_attempt_id"]
        connection.execute(
            "INSERT INTO task_attempts(attempt_id,task_id,parent_attempt_id,kind,state,started_at) "
            "VALUES(?,?,?,?,?,?)", (attempt_id, str(task_id), parent_id, mode, "queued", now),
        )
        connection.execute(
            "UPDATE tasks SET state='queued',active_attempt_id=?,cancel_requested=0,lease_owner=NULL,"
            "lease_expires=NULL,updated_at=? WHERE task_id=?", (attempt_id, now, str(task_id)),
        )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return attempt_id


def reorder_queued_task(project: Any, task_id: str, new_ordinal: int) -> None:
    """Move one queued row to an ordinal; running and terminal history stays fixed."""
    if isinstance(new_ordinal, bool) or int(new_ordinal) < 1:
        raise ValueError("task_ordinal_invalid")
    path = _project_path(project)
    project_id = str(project.get("project_id")) if isinstance(project, dict) else None
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        if project_id is None:
            found = connection.execute("SELECT project_id FROM projects WHERE project_path=?", (str(path),)).fetchone()
            if found is None:
                raise ValueError("project_not_open")
            project_id = str(found["project_id"])
        target = connection.execute("SELECT state,ordinal FROM tasks WHERE project_id=? AND task_id=?",
                                    (project_id, str(task_id))).fetchone()
        if target is None:
            raise KeyError("task_not_found")
        if target["state"] != "queued":
            raise RuntimeError("only_queued_tasks_can_move")
        queued = connection.execute(
            "SELECT task_id,ordinal FROM tasks WHERE project_id=? AND state='queued' ORDER BY ordinal", (project_id,)
        ).fetchall()
        slots = [int(row["ordinal"]) for row in queued]
        order = [row["task_id"] for row in queued if row["task_id"] != str(task_id)]
        order.insert(min(int(new_ordinal) - 1, len(order)), str(task_id))
        max_ordinal = int(connection.execute(
            "SELECT COALESCE(MAX(ordinal),0) FROM tasks WHERE project_id=?", (project_id,)
        ).fetchone()[0])
        # Temporary ordinals above every saved task avoid CHECK and UNIQUE constraints.
        for index, identifier in enumerate(order, 1):
            connection.execute("UPDATE tasks SET ordinal=? WHERE task_id=?", (max_ordinal + index, identifier))
        for slot, identifier in zip(slots, order):
            connection.execute("UPDATE tasks SET ordinal=?,updated_at=? WHERE task_id=?",
                               (slot, _utc_now(), identifier))
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()

