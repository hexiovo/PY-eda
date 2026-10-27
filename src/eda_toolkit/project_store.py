"""Local project identity, revision snapshots, and SQLite schema management.

Project data stays on the selected local project directory. This module has no Qt
dependency; callers pass JSON-compatible snapshots and keep widgets on the UI
thread.
"""

from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ENTRY = "project.json"
DATABASE_RELATIVE = Path(".eda") / "state.sqlite"
LAST_GOOD_DRAFT_RELATIVE = Path(".eda") / "last-good-draft.json"
_VALID_REVISION_KINDS = {"autosave", "project", "marker", "qc", "recovery"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _project_path(project: Any) -> Path:
    value = project.get("path") if isinstance(project, dict) else project
    if value is None:
        raise ValueError("project_path_required")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError("project_path_must_be_directory")
    return path


def _database_path(project: Any) -> Path:
    return _project_path(project) / DATABASE_RELATIVE


def _connect(project: Any) -> sqlite3.Connection:
    path = _database_path(project)
    if not path.is_file():
        raise FileNotFoundError(f"project_database_missing:{path}")
    connection = sqlite3.connect(path, timeout=10.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=10000")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _initialize_database(connection: sqlite3.Connection, project_id: str, path: Path) -> None:
    """Apply the initial durable schema; refuse an unknown future schema."""
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if user_version > 2:
        raise ValueError(f"project_schema_newer_than_application:{user_version}")
    connection.execute("PRAGMA journal_mode=WAL")
    if user_version == 0:
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                project_path TEXT NOT NULL UNIQUE,
                project_revision INTEGER NOT NULL DEFAULT 0 CHECK(project_revision >= 0),
                draft_revision INTEGER NOT NULL DEFAULT 0 CHECK(draft_revision >= 0),
                marker_revision INTEGER NOT NULL DEFAULT 0 CHECK(marker_revision >= 0),
                qc_revision INTEGER NOT NULL DEFAULT 0 CHECK(qc_revision >= 0),
                current_snapshot_revision INTEGER NOT NULL DEFAULT 0 CHECK(current_snapshot_revision >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS revisions (
                project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                revision INTEGER NOT NULL CHECK(revision > 0),
                parent_revision INTEGER NOT NULL DEFAULT 0 CHECK(parent_revision >= 0),
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(project_id, revision)
            );
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL CHECK(ordinal > 0),
                created_at TEXT NOT NULL,
                state TEXT NOT NULL,
                draft_revision INTEGER NOT NULL CHECK(draft_revision >= 0),
                config_json TEXT NOT NULL,
                config_sha256 TEXT NOT NULL,
                input_manifest_json TEXT NOT NULL,
                inputs_hash TEXT NOT NULL,
                marker_revision INTEGER NOT NULL CHECK(marker_revision >= 0),
                qc_revision INTEGER NOT NULL CHECK(qc_revision >= 0),
                preflight_report_json TEXT NOT NULL,
                run_id TEXT,
                active_attempt_id TEXT,
                lease_owner TEXT,
                lease_expires TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN (0,1)),
                updated_at TEXT NOT NULL,
                UNIQUE(project_id, ordinal)
            );
            CREATE INDEX IF NOT EXISTS tasks_queue_order
                ON tasks(project_id, state, ordinal);
            CREATE TABLE IF NOT EXISTS task_attempts (
                attempt_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                parent_attempt_id TEXT,
                kind TEXT NOT NULL,
                state TEXT NOT NULL,
                pid INTEGER,
                process_start_identity TEXT,
                run_directory TEXT,
                started_at TEXT,
                ended_at TEXT,
                manifest_sha256 TEXT,
                error_json TEXT
            );
            CREATE INDEX IF NOT EXISTS task_attempts_task
                ON task_attempts(task_id, started_at);
            CREATE TABLE IF NOT EXISTS marker_actions (
                action_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                revision INTEGER NOT NULL CHECK(revision > 0),
                parent_revision INTEGER NOT NULL CHECK(parent_revision >= 0),
                operator TEXT NOT NULL,
                created_at TEXT NOT NULL,
                operation TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT,
                reason TEXT
            );
            CREATE TABLE IF NOT EXISTS qc_actions (
                action_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
                revision INTEGER NOT NULL CHECK(revision > 0),
                parent_revision INTEGER NOT NULL CHECK(parent_revision >= 0),
                operator TEXT NOT NULL,
                created_at TEXT NOT NULL,
                operation TEXT NOT NULL,
                before_json TEXT,
                after_json TEXT,
                reason TEXT,
                snapshot_sha256 TEXT NOT NULL
            );
            PRAGMA user_version=2;
            COMMIT;
            """
        )
    elif user_version == 1:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(task_attempts)").fetchall()}
        if "run_directory" not in columns:
            connection.execute("ALTER TABLE task_attempts ADD COLUMN run_directory TEXT")
        connection.execute("PRAGMA user_version=2")
    existing = connection.execute(
        "SELECT project_id FROM projects WHERE project_path=?", (str(path),)
    ).fetchone()
    other = connection.execute(
        "SELECT project_path FROM projects WHERE project_id=?", (project_id,)
    ).fetchone()
    if existing is None and other is not None and str(Path(other["project_path"]).resolve()) != str(path):
        raise ValueError("project_id_path_mismatch")
    if existing is None:
        now = _utc_now()
        connection.execute(
            "INSERT INTO projects(project_id, project_path, created_at, updated_at) VALUES(?,?,?,?)",
            (project_id, str(path), now, now),
        )
    elif existing["project_id"] != project_id:
        raise ValueError("project_entry_database_id_mismatch")


def _write_project_entry(path: Path, project_id: str, created_at: str, revision: int) -> None:
    entry_path = path / PROJECT_ENTRY
    entry = {
        "schema_version": "1.0",
        "project_id": project_id,
        "database": str(DATABASE_RELATIVE.as_posix()),
        "created_at": created_at,
        "updated_at": _utc_now(),
        "last_revision": int(revision),
    }
    encoded = json.dumps(entry, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.mkdir(parents=True, exist_ok=True)
    with _project_metadata_lock(path):
        if entry_path.is_file():
            backup = path / f"{PROJECT_ENTRY}.last-good"
            previous = entry_path.read_bytes()
            _atomic_write_bytes(backup, previous, prefix=f".{backup.name}.")
        _atomic_write_bytes(entry_path, encoded.encode("utf-8"), prefix=f".{PROJECT_ENTRY}.")


@contextlib.contextmanager
def _project_metadata_lock(path: Path):
    """Serialize project.json/last-good exports across windows/processes."""
    lock_path = path / ".eda" / "metadata.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _atomic_write_bytes(target: Path, payload: bytes, *, prefix: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=prefix,
                                         suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def open_project(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Open or initialize a project folder without overwriting a corrupt database."""
    project_path = _project_path(path)
    project_path.mkdir(parents=True, exist_ok=True)
    entry_path = project_path / PROJECT_ENTRY
    database = project_path / DATABASE_RELATIVE
    entry: dict[str, Any] | None = None
    if entry_path.is_file():
        try:
            parsed = json.loads(entry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            backup = project_path / f"{PROJECT_ENTRY}.last-good"
            raise ValueError(f"project_entry_corrupt:{backup if backup.exists() else entry_path}") from error
        if not isinstance(parsed, dict) or not isinstance(parsed.get("project_id"), str):
            raise ValueError("project_entry_invalid")
        try:
            uuid.UUID(parsed["project_id"])
        except (ValueError, AttributeError) as error:
            raise ValueError("project_id_invalid") from error
        entry = parsed
    elif database.exists():
        # A crash may occur after SQLite initialization but before project.json.
        # Recover the id only from a valid database; never replace unreadable state.
        connection = sqlite3.connect(database, timeout=10.0)
        try:
            row = connection.execute("SELECT project_id, created_at FROM projects LIMIT 1").fetchone()
            if row is None:
                raise ValueError("project_entry_missing_database_has_no_identity")
            project_id = str(row[0])
            created_at = str(row[1])
            uuid.UUID(project_id)
            entry = {"project_id": project_id, "created_at": created_at}
        except sqlite3.DatabaseError as error:
            raise ValueError("project_database_corrupt_not_overwritten") from error
        finally:
            connection.close()
    else:
        entry = {"project_id": str(uuid.uuid4()), "created_at": _utc_now()}

    project = {"path": str(project_path), "project_id": entry["project_id"]}
    if database.exists():
        connection = sqlite3.connect(database, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
    else:
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database, timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        _initialize_database(connection, entry["project_id"], project_path)
        row = connection.execute("SELECT * FROM projects WHERE project_id=?", (entry["project_id"],)).fetchone()
        if row is None:
            raise ValueError("project_database_identity_missing")
        project.update({
            "project_revision": int(row["project_revision"]),
            "draft_revision": int(row["draft_revision"]),
            "marker_revision": int(row["marker_revision"]),
            "qc_revision": int(row["qc_revision"]),
            "current_snapshot_revision": int(row["current_snapshot_revision"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "db_path": str(database),
        })
        _write_project_entry(project_path, project["project_id"], project["created_at"],
                              project["current_snapshot_revision"])
    except sqlite3.DatabaseError as error:
        raise ValueError("project_database_corrupt_not_overwritten") from error
    finally:
        connection.close()
    return project


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_project_snapshot(project: Any, revision: int | None = None) -> dict[str, Any]:
    """Return a verified JSON snapshot. The newest revision is used by default."""
    path = _project_path(project)
    project_id = project.get("project_id") if isinstance(project, dict) else open_project(path)["project_id"]
    connection = _connect(path)
    try:
        if revision is None:
            row = connection.execute(
                "SELECT * FROM revisions WHERE project_id=? ORDER BY revision DESC LIMIT 1", (project_id,)
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM revisions WHERE project_id=? AND revision=?", (project_id, int(revision))
            ).fetchone()
        if row is None:
            return {}
        payload_json = str(row["payload_json"])
        if _sha256_text(payload_json) != row["payload_sha256"]:
            raise ValueError("project_snapshot_hash_mismatch")
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("project_snapshot_not_object")
        return payload
    finally:
        connection.close()


def save_project_revision(
    project: dict[str, Any],
    patch: dict[str, Any],
    expected_revision: int,
    kind: str = "autosave",
    *,
    audit_table: str | None = None,
    audit_action: dict[str, Any] | None = None,
) -> int:
    """Atomically save a full JSON-compatible draft revision with CAS protection."""
    if not isinstance(project, dict) or not isinstance(patch, dict):
        raise TypeError("project_and_patch_must_be_objects")
    if kind not in _VALID_REVISION_KINDS:
        raise ValueError("project_revision_kind_invalid")
    if (audit_table is None) != (audit_action is None):
        raise ValueError("audit_table_and_action_must_be_provided_together")
    if audit_table is not None and audit_table not in {"marker_actions", "qc_actions"}:
        raise ValueError("audit_table_invalid")
    if audit_action is not None and not isinstance(audit_action, dict):
        raise TypeError("audit_action_must_be_object")
    project_path = _project_path(project)
    project_id = str(project.get("project_id", ""))
    uuid.UUID(project_id)
    connection = _connect(project_path)
    created_at = str(project.get("created_at") or _utc_now())
    now = _utc_now()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if row is None:
            raise ValueError("project_not_open")
        actual_revision = int(row["project_revision"])
        if actual_revision != int(expected_revision):
            raise RuntimeError(f"project_revision_conflict:expected={expected_revision}:actual={actual_revision}")
        latest = connection.execute(
            "SELECT payload_json,payload_sha256 FROM revisions WHERE project_id=? ORDER BY revision DESC LIMIT 1", (project_id,)
        ).fetchone()
        current_json = str(latest["payload_json"]) if latest else "{}"
        if latest and _sha256_text(current_json) != latest["payload_sha256"]:
            raise ValueError("project_snapshot_hash_mismatch")
        current = json.loads(current_json)
        if not isinstance(current, dict):
            raise ValueError("project_snapshot_not_object")
        payload = _merge_dict(current, patch)
        payload_json = _canonical_json(payload)
        if payload_json == (str(latest["payload_json"]) if latest else "{}") and audit_action is None:
            connection.execute("COMMIT")
            _write_last_good_draft_export(project, payload, actual_revision)
            return actual_revision
        payload_hash = _sha256_text(payload_json)
        revision = actual_revision + 1
        parent = int(row["current_snapshot_revision"])
        marker_revision = int(row["marker_revision"]) + (1 if kind == "marker" else 0)
        qc_revision = int(row["qc_revision"]) + (1 if kind == "qc" else 0)
        connection.execute(
            "INSERT INTO revisions(project_id,revision,parent_revision,kind,payload_json,payload_sha256,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (project_id, revision, parent, kind, payload_json, payload_hash, now),
        )
        connection.execute(
            "UPDATE projects SET project_revision=?,draft_revision=?,marker_revision=?,qc_revision=?,"
            "current_snapshot_revision=?,updated_at=? WHERE project_id=?",
            (revision, revision, marker_revision, qc_revision, revision, now, project_id),
        )
        if audit_action is not None:
            action_id = str(audit_action.get("action_id") or uuid.uuid4())
            uuid.UUID(action_id)
            action_revision = marker_revision if audit_table == "marker_actions" else qc_revision
            before_json = (None if audit_action.get("before") is None
                           else _canonical_json(audit_action["before"]))
            after_json = (None if audit_action.get("after") is None
                          else _canonical_json(audit_action["after"]))
            common = (
                action_id, project_id, action_revision, action_revision - 1,
                str(audit_action.get("operator") or "unknown"),
                str(audit_action.get("time") or now), str(audit_action.get("operation") or "edit"),
                before_json, after_json, audit_action.get("reason"),
            )
            if audit_table == "marker_actions":
                connection.execute(
                    "INSERT INTO marker_actions(action_id,project_id,revision,parent_revision,operator,created_at,"
                    "operation,before_json,after_json,reason) VALUES(?,?,?,?,?,?,?,?,?,?)", common,
                )
            else:
                snapshot_hash = audit_action.get("snapshot_sha256")
                if not isinstance(snapshot_hash, str) or len(snapshot_hash) != 64:
                    raise ValueError("qc_snapshot_hash_required")
                connection.execute(
                    "INSERT INTO qc_actions(action_id,project_id,revision,parent_revision,operator,created_at,"
                    "operation,before_json,after_json,reason,snapshot_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (*common, snapshot_hash),
                )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    project.update({
        "project_revision": revision, "draft_revision": revision,
        "marker_revision": marker_revision, "qc_revision": qc_revision,
        "current_snapshot_revision": revision, "updated_at": now,
    })
    _write_last_good_draft_export(project, payload, revision)
    _write_project_entry(project_path, project_id, created_at, revision)
    return revision


def load_recovery_candidates(project: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    """List recent saved drafts without automatically choosing or restoring one."""
    if isinstance(limit, bool) or int(limit) < 1 or int(limit) > 1000:
        raise ValueError("recovery_limit_out_of_range")
    path = _project_path(project)
    project_id = project.get("project_id") if isinstance(project, dict) else open_project(path)["project_id"]
    connection = _connect(path)
    try:
        rows = connection.execute(
            "SELECT revision,parent_revision,kind,payload_json,payload_sha256,created_at FROM revisions "
            "WHERE project_id=? ORDER BY revision DESC LIMIT ?", (project_id, int(limit))
        ).fetchall()
    finally:
        connection.close()
    candidates = []
    for row in rows:
        candidate = {
            "revision": int(row["revision"]), "parent_revision": int(row["parent_revision"]),
            "kind": str(row["kind"]), "sha256": str(row["payload_sha256"]),
            "created_at": str(row["created_at"]),
        }
        try:
            payload_json = str(row["payload_json"])
            if _sha256_text(payload_json) != row["payload_sha256"]:
                raise ValueError("project_snapshot_hash_mismatch")
            payload = json.loads(payload_json)
            if not isinstance(payload, dict):
                raise ValueError("project_snapshot_not_object")
            candidate.update({"valid": True, "payload": payload})
        except (ValueError, json.JSONDecodeError, TypeError) as error:
            candidate.update({"valid": False, "payload": None, "error": str(error)})
        candidates.append(candidate)
    return candidates


def _write_last_good_draft_export(project: dict[str, Any], snapshot: dict[str, Any], revision: int) -> Path:
    """Atomically persist a verifiable draft outside SQLite for explicit recovery."""
    path = _project_path(project)
    target = path / LAST_GOOD_DRAFT_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0", "project_id": str(project["project_id"]),
        "project_revision": int(revision), "saved_at": _utc_now(),
        "snapshot_sha256": _sha256_text(_canonical_json(snapshot)),
        "snapshot": copy.deepcopy(snapshot),
    }
    with _project_metadata_lock(path):
        # A delayed writer from a lower revision may never replace a newer backup.
        if target.is_file():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
                if int(existing.get("project_revision", -1)) > int(revision):
                    return target
            except (OSError, ValueError, json.JSONDecodeError, TypeError):
                pass
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        _atomic_write_bytes(target, encoded.encode("utf-8"), prefix=".last-good-draft.")
    return target


def load_last_good_draft_export(project: Any) -> dict[str, Any]:
    """Read and verify the full last-good snapshot without touching SQLite."""
    path = _project_path(project) / LAST_GOOD_DRAFT_RELATIVE
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"last_good_draft_unreadable:{path}") from error
    if not isinstance(envelope, dict) or not isinstance(envelope.get("snapshot"), dict):
        raise ValueError("last_good_draft_invalid")
    if envelope.get("schema_version") != "1.0":
        raise ValueError("last_good_draft_schema_unsupported")
    try:
        uuid.UUID(str(envelope.get("project_id")))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("last_good_draft_project_id_invalid") from error
    actual_hash = _sha256_text(_canonical_json(envelope["snapshot"]))
    if actual_hash != envelope.get("snapshot_sha256"):
        raise ValueError("last_good_draft_hash_mismatch")
    return envelope


def restore_last_good_draft_export(source_project: Any, destination: str | os.PathLike[str]) -> dict[str, Any]:
    """Restore a verified exported draft into a separate new project directory."""
    source = _project_path(source_project)
    target = _project_path(destination)
    if source == target:
        raise ValueError("recovery_destination_must_differ_from_corrupt_source")
    entry = target / PROJECT_ENTRY
    database = target / DATABASE_RELATIVE
    if entry.exists() or database.exists() or (target.exists() and any(target.iterdir())):
        raise ValueError("recovery_destination_already_contains_project")
    exported = load_last_good_draft_export(source)
    snapshot = copy.deepcopy(exported["snapshot"])
    config = snapshot.get("config")
    if isinstance(config, dict):
        config.setdefault("workspace", {})["project_directory"] = str(target)
    restored = open_project(target)
    save_project_revision(restored, snapshot, 0, kind="recovery")
    return restored


def append_audit_action(
    project: dict[str, Any], table: str, action: dict[str, Any], snapshot_sha256: str | None = None,
) -> None:
    """Persist append-only marker/QC action metadata in the project database."""
    if table not in {"marker_actions", "qc_actions"}:
        raise ValueError("audit_table_invalid")
    required = ("action_id", "revision", "parent_revision", "operator", "operation")
    if any(key not in action for key in required):
        raise ValueError("audit_action_missing_fields")
    project_path = _project_path(project)
    project_id = str(project["project_id"])
    action_id = str(action["action_id"])
    uuid.UUID(action_id)
    before_json = None if action.get("before") is None else _canonical_json(action["before"])
    after_json = None if action.get("after") is None else _canonical_json(action["after"])
    connection = _connect(project_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        if table == "marker_actions":
            connection.execute(
                "INSERT INTO marker_actions(action_id,project_id,revision,parent_revision,operator,created_at,"
                "operation,before_json,after_json,reason) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (action_id, project_id, int(action["revision"]), int(action["parent_revision"]),
                 str(action["operator"]), str(action.get("time") or _utc_now()), str(action["operation"]),
                 before_json, after_json, action.get("reason")),
            )
        else:
            if not isinstance(snapshot_sha256, str) or len(snapshot_sha256) != 64:
                raise ValueError("qc_snapshot_hash_required")
            connection.execute(
                "INSERT INTO qc_actions(action_id,project_id,revision,parent_revision,operator,created_at,"
                "operation,before_json,after_json,reason,snapshot_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (action_id, project_id, int(action["revision"]), int(action["parent_revision"]),
                 str(action["operator"]), str(action.get("time") or _utc_now()), str(action["operation"]),
                 before_json, after_json, action.get("reason"), snapshot_sha256),
            )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def list_audit_actions(project: Any, table: str = "marker_actions") -> list[dict[str, Any]]:
    """Return marker or QC audit history in append order."""
    if table not in {"marker_actions", "qc_actions"}:
        raise ValueError("audit_table_invalid")
    path = _project_path(project)
    project_id = project.get("project_id") if isinstance(project, dict) else open_project(path)["project_id"]
    connection = _connect(path)
    try:
        rows = connection.execute(f"SELECT * FROM {table} WHERE project_id=? ORDER BY revision,created_at,action_id",
                                  (project_id,)).fetchall()
    finally:
        connection.close()
    result = []
    for row in rows:
        item = dict(row)
        for source, target in (("before_json", "before"), ("after_json", "after")):
            item[target] = json.loads(item.pop(source)) if item.get(source) is not None else None
        result.append(item)
    return result

