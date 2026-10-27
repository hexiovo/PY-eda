"""Explicit-start task dispatcher with OS locking and crash reconciliation."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .project_store import _canonical_json, _connect, _project_path
from .task_store import get_task, list_tasks, request_task_cancel, retry_task


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _process_start_identity(pid: int | None = None) -> str | None:
    """Return an OS process creation identity where the platform exposes one."""
    target = int(pid or os.getpid())
    if os.name == "nt":
        class _FileTime(ctypes.Structure):
            _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetProcessTimes.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FileTime),
                                             ctypes.POINTER(_FileTime), ctypes.POINTER(_FileTime),
                                             ctypes.POINTER(_FileTime)]
        kernel32.GetProcessTimes.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(0x1000, False, target)
        if not handle:
            return None
        try:
            created, exited, kernel, user = _FileTime(), _FileTime(), _FileTime(), _FileTime()
            if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                            ctypes.byref(kernel), ctypes.byref(user)):
                return None
            return f"win-filetime:{(created.high << 32) | created.low}"
        finally:
            kernel32.CloseHandle(handle)
    stat = Path(f"/proc/{target}/stat")
    if stat.is_file():
        try:
            fields = stat.read_text(encoding="ascii").split()
            return f"proc-startticks:{fields[21]}"
        except (OSError, IndexError):
            return None
    return None


def _validate_queued_input_snapshot(task: dict[str, Any], project_handle: dict[str, Any]) -> dict[str, str]:
    """Recheck a task against its own frozen preflight, never the current draft revision."""
    config = task.get("config")
    input_manifest = task.get("input_manifest")
    report = task.get("preflight_report")
    if not isinstance(config, dict) or not isinstance(report, dict):
        raise RuntimeError("queued_task_preflight_snapshot_missing")
    if hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest() != task.get("config_sha256"):
        raise RuntimeError("queued_task_config_hash_mismatch")
    if hashlib.sha256(_canonical_json(input_manifest).encode("utf-8")).hexdigest() != task.get("inputs_hash"):
        raise RuntimeError("queued_task_input_manifest_hash_mismatch")
    if report.get("project_id") != task.get("project_id"):
        raise RuntimeError("queued_task_preflight_project_mismatch")
    frozen_revisions = {
        "project_id": task["project_id"],
        "project_revision": report.get("project_revision"),
        "draft_revision": task.get("draft_revision"),
        "marker_revision": task.get("marker_revision"),
        "qc_revision": task.get("qc_revision"),
        "project_directory": str(Path(project_handle["path"]).resolve()),
    }
    for field in ("project_revision", "draft_revision", "marker_revision", "qc_revision"):
        value = frozen_revisions[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"queued_task_frozen_{field}_invalid")
        if report.get(field) != value:
            raise RuntimeError(f"queued_task_frozen_{field}_mismatch")
    expected_fingerprints = report.get("input_fingerprints")
    if not isinstance(expected_fingerprints, dict):
        raise RuntimeError("queued_task_preflight_fingerprints_missing")
    from .runtime_preflight import validate_preflight_snapshot
    freshness = validate_preflight_snapshot(report, config, input_manifest, frozen_revisions)
    if (not isinstance(freshness, dict) or freshness.get("valid") is not True or
            freshness.get("stale") is True):
        errors = freshness.get("errors", []) if isinstance(freshness, dict) else []
        detail = _canonical_json(errors)[:4000]
        raise RuntimeError("queued_task_preflight_stale: " + detail)
    return {str(path): str(digest) for path, digest in expected_fingerprints.items()}


@contextlib.contextmanager
def _dispatcher_lock(project: Any):
    path = _project_path(project) / ".eda" / "dispatcher.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            raise RuntimeError("project_dispatcher_lock_held") from error
        try:
            yield handle
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


def _project_id(project: Any) -> str:
    if isinstance(project, dict) and project.get("project_id"):
        return str(project["project_id"])
    path = _project_path(project)
    connection = _connect(path)
    try:
        row = connection.execute("SELECT project_id FROM projects WHERE project_path=?", (str(path),)).fetchone()
        if row is None:
            raise ValueError("project_not_open")
        return str(row["project_id"])
    finally:
        connection.close()


def claim_next_task(project: Any, owner_token: str) -> dict[str, Any] | None:
    """Claim the next queued task atomically; this does not launch it."""
    if not isinstance(owner_token, str) or not owner_token.strip():
        raise ValueError("owner_token_required")
    path = _project_path(project)
    project_id = _project_id(project)
    connection = _connect(path)
    now = _utc_now()
    expiry = (datetime.now(timezone.utc) + timedelta(seconds=90)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    attempt_id = None
    try:
        connection.execute("BEGIN IMMEDIATE")
        active = connection.execute(
            "SELECT task_id FROM tasks WHERE project_id=? AND state IN ('claimed','running','cancellation_requested') LIMIT 1",
            (project_id,),
        ).fetchone()
        if active:
            connection.execute("COMMIT")
            return None
        row = connection.execute(
            "SELECT * FROM tasks WHERE project_id=? AND state='queued' ORDER BY ordinal LIMIT 1", (project_id,)
        ).fetchone()
        if row is None:
            connection.execute("COMMIT")
            return None
        attempt_id = row["active_attempt_id"]
        if attempt_id:
            attempt = connection.execute("SELECT state FROM task_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        else:
            attempt = None
        if attempt is None:
            attempt_id = str(uuid.uuid4())
            connection.execute(
                "INSERT INTO task_attempts(attempt_id,task_id,parent_attempt_id,kind,state,pid,process_start_identity,started_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (attempt_id, row["task_id"], None, "run", "claimed", os.getpid(), _process_start_identity(), now),
            )
        else:
            connection.execute(
                "UPDATE task_attempts SET state='claimed',pid=?,process_start_identity=?,started_at=?,ended_at=NULL "
                "WHERE attempt_id=?",
                (os.getpid(), _process_start_identity(), now, attempt_id),
            )
        connection.execute(
            "UPDATE tasks SET state='claimed',active_attempt_id=?,lease_owner=?,lease_expires=?,"
            "cancel_requested=0,updated_at=? WHERE task_id=?",
            (attempt_id, owner_token, expiry, now, row["task_id"]),
        )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    task = get_task({"path": str(path), "project_id": project_id}, row["task_id"])
    if task is None:
        raise RuntimeError("claimed_task_disappeared")
    task["attempt_id"] = attempt_id
    task["owner_token"] = owner_token
    task["project"] = {"path": str(path), "project_id": project_id}
    return task


def _resolve_output_root(config: dict[str, Any]) -> Path:
    output = Path(config.get("output", {}).get("directory", "results")).expanduser()
    if not output.is_absolute():
        base = config.get("input", {}).get("base_directory") or Path.cwd()
        output = Path(base).expanduser() / output
    return output.resolve()


def _locate_live_run(config: dict[str, Any], started_at: str) -> Path | None:
    output = _resolve_output_root(config)
    if not output.is_dir():
        return None
    try:
        started_epoch = datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        started_epoch = 0.0
    matches = []
    for child in output.iterdir():
        state_path = child / "pipeline_state.json"
        if not child.is_dir() or not state_path.is_file() or not (child / ".running.lock").is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("status") == "running" and state_path.stat().st_mtime >= started_epoch - 5:
            matches.append((state_path.stat().st_mtime, child, state))
    if len(matches) != 1:
        return None
    _mtime, run_directory, run_state = matches[0]
    return run_directory


def _write_task_binding(run_directory: Path, task: dict[str, Any]) -> dict[str, Any]:
    state_path = run_directory / "pipeline_state.json"
    run_state = json.loads(state_path.read_text(encoding="utf-8"))
    binding = {
        "schema_version": "1.0",
        "task_id": task["task_id"],
        "attempt_id": task["attempt_id"],
        "project_id": task["project_id"],
        "run_id": run_state.get("run_id"),
        "config_sha256": task["config_sha256"],
        "inputs_hash": task["inputs_hash"],
        "bound_at": _utc_now(),
    }
    target = run_directory / "task_binding.json"
    temp = run_directory / f".task_binding.{uuid.uuid4().hex}.tmp"
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(binding, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)
    return {**binding, "run_directory": str(run_directory.resolve())}


def _update_run_binding(project: Any, task: dict[str, Any], run_directory: Path) -> dict[str, Any]:
    binding = _write_task_binding(run_directory, task)
    path = _project_path(project)
    connection = _connect(path)
    now = _utc_now()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("UPDATE tasks SET run_id=?,updated_at=? WHERE task_id=?",
                           (binding["run_id"], now, task["task_id"]))
        connection.execute("UPDATE task_attempts SET run_directory=? WHERE attempt_id=?",
                           (binding["run_directory"], task["attempt_id"]))
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    task.update(binding)
    return binding


def _refresh_lease(project: Any, task: dict[str, Any]) -> bool:
    path = _project_path(project)
    connection = _connect(path)
    now = _utc_now()
    expiry = (datetime.now(timezone.utc) + timedelta(seconds=90)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT state,cancel_requested FROM tasks WHERE task_id=?", (task["task_id"],)).fetchone()
        if row is None:
            connection.execute("ROLLBACK")
            raise KeyError("task_not_found")
        cancelled = bool(row["cancel_requested"]) or row["state"] == "cancellation_requested"
        connection.execute("UPDATE tasks SET lease_expires=?,updated_at=? WHERE task_id=?",
                           (expiry, now, task["task_id"]))
        connection.execute("COMMIT")
        return cancelled
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _publish(event_sink: Any, event: dict[str, Any]) -> None:
    try:
        from .runtime_events import emit_run_event
    except ImportError:
        if callable(event_sink):
            event_sink(event)
        else:
            raise RuntimeError("runtime_event_sink_invalid")
    else:
        emit_run_event(event_sink, event)


def _set_task_and_attempt(project: Any, task: dict[str, Any], state: str, *, error: Any = None,
                          manifest_sha256: str | None = None, run_id: str | None = None) -> None:
    path = _project_path(project)
    connection = _connect(path)
    now = _utc_now()
    error_json = None if error is None else _canonical_json(error)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT state,active_attempt_id FROM tasks WHERE task_id=?", (task["task_id"],)).fetchone()
        if row is None:
            raise KeyError("task_not_found")
        old_state = str(row["state"])
        if state != old_state:
            from .task_store import _TRANSITIONS
            if state not in _TRANSITIONS.get(old_state, set()):
                raise RuntimeError(f"invalid_task_transition:{old_state}->{state}")
        attempt_id = row["active_attempt_id"]
        if attempt_id:
            connection.execute(
                "UPDATE task_attempts SET state=?,ended_at=?,manifest_sha256=COALESCE(?,manifest_sha256),"
                "error_json=? WHERE attempt_id=?",
                (state, now if state in {"succeeded", "partial", "failed", "cancelled", "interrupted"} else None,
                 manifest_sha256, error_json, attempt_id),
            )
        connection.execute(
            "UPDATE tasks SET state=?,run_id=COALESCE(?,run_id),lease_owner=NULL,lease_expires=NULL,"
            "cancel_requested=?,updated_at=? WHERE task_id=?",
            (state, run_id, 0 if state in {"succeeded", "partial", "failed", "cancelled"} else 1,
             now, task["task_id"]),
        )
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def execute_claimed_task(task_id: str, event_sink: Any, project: Any = None,
                         runtime_session: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one claimed frozen task while holding the project dispatcher lock."""
    if project is None:
        raise ValueError("project_handle_required_for_task_execution")
    task = get_task(project, task_id)
    if task is None:
        raise KeyError("task_not_found")
    if task["state"] not in {"claimed", "cancellation_requested"}:
        raise RuntimeError("task_must_be_claimed_before_execution")
    attempt_id = task.get("active_attempt_id")
    if not attempt_id:
        raise RuntimeError("task_attempt_missing")
    task["attempt_id"] = attempt_id
    project_id = task["project_id"]
    project_handle = {"path": str(_project_path(project)), "project_id": project_id}
    task["attempt_info"] = _attempt_row(project_handle, attempt_id) or {}
    started_mono = time.monotonic()
    run_directory = None
    cancel_event_published = False

    with _dispatcher_lock(project_handle):
        connection = _connect(project_handle)
        now = _utc_now()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT state FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None or row["state"] not in {"claimed", "cancellation_requested"}:
                raise RuntimeError("claimed_task_state_changed")
            if row["state"] == "claimed":
                connection.execute("UPDATE tasks SET state='running',updated_at=? WHERE task_id=?", (now, task_id))
                connection.execute("UPDATE task_attempts SET state='running',pid=?,process_start_identity=? WHERE attempt_id=?",
                                   (os.getpid(), _process_start_identity(), attempt_id))
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

        try:
            from .pipeline import resume_run, run_pipeline
            config = task["config"]
            input_manifest = task["input_manifest"]
            expected_input_fingerprints = _validate_queued_input_snapshot(task, project_handle)
            runtime_context = {"expected_input_fingerprints": expected_input_fingerprints}
            attempt_kind = str(task["attempt_info"].get("kind") or "run")
            resume_source = None
            if attempt_kind == "resume":
                parent = _attempt_row(project_handle, task["attempt_info"].get("parent_attempt_id"))
                if not parent or not parent.get("run_directory"):
                    raise RuntimeError("resume_source_run_directory_missing")
                resume_source = Path(parent["run_directory"])

            def cancellation_requested() -> bool:
                return _refresh_lease(project_handle, task)

            def progress_callback(raw: dict[str, Any]) -> None:
                nonlocal run_directory, cancel_event_published
                if run_directory is None:
                    run_directory = _locate_live_run(
                        config, task["attempt_info"].get("started_at") or task["created_at"])
                    if run_directory is not None:
                        binding = _update_run_binding(project_handle, task, run_directory)
                        task.update(binding)
                cancelled = cancellation_requested()
                stage = str(raw.get("stage") or "pipeline")
                # Detailed stage/work-unit lifecycle is emitted directly by pipeline
                # through this same runtime controller. This compatibility event only
                # carries a human-readable progress message and never counts work twice.
                raw_details = {"legacy_pipeline_progress": raw}
                event = {
                    "schema_version": "1.0", "event_id": str(uuid.uuid4()), "seq": None,
                    "timestamp_utc": _utc_now(), "monotonic_s": time.monotonic(),
                    "level": "INFO", "project_id": project_id, "task_id": task_id,
                    "attempt_id": attempt_id, "run_id": task.get("run_id"), "stage": stage,
                    "record_id": str(raw.get("record_id") or "") or None, "kind": "progress",
                    "message": str(raw.get("message") or ""), "units_done": 0,
                    "units_total": None, "unit_type": "",
                    "details": raw_details,
                }
                _publish(event_sink, event)
                if cancelled and not cancel_event_published:
                    cancel_event_published = True
                    _publish(event_sink, {
                        **event, "event_id": str(uuid.uuid4()), "seq": None,
                        "kind": "cancel_requested", "message": "任务收到取消请求，将在安全检查点停止。",
                    })

            if resume_source is not None:
                summary = resume_run(resume_source, progress_callback=progress_callback,
                                     cancel_callback=cancellation_requested,
                                     event_sink=event_sink, runtime_session=runtime_session,
                                     project_id=project_id, task_id=task_id, attempt_id=attempt_id,
                                     runtime_context=runtime_context)
            else:
                summary = run_pipeline(config, input_manifest=input_manifest,
                                       progress_callback=progress_callback,
                                       cancel_callback=cancellation_requested,
                                       event_sink=event_sink, runtime_session=runtime_session,
                                       project_id=project_id, task_id=task_id, attempt_id=attempt_id,
                                       runtime_context=runtime_context)
            directory_text = summary.get("directory")
            directory = Path(directory_text) if directory_text else run_directory
            manifest_hash = None
            manifest_status = None
            if directory and (directory / "manifest.json").is_file():
                from .core_export import sha256_file, verify_manifest
                if verify_manifest(directory)["valid"]:
                    manifest_hash = sha256_file(directory / "manifest.json")
                    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                    manifest_status = manifest.get("status")
            outcome = str(summary.get("status", "failed"))
            if outcome == "succeeded" and manifest_hash is None:
                outcome = "partial"
                summary.setdefault("warnings", []).append("manifest未通过完整性校验，任务不记为succeeded。")
            outcome = outcome if outcome in {"succeeded", "partial", "failed", "cancelled"} else "failed"
            _set_task_and_attempt(project_handle, task, outcome,
                                  error=summary.get("errors") if summary.get("errors") else None,
                                  manifest_sha256=manifest_hash, run_id=summary.get("run_id"))
            event_kind = {"succeeded": "completed", "partial": "warning",
                          "failed": "failed", "cancelled": "cancelled"}[outcome]
            completed_records = int(summary.get("completed_records", 0) or 0)
            total_records = len(summary.get("record_statuses", []))
            _publish(event_sink, {
                "schema_version": "1.0", "event_id": str(uuid.uuid4()), "seq": None,
                "timestamp_utc": _utc_now(), "monotonic_s": time.monotonic(),
                "level": "INFO" if outcome == "succeeded" else "WARNING", "project_id": project_id,
                "task_id": task_id, "attempt_id": attempt_id, "run_id": summary.get("run_id"),
                "stage": "task", "record_id": "", "kind": event_kind,
                "message": f"任务结束：{outcome}", "units_done": completed_records,
                "units_total": total_records if outcome == "succeeded" else None, "unit_type": "record",
                "details": {"directory": str(directory) if directory else None, "manifest_status": manifest_status,
                            "run_status": outcome, "manifest_finalized": bool(manifest_hash and outcome == "succeeded")},
            })
            return summary
        except BaseException as error:
            _set_task_and_attempt(project_handle, task, "failed", error={"type": type(error).__name__, "message": str(error)})
            _publish(event_sink, {
                "schema_version": "1.0", "event_id": str(uuid.uuid4()), "seq": None,
                "timestamp_utc": _utc_now(), "monotonic_s": time.monotonic(),
                "level": "ERROR", "project_id": project_id, "task_id": task_id,
                "attempt_id": attempt_id, "run_id": task.get("run_id"),
                "stage": "preflight" if str(error).startswith("queued_task_preflight") else "task",
                "record_id": "", "kind": "failed",
                "message": f"任务在分析前/执行期间失败：{error}",
                "units_done": 0, "units_total": None, "unit_type": "",
                "details": {"error_type": type(error).__name__, "reason": str(error),
                            "remediation": "若输入、配置或环境已变化，请重新运行预检并创建新任务。"},
            })
            raise


def reconcile_interrupted_tasks(project: Any, live_owner_probe: Callable[[dict[str, Any]], bool | None]) -> dict[str, Any]:
    """Reconcile only after dispatcher exclusion; never steal a task with uncertain liveness."""
    if not callable(live_owner_probe):
        raise TypeError("live_owner_probe_must_be_callable")
    path = _project_path(project)
    project_handle = {"path": str(path), "project_id": _project_id(project)}
    try:
        lock = _dispatcher_lock(project_handle)
        lock.__enter__()
    except RuntimeError:
        return {"status": "attention", "reason": "dispatcher_lock_held", "tasks": []}
    examined, recovered, interrupted, attention = [], [], [], []
    try:
        connection = _connect(project_handle)
        try:
            rows = connection.execute(
                "SELECT * FROM tasks WHERE project_id=? AND state IN ('claimed','running','cancellation_requested','attention') ORDER BY ordinal",
                (project_handle["project_id"],),
            ).fetchall()
            tasks = [_decode_row(row) for row in rows]
        finally:
            connection.close()
        for task in tasks:
            attempt = _attempt_row(project_handle, task.get("active_attempt_id"))
            task["attempt"] = attempt
            examined.append(task["task_id"])
            try:
                live = live_owner_probe(task)
            except Exception:
                live = None
            if live is True:
                attention.append({"task_id": task["task_id"], "reason": "owner_process_live"})
                continue
            if live is not False:
                _transition_reconcile(project_handle, task, "attention", "owner_liveness_unproven")
                attention.append({"task_id": task["task_id"], "reason": "owner_liveness_unproven"})
                continue
            directory_text = (attempt or {}).get("run_directory")
            recovered_state = None
            if directory_text:
                directory = Path(directory_text)
                binding_path = directory / "task_binding.json"
                manifest_path = directory / "manifest.json"
                if binding_path.is_file() and manifest_path.is_file():
                    try:
                        binding = json.loads(binding_path.read_text(encoding="utf-8"))
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        from .core_export import verify_manifest
                        verified = verify_manifest(directory)
                        bound = (binding.get("task_id") == task["task_id"] and
                                 binding.get("attempt_id") == task.get("active_attempt_id") and
                                 binding.get("project_id") == project_handle["project_id"])
                        if verified.get("valid") and bound and manifest.get("status") in {"succeeded", "partial", "failed", "cancelled"}:
                            recovered_state = manifest["status"]
                            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                            _transition_reconcile(project_handle, task, recovered_state, "complete_manifest_reconciled",
                                                  manifest_hash=manifest_hash)
                            recovered.append({"task_id": task["task_id"], "state": recovered_state})
                            continue
                    except (OSError, ValueError, json.JSONDecodeError):
                        recovered_state = None
            _transition_reconcile(project_handle, task, "interrupted", "no_verified_complete_manifest")
            interrupted.append(task["task_id"])
    finally:
        lock.__exit__(None, None, None)
    return {"status": "reconciled", "examined": examined, "recovered": recovered,
            "interrupted": interrupted, "attention": attention}


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("config_json", "input_manifest_json", "preflight_report_json"):
        item[key.removesuffix("_json")] = json.loads(item[key])
    return item


def _attempt_row(project: Any, attempt_id: str | None) -> dict[str, Any] | None:
    if not attempt_id:
        return None
    connection = _connect(project)
    try:
        row = connection.execute("SELECT * FROM task_attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def _transition_reconcile(project: Any, task: dict[str, Any], state: str, reason: str,
                          manifest_hash: str | None = None) -> None:
    path = _project_path(project)
    connection = _connect(path)
    now = _utc_now()
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT state,active_attempt_id FROM tasks WHERE task_id=?", (task["task_id"],)).fetchone()
        if row is None:
            raise KeyError("task_not_found")
        current = str(row["state"])
        if state != current:
            from .task_store import _TRANSITIONS
            if state not in _TRANSITIONS.get(current, set()):
                # `attention` may be surfaced from a previously marked interrupted task.
                if not (current == "attention" and state == "interrupted"):
                    raise RuntimeError(f"invalid_task_transition:{current}->{state}")
        error = _canonical_json({"reason": reason})
        attempt_id = row["active_attempt_id"]
        if attempt_id:
            connection.execute("UPDATE task_attempts SET state=?,ended_at=?,manifest_sha256=COALESCE(?,manifest_sha256),error_json=? WHERE attempt_id=?",
                               (state, now, manifest_hash, error, attempt_id))
        connection.execute("UPDATE tasks SET state=?,lease_owner=NULL,lease_expires=NULL,updated_at=? WHERE task_id=?",
                           (state, now, task["task_id"]))
        connection.execute("COMMIT")
    except Exception:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()

