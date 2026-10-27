"""JSONL/text运行日志、controller事件队列、检索导出与异常记录。"""
from __future__ import annotations

import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
import uuid

from .errors import fail
from .runtime_events import make_run_event


_STOP = object()
_LEVEL_RANK = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_PATH_RE = re.compile(r"(?:(?:[A-Za-z]:\\|\\\\)[^\s\"'<>|]+|/(?:[^\s\"'<>|]+/)*[^\s\"'<>|]+)")
_HOOK_LOCK = threading.RLock()
_HOOK_SESSIONS = {}
_PREVIOUS_SYS_HOOK = None
_PREVIOUS_THREAD_HOOK = None
_FAULT_SESSION = None


def _project_path(project):
    if isinstance(project, dict):
        for name in ("project_directory", "directory", "path"):
            if project.get(name):
                project = project[name]
                break
    path = Path(project).expanduser().resolve()
    return path


def _sanitize_text(value, context=None):
    text = str(value)
    if context is not None:
        home = str(context.get("home_directory", ""))
        if home:
            text = text.replace(home, "<home>")
    return _PATH_RE.sub("<path>", text)


def _sanitize_value(value, context=None, key=""):
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            if str(name).casefold() in {"path", "file_path", "source_path", "directory", "project_directory"}:
                result[str(name)] = "<path>" if item is not None else None
            else:
                result[str(name)] = _sanitize_value(item, context, str(name))
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, context, key) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value, context)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _sanitize_text(value, context)


def _normalized_event(event, sequence, context):
    row = make_run_event(event)
    row["seq"] = sequence
    if row.get("project_id") is None:
        row["project_id"] = context.get("project_id")
    row["message"] = _sanitize_text(row["message"], context)
    row["details"] = _sanitize_value(row["details"], context)
    return row


def _text_line(event):
    return (f"{event['timestamp_utc']} [{event['level']}] #{event['seq']} "
            f"{event.get('stage') or '-'} {event.get('kind')}: {event.get('message') or ''}")


def _rotated_path(path, index):
    return path.with_name(path.name + f".{index}")


def _rotate_one(path, stream_key, context, incoming_size):
    stream = context.get(stream_key)
    max_bytes = context["log_max_bytes"]
    if stream is None or path.stat().st_size + incoming_size <= max_bytes:
        return
    protected = context.get("protected_paths", set())
    rotation_targets = [path, *(_rotated_path(path, index) for index in range(1, context["log_backup_count"] + 1))]
    if any(candidate.exists() and str(candidate.resolve()) in protected for candidate in rotation_targets):
        # manifest引用中的日志必须保留原路径；宁可让当前session日志超过软上限。
        return
    try:
        stream.flush()
    finally:
        stream.close()
    context[stream_key] = None
    backup_count = context["log_backup_count"]
    for index in range(backup_count, 0, -1):
        destination = _rotated_path(path, index)
        source = path if index == 1 else _rotated_path(path, index - 1)
        if not source.exists():
            continue
        if destination.exists():
            if str(destination.resolve()) in context["protected_paths"]:
                archived = destination.with_name(destination.name + ".protected-" + uuid.uuid4().hex[:8])
                os.replace(destination, archived)
            else:
                destination.unlink()
        os.replace(source, destination)
    context[stream_key] = path.open("a", encoding="utf-8", newline="\n")


def _write_log_event(context, event):
    json_line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    text_line = _text_line(event) + "\n"
    _rotate_one(context["jsonl_path"], "jsonl_stream", context, len(json_line.encode("utf-8")))
    _rotate_one(context["text_path"], "text_stream", context, len(text_line.encode("utf-8")))
    context["jsonl_stream"].write(json_line)
    context["text_stream"].write(text_line)
    terminal = event["kind"] in {"completed", "failed", "cancelled", "cancel_requested"}
    critical = _LEVEL_RANK.get(event["level"], 20) >= _LEVEL_RANK[context["log_level"]]
    if terminal or critical:
        context["jsonl_stream"].flush()
        context["text_stream"].flush()


def _notify_subscribers(context, event):
    for subscriber in tuple(context.get("subscribers", ())):
        try:
            if callable(subscriber):
                subscriber(event)
            elif callable(getattr(subscriber, "put", None)):
                subscriber.put(event)
        except Exception:
            # UI订阅者故障不应中断唯一日志写者或分析计算。
            continue


def _controller_loop(context):
    while True:
        item = context["event_queue"].get()
        if isinstance(item, tuple) and len(item) == 2 and item[0] is _STOP:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "flush":
            try:
                context["jsonl_stream"].flush()
                context["text_stream"].flush()
            finally:
                item[1].set()
            continue
        try:
            with context["sequence_lock"]:
                context["sequence"] += 1
                sequence = context["sequence"]
            event = _normalized_event(item, sequence, context)
            if _LEVEL_RANK.get(event["level"], 20) >= _LEVEL_RANK[context["log_level"]] or event["kind"] in {
                    "completed", "failed", "cancelled", "cancel_requested"}:
                _write_log_event(context, event)
            _notify_subscribers(context, event)
        except Exception as error:
            context["controller_error"] = _sanitize_text(error, context)
            _append_crash(context, "runtime event controller failure: " + traceback.format_exc())
    for name in ("jsonl_stream", "text_stream"):
        stream = context.get(name)
        if stream is not None:
            stream.flush()
            stream.close()
            context[name] = None


def _append_crash(context, message):
    crash_path = context.get("crash_path")
    if not crash_path:
        return
    with context["crash_lock"]:
        try:
            with Path(crash_path).open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(_sanitize_text(message, context).rstrip() + "\n")
        except OSError:
            pass


def _global_sys_hook(exc_type, exc_value, exc_tb):
    rendered = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    with _HOOK_LOCK:
        sessions = tuple(_HOOK_SESSIONS.values())
        previous = _PREVIOUS_SYS_HOOK
    for context in sessions:
        _append_crash(context, rendered)
    if previous is not None:
        previous(exc_type, exc_value, exc_tb)


def _global_thread_hook(args):
    rendered = f"Unhandled exception in thread {getattr(args.thread, 'name', '')}:\n" + "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    with _HOOK_LOCK:
        sessions = tuple(_HOOK_SESSIONS.values())
        previous = _PREVIOUS_THREAD_HOOK
    for context in sessions:
        _append_crash(context, rendered)
    if previous is not None:
        previous(args)


def _install_exception_handlers(context):
    global _PREVIOUS_SYS_HOOK, _PREVIOUS_THREAD_HOOK, _FAULT_SESSION
    with _HOOK_LOCK:
        _HOOK_SESSIONS[context["session_id"]] = context
        if sys.excepthook is not _global_sys_hook:
            _PREVIOUS_SYS_HOOK = sys.excepthook
            sys.excepthook = _global_sys_hook
        if hasattr(threading, "excepthook") and threading.excepthook is not _global_thread_hook:
            _PREVIOUS_THREAD_HOOK = threading.excepthook
            threading.excepthook = _global_thread_hook
        if _FAULT_SESSION is None and not faulthandler.is_enabled():
            try:
                context["fault_stream"] = Path(context["crash_path"]).open("a", encoding="utf-8", newline="\n")
                faulthandler.enable(file=context["fault_stream"], all_threads=True)
                context["owns_faulthandler"] = True
                _FAULT_SESSION = context["session_id"]
            except (OSError, RuntimeError, ValueError):
                context["fault_stream"] = None


def _uninstall_exception_handlers(context):
    global _PREVIOUS_SYS_HOOK, _PREVIOUS_THREAD_HOOK, _FAULT_SESSION
    with _HOOK_LOCK:
        _HOOK_SESSIONS.pop(context["session_id"], None)
        if not _HOOK_SESSIONS:
            if sys.excepthook is _global_sys_hook and _PREVIOUS_SYS_HOOK is not None:
                sys.excepthook = _PREVIOUS_SYS_HOOK
            if hasattr(threading, "excepthook") and threading.excepthook is _global_thread_hook and _PREVIOUS_THREAD_HOOK is not None:
                threading.excepthook = _PREVIOUS_THREAD_HOOK
            _PREVIOUS_SYS_HOOK = None
            _PREVIOUS_THREAD_HOOK = None
        if _FAULT_SESSION == context["session_id"]:
            if faulthandler.is_enabled():
                faulthandler.disable()
            _FAULT_SESSION = None
    fault_stream = context.get("fault_stream")
    if fault_stream is not None:
        fault_stream.flush()
        fault_stream.close()
        context["fault_stream"] = None


def create_runtime_session(project_directory, *, log_level="INFO", log_max_bytes=10485760,
                           log_backup_count=5, subscribers=None, session_id=None, install_hooks=True,
                           project_id=None):
    """创建唯一controller日志线程；调用方把返回的event_sink传给pipeline/worker。"""
    level = str(log_level).upper()
    if level not in _LEVEL_RANK:
        fail("runtime_log_level", "未知日志级别。", category="configuration", field="runtime.log_level")
    if isinstance(log_max_bytes, bool) or not isinstance(log_max_bytes, int) or log_max_bytes < 1024:
        fail("runtime_log_size", "log_max_bytes 必须至少为1024。", category="configuration", field="runtime.log_max_bytes")
    if isinstance(log_backup_count, bool) or not isinstance(log_backup_count, int) or not 1 <= log_backup_count <= 100:
        fail("runtime_log_backups", "log_backup_count 必须在1..100内。", category="configuration", field="runtime.log_backup_count")
    project = _project_path(project_directory)
    if project_id is None:
        project_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "eda-project:" + str(project)))
    try:
        project_id = str(uuid.UUID(str(project_id)))
    except (ValueError, TypeError, AttributeError):
        fail("runtime_project_id", "project_id必须是UUID字符串。", category="configuration", field="project_id")
    logs = project / ".eda" / "logs"
    try:
        logs.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        fail("runtime_log_directory", f"无法创建项目日志目录：{_sanitize_text(error)}。", category="io", field="workspace.project_directory")
    sid = str(session_id or uuid.uuid4())
    try:
        uuid.UUID(sid)
    except (ValueError, TypeError, AttributeError):
        fail("runtime_session_id", "session_id 必须为UUID。", category="configuration")
    jsonl_path = logs / f"session-{sid}.jsonl"
    text_path = logs / f"session-{sid}.txt"
    crash_path = logs / "crash.txt"
    try:
        jsonl_stream = jsonl_path.open("a", encoding="utf-8", newline="\n")
        text_stream = text_path.open("a", encoding="utf-8", newline="\n")
    except OSError as error:
        fail("runtime_log_open", f"无法打开项目日志文件：{_sanitize_text(error)}。", category="io")
    subscriber_list = []
    if subscribers is not None:
        subscriber_list = list(subscribers) if isinstance(subscribers, (list, tuple, set)) else [subscribers]
    context = {
        "session_id": sid, "project_id": project_id, "project_directory": str(project), "logs_directory": str(logs),
        "jsonl_path": jsonl_path, "text_path": text_path, "crash_path": crash_path,
        "jsonl_stream": jsonl_stream, "text_stream": text_stream,
        "log_level": level, "log_max_bytes": log_max_bytes, "log_backup_count": log_backup_count,
        "event_queue": queue.Queue(), "sequence": 0, "sequence_lock": threading.Lock(),
        "crash_lock": threading.Lock(), "subscribers": subscriber_list,
        "protected_paths": set(referenced_log_paths(project)),
        "home_directory": str(Path.home()), "closed": False, "controller_error": None,
        "fault_stream": None, "owns_faulthandler": False,
    }
    context["controller_thread"] = threading.Thread(target=_controller_loop, args=(context,),
                                                    name=f"eda-log-{sid[:8]}", daemon=True)
    context["controller_thread"].start()

    def event_sink(event):
        if context["closed"]:
            raise RuntimeError("runtime event session is closed")
        context["event_queue"].put(event)

    def flush(timeout=5.0):
        if context["closed"]:
            return
        acknowledgement = threading.Event()
        context["event_queue"].put(("flush", acknowledgement))
        acknowledgement.wait(timeout=max(0.0, float(timeout)))

    def close(timeout=5.0):
        if context["closed"]:
            return
        flush(timeout=timeout)
        context["closed"] = True
        context["event_queue"].put((_STOP, None))
        context["controller_thread"].join(timeout=max(0.0, float(timeout)))
        _uninstall_exception_handlers(context)

    context["event_sink"] = event_sink
    context["flush"] = flush
    context["close"] = close
    context["subscribe"] = lambda subscriber: context["subscribers"].append(subscriber)
    if install_hooks:
        _install_exception_handlers(context)
    return context


def _json_log_files(project):
    logs = _project_path(project) / ".eda" / "logs"
    if not logs.is_dir():
        return []
    return sorted((path for path in logs.glob("session-*.jsonl*") if path.is_file()),
                  key=lambda path: (path.stat().st_mtime_ns, path.name))


def _iter_events(project):
    for path in _json_log_files(project):
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(row, dict):
                        yield row
        except OSError:
            continue


def search_logs(project, query="", levels=None, task_id=None, run_id=None, limit=500, cursor=None):
    """跨session分页查询持久事件；cursor为匹配行的十进制offset。"""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
        fail("log_limit", "limit 必须在1..5000内。", category="input", field="limit")
    try:
        offset = 0 if cursor is None else int(cursor)
    except (TypeError, ValueError):
        fail("log_cursor", "cursor 必须为非负整数字符串或null。", category="input", field="cursor")
    if offset < 0:
        fail("log_cursor", "cursor 必须为非负整数字符串或null。", category="input", field="cursor")
    accepted_levels = None
    if levels is not None:
        values = [levels] if isinstance(levels, str) else list(levels)
        accepted_levels = {str(value).upper() for value in values}
        if not accepted_levels <= set(_LEVEL_RANK):
            fail("log_levels", "levels 含未知日志级别。", category="input", field="levels")
    needle = str(query or "").casefold()
    page = []
    matched = 0
    has_more = False
    for event in _iter_events(project):
        if accepted_levels is not None and str(event.get("level", "")).upper() not in accepted_levels:
            continue
        if task_id is not None and event.get("task_id") != task_id:
            continue
        if run_id is not None and event.get("run_id") != run_id:
            continue
        searchable = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str).casefold()
        if needle and needle not in searchable:
            continue
        if matched >= offset:
            if len(page) < limit:
                page.append(event)
            else:
                has_more = True
                break
        matched += 1
    next_cursor = str(offset + len(page)) if has_more else None
    return page, next_cursor


def export_logs(project, format="jsonl", destination=None, *, query="", levels=None, task_id=None, run_id=None):
    """导出持久日志筛选结果到新文件，不覆盖已有文件。"""
    output_format = str(format).lower()
    if output_format not in {"jsonl", "text"}:
        fail("log_export_format", "format 必须为jsonl或text。", category="input", field="format")
    project_path = _project_path(project)
    if destination is None:
        export_dir = project_path / ".eda" / "logs" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = export_dir / f"runtime-log-{stamp}-{uuid.uuid4().hex[:8]}.{output_format}"
    else:
        output = Path(destination).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            for event in _iter_events(project_path):
                if levels is not None:
                    accepted = {str(level).upper() for level in ([levels] if isinstance(levels, str) else levels)}
                    if str(event.get("level", "")).upper() not in accepted:
                        continue
                if task_id is not None and event.get("task_id") != task_id:
                    continue
                if run_id is not None and event.get("run_id") != run_id:
                    continue
                searchable = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str).casefold()
                if query and str(query).casefold() not in searchable:
                    continue
                if output_format == "jsonl":
                    stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                else:
                    stream.write(_text_line(event) + "\n")
    except FileExistsError:
        fail("log_export_exists", "导出目标已存在；请指定新文件名。", category="io", field="destination")
    return str(output)


def export_run_logs(session, run_id, destination):
    """将session中指定run的过滤日志复制到run目录，供manifest纳入哈希。"""
    if isinstance(session, dict):
        flush = session.get("flush")
        if callable(flush):
            flush()
        project = session.get("project_directory")
    else:
        project = session
    if not project:
        return []
    output_dir = Path(destination).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_out = output_dir / "runtime_log.jsonl"
    text_out = output_dir / "runtime_log.txt"
    count = 0
    with jsonl_out.open("w", encoding="utf-8", newline="\n") as jsonl_stream, text_out.open("w", encoding="utf-8", newline="\n") as text_stream:
        for event in _iter_events(project):
            if event.get("run_id") != run_id:
                continue
            jsonl_stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            text_stream.write(_text_line(event) + "\n")
            count += 1
    if count == 0:
        jsonl_out.unlink(missing_ok=True)
        text_out.unlink(missing_ok=True)
        return []
    return [str(jsonl_out), str(text_out)]


def load_eta_history(project, *, limit=20000, exclude_run_id=None):
    """从持久session事件抽取有实际耗时的成功工作单位，供estimate_progress匹配。"""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100000:
        fail("eta_history_limit", "ETA历史limit必须在1..100000内。", category="input", field="limit")
    rows = []
    for event in _iter_events(project):
        if event.get("kind") != "completed" or event.get("run_id") == exclude_run_id:
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        duration = details.get("duration_s")
        method = details.get("method")
        rate = details.get("rate_hz")
        unit_type = event.get("unit_type")
        if (not isinstance(duration, (int, float)) or isinstance(duration, bool) or
                not math.isfinite(duration) or duration < 0 or not method or not unit_type):
            continue
        rows.append({"run_id": event.get("run_id"), "method": method,
                     "rate_hz": rate, "unit_type": unit_type,
                     "duration_s": float(duration), "stage": event.get("stage"),
                     "record_id": event.get("record_id"),
                     "work_id": details.get("work_id")})
    return rows[-limit:]


def referenced_log_paths(project):
    """收集run manifest中引用的runtime日志相对路径，供轮转保护使用。"""
    root = _project_path(project)
    referenced = set()
    for manifest_path in root.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for artifact in manifest.get("artifacts", []):
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                continue
            candidate = (manifest_path.parent / artifact["path"]).resolve()
            if candidate.suffix.lower() in {".jsonl", ".txt"} and "runtime" in candidate.name.casefold():
                referenced.add(str(candidate))
    return referenced


__all__ = ["create_runtime_session", "search_logs", "export_logs", "export_run_logs",
           "load_eta_history", "referenced_log_paths"]
