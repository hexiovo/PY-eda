"""无界面入口；与GUI调用同一个运行控制器。"""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import uuid
from . import __version__
from .capabilities import get_capabilities
from .config import PRESETS, get_preset, resolve_config, save_config
from .core_export import json_safe, sha256_file, verify_manifest
from .errors import error_to_dict


def _parser():
    parser = argparse.ArgumentParser(prog="eda_cli", description="EDA 皮电分析工作台 · 默认参数为研究示例")
    parser.add_argument("--json", action="store_true", help="机器可读JSON输出")
    sub = parser.add_subparsers(dest="command")
    for name in ("version", "capabilities", "resources", "validate", "preflight", "plan", "run", "resume", "retry", "export", "presets", "templates", "demo", "queue-start", "queue-list"):
        command = sub.add_parser(name)
        command.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        if name in ("validate", "preflight", "plan", "run"):
            command.add_argument("--config", type=Path)
            command.add_argument("--manifest", type=Path)
            command.add_argument("--input", action="append", default=[], help="信号CSV，可重复")
            command.add_argument("--unit", choices=["uS", "mS", "S", "ohm", "kohm", "Mohm"])
            command.add_argument("--fs", type=float, help="无时间轴时必须提供采样率Hz")
            command.add_argument("--output", type=Path)
            command.add_argument("--workers", type=int)
            command.add_argument("--preset", choices=PRESETS, default="single_standard")
        if name in ("resume", "retry", "export"):
            command.add_argument("directory", type=Path)
            if name == "export":
                command.add_argument("--output", required=True, type=Path)
        if name == "presets":
            command.add_argument("--name", choices=PRESETS)
            command.add_argument("--output", type=Path)
        if name == "templates":
            command.add_argument("--name")
            command.add_argument("--output", type=Path)
        if name == "demo":
            command.add_argument("--directory", type=Path, required=True)
        if name in ("queue-start", "queue-list"):
            command.add_argument("--project", type=Path, required=True, help="持久项目目录")
        if name == "queue-list":
            command.add_argument("--limit", type=int, default=500)
    return parser


def _config(arguments):
    from .pipeline import resolve_run_config
    config = resolve_run_config(arguments.config) if arguments.config else get_preset(arguments.preset)
    if arguments.input:
        config["input"]["signals"] = [str(Path(path).resolve()) for path in arguments.input]
        config["input"]["manifest"] = None
    if arguments.manifest:
        config["input"]["manifest"] = str(arguments.manifest.resolve())
    if arguments.unit:
        config["mapping"]["eda_unit"] = arguments.unit
    if arguments.fs is not None:
        config["mapping"]["sampling_rate_hz"] = arguments.fs
    if arguments.output:
        config["output"]["directory"] = str(arguments.output.resolve())
    if arguments.workers is not None:
        config["compute"]["workers"] = arguments.workers
    return resolve_config(config)


def _project_directory(config, config_path=None):
    workspace = config.get("workspace", {})
    configured = workspace.get("project_directory") or config.get("input", {}).get("base_directory")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute() and config_path:
            candidate = Path(config_path).resolve().parent / candidate
        return candidate.resolve()
    if config_path:
        return Path(config_path).resolve().parent
    return Path.cwd().resolve()


def _runtime_session(config, config_path=None, *, machine=False):
    from .runtime_logging import create_runtime_session
    runtime = config.get("runtime", {})
    session = create_runtime_session(
        _project_directory(config, config_path),
        log_level=runtime.get("log_level", "INFO"),
        log_max_bytes=runtime.get("log_max_bytes", 10485760),
        log_backup_count=runtime.get("log_backup_count", 5),
    )

    def print_event(event):
        if machine:
            return
        if event.get("kind") in {"run", "stage", "work_started", "completed", "failed", "cancelled", "warning"}:
            detail = event.get("details", {})
            work = detail.get("work_id", "") if isinstance(detail, dict) else ""
            print(f"[{event.get('stage','')}] {event.get('message','')} {work}".rstrip(),
                  file=sys.stderr, flush=True)

    session["subscribe"](print_event)
    return session


def _queue_runtime_session(project, *, machine=False):
    from .runtime_logging import create_runtime_session
    session = create_runtime_session(project["path"], project_id=project["project_id"])

    def print_event(event):
        if machine:
            return
        if event.get("kind") in {"run", "stage", "work_started", "completed", "failed", "cancel_requested", "cancelled", "warning"}:
            detail = event.get("details", {})
            work = detail.get("work_id", "") if isinstance(detail, dict) else ""
            print(f"[{event.get('stage','')}] {event.get('message','')} {work}".rstrip(),
                  file=sys.stderr, flush=True)

    session["subscribe"](print_event)
    return session


def _queue_list(project_path, limit):
    from .project_store import open_project
    from .task_store import list_tasks
    project = open_project(project_path)
    tasks = list_tasks(project, limit=limit)
    columns = ("task_id", "ordinal", "state", "run_id", "active_attempt_id",
               "draft_revision", "marker_revision", "qc_revision", "created_at", "updated_at")
    return {"project_id": project["project_id"], "project_directory": project["path"],
            "task_count": len(tasks), "tasks": [{key: task.get(key) for key in columns} for task in tasks]}


def _runtime_resources():
    """报告本进程实际解析到的资源路径与摘要，冻结验收可拒绝源码回退。"""
    from .runtime_identity import icon_path

    package_root = Path(__file__).resolve().parent
    files = {
        "icon": Path(icon_path()).resolve(),
        "font": (package_root / "resources" / "help" / "fonts" / "NotoSansSC-Regular.ttf").resolve(),
        "help": (package_root / "resources" / "help" / "EDA使用指南.pdf").resolve(),
    }
    frozen = bool(getattr(sys, "frozen", False))
    runtime_root = Path(getattr(sys, "_MEIPASS", package_root.parent)).resolve()
    resources = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"发行资源缺失：{name}: {path}")
        if frozen and not path.is_relative_to(runtime_root):
            raise RuntimeError(f"冻结进程资源解析到了打包目录之外：{name}: {path}")
        resources[name] = {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
    return {"frozen": frozen, "runtime_root": str(runtime_root), "resources": resources}


def _queue_start(project_path, *, machine=False):
    from .project_store import open_project
    from .task_queue import claim_next_task, execute_claimed_task
    from .task_store import request_task_cancel

    project = open_project(project_path)
    session = _queue_runtime_session(project, machine=machine)
    owner_token = f"cli:{os.getpid()}:{uuid.uuid4()}"
    active_task = {"task_id": None}
    stop_requested = {"value": False}
    outcomes = []

    def interrupt_handler(_signum, _frame):
        stop_requested["value"] = True
        task_id = active_task.get("task_id")
        if task_id:
            try:
                request_task_cancel(project, task_id)
            except Exception:
                pass

    old_sigint = signal.signal(signal.SIGINT, interrupt_handler)
    old_sigbreak = None
    if hasattr(signal, "SIGBREAK"):
        old_sigbreak = signal.signal(signal.SIGBREAK, interrupt_handler)
    started = time.monotonic()
    try:
        while not stop_requested["value"]:
            task = claim_next_task(project, owner_token=owner_token)
            if task is None:
                break
            active_task["task_id"] = task["task_id"]
            # A signal may arrive while claim_next_task is inside its SQLite
            # transaction, before active_task is published to the handler.
            # Retire that race by persisting cancellation after the claim.
            if stop_requested["value"]:
                try:
                    request_task_cancel(project, task["task_id"])
                except Exception:
                    pass
            summary = execute_claimed_task(task["task_id"], session["event_sink"],
                                           project=project, runtime_session=session)
            outcomes.append({"task_id": task["task_id"], "attempt_id": task.get("attempt_id"),
                             "status": summary.get("status", "failed"), "run_id": summary.get("run_id"),
                             "directory": summary.get("directory")})
            active_task["task_id"] = None
            if summary.get("status") == "cancelled" and stop_requested["value"]:
                break
    finally:
        active_task["task_id"] = None
        signal.signal(signal.SIGINT, old_sigint)
        if old_sigbreak is not None:
            signal.signal(signal.SIGBREAK, old_sigbreak)
        session["flush"]()
        session["close"]()

    states = {item["status"] for item in outcomes}
    if stop_requested["value"]:
        status = "cancelled"
    elif "failed" in states:
        status = "failed"
    elif "partial" in states:
        status = "partial"
    elif "cancelled" in states:
        status = "cancelled"
    else:
        status = "succeeded"
    return {"status": status, "project_id": project["project_id"],
            "project_directory": project["path"], "started_count": len(outcomes),
            "queue_exhausted": not stop_requested["value"],
            "interrupted": stop_requested["value"], "elapsed_s": time.monotonic() - started,
            "tasks": outcomes}


def _emit(value, machine):
    print(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) if machine or not isinstance(value, str) else value)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    machine = args.json
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "version":
            value = {"version": __version__, "schema_version": "1.1"}
        elif args.command == "capabilities":
            value = get_capabilities()
        elif args.command == "resources":
            value = _runtime_resources()
        elif args.command == "presets":
            value = get_preset(args.name) if args.name else {"presets": list(PRESETS), "qualification": "research_examples"}
            if args.output:
                if not args.name:
                    raise ValueError("写出预设时必须指定 --name。")
                save_config(value, args.output)
        elif args.command == "templates":
            from .core_io import export_template, list_templates, read_template
            value = list_templates() if not args.name else read_template(args.name)
            if args.name and args.output:
                value = {"path": str(export_template(args.name, args.output))}
        elif args.command == "demo":
            from .demo import create_demo
            value = create_demo(args.directory)
        elif args.command == "queue-list":
            if isinstance(args.limit, bool) or not 1 <= args.limit <= 10000:
                raise ValueError("--limit 必须在1..10000内。")
            value = _queue_list(args.project, args.limit)
        elif args.command == "queue-start":
            value = _queue_start(args.project, machine=machine)
        elif args.command == "export":
            if not verify_manifest(args.directory)["valid"]:
                raise ValueError("来源运行的完整性校验失败。")
            if args.output.exists():
                raise ValueError("导出目标已存在，请选择新的目录。")
            shutil.copytree(args.directory, args.output)
            value = {"directory": str(args.output.resolve()), "verified": verify_manifest(args.output)["valid"]}
        elif args.command in ("resume", "retry"):
            from .pipeline import resume_run
            saved_config = resolve_config(args.directory / "resolved_config.json")
            session = _runtime_session(saved_config, args.directory / "resolved_config.json", machine=machine)
            try:
                value = resume_run(args.directory, event_sink=session["event_sink"],
                                   runtime_session=session, project_id=session["project_id"],
                                   attempt_id=str(uuid.uuid4()))
            finally:
                session["flush"]()
                session["close"]()
        else:
            from .pipeline import preview_plan, run_pipeline
            config = _config(args)
            if args.command == "validate":
                from .core_io import load_recordings
                records, tables, report = load_recordings(config["input"].get("manifest"), config["mapping"], config)
                value = {"valid": True, "records": len(records), "tables": list(tables), "import_report": report}
            elif args.command == "preflight":
                session = _runtime_session(config, args.config, machine=machine)
                try:
                    from .runtime_preflight import preflight_run
                    project_state = {"project_id": session["project_id"], "project_revision": 0,
                                     "draft_revision": 0, "marker_revision": 0, "qc_revision": 0}
                    value = preflight_run(config, config["input"].get("manifest"), project_state,
                                          event_sink=session["event_sink"])
                finally:
                    session["flush"]()
                    session["close"]()
            elif args.command == "plan":
                value = preview_plan(config)
            else:
                session = _runtime_session(config, args.config, machine=machine)
                try:
                    value = run_pipeline(config, cancel_callback=None,
                                         event_sink=session["event_sink"], runtime_session=session,
                                         project_id=session["project_id"], attempt_id=str(uuid.uuid4()))
                finally:
                    session["flush"]()
                    session["close"]()
        _emit(value, machine)
        if isinstance(value, dict) and "status" in value:
            if value["status"] == "failed" and value.get("errors") and all(error.get("category") in ("input", "configuration") for error in value["errors"]):
                return 2
            return {"succeeded": 0, "partial": 5, "failed": 4, "cancelled": 130}.get(value["status"], 0)
        return 0
    except KeyboardInterrupt:
        _emit({"status": "cancelled", "message": "已中断；已提交缓存可用于恢复。"}, machine)
        return 130
    except Exception as error:
        details = error_to_dict(error, stage="cli")
        _emit({"status": "failed", "error": details}, True)
        return 3 if details["category"] == "dependency" else 2 if isinstance(error, (ValueError, FileNotFoundError)) else 4


if __name__ == "__main__":
    raise SystemExit(main())
