#!/usr/bin/env python3
"""Prepare and statically check the supplied PYEDA study inputs.

The script follows the local eda-study-config workflow.  It never modifies the
source workbook, source spreadsheets, or source ZIP archives.  Each invocation
creates a new run directory and leaves a draft configuration when identity,
device, or outcome facts still need confirmation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


MATCH_ACCEPT_S = 10 * 60
MATCH_GAP_S = 10 * 60
SAMPLE_RATE_HZ = 4.0
OHM_TO_US = 1_000_000.0


class WorkflowCancelled(RuntimeError):
    """Raised when the interactive wrapper declines a blocking question."""


def _emit_progress(callback: Any, *, stage: str, done: int, total: int,
                   message: str, details: dict[str, Any] | None = None) -> None:
    if callback is None:
        return
    callback({
        "stage": stage,
        "done": int(done),
        "total": int(total),
        "message": message,
        "details": details or {},
    })


def _runtime_python() -> str:
    """Use the console interpreter for child CLI calls even when GUI uses pythonw."""
    project_python = Path(r"D:\program\EDA\.venv\Scripts\python.exe")
    if project_python.is_file():
        return str(project_python)
    executable = Path(sys.executable)
    if executable.name.casefold() == "pythonw.exe":
        console = executable.with_name("python.exe")
        if console.is_file():
            return str(console)
    return str(executable)


def _hidden_process_kwargs() -> dict[str, Any]:
    """Prevent child console windows when the workflow is launched from GUI."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_json(command: list[str]) -> tuple[int, Any, str, str]:
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_hidden_process_kwargs(),
    )
    parsed: Any = None
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pass
    return completed.returncode, parsed, completed.stdout, completed.stderr


class AnalysisBlocked(ValueError):
    """The selected bundle lacks current authorization evidence."""


def run_verified_bundle(config_path: Path, eda_root: Path, progress_callback=None):
    """Run a verified pilot or formal bundle using the current local toolkit."""
    from eda_toolkit.core_export import verify_manifest
    from eda_toolkit.pipeline import resolve_run_config, run_pipeline
    from eda_toolkit.runtime_events import estimate_progress
    from eda_toolkit.runtime_preflight import validate_preflight_snapshot

    config_path = config_path.resolve()
    folder = config_path.parent
    if config_path.name != "analysis.json":
        raise AnalysisBlocked("请选择核验包中的 analysis.json。整理阶段生成的草案只用于复核。")
    _emit_progress(progress_callback, stage="analysis_checks", done=0, total=3,
                   message="读取配置和核验证据…")
    decision_path = folder / "decision-log.json"
    evidence = {step: folder / "verification" / f"{step}.json"
                for step in ("validate", "plan", "preflight")}
    if not decision_path.is_file():
        raise AnalysisBlocked("核验包需要同级的 decision-log.json。")
    decision = json.loads(decision_path.read_text(encoding="utf-8-sig"))
    mode = decision.get("status")
    if decision.get("unresolved") != [] or mode not in {"ready_for_pilot", "ready_for_execution"}:
        raise AnalysisBlocked("请先按决策记录补齐影响分析的资料，再生成已核验的试跑或正式分析包。")
    if decision.get("provenance") == "SYNTHETIC_ONLY" and mode == "ready_for_execution":
        raise AnalysisBlocked("合成数据包仅用于工程试跑。")
    if any(not path.is_file() for path in evidence.values()):
        raise AnalysisBlocked("请补齐 verification 目录中的 validate.json、plan.json 和 preflight.json。")
    code, checked, stdout, stderr = run_json([
        _runtime_python(), "-X", "utf8",
        str(eda_root / "skills" / "eda-study-config" / "scripts" / "check_bundle.py"),
        str(config_path), "--decision-log", str(decision_path), "--toolkit-root", str(eda_root),
    ])
    if code or not isinstance(checked, dict) or checked.get("static_valid") is not True:
        raise AnalysisBlocked("配置核对未通过。\n" + (stderr or stdout))
    config = resolve_run_config(config_path)
    if config["output"]["overwrite"]:
        raise AnalysisBlocked("请在核验包中指定新结果目录并关闭覆盖。")
    report = json.loads(evidence["preflight"].read_text(encoding="utf-8-sig"))
    validated = json.loads(evidence["validate"].read_text(encoding="utf-8-sig"))
    planned = json.loads(evidence["plan"].read_text(encoding="utf-8-sig"))
    if (validated.get("valid") is not True or not validated.get("records")
            or validated.get("import_report", {}).get("consumed_inputs") != report.get("input_fingerprints")
            or not planned.get("stages") or planned.get("config_sha256") != report.get("config_hash")):
        raise AnalysisBlocked("导入校验、计划和预检的证据不一致。请对同一分析配置重新核验。")
    _emit_progress(progress_callback, stage="analysis_checks", done=1, total=3,
                   message="复核输入文件、执行器和输出空间…")
    project_state = {field: report[field] for field in (
        "project_id", "project_revision", "draft_revision", "marker_revision", "qc_revision"
    )}
    snapshot = validate_preflight_snapshot(report, config, config["input"]["manifest"], project_state)
    if snapshot.get("valid") is not True:
        reasons = "\n".join(str(issue.get("message", "")) for issue in snapshot["errors"])
        raise AnalysisBlocked("预检证据已失效或仍有阻断项。\n" + reasons)
    _emit_progress(progress_callback, stage="analysis_checks", done=2, total=3,
                   message="核对试跑验收记录…")
    if mode == "ready_for_execution":
        acceptance = decision.get("pilot_acceptance") or {}
        pilot = folder / acceptance.get("directory", "")
        manifest_path = pilot / "manifest.json"
        if (acceptance.get("accepted") is not True or not str(acceptance.get("review", "")).strip()
                or acceptance.get("formal_preflight_sha256") != report["report_sha256"]
                or not manifest_path.is_file()
                or acceptance.get("manifest_sha256") != sha256_file(manifest_path)):
            raise AnalysisBlocked("正式运行需要绑定当前预检和试跑结果的验收记录。")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        pilot_state = json.loads((pilot / "pipeline_state.json").read_text(encoding="utf-8-sig"))
        if (manifest.get("status") != "succeeded" or not verify_manifest(pilot, manifest)["valid"]
                or pilot_state.get("source_fingerprint") != report["source_fingerprint"]):
            raise AnalysisBlocked("试跑结果不完整，或试跑所用执行器已变化。请重新试跑并验收。")

    events = []
    plan = report["work_plan"]

    def report_event(event):
        events.append(event)
        update = estimate_progress(events, plan, [])
        _emit_progress(progress_callback, stage="analysis_run",
                       done=update["global_done"], total=update["global_total"],
                       message=event.get("message") or "分析处理中…",
                       details={"eta_s": update["eta_s"]})

    _emit_progress(progress_callback, stage="analysis_run", done=0, total=plan["total_units"],
                   message="开始正式分析…" if mode == "ready_for_execution" else "开始代表性试跑…")
    result = run_pipeline(config, event_sink=report_event, runtime_context={
        "expected_input_fingerprints": report["input_fingerprints"],
    })
    if result["status"] == "succeeded" and not verify_manifest(result["directory"])["valid"]:
        raise RuntimeError(f"结果文件完整性校验失败。目录：{result['directory']}")
    return {**result, "analysis_mode": mode, "run_directory": result["directory"]}


def parse_mmdd(value: Any) -> tuple[int, int]:
    """Parse the workbook's mixed numeric/string month.day representation."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        text = f"{value:.15g}"
    else:
        text = str(value).strip()
    match = re.search(r"(\d{1,2})\D+(\d{1,2})", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    if re.fullmatch(r"\d{1,2}", text):
        return int(text), 1
    raise ValueError(f"无法解析实验日期 {value!r}")


def parse_time(value: Any) -> tuple[int, int, int]:
    if isinstance(value, dt.time):
        return value.hour, value.minute, value.second
    match = re.search(r"(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?", str(value))
    if not match:
        raise ValueError(f"无法解析开始时间 {value!r}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def load_design(path: Path) -> list[dict[str, Any]]:
    import openpyxl

    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    rows: list[dict[str, Any]] = []
    for sheet in workbook.worksheets:
        values = list(sheet.iter_rows(values_only=True))
        if not values:
            continue
        for source_row, row in enumerate(values[1:], start=2):
            if len(row) < 8 or row[2] in (None, ""):
                continue
            month, day = parse_mmdd(row[0])
            hour, minute, second = parse_time(row[6])
            stamp = dt.datetime(2025, month, day, hour, minute, second)
            rows.append(
                {
                    "sheet": str(sheet.title),
                    "source_row": source_row,
                    "date": stamp.date().isoformat(),
                    "start_time": stamp.strftime("%H:%M:%S"),
                    "datetime": stamp,
                    "group_id": str(row[2]),
                    "subject_id": str(row[3]),
                    "participant_label": str(row[4] or ""),
                    "ring_id": str(row[7] or ""),
                    "note": str(row[8] if len(row) > 8 and row[8] else ""),
                }
            )
    return rows


def first_timestamp(member: zipfile.ZipFile, name: str) -> int | None:
    with member.open(name) as raw:
        raw.readline()
        for _ in range(64):
            line = raw.readline()
            if not line:
                break
            first = line.decode("utf-8-sig", errors="replace").split(",", 1)[0].strip()
            if first.isdigit():
                return int(first)
    return None


def collect_raw_members(wrist_dir: Path) -> list[dict[str, Any]]:
    raw_dir = wrist_dir / "原始数据"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"未找到原始手环目录 {raw_dir}")
    members: list[dict[str, Any]] = []
    for archive in sorted(raw_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as package:
            for name in sorted(package.namelist()):
                if not name.startswith("GSR/") or not name.lower().endswith(".csv"):
                    continue
                timestamp = first_timestamp(package, name)
                if timestamp is None:
                    members.append(
                        {
                            "archive": archive,
                            "member": name,
                            "timestamp_ms": None,
                            "datetime": None,
                            "recording_id": Path(name).stem,
                        }
                    )
                    continue
                local = dt.datetime.fromtimestamp(
                    timestamp / 1000.0,
                    tz=dt.timezone(dt.timedelta(hours=8)),
                ).replace(tzinfo=None)
                members.append(
                    {
                        "archive": archive,
                        "member": name,
                        "timestamp_ms": timestamp,
                        "datetime": local,
                        "recording_id": Path(name).stem,
                    }
                )
    return members


def sequence_match(
    designs: list[dict[str, Any]], members: list[dict[str, Any]]
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Monotone time alignment with explicit gap penalties.

    This is a proposed time-based linkage only.  It is deliberately kept as a
    reviewable alignment table and does not silently create a confirmed join.
    """
    designs = sorted(designs, key=lambda item: item["datetime"])
    members = sorted(
        [item for item in members if item["datetime"] is not None],
        key=lambda item: item["datetime"],
    )
    n, m = len(designs), len(members)
    dp = [[math.inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            current = dp[i][j]
            if not math.isfinite(current):
                continue
            if i < n and current + MATCH_GAP_S < dp[i + 1][j]:
                dp[i + 1][j] = current + MATCH_GAP_S
                back[i + 1][j] = "skip_design"
            if j < m and current + MATCH_GAP_S < dp[i][j + 1]:
                dp[i][j + 1] = current + MATCH_GAP_S
                back[i][j + 1] = "skip_member"
            if i < n and j < m:
                delta = abs((designs[i]["datetime"] - members[j]["datetime"]).total_seconds())
                if current + delta < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1] = current + delta
                    back[i + 1][j + 1] = "match"
    pairs: list[tuple[int, int, float]] = []
    unmatched_design: list[int] = []
    unmatched_member: list[int] = []
    i, j = n, m
    while i or j:
        action = back[i][j]
        if action == "match":
            delta = abs((designs[i - 1]["datetime"] - members[j - 1]["datetime"]).total_seconds())
            pairs.append((i - 1, j - 1, delta))
            i -= 1
            j -= 1
        elif action == "skip_design":
            unmatched_design.append(i - 1)
            i -= 1
        elif action == "skip_member":
            unmatched_member.append(j - 1)
            j -= 1
        else:
            if i:
                unmatched_design.append(i - 1)
                i -= 1
            elif j:
                unmatched_member.append(j - 1)
                j -= 1
    return list(reversed(pairs)), list(reversed(unmatched_design)), list(reversed(unmatched_member))


def make_linkage(
    design: list[dict[str, Any]], members: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_date_design: dict[str, list[dict[str, Any]]] = {}
    by_date_member: dict[str, list[dict[str, Any]]] = {}
    without_timestamp = []
    for item in design:
        by_date_design.setdefault(item["date"], []).append(item)
    for item in members:
        if item["datetime"] is not None:
            by_date_member.setdefault(item["datetime"].date().isoformat(), []).append(item)
        else:
            without_timestamp.append(item)
    subject_contexts: dict[str, set[tuple[str, str]]] = {}
    for item in design:
        subject_contexts.setdefault(item["subject_id"], set()).add((item["sheet"], item["group_id"]))
    duplicate_subject_ids = sorted(
        subject_id for subject_id, contexts in subject_contexts.items() if len(contexts) > 1
    )
    duplicate_subject_set = set(duplicate_subject_ids)
    proposed: dict[str, dict[str, Any]] = {}
    review_rows: list[dict[str, Any]] = []
    summary = {
        "design_rows": len(design),
        "raw_gsr_members": len(members),
        "accepted_matches": 0,
        "review_matches": 0,
        "unmatched_design_rows": 0,
        "unmatched_raw_members": 0,
        "duplicate_design_subject_ids": duplicate_subject_ids,
    }
    unmatched_raw = list(without_timestamp)
    dates = sorted(set(by_date_design) | set(by_date_member))
    for date in dates:
        ditems = sorted(by_date_design.get(date, []), key=lambda item: item["datetime"])
        mitems = sorted(by_date_member.get(date, []), key=lambda item: item["datetime"])
        pairs, missing_d, missing_m = sequence_match(ditems, mitems)
        for dindex, mindex, delta in pairs:
            drow, mrow = ditems[dindex], mitems[mindex]
            status = "proposed_match" if delta <= MATCH_ACCEPT_S else "needs_review"
            if drow["subject_id"] in duplicate_subject_set:
                status = "needs_review"
            if status == "proposed_match":
                summary["accepted_matches"] += 1
                proposed[mrow["recording_id"]] = drow
            else:
                summary["review_matches"] += 1
            review_rows.append(
                {
                    "row_type": "matched",
                    "status": status,
                    "recording_id": mrow["recording_id"],
                    "archive": str(mrow["archive"]),
                    "member": mrow["member"],
                    "raw_start": mrow["datetime"].isoformat(" ") if mrow["datetime"] else "",
                    "design_sheet": drow["sheet"],
                    "design_source_row": drow["source_row"],
                    "group_id": drow["group_id"],
                    "subject_id": drow["subject_id"],
                    "ring_id": drow["ring_id"],
                    "design_start": f"{drow['date']} {drow['start_time']}",
                    "delta_minutes": round(delta / 60.0, 3),
                    "note": drow["note"],
                }
            )
        for dindex in missing_d:
            drow = ditems[dindex]
            summary["unmatched_design_rows"] += 1
            review_rows.append(
                {
                    "row_type": "design_without_raw",
                    "status": "unmatched",
                    "recording_id": "",
                    "archive": "",
                    "member": "",
                    "raw_start": "",
                    "design_sheet": drow["sheet"],
                    "design_source_row": drow["source_row"],
                    "group_id": drow["group_id"],
                    "subject_id": drow["subject_id"],
                    "ring_id": drow["ring_id"],
                    "design_start": f"{drow['date']} {drow['start_time']}",
                    "delta_minutes": "",
                    "note": drow["note"],
                }
            )
        unmatched_raw.extend(mitems[index] for index in missing_m)
    summary["unmatched_raw_members"] = len(unmatched_raw)
    for mrow in unmatched_raw:
        review_rows.append(
            {
                "row_type": "raw_without_design",
                "status": "unmatched",
                "recording_id": mrow["recording_id"],
                "archive": str(mrow["archive"]),
                "member": mrow["member"],
                "raw_start": mrow["datetime"].isoformat(" ") if mrow["datetime"] else "",
                "design_sheet": "",
                "design_source_row": "",
                "group_id": "",
                "subject_id": "",
                "ring_id": "",
                "design_start": "",
                "delta_minutes": "",
                "note": "" if mrow["datetime"] else "no_source_timestamp",
            }
        )
    return proposed, review_rows, summary


def finite_positive(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def convert_member(
    member: dict[str, Any],
    output: Path,
    linkage: dict[str, Any] | None,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    subject_id = linkage["subject_id"] if linkage else f"unknown_{member['recording_id']}"
    group_id = linkage["group_id"] if linkage else ""
    design_sheet = linkage["sheet"] if linkage else ""
    ring_id = linkage["ring_id"] if linkage else ""
    session_id = member["datetime"].date().isoformat() if member["datetime"] else "unknown"
    headers = [
        "recording_id",
        "subject_id",
        "session_id",
        "channel",
        "time_s",
        "eda_uS",
        "source_time_ms",
        "raw_resistance_ohm",
        "event_name",
        "group_id",
        "design_sheet",
        "ring_id",
    ]
    rows = 0
    numeric = 0
    missing = 0
    invalid = 0
    time_errors = 0
    event_markers: list[dict[str, Any]] = []
    previous_time: float | None = None
    first_time_ms: int | None = None
    last_timestamp_ms: int | None = None
    blanks_after_timestamp = 0
    with zipfile.ZipFile(member["archive"]) as package:
        with package.open(member["member"]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.DictReader(text)
            with output.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                for source_row in reader:
                    timestamp_text = str(source_row.get("timeStamp") or "").strip()
                    event_name = str(source_row.get("eventName") or "")
                    raw_value_text = str(source_row.get("data") or "").strip()
                    # recEnd is a marker, not an EDA sample.  Its timestamp is
                    # often a short fraction before the inferred fourth sample
                    # of the last block.  Keep it in a separate audit table so
                    # the signal time grid remains strictly increasing.
                    if event_name == "recEnd" and not raw_value_text:
                        event_markers.append(
                            {
                                "recording_id": member["recording_id"],
                                "event_name": event_name,
                                "source_time_ms": timestamp_text,
                                "source_archive": str(member["archive"]),
                                "source_member": member["member"],
                            }
                        )
                        continue
                    if timestamp_text.isdigit():
                        last_timestamp_ms = int(timestamp_text)
                        if first_time_ms is None:
                            first_time_ms = last_timestamp_ms
                        blanks_after_timestamp = 0
                        inferred_ms = last_timestamp_ms
                        source_time_ms = str(last_timestamp_ms)
                    elif last_timestamp_ms is not None:
                        blanks_after_timestamp += 1
                        inferred_ms = last_timestamp_ms + int(round(1000.0 * blanks_after_timestamp / SAMPLE_RATE_HZ))
                        source_time_ms = ""
                    else:
                        inferred_ms = rows * int(round(1000.0 / SAMPLE_RATE_HZ))
                        source_time_ms = ""
                    if first_time_ms is None:
                        first_time_ms = inferred_ms
                    time_s = (inferred_ms - first_time_ms) / 1000.0
                    if previous_time is not None and time_s <= previous_time:
                        time_errors += 1
                    previous_time = time_s
                    raw_value = finite_positive(raw_value_text)
                    if raw_value is None:
                        eda = ""
                        if not raw_value_text or raw_value_text.lower() in {"na", "nan"}:
                            missing += 1
                        else:
                            invalid += 1
                    else:
                        eda = f"{OHM_TO_US / raw_value:.12g}"
                        numeric += 1
                    writer.writerow(
                        {
                            "recording_id": member["recording_id"],
                            "subject_id": subject_id,
                            "session_id": session_id,
                            "channel": "GSR",
                            "time_s": f"{time_s:.6f}",
                            "eda_uS": eda,
                            "source_time_ms": source_time_ms,
                            "raw_resistance_ohm": raw_value_text,
                            "event_name": event_name,
                            "group_id": group_id,
                            "design_sheet": design_sheet,
                            "ring_id": ring_id,
                        }
                    )
                    rows += 1
    return {
        "recording_id": member["recording_id"],
        "rows": rows,
        "numeric_eda": numeric,
        "missing_eda": missing,
        "invalid_eda": invalid,
        "time_errors": time_errors,
        "output": str(output),
        "source_archive": str(member["archive"]),
        "source_member": member["member"],
        "subject_id": subject_id,
        "group_id": group_id,
        "event_markers": event_markers,
    }


def export_workbook_sheets(source: Path, directory: Path) -> list[dict[str, Any]]:
    import openpyxl

    directory.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.load_workbook(source, data_only=False, read_only=True)
    rows: list[dict[str, Any]] = []
    for sheet in workbook.worksheets:
        destination = directory / f"{source.stem}__{sheet.title}.csv"
        values = list(sheet.iter_rows(values_only=True))
        width = max((len(row) for row in values), default=0)
        headers: list[str] = []
        seen: set[str] = set()
        for index, value in enumerate(values[0] if values else [], start=1):
            text = str(value or f"column_{index}").strip() or f"column_{index}"
            if text in seen:
                text = f"{text}_{index}"
            seen.add(text)
            headers.append(text)
        while len(headers) < width:
            headers.append(f"column_{len(headers) + 1}")
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in values[1:]:
                writer.writerow(["" if value is None else value for value in row] + [""] * (width - len(row)))
        rows.append(
            {
                "source": str(source),
                "sheet": sheet.title,
                "rows": max(0, len(values) - 1),
                "columns": width,
                "csv": str(destination),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def feature_conversion_check(wrist_dir: Path, members: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare a few raw resistance conversions with device feature exports.

    This is provenance evidence for the proposed 1e6/R conversion only.  It is
    not a device validation or a scientific qualification claim.
    """
    feature_dir = wrist_dir / "特征数据"
    feature_members: dict[str, tuple[Path, str]] = {}
    if feature_dir.is_dir():
        for archive in sorted(feature_dir.glob("*.zip")):
            with zipfile.ZipFile(archive) as package:
                for name in package.namelist():
                    if name.startswith("GSR/") and name.lower().endswith(".csv"):
                        prefix = Path(name).name.rsplit("-", 1)[0]
                        feature_members.setdefault(prefix, (archive, name))
    checks: list[dict[str, Any]] = []
    for member in members[:10]:
        prefix = member["recording_id"].rsplit("-", 1)[0]
        target = feature_members.get(prefix)
        if not target:
            continue
        raw_values: list[float] = []
        feature_values: list[float] = []
        with zipfile.ZipFile(member["archive"]) as package:
            with package.open(member["member"]) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace"))
                for row in reader:
                    value = finite_positive(str(row.get("data") or "").strip())
                    if value is not None:
                        raw_values.append(OHM_TO_US / value)
                    if len(raw_values) >= 80:
                        break
        with zipfile.ZipFile(target[0]) as package:
            with package.open(target[1]) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace"))
                for row in reader:
                    value = finite_positive(str(row.get("mean_GSR") or "").strip())
                    if value is not None:
                        feature_values.append(value)
                    if len(feature_values) >= 20:
                        break
        checks.append(
            {
                "recording_id": member["recording_id"],
                "feature_member": target[1],
                "raw_conversion_preview_uS": raw_values[:8],
                "feature_mean_GSR_preview_uS": feature_values[:8],
                "interpretation": "same scale is observed in a preview; exact device conversion remains to be confirmed",
            }
        )
    return {"checked": len(checks), "checks": checks}


def build_config(run_root: Path, preset: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(preset))
    config["pipeline_id"] = "single_standard"
    config["input"].update(
        {
            "manifest": "input_manifest.csv",
            "signals": [],
            "participants": "participants.csv",
            "events": None,
            "conditions": None,
            "group_membership": None,
            "sync_anchors": None,
            "manual_qc": None,
            "statistical_design": None,
            "base_directory": str(run_root),
            "encoding": "utf-8-sig",
            "delimiter": ",",
        }
    )
    config["mapping"].update(
        {
            "columns": {},
            "constants": {},
            "eda_unit": "uS",
            "sampling_rate_hz": SAMPLE_RATE_HZ,
            "time_unit": "s",
            "clock_domain": "recording_local",
        }
    )
    config["windows"].update(
        {
            "whole_record": True,
            "fixed_length_s": None,
            "fixed_step_s": 30.0,
        }
    )
    config["features"].update(
        {
            "ids": [
                "valid_sample_proportion",
                "mean",
                "median",
                "sd",
                "tonic_mean",
                "phasic_auc",
                "scr_count",
                "scr_rate",
                "scr_amplitude_mean",
            ],
            "exclude_interpolated": True,
        }
    )
    config["features"]["extended"]["enabled"] = False
    config["statistics"].update(
        {
            "enabled": False,
            "designs": [],
            "planned_hypotheses": [],
        }
    )
    config["synchrony"]["enabled"] = False
    config["output"].update(
        {
            "directory": "results",
            "overwrite": False,
            "save_processed_signals": True,
        }
    )
    config["workspace"]["project_directory"] = str(run_root)
    return config


def write_operator_card(run_root: Path, summary: dict[str, Any]) -> None:
    text = f"""# PYEDA 一站式操作卡

当前状态是 **blocked**。本 CMD 已完成来源清单、只读提取草案和静态配置准备，但没有把时间近邻配对自动视为确认身份，也没有选择主要因变量、任务时间窗或推断统计。

## 双击运行

双击 `D:\\data\\PYEDA\\一站式PYEDA检查.lnk`。窗口显示当前阶段、进度、已用时和预计剩余时间。每次整理在 `study-config` 下创建新目录。

## 已完成的自动步骤

1. 读取 `实验情况记录表.xlsx` 的三个工作表。
2. 只读读取 `手环数据\\原始数据` 中的 GSR CSV，生成新的标准化信号 CSV。
3. 将原始 `data` 作为电阻欧姆值的候选解释，按 `1,000,000 / R` 生成候选 `eda_uS`。该换算已与少量设备特征文件做尺度预览，但仍需设备导出说明确认。
4. 依据同日时间顺序生成 `mapping_review.csv`。状态为 `proposed_match` 的行也需要人工确认，不能直接当作正式身份联结。
5. 将因变量工作簿导出为派生 CSV，并保留原始工作簿路径和哈希。
6. 生成 `analysis.draft.json`、`decision-log.json`、`数据准备清单.md` 和验证证据。

## 成功判断

应看到新的运行目录，目录中有 `prepared\\signals`、`prepared\\dv`、`mapping_review.csv`、`event_markers.csv` 和 `analysis.draft.json`。若窗口弹出身份配对问题，选择继续只会生成草案，选择取消会停止本次整理。不要把 draft 改名为正式配置。

## 当前阻断项

- 身份配对仍是同日时间近邻的可审阅提议。原始 GSR 199 个，设计表 189 行。本次提取的匹配统计为 `{summary['accepted_matches']} 个建议匹配、{summary['review_matches']} 个需复核匹配、{summary['unmatched_design_rows']} 个设计行无原始记录、{summary['unmatched_raw_members']} 个原始记录无设计行`。
- 研究设计没有给出 EDA 事件/条件时间窗，也没有声明 EDA 的主要结局和估计量。
- 三个任务数据表和两个问卷表的因变量层级、对应任务和与 EDA 会话的联结规则尚未声明。
- 手环原始 `data` 的设备单位和 4 Hz 解释需要设备导出说明或研究记录确认。

## 确认后再做什么

提供 `mapping_review.csv` 的核对结果、设备说明、任务时间窗和主要比较。由配置助手生成独立的试跑包，通过“载入已核验分析包”运行。验收试跑并生成正式分析包后，从同一入口选择正式包，运行并导出结果。

新信号由已验证的设备软件采集，导出后可在窗口选择“导入新采集的文件”。
"""
    (run_root / "主试操作卡.md").write_text(text, encoding="utf-8")


def write_data_checklist(run_root: Path, design: Path, wrist_dir: Path, dv_dir: Path) -> None:
    text = f"""# 数据准备清单

## 原始来源

- `{design}`
- `{wrist_dir / "原始数据" / "*.zip"}`
- `{wrist_dir / "特征数据" / "*.zip"}`，用于转换尺度预览
- `{dv_dir / "*.xlsx"}`

## 生成的标准信号

`prepared\\signals\\*.csv` 使用 UTF-8 BOM。每个文件含 `recording_id`、`subject_id`、`session_id`、`channel`、`time_s`、`eda_uS` 以及原始时间、电阻和事件字段。`eda_uS` 是候选换算值，未得到设备说明前不能当成已确认的物理单位。

## 身份和时间

`mapping_review.csv` 是人工确认表。`proposed_match` 只表示同日时间顺序匹配在十分钟内，`needs_review`、`unmatched` 都不能用于正式推断。不要按文件数量补齐缺失人，不要把未知身份改成已有被试号。

## 因变量

`prepared\\dv\\*.csv` 是工作簿的只读派生副本。原始工作簿仍是唯一来源。当前没有把任何因变量自动写入 EDA 统计设计，因为研究方案没有声明主要因变量、时间窗、汇总层级和联结键。

## 不能自行修改的内容

- 不把空值填成零。
- 不对时间排序、删除重复记录或跨缺口插值。
- 不用 `mean_GSR` 等设备特征替代原始 GSR 后再声称完成 EDA 分解。
- 不在结果出来后挑选窗口、阈值、结局或比较。
"""
    (run_root / "数据准备清单.md").write_text(text, encoding="utf-8")


def main(
    argv: list[str] | None = None,
    *,
    progress_callback: Any = None,
    prompt_callback: Any = None,
    return_result: bool = False,
) -> int | dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda-root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--dv-dir", type=Path, required=True)
    parser.add_argument("--wrist-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    for path in (args.eda_root, args.design, args.dv_dir, args.wrist_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output_root / f"run_{timestamp}").resolve()
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True)

    _emit_progress(progress_callback, stage="inventory", done=0, total=6, message="读取实验设计表…")
    design = load_design(args.design)
    _emit_progress(progress_callback, stage="inventory", done=1, total=6, message=f"已读取 {len(design)} 行实验设计")
    members = collect_raw_members(args.wrist_dir)
    if not members:
        raise ValueError(f"未在 {args.wrist_dir / '原始数据'} 找到含 GSR CSV 的设备导出 ZIP。")
    _emit_progress(progress_callback, stage="inventory", done=2, total=6, message=f"已发现 {len(members)} 个原始 GSR 记录")
    proposed, review_rows, linkage_summary = make_linkage(design, members)
    _emit_progress(progress_callback, stage="inventory", done=3, total=6, message="已生成时间近邻配对复核表", details=linkage_summary)
    if prompt_callback is not None and (
        linkage_summary.get("review_matches", 0)
        or linkage_summary.get("unmatched_design_rows", 0)
        or linkage_summary.get("unmatched_raw_members", 0)
        or linkage_summary.get("duplicate_design_subject_ids")
    ):
        question = {
            "id": "linkage_review",
            "title": "发现身份配对需要复核",
            "message": (
                f"检测到 {linkage_summary.get('review_matches', 0)} 个需复核匹配、"
                f"{linkage_summary.get('unmatched_design_rows', 0)} 个无原始记录设计行、"
                f"{linkage_summary.get('unmatched_raw_members', 0)} 个无设计行原始记录。"
                "是否继续生成只读整理包和分析草案？"
            ),
            "details": linkage_summary,
            "default": True,
        }
        if not prompt_callback(question):
            raise WorkflowCancelled("用户取消了身份配对复核后的整理")
    _emit_progress(progress_callback, stage="inventory", done=4, total=6, message="已通过输入问题确认")

    prepared_signal_dir = run_root / "prepared" / "signals"
    conversion_rows: list[dict[str, Any]] = []
    event_marker_rows: list[dict[str, Any]] = []
    for index, member in enumerate(members, start=1):
        linkage = proposed.get(member["recording_id"])
        output = prepared_signal_dir / f"{member['recording_id']}.csv"
        converted = convert_member(member, output, linkage)
        event_marker_rows.extend(converted.pop("event_markers", []))
        conversion_rows.append(converted)
        _emit_progress(
            progress_callback,
            stage="conversion",
            done=index,
            total=len(members),
            message=f"转换 GSR {index}/{len(members)}：{member['recording_id']}",
        )

    manifest_rows: list[dict[str, Any]] = []
    for member in members:
        linkage = proposed.get(member["recording_id"])
        manifest_rows.append(
            {
                "file_path": str((Path("prepared") / "signals" / f"{member['recording_id']}.csv").as_posix()),
                "recording_id": member["recording_id"],
                "subject_id": linkage["subject_id"] if linkage else f"unknown_{member['recording_id']}",
                "session_id": member["datetime"].date().isoformat() if member["datetime"] else "unknown",
                "channel": "GSR",
                "sampling_rate_hz": SAMPLE_RATE_HZ,
                "eda_unit": "uS",
                "clock_domain": "recording_local",
                "time_origin": "",
                "time_unit": "s",
            }
        )
    write_csv(
        run_root / "input_manifest.csv",
        manifest_rows,
        [
            "file_path", "recording_id", "subject_id", "session_id", "channel",
            "sampling_rate_hz", "eda_unit", "clock_domain", "time_origin", "time_unit",
        ],
    )
    _emit_progress(progress_callback, stage="conversion_finalize", done=1, total=1, message="已写入输入清单和事件标记")

    write_csv(
        run_root / "event_markers.csv",
        event_marker_rows,
        ["recording_id", "event_name", "source_time_ms", "source_archive", "source_member"],
    )

    review_fields = [
        "row_type", "status", "recording_id", "archive", "member", "raw_start",
        "design_sheet", "design_source_row", "group_id", "subject_id", "ring_id",
        "design_start", "delta_minutes", "note",
    ]
    write_csv(run_root / "mapping_review.csv", review_rows, review_fields)

    participant_rows: list[dict[str, Any]] = []
    duplicate_subject_ids = set(linkage_summary.get("duplicate_design_subject_ids", []))
    for item in design:
        table_subject_id = item["subject_id"]
        if item["subject_id"] in duplicate_subject_ids:
            table_subject_id = f"unresolved_{item['subject_id']}_{item['sheet']}_{item['group_id']}"
        participant_rows.append(
            {
                "subject_id": table_subject_id,
                "original_subject_id": item["subject_id"],
                "experimental_group": item["sheet"],
                "group_id": item["group_id"],
                "design_sheet": item["sheet"],
                "ring_id": item["ring_id"],
                "design_date": item["date"],
                "start_time": item["start_time"],
                "source_row": item["source_row"],
                "note": item["note"],
            }
        )
    known_subjects = {row["subject_id"] for row in participant_rows}
    for item in members:
        if item["recording_id"] not in proposed:
            unknown = f"unknown_{item['recording_id']}"
            if unknown not in known_subjects:
                participant_rows.append(
                    {
                        "subject_id": unknown,
                        "original_subject_id": "",
                        "experimental_group": "",
                        "group_id": "",
                        "design_sheet": "",
                        "ring_id": "",
                        "design_date": "",
                        "start_time": "",
                        "source_row": "",
                        "note": "自动生成占位身份，待人工联结",
                    }
                )
                known_subjects.add(unknown)
    write_csv(
        run_root / "participants.csv",
        participant_rows,
        [
            "subject_id", "original_subject_id", "experimental_group", "group_id", "design_sheet", "ring_id",
            "design_date", "start_time", "source_row", "note",
        ],
    )

    dv_inventory: list[dict[str, Any]] = []
    dv_sources = sorted(args.dv_dir.glob("*.xlsx"))
    for index, source in enumerate(dv_sources, start=1):
        dv_inventory.extend(export_workbook_sheets(source, run_root / "prepared" / "dv"))
        _emit_progress(progress_callback, stage="dependent_variables", done=index, total=len(dv_sources), message=f"导出依赖变量 {index}/{len(dv_sources)}：{source.name}")
    json_dump(run_root / "dv_inventory.json", dv_inventory)
    write_csv(
        run_root / "dv_inventory.csv",
        dv_inventory,
        ["source", "sheet", "rows", "columns", "csv"],
    )

    code, preset, stdout, stderr = run_json(
        [_runtime_python(), "-X", "utf8", "-m", "eda_toolkit", "presets", "--name", "single_standard", "--json"]
    )
    if code != 0 or not isinstance(preset, dict):
        raise RuntimeError(f"无法读取当前执行器 preset。{stderr or stdout}")
    config = build_config(run_root, preset)
    json_dump(run_root / "analysis.draft.json", config)
    _emit_progress(progress_callback, stage="configuration", done=1, total=1, message="已生成 analysis.draft.json")

    design_sources = [
        {
            "path": str(args.design),
            "known": [
                "三个工作表分别记录 human_human1、AI_Human1、human_human2 设计行",
                "日期缺少年份，按原始手环时间戳与 2025 年记录对齐",
                "subject_id、group_id、ring_id 和开始时间来自原表",
            ],
        },
        {
            "path": str(args.wrist_dir / "原始数据"),
            "known": ["发现原始 GSR ZIP", "每个 GSR CSV 含 timeStamp、data、eventName"],
            "proposed": ["data 作为欧姆值，按 1e6/R 候选换算为 uS", "有效采样区按 4 Hz 重建空时间戳行"],
        },
        {
            "path": str(args.dv_dir),
            "known": ["发现 AUT、PIT 和 survey 工作簿并导出只读副本"],
            "missing": ["EDA 与这些结局的任务、层级、键和主要结局定义"],
        },
    ]
    parameter_ledger = [
        {
            "path": "mapping.eda_unit",
            "value": "uS",
            "unit": "uS",
            "source": "proposed:raw GSR resistance preview and device feature scale",
            "rationale": "以 1e6/R 生成候选电导，少量 mean_GSR 预览处于同一数量级",
            "validation": "runtime_unverified",
        },
        {
            "path": "mapping.sampling_rate_hz",
            "value": SAMPLE_RATE_HZ,
            "unit": "Hz",
            "source": "proposed:raw GSR row layout",
            "rationale": "有效记录区通常为每秒四行，空 timeStamp 行按 250 ms 展开",
            "validation": "runtime_unverified",
        },
        {
            "path": "windows.whole_record",
            "value": True,
            "unit": "boolean",
            "source": "proposed:missing event/condition timeline",
            "rationale": "在没有任务事件时间窗之前只生成整段描述性特征",
            "validation": "runtime_unverified",
        },
        {
            "path": "statistics.enabled",
            "value": False,
            "unit": "boolean",
            "source": "missing:primary outcome and estimand",
            "rationale": "不在缺少主要结局、独立单位和联结键时编造推断设计",
            "validation": "by_design",
        },
    ]
    unresolved = [
        {
            "id": "identity_linkage",
            "question": "请确认 mapping_review.csv 中每个原始 GSR 文件对应哪个 subject_id、group_id 和 ring_id",
            "impact": "身份错误会把同一人或错误组别送入后续汇总和统计。设计表中的 subject_id=71 在两个不同组/工作表中重复，已被保留为未决冲突。",
        },
        {
            "id": "device_unit_and_rate",
            "question": "请提供手环 GSR 导出说明，确认 data 的单位、1e6/R 换算和有效采样率",
            "impact": "单位和采样率改变分解、SCR 阈值、时间窗覆盖与数值解释",
        },
        {
            "id": "eda_windows_and_outcomes",
            "question": "请指定 EDA 主要结局、任务/事件时间窗、汇总层级和与 AUT/PIT/survey 的联结键",
            "impact": "没有这些事实不能生成正式 inferential statistics 或因变量关联结果",
        },
    ]
    unresolved[0]["next_step"] = "打开 mapping_review.csv，按手环编号和原始记录核对，注明重复 71 的正确编号或重复会话。"
    unresolved[1]["next_step"] = "提供手环型号和设备导出说明，或提供主试留存的单位、采样率及掉线标记记录。"
    unresolved[2]["next_step"] = "提供实验流程、AUT/PIT 的起止时间或事件日志，以及主要比较和因变量列。"
    verification = [
        {
            "step": "source_inventory",
            "status": "completed",
            "evidence": "design, raw GSR, feature archives and dependent-variable workbooks inventoried",
        },
        {
            "step": "conversion",
            "status": "completed_with_proposed_parameters",
            "evidence": "new standardized CSVs and mapping_review.csv written",
        },
        {
            "step": "validate_plan_preflight_pilot",
            "status": "pending_user_confirmation",
            "evidence": "formal analysis is blocked until unresolved facts are supplied",
        },
    ]
    decision_log = {
        "status": "blocked",
        "design_sources": design_sources,
        "hypothesis_chain": [],
        "parameter_ledger": parameter_ledger,
        "unresolved": unresolved,
        "verification": verification,
        "linkage_summary": linkage_summary,
        "conversion_summary": conversion_rows,
    }
    json_dump(run_root / "decision-log.json", decision_log)

    source_paths = [args.design, *sorted(args.dv_dir.glob("*.xlsx")), *sorted((args.wrist_dir / "原始数据").glob("*.zip")), *sorted((args.wrist_dir / "特征数据").glob("*.zip"))]
    source_hashes = {}
    for index, path in enumerate(source_paths, start=1):
        source_hashes[str(path)] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
        _emit_progress(progress_callback, stage="hashing", done=index, total=len(source_paths), message=f"计算来源哈希 {index}/{len(source_paths)}")
    json_dump(run_root / "verification" / "source_hashes.json", source_hashes)
    json_dump(run_root / "verification" / "linkage_summary.json", linkage_summary)
    _emit_progress(progress_callback, stage="evidence", done=0, total=4, message="核对设备特征转换预览…")
    json_dump(run_root / "verification" / "conversion_preview.json", feature_conversion_check(args.wrist_dir, members))
    _emit_progress(progress_callback, stage="evidence", done=1, total=4, message="已保存设备特征转换预览")
    json_dump(run_root / "verification" / "extraction_summary.json", {"files": conversion_rows})
    json_dump(run_root / "verification" / "executor_preset.json", preset)
    _emit_progress(progress_callback, stage="evidence", done=2, total=4, message="读取执行器版本和 preset…")
    version_code, version, version_stdout, version_stderr = run_json(
        [_runtime_python(), "-X", "utf8", "-m", "eda_toolkit", "version", "--json"]
    )
    cap_code, capabilities, cap_stdout, cap_stderr = run_json(
        [_runtime_python(), "-X", "utf8", "-m", "eda_toolkit", "capabilities", "--json"]
    )
    json_dump(run_root / "verification" / "executor_version.json", version if isinstance(version, dict) else {"stdout": version_stdout, "stderr": version_stderr, "code": version_code})
    json_dump(run_root / "verification" / "executor_capabilities.json", capabilities if isinstance(capabilities, dict) else {"stdout": cap_stdout, "stderr": cap_stderr, "code": cap_code})
    _emit_progress(progress_callback, stage="evidence", done=3, total=4, message="已保存执行器能力快照")

    check_script = args.eda_root / "skills" / "eda-study-config" / "scripts" / "check_bundle.py"
    check_output_path = run_root / "verification" / "static_check.json"
    check = subprocess.run(
        [
            _runtime_python(),
            str(check_script),
            str(run_root / "analysis.draft.json"),
            "--decision-log",
            str(run_root / "decision-log.json"),
            "--toolkit-root",
            str(args.eda_root),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_hidden_process_kwargs(),
    )
    check_output_path.write_text(check.stdout, encoding="utf-8")
    _emit_progress(progress_callback, stage="evidence", done=4, total=4, message="已完成静态配置检查", details={"exit_code": check.returncode})
    runtime_checks: list[dict[str, Any]] = []
    for index, command_name in enumerate(("validate", "plan", "preflight"), start=1):
        _emit_progress(progress_callback, stage="verification", done=index - 1, total=3,
                       message=f"正在运行 {command_name}…")
        command = subprocess.run(
            [
                _runtime_python(),
                "-X",
                "utf8",
                "-m",
                "eda_toolkit",
                command_name,
                "--config",
                str(run_root / "analysis.draft.json"),
                "--json",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **_hidden_process_kwargs(),
        )
        (run_root / "verification" / f"{command_name}.json").write_text(
            command.stdout,
            encoding="utf-8",
        )
        try:
            payload = json.loads(command.stdout)
        except json.JSONDecodeError:
            payload = None
        expected_key = {"validate": "valid", "plan": "stages", "preflight": "eligible_to_enqueue"}[command_name]
        passed = (command.returncode == 0 and isinstance(payload, dict)
                  and bool(payload.get(expected_key)))
        runtime_checks.append(
            {
                "step": command_name,
                "status": "completed" if passed else "failed",
                "exit_code": command.returncode,
                "evidence": f"verification/{command_name}.json",
            }
        )
        json_dump(run_root / "verification" / "check-status.json", runtime_checks)
        _emit_progress(progress_callback, stage="verification", done=index, total=3, message=f"完成 {command_name} 核验", details={"exit_code": command.returncode})
        if not passed and prompt_callback is not None:
            if not prompt_callback({
                "id": f"verification_{command_name}",
                "title": f"{command_name} 核验未通过",
                "message": f"{command_name} 未通过。是否保留当前证据并继续整理？",
                "details": {"stdout": command.stdout[-12000:], "stderr": command.stderr[-12000:]},
                "default": False,
            }):
                raise WorkflowCancelled(f"用户取消了 {command_name} 核验后的整理")
    decision_log["verification"].extend(runtime_checks)
    json_dump(run_root / "decision-log.json", decision_log)
    write_operator_card(run_root, linkage_summary)
    write_data_checklist(run_root, args.design, args.wrist_dir, args.dv_dir)
    (args.output_root / "LATEST_RUN.txt").write_text(str(run_root) + "\n", encoding="utf-8")

    result = {
        "status": "blocked",
        "preparation_status": "completed" if all(item["status"] == "completed" for item in runtime_checks) else "check_failed",
        "unresolved": unresolved,
        "run_directory": str(run_root),
        "analysis_draft": str(run_root / "analysis.draft.json"),
        "mapping_review": str(run_root / "mapping_review.csv"),
        "static_check_exit_code": check.returncode,
        "linkage_summary": linkage_summary,
        "message": "已生成只读整理包和草案。补齐 decision-log unresolved 后再编译正式配置。",
    }
    _emit_progress(progress_callback, stage="complete", done=1, total=1, message="整理完成，等待人工确认", details=result)
    if return_result:
        return result
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
