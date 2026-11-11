"""Process this study's AUT/PIT chat intervals and team outcomes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
import datetime as dt
from importlib.metadata import version
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
from threading import Lock
import time
import traceback
from uuid import uuid4
import zipfile

import numpy as np
import openpyxl
import pandas as pd
from scipy import signal
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from eda_toolkit.capabilities import source_fingerprint
from eda_toolkit.core_export import atomic_json

from pyeda_one_stop import (
    _emit_progress, collect_raw_members, json_dump, load_design, make_linkage, sha256_file,
)


TASKS = {
    "AUT": ("_aut_rawdata.xlsx", "chat_data", "group_data"),
    "PIT": ("_pit_rawdata_rerate.xlsx", "chat", "group"),
}
FEATURES = ["tonic_median_log_ratio", "phasic_rms_log_ratio"]
OUTCOMES = ["novelty_mean", "usefulness_mean"]
CUTOFF_HZ = 0.05
EDGE_SECONDS = 40
MIN_WINDOW_SECONDS = 60
MIN_COVERAGE = 0.8
LOCAL_TIMEZONE = dt.timezone(dt.timedelta(hours=8))
TABLES = ("recording_qc", "record_task_features", "team_analysis_data", "team_descriptives",
          "behavior_models", "team_inclusion_flow")
FIGURES = ("figure_team_novelty", "figure_model_estimates", "figure_sample_flow")
HANDOFF = "APA_报告交接.md"
LEGACY_ANALYSIS = "99642f803dfcc806ea4aa70457ca4aba674527828592f69d324485dc7a90dead"
INTAKE_FILES = ("analysis_settings.json", "task_windows.csv", "mapping_review.csv")
TABLE_PATHS = {
    name: f"{'features' if name in ('recording_qc', 'record_task_features') else 'statistics'}/{name}.csv"
    for name in TABLES
}
FIGURE_PATHS = [f"figures/{name}.{ext}" for name in FIGURES for ext in ("png", "svg")]


class RecordingFormatError(ValueError):
    """An individual GSR CSV cannot be interpreted."""


class StudyPaused(Exception):
    """The operator requested a stop at the next checkpoint."""


@contextmanager
def study_lock(output_root):
    """An OS lock is released even if the window is terminated."""
    with (output_root / ".aut-pit.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("另一个 PYEDA 窗口正在处理此输出目录，请使用已打开的窗口。") from error
        yield


def append_log(path, message):
    stamp = dt.datetime.now(LOCAL_TIMEZONE).isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{stamp} {message}\n")


def artifacts_match(root, hashes):
    try:
        return bool(hashes) and all(sha256_file(root / name) == digest for name, digest in hashes.items())
    except OSError:
        return False


def latest_checkpoint(output_root, sources):
    paths = sorted(output_root.glob("AUT-PIT-EDA-*/checkpoint.json"),
                   key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in paths:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(state, dict) and state.get("version") in (1, 2) and state.get("sources") == sources
                and isinstance(state.get("stages"), dict)):
            return path.parent, state
    return None, None


def source_snapshot(wrist_dir, workbooks, prepared, progress=None):
    archives = sorted((wrist_dir / "原始数据").glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"请将手环 ZIP 放到 {wrist_dir / '原始数据'} 后重试。")
    paths = [*archives, *workbooks]
    if prepared is not None:
        paths.extend([prepared / "mapping_review.csv", prepared / "verification/source_hashes.json"])
    hashes = {}
    for index, path in enumerate(paths, 1):
        if progress:
            progress(index, len(paths))
        hashes[str(path)] = {"sha256": sha256_file(path)}
    return archives, hashes


def read_record(path):
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(entry["qc"], dict) and isinstance(entry["features"], list) and entry["sha256"]:
            return entry
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def archived_markers(member):
    """Recover only marker metadata omitted by the previous checkpoint format."""
    starts, ends = [], []
    with zipfile.ZipFile(member["archive"]) as package, package.open(member["member"]) as raw:
        rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        for row in rows:
            marker = row.get("eventName")
            if marker == "recStart":
                starts.append(int(row["timeStamp"].strip()))
            elif marker == "recEnd":
                ends.append(int(row["timeStamp"].strip()))
    return starts, ends


def copy_verified(source, target, digest):
    if not source.is_file():
        return False
    temporary = target.with_suffix(".copying")
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != digest:
        temporary.unlink()
        return False
    temporary.replace(target)
    return True


def resource_budget(requested, members):
    if requested == 1:
        return 1, "采用串行"
    available = None
    if os.name == "nt":
        import ctypes

        buffer = ctypes.create_string_buffer(struct.pack("II7Q", 64, 0, *([0] * 7)))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(buffer)):
            available = struct.unpack_from("II7Q", buffer.raw)[3] // (1024 ** 2)
    if available is None:
        return 1, "无法测量可用内存，回退串行"
    archive_sizes = {}
    for archive in {item["archive"] for item in members}:
        with zipfile.ZipFile(archive) as package:
            archive_sizes[archive] = {entry.filename: entry.file_size for entry in package.infolist()}
    largest = max(archive_sizes[item["archive"]][item["member"]] for item in members)
    per_worker_mb = max(128, math.ceil(largest * 20 / 1024 ** 2) + 64)
    workers = max(1, min(requested, os.cpu_count() or 1, max(0, available - 512) // per_worker_mb))
    return workers, f"可用内存 {available} MiB，每个工作线程按 {per_worker_mb} MiB 估算"


def task_tables(dv_dir):
    chat_parts, outcome_parts = [], []
    for task, (filename, chat_sheet, group_sheet) in TASKS.items():
        with (dv_dir / filename).open("rb") as stream:
            book = openpyxl.load_workbook(stream, read_only=True, data_only=True)
            try:
                for name, keys in ((chat_sheet, ["group_id", "time"]),
                                   (group_sheet, ["group_id", "group_type", *OUTCOMES])):
                    rows = book[name].values
                    header = next(rows)
                    positions = [header.index(key) for key in keys]
                    table = pd.DataFrame(([row[i] for i in positions] for row in rows), columns=keys)
                    table = table.dropna(subset=["group_id"])
                    table["group_id"] = table["group_id"].astype(str).str.strip().str.upper()
                    table["task"] = task
                    (chat_parts if name == chat_sheet else outcome_parts).append(table)
            finally:
                book.close()
    chat = pd.concat(chat_parts, ignore_index=True)
    chat["time"] = pd.to_datetime(chat["time"], errors="raise")
    if chat["time"].isna().any():
        raise ValueError("聊天表有缺失时间戳，请先核对。")
    windows = (chat.groupby(["task", "group_id"], as_index=False)
               .agg(first_chat=("time", "min"), last_chat=("time", "max"),
                    messages=("time", "size"), dates=("time", lambda t: t.dt.normalize().nunique())))
    windows["start_ms"] = windows["first_chat"].dt.tz_localize(LOCAL_TIMEZONE).astype("datetime64[ms, UTC]").astype("int64")
    windows["stop_ms"] = windows["last_chat"].dt.tz_localize(LOCAL_TIMEZONE).astype("datetime64[ms, UTC]").astype("int64") + 1000
    outcomes = pd.concat(outcome_parts, ignore_index=True)
    outcomes["group_type"] = outcomes["group_type"].astype("string").str.strip().str.casefold()
    if outcomes.duplicated(["task", "group_id"]).any():
        raise ValueError("团队行为表存在重复任务×团队键。")
    for column in OUTCOMES:
        outcomes[column] = pd.to_numeric(outcomes[column], errors="raise")
    return windows, outcomes


def raw_blocks(member):
    """Use only timestamped blocks with four numeric samples in source order."""
    times, block_values, starts, ends = [], [], [], []
    with zipfile.ZipFile(member["archive"]) as package, package.open(member["member"]) as raw:
        rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        if not {"timeStamp", "data", "eventName"}.issubset(rows.fieldnames or []):
            raise RecordingFormatError("缺少 timeStamp、data 或 eventName 列。")
        for row in rows:
            stamp = (row.get("timeStamp") or "").strip()
            raw_value = (row.get("data") or "").strip()
            marker = row.get("eventName")
            if (stamp or marker in ("recStart", "recEnd")) and not stamp.isdecimal():
                raise RecordingFormatError(f"第 {rows.line_num} 行缺少有效时间戳。")
            if marker == "recStart":
                starts.append(int(stamp))
            if marker == "recEnd":
                ends.append(int(stamp))
                if not raw_value:
                    continue
            if stamp:
                times.append(int(stamp))
                block_values.append([])
            if not block_values:
                raise RecordingFormatError(f"第 {rows.line_num} 行出现在第一个时间戳之前。")
            try:
                value = float(raw_value)
            except ValueError:
                value = np.nan
            block_values[-1].append(value if math.isfinite(value) and value > 0 else np.nan)
    timestamps = np.asarray(times, dtype=np.int64)
    values = np.array([np.median(np.log(block)) if len(block) == 4 and np.isfinite(block).all()
                       else np.nan for block in block_values], dtype=float)
    return timestamps, values, starts, ends


def preprocess(timestamps, values, sos):
    finite = np.flatnonzero(np.isfinite(values))
    centered = np.full(len(values), np.nan)
    tonic = centered.copy()
    if not len(finite):
        return centered, tonic, tonic.copy()
    centered[finite] = np.nanmedian(values) - values[finite]
    splits = np.flatnonzero(np.diff(timestamps[finite]) != 1000) + 1
    for indices in np.split(finite, splits):
        if len(indices) >= 2 * EDGE_SECONDS + 1:
            filtered = signal.sosfiltfilt(sos, centered[indices])
            tonic[indices[EDGE_SECONDS:-EDGE_SECONDS]] = filtered[EDGE_SECONDS:-EDGE_SECONDS]
    return centered, tonic, centered - tonic


def analyze_window(window, timestamps, tonic, phasic, starts, ends, ambiguous):
    start, stop = window.start_ms, window.stop_ms
    expected = math.ceil((stop - start) / 1000)
    selection = (timestamps >= start) & (timestamps < stop) & np.isfinite(tonic)
    samples = int(selection.sum())
    coverage = min(samples / expected, 1.0)
    reason = ""
    if len(starts) != 1 or len(ends) != 1 or not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        reason = "recording_markers_or_clock_invalid"
    elif window.dates != 1 or expected < MIN_WINDOW_SECONDS:
        reason = "short_or_multi_date_chat_span"
    elif not (starts[0] <= start and stop <= ends[0]):
        reason = "chat_span_outside_recording"
    elif ambiguous:
        reason = "recording_covers_multiple_team_chat_spans"
    elif coverage < MIN_COVERAGE:
        reason = "insufficient_filtered_coverage"
    return {
        "expected_blocks": expected, "eligible_blocks": samples, "coverage": coverage,
        "status": "unavailable" if reason else "valid", "reason": reason,
        FEATURES[0]: np.nan if reason else float(np.median(tonic[selection])),
        FEATURES[1]: np.nan if reason else float(np.sqrt(np.mean(np.square(phasic[selection])))),
    }


def fit_models(teams):
    rows = []
    for task in TASKS:
        task_rows = teams[teams["task"].eq(task) & teams["team_eda_complete"]]
        for feature in FEATURES:
            for outcome in OUTCOMES:
                data = task_rows[[feature, outcome, "group_type"]].dropna()
                result = {"task": task, "feature": feature, "outcome": outcome,
                          "n": len(data), "status": "unavailable", "reason": "insufficient_data_or_variation",
                          **dict.fromkeys(["estimate", "se_hc3", "ci_low", "ci_high", "t", "df",
                                           "p", "r_squared", "max_leverage", "max_cooks_distance"], np.nan)}
                groups = pd.get_dummies(data["group_type"].astype(str), drop_first=True, dtype=float)
                if len(data) >= max(12, len(groups.columns) + 5) and data[feature].std() > 0 and data[outcome].std() > 0:
                    standardized = (data[feature] - data[feature].mean()) / data[feature].std()
                    predictors = sm.add_constant(pd.concat([standardized, groups], axis=1))
                    fit = sm.OLS(data[outcome], predictors).fit(cov_type="HC3", use_t=True)
                    ci = fit.conf_int().loc[feature]
                    if np.isfinite([fit.bse[feature], fit.pvalues[feature], *ci]).all():
                        influence = fit.get_influence()
                        result.update(status="estimated", reason="", estimate=fit.params[feature],
                                      se_hc3=fit.bse[feature], ci_low=ci.iloc[0], ci_high=ci.iloc[1],
                                      t=fit.tvalues[feature], df=fit.df_resid, p=fit.pvalues[feature],
                                      r_squared=fit.rsquared, max_leverage=influence.hat_matrix_diag.max(),
                                      max_cooks_distance=influence.cooks_distance[0].max())
                rows.append(result)
    table = pd.DataFrame(rows)
    corrected = multipletests(table["p"].fillna(1), method="holm")[1]
    table["p_holm"] = np.where(table["p"].notna(), corrected, np.nan)
    return table


def write_figures(root, teams, models):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(figure)
    for ax, (task, feature) in zip(figure.subplots(2, 2).flat,
                                   [(task, feature) for task in TASKS for feature in FEATURES]):
        data = teams.loc[teams["task"].eq(task) & teams["team_eda_complete"]].dropna(
            subset=[feature, "novelty_mean", "group_type"])
        for group_type, group in data.groupby("group_type"):
            ax.scatter(group[feature], group["novelty_mean"], s=14, alpha=0.72, label=group_type)
        ax.set(title=f"{task} (n = {len(data)} teams)", xlabel=feature.replace("_", " "), ylabel="Team novelty score")
        if len(data):
            ax.legend(frameon=False)
    figure.savefig(root / "figure_team_novelty.png", dpi=300)
    figure.savefig(root / "figure_team_novelty.svg")

    figure = Figure(figsize=(9, 5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    plotted = models[models["status"].eq("estimated")].reset_index(drop=True)
    ax.errorbar(plotted["estimate"], np.arange(len(plotted)),
                xerr=[plotted["estimate"] - plotted["ci_low"], plotted["ci_high"] - plotted["estimate"]],
                fmt="o", color="#215989", capsize=3)
    ax.set_yticks(np.arange(len(plotted)), [f"{r.task} | {r.feature} | {r.outcome}" for r in plotted.itertuples()])
    ax.axvline(0, color="0.55", linewidth=1)
    ax.set(xlabel="Behavior score change per 1 SD of EDA predictor (95% CI)")
    ax.invert_yaxis()
    figure.savefig(root / "figure_model_estimates.png", dpi=300)
    figure.savefig(root / "figure_model_estimates.svg")

    figure = Figure(figsize=(8, 4), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    categories = ["Chat span available", "Complete team EDA", "EDA and both scores"]
    positions = np.arange(len(categories))
    for offset, task in zip((-0.18, 0.18), TASKS):
        data = teams[teams["task"].eq(task)]
        values = [data["first_chat"].notna().sum(), data["team_eda_complete"].sum(),
                  (data["team_eda_complete"] & data[OUTCOMES].notna().all(axis=1)).sum()]
        bars = ax.bar(positions + offset, values, width=0.34, label=task)
        ax.bar_label(bars, padding=3)
    ax.set(xticks=positions, xticklabels=categories, ylabel="Teams")
    ax.margins(y=0.12)
    ax.legend(frameon=False)
    figure.savefig(root / "figure_sample_flow.png", dpi=300)
    figure.savefig(root / "figure_sample_flow.svg")


def run_study(prepared: Path | None, wrist_dir: Path, dv_dir: Path, progress_callback=None,
              *, design: Path | None = None, output_root: Path | None = None,
              cancel_requested=None, force_new=False, max_workers=1):
    if prepared is None and (design is None or output_root is None):
        raise ValueError("请指定实验设计表和输出根目录。")
    if type(max_workers) is not int or not 1 <= max_workers <= 4:
        raise ValueError("并行上限须为 1 至 4。")
    output_root = (output_root or prepared.resolve().parent).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with study_lock(output_root):
        result = _run_checkpointed(output_root, design, prepared, wrist_dir.resolve(), dv_dir.resolve(),
                                   progress_callback, cancel_requested, force_new, max_workers)
        if result["status"] == "succeeded":
            pointer = output_root / "LATEST_AUT_PIT_RUN.txt"
            target = result["run_directory"] + "\n"
            if not pointer.is_file() or pointer.read_text(encoding="utf-8") != target:
                temporary = pointer.with_suffix(".tmp")
                temporary.write_text(target, encoding="utf-8")
                temporary.replace(pointer)
        return result


def _run_checkpointed(output_root, design, prepared, wrist_dir, dv_dir,
                      progress_callback, cancel_requested, force_new, max_workers):
    design = design.resolve() if design is not None else None
    prepared = prepared.resolve() if prepared is not None else None
    sources = [str(path) if path is not None else None for path in (design, prepared, wrist_dir, dv_dir)]
    workbooks = [(dv_dir / names[0]).resolve() for names in TASKS.values()]
    if design is not None:
        workbooks.insert(0, design)
    execution = {
        "toolkit": source_fingerprint(),
        "analysis": sha256_file(Path(__file__)),
        "intake": sha256_file(Path(__file__).with_name("pyeda_one_stop.py")),
        "python": sys.version,
        "dependencies": {name: version(name) for name in
                         ("numpy", "pandas", "scipy", "statsmodels", "openpyxl", "matplotlib")},
    }
    # Scheduling, display and storage policy are outside the numerical identity.
    numeric_sources = (task_tables, raw_blocks, preprocess, analyze_window, fit_models,
                       prepare_record, extract_record, write_statistics, write_figures)
    numeric = "\n".join(inspect.getsource(function) for function in numeric_sources)
    parameters = (TASKS, FEATURES, OUTCOMES, CUTOFF_HZ, EDGE_SECONDS, MIN_WINDOW_SECONDS, MIN_COVERAGE)
    execution["numeric"] = hashlib.sha256((numeric + repr(parameters)).encode("utf-8")).hexdigest()
    root, state = latest_checkpoint(output_root, sources) if not force_new else (None, None)
    previous = None

    def start_directory():
        new_root = output_root / f"AUT-PIT-EDA-{dt.datetime.now(LOCAL_TIMEZONE):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
        new_root.mkdir()
        return new_root, {"version": 2, "sources": sources, "execution": execution, "stages": {}}

    def compatible(candidate):
        if candidate is None or candidate.get("inputs_changed") or candidate.get("sources") != sources:
            return False
        if candidate["version"] == 1:
            expected = {key: value for key, value in execution.items() if key != "numeric"}
            expected["analysis"] = LEGACY_ANALYSIS
            return candidate.get("execution") == expected
        old_identity = {key: value for key, value in candidate.get("execution", {}).items() if key != "analysis"}
        return old_identity == {key: value for key, value in execution.items() if key != "analysis"}

    if compatible(state):
        previous = (root, state)
    if previous is None or state["version"] != 2:
        root, state = start_directory()
    checkpoint = root / "checkpoint.json"

    def save():
        atomic_json(checkpoint, state)

    def report(*, stage, done, total, message, details=None):
        append_log(root / "run.log", message)
        _emit_progress(progress_callback, stage=stage, done=done, total=total, message=message,
                       details={**(details or {}), "run_directory": str(root)})

    def check_pause():
        if cancel_requested is not None and cancel_requested():
            raise StudyPaused

    try:
        report(stage="analysis_checks", done=0, total=3, message="检查上次进度并核对输入文件…")
        def hash_progress(index, total):
            _emit_progress(progress_callback, stage="analysis_checks", done=0, total=3,
                           message=f"正在核验源文件 {index}/{total}", details={"run_directory": str(root)})

        archives, snapshots = source_snapshot(wrist_dir, workbooks, prepared, hash_progress)
        if state.get("inputs") is not None and state["inputs"] != snapshots:
            append_log(root / "run.log", "检测到源文件变化，保留旧断点，开始独立运行。")
            root, state = start_directory()
            checkpoint = root / "checkpoint.json"
            report(stage="analysis_checks", done=0, total=3, message="输入文件有变化，正在建立新运行…")
        state["inputs"] = snapshots
        if state.get("status") == "complete" and artifacts_match(root, {"manifest.json": state.get("manifest_sha256")}):
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if artifacts_match(root, {item["path"]: item["sha256"] for item in manifest["artifacts"]}):
                report(stage="analysis_export", done=4, total=4, message="相同输入已完成，已核验并打开原结果。")
                return json.loads((root / "summary.json").read_text(encoding="utf-8"))
        manifest_path = root / "manifest.json"
        old_hashes = {}
        if artifacts_match(root, {"manifest.json": state.get("manifest_sha256")}):
            old_hashes = {item["path"]: item["sha256"]
                          for item in json.loads(manifest_path.read_text(encoding="utf-8"))["artifacts"]}
        if previous and previous[0] != root:
            state["reuse_directory"] = previous[0].name
        else:
            previous = None
            name = state.get("reuse_directory", "")
            if name.startswith("AUT-PIT-EDA-") and Path(name).name == name and name != root.name:
                reuse_root = output_root / name
                try:
                    reuse_state = json.loads((reuse_root / "checkpoint.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    reuse_state = None
                if compatible(reuse_state):
                    previous = (reuse_root, reuse_state)
        state["status"] = "running"
        state["stage"] = "checks"
        state.pop("error", None)
        state.pop("active_recording", None)
        save()
        manifest_path.unlink(missing_ok=True)
        check_pause()
        result = _analyze_sources(root, design, prepared, wrist_dir, dv_dir, workbooks, archives,
                                  snapshots, state, save, report, check_pause, previous, old_hashes, max_workers)
        state.update(status="complete", stage="complete", manifest_sha256=sha256_file(root / "manifest.json"))
        save()
        return result
    except StudyPaused:
        state["status"] = "paused"
        save()
        done = state.get("records_saved", 0)
        append_log(root / "run.log", f"已暂停，保存了 {done} 条记录的断点。")
        return {"status": "paused", "run_directory": str(root), "records_saved": done}
    except Exception as error:
        detail = traceback.format_exc()
        try:
            append_log(root / "run.log", f"失败于 {state.get('stage', 'checks')}，"
                       f"已保存 {state.get('records_saved', 0)} 条。{error}")
            append_log(root / "error.log", detail)
            if state.get("status") != "complete":
                status = "blocked" if isinstance(error, FileNotFoundError) and state.get("stage", "checks") == "checks" else "failed"
                state.update(status=status, error=str(error))
                save()
                atomic_json(root / "summary.json", {
                    "status": status, "run_directory": str(root), "errors": [str(error)],
                })
        except OSError as logging_error:
            error.add_note(f"保存失败现场时另遇到：{logging_error}")
        raise RuntimeError(f"{error}\n本次目录：{root}\n运行日志：{root / 'run.log'}\n错误日志：{root / 'error.log'}") from error


def _analyze_sources(result_root, design, prepared, wrist_dir, dv_dir, workbooks, archives,
                     snapshots, state, save, report, check_pause, previous, old_hashes, max_workers):
    report(stage="analysis_checks", done=1, total=3, message="读取研究名册与手环 ZIP，生成配对表…")
    members = collect_raw_members(wrist_dir)
    if not members:
        raise ValueError(f"在 {wrist_dir / '原始数据'} 的 ZIP 中找不到 GSR CSV。")
    recording_ids = {item["recording_id"] for item in members}
    if len(recording_ids) != len(members):
        raise ValueError("手环 ZIP 中存在重复的 GSR 记录名。请核对是否导入了重叠或重复的 ZIP。")
    for folder in ("intake", "preprocessed", "features/records", "statistics", "figures"):
        (result_root / folder).mkdir(parents=True, exist_ok=True)
    mapping = result_root / "intake/mapping_review.csv"
    if prepared is None:
        design_rows = load_design(design)
        if not design_rows:
            raise ValueError(f"实验设计表没有可用名册行。请核对 {design}。")
        _, review_rows, linkage = make_linkage(design_rows, members)
        pd.DataFrame(review_rows).to_csv(mapping, index=False, encoding="utf-8-sig")
        json_dump(result_root / "intake/linkage_summary.json", linkage)
    else:
        prepared = prepared.resolve()
        original = json.loads((prepared / "verification/source_hashes.json").read_text(encoding="utf-8"))
        for path in [*sorted({item["archive"] for item in members}), *workbooks]:
            if snapshots[str(path)]["sha256"] != original.get(str(path), {}).get("sha256"):
                raise ValueError(f"输入与原整理包来源不一致。请重新整理 {path.name}。")
        mapping.write_bytes((prepared / "mapping_review.csv").read_bytes())
    roster = pd.read_csv(mapping, dtype=str, keep_default_na=False)
    used_recordings = roster.loc[roster["recording_id"].ne(""), "recording_id"]
    if used_recordings.duplicated().any() or set(used_recordings) != recording_ids:
        raise ValueError("手环记录与配对复核表不一致。")
    report(stage="analysis_checks", done=2, total=3, message="读取 AUT/PIT 聊天时间和团队评分…")
    windows, outcomes = task_tables(dv_dir)
    roster = roster[roster["group_id"].ne("")].copy()
    roster["group_id"] = roster["group_id"].str.strip().str.upper()
    # 仅用组级联结。组内两人的手环交换不影响完整团队的均值。
    ambiguous_ids = set(roster.loc[roster["subject_id"].duplicated(keep=False), "group_id"])
    eligible_roster = roster[roster["status"].eq("proposed_match") & ~roster["group_id"].isin(ambiguous_ids)]
    by_recording = dict(zip(eligible_roster["recording_id"], eligible_roster["group_id"]))
    task_windows = {gid: list(frame.itertuples(index=False)) for gid, frame in windows.groupby("group_id")}
    signals = result_root / "preprocessed"
    settings = {
        "qualification": "exploratory_chat_interval_proxy",
        "source_preparation": str(prepared) if prepared is not None else "automatic_direct_ingest",
        "vendor": "https://www.psychtech.cn/emotion-monitor-terminal.html",
        "clock_assumption": "chat UTC+08 and device Unix millisecond timestamps, no independent clock anchors",
        "window": "[first chat timestamp, last chat timestamp + 1 s)",
        "sample_representation": "1 Hz blocks with four positive finite samples, median natural log, no within-block time assignment",
        "signal": "median of recording log(raw) minus median log(raw) in block; dimensionless",
        "filter": {"method": "zero-phase Butterworth SOS, contiguous runs", "order": 2,
                   "cutoff_hz": CUTOFF_HZ, "trim_each_edge_s": EDGE_SECONDS, "interpolation": False},
        "min_window_s": MIN_WINDOW_SECONDS, "min_coverage": MIN_COVERAGE,
        "team_unit": "complete roster, proposed_match only, excluded duplicate IDs and ambiguous record-to-team windows",
        "model": "separate OLS by task; each predictor standardized in complete sample; group_type fixed effect; HC3 t inference",
        "missingness": "no interpolation; model-wise complete cases; all roster members required for team features",
        "predictors": FEATURES, "outcomes": OUTCOMES,
        "multiplicity": "Holm across all eight planned tests; failed fits retain family slots",
        "source_fingerprint": state["execution"]["toolkit"], "script_sha256": state["execution"]["analysis"],
        "intake_script_sha256": state["execution"]["intake"],
        "inputs_sha256": {"mapping_review.csv": sha256_file(mapping),
                          **{str(path): snapshots[str(path)]["sha256"] for path in workbooks}},
        "raw_archive_fingerprints": {str(path): snapshots[str(path)] for path in archives},
    }
    json_dump(result_root / "intake/analysis_settings.json", settings)
    windows.to_csv(result_root / "intake/task_windows.csv", index=False, encoding="utf-8-sig")
    report(stage="analysis_checks", done=3, total=3, message="已核对探索性分析设置")
    records_dir = result_root / "features/records"
    saved = {}
    for rid in recording_ids:
        path = records_dir / f"{rid}.json"
        old_hash = old_hashes.get(f"features/records/{path.name}")
        if old_hash and not artifacts_match(records_dir, {path.name: old_hash}):
            path.unlink(missing_ok=True)
            continue
        entry = read_record(path)
        if entry and artifacts_match(signals, {f"{rid}.csv.gz": entry["sha256"]}):
            saved[rid] = entry["sha256"]
    if len(saved) < state.get("records_saved", 0):
        report(stage="analysis_checks", done=3, total=3, message="部分断点文件缺失或已变化，将补跑这些记录。")
        state["stages"].clear()
    if len(saved) != len(members):
        state["stages"].clear()
    state["total_records"] = len(members)
    state["records_saved"] = len(saved)
    state["stage"] = "records"
    save()
    if saved:
        report(stage="analysis_run", done=len(saved), total=len(members),
               message=f"已恢复 {len(saved)}/{len(members)} 条记录的断点，继续处理剩余部分。")
    sos = signal.butter(2, CUTOFF_HZ, fs=1.0, output="sos")
    pending = [member for member in members if member["recording_id"] not in saved]
    actual_workers, budget = resource_budget(max_workers, pending) if pending else (0, "全部记录已缓存")
    resources = {"requested_workers": max_workers, "actual_worker_cap": actual_workers,
                 "backend": "cpu", "gpu_executed": False, "reason": budget, "peak_workers": 0}
    atomic_json(result_root / "resources.json", resources)
    if pending:
        report(stage="analysis_run", done=len(saved), total=len(members),
               message=f"CPU 记录处理 · 请求 {max_workers} 并行 · 实际上限 {actual_workers} · {budget}")
    started_records = time.monotonic()
    processed_now = 0
    failures = {}
    worker_lock = Lock()
    active_workers = 0

    def process(member):
        rid = member["recording_id"]
        target = signals / f"{rid}.csv.gz"
        group = by_recording.get(rid, "")
        key = [member["member"], snapshots[str(member["archive"])]["sha256"]]
        if previous:
            old_root, old_state = previous
            same_inputs = old_state.get("inputs") == snapshots
            if old_state["version"] == 1:
                old = old_state.get("records", {}).get(rid) if same_inputs else None
            else:
                old = read_record(old_root / "features/records" / f"{rid}.json")
                if old and old.get("source_key") != key:
                    old = None
            if old and copy_verified(old_root / "preprocessed" / target.name, target, old["sha256"]):
                if same_inputs:
                    return {**old, "source_key": key, "reused_exact": old_root.name}, "恢复旧结果"
                samples = pd.read_csv(target, usecols=["timestamp_ms", "tonic", "phasic"], float_precision="round_trip")
                markers = old.get("markers")
                if markers is None:
                    markers = ([], []) if old["qc"]["parse_issue"] else archived_markers(member)
                starts, ends = markers
                data = (samples["timestamp_ms"].to_numpy(dtype=np.int64),
                        samples["tonic"].to_numpy(dtype=float), samples["phasic"].to_numpy(dtype=float),
                        starts, ends)
                record_qc = {**old["qc"], "candidate_team": group}
                features = extract_record(rid, group, data, record_qc, task_windows, windows)
                return {**old, "qc": record_qc, "features": features, "markers": markers}, "复用预处理"
        data, record_qc = prepare_record(member, target, group, sos)
        features = extract_record(rid, group, data, record_qc, task_windows, windows)
        return {"sha256": sha256_file(target), "qc": record_qc, "features": features,
                "source_key": key, "markers": list(data[-2:])}, "计算记录"

    def tracked_process(member):
        nonlocal active_workers
        with worker_lock:
            active_workers += 1
            resources["peak_workers"] = max(resources["peak_workers"], active_workers)
        try:
            return process(member)
        finally:
            with worker_lock:
                active_workers -= 1

    def commit(member, result):
        nonlocal processed_now
        rid = member["recording_id"]
        entry, action = result
        atomic_json(records_dir / f"{rid}.json", entry)
        saved[rid] = entry["sha256"]
        state["records_saved"] = len(saved)
        processed_now += 1
        eta = (time.monotonic() - started_records) * (len(members) - len(saved)) / processed_now
        report(stage="analysis_run", done=len(saved), total=len(members),
               message=f"{action} {len(saved)}/{len(members)} · {rid}",
               details={"eta_s": eta if processed_now >= 3 and len(saved) < len(members) else None})

    def failure(member, error):
        rid = member["recording_id"]
        failures[rid] = str(error)
        append_log(result_root / "error.log", f"{rid}\n{traceback.format_exc()}")
        report(stage="analysis_run", done=len(saved), total=len(members), message=f"{rid} 处理失败：{error}")

    with ThreadPoolExecutor(max_workers=max(1, actual_workers)) as pool:
        while pending:
            check_pause()
            batch, pending = pending[:actual_workers], pending[actual_workers:]
            if actual_workers == 1:
                member = batch[0]
                try:
                    result = tracked_process(member)
                except (OSError, MemoryError) as error:
                    failure(member, error)
                else:
                    commit(member, result)
                continue
            futures = {pool.submit(tracked_process, member): member for member in batch}
            for future in as_completed(futures):
                member = futures[future]
                try:
                    result = future.result()
                except MemoryError:
                    pending.insert(0, member)
                    actual_workers = 1
                    resources.update(actual_worker_cap=1, reason="运行时内存不足，改为串行重试")
                    report(stage="analysis_run", done=len(saved), total=len(members), message=resources["reason"])
                except OSError as error:
                    failure(member, error)
                else:
                    commit(member, result)
    atomic_json(result_root / "resources.json", resources)
    if failures:
        raise RuntimeError(f"{len(failures)} 条记录处理失败，其他已完成记录已保存。失败记录见 error.log。")
    check_pause()
    if previous and not state["stages"] and all(
            read_record(records_dir / f"{rid}.json").get("reused_exact") == previous[0].name for rid in saved):
        restore_export_stages(previous, result_root, state, save)

    def verify_sources():
        try:
            current = source_snapshot(wrist_dir, workbooks, prepared)
        except FileNotFoundError:
            state["inputs_changed"] = True
            raise
        if current != (archives, snapshots):
            state["inputs_changed"] = True
            raise ValueError("分析期间输入发生变化，请等文件写入完成再重试。")

    def compute_statistics():
        qc, features = [], []
        for member in members:
            entry = read_record(records_dir / f"{member['recording_id']}.json")
            qc.append(entry["qc"])
            features.extend(entry["features"])
        return write_statistics(result_root, settings, qc, features, windows, outcomes, roster, ambiguous_ids)

    return export_results(result_root, state, save, report, check_pause, compute_statistics, verify_sources, saved)


def prepare_record(member, target, group, sos):
    rid = member["recording_id"]
    parse_issue = ""
    try:
        ts, values, starts, ends = raw_blocks(member)
    except (RecordingFormatError, UnicodeDecodeError, csv.Error) as error:
        parse_issue = str(error)
        ts, values, starts, ends = np.array([], dtype=np.int64), np.array([]), [], []
    if len(starts) == len(ends) == 1:
        values[(ts < starts[0]) | (ts + 1000 > ends[0])] = np.nan
    centered, tonic, phasic = preprocess(ts, values, sos)
    if len(starts) != 1 or len(ends) != 1 or np.any(np.diff(ts) <= 0):
        tonic.fill(np.nan)
        phasic.fill(np.nan)
    temporary = target.with_suffix(".tmp")
    pd.DataFrame({"timestamp_ms": ts, "relative_log_signal": centered,
                  "tonic": tonic, "phasic": phasic, "eligible": np.isfinite(tonic)}).to_csv(
        temporary, index=False, compression="gzip")
    temporary.replace(target)
    qc = {"recording_id": rid, "source_member": member["member"], "candidate_team": group,
          "parse_issue": parse_issue, "rec_start_count": len(starts), "rec_end_count": len(ends),
          "clock_reverse_or_duplicate": int((np.diff(ts) <= 0).sum()),
          "blocks": len(ts), "finite_blocks": int(np.isfinite(values).sum()),
          "filtered_blocks": int(np.isfinite(tonic).sum())}
    return (ts, tonic, phasic, starts, ends), qc


def extract_record(rid, group, data, qc, task_windows, windows):
    ts, tonic, phasic, starts, ends = data
    features = []
    for window in task_windows.get(group, []):
        overlaps = (windows["task"].eq(window.task) & windows["group_id"].ne(group)
                    & windows["start_ms"].ge(starts[0]) & windows["stop_ms"].le(ends[0])
                    & windows["start_ms"].lt(window.stop_ms) & windows["stop_ms"].gt(window.start_ms)
                    ).any() if len(starts) == len(ends) == 1 else False
        feature = analyze_window(window, ts, tonic, phasic, starts, ends, overlaps)
        if qc["parse_issue"]:
            feature["reason"] = "recording_parse_error"
        features.append({"task": window.task, "group_id": group, "recording_id": rid, **feature})
    return features


def write_statistics(result_root, settings, qc, record_features, windows, outcomes, roster, ambiguous_ids):
    features = pd.DataFrame(record_features, columns=[
        "task", "group_id", "recording_id", "expected_blocks", "eligible_blocks",
        "coverage", "status", "reason", *FEATURES,
    ])
    features[FEATURES] = features[FEATURES].astype(float)
    good = features[features["status"].eq("valid")]
    aggregate = good.groupby(["task", "group_id"])[FEATURES].mean().reset_index()
    counts = good.groupby(["task", "group_id"]).size().rename("eligible_recordings")
    teams = windows.merge(outcomes, on=["task", "group_id"], how="outer", validate="1:1")
    teams = teams.merge(aggregate, on=["task", "group_id"], how="left", validate="1:1")
    teams = teams.join(counts, on=["task", "group_id"])
    teams["roster_size"] = teams["group_id"].map(roster.groupby("group_id").size())
    teams["eligible_recordings"] = teams["eligible_recordings"].fillna(0).astype(int)
    teams["team_eda_complete"] = teams["eligible_recordings"].eq(teams["roster_size"])
    teams["team_eda_status"] = np.select(
        [teams["roster_size"].isna(), teams["first_chat"].isna(),
         teams["group_id"].isin(ambiguous_ids), teams["team_eda_complete"]],
        ["no_team_roster", "no_chat_window", "duplicate_subject_id", "valid"],
        default="incomplete_candidate_roster_or_window_qc",
    )
    teams.loc[~teams["team_eda_complete"], FEATURES] = np.nan
    models = fit_models(teams)
    descriptive = teams[teams["team_eda_complete"]].groupby(["task", "group_type"])[FEATURES + OUTCOMES].agg(
        ["count", "mean", "std", "median", "min", "max"])
    descriptive.columns = [f"{variable}_{statistic}" for variable, statistic in descriptive.columns]
    flow = (teams.groupby(["task", "team_eda_status"], dropna=False).size().rename("teams").reset_index())
    for name, table in [("recording_qc", pd.DataFrame(qc)), ("record_task_features", features),
                        ("team_analysis_data", teams), ("team_descriptives", descriptive.reset_index()),
                        ("behavior_models", models), ("team_inclusion_flow", flow)]:
        table.to_csv(result_root / TABLE_PATHS[name], index=False, encoding="utf-8-sig")
    summary = {
        "status": "succeeded", "qualification": settings["qualification"],
        "run_directory": str(result_root),
        "preprocessed_records": sum(not row["parse_issue"] for row in qc),
        "recording_parse_errors": sum(bool(row["parse_issue"]) for row in qc),
        "record_task_features_valid": len(good), "models_estimated": int(models["status"].eq("estimated").sum()),
        "complete_teams": teams.groupby("task")["team_eda_complete"].sum().astype(int).to_dict(),
    }
    if summary["models_estimated"] != len(models):
        summary["status"] = "partial"
        summary["retryable"] = False
        summary["errors"] = [f"估计完成 {summary['models_estimated']}/{len(models)} 个模型。请查看 behavior_models.csv 中的原因。"]
    return summary


def write_handoff(result_root):
    (result_root / HANDOFF).write_text(
        "# AUT/PIT EDA 报告交接\n\n"
        "`summary.json` 列出各阶段结果位置，`manifest.json` 记录文件哈希。"
        "模型与描述统计位于 `statistics/`，图件位于 `figures/`，质控与特征位于 `features/`。\n\n"
        "把本目录的路径交给 Codex。根据分析设置、质控表、窗口、特征、联结数据、模型和图件撰写 APA 第 7 版报告。"
        "报告说明聊天跨度代理、UTC+08 时钟假定、团队配对规则、幅值单位未知和模型限制。"
        "报告每个模型的团队数、效应、95% CI、原始及 Holm 校正 p 值。"
        "配套图表包括团队纳入流程、分组描述统计、模型估计与区间、团队新颖性散点图。"
        "图表按 APA 第 7 版编号并附标题、图表注和变量定义。\n\n"
        "- [中科心研 PTES100 设备资料](https://www.psychtech.cn/emotion-monitor-terminal.html)\n"
        "- [Boucsein 等人（2012）的 EDA 测量指南](https://doi.org/10.1111/j.1469-8986.2012.01384.x)\n"
        "- [MacKinnon 与 White（1985）的异方差稳健协方差估计](https://doi.org/10.1016/0304-4076(85)90158-7)\n"
        "- [Holm（1979）的多重检验校正](https://www.jstor.org/stable/4615733)\n"
        "- [APA 图表指南](https://apastyle.apa.org/style-grammar-guidelines/tables-figures)\n",
        encoding="utf-8",
    )


def restore_export_stages(previous, root, state, save):
    old_root, old_state = previous
    statistics = old_state["stages"].get("statistics", {})
    old_files = statistics.get("files", {})
    figures = old_state["stages"].get("figures", {})

    def restore(targets, hashes):
        restored = {}
        for name in targets:
            old_name = Path(name).name if old_state["version"] == 1 else name
            if not hashes.get(old_name) or not copy_verified(old_root / old_name, root / name, hashes[old_name]):
                return None
            restored[name] = hashes[old_name]
        return restored

    files = restore(TABLE_PATHS.values(), old_files)
    if files:
        summary = {**statistics["summary"], "run_directory": str(root)}
        if summary["status"] == "partial":
            summary["retryable"] = False
        state["stages"]["statistics"] = {"summary": summary, "files": files}
        files = restore(FIGURE_PATHS, figures)
        if files:
            state["stages"]["figures"] = files
        save()


def export_results(root, state, save, report, check_pause, compute_statistics, verify_sources, saved):
    stages = state["stages"]
    statistics = stages.get("statistics", {})
    state["stage"] = "statistics"
    if artifacts_match(root, statistics.get("files")):
        summary = statistics["summary"]
        report(stage="analysis_export", done=0, total=4, message="已恢复统计表断点。")
    else:
        stages.clear()
        save()
        report(stage="analysis_export", done=0, total=4, message="汇总完整团队，计算模型和诊断…")
        summary = compute_statistics()
        stages["statistics"] = {
            "summary": summary,
            "files": {name: sha256_file(root / name) for name in TABLE_PATHS.values()},
        }
        save()
    check_pause()
    state["stage"] = "figures"
    if artifacts_match(root, stages.get("figures")):
        report(stage="analysis_export", done=1, total=4, message="已恢复图表和报告交接断点。")
    else:
        save()
        report(stage="analysis_export", done=1, total=4, message="统计表已导出，正在生成图和报告交接清单…")
        teams = pd.read_csv(root / TABLE_PATHS["team_analysis_data"], float_precision="round_trip")
        models = pd.read_csv(root / TABLE_PATHS["behavior_models"], float_precision="round_trip")
        write_figures(root / "figures", teams, models)
        stages["figures"] = {name: sha256_file(root / name) for name in FIGURE_PATHS}
        save()
    check_pause()
    state["stage"] = "verification"
    save()
    report(stage="analysis_export", done=2, total=4, message="复核原始输入在运行期间未变化…")
    verify_sources()
    check_pause()
    summary = {**summary, "results": {
        "intake": "intake/analysis_settings.json", "preprocessed": "preprocessed/",
        "features": TABLE_PATHS["record_task_features"], "qc": TABLE_PATHS["recording_qc"],
        "models": TABLE_PATHS["behavior_models"], "descriptives": TABLE_PATHS["team_descriptives"],
        "figures": "figures/", "report_handoff": HANDOFF, "manifest": "manifest.json",
    }}
    atomic_json(root / "summary.json", summary)
    write_handoff(root)
    report(stage="analysis_export", done=3, total=4, message="正在计算输出文件的完整性清单…")
    files = [*(f"intake/{name}" for name in INTAKE_FILES), "summary.json", HANDOFF, "resources.json",
             *(f"features/records/{rid}.json" for rid in saved)]
    if state["sources"][1] is None:
        files.append("intake/linkage_summary.json")
    hashes = {name: sha256_file(root / name) for name in files}
    hashes.update(stages["statistics"]["files"])
    hashes.update(stages["figures"])
    hashes.update({f"preprocessed/{rid}.csv.gz": digest for rid, digest in saved.items()})
    atomic_json(root / "manifest.json", {
        "artifacts": [{"path": name, "sha256": digest} for name, digest in sorted(hashes.items())],
    })
    report(stage="analysis_export", done=4, total=4, message="结果、图件及报告交接已保存")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prepared", type=Path)
    source.add_argument("--design", type=Path)
    parser.add_argument("--wrist-dir", required=True, type=Path)
    parser.add_argument("--dv-dir", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--force-new", action="store_true", help="Start a separate run without reusing a checkpoint.")
    parser.add_argument("--max-workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    if args.design is not None and args.output_root is None:
        parser.error("--design requires --output-root")
    result = run_study(args.prepared, args.wrist_dir, args.dv_dir, design=args.design, output_root=args.output_root,
                       force_new=args.force_new, max_workers=args.max_workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "succeeded" else 1)
