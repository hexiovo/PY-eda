"""Workload-based ETA estimates for the one-stop EDA GUI."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import zipfile


DEFAULT_PROFILE = {
    "version": 1,
    "source": "observed_completed_run",
    "safety_factor": 1.20,
    "base": {
        "records": 199,
        "mean_signal_bytes": 223688.31,
        "mean_signal_seconds": 3559.58,
        "elapsed_s": 34.0,
        "qualification": "resumed_source_reuse",
    },
    "response": {
        "records": 141,
        "source_records": 199,
        "mean_signal_bytes": 223688.31,
        "mean_signal_seconds": 3559.58,
        "feature_elapsed_s": 510.0,
        "statistics_elapsed_s": 29.0,
        "windows": 13764,
    },
}


def profile_path(output_root: Path) -> Path:
    return output_root / "eta_baseline.json"


def load_profile(output_root: Path) -> dict:
    path = profile_path(output_root)
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("version") == 1:
                return value
        except (OSError, ValueError):
            pass
    return DEFAULT_PROFILE.copy()


def save_profile(output_root: Path, profile: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = profile_path(output_root).with_suffix(".tmp")
    temporary.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(profile_path(output_root))


def _workload_key(wrist_dir: Path, dv_dir: Path) -> str:
    paths = sorted((wrist_dir / "原始数据").glob("*.zip"))
    paths.extend((dv_dir / name) for name in ("_aut_rawdata.xlsx", "_pit_rawdata_rerate.xlsx")
                 if (dv_dir / name).is_file())
    parts = []
    for path in paths:
        stat = path.stat()
        parts.append(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def input_workload(wrist_dir: Path, dv_dir: Path, profile: dict | None = None) -> dict:
    key = _workload_key(wrist_dir, dv_dir)
    cached = (profile or {}).get("workload_cache", {}).get(key)
    if isinstance(cached, dict) and cached.get("records", 0) > 0:
        return {**cached, "cache": "reused", "cache_key": key}
    sizes = []
    samples = []
    for archive in sorted((wrist_dir / "原始数据").glob("*.zip")):
        with zipfile.ZipFile(archive) as package:
            for entry in package.infolist():
                if not entry.filename.startswith("GSR/") or not entry.filename.lower().endswith(".csv"):
                    continue
                sizes.append(entry.file_size)
                # 中科心研导出每秒约四个 GSR 样本。行数只在输入指纹变化时读取，
                # 避免每次启动都重复扫描大型 ZIP。
                with package.open(entry) as raw:
                    samples.append(max(0, sum(1 for _ in raw) - 1))
    outcome_bytes = sum((dv_dir / name).stat().st_size for name in
                        ("_aut_rawdata.xlsx", "_pit_rawdata_rerate.xlsx")
                        if (dv_dir / name).is_file())
    mean_samples = sum(samples) / len(samples) if samples else 0.0
    workload = {
        "records": len(sizes),
        "mean_signal_bytes": sum(sizes) / len(sizes) if sizes else 0.0,
        "mean_signal_samples": mean_samples,
        "mean_signal_seconds": mean_samples / 4.0 if mean_samples else 0.0,
        "total_signal_seconds": sum(samples) / 4.0 if samples else 0.0,
        "outcome_bytes": outcome_bytes,
        "cache": "scanned",
        "cache_key": key,
    }
    if profile is not None and workload["records"]:
        profile.setdefault("workload_cache", {})[key] = {
            name: value for name, value in workload.items() if name not in {"cache", "cache_key"}
        }
    return workload


def estimate_total_seconds(output_root: Path, wrist_dir: Path, dv_dir: Path) -> dict:
    profile = load_profile(output_root)
    workload = input_workload(wrist_dir, dv_dir, profile)
    if workload["records"] == 0:
        return {"status": "unavailable", "reason": "no_signal_records", "seconds": None}
    if workload.get("cache") == "scanned":
        save_profile(output_root, profile)
    base = profile["base"]
    response = profile["response"]
    record_scale = workload["records"] / max(1, response.get("source_records", response["records"]))
    duration_scale = workload["mean_signal_seconds"] / max(1.0, response.get("mean_signal_seconds", 1.0))
    volume_scale = workload["mean_signal_bytes"] / max(1.0, response["mean_signal_bytes"])
    signal_scale = max(record_scale * duration_scale, 0.5)
    outcome_scale = workload["outcome_bytes"] / max(1.0, 240782 + 435619)
    base_seconds = base["elapsed_s"] * workload["records"] / max(1, base["records"]) * duration_scale
    feature_seconds = response["feature_elapsed_s"] * signal_scale
    statistics_seconds = response["statistics_elapsed_s"] * max(0.75, signal_scale, outcome_scale)
    seconds = (base_seconds + feature_seconds + statistics_seconds) * profile["safety_factor"]
    return {
        "status": "estimated",
        "seconds": seconds,
        "workload": workload,
        "record_scale": record_scale,
        "duration_scale": duration_scale,
        "volume_scale": volume_scale,
        "signal_scale": signal_scale,
        "outcome_scale": outcome_scale,
        "baseline": {
            "response_records": response["records"],
            "response_elapsed_s": response["feature_elapsed_s"] + response["statistics_elapsed_s"],
            "safety_factor": profile["safety_factor"],
            "qualification": response.get("qualification", "observed_completed_run"),
        },
        "message": "根据上一轮已完成运行、被试数、信号时长和行为工作簿体量估算",
    }


def update_response_profile(output_root: Path, *, elapsed_s: float, records: int, windows: int) -> None:
    if elapsed_s < 60 or records < 1:
        return
    profile = load_profile(output_root)
    response = profile["response"]
    old_weight = max(1, response["records"])
    weight = old_weight + records
    total_observed = response["feature_elapsed_s"] + response["statistics_elapsed_s"]
    per_record = elapsed_s / records
    previous_per_record = total_observed / old_weight
    blended = (previous_per_record * old_weight + per_record * records) / weight
    response["records"] = weight
    response["mean_signal_seconds"] = response.get("mean_signal_seconds", 3559.58)
    response["feature_elapsed_s"] = blended * weight * 0.94
    response["statistics_elapsed_s"] = blended * weight * 0.06
    response["windows"] = max(response.get("windows", 0), windows)
    response["qualification"] = "observed_completed_run"
    profile["last_observed"] = {
        "elapsed_s": elapsed_s,
        "records": records,
        "windows": windows,
    }
    profile["last_observed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save_profile(output_root, profile)
