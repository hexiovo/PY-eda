"""Generic one-stop EDA template with logs, checkpoints and APA handoff metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from uuid import uuid4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output"]["directory"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / f"EDA-RUN-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    run_root.mkdir()
    checkpoint = run_root / "checkpoint.json"
    run_log = run_root / "run.log"
    state = {"status": "running", "stage": "input_checks", "records_saved": 0, "config": str(config_path)}
    atomic_json(checkpoint, state)
    log(run_log, "input_checks started")

    input_files = [Path(config["input"][name]).expanduser().resolve()
                   for name in ("design_file", "manifest")]
    signal_directory = Path(config["input"]["signal_directory"]).expanduser().resolve()
    outcome_directory = Path(config["input"]["outcome_directory"]).expanduser().resolve()
    if not all(path.is_file() for path in input_files):
        raise FileNotFoundError("design_file or manifest is missing")
    if not signal_directory.is_dir() or not outcome_directory.is_dir():
        raise FileNotFoundError("signal_directory or outcome_directory is missing")

    fingerprints = {str(path): sha256(path) for path in input_files}
    atomic_json(run_root / "input_fingerprints.json", fingerprints)
    log(run_log, "input_checks completed")

    # Insert the validated project-specific adapters here. Keep each stage in its own directory.
    for stage in ("preprocessed", "features", "statistics", "figures"):
        (run_root / stage).mkdir()
    (run_root / "analysis_settings.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_root / "APA7_报告交接.md").write_text(
        "# EDA APA 7 报告交接\n\n"
        "请依据本目录的分析设置、特征台账、质控、统计表、图件和 manifest 撰写报告。\n",
        encoding="utf-8")
    state.update(status="succeeded", stage="complete")
    atomic_json(checkpoint, state)
    artifacts = [
        {"path": path.relative_to(run_root).as_posix(), "sha256": sha256(path)}
        for path in sorted(run_root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "checkpoint.json", "run.log"}
    ]
    atomic_json(run_root / "manifest.json", {"artifacts": artifacts})
    log(run_log, "complete")
    return {"status": "succeeded", "run_directory": str(run_root), "artifacts": len(artifacts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
