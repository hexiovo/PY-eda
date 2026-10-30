"""Create NEW synthetic-only paired-analysis materials; never use them as participant data."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit-root", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    root = args.toolkit_root.resolve()
    if not (root / "src" / "eda_toolkit" / "demo.py").is_file():
        raise ValueError("toolkit root must contain src/eda_toolkit/demo.py")
    sys.path.insert(0, str(root / "src"))
    from eda_toolkit.demo import create_demo
    from eda_toolkit.config import resolve_config, save_config
    target = args.directory.resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError("directory must be empty/new; never overwrite")
    # Keep the toolkit's 600 s demo duration: two 300 s conditions retain
    # sufficient support after the preset's filter-edge qualification.
    create_demo(target, duration_s=600, sampling_rate_hz=4, seed=2026)
    config = resolve_config(target / "statistics.json")
    config["statistics"].update({
        "designs": [{"design_id": "synthetic_paired_mean", "outcome": "mean", "design_type": "paired",
                     "independent_unit": "subject_id", "condition_column": "condition", "conditions": ["A", "B"],
                     "aggregate": "mean", "alternative": "two-sided", "correction_family": "synthetic_primary"}],
        "planned_hypotheses": [{"hypothesis_id": "synthetic_paired_mean", "correction_family": "synthetic_primary"}],
    })
    config["output"].update({"directory": str(target / "results"), "formats": ["csv", "json", "html", "png"]})
    save_config(config, target / "analysis.json")
    log = {
        "status": "pending_real_data", "provenance": "SYNTHETIC_ONLY", "unresolved": [],
        "design_sources": ["EDA Toolkit deterministic engineering demo; not a scientific study"],
        "hypothesis_chain": [{"id": "synthetic_paired_mean", "outcome": "mean", "basis": "clean_uS",
                              "window_kind": "condition", "independent_unit": "subject_id", "contrast": "A - B",
                              "purpose": "engineering validation only"}],
        "parameter_ledger": [{"path": "*", "source": "synthetic generator + runtime preset",
                              "validation": "engineering only; populate per-leaf ledger for a real study"}],
        "verification": [],
    }
    (target / "decision-log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"directory": str(target), "config": str(target / "analysis.json"),
                      "provenance": "SYNTHETIC_ONLY", "execution_performed": False}, indent=2))


if __name__ == "__main__":
    main()
