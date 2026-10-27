"""Verify the v0.3 portable processes against the same source pipeline."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from eda_toolkit.config import resolve_config
from eda_toolkit.core_export import safe_name, sha256_file, verify_manifest
from eda_toolkit.demo import create_demo
from eda_toolkit.metrics_driver import DRIVER_IDS
from eda_toolkit.pipeline import run_pipeline

ROOT = PROJECT / "artifacts/verification/driver-v0.3.0"
BUNDLE = PROJECT / "dist/v0.3/EDA_Toolkit-v0.3.0"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1")
    env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    cases = []

    def run(exe, args, name, json_output=True):
        result = subprocess.run([str(BUNDLE / exe), *map(str, args)], capture_output=True,
                                cwd=BUNDLE, env=env, timeout=240,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        (ROOT / f"{name}.stdout.txt").write_bytes(result.stdout)
        (ROOT / f"{name}.stderr.txt").write_bytes(result.stderr)
        cases.append({"name": name, "returncode": result.returncode})
        assert result.returncode == 0, (name, result.stderr[-2000:])
        return json.loads(result.stdout.decode("utf-8-sig")) if json_output else None

    for preset, count in (("single_standard",124), ("group_sync",135), ("single_driver",133)):
        plan = run("eda_cli.exe", ["plan", "--preset", preset, "--json"], f"plan-{preset}")
        assert len(plan["feature_definitions"]) == count
        if preset == "single_driver":
            assert DRIVER_IDS <= plan["feature_definitions"].keys()
    capabilities = run("eda_cli.exe", ["capabilities", "--json"], "capabilities")
    info = json.loads((BUNDLE / "_internal/eda_toolkit/resources/build_info.json").read_text(encoding="utf-8"))
    assert info["version"] == "0.3.0"
    assert capabilities["source_fingerprint"] == info["source_fingerprint"]
    demo = ROOT / "frozen-driver-demo"
    create_demo(demo, duration_s=180, sampling_rate_hz=4, seed=271026)
    config = resolve_config(demo / "single.json")
    config["decomposition"]["method"] = "bateman_sparse"
    config["features"]["ids"] = sorted(DRIVER_IDS)
    config["output"].update(directory=str(ROOT / "source-results"), formats=["csv", "json", "xlsx"])
    source = run_pipeline(config)
    assert source["status"] == "succeeded", source
    config["output"]["directory"] = str(ROOT / "frozen-results")
    config_file = ROOT / "frozen-driver-config.json"
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    frozen = run("eda_cli.exe", ["run", "--config", config_file, "--json"], "driver-run")
    assert frozen["status"] == "succeeded", frozen
    source_root, frozen_root = Path(source["directory"]), Path(frozen["directory"])
    for root in (source_root, frozen_root):
        assert verify_manifest(root)["valid"]
    compared = {}
    for name in ("features", "signals", "driver_bursts", "driver_diagnostics"):
        left = pd.read_csv(source_root / "tables" / f"{safe_name(name)}.csv")
        right = pd.read_csv(frozen_root / "tables" / f"{safe_name(name)}.csv")
        pd.testing.assert_frame_equal(left, right, check_exact=True)
        compared[name] = len(left)
        if name == "features":
            assert set(left.feature_id) == DRIVER_IDS
            assert left.loc[left.feature_id.eq("driver_integral_uS"),"status"].eq("valid").any()
        if name == "driver_diagnostics":
            assert left.status.eq("succeeded").any()
    run("eda_gui.exe", ["--smoke", "--screenshot", ROOT / "frozen-gui.png"], "gui", False)
    assert (ROOT / "frozen-gui.png").stat().st_size > 1000
    pdf = "eda_toolkit/resources/help/EDA使用指南.pdf"
    assert sha256_file(PROJECT / "src" / pdf) == sha256_file(BUNDLE / "_internal" / pdf)
    report = {"status": "passed", "version": "0.3.0", "cases": cases,
              "source_fingerprint": info["source_fingerprint"], "bundle": str(BUNDLE),
              "source_frozen_exact_comparison_rows": compared,
              "executables": {name: sha256_file(BUNDLE / name) for name in ("eda_gui.exe", "eda_cli.exe")},
              "pdf_sha256": sha256_file(BUNDLE / "_internal" / pdf),
              "scope": "local Windows; restricted PATH; synthetic data; offscreen GUI; clean Windows not tested"}
    (ROOT / "frozen-verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
