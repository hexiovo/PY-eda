"""Environment inspection and isolated, missing-only dependency installation."""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

from .dependency_paths import (
    activate_application_dependency_paths,
    app_data_root as _app_data_root,
    base_dependency_root,
    gpu_dependency_root,
)
from .gpu_support import (
    _emit,
    _run_streaming,
    find_compatible_python,
    inspect_gpu_environment,
    install_gpu_dependencies,
)


BASE_REQUIREMENTS = (
    ("numpy", "numpy>=2.0", "numpy"),
    ("scipy", "scipy>=1.14", "scipy"),
    ("pandas", "pandas>=2.2", "pandas"),
    ("PySide6", "PySide6>=6.8", "PySide6"),
    ("matplotlib", "matplotlib>=3.9", "matplotlib"),
    ("openpyxl", "openpyxl>=3.1", "openpyxl"),
    ("reportlab", "reportlab>=4.2", "reportlab"),
    ("statsmodels", "statsmodels>=0.14.4", "statsmodels"),
)


def _is_importable(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _version_of(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _satisfies_minimum(version: str | None, minimum: str) -> bool:
    if version is None or version == "bundled":
        return version == "bundled"
    try:
        from packaging.version import Version
        return Version(version) >= Version(minimum)
    except Exception:
        def numeric_parts(value: str) -> tuple[int, ...]:
            result = []
            for part in value.split("."):
                digits = "".join(char for char in part if char.isdigit())
                if not digits:
                    break
                result.append(int(digits))
            return tuple(result)
        return numeric_parts(version) >= numeric_parts(minimum)


def inspect_environment(*, refresh: bool = False) -> dict:
    """Return JSON-safe package and GPU availability for the current process."""
    activate_application_dependency_paths()
    packages: dict[str, dict] = {}
    missing: list[str] = []
    incompatible: list[dict] = []
    for name, requirement, module_name in BASE_REQUIREMENTS:
        version = _version_of(name)
        importable = _is_importable(module_name)
        if version is None and importable:
            version = "bundled"
        minimum = requirement.split(">=", 1)[1]
        available = importable or version is not None
        meets = _satisfies_minimum(version, minimum)
        packages[name] = {"version": version, "available": available, "requirement": requirement,
                          "satisfies_requirement": meets}
        if not available:
            missing.append(requirement)
        elif not meets:
            incompatible.append({"name": name, "installed_version": version,
                                 "required": requirement,
                                 "reason": "已安装版本不满足项目要求；自动安装不会升级现有依赖。"})
    gpu = inspect_gpu_environment(refresh=refresh)
    installer = find_compatible_python(refresh=refresh)
    return {
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "implementation": getattr(sys.implementation, "name", "unknown"),
        "platform": sys.platform, "machine": platform.machine(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "packages": packages, "missing": missing, "incompatible": incompatible,
        "installer": installer,
        "gpu": gpu,
        "isolated_dependency_directory": str(base_dependency_root()),
        "isolated_gpu_directory": str(gpu_dependency_root()),
    }


def _dependency_log_path() -> Path:
    directory = _app_data_root() / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"dependency-install-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.log"


def _activate_dependency_directory(directory: Path) -> None:
    if not directory.is_dir():
        return
    resolved = str(directory.resolve())
    current = {str(Path(item).resolve()) for item in sys.path if item}
    if resolved not in current:
        # Preserve distributions already imported from the application bundle.
        sys.path.append(resolved)
    importlib.invalidate_caches()


def _run_base_install(requirements: list[str], *, log_path: Path, progress_callback=None,
                      cancel_callback=None) -> dict:
    installer = find_compatible_python()
    if not installer.get("available"):
        return {"status": "blocked", "reason": installer.get("reason", "matching_external_python_not_found"),
                "installed": [], "output": ""}
    import shutil
    root = base_dependency_root()
    root.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"installer_python={installer['executable']} version={installer.get('version')}\n")
        stream.write(f"base_requirements={json.dumps(requirements, ensure_ascii=False)}\n")
    last_output = ""
    for attempt in (1, 2):
        if cancel_callback is not None and cancel_callback():
            return {"status": "cancelled", "reason": "cancelled", "installed": [], "output": last_output}
        staging = root / f".stage-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        final_target = root / f"install-{uuid.uuid4().hex[:12]}"
        command = [installer["executable"], "-m", "pip", "install", "--disable-pip-version-check",
                   "--no-input", "--only-binary=:all:", "--no-warn-script-location", "--index-url",
                   "https://pypi.org/simple", "--target", str(staging), *requirements]
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"attempt={attempt} argv={json.dumps(command, ensure_ascii=False)}\n")
        _emit(progress_callback, "prepare", "正在为缺失的基础依赖选择匹配的Python wheel",
              progress=None, attempt=attempt)
        run = _run_streaming(command, log_path=log_path, attempt=attempt,
                             progress_callback=progress_callback, cancel_callback=cancel_callback)
        if run.get("cancelled"):
            shutil.rmtree(staging, ignore_errors=True)
            return {"status": "cancelled", "reason": "cancelled", "installed": [],
                    "output": run.get("output", "")}
        if run.get("returncode") == 0:
            staging.replace(final_target)
            _activate_dependency_directory(final_target)
            return {"status": "succeeded", "reason": None, "installed": requirements,
                    "output": run.get("output", "")}
        shutil.rmtree(staging, ignore_errors=True)
        last_output = run.get("output", "")[-2000:] or run.get("error", "pip_install_failed")
        lower = last_output.lower()
        if any(marker in lower for marker in ("no matching distribution", "could not find a version",
                                               "not a supported wheel", "requires a different python")):
            reason = "找不到适用于当前Python与平台的基础依赖wheel；已禁止源码编译。"
            return {"status": "blocked", "reason": reason, "installed": [], "output": last_output}
        transient = any(marker in lower for marker in (
            "readtimeout", "connection reset", "connection aborted", "temporary failure in name resolution",
            "connection broken", "max retries exceeded", "http error 5", "tlsv1 alert",
        ))
        if attempt == 1 and transient:
            _emit(progress_callback, "retry", "检测到临时下载错误，清理临时目录后重试一次",
                  progress=None, attempt=attempt)
            continue
        return {"status": "failed", "reason": "基础依赖安装失败。", "installed": [], "output": last_output}
    return {"status": "failed", "reason": "基础依赖安装失败。", "installed": [], "output": last_output}


def install_missing_dependencies(*, include_gpu: bool = False, progress_callback=None,
                                 cancel_callback=None) -> dict:
    """Install only missing base packages and optionally the supported CuPy wheel.

    Work is written under the current user's application data directory.  The
    synchronous function is intended to run in the UI's worker thread.
    Existing base packages are never upgraded; incompatible installed versions
    are reported for manual resolution.
    """
    initial = inspect_environment(refresh=True)
    missing_names = {item.split(">=", 1)[0].casefold() for item in initial["missing"]}
    base_reqs = [requirement for name, requirement, _module in BASE_REQUIREMENTS
                 if name.casefold() in missing_names]
    already_installed = [name for name, info in initial["packages"].items()
                         if info["available"] and info["satisfies_requirement"]]
    blocked: list[str] = [f"{item['name']}: {item['reason']}" for item in initial["incompatible"]]
    errors: list[str] = []
    installed: list[str] = []
    log_path: Path | None = None
    gpu_result = None

    gpu_ready = bool(initial["gpu"].get("backend_enabled"))
    gpu_install_needed = include_gpu and not gpu_ready
    if base_reqs or gpu_install_needed:
        log_path = _dependency_log_path()
    if base_reqs:
        result = _run_base_install(base_reqs, log_path=log_path, progress_callback=progress_callback,
                                   cancel_callback=cancel_callback)
        if result["status"] == "succeeded":
            installed.extend(result["installed"])
        elif result["status"] == "cancelled":
            return {"status": "cancelled", "installed": installed, "already_installed": already_installed,
                    "missing": initial["missing"], "blocked": blocked, "errors": [],
                    "log_path": str(log_path) if log_path else None, "gpu": initial["gpu"]}
        else:
            (blocked if result["status"] == "blocked" else errors).append(result["reason"])
            if result.get("output"):
                errors.append(result["output"])

    if include_gpu:
        if gpu_ready:
            gpu_result = {"status": "succeeded", "installed": [],
                          "already_installed": [initial["gpu"].get("cupy_distribution") or "CuPy"],
                          "missing": [], "blocked": [], "errors": [], "gpu": initial["gpu"]}
        else:
            gpu_result = install_gpu_dependencies(progress_callback=progress_callback,
                                                  cancel_callback=cancel_callback,
                                                  log_path=log_path)
            installed.extend(gpu_result.get("installed", []))
            already_installed.extend(gpu_result.get("already_installed", []))
            blocked.extend(str(item) for item in gpu_result.get("blocked", []))
            errors.extend(str(item) for item in gpu_result.get("errors", []))

    final = inspect_environment(refresh=True)
    status = "succeeded"
    if blocked:
        status = "blocked"
    elif errors:
        status = "failed"
    elif (base_reqs and final["missing"]) or (include_gpu and not final["gpu"].get("backend_enabled")):
        status = "partial"
    if gpu_result and gpu_result.get("status") == "cancelled":
        status = "cancelled"
    return {
        "status": status, "installed": installed, "already_installed": already_installed,
        "missing": final["missing"], "incompatible": final["incompatible"],
        "blocked": blocked, "errors": errors,
        "log_path": str(log_path) if log_path else (gpu_result or {}).get("log_path"),
        "gpu": final["gpu"], "environment": final,
    }


__all__ = ["BASE_REQUIREMENTS", "activate_application_dependency_paths", "inspect_environment",
           "inspect_gpu_environment", "install_missing_dependencies"]
