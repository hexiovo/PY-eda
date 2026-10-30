"""Optional CuPy runtime discovery and isolated installation helpers.

CuPy is deliberately kept out of the base dependency set.  GPU selection is
available only when this module can load CuPy and execute a CUDA operation on
an NVIDIA device; an ``nvidia-smi`` result alone never enables the backend.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

from .dependency_paths import app_data_root as _app_data_root, base_dependency_root, gpu_dependency_root


CUPY_VERSION = "14.2.0"
PYPI_INDEX_URL = "https://pypi.org/simple"
SUPPORTED_CUDA_MAJORS = (12, 13)

_PYTHON_INFO_SCRIPT = (
    "import importlib.util,json,platform,sys,sysconfig; "
    "print(json.dumps({'version':list(sys.version_info[:3]),"
    "'implementation':sys.implementation.name,'machine':platform.machine(),"
    "'executable':sys.executable,'platform':sysconfig.get_platform(),"
    "'pip':importlib.util.find_spec('pip') is not None}))"
)
_VERIFY_SCRIPT = (
    "import glob,json,os,sys; "
    "target=os.environ.get('EDA_GPU_VERIFY_TARGET',''); "
    "handles=[os.add_dll_directory(p) for p in glob.glob(os.path.join(target,'nvidia','*','bin')) "
    "if os.name=='nt' and os.path.isdir(p)]; "
    "import cupy as cp; "
    "count=int(cp.cuda.runtime.getDeviceCount()); "
    "d=cp.cuda.runtime.getDeviceProperties(0); "
    "x=cp.asarray([1.0,2.0],dtype=cp.float64); y=cp.fft.rfft(x); "
    "cp.cuda.Stream.null.synchronize(); "
    "name=d.get('name',b'unknown'); "
    "name=name.decode(errors='replace') if isinstance(name,bytes) else str(name); "
    "print(json.dumps({'device_count':count,'device':name,'cupy_version':cp.__version__,"
    "'runtime_version':cp.cuda.runtime.runtimeGetVersion(),"
    "'driver_version':cp.cuda.runtime.driverGetVersion(),'fft_size':int(y.size)}))"
)

_STATE_LOCK = threading.RLock()
_GPU_COMPUTE_LOCK = threading.RLock()
_STATUS_CACHE: dict | None = None
_NVIDIA_CACHE: dict | None = None
_INSTALLER_CACHE: dict | None = None
_GPU_CONTEXT: dict | None = None
_DLL_DIRECTORY_HANDLES: list = []
_DLL_DIRECTORY_PATHS: set[str] = set()
_ACTIVE_INSTALLS: set[str] = set()


class GPUUnavailableError(RuntimeError):
    """Raised when a requested CuPy/NVIDIA runtime cannot be used."""


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _normalize_machine(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("-", "_")
    if normalized in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if normalized in {"arm64", "aarch64"}:
        return "aarch64"
    return normalized


def _query_python(executable: str, prefix_args: tuple[str, ...] = ()) -> dict | None:
    try:
        result = subprocess.run(
            [executable, *prefix_args, "-c", _PYTHON_INFO_SCRIPT],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=6, check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        details = json.loads(result.stdout.strip().splitlines()[-1])
        details["launcher"] = executable
        return details
    except (ValueError, IndexError, TypeError):
        return None


def _python_launcher_candidates() -> list[str]:
    candidates: list[str] = []
    if os.name == "nt":
        launcher = shutil.which("py")
        if launcher:
            try:
                result = subprocess.run(
                    [launcher, "-0p"], capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=6,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                for line in result.stdout.splitlines():
                    match = re.search(r"((?:[A-Za-z]:\\|\\\\).+?python(?:w)?\.exe)\s*$", line, re.IGNORECASE)
                    if match:
                        candidates.append(match.group(1).strip())
            except (OSError, subprocess.TimeoutExpired):
                pass
        where = shutil.which("where.exe") or shutil.which("where")
        if where:
            try:
                result = subprocess.run(
                    [where, "python.exe"], capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=4,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                candidates.extend(line.strip() for line in result.stdout.splitlines() if line.strip())
            except (OSError, subprocess.TimeoutExpired):
                pass
    for command in ("python", "python3"):
        executable = shutil.which(command)
        if executable:
            candidates.append(executable)
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def find_compatible_python(*, refresh: bool = False) -> dict:
    """Find an external Python whose CPython ABI matches the running app.

    A PyInstaller executable is never passed to ``-m pip``.  Source installs
    may use the current interpreter; frozen apps must find a real Python binary.
    """
    global _INSTALLER_CACHE
    with _STATE_LOCK:
        if _INSTALLER_CACHE is not None and not refresh:
            return deepcopy(_INSTALLER_CACHE)
    frozen = bool(getattr(sys, "frozen", False))
    current_machine = _normalize_machine(platform.machine())
    required = (sys.version_info.major, sys.version_info.minor)
    candidates: list[str] = []
    if not frozen and Path(sys.executable).name.lower() not in {"eda-toolkit.exe", "eda-cli.exe"}:
        candidates.append(sys.executable)
    candidates.extend(_python_launcher_candidates())
    rejected: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        details = _query_python(candidate)
        if details is None:
            rejected.append({"executable": candidate, "reason": "python_probe_failed"})
            continue
        version = details.get("version", [])
        if details.get("implementation") != "cpython":
            rejected.append({"executable": candidate, "reason": "cpython_required", "version": version})
            continue
        if tuple(version[:2]) != required:
            rejected.append({"executable": candidate, "reason": "python_abi_mismatch", "version": version})
            continue
        if _normalize_machine(details.get("machine")) != current_machine:
            rejected.append({"executable": candidate, "reason": "architecture_mismatch", "machine": details.get("machine")})
            continue
        if not details.get("pip"):
            rejected.append({"executable": candidate, "reason": "pip_unavailable", "version": version})
            continue
        details["available"] = True
        details["frozen"] = frozen
        details["rejected_candidates"] = rejected
        with _STATE_LOCK:
            _INSTALLER_CACHE = deepcopy(details)
        return details
    result = {"available": False, "required_python": list(required), "required_machine": current_machine,
              "frozen": frozen, "reason": "matching_external_python_not_found", "rejected_candidates": rejected}
    with _STATE_LOCK:
        _INSTALLER_CACHE = deepcopy(result)
    return result


def _nvidia_smi_executable() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable:
        return executable
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        candidate = Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def _probe_nvidia(refresh: bool = False) -> dict:
    global _NVIDIA_CACHE
    with _STATE_LOCK:
        if _NVIDIA_CACHE is not None and not refresh:
            return deepcopy(_NVIDIA_CACHE)
    executable = _nvidia_smi_executable()
    result = {"probe_status": "nvidia_smi_not_found", "devices": [], "driver_version": None,
              "cuda_version": None, "nvidia_smi": None}
    if executable:
        result["nvidia_smi"] = executable
        try:
            query = subprocess.run(
                [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=4,
                check=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if query.returncode == 0:
                import csv
                from io import StringIO
                rows = []
                for fields in csv.reader(StringIO(query.stdout)):
                    if not fields:
                        continue
                    fields = [field.strip() for field in fields]
                    name = fields[0] if fields else "unknown"
                    try:
                        memory_mb = int(float(fields[1])) if len(fields) > 1 else None
                    except ValueError:
                        memory_mb = None
                    driver = fields[2] if len(fields) > 2 else None
                    rows.append({"name": name, "memory_total_mb": memory_mb})
                    if not result["driver_version"] and driver:
                        result["driver_version"] = driver
                result["devices"] = rows
                result["probe_status"] = "measured" if rows else "no_nvidia_device"
            else:
                result["probe_status"] = "driver_query_failed"
            try:
                summary = subprocess.run(
                    [executable], capture_output=True, text=True, encoding="utf-8", errors="replace",
                    timeout=4, check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                version_match = re.search(r"CUDA Version:\s*(\d+(?:\.\d+)?)", summary.stdout)
                if version_match:
                    result["cuda_version"] = version_match.group(1)
                driver_match = re.search(r"Driver Version:\s*([\d.]+)", summary.stdout)
                if driver_match:
                    result["driver_version"] = result["driver_version"] or driver_match.group(1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except (OSError, subprocess.TimeoutExpired):
            result["probe_status"] = "query_unavailable"
    with _STATE_LOCK:
        _NVIDIA_CACHE = deepcopy(result)
    return result


def _supported_runtime_target(nvidia: dict) -> tuple[str | None, str | None]:
    raw = nvidia.get("cuda_version")
    if not raw:
        return None, "cuda_driver_version_unavailable"
    try:
        major = int(str(raw).split(".", 1)[0])
    except ValueError:
        return None, "cuda_driver_version_unavailable"
    if major == 12:
        return "cupy-cuda12x", None
    if major == 13:
        return "cupy-cuda13x", None
    return None, "unsupported_cuda_driver_version"


def _cupy_distribution_version() -> tuple[str | None, str | None]:
    for distribution in ("cupy-cuda12x", "cupy-cuda13x", "cupy"):
        try:
            return importlib.metadata.version(distribution), distribution
        except importlib.metadata.PackageNotFoundError:
            continue
    return None, None


def _isolated_gpu_target(cuda_package: str) -> Path:
    return gpu_dependency_root() / f"{cuda_package}-{CUPY_VERSION}-{_python_tag()}"


def _prepare_isolated_cuda_root(path: Path) -> Path | None:
    """Expose isolated CUDA component DLLs through a toolkit-shaped root.

    ``cuda.pathfinder`` scans ``site.getsitepackages()`` for NVIDIA wheels.
    Packages installed with pip's ``--target`` are on ``PYTHONPATH`` but are
    not returned by that function.  Its documented ``CUDA_PATH`` fallback
    expects the traditional ``<root>/bin`` layout, so hard-link (or, where
    hard links are unavailable, copy) the isolated component DLLs into the
    already-installed CUDA runtime package's ``bin`` directory.
    """
    if os.name != "nt":
        return None
    runtime_root = path / "nvidia" / "cuda_runtime"
    bin_directory = runtime_root / "bin"
    if not bin_directory.is_dir():
        return None
    with _STATE_LOCK:
        for source in sorted(path.glob("nvidia/*/bin/*.dll")):
            destination = bin_directory / source.name
            try:
                if source.resolve() == destination.resolve() or os.path.samefile(source, destination):
                    continue
            except OSError:
                pass
            if destination.exists():
                source_stat = source.stat()
                destination_stat = destination.stat()
                # copy2 preserves both values. This also avoids re-copying the
                # toolkit-shaped layout on every environment inspection.
                if (source_stat.st_size == destination_stat.st_size
                        and source_stat.st_mtime_ns == destination_stat.st_mtime_ns):
                    continue
                raise GPUUnavailableError(
                    f"隔离CUDA运行时目录中存在同名DLL冲突：{destination.name}。"
                )
            try:
                os.link(source, destination)
                continue
            except OSError:
                pass
            temporary = bin_directory / f".{source.name}.{uuid.uuid4().hex}.tmp"
            try:
                shutil.copy2(source, temporary)
                temporary.replace(destination)
            except OSError as error:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
                raise GPUUnavailableError(
                    f"无法为CuPy加载隔离CUDA组件DLL {source.name}：{error}"
                ) from error
    return runtime_root


def _activate_dependency_path(path: Path) -> None:
    if not path.is_dir():
        return
    cuda_root = _prepare_isolated_cuda_root(path)
    if cuda_root is not None:
        # This is process-local and is inherited only by subprocesses started
        # from this app; no persistent Windows environment setting is changed.
        os.environ["CUDA_PATH"] = str(cuda_root.resolve())
    resolved = str(path.resolve())
    if resolved not in {str(Path(item).resolve()) for item in sys.path if item}:
        # Appending preserves the exact NumPy/SciPy versions already loaded by
        # the source interpreter or PyInstaller bundle.
        sys.path.append(resolved)
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in path.glob("nvidia/*/bin"):
            if directory.is_dir():
                resolved_dir = str(directory.resolve())
                if resolved_dir not in _DLL_DIRECTORY_PATHS:
                    try:
                        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(resolved_dir))
                        _DLL_DIRECTORY_PATHS.add(resolved_dir)
                    except OSError:
                        continue
    importlib.invalidate_caches()


def _runtime_details(cupy_module) -> dict:
    count = int(cupy_module.cuda.runtime.getDeviceCount())
    if count <= 0:
        raise GPUUnavailableError("没有可用的 NVIDIA CUDA 设备。")
    properties = cupy_module.cuda.runtime.getDeviceProperties(0)
    device_name = properties.get("name", b"unknown")
    if isinstance(device_name, bytes):
        device_name = device_name.decode("utf-8", errors="replace")
    x = cupy_module.asarray([1.0, 2.0], dtype=cupy_module.float64)
    spectrum = cupy_module.fft.rfft(x)
    cupy_module.cuda.Stream.null.synchronize()
    nvidia = _probe_nvidia()
    version, distribution = _cupy_distribution_version()
    if version is None:
        version = getattr(cupy_module, "__version__", None)
    cupy_path = Path(getattr(cupy_module, "__file__", "")).resolve()
    try:
        isolated = cupy_path.is_relative_to(gpu_dependency_root().resolve())
    except (OSError, ValueError):
        isolated = False
    return {
        "probe_status": "measured", "backend_enabled": True, "available": True,
        "devices": [f"{item['name']} ({item['memory_total_mb']} MiB)" if item.get("memory_total_mb") is not None
                    else str(item["name"]) for item in nvidia.get("devices", [])]
                   or [str(device_name)],
        "device_details": nvidia.get("devices") or [{"name": str(device_name), "memory_total_mb": None}],
        "device": str(device_name), "device_count": count,
        "driver_version": nvidia.get("driver_version"), "cuda_driver_version": nvidia.get("cuda_version"),
        "cuda_runtime_version": int(cupy_module.cuda.runtime.runtimeGetVersion()),
        "cupy_version": version, "cupy_distribution": distribution,
        "compute_capability": f"{properties.get('major', '?')}.{properties.get('minor', '?')}",
        "fft_probe_length": int(spectrum.size), "source": "isolated" if isolated else "environment",
        "blocked_reason": None,
    }


def load_gpu_runtime() -> dict:
    """Load and cache CuPy after one real device/FFT probe."""
    global _GPU_CONTEXT, _STATUS_CACHE
    with _STATE_LOCK:
        if _GPU_CONTEXT is not None:
            return _GPU_CONTEXT
        nvidia = _probe_nvidia()
        cuda_package, target_error = _supported_runtime_target(nvidia)
        if cuda_package:
            _activate_dependency_path(_isolated_gpu_target(cuda_package))
        try:
            cupy_module = importlib.import_module("cupy")
            details = _runtime_details(cupy_module)
        except Exception as error:
            raise GPUUnavailableError(f"CuPy GPU runtime 不可用：{type(error).__name__}: {error}") from error
        if target_error and not details.get("cuda_driver_version"):
            details["blocked_reason"] = target_error
        _GPU_CONTEXT = {"cupy": cupy_module, "details": details, "lock": _GPU_COMPUTE_LOCK}
        _STATUS_CACHE = deepcopy(details)
        return _GPU_CONTEXT


def inspect_gpu_environment(*, refresh: bool = False) -> dict:
    """Return stable JSON-safe GPU readiness, including real CuPy execution."""
    global _STATUS_CACHE
    with _STATE_LOCK:
        if _STATUS_CACHE is not None and not refresh:
            return deepcopy(_STATUS_CACHE)
    nvidia = _probe_nvidia(refresh=refresh)
    package_version, package_name = _cupy_distribution_version()
    cuda_package, cuda_error = _supported_runtime_target(nvidia)
    supported_python = (sys.implementation.name == "cpython" and sys.version_info.major == 3
                        and 11 <= sys.version_info.minor <= 14)
    machine = _normalize_machine(platform.machine())
    supported_machine = machine in {"x86_64", "aarch64"}
    base = {
        "probe_status": nvidia["probe_status"], "available": False, "backend_enabled": False,
        "devices": [f"{item['name']} ({item['memory_total_mb']} MiB)" if item.get("memory_total_mb") is not None
                    else str(item["name"]) for item in nvidia.get("devices", [])],
        "device_details": nvidia.get("devices", []), "device": None,
        "device_count": len(nvidia.get("devices", [])),
        "driver_version": nvidia.get("driver_version"), "cuda_driver_version": nvidia.get("cuda_version"),
        "cuda_runtime_version": None, "cupy_version": package_version, "cupy_distribution": package_name,
        "cuda_package": cuda_package, "compute_capability": None,
        "python_version": sys.version.split()[0], "python_abi": _python_tag(),
        "platform_machine": platform.machine(), "install_supported": False,
        "installer_python": None, "blocked_reason": None,
    }
    if cuda_package:
        _activate_dependency_path(_isolated_gpu_target(cuda_package))
    package_version, package_name = _cupy_distribution_version()
    base["cupy_version"] = package_version
    base["cupy_distribution"] = package_name
    if not nvidia.get("devices"):
        base["blocked_reason"] = "nvidia_gpu_not_detected"
    elif not supported_python:
        base["blocked_reason"] = "unsupported_python_abi"
    elif not supported_machine:
        base["blocked_reason"] = "unsupported_platform_architecture"
    elif not cuda_package:
        base["blocked_reason"] = cuda_error
    elif package_version is None:
        installer = find_compatible_python(refresh=refresh)
        if installer.get("available"):
            base["installer_python"] = installer.get("executable")
        base["install_supported"] = bool(installer.get("available"))
        if not installer.get("available"):
            base["blocked_reason"] = installer.get("reason")
    else:
        base["install_supported"] = True
    try:
        if importlib.util.find_spec("cupy") is not None:
            context = load_gpu_runtime()
            base.update(deepcopy(context["details"]))
            base["cuda_package"] = cuda_package or context["details"].get("cupy_distribution")
            base["install_supported"] = True
        elif base["blocked_reason"] is None:
            base["blocked_reason"] = "cupy_not_installed"
    except Exception as error:
        base["backend_enabled"] = False
        base["available"] = False
        base["blocked_reason"] = f"cupy_runtime_unavailable: {type(error).__name__}: {error}"
    with _STATE_LOCK:
        _STATUS_CACHE = deepcopy(base)
    return base


def _emit(progress_callback, phase: str, message: str, *, progress=None, attempt: int = 0) -> None:
    if not callable(progress_callback):
        return
    try:
        progress_callback({"phase": phase, "message": message, "progress": progress, "attempt": attempt})
    except Exception:
        return


def _log_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line.rstrip("\r\n") + "\n")


def _clean_partial_cupy_import() -> None:
    for name in tuple(sys.modules):
        if name == "cupy" or name.startswith("cupy.") or name == "cupyx" or name.startswith("cupyx."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()


def _run_streaming(command: list[str], *, log_path: Path, attempt: int,
                   progress_callback=None, cancel_callback=None, env=None,
                   timeout_seconds: float | None = None) -> dict:
    """Run a fixed argv without a shell; stream logs and honor cancellation."""
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=env, shell=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
    except OSError as error:
        _log_line(log_path, f"process start failed: {type(error).__name__}: {error}")
        return {"returncode": None, "error": str(error), "cancelled": False, "output": ""}
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, name="eda-gpu-installer-output", daemon=True)
    reader.start()
    output_tail: deque[str] = deque(maxlen=80)
    sentinel_seen = False
    cancelled = False
    timed_out = False
    started_at = time.monotonic()
    while process.poll() is None or not sentinel_seen:
        if cancel_callback is not None:
            try:
                requested = bool(cancel_callback())
            except Exception:
                requested = False
            if requested and process.poll() is None:
                cancelled = True
                _terminate_process_tree(process)
        if (timeout_seconds is not None and not cancelled and not timed_out
                and time.monotonic() - started_at >= timeout_seconds and process.poll() is None):
            timed_out = True
            _log_line(log_path, f"process timed out after {timeout_seconds:g} seconds")
            _emit(progress_callback, "timeout", f"验证进程超过 {timeout_seconds:g} 秒，已停止进程树",
                  progress=None, attempt=attempt)
            _terminate_process_tree(process)
        try:
            line = lines.get(timeout=0.15)
        except queue.Empty:
            continue
        if line is None:
            sentinel_seen = True
            continue
        line = line.rstrip("\r\n")
        output_tail.append(line)
        _log_line(log_path, line)
        _emit(progress_callback, "install", line[:400], progress=None, attempt=attempt)
    return {"returncode": process.wait(), "error": None, "cancelled": cancelled,
            "timed_out": timed_out, "output": "\n".join(output_tail)}


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Stop a timed-out/cancelled helper and any child processes it started."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        taskkill = shutil.which("taskkill.exe") or shutil.which("taskkill")
        if taskkill:
            try:
                subprocess.run(
                    [taskkill, "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, text=True, timeout=5, check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
    else:
        import signal
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                import signal
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
            try:
                process.kill()
            except OSError:
                pass
            process.wait(timeout=4)


def _verify_installed_target(python_executable: str, prefix_args: tuple[str, ...], target: Path,
                             *, log_path: Path, attempt: int, progress_callback=None,
                             cancel_callback=None) -> dict:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(target) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    env["EDA_GPU_VERIFY_TARGET"] = str(target)
    try:
        cuda_root = _prepare_isolated_cuda_root(target)
    except GPUUnavailableError as error:
        _log_line(log_path, f"isolated CUDA path setup failed: {error}")
        return {"ok": False, "cancelled": False, "reason": str(error)}
    if cuda_root is not None:
        env["CUDA_PATH"] = str(cuda_root.resolve())
    command = [python_executable, *prefix_args, "-c", _VERIFY_SCRIPT]
    _emit(progress_callback, "verify", "验证CuPy wheel、CUDA运行时与设备", progress=None, attempt=attempt)
    run = _run_streaming(command, log_path=log_path, attempt=attempt,
                         progress_callback=progress_callback, cancel_callback=cancel_callback, env=env,
                         timeout_seconds=60.0)
    if run["cancelled"]:
        return {"ok": False, "cancelled": True, "reason": "cancelled"}
    if run.get("timed_out"):
        return {"ok": False, "cancelled": False, "timed_out": True,
                "reason": "cupy_cuda_runtime_verification_timed_out_after_60s"}
    if run["returncode"] != 0:
        return {"ok": False, "cancelled": False, "reason": run["output"][-1500:] or run["error"] or "cuda_verification_failed"}
    try:
        last_line = run["output"].splitlines()[-1]
        details = json.loads(last_line)
    except (IndexError, ValueError):
        details = {"verification_output": run["output"][-500:]}
    return {"ok": True, "cancelled": False, "details": details}


def install_gpu_dependencies(*, progress_callback=None, cancel_callback=None, log_path: str | Path | None = None) -> dict:
    """Install pinned CuPy wheels in the app-private runtime directory.

    Source and frozen applications use an ABI-matched CPython installer.  A
    frozen executable is never invoked as ``python -m pip``.  Source builds are
    disabled so missing wheel support becomes an explicit, logged failure.
    """
    global _STATUS_CACHE, _GPU_CONTEXT, _NVIDIA_CACHE
    status = inspect_gpu_environment(refresh=True)
    if status.get("backend_enabled"):
        return {"status": "succeeded", "installed": [], "already_installed": [status.get("cupy_distribution") or "CuPy"],
                "missing": [], "blocked": [], "errors": [], "log_path": str(log_path) if log_path else None,
                "gpu": status}
    if status.get("blocked_reason") not in {"cupy_not_installed", None}:
        if not status.get("install_supported"):
            return {"status": "blocked", "installed": [], "already_installed": [], "missing": ["CuPy"],
                    "blocked": [status.get("blocked_reason")], "errors": [], "log_path": str(log_path) if log_path else None,
                    "gpu": status}
        if status.get("cupy_version"):
            return {"status": "blocked", "installed": [], "already_installed": [status.get("cupy_distribution") or "CuPy"],
                    "missing": [], "blocked": [status.get("blocked_reason")], "errors": [],
                    "log_path": str(log_path) if log_path else None, "gpu": status}
    nvidia = _probe_nvidia()
    cuda_package, error = _supported_runtime_target(nvidia)
    if not nvidia.get("devices"):
        return {"status": "blocked", "installed": [], "already_installed": [], "missing": ["CuPy"],
                "blocked": ["nvidia_gpu_not_detected"], "errors": [], "log_path": str(log_path) if log_path else None,
                "gpu": status}
    if error or not cuda_package:
        return {"status": "blocked", "installed": [], "already_installed": [], "missing": ["CuPy"],
                "blocked": [error or "unsupported_cuda_driver_version"], "errors": [],
                "log_path": str(log_path) if log_path else None, "gpu": status}
    installer = find_compatible_python()
    if not installer.get("available"):
        return {"status": "blocked", "installed": [], "already_installed": [], "missing": ["CuPy"],
                "blocked": [installer.get("reason", "matching_external_python_not_found")], "errors": [],
                "log_path": str(log_path) if log_path else None, "gpu": status}
    target = _isolated_gpu_target(cuda_package)
    if target.exists():
        _activate_dependency_path(target)
        _clean_partial_cupy_import()
        try:
            context = load_gpu_runtime()
            refreshed = inspect_gpu_environment(refresh=True)
            return {"status": "succeeded", "installed": [], "already_installed": [f"{cuda_package}=={CUPY_VERSION}"],
                    "missing": [], "blocked": [], "errors": [], "log_path": str(log_path) if log_path else None,
                    "gpu": refreshed, "runtime": context["details"]}
        except Exception as error:
            return {"status": "blocked", "installed": [], "already_installed": [f"{cuda_package}=={CUPY_VERSION}"],
                    "missing": [], "blocked": [f"existing_isolated_cupy_unusable: {error}"], "errors": [],
                    "log_path": str(log_path) if log_path else None, "gpu": inspect_gpu_environment(refresh=True)}
    root = gpu_dependency_root()
    root.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_dir = _app_data_root() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"gpu-install-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.log"
    else:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(target.resolve())
    with _STATE_LOCK:
        if key in _ACTIVE_INSTALLS:
            return {"status": "blocked", "installed": [], "already_installed": [], "missing": ["CuPy"],
                    "blocked": ["gpu_install_already_running"], "errors": [], "log_path": str(log_path),
                    "gpu": status}
        _ACTIVE_INSTALLS.add(key)
    try:
        _log_line(log_path, f"target={target}")
        _log_line(log_path, f"installer_python={installer.get('executable')} version={installer.get('version')}")
        _log_line(log_path, f"package={cuda_package}[ctk]=={CUPY_VERSION} index={PYPI_INDEX_URL}")
        attempts = 2
        last_error = ""
        for attempt in range(1, attempts + 1):
            if cancel_callback is not None and cancel_callback():
                return {"status": "cancelled", "installed": [], "already_installed": [], "missing": ["CuPy"],
                        "blocked": [], "errors": [], "log_path": str(log_path), "gpu": status}
            staging = root / f".stage-{uuid.uuid4().hex}"
            staging.mkdir(parents=True, exist_ok=False)
            command = [
                installer["executable"], "-m", "pip", "install",
                "--disable-pip-version-check", "--no-input", "--only-binary=:all:",
                "--no-warn-script-location", "--index-url", PYPI_INDEX_URL,
                "--target", str(staging), f"{cuda_package}[ctk]=={CUPY_VERSION}",
            ]
            _emit(progress_callback, "prepare", f"开始安装 {cuda_package} {CUPY_VERSION}（应用隔离目录）",
                  progress=None, attempt=attempt)
            _log_line(log_path, f"attempt={attempt} argv={json.dumps(command, ensure_ascii=False)}")
            run = _run_streaming(command, log_path=log_path, attempt=attempt,
                                 progress_callback=progress_callback, cancel_callback=cancel_callback)
            if run["cancelled"]:
                shutil.rmtree(staging, ignore_errors=True)
                return {"status": "cancelled", "installed": [], "already_installed": [], "missing": ["CuPy"],
                        "blocked": [], "errors": [], "log_path": str(log_path), "gpu": status}
            if run["returncode"] != 0:
                last_error = run["output"][-2000:] or run["error"] or "pip_install_failed"
                shutil.rmtree(staging, ignore_errors=True)
                no_wheel = any(marker in last_error.lower() for marker in (
                    "no matching distribution", "no matching version", "could not find a version",
                    "not a supported wheel", "requires a different python",
                ))
                if no_wheel:
                    message = (f"未找到与 Python {_python_tag()}、{platform.system()}、{platform.machine()} 匹配的"
                               f"{cuda_package}=={CUPY_VERSION} 官方 wheel；已禁止源码编译。")
                    _log_line(log_path, "blocked: " + message)
                    return {"status": "blocked", "installed": [], "already_installed": [], "missing": ["CuPy"],
                            "blocked": [message], "errors": [last_error], "log_path": str(log_path),
                            "gpu": inspect_gpu_environment(refresh=True)}
                transient = any(marker in last_error.lower() for marker in (
                    "readtimeout", "connection reset", "connection aborted", "temporary failure in name resolution",
                    "connection broken", "max retries exceeded", "http error 5", "tlsv1 alert",
                ))
                if attempt < attempts and transient:
                    _emit(progress_callback, "retry", "安装失败，清理临时文件后重试一次；详情已写入日志",
                          progress=None, attempt=attempt)
                    _log_line(log_path, "retrying once after failed pip attempt")
                    continue
                return {"status": "failed", "installed": [], "already_installed": [], "missing": ["CuPy"],
                        "blocked": [], "errors": [last_error], "log_path": str(log_path),
                        "gpu": inspect_gpu_environment(refresh=True)}
            verified = _verify_installed_target(installer["executable"], (), staging, log_path=log_path,
                                                attempt=attempt, progress_callback=progress_callback,
                                                cancel_callback=cancel_callback)
            if not verified.get("ok"):
                shutil.rmtree(staging, ignore_errors=True)
                if verified.get("cancelled"):
                    return {"status": "cancelled", "installed": [], "already_installed": [], "missing": ["CuPy"],
                            "blocked": [], "errors": [], "log_path": str(log_path), "gpu": status}
                last_error = str(verified.get("reason"))
                return {"status": "failed", "installed": [], "already_installed": [], "missing": ["CuPy"],
                        "blocked": ["cupy_cuda_runtime_verification_failed"], "errors": [last_error],
                        "log_path": str(log_path), "gpu": inspect_gpu_environment(refresh=True)}
            if target.exists():
                shutil.rmtree(staging, ignore_errors=True)
                return {"status": "blocked", "installed": [], "already_installed": [], "missing": ["CuPy"],
                        "blocked": ["isolated_gpu_target_created_by_another_install"], "errors": [],
                        "log_path": str(log_path), "gpu": inspect_gpu_environment(refresh=True)}
            staging.replace(target)
            _activate_dependency_path(target)
            _clean_partial_cupy_import()
            with _STATE_LOCK:
                _GPU_CONTEXT = None
                _STATUS_CACHE = None
            try:
                context = load_gpu_runtime()
            except Exception as error:
                return {"status": "failed", "installed": [f"{cuda_package}=={CUPY_VERSION}"],
                        "already_installed": [], "missing": [], "blocked": ["app_runtime_verification_failed"],
                        "errors": [f"{type(error).__name__}: {error}"], "log_path": str(log_path),
                        "gpu": inspect_gpu_environment(refresh=True)}
            refreshed = inspect_gpu_environment(refresh=True)
            _emit(progress_callback, "complete", f"CuPy GPU runtime 已就绪：{refreshed.get('device')}",
                  progress=1.0, attempt=attempt)
            return {"status": "succeeded", "installed": [f"{cuda_package}=={CUPY_VERSION}"],
                    "already_installed": [], "missing": [], "blocked": [], "errors": [],
                    "log_path": str(log_path), "gpu": refreshed, "runtime": context["details"]}
        return {"status": "failed", "installed": [], "already_installed": [], "missing": ["CuPy"],
                "blocked": [], "errors": [last_error or "gpu_install_failed"], "log_path": str(log_path),
                "gpu": inspect_gpu_environment(refresh=True)}
    finally:
        with _STATE_LOCK:
            _ACTIVE_INSTALLS.discard(key)


__all__ = [
    "CUPY_VERSION", "GPUUnavailableError", "base_dependency_root", "find_compatible_python",
    "gpu_dependency_root", "inspect_gpu_environment", "install_gpu_dependencies", "load_gpu_runtime",
]
