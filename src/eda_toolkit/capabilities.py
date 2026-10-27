"""实际能力登记；禁用项不能在选择后静默替换为另一算法。"""
import importlib.metadata
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile

from .errors import fail

SOURCE_FINGERPRINT_SCOPE = "package_python_except_ui_v1"


def source_fingerprint():
    """递归绑定非UI包源码；科学实现或运行编排变化后拒绝复用旧缓存。

    UI文件不参与科学缓存指纹；core、公共契约、配置、runtime、项目/任务服务与
    其他随包Python模块全部纳入，宁可因非数值改动重算，也不漏掉新指标模块。
    冻结发行包读取由相同源码清单生成的resources/build_info.json。
    """
    root = Path(__file__).resolve().parent
    if getattr(sys, "frozen", False):
        info = root / "resources" / "build_info.json"
        if not info.is_file():
            fail("build_fingerprint_missing", "发行包缺少构建指纹。", category="dependency")
        metadata = json.loads(info.read_text(encoding="utf-8"))
        if metadata.get("fingerprint_scope") != SOURCE_FINGERPRINT_SCOPE:
            fail("build_fingerprint_scope", "发行包使用了未知的源码指纹范围。", category="dependency")
        return metadata["source_fingerprint"]
    items = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == "ui":
            continue
        items[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"fingerprint_scope": SOURCE_FINGERPRINT_SCOPE, "files": items}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def numerical_environment_fingerprint():
    """数值依赖/解释器/线程环境改变时，旧记录缓存不再作为兼容结果复用。"""
    versions = {}
    for name in ("numpy", "scipy", "pandas", "statsmodels"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    descriptor = {"python": sys.version.split()[0], "machine": platform.machine(), "packages": versions,
                  "thread_environment": {name: os.environ.get(name) for name in
                                         ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode("utf-8")).hexdigest()


def inspect_output_directory(directory):
    """以最近已存在目录中的临时文件检查写入能力，不创建/覆盖运行结果。"""
    target = Path(directory).expanduser().resolve()
    if target.exists() and not target.is_dir():
        return {"directory": str(target), "writable": False, "reason": "output_path_is_file"}
    parent = target
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".eda-write-probe-", dir=parent)
        os.close(descriptor)
        return {"directory": str(target), "writable": True, "probe_parent": str(parent), "reason": "temporary_file_probe_passed"}
    except OSError as error:
        return {"directory": str(target), "writable": False, "reason": str(error)}
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def get_capabilities(output_directory=None):
    packages = {}
    for distribution in ("numpy", "scipy", "pandas", "PySide6", "matplotlib", "statsmodels", "openpyxl", "reportlab"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    memory = {"total_mb": None, "available_mb": None, "probe_status": "not_available"}
    if os.name == "nt":
        import ctypes
        buffer = ctypes.create_string_buffer(struct.pack("II7Q", 64, 0, *([0] * 7)))
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(buffer)):
            values = struct.unpack("II7Q", buffer.raw[:64])
            memory = {"total_mb": values[2] // 1024 ** 2, "available_mb": values[3] // 1024 ** 2, "probe_status": "measured"}
    gpu = {"probe_status": "nvidia_smi_not_found", "devices": [], "backend_enabled": False}
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            completed = subprocess.run([executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=2, creationflags=0x08000000 if os.name == "nt" else 0)
            gpu.update({"probe_status": "measured" if completed.returncode == 0 else "driver_query_failed",
                        "devices": completed.stdout.strip().splitlines() if completed.returncode == 0 else []})
        except (OSError, subprocess.TimeoutExpired):
            gpu["probe_status"] = "query_unavailable"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count(), "memory": memory, "gpu": gpu,
            "source_fingerprint": source_fingerprint(),
            "numerical_environment_fingerprint": numerical_environment_fingerprint(),
            "output_directory": inspect_output_directory(output_directory) if output_directory is not None else None,
            "packages": packages,
            "methods": {
                "bateman_sparse": {"enabled": packages["scipy"] is not None, "readiness": "research", "reason": "本地 Bateman 非负稀疏反卷积；输出 driver 与收敛诊断。"},
                "filter_separation": {"enabled": packages["scipy"] is not None, "readiness": "research", "reason": "独立SciPy滤波分离；真实设备适用性待验证。"},
                "cvxeda": {"enabled": False, "readiness": "blocked_license", "reason": "尚未完成GPL/求解器发行决策及固定实现数值验证。"},
                "ledalab": {"enabled": False, "readiness": "unavailable_dependency", "reason": "未包含经过验证的Python CDA/DDA移植；基础版不依赖MATLAB。"},
                "gpu": {"enabled": False, "readiness": "experimental", "reason": "未取得端到端性能收益及CPU/GPU数值一致性证据。"},
            }, "science_qualification": "pending_representative_real_data"}


def require_capabilities(config):
    capabilities = get_capabilities()
    selected = config["decomposition"]["method"]
    if not capabilities["methods"][selected]["enabled"]:
        fail("method_unavailable", capabilities["methods"][selected]["reason"], category="dependency", stage="decomposition")
    if config["compute"]["backend"] != "cpu":
        fail("gpu_unavailable", capabilities["methods"]["gpu"]["reason"], category="dependency", stage="compute")
    if config["statistics"]["enabled"] and capabilities["packages"]["statsmodels"] is None:
        fail("statsmodels_missing", "统计功能需要安装当前锁定的 statsmodels。", category="dependency", stage="statistics")
    return capabilities
