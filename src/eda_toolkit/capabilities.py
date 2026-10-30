"""实际能力登记；禁用项不能在选择后静默替换为另一算法。"""
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile

from .errors import fail
from .gpu_support import inspect_gpu_environment

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


def numerical_environment_fingerprint(backend="cpu"):
    """数值依赖/解释器/线程环境改变时，旧记录缓存不再作为兼容结果复用。"""
    versions = {}
    names = ["numpy", "scipy", "pandas", "statsmodels"]
    if backend == "gpu":
        names.extend(("cupy", "cupy-cuda12x", "cupy-cuda13x"))
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    descriptor = {"python": sys.version.split()[0], "machine": platform.machine(), "packages": versions,
                  "compute_backend": backend,
                  "thread_environment": {name: os.environ.get(name) for name in
                                         ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    if backend == "gpu":
        gpu = inspect_gpu_environment()
        descriptor["gpu_runtime"] = {name: gpu.get(name) for name in
                                     ("device", "compute_capability", "driver_version", "cuda_driver_version",
                                      "cuda_runtime_version", "cupy_version", "cupy_distribution")}
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
    for distribution in ("numpy", "scipy", "pandas", "PySide6", "matplotlib", "statsmodels", "openpyxl",
                         "reportlab", "cupy-cuda12x", "cupy-cuda13x"):
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
    gpu = inspect_gpu_environment()
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
                "gpu": {"enabled": bool(gpu.get("backend_enabled")),
                        "readiness": "ready" if gpu.get("backend_enabled") else "unavailable_dependency",
                        "reason": "CuPy/Bateman GPU运行时已通过设备与FFT探测。" if gpu.get("backend_enabled")
                        else gpu.get("blocked_reason", "CuPy GPU运行时未就绪。")},
            }, "science_qualification": "pending_representative_real_data"}


def require_capabilities(config):
    capabilities = get_capabilities()
    selected = config["decomposition"]["method"]
    if not capabilities["methods"][selected]["enabled"]:
        fail("method_unavailable", capabilities["methods"][selected]["reason"], category="dependency", stage="decomposition")
    if config["compute"]["backend"] == "gpu":
        if selected != "bateman_sparse":
            fail("gpu_method_unsupported", "当前GPU后端只加速 Bateman sparse 分解；该分解以外的方法不支持GPU。",
                 category="configuration", stage="compute", field="compute.backend",
                 remediation="将分解方法设为 bateman_sparse，或把 compute.backend 设为 cpu。")
        if not capabilities["gpu"].get("backend_enabled"):
            reason = capabilities["gpu"].get("blocked_reason") or "CuPy GPU运行时未就绪。"
            fail("gpu_unavailable", f"已选择GPU，但后端没有可用的CuPy/NVIDIA运行时：{reason}",
                 category="dependency", stage="compute", field="compute.backend",
                 remediation="检查GPU环境，安装匹配的CuPy wheel和CUDA组件，再重新选择GPU。")
    if config["statistics"]["enabled"] and capabilities["packages"]["statsmodels"] is None:
        fail("statsmodels_missing", "统计功能需要安装当前锁定的 statsmodels。", category="dependency", stage="statistics")
    return capabilities
