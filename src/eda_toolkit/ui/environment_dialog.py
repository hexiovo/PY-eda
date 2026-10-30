"""Asynchronous desktop UI for checking and repairing the local runtime."""

from __future__ import annotations

import importlib
import inspect
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_KEY_LABELS = {
    "python": "Python",
    "version": "版本",
    "status": "状态",
    "message": "说明",
    "missing": "缺少依赖",
    "installed": "已安装",
    "already_installed": "已具备",
    "blocked": "无法安装",
    "blocked_reason": "阻止原因",
    "errors": "错误",
    "packages": "依赖包",
    "gpu": "GPU",
    "cuda": "CUDA",
    "driver": "驱动",
    "devices": "设备",
    "backend_enabled": "GPU 算法后端",
    "supported_methods": "支持的方法",
    "probe_status": "探测结果",
    "probe_reason": "探测原因",
    "readiness": "可用性",
    "log_path": "安装日志",
    "include_gpu": "包含 GPU 依赖",
    "cpu_count": "CPU 核数",
    "memory": "内存",
    "total_mb": "总内存 (MiB)",
    "available_mb": "可用内存 (MiB)",
    "output_directory": "输出目录探测",
    "writable": "可写",
    "directory": "目录",
    "reason": "探测说明",
    "methods": "方法状态",
    "science_qualification": "科学资格",
}
_STATUS_LABELS = {
    "ready": "就绪",
    "available": "可用",
    "missing": "缺少依赖",
    "not_available": "不可用",
    "not_installed": "未安装",
    "succeeded": "完成",
    "partial": "部分完成",
    "blocked": "受阻",
    "failed": "失败",
    "cancelled": "已取消",
    "unavailable": "不可用",
    "unsupported": "不支持",
    "enabled": "已启用",
    "disabled": "未启用",
}


def _display_value(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "未探测"
    if isinstance(value, (list, tuple, set)):
        return "、".join(_display_value(item) for item in value) or "无"
    if isinstance(value, dict):
        if not value:
            return "无"
        return "；".join(f"{_KEY_LABELS.get(str(key), str(key))}：{_display_value(item)}"
                         for key, item in value.items())
    text = str(value)
    return _STATUS_LABELS.get(text.lower(), text)


def _format_tree(data: Any, *, indent: int = 0) -> list[str]:
    prefix = "  " * indent
    if not isinstance(data, dict):
        return [prefix + _display_value(data)]
    lines: list[str] = []
    for key, value in data.items():
        label = _KEY_LABELS.get(str(key), str(key).replace("_", " "))
        if isinstance(value, dict) and value:
            lines.append(f"{prefix}{label}：")
            lines.extend(_format_tree(value, indent=indent + 1))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            lines.append(f"{prefix}{label}：")
            for index, item in enumerate(value, 1):
                lines.append(f"{prefix}  {index}.")
                lines.extend(_format_tree(item, indent=indent + 2))
        else:
            lines.append(f"{prefix}{label}：{_display_value(value)}")
    return lines


class EnvironmentDialog(QDialog):
    """Checks Python/GPU readiness and runs the installer away from the UI thread."""

    def __init__(self, parent: QWidget | None = None, *, service: Any = None,
                 capability_provider: Callable[[], dict[str, Any]] | None = None,
                 on_status_changed: Callable[[dict[str, Any]], None] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("环境检查")
        self.setObjectName("environmentCheckDialog")
        self.setModal(False)
        self.resize(760, 540)
        self.setMinimumSize(620, 420)
        self._service = service
        self._capability_provider = capability_provider
        self._on_status_changed = on_status_changed
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eda-environment")
        self._future: Future[Any] | None = None
        self._operation: str | None = None
        self._cancel_event = threading.Event()
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._closed = False
        self._snapshot: dict[str, Any] | None = None
        self._post_install_summary = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)
        heading = QLabel("检查本机能力、Python 依赖、方法可用性与 GPU 运行时。安装操作在后台执行。")
        heading.setWordWrap(True)
        heading.setObjectName("subtle")
        layout.addWidget(heading)

        self.details = QPlainTextEdit()
        self.details.setObjectName("environmentDetails")
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("正在读取本机环境…")
        self.details.setMinimumHeight(280)
        layout.addWidget(self.details, 1)

        self.progress_label = QLabel("正在检查环境…")
        self.progress_label.setObjectName("environmentProgressLabel")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setObjectName("environmentProgress")
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        bottom = QHBoxLayout()
        self.gpu_checkbox = QCheckBox("同时安装 GPU 依赖（仅供 bateman_sparse）")
        self.gpu_checkbox.setObjectName("installGpuDependencies")
        self.gpu_checkbox.setToolTip("仅当 CUDA 驱动和 GPU 可用时才适合启用；其他方法不会自动切换算法。")
        self.gpu_checkbox.toggled.connect(lambda *_: self.install_button.setEnabled(
            self._operation is None and self._can_install()))
        bottom.addWidget(self.gpu_checkbox)
        bottom.addStretch(1)
        self.refresh_button = QPushButton("重新检查")
        self.refresh_button.setObjectName("refreshEnvironment")
        self.install_button = QPushButton("一键安装缺失依赖")
        self.install_button.setObjectName("installDependencies")
        self.install_button.setObjectName("primary")
        self.install_button.setEnabled(False)
        self.cancel_button = QPushButton("取消安装")
        self.cancel_button.setObjectName("cancelEnvironmentInstall")
        self.cancel_button.setVisible(False)
        self.close_button = QPushButton("关闭")
        for button in (self.refresh_button, self.install_button, self.cancel_button, self.close_button):
            button.setMinimumHeight(32)
        bottom.addWidget(self.refresh_button)
        bottom.addWidget(self.install_button)
        bottom.addWidget(self.cancel_button)
        bottom.addWidget(self.close_button)
        layout.addLayout(bottom)

        self.refresh_button.clicked.connect(self.refresh)
        self.install_button.clicked.connect(self.install_missing)
        self.cancel_button.clicked.connect(self.request_cancel)
        self.close_button.clicked.connect(self.close)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()
        self.refresh()

    def _get_service(self) -> Any:
        if self._service is None:
            self._service = importlib.import_module("eda_toolkit.environment_service")
        return self._service

    def _set_busy(self, operation: str | None) -> None:
        self._operation = operation
        busy = operation is not None
        self.refresh_button.setEnabled(not busy)
        self.install_button.setEnabled(not busy and self._can_install())
        self.gpu_checkbox.setEnabled(not busy)
        self.cancel_button.setVisible(operation == "install")
        self.cancel_button.setEnabled(operation == "install" and not self._cancel_event.is_set())
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)

    def _can_install(self) -> bool:
        if not self._snapshot:
            return False
        environment = self._snapshot.get("environment", {})
        gpu = self._snapshot.get("gpu", {})
        return _has_missing(environment) or (self.gpu_checkbox.isChecked() and _gpu_needs_install(gpu))

    def refresh(self) -> None:
        if self._operation is not None:
            return
        self.details.setPlainText("正在检查本机运行环境…")
        self.progress_label.setText("正在检查环境…")
        self._set_busy("inspect")
        self._future = self._executor.submit(self._inspect_all)

    def _inspect_all(self) -> dict[str, Any]:
        service = self._get_service()
        environment = service.inspect_environment()
        gpu_method = service.inspect_gpu_environment
        try:
            supports_refresh = "refresh" in inspect.signature(gpu_method).parameters
        except (TypeError, ValueError):
            supports_refresh = False
        gpu = gpu_method(refresh=True) if supports_refresh else gpu_method()
        capabilities: dict[str, Any] = {}
        if self._capability_provider is not None:
            try:
                capabilities = self._capability_provider()
            except Exception as error:
                capabilities = {"probe_error": f"{type(error).__name__}: {error}"}
        return {"environment": environment, "gpu": gpu, "capabilities": capabilities}

    def install_missing(self) -> None:
        if self._operation is not None:
            return
        self._cancel_event.clear()
        include_gpu = self.gpu_checkbox.isChecked()
        self.details.appendPlainText("\n正在安装缺失依赖…")
        self.progress_label.setText("安装已在后台开始；下载期间可随时取消。")
        self._set_busy("install")
        self._future = self._executor.submit(self._install, include_gpu)

    def _install(self, include_gpu: bool) -> Any:
        service = self._get_service()
        method = service.install_missing_dependencies
        return method(include_gpu=include_gpu, progress_callback=self._events.put,
                      cancel_callback=self._cancel_event.is_set)

    def request_cancel(self) -> None:
        if self._operation == "install":
            self._cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.progress_label.setText("正在请求取消；安装器会清理本次临时文件…")

    def _poll(self) -> None:
        if self._closed:
            return
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if not isinstance(event, dict):
                continue
            message = str(event.get("message") or event.get("phase") or "安装进度更新")
            self.progress_label.setText(message)
            progress = event.get("progress")
            if isinstance(progress, (int, float)) and 0 <= progress <= 100:
                self.progress.setRange(0, 100)
                self.progress.setValue(int(progress))
            else:
                self.progress.setRange(0, 0)
            self.details.appendPlainText(message)

        future = self._future
        if future is None or not future.done():
            return
        operation = self._operation
        self._future = None
        try:
            result = future.result()
        except Exception as error:  # service errors stay visible in the dialog
            self._set_busy(None)
            self.progress_label.setText(f"环境操作失败：{type(error).__name__}: {error}")
            self.details.appendPlainText(self.progress_label.text())
            return

        if operation == "inspect":
            self._snapshot = result if isinstance(result, dict) else {"environment": {"result": result}}
            self._render_snapshot()
            self._set_busy(None)
            self.progress_label.setText("检查完成。")
            if self._on_status_changed:
                self._on_status_changed(self._snapshot)
        elif operation == "install":
            self._render_install_result(result)
            self._set_busy(None)
            self.progress_label.setText("安装操作已结束；正在重新检查环境…")
            self.refresh()

    def _render_snapshot(self) -> None:
        snapshot = self._snapshot or {}
        lines = ["本机能力", *_format_tree(snapshot.get("capabilities", {})), "", "Python 依赖",
                 *_format_tree(snapshot.get("environment", {})), "", "GPU 环境",
                 *_format_tree(snapshot.get("gpu", {}))]
        if self._post_install_summary:
            lines.extend(["", self._post_install_summary])
        self.details.setPlainText("\n".join(lines))
        environment = snapshot.get("environment", {})
        if _has_missing(environment):
            self.progress_label.setText("发现缺失依赖。可以一次安装常规依赖；GPU 依赖可选。")
        else:
            self.progress_label.setText("常规依赖检查完成。")
        self.install_button.setEnabled(self._can_install())

    def _render_install_result(self, result: Any) -> None:
        if isinstance(result, dict):
            lines = ["安装结果", *_format_tree(result)]
            status = _STATUS_LABELS.get(str(result.get("status", "")).lower(), result.get("status", "已返回"))
        else:
            lines = ["安装结果", _display_value(result)]
            status = "已返回"
        self._post_install_summary = "\n".join(lines)
        self.progress_label.setText(f"安装结果：{status}。正在重新检查常规依赖与 GPU。")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._operation == "install":
            self.request_cancel()
            event.ignore()
            return
        self._closed = True
        self._poll_timer.stop()
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def shutdown(self) -> None:
        """Cancel an active install and wait for its staging cleanup at app exit."""
        if self._operation == "install":
            self._cancel_event.set()
        self._closed = True
        self._poll_timer.stop()
        self._executor.shutdown(wait=True, cancel_futures=True)


def _has_missing(environment: Any) -> bool:
    if not isinstance(environment, dict):
        return False
    missing = environment.get("missing")
    if isinstance(missing, dict):
        return any(bool(value) for value in missing.values())
    if isinstance(missing, (list, tuple, set, str)):
        return bool(missing)
    return str(environment.get("status", "")).lower() in {"missing", "partial", "blocked"}


def _gpu_needs_install(gpu: Any) -> bool:
    if not isinstance(gpu, dict) or not gpu:
        return False
    if _has_missing(gpu):
        return True
    for key in ("cupy", "runtime", "dependencies"):
        component = gpu.get(key)
        if isinstance(component, dict) and _has_missing(component):
            return True
    return bool(gpu.get("install_supported")) and not bool(gpu.get("backend_enabled"))
