"""PySide6 中文研究工作台；界面只调用公开的数据、质控和 pipeline API。"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QFont, QFontDatabase, QKeySequence, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLayout, QLineEdit, QListWidget, QMenu,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QStatusBar,
    QTextBrowser, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem,
    QToolButton, QVBoxLayout, QWidget,
)


PAGES = ("项目", "导入与映射", "质控", "预处理", "分解与 SCR", "多人同步", "统计设计", "结果",
         "Marker 编辑", "任务与日志")
INPUT_KEYS = ("manifest", "participants", "events", "conditions", "group_membership",
              "sync_anchors", "manual_qc", "statistical_design")


def _table_widget(frame: Any, *, max_rows: int = 200, parent: QWidget | None = None) -> QTableWidget:
    """将已有 DataFrame 展示为有行数上限的只读 Qt 表，不复制或解析输入文件。"""
    import pandas as pd

    data = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame or [])
    data = data.head(max_rows)
    table = QTableWidget(len(data), len(data.columns), parent)
    table.setHorizontalHeaderLabels([str(item) for item in data.columns])
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    for row_index, row in enumerate(data.itertuples(index=False, name=None)):
        for column_index, value in enumerate(row):
            item = QTableWidgetItem("" if pd.isna(value) else str(value))
            table.setItem(row_index, column_index, item)
    table.horizontalHeader().setStretchLastSection(True)
    table.resizeColumnsToContents()
    table.setMinimumHeight(150)
    return table


def _safe_config_text(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)


def _cjk_font_candidates() -> tuple[Path, ...]:
    """Return the bundled CJK font first for consistent exported plot labels."""
    bundled = (Path(__file__).resolve().parent.parent / "resources" / "help" / "fonts" /
               "NotoSansSC-Regular.ttf")
    return (bundled, Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"), Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"))


def _ui_font_candidates() -> tuple[Path, ...]:
    """Prefer the Windows system sans CJK fonts for readable native controls."""
    bundled = (Path(__file__).resolve().parent.parent / "resources" / "help" / "fonts" /
               "NotoSansSC-Regular.ttf")
    return (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"),
            bundled, Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"))


def _resolved_output_directory(config: dict[str, Any]) -> Path:
    """把相对输出路径锚定到已解析的项目 base_directory。"""
    output = Path(config["output"]["directory"]).expanduser()
    if not output.is_absolute():
        base = config.get("input", {}).get("base_directory") or Path.cwd()
        output = Path(base).expanduser() / output
    return output.resolve()


def _capability_summary(caps: dict[str, Any]) -> str:
    """以真实探针结果汇总本机资源、方法可用性和输出写入状态。"""
    memory = caps.get("memory", {}) or {}
    total_mb = memory.get("total_mb")
    available_mb = memory.get("available_mb")
    ram = (f"{total_mb} MiB total · {available_mb} MiB available" if total_mb is not None and available_mb is not None
           else f"状态 {memory.get('probe_status', 'not_available')}")
    gpu = caps.get("gpu", {}) or {}
    devices = "; ".join(str(item) for item in gpu.get("devices", [])) or "未发现可查询设备"
    output = caps.get("output_directory") or {}
    output_state = "可写" if output.get("writable") else "不可写/未探测"
    output_reason = output.get("reason", "probe_not_requested")
    packages = caps.get("packages", {})
    methods = caps.get("methods", {})
    return "\n".join([
        f"Python {caps.get('python')}  ·  {sys.platform}  ·  CPU {caps.get('cpu_count') or '未知'} 核",
        f"RAM：{ram}（{memory.get('probe_status', 'not_available')}）",
        f"GPU：{devices} · 探针 {gpu.get('probe_status', 'not_available')} · GPU 算法{'已启用' if gpu.get('backend_enabled') else '未启用'}",
        f"输出探针：{output_state} · {output.get('directory', '未指定')} · {output_reason}",
        "依赖：" + "，".join(f"{name} {version or '未安装'}" for name, version in packages.items()),
        "方法：" + "；".join(f"{name}={'可用' if item.get('enabled') else item.get('readiness')}" for name, item in methods.items()),
        "科学资格：" + caps.get("science_qualification", "未声明"),
    ])


def _compute_compatibility_error(config: dict[str, Any]) -> str | None:
    if (config.get("compute", {}).get("backend") == "gpu" and
            config.get("decomposition", {}).get("method") != "bateman_sparse"):
        return "GPU 后端仅支持 bateman_sparse。请修改计算后端或分解方法；不会自动切换算法。"
    return None


def _format_error(error: Any) -> str:
    details = getattr(error, "eda_details", None)
    if isinstance(details, dict):
        code = details.get("code", "运行错误")
        remedy = details.get("remediation")
        text = f"{code}：{error}"
        return text + (f"\n建议：{remedy}" if remedy else "")
    return f"{type(error).__name__}: {error}"


def warm_runtime_imports(progress_callback=None) -> None:
    """Preload expensive runtime modules before the main UI is constructed."""
    if progress_callback:
        progress_callback("正在加载 NumPy、pandas 和 SciPy…")
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy  # noqa: F401
    if progress_callback:
        progress_callback("正在准备工作台运行组件…")
    from ..capabilities import get_capabilities  # noqa: F401
    from ..config import PRESETS, get_preset, parse_json_text, resolve_config, save_config  # noqa: F401
    from ..core_io import (export_template, list_templates, load_recordings, preview_csv,
                           read_signal_csv, read_template)  # noqa: F401
    from ..core_qc import append_annotation, build_quality_mask, redo_annotation, undo_annotation  # noqa: F401
    from ..pipeline import resolve_run_config  # noqa: F401
    from ..project_store import (load_recovery_candidates, open_project, read_project_snapshot,
                                 save_project_revision, load_last_good_draft_export,
                                 restore_last_good_draft_export)  # noqa: F401
    from ..task_store import (enqueue_task, get_task, list_task_attempts, list_tasks,
                              query_tasks, reorder_queued_task, request_task_cancel, retry_task)  # noqa: F401
    from ..task_queue import (claim_next_task, execute_claimed_task, reconcile_interrupted_tasks,
                              _process_start_identity)  # noqa: F401
    from ..marker_store import (delete_marker, export_marker_snapshot, get_markers, import_markers,
                                list_marker_source_tombstones, merge_marker_event_rows, read_marker_csv,
                                redo_marker_action, undo_marker_action, update_marker)  # noqa: F401
    from ..runtime_preflight import preflight_run  # noqa: F401
    from ..runtime_logging import create_runtime_session, export_logs, load_eta_history, search_logs  # noqa: F401


def create_window(initial_config: Any = None):
    """创建工作台并返回 ``(window, state)``；state 暴露控件索引供自动化和集成使用。

    初始配置先由 ``resolve_config`` 完整校验。分析任务通过单线程 executor 调用
    同一 ``pipeline.run_pipeline``；worker 只读深拷贝快照，所有控件更新由主线程
    QTimer 消费队列事件。窗口和其子页面均使用 Qt 自带控件，无自定义 widget 类。
    """
    import numpy as np
    import pandas as pd

    from ..capabilities import get_capabilities
    from ..config import PRESETS, get_preset, parse_json_text, resolve_config, save_config
    from ..core_io import (export_template, list_templates, load_recordings, preview_csv,
                           read_signal_csv, read_template)
    from ..core_qc import append_annotation, build_quality_mask, redo_annotation, undo_annotation
    from ..pipeline import resolve_run_config
    from ..project_store import (load_recovery_candidates, open_project, read_project_snapshot,
                                 save_project_revision, load_last_good_draft_export,
                                 restore_last_good_draft_export)
    from ..task_store import (enqueue_task, get_task, list_task_attempts, list_tasks,
                              query_tasks, reorder_queued_task, request_task_cancel, retry_task)
    from ..task_queue import (claim_next_task, execute_claimed_task, reconcile_interrupted_tasks,
                             _process_start_identity)
    from ..marker_store import (delete_marker, export_marker_snapshot, get_markers, import_markers,
                                list_marker_source_tombstones, merge_marker_event_rows, read_marker_csv,
                                redo_marker_action, undo_marker_action, update_marker)
    from ..runtime_preflight import preflight_run
    from ..runtime_logging import create_runtime_session, export_logs, load_eta_history, search_logs

    from ..runtime_identity import configure_app_identity, icon_path
    from .parameter_form import ParameterForm
    from .structured_design import ClickableComboBox, DriverParameterForm, StatisticsDesignForm

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    # Qt may assign its own Windows AppUserModelID during QApplication setup.
    # Set the application's stable identity after that setup and before showing windows.
    identity_result = configure_app_identity()
    ui_family = "Segoe UI"
    ui_font_path = None
    ui_font_source = "platform_default"
    for font_path in _ui_font_candidates():
        if font_path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(font_path))
            families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
            if families:
                ui_family = families[0]
                ui_font_path = str(font_path.resolve())
                ui_font_source = "bundled" if font_path.name == "NotoSansSC-Regular.ttf" else "system_fallback"
                break
    app.setFont(QFont(ui_family, 11))
    config = resolve_run_config(initial_config) if initial_config is not None else resolve_config({})
    state: dict[str, Any] = {
        "app": app, "config": config, "app_identity": identity_result,
        "font_metadata": {"family": ui_family, "path": ui_font_path,
                                                              "source": ui_font_source},
        "records": [], "associated": {}, "qc_result": None,
        "run_future": None, "cancel_event": None, "run_queue": queue.Queue(), "run_id": 0,
        "preview_id": 0, "mapping_preview_snapshot": None, "busy": False, "closed": False, "last_summary": None,
        "project": None, "runtime_session": None, "autosave_dirty": False, "autosave_state": "未配置项目",
        "compute_configuration_error": _compute_compatibility_error(config),
        "preflight_future": None, "preflight_report": None, "pending_enqueue": None,
        "preflight_cancel_event": None, "preflight_generation": 0, "preflight_context": None,
        "log_export_future": None, "log_export_target": None,
        "runtime_events": [], "runtime_work_plan": None, "runtime_history": [], "selected_task_id": None,
        "qc_records_preview_id": None, "qc_records_config_sha256": None,
        "marker_waveform_request_id": 0, "marker_waveform_pending": None,
        "selected_marker_id": None,
        "log_message_count": 0,
        "last_figures": [], "temp_demo": None, "widget_bindings": [], "template_buttons": {},
        "table_preview": None, "last_loaded_path": None, "executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="eda-pipeline"),
        "io_executor": ThreadPoolExecutor(max_workers=1, thread_name_prefix="eda-preview"),
    }

    def invalidate_mapping_preview() -> int:
        """使所有已启动的列映射读取预览失效，防止旧文件/配置回写界面。"""
        state["preview_id"] += 1
        state["mapping_preview_snapshot"] = None
        return state["preview_id"]

    def qc_source_config_sha256(source_config: dict[str, Any]) -> str:
        payload = {"input": source_config.get("input", {}), "mapping": source_config.get("mapping", {})}
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def canonical_config_sha256(source_config: dict[str, Any]) -> str:
        canonical = json.dumps(source_config, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), default=str, allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    state["invalidate_mapping_preview"] = invalidate_mapping_preview
    widgets: dict[str, Any] = {}
    window = QMainWindow()
    if callable(icon_path):
        try:
            configured_icon = icon_path()
            if configured_icon and Path(configured_icon).is_file():
                app.setWindowIcon(QIcon(str(configured_icon)))
                window.setWindowIcon(QIcon(str(configured_icon)))
        except Exception:
            pass
    window.setWindowTitle("EDA Toolkit · 皮肤电研究工作台")
    screen = app.primaryScreen()
    available = screen.availableGeometry() if screen is not None else None
    if available is not None and available.width() > 0 and available.height() > 0:
        # QScreen geometry is already expressed in Qt logical pixels. Leave
        # room for native title-bar/frame decoration and keep the whole window
        # inside the usable work area, including the taskbar inset.
        width_budget = max(320, int(available.width()) - 24)
        height_budget = max(320, int(available.height()) - 60)
        minimum_width = min(1120, width_budget)
        minimum_height = min(760, height_budget)
        initial_width = min(1420, width_budget)
        initial_height = min(920, height_budget)
        window.setMinimumSize(minimum_width, minimum_height)
        window.resize(initial_width, initial_height)
        x = int(available.x()) + max(8, (int(available.width()) - initial_width) // 2)
        y = int(available.y()) + 8
        window.move(x, y)
    else:
        window.resize(1420, 920)
        window.setMinimumSize(1120, 760)
    window.setObjectName("mainWindow")
    combo_arrow_path = str((Path(__file__).resolve().parent.parent / "resources" / "icons" /
                            "chevron_down.svg").resolve()).replace("\\", "/")
    window.setStyleSheet(f"""
        QWidget {{ background: #F3F6FA; color: #22364B; font-family: '{ui_family}', sans-serif; font-size: 10pt; }}
        QMainWindow {{ background: #F3F6FA; }}
        QLabel {{ background: transparent; }}
        QLabel#eyebrow {{ color: #2563EB; font-size: 9pt; font-weight: 700; letter-spacing: 1px; }}
        QLabel#pageTitle {{ color: #142B45; font-size: 18pt; font-weight: 700; }}
        QLabel#subtle {{ color: #607286; font-size: 9pt; }}
        QFrame#header, QGroupBox {{ background: #FFFFFF; border: 1px solid #DCE4EE; border-radius: 9px; }}
        QGroupBox {{ margin-top: 12px; padding: 12px 10px 8px 10px; font-weight: 650; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 5px; color: #334C68; }}
        QPushButton {{ background: #FFFFFF; border: 1px solid #C8D4E1; border-radius: 6px; padding: 7px 12px; min-height: 26px; }}
        QPushButton:hover {{ border-color: #2563EB; background: #EFF5FF; }}
        QPushButton:pressed {{ background: #DFEAFF; }}
        QPushButton:disabled {{ color: #8999A5; background: #EDF1F4; }}
        QToolButton {{ background: #FFFFFF; border: 1px solid #C8D4E1; border-radius: 6px; padding: 7px 12px; min-height: 26px; }}
        QToolButton:hover {{ border-color: #2563EB; background: #EFF5FF; }}
        QToolButton:pressed {{ background: #DFEAFF; }}
        QToolButton:disabled {{ color: #8999A5; background: #EDF1F4; }}
        QPushButton#primary {{ background: #2563EB; color: white; border: 1px solid #2563EB; font-weight: 700; }}
        QPushButton#primary:hover {{ background: #1D4ED8; }}
        QPushButton#danger {{ color: #9A3D39; border-color: #D9A7A3; }}
        QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {{ background: white; border: 1px solid #C8D4E1; border-radius: 5px; padding: 5px 7px; selection-background-color: #2563EB; }}
        QComboBox {{ min-height: 28px; padding-right: 22px; }}
        QComboBox::drop-down {{ width: 23px; border-left: 1px solid #DCE4EE; border-top-right-radius: 5px; border-bottom-right-radius: 5px; background: #F7F9FC; }}
        QComboBox::down-arrow {{ image: url("{combo_arrow_path}"); width: 10px; height: 10px; }}
        QComboBox QAbstractItemView {{ background: white; selection-background-color: #E7EFFF; selection-color: #142B45; border: 1px solid #C8D4E1; }}
        QDoubleSpinBox, QSpinBox {{ max-width: 210px; }}
        QPlainTextEdit {{ font-family: Consolas, '{ui_family}', monospace; font-size: 9pt; }}
        QFrame#sidebar {{ background: #142B45; border: none; }}
        QLabel#sidebarBrand {{ color: #FFFFFF; font-size: 13pt; font-weight: 700; }}
        QLabel#sidebarCaption {{ color: #AFC1D4; font-size: 9pt; }}
        QListWidget#sidebarNav {{ background: #142B45; border: none; padding: 8px 6px; outline: none; }}
        QListWidget#sidebarNav::item {{ padding: 9px 8px; margin: 3px 0; border-radius: 6px; color: #B8C9DB; font-size: 9pt; }}
        QListWidget#sidebarNav::item:selected {{ background: #2563EB; color: #FFFFFF; font-weight: 700; }}
        QListWidget#sidebarNav::item:hover:!selected {{ background: #213E5E; color: #FFFFFF; }}
        QTableWidget {{ background: white; alternate-background-color: #F5F8FC; gridline-color: #E2E8F0; border: 1px solid #DCE4EE; }}
        QHeaderView::section {{ background: #EEF3F9; color: #334C68; border: 0; padding: 6px; font-weight: 650; }}
        QProgressBar {{ background: #E2E9EE; border: none; border-radius: 5px; text-align: center; min-height: 15px; }}
        QProgressBar::chunk {{ background: #2563EB; border-radius: 5px; }}
        QScrollArea {{ border: none; background: transparent; }}
        QScrollBar:vertical {{ background: #EDF2F5; width: 12px; margin: 2px; border-radius: 6px; }}
        QScrollBar::handle:vertical {{ background: #B8C9D4; min-height: 28px; border-radius: 5px; }}
        QScrollBar::handle:vertical:hover {{ background: #91A9B8; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{ background: #EDF2F5; height: 12px; margin: 2px; border-radius: 6px; }}
        QScrollBar::handle:horizontal {{ background: #B8C9D4; min-width: 28px; border-radius: 5px; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    """)

    root = QWidget()
    root_layout = QVBoxLayout(root)
    root_layout.setContentsMargins(12, 10, 12, 8)
    root_layout.setSpacing(7)

    header = QFrame()
    header.setObjectName("header")
    header_layout = QVBoxLayout(header)
    header_layout.setContentsMargins(12, 7, 12, 7)
    top_row = QHBoxLayout()
    identity = QVBoxLayout()
    eyebrow = QLabel("EDA TOOLKIT")
    eyebrow.setObjectName("eyebrow")
    title = QLabel("皮肤电研究工作台")
    title.setObjectName("pageTitle")
    identity.addWidget(eyebrow)
    identity.addWidget(title)
    top_row.addLayout(identity)
    top_row.addStretch(1)
    preset = QComboBox()
    preset.setObjectName("presetSelector")
    preset_labels = {
        "single_standard": "单人基础分析",
        "single_event": "事件相关分析",
        "single_advanced": "单人扩展分析",
        "single_driver": "Driver 分解分析",
        "group_sync": "多人同步分析",
        "custom": "自定义方案",
    }
    for preset_id in PRESETS:
        preset.addItem(preset_labels[preset_id], preset_id)
    preset.setCurrentIndex(max(0, preset.findData(config["pipeline_id"])))
    widgets["preset"] = preset
    top_row.addWidget(QLabel("研究预设"))
    top_row.addWidget(preset)
    header_layout.addLayout(top_row)

    # Keep button objects as hidden compatibility bridges for existing signal
    # handlers and automation. User-facing access lives in menus below.
    action_specs = (
        ("import_config", "导入设置", "导入完整设置文件（JSON）"),
        ("save_config", "导出设置", "导出当前控件/已应用设置为 JSON；每次选择保存位置"),
        ("plan", "计划预览", "查看当前已应用设置对应的运行计划"),
        ("preflight", "运行前检查", "检查当前设置和项目输入"),
        ("trial", "单文件预检+入队", "选择单个信号文件，完成运行前检查后入队"),
        ("batch", "批量预检+入队", "使用当前项目输入完成运行前检查后入队"),
    )
    for key, caption, tooltip in action_specs:
        button = QPushButton(caption)
        button.setToolTip(tooltip)
        button.hide()
        if key in ("preflight", "trial", "batch"):
            button.setEnabled(False)
        widgets[key] = button

    run_row = QHBoxLayout()
    add_task_button = QToolButton()
    add_task_button.setText("添加任务…")
    add_task_button.setMinimumHeight(30)
    add_task_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    add_task_menu = QMenu(add_task_button)
    add_single_action = add_task_menu.addAction("单文件预检并入队…")
    add_batch_action = add_task_menu.addAction("批量预检并入队")
    add_task_button.setEnabled(False)
    add_single_action.setEnabled(False)
    add_batch_action.setEnabled(False)
    add_task_button.setMenu(add_task_menu)
    widgets.update({"add_task_button": add_task_button, "add_task_menu": add_task_menu,
                    "add_single_task_action": add_single_action,
                    "add_batch_task_action": add_batch_action})
    run_row.addWidget(add_task_button)
    for key, caption in (("start_queue", "开始队列"), ("cancel", "取消任务")):
        button = QPushButton(caption)
        button.setMinimumHeight(30)
        button.setObjectName("primary" if key == "start_queue" else "danger")
        button.setEnabled(False)
        if key == "cancel":
            button.hide()
        run_row.addWidget(button)
        widgets[key] = button
    run_row.addStretch(1)
    header_layout.addLayout(run_row)
    root_layout.addWidget(header)

    body = QSplitter(Qt.Orientation.Horizontal)
    sidebar = QFrame()
    sidebar.setObjectName("sidebar")
    sidebar.setMinimumWidth(190)
    sidebar.setMaximumWidth(220)
    sidebar_layout = QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(12, 16, 12, 14)
    sidebar_layout.setSpacing(8)
    sidebar_brand = QLabel("EDA Toolkit")
    sidebar_brand.setObjectName("sidebarBrand")
    sidebar_caption = QLabel("皮肤电研究工作台")
    sidebar_caption.setObjectName("sidebarCaption")
    sidebar_layout.addWidget(sidebar_brand)
    sidebar_layout.addWidget(sidebar_caption)
    sidebar_rule = QFrame()
    sidebar_rule.setFrameShape(QFrame.Shape.HLine)
    sidebar_rule.setStyleSheet("color: #38536F;")
    sidebar_layout.addWidget(sidebar_rule)
    navigation = QListWidget()
    navigation.setObjectName("sidebarNav")
    navigation.addItems([f"{index:02d}   {name}" for index, name in enumerate(PAGES, start=1)])
    navigation.setCurrentRow(0)
    navigation.setUniformItemSizes(True)
    sidebar_layout.addWidget(navigation, 1)
    sidebar_footer = QLabel("配置  ·  处理  ·  结果")
    sidebar_footer.setObjectName("sidebarCaption")
    sidebar_layout.addWidget(sidebar_footer)
    stack = QStackedWidget()
    widgets["navigation"] = navigation
    widgets["stack"] = stack
    widgets["pages"] = {}
    body.addWidget(sidebar)
    body.addWidget(stack)
    body.setStretchFactor(0, 0)
    body.setStretchFactor(1, 1)
    body.setSizes([204, 1160])
    root_layout.addWidget(body, 1)

    progress = QProgressBar()
    progress.setRange(0, 100)
    progress.setValue(0)
    progress.setFixedWidth(170)
    progress.setMaximumHeight(15)
    progress.setTextVisible(False)
    status = QLabel("就绪 · 参数为研究示例，尚未按设备/任务科学资格验证")
    status.setObjectName("subtle")
    status.setMinimumWidth(0)
    status_bar = QStatusBar(window)
    status_bar.setSizeGripEnabled(False)
    status_bar.setFixedHeight(27)
    status_bar.addWidget(status, 1)
    status_bar.addPermanentWidget(progress)
    window.setStatusBar(status_bar)
    log_header = QLabel("本次会话暂无消息")
    log_header.setObjectName("subtle")
    log_box = QPlainTextEdit()
    log_box.setReadOnly(True)
    log_box.setMaximumBlockCount(1200)
    log_box.setPlaceholderText("本次会话暂无消息。任务进度和错误会显示在这里。")
    log_box.setFixedHeight(88)
    window.setCentralWidget(root)
    widgets.update({"progress": progress, "status": status, "log": log_box,
                    "log_header": log_header, "status_bar": status_bar,
                    "header": header, "body": body})

    def log(message: Any) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        log_box.appendPlainText(f"{stamp}  {message}")
        state["log_message_count"] += 1
        log_header.setText(f"本次会话 · {state['log_message_count']} 条消息")
        session = state.get("runtime_session")
        if session and callable(session.get("event_sink")):
            try:
                from ..runtime_events import emit_run_event
                project = state.get("project") or {}
                emit_run_event(session["event_sink"], {
                    "project_id": project.get("project_id"), "stage": "ui", "kind": "progress",
                    "level": "INFO", "message": str(message), "unit_type": "",
                    "details": {"source": "desktop_ui"},
                })
            except Exception:
                pass

    def set_status(message: str, percent: int | None = None) -> None:
        status.setText(message)
        if percent is not None:
            progress.setRange(0, 100)
            progress.setValue(max(0, min(100, int(percent))))

    def set_progress(percent: float | None, message: str) -> None:
        status.setText(message)
        if percent is None:
            progress.setRange(0, 0)
        else:
            progress.setRange(0, 100)
            progress.setValue(max(0, min(100, int(percent))))

    autosave_timer = QTimer(window)
    autosave_timer.setSingleShot(True)
    state["autosave_timer"] = autosave_timer

    def project_state_snapshot() -> dict[str, Any] | None:
        project = state.get("project")
        if not project:
            return None
        return {"project_id": project["project_id"],
                "project_revision": int(project.get("project_revision", 0)),
                "draft_revision": int(project.get("draft_revision", 0)),
                "marker_revision": int(project.get("marker_revision", 0)),
                "qc_revision": int(project.get("qc_revision", 0)),
                "project_directory": project["path"]}

    def materialize_project_events(snapshot: dict[str, Any]) -> None:
        """Freeze managed markers together with untouched source events for one task."""
        project = state.get("project")
        managed = export_marker_snapshot(project) if project else []
        tombstones = list_marker_source_tombstones(project) if project else []
        if not managed and not tombstones:
            return
        source_rows: list[dict[str, Any]] = []
        source_report: dict[str, Any] = {}
        configured_source = snapshot.get("input", {}).get("events")
        if isinstance(configured_source, str) and configured_source.strip():
            source_path = Path(configured_source).expanduser()
            if not source_path.is_absolute():
                base = snapshot.get("input", {}).get("base_directory") or project["path"]
                source_path = Path(base).expanduser() / source_path
            source_path = source_path.resolve()
            if source_path.is_file():
                source_rows, source_report = read_marker_csv(source_path)
            else:
                raise FileNotFoundError(f"configured_events_file_missing:{source_path}")
        rows = merge_marker_event_rows(source_rows, source_report, managed, tombstones)
        fields = ("recording_id", "event_id", "source_event_id", "source_event_id_original",
                  "marker_id", "onset_s", "duration_s", "event_type", "condition", "channel",
                  "clock_domain", "source_path", "source_sha256", "marker_revision")
        from io import StringIO
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        encoded = stream.getvalue().encode("utf-8-sig")
        digest = hashlib.sha256(encoded).hexdigest()
        target = Path(project["path"]) / ".eda" / "snapshots" / f"events-r{project.get('marker_revision', 0)}-{digest[:12]}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as output_stream:
                    output_stream.write(encoded)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        snapshot["input"]["events"] = str(target.resolve())

    def persist_project_draft(*, kind: str = "autosave", audit_action: dict[str, Any] | None = None) -> int | None:
        if state.get("preflight_future") is not None and not state["preflight_future"].done():
            invalidate_preflight("项目草稿正在更新")
        project = state.get("project")
        if not project:
            state["autosave_dirty"] = True
            state["autosave_state"] = "未配置项目；草稿尚未落盘"
            if "project_autosave_status" in widgets:
                widgets["project_autosave_status"].setText(state["autosave_state"])
            return None
        autosave_timer.stop()
        patch = {"config": copy.deepcopy(state["config"]),
                 "ui": {"current_page": max(0, navigation.currentRow()) if "navigation" in widgets else 0}}
        try:
            revision = save_project_revision(project, patch,
                                             int(project.get("project_revision", 0)), kind=kind,
                                             audit_table="qc_actions" if audit_action else None,
                                             audit_action=audit_action)
            state["autosave_dirty"] = False
            state["autosave_state"] = f"已保存 · revision {revision} · {datetime.now().strftime('%H:%M:%S')}"
            if "project_autosave_status" in widgets:
                widgets["project_autosave_status"].setText(state["autosave_state"])
            return revision
        except Exception as error:
            state["autosave_state"] = f"保存失败 · {_format_error(error)}"
            if "project_autosave_status" in widgets:
                widgets["project_autosave_status"].setText(state["autosave_state"])
            log(state["autosave_state"])
            return None

    def schedule_project_autosave() -> None:
        if state.get("loading_snapshot"):
            return
        state["autosave_dirty"] = True
        project = state.get("project")
        if not project:
            state["autosave_state"] = "未配置项目；草稿尚未落盘"
            if "project_autosave_status" in widgets:
                widgets["project_autosave_status"].setText(state["autosave_state"])
            return
        if not state["config"].get("workspace", {}).get("autosave_enabled", True):
            state["autosave_state"] = "自动保存已关闭；请手动打开项目页保存"
            if "project_autosave_status" in widgets:
                widgets["project_autosave_status"].setText(state["autosave_state"])
            return
        delay = int(state["config"].get("workspace", {}).get("autosave_debounce_ms", 750))
        autosave_timer.start(delay)

    autosave_timer.timeout.connect(lambda: persist_project_draft())

    def add_page(name: str, subtitle: str = "") -> tuple[QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(9)
        heading = QLabel(name)
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("subtle")
            sub.setWordWrap(True)
            layout.addWidget(sub)
        layout.addStretch(1)
        scroll.setWidget(page)
        stack.addWidget(scroll)
        widgets["pages"][name] = {"scroll": scroll, "content": page, "layout": layout}
        return page, layout

    def put_content_before_stretch(layout: QVBoxLayout, item: QWidget) -> None:
        index = max(0, layout.count() - 1)
        layout.insertWidget(index, item)

    def path_row(layout: QFormLayout, key: str, label: str, default: str = "") -> QLineEdit:
        line = QLineEdit(default)
        line.setPlaceholderText("选择文件或输入路径")
        browse = QPushButton("浏览…")
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line, 1)
        row.addWidget(browse)
        layout.addRow(label, holder)
        widgets[key] = line
        widgets[key + "_browse"] = browse

        def choose_file() -> None:
            filename, _ = QFileDialog.getOpenFileName(window, f"选择{label}", line.text(),
                                                     "数据与配置 (*.csv *.tsv *.json);;所有文件 (*.*)")
            if filename:
                line.setText(filename)

        browse.clicked.connect(choose_file)
        return line

    def bind_value(key: str, control: Any, cast: Any = None, *, nullable: bool = False) -> None:
        widgets[key] = control
        state["widget_bindings"].append((key, control, cast, nullable))

        def changed(*_args: Any) -> None:
            if state["busy"]:
                return
            section, field = key.split(".", 1)
            value = control.value() if hasattr(control, "value") else control.currentData()
            if value is None and not nullable and hasattr(control, "currentText"):
                value = control.currentText()
            if cast is not None and value is not None:
                try:
                    value = cast(value)
                except (TypeError, ValueError):
                    return
            state["config"][section][field] = value
            invalidate_mapping_preview()
            update_config_editor()

        signal = getattr(control, "valueChanged", None) or getattr(control, "currentIndexChanged", None)
        if signal is not None:
            signal.connect(changed)

    def update_config_editor() -> None:
        # The editable snapshot represents the whole analysis config; changing it
        # invalidates a preview that was launched against an earlier revision.
        invalidate_mapping_preview()
        if "config_json" in widgets:
            widgets["config_json"].setPlainText(_safe_config_text(state["config"]))
        schedule_project_autosave()

    def refresh_bound_controls() -> None:
        for key, control, _cast, _nullable in state["widget_bindings"]:
            section, field = key.split(".", 1)
            value = state["config"].get(section, {}).get(field)
            control.blockSignals(True)
            if isinstance(control, QComboBox):
                index = control.findData(value)
                if index < 0:
                    index = control.findText(str(value) if value is not None else "")
                if index >= 0:
                    control.setCurrentIndex(index)
            elif isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, QSpinBox):
                control.setValue(int(value or 0))
            elif isinstance(control, QDoubleSpinBox):
                control.setValue(float(value or 0))
            elif isinstance(control, QLineEdit) and isinstance(value, list):
                control.setText(",".join(map(str, value)))
            control.blockSignals(False)
        preset.blockSignals(True)
        preset.setCurrentIndex(max(0, preset.findData(state["config"]["pipeline_id"])))
        preset.blockSignals(False)
        if "output_directory" in widgets:
            widgets["output_directory"].setText(str(state["config"]["output"]["directory"]))
        if "compute_workers" in widgets:
            widgets["compute_workers"].blockSignals(True)
            widgets["compute_workers"].setValue(int(state["config"]["compute"]["workers"]))
            widgets["compute_workers"].blockSignals(False)
        if "compute_backend" in widgets:
            backend_box = widgets["compute_backend"]
            backend_box.blockSignals(True)
            backend_box.setCurrentIndex(max(0, backend_box.findData(state["config"]["compute"]["backend"])))
            backend_box.blockSignals(False)
        if "baseline_edit" in widgets:
            baseline = state["config"]["preprocess"].get("baseline")
            widgets["baseline_edit"].setText(",".join(map(str, baseline)) if baseline else "")
        if "feature_registry_table" in widgets:
            refresh_feature_selector()
        if "extended_parameter_editors" in widgets:
            refresh_extended_parameter_editors()
        if "driver_parameters" in widgets:
            widgets["driver_parameters"].set_config(state["config"]["decomposition"]["driver"])
        if "statistics_designs_layout" in widgets:
            refresh_statistics_forms()
        if "preprocess_reference_labels" in widgets:
            refresh_preprocess_references()
        if "workbook_status_table" in widgets:
            refresh_workbook_status_table(state.get("workbook_statuses"))
        if "mapping_unit" in widgets:
            unit = state["config"]["mapping"].get("eda_unit")
            widgets["mapping_unit"].setCurrentIndex(max(0, widgets["mapping_unit"].findData(unit)))
            rate = state["config"]["mapping"].get("sampling_rate_hz")
            widgets["mapping_rate"].setText("" if rate is None else str(rate))
            widgets["mapping_clock"].setText(str(state["config"]["mapping"].get("clock_domain", "recording_local")))
            for key, box in widgets.get("mapping_columns", {}).items():
                source = state["config"]["mapping"].get("columns", {}).get(key)
                index = box.findText(str(source)) if source else -1
                if index >= 0:
                    box.setCurrentIndex(index)
        update_config_editor()

    def apply_config(candidate: Any, *, source: str = "配置") -> bool:
        try:
            resolved = resolve_config(candidate)
        except Exception as error:
            log(f"{source}未应用：{_format_error(error)}")
            QMessageBox.warning(window, "配置校验未通过", _format_error(error))
            return False
        if (resolved["compute"].get("backend") == "gpu" and
                resolved["decomposition"].get("method") != "bateman_sparse"):
            detail = "GPU 后端仅支持 bateman_sparse。请修改计算后端或分解方法；系统不会自动切换算法。"
            QMessageBox.warning(window, "计算后端与方法不兼容", detail)
            log(f"{source}未应用：{detail}")
            return False
        invalidate_mapping_preview()
        state["config"] = resolved
        refresh_bound_controls()
        refresh_compute_guard()
        set_status(f"{source}已校验 · {resolved['pipeline_id']}")
        log(f"已应用完整配置：{source}")
        return True

    def import_config() -> None:
        filename, _ = QFileDialog.getOpenFileName(window, "导入完整 JSON 配置", "", "JSON (*.json)")
        if not filename:
            return
        try:
            candidate = resolve_run_config(filename)
        except Exception as error:
            QMessageBox.critical(window, "JSON 读取失败", _format_error(error))
            return
        if apply_config(candidate, source=Path(filename).name):
            widgets["config_path"].setText(filename)
            state["last_loaded_path"] = filename

    def save_config_action() -> None:
        focused = QApplication.focusWidget()
        if focused is not None:
            focused.clearFocus()
            app.processEvents()
        try:
            # Statistics cards are the authoritative editor; gather them before
            # export so a valid form edit cannot be lost behind stale state.
            candidate = resolve_run_config(config_with_analysis_forms())
        except Exception as error:
            QMessageBox.warning(window, "无法导出当前设置", _format_error(error))
            return
        filename, _ = QFileDialog.getSaveFileName(
            window, "导出设置为 JSON", "eda-settings.json", "JSON (*.json)")
        if not filename:
            return
        try:
            save_config(candidate, filename)
            state["config"] = candidate
            widgets["config_path"].setText(filename)
            persist_project_draft()
            set_status(f"配置已保存：{Path(filename).name}")
            log(f"已保存配置：{filename}")
        except Exception as error:
            QMessageBox.critical(window, "保存失败", _format_error(error))

    def apply_json_editor() -> None:
        try:
            candidate = resolve_run_config(parse_json_text(widgets["config_json"].toPlainText()))
        except Exception as error:
            QMessageBox.warning(window, "JSON 语法错误", _format_error(error))
            return
        apply_config(candidate, source="高级 JSON 配置")

    def show_plan() -> None:
        try:
            resolved = resolve_run_config(config_with_analysis_forms())
            caps = get_capabilities(_resolved_output_directory(resolved))
            state["capabilities"] = caps
            caps_label.setText(_capability_summary(caps))
            method = caps["methods"][resolved["decomposition"]["method"]]
            output_probe = caps.get("output_directory") or {}
            lines = [f"预设：{resolved['pipeline_id']}", f"分解：{resolved['decomposition']['method']} · {method['readiness']}",
                     f"输入 manifest：{resolved['input'].get('manifest') or 'signals list'}",
                     f"项目 base：{resolved['input'].get('base_directory')}",
                     f"输出根目录：{output_probe.get('directory', _resolved_output_directory(resolved))}",
                     f"输出探针：{'可写' if output_probe.get('writable') else '不可写'} · {output_probe.get('reason', 'unknown')}",
                     f"采样率：{resolved['mapping'].get('sampling_rate_hz') or '从时间轴/输入声明估计'} Hz",
                     f"同步分析：{'启用' if resolved['synchrony']['enabled'] else '关闭'}",
                     f"统计分析：{'启用' if resolved['statistics']['enabled'] else '关闭'}",
                     f"Workers：{resolved['compute']['workers']}",
                     f"能力状态：{caps['science_qualification']}",
                     "运行时将冻结完整配置快照，并对每个record单独记录失败或成功。"]
            if resolved["decomposition"]["method"] != "filter_separation":
                lines.append(f"方法不可用原因：{method['reason']}")
            QMessageBox.information(window, "运行计划预览", "\n".join(lines))
            log("计划预览已生成（仅展示，不启动计算）")
        except Exception as error:
            QMessageBox.warning(window, "计划配置无效", _format_error(error))

    # ── 项目与完整 JSON 配置 ──────────────────────────────────────────────
    project_page, project_layout = add_page("项目", "选择本地数据与输出位置；项目文件不离开本机。")
    project_group = QGroupBox("项目位置")
    project_form = QFormLayout(project_group)
    config_path = QLineEdit("")
    config_path.setPlaceholderText("JSON 配置文件路径（可选）")
    widgets["config_path"] = config_path
    project_form.addRow("当前配置文件", config_path)
    project_directory_line = QLineEdit(str(config.get("workspace", {}).get("project_directory") or ""))
    project_directory_line.setPlaceholderText("先打开项目目录，之后草稿、marker、队列与日志才会持久化")
    project_open_button = QPushButton("打开/创建项目…")
    project_directory_holder = QWidget()
    project_directory_row = QHBoxLayout(project_directory_holder)
    project_directory_row.setContentsMargins(0, 0, 0, 0)
    project_directory_row.addWidget(project_directory_line, 1)
    project_directory_row.addWidget(project_open_button)
    project_form.addRow("持久化项目", project_directory_holder)
    project_autosave_status = QLabel("未配置项目；草稿尚未落盘")
    project_autosave_status.setObjectName("subtle")
    project_autosave_status.setWordWrap(True)
    project_form.addRow("自动保存状态", project_autosave_status)
    widgets.update({"project_directory": project_directory_line,
                    "project_open": project_open_button,
                    "project_autosave_status": project_autosave_status})
    output_line = QLineEdit(str(config["output"]["directory"]))
    output_button = QPushButton("选择目录…")
    output_holder = QWidget()
    output_row = QHBoxLayout(output_holder)
    output_row.setContentsMargins(0, 0, 0, 0)
    output_row.addWidget(output_line, 1)
    output_row.addWidget(output_button)
    project_form.addRow("结果输出目录", output_holder)
    widgets["output_directory"] = output_line

    def choose_output() -> None:
        directory = QFileDialog.getExistingDirectory(window, "选择结果输出目录", output_line.text() or str(Path.cwd()))
        if directory:
            output_line.setText(directory)
            state["config"]["output"]["directory"] = directory
            update_config_editor()

    output_button.clicked.connect(choose_output)
    output_line.editingFinished.connect(lambda: (state["config"]["output"].update(directory=output_line.text().strip()), update_config_editor()))
    put_content_before_stretch(project_layout, project_group)

    def _live_task_probe(task: dict[str, Any]) -> bool | None:
        attempt = task.get("attempt") if isinstance(task.get("attempt"), dict) else {}
        pid = attempt.get("pid")
        expected_identity = attempt.get("process_start_identity")
        if not isinstance(pid, int) or not expected_identity:
            return None
        actual_identity = _process_start_identity(pid)
        if actual_identity is not None:
            return actual_identity == expected_identity
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.WaitForSingleObject.restype = ctypes.c_uint32
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if not handle:
                return False if ctypes.get_last_error() == 87 else None  # ERROR_INVALID_PARAMETER means no such PID.
            try:
                wait_result = int(kernel32.WaitForSingleObject(handle, 0))
                if wait_result == 0:  # WAIT_OBJECT_0: the process has exited.
                    return False
                if wait_result != 258:  # WAIT_TIMEOUT: process is alive; identity could not be read.
                    return None
            finally:
                kernel32.CloseHandle(handle)
            return None
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
        return None

    def invalidate_preflight(reason: str) -> bool:
        """Cancel and invalidate an in-flight preflight without applying its report."""
        future = state.get("preflight_future")
        if future is None:
            return False
        context = state.get("preflight_context")
        if isinstance(context, dict):
            context["invalidated_reason"] = reason
        state["preflight_generation"] = int(state.get("preflight_generation", 0)) + 1
        cancel_event = state.get("preflight_cancel_event")
        if cancel_event is not None:
            cancel_event.set()
        widgets["cancel"].setEnabled(False)
        widgets["cancel"].hide()
        state["pending_enqueue"] = None
        state["preflight_report"] = None
        if future.done():
            state["preflight_future"] = None
            state["preflight_context"] = None
            state["preflight_cancel_event"] = None
            widgets["cancel"].setEnabled(False)
            widgets["cancel"].hide()
        set_progress(0, f"运行前检查已取消/过期：{reason} · 未入队、未启动分析")
        task_status_label.setText(f"运行前检查已取消/过期：{reason}；请按当前项目状态重新检查。")
        log(f"丢弃运行前检查 generation：{reason}")
        return not future.done()

    def recovery_diff_text(saved_payload: dict[str, Any], previous_payload: dict[str, Any] | None,
                           saved_revision: int) -> str:
        def display(value: Any) -> str:
            try:
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            except (TypeError, ValueError):
                rendered = str(value)
            return rendered if len(rendered) <= 80 else rendered[:77] + "…"

        def field_diffs(left: Any, right: Any, prefix: str = "") -> list[str]:
            if isinstance(left, dict) and isinstance(right, dict):
                result: list[str] = []
                for key in sorted(set(left) | set(right)):
                    path = f"{prefix}.{key}" if prefix else str(key)
                    result.extend(field_diffs(left.get(key), right.get(key), path))
                return result
            if left == right:
                return []
            return [f"{prefix}: {display(left)} → {display(right)}"]

        lines = [f"将恢复草稿 revision {saved_revision}。"]
        previous = previous_payload or {}
        prior_config = previous.get("config") if isinstance(previous.get("config"), dict) else {}
        saved_config = saved_payload.get("config") if isinstance(saved_payload.get("config"), dict) else {}
        prior_fields = field_diffs(prior_config, saved_config)
        lines.append(f"相对上一有效草稿配置变更 {len(prior_fields)} 项：")
        lines.extend("  · " + item for item in prior_fields[:12])
        if len(prior_fields) > 12:
            lines.append(f"  · 另有 {len(prior_fields) - 12} 项配置变化")

        previous_markers = previous.get("markers", []) if isinstance(previous.get("markers", []), list) else []
        saved_markers = saved_payload.get("markers", []) if isinstance(saved_payload.get("markers", []), list) else []
        previous_by_id = {str(item.get("marker_id")): item for item in previous_markers if isinstance(item, dict)}
        saved_by_id = {str(item.get("marker_id")): item for item in saved_markers if isinstance(item, dict)}
        added = sorted(set(saved_by_id) - set(previous_by_id))
        removed = sorted(set(previous_by_id) - set(saved_by_id))
        edited = sorted(key for key in set(previous_by_id) & set(saved_by_id)
                        if previous_by_id[key] != saved_by_id[key])
        lines.append(f"marker：{len(previous_markers)} → {len(saved_markers)} 条；新增 {len(added)}、删除 {len(removed)}、编辑 {len(edited)}。")
        for marker_id in (added + removed + edited)[:8]:
            item = saved_by_id.get(marker_id) or previous_by_id.get(marker_id) or {}
            action = "新增" if marker_id in added else "删除" if marker_id in removed else "编辑"
            lines.append(f"  · {action} {item.get('source_event_id', marker_id)} · {item.get('recording_id', '')}/{item.get('channel_scope', item.get('channel', ''))}")

        prior_qc = prior_config.get("qc", {}).get("annotations", []) if isinstance(prior_config.get("qc"), dict) else []
        saved_qc = saved_config.get("qc", {}).get("annotations", []) if isinstance(saved_config.get("qc"), dict) else []
        prior_manual = prior_config.get("input", {}).get("manual_qc") if isinstance(prior_config.get("input"), dict) else None
        saved_manual = saved_config.get("input", {}).get("manual_qc") if isinstance(saved_config.get("input"), dict) else None
        lines.append(f"QC：内嵌审计标注 {len(prior_qc) if isinstance(prior_qc, list) else '未知'} → {len(saved_qc) if isinstance(saved_qc, list) else '未知'} 条；manual_qc 来源 {display(prior_manual)} → {display(saved_manual)}。")

        current_diffs = field_diffs(state.get("config", {}), saved_config)
        lines.append(f"当前工作配置与恢复草稿字段差异 {len(current_diffs)} 项：")
        lines.extend("  · " + item for item in current_diffs[:8])
        if len(current_diffs) > 8:
            lines.append(f"  · 另有 {len(current_diffs) - 8} 项差异")
        lines.append("选择“是”恢复上述草稿；选择“否”保留当前工作配置。")
        return "\n".join(lines)

    def open_project_at(path: str, *, ask_recovery: bool = True) -> bool:
        if state.get("run_future") is not None and not state["run_future"].done():
            QMessageBox.warning(window, "任务正在运行", "请等当前任务结束后再切换项目。")
            return False
        preflight_future = state.get("preflight_future")
        if preflight_future is not None:
            still_running = invalidate_preflight("切换项目")
            if still_running:
                QMessageBox.information(window, "预检正在取消", "已请求取消当前项目的运行前检查。待检查线程结束后，请再次切换项目；旧报告不会用于新项目。")
                return False
        current_project = state.get("project")
        target_path = Path(path).expanduser().resolve()
        if current_project and Path(current_project["path"]).expanduser().resolve() != target_path:
            if autosave_timer.isActive() or state.get("autosave_dirty"):
                if persist_project_draft() is None:
                    QMessageBox.warning(window, "项目草稿未保存", "旧项目的自动保存尚未完成；为避免丢失编辑，未切换项目。请检查保存状态后重试。")
                    return False
        try:
            project_handle = open_project(target_path)
            candidates = load_recovery_candidates(project_handle)
            corrupt_rows = [item for item in candidates if item.get("valid") is False]
            if corrupt_rows:
                detail = ", ".join(f"r{item['revision']}:{item.get('error')}" for item in corrupt_rows)
                raise ValueError(f"project_snapshot_corruption:{detail}")
            latest = next((item for item in candidates if item.get("valid") is not False), None)
            snapshot = latest.get("payload", {}) if latest else {}
            saved_config = snapshot.get("config") if isinstance(snapshot, dict) else None
            restore = False
            if ask_recovery and isinstance(saved_config, dict):
                previous_candidate = next((item for item in candidates
                                           if item.get("valid") is True and
                                           item.get("revision", latest["revision"]) < latest["revision"]), None)
                answer = QMessageBox.question(
                    window, "发现可恢复草稿",
                    recovery_diff_text(snapshot,
                                       previous_candidate.get("payload") if previous_candidate else None,
                                       latest["revision"]),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes)
                restore = answer == QMessageBox.StandardButton.Yes
            if restore:
                try:
                    state["config"] = resolve_config(saved_config)
                except Exception as error:
                    QMessageBox.warning(window, "草稿无法恢复", _format_error(error))
                    restore = False
            old_session = state.get("runtime_session")
            if old_session:
                old_session["flush"]()
                old_session["close"]()
            state["run_queue"] = queue.Queue()
            state["runtime_events"] = []
            state["project"] = project_handle
            state["config"].setdefault("workspace", {})["project_directory"] = str(Path(path).expanduser().resolve())
            runtime_config = state["config"].get("runtime", {})
            session = create_runtime_session(
                project_handle["path"], log_level=runtime_config.get("log_level", "INFO"),
                log_max_bytes=runtime_config.get("log_max_bytes", 10485760),
                log_backup_count=runtime_config.get("log_backup_count", 5),
                subscribers=[state["run_queue"].put], install_hooks=True,
                project_id=project_handle["project_id"])
            try:
                state["runtime_history"] = load_eta_history({"path": project_handle["path"]}, limit=20000)
            except Exception:
                state["runtime_history"] = []
            state["runtime_session"] = session
            try:
                reconciliation = reconcile_interrupted_tasks(project_handle, _live_task_probe)
                state["task_reconciliation"] = reconciliation
            except Exception as error:
                state["task_reconciliation"] = {"status": "error", "message": _format_error(error)}
            project_directory_line.setText(project_handle["path"])
            refresh_bound_controls()
            if not restore:
                state["autosave_dirty"] = True
            state["autosave_state"] = f"项目已打开 · {project_handle['path']} · revision {project_handle.get('project_revision', 0)}"
            project_autosave_status.setText(state["autosave_state"])
            if restore:
                schedule_project_autosave()
                log(f"已恢复项目草稿 revision {latest['revision']}；任务队列不会自动执行。")
            else:
                persist_project_draft(kind="project")
            if state.get("refresh_markers"):
                state["refresh_markers"]()
            if state.get("refresh_tasks"):
                state["refresh_tasks"]()
            refresh_compute_guard()
            set_status(f"项目已打开 · {Path(path).name} · 队列需显式启动")
            log(f"已打开项目：{project_handle['path']} · 恢复状态 {state['task_reconciliation'].get('status', 'not_checked')}")
            return True
        except Exception as error:
            backup = Path(path).expanduser().resolve() / ".eda" / "last-good-draft.json"
            detail = _format_error(error)
            if backup.is_file():
                try:
                    exported = load_last_good_draft_export(path)
                    answer = QMessageBox.question(
                        window, "项目状态损坏",
                        f"{detail}\n\n检测到可独立校验的完整草稿导出（revision {exported['project_revision']}）。"
                        f"数据库不会被覆盖。是否把草稿恢复到一个新项目目录？",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No)
                    if answer == QMessageBox.StandardButton.Yes:
                        target = QFileDialog.getExistingDirectory(
                            window, "选择新的恢复项目目录（不能是损坏项目目录）",
                            str(Path(path).expanduser().resolve().parent / (Path(path).name + "-recovered")))
                        if target:
                            restored = restore_last_good_draft_export(path, target)
                            recovered_snapshot = read_project_snapshot(restored)
                            if isinstance(recovered_snapshot.get("config"), dict):
                                state["config"] = resolve_config(recovered_snapshot["config"])
                            refresh_bound_controls()
                            return open_project_at(restored["path"], ask_recovery=False)
                        detail += f"\n恢复未执行；备份仍可读取：{backup}。"
                    else:
                        detail += f"\n恢复未执行；备份仍可读取：{backup}。"
                except Exception as backup_error:
                    detail += f"\n独立草稿备份不可恢复：{_format_error(backup_error)}"
            QMessageBox.critical(window, "项目无法打开", detail)
            project_autosave_status.setText(f"打开失败 · {detail}")
            return False

    def choose_project_directory() -> None:
        directory = QFileDialog.getExistingDirectory(window, "选择本地项目目录",
                                                      project_directory_line.text() or str(Path.cwd()))
        if directory:
            open_project_at(directory)

    project_open_button.clicked.connect(choose_project_directory)
    project_directory_line.editingFinished.connect(
        lambda: open_project_at(project_directory_line.text().strip()) if project_directory_line.text().strip() else None)

    demo_group = QGroupBox("工程演示数据")
    demo_layout = QVBoxLayout(demo_group)
    demo_note = QLabel("创建 4 条确定性 SYNTHETIC_ONLY 信号（4 Hz × 600 s）及事件、条件、配对与同步锚点。数据只验证软件路径，不代表设备或科学适用性。演示目录必须为空，绝不覆盖已有文件。")
    demo_note.setWordWrap(True)
    demo_button = QPushButton("在输出目录创建并载入合成演示")
    demo_button.setObjectName("primary")
    demo_layout.addWidget(demo_note)
    demo_layout.addWidget(demo_button, alignment=Qt.AlignmentFlag.AlignLeft)
    put_content_before_stretch(project_layout, demo_group)

    def load_demo() -> None:
        try:
            from ..demo import create_demo
            base = Path(output_line.text() or "results").expanduser().resolve()
            destination = base / "synthetic_demo"
            demo = create_demo(destination)
            candidate = resolve_run_config(demo["single_config"])
            if apply_config(candidate, source="SYNTHETIC_ONLY 演示配置"):
                config_path.setText(demo["single_config"])
                state["last_loaded_path"] = demo["single_config"]
                log(f"已创建并载入合成演示目录：{destination}")
                set_status("已载入合成演示 · 工程数据，不是科学验证")
        except Exception as error:
            QMessageBox.warning(window, "演示数据未创建", _format_error(error))
            log(f"合成演示不可用：{_format_error(error)}")

    demo_button.clicked.connect(load_demo)
    caps = get_capabilities(_resolved_output_directory(state["config"]))
    state["capabilities"] = caps
    caps_label = QLabel(_capability_summary(caps))
    caps_label.setWordWrap(True)
    caps_label.setObjectName("subtle")
    caps_label.hide()
    widgets["capabilities_label"] = caps_label
    environment_button = QPushButton("环境监测")
    environment_button.setObjectName("environmentCheckButton")
    environment_button.setToolTip("手动刷新本机依赖、GPU 与方法状态；可从弹窗查看和安装缺少的运行依赖。")
    environment_button.setMinimumHeight(34)
    widgets["environment_check_button"] = environment_button

    compute_group = QGroupBox("计算资源")
    compute_form = QFormLayout(compute_group)
    compute_workers = QSpinBox()
    compute_workers.setObjectName("computeWorkers")
    compute_workers.setRange(1, 32)
    compute_workers.setSuffix(" 个")
    compute_workers.setValue(int(config["compute"]["workers"]))
    compute_workers.setToolTip("同时处理的记录数；运行时按待处理记录数限制实际并行数（1–32）。")
    compute_backend = QComboBox()
    compute_backend.setObjectName("computeBackend")
    compute_backend.addItem("CPU", "cpu")
    compute_backend.addItem("GPU · 仅 bateman_sparse", "gpu")
    compute_backend.setCurrentIndex(max(0, compute_backend.findData(config["compute"]["backend"])))
    compute_backend.setToolTip("GPU 后端只支持 bateman_sparse；选择其他方法会被界面阻止，不会自动换算法。")
    compute_status_note = QLabel("并行数会写入当前计算配置。GPU 仅支持 bateman_sparse；依赖状态可在“环境监测”中手动刷新。")
    compute_status_note.setObjectName("subtle")
    compute_status_note.setWordWrap(True)
    compute_form.addRow("并行记录数 (workers)", compute_workers)
    compute_form.addRow("计算后端", compute_backend)
    compute_form.addRow("说明", compute_status_note)
    widgets.update({"compute_workers": compute_workers, "compute_backend": compute_backend,
                    "compute_status_note": compute_status_note})

    def update_compute_workers(value: int) -> None:
        if state["busy"]:
            return
        state["config"]["compute"]["workers"] = int(value)
        update_config_editor()

    def update_compute_backend(_index: int = -1) -> None:
        if state["busy"]:
            return
        selected = compute_backend.currentData()
        if selected == "gpu" and state["config"]["decomposition"].get("method") != "bateman_sparse":
            compute_backend.blockSignals(True)
            compute_backend.setCurrentIndex(compute_backend.findData(state["config"]["compute"]["backend"]))
            compute_backend.blockSignals(False)
            QMessageBox.warning(window, "GPU 后端不可用于当前方法",
                                "GPU 仅支持 bateman_sparse。请先选择 bateman_sparse，再设置 GPU；不会自动切换分解方法。")
            return
        state["config"]["compute"]["backend"] = str(selected)
        refresh_compute_guard()
        update_config_editor()

    def refresh_compute_guard() -> None:
        error = _compute_compatibility_error(state["config"])
        state["compute_configuration_error"] = error
        if error:
            compute_status_note.setText(error)
            compute_status_note.setStyleSheet("color: #9A3D39; font-weight: 600;")
        else:
            compute_status_note.setText(
                "并行数会写入当前计算配置。GPU 仅支持 bateman_sparse；依赖状态可在“环境监测”中手动刷新。")
            compute_status_note.setStyleSheet("")
        allowed = error is None and state.get("project") is not None and not state.get("busy", False)
        for key in ("preflight", "trial", "batch", "start_queue", "add_task_button",
                    "add_single_task_action", "add_batch_task_action", "task_single_action",
                    "task_batch_action", "preflight_action"):
            action = widgets.get(key)
            if action is not None:
                action.setEnabled(allowed)

    compute_workers.valueChanged.connect(update_compute_workers)
    compute_backend.currentIndexChanged.connect(update_compute_backend)
    refresh_compute_guard()

    def open_environment_dialog() -> None:
        dialog = state.get("environment_dialog")
        if dialog is not None and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return
        from .environment_dialog import EnvironmentDialog

        config_snapshot = copy.deepcopy(state["config"])

        def environment_updated(snapshot: dict[str, Any]) -> None:
            state["environment_status"] = copy.deepcopy(snapshot)
            refreshed_caps = snapshot.get("capabilities")
            if isinstance(refreshed_caps, dict) and "probe_error" not in refreshed_caps:
                state["capabilities"] = refreshed_caps
                caps_label.setText(_capability_summary(refreshed_caps))
            gpu = snapshot.get("gpu", {}) if isinstance(snapshot, dict) else {}
            readiness = gpu.get("readiness", gpu.get("status", "已探测")) if isinstance(gpu, dict) else "已探测"
            if state.get("compute_configuration_error"):
                refresh_compute_guard()
            else:
                compute_status_note.setText(
                    f"GPU 环境：{readiness}。GPU 仅支持 bateman_sparse，不会自动切换算法。")

        dialog = EnvironmentDialog(
            window,
            capability_provider=lambda: get_capabilities(_resolved_output_directory(config_snapshot)),
            on_status_changed=environment_updated)
        state["environment_dialog"] = dialog
        dialog.finished.connect(lambda *_: state.update(environment_dialog=None))
        dialog.open()

    environment_button.clicked.connect(open_environment_dialog)
    put_content_before_stretch(project_layout, environment_button)
    put_content_before_stretch(project_layout, compute_group)

    # Retain hidden mirrors as compatibility handles for older integrations.
    # They are deliberately not placed in the project page or used as config input.
    json_toggle = QPushButton("高级 JSON 编辑已移除")
    json_toggle.hide()
    widgets["json_toggle"] = json_toggle
    json_group = QGroupBox("高级 JSON 编辑已移除")
    json_group.hide()
    widgets["json_group"] = json_group
    config_json = QPlainTextEdit()
    config_json.setObjectName("configJson")
    config_json.setPlainText(_safe_config_text(config))
    config_json.hide()
    widgets["config_json"] = config_json
    apply_json_button = QPushButton("校验并应用 JSON")
    apply_json_button.hide()
    widgets["apply_json"] = apply_json_button
    apply_json_button.clicked.connect(apply_json_editor)

    # ── 九类 core_io 模板与真实列映射 ─────────────────────────────────────
    import_page, import_layout = add_page("导入与映射", "下载一个工作簿，填写信号和研究信息后导入检查完整性。电导单位需要明确选择。")
    template_group = QGroupBox("单工作簿输入")
    template_layout = QVBoxLayout(template_group)
    template_info = QLabel("使用一个 Excel 工作簿查看全部九类表格。导入时按工作表、字段和行列检查必填数据；运行前检查仍会执行完整跨表校验。")
    template_info.setWordWrap(True)
    template_info.setObjectName("subtle")
    template_layout.addWidget(template_info)
    workbook_toolbar = QHBoxLayout()
    download_workbook_button = QPushButton("下载 Excel 工作簿")
    download_workbook_button.setObjectName("primary")
    import_workbook_button = QPushButton("导入并校验工作簿…")
    workbook_toolbar.addWidget(download_workbook_button)
    workbook_toolbar.addWidget(import_workbook_button)
    workbook_toolbar.addStretch(1)
    template_layout.addLayout(workbook_toolbar)
    workbook_status_table = QTableWidget(0, 4)
    workbook_status_table.setObjectName("workbookStatusTable")
    workbook_status_table.setHorizontalHeaderLabels(["表名", "要求", "填写状态", "缺失项 / 错误"])
    workbook_status_table.verticalHeader().setVisible(False)
    workbook_status_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    workbook_status_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    workbook_status_table.setAlternatingRowColors(True)
    workbook_status_table.setMaximumHeight(290)
    workbook_status_table.horizontalHeader().setStretchLastSection(True)
    template_layout.addWidget(workbook_status_table)
    widgets.update({"download_workbook": download_workbook_button,
                    "import_workbook": import_workbook_button,
                    "workbook_status_table": workbook_status_table})
    template_container = QWidget()
    template_grid = QGridLayout(template_container)
    template_grid.setContentsMargins(0, 0, 0, 0)
    template_grid.setHorizontalSpacing(9)
    template_grid.setVerticalSpacing(7)
    template_grid.addWidget(QLabel("类型"), 0, 0)
    template_grid.addWidget(QLabel("说明"), 0, 1)
    template_grid.addWidget(QLabel(""), 0, 2)
    template_grid.addWidget(QLabel(""), 0, 3)
    template_grid.addWidget(QLabel(""), 0, 4)
    template_meta = list_templates()
    widgets["template_buttons"] = state["template_buttons"]
    for row_index, item in enumerate(template_meta, start=1):
        template_name = item["name"]
        template_grid.addWidget(QLabel(item["display_name"]), row_index, 0)
        description = QLabel(item.get("description", ""))
        description.setObjectName("subtle")
        description.setWordWrap(True)
        template_grid.addWidget(description, row_index, 1)
        button_set = {}
        for column, (action, label) in enumerate((("view", "查看"), ("download", "下载"), ("import", "导入")), start=2):
            button = QPushButton(label)
            template_grid.addWidget(button, row_index, column)
            button_set[action] = button
        state["template_buttons"][template_name] = button_set

        def view_template(_checked: bool = False, name: str = template_name,
                          metadata: dict[str, Any] = item) -> None:
            from PySide6.QtWidgets import QDialog, QDialogButtonBox
            try:
                frame = read_template(name, "example")
                schema = read_template(name, "schema")
                dialog = QDialog(window)
                dialog.setWindowTitle(f"模板预览 · {metadata['display_name']} · {metadata.get('example_label', 'SYNTHETIC_ONLY')}")
                dialog.resize(900, 620)
                layout = QVBoxLayout(dialog)
                label = QLabel(f"{metadata.get('description', '')}\n示例文件标记：{metadata.get('example_label', 'SYNTHETIC_ONLY')}；字段 schema 有 {len(schema.get('fields', []))} 项。")
                label.setWordWrap(True)
                layout.addWidget(label)
                layout.addWidget(_table_widget(frame, parent=dialog), 1)
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
                buttons.rejected.connect(dialog.reject)
                buttons.accepted.connect(dialog.accept)
                layout.addWidget(buttons)
                dialog.exec()
            except Exception as error:
                QMessageBox.critical(window, "模板读取失败", _format_error(error))

        def download_template(_checked: bool = False, name: str = template_name,
                              metadata: dict[str, Any] = item) -> None:
            target, _ = QFileDialog.getSaveFileName(window, f"下载{metadata['display_name']}模板",
                                                   metadata.get("template_file", f"{name}.csv"),
                                                   "CSV (*.csv);;JSON (*.json)")
            if not target:
                return
            try:
                exported = export_template(name, target, "template")
                log(f"已导出 {name} 模板：{exported}")
                set_status(f"模板已导出：{Path(exported).name}")
            except Exception as error:
                QMessageBox.critical(window, "模板导出失败", _format_error(error))

        def import_template(_checked: bool = False, name: str = template_name,
                            metadata: dict[str, Any] = item) -> None:
            filename, _ = QFileDialog.getOpenFileName(window, f"导入{metadata['display_name']}表",
                                                       "", "CSV (*.csv *.tsv);;所有文件 (*.*)")
            if not filename:
                return
            try:
                result = preview_csv(filename, max_rows=60, options=state["config"])
                if name == "signals":
                    state["config"]["input"]["manifest"] = None
                    state["config"]["input"]["signals"] = [filename]
                elif name == "input_manifest":
                    state["config"]["input"]["manifest"] = filename
                elif name in INPUT_KEYS:
                    state["config"]["input"][name] = filename
                state["config"]["input"]["base_directory"] = str(Path(filename).resolve().parent)
                update_config_editor()
                widgets["preview_title"].setText(f"{metadata['display_name']} · {result['total_rows']} 行 · 预览 {len(result['preview'])} 行 · {Path(filename).name}")
                widgets["preview_host"].setWidget(_table_widget(result["preview"], parent=widgets["preview_host"]))
                log(f"已绑定 {name} 输入，core_io 预览 {result['total_rows']} 行；完整校验将在 pipeline 加载时执行。")
                set_status(f"已绑定 {metadata['display_name']} · {result['total_rows']} 行")
            except Exception as error:
                QMessageBox.critical(window, "输入预览失败", _format_error(error))

        button_set["view"].clicked.connect(view_template)
        button_set["download"].clicked.connect(download_template)
        button_set["import"].clicked.connect(import_template)
    template_scroll = QScrollArea()
    template_scroll.setWidgetResizable(True)
    template_scroll.setMinimumHeight(340)
    template_scroll.setWidget(template_container)
    template_scroll.hide()
    template_layout.addWidget(template_scroll)
    put_content_before_stretch(import_layout, template_group)

    preview_group = QGroupBox("有界预览")
    preview_layout = QVBoxLayout(preview_group)
    preview_title = QLabel("尚未绑定输入。预览显示总数据行数及最多 60 行。")
    preview_title.setObjectName("subtle")
    preview_host = QScrollArea()
    preview_host.setWidgetResizable(True)
    preview_host.setMinimumHeight(170)
    preview_host.setWidget(QWidget())
    preview_layout.addWidget(preview_title)
    preview_layout.addWidget(preview_host)
    widgets.update({"preview_title": preview_title, "preview_host": preview_host})

    def refresh_workbook_status_table(statuses: dict[str, Any] | None = None) -> None:
        from .workbook import table_requirements

        table = widgets["workbook_status_table"]
        requirements = table_requirements(state["config"])
        status_map = statuses or {}
        table.setRowCount(len(template_meta))
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["表名", "要求", "填写状态", "缺失项 / 错误"])
        table_name_map = {item["name"]: item for item in template_meta}
        for row, table_id in enumerate(table_name_map):
            template = table_name_map[table_id]
            requirement = requirements.get(table_id, {})
            detail = status_map.get(table_id, {})
            input_key = requirement.get("input_key", table_id)
            bound_value = state["config"]["input"].get(input_key)
            bound = bool(bound_value)
            required = bool(requirement.get("required_by_config", detail.get("required_by_config", False)))
            if table_id in ("signals", "input_manifest"):
                signal_source_missing = not bool(state["config"]["input"].get("signals") or
                                                 state["config"]["input"].get("manifest"))
                required = signal_source_missing
                requirement_text = "至少一种信号来源必需" if signal_source_missing else "信号来源（二选一）"
            else:
                requirement_text = "必需" if required else "可选"
            raw_status = detail.get("status")
            labels = {"complete": "已填写", "invalid": "有错误", "missing_required": "缺少必填字段",
                      "optional_empty": "可选，未填写", "not_provided": "未提供"}
            if raw_status:
                completion = labels.get(str(raw_status), str(raw_status))
            elif bound:
                completion = "已绑定 · 待预检"
            else:
                completion = "缺失" if required else "未填写"
            missing = [str(value) for value in detail.get("missing_fields", [])]
            errors = detail.get("errors", [])
            diagnostics = []
            for error in errors:
                location = ""
                if error.get("sheet") or error.get("table"):
                    location += str(error.get("sheet", error.get("table")))
                if error.get("row") is not None:
                    location += f" 第 {error['row']} 行"
                if error.get("column"):
                    location += f"「{error['column']}」"
                diagnostics.append((location + "：" if location else "") + str(error.get("message", error.get("code", "导入错误"))))
            missing_text = "、".join(missing + diagnostics)
            if not missing_text and raw_status == "complete":
                missing_text = "无"
            if not missing_text and not bound and required:
                required_fields = requirement.get("required_fields", [])
                missing_text = "尚未提供工作表" + (f"；需填：{'、'.join(map(str, required_fields))}" if required_fields else "")
            values = (detail.get("sheet_name", requirement.get("sheet_name", template["display_name"])),
                      requirement_text, completion, missing_text)
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if column == 3 and (diagnostics or missing):
                    cell.setToolTip("\n".join(diagnostics + ([f"缺少字段：{'、'.join(missing)}"] if missing else [])))
                table.setItem(row, column, cell)
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 140)
        table.setColumnWidth(1, 190)
        table.setColumnWidth(2, 150)
        table.horizontalHeader().setStretchLastSection(True)

    def download_workbook() -> None:
        from .workbook import export_workbook

        target, _ = QFileDialog.getSaveFileName(window, "下载 Excel 工作簿", "eda_input_workbook.xlsx",
                                                "Excel 工作簿 (*.xlsx)")
        if not target:
            return
        try:
            exported = export_workbook(target)
            set_status(f"工作簿已导出：{Path(exported).name}")
            log(f"已导出九表工作簿：{exported}")
        except Exception as error:
            QMessageBox.critical(window, "工作簿导出失败", _format_error(error))

    def import_workbook_file() -> None:
        from .workbook import import_workbook

        filename, _ = QFileDialog.getOpenFileName(window, "导入 Excel 工作簿", "", "Excel 工作簿 (*.xlsx)")
        if not filename:
            return
        destination = state["project"]["path"] if state.get("project") else str(Path(filename).resolve().parent)
        try:
            result = import_workbook(filename, destination, config=state["config"])
            candidate = copy.deepcopy(state["config"])
            candidate["input"] = copy.deepcopy(result["bindings"])
            if result.get("statistics_designs") is not None:
                candidate["statistics"]["designs"] = copy.deepcopy(result["statistics_designs"])
            if not apply_config(candidate, source=f"工作簿 {Path(filename).name}"):
                return
            state["workbook_statuses"] = copy.deepcopy(result["statuses"])
            refresh_workbook_status_table(state["workbook_statuses"])
            invalid = sum(1 for item in result["statuses"].values() if item.get("status") != "complete")
            set_status(f"工作簿已导入 · {len(result['statuses']) - invalid}/{len(result['statuses'])} 张表完整")
            log(f"已导入工作簿并检查 {len(result['statuses'])} 张表：{filename}")
            error_count = sum(len(item.get("errors", [])) for item in result["statuses"].values())
            if error_count:
                QMessageBox.warning(window, "工作簿存在待修正内容",
                                    f"发现 {error_count} 项工作表错误。请查看表格中的行、列和缺失字段，再运行前完整检查。")
        except Exception as error:
            QMessageBox.critical(window, "工作簿导入失败", _format_error(error))

    download_workbook_button.clicked.connect(download_workbook)
    import_workbook_button.clicked.connect(import_workbook_file)
    state["workbook_statuses"] = {}
    refresh_workbook_status_table()
    put_content_before_stretch(import_layout, preview_group)

    mapping_group = QGroupBox("信号列映射与实际读取预览")
    mapping_form = QFormLayout(mapping_group)
    mapping_file = QLineEdit("")
    mapping_file.setPlaceholderText("选择一条信号 CSV 做映射预览")
    map_browse = QPushButton("选择信号文件…")
    map_file_holder = QWidget()
    map_file_row = QHBoxLayout(map_file_holder)
    map_file_row.setContentsMargins(0, 0, 0, 0)
    map_file_row.addWidget(mapping_file, 1)
    map_file_row.addWidget(map_browse)
    mapping_form.addRow("CSV 文件", map_file_holder)
    column_controls: dict[str, QComboBox] = {}
    for key, caption in (("time_s", "来源时间列"), ("eda_uS", "EDA 列"), ("recording_id", "记录编号列"),
                         ("subject_id", "参与者编号列"), ("session_id", "会话编号列"), ("channel", "通道列"),
                         ("clock_domain", "时钟域列")):
        box = QComboBox()
        box.setEditable(False)
        box.addItem("— 不映射 —", None)
        if key in ("time_s", "eda_uS"):
            box.setCurrentIndex(0)
        mapping_form.addRow(caption, box)
        column_controls[key] = box
    unit_box = QComboBox()
    unit_box.addItem("未声明（未知，需显式选择）", None)
    for unit in ("uS", "mS", "S", "ohm", "kohm", "Mohm"):
        unit_box.addItem(unit, unit)
    unit_box.setCurrentIndex(0)
    widgets["mapping_unit"] = unit_box
    mapping_form.addRow("EDA 单位", unit_box)
    fs_edit = QLineEdit()
    fs_edit.setPlaceholderText("仅当无可信时间列时填写；例如 4 Hz")
    mapping_form.addRow("采样率 Hz", fs_edit)
    clock_edit = QLineEdit("recording_local")
    mapping_form.addRow("时钟域常量", clock_edit)
    widgets.update({"mapping_rate": fs_edit, "mapping_clock": clock_edit, "mapping_columns": column_controls})
    mapping_preview_button = QPushButton("用 core_io 读取并预览")
    mapping_preview_button.setObjectName("primary")
    mapping_form.addRow("", mapping_preview_button)
    mapping_result = QLabel("列选择只展示真实 CSV 表头；EDA 单位不会自动猜测。")
    mapping_result.setWordWrap(True)
    mapping_result.setObjectName("subtle")
    mapping_form.addRow("读取诊断", mapping_result)
    widgets.update({"mapping_file": mapping_file, "mapping_preview_button": mapping_preview_button,
                    "mapping_result": mapping_result})
    put_content_before_stretch(import_layout, mapping_group)

    def populate_mapping_columns() -> None:
        filename = mapping_file.text().strip()
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(window, "选择信号 CSV", "", "CSV (*.csv *.tsv);;所有文件 (*.*)")
            if not filename:
                return
            mapping_file.setText(filename)
        invalidate_mapping_preview()
        try:
            preview = preview_csv(filename, max_rows=2, options=state["config"])
            headers = list(preview["preview"].columns)
            for key, box in column_controls.items():
                box.blockSignals(True)
                box.clear()
                box.addItem("— 不映射 —", None)
                for header in headers:
                    # currentData() is consumed below by core_io. Keep the visible
                    # CSV header as its userData instead of creating text-only items.
                    box.addItem(str(header), str(header))
                configured = state["config"]["mapping"].get("columns", {}).get(key)
                expected = configured if configured in headers else key if key in headers else None
                if expected:
                    index = box.findData(expected)
                    if index >= 0:
                        box.setCurrentIndex(index)
                box.blockSignals(False)
            configured_unit = state["config"]["mapping"].get("eda_unit")
            unit_box.blockSignals(True)
            unit_box.setCurrentIndex(max(0, unit_box.findData(configured_unit)))
            unit_box.blockSignals(False)
            configured_rate = state["config"]["mapping"].get("sampling_rate_hz")
            fs_edit.blockSignals(True)
            fs_edit.setText("" if configured_rate is None else str(configured_rate))
            fs_edit.blockSignals(False)
            clock_edit.blockSignals(True)
            clock_edit.setText(str(state["config"]["mapping"].get("clock_domain", "recording_local")))
            clock_edit.blockSignals(False)
            widgets["preview_title"].setText(f"映射文件 {Path(filename).name} · 总计 {preview['total_rows']} 行 · 表头来自 core_io.preview_csv")
            widgets["preview_host"].setWidget(_table_widget(preview["preview"], parent=widgets["preview_host"]))
            log(f"读取 {len(headers)} 个真实列名：{', '.join(headers)}")
        except Exception as error:
            QMessageBox.warning(window, "无法读取列名", _format_error(error))

    map_browse.clicked.connect(populate_mapping_columns)
    mapping_file.editingFinished.connect(populate_mapping_columns)

    def current_mapping_snapshot() -> dict[str, Any]:
        """复制当前文件、列映射控件与整份配置，供异步读取隔离使用。"""
        raw_rate = fs_edit.text().strip()
        try:
            rate: Any = float(raw_rate) if raw_rate else None
        except ValueError:
            rate = {"invalid_sampling_rate_text": raw_rate}
        columns = {}
        for key, box in column_controls.items():
            value = box.currentData()
            if value is None and box.currentIndex() > 0:
                value = box.currentText()
            if value:
                columns[key] = str(value)
        mapping = {"columns": columns, "eda_unit": unit_box.currentData(),
                   "sampling_rate_hz": rate, "time_unit": "s",
                   "clock_domain": clock_edit.text().strip() or "recording_local"}
        return {"filename": mapping_file.text().strip(), "mapping": mapping,
                "config": copy.deepcopy(state["config"])}

    def mapping_snapshot_is_current(snapshot: dict[str, Any]) -> bool:
        return snapshot == current_mapping_snapshot()

    # Any file or mapping control edit makes in-flight preview results stale.
    mapping_file.textChanged.connect(lambda *_: invalidate_mapping_preview())
    for box in column_controls.values():
        box.currentIndexChanged.connect(lambda *_: invalidate_mapping_preview())
    unit_box.currentIndexChanged.connect(lambda *_: invalidate_mapping_preview())
    fs_edit.textChanged.connect(lambda *_: invalidate_mapping_preview())
    clock_edit.textChanged.connect(lambda *_: invalidate_mapping_preview())

    def run_mapping_preview() -> None:
        snapshot = current_mapping_snapshot()
        filename = snapshot["filename"]
        if not filename:
            QMessageBox.information(window, "选择信号文件", "请先选择 CSV 文件。")
            return
        columns = snapshot["mapping"]["columns"]
        if "eda_uS" not in columns:
            QMessageBox.warning(window, "缺少 EDA 映射", "请明确选择 EDA 列。")
            return
        mapping = snapshot["mapping"]
        if isinstance(mapping["sampling_rate_hz"], dict):
            QMessageBox.warning(window, "采样率无效", "采样率必须是正的有限数值 Hz。")
            return
        preview_id = invalidate_mapping_preview()
        state["mapping_preview_snapshot"] = copy.deepcopy(snapshot)
        set_status("core_io 正在读取信号并校验映射…")
        future = state["io_executor"].submit(read_signal_csv, filename, copy.deepcopy(mapping),
                                              copy.deepcopy(snapshot["config"]))
        state["mapping_future"] = (preview_id, future, copy.deepcopy(snapshot))
        log(f"开始以显式映射读取 {Path(filename).name}")

    mapping_preview_button.clicked.connect(run_mapping_preview)

    # ── QC 原始信号、选择区间与追加审计 ────────────────────────────────────
    qc_page, qc_layout = add_page("质控", "质控来源于真实输入。拖动曲线可选择 [start,end) 时间段；人工决定会追加到 qc.annotations，并随下一次运行冻结。")
    qc_top = QHBoxLayout()
    qc_load_button = QPushButton("按当前输入载入记录")
    qc_load_button.setObjectName("primary")
    qc_record_combo = QComboBox()
    qc_record_combo.setMinimumWidth(280)
    qc_summary_label = QLabel("尚未载入记录")
    qc_summary_label.setObjectName("subtle")
    qc_top.addWidget(qc_load_button)
    qc_top.addWidget(QLabel("记录"))
    qc_top.addWidget(qc_record_combo)
    qc_top.addWidget(qc_summary_label, 1)
    qc_layout.insertLayout(max(0, qc_layout.count() - 1), qc_top)

    qc_plot_group = QGroupBox("原始 EDA 与当前 QC")
    qc_plot_layout = QVBoxLayout(qc_plot_group)
    qc_plot_host = QWidget()
    qc_plot_host_layout = QVBoxLayout(qc_plot_host)
    qc_plot_host_layout.setContentsMargins(0, 0, 0, 0)
    qc_plot_layout.addWidget(qc_plot_host)
    put_content_before_stretch(qc_layout, qc_plot_group)

    annotation_group = QGroupBox("人工区间标注与审计")
    annotation_layout = QVBoxLayout(annotation_group)
    selection_form = QFormLayout()
    start_spin, end_spin = QDoubleSpinBox(), QDoubleSpinBox()
    for spin in (start_spin, end_spin):
        spin.setDecimals(4)
        spin.setRange(-1e9, 1e9)
        spin.setSuffix(" s")
    selection_form.addRow("起点（含）", start_spin)
    selection_form.addRow("终点（不含）", end_spin)
    operator_edit = QLineEdit(os.environ.get("USERNAME", "operator"))
    reason_edit = QLineEdit()
    reason_edit.setPlaceholderText("说明排除或恢复依据（必填）")
    selection_form.addRow("操作者", operator_edit)
    selection_form.addRow("原因", reason_edit)
    annotation_layout.addLayout(selection_form)
    annotation_buttons = QHBoxLayout()
    exclude_button, restore_button = QPushButton("排除区间"), QPushButton("恢复自动排除")
    undo_button, redo_button = QPushButton("撤销选中决定"), QPushButton("重做选中决定")
    annotation_buttons.addWidget(exclude_button)
    annotation_buttons.addWidget(restore_button)
    annotation_buttons.addWidget(undo_button)
    annotation_buttons.addWidget(redo_button)
    annotation_buttons.addStretch(1)
    annotation_layout.addLayout(annotation_buttons)
    qc_target_combo = QComboBox()
    qc_target_combo.setMinimumWidth(260)
    annotation_layout.addWidget(QLabel("可撤销/重做的决定"))
    annotation_layout.addWidget(qc_target_combo)
    audit_table = QTableWidget(0, 0)
    audit_table.setMinimumHeight(150)
    annotation_layout.addWidget(audit_table)
    put_content_before_stretch(qc_layout, annotation_group)
    widgets.update({"qc_load": qc_load_button, "qc_record": qc_record_combo, "qc_summary": qc_summary_label,
                    "qc_start": start_spin, "qc_end": end_spin, "qc_operator": operator_edit,
                    "qc_reason": reason_edit, "qc_exclude": exclude_button, "qc_restore": restore_button,
                    "qc_undo": undo_button, "qc_redo": redo_button, "qc_target": qc_target_combo,
                    "qc_audit": audit_table, "qc_plot_host": qc_plot_host})

    def current_qc_history() -> list[dict[str, Any]]:
        imported = state.get("associated", {}).get("manual_qc")
        if isinstance(imported, pd.DataFrame) and not imported.empty:
            return imported.to_dict(orient="records")
        return [dict(item) for item in state["config"]["qc"].get("annotations", [])]

    def qc_history_sources_conflict() -> bool:
        imported = state.get("associated", {}).get("manual_qc")
        return bool(state["config"]["qc"].get("annotations")) and isinstance(imported, pd.DataFrame) and not imported.empty

    def refresh_audit_table() -> None:
        history = current_qc_history()
        if not history:
            audit_table.setRowCount(0)
            audit_table.setColumnCount(0)
            qc_target_combo.clear()
            return
        columns = ["revision", "action", "recording_id", "start_s", "end_s", "reason", "operator", "timestamp", "target_annotation_id"]
        frame = pd.DataFrame(history)
        for col in columns:
            if col not in frame:
                frame[col] = ""
        audit_table.setRowCount(len(frame))
        audit_table.setColumnCount(len(columns))
        audit_table.setHorizontalHeaderLabels(columns)
        audit_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, values in enumerate(frame[columns].itertuples(index=False, name=None)):
            for col, value in enumerate(values):
                audit_table.setItem(row, col, QTableWidgetItem("" if pd.isna(value) else str(value)))
        audit_table.resizeColumnsToContents()
        qc_target_combo.clear()
        decisions: dict[str, dict[str, Any]] = {}
        undone: set[str] = set()
        for entry in history:
            action = str(entry.get("action", "")).casefold()
            target = str(entry.get("target_annotation_id", ""))
            if action in ("exclude", "restore"):
                decisions[str(entry.get("annotation_id"))] = entry
            elif action == "undo" and target in decisions:
                undone.add(target)
            elif action == "redo" and target in decisions:
                undone.discard(target)
        for annotation_id, entry in decisions.items():
            decision_state = "已撤销" if annotation_id in undone else "生效"
            qc_target_combo.addItem(f"{decision_state} · r{entry.get('revision')} · {entry.get('action')} · {entry.get('recording_id')} [{entry.get('start_s')}, {entry.get('end_s')})", annotation_id)

    def show_qc_record() -> None:
        index = qc_record_combo.currentData()
        records = state.get("records", [])
        if index is None or not (0 <= int(index) < len(records)):
            return
        record = records[int(index)]
        metadata = record.get("metadata", {})
        try:
            result = build_quality_mask(record, state["config"], current_qc_history())
            state["qc_result"] = result
        except Exception as error:
            qc_summary_label.setText(_format_error(error))
            return
        summary = result["summary"]
        qc_summary_label.setText(f"有效样本 {summary['eligible_sample_count']}/{summary['sample_count']} · 可用相邻时长 {summary['usable_duration_s']:.3f} s · QC 修订 {result['annotation_revision']}")
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.font_manager import FontProperties
            from matplotlib.figure import Figure
            from matplotlib.widgets import SpanSelector
            while qc_plot_host_layout.count():
                item = qc_plot_host_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            time = np.asarray(record["time_s"], dtype=float)
            signal = np.asarray(record["eda_uS"], dtype=float)
            stride = max(1, int(math.ceil(len(time) / 8000)))
            fig = Figure(figsize=(11, 3.5), constrained_layout=True)
            ax = fig.subplots()
            ax.plot(time[::stride], signal[::stride], color="#47728F", linewidth=0.8, label="原始 EDA · μS")
            eligible = np.asarray(result["eligible"], dtype=bool)
            bad = np.isfinite(signal) & ~eligible
            ax.scatter(time[bad][::stride], signal[bad][::stride], s=7, color="#C56D43", alpha=0.65, label="不可用 / 已排除")
            ax.set_xlabel("时间 (s)")
            ax.set_ylabel("电导 (μS)")
            ax.set_title(f"{metadata.get('recording_id', '')} · {metadata.get('subject_id', '')} · 原始信号与当前资格", loc="left")
            ax.grid(color="#D9E2EA", linewidth=0.5)
            ax.legend(frameon=False, ncol=2)
            cjk_path = next((path for path in _cjk_font_candidates() if path.is_file()), None)
            plot_font = FontProperties(fname=str(cjk_path)) if cjk_path else FontProperties(family=ui_family)
            for text_item in fig.findobj(match=lambda item: hasattr(item, "set_fontproperties")):
                try:
                    text_item.set_fontproperties(plot_font)
                except (AttributeError, TypeError):
                    pass
            canvas = FigureCanvasQTAgg(fig)

            def selected_span(left: float, right: float) -> None:
                if math.isfinite(left) and math.isfinite(right) and left < right:
                    start_spin.setValue(left)
                    end_spin.setValue(right)

            selector = SpanSelector(ax, selected_span, "horizontal", useblit=True, props={"facecolor": "#168A8A", "alpha": 0.22}, interactive=True)
            state["qc_span_selector"] = selector
            state["qc_canvas"] = canvas
            toolbar = NavigationToolbar2QT(canvas, window)
            qc_plot_host_layout.addWidget(toolbar)
            qc_plot_host_layout.addWidget(canvas)
            state["qc_toolbar"] = toolbar
            canvas.draw_idle()
        except Exception as error:
            log(f"质控图无法绘制：{_format_error(error)}")
        refresh_audit_table()

    def on_qc_records_loaded(records: list[dict[str, Any]], tables: dict[str, Any], report: dict[str, Any],
                             preview_id: int) -> None:
        state["records"], state["associated"] = records, tables
        state["qc_records_preview_id"] = preview_id
        state["qc_records_config_sha256"] = state.get("qc_load_config_sha256")
        qc_record_combo.clear()
        for index, record in enumerate(records):
            meta = record.get("metadata", {})
            qc_record_combo.addItem(f"{meta.get('recording_id', '')} · {meta.get('subject_id', '')} · {meta.get('channel', '')}", index)
        if records:
            show_qc_record()
            refresh_marker_record_choices()
            log(f"core_io 已载入 {len(records)} 条记录作 QC 预览；完整报告含 {len(report.get('warnings', []))} 条导入警告。")
            set_status(f"已载入 {len(records)} 条记录供 QC 预览")

    def load_qc_records() -> None:
        snapshot = copy.deepcopy(state["config"])
        preview_id = state["preview_id"] = state["preview_id"] + 1
        state["qc_load_config_sha256"] = qc_source_config_sha256(snapshot)
        future = state["io_executor"].submit(load_recordings, None, snapshot["mapping"], snapshot)
        state["qc_future"] = (preview_id, future)
        set_status("core_io 正在导入记录与伴随表…")

    qc_load_button.clicked.connect(load_qc_records)
    qc_record_combo.currentIndexChanged.connect(lambda *_: show_qc_record())

    def annotate_qc(action: str) -> None:
        if qc_history_sources_conflict():
            QMessageBox.warning(window, "QC 历史来源冲突", "配置同时含 qc.annotations 与 input.manual_qc 表；为避免丢弃或重排审计记录，请先在导入页只绑定一个历史来源。")
            return
        record_index = qc_record_combo.currentData()
        if record_index is None:
            QMessageBox.information(window, "尚无 QC 记录", "请先载入当前输入。")
            return
        start, end = start_spin.value(), end_spin.value()
        reason, operator = reason_edit.text().strip(), operator_edit.text().strip()
        if not start < end or not reason or not operator:
            QMessageBox.warning(window, "标注字段不完整", "请检查非空时间区间，并填写操作者与原因。")
            return
        record = state["records"][int(record_index)]
        metadata = record.get("metadata", {})
        history = current_qc_history()
        revision = state.get("qc_result", {}).get("annotation_revision", len(history))
        before_history = copy.deepcopy(history)
        annotation = {"annotation_id": f"qc-{uuid.uuid4().hex}",
                      "recording_id": str(metadata.get("recording_id", "")),
                      "channel": str(metadata.get("channel", "unknown")),
                      "start_s": start, "end_s": end, "action": action,
                      "reason": reason, "operator": operator,
                      "timestamp": datetime.now(timezone.utc).isoformat(),
                      "clock_domain": str(metadata.get("clock_domain", ""))}
        try:
            new_history = append_annotation(history, annotation, expected_parent_revision=revision)
            state["config"]["qc"]["annotations"] = new_history
            if state["config"]["input"].get("manual_qc") is not None:
                state["config"]["input"]["manual_qc"] = None
                log("导入的 manual_qc 审计已保留在 qc.annotations；新 run 改用该冻结历史作为唯一 QC 来源。")
            update_config_editor()
            audit_hash = hashlib.sha256(json.dumps(new_history, ensure_ascii=False, sort_keys=True,
                                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            persist_project_draft(kind="qc", audit_action={
                "action_id": str(uuid.uuid4()), "operator": operator, "operation": action,
                "before": before_history, "after": new_history, "reason": reason,
                "snapshot_sha256": audit_hash,
            })
            show_qc_record()
            update_result_freshness()
            log(f"追加 QC {action}：{metadata.get('recording_id')} [{start:.4f}, {end:.4f}) · r{revision + 1}")
            set_status(f"QC {action} 已追加 · 修订 {revision + 1}")
        except Exception as error:
            QMessageBox.critical(window, "QC 标注失败", _format_error(error))

    exclude_button.clicked.connect(lambda: annotate_qc("exclude"))
    restore_button.clicked.connect(lambda: annotate_qc("restore"))

    def history_command(action: str) -> None:
        if qc_history_sources_conflict():
            QMessageBox.warning(window, "QC 历史来源冲突", "请先选择一个唯一的 QC 历史来源，再追加撤销/重做记录。")
            return
        target = qc_target_combo.currentData()
        reason, operator = reason_edit.text().strip(), operator_edit.text().strip()
        if not target or not reason or not operator:
            QMessageBox.warning(window, "无法更新审计历史", "选择有效决定，并填写操作者与本次原因。")
            return
        history = current_qc_history()
        revision = state.get("qc_result", {}).get("annotation_revision", len(history))
        before_history = copy.deepcopy(history)
        fn = undo_annotation if action == "undo" else redo_annotation
        try:
            new_history = fn(history, str(target), expected_parent_revision=revision,
                             operator=operator, reason=reason)
            state["config"]["qc"]["annotations"] = new_history
            if state["config"]["input"].get("manual_qc") is not None:
                state["config"]["input"]["manual_qc"] = None
            update_config_editor()
            audit_hash = hashlib.sha256(json.dumps(new_history, ensure_ascii=False, sort_keys=True,
                                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            persist_project_draft(kind="qc", audit_action={
                "action_id": str(uuid.uuid4()), "operator": operator, "operation": action,
                "before": before_history, "after": new_history, "reason": reason,
                "snapshot_sha256": audit_hash,
            })
            show_qc_record()
            update_result_freshness()
            log(f"已追加 {action} 事件，目标 {target}，新修订 {revision + 1}")
        except Exception as error:
            QMessageBox.warning(window, "审计命令失败", _format_error(error))

    undo_button.clicked.connect(lambda: history_command("undo"))
    redo_button.clicked.connect(lambda: history_command("redo"))
    refresh_audit_table()

    # ── 数值参数：控件直连完整配置，不另造运行路径 ─────────────────────────
    preprocess_page, preprocess_layout = add_page("预处理", "设置缺口处理、平滑、滤波和时间网格。清洁后的信号保留 μS；缩放结果作为单独分析分支保存。")
    preprocess_group = QGroupBox("缺口、平滑、滤波与时间网格")
    pp_form = QFormLayout(preprocess_group)

    def spin_box(value: Any, minimum: float, maximum: float, step: float, decimals: int = 3) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setValue(float(value or 0))
        return box

    gap_spin = spin_box(config["preprocess"]["interpolate_max_gap_s"], 0, 3600, 0.1, 3)
    preprocess_reference_labels: dict[str, QLabel] = {}

    def add_preprocess_field(label: str, key: str, control: QWidget) -> None:
        reference = QLabel()
        reference.setObjectName("subtle")
        reference.setMinimumWidth(180)
        reference.setWordWrap(True)
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addWidget(control)
        row.addWidget(reference, 1)
        pp_form.addRow(label, holder)
        preprocess_reference_labels[key] = reference

    def _reference_text(key: str, value: Any) -> str:
        if key in ("interpolation", "smoothing", "scaling"):
            labels = {"linear": "线性", "pchip": "保形插值", "none": "关闭", "moving": "滑动平均",
                      "median": "中位数平滑", "zscore": "Z 分数", "range": "范围缩放", "log": "对数缩放"}
            return labels.get(str(value), str(value))
        if value is None:
            return "关闭（0）" if key in ("lowpass_hz", "resample_hz") else "未设置"
        if isinstance(value, list):
            return "未设置" if not value else f"{value[0]}–{value[1]} s"
        units = {"interpolate_max_gap_s": " s", "smoothing_window_s": " s", "lowpass_hz": " Hz",
                 "resample_hz": " Hz", "min_segment_duration_s": " s", "filter_order": " 阶"}
        return f"{value:g}{units.get(key, '')}" if isinstance(value, (int, float)) else str(value)

    def refresh_preprocess_references() -> None:
        reference = get_preset(state["config"]["pipeline_id"])["preprocess"]
        for field, label_widget in preprocess_reference_labels.items():
            action = {
                "interpolate_max_gap_s": "短缺口插值上限",
                "interpolation": "插值方式",
                "smoothing": "平滑方式",
                "smoothing_window_s": "平滑窗口",
                "lowpass_hz": "低通截止；需低于 Nyquist 频率",
                "resample_hz": "目标采样率",
                "edge_tolerance": "settling 边缘容差",
                "min_segment_duration_s": "最短处理段",
                "jitter_tolerance": "采样间隔相对抖动上限",
                "baseline": "固定基线区间",
                "scaling": "独立缩放分支",
                "filter_order": "低通滤波阶数",
            }[field]
            label_widget.setText(f"方案参考 · {action}：{_reference_text(field, reference.get(field))}")

    add_preprocess_field("仅短缺口插值上限 (s；0=禁用)", "interpolate_max_gap_s", gap_spin)
    bind_value("preprocess.interpolate_max_gap_s", gap_spin, float)
    interpolation_box = ClickableComboBox()
    interpolation_box.addItem("线性插值", "linear")
    interpolation_box.addItem("PCHIP 保形插值", "pchip")
    interpolation_box.setCurrentIndex(interpolation_box.findData(config["preprocess"]["interpolation"]))
    add_preprocess_field("插值方法（不外推）", "interpolation", interpolation_box)
    bind_value("preprocess.interpolation", interpolation_box)
    smoothing_box = ClickableComboBox()
    smoothing_box.addItem("不平滑", "none")
    smoothing_box.addItem("滑动平均", "moving")
    smoothing_box.addItem("中位数平滑", "median")
    smoothing_box.setCurrentIndex(smoothing_box.findData(config["preprocess"]["smoothing"]))
    add_preprocess_field("平滑方法", "smoothing", smoothing_box)
    bind_value("preprocess.smoothing", smoothing_box)
    for key, label, maximum in (("smoothing_window_s", "平滑窗口 (s)", 3600),
                                ("lowpass_hz", "低通截止 (Hz；0=关闭)", 100000),
                                ("resample_hz", "重采样目标 (Hz；0=关闭)", 100000),
                                ("edge_tolerance", "边缘容差（相对 settling 幅值）", 0.1),
                                ("min_segment_duration_s", "最短处理段 (s)", 86400),
                                ("jitter_tolerance", "单段采样间隔相对抖动上限", 0.5),
                                ("filter_order", "低通滤波阶数", 12)):
        value = config["preprocess"][key]
        if key == "filter_order":
            box = QSpinBox()
            box.setRange(1, int(maximum))
            box.setValue(int(value))
            box.setMaximumWidth(210)
        else:
            box = spin_box(value, 0, maximum, 0.01 if key not in ("min_segment_duration_s",) else 1,
                       6 if key in ("edge_tolerance", "jitter_tolerance") else 3)
        add_preprocess_field(label, key, box)
        cast = (lambda v: None if float(v) <= 0 else float(v)) if key in ("lowpass_hz", "resample_hz") else float
        bind_value(f"preprocess.{key}", box, int if key == "filter_order" else cast,
                   nullable=key in ("lowpass_hz", "resample_hz"))
    baseline_edit = QLineEdit("")
    widgets["baseline_edit"] = baseline_edit
    baseline_edit.setPlaceholderText("留空或填 start,end（秒）；按该区间清洁样本均值减基线")
    baseline_edit.setText(",".join(map(str, config["preprocess"]["baseline"])) if config["preprocess"]["baseline"] else "")
    add_preprocess_field("固定基线区间 (s)", "baseline", baseline_edit)
    scaling_box = ClickableComboBox()
    scaling_box.addItem("不缩放", "none")
    scaling_box.addItem("Z 分数", "zscore")
    scaling_box.addItem("范围缩放", "range")
    scaling_box.addItem("对数缩放", "log")
    scaling_box.setCurrentIndex(scaling_box.findData(config["preprocess"]["scaling"]))
    add_preprocess_field("独立缩放分支", "scaling", scaling_box)
    bind_value("preprocess.scaling", scaling_box)
    baseline_hint = QLabel("滤波 cutoff 必须低于 Nyquist；QC 连续段会分别处理。边缘影响由冲激/settling 与保守边缘半径标出。非均匀时间若超过 jitter 上限会拒绝统一采样假设。")
    baseline_hint.setWordWrap(True)
    baseline_hint.setObjectName("subtle")
    pp_form.addRow("处理边界", baseline_hint)
    widgets["preprocess_reference_labels"] = preprocess_reference_labels
    refresh_preprocess_references()
    put_content_before_stretch(preprocess_layout, preprocess_group)

    def update_baseline() -> None:
        text = baseline_edit.text().strip()
        try:
            if not text:
                baseline = None
            else:
                endpoints = [float(item.strip()) for item in text.replace(";", ",").split(",")]
                if len(endpoints) != 2 or not all(math.isfinite(item) for item in endpoints) or endpoints[0] >= endpoints[1]:
                    raise ValueError("请提供 start,end 两个有限递增秒数。")
                baseline = endpoints
            state["config"]["preprocess"]["baseline"] = baseline
            update_config_editor()
        except Exception:
            baseline_edit.setStyleSheet("border: 1px solid #B94A48; background: #FFF2F0;")
            return
        baseline_edit.setStyleSheet("")

    baseline_edit.editingFinished.connect(update_baseline)

    analysis_page, analysis_layout = add_page("分解与 SCR", "支持独立滤波和非负 Bateman Driver 分解；SCR 使用 trough-to-peak 检测。方法与阈值尚需设备或任务验证。")
    decomposition_group = QGroupBox("信号分解")
    decomp_form = QFormLayout(decomposition_group)
    method_box = QComboBox()
    for method_name in ("filter_separation", "bateman_sparse", "cvxeda", "ledalab"):
        info = get_capabilities()["methods"][method_name]
        method_box.addItem(f"{method_name} · {'可用' if info['enabled'] else info['readiness']}", method_name)
    method_box.setCurrentIndex(max(0, method_box.findData(config["decomposition"]["method"])))
    decomp_form.addRow("分解方法", method_box)
    cutoff_spin = spin_box(config["decomposition"]["cutoff_hz"], 0.001, 100000, 0.01, 4)
    order_spin = QSpinBox()
    order_spin.setRange(1, 12)
    order_spin.setValue(config["decomposition"]["order"])
    decomp_form.addRow("分解截止 (Hz)", cutoff_spin)
    decomp_form.addRow("滤波阶数", order_spin)
    put_content_before_stretch(analysis_layout, decomposition_group)
    widgets.update({"decomp_method": method_box, "decomp_cutoff": cutoff_spin, "decomp_order": order_spin})

    def bind_section_control(section: str, field: str, control: Any, *, transform: Any = None) -> None:
        widgets[f"{section}.{field}"] = control
        state["widget_bindings"].append((f"{section}.{field}", control, transform, False))
        signal = (getattr(control, "valueChanged", None) or getattr(control, "currentIndexChanged", None)
                  or getattr(control, "toggled", None))
        if signal is None:
            return

        def save(*_args: Any) -> None:
            if state["busy"]:
                return
            value = control.isChecked() if isinstance(control, QCheckBox) else control.value() if hasattr(control, "value") else control.currentData()
            if value is None and hasattr(control, "currentText"):
                value = control.currentText()
            if (section == "decomposition" and field == "method" and
                    state["config"]["compute"].get("backend") == "gpu" and value != "bateman_sparse"):
                control.blockSignals(True)
                control.setCurrentIndex(control.findData(state["config"][section][field]))
                control.blockSignals(False)
                QMessageBox.warning(window, "GPU 后端不可用于当前方法",
                                    "当前计算后端为 GPU，且 GPU 仅支持 bateman_sparse。请先切换到 CPU，再选择其他方法。")
                return
            if section == "decomposition" and field == "method" and value != "bateman_sparse":
                from ..metrics_driver import DRIVER_IDS
                if DRIVER_IDS.intersection(state["config"]["features"].get("ids", [])):
                    control.blockSignals(True)
                    control.setCurrentIndex(control.findData(state["config"][section][field]))
                    control.blockSignals(False)
                    QMessageBox.warning(window, "仍有 Driver 指标被选中", "请先取消选择 Driver 指标，再切换分解方法。")
                    return
            state["config"][section][field] = transform(value) if transform else value
            if section == "decomposition" and field == "method":
                refresh_compute_guard()
            update_config_editor()
            if section == "decomposition" and field == "method" and "feature_registry_table" in widgets:
                refresh_feature_selector()

        signal.connect(save)

    bind_section_control("decomposition", "method", method_box)
    bind_section_control("decomposition", "cutoff_hz", cutoff_spin, transform=float)
    bind_section_control("decomposition", "order", order_spin, transform=int)

    driver_editor = DriverParameterForm(config["decomposition"]["driver"])
    driver_editor.setObjectName("driverParameters")
    driver_apply = QPushButton("校验并应用 Driver 参数")
    decomp_form.addRow("Bateman driver 参数", driver_editor)
    decomp_form.addRow("", driver_apply)
    widgets.update(driver_parameters=driver_editor, driver_apply=driver_apply)

    def apply_driver_parameters() -> None:
        try:
            candidate = copy.deepcopy(state["config"])
            candidate = config_with_analysis_forms()
            apply_config(candidate, source="Driver 参数")
        except Exception as error:
            QMessageBox.warning(window, "Driver 参数无效", _format_error(error))

    driver_apply.clicked.connect(apply_driver_parameters)

    qc_parameter_group = QGroupBox("自动质量阈值")
    qc_parameter_form = QFormLayout(qc_parameter_group)
    exclude_auto_check = QCheckBox("分析时排除自动候选")
    exclude_auto_check.setChecked(config["qc"]["exclude_auto"])
    qc_parameter_form.addRow("自动候选", exclude_auto_check)
    bind_section_control("qc", "exclude_auto", exclude_auto_check, transform=bool)
    qc_references = {
        "min_uS": ("文献示例 0.05 μS", "低于此电导值的样本会成为自动质量候选；0 表示关闭。"),
        "max_uS": ("文献设备上限示例 60 μS", "高于此电导值的样本会成为自动质量候选；0 表示关闭。"),
        "max_slope_uS_s": ("文献示例 10 μS/s", "超过此变化速率的样本会成为自动质量候选；0 表示关闭。"),
        "max_jump_uS": ("程序参考：关闭", "相邻样本变化超过此阈值时标记候选；0 表示关闭。"),
        "flatline_duration_s": ("程序参考：5 s", "变化范围不超过平线容差且持续达到该时长时标记候选。"),
        "flatline_tolerance_uS": ("程序参考：0 μS", "平线检测允许的电导变化幅度。"),
        "gap_factor": ("程序参考：1.5 × 中位间隔", "采样间隔超过典型间隔的该倍数时分段。"),
    }
    for key, label, high in (("min_uS", "最低电导 (μS；0=关闭)", 1e6),
                             ("max_uS", "最高电导 (μS；0=关闭)", 1e6),
                             ("max_slope_uS_s", "最大斜率 (μS/s；0=关闭)", 1e9),
                             ("max_jump_uS", "最大相邻跳变 (μS；0=关闭)", 1e6),
                             ("flatline_duration_s", "平线最短时长 (s)", 86400),
                             ("flatline_tolerance_uS", "平线容差 (μS)", 1e6),
                             ("gap_factor", "缺口分段倍数", 100)):
        current = config["qc"][key]
        box = spin_box(current, 0, high, 0.1, 5)
        box.setToolTip(qc_references[key][1])
        note = QLabel(qc_references[key][0])
        note.setObjectName("subtle")
        note.setMinimumWidth(170)
        note.setWordWrap(True)
        field = QWidget()
        row = QHBoxLayout(field)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addWidget(box)
        row.addWidget(note, 1)
        qc_parameter_form.addRow(label, field)
        optional = key in ("min_uS", "max_uS", "max_slope_uS_s", "max_jump_uS")
        bind_section_control("qc", key, box,
                             transform=(lambda value: None if float(value) <= 0 else float(value)) if optional else float)
    qc_source = QLabel(
        '<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5880745/">'
        "Kleckner 等（2017）表 1 文献示例</a>：限值与设备和研究情境有关，不是通用标准。"
    )
    qc_source.setOpenExternalLinks(True)
    qc_source.setWordWrap(True)
    qc_source.setObjectName("subtle")
    qc_parameter_form.addRow("参考来源", qc_source)
    qc_layout.insertWidget(qc_layout.indexOf(annotation_group), qc_parameter_group)

    scr_group = QGroupBox("SCR 事件")
    scr_form = QFormLayout(scr_group)
    basis_box = QComboBox()
    basis_box.addItems(["phasic", "clean"])
    basis_box.setCurrentText(config["scr"]["signal_basis"])
    threshold_box = QComboBox()
    threshold_box.addItems(["absolute", "relative"])
    threshold_box.setCurrentText(config["scr"]["threshold_type"])
    scr_form.addRow("检测信号", basis_box)
    scr_form.addRow("阈值类型", threshold_box)
    bind_section_control("scr", "signal_basis", basis_box)
    bind_section_control("scr", "threshold_type", threshold_box)
    for key, label, maximum in (("amplitude_min_uS", "绝对幅度阈值 (μS)", 1e6),
                                ("relative_fraction", "相对录音最大候选峰幅比例", 1),
                                ("min_rise_s", "最短上升时长 (s)", 3600),
                                ("max_rise_s", "最长上升时长 (s)", 3600),
                                ("min_interval_s", "响应最小间隔 (s)", 86400),
                                ("recovery_fraction_returned", "恢复判定返回幅度比例", 1)):
        value = config["scr"][key]
        box = spin_box(value, 0 if key in ("amplitude_min_uS", "min_rise_s", "max_rise_s", "min_interval_s") else 1e-6,
                       maximum, 0.05, 5)
        scr_form.addRow(label, box)
        bind_section_control("scr", key, box, transform=float)
    for key, label in (("onset_left_context_s", "Onset 左上下文 (s；同一合格段内)"),
                       ("onset_right_context_s", "Onset 右上下文 (s；≥最长上升时长)")):
        value = config["scr"].get(key)
        if value is None:
            value = config["scr"]["max_rise_s"]
        box = spin_box(value, 0, 86400, 0.25, 4)
        scr_form.addRow(label, box)
        bind_section_control("scr", key, box, transform=float)
        if key == "onset_right_context_s":
            max_rise_control = widgets["scr.max_rise_s"]
            max_rise_control.valueChanged.connect(lambda value, right=box: right.setValue(max(right.value(), value)))
    window_group = QGroupBox("分析窗与特征")
    window_form = QFormLayout(window_group)
    for key, label in (("fixed_length_s", "固定窗长 (s；0=关闭)"), ("fixed_step_s", "固定步长 (s；0=关闭)"),
                       ("min_coverage", "最低覆盖比例")):
        value = config["windows"][key]
        box = spin_box(value, 0 if key != "min_coverage" else 0, 100000 if key != "min_coverage" else 1, 1, 4)
        window_form.addRow(label, box)
        section = "windows"
        cast = (lambda v: None if float(v) <= 0 else float(v)) if key in ("fixed_length_s", "fixed_step_s") else float
        bind_section_control(section, key, box, transform=cast)
    tail_box = QComboBox()
    tail_box.addItems(["drop", "include"])
    tail_box.setCurrentText(config["windows"]["tail_policy"])
    window_form.addRow("固定窗尾处理", tail_box)
    bind_section_control("windows", "tail_policy", tail_box)
    selector_group = QGroupBox("指标注册表")
    selector_layout = QVBoxLayout(selector_group)
    selector_toolbar = QHBoxLayout()
    feature_family = QComboBox()
    feature_family.setObjectName("featureFamilyFilter")
    feature_family.addItem("全部指标族", "")
    feature_family.setMinimumWidth(150)
    feature_search = QLineEdit()
    feature_search.setObjectName("featureSearch")
    feature_search.setPlaceholderText("按中文名称、ID、单位或资格说明筛选")
    feature_search.setClearButtonEnabled(True)
    select_visible = QPushButton("全选当前筛选")
    select_visible.setObjectName("selectVisibleFeatures")
    clear_visible = QPushButton("清除当前筛选")
    clear_visible.setObjectName("clearVisibleFeatures")
    select_current = QPushButton("仅保留所选指标")
    select_current.setObjectName("selectCurrentFeatureOnly")
    selector_toolbar.addWidget(feature_family)
    selector_toolbar.addWidget(feature_search, 1)
    selector_toolbar.addWidget(select_current)
    selector_toolbar.addWidget(select_visible)
    selector_toolbar.addWidget(clear_visible)
    selector_layout.addLayout(selector_toolbar)
    feature_table = QTableWidget(0, 5)
    feature_table.setObjectName("featureRegistryTable")
    feature_table.setHorizontalHeaderLabels(["选择", "指标名称", "feature ID", "单位", "资格"])
    feature_table.verticalHeader().setVisible(False)
    feature_table.setAlternatingRowColors(True)
    feature_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    feature_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    feature_table.setMinimumHeight(300)
    feature_table.setMaximumHeight(500)
    feature_table.setColumnWidth(0, 52)
    feature_table.setColumnWidth(1, 220)
    feature_table.setColumnWidth(2, 250)
    feature_table.setColumnWidth(3, 90)
    feature_table.horizontalHeader().setStretchLastSection(True)
    selector_layout.addWidget(feature_table)
    feature_selection_status = QLabel("")
    feature_selection_status.setObjectName("featureSelectionStatus")
    feature_selection_status.setWordWrap(True)
    feature_selection_status.setObjectName("subtle")
    selector_layout.addWidget(feature_selection_status)
    window_form.addRow("指标选择", selector_group)

    extended_group = QGroupBox("扩展指标参数")
    extended_layout = QVBoxLayout(extended_group)
    extended_enabled = QCheckBox("启用扩展指标计算")
    extended_enabled.setObjectName("extendedFeaturesEnabled")
    extended_enabled.setChecked(bool(config["features"].get("extended", {}).get("enabled", False)))
    extended_layout.addWidget(extended_enabled)
    extended_note = QLabel(
        "在各页签直接填写参数，点击应用后检查参数并保存到当前项目。"
        "指标定义、单位、有效支持和资格说明可在注册表行悬停查看。"
    )
    extended_note.setObjectName("subtle")
    extended_note.setWordWrap(True)
    extended_layout.addWidget(extended_note)
    baseline_row = QHBoxLayout()
    baseline_row.addWidget(QLabel("基线分母下限 (μS)"))
    baseline_min_spin = QDoubleSpinBox()
    baseline_min_spin.setObjectName("featureBaselineMin")
    baseline_min_spin.setDecimals(12)
    baseline_min_spin.setRange(1e-12, 1e6)
    baseline_min_spin.setSingleStep(1e-6)
    baseline_min_spin.setValue(float(config["features"].get("extended", {}).get("baseline_min_uS", 1e-6)))
    baseline_row.addWidget(baseline_min_spin)
    baseline_row.addStretch(1)
    extended_layout.addLayout(baseline_row)
    parameter_tabs = QTabWidget()
    parameter_tabs.setObjectName("extendedParameterTabs")
    parameter_labels = {
        "trial_series": "试次序列", "nonspecific": "非特异 SCR", "spectral": "频谱",
        "autocorrelation": "自相关", "complexity": "复杂度", "group_temporal": "群体时序",
    }
    parameter_editors: dict[str, ParameterForm] = {}
    parameter_validation_labels: dict[str, QLabel] = {}
    for parameter_key, parameter_label in parameter_labels.items():
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        editor = ParameterForm(parameter_key,
                               config["features"].get("extended", {}).get(parameter_key, {}),
                               parent=tab)
        scroll = QScrollArea(tab)
        scroll.setObjectName(f"extendedParameterScroll_{parameter_key}")
        scroll.setWidgetResizable(True)
        scroll.setWidget(editor)
        tab_layout.addWidget(scroll, 1)
        apply_parameters = QPushButton(f"应用{parameter_label}参数")
        apply_parameters.setObjectName(f"applyExtended_{parameter_key}")
        apply_parameters.setMinimumHeight(32)
        validation_label = QLabel("")
        validation_label.setObjectName(f"extendedParameterStatus_{parameter_key}")
        validation_label.setWordWrap(True)
        validation_label.setObjectName("subtle")
        tab_layout.addWidget(validation_label)
        tab_layout.addWidget(apply_parameters, alignment=Qt.AlignmentFlag.AlignRight)
        parameter_tabs.addTab(tab, parameter_label)
        parameter_editors[parameter_key] = editor
        parameter_validation_labels[parameter_key] = validation_label
        widgets[f"apply_extended_{parameter_key}"] = apply_parameters
        widgets[f"extended_parameter_status_{parameter_key}"] = validation_label
    extended_layout.addWidget(parameter_tabs)
    window_form.addRow("扩展参数", extended_group)
    whole_record_check = QCheckBox("包含全记录窗口")
    whole_record_check.setChecked(config["windows"]["whole_record"])
    window_form.addRow("全记录窗口", whole_record_check)
    bind_section_control("windows", "whole_record", whole_record_check, transform=bool)
    for key, label in (("event_baseline_s", "事件 baseline 相对区间 (s)"),
                       ("event_response_s", "事件 response 相对区间 (s)"),
                       ("event_peak_s", "事件 peak 搜索区间 (s)"),
                       ("event_recovery_s", "事件 recovery 相对区间 (s)")):
        range_edit = QLineEdit(",".join(map(str, config["windows"][key])))
        range_edit.setPlaceholderText("start,end；相对事件时间")
        window_form.addRow(label, range_edit)
        widgets[f"windows.{key}"] = range_edit

        def update_event_range(field: str = key, edit: QLineEdit = range_edit) -> None:
            try:
                limits = [float(item.strip()) for item in edit.text().replace(";", ",").split(",")]
                if len(limits) != 2 or not all(math.isfinite(x) for x in limits) or limits[0] >= limits[1]:
                    raise ValueError("需要两个有限且递增的相对秒数。")
                candidate = copy.deepcopy(state["config"])
                candidate["windows"][field] = limits
                state["config"] = resolve_config(candidate)
                edit.setStyleSheet("")
                update_config_editor()
            except Exception:
                edit.setStyleSheet("border: 1px solid #B94A48; background: #FFF2F0;")

        range_edit.editingFinished.connect(update_event_range)
    exclude_interpolated = QCheckBox("特征分析排除插值影响样本")
    exclude_interpolated.setChecked(config["features"]["exclude_interpolated"])
    window_form.addRow("插值策略", exclude_interpolated)
    bind_section_control("features", "exclude_interpolated", exclude_interpolated, transform=bool)
    put_content_before_stretch(analysis_layout, scr_group)
    put_content_before_stretch(analysis_layout, window_group)

    widgets.update({"feature_registry_table": feature_table, "feature_family_filter": feature_family,
                    "feature_search": feature_search, "feature_selection_status": feature_selection_status,
                    "select_visible_features": select_visible, "clear_visible_features": clear_visible,
                    "select_current_feature_only": select_current,
                    "extended_features_enabled": extended_enabled, "feature_baseline_min": baseline_min_spin,
                    "extended_parameter_tabs": parameter_tabs,
                    "extended_parameter_editors": parameter_editors})

    def current_feature_registry(candidate: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
        source = candidate if candidate is not None else state["config"]
        from ..core_features import list_feature_definitions
        catalog_config = copy.deepcopy(source)
        extended = catalog_config.setdefault("features", {}).setdefault("extended", {})
        extended["enabled"] = True
        for family in ("nonspecific", "spectral", "autocorrelation", "complexity", "group_temporal"):
            if isinstance(extended.get(family), dict) and "enabled" in extended[family]:
                extended[family]["enabled"] = True
        catalog_config.setdefault("synchrony", {})["enabled"] = True
        catalog_config.setdefault("decomposition", {})["method"] = "bateman_sparse"
        registry = list_feature_definitions(catalog_config)
        return {str(key): dict(value) for key, value in registry.items()}

    def feature_activation_requirement(feature_id: str, definition: dict[str, Any],
                                       candidate: dict[str, Any] | None = None) -> str:
        from ..core_features import LEGACY_FEATURE_IDS
        if feature_id in LEGACY_FEATURE_IDS:
            return ""
        cfg = candidate if candidate is not None else state["config"]
        extended = cfg.get("features", {}).get("extended", {})
        if not extended.get("enabled", False):
            return "启用“扩展指标计算”"
        family = str(definition.get("family", ""))
        if family == "driver" and cfg.get("decomposition", {}).get("method") != "bateman_sparse":
            return "选择 bateman_sparse 分解方法或 single_driver 预设"
        if family == "group_temporal" and not cfg.get("synchrony", {}).get("enabled", False):
            return "启用“多人同步分析”"
        family_switch = {"spectral": "spectral", "autocorrelation": "autocorrelation",
                         "complexity": "complexity", "nonspecific": "nonspecific",
                         "group_temporal": "group_temporal"}.get(family)
        if family_switch and not extended.get(family_switch, {}).get("enabled", False):
            return f"在扩展参数中启用 {feature_family_label(family)}"
        return ""

    def active_feature_ids(candidate: dict[str, Any],
                           registry: dict[str, dict[str, Any]] | None = None) -> set[str]:
        definitions = registry if registry is not None else current_feature_registry(candidate)
        return {feature_id for feature_id, definition in definitions.items()
                if not feature_activation_requirement(feature_id, definition, candidate)}

    def feature_family_label(family: str) -> str:
        return {"legacy": "原有 33 项", "sample": "采样统计", "tonic": "Tonic", "driver": "Driver 驱动",
                "phasic": "Phasic", "phasic_integral": "Phasic 积分",
                "scr": "SCR", "scr_distribution": "SCR 幅值分布与变换", "scr_rise": "SCR 上升时间与斜率",
                "scr_recovery": "SCR 恢复", "scr_width": "SCR 半高宽", "scr_isi": "SCR 起始间隔",
                "event": "事件", "event_trial": "事件试次", "trial": "试次",
                "trial_series": "试次序列", "habituation": "习惯化趋势（描述性）",
                "nonspecific": "非特异 SCR", "spectral": "频谱", "autocorrelation": "自相关",
                "complexity": "复杂度", "group_temporal": "群体时序", "group": "群体同步"}.get(
                    family, family.replace("_", " ") if family else "未分类")

    def update_feature_filters() -> None:
        query = feature_search.text().strip().casefold()
        family = str(feature_family.currentData() or "")
        for row in range(feature_table.rowCount()):
            family_item = feature_table.item(row, 4)
            search_text = " ".join(str(feature_table.item(row, col).data(Qt.ItemDataRole.UserRole) or "")
                                    for col in range(1, feature_table.columnCount())).casefold()
            match = (not family or str(family_item.data(Qt.ItemDataRole.UserRole) or "") == family)
            feature_table.setRowHidden(row, not match or (query not in search_text))

    def update_feature_selection_status(message: str | None = None) -> None:
        active_ids = set(state.get("active_feature_ids", set()))
        total = len(active_ids)
        selected = sum(1 for row in range(feature_table.rowCount())
                       if str(feature_table.item(row, 0).data(Qt.ItemDataRole.UserRole)) in active_ids and
                       feature_table.item(row, 0).checkState() == Qt.CheckState.Checked)
        inactive_n = feature_table.rowCount() - total
        note = message or ("空的 features.ids 表示选择当前配置中全部可用指标。"
                           if not state["config"]["features"].get("ids") else "仅计算显式勾选且当前启用的指标。")
        if inactive_n:
            note += f" 另有 {inactive_n} 项因扩展家族或多人同步未启用而灰显。"
        feature_selection_status.setText(f"已选择 {selected}/{total} 项可用指标。{note}")

    def commit_feature_selection(*, message: str | None = None) -> None:
        active_ids = set(state.get("active_feature_ids", set()))
        selected_ids = [str(feature_table.item(row, 0).data(Qt.ItemDataRole.UserRole))
                        for row in range(feature_table.rowCount())
                        if feature_table.item(row, 0).checkState() == Qt.CheckState.Checked]
        active_selected = set(selected_ids).intersection(active_ids)
        if not active_selected:
            previous_ids = set(state["config"]["features"].get("ids", []))
            restore_all = not previous_ids
            previous_active = previous_ids.intersection(active_ids)
            if previous_ids and not previous_active:
                update_feature_selection_status("至少保留一项当前可用指标；已恢复上次选择。可先启用对应指标族。")
                refresh_feature_selector()
                return
            feature_table.blockSignals(True)
            for row in range(feature_table.rowCount()):
                item = feature_table.item(row, 0)
                feature_id = str(item.data(Qt.ItemDataRole.UserRole))
                item.setCheckState(Qt.CheckState.Checked if feature_id in active_ids and
                                   (restore_all or feature_id in previous_active)
                                   else Qt.CheckState.Unchecked)
            feature_table.blockSignals(False)
            update_feature_selection_status(
                "至少保留一项当前可用指标；已恢复上次选择。空的 features.ids 表示选择全部可用指标。")
            return
        state["config"]["features"]["ids"] = ([] if set(selected_ids) == active_ids else sorted(selected_ids))
        update_feature_selection_status(message)
        update_config_editor()

    def refresh_feature_selector() -> None:
        registry = current_feature_registry()
        active_ids = active_feature_ids(state["config"], registry)
        state["feature_registry"] = registry
        state["active_feature_ids"] = active_ids
        previous_family = str(feature_family.currentData() or "")
        families = sorted({str(definition.get("family", "")) for definition in registry.values()})
        feature_family.blockSignals(True)
        feature_family.clear()
        feature_family.addItem("全部指标族", "")
        for family in families:
            feature_family.addItem(feature_family_label(family), family)
        restored_index = feature_family.findData(previous_family)
        if restored_index >= 0:
            feature_family.setCurrentIndex(restored_index)
        feature_family.blockSignals(False)
        explicitly_selected = set(state["config"]["features"].get("ids", []))
        select_all = not explicitly_selected
        ordered = sorted(registry.items(), key=lambda pair: (str(pair[1].get("family", "")),
                                                              str(pair[1].get("label_zh", pair[0])), pair[0]))
        feature_table.blockSignals(True)
        feature_table.setRowCount(len(ordered))
        for row, (feature_id, definition) in enumerate(ordered):
            label = str(definition.get("label_zh") or feature_id)
            family = str(definition.get("family", ""))
            qualification = str(definition.get("qualification", ""))
            activation_requirement = feature_activation_requirement(feature_id, definition)
            active = feature_id in active_ids
            display_qualification = (f"未启用 · {activation_requirement}" if activation_requirement
                                    else qualification or "未声明")
            tooltip_lines = [f"{label} ({feature_id})", f"指标族：{feature_family_label(family)}",
                             f"单位：{definition.get('unit', '')}", f"basis：{definition.get('basis', '')}",
                             f"公式：{definition.get('formula', '')}",
                             f"最小样本数：{definition.get('min_n', '')}",
                             f"最短有效时长：{definition.get('min_duration_s', '')}",
                             f"参数：{json.dumps(definition.get('parameters', {}), ensure_ascii=False, sort_keys=True)}",
                             f"有效性：{definition.get('validity', '')}", f"资格：{qualification}",
                             f"当前配置：{'可用' if active else f'未启用，需{activation_requirement}'}"]
            searchable = " ".join([feature_id, label, family, str(definition.get("unit", "")),
                                   display_qualification, str(definition.get("formula", ""))])
            toggle = QTableWidgetItem()
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if active or feature_id in explicitly_selected:
                flags |= Qt.ItemFlag.ItemIsUserCheckable
            toggle.setFlags(flags)
            toggle.setCheckState(Qt.CheckState.Checked if (select_all and active) or
                                 feature_id in explicitly_selected
                                 else Qt.CheckState.Unchecked)
            toggle.setData(Qt.ItemDataRole.UserRole, feature_id)
            toggle.setData(Qt.ItemDataRole.UserRole + 1, searchable)
            toggle.setData(Qt.ItemDataRole.UserRole + 2, active)
            toggle.setToolTip("\n".join(tooltip_lines))
            feature_table.setItem(row, 0, toggle)
            for column, value in ((1, label), (2, feature_id), (3, str(definition.get("unit", ""))),
                                  (4, display_qualification)):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setData(Qt.ItemDataRole.UserRole, family if column == 4 else searchable)
                item.setToolTip("\n".join(tooltip_lines))
                if not active:
                    item.setForeground(QBrush(QColor("#81909A")))
                feature_table.setItem(row, column, item)
            feature_table.setRowHeight(row, 27)
        feature_table.blockSignals(False)
        update_feature_filters()
        update_feature_selection_status()

    def apply_feature_family_parameters(parameter_key: str, editor: ParameterForm) -> None:
        validation_label = parameter_validation_labels[parameter_key]
        validation_label.setText("")
        try:
            parsed = editor.value()
            candidate = copy.deepcopy(state["config"])
            candidate["features"]["extended"][parameter_key] = parsed
            old_ids = list(candidate["features"].get("ids", []))
            if old_ids:
                old_registry = current_feature_registry(state["config"])
                new_registry = current_feature_registry(candidate)
                new_ids = set(new_registry)
                removed = set(old_ids) - new_ids
                adjusted = [item for item in old_ids if item in new_ids]
                changed_families: set[str] = set()
                for removed_id in removed:
                    if removed_id.startswith("clean_bandpower_"):
                        changed_families.add("band_abs" if removed_id.endswith("_uS2") else "band_norm")
                    elif removed_id.startswith("clean_autocorr_lag_"):
                        changed_families.add("autocorrelation")
                if "band_abs" in changed_families:
                    adjusted.extend(item for item in new_ids if item.startswith("clean_bandpower_") and
                                    item.endswith("_uS2"))
                if "band_norm" in changed_families:
                    adjusted.extend(item for item in new_ids if item.startswith("clean_bandpower_") and
                                    item.endswith("_normalized"))
                if "autocorrelation" in changed_families:
                    adjusted.extend(item for item in new_ids if item.startswith("clean_autocorr_lag_"))
                adjusted_ids = set(adjusted)
                if not adjusted_ids:
                    raise ValueError("参数修改会清空显式指标选择。请至少保留一项可用指标，避免空列表被解释为全选。")
                newly_inactive = adjusted_ids - active_feature_ids(candidate, new_registry)
                if newly_inactive:
                    names = ", ".join(sorted(newly_inactive)[:6])
                    suffix = " …" if len(newly_inactive) > 6 else ""
                    raise ValueError(f"先取消选择当前将停用的指标，再关闭对应家族：{names}{suffix}")
                candidate["features"]["ids"] = sorted(adjusted_ids)
            if not apply_config(candidate, source=f"{parameter_labels[parameter_key]}参数"):
                validation_label.setText("参数未应用。请按配置校验提示修改字段后再试。")
                return
            validation_label.setText("已校验并保存到当前配置；项目打开时会同步持久化。")
            update_feature_selection_status("参数已校验并保存；如 band/lag 配置改变，已选中的对应动态指标族随之更新。")
        except Exception as error:
            detail = _format_error(error)
            validation_label.setText(f"未应用：{detail}")
            QMessageBox.warning(window, "扩展参数无效", detail)

    def update_extended_enablement(enabled: bool) -> None:
        selected_ids = set(state["config"]["features"].get("ids", []))
        if not enabled and selected_ids:
            from ..core_features import LEGACY_FEATURE_IDS
            extended_selected = selected_ids - set(LEGACY_FEATURE_IDS)
            if extended_selected:
                extended_enabled.blockSignals(True)
                extended_enabled.setChecked(True)
                extended_enabled.blockSignals(False)
                names = ", ".join(sorted(extended_selected)[:6])
                suffix = " …" if len(extended_selected) > 6 else ""
                QMessageBox.warning(window, "仍有扩展指标被选中",
                                    f"先取消选择扩展指标，再关闭扩展计算：{names}{suffix}")
                return
        state["config"]["features"]["extended"]["enabled"] = bool(enabled)
        update_config_editor()
        refresh_feature_selector()
        update_feature_selection_status()

    def update_extended_baseline(value: float) -> None:
        state["config"]["features"]["extended"]["baseline_min_uS"] = float(value)
        update_config_editor()

    def refresh_extended_parameter_editors() -> None:
        driver_editor.set_config(state["config"]["decomposition"]["driver"])
        extended = state["config"].get("features", {}).get("extended", {})
        extended_enabled.blockSignals(True)
        extended_enabled.setChecked(bool(extended.get("enabled", False)))
        extended_enabled.blockSignals(False)
        baseline_min_spin.blockSignals(True)
        baseline_min_spin.setValue(float(extended.get("baseline_min_uS", 1e-6)))
        baseline_min_spin.blockSignals(False)
        for parameter_key, editor in parameter_editors.items():
            editor.set_value(extended.get(parameter_key, {}))
            parameter_validation_labels[parameter_key].clear()

    def handle_feature_item_changed(item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        active = bool(item.data(Qt.ItemDataRole.UserRole + 2))
        if not active and item.checkState() == Qt.CheckState.Checked:
            feature_table.blockSignals(True)
            item.setCheckState(Qt.CheckState.Unchecked)
            feature_table.blockSignals(False)
            update_feature_selection_status("该指标当前未启用，需先开启对应功能后才能选择。")
            return
        commit_feature_selection()

    feature_table.itemChanged.connect(handle_feature_item_changed)
    feature_family.currentIndexChanged.connect(lambda *_: update_feature_filters())
    feature_search.textChanged.connect(lambda *_: update_feature_filters())

    def set_visible_features(selected: bool) -> None:
        feature_table.blockSignals(True)
        for row in range(feature_table.rowCount()):
            if not feature_table.isRowHidden(row) and bool(
                    feature_table.item(row, 0).data(Qt.ItemDataRole.UserRole + 2)):
                feature_table.item(row, 0).setCheckState(
                    Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked)
        feature_table.blockSignals(False)
        commit_feature_selection()

    def select_only_current_feature() -> None:
        selected_row = feature_table.currentRow()
        if selected_row < 0 or feature_table.item(selected_row, 0) is None:
            update_feature_selection_status("先选择一行指标，再使用“仅保留所选指标”。")
            return
        if not bool(feature_table.item(selected_row, 0).data(Qt.ItemDataRole.UserRole + 2)):
            update_feature_selection_status("当前指标尚未启用；开启相应家族后才能单独选择。")
            return
        feature_table.blockSignals(True)
        for row in range(feature_table.rowCount()):
            if not bool(feature_table.item(row, 0).data(Qt.ItemDataRole.UserRole + 2)):
                feature_table.item(row, 0).setCheckState(Qt.CheckState.Unchecked)
                continue
            feature_table.item(row, 0).setCheckState(
                Qt.CheckState.Checked if row == selected_row else Qt.CheckState.Unchecked)
        feature_table.blockSignals(False)
        commit_feature_selection(message="当前只选择了一项指标。")

    select_visible.clicked.connect(lambda: set_visible_features(True))
    clear_visible.clicked.connect(lambda: set_visible_features(False))
    select_current.clicked.connect(select_only_current_feature)
    extended_enabled.toggled.connect(update_extended_enablement)
    baseline_min_spin.valueChanged.connect(update_extended_baseline)
    for parameter_key, editor in parameter_editors.items():
        widgets[f"apply_extended_{parameter_key}"].clicked.connect(
            lambda _checked=False, key=parameter_key, target=editor:
            apply_feature_family_parameters(key, target))
    refresh_feature_selector()
    refresh_extended_parameter_editors()

    # ── 多人同步与结构化统计 ──────────────────────────────────────────────
    sync_page, sync_layout = add_page("多人同步", "真实 clock anchors、成员关系与方法选择由 core_sync/pipeline 校验。秒单位不意味着共享时钟。")
    sync_group = QGroupBox("同步方案")
    sync_form = QFormLayout(sync_group)
    sync_enabled = QCheckBox("运行多人同步分析")
    sync_enabled.setChecked(config["synchrony"]["enabled"])
    sync_form.addRow("启用", sync_enabled)
    widgets["synchrony.enabled"] = sync_enabled
    state["widget_bindings"].append(("synchrony.enabled", sync_enabled, bool, False))

    def update_synchrony_enablement(enabled: bool) -> None:
        selected_ids = set(state["config"]["features"].get("ids", []))
        if not enabled and selected_ids:
            group_ids = {feature_id for feature_id, definition in current_feature_registry().items()
                         if definition.get("family") == "group_temporal"}
            selected_group_ids = selected_ids.intersection(group_ids)
            if selected_group_ids:
                sync_enabled.blockSignals(True)
                sync_enabled.setChecked(True)
                sync_enabled.blockSignals(False)
                names = ", ".join(sorted(selected_group_ids)[:6])
                suffix = " …" if len(selected_group_ids) > 6 else ""
                QMessageBox.warning(window, "仍有群体时序指标被选中",
                                    f"先取消选择群体时序指标，再关闭多人同步分析：{names}{suffix}")
                return
        state["config"]["synchrony"]["enabled"] = bool(enabled)
        update_config_editor()
        refresh_feature_selector()

    sync_enabled.toggled.connect(update_synchrony_enablement)
    for key, label, values in (("basis", "信号基线", ["clean", "tonic", "phasic"]),
                               ("metric", "相关度量", ["pearson", "spearman"]),
                               ("null_method", "Null 方法", ["circular_shift", "pseudodyad", "none"]),
                               ("tail", "检验方向", ["two-sided", "greater", "less"])):
        box = QComboBox()
        box.addItems(values)
        box.setCurrentText(config["synchrony"][key])
        sync_form.addRow(label, box)
        bind_section_control("synchrony", key, box)
    for key, label, high in (("window_s", "窗长 (s)", 86400), ("step_s", "步长 (s)", 86400),
                             ("max_lag_s", "最大滞后 (s)", 3600), ("lag_step_s", "滞后分辨率 (s)", 3600),
                             ("analysis_rate_hz", "共同分析采样率 (Hz)", 100000),
                             ("timing_tolerance_s", "时间容差 (s)", 100),
                             ("min_coverage", "最低覆盖比例", 1),
                             ("min_shift_s", "最小循环移位 (s)", 86400),
                             ("null_replicates", "Null 重复数", 100000)):
        integer = key == "null_replicates"
        box = QSpinBox() if integer else spin_box(config["synchrony"][key], 0, high, 0.1, 4)
        if integer:
            box.setRange(0, high)
            box.setValue(int(config["synchrony"][key]))
        elif key in ("min_coverage",):
            box.setRange(0, 1)
        else:
            box.setRange(0 if key in ("max_lag_s", "timing_tolerance_s", "min_shift_s") else 1e-6, high)
        sync_form.addRow(label, box)
        bind_section_control("synchrony", key, box, transform=int if integer else float)
    shared_clock = QCheckBox("已通过独立时钟证据验证共享参考时钟")
    shared_clock.setChecked(config["synchrony"]["shared_clock_verified"])
    sync_form.addRow("时钟资格声明", shared_clock)
    bind_section_control("synchrony", "shared_clock_verified", shared_clock, transform=bool)
    sync_note = QLabel("不启用独立时钟锚点时，不能仅凭时间戳或 EDA 相似性声称同步。Null 表仅导出 mean/SD/尾概率摘要时，图形不会伪造经验分布。")
    sync_note.setWordWrap(True)
    sync_note.setObjectName("subtle")
    sync_form.addRow("方法边界", sync_note)
    put_content_before_stretch(sync_layout, sync_group)

    stats_page, stats_layout = add_page("统计设计", "选择分析方法并填写字段。设计采用结构化中文表单；高级字段保留原配置，公式或代码不会执行。")
    stats_group = QGroupBox("已计划的统计分析")
    stats_layout_inner = QVBoxLayout(stats_group)
    stats_enabled = QCheckBox("运行统计设计")
    stats_enabled.setChecked(config["statistics"]["enabled"])
    stats_layout_inner.addWidget(stats_enabled)
    bind_section_control("statistics", "enabled", stats_enabled, transform=bool)
    correction_row = QHBoxLayout()
    correction_row.addWidget(QLabel("多重比较校正"))
    correction_box = QComboBox()
    correction_box.addItems(["holm", "fdr_bh", "fdr_by", "none"])
    correction_box.setCurrentText(config["statistics"]["correction"])
    correction_row.addWidget(correction_box)
    correction_row.addStretch(1)
    stats_layout_inner.addLayout(correction_row)
    bind_section_control("statistics", "correction", correction_box)
    design_toolbar = QHBoxLayout()
    add_design_button = QPushButton("添加统计设计")
    add_design_button.setObjectName("primary")
    design_toolbar.addWidget(add_design_button)
    design_toolbar.addStretch(1)
    stats_layout_inner.addLayout(design_toolbar)
    design_host = QWidget()
    design_cards_layout = QVBoxLayout(design_host)
    design_cards_layout.setContentsMargins(0, 0, 0, 0)
    design_cards_layout.setSpacing(10)
    stats_layout_inner.addWidget(design_host)
    design_apply = QPushButton("校验并应用统计设计")
    design_apply.setObjectName("primary")
    stats_layout_inner.addWidget(design_apply, alignment=Qt.AlignmentFlag.AlignRight)
    participant_note = QLabel("在“导入与映射”的统一工作簿中填写参与者表。统计按被试、互动组或会话声明独立单位，不会把重复测量自动视为独立样本。")
    participant_note.setWordWrap(True)
    participant_note.setObjectName("subtle")
    stats_layout_inner.addWidget(participant_note)
    put_content_before_stretch(stats_layout, stats_group)

    widgets.update({"statistics_designs": design_host, "statistics_designs_layout": design_cards_layout,
                    "statistics_design_forms": [], "statistics_add_design": add_design_button,
                    "statistics_apply": design_apply})

    def remove_statistics_form(form: StatisticsDesignForm) -> None:
        if form in widgets["statistics_design_forms"]:
            widgets["statistics_design_forms"].remove(form)
            design_cards_layout.removeWidget(form)
            form.setParent(None)
            form.deleteLater()
            for index, remaining in enumerate(widgets["statistics_design_forms"], start=1):
                remaining._update_heading(index)

    def add_statistics_form(design: dict[str, Any], index: int) -> StatisticsDesignForm:
        form = StatisticsDesignForm(design, index=index, remove_callback=remove_statistics_form,
                                   parent=design_host)
        form.remove_button.setObjectName("danger")
        widgets["statistics_design_forms"].append(form)
        design_cards_layout.addWidget(form)
        return form

    def refresh_statistics_forms() -> None:
        layout = widgets["statistics_designs_layout"]
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        widgets["statistics_design_forms"] = []
        for index, design in enumerate(state["config"]["statistics"].get("designs", []), start=1):
            add_statistics_form(design, index)

    def config_with_analysis_forms() -> dict[str, Any]:
        candidate = copy.deepcopy(state["config"])
        candidate["statistics"]["designs"] = [form.to_config() for form in widgets["statistics_design_forms"]]
        candidate["decomposition"]["driver"] = driver_editor.to_config()
        return candidate

    def add_new_statistics_design() -> None:
        index = len(widgets["statistics_design_forms"]) + 1
        existing_ids = {form.design_id.text().strip() for form in widgets["statistics_design_forms"]}
        design_id = f"design_{index}"
        suffix = index
        while design_id in existing_ids:
            suffix += 1
            design_id = f"design_{suffix}"
        add_statistics_form({"design_id": design_id, "outcome": "", "design_type": "descriptive",
                             "independent_unit": "subject_id"}, index)

    add_design_button.clicked.connect(add_new_statistics_design)
    refresh_statistics_forms()

    def apply_design() -> None:
        try:
            resolved = resolve_config(config_with_analysis_forms())
            count = len(resolved["statistics"]["designs"])
            state["config"] = resolved
            refresh_bound_controls()
            log(f"统计设计校验通过：{count} 项")
        except Exception as error:
            QMessageBox.warning(window, "统计设计未通过", _format_error(error))

    design_apply.clicked.connect(apply_design)

    # ── 结果、产物预览和真实后台运行 ───────────────────────────────────────
    results_page, results_layout = add_page("结果", "显示当前 pipeline 真实返回的 summary、导出表预览、图形和运行目录；不会用示例数据填补缺项。")
    result_controls = QHBoxLayout()
    result_dir_edit = QLineEdit("")
    result_dir_edit.setPlaceholderText("已有运行目录，可选择后执行校验恢复")
    result_browse = QPushButton("选择运行目录…")
    resume_button = QPushButton("校验并恢复")
    run_metadata_button = QPushButton("查看本次运行定义…")
    result_controls.addWidget(result_dir_edit, 1)
    result_controls.addWidget(result_browse)
    result_controls.addWidget(resume_button)
    result_controls.addWidget(run_metadata_button)
    results_layout.insertLayout(max(0, results_layout.count() - 1), result_controls)
    result_summary = QLabel("尚无运行结果")
    result_summary.setWordWrap(True)
    result_summary.setObjectName("subtle")
    put_content_before_stretch(results_layout, result_summary)
    result_revision_status = QLabel("结果快照状态：尚无可比较结果")
    result_revision_status.setWordWrap(True)
    result_revision_status.setObjectName("subtle")
    put_content_before_stretch(results_layout, result_revision_status)
    artifact_controls = QHBoxLayout()
    artifact_combo = QComboBox()
    artifact_combo.setMinimumWidth(420)
    preview_artifact_button = QPushButton("预览所选表")
    copy_artifact_button = QPushButton("导出所选文件副本…")
    open_dir_button = QPushButton("打开运行目录")
    open_html_button = QPushButton("打开 HTML 报告")
    for control in (artifact_combo, preview_artifact_button, copy_artifact_button, open_dir_button, open_html_button):
        artifact_controls.addWidget(control)
    results_layout.insertLayout(max(0, results_layout.count() - 1), artifact_controls)
    artifact_preview_host = QScrollArea()
    artifact_preview_host.setWidgetResizable(True)
    artifact_preview_host.setMinimumHeight(320)
    artifact_preview_host.setWidget(QWidget())
    put_content_before_stretch(results_layout, artifact_preview_host)
    widgets.update({"result_dir": result_dir_edit, "result_browse": result_browse,
                    "resume": resume_button, "result_summary": result_summary,
                    "run_metadata_help": run_metadata_button,
                    "result_revision_status": result_revision_status,
                    "artifact_combo": artifact_combo, "preview_artifact": preview_artifact_button,
                    "copy_artifact": copy_artifact_button, "open_result_dir": open_dir_button,
                    "open_html": open_html_button, "artifact_preview": artifact_preview_host})

    # ── Marker editor: persisted project revision and auditable user IDs ──
    marker_page, marker_layout = add_page(
        "Marker 编辑", "事件标记保存在本地项目快照中。用户事件 ID 与内部稳定 ID 分开保存；编辑后新任务会使用新的 marker 修订快照。")
    marker_actions = QHBoxLayout()
    marker_import_button = QPushButton("导入 CSV…")
    marker_export_button = QPushButton("导出 CSV…")
    marker_undo_button = QPushButton("撤销")
    marker_redo_button = QPushButton("重做")
    marker_refresh_button = QPushButton("刷新")
    marker_conflict = QComboBox()
    marker_conflict.addItem("重复时拒绝", "reject")
    marker_conflict.addItem("显式替换", "replace")
    marker_conflict.addItem("显式重命名", "rename")
    marker_actions.addWidget(marker_import_button)
    marker_actions.addWidget(marker_export_button)
    marker_actions.addWidget(marker_undo_button)
    marker_actions.addWidget(marker_redo_button)
    marker_actions.addWidget(marker_refresh_button)
    marker_actions.addWidget(QLabel("导入冲突"))
    marker_actions.addWidget(marker_conflict)
    put_content_before_stretch(marker_layout, QWidget()) if False else None
    marker_layout.insertLayout(max(0, marker_layout.count() - 1), marker_actions)
    marker_table = QTableWidget(0, 7)
    marker_table.setHorizontalHeaderLabels(["recording_id", "channel", "用户 event_id", "onset_s", "duration_s", "event_type", "condition"])
    marker_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    marker_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    marker_table.setAlternatingRowColors(True)
    marker_table.setMinimumHeight(260)
    marker_layout.insertWidget(max(0, marker_layout.count() - 1), marker_table, 2)
    marker_waveform_group = QGroupBox("波形时间定位 · recording-local 秒")
    marker_waveform_layout = QVBoxLayout(marker_waveform_group)
    marker_waveform_controls = QHBoxLayout()
    marker_record_combo = QComboBox()
    marker_record_combo.setMinimumWidth(360)
    marker_waveform_controls.addWidget(QLabel("当前信号通道"))
    marker_waveform_controls.addWidget(marker_record_combo, 1)
    marker_waveform_layout.addLayout(marker_waveform_controls)
    marker_waveform_status = QLabel("请先在“质控”页异步加载当前信号；点击波形可定位所选 marker 的 onset。")
    marker_waveform_status.setObjectName("subtle")
    marker_waveform_status.setWordWrap(True)
    marker_waveform_layout.addWidget(marker_waveform_status)
    marker_waveform_host = QWidget()
    marker_waveform_host_layout = QVBoxLayout(marker_waveform_host)
    marker_waveform_host_layout.setContentsMargins(0, 0, 0, 0)
    marker_waveform_layout.addWidget(marker_waveform_host)
    marker_layout.insertWidget(max(0, marker_layout.count() - 1), marker_waveform_group, 2)
    marker_form_group = QGroupBox("新增或编辑所选 marker")
    marker_form = QFormLayout(marker_form_group)
    marker_fields = {}
    for name, caption in (("recording_id", "recording_id"), ("channel", "channel（空或 * 表示广播）"),
                          ("source_event_id", "用户 event_id"), ("onset_s", "onset_s"),
                          ("duration_s", "duration_s"), ("event_type", "event_type"),
                          ("condition", "condition"), ("clock_domain", "clock_domain"),
                          ("operator", "操作者"), ("reason", "变更原因")):
        line = QLineEdit()
        if name == "clock_domain":
            line.setText("recording_local")
        if name in {"operator", "reason"}:
            line.setPlaceholderText("必填；用于审计")
        marker_form.addRow(caption, line)
        marker_fields[name] = line
    marker_save_buttons = QHBoxLayout()
    marker_add_button = QPushButton("新增 marker")
    marker_save_button = QPushButton("保存编辑")
    marker_delete_button = QPushButton("删除所选")
    marker_save_buttons.addWidget(marker_add_button)
    marker_save_buttons.addWidget(marker_save_button)
    marker_save_buttons.addWidget(marker_delete_button)
    marker_form.addRow(marker_save_buttons)
    marker_layout.insertWidget(max(0, marker_layout.count() - 1), marker_form_group)
    marker_state_label = QLabel("请先打开项目目录，才能保存 marker。")
    marker_state_label.setObjectName("subtle")
    marker_state_label.setWordWrap(True)
    marker_layout.insertWidget(max(0, marker_layout.count() - 1), marker_state_label)
    widgets.update({"marker_table": marker_table, "marker_import": marker_import_button,
                    "marker_export": marker_export_button, "marker_undo": marker_undo_button,
                    "marker_redo": marker_redo_button, "marker_refresh": marker_refresh_button,
                    "marker_conflict": marker_conflict, "marker_fields": marker_fields,
                    "marker_add": marker_add_button, "marker_save": marker_save_button,
                    "marker_delete": marker_delete_button, "marker_status": marker_state_label,
                    "marker_record_combo": marker_record_combo,
                    "marker_waveform_status": marker_waveform_status,
                    "marker_waveform_host": marker_waveform_host})

    def refresh_markers() -> None:
        project = state.get("project")
        preserve_marker_id = state.get("selected_marker_id")
        marker_table.blockSignals(True)
        marker_table.setRowCount(0)
        if not project:
            marker_table.blockSignals(False)
            state["selected_marker_id"] = None
            marker_state_label.setText("未配置项目；marker 编辑暂不能持久化。请到“项目”页打开或创建项目目录。")
            show_marker_waveform()
            return
        try:
            rows = get_markers(project)
            marker_table.setRowCount(len(rows))
            for row_index, marker in enumerate(rows):
                values = (marker.get("recording_id"), marker.get("channel_scope", marker.get("channel")),
                          marker.get("source_event_id"), marker.get("onset_s"), marker.get("duration_s"),
                          marker.get("event_type"), marker.get("condition"))
                for column, value in enumerate(values):
                    item = QTableWidgetItem("" if value is None else str(value))
                    if column == 0:
                        item.setData(Qt.ItemDataRole.UserRole, marker["marker_id"])
                    marker_table.setItem(row_index, column, item)
            marker_state_label.setText(
                f"已加载 {len(rows)} 个 marker · project r{project.get('project_revision', 0)} · "
                f"marker r{project.get('marker_revision', 0)} · 编辑后任务快照会重新生成")
            marker_undo_button.setEnabled(bool(read_project_snapshot(project).get("marker_undo_stack")))
            marker_redo_button.setEnabled(bool(read_project_snapshot(project).get("marker_redo_stack")))
            marker_table.blockSignals(False)
            matching_row = next((row_index for row_index in range(marker_table.rowCount())
                                 if marker_table.item(row_index, 0) and
                                 marker_table.item(row_index, 0).data(Qt.ItemDataRole.UserRole) == preserve_marker_id), None)
            if matching_row is not None:
                marker_table.setCurrentCell(matching_row, 0)
            else:
                state["selected_marker_id"] = None
                show_marker_waveform()
        except Exception as error:
            marker_table.blockSignals(False)
            marker_state_label.setText(f"读取 marker 失败：{_format_error(error)}")

    state["refresh_markers"] = refresh_markers

    def selected_marker() -> dict[str, Any] | None:
        row_index = marker_table.currentRow()
        if row_index < 0 or marker_table.item(row_index, 0) is None:
            return None
        marker_id = marker_table.item(row_index, 0).data(Qt.ItemDataRole.UserRole)
        return next((item for item in get_markers(state["project"]) if item["marker_id"] == marker_id), None)

    def _marker_broadcast_channel(value: Any) -> bool:
        return str(value or "").strip().casefold() in {"", "na", "nan", "none", "*", "unknown"}

    def _clear_marker_waveform() -> None:
        while marker_waveform_host_layout.count():
            item = marker_waveform_host_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        marker_waveform_host.setMinimumHeight(0)
        state.pop("marker_waveform_canvas", None)
        state.pop("marker_waveform_toolbar", None)
        state.pop("marker_waveform_axes", None)
        state.pop("marker_preview_line", None)

    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _marker_record_identity(record: dict[str, Any]) -> tuple[str, str, str]:
        metadata = record.get("metadata", {})
        return (str(metadata.get("recording_id", "")), str(metadata.get("channel", "")),
                str(metadata.get("clock_domain", "")))

    def _is_current_marker_source() -> bool:
        return (state.get("qc_records_preview_id") is not None and
                state.get("qc_records_config_sha256") == qc_source_config_sha256(state.get("config", {})))

    def refresh_marker_record_choices() -> None:
        previous = marker_record_combo.currentData()
        marker_record_combo.blockSignals(True)
        marker_record_combo.clear()
        records = state.get("records", [])
        for index, record in enumerate(records):
            metadata = record.get("metadata", {})
            label = (f"{metadata.get('recording_id', '')} · {metadata.get('channel', '')} · "
                     f"{metadata.get('clock_domain', 'clock unknown')}")
            marker_record_combo.addItem(label, index)
        selected_index = -1
        if isinstance(previous, int) and 0 <= previous < marker_record_combo.count():
            selected_index = previous
        elif state.get("selected_marker_id"):
            marker = next((item for item in get_markers(state["project"])
                           if item.get("marker_id") == state["selected_marker_id"]), None) if state.get("project") else None
            if marker:
                candidates = []
                for index, record in enumerate(records):
                    recording_id, channel, domain = _marker_record_identity(record)
                    marker_channel = marker.get("channel_scope", marker.get("channel"))
                    if (recording_id == str(marker.get("recording_id", "")) and
                            domain == str(marker.get("clock_domain", "")) and
                            (_marker_broadcast_channel(marker_channel) or channel == str(marker_channel))):
                        candidates.append(index)
                if len(candidates) == 1:
                    selected_index = candidates[0]
        if selected_index < 0 and marker_record_combo.count():
            selected_index = 0
        if selected_index >= 0:
            marker_record_combo.setCurrentIndex(selected_index)
        marker_record_combo.blockSignals(False)
        show_marker_waveform()

    def _render_marker_waveform(record_index: int, request_id: int, source_sha256: str) -> None:
        if request_id != state.get("marker_waveform_request_id"):
            return
        if not _is_current_marker_source():
            marker_waveform_status.setText("输入配置或映射已变更；为避免显示旧通道波形，请到质控页重新加载当前输入。")
            return
        records = state.get("records", [])
        if not (0 <= record_index < len(records)):
            marker_waveform_status.setText("所选记录已不在当前加载结果中；请重新加载当前输入。")
            return
        record = records[record_index]
        metadata = record.get("metadata", {})
        recording_id, channel, clock_domain = _marker_record_identity(record)
        if not clock_domain:
            marker_waveform_status.setText("该信号未声明 clock_domain；为避免把 marker 秒数映射到错误时钟，不能进行波形定位。")
            return
        try:
            time_values = np.asarray(record["time_s"], dtype=float)
            signal = np.asarray(record["eda_uS"], dtype=float)
            if time_values.ndim != 1 or signal.ndim != 1 or len(time_values) != len(signal) or len(time_values) < 2:
                raise ValueError("当前源信号时间与幅值网格不匹配。")
        except Exception as error:
            marker_waveform_status.setText(f"当前源信号不适合波形定位：{_format_error(error)}")
            return
        selected = selected_marker()
        marker_rows = get_markers(state["project"]) if state.get("project") else []
        channel_markers = []
        mismatched_clock = 0
        for marker in marker_rows:
            marker_channel = marker.get("channel_scope", marker.get("channel"))
            if marker.get("recording_id") != recording_id or not (
                    _marker_broadcast_channel(marker_channel) or str(marker_channel) == channel):
                continue
            if str(marker.get("clock_domain", "")) != clock_domain:
                mismatched_clock += 1
                continue
            channel_markers.append(marker)
        compatible_selected = bool(selected and selected.get("recording_id") == recording_id and
                                   ( _marker_broadcast_channel(selected.get("channel_scope", selected.get("channel"))) or
                                    str(selected.get("channel_scope", selected.get("channel"))) == channel) and
                                   str(selected.get("clock_domain", "")) == clock_domain)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.font_manager import FontProperties
            from matplotlib.figure import Figure
            _clear_marker_waveform()
            stride = max(1, int(math.ceil(len(time_values) / 6000)))
            figure = Figure(figsize=(11, 2.8), constrained_layout=True)
            axes = figure.subplots()
            axes.plot(time_values[::stride], signal[::stride], color="#47728F", linewidth=0.85,
                      label="当前源信号 · μS")
            for marker in channel_markers:
                try:
                    onset = float(marker.get("onset_s"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(onset):
                    continue
                is_selected = bool(selected and marker.get("marker_id") == selected.get("marker_id"))
                axes.axvline(onset, color="#C56D43" if is_selected else "#148A8A",
                             linewidth=1.4 if is_selected else 0.9,
                             linestyle="--" if is_selected else ":",
                             label="选中 marker" if is_selected else None)
                try:
                    duration = float(marker.get("duration_s") or 0)
                except (TypeError, ValueError):
                    duration = 0.0
                if duration > 0 and math.isfinite(duration):
                    axes.axvspan(onset, onset + duration, color="#148A8A", alpha=0.07)
            preview_recording = str(marker_fields["recording_id"].text()).strip()
            preview_channel = marker_fields["channel"].text().strip() or "*"
            preview_domain = marker_fields["clock_domain"].text().strip()
            preview_matches = (preview_recording == recording_id and
                               (_marker_broadcast_channel(preview_channel) or preview_channel == channel) and
                               preview_domain == clock_domain)
            if preview_matches:
                try:
                    preview_onset = float(marker_fields["onset_s"].text())
                except (TypeError, ValueError):
                    preview_onset = math.nan
                if math.isfinite(preview_onset):
                    state["marker_preview_line"] = axes.axvline(
                        preview_onset, color="#9A3D39", linewidth=1.8, linestyle="--",
                        label="待保存定位时间")
            axes.set_xlabel(f"recording-local time (s) · clock_domain={clock_domain}")
            axes.set_ylabel("电导 (μS)")
            axes.set_title(f"{recording_id} · channel {channel} · 源 SHA256 {source_sha256[:12]}", loc="left")
            axes.grid(color="#D9E2EA", linewidth=0.5)
            handles, labels = axes.get_legend_handles_labels()
            if handles:
                axes.legend(handles, labels, frameon=False, loc="upper right")
            cjk_path = next((path for path in _cjk_font_candidates() if path.is_file()), None)
            plot_font = FontProperties(fname=str(cjk_path)) if cjk_path else FontProperties(family=ui_family)
            for text_item in figure.findobj(match=lambda item: hasattr(item, "set_fontproperties")):
                try:
                    text_item.set_fontproperties(plot_font)
                except (AttributeError, TypeError):
                    pass
            canvas = FigureCanvasQTAgg(figure)
            # QTAgg's empty sizeHint can otherwise let the canvas collapse to
            # zero height inside the page's resizable scroll-area layout.
            canvas.setMinimumHeight(240)
            marker_waveform_host.setMinimumHeight(270)
            state["marker_waveform_axes"] = axes
            state["marker_waveform_canvas"] = canvas

            def locate_on_waveform(event: Any) -> None:
                if event.inaxes is not axes or event.button != 1 or event.xdata is None:
                    return
                if request_id != state.get("marker_waveform_request_id") or not _is_current_marker_source():
                    marker_waveform_status.setText("输入来源已变化；本次定位已取消，请重新加载当前波形。")
                    return
                onset = float(event.xdata)
                if not (float(time_values[0]) <= onset <= float(time_values[-1])):
                    marker_waveform_status.setText("点击位置超出当前记录时间范围，未修改 marker。")
                    return
                if selected and not compatible_selected:
                    marker_waveform_status.setText(
                        "当前选中 marker 与波形的 recording/channel/clock_domain 不一致；未修改，不能自动换算。")
                    return
                if selected is None:
                    marker_fields["recording_id"].setText(recording_id)
                    marker_fields["channel"].setText(channel)
                    marker_fields["clock_domain"].setText(clock_domain)
                marker_fields["onset_s"].setText(f"{onset:.6f}")
                preview_line = state.get("marker_preview_line")
                if preview_line is not None:
                    try:
                        preview_line.remove()
                    except (AttributeError, ValueError):
                        pass
                state["marker_preview_line"] = axes.axvline(
                    onset, color="#9A3D39", linewidth=1.8, linestyle="--",
                    label="待保存定位时间")
                canvas.draw_idle()
                action = "选中 marker 的 onset 已定位" if selected else "新 marker 的 onset 已定位"
                marker_waveform_status.setText(
                    f"{action}到 {onset:.6f} s · 时钟 {clock_domain}。请点击“保存编辑”或“新增 marker”提交修订。")

            canvas.mpl_connect("button_press_event", locate_on_waveform)
            toolbar = NavigationToolbar2QT(canvas, window)
            marker_waveform_host_layout.addWidget(toolbar)
            marker_waveform_host_layout.addWidget(canvas)
            state["marker_waveform_toolbar"] = toolbar
            canvas.draw_idle()
            if selected and not compatible_selected:
                message = "当前选中 marker 属于其他 recording/channel/clock_domain；波形仅供查看，不能直接定位该 marker。"
            else:
                message = ("点击波形可定位 onset；保存按钮才会写入新的 marker 修订。"
                           if compatible_selected or selected is None else "请选择同源 marker 后再定位。")
            if mismatched_clock:
                message += f" 另有 {mismatched_clock} 个 marker 属于不同 clock_domain，未转换也未显示。"
            marker_waveform_status.setText(
                f"已校验当前输入文件 SHA256；{message}  坐标：{recording_id} / {channel} / {clock_domain}。")
        except Exception as error:
            marker_waveform_status.setText(f"波形绘制失败：{_format_error(error)}")

    def show_marker_waveform(*_args: Any) -> None:
        request_id = int(state.get("marker_waveform_request_id", 0)) + 1
        state["marker_waveform_request_id"] = request_id
        state["marker_waveform_pending"] = None
        _clear_marker_waveform()
        records = state.get("records", [])
        if not records:
            marker_waveform_status.setText("尚未载入信号；请到“质控”页异步加载当前输入后再定位 marker。")
            return
        if not _is_current_marker_source():
            marker_waveform_status.setText("当前输入配置或映射已变化；旧波形不会用于 marker 编辑，请重新加载质控记录。")
            return
        record_index = marker_record_combo.currentData()
        if isinstance(record_index, bool) or not isinstance(record_index, int) or not (0 <= record_index < len(records)):
            marker_waveform_status.setText("请选择有效 recording/channel。")
            return
        metadata = records[record_index].get("metadata", {})
        source_path = metadata.get("source_path")
        expected_sha256 = str(metadata.get("source_sha256", "")).lower()
        if not source_path or len(expected_sha256) != 64:
            marker_waveform_status.setText("当前记录缺少源路径或 SHA256；为避免显示来源不明的波形，定位不可用。")
            return
        marker_waveform_status.setText(
            f"正在后台核对当前源文件并准备绘制 {metadata.get('recording_id')} / {metadata.get('channel')}…")
        future = state["io_executor"].submit(_sha256_file, str(Path(source_path).resolve()))
        state["marker_waveform_pending"] = {
            "request_id": request_id, "record_index": record_index,
            "expected_sha256": expected_sha256, "future": future,
        }

    def poll_marker_waveform() -> None:
        pending = state.get("marker_waveform_pending")
        if not isinstance(pending, dict) or not pending["future"].done():
            return
        state["marker_waveform_pending"] = None
        if pending["request_id"] != state.get("marker_waveform_request_id"):
            return
        try:
            observed_sha256 = str(pending["future"].result()).lower()
        except Exception as error:
            marker_waveform_status.setText(f"源信号读取失败；没有绘制波形：{_format_error(error)}")
            return
        if observed_sha256 != pending["expected_sha256"]:
            marker_waveform_status.setText(
                "源文件内容已变化；为防止把旧通道或旧时间轴画成当前来源，本次未显示波形。请重新载入质控记录。")
            return
        if not _is_current_marker_source():
            marker_waveform_status.setText("输入配置已变化；本次波形请求已过期，没有绘制。请重新加载当前输入。")
            return
        _render_marker_waveform(pending["record_index"], pending["request_id"], observed_sha256)

    marker_record_combo.currentIndexChanged.connect(show_marker_waveform)

    def select_marker_row(*_args: Any) -> None:
        marker = selected_marker()
        if marker is None:
            state["selected_marker_id"] = None
            show_marker_waveform()
            return
        state["selected_marker_id"] = marker.get("marker_id")
        values = {"recording_id": marker.get("recording_id"),
                  "channel": marker.get("channel_scope", marker.get("channel")),
                  "source_event_id": marker.get("source_event_id"), "onset_s": marker.get("onset_s"),
                  "duration_s": marker.get("duration_s"), "event_type": marker.get("event_type"),
                  "condition": marker.get("condition"), "clock_domain": marker.get("clock_domain")}
        for name, value in values.items():
            marker_fields[name].setText("" if value is None else str(value))
        refresh_marker_record_choices()

    marker_table.itemSelectionChanged.connect(select_marker_row)

    def marker_operator_reason() -> tuple[str, str]:
        operator = marker_fields["operator"].text().strip()
        reason = marker_fields["reason"].text().strip()
        if not operator or not reason:
            raise ValueError("请填写操作者和变更原因。")
        return operator, reason

    def configured_source_event_conflict(marker: dict[str, Any]) -> str | None:
        configured = state.get("config", {}).get("input", {}).get("events")
        if not isinstance(configured, str) or not configured.strip():
            return None
        source_path = Path(configured).expanduser()
        if not source_path.is_absolute():
            base = state.get("config", {}).get("input", {}).get("base_directory")
            if not base:
                project = state.get("project")
                base = project.get("path") if project else None
            if base:
                source_path = Path(base).expanduser() / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"configured_events_file_missing:{source_path}")
        rows, _report = read_marker_csv(source_path)
        target_recording = str(marker.get("recording_id", ""))
        target_event_id = str(marker.get("source_event_id") or marker.get("event_id") or "")
        target_channel = str(marker.get("channel", "*") or "*").strip()
        def normalized_channel(value: Any) -> str:
            text = "" if value is None else str(value).strip()
            return "*" if not text or text.casefold() in {"na", "nan", "null", "*"} else text
        target_channel = normalized_channel(target_channel)
        for source_row in rows:
            source_id = str(source_row.get("source_event_id") or source_row.get("event_id") or "")
            if str(source_row.get("recording_id", "")) != target_recording or source_id != target_event_id:
                continue
            source_channel = normalized_channel(source_row.get("channel"))
            if target_channel == "*" or source_channel == "*" or target_channel == source_channel:
                return str(source_path)
        return None

    def marker_import_action() -> None:
        project = state.get("project")
        if not project:
            QMessageBox.warning(window, "项目未打开", "marker 导入需要先打开本地项目目录。")
            return
        filename, _ = QFileDialog.getOpenFileName(window, "导入事件 marker CSV", "", "CSV (*.csv);;所有文件 (*.*)")
        if not filename:
            return
        try:
            operator, reason = marker_operator_reason()
            rows, _source = read_marker_csv(filename)
            persist_project_draft()
            result = import_markers(project, rows, operator=operator, source_path=filename,
                                    conflict_policy=marker_conflict.currentData(), reason=reason,
                                    default_clock_domain=state["config"]["mapping"].get("clock_domain", "recording_local"))
            refresh_markers()
            update_result_freshness()
            schedule_project_autosave()
            log(f"marker 导入完成：新增/变更 {result.get('imported_count', 0)} · 修订 {result.get('marker_revision', 0)}")
        except Exception as error:
            QMessageBox.warning(window, "marker 导入失败", _format_error(error))

    def marker_export_action() -> None:
        project = state.get("project")
        if not project:
            QMessageBox.information(window, "项目未打开", "先打开项目后才能导出持久化 marker。")
            return
        filename, _ = QFileDialog.getSaveFileName(window, "导出事件 marker CSV", "events-markers.csv", "CSV (*.csv)")
        if filename:
            try:
                export_marker_snapshot(project, filename)
                set_status(f"marker 已导出：{Path(filename).name}")
                log(f"marker 导出：{filename}")
            except Exception as error:
                QMessageBox.critical(window, "marker 导出失败", _format_error(error))

    def marker_add_action() -> None:
        project = state.get("project")
        if not project:
            QMessageBox.warning(window, "项目未打开", "请先打开项目目录。")
            return
        try:
            operator, reason = marker_operator_reason()
            row = {"recording_id": marker_fields["recording_id"].text().strip(),
                   "channel": marker_fields["channel"].text().strip() or "*",
                   "event_id": marker_fields["source_event_id"].text(),
                   "onset_s": marker_fields["onset_s"].text(),
                   "duration_s": marker_fields["duration_s"].text() or 0,
                   "event_type": marker_fields["event_type"].text().strip(),
                   "condition": marker_fields["condition"].text().strip() or None,
                   "clock_domain": marker_fields["clock_domain"].text().strip()}
            conflict_path = configured_source_event_conflict(row)
            if conflict_path:
                raise ValueError(
                    f"事件 ID {row['event_id']} 已存在于配置的只读 events CSV（{conflict_path}）。"
                    "请先导入该 CSV 中的 marker，再编辑现有行；或使用其他 event ID。"
                )
            persist_project_draft()
            result = import_markers(project, [row], operator=operator, conflict_policy="reject", reason=reason)
            refresh_markers()
            update_result_freshness()
            log(f"已新增 marker {row['event_id']}，marker 修订 {result.get('marker_revision', 0)}")
        except Exception as error:
            QMessageBox.warning(window, "marker 未新增", _format_error(error))

    def marker_save_action() -> None:
        project = state.get("project")
        marker = selected_marker()
        if not project or not marker:
            QMessageBox.information(window, "未选择 marker", "请先选择一个 marker 行。")
            return
        try:
            operator, reason = marker_operator_reason()
            changes = {"recording_id": marker_fields["recording_id"].text().strip(),
                       "channel": marker_fields["channel"].text().strip() or "*",
                       "source_event_id": marker_fields["source_event_id"].text(),
                       "onset_s": marker_fields["onset_s"].text(),
                       "duration_s": marker_fields["duration_s"].text(),
                       "event_type": marker_fields["event_type"].text().strip(),
                       "condition": marker_fields["condition"].text().strip() or None,
                       "clock_domain": marker_fields["clock_domain"].text().strip()}
            persist_project_draft()
            result = update_marker(project, marker["marker_id"], changes, operator=operator, reason=reason)
            refresh_markers()
            update_result_freshness()
            log(f"marker {changes['source_event_id']} 已保存；稳定内部 ID 保持不变，修订 {result.get('marker_revision', 0)}")
        except Exception as error:
            QMessageBox.warning(window, "marker 未保存", _format_error(error))

    def marker_delete_action() -> None:
        project = state.get("project")
        marker = selected_marker()
        if not project or not marker:
            return
        try:
            operator, reason = marker_operator_reason()
            delete_marker(project, marker["marker_id"], operator=operator, reason=reason)
            refresh_markers()
            update_result_freshness()
            log(f"已删除 marker {marker.get('source_event_id')}；变更已追加到审计历史")
        except Exception as error:
            QMessageBox.warning(window, "marker 未删除", _format_error(error))

    def marker_history_action(operation: str) -> None:
        project = state.get("project")
        if not project:
            return
        try:
            operator, reason = marker_operator_reason()
            fn = undo_marker_action if operation == "undo" else redo_marker_action
            fn(project, operator=operator, reason=reason)
            refresh_markers()
            update_result_freshness()
            log(f"marker {operation} 已追加审计修订")
        except Exception as error:
            QMessageBox.information(window, f"marker {operation} 不可用", _format_error(error))

    marker_import_button.clicked.connect(marker_import_action)
    marker_export_button.clicked.connect(marker_export_action)
    marker_add_button.clicked.connect(marker_add_action)
    marker_save_button.clicked.connect(marker_save_action)
    marker_delete_button.clicked.connect(marker_delete_action)
    marker_undo_button.clicked.connect(lambda: marker_history_action("undo"))
    marker_redo_button.clicked.connect(lambda: marker_history_action("redo"))
    marker_refresh_button.clicked.connect(refresh_markers)

    # ── Persistent task history and project-wide log search ────────────────
    task_page, task_layout = add_page(
        "任务与日志", "运行前预检和结果快照会绑定项目 revision。排队任务不会随应用启动自动执行；请使用“开始队列”显式启动。")
    task_controls = QHBoxLayout()
    task_refresh_button = QPushButton("刷新历史")
    task_up_button = QPushButton("队列上移")
    task_down_button = QPushButton("队列下移")
    task_cancel_button = QPushButton("取消选中任务")
    task_resume_button = QPushButton("恢复失败/中断任务")
    task_retry_button = QPushButton("新运行重试")
    task_open_result_button = QPushButton("查看选中结果")
    task_open_result_button.setEnabled(False)
    for control in (task_refresh_button, task_up_button, task_down_button, task_cancel_button,
                    task_resume_button, task_retry_button, task_open_result_button):
        task_controls.addWidget(control)
    task_layout.insertLayout(max(0, task_layout.count() - 1), task_controls)
    task_table = QTableWidget(0, 5)
    task_table.setHorizontalHeaderLabels(["序号", "状态", "task_id", "创建时间", "run_id"])
    task_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    task_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    task_table.setAlternatingRowColors(True)
    task_table.setMinimumHeight(230)
    task_layout.insertWidget(max(0, task_layout.count() - 1), task_table, 2)
    attempt_detail = QPlainTextEdit()
    attempt_detail.setReadOnly(True)
    attempt_detail.setMaximumHeight(120)
    task_layout.insertWidget(max(0, task_layout.count() - 1), attempt_detail)
    session_log_group = QGroupBox("本次会话消息")
    session_log_layout = QVBoxLayout(session_log_group)
    session_log_layout.setContentsMargins(8, 10, 8, 8)
    session_log_layout.setSpacing(4)
    session_log_layout.addWidget(log_header)
    session_log_layout.addWidget(log_box)
    task_layout.insertWidget(max(0, task_layout.count() - 1), session_log_group)
    widgets["session_log_group"] = session_log_group
    log_group = QGroupBox("项目持久化日志")
    log_layout = QVBoxLayout(log_group)
    log_search_row = QHBoxLayout()
    log_query = QLineEdit()
    log_query.setPlaceholderText("全文关键字")
    log_level = QComboBox()
    log_level.addItems(["全部级别", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    log_task_filter = QLineEdit()
    log_task_filter.setPlaceholderText("task_id（可选）")
    log_run_filter = QLineEdit()
    log_run_filter.setPlaceholderText("run_id（可选）")
    log_search_button = QPushButton("搜索日志")
    log_next_button = QPushButton("下一页")
    log_export_button = QPushButton("导出日志…")
    for control in (log_query, log_level, log_task_filter, log_run_filter,
                    log_search_button, log_next_button, log_export_button):
        log_search_row.addWidget(control)
    log_layout.addLayout(log_search_row)
    log_results = QPlainTextEdit()
    log_results.setReadOnly(True)
    log_results.setMaximumHeight(210)
    log_layout.addWidget(log_results)
    task_layout.insertWidget(max(0, task_layout.count() - 1), log_group)
    task_status_label = QLabel("未配置项目；任务历史与日志尚不可用。")
    task_status_label.setObjectName("subtle")
    task_layout.insertWidget(max(0, task_layout.count() - 1), task_status_label)
    widgets.update({"task_table": task_table, "task_refresh": task_refresh_button,
                    "task_up": task_up_button, "task_down": task_down_button,
                    "task_cancel": task_cancel_button, "task_resume": task_resume_button,
                    "task_retry": task_retry_button, "task_open_result": task_open_result_button,
                    "task_detail": attempt_detail,
                    "task_status": task_status_label, "log_query": log_query, "log_level": log_level,
                    "log_task_filter": log_task_filter, "log_run_filter": log_run_filter,
                    "log_search": log_search_button, "log_next": log_next_button,
                    "log_export": log_export_button, "log_results": log_results})

    def refresh_tasks() -> None:
        project = state.get("project")
        task_table.setRowCount(0)
        if not project:
            task_status_label.setText("未配置项目；任务历史与日志尚不可用。")
            return
        try:
            rows = list_tasks(project, limit=500)
            task_table.setRowCount(len(rows))
            for row_index, task in enumerate(rows):
                values = (task.get("ordinal"), task.get("state"), task.get("task_id"),
                          task.get("created_at"), task.get("run_id"))
                for column, value in enumerate(values):
                    item = QTableWidgetItem("" if value is None else str(value))
                    if column == 2:
                        item.setData(Qt.ItemDataRole.UserRole, task["task_id"])
                    task_table.setItem(row_index, column, item)
            active = next((task for task in rows if task.get("state") in
                           {"claimed", "running", "cancellation_requested"}), None)
            state["active_task_id"] = active.get("task_id") if active else None
            task_status_label.setText(
                f"{len(rows)} 个持久任务 · 活动任务 {state.get('active_task_id') or '无'} · "
                f"恢复检查 {state.get('task_reconciliation', {}).get('status', '未执行')} · 队列不自动启动")
        except Exception as error:
            task_status_label.setText(f"读取任务历史失败：{_format_error(error)}")

    state["refresh_tasks"] = refresh_tasks

    def selected_task_id() -> str | None:
        row_index = task_table.currentRow()
        if row_index < 0 or task_table.item(row_index, 2) is None:
            return None
        return task_table.item(row_index, 2).data(Qt.ItemDataRole.UserRole)

    def show_selected_task(*_args: Any) -> None:
        project = state.get("project")
        task_id = selected_task_id()
        if not project or not task_id:
            return
        state["selected_task_id"] = task_id
        task = get_task(project, task_id)
        if not task:
            attempt_detail.setPlainText("所选任务已不存在；请刷新历史。")
            task_open_result_button.setEnabled(False)
            return
        attempts = list_task_attempts(project, task_id)
        attempt_detail.setPlainText(json.dumps({"task": task, "attempts": attempts},
                                               ensure_ascii=False, indent=2, default=str)[:20000])
        preferred_attempt = next((item for item in reversed(attempts)
                                  if item.get("attempt_id") == task.get("active_attempt_id") and item.get("run_directory")), None)
        latest_attempt = preferred_attempt or next((item for item in reversed(attempts)
                                                    if item.get("run_directory")), None)
        result_directory = Path(latest_attempt["run_directory"]) if latest_attempt else None
        task_open_result_button.setEnabled(bool(result_directory and result_directory.is_dir()))

    task_table.itemSelectionChanged.connect(show_selected_task)

    def reorder_selected(delta: int) -> None:
        project = state.get("project")
        task_id = selected_task_id()
        if not project or not task_id:
            return
        task = get_task(project, task_id)
        if not task or task.get("state") != "queued":
            return
        try:
            reorder_queued_task(project, task_id, max(1, int(task["ordinal"]) + delta))
            refresh_tasks()
        except Exception as error:
            QMessageBox.warning(window, "队列顺序未更改", _format_error(error))

    def cancel_selected_task() -> None:
        project, task_id = state.get("project"), selected_task_id()
        if project and task_id:
            try:
                request_task_cancel(project, task_id)
                refresh_tasks()
                log(f"已请求取消任务 {task_id}；运行会在安全检查点停止。")
            except Exception as error:
                QMessageBox.warning(window, "无法取消任务", _format_error(error))

    def retry_selected_task(mode: str) -> None:
        project, task_id = state.get("project"), selected_task_id()
        if not project or not task_id:
            return
        try:
            attempt_id = retry_task(project, task_id, mode=mode)
            refresh_tasks()
            log(f"已为任务创建显式 {mode} attempt {attempt_id}；仍需点击“开始队列”。")
        except Exception as error:
            QMessageBox.warning(window, "无法创建重试", _format_error(error))

    task_refresh_button.clicked.connect(refresh_tasks)
    task_up_button.clicked.connect(lambda: reorder_selected(-1))
    task_down_button.clicked.connect(lambda: reorder_selected(1))
    task_cancel_button.clicked.connect(cancel_selected_task)
    task_resume_button.clicked.connect(lambda: retry_selected_task("resume"))
    task_retry_button.clicked.connect(lambda: retry_selected_task("retry"))

    def begin_log_search(*, next_page: bool = False) -> None:
        project = state.get("project")
        if not project:
            log_results.setPlainText("未配置项目；日志保存在项目目录 .eda/logs 中。")
            return
        if state.get("log_search_future") and not state["log_search_future"].done():
            return
        if not next_page:
            state["log_cursor"] = None
        cursor = state.get("log_cursor") if next_page else None
        query = log_query.text().strip()
        selected_level = log_level.currentText()
        levels = None if selected_level == "全部级别" else [selected_level]
        task_filter = log_task_filter.text().strip() or None
        run_filter = log_run_filter.text().strip() or None
        state["log_search_future"] = state["io_executor"].submit(
            search_logs, {"path": project["path"]}, query, levels, task_filter, run_filter, 500, cursor)
        state["log_search_append"] = bool(next_page)

    def export_project_logs() -> None:
        project = state.get("project")
        if not project:
            return
        filename, selected_filter = QFileDialog.getSaveFileName(window, "导出项目日志", "runtime-log.jsonl",
                                                                 "JSONL (*.jsonl);;文本 (*.txt)")
        if not filename:
            return
        format_name = "text" if filename.casefold().endswith(".txt") or selected_filter.startswith("文本") else "jsonl"
        if state.get("log_export_future") is not None and not state["log_export_future"].done():
            return
        state["log_export_target"] = filename
        state["log_export_context"] = {"project_id": project.get("project_id"), "project_path": project["path"]}
        state["log_export_future"] = state["io_executor"].submit(
            export_logs, {"path": project["path"]}, format_name, filename,
            query=log_query.text().strip(),
            levels=None if log_level.currentText() == "全部级别" else [log_level.currentText()],
            task_id=log_task_filter.text().strip() or None,
            run_id=log_run_filter.text().strip() or None)
        log_export_button.setEnabled(False)
        set_status("日志正在后台导出；界面仍可继续操作。")
        log(f"已在后台开始导出项目日志：{filename}")

    log_search_button.clicked.connect(lambda: begin_log_search())
    log_next_button.clicked.connect(lambda: begin_log_search(next_page=True))
    log_export_button.clicked.connect(export_project_logs)

    def selected_run_directory() -> Path | None:
        text = result_dir_edit.text().strip()
        if text:
            path = Path(text).expanduser()
            if path.is_dir():
                return path.resolve()
            return None
        summary = state.get("last_summary")
        if isinstance(summary, dict) and summary.get("directory"):
            return Path(summary["directory"]).resolve()
        return None

    def open_run_metadata_help() -> None:
        """Show definitions and table schemas frozen with the selected run."""
        from html import escape

        directory = selected_run_directory()
        if directory is None:
            QMessageBox.information(window, "尚无运行定义", "先打开一个已有运行目录或从任务历史打开结果。")
            return

        def read_json(filename: str) -> dict[str, Any] | None:
            path = directory / filename
            if not path.is_file():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                return None
            return value if isinstance(value, dict) else None

        manifest = read_json("manifest.json")
        provenance = read_json("provenance.json") or {}
        frozen_config = read_json("resolved_config.json")
        definitions = read_json("feature_definitions.json")
        if definitions is None:
            candidate = provenance.get("feature_definitions")
            definitions = candidate if isinstance(candidate, dict) else None
        verification_text = "未找到 manifest.json，不能验证该运行目录的文件完整性。"
        if manifest is not None:
            try:
                from ..core_export import verify_manifest
                verification = verify_manifest(directory)
                verification_text = ("manifest 完整性：通过。" if verification.get("valid") else
                                     f"manifest 完整性：未通过，{len(verification.get('failures', []))} 项缺失或损坏。")
            except Exception as error:
                verification_text = f"manifest 校验出错：{_format_error(error)}"

        run_id = str((manifest or {}).get("run_id") or provenance.get("run_id") or "未知")
        pipeline_id = str((frozen_config or {}).get("pipeline_id") or "未保存")
        schema_version = str((frozen_config or {}).get("schema_version") or "未保存")
        frozen_ids = ((frozen_config or {}).get("features") or {}).get("ids", None)
        if frozen_ids == []:
            selection_text = "features.ids=[]：本次运行使用其冻结配置当时全部适用指标。"
        elif isinstance(frozen_ids, list):
            selection_text = f"本次运行显式选择 {len(frozen_ids)} 项指标。"
        else:
            selection_text = "旧运行未保存可识别的指标选择字段。"

        paragraphs = [
            f"<p><b>运行 ID：</b>{escape(run_id)}<br><b>配置预设：</b>{escape(pipeline_id)}；schema {escape(schema_version)}<br>{escape(verification_text)}</p>",
            "<p>以下定义读取自该运行目录的 <code>feature_definitions.json</code>（旧运行可读 provenance 中的冻结副本）。"
            "不会使用当前草稿或当前安装的 registry 反推旧动态 band/lag 指标。</p>",
            f"<p>{escape(selection_text)}</p>",
        ]
        if frozen_config is None:
            paragraphs.append("<p><b>配置快照缺失：</b>无法解释该旧运行的参数。请勿用当前编辑器配置替代。</p>")
        if not isinstance(definitions, dict):
            paragraphs.append("<p><b>冻结指标定义缺失：</b>该运行没有保存逐指标定义，因此这里只能查看实际表头，不能从当前草稿补写含义。</p>")
        else:
            rows = []
            for feature_id, definition in sorted(definitions.items(), key=lambda pair: str(pair[0])):
                if not isinstance(definition, dict):
                    continue
                label = str(definition.get("label_zh") or feature_id)
                parameters = json.dumps(definition.get("parameters", {}), ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":"))
                support = f"N≥{definition.get('min_n', '')}; 时长≥{definition.get('min_duration_s', '')} s"
                cells = [label, str(feature_id), str(definition.get("unit", "")),
                         str(definition.get("basis", "")), str(definition.get("formula", "")),
                         support, parameters, str(definition.get("validity", "")),
                         str(definition.get("qualification", ""))]
                rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in cells) + "</tr>")
            paragraphs.append(
                "<h3>本次运行采用的冻结指标定义</h3>"
                "<table border='1' cellspacing='0' cellpadding='4'><thead><tr>"
                "<th>指标</th><th>ID</th><th>单位</th><th>basis</th><th>定义</th><th>最低支持</th>"
                "<th>本次参数</th><th>有效性</th><th>资格</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
            )

        table_descriptions = {
            "trial_table": "逐事件试次及其响应、基线、归属与排除状态。",
            "trial_summaries": "按被试、会话、条件和事件类型汇总计划/有效/排除试次数。",
            "trial_fit_diagnostics": "事件序列趋势模型的拟合状态、参数和诊断。",
            "spectra": "Welch 频谱估计的频率 bin、PSD、采样率与参数。",
            "complexity_diagnostics": "自相关与复杂度指标的参数、状态及诊断。",
            "ns_classification": "逐 SCR 的非特异/事件相关分类及注释覆盖理由。",
            "scr_events_extended": "SCR 候选目录及扩展斜率、恢复和半高宽字段。",
            "event_scr_links": "事件与 SCR 的唯一/歧义归属关系。",
            "group_temporal_features": "固定成员/边集合的群体时序摘要。",
            "sync_episodes": "正/负同步连续时段及其删失状态。",
            "node_strength": "节点固定图上的有符号、正、负和绝对强度。",
            "group_temporal_diagnostics": "共同分析时长、有效/未知支持及固定成员/边诊断。",
        }
        artifacts = (manifest or {}).get("artifacts", [])
        table_rows = []
        if isinstance(artifacts, list):
            for artifact in artifacts:
                relative = artifact.get("path") if isinstance(artifact, dict) else None
                if not isinstance(relative, str) or not relative.replace("\\", "/").startswith("tables/"):
                    continue
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory.resolve()) or not path.is_file():
                    continue
                stem = path.stem
                key = stem.rsplit("_", 1)[0] if "_" in stem else stem
                description = table_descriptions.get(key, "结果表；请结合本次运行冻结的状态、单位和定义解释。")
                columns: list[str] = []
                if path.suffix.casefold() == ".csv":
                    try:
                        with path.open("r", encoding="utf-8-sig", newline="") as stream:
                            columns = next(csv.reader(stream), [])
                    except Exception:
                        columns = []
                elif path.suffix.casefold() == ".json":
                    try:
                        with path.open("rb") as stream:
                            prefix = stream.read(262144).decode("utf-8-sig")
                        object_start = prefix.find("{")
                        if object_start >= 0:
                            first_row, _ = json.JSONDecoder().raw_decode(prefix, object_start)
                            if isinstance(first_row, dict):
                                columns = list(first_row)
                    except Exception:
                        columns = []
                detail = f"列：{', '.join(columns)}" if columns else "未能从表文件读取列名。"
                table_rows.append(
                    f"<tr><td>{escape(relative)}</td><td>{escape(description)}<br><small>{escape(detail)}</small></td></tr>"
                )
        if table_rows:
            paragraphs.append(
                "<h3>本次运行实际生成的表</h3><table border='1' cellspacing='0' cellpadding='4'>"
                "<thead><tr><th>运行目录中的文件</th><th>内容说明与实际列</th></tr></thead><tbody>"
                + "".join(table_rows) + "</tbody></table>"
            )
        else:
            paragraphs.append("<h3>本次运行实际生成的表</h3><p>manifest 未列出 tables/ 下的表文件。</p>")

        dialog = QDialog(window)
        dialog.setWindowTitle(f"本次运行定义 · {run_id}")
        dialog.resize(1100, 760)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser(dialog)
        browser.setOpenLinks(False)
        browser.setHtml("<html><body>" + "".join(paragraphs) + "</body></html>")
        layout.addWidget(browser, 1)
        close_button = QPushButton("关闭", dialog)
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        state["run_metadata_dialog"] = dialog
        state["run_metadata_browser"] = browser
        dialog.exec()

    run_metadata_button.clicked.connect(open_run_metadata_help)

    def update_result_freshness() -> None:
        """Compare the displayed run's frozen marker/QC revisions with the live project."""
        summary = state.get("last_summary")
        if not isinstance(summary, dict):
            result_revision_status.setText("结果快照状态：尚无可比较结果")
            result_revision_status.setStyleSheet("color: #5D7182;")
            return
        frozen = summary.get("project_snapshot")
        required = ("marker_revision", "qc_revision")
        if not isinstance(frozen, dict) or any(
                isinstance(frozen.get(field), bool) or not isinstance(frozen.get(field), int)
                or frozen.get(field) < 0 for field in required):
            result_revision_status.setText(
                "结果快照状态：无法核对（该旧结果未记录 marker/QC 修订；工件仍可查看）")
            result_revision_status.setStyleSheet("color: #8A5A22; font-weight: 650;")
            return
        project = state.get("project")
        if not project:
            result_revision_status.setText("结果快照状态：未打开项目，无法与当前 marker/QC 修订核对")
            result_revision_status.setStyleSheet("color: #8A5A22; font-weight: 650;")
            return
        try:
            current = open_project(project["path"])
            state["project"] = current
        except Exception as error:
            result_revision_status.setText(f"结果快照状态：读取当前项目修订失败 · {_format_error(error)}")
            result_revision_status.setStyleSheet("color: #9A3D39; font-weight: 650;")
            return
        current_values = {field: current.get(field) for field in required}
        if frozen.get("project_id") and frozen.get("project_id") != current.get("project_id"):
            result_revision_status.setText("结果快照状态：来自其他项目；当前项目修订不可比较，旧工件仍保留")
            result_revision_status.setStyleSheet("color: #8A5A22; font-weight: 650;")
            return
        stale = [field for field in required if frozen[field] != current_values[field]]
        revision_text = (f"运行快照 marker r{frozen['marker_revision']} / QC r{frozen['qc_revision']}；"
                         f"当前项目 marker r{current_values['marker_revision']} / QC r{current_values['qc_revision']}。")
        if stale:
            result_revision_status.setText("结果快照状态：已过期（" + revision_text + "旧工件保留，可按原快照查看/复现。）")
            result_revision_status.setStyleSheet("color: #9A3D39; font-weight: 700;")
        else:
            result_revision_status.setText("结果快照状态：与当前 marker/QC 修订一致（" + revision_text + "）")
            result_revision_status.setStyleSheet("color: #087E83; font-weight: 650;")

    def update_result_summary(summary: dict[str, Any]) -> None:
        state["last_summary"] = summary
        directory = summary.get("directory", "")
        if directory:
            result_dir_edit.setText(directory)
        statuses = summary.get("record_statuses", [])
        completed = summary.get("completed_records", len([x for x in statuses if x.get("status") == "succeeded"]))
        result_summary.setText("\n".join([
            f"运行状态：{summary.get('status', 'unknown')}  ·  run_id={summary.get('run_id', '')}",
            f"已完成记录：{completed}  ·  耗时：{summary.get('elapsed_s', 0):.2f} s" if isinstance(summary.get("elapsed_s"), (int, float)) else f"已完成记录：{completed}",
            f"科学资格：{summary.get('science_qualification', 'pending_representative_real_data')}",
            f"目录：{directory}",
            f"错误 {len(summary.get('errors', []))} 项 · 警告 {len(summary.get('warnings', []))} 项",
        ]))
        artifact_combo.clear()
        state["artifact_paths"] = {}
        for artifact in summary.get("artifacts", []):
            relative = artifact.get("path") if isinstance(artifact, dict) else None
            if not relative:
                continue
            absolute = (Path(directory) / relative).resolve()
            state["artifact_paths"][relative] = absolute
            artifact_combo.addItem(relative, relative)
        if not artifact_combo.count() and directory:
            for candidate in ("summary.json", "provenance.json", "report.html"):
                if (Path(directory) / candidate).is_file():
                    state["artifact_paths"][candidate] = Path(directory) / candidate
                    artifact_combo.addItem(candidate, candidate)
        resume_button.setEnabled(summary.get("status") in ("partial", "failed", "cancelled"))
        update_result_freshness()
        for error in summary.get("errors", []):
            log(f"运行诊断：{error.get('code', '')} · {error.get('message', error)}")

    def open_selected_task_result() -> None:
        project = state.get("project")
        task_id = selected_task_id()
        if not project or not task_id:
            return
        task = get_task(project, task_id)
        if not task:
            task_status_label.setText("所选任务已不存在；请刷新历史。")
            return
        attempts = list_task_attempts(project, task_id)
        attempt = next((item for item in reversed(attempts)
                        if item.get("attempt_id") == task.get("active_attempt_id") and item.get("run_directory")), None)
        attempt = attempt or next((item for item in reversed(attempts) if item.get("run_directory")), None)
        if not attempt:
            task_status_label.setText("所选任务还没有结果目录。")
            return
        directory = Path(attempt["run_directory"]).expanduser().resolve()
        if not directory.is_dir():
            task_status_label.setText(f"所选任务结果目录不存在：{directory}")
            return
        manifest_path = directory / "manifest.json"
        manifest_payload: dict[str, Any] = {}
        saved_summary: dict[str, Any] = {}
        verified = {"valid": False, "failures": [{"reason": "manifest_missing"}]}
        try:
            from ..core_export import verify_manifest
            if manifest_path.is_file():
                verified = verify_manifest(directory)
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary_path = directory / "summary.json"
            if summary_path.is_file():
                saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as error:
            task_status_label.setText(f"读取所选任务结果失败：{_format_error(error)}")
            return
        result = {**saved_summary, "directory": str(directory),
                  "run_id": manifest_payload.get("run_id", task.get("run_id")),
                  "status": manifest_payload.get("status", attempt.get("state", task.get("state"))),
                  "artifacts": manifest_payload.get("artifacts", saved_summary.get("artifacts", [])),
                  "project_snapshot": {
                      "project_id": task.get("project_id"),
                      "marker_revision": task.get("marker_revision"),
                      "qc_revision": task.get("qc_revision"),
                      "task_id": task_id,
                  }}
        update_result_summary(result)
        state["last_existing_verification"] = verified
        if not verified.get("valid"):
            result_summary.setText(result_summary.text() +
                                   f"\n产物完整性校验未通过：{len(verified.get('failures', []))} 项。")
        navigation.setCurrentRow(PAGES.index("结果"))
        task_status_label.setText("已从所选任务打开运行结果；请检查产物完整性状态。")
        if artifact_combo.count():
            html_index = next((index for index in range(artifact_combo.count())
                               if str(artifact_combo.itemData(index)).casefold().endswith(".html")), -1)
            if html_index >= 0:
                artifact_combo.setCurrentIndex(html_index)
        log(f"已从历史任务打开结果：task_id={task_id} · run_id={result.get('run_id')} · 校验={bool(verified.get('valid'))}")

    def render_artifact_preview() -> None:
        relative = artifact_combo.currentData()
        path = state.get("artifact_paths", {}).get(relative)
        if not path or not Path(path).is_file():
            return
        old = artifact_preview_host.takeWidget()
        if old:
            old.deleteLater()
        suffix = Path(path).suffix.casefold()
        if suffix in (".csv", ".tsv"):
            try:
                result = preview_csv(path, max_rows=100, options=state["config"])
                frame = result["preview"]
                host = QWidget()
                layout = QVBoxLayout(host)
                layout.addWidget(QLabel(f"{relative} · 共 {result['total_rows']} 行 · 展示前 {len(frame)} 行"))
                layout.addWidget(_table_widget(frame, max_rows=100, parent=host), 1)
                artifact_preview_host.setWidget(host)
            except Exception as error:
                artifact_preview_host.setWidget(QLabel(f"表预览失败：{_format_error(error)}"))
        elif suffix == ".png":
            label = QLabel()
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                label.setPixmap(pixmap.scaled(1100, 680, Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation))
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                label.setText("无法显示此图形文件。")
            artifact_preview_host.setWidget(label)
        elif suffix == ".html":
            browser = QTextBrowser()
            browser.setOpenLinks(False)
            browser.setOpenExternalLinks(False)
            try:
                browser.setHtml(Path(path).read_text(encoding="utf-8-sig")[:2_000_000])
            except Exception as error:
                browser.setPlainText(_format_error(error))
            artifact_preview_host.setWidget(browser)
        elif suffix in (".json", ".txt", ".md"):
            label = QPlainTextEdit()
            label.setReadOnly(True)
            try:
                label.setPlainText(Path(path).read_text(encoding="utf-8-sig")[:150000])
            except Exception as error:
                label.setPlainText(_format_error(error))
            artifact_preview_host.setWidget(label)
        else:
            artifact_preview_host.setWidget(QLabel(f"{relative}\n该格式保留为原文件，可导出副本。"))

    preview_artifact_button.clicked.connect(render_artifact_preview)
    artifact_combo.currentIndexChanged.connect(lambda *_: render_artifact_preview())

    def browse_result_dir() -> None:
        directory = QFileDialog.getExistingDirectory(window, "选择已有运行目录", result_dir_edit.text() or str(Path.cwd()))
        if directory:
            result_dir_edit.setText(directory)
            manifest = Path(directory) / "manifest.json"
            try:
                from ..core_export import verify_manifest
                verified = verify_manifest(directory)
                result_summary.setText(f"已有运行目录：{directory}\n产物校验：{'通过' if verified['valid'] else '失败'} · 缺失/损坏 {len(verified['failures'])}")
                state["last_existing_verification"] = verified
                if manifest.is_file():
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    summary_path = Path(directory) / "summary.json"
                    saved_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
                    update_result_summary({**saved_summary, "directory": directory, "artifacts": payload.get("artifacts", [])})
            except Exception as error:
                log(f"运行目录检查失败：{_format_error(error)}")

    result_browse.clicked.connect(browse_result_dir)

    def copy_artifact() -> None:
        relative = artifact_combo.currentData()
        source = state.get("artifact_paths", {}).get(relative)
        if not source or not Path(source).is_file():
            return
        target, _ = QFileDialog.getSaveFileName(window, "导出文件副本", Path(source).name, "所有文件 (*.*)")
        if not target:
            return
        try:
            import shutil
            shutil.copy2(source, target)
            log(f"已导出文件副本：{target}")
            set_status(f"文件副本已导出：{Path(target).name}")
        except Exception as error:
            QMessageBox.critical(window, "导出副本失败", _format_error(error))

    copy_artifact_button.clicked.connect(copy_artifact)

    def open_run_directory() -> None:
        directory = selected_run_directory()
        if directory and directory.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def open_html_report() -> None:
        directory = selected_run_directory()
        if not directory:
            return
        reports = sorted(directory.rglob("*.html"))
        if reports:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(reports[0])))
        else:
            QMessageBox.information(window, "暂无 HTML 报告", "当前运行目录没有 HTML 报告产物。")

    open_dir_button.clicked.connect(open_run_directory)
    open_html_button.clicked.connect(open_html_report)
    task_open_result_button.clicked.connect(lambda: open_selected_task_result())

    def set_busy(is_busy: bool) -> None:
        state["busy"] = is_busy
        has_project = state.get("project") is not None
        compute_ok = state.get("compute_configuration_error") is None
        preset.setEnabled(not is_busy)
        add_task_button.setEnabled(not is_busy and has_project and compute_ok)
        widgets["import_config"].setEnabled(not is_busy)
        widgets["save_config"].setEnabled(not is_busy)
        widgets["plan"].setEnabled(not is_busy)
        widgets["preflight"].setEnabled(not is_busy and has_project and compute_ok)
        widgets["trial"].setEnabled(not is_busy and has_project and compute_ok)
        widgets["batch"].setEnabled(not is_busy and has_project and compute_ok)
        widgets["cancel"].setEnabled(is_busy)
        widgets["cancel"].setVisible(is_busy)
        widgets["start_queue"].setEnabled(not is_busy and has_project and compute_ok)
        for action_key, allowed in (("add_single_task_action", not is_busy and has_project and compute_ok),
                                    ("add_batch_task_action", not is_busy and has_project and compute_ok),
                                    ("task_single_action", not is_busy and has_project and compute_ok),
                                    ("task_batch_action", not is_busy and has_project and compute_ok),
                                    ("preflight_action", not is_busy and has_project and compute_ok),
                                    ("plan_action", not is_busy),
                                    ("import_action", not is_busy),
                                    ("export_action", not is_busy)):
            action = widgets.get(action_key)
            if action is not None:
                action.setEnabled(allowed)

    def submit_preflight(snapshot: dict[str, Any], project_state: dict[str, Any], *,
                         pending_kind: str | None) -> None:
        generation = int(state.get("preflight_generation", 0)) + 1
        cancel_event = threading.Event()
        context = {
            "generation": generation,
            "project_id": project_state["project_id"],
            "project_state": copy.deepcopy(project_state),
            "project_path": str(Path(project_state["project_directory"]).resolve()),
            "config_sha256": canonical_config_sha256(state["config"]),
            "invalidated_reason": None,
        }
        state["preflight_generation"] = generation
        state["preflight_cancel_event"] = cancel_event
        state["preflight_context"] = context
        state["preflight_report"] = None
        state["runtime_work_plan"] = None
        state["runtime_events"] = []
        state["pending_enqueue"] = ({"snapshot": copy.deepcopy(snapshot), "kind": pending_kind}
                                     if pending_kind else None)
        state["preflight_future"] = state["io_executor"].submit(
            preflight_run, copy.deepcopy(snapshot), None, copy.deepcopy(project_state),
            cancel_event.is_set, state["runtime_session"]["event_sink"] if state.get("runtime_session") else None)
        widgets["cancel"].setEnabled(True)
        widgets["cancel"].setVisible(True)

    def preflight_context_is_current(context: dict[str, Any] | None) -> tuple[bool, str]:
        if not isinstance(context, dict):
            return False, "缺少预检代次上下文"
        if context.get("invalidated_reason"):
            return False, str(context["invalidated_reason"])
        if context.get("generation") != state.get("preflight_generation"):
            return False, "预检代次已更新"
        project = state.get("project")
        if not project or project.get("project_id") != context.get("project_id"):
            return False, "当前项目已改变"
        current_state = project_state_snapshot()
        if current_state != context.get("project_state"):
            return False, "项目、marker 或 QC 修订已改变"
        if canonical_config_sha256(state.get("config", {})) != context.get("config_sha256"):
            return False, "当前配置已改变"
        return True, ""

    def start_pipeline(*, trial: bool) -> None:
        project = state.get("project")
        if not project:
            QMessageBox.information(window, "请先打开项目", "任务需要本地项目目录保存配置快照、队列历史和持久日志。请到“项目”页打开或创建项目。")
            navigation.setCurrentRow(0)
            return
        if state.get("preflight_future") is not None or state.get("run_future") is not None:
            QMessageBox.information(window, "已有后台工作", "请等待当前预检或运行结束。")
            return
        try:
            snapshot = resolve_run_config(config_with_analysis_forms())
        except Exception as error:
            QMessageBox.warning(window, "配置未通过校验", _format_error(error))
            return
        state["config"] = copy.deepcopy(snapshot)
        refresh_bound_controls()
        if trial:
            filename = mapping_file.text().strip()
            if not filename and snapshot["input"].get("signals"):
                first = snapshot["input"]["signals"][0]
                filename = first.get("file_path", "") if isinstance(first, dict) else str(first)
            if not filename:
                filename, _ = QFileDialog.getOpenFileName(window, "选择单个信号文件试跑", "", "CSV (*.csv *.tsv);;所有文件 (*.*)")
            if not filename:
                return
            mapped_eda_column = snapshot["mapping"].get("columns", {}).get("eda_uS", "eda_uS")
            if snapshot["mapping"].get("eda_unit") is None and mapped_eda_column != "eda_uS":
                QMessageBox.warning(window, "需要明确 EDA 单位", "映射页的 EDA 单位为未知。请选择来源文件对应单位后再试跑；不会默认猜成 μS。")
                navigation.setCurrentRow(1)
                return
            if snapshot["mapping"].get("eda_unit") is None and mapped_eda_column == "eda_uS":
                # 标准列名由 core_io 契约明确声明单位 uS；只有该精确标准字段才可沿用该声明。
                snapshot["mapping"]["eda_unit"] = "uS"
            snapshot["input"].update({"manifest": None, "signals": [str(Path(filename).resolve())], "base_directory": str(Path(filename).resolve().parent)})
            for key in ("participants", "group_membership", "sync_anchors", "manual_qc", "statistical_design"):
                snapshot["input"][key] = None
            snapshot["synchrony"]["enabled"] = False
            snapshot["statistics"]["enabled"] = False
            log("单文件任务：仅关闭多人同步和群体统计；已有 events/conditions 仍随冻结输入参与相应窗口分析。")
        if not trial:
            state["config"] = copy.deepcopy(snapshot)
            persist_project_draft()
        elif persist_project_draft() is None:
            QMessageBox.warning(window, "项目草稿未保存", "当前项目草稿无法持久化，未提交预检。")
            return
        materialize_project_events(snapshot)
        if not trial:
            # Keep the live editor aligned with the exact config handed to preflight.
            refresh_bound_controls()
        project_state = project_state_snapshot()
        if project_state is None:
            return
        submit_preflight(snapshot, project_state, pending_kind="trial" if trial else "batch")
        set_progress(None, "运行前检查进行中 · 分析未启动")
        task_status_label.setText("正在解析输入、配置、时钟/QC/markers 和资源预算；本步骤不执行完整分析。")
        navigation.setCurrentRow(9)
        log(f"已提交 {state['pending_enqueue']['kind']} 的运行前预检；等待报告后才入队。")

    widgets["trial"].clicked.connect(lambda: start_pipeline(trial=True))
    widgets["batch"].clicked.connect(lambda: start_pipeline(trial=False))
    widgets["preflight"].clicked.connect(lambda: _start_preflight_only())

    def _start_preflight_only() -> None:
        if state.get("project") is None:
            QMessageBox.information(window, "请先打开项目", "运行前检查报告需要绑定项目 revision。")
            navigation.setCurrentRow(0)
            return
        if state.get("preflight_future") is not None or state.get("run_future") is not None:
            return
        try:
            snapshot = resolve_run_config(config_with_analysis_forms())
            state["config"] = copy.deepcopy(snapshot)
            refresh_bound_controls()
            if persist_project_draft() is None:
                raise RuntimeError("项目草稿保存失败，不能生成可复核的预检报告。")
            materialize_project_events(snapshot)
            state["pending_enqueue"] = None
            project_state = project_state_snapshot()
            if project_state is None:
                return
            submit_preflight(snapshot, project_state, pending_kind=None)
            set_progress(None, "运行前检查进行中 · 分析未启动")
            navigation.setCurrentRow(9)
        except Exception as error:
            QMessageBox.warning(window, "无法开始运行前检查", _format_error(error))

    def cancel_pipeline() -> None:
        if state.get("preflight_future") is not None:
            invalidate_preflight("用户请求取消")
            return
        project = state.get("project")
        task_id = state.get("active_task_id")
        if project and task_id:
            try:
                request_task_cancel(project, task_id)
                set_status("已请求安全取消 · 等待任务到达检查点")
                log(f"已请求取消任务 {task_id}；pipeline 会保留已提交记录。")
                refresh_tasks()
            except Exception as error:
                QMessageBox.warning(window, "取消请求失败", _format_error(error))

    widgets["cancel"].clicked.connect(cancel_pipeline)

    def resume_pipeline() -> None:
        project, task_id = state.get("project"), state.get("selected_task_id")
        if not project or not task_id:
            QMessageBox.information(window, "选择任务", "请到“任务与日志”页选择一个可恢复任务。")
            return
        retry_selected_task("resume")

    resume_button.clicked.connect(resume_pipeline)

    def start_queued_task() -> None:
        project = state.get("project")
        session = state.get("runtime_session")
        if not project or not session:
            QMessageBox.information(window, "项目未打开", "请先打开项目目录。")
            return
        if state.get("run_future") is not None or state.get("preflight_future") is not None:
            return
        try:
            task = claim_next_task(project, owner_token=f"gui:{os.getpid()}:{uuid.uuid4()}")
            if task is None:
                QMessageBox.information(window, "队列为空", "没有可启动的队列任务；任务历史页列出当前状态。")
                return
            state["active_task_id"] = task["task_id"]
            state["selected_task_id"] = task["task_id"]
            state["runtime_events"] = []
            state["runtime_work_plan"] = (task.get("preflight_report") or {}).get("work_plan")
            state["claimed_task"] = task
            state["run_future"] = state["executor"].submit(
                execute_claimed_task, task["task_id"], session["event_sink"], project=project,
                runtime_session=session)
            state["current_run_kind"] = "queued task"
            state["progress_started_at"] = time.monotonic()
            set_busy(True)
            set_progress(None, f"任务已开始 · {task['task_id']} · 根据实际完成单位更新进度")
            log(f"显式启动队列任务 {task['task_id']} · attempt {task.get('attempt_id')}")
            refresh_tasks()
            navigation.setCurrentRow(9)
        except Exception as error:
            QMessageBox.warning(window, "队列任务未启动", _format_error(error))

    widgets["start_queue"].clicked.connect(start_queued_task)

    def poll_background_tasks() -> None:
        # Import preview, QC load and pipeline all return data through main-thread polling.
        poll_marker_waveform()
        mapping_task = state.get("mapping_future")
        if mapping_task:
            preview_id, future = mapping_task[:2]
            snapshot = mapping_task[2] if len(mapping_task) > 2 else None
            if future.done():
                state["mapping_future"] = None
                snapshot_current = (snapshot is None or mapping_snapshot_is_current(snapshot))
                if preview_id == state["preview_id"] and snapshot_current:
                    try:
                        samples, metadata, report = future.result()
                        accepted_snapshot = snapshot or current_mapping_snapshot()
                        source_filename = accepted_snapshot["filename"]
                        frame = samples.head(100)
                        widgets["preview_title"].setText(f"{Path(source_filename).name} · core_io 读取 {len(samples)} 行 · 预览前 {len(frame)} 行 · 单位 {metadata.get('unit', '未声明')} · fs {metadata.get('fs_estimated_hz', metadata.get('fs_declared_hz', 'NA'))} Hz")
                        widgets["preview_host"].setWidget(_table_widget(frame, parent=widgets["preview_host"]))
                        mapping_result.setText(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:5000])
                        accepted_config = copy.deepcopy(accepted_snapshot["config"])
                        accepted_config["mapping"].update(copy.deepcopy(accepted_snapshot["mapping"]))
                        accepted_config["input"]["signals"] = [source_filename]
                        accepted_config["input"]["manifest"] = None
                        accepted_config["input"]["base_directory"] = str(Path(source_filename).resolve().parent)
                        state["config"] = resolve_config(accepted_config)
                        update_config_editor()
                        set_status("信号已由 core_io 读取并完成映射预览")
                        log(f"信号读取成功：{len(samples)} 行；来源 {report.get('source_sha256', report.get('sha256', 'hash 已记录'))}")
                    except Exception as error:
                        mapping_result.setText(_format_error(error))
                        set_status("映射读取失败")
                        log(_format_error(error))

        qc_task = state.get("qc_future")
        if qc_task:
            preview_id, future = qc_task
            if future.done():
                state["qc_future"] = None
                if preview_id == state["preview_id"]:
                    try:
                        records, tables, report = future.result()
                        on_qc_records_loaded(records, tables, report, preview_id)
                    except Exception as error:
                        QMessageBox.warning(window, "记录读取失败", _format_error(error))
                        set_status("QC 记录载入失败")
                        log(_format_error(error))

        preflight_future = state.get("preflight_future")
        if preflight_future is not None and preflight_future.done():
            state["preflight_future"] = None
            state["preflight_cancel_event"] = None
            context = state.get("preflight_context")
            state["preflight_context"] = None
            widgets["cancel"].setEnabled(False)
            widgets["cancel"].hide()
            pending = state.get("pending_enqueue")
            current, stale_reason = preflight_context_is_current(context)
            if not current:
                state["pending_enqueue"] = None
                state["preflight_report"] = None
                state["runtime_work_plan"] = None
                task_status_label.setText(f"预检报告已丢弃：{stale_reason}；请对当前项目重新检查。")
                set_progress(0, f"运行前检查已过期：{stale_reason} · 未入队、未启动分析")
                log(f"预检结果丢弃 generation={context.get('generation') if isinstance(context, dict) else 'unknown'} · {stale_reason}")
                return
            try:
                report = preflight_future.result()
                state["preflight_report"] = report
                state["runtime_work_plan"] = report.get("work_plan")
                errors = report.get("errors", [])
                warnings = report.get("warnings", [])
                preflight_text = [
                    f"报告 ID：{report.get('report_id')}",
                    f"配置 SHA256：{report.get('config_hash')}",
                    f"可入队：{'是' if report.get('eligible_to_enqueue') else '否'}",
                    f"计划工作单位：{report.get('work_unit_count', (report.get('work_plan') or {}).get('total_units'))}",
                    f"阻断项 {len(errors)} · 警告 {len(warnings)}",
                ]
                for item in errors[:30]:
                    preflight_text.append(f"阻断 [{item.get('code')}] {item.get('path')}: {item.get('message')} · {item.get('remediation')}")
                for item in warnings[:20]:
                    preflight_text.append(f"警告 [{item.get('code')}] {item.get('path')}: {item.get('message')}")
                attempt_detail.setPlainText("\n".join(preflight_text))
                if report.get("eligible_to_enqueue") and pending:
                    current_state = project_state_snapshot()
                    snapshot = pending["snapshot"]
                    frozen_spec = {"config": snapshot, "input_manifest": None,
                                   "project_revision": report.get("project_revision"),
                                   "draft_revision": report.get("draft_revision"),
                                   "marker_revision": report.get("marker_revision"),
                                   "qc_revision": report.get("qc_revision")}
                    task_id = enqueue_task(state["project"], frozen_spec, report)
                    task_status_label.setText(
                        f"预检报告已批准条件：合格，可入队 · task {task_id} · 尚未运行。点击“开始队列”后才会计算。")
                    log(f"预检通过并入队：{pending['kind']} task {task_id} · 报告 {report.get('report_id')}")
                    refresh_tasks()
                else:
                    task_status_label.setText(
                        f"运行前检查完成 · {'可入队' if report.get('eligible_to_enqueue') else '存在阻断项'} · "
                        f"{len(errors)} 个阻断项 / {len(warnings)} 个警告 · 报告仅检查，没有启动分析")
                    log(f"运行前检查完成：eligible={report.get('eligible_to_enqueue')} · 报告 {report.get('report_id')}")
                state["pending_enqueue"] = None
                set_progress(0 if not report.get("eligible_to_enqueue") else None,
                             "运行前检查完成 · 等待用户检查报告并显式开始队列")
            except Exception as error:
                state["pending_enqueue"] = None
                state["preflight_context"] = None
                task_status_label.setText(f"运行前检查/入队失败：{_format_error(error)}")
                set_progress(None, "运行前检查失败；未启动分析")
                log(f"运行前检查/入队失败：{_format_error(error)}")

        log_future = state.get("log_search_future")
        if log_future is not None and log_future.done():
            state["log_search_future"] = None
            try:
                rows, cursor = log_future.result()
                state["log_cursor"] = cursor
                rendered = "\n".join(
                    f"{row.get('timestamp_utc', '')} [{row.get('level')}] #{row.get('seq')} "
                    f"{row.get('stage')} {row.get('kind')}: {row.get('message')}"
                    for row in rows)
                if state.pop("log_search_append", False):
                    rendered = log_results.toPlainText() + ("\n" if rendered else "") + rendered
                log_results.setPlainText(rendered or "没有匹配的持久日志事件。")
                log_next_button.setEnabled(cursor is not None)
            except Exception as error:
                log_results.setPlainText(f"日志检索失败：{_format_error(error)}")

        log_export_future = state.get("log_export_future")
        if log_export_future is not None and log_export_future.done():
            state["log_export_future"] = None
            log_export_button.setEnabled(True)
            try:
                exported_path = log_export_future.result()
                context = state.get("log_export_context") or {}
                state["log_export_context"] = None
                if state.get("project", {}).get("project_id") == context.get("project_id"):
                    set_status(f"日志已导出：{Path(exported_path).name}")
                    log(f"项目日志已导出：{exported_path}")
            except Exception as error:
                state["log_export_context"] = None
                log_results.setPlainText(f"日志导出失败：{_format_error(error)}")
                QMessageBox.warning(window, "日志导出失败", _format_error(error))

        # Runtime controller is the sole event sequencer/writer. The UI only renders
        # the same events it receives after durable JSONL append.
        while True:
            try:
                event = state["run_queue"].get_nowait()
            except queue.Empty:
                break
            if not isinstance(event, dict):
                continue
            state["runtime_events"].append(event)
            if len(state["runtime_events"]) > 10000:
                state["runtime_events"] = state["runtime_events"][-8000:]
            if event.get("kind") == "plan_revised" and isinstance(event.get("details"), dict):
                revised_plan = event["details"].get("work_plan")
                if isinstance(revised_plan, dict):
                    state["runtime_work_plan"] = revised_plan
                    log(f"运行计划已按真实输入更新 · revision {revised_plan.get('revision', 'unknown')}")
            if event.get("kind") == "completed" and isinstance(event.get("details"), dict):
                detail = event["details"]
                if detail.get("duration_s") is not None and detail.get("method") is not None and detail.get("rate_hz") is not None:
                    state["runtime_history"].append(detail)
            if event.get("stage") != "ui":
                message = str(event.get("message") or "")
                if message:
                    local_record = f" · {event.get('record_id')}" if event.get("record_id") else ""
                    log(f"{event.get('stage') or 'runtime'}：{message}{local_record}")
            if state.get("runtime_work_plan"):
                try:
                    progress_info = __import__("eda_toolkit.runtime_events", fromlist=["estimate_progress"]).estimate_progress(
                        state["runtime_events"], state["runtime_work_plan"], state.get("runtime_history", []))
                    percent = progress_info.get("percent")
                    eta = progress_info.get("eta_s")
                    eta_label = (f"ETA {eta:.0f} 秒 · {progress_info.get('confidence')} 信心" if eta is not None
                                 else f"ETA 未知 · {progress_info.get('reason')}")
                    stage_counts = (f" · {progress_info.get('stage')} {progress_info.get('done')}/"
                                    f"{progress_info.get('total')}" if progress_info.get("stage") else "")
                    set_progress(percent, f"{event.get('message') or event.get('stage')}{stage_counts} · {eta_label}")
                except Exception as error:
                    set_progress(None, f"事件进度解析不可用：{_format_error(error)}")

        current_future = state.get("run_future")
        if current_future is not None:
            if current_future.done():
                try:
                    summary = current_future.result()
                    claimed_task = state.get("claimed_task")
                    if isinstance(summary, dict) and isinstance(claimed_task, dict):
                        summary = {**summary, "project_snapshot": {
                            "project_id": claimed_task.get("project_id"),
                            "marker_revision": claimed_task.get("marker_revision"),
                            "qc_revision": claimed_task.get("qc_revision"),
                            "task_id": claimed_task.get("task_id"),
                        }}
                    update_result_summary(summary)
                    status_word = summary.get("status", "unknown")
                    verified_success = False
                    if status_word == "succeeded" and summary.get("directory"):
                        try:
                            from ..core_export import verify_manifest
                            verified_success = bool(verify_manifest(summary["directory"]).get("valid"))
                        except Exception:
                            verified_success = False
                    set_progress(100 if verified_success else None,
                                 f"pipeline {status_word} · {summary.get('directory', '')}")
                    log(f"pipeline 完成：{status_word}，已完成 {summary.get('completed_records', 0)} 条")
                    if summary.get("directory"):
                        navigation.setCurrentRow(7)
                        if artifact_combo.count():
                            render_artifact_preview()
                except Exception as error:
                    summary = {"status": "failed", "errors": [{"message": _format_error(error)}]}
                    update_result_summary(summary)
                    set_status("后台任务未正常返回")
                    log(f"后台任务异常：{_format_error(error)}")
                state["run_future"] = None
                state["cancel_event"] = None
                state["active_task_id"] = None
                set_busy(False)
                refresh_tasks()

    timer = QTimer(window)
    timer.setInterval(150)
    timer.timeout.connect(poll_background_tasks)
    timer.start()
    state["timer"] = timer

    # ── 页面导航、帮助与错误解释 ──────────────────────────────────────────
    def change_page(row: int) -> None:
        stack.setCurrentIndex(max(0, row))
        if state.get("project"):
            schedule_project_autosave()

    navigation.currentRowChanged.connect(change_page)
    widgets["import_config"].clicked.connect(import_config)
    widgets["save_config"].clicked.connect(save_config_action)
    widgets["plan"].clicked.connect(show_plan)

    def change_preset(_index: int = -1) -> None:
        name = preset.currentData()
        if not name or name == state["config"]["pipeline_id"] or state["busy"]:
            return
        try:
            old = resolve_config(config_with_analysis_forms())
        except Exception as error:
            preset.blockSignals(True)
            preset.setCurrentIndex(max(0, preset.findData(state["config"]["pipeline_id"])))
            preset.blockSignals(False)
            QMessageBox.warning(window, "预设未切换", f"请先修正当前表单：{_format_error(error)}")
            return
        candidate = get_preset(name)
        # 预设调整流程参数，继续保留用户显式选择的输入、单位/映射、QC 标注和输出位置。
        for section in ("input", "mapping", "qc", "output", "workspace", "runtime"):
            candidate[section] = copy.deepcopy(old[section])
        candidate["compute"] = copy.deepcopy(old["compute"])
        candidate["preprocess"] = copy.deepcopy(old["preprocess"])
        candidate["decomposition"] = copy.deepcopy(old["decomposition"])
        candidate["scr"] = copy.deepcopy(old["scr"])
        candidate["windows"] = copy.deepcopy(old["windows"])
        candidate["features"] = copy.deepcopy(old["features"])
        candidate["statistics"] = copy.deepcopy(old["statistics"])
        if name != "group_sync":
            candidate["synchrony"] = copy.deepcopy(old["synchrony"])
        apply_config(candidate, source=f"预设 {name}")

    preset.currentIndexChanged.connect(change_preset)

    def open_help() -> None:
        from importlib import resources
        candidate = resources.files("eda_toolkit").joinpath("resources", "help", "EDA使用指南.pdf")
        if candidate.is_file():
            try:
                # Frozen/zip resources may only exist inside a temporary extraction context;
                # persist a user-readable local copy before asking the OS PDF viewer to open it.
                filepath = Path(tempfile.gettempdir()) / "eda-toolkit-中文帮助.pdf"
                filepath.write_bytes(candidate.read_bytes())
                state["help_copy"] = str(filepath)
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(filepath)))
            except Exception as error:
                QMessageBox.warning(window, "帮助无法打开", _format_error(error))
        else:
            QMessageBox.information(window, "帮助文档未构建", f"内置中文 PDF 尚未安装。预期资源路径：{candidate}")

    def explain_errors() -> None:
        QMessageBox.information(window, "错误解释", "运行错误保存在结果页的 summary.errors 和日志。诊断含稳定 code、stage、record_id、field 与可选 remediation；修正输入或配置后，使用“校验并恢复”验证已提交缓存和输入哈希。未满足方法依赖时不会自动换用其他算法。")

    settings_menu = window.menuBar().addMenu("设置")
    import_action = QAction("导入设置…", window)
    import_action.setToolTip("从 JSON 文件导入完整设置")
    import_action.triggered.connect(lambda: widgets["import_config"].click())
    export_action = QAction("导出设置为 JSON…", window)
    export_action.setToolTip("导出当前控件/已应用设置，并选择保存位置")
    export_action.triggered.connect(lambda: widgets["save_config"].click())
    settings_menu.addAction(import_action)
    settings_menu.addAction(export_action)

    task_menu = window.menuBar().addMenu("任务")
    plan_action = QAction("计划预览", window)
    plan_action.triggered.connect(lambda: widgets["plan"].click())
    preflight_action = QAction("运行前检查…", window)
    preflight_action.triggered.connect(lambda: widgets["preflight"].click())
    task_menu.addAction(plan_action)
    task_menu.addAction(preflight_action)
    task_menu.addSeparator()
    task_single_action = QAction("单文件预检并入队…", window)
    task_single_action.triggered.connect(lambda: widgets["trial"].click())
    task_batch_action = QAction("批量预检并入队", window)
    task_batch_action.triggered.connect(lambda: widgets["batch"].click())
    task_menu.addAction(task_single_action)
    task_menu.addAction(task_batch_action)
    add_single_action.triggered.connect(lambda: widgets["trial"].click())
    add_batch_action.triggered.connect(lambda: widgets["batch"].click())

    has_project = state.get("project") is not None
    for action, enabled in ((import_action, True), (export_action, True), (plan_action, True),
                            (preflight_action, has_project), (task_single_action, has_project),
                            (task_batch_action, has_project)):
        action.setEnabled(enabled)
    widgets.update({"import_action": import_action, "export_action": export_action,
                    "plan_action": plan_action, "preflight_action": preflight_action,
                    "task_single_action": task_single_action,
                    "task_batch_action": task_batch_action})

    help_action = QAction("中文帮助", window)
    help_action.setShortcut(QKeySequence("F1"))
    help_action.triggered.connect(open_help)
    error_action = QAction("错误解释", window)
    error_action.triggered.connect(explain_errors)
    menu = window.menuBar().addMenu("帮助")
    menu.addAction(help_action)
    menu.addAction(error_action)
    window.setStatusTip("参数为研究示例；每次运行会冻结配置并写出完整来源与状态。")
    widgets.update({"help": help_action, "explain_errors": error_action})

    def shutdown() -> None:
        if state["closed"]:
            return
        state["closed"] = True
        timer.stop()
        autosave_timer.stop()
        environment_dialog = state.get("environment_dialog")
        if environment_dialog is not None:
            environment_dialog.shutdown()
        project = state.get("project")
        active_task_id = state.get("active_task_id")
        if project and active_task_id:
            try:
                request_task_cancel(project, active_task_id)
            except Exception:
                pass
        state["io_executor"].shutdown(wait=True, cancel_futures=True)
        state["executor"].shutdown(wait=True, cancel_futures=True)
        if state.get("autosave_dirty"):
            persist_project_draft()
        session = state.get("runtime_session")
        if session:
            try:
                session["flush"]()
                session["close"]()
            except Exception:
                pass

    app.aboutToQuit.connect(shutdown)
    window.destroyed.connect(lambda *_: shutdown())
    if initial_config is not None:
        refresh_bound_controls()
        if isinstance(initial_config, (str, Path)):
            config_path.setText(str(Path(initial_config).expanduser().resolve()))
            state["last_loaded_path"] = str(Path(initial_config).expanduser().resolve())
    widgets["project_autosave_status"].setText(state["autosave_state"])
    project_directory = state["config"].get("workspace", {}).get("project_directory")
    if project_directory:
        open_project_at(str(project_directory), ask_recovery=True)
    else:
        widgets["preflight"].setEnabled(False)
        widgets["trial"].setEnabled(False)
        widgets["batch"].setEnabled(False)
        widgets["add_task_button"].setEnabled(False)
        widgets["start_queue"].setEnabled(False)
        task_status_label.setText("未配置项目；打开项目后才会启用运行前检查、任务队列和持久日志。")
        marker_state_label.setText("未配置项目；打开项目后才能持久化 marker 编辑。")
        set_status("就绪 · 未打开项目；自动保存、任务历史和持久日志不可用")
    state["widgets"] = widgets
    state["shutdown"] = shutdown
    return window, state


def main(argv: list[str] | None = None) -> int:
    """Start through the lightweight bootstrap so feedback appears before imports."""
    from .bootstrap import main as bootstrap_main

    return bootstrap_main(argv, ui_module=sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())





