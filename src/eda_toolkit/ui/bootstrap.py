"""Lightweight GUI startup path that paints feedback before analysis imports."""
from __future__ import annotations

import argparse
import importlib
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase, QIcon
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QMessageBox, QProgressBar, QVBoxLayout

from ..runtime_identity import configure_app_identity, configure_window_icons, icon_path


def _show_error(parent: QDialog, title: str, error: BaseException) -> None:
    box = QMessageBox(QMessageBox.Icon.Critical, title,
                      f"工作台未能启动：{type(error).__name__}: {error}",
                      QMessageBox.StandardButton.Ok, parent)
    box.setDetailedText("".join(traceback.format_exception(type(error), error, error.__traceback__)))
    box.exec()


def main(argv: list[str] | None = None, *, ui_module: ModuleType | None = None) -> int:
    """Start the GUI with visible loading status before importing heavy modules."""
    parser = argparse.ArgumentParser(prog="eda-gui", description="皮肤电研究工作台")
    parser.add_argument("--smoke", action="store_true", help="显示一次真实 Qt 窗口后退出")
    parser.add_argument("--screenshot", help="把当前真实窗口画面保存为 PNG")
    parser.add_argument("--startup-screenshot", help="保存重依赖导入前显示的启动提示 PNG")
    parser.add_argument("--config", help="启动时读取完整 JSON 配置")
    args, _unknown = parser.parse_known_args(argv)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setApplicationName("EDA Toolkit")
    app.setApplicationVersion("0.3.1")
    installed_families = {name.casefold(): name for name in QFontDatabase.families()}
    startup_font_family = next((installed_families[name.casefold()]
                                for name in ("Microsoft YaHei UI", "Microsoft YaHei", "微软雅黑", "Segoe UI")
                                if name.casefold() in installed_families), "Microsoft YaHei UI")
    app.setFont(QFont(startup_font_family, 10))
    desktop_name_setter = getattr(app, "setDesktopFileName", None)
    if callable(desktop_name_setter):
        desktop_name_setter("eda-toolkit")

    # Qt can assign a process identity while QApplication is created. Reapply
    # the stable application ID now, before any startup or workbench window shows.
    configure_app_identity()
    try:
        icon = QIcon(icon_path())
        app.setWindowIcon(icon)
    except Exception:
        icon = QIcon()

    splash = QDialog()
    splash.setWindowTitle("EDA Toolkit · 正在启动")
    splash.setWindowIcon(icon)
    splash.setWindowModality(Qt.WindowModality.ApplicationModal)
    splash.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
    splash.setFixedSize(470, 154)
    splash.setStyleSheet("""
        QDialog { background: #F4F7FA; color: #183247; }
        QLabel#startupTitle { color: #17324D; font-size: 15pt; font-weight: 700; }
        QLabel#startupStatus { color: #5D7182; font-size: 10.5pt; }
        QProgressBar { background: #E2E9EE; border: none; border-radius: 5px; min-height: 10px; }
        QProgressBar::chunk { background: #148A8A; border-radius: 5px; }
    """)
    splash_layout = QVBoxLayout(splash)
    splash_layout.setContentsMargins(24, 18, 24, 18)
    splash_layout.setSpacing(12)
    title = QLabel("EDA 皮电研究工作台")
    title.setObjectName("startupTitle")
    title.setFont(QFont(startup_font_family, 15, QFont.Weight.DemiBold))
    status = QLabel("正在开启工作台，载入分析模块中…")
    status.setObjectName("startupStatus")
    status.setFont(QFont(startup_font_family, 10))
    progress = QProgressBar()
    progress.setRange(0, 0)
    progress.setTextVisible(False)
    splash_layout.addWidget(title)
    splash_layout.addWidget(status)
    splash_layout.addWidget(progress)
    splash.show()
    app.processEvents()
    if args.startup_screenshot:
        try:
            startup_target = Path(args.startup_screenshot).expanduser().resolve()
            startup_target.parent.mkdir(parents=True, exist_ok=True)
            if splash.grab().save(str(startup_target), "PNG"):
                print(f"已保存重依赖导入前的启动提示截图：{startup_target}")
            else:
                print(f"无法保存启动提示截图：{startup_target}", file=sys.stderr)
        except Exception as error:
            print(f"无法保存启动提示截图：{error}", file=sys.stderr)

    phases: queue.Queue[str] = queue.Queue()
    finished = threading.Event()
    outcome: dict[str, Any] = {}
    started_at = time.monotonic()

    def load_runtime() -> None:
        try:
            module = ui_module or importlib.import_module("eda_toolkit.ui.app")
            module.warm_runtime_imports(phases.put)
            outcome["module"] = module
        except BaseException as error:
            outcome["error"] = error
        finally:
            finished.set()

    threading.Thread(target=load_runtime, name="eda-gui-startup", daemon=True).start()
    timer = QTimer(app)

    def continue_startup() -> None:
        while True:
            try:
                status.setText(phases.get_nowait())
            except queue.Empty:
                break
        if not finished.is_set() or time.monotonic() - started_at < 0.25:
            return
        timer.stop()
        error = outcome.get("error")
        if error is not None:
            _show_error(splash, "工作台启动失败", error)
            splash.close()
            app.exit(1)
            return

        status.setText("正在创建工作台界面…")
        app.processEvents()
        state = None
        try:
            module = outcome["module"]
            window, state = module.create_window(args.config)
            if not icon.isNull():
                window.setWindowIcon(icon)
            window.show()
            app.processEvents()
            native_icon_result = configure_window_icons(window)
            state["native_icon_result"] = native_icon_result
            splash.close()
            if args.screenshot:
                target = Path(args.screenshot).expanduser().resolve()
                target.parent.mkdir(parents=True, exist_ok=True)
                if not window.grab().save(str(target), "PNG"):
                    print(f"无法保存真实 Qt 窗口截图：{target}", file=sys.stderr)
                    window.close()
                    state["shutdown"]()
                    app.exit(2)
                    return
                print(f"已保存真实 Qt 窗口截图：{target}")
            if args.smoke:
                window.close()
                state["shutdown"]()
                app.exit(0)
        except BaseException as error:
            _show_error(splash, "工作台启动失败", error)
            splash.close()
            if state is not None:
                try:
                    state["shutdown"]()
                except Exception:
                    pass
            app.exit(1)

    timer.timeout.connect(continue_startup)
    timer.start(35)
    return app.exec()


__all__ = ["main"]

