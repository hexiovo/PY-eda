"""Window for automatic PYEDA analysis from the original study files."""

from __future__ import annotations

import sys
import hashlib
import os
import time
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    EDA_ROOT = Path(sys._MEIPASS)
else:
    EDA_ROOT = Path(__file__).resolve().parent.parent
if not (EDA_ROOT / "skills" / "eda-study-config" / "scripts" / "check_bundle.py").is_file():
    EDA_ROOT = Path(r"D:\program\EDA")
PYEDA_ROOT = Path(r"D:\data\PYEDA")


class AnalysisWorker(QThread):
    progress = Signal(dict)
    completed = Signal(dict)
    failed = Signal(str, str)
    blocked = Signal(str)

    def __init__(self, wrist_dir: Path, config: Path | None = None, *,
                 force_new: bool = False, max_workers: int = 1) -> None:
        super().__init__()
        self.config = config
        self.wrist_dir = wrist_dir
        self.force_new = force_new
        self.max_workers = max_workers

    def run(self) -> None:
        try:
            if self.config is not None:
                import pyeda_one_stop

                try:
                    self.completed.emit(pyeda_one_stop.run_verified_bundle(self.config, EDA_ROOT, self.progress.emit))
                except pyeda_one_stop.AnalysisBlocked as error:
                    self.blocked.emit(str(error))
                return
            from pyeda_aut_pit import run_study

            base = run_study(
                None, self.wrist_dir, PYEDA_ROOT / "_rawdata" / "_rawdata", self.progress.emit,
                design=PYEDA_ROOT / "实验情况记录表.xlsx", output_root=PYEDA_ROOT / "study-config",
                cancel_requested=self.isInterruptionRequested, force_new=self.force_new,
                max_workers=self.max_workers,
            )
            if base["status"] in ("succeeded", "partial"):
                from pyeda_turn_analysis import run_turn_analysis

                self.completed.emit(run_turn_analysis(Path(base["run_directory"]), self.progress.emit,
                                                       self.isInterruptionRequested))
            else:
                self.completed.emit(base)
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PYEDA 一站式导入与分析")
        self.setMinimumSize(760, 520)
        self.resize(860, 600)
        self.started_at: float | None = None
        self.current_done = 0
        self.current_total = 1
        self.current_stage = ""
        self.analysis_eta = None
        self.baseline_eta = None
        self.baseline_eta_message = ""
        self.wrist_dir = PYEDA_ROOT / "手环数据"
        self.latest_directory: Path | None = None
        self.running = False
        self.close_when_finished = False
        self.last_config: Path | None = None
        self.last_status = ""
        self.worker: AnalysisWorker | None = None

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        title = QLabel("PYEDA 一站式导入与分析")
        title.setFont(QFont("Microsoft YaHei UI", 18, QFont.Weight.Bold))
        subtitle = QLabel("手环与行为数据 · 个人回答情境、IOC 与扩展统计 · 探索性分析")
        subtitle.setStyleSheet("color: #607586;")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.stage_label = QLabel("准备启动…")
        self.stage_label.setWordWrap(True)
        self.stage_label.setStyleSheet("font-size: 12pt; color: #183247;")
        layout.addWidget(self.stage_label)

        self.overall_label = QLabel("总体进度 0%")
        self.overall_label.setStyleSheet("font-size: 10.5pt; color: #476176;")
        layout.addWidget(self.overall_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(24)
        layout.addWidget(self.progress_bar)

        metrics = QHBoxLayout()
        self.elapsed_label = QLabel("已用时 00:00")
        self.eta_label = QLabel("预计剩余时间 计算中")
        self.detail_label = QLabel("0 / 0")
        metrics.addWidget(self.elapsed_label)
        metrics.addStretch(1)
        metrics.addWidget(self.eta_label)
        metrics.addStretch(1)
        metrics.addWidget(self.detail_label)
        layout.addLayout(metrics)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("运行日志会显示在这里")
        layout.addWidget(self.log, 1)

        buttons = QHBoxLayout()
        self.open_button = QPushButton("打开本次目录")
        self.open_button.clicked.connect(self._open_latest)
        self.open_button.setEnabled(False)
        self.log_button = QPushButton("打开日志")
        self.log_button.clicked.connect(self._open_log)
        self.log_button.setEnabled(False)
        self.bundle_button = QPushButton("载入已核验分析包")
        self.bundle_button.clicked.connect(self._select_bundle)
        self.import_button = QPushButton("选择其他手环目录")
        self.import_button.clicked.connect(self._select_export)
        self.copy_button = QPushButton("复制输出目录")
        self.copy_button.clicked.connect(self._copy_output)
        self.copy_button.setEnabled(False)
        self.restart_button = QPushButton("继续／重试")
        self.restart_button.clicked.connect(self._restart)
        self.restart_button.setEnabled(False)
        self.force_close_button = QPushButton("立即退出")
        self.force_close_button.hide()
        self.force_close_button.clicked.connect(self._force_close)
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.log_button)
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.bundle_button)
        buttons.addWidget(self.copy_button)
        buttons.addStretch(1)
        buttons.addWidget(self.restart_button)
        buttons.addWidget(self.force_close_button)
        layout.addLayout(buttons)
        self.setCentralWidget(central)

        self.clock = QTimer(self)
        self.clock.timeout.connect(self._refresh_clock)
        self.clock.start(500)

    @Slot()
    def start_work(self) -> None:
        self._start_worker()

    def _select_bundle(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择核验包中的 analysis.json", str(PYEDA_ROOT / "study-config"),
            "分析配置 (analysis.json)",
        )
        if selected:
            self._start_worker(Path(selected))

    def _select_export(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "选择包含“原始数据”子目录的手环导出目录", str(PYEDA_ROOT / "手环数据"),
        )
        if chosen:
            self._start_worker(wrist_dir=Path(chosen))

    def _copy_output(self) -> None:
        if self.latest_directory:
            QApplication.clipboard().setText(str(self.latest_directory))
            self.log.appendPlainText("已复制本次输出目录。")

    def _start_worker(self, config: Path | None = None, *, wrist_dir: Path | None = None,
                      force_new: bool = False, max_workers: int = 1) -> None:
        if self.running:
            return
        if wrist_dir is not None:
            self.wrist_dir = wrist_dir
        self.latest_directory = None
        self.last_config = config
        self.last_status = ""
        self.analysis_eta = None
        self.baseline_eta = None
        self.baseline_eta_message = ""
        self.current_done = 0
        self.current_total = 1
        self.current_stage = ""
        self.log.clear()
        try:
            from pyeda_eta import estimate_total_seconds

            estimate = estimate_total_seconds(PYEDA_ROOT / "study-config", self.wrist_dir,
                                               PYEDA_ROOT / "_rawdata" / "_rawdata")
            if estimate.get("status") == "estimated":
                self.baseline_eta = float(estimate["seconds"])
                self.baseline_eta_message = estimate["message"]
                self.log.appendPlainText(
                    f"预计总耗时约 {self._format_seconds(self.baseline_eta)}。{self.baseline_eta_message}。"
                )
        except Exception as error:
            self.log.appendPlainText(f"未能生成运行前时间估计：{error}")
        self._set_overall(0)
        self.running = True
        self.started_at = time.monotonic()
        self.restart_button.setEnabled(False)
        self.bundle_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.log_button.setEnabled(False)
        self.copy_button.setEnabled(False)
        self.stage_label.setText("正在准备分析…")
        self.worker = AnalysisWorker(self.wrist_dir, config, force_new=force_new, max_workers=max_workers)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.blocked.connect(self._on_blocked)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    @Slot(dict)
    def _on_progress(self, event: dict[str, Any]) -> None:
        stage = str(event.get("stage", "working"))
        done = int(event.get("done", 0) or 0)
        total = max(1, int(event.get("total", 1) or 1))
        message = str(event.get("message", ""))
        self.current_done = done
        self.current_total = total
        self.current_stage = stage
        details = event.get("details", {})
        self.analysis_eta = details.get("eta_s")
        if details.get("run_directory"):
            self._set_run_directory(Path(details["run_directory"]))
        if not self.close_when_finished:
            self.stage_label.setText(message or stage)
        start, end = {
            "analysis_checks": (0, 10 if self.last_config else 5),
            "analysis_run": (10, 99) if self.last_config else (5, 40),
            "analysis_export": (92, 99) if self.last_config else (40, 45),
            "extended_checks": (45, 50),
            "extended_features": (50, 78),
            "extended_statistics": (78, 97),
            "extended_export": (97, 99),
        }[stage]
        fraction = min(1.0, max(0.0, done / total))
        self._set_overall(max(self.progress_bar.value(), start + int((end - start) * fraction)))
        if message:
            self.log.appendPlainText(message)

    def _set_overall(self, percent: int) -> None:
        self.progress_bar.setValue(percent)
        self.overall_label.setText(f"总体进度 {percent}%")
        self.detail_label.setText(f"{self.current_done} / {self.current_total}")

    @Slot()
    def _refresh_clock(self) -> None:
        if self.started_at is None:
            return
        elapsed = max(0.0, time.monotonic() - self.started_at)
        self.elapsed_label.setText(f"已用时 {self._format_seconds(elapsed)}")
        if self.running:
            eta = self.analysis_eta
            if eta is not None:
                scope = "当前记录批次估算" if self.last_config is None else "估算"
                text = f"{self._format_seconds(eta)}（{scope}）"
            elif self.baseline_eta is not None:
                remaining = max(0.0, self.baseline_eta - elapsed)
                text = f"约 {self._format_seconds(remaining)}（上次运行基数）"
            elif self.current_stage in ("analysis_export", "extended_export", "extended_statistics"):
                text = "正在收尾"
            else:
                text = "正在收集速度"
            self.eta_label.setText(f"预计剩余时间 {text}")
        else:
            self.eta_label.setText("预计剩余时间 —")

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0, int(round(value)))
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    @Slot(dict)
    def _on_completed(self, result: dict[str, Any]) -> None:
        observed_elapsed = None
        if self.started_at is not None:
            observed_elapsed = max(0.0, time.monotonic() - self.started_at)
        self._set_run_directory(Path(result["run_directory"]))
        self.last_status = result["status"]
        if (result.get("qualification") == "exploratory_response_posting_proxy"
                and observed_elapsed is not None):
            try:
                from pyeda_eta import update_response_profile

                update_response_profile(PYEDA_ROOT / "study-config", elapsed_s=observed_elapsed,
                                        records=int(result.get("processed_records", 0) or 0),
                                        windows=int(result.get("response_feature_windows", 0) or 0))
                self.log.appendPlainText(f"已用本次实测耗时 {self._format_seconds(observed_elapsed)} 更新后续估时基数。")
            except Exception as error:
                self.log.appendPlainText(f"运行完成，但未更新估时基数：{error}")
        self.log.appendPlainText(f"输出目录：{self.latest_directory}")
        if self.last_status == "paused":
            self.stage_label.setText(f"已暂停，保存了 {result['records_saved']} 条记录。再次打开时自动继续。")
            return
        mode = result.get("analysis_mode")
        label = {"ready_for_execution": "正式分析", "ready_for_pilot": "试跑"}.get(mode, "AUT/PIT 探索分析")
        if result["status"] == "succeeded":
            self._set_overall(100)
            self.stage_label.setText(f"{label}已完成并导出")
            if mode is None:
                self.log.appendPlainText("可复制本次目录，交给 Codex 撰写 APA 报告。")
        else:
            issues = "\n".join(map(str, result.get("errors", [])))
            self.stage_label.setText(f"{label}已导出部分结果，请查看原因")
            self.log.appendPlainText(issues)
            if not self.close_when_finished and result.get("retryable", True):
                QMessageBox.warning(self, "分析未全部完成", f"{issues}\n{self.latest_directory}")

    def _set_run_directory(self, root: Path) -> None:
        if self.latest_directory == root:
            return
        self.latest_directory = root
        self.open_button.setEnabled(True)
        self.copy_button.setEnabled(True)
        self.log_button.setEnabled((root / "run.log").is_file())

    @Slot(str)
    def _on_blocked(self, message: str) -> None:
        self.stage_label.setText("分析未通过检查，请查看原因")
        self.log.appendPlainText(message)
        QMessageBox.warning(self, "分析检查", message)

    @Slot(str, str)
    def _on_failed(self, message: str, details: str) -> None:
        self.stage_label.setText("运行失败，已保留错误信息")
        self.log.appendPlainText(details)
        if self.close_when_finished:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("PYEDA 运行失败")
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(message)
        box.setInformativeText("核对提示后，点击“继续／重试”可从已保存的位置继续。")
        box.setDetailedText(details)
        box.exec()

    @Slot()
    def _worker_finished(self) -> None:
        self.running = False
        self._refresh_clock()
        self.started_at = None
        self.restart_button.setText("重新分析" if self.last_status == "succeeded" and self.last_config is None
                                    else "继续／重试")
        self.restart_button.setEnabled(True)
        self.bundle_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self.worker.deleteLater()
        self.worker = None
        if self.close_when_finished:
            self.close()

    def closeEvent(self, event) -> None:
        if self.running:
            if self.last_config is None:
                if not self.close_when_finished:
                    self.close_when_finished = True
                    self.worker.requestInterruption()
                    self.stage_label.setText("正在保存断点，随后关闭窗口…")
                    self.force_close_button.show()
            else:
                self.log.appendPlainText("已核验分析包正在运行，完成后可以关闭窗口。")
            event.ignore()
            return
        super().closeEvent(event)

    def _force_close(self) -> None:
        if self.close_when_finished and self.running:
            os._exit(0)

    def _restart(self) -> None:
        if self.last_config is not None:
            self._start_worker(self.last_config)
            return
        workers, accepted = QInputDialog.getInt(
            self, "本次并行设置", "CPU 并行上限（1 为串行，内存不足会自动降低）", 1, 1, 4)
        if accepted:
            self._start_worker(force_new=self.last_status == "succeeded", max_workers=workers)

    def _open_latest(self) -> None:
        if self.latest_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.latest_directory)))

    def _open_log(self) -> None:
        if self.latest_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.latest_directory / "run.log")))


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PYEDA 一站式导入与分析")
    app.setStyle("Fusion")
    name = "PYEDA-" + hashlib.sha256(str(PYEDA_ROOT.resolve()).encode("utf-8")).hexdigest()[:16]
    existing = QLocalSocket()
    existing.connectToServer(name)
    if existing.waitForConnected(250):
        existing.disconnectFromServer()
        return 0
    server = QLocalServer()
    if not server.listen(name):
        # A second launcher can arrive while the first one binds its endpoint.
        existing.abort()
        existing.connectToServer(name)
        if existing.waitForConnected(250):
            existing.disconnectFromServer()
            return 0
        QMessageBox.critical(None, "PYEDA 启动失败", server.errorString())
        return 1
    window = MainWindow()

    def show_existing():
        while server.hasPendingConnections():
            connection = server.nextPendingConnection()
            connection.disconnectFromServer()
            connection.deleteLater()
        if window.isMinimized():
            window.showNormal()
        window.raise_()
        window.activateWindow()

    server.newConnection.connect(show_existing)
    window.show()
    QTimer.singleShot(250, window.start_work)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
