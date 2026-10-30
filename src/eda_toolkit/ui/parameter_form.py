"""Structured controls for the approved extended-feature parameters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


_SERIES_FIELDS = (
    ("condition", "条件"),
    ("event_type", "事件类型"),
    ("stimulus_id", "刺激 ID"),
)


class CompactDoubleSpinBox(QDoubleSpinBox):
    """Keep full numeric precision without displaying fixed trailing zeroes."""

    def textFromValue(self, value: float) -> str:  # noqa: N802
        if value == 0:
            return "0"
        text = f"{value:.{self.decimals()}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text


class NumericListEditor(QWidget):
    """Editable ordered list of positive values with visible units."""

    def __init__(self, *, unit: str, minimum: float = 1e-12,
                 maximum: float = 1_000_000_000.0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self._unit = unit
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(5)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self._layout.addLayout(self._rows_layout)
        self.add_button = QPushButton(f"添加{unit}值")
        self.add_button.setMinimumHeight(30)
        self.add_button.clicked.connect(lambda: self.add_value())
        self._layout.addWidget(self.add_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self._spins: list[QDoubleSpinBox] = []

    def add_value(self, value: float | None = None) -> QDoubleSpinBox:
        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        spin = CompactDoubleSpinBox(row)
        spin.setRange(self._minimum, self._maximum)
        spin.setDecimals(12)
        spin.setSingleStep(0.5 if self._unit == "s" else 0.01)
        spin.setSuffix(f" {self._unit}")
        spin.setValue(float(value) if value is not None else (self._spins[-1].value() + 1.0 if self._spins else 1.0))
        spin.setObjectName(f"numericListValue_{len(self._spins)}")
        remove = QPushButton("移除")
        remove.setMinimumHeight(30)
        remove.setAccessibleName(f"移除第 {len(self._spins) + 1} 项")
        row_layout.addWidget(spin, 1)
        row_layout.addWidget(remove)
        self._rows_layout.addWidget(row)
        self._spins.append(spin)

        def delete_row() -> None:
            index = self._spins.index(spin)
            self._spins.pop(index)
            self._rows_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
            for position, control in enumerate(self._spins):
                control.setObjectName(f"numericListValue_{position}")

        remove.clicked.connect(delete_row)
        return spin

    def value(self) -> list[float]:
        return [float(spin.value()) for spin in self._spins]

    def set_value(self, values: list[float]) -> None:
        while self._spins:
            spin = self._spins.pop()
            row = spin.parentWidget()
            self._rows_layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        for value in values:
            self.add_value(float(value))


class NumericRangeEditor(QWidget):
    """Two-number interval editor used for normalization frequency bounds."""

    def __init__(self, *, unit: str, minimum: float, maximum: float,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.low = CompactDoubleSpinBox(self)
        self.high = CompactDoubleSpinBox(self)
        for spin in (self.low, self.high):
            spin.setRange(minimum, maximum)
            spin.setDecimals(12)
            spin.setSingleStep(0.01)
            spin.setSuffix(f" {unit}")
        self.low.setObjectName("normalizationBandLow")
        self.high.setObjectName("normalizationBandHigh")
        layout.addWidget(QLabel("下限", self))
        layout.addWidget(self.low, 1)
        layout.addWidget(QLabel("上限", self))
        layout.addWidget(self.high, 1)

    def value(self) -> list[float]:
        return [float(self.low.value()), float(self.high.value())]

    def set_value(self, values: list[float]) -> None:
        if len(values) != 2:
            return
        self.low.setValue(float(values[0]))
        self.high.setValue(float(values[1]))


class CoverageTable(QWidget):
    """Structured editor for complete annotation-coverage declarations."""

    HEADERS = ("记录 ID", "通道", "开始时间 (s)", "结束时间 (s)", "时钟域", "完整覆盖")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.table = QTableWidget(0, len(self.HEADERS), self)
        self.table.setObjectName("annotationCoverageTable")
        self.table.setHorizontalHeaderLabels(list(self.HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(112)
        self.table.setMaximumHeight(210)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 115)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 118)
        self.table.setColumnWidth(3, 118)
        self.table.setColumnWidth(4, 120)
        layout.addWidget(self.table)
        controls = QHBoxLayout()
        self.add_button = QPushButton("添加覆盖区间")
        self.remove_button = QPushButton("移除所选")
        for button in (self.add_button, self.remove_button):
            button.setMinimumHeight(30)
        controls.addWidget(self.add_button)
        controls.addWidget(self.remove_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.add_button.clicked.connect(self.add_row)
        self.remove_button.clicked.connect(self.remove_selected)

    @staticmethod
    def _text_cell(text: str = "") -> QLineEdit:
        edit = QLineEdit(text)
        edit.setMinimumWidth(75)
        return edit

    @staticmethod
    def _time_cell(value: float = 0.0) -> QDoubleSpinBox:
        spin = CompactDoubleSpinBox()
        spin.setRange(-1_000_000_000_000.0, 1_000_000_000_000.0)
        spin.setDecimals(9)
        spin.setSingleStep(1.0)
        spin.setSuffix(" s")
        spin.setValue(float(value))
        return spin

    def add_row(self, row_value: dict[str, Any] | None = None) -> None:
        row_value = row_value or {}
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setCellWidget(row, 0, self._text_cell(str(row_value.get("recording_id", ""))))
        self.table.setCellWidget(row, 1, self._text_cell(str(row_value.get("channel", ""))))
        self.table.setCellWidget(row, 2, self._time_cell(float(row_value.get("start_s", 0.0))))
        self.table.setCellWidget(row, 3, self._time_cell(float(row_value.get("end_s", 1.0))))
        self.table.setCellWidget(row, 4, self._text_cell(str(row_value.get("clock_domain", ""))))
        complete = QCheckBox("是")
        complete.setChecked(bool(row_value.get("complete", True)))
        complete.setToolTip("扩展指标声明只接受 complete=true。")
        self.table.setCellWidget(row, 5, complete)
        self.table.setRowHeight(row, 31)

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def value(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in range(self.table.rowCount()):
            result.append({
                "recording_id": self.table.cellWidget(row, 0).text().strip(),
                "channel": self.table.cellWidget(row, 1).text().strip(),
                "start_s": float(self.table.cellWidget(row, 2).value()),
                "end_s": float(self.table.cellWidget(row, 3).value()),
                "clock_domain": self.table.cellWidget(row, 4).text().strip(),
                "complete": self.table.cellWidget(row, 5).isChecked(),
            })
        return result

    def set_value(self, rows: list[dict[str, Any]]) -> None:
        self.table.setRowCount(0)
        for row in rows:
            self.add_row(row)


class SpectralBandsTable(QWidget):
    """Add/remove table for dynamic spectral bands."""

    HEADERS = ("频带 ID", "低频 (Hz)", "高频 (Hz)")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.table = QTableWidget(0, len(self.HEADERS), self)
        self.table.setObjectName("spectralBandsTable")
        self.table.setHorizontalHeaderLabels(list(self.HEADERS))
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(105)
        self.table.setMaximumHeight(190)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 190)
        layout.addWidget(self.table)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("添加频带")
        self.remove_button = QPushButton("移除所选")
        for button in (self.add_button, self.remove_button):
            button.setMinimumHeight(30)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.add_button.clicked.connect(self.add_row)
        self.remove_button.clicked.connect(self.remove_selected)

    @staticmethod
    def _frequency_spin(value: float) -> QDoubleSpinBox:
        spin = CompactDoubleSpinBox()
        spin.setRange(1e-12, 1_000_000_000.0)
        spin.setDecimals(12)
        spin.setSingleStep(0.01)
        spin.setSuffix(" Hz")
        spin.setMaximumWidth(185)
        spin.setValue(float(value))
        return spin

    def add_row(self, row_value: dict[str, Any] | None = None) -> None:
        row_value = row_value or {}
        row = self.table.rowCount()
        self.table.insertRow(row)
        band_id = str(row_value.get("id", f"custom_band_{row + 1}"))
        self.table.setCellWidget(row, 0, QLineEdit(band_id))
        self.table.setCellWidget(row, 1, self._frequency_spin(float(row_value.get("low_hz", 0.01))))
        self.table.setCellWidget(row, 2, self._frequency_spin(float(row_value.get("high_hz", 0.1))))
        self.table.setRowHeight(row, 31)

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def value(self) -> list[dict[str, Any]]:
        return [
            {
                "id": self.table.cellWidget(row, 0).text().strip(),
                "low_hz": float(self.table.cellWidget(row, 1).value()),
                "high_hz": float(self.table.cellWidget(row, 2).value()),
            }
            for row in range(self.table.rowCount())
        ]

    def set_value(self, rows: list[dict[str, Any]]) -> None:
        self.table.setRowCount(0)
        for row in rows:
            self.add_row(row)


class ParameterForm(QWidget):
    """A schema-shaped, JSON-free editor for one extended metric family."""

    TITLES = {
        "trial_series": "试次序列",
        "nonspecific": "非特异 SCR",
        "spectral": "频谱",
        "autocorrelation": "自相关",
        "complexity": "复杂度",
        "group_temporal": "群体时序",
    }

    def __init__(self, family: str, value: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if family not in self.TITLES:
            raise ValueError(f"未知扩展指标参数组：{family}")
        self.family = family
        self.field_widgets: dict[str, QWidget] = {}
        self._readers: dict[str, Callable[[], Any]] = {}
        self._writers: dict[str, Callable[[Any], None]] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(9)
        self._build()
        self.set_value(value)

    def _register(self, path: str, control: QWidget, reader: Callable[[], Any],
                  writer: Callable[[Any], None] | None = None) -> QWidget:
        control.setObjectName(f"parameter_{self.family}_{path.replace('.', '_')}")
        self.field_widgets[path] = control
        self._readers[path] = reader
        if writer is None:
            if isinstance(control, QCheckBox):
                writer = lambda value, target=control: target.setChecked(bool(value))
            elif isinstance(control, QComboBox):
                writer = lambda value, target=control: target.setCurrentIndex(max(0, target.findData(value)))
            elif isinstance(control, QSpinBox):
                writer = lambda value, target=control: target.setValue(int(value))
            elif isinstance(control, QDoubleSpinBox):
                writer = lambda value, target=control: target.setValue(float(value))
        if writer is not None:
            self._writers[path] = writer
        return control

    @staticmethod
    def _label(text: str, hint: str) -> QLabel:
        label = QLabel(text)
        label.setToolTip(hint)
        return label

    def _check(self, form: QFormLayout, path: str, label: str, *, hint: str = "") -> None:
        control = QCheckBox(label)
        control.setAccessibleName(label)
        control.setToolTip(hint)
        self._register(path, control, control.isChecked)
        form.addRow("", control)

    def _int(self, form: QFormLayout, path: str, label: str, *, minimum: int = 1,
             maximum: int = 1_000_000, suffix: str = "", step: int = 1) -> None:
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        control.setSuffix(suffix)
        control.setToolTip(f"允许范围：{minimum:,}–{maximum:,}{suffix}。")
        self._register(path, control, control.value)
        form.addRow(self._label(label, control.toolTip()), control)

    def _float(self, form: QFormLayout, path: str, label: str, *, minimum: float = 0.0,
               maximum: float = 1_000_000_000.0, decimals: int = 6,
               suffix: str = "", step: float = 0.1) -> None:
        control = CompactDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(max(decimals, 12 if minimum < 1e-9 else decimals))
        control.setSingleStep(step)
        control.setSuffix(suffix)
        control.setToolTip(f"允许范围：{minimum:g}–{maximum:g}{suffix}。")
        self._register(path, control, control.value)
        form.addRow(self._label(label, control.toolTip()), control)

    def _choice(self, form: QFormLayout, path: str, label: str,
                options: tuple[tuple[str, str], ...]) -> None:
        control = QComboBox()
        for value, text in options:
            control.addItem(text, value)
        self._register(path, control, control.currentData)
        form.addRow(label, control)

    def _group(self, title: str) -> QFormLayout:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._layout.addWidget(group)
        return form

    def _list_with_label(self, form: QFormLayout, path: str, label: str,
                         editor: QWidget, hint: str = "") -> None:
        self._register(path, editor, editor.value, editor.set_value)  # type: ignore[attr-defined]
        form.addRow(self._label(label, hint), editor)

    def _build(self) -> None:
        family = self.family
        title = self.TITLES[family]
        if family == "trial_series":
            form = self._group(title)
            series = QListSeriesEditor()
            self._list_with_label(form, "series_by", "序列分组字段", series,
                                  "勾选条件、事件类型或刺激 ID；至少选择一项。")
            for key, label in (
                ("min_slope_trials", "斜率拟合最少试次数"),
                ("min_logit_trials", "Logit 拟合最少试次数"),
                ("min_decay_trials", "衰减拟合最少试次数"),
                ("decay_max_nfev", "衰减拟合最大迭代次数"),
            ):
                self._int(form, key, label)
            self._float(form, "decay_max_rate", "衰减率上限 (1/s)", minimum=1e-12,
                        decimals=9, step=0.1)
            return

        if family == "nonspecific":
            form = self._group(title)
            self._check(form, "enabled", "启用非特异 SCR 指标")
            coverage = CoverageTable()
            self._list_with_label(form, "annotation_coverage", "人工覆盖声明", coverage,
                                  "记录 ID、通道、时钟域和非空 [开始, 结束) 秒区间；complete 必须为是。")
            return

        if family == "spectral":
            form = self._group("谱分析")
            self._check(form, "enabled", "启用频谱指标")
            self._choice(form, "basis", "信号基础", (("clean", "clean · 清洁信号"),))
            self._float(form, "segment_s", "Welch 分段长度 (s)", minimum=1e-12,
                        decimals=6, suffix=" s", step=1.0)
            self._float(form, "overlap", "分段重叠比例", minimum=0, maximum=0.999999999,
                        decimals=6, step=0.05)
            self._choice(form, "window", "窗函数", (("hann", "Hann"), ("boxcar", "矩形窗")))
            self._choice(form, "detrend", "去趋势", (("constant", "常量"), ("linear", "线性")))
            self._int(form, "min_segments", "最少分段数", maximum=100_000)
            self._float(form, "min_cycles", "最少周期数", minimum=1e-12, decimals=6,
                        suffix=" 周期", step=0.5)
            bands = SpectralBandsTable()
            self._list_with_label(form, "bands", "频带表", bands,
                                  "ID 需唯一并符合小写字母开头的 feature ID；高频必须大于低频。")
            interval = NumericRangeEditor(unit="Hz", minimum=1e-12, maximum=1_000_000_000.0)
            self._list_with_label(form, "normalization_band_hz", "归一化频带 (Hz)", interval,
                                  "必须有两个正端点，且上限大于下限。")
            return

        if family == "autocorrelation":
            form = self._group(title)
            self._check(form, "enabled", "启用自相关指标")
            lags = NumericListEditor(unit="s")
            self._list_with_label(form, "lags_s", "滞后列表 (s)", lags,
                                  "至少一个正值，且严格递增、不重复。")
            self._int(form, "min_pairs", "最少有效样本对数", maximum=1_000_000)
            return

        if family == "complexity":
            overview = self._group("复杂度共用设置")
            self._check(overview, "enabled", "启用复杂度指标")
            self._choice(overview, "basis", "信号基础", (("clean", "clean · 清洁信号"),))
            self._int(overview, "min_samples", "最少样本数", maximum=1_000_000_000)
            self._int(overview, "max_samples", "最多样本数", maximum=1_000_000_000)
            self._int(overview, "max_pair_comparisons", "最大成对比较次数", maximum=1_000_000_000)

            sampen = self._group("样本熵 (SampEn)")
            self._int(sampen, "sampen.m", "模板维数 m", minimum=2, maximum=5)
            self._int(sampen, "sampen.delay", "延迟", minimum=1, maximum=1000, suffix=" 个样本")
            self._choice(sampen, "sampen.r_mode", "容差模式",
                         (("sd_fraction", "标准差比例"), ("absolute", "绝对阈值")))
            self._float(sampen, "sampen.r", "容差 r", minimum=1e-12, decimals=9, step=0.05)

            permen = self._group("排列熵 (PermEn)")
            self._int(permen, "permen.dimension", "嵌入维数", minimum=2, maximum=7)
            self._int(permen, "permen.delay", "延迟", minimum=1, maximum=1000, suffix=" 个样本")
            self._check(permen, "permen.normalize", "归一化排列熵")
            self._choice(permen, "permen.ties", "并列值处理", (("stable", "稳定排序"), ("reject", "遇到并列值时拒绝")))

            dfa = self._group("去趋势波动分析 (DFA)")
            self._int(dfa, "dfa.order", "拟合阶数", minimum=1, maximum=1)
            self._int(dfa, "dfa.min_samples", "DFA 最少样本数", maximum=1_000_000)
            self._int(dfa, "dfa.min_scale", "最小尺度", minimum=2, maximum=1_000_000, suffix=" 个样本")
            self._float(dfa, "dfa.max_scale_fraction", "最大尺度比例", minimum=1e-12, maximum=1,
                        decimals=6, step=0.05)
            self._int(dfa, "dfa.scale_count", "尺度数", minimum=2, maximum=1_000_000)
            self._int(dfa, "dfa.min_scales", "最少有效尺度数", minimum=2, maximum=1_000_000)
            return

        if family == "group_temporal":
            form = self._group(title)
            self._check(form, "enabled", "启用群体时序指标")
            self._float(form, "correlation_threshold", "相关系数阈值", minimum=-1, maximum=1,
                        decimals=6, step=0.05)
            self._float(form, "lag_stability_tolerance_s", "滞后稳定容差 (s)", minimum=0,
                        decimals=6, suffix=" s", step=0.1)
            self._int(form, "min_valid_windows", "最少有效窗口数", maximum=1_000_000)

    def value(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for path, read in self._readers.items():
            parts = path.split(".")
            target = result
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = read()
        return result

    def set_value(self, value: dict[str, Any]) -> None:
        for path, write in self._writers.items():
            current: Any = value
            for part in path.split("."):
                if not isinstance(current, dict) or part not in current:
                    current = None
                    break
                current = current[part]
            if current is not None:
                write(deepcopy(current))


class QListSeriesEditor(QWidget):
    """Checkable set editor for the supported trial grouping dimensions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.checks: dict[str, QCheckBox] = {}
        for value, label in _SERIES_FIELDS:
            check = QCheckBox(label)
            check.setObjectName(f"seriesBy_{value}")
            check.setProperty("seriesValue", value)
            self.checks[value] = check
            layout.addWidget(check)

    def value(self) -> list[str]:
        return [value for value, _label in _SERIES_FIELDS if self.checks[value].isChecked()]

    def set_value(self, values: list[str]) -> None:
        selected = set(values)
        for value, check in self.checks.items():
            check.setChecked(value in selected)
