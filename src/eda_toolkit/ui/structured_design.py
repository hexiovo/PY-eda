"""Reusable, Chinese-labeled configuration forms for analysis designs."""

from __future__ import annotations

import copy
import sys
from typing import Any

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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class ClickableComboBox(QComboBox):
    """Open the choices when any part of the combo is clicked."""

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.button() == Qt.MouseButton.LeftButton:
            self.showPopup()
            event.accept()
            return
        super().mousePressEvent(event)


class PreciseDriverSpinBox(QDoubleSpinBox):
    """Display full precision compactly instead of padding 17 decimal places."""

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt API name
        return "0" if value == 0 else format(value, ".15g")


DRIVER_FIELDS = (
    ("tau_rise_s", "上升时间常数 τ₁ (s)", 1e-12, sys.float_info.max, "Bateman 响应核的上升时间。"),
    ("tau_decay_s", "衰减时间常数 τ₂ (s)", 1e-12, sys.float_info.max, "必须大于上升时间的 1.01 倍。"),
    ("tonic_knot_s", "Tonic 节点间隔 (s)", 1e-12, sys.float_info.max, "Tonic 平滑曲线的节点间隔，不能短于衰减时间。"),
    ("sparsity", "稀疏惩罚系数", 0.0, sys.float_info.max, "控制 Driver 稀疏程度。"),
    ("ridge", "Ridge 稳定化系数", 1e-12, sys.float_info.max, "用于数值稳定的二次惩罚系数。"),
    ("tonic_smoothness", "Tonic 平滑系数", 0.0, sys.float_info.max, "控制 Tonic 曲线的平滑程度。"),
    ("kernel_tail_tolerance", "响应核尾部容差", 1e-8, 0.01, "用于截断响应核的尾部容差。"),
    ("max_iterations", "最大迭代次数", 1, 100000, "优化器允许的最大迭代数。"),
    ("gradient_tolerance", "梯度收敛容差", 1e-12, sys.float_info.max, "优化器的梯度收敛阈值。"),
    ("function_tolerance", "目标函数收敛容差", 1e-12, sys.float_info.max, "优化器的目标函数变化阈值。"),
    ("kkt_tolerance", "KKT 收敛容差", 1e-12, sys.float_info.max, "优化器的 KKT 条件阈值。"),
    ("max_segment_samples", "单段最大样本数", 10, 1000000, "超出预算的片段会报告为不可处理。"),
    ("burst_threshold_uS_s", "爆发阈值 (μS/s)", 1e-12, sys.float_info.max, "用于 Driver 爆发识别的阈值。"),
)


class DriverParameterForm(QWidget):
    """Numeric form for every supported Bateman Driver parameter."""

    def __init__(self, values: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.original = copy.deepcopy(values)
        self._dirty: set[str] = set()
        self.controls: dict[str, QWidget] = {}
        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        for key, label, minimum, maximum, tooltip in DRIVER_FIELDS:
            if key in ("max_iterations", "max_segment_samples"):
                control: QWidget = QSpinBox(self)
                control.setRange(int(minimum), int(maximum))
                control.setValue(int(values[key]))
            else:
                control = PreciseDriverSpinBox(self)
                control.setDecimals(17)
                control.setRange(float(minimum), float(maximum))
                control.setSingleStep(0.01 if minimum == 0 else max(float(minimum), 0.01))
                control.setValue(float(values[key]))
            control.setMaximumWidth(210)
            control.setToolTip(tooltip)
            control.setObjectName(f"driver_{key}")
            self.controls[key] = control
            control.valueChanged.connect(lambda *_args, field=key: self._dirty.add(field))
            note = QLabel(tooltip, self)
            note.setObjectName("subtle")
            note.setMinimumWidth(280)
            note.setWordWrap(True)
            field = QWidget(self)
            row = QHBoxLayout(field)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            row.addWidget(control)
            row.addWidget(note, 1)
            layout.addRow(label, field)

    def to_config(self) -> dict[str, int | float]:
        result = copy.deepcopy(self.original)
        for key, control in self.controls.items():
            if key in self._dirty:
                result[key] = int(control.value()) if isinstance(control, QSpinBox) else float(control.value())
        return result

    def set_config(self, values: dict[str, Any]) -> None:
        self.original = copy.deepcopy(values)
        self._dirty.clear()
        for key, control in self.controls.items():
            control.blockSignals(True)
            control.setValue(int(values[key]) if isinstance(control, QSpinBox) else float(values[key]))
            control.blockSignals(False)


STATISTICAL_DESIGNS = (
    ("descriptive", "描述性统计"),
    ("one_sample", "单样本 t 检验"),
    ("paired", "配对 t 检验"),
    ("welch", "Welch 独立组 t 检验"),
    ("correlation", "Pearson 相关"),
    ("mixed_effects", "随机截距混合模型"),
    ("wilcoxon", "Wilcoxon 配对秩检验"),
    ("mannwhitney", "Mann–Whitney U 检验"),
)


def _text_edit(value: str = "", *, placeholder: str = "") -> QPlainTextEdit:
    edit = QPlainTextEdit()
    edit.setPlainText(value)
    edit.setPlaceholderText(placeholder)
    edit.setMaximumHeight(62)
    return edit


def _line_edit(value: str = "", *, placeholder: str = "") -> QLineEdit:
    edit = QLineEdit(value)
    edit.setPlaceholderText(placeholder)
    edit.setMaximumWidth(380)
    return edit


class StatisticsDesignForm(QGroupBox):
    """One structured statistics design, including advanced supported fields."""

    def __init__(self, design: dict[str, Any] | None = None, *, index: int = 1,
                 remove_callback=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.original = copy.deepcopy(design or {})
        self.remove_callback = remove_callback
        self._dirty: set[str] = set()
        self.setObjectName("statisticsDesignCard")
        outer = QVBoxLayout(self)
        heading_row = QHBoxLayout()
        self.card_heading = QLabel()
        self.card_heading.setStyleSheet("font-weight: 700; color: #142B45;")
        heading_row.addWidget(self.card_heading)
        heading_row.addStretch(1)
        self.remove_button = QPushButton("删除此设计")
        self.remove_button.setObjectName("danger")
        self.remove_button.setMaximumWidth(140)
        self.remove_button.clicked.connect(lambda: self.remove_callback(self) if self.remove_callback else None)
        heading_row.addWidget(self.remove_button)
        outer.addLayout(heading_row)

        common_form = QFormLayout()
        self.design_id = _line_edit(str(self.original.get("design_id", f"design_{index}")),
                                    placeholder="例如 primary_outcome")
        self.outcome = _line_edit(str(self.original.get("outcome", "")),
                                  placeholder="分析结果列名或 feature ID")
        self.kind = ClickableComboBox(self)
        for data, label in STATISTICAL_DESIGNS:
            self.kind.addItem(label, data)
        self.kind.setCurrentIndex(max(0, self.kind.findData(self.original.get("design_type", "descriptive"))))
        self.independent_unit = ClickableComboBox(self)
        for data, label in (("subject_id", "被试 ID"), ("group_id", "互动组 ID"), ("session_id", "会话 ID")):
            self.independent_unit.addItem(label, data)
        unit_value = self.original.get("independent_unit", "subject_id")
        unit_index = self.independent_unit.findData(unit_value)
        if unit_index < 0 and "independent_unit" in self.original:
            self.independent_unit.insertItem(0, f"自定义字段 · {unit_value}", unit_value)
            unit_index = 0
        self.independent_unit.setCurrentIndex(max(0, unit_index))
        self.independent_unit.setMaximumWidth(250)
        common_form.addRow("设计 ID", self.design_id)
        common_form.addRow("结局字段", self.outcome)
        common_form.addRow("统计方法", self.kind)
        common_form.addRow("独立单位", self.independent_unit)
        common_box = QGroupBox("基本信息")
        common_box.setLayout(common_form)
        outer.addWidget(common_box)

        method_box = QGroupBox("条件、组别与模型参数")
        method_form = QFormLayout(method_box)
        self.condition_column = _line_edit(str(self.original.get("condition_column", "condition")),
                                           placeholder="例如 condition")
        self.conditions = _text_edit("\n".join(map(str, self.original.get("conditions", []))),
                                     placeholder="每行一个条件水平；配对检验需两个水平")
        self.group_column = _line_edit(str(self.original.get("group_column", "experimental_group")),
                                       placeholder="例如 experimental_group")
        self.groups = _text_edit("\n".join(map(str, self.original.get("groups", []))),
                                 placeholder="每行一个组别；独立组比较需两个组")
        self.predictor = _line_edit(str(self.original.get("predictor", "")), placeholder="相关分析的预测变量字段")
        self.reference = _line_edit(str(self.original.get("reference", "")), placeholder="单样本检验的数值参照")
        method_form.addRow("条件字段", self.condition_column)
        method_form.addRow("条件水平", self.conditions)
        method_form.addRow("组别字段", self.group_column)
        method_form.addRow("组别水平", self.groups)
        method_form.addRow("相关预测变量", self.predictor)
        method_form.addRow("单样本参照值", self.reference)
        outer.addWidget(method_box)

        advanced_toggle = QToolButton(self)
        advanced_toggle.setText("显示高级模型字段")
        advanced_toggle.setCheckable(True)
        advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        advanced_toggle.toggled.connect(self._toggle_advanced)
        outer.addWidget(advanced_toggle)
        self.advanced_box = QGroupBox("高级模型字段（保存时保留未修改的原值）", self)
        advanced_form = QFormLayout(self.advanced_box)
        self.subject_column = _line_edit(str(self.original.get("subject_column", "subject_id")))
        self.session_column = _line_edit(str(self.original.get("session_column", "session_id")))
        self.group_id_column = _line_edit(str(self.original.get("group_id_column", "group_id")))
        self.covariates = _text_edit("\n".join(map(str, self.original.get("covariates", []))),
                                     placeholder="每行一个数值协变量")
        self.random_intercept = QCheckBox("包含被试随机截距")
        self.random_intercept.setChecked(bool(self.original.get("random_intercept", True)))
        self.correction_family = _line_edit(str(self.original.get("correction_family", "default")))
        self.alternative = ClickableComboBox(self)
        for data, label in (("two-sided", "双侧"), ("less", "小于"), ("greater", "大于")):
            self.alternative.addItem(label, data)
        self.alternative.setCurrentIndex(max(0, self.alternative.findData(
            self.original.get("alternative", "two-sided"))))
        self.aggregate = ClickableComboBox(self)
        for data, label in (("", "不指定"), ("mean", "均值"), ("median", "中位数")):
            self.aggregate.addItem(label, data)
        aggregate_value = self.original.get("aggregate", "")
        self.aggregate.setCurrentIndex(max(0, self.aggregate.findData(aggregate_value)))
        for key, label, control in (
            ("subject_column", "混合模型被试字段", self.subject_column),
            ("session_column", "会话字段", self.session_column),
            ("group_id_column", "互动组字段", self.group_id_column),
            ("covariates", "数值协变量", self.covariates),
            ("random_intercept", "随机结构", self.random_intercept),
            ("correction_family", "多重比较分组", self.correction_family),
            ("alternative", "检验方向", self.alternative),
            ("aggregate", "重复窗口汇总", self.aggregate),
        ):
            advanced_form.addRow(label, control)

        contrast_group = QGroupBox("结构化 contrast（可选）", self.advanced_box)
        contrast_form = QFormLayout(contrast_group)
        contrast = self.original.get("contrast", {})
        self.contrast_condition_column = _line_edit(str(contrast.get("condition_column", "")))
        self.contrast_conditions = _text_edit("\n".join(map(str, contrast.get("conditions", []))))
        self.contrast_group_column = _line_edit(str(contrast.get("group_column", "")))
        self.contrast_groups = _text_edit("\n".join(map(str, contrast.get("groups", []))))
        self.contrast_reference = _line_edit(str(contrast.get("reference", "")))
        self.contrast_predictor = _line_edit(str(contrast.get("predictor", "")))
        contrast_form.addRow("Contrast 条件字段", self.contrast_condition_column)
        contrast_form.addRow("Contrast 条件水平", self.contrast_conditions)
        contrast_form.addRow("Contrast 组别字段", self.contrast_group_column)
        contrast_form.addRow("Contrast 组别水平", self.contrast_groups)
        contrast_form.addRow("Contrast 参照值", self.contrast_reference)
        contrast_form.addRow("Contrast 预测变量", self.contrast_predictor)
        advanced_form.addRow(contrast_group)
        self.advanced_box.setVisible(False)
        outer.addWidget(self.advanced_box)

        self._connect_dirty(self.design_id, "design_id")
        self._connect_dirty(self.outcome, "outcome")
        self._connect_dirty(self.kind, "design_type")
        self._connect_dirty(self.independent_unit, "independent_unit")
        for field_name in ("condition_column", "conditions", "group_column", "groups", "predictor", "reference",
                           "subject_column", "session_column", "group_id_column", "covariates",
                           "random_intercept", "correction_family", "alternative", "aggregate"):
            self._connect_dirty(getattr(self, field_name), field_name)
        for control in (self.contrast_condition_column, self.contrast_conditions, self.contrast_group_column,
                        self.contrast_groups, self.contrast_reference, self.contrast_predictor):
            self._connect_dirty(control, "contrast")
        self.kind.currentIndexChanged.connect(lambda *_: self._update_heading(index))
        self._update_heading(index)

    def _toggle_advanced(self, expanded: bool) -> None:
        self.advanced_box.setVisible(expanded)
        self.sender().setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.sender().setText("收起高级模型字段" if expanded else "显示高级模型字段")

    def _update_heading(self, index: int) -> None:
        method_name = self.kind.currentText()
        self.card_heading.setText(f"设计 {index} · {method_name}")

    def _connect_dirty(self, control: QWidget, field: str) -> None:
        if isinstance(control, QLineEdit):
            control.textChanged.connect(lambda *_: self._dirty.add(field))
        elif isinstance(control, QPlainTextEdit):
            control.textChanged.connect(lambda: self._dirty.add(field))
        elif isinstance(control, QCheckBox):
            control.toggled.connect(lambda *_: self._dirty.add(field))
        elif isinstance(control, QComboBox):
            control.currentIndexChanged.connect(lambda *_: self._dirty.add(field))

    @staticmethod
    def _lines(edit: QPlainTextEdit) -> list[str]:
        return [line.strip() for line in edit.toPlainText().splitlines() if line.strip()]

    def to_config(self) -> dict[str, Any]:
        result = copy.deepcopy(self.original)
        result.update({
            "design_id": self.design_id.text().strip(),
            "outcome": self.outcome.text().strip(),
            "design_type": self.kind.currentData(),
            "independent_unit": self.independent_unit.currentData(),
        })
        values = {
            "condition_column": self.condition_column.text().strip(),
            "conditions": self._lines(self.conditions),
            "group_column": self.group_column.text().strip(),
            "groups": self._lines(self.groups),
            "predictor": self.predictor.text().strip(),
            "subject_column": self.subject_column.text().strip(),
            "session_column": self.session_column.text().strip(),
            "group_id_column": self.group_id_column.text().strip(),
            "covariates": self._lines(self.covariates),
            "correction_family": self.correction_family.text().strip(),
            "alternative": self.alternative.currentData(),
            "aggregate": self.aggregate.currentData(),
        }
        for field, value in values.items():
            if field in self._dirty:
                if value or field == "aggregate" and value:
                    result[field] = value
                else:
                    result.pop(field, None)
        if "random_intercept" in self._dirty:
            result["random_intercept"] = self.random_intercept.isChecked()
        if "reference" in self._dirty:
            text = self.reference.text().strip()
            if text:
                result["reference"] = float(text)
            else:
                result.pop("reference", None)
        if "contrast" in self._dirty:
            contrast: dict[str, Any] = {}
            for key, value in (
                ("condition_column", self.contrast_condition_column.text().strip()),
                ("conditions", self._lines(self.contrast_conditions)),
                ("group_column", self.contrast_group_column.text().strip()),
                ("groups", self._lines(self.contrast_groups)),
                ("predictor", self.contrast_predictor.text().strip()),
            ):
                if value:
                    contrast[key] = value
            reference = self.contrast_reference.text().strip()
            if reference:
                contrast["reference"] = float(reference)
            if contrast:
                result["contrast"] = contrast
            else:
                result.pop("contrast", None)
        return result


__all__ = ["ClickableComboBox", "DRIVER_FIELDS", "DriverParameterForm", "STATISTICAL_DESIGNS",
           "StatisticsDesignForm"]
