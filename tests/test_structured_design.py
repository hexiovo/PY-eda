from __future__ import annotations

import copy
import os


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


def test_driver_form_exposes_every_field_and_preserves_loaded_precision():
    _app()
    from eda_toolkit.config import get_preset
    from eda_toolkit.ui.structured_design import DRIVER_FIELDS, DriverParameterForm

    driver = copy.deepcopy(get_preset()["decomposition"]["driver"])
    driver["gradient_tolerance"] = 1.234567890123e-12
    form = DriverParameterForm(driver)

    assert set(form.controls) == {field[0] for field in DRIVER_FIELDS} == set(driver)
    assert form.to_config() == driver
    form.controls["sparsity"].setValue(0.02)
    saved = form.to_config()
    assert saved["sparsity"] == 0.02
    assert saved["gradient_tolerance"] == driver["gradient_tolerance"]


def test_statistics_form_supports_all_methods_and_roundtrips_advanced_fields():
    _app()
    from eda_toolkit.ui.structured_design import STATISTICAL_DESIGNS, StatisticsDesignForm

    source = {
        "design_id": "primary", "outcome": "mean", "design_type": "mixed_effects",
        "independent_unit": "group", "condition_column": "phase", "conditions": ["pre", "post"],
        "group_column": "arm", "groups": ["control", "active"], "subject_column": "participant",
        "session_column": "visit", "group_id_column": "dyad", "reference": 0.25,
        "covariates": ["age", "baseline"], "random_intercept": False,
        "correction_family": "primary", "alternative": "greater", "aggregate": "median",
        "predictor": "age", "contrast": {"conditions": ["pre", "post"], "reference": 0.0},
    }
    form = StatisticsDesignForm(source)

    assert {form.kind.itemData(index) for index in range(form.kind.count())} == {
        identifier for identifier, _label in STATISTICAL_DESIGNS
    }
    assert form.independent_unit.currentData() == "group"
    assert form.to_config() == source

    form.outcome.setText("tonic_mean")
    saved = form.to_config()
    assert saved["outcome"] == "tonic_mean"
    assert {key: value for key, value in saved.items() if key != "outcome"} == {
        key: value for key, value in source.items() if key != "outcome"
    }


def test_clickable_combo_opens_from_the_center_of_the_control(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = _app()
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from eda_toolkit.ui.structured_design import ClickableComboBox

    combo = ClickableComboBox()
    combo.addItems(["线性", "保形插值"])
    combo.resize(220, 32)
    combo.show()
    app.processEvents()
    QTest.mouseClick(combo, Qt.MouseButton.LeftButton, pos=QPoint(80, 16))
    app.processEvents()
    assert combo.view().isVisible()
    combo.hidePopup()
