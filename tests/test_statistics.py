import numpy as np
import pandas as pd
import pytest
from scipy import stats

from eda_toolkit.config import resolve_config
from eda_toolkit.core_statistics import run_statistics


def frame(values):
    return pd.DataFrame({"subject_id": [f"{i:03d}" for i in range(len(values))], "value": values,
                         "session_id": "s", "status": "valid"})


def run(data, design, extra=None):
    config = resolve_config({"statistics": {"enabled": True, "designs": [design], **(extra or {})}})
    return run_statistics(data, None, config)


def test_one_sample_matches_reference_and_fixed_family():
    result, diagnostic = run(frame([1, 2, 3, 4]), {"design_id": "h1", "outcome": "value", "design_type": "one_sample"},
                             {"planned_hypotheses": ["h1", "h2", "h3"]})
    expected = stats.ttest_1samp([1, 2, 3, 4], 0)
    assert result.iloc[0].estimate == 2.5
    assert result.iloc[0].p_raw == pytest.approx(expected.pvalue)
    assert result.iloc[0].p_adjusted == pytest.approx(min(1, 3 * expected.pvalue))
    assert result.iloc[1:].p_raw.isna().all()
    assert diagnostic.iloc[0].status == "succeeded"


def test_repeated_windows_require_explicit_aggregation():
    data = pd.concat([frame([1, 2, 4]), frame([2, 3, 7])], ignore_index=True)
    design = {"design_id": "one", "outcome": "value", "design_type": "one_sample"}
    result, diagnostic = run(data, design)
    assert result.iloc[0].status == "unavailable"
    assert diagnostic.iloc[0].code == "pseudoreplication"
    result, _ = run(data, {**design, "aggregate": "mean"})
    assert result.iloc[0].n_observations == 3
    assert result.iloc[0].estimate == pytest.approx(np.mean([1.5, 2.5, 5.5]))


def test_paired_differences_and_missing_partner_are_reported():
    a, b = frame([3, 5, 8, 9]), frame([1, 2, 4, 7])
    a["condition"], b["condition"] = "A", "B"
    data = pd.concat([a, b.iloc[:3]], ignore_index=True)
    result, diagnostic = run(data, {"design_id": "paired", "outcome": "value", "design_type": "paired", "conditions": ["A", "B"]})
    assert result.iloc[0].estimate == 3
    assert result.iloc[0].p_raw == pytest.approx(stats.ttest_rel([3, 5, 8], [1, 2, 4]).pvalue)
    assert "003" in diagnostic.iloc[0].details


def test_welch_and_safe_design():
    data = frame([1, 3, 2, 5, 9, 8])
    data["experimental_group"] = ["A"] * 3 + ["B"] * 3
    result, _ = run(data, {"design_id": "w", "outcome": "value", "design_type": "welch", "groups": ["A", "B"]})
    assert result.iloc[0].p_raw == pytest.approx(stats.ttest_ind([1, 3, 2], [5, 9, 8], equal_var=False).pvalue)
    with pytest.raises(ValueError):
        run_statistics(data, {"outcome": "value", "formula": "__import__('os')"}, {})


def test_real_random_intercept_fit():
    rng = np.random.default_rng(41)
    rows = []
    for subject in range(16):
        intercept = rng.normal(0, 1.2)
        for condition in ("A", "B"):
            rows.append({"subject_id": str(subject), "session_id": "s", "condition": condition,
                         "value": intercept + (1.1 if condition == "B" else 0) + rng.normal(0, .2)})
    result, diagnostic = run(pd.DataFrame(rows), {"design_id": "lmm", "outcome": "value", "design_type": "mixed_effects", "conditions": ["A", "B"]})
    assert diagnostic.iloc[0].status == "succeeded", diagnostic.to_dict("records")
    coefficient = result.loc[result.contrast.eq("condition:B")].iloc[0]
    assert coefficient.status == "valid"
    assert .8 < coefficient.estimate < 1.4


def test_missing_condition_is_excluded_and_reported_instead_of_reference():
    rng = np.random.default_rng(23)
    rows = []
    for subject in range(12):
        base = rng.normal()
        for condition in ("A", "B"):
            rows.append({"subject_id": str(subject), "condition": condition,
                         "value": base + (condition == "B") + rng.normal(0, .1)})
    clean = pd.DataFrame(rows)
    missing = clean.loc[clean.condition.eq("A")].copy()
    missing["condition"] = None
    missing["value"] += 50
    design = {"design_id": "lmm", "outcome": "value", "design_type": "mixed_effects", "conditions": ["A", "B"]}
    baseline, _ = run(clean, design)
    result, diagnostics = run(pd.concat([clean, missing], ignore_index=True), design)
    assert result.estimate.to_list() == pytest.approx(baseline.estimate.to_list())
    assert diagnostics.iloc[0].excluded_n == 12
