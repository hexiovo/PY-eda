"""WB02 known-answer tests for group temporal feature extraction."""

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.config import get_preset
from eda_toolkit.core_sync import (extract_group_temporal_features,
                                   summarize_group_synchrony)


def _aligned_three():
    time = np.arange(10.0)
    keys = ["r1::eda", "r2::eda", "r3::eda"]
    subjects = ["s1", "s2", "s3"]
    membership = pd.DataFrame([{"group_id": "g1", "session_id": "ss1", "subject_id": subject,
                                "recording_id": f"r{i}", "channel": "eda"}
                               for i, subject in enumerate(subjects, 1)])
    return {"time_s": time, "signals": {key: np.arange(len(time), dtype=float) for key in keys},
            "eligible": {key: np.ones(len(time), bool) for key in keys},
            "metadata": {key: {"recording_id": f"r{i}", "channel": "eda", "subject_id": subjects[i - 1],
                               "session_id": "ss1"} for i, key in enumerate(keys, 1)},
            "membership": membership, "analysis_intervals_s": [[0.0, 9.0]],
            "alignment_report": {"analysis_rate_hz": 1.0, "formal_synchrony_allowed": True}}


def _pair_values():
    edge_rows = [("s1", "s2", 0.5), ("s1", "s3", -0.2), ("s2", "s3", 0.4)]
    windows = [("w1", 0.0, 4.0, 0.5), ("w2", 2.0, 6.0, 0.5), ("w3", 6.0, 9.0, -0.5)]
    rows = []
    for window_id, start, end, _ in windows:
        for a, b, correlation in edge_rows:
            rows.append({"group_id": "g1", "session_id": "ss1", "window_id": window_id,
                         "window_start_s": start, "window_end_s": end,
                         "pair_id": f"g1::ss1::{a}::{b}", "subject_id_a": a, "subject_id_b": b,
                         "record_key_a": f"{a.replace('s', 'r')}::eda",
                         "record_key_b": f"{b.replace('s', 'r')}::eda",
                         "metric": "pearson", "basis": "clean_uS", "lag_s": 0.0,
                         "r": correlation if window_id != "w3" else -correlation,
                         "status": "valid", "reason": "", "valid_n": 4, "duration_s": end - start,
                         "coverage": 1.0})
    return pd.DataFrame(rows)


def test_group_cells_do_not_double_count_overlap_and_episodes_stop_at_support_gap():
    aligned = _aligned_three()
    values = _pair_values()
    config = get_preset()
    summary = summarize_group_synchrony(values, aligned, config)
    features, episodes, nodes, diagnostics = extract_group_temporal_features(values, summary, aligned, config)
    temporal = features.loc[features["feature_id"].eq("sync_positive_time_proportion")].iloc[0]
    assert temporal["value"] == pytest.approx(5.75 / 8.75)
    diag = diagnostics.loc[diagnostics["group_id"].eq("g1") & diagnostics["metric"].eq("pearson")].iloc[0]
    assert diag["valid_class_duration_s"] == pytest.approx(8.75)
    assert diag["unknown_duration_s"] == pytest.approx(0.25)
    positive = episodes.loc[episodes["sign"].eq("positive")].iloc[0]
    assert positive["observed_duration_s"] == pytest.approx(5.75)
    assert bool(positive["left_censored"])
    assert bool(positive["right_censored"])
    assert features.loc[features.feature_id.eq("sync_positive_episode_duration_mean_s"), "status"].iloc[0] == "unavailable"


def test_lag_stability_uses_tail_selected_pair_lags_and_node_strength_keeps_signed_edges():
    aligned = _aligned_three()
    values = _pair_values()
    for row_idx, row in values.iterrows():
        # Add a second configured lag. For two-sided search, selected lag remains
        # the larger absolute correlation at lag zero for s1-s2 and s1-s3, etc.
        extra = row.copy()
        extra["lag_s"] = 1.0
        extra["r"] = (-0.9 if row["window_id"] == "w3" and row["subject_id_a"] == "s1" and
                       row["subject_id_b"] == "s3" else 0.1)
        values.loc[len(values)] = extra
    config = get_preset()
    config["synchrony"]["tail"] = "two-sided"
    summary = summarize_group_synchrony(values, aligned, config)
    features, _, nodes, _ = extract_group_temporal_features(values, summary, aligned, config)
    a = nodes.loc[nodes["subject_id"].eq("s1") & nodes["window_id"].eq("w1")].iloc[0]
    assert a["signed"] == pytest.approx(0.3)
    assert a["positive"] == pytest.approx(0.5)
    assert a["negative"] == pytest.approx(-0.2)
    assert a["absolute"] == pytest.approx(0.7)
    assert a["normalized_signed"] == pytest.approx(0.15)
    assert a["effective_degree"] == a["expected_degree"] == 2
    pair_sd = features.loc[features.feature_id.eq("sync_lag_sd_s") &
                           features.window_id.str.startswith("sync_pair") &
                           features.window_id.str.contains("s1::s3")]
    assert not pair_sd.empty
    assert (pair_sd["value"].dropna() > 0.0).all()


def _r2_group_fixture(*, omit_bc=False):
    time = np.arange(11.0)
    subjects = ["s1", "s2", "s3"]
    keys = [f"r{i}::c" for i in range(1, 4)]
    aligned = {"time_s": time,
               "signals": {key: np.arange(len(time), dtype=float) for key in keys},
               "eligible": {key: np.ones(len(time), dtype=bool) for key in keys},
               "metadata": {key: {"recording_id": f"r{i}", "subject_id": subjects[i - 1],
                                  "session_id": "ss", "channel": "c"}
                            for i, key in enumerate(keys, 1)},
               "membership": pd.DataFrame([{"group_id": "g", "session_id": "ss",
                                            "subject_id": subject, "recording_id": f"r{i}",
                                            "channel": "c"}
                                           for i, subject in enumerate(subjects, 1)]),
               "analysis_intervals_s": [[0.0, 10.0]],
               "alignment_report": {"analysis_rate_hz": 1.0, "formal_synchrony_allowed": True}}
    windows = [("w1", 0.0, 4.0), ("w2", 2.0, 6.0), ("w3", 6.0, 10.0)]
    edge_pairs = [("s1", "s2"), ("s1", "s3"), ("s2", "s3")]
    rows = []
    for window_id, start, end in windows:
        for a, b in edge_pairs:
            if omit_bc and (a, b) == ("s2", "s3"):
                continue
            for lag in (0.0, 2.0):
                if (a, b) == ("s1", "s2"):
                    corr = {("w1", 0.0): 0.7, ("w1", 2.0): 0.2,
                            ("w2", 0.0): 0.6, ("w2", 2.0): 0.1,
                            ("w3", 0.0): 0.1, ("w3", 2.0): -0.9}[(window_id, lag)]
                else:
                    corr = 0.4 if lag == 0.0 else 0.1
                rows.append({"group_id": "g", "session_id": "ss", "window_id": window_id,
                             "window_start_s": start, "window_end_s": end,
                             "pair_id": f"g::ss::{a}::{b}", "subject_id_a": a, "subject_id_b": b,
                             "record_key_a": f"r{subjects.index(a) + 1}::c",
                             "record_key_b": f"r{subjects.index(b) + 1}::c",
                             "metric": "pearson", "basis": "clean_uS", "lag_s": lag,
                             "r": corr, "status": "valid", "reason": "", "valid_n": 4,
                             "duration_s": end - start, "coverage": 1.0})
    return aligned, pd.DataFrame(rows)


def test_lag_weights_use_nonoverlapping_time_cells_and_pair_support_duration():
    aligned, values = _r2_group_fixture()
    config = get_preset("group_sync")
    config["synchrony"]["tail"] = "two-sided"
    config["features"]["extended"]["group_temporal"]["lag_stability_tolerance_s"] = 1.0
    summary = summarize_group_synchrony(values, aligned, config)
    features, _, _, diagnostics = extract_group_temporal_features(values, summary, aligned, config)
    pair = features.loc[features.feature_id.eq("sync_lag_sd_s") &
                        features.window_id.str.contains("s1::s2")].iloc[0]
    stable = features.loc[features.feature_id.eq("sync_lag_stability_proportion") &
                          features.window_id.str.contains("s1::s2")].iloc[0]
    assert pair.value == pytest.approx(np.sqrt(0.96))
    assert stable.value == pytest.approx(0.6)
    assert pair.valid_duration_s == pytest.approx(10.0)
    assert diagnostics.loc[diagnostics.group_id.eq("g") &
                           diagnostics.window_count.eq(3)].iloc[0].window_count == 3


def test_fixed_roster_degree_survives_an_edge_missing_from_all_windows():
    aligned, values = _r2_group_fixture(omit_bc=True)
    config = get_preset("group_sync")
    summary = summarize_group_synchrony(values, aligned, config)
    features, _, nodes, _ = extract_group_temporal_features(values, summary, aligned, config)
    w1 = nodes.loc[nodes.window_id.eq("w1")].set_index("subject_id")
    assert summary.loc[summary.summary_type.eq("all_fixed_members") & summary.window_id.eq("w1"),
                       "reason"].iloc[0] == "missing_fixed_member_edge"
    assert w1.loc["s1", "status"] == "valid"
    assert w1.loc["s1", "expected_degree"] == 2
    for subject in ("s2", "s3"):
        assert w1.loc[subject, "expected_degree"] == 2
        assert w1.loc[subject, "effective_degree"] == 1
        assert w1.loc[subject, "status"] == "unavailable"
        assert w1.loc[subject, "reason"] == "NA_missing_fixed_incident_edge"
        assert pd.isna(w1.loc[subject, "signed"])
    b_strength = features.loc[features.subject_id.eq("s2") &
                              features.window_id.str.endswith(":w1:pearson")]
    assert set(b_strength.status) == {"unavailable"}


def test_observed_edges_do_not_define_a_complete_graph_without_frozen_roster():
    aligned, values = _r2_group_fixture()
    aligned["membership"] = pd.DataFrame(columns=["group_id", "session_id", "subject_id",
                                                   "recording_id", "channel"])
    config = get_preset("group_sync")
    summary = summarize_group_synchrony(values, aligned, config)
    _, _, nodes, _ = extract_group_temporal_features(values, summary, aligned, config)
    w1 = nodes.loc[nodes.window_id.eq("w1")]
    assert set(w1.subject_id) == {"s1", "s2", "s3"}
    assert w1.expected_degree.isna().all()
    assert set(w1.status) == {"unavailable"}
    assert set(w1.reason) == {"NA_expected_graph_unknown"}
