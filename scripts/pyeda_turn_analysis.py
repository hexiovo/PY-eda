"""Participant-level response-linked EDA features and exploratory inference."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
import unicodedata
from uuid import uuid4
import warnings

import numpy as np
import openpyxl
import pandas as pd
from scipy import signal, stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from eda_toolkit.core_export import atomic_json

from pyeda_one_stop import _emit_progress, load_design, sha256_file
from pyeda_aut_pit import (
    LOCAL_TIMEZONE, StudyPaused, append_log, archived_markers, artifacts_match,
    collect_raw_members, study_lock,
)

DATA_ROOT = Path(r"D:\data\PYEDA")
TASK_INPUTS = {
    "AUT": ("_aut_rawdata.xlsx", "chat_data", "individual_data"),
    "PIT": ("_pit_rawdata_rerate.xlsx", "chat", "individual"),
}
OUTCOMES = ["novelty_mean", "usefulness_mean", "fluency_ind", "flexibility_ind", "ioc_proportion"]
WINDOWS = ["turn_interval", "post_1_6s"]
RULES = {
    "time": "UTC+08 assumed; posting timestamps, unverified clock anchors",
    "turn_interval": "[current posting, next posting), last response unavailable",
    "post_1_6s": "[posting + 1 s, min(posting + 6 s, next posting)); at least 3 s",
    "baseline": "[posting - 5 s, posting), at least four eligible blocks and 80% coverage",
    "ioc": "within task x team, ordered by timestamp, idea_sequence and source row; 1 if ideatype_rater2 equals previous, else 0",
    "missing_ioc": "first response or either missing category -> missing; no cross-task, cross-team or missing-category skip",
    "identity": "proposed time linkage plus unique same-team surname prefix AND worksheet role agreement",
    "coverage": .8, "minimum_blocks": 3,
    "peak": "relative local maxima in contiguous phasic runs, prominence >= 0.5 x recording MAD x 1.4826, distance >= 2 s",
    "averaging": "equal eligible event weights within participant x task x role x window; count features expressed as rates",
    "independence": "teams for t/Welch/Levene/correlation; team-clustered GEE for participant and event regression",
    "multiplicity": "Holm separately by statistical method and outcome, including unavailable planned slots",
}


def feature_definitions():
    rows = []
    def add(name, family, formula, minimum=3):
        rows.append(dict(feature=name, family=family, definition=formula, minimum_blocks=minimum,
                         unit="dimensionless or derived seconds/minutes", support="supported"))
    for family in ("relative", "tonic", "phasic"):
        for suffix, formula in (
            ("mean", "arithmetic mean"), ("median", "median"), ("sd", "sample SD, ddof=1"),
            ("iqr", "75th minus 25th percentile"), ("range", "max minus min"),
        ):
            add(f"{family}_{suffix}", family, formula)
    for suffix, formula, minimum in (
        ("p05", "5th percentile", 5), ("p95", "95th percentile", 5),
        ("rmssd", "RMS of differences of adjacent eligible blocks only", 4),
        ("mean_abs_derivative", "mean absolute difference of adjacent eligible 1 s blocks", 4),
        ("slope_per_min", "OLS slope on actual elapsed minutes", 5),
        ("skew", "bias-corrected sample skewness", 16),
        ("excess_kurtosis", "bias-corrected Fisher excess kurtosis", 20),
    ):
        add(f"relative_{suffix}", "relative", formula, minimum)
    add("tonic_slope_per_min", "tonic", "OLS slope on actual elapsed minutes", 5)
    for suffix, formula in (
        ("rms", "sqrt(mean(phasic squared))"), ("abs_mean", "mean absolute phasic"),
        ("p95", "95th percentile of phasic"), ("positive_area_per_s", "mean positive phasic over eligible 1 s blocks"),
    ):
        add(f"phasic_{suffix}", "phasic", formula)
    for suffix, formula in (
        ("rate_per_min", "relative peak count / eligible seconds x 60"),
        ("mean_prominence", "mean prominence; zero with sufficient exposure and no detected peaks"),
        ("max_prominence", "maximum prominence; zero with sufficient exposure and no detected peaks"),
    ):
        add(f"relative_peak_{suffix}", "relative_peak", formula, 8)
    for name, formula in (
        ("relative_mean_change", "window mean relative minus pre-posting baseline mean relative"),
        ("tonic_mean_change", "window mean tonic minus pre-posting baseline mean tonic"),
        ("phasic_rms_change", "window phasic RMS minus pre-posting baseline phasic RMS"),
    ):
        add(name, "baseline_change", formula)
    add("relative_psd_001_005", "spectrum", "Welch integral over 0.01-0.05 Hz, contiguous segment >= 300 s", 300)
    add("relative_psd_005_025", "spectrum", "Welch integral over 0.05-0.25 Hz, contiguous segment >= 60 s", 60)
    return pd.DataFrame(rows)


DEFINITIONS = feature_definitions()
FEATURES = DEFINITIONS.feature.tolist()


def read_sheet(path, sheet):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = book[sheet].values
        header = list(next(rows))
        keep = [(i, name) for i, name in enumerate(header) if name]
        return pd.DataFrame(([row[i] for i, _ in keep] for row in rows), columns=[name for _, name in keep])
    finally:
        book.close()


def load_events_and_people(dv_dir):
    event_parts, people_parts = [], []
    for task, (filename, chat, individual) in TASK_INPUTS.items():
        source = read_sheet(dv_dir / filename, chat)
        required = ["group_id", "time", "participant_label", "group_type", "identity", "ideatype_rater2", "idea_sequence"]
        if not set(required).issubset(source):
            raise ValueError(f"{filename} 缺少所需列。")
        e = source[required].copy()
        e["source_row"] = np.arange(len(e)) + 2
        e["task"] = task
        e["group_id"] = e.group_id.astype(str).str.strip().str.upper()
        e["group_type"] = e.group_type.astype(str).str.casefold()
        e["participant_label"] = e.participant_label.astype(str).str.casefold()
        e["time"] = pd.to_datetime(e.time, errors="raise")
        if e.time.isna().any() or not e.group_type.isin(["ai", "human"]).all():
            raise ValueError(f"{filename} 有不可用的时间或团队类型。")
        e["idea_sequence"] = pd.to_numeric(e.idea_sequence, errors="raise")
        if e.duplicated(["group_id", "idea_sequence"]).any():
            raise ValueError(f"{filename} 同一团队存在重复回答序号。")
        e = e.sort_values(["group_id", "time", "idea_sequence", "source_row"], kind="stable")
        e["category"] = e.ideatype_rater2.map(
            lambda v: None if pd.isna(v) or not str(v).strip() else unicodedata.normalize("NFKC", str(v)).strip())
        prior = e.groupby("group_id", sort=False).category.shift()
        first = e.groupby("group_id", sort=False).cumcount().eq(0)
        e["ioc"] = np.where(first | e.category.isna() | prior.isna(), np.nan, e.category.eq(prior).astype(float))
        e["previous_actor"] = e.groupby("group_id", sort=False).participant_label.shift()
        e["speaker_changed"] = ~e.participant_label.eq(e.previous_actor)
        e.loc[first, "speaker_changed"] = False
        e["timestamp_ms"] = e.time.dt.tz_localize(LOCAL_TIMEZONE).astype("datetime64[ms, UTC]").astype("int64")
        e["next_ms"] = e.groupby("group_id", sort=False).timestamp_ms.shift(-1)
        e["event_id"] = [f"{task}-{g}-{int(i)}" for g, i in zip(e.group_id, e.idea_sequence)]
        event_parts.append(e)
        people = read_sheet(dv_dir / filename, individual)
        people["task"] = task
        people["group_id"] = people.group_id.astype(str).str.strip().str.upper()
        people["participant_label"] = people.participant_label.astype(str).str.casefold()
        keep = ["task", "group_id", "participant_label", *[x for x in OUTCOMES[:-1] if x in people]]
        people = people[keep].copy()
        if people.duplicated(["task", "group_id", "participant_label"]).any():
            raise ValueError(f"{filename} 个人表存在重复人员键。")
        people_parts.append(people)
    events = pd.concat(event_parts, ignore_index=True)
    people = pd.concat(people_parts, ignore_index=True)
    ioc = events[events.participant_label.ne("ai")].groupby(["task", "group_id", "participant_label"]).ioc.agg(
        ioc_successes="sum", ioc_trials="count", ioc_proportion="mean").reset_index()
    return events, people.merge(ioc, how="left", on=["task", "group_id", "participant_label"], validate="1:1")


def identity_mapping(base, events, design_path):
    roster = pd.read_csv(base / "intake/mapping_review.csv", dtype=str, keep_default_na=False)
    designs = {(d["sheet"], str(d["source_row"])): d for d in load_design(design_path)}
    rows = []
    for r in roster.to_dict("records"):
        if not r["recording_id"]:
            continue
        for task in TASK_INPUTS:
            row = {key: r[key] for key in ("recording_id", "subject_id", "group_id", "design_sheet", "status", "archive", "member")}
            row.update(task=task, participant_label="", identity_status="raw_design_link_needs_review")
            if r["status"] == "proposed_match":
                d = designs[(r["design_sheet"], r["design_source_row"])]
                surname = "".join(unicodedata.normalize("NFKC", d["participant_label"]).split())
                candidates = events[(events.task == task) & (events.group_id == r["group_id"])
                                    & events.participant_label.isin(["human1", "human2"])][["identity", "participant_label"]].drop_duplicates()
                matches = candidates[candidates.identity.astype(str).str.startswith(surname)]
                expected = "human2" if r["design_sheet"] == "human_human2" else "human1"
                row["identity_status"] = "missing_or_ambiguous_surname"
                if len(matches) == 1 and matches.iloc[0].participant_label == expected:
                    row.update(participant_label=expected, identity_status="candidate_surname_and_sheet_agree")
            rows.append(row)
    frame = pd.DataFrame(rows)
    usable = frame.identity_status.eq("candidate_surname_and_sheet_agree")
    duplicated = frame.loc[usable].duplicated(["task", "group_id", "participant_label"], keep=False)
    frame.loc[frame.loc[usable].index[duplicated], "identity_status"] = "duplicate_recording_for_person"
    return frame


def relative_peaks(ts, phasic):
    finite = phasic[np.isfinite(phasic)]
    scale = stats.median_abs_deviation(finite, scale="normal") if len(finite) else np.nan
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.array([], dtype=np.int64), np.array([]), False
    valid = np.flatnonzero(np.isfinite(phasic))
    splits = np.flatnonzero(np.diff(ts[valid]) != 1000) + 1
    locations, prominences = [], []
    for segment in np.split(valid, splits):
        if len(segment) >= 3:
            indices, values = signal.find_peaks(phasic[segment], prominence=.5 * scale, distance=2)
            locations.extend(ts[segment[indices]])
            prominences.extend(values["prominences"])
    return np.asarray(locations), np.asarray(prominences), True


def window_features(ts, arrays, start, stop, baseline, peaks):
    result = dict.fromkeys(FEATURES, np.nan)
    expected = (stop - start) / 1000 if np.isfinite(stop) else 0
    mask = (ts >= start) & (ts < stop) & np.isfinite(arrays["tonic"]) if expected > 0 else np.zeros(len(ts), bool)
    n = int(mask.sum())
    coverage = min(n / expected, 1.) if expected > 0 else 0.
    status = "valid" if expected >= 3 and n >= 3 and coverage >= RULES["coverage"] else "insufficient_exposure"
    if status != "valid":
        return result, n, coverage, status
    t = ts[mask]
    adjacent = np.diff(t) == 1000
    for prefix, x in arrays.items():
        x = x[mask]
        result.update({f"{prefix}_mean": np.mean(x), f"{prefix}_median": np.median(x),
                       f"{prefix}_sd": np.std(x, ddof=1), f"{prefix}_iqr": np.ptp(np.quantile(x, [.25, .75])),
                       f"{prefix}_range": np.ptp(x)})
        if prefix in ("relative", "tonic") and n >= 5:
            result[f"{prefix}_slope_per_min"] = stats.linregress((t-t[0])/60000, x).slope
        if prefix == "relative":
            if n >= 5:
                result.update(relative_p05=np.quantile(x, .05), relative_p95=np.quantile(x, .95))
            if n >= 4 and adjacent.any():
                dx = np.diff(x)[adjacent]
                result.update(relative_rmssd=np.sqrt(np.mean(dx**2)), relative_mean_abs_derivative=np.mean(np.abs(dx)))
            if np.ptp(x) > 0 and n >= 16:
                result["relative_skew"] = stats.skew(x, bias=False)
            if np.ptp(x) > 0 and n >= 20:
                result["relative_excess_kurtosis"] = stats.kurtosis(x, fisher=True, bias=False)
            if adjacent.all() and n >= 60:
                hz, psd = signal.welch(x, fs=1, nperseg=min(256, n), detrend="linear")
                for lower, upper, label, minimum in ((.01, .05, "001_005", 300), (.05, .25, "005_025", 60)):
                    band = (hz >= lower) & (hz <= upper)
                    if n >= minimum and band.sum() >= 2:
                        result[f"relative_psd_{label}"] = np.trapezoid(psd[band], hz[band])
        if prefix == "phasic":
            result.update(phasic_rms=np.sqrt(np.mean(x**2)), phasic_abs_mean=np.mean(np.abs(x)),
                          phasic_p95=np.quantile(x, .95), phasic_positive_area_per_s=np.mean(np.maximum(x, 0)))
    locations, prominence, calibrated = peaks
    if n >= 8 and calibrated:
        selected = (locations >= start) & (locations < stop)
        result.update(relative_peak_rate_per_min=selected.sum()/n*60,
                      relative_peak_mean_prominence=float(prominence[selected].mean()) if selected.any() else 0.,
                      relative_peak_max_prominence=float(prominence[selected].max()) if selected.any() else 0.)
    bm = (ts >= baseline-5000) & (ts < baseline) & np.isfinite(arrays["tonic"])
    if bm.sum() >= 4:
        result.update(relative_mean_change=result["relative_mean"]-arrays["relative"][bm].mean(),
                      tonic_mean_change=result["tonic_mean"]-arrays["tonic"][bm].mean(),
                      phasic_rms_change=result["phasic_rms"]-np.sqrt(np.mean(arrays["phasic"][bm]**2)))
    return result, n, coverage, status


def extract_record(base, recording_id, identities, events, target):
    data = pd.read_csv(base / "preprocessed" / f"{recording_id}.csv.gz")
    ts = data.timestamp_ms.to_numpy(dtype=np.int64)
    arrays = {"relative": data.relative_log_signal.to_numpy(float),
              "tonic": data.tonic.to_numpy(float), "phasic": data.phasic.to_numpy(float)}
    peaks = relative_peaks(ts, arrays["phasic"])
    identity = identities.iloc[0]
    starts, ends = archived_markers({"archive": Path(identity.archive), "member": identity.member})
    rows = []
    for person in identities.itertuples(index=False):
        selected = events[(events.task == person.task) & (events.group_id == person.group_id)]
        for e in selected.itertuples(index=False):
            for window in WINDOWS:
                start = e.timestamp_ms if window == "turn_interval" else e.timestamp_ms + 1000
                end = e.next_ms if window == "turn_interval" else min(e.timestamp_ms + 6000, e.next_ms) if np.isfinite(e.next_ms) else e.timestamp_ms + 6000
                values, n, coverage, status = window_features(ts, arrays, start, end, e.timestamp_ms, peaks)
                if len(starts) != 1 or len(ends) != 1 or not (starts[0] <= start and np.isfinite(end) and end <= ends[0]):
                    status = "outside_recording_or_invalid_markers"
                    values = dict.fromkeys(FEATURES, np.nan)
                rows.append(dict(recording_id=recording_id, subject_id=person.subject_id, task=e.task,
                                 group_id=e.group_id, group_type=e.group_type, participant_label=person.participant_label,
                                 event_id=e.event_id, role="self" if e.participant_label == person.participant_label else "other",
                                 actor=e.participant_label, window=window, ioc=e.ioc,
                                 speaker_changed=e.speaker_changed, sequence=e.idea_sequence, expected_s=(end-start)/1000,
                                 eligible_s=n, coverage=coverage, status=status, **values))
    table = pd.DataFrame(rows)
    temporary = target.with_suffix(".tmp")
    table.to_csv(temporary, index=False, compression="gzip", encoding="utf-8-sig")
    temporary.replace(target)
    return len(table)


def corrected(table, family_columns):
    table = pd.DataFrame(table)
    table["p_holm"] = np.nan
    for _, group in table.groupby(family_columns, dropna=False, sort=False):
        values = multipletests(group.p.fillna(1), method="holm")[1]
        table.loc[group.index, "p_holm"] = np.where(group.p.notna(), values, np.nan)
        table.loc[group.index, "family_size"] = len(group)
        if "p_bias_reduced" in table:
            values = multipletests(group.p_bias_reduced.fillna(1), method="holm")[1]
            table.loc[group.index, "p_holm_bias_reduced"] = np.where(group.p_bias_reduced.notna(), values, np.nan)
    return table


def base_result(**values):
    return dict(status="unavailable", reason="insufficient_sample_or_variation", p=np.nan, **values)


def fit_gee(data, outcome, feature, binary=False, event=False):
    result = base_result(n=0, n_teams=0, outcome=outcome, feature=feature)
    required = [outcome, feature, "group_id", "group_type"]
    if binary and not event:
        required += ["ioc_trials"]
    data = data.dropna(subset=required).copy()
    result.update(n=len(data), n_teams=data.group_id.nunique())
    if len(data) < 30 or data.group_id.nunique() < 15 or data[feature].std() <= 1e-12 or data[outcome].nunique() < 2:
        return result
    data["x"] = (data[feature]-data[feature].mean())/data[feature].std(ddof=1)
    X = pd.DataFrame({"intercept": 1., "eda": data.x, "human": data.group_type.eq("human").astype(float)}, index=data.index)
    if event:
        X["pit"] = data.task.eq("PIT").astype(float)
        X["sequence"] = (data.sequence-data.sequence.mean())/data.sequence.std(ddof=1)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        result["reason"] = "rank_deficient"
        return result
    family = sm.families.Binomial() if binary else sm.families.Poisson() if outcome in ("fluency_ind", "flexibility_ind") else sm.families.Gaussian()
    weights = data.ioc_trials.to_numpy(dtype=float) if binary and not event else None
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            fit = sm.GEE(data[outcome], X, groups=data.group_id, family=family,
                         cov_struct=sm.cov_struct.Independence(), weights=weights).fit(maxiter=100)
        ci = fit.conf_int().loc["eda"]
        if fit.converged and np.isfinite([fit.params.eda, fit.bse.eda, fit.pvalues.eda, *ci]).all():
            ratios = binary or isinstance(family, sm.families.Poisson)
            result.update(status="estimated", reason="", estimate=fit.params.eda, se=fit.bse.eda,
                          ci_low=ci.iloc[0], ci_high=ci.iloc[1], statistic=fit.tvalues.eda, p=fit.pvalues.eda,
                          model=family.__class__.__name__, predictor_mean=data[feature].mean(),
                          predictor_sd=data[feature].std(ddof=1),
                          ratio=np.exp(fit.params.eda) if ratios and abs(fit.params.eda) < 700 else np.nan,
                          ratio_ci_low=np.exp(ci.iloc[0]) if ratios and abs(ci.iloc[0]) < 700 else np.nan,
                          ratio_ci_high=np.exp(ci.iloc[1]) if ratios and abs(ci.iloc[1]) < 700 else np.nan,
                          warnings=" | ".join(sorted({str(w.message) for w in recorded})))
            fisher_weights = np.asarray(family.weights(fit.fittedvalues))
            if weights is not None:
                fisher_weights = fisher_weights*weights
            design = X.to_numpy(float)
            bread = np.linalg.pinv(design.T@(design*fisher_weights[:,None]))
            leverage = fisher_weights*np.einsum("ij,jk,ik->i",design,bread,design)
            result.update(max_design_leverage=float(leverage.max()),
                          max_standardized_predictor=float(np.abs(data.x).max()),
                          max_cluster_leverage=float(pd.Series(leverage).groupby(data.group_id.to_numpy()).sum().max()))
            try:
                with warnings.catch_warnings(record=True):
                    sensitivity = fit.model.fit(start_params=fit.params, maxiter=10, cov_type="bias_reduced")
                bci = sensitivity.conf_int().loc["eda"]
                if sensitivity.converged and np.isfinite([sensitivity.bse.eda,sensitivity.pvalues.eda,*bci]).all():
                    result.update(se_bias_reduced=sensitivity.bse.eda,p_bias_reduced=sensitivity.pvalues.eda,
                                  ci_low_bias_reduced=bci.iloc[0],ci_high_bias_reduced=bci.iloc[1])
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                pass
        else:
            result["reason"] = "not_converged_or_nonfinite"
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as error:
        result["reason"] = str(error)
    return result


def statistics(participants, event_files, out, progress):
    tables = {}
    keys = ["task", "window", "role", "group_id", "group_type"]
    team = participants.groupby(keys, as_index=False)[FEATURES + OUTCOMES].mean()
    team.to_csv(out / "team_role_features.csv", index=False, encoding="utf-8-sig")
    results, correlation, regression, omnibus = [], [], [], []
    combinations = [(t, w, r) for t in TASK_INPUTS for w in WINDOWS for r in ("self", "other")]
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    for done, (task, window, role) in enumerate(combinations, 1):
        checkpoint = parts / f"{task}-{window}-{role}.json"
        if checkpoint.is_file():
            part = json.loads(checkpoint.read_text(encoding="utf-8"))
            results.extend(part["comparisons"])
            correlation.extend(part["correlations"])
            regression.extend(part["regressions"])
            progress(done, len(combinations)+len(WINDOWS), f"恢复统计分段 {task} / {window} / {role}")
            continue
        offsets = len(results), len(correlation), len(regression)
        subset = team[(team.task == task) & (team.window == window) & (team.role == role)]
        sample = participants[(participants.task == task) & (participants.window == window) & (participants.role == role)]
        for feature in FEATURES:
            a = subset.loc[subset.group_type.eq("ai"), feature].dropna().to_numpy()
            b = subset.loc[subset.group_type.eq("human"), feature].dropna().to_numpy()
            for test in ("welch", "levene_median"):
                row = base_result(method=test, task=task, window=window, role=role, feature=feature, n_ai=len(a), n_human=len(b))
                if min(len(a), len(b)) >= 5 and np.var(np.r_[a, b]) > 1e-24:
                    if test == "welch":
                        fit = stats.ttest_ind(a, b, equal_var=False)
                        se = np.sqrt(np.var(a, ddof=1)/len(a)+np.var(b, ddof=1)/len(b))
                        delta = np.mean(a)-np.mean(b)
                        row.update(estimate=delta, df=float(fit.df), ci_low=delta-stats.t.ppf(.975, fit.df)*se,
                                   ci_high=delta+stats.t.ppf(.975, fit.df)*se)
                        pooled = np.sqrt(((len(a)-1)*np.var(a,ddof=1)+(len(b)-1)*np.var(b,ddof=1))/(len(a)+len(b)-2))
                        row["hedges_g"] = (1-3/(4*(len(a)+len(b))-9))*delta/pooled if pooled > 0 else np.nan
                    else:
                        fit = stats.levene(a, b, center="median")
                        row.update(df1=1, df2=len(a)+len(b)-2, estimate=np.var(a, ddof=1)/np.var(b, ddof=1) if np.var(b) > 0 else np.nan)
                    if np.isfinite([fit.statistic, fit.pvalue]).all():
                        row.update(status="estimated", reason="", statistic=fit.statistic, p=fit.pvalue)
                results.append(row)
            for outcome in OUTCOMES:
                d = subset[[feature, outcome]].dropna()
                for method in ("pearson", "spearman"):
                    row = base_result(method=method, task=task, window=window, role=role,
                                      feature=feature, outcome=outcome, n=len(d))
                    if len(d) >= 10 and d[feature].std() > 1e-12 and d[outcome].std() > 1e-12:
                        fit = stats.pearsonr(d[feature], d[outcome]) if method == "pearson" else stats.spearmanr(d[feature], d[outcome])
                        row.update(status="estimated", reason="", estimate=float(fit.statistic), p=float(fit.pvalue))
                        if method == "pearson":
                            ci = fit.confidence_interval()
                            row.update(ci_low=ci.low, ci_high=ci.high)
                    correlation.append(row)
                fit = fit_gee(sample, outcome, feature, binary=outcome == "ioc_proportion")
                regression.append(dict(task=task, window=window, role=role, method="GEE", **fit))
        atomic_json(checkpoint, dict(comparisons=results[offsets[0]:], correlations=correlation[offsets[1]:],
                                    regressions=regression[offsets[2]:]))
        progress(done, len(combinations)+len(WINDOWS), f"组间、方差、相关和回归 {task} / {window} / {role}")
    for task in TASK_INPUTS:
        for window in WINDOWS:
            s = participants[(participants.task == task) & (participants.window == window)]
            for feature in FEATURES:
                paired = s.pivot(index=["subject_id", "group_id"], columns="role", values=feature).dropna()
                paired = paired.groupby("group_id")[["self", "other"]].mean()
                row = base_result(method="paired_team", task=task, window=window, role="self-minus-other", feature=feature, n=len(paired))
                if len(paired) >= 8 and (paired["self"]-paired["other"]).std() > 1e-12:
                    d = paired["self"]-paired["other"]
                    fit = stats.ttest_rel(paired["self"], paired["other"])
                    ci = fit.confidence_interval()
                    row.update(status="estimated", reason="", estimate=d.mean(), statistic=fit.statistic,
                               df=float(fit.df), p=fit.pvalue, ci_low=ci.low, ci_high=ci.high,
                               effect_dz=d.mean()/d.std(ddof=1))
                results.append(row)
    for window in WINDOWS:
        s = participants[participants.window.eq(window)]
        for feature in FEATURES:
            d = s.dropna(subset=[feature]).copy()
            terms = ("role", "group_type", "role_x_group_type", "task")
            rows = [base_result(method="GEE_factorial_Wald", window=window, feature=feature, term=term,
                                n=len(d), n_teams=d.group_id.nunique()) for term in terms]
            if len(d) >= 40 and d.group_id.nunique() >= 15 and d[feature].std() > 1e-12:
                X = pd.DataFrame({"intercept": 1., "role": d.role.eq("self").astype(float),
                                  "group_type": d.group_type.eq("human").astype(float),
                                  "task": d.task.eq("PIT").astype(float)}, index=d.index)
                X["role_x_group_type"] = X.role*X.group_type
                if np.linalg.matrix_rank(X) == X.shape[1]:
                    try:
                        with warnings.catch_warnings(record=True):
                            fit = sm.GEE(d[feature], X, groups=d.group_id, family=sm.families.Gaussian()).fit()
                        for row in rows:
                            term = row["term"]
                            ci = fit.conf_int().loc[term]
                            if fit.converged and np.isfinite([fit.bse[term], fit.pvalues[term], *ci]).all():
                                row.update(status="estimated", reason="", estimate=fit.params[term], se=fit.bse[term],
                                           statistic=fit.tvalues[term]**2, df=1, p=fit.pvalues[term], ci_low=ci.iloc[0], ci_high=ci.iloc[1])
                    except (ValueError, np.linalg.LinAlgError):
                        pass
            omnibus.extend(rows)
    event_tests = []
    for index, window in enumerate(WINDOWS, 1):
        frames = []
        for path in event_files:
            t = pd.read_csv(path, dtype={"subject_id": str})
            frames.append(t[t.window.eq(window) & t.role.eq("self") & t.status.eq("valid")])
        self_events = pd.concat(frames, ignore_index=True)
        for feature in FEATURES:
            event_tests.append(dict(method="event_IOC_logistic_GEE", window=window,
                                    **fit_gee(self_events, "ioc", feature, binary=True, event=True)))
        progress(len(combinations)+index, len(combinations)+len(WINDOWS), f"事件级 IOC 二项回归 / {window}")
    tables["comparisons"] = corrected(results, ["method"])
    tables["correlations"] = corrected(correlation, ["method", "outcome"])
    tables["participant_regressions"] = corrected(regression, ["method", "outcome"])
    tables["factorial_tests"] = corrected(omnibus, ["method"])
    tables["event_ioc_regressions"] = corrected(event_tests, ["method"])
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    return tables


def run_turn_analysis(base, progress_callback=None, cancel_requested=None):
    from pyeda_turn_outputs import write_outputs

    base = Path(base).resolve()
    output_root = base.parent
    inputs = [base / "manifest.json", DATA_ROOT / "实验情况记录表.xlsx",
              *(DATA_ROOT / "_rawdata/_rawdata" / names[0] for names in TASK_INPUTS.values())]
    fingerprints = {str(path): sha256_file(path) for path in inputs}
    extraction = "\n".join(inspect.getsource(f) for f in
                           (feature_definitions, load_events_and_people, identity_mapping, relative_peaks, window_features, extract_record))
    extraction = hashlib.sha256(extraction.encode()).hexdigest()
    identity = hashlib.sha256(json.dumps(dict(inputs=fingerprints, extraction=extraction, rules=RULES),
                                      sort_keys=True).encode()).hexdigest()
    legacy_identity = hashlib.sha256(json.dumps(dict(inputs=fingerprints,
        script="8654462457f48fbc6167903ab2cf59697ec972f6ac9cc416bf81c4dab422f37c", rules=RULES),
        sort_keys=True).encode()).hexdigest()
    analysis_signature = hashlib.sha256((inspect.getsource(statistics)+inspect.getsource(fit_gee)).encode()).hexdigest()
    graphics_signature = sha256_file(Path(__file__).with_name("pyeda_turn_outputs.py"))
    with study_lock(output_root):
        candidates = sorted(output_root.glob("AUT-PIT-RESPONSES-*/checkpoint.json"),
                            key=lambda path: path.stat().st_mtime_ns, reverse=True)
        root = None
        for path in candidates:
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if old.get("identity") in (identity, legacy_identity):
                root = path.parent
                break
        if root is None:
            root = output_root / f"AUT-PIT-RESPONSES-{dt.datetime.now(LOCAL_TIMEZONE):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
            root.mkdir()
        state = {"identity": identity, "base_run": str(base), "status": "running",
                 "analysis_signature": analysis_signature}
        def report(stage, done, total, message):
            append_log(root / "run.log", message)
            _emit_progress(progress_callback, stage=stage, done=done, total=total, message=message,
                           details={"run_directory": str(root)})
        def save(stage):
            state["stage"] = stage
            atomic_json(root / "checkpoint.json", state)
        def pause():
            if cancel_requested is not None and cancel_requested():
                raise StudyPaused
        try:
            report("extended_checks", 0, 1, "核对记录、发言身份与 IOC 定义…")
            if (root / "manifest.json").is_file():
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("analysis_signature") == analysis_signature and artifacts_match(root, {e["path"]: e["sha256"] for e in manifest["artifacts"]}):
                    if manifest.get("graphics_signature") != graphics_signature:
                        report("extended_export", 0, 1, "复用已完成统计，导出扩展图表…")
                        write_outputs(root)
                        files = {e["path"]:e["sha256"] for e in manifest["artifacts"]}
                        files.update({p.relative_to(root).as_posix():sha256_file(p) for p in (root/"figures").iterdir() if p.is_file()})
                        atomic_json(root/"manifest.json",dict(analysis_signature=analysis_signature,
                                    graphics_signature=graphics_signature,
                                    artifacts=[dict(path=path,sha256=digest) for path,digest in sorted(files.items())]))
                    report("extended_export", 1, 1, "个人回答情境分析已完成，复用原结果。")
                    return json.loads((root / "summary.json").read_text(encoding="utf-8"))
            for folder in ("intake", "features/records", "statistics", "figures"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            stage_signature = root / "statistics/signature.json"
            compatible_statistics = (stage_signature.is_file()
                and json.loads(stage_signature.read_text(encoding="utf-8")).get("analysis_signature") == analysis_signature)
            if not compatible_statistics:
                for path in (root / "statistics/parts").glob("*.json"):
                    path.unlink()
            atomic_json(stage_signature, {"analysis_signature": analysis_signature})
            settings = json.loads((base / "intake/analysis_settings.json").read_text(encoding="utf-8"))
            for path, digest in settings["inputs_sha256"].items():
                if Path(path).is_absolute() and sha256_file(Path(path)) != digest:
                    raise ValueError("原始工作簿与基础预处理批次不一致，请先更新基础分析。")
            sources = json.loads((base / "manifest.json").read_text(encoding="utf-8"))["artifacts"]
            hashes = {e["path"]: e["sha256"] for e in sources}
            events, people = load_events_and_people(DATA_ROOT / "_rawdata/_rawdata")
            mapping = identity_mapping(base, events, DATA_ROOT / "实验情况记录表.xlsx")
            usable = mapping[mapping.identity_status.eq("candidate_surname_and_sheet_agree")]
            exports = events.drop(columns=["identity", "ideatype_rater2"]).copy()
            exports.to_csv(root / "intake/response_events.csv", index=False, encoding="utf-8-sig")
            mapping.to_csv(root / "intake/person_record_mapping.csv", index=False, encoding="utf-8-sig")
            DEFINITIONS.to_csv(root / "intake/feature_definitions.csv", index=False, encoding="utf-8-sig")
            people.to_csv(root / "intake/individual_outcomes.csv", index=False, encoding="utf-8-sig")
            atomic_json(root / "intake/analysis_settings.json", dict(base_run=str(base), rules=RULES, inputs=fingerprints,
                                                                  features=FEATURES, outcomes=OUTCOMES, qualification="exploratory"))
            save("features")
            report("extended_checks", 1, 1, "身份核对完成，开始逐记录计算回答情境特征…")
            records = []
            if usable.empty:
                raise ValueError("没有完成个人身份核对的手环记录。")
            for done, (rid, identity_rows) in enumerate(usable.groupby("recording_id", sort=True), 1):
                pause()
                target = root / "features/records" / f"{rid}.csv.gz"
                checkpoint = target.with_suffix(".json")
                source_hash = hashes[f"preprocessed/{rid}.csv.gz"]
                if sha256_file(base / "preprocessed" / f"{rid}.csv.gz") != source_hash:
                    raise ValueError(f"基础信号完整性校验失败：{rid}")
                valid = False
                if checkpoint.is_file():
                    meta = json.loads(checkpoint.read_text(encoding="utf-8"))
                    valid = artifacts_match(target.parent, {target.name: meta["sha256"]})
                if not valid:
                    count = extract_record(base, rid, identity_rows, events, target)
                    atomic_json(checkpoint, dict(sha256=sha256_file(target), rows=count))
                records.append(target)
                report("extended_features", done, usable.recording_id.nunique(), f"已保存回答情境记录 {done}/{usable.recording_id.nunique()}")
            pause()
            save("statistics")
            frames = [pd.read_csv(path, dtype={"subject_id": str}) for path in records]
            trials = pd.concat(frames, ignore_index=True)
            group_keys = ["subject_id", "task", "group_id", "group_type", "participant_label", "window", "role"]
            participants = trials[trials.status.eq("valid")].groupby(group_keys)[FEATURES].mean().reset_index()
            counts = trials[trials.status.eq("valid")].groupby(group_keys).size().rename("valid_events")
            participants = participants.join(counts, on=group_keys)
            exposures = trials[trials.status.eq("valid")].groupby(group_keys)[FEATURES].count()
            exposures.to_csv(root / "features/participant_feature_exposure.csv", encoding="utf-8-sig")
            feature_counts = participants[group_keys].merge(exposures.reset_index(), on=group_keys,validate="1:1")[FEATURES]
            participants[FEATURES] = participants[FEATURES].where(feature_counts.ge(3))
            participants = participants.merge(people, how="left", on=["task", "group_id", "participant_label"], validate="m:1")
            participants.to_csv(root / "features/participant_task_role_features.csv", index=False, encoding="utf-8-sig")
            availability = pd.DataFrame([dict(feature=f, events_available=int(trials[f].notna().sum()),
                                               subjects_available=int(participants.loc[participants[f].notna(), "subject_id"].nunique())) for f in FEATURES])
            availability.to_csv(root / "features/feature_availability.csv", index=False, encoding="utf-8-sig")
            trials.groupby(["task", "window", "role", "status"]).size().rename("windows").reset_index().to_csv(
                root / "features/window_qc.csv", index=False, encoding="utf-8-sig")
            tables = statistics(participants, records, root / "statistics",
                                lambda done, total, text: report("extended_statistics", done, total, text))
            pause()
            save("export")
            write_outputs(root)
            summary = dict(status="succeeded", qualification="exploratory_response_posting_proxy",
                           run_directory=str(root), base_run=str(base), features=len(FEATURES), processed_records=len(records),
                           subjects=int(participants.subject_id.nunique()), feature_rows=len(participants),
                           response_feature_windows=len(trials), valid_feature_windows=int(trials.status.eq("valid").sum()),
                           analysis_tables={name:dict(planned=len(t),estimated=int(t.status.eq("estimated").sum()),
                                                     holm_below_05=int(t.p_holm.lt(.05).sum())) for name,t in tables.items()})
            atomic_json(root / "summary.json", summary)
            (root / "分析说明.md").write_text(
                "# 个人回答情境 EDA 扩展分析\n\n"
                "participant_task_role_features.csv 位于 features，逐记录逐回答结果位于 features/records。"
                "self 和 other 以目标被试为参照。时间为发送后的代理窗。IOC 按任务及团队内相邻回答的 ideatype_rater2 一致性计算，"
                "首条和缺失类别保留为空。连续本人回答亦按用户定义比较前一条，speaker_changed 可用于审计。\n\n"
                "Welch、方差检验和相关以团队平均值为独立单位。本人/对方配对在两种角色都有有效值的被试中先求团队均值。"
                "多因素 Wald 和回归用团队聚类的 GEE。IOC 比例使用有效回答数加权的二项 GEE，事件 IOC 另拟合二项 GEE。"
                "所有检验为探索性。每个统计方法和结局内对全部预定检验作 Holm 校正，缺失检验保留家族位置。"
                "缺少样本、无变异和不可估计的项目在结果表中记录原因。"
                "相对峰是无量纲信号的局部峰代理，谱指标需足够长的连续窗口。\n",
                encoding="utf-8")
            atomic_json(root / "manifest.json", {"analysis_signature": analysis_signature, "graphics_signature":graphics_signature,
                "artifacts":[dict(path=p.relative_to(root).as_posix(),sha256=sha256_file(p))
                          for p in sorted(root.rglob("*")) if p.is_file() and p.name not in ("manifest.json","checkpoint.json","run.log","error.log")]})
            state["status"] = "complete"
            save("complete")
            (output_root / "LATEST_RESPONSE_RUN.txt").write_text(str(root)+"\n",encoding="utf-8")
            report("extended_export", 1, 1, "个人、回答情境、IOC 和扩展统计结果已导出。")
            return summary
        except StudyPaused:
            state["status"] = "paused"
            save(state.get("stage", "checks"))
            return dict(status="paused",run_directory=str(root),records_saved=len(list((root/"features/records").glob("*.csv.gz"))))
        except Exception as error:
            state.update(status="failed",error=str(error))
            save(state.get("stage","checks"))
            append_log(root/"error.log",traceback.format_exc())
            raise RuntimeError(f"{error}\n输出目录：{root}") from error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_turn_analysis(args.base_run), ensure_ascii=False, indent=2))
