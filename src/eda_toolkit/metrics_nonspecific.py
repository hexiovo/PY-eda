"""基于显式完整 marker 覆盖的 non-event-associated SCR 分类。"""

from __future__ import annotations

import numpy as np
import pandas as pd


NS_COLUMNS = ["recording_id", "channel", "scr_id", "onset_s", "amplitude_uS", "class",
              "status", "reason", "exposure_policy", "annotation_coverage_ids"]


def _merge(intervals):
    output = []
    for lo, hi in sorted((float(a), float(b)) for a, b in intervals if b > a):
        # Only actual overlap or exact adjacency is merged. A positive-width
        # gap, however small, remains an unsupported interval.
        if output and lo <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], hi))
        else:
            output.append((lo, hi))
    return output


def _intersect(a, b):
    out = []
    for alo, ahi in a:
        for blo, bhi in b:
            lo, hi = max(alo, blo), min(ahi, bhi)
            if hi > lo:
                out.append((lo, hi))
    return _merge(out)


def _subtract(base, removed):
    result = []
    removed = _merge(removed)
    for lo, hi in _merge(base):
        parts = [(lo, hi)]
        for rlo, rhi in removed:
            next_parts = []
            for a, b in parts:
                if rhi <= a or rlo >= b:
                    next_parts.append((a, b))
                else:
                    if rlo > a:
                        next_parts.append((a, rlo))
                    if rhi < b:
                        next_parts.append((rhi, b))
            parts = next_parts
        result.extend(parts)
    return _merge(result)


def _inside(value, intervals):
    return any(lo <= value < hi for lo, hi in intervals)


def build_nonspecific_support(bundle, windows, onset_intervals, config):
    """根据完整 event-marker coverage 推导可称为 non-event-associated 的 onset 支持。"""
    cfg = config.get("features", {}).get("extended", {}).get("nonspecific", {})
    if not cfg.get("enabled", True):
        return [], [], "nonspecific_disabled"
    rows = cfg.get("annotation_coverage", [])
    metadata = bundle.get("metadata", {})
    rid = str(metadata.get("recording_id", bundle.get("recording_id", "")))
    channel = str(metadata.get("channel", bundle.get("channel", "eda")))
    domain = str(metadata.get("clock_domain", "recording_local"))
    matched = [r for r in rows if r.get("complete") is True and str(r.get("recording_id")) == rid
               and str(r.get("channel")) == channel and str(r.get("clock_domain")) == domain]
    if not matched:
        return [], [], "NA_annotation_incomplete"
    response_lo, response_hi = map(float, config.get("windows", {}).get("event_response_s", [1.0, 4.0]))
    # Union complete annotation coverage before mapping it to marker-onset
    # support. Transforming each source row first would create an artificial
    # response-width gap at every merely administrative partition boundary.
    coverage_union = _merge((float(r["start_s"]), float(r["end_s"])) for r in matched)
    coverage_onset = _merge((start + response_hi, end + response_lo)
                            for start, end in coverage_union)
    if not coverage_onset:
        return [], matched, "NA_annotation_incomplete"
    supported = _intersect(onset_intervals, coverage_onset)
    scheduled = []
    for row in windows.to_dict("records") if isinstance(windows, pd.DataFrame) else windows:
        if row.get("kind") != "event_response" or row.get("event_onset_s") is None or pd.isna(row.get("event_onset_s")):
            continue
        marker_onset = float(row["event_onset_s"])
        scheduled.append((marker_onset + response_lo, marker_onset + response_hi))
    return _subtract(supported, scheduled), matched, ""


def classify_nonspecific_scr(bundle, scr, windows, onset_intervals, event_links, config):
    """返回逐SCR分类表及可用于窗口特征的NS onset exposure区间。"""
    intervals, coverage_rows, policy_reason = build_nonspecific_support(bundle, windows, onset_intervals, config)
    metadata = bundle.get("metadata", {})
    rid = str(metadata.get("recording_id", bundle.get("recording_id", "")))
    channel = str(metadata.get("channel", bundle.get("channel", "eda")))
    scr_rows = scr.to_dict("records") if isinstance(scr, pd.DataFrame) else []
    ambiguous = set()
    if isinstance(event_links, pd.DataFrame) and not event_links.empty:
        ambiguous = set(event_links.loc[event_links["link_status"].eq("ambiguous"), "scr_id"].dropna().astype(str))
    rows = []
    for item in scr_rows:
        scr_id = str(item.get("scr_id", ""))
        onset = float(item.get("onset_s", np.nan))
        amplitude = float(item.get("amplitude_uS", np.nan))
        eligible = bool(item.get("onset_eligible", True))
        if not eligible:
            cls, status, reason = "unavailable", "unavailable", str(item.get("qualification_reason") or "unavailable_onset_context")
        elif scr_id in ambiguous:
            cls, status, reason = "ambiguous", "ambiguous", "ambiguous_event_scr_assignment"
        elif policy_reason:
            cls, status, reason = "unavailable", "unavailable", policy_reason
        elif _inside(onset, intervals):
            cls, status, reason = "non_event_associated", "valid", ""
        else:
            cls, status, reason = "event_associated_or_outside", "valid", "outside_non_event_onset_support"
        rows.append({"recording_id": rid, "channel": channel, "scr_id": scr_id, "onset_s": onset,
                     "amplitude_uS": amplitude, "class": cls, "status": status, "reason": reason,
                     "exposure_policy": "complete_marker_coverage; onset_support minus scheduled response windows",
                     "annotation_coverage_ids": [f"coverage:{i}" for i, _ in enumerate(coverage_rows)]})
    return pd.DataFrame(rows, columns=NS_COLUMNS), intervals, policy_reason
