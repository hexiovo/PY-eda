"""Driver feature definitions and linear-time-support measurements."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core_preprocess import _segments
from .driver_decompose import driver_settings


DRIVER_SPECS = [
    ("driver_integral_uS", "Driver 积分", "uS", "integral of piecewise-linear driver over eligible adjacent support"),
    ("driver_time_mean_uS_s", "Driver 时间均值", "uS/s", "driver integral / eligible adjacent duration"),
    ("driver_max_uS_s", "Driver 最大值", "uS/s", "maximum eligible driver sample in half-open window"),
    ("driver_active_time_proportion", "Driver 活跃时间比例", "fraction", "duration at or above threshold / eligible adjacent duration"),
    ("driver_burst_count", "Driver 完整脉冲计数", "count", "complete above-threshold episodes with peak in half-open window"),
    ("driver_burst_rate_per_min", "Driver 完整脉冲频率", "bursts/min", "60*complete burst count/eligible adjacent duration"),
    ("event_driver_integral_uS", "事件 Driver 积分", "uS", "driver integral within event response window; no baseline subtraction"),
    ("event_driver_max_uS_s", "事件 Driver 最大值", "uS/s", "maximum eligible driver sample within event response window"),
    ("event_driver_peak_latency_s", "事件 Driver 峰潜伏期", "s", "time of earliest maximum driver sample above threshold minus event onset"),
]
DRIVER_IDS = frozenset(item[0] for item in DRIVER_SPECS)
BURST_COLUMNS = ["driver_burst_id", "start_s", "end_s", "peak_s", "peak_driver_uS_s", "integral_uS",
                 "threshold_uS_s", "segment_id", "status", "reason", "left_censored", "right_censored"]


def linear_window(time, values, eligible, start, end, max_gap, threshold):
    """Clip adjacent line segments exactly; do not bridge a missing point or extrapolate."""
    area = duration = active = 0.
    lo = max(0, int(np.searchsorted(time, start)) - 1)
    hi = min(len(time)-1, int(np.searchsorted(time, end, side="right")))
    for i in range(lo, hi):
        dt = time[i+1] - time[i]
        if not (eligible[i] and eligible[i+1] and dt > 0 and dt <= max_gap):
            continue
        a, b = max(start, time[i]), min(end, time[i+1])
        if b <= a:
            continue
        ya = values[i] + (values[i+1] - values[i]) * (a-time[i]) / dt
        yb = values[i] + (values[i+1] - values[i]) * (b-time[i]) / dt
        span = b-a
        area += .5 * (ya+yb) * span
        duration += span
        if ya >= threshold and yb >= threshold:
            active += span
        elif ya >= threshold or yb >= threshold:
            active += span * (max(ya, yb)-threshold) / abs(yb-ya)
    return {"integral": float(area), "duration": float(duration), "active": float(active)}


def driver_bursts(bundle, config):
    if bundle.get("method") != "bateman_sparse":
        return pd.DataFrame(columns=BURST_COLUMNS)
    time = np.asarray(bundle["time_s"], float)
    values = np.asarray(bundle["driver_uS_s"], float)
    mask = np.asarray(bundle["quality"]["driver_eligible"], bool) & np.isfinite(values)
    threshold = float(driver_settings(config)["burst_threshold_uS_s"])
    max_gap = bundle["max_gap_s"]
    rows = []
    for segment_id, (lo, hi) in enumerate(_segments(mask, time, max_gap)):
        above = values[lo:hi] >= threshold
        starts = np.flatnonzero(np.diff(np.r_[False, above].astype(int)) == 1) + lo
        stops = np.flatnonzero(np.diff(np.r_[above, False].astype(int)) == -1) + lo + 1
        for start, stop in zip(starts, stops):
            left, right = start == lo, stop == hi
            onset = time[start] if left else time[start-1] + (time[start]-time[start-1]) * (
                threshold-values[start-1]) / (values[start]-values[start-1])
            offset = time[stop-1] if right else time[stop-1] + (time[stop]-time[stop-1]) * (
                values[stop-1]-threshold) / (values[stop-1]-values[stop])
            peak = start + int(np.argmax(values[start:stop]))
            integral = linear_window(time, values, mask, onset, offset, max_gap, threshold)["integral"]
            rows.append({"driver_burst_id": f"driver-{len(rows)+1}", "start_s": float(onset), "end_s": float(offset),
                         "peak_s": float(time[peak]), "peak_driver_uS_s": float(values[peak]), "integral_uS": integral,
                         "threshold_uS_s": threshold, "segment_id": segment_id,
                         "status": "censored" if left or right else "valid",
                         "reason": "support_boundary_censored" if left or right else "",
                         "left_censored": left, "right_censored": right})
    return pd.DataFrame(rows, columns=BURST_COLUMNS)


def driver_window_metrics(bundle, window, config, catalogue):
    ids = [row[0] for row in DRIVER_SPECS if not row[0].startswith("event_") or window["kind"] == "event_response"]
    def unavailable(reason):
        return {fid: {"value": None, "status": "unavailable", "reason": reason, "n": 0, "duration": 0.} for fid in ids}
    if bundle.get("method") != "bateman_sparse" or "driver_uS_s" not in bundle:
        return unavailable("requires_bateman_sparse")
    time, values = np.asarray(bundle["time_s"], float), np.asarray(bundle["driver_uS_s"], float)
    mask = np.asarray(bundle["quality"]["driver_eligible"], bool) & np.isfinite(values) & (values >= 0)
    start, end = float(window["start_s"]), float(window["end_s"])
    threshold = float(driver_settings(config)["burst_threshold_uS_s"])
    support = linear_window(time, values, mask, start, end, bundle["max_gap_s"], threshold)
    sample_indices = np.flatnonzero(mask & (time >= start) & (time < end))
    expected = end-start
    # True clock gaps invalidate a window; excluded points reduce its coverage.
    gaps = (np.diff(time) > bundle["max_gap_s"]) & (time[:-1] < end) & (time[1:] > start)
    if (expected <= 0 or not len(sample_indices) or support["duration"] <= 0 or np.any(gaps) or
            start < time[0] or end > np.nextafter(time[-1], np.inf) or
            support["duration"] / expected + 1e-12 < config["windows"]["min_coverage"] or
            window.get("window_status") in ("invalid", "unavailable") or
            (pd.notna(window.get("condition_crossing")) and bool(window.get("condition_crossing")))):
        return unavailable("insufficient_driver_support_or_invalid_window")
    peak_i = sample_indices[int(np.argmax(values[sample_indices]))]
    complete = catalogue.loc[catalogue["status"].eq("valid") & catalogue["peak_s"].ge(start) & catalogue["peak_s"].lt(end)]
    numbers = {"driver_integral_uS": support["integral"], "driver_time_mean_uS_s": support["integral"] / support["duration"],
               "driver_max_uS_s": float(values[peak_i]), "driver_active_time_proportion": support["active"] / support["duration"],
               "driver_burst_count": len(complete), "driver_burst_rate_per_min": 60 * len(complete) / support["duration"],
               "event_driver_integral_uS": support["integral"], "event_driver_max_uS_s": float(values[peak_i])}
    results = {fid: {"value": numbers.get(fid), "status": "valid", "reason": "", "n": len(sample_indices),
                     "duration": support["duration"]} for fid in ids}
    if "event_driver_peak_latency_s" in results:
        row = results["event_driver_peak_latency_s"]
        onset = window.get("event_onset_s")
        if onset is None or not np.isfinite(onset) or values[peak_i] < threshold:
            row.update(status="unavailable", reason="no_above_threshold_driver_peak_or_event_onset")
        else:
            row["value"] = float(time[peak_i]-onset)
    return results
