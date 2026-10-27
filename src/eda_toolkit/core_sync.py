"""Clock alignment and group synchrony calculations.

All time values are seconds. The module never infers clock alignment from EDA
signals. Results from configured surrogate null methods are experimental
surrogate probabilities until their operating characteristics are calibrated.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from .contracts import FEATURE_COLUMNS, continuous_segments
from .core_preprocess import _resample_arrays
from .errors import fail


TIME_MAPPING_VERSION = "affine-clock-map/1"
_TABLE_TIME_COLUMNS = {
    "events": ("onset_s", "start_s", "end_s"),
    "conditions": ("start_s", "end_s"),
    "manual_qc": ("start_s", "end_s"),
}
_TABLE_DURATION_COLUMNS = {"events": ("duration_s",)}
_SIGNAL_FOR_BASIS = {"clean": "clean_uS", "tonic": "tonic_uS", "phasic": "phasic_uS"}


def _missing(value):
    if value is None:
        return True
    return bool(pd.isna(value)) if np.isscalar(value) else False


def _present(value):
    return not _missing(value) and value != ""


def _cancel(cancel_callback):
    if cancel_callback is not None and cancel_callback():
        raise InterruptedError("用户请求取消同步分析。")


def _as_rows(table):
    if table is None:
        return []
    if isinstance(table, pd.DataFrame):
        return table.to_dict(orient="records")
    if isinstance(table, dict):
        return [dict(table)]
    try:
        return [dict(row) for row in table]
    except (TypeError, ValueError):
        return []


def _copy_table(table):
    if isinstance(table, pd.DataFrame):
        result = table.copy(deep=True)
        result.attrs = copy.deepcopy(table.attrs)
        return result
    return copy.deepcopy(table)


def _copy_record(record):
    result = {}
    for key, value in record.items():
        if isinstance(value, np.ndarray):
            result[key] = np.array(value, copy=True)
        elif isinstance(value, pd.DataFrame):
            result[key] = _copy_table(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _recording_id(item):
    metadata = item.get("metadata", {})
    return str(metadata.get("recording_id", item.get("recording_id", "")))


def _channel(item):
    metadata = item.get("metadata", {})
    return str(metadata.get("channel", item.get("channel", "eda")))


def _record_key(item):
    return f"{_recording_id(item)}::{_channel(item)}"


def _clock_domain(item, config):
    metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
    domain = metadata.get("clock_domain")
    if domain:
        return str(domain)
    mapping = config.get("mapping", {}) if isinstance(config, dict) else {}
    return str(mapping.get("clock_domain", "recording_local"))


def _anchors_for(anchors, item):
    rows = _as_rows(anchors)
    rid, channel = _recording_id(item), _channel(item)
    selected = [row for row in rows if str(row.get("recording_id", "")) == rid]
    channel_rows = [row for row in selected if _present(row.get("channel"))]
    if channel_rows:
        selected = [row for row in selected if not _present(row.get("channel")) or str(row.get("channel")) == channel]
    return selected


def _canonical_hash(rows):
    normalized = []
    for row in rows:
        clean = {}
        for key, value in sorted(row.items()):
            if isinstance(value, np.generic):
                value = value.item()
            if value is None or (np.isscalar(value) and pd.isna(value)):
                clean[key] = None
            elif isinstance(value, (list, tuple, np.ndarray)):
                clean[key] = list(value)
            else:
                clean[key] = value
        normalized.append(clean)
    normalized.sort(key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False, default=str))
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fit_clock_map(rows):
    valid = []
    rejected = []
    for row in rows:
        try:
            local = float(row["local_time_s"])
            reference = float(row["reference_time_s"])
        except (KeyError, TypeError, ValueError):
            rejected.append({"anchor_id": row.get("anchor_id"), "reason": "missing_or_nonnumeric_time"})
            continue
        if np.isfinite(local) and np.isfinite(reference):
            valid.append((local, reference))
        else:
            rejected.append({"anchor_id": row.get("anchor_id"), "reason": "nonfinite_time"})
    reference_domains = sorted({str(row.get("reference_clock_domain")) for row in rows
                                if _present(row.get("reference_clock_domain"))})
    if len(reference_domains) > 1:
        return {"status": "unverified", "reason": "anchor_reference_domain_mismatch", "anchor_count": 0,
                "reference_clock_domains": reference_domains, "anchor_sha256": _canonical_hash(rows),
                "invalid_anchor_count": len(rejected), "invalid_anchor_rows": rejected}
    if not valid:
        return {"status": "unverified", "reason": "no_valid_anchors", "anchor_count": 0,
                "anchor_sha256": _canonical_hash(rows), "invalid_anchor_count": len(rejected),
                "invalid_anchor_rows": rejected}
    x = np.asarray([pair[0] for pair in valid], dtype=float)
    y = np.asarray([pair[1] for pair in valid], dtype=float)
    if len(np.unique(x)) != len(x):
        return {"status": "unverified", "reason": "duplicate_local_anchor_time", "anchor_count": len(valid),
                "anchor_sha256": _canonical_hash(rows), "invalid_anchor_count": len(rejected),
                "invalid_anchor_rows": rejected}
    if len(valid) == 1:
        slope = 1.0
        intercept = float(y[0] - x[0])
        residual = np.array([0.0])
        fit_status = "one_anchor_offset_only"
        uncertainty_status = "not_estimable_one_anchor"
    else:
        design = np.column_stack((np.ones(len(x)), x))
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 2:
            return {"status": "unverified", "reason": "anchor_times_do_not_span", "anchor_count": len(valid),
                    "anchor_sha256": _canonical_hash(rows), "invalid_anchor_count": len(rejected),
                    "invalid_anchor_rows": rejected}
        intercept, slope = map(float, coefficients)
        residual = y - (intercept + slope * x)
        fit_status = "two_anchors_no_empirical_uncertainty" if len(valid) == 2 else "ols_all_anchors"
        uncertainty_status = "not_estimable_two_anchors" if len(valid) == 2 else "residual_descriptive_not_device_qualified"
    if not np.isfinite(slope) or slope <= 0 or not np.isfinite(intercept):
        return {"status": "unverified", "reason": "nonpositive_or_nonfinite_slope", "anchor_count": len(valid),
                "anchor_sha256": _canonical_hash(rows), "a": intercept if np.isfinite(intercept) else None,
                "b": slope if np.isfinite(slope) else None, "invalid_anchor_count": len(rejected),
                "invalid_anchor_rows": rejected}
    ordered = np.sort(x)
    span = float(ordered[-1] - ordered[0]) if len(x) > 1 else 0.0
    resid_sd = float(np.std(residual, ddof=1)) if len(residual) > 1 else None
    return {"status": "aligned", "reason": "", "a": intercept, "b": slope,
            "anchor_count": len(valid), "anchor_span_s": span,
            "anchor_min_local_s": float(ordered[0]), "anchor_max_local_s": float(ordered[-1]),
            "residual_rms_s": float(np.sqrt(np.mean(residual ** 2))),
            "residual_sd_s": resid_sd,
            "max_abs_residual_s": float(np.max(np.abs(residual))),
            "fit_status": fit_status, "uncertainty_status": uncertainty_status,
            "outliers_removed": 0, "anchor_sha256": _canonical_hash(rows),
            "invalid_anchor_count": len(rejected), "invalid_anchor_rows": rejected,
            "reference_clock_domain": reference_domains[0] if reference_domains else "unspecified_reference"}


def _mapping_for(item, anchors, config):
    metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
    domain = _clock_domain(item, config)
    if domain.lower() == "reference":
        return {"status": "already_reference", "reason": "metadata_clock_domain_reference",
                "a": 0.0, "b": 1.0, "anchor_count": 0, "anchor_sha256": _canonical_hash([]),
                "original_clock_domain": domain, "fs_source": "metadata_reference"}
    rows = _anchors_for(anchors, item)
    if rows:
        fitted = _fit_clock_map(rows)
        fitted["original_clock_domain"] = domain
        fitted["fs_source"] = "local_rate_divided_by_affine_slope"
        return fitted
    sync = config.get("synchrony", {}) if isinstance(config, dict) else {}
    if sync.get("shared_clock_verified") is True:
        return {"status": "shared_clock_verified", "reason": "explicit_configuration_assertion",
                "a": 0.0, "b": 1.0, "anchor_count": 0, "anchor_sha256": _canonical_hash([]),
                "original_clock_domain": domain, "fs_source": "explicit_shared_clock_configuration"}
    return {"status": "unverified", "reason": "anchors_required_for_local_clock",
            "anchor_count": 0, "anchor_sha256": _canonical_hash([]), "original_clock_domain": domain}


def _local_rate(item, config):
    metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
    for key in ("fs_local_hz", "source_fs_hz", "sampling_rate_hz", "fs_hz"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0:
            return float(value), f"metadata.{key}"
    mapping = config.get("mapping", {}) if isinstance(config, dict) else {}
    value = mapping.get("sampling_rate_hz")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0:
        return float(value), "config.mapping.sampling_rate_hz"
    time = np.asarray(item.get("time_s", []), dtype=float)
    delta = np.diff(time)
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if len(delta):
        return float(1.0 / np.median(delta)), "median_local_time_difference"
    return None, "unavailable"


def _map_time_values(frame, row_indices, cols, mapping):
    if not isinstance(frame, pd.DataFrame):
        return frame, 0
    changed = 0
    for index in row_indices:
        for col in cols:
            if col not in frame.columns:
                continue
            value = frame.at[index, col]
            if pd.isna(value):
                continue
            local_col = f"local_{col}"
            if local_col not in frame.columns:
                frame[local_col] = np.nan
            if pd.isna(frame.at[index, local_col]):
                frame.at[index, local_col] = float(value)
            frame.at[index, col] = float(mapping["a"] + mapping["b"] * float(value))
            changed += 1
    return frame, changed


def _map_duration_values(frame, row_indices, cols, mapping):
    if not isinstance(frame, pd.DataFrame):
        return frame, 0
    changed = 0
    for index in row_indices:
        for col in cols:
            if col not in frame.columns:
                continue
            value = frame.at[index, col]
            if _missing(value):
                continue
            local_col = f"local_{col}"
            if local_col not in frame.columns:
                frame[local_col] = np.nan
            if _missing(frame.at[index, local_col]):
                frame.at[index, local_col] = float(value)
            frame.at[index, col] = float(mapping["b"] * float(value))
            changed += 1
    return frame, changed


def apply_clock_alignment(records, tables, config, *, cancel_callback=None):
    """Map local record clocks and associated time tables to the declared reference.

    The function copies records and tables. A local clock with no anchors remains
    explicitly unverified unless ``synchrony.shared_clock_verified`` is true.
    """
    if not isinstance(records, (list, tuple)):
        fail("sync_records", "records 必须是记录列表。", stage="synchrony")
    if not isinstance(tables, dict):
        fail("sync_tables", "tables 必须是关联表字典。", stage="synchrony")
    copied_records = [_copy_record(record) for record in records]
    copied_tables = {key: _copy_table(value) for key, value in tables.items()}
    anchors = copied_tables.get("sync_anchors")
    declared_reference_domains = {str(row.get("reference_clock_domain")) for row in _as_rows(anchors)
                                  if _present(row.get("reference_clock_domain"))}
    if len(declared_reference_domains) > 1:
        fail("sync_reference_domain_mismatch", "同步锚点声明了多个不同reference_clock_domain，不能放入同一共同网格。",
             field="sync_anchors.reference_clock_domain", stage="clock_alignment")
    maps = {}
    record_report = []
    for record in copied_records:
        _cancel(cancel_callback)
        key = _record_key(record)
        metadata = record.setdefault("metadata", {})
        old_time = np.asarray(record.get("time_s", []), dtype=float)
        fit = _mapping_for(record, anchors, config)
        local_rate, rate_source = _local_rate(record, config)
        fit["record_key"] = key
        fit["recording_id"] = _recording_id(record)
        fit["fs_local_hz"] = local_rate
        fit["fs_source"] = fit.get("fs_source", rate_source)
        if fit["status"] == "already_reference":
            metadata.setdefault("original_clock_domain", "reference")
            metadata.setdefault("clock_map_a", 0.0)
            metadata.setdefault("clock_map_b", 1.0)
            metadata.setdefault("time_mapping", {"a": metadata["clock_map_a"], "b": metadata["clock_map_b"]})
            metadata.setdefault("anchor_sha256", _canonical_hash([]))
            metadata.setdefault("time_mapping_version", TIME_MAPPING_VERSION)
            reference_rate = metadata.get("fs_reference_hz", metadata.get("fs_hz", local_rate))
            if reference_rate is not None and np.isfinite(reference_rate) and reference_rate > 0:
                metadata.setdefault("fs_reference_hz", float(reference_rate))
                metadata.setdefault("fs_hz", float(reference_rate))
            metadata["clock_domain"] = "reference"
            metadata["clock_alignment_status"] = "already_reference"
            metadata["clock_alignment_uncertainty_status"] = "preserved_existing_reference_evidence"
            fit["a"] = float(metadata["clock_map_a"])
            fit["b"] = float(metadata["clock_map_b"])
            fit["anchor_sha256"] = metadata["anchor_sha256"]
            fit["time_mapping_version"] = metadata.get("time_mapping_version")
            fit["fs_reference_hz"] = reference_rate
        elif fit["status"] in ("aligned", "shared_clock_verified"):
            if "local_time_s" not in record:
                record["local_time_s"] = np.array(old_time, copy=True)
            metadata.setdefault("local_time_s_preserved", True)
            metadata["original_clock_domain"] = fit.get("original_clock_domain", metadata.get("original_clock_domain", "reference"))
            metadata["clock_map_a"] = float(fit["a"])
            metadata["clock_map_b"] = float(fit["b"])
            metadata["time_mapping"] = {"a": float(fit["a"]), "b": float(fit["b"])}
            metadata["anchor_sha256"] = fit["anchor_sha256"]
            metadata["time_mapping_version"] = TIME_MAPPING_VERSION
            if fit.get("reference_clock_domain"):
                metadata["reference_clock_domain"] = fit["reference_clock_domain"]
            if fit["status"] != "already_reference":
                record["time_s"] = fit["a"] + fit["b"] * old_time
            metadata["clock_domain"] = "reference"
            fs_ref = local_rate / float(fit["b"]) if local_rate is not None else None
            if fs_ref is not None and np.isfinite(fs_ref) and fs_ref > 0:
                metadata["source_fs_hz"] = local_rate
                metadata["fs_local_hz"] = local_rate
                metadata["fs_hz"] = fs_ref
                metadata["fs_reference_hz"] = fs_ref
            metadata["clock_alignment_status"] = fit["status"]
            metadata["clock_alignment_uncertainty_status"] = fit.get("uncertainty_status", "not_applicable_reference_clock")
            if len(old_time):
                lo, hi = float(np.min(old_time)), float(np.max(old_time))
                anchor_lo, anchor_hi = fit.get("anchor_min_local_s"), fit.get("anchor_max_local_s")
                fit["extrapolation_before_s"] = max(0.0, anchor_lo - lo) if anchor_lo is not None else 0.0
                fit["extrapolation_after_s"] = max(0.0, hi - anchor_hi) if anchor_hi is not None else 0.0
                fit["extrapolated_sample_fraction"] = (
                    float(np.mean((old_time < anchor_lo) | (old_time > anchor_hi)))
                    if anchor_lo is not None and anchor_hi is not None else 0.0
                )
            else:
                fit.update({"extrapolation_before_s": 0.0, "extrapolation_after_s": 0.0,
                            "extrapolated_sample_fraction": 0.0})
        else:
            metadata["clock_alignment_status"] = "unverified"
            metadata["clock_alignment_reason"] = fit["reason"]
            metadata["original_clock_domain"] = fit.get("original_clock_domain")
            fit.update({"extrapolation_before_s": None, "extrapolation_after_s": None,
                        "extrapolated_sample_fraction": None, "fs_reference_hz": None})
        maps[key] = fit
        record_report.append(fit)

    table_report = []
    for table_name, time_columns in _TABLE_TIME_COLUMNS.items():
        table = copied_tables.get(table_name)
        if not isinstance(table, pd.DataFrame) or table.empty or "recording_id" not in table.columns:
            continue
        changed = 0
        inherited_domain_rows = []
        preserved_local_rows = []
        unresolved = []
        for index, row in table.iterrows():
            _cancel(cancel_callback)
            rid = str(row.get("recording_id", ""))
            candidates = [mapping for key, mapping in maps.items()
                          if mapping.get("recording_id") == rid]
            channel = row.get("channel")
            if _present(channel):
                candidates = [mapping for mapping in candidates
                              if mapping.get("record_key", "").endswith("::" + str(channel))]
            if not candidates:
                unresolved.append({"recording_id": rid, "row": int(index),
                                   "reason": "record_or_channel_clock_map_missing"})
                continue
            row_domain = row.get("clock_domain")
            if not _present(row_domain):
                row_domains = {str(mapping.get("original_clock_domain", "")) for mapping in candidates}
                if len(row_domains) != 1:
                    unresolved.append({"recording_id": rid, "row": int(index),
                                       "reason": "ambiguous_missing_table_clock_domain"})
                    continue
                local_domain = next(iter(row_domains))
                inherited_domain_rows.append({"recording_id": rid, "row": int(index),
                                              "inherited_clock_domain": local_domain})
            else:
                local_domain = str(row_domain)
            if local_domain.lower() == "reference":
                if any(mapping.get("status") == "unverified" for mapping in candidates):
                    unresolved.append({"recording_id": rid, "row": int(index),
                                       "reason": "reference_table_clock_without_record_mapping"})
                continue
            domain_matches = [mapping for mapping in candidates
                              if str(mapping.get("original_clock_domain", "")) == local_domain]
            if not domain_matches:
                unresolved.append({"recording_id": rid, "row": int(index),
                                   "reason": "clock_domain_mismatch", "table_clock_domain": local_domain,
                                   "record_clock_domains": sorted({str(mapping.get("original_clock_domain", ""))
                                                                    for mapping in candidates})})
                continue
            candidates = domain_matches
            unique_maps = {(mapping.get("status"), round(float(mapping.get("a", 0.0)), 15),
                            round(float(mapping.get("b", 1.0)), 15)) for mapping in candidates}
            if len(unique_maps) != 1:
                unresolved.append({"recording_id": rid, "row": int(index), "reason": "ambiguous_channel_clock_map"})
                continue
            chosen = candidates[0]
            if chosen.get("status") == "unverified":
                if "clock_domain" not in table.columns:
                    table["clock_domain"] = ""
                if not _present(row_domain):
                    table.at[index, "clock_domain"] = local_domain
                preserved_local_rows.append({"recording_id": rid, "row": int(index),
                                             "clock_domain": local_domain,
                                             "reason": chosen.get("reason", "clock_unverified")})
                continue
            if chosen.get("status") == "already_reference":
                if local_domain.lower() == "reference":
                    continue
                # A reference record may still have a linked table in its preserved
                # original local domain; reuse the original map exactly once.
                if "time_mapping_version" not in chosen:
                    unresolved.append({"recording_id": rid, "row": int(index),
                                       "reason": "local_table_clock_map_not_available_for_reference_record"})
                    continue
            local_domain = row_domain if _present(row_domain) else chosen.get("original_clock_domain")
            if "clock_domain" not in table.columns:
                table["clock_domain"] = ""
            if "local_clock_domain" not in table.columns:
                table["local_clock_domain"] = ""
            if not _present(table.at[index, "local_clock_domain"]):
                table.at[index, "local_clock_domain"] = local_domain
            table.at[index, "clock_domain"] = "reference"
            table, n_changed = _map_time_values(table, [index], time_columns, chosen)
            table, n_duration_changed = _map_duration_values(
                table, [index], _TABLE_DURATION_COLUMNS.get(table_name, ()), chosen)
            changed += n_changed + n_duration_changed
        copied_tables[table_name] = table
        table_report.append({"table": table_name, "mapped_values": changed,
                             "inherited_domain_rows": inherited_domain_rows,
                             "preserved_local_rows": preserved_local_rows,
                             "unresolved_rows": unresolved})
        if unresolved:
            fail("sync_table_clock_unresolved",
                 f"{table_name} 有 {len(unresolved)} 行的record/channel clock domain无法安全对应；已拒绝将局部秒数当作reference时间。",
                 field=table_name, stage="clock_alignment",
                 remediation="为对应时间表声明与信号记录一致的clock_domain，并提供可验证的时钟映射后重试。")
    report = {"status": "aligned" if all(x["status"] in ("aligned", "shared_clock_verified", "already_reference") for x in record_report)
              else "unverified",
              "time_mapping_version": TIME_MAPPING_VERSION,
              "reference_clock": "reference",
              "records": record_report,
              "tables": table_report,
              "formal_synchrony_allowed": all(x["status"] in ("aligned", "shared_clock_verified", "already_reference") for x in record_report),
              "notes": ["不从EDA相关性推断时钟映射。", "OLS保留全部有效锚点，不自动剔除异常点。",
                        "单锚点只支持offset；双锚点没有经验残差不确定度。"]}
    return copied_records, copied_tables, report


def _bundle_clock_map(bundle, anchors, config):
    metadata = bundle.get("metadata", {}) if isinstance(bundle.get("metadata"), dict) else {}
    timebase = bundle.get("timebase_metadata", {}) if isinstance(bundle.get("timebase_metadata"), dict) else {}
    if str(timebase.get("clock_domain", metadata.get("clock_domain", ""))).lower() == "reference":
        return {"status": "already_reference", "a": 0.0, "b": 1.0,
                "reason": "metadata_clock_domain_reference", "record_key": _record_key(bundle)}
    mapping = _mapping_for(bundle, anchors, config)
    mapping["record_key"] = _record_key(bundle)
    return mapping


def _membership_table(bundles):
    tables = []
    for bundle in bundles:
        membership = bundle.get("membership")
        if isinstance(membership, pd.DataFrame) and not membership.empty:
            tables.append(membership.copy(deep=True))
        elif membership:
            rows = _as_rows(membership)
            if rows:
                tables.append(pd.DataFrame(rows))
    if tables:
        return pd.concat(tables, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return pd.DataFrame()


def _rate_and_mask(bundle, key, config):
    if key not in bundle:
        fail("sync_basis_missing", f"同步basis信号缺少 {key}。", stage="synchrony", field=key)
    time = np.asarray(bundle.get("time_s", []), dtype=float)
    values = np.asarray(bundle[key], dtype=float)
    if time.ndim != 1 or values.ndim != 1 or len(time) != len(values) or not len(time):
        fail("sync_signal_shape", "同步信号与时间必须是等长非空一维数组。", stage="synchrony")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        fail("sync_timebase", "同步输入时间必须有限且严格递增。", stage="synchrony")
    quality = bundle.get("quality", {}) if isinstance(bundle.get("quality"), dict) else {}
    analysis = quality.get("analysis_eligible", quality.get("eligible"))
    if analysis is None:
        analysis = np.ones(len(time), dtype=bool)
    analysis = np.array(analysis, dtype=bool, copy=True)
    if analysis.shape != time.shape:
        fail("sync_mask_shape", "analysis_eligible必须与信号等长。", stage="synchrony")
    sync_cfg = config.get("synchrony", {})
    if not sync_cfg.get("allow_provisional", False):
        detector = bundle.get("detector_eligible", quality.get("detector_eligible"))
        if detector is not None:
            detector = np.asarray(detector, dtype=bool)
            if detector.shape != time.shape:
                fail("sync_mask_shape", "detector_eligible必须与信号等长。", stage="synchrony")
            analysis &= detector
        else:
            analysis &= ~np.asarray(quality.get("interpolated", np.zeros(len(time), bool)), dtype=bool)
            analysis &= ~np.asarray(quality.get("artifact", np.zeros(len(time), bool)), dtype=bool)
    analysis &= np.isfinite(values)
    max_gap = bundle.get("max_gap_s")
    if max_gap is None:
        valid_d = np.diff(time)[analysis[:-1] & analysis[1:]]
        valid_d = valid_d[np.isfinite(valid_d) & (valid_d > 0)]
        max_gap = float(np.median(valid_d) * 1.5) if len(valid_d) else None
    deltas = []
    segments = continuous_segments(analysis, time, max_gap)
    for start, end in segments:
        if end - start > 1:
            deltas.extend(np.diff(time[start:end]).tolist())
    if not deltas:
        fail("sync_sampling_rate", "同步记录没有足够的连续样本估计采样率。", stage="synchrony")
    delta = np.asarray(deltas, dtype=float)
    median_dt = float(np.median(delta))
    jitter_tol = float(config.get("preprocess", {}).get("jitter_tolerance", 0.01))
    relative = np.abs(delta - median_dt) / median_dt
    if np.any(relative > jitter_tol + 1e-12):
        fail("sync_irregular_timebase", "同步网格源采样在连续段内超出显式jitter容差。", stage="synchrony",
             remediation="先用有界、经说明的重采样处理不规则时间；不能把它直接插值成均匀网格。")
    fs_time = 1.0 / median_dt
    metadata = bundle.get("metadata", {}) if isinstance(bundle.get("metadata"), dict) else {}
    timebase = bundle.get("timebase_metadata", {}) if isinstance(bundle.get("timebase_metadata"), dict) else {}
    fs_declared = timebase.get("fs_reference_hz", timebase.get(
        "fs_hz", metadata.get("fs_reference_hz", metadata.get(
            "fs_hz", metadata.get("sampling_rate_hz", bundle.get("sampling_rate_hz"))))))
    slope = metadata.get("clock_map_b", timebase.get("clock_map_b", 1.0))
    clock_domain = timebase.get("clock_domain", metadata.get("clock_domain"))
    if fs_declared is not None and clock_domain != "reference" and slope and "fs_reference_hz" not in timebase:
        fs_declared = float(fs_declared) / float(slope)
    if fs_declared is not None:
        if not isinstance(fs_declared, (int, float)) or not np.isfinite(fs_declared) or fs_declared <= 0:
            fail("sync_reference_rate", "fs_reference_hz必须为正有限数。", stage="synchrony")
        if abs(float(fs_declared) - fs_time) / fs_time > max(jitter_tol, 1e-6):
            fail("sync_reference_rate_mismatch", "声明的参考采样率与处理时间网格不符。", stage="synchrony",
                 remediation="检查已映射time_s与timebase_metadata；不能静默改写采样率。")
    return time, values, analysis, max_gap, fs_time


def _empty_aligned(bundles, basis, unit, membership, report):
    keys = [_record_key(bundle) for bundle in bundles]
    return {"time_s": np.array([], dtype=float),
            "signals": {key: np.array([], dtype=float) for key in keys},
            "eligible": {key: np.array([], dtype=bool) for key in keys},
            "metadata": {key: copy.deepcopy(bundle.get("metadata", {})) for key, bundle in zip(keys, bundles)},
            "membership": membership.copy(deep=True), "basis": basis, "unit": unit,
            "alignment_report": copy.deepcopy(report)}


def align_recordings(processed_bundles, anchors=None, config=None, *, cancel_callback=None):
    """Resample processed bundles onto a shared reference-clock grid.

    Downsampling uses the preprocessing module's polyphase anti-alias FIR and
    support propagation. Interpolation is only used afterwards to reconcile the
    phase of already band-limited target-rate grids, never as a downsampler.
    """
    config = config or {}
    bundles = list(processed_bundles or [])
    sync_cfg = config.get("synchrony", {})
    basis_name = str(sync_cfg.get("basis", "phasic"))
    signal_name = _SIGNAL_FOR_BASIS.get(basis_name)
    if signal_name is None:
        fail("sync_basis", f"不支持同步basis：{basis_name}。", stage="synchrony")
    unit = "uS"
    membership = _membership_table(bundles)
    maps = [_bundle_clock_map(bundle, anchors, config) for bundle in bundles]
    allowed = all(item.get("status") in ("already_reference", "aligned", "shared_clock_verified") for item in maps)
    base_report = {"status": "aligned" if allowed else "unverified", "maps": maps,
                   "formal_synchrony_allowed": allowed, "basis": f"{basis_name}_uS", "unit": unit,
                   "resampling": "scipy.signal.resample_poly via core_preprocess._resample_arrays"}
    if not bundles:
        base_report.update({"status": "unavailable", "reason": "no_processed_bundles", "formal_synchrony_allowed": False})
        return _empty_aligned([], f"{basis_name}_uS", unit, membership, base_report), base_report
    if not allowed:
        base_report["reason"] = "unverified_clock_domain_or_missing_anchors"
        return _empty_aligned(bundles, f"{basis_name}_uS", unit, membership, base_report), base_report

    target_hz = float(sync_cfg.get("analysis_rate_hz", 4.0))
    if not np.isfinite(target_hz) or target_hz <= 0:
        fail("sync_analysis_rate", "analysis_rate_hz必须为正有限数。", category="configuration", stage="synchrony")
    prepared = {}
    map_by_key = {mapping["record_key"]: mapping for mapping in maps}
    for bundle in bundles:
        _cancel(cancel_callback)
        key = _record_key(bundle)
        if key in prepared:
            fail("sync_duplicate_record_key", f"同步记录键重复：{key}。", stage="synchrony")
        time, values, mask, max_gap, old_hz = _rate_and_mask(bundle, signal_name, config)
        clock_map = map_by_key[key]
        metadata_out = copy.deepcopy(bundle.get("metadata", {}))
        if clock_map.get("status") == "aligned":
            time = float(clock_map["a"]) + float(clock_map["b"]) * time
            old_hz = old_hz / float(clock_map["b"])
            metadata_out.update({"original_clock_domain": clock_map.get("original_clock_domain"),
                                 "clock_domain": "reference", "clock_map_a": float(clock_map["a"]),
                                 "clock_map_b": float(clock_map["b"]),
                                 "time_mapping": {"a": float(clock_map["a"]), "b": float(clock_map["b"])},
                                 "anchor_sha256": clock_map["anchor_sha256"],
                                 "time_mapping_version": TIME_MAPPING_VERSION,
                                 "fs_local_hz": old_hz * float(clock_map["b"]),
                                 "source_fs_hz": old_hz * float(clock_map["b"]),
                                 "fs_hz": old_hz, "fs_reference_hz": old_hz,
                                 "reference_clock_domain": clock_map.get("reference_clock_domain", "unspecified_reference"),
                                 "clock_alignment_status": "aligned"})
        elif clock_map.get("status") == "shared_clock_verified":
            metadata_out.update({"clock_domain": "reference", "clock_map_a": 0.0,
                                 "clock_map_b": 1.0,
                                 "anchor_sha256": clock_map["anchor_sha256"],
                                 "time_mapping_version": TIME_MAPPING_VERSION,
                                 "fs_hz": old_hz, "fs_reference_hz": old_hz,
                                 "clock_alignment_status": "shared_clock_verified"})
        else:
            metadata_out.update({"clock_domain": "reference", "clock_alignment_status": "already_reference"})
        # Use exactly the same polyphase FIR and conservative support rules as P04.
        quality = {
            "original_missing": ~np.isfinite(values), "nonfinite": ~np.isfinite(values),
            "auto_candidate": np.zeros(len(values), bool), "manual_exclude": np.zeros(len(values), bool),
            "manual_restore": np.zeros(len(values), bool),
            "interpolated": np.asarray(bundle.get("quality", {}).get("interpolated", np.zeros(len(values), bool)), bool),
            "edge_affected": np.asarray(bundle.get("quality", {}).get("edge_affected", np.zeros(len(values), bool)), bool),
            "eligible": mask.copy(), "processing_eligible": mask.copy(), "analysis_eligible": mask.copy(),
            "detector_eligible": mask.copy(), "artifact": np.asarray(bundle.get("quality", {}).get("artifact", np.zeros(len(values), bool)), bool),
            "source_eligible": mask.copy(),
        }
        segments = continuous_segments(mask, time, max_gap)
        out_time, out_values, out_quality, resample_info = _resample_arrays(
            time, values, quality, mask, target_hz, old_hz, segments)
        out_mask = out_quality.get("eligible", np.zeros(len(out_time), bool))
        # Also reject points whose detector-only/provisional support was propagated as invalid.
        out_mask &= out_quality.get("detector_eligible", out_mask)
        out_mask &= np.isfinite(out_values)
        prepared[key] = {"time": out_time, "values": out_values, "mask": out_mask,
                         "metadata": metadata_out,
                         "resampling": resample_info, "max_gap_s": max_gap}
    valid_extents = []
    for item in prepared.values():
        indices = np.flatnonzero(item["mask"])
        if len(indices):
            valid_extents.append((float(item["time"][indices[0]]), float(item["time"][indices[-1]])))
    if len(valid_extents) != len(prepared):
        base_report.update({"status": "unavailable", "reason": "no_common_valid_support",
                            "formal_synchrony_allowed": False})
        return _empty_aligned(bundles, f"{basis_name}_uS", unit, membership, base_report), base_report
    common_start = max(lo for lo, _ in valid_extents)
    common_end = min(hi for _, hi in valid_extents)
    first = int(math.ceil(common_start * target_hz - 1e-10))
    last = int(math.floor(common_end * target_hz + 1e-10))
    if last < first:
        base_report.update({"status": "unavailable", "reason": "empty_intersection", "formal_synchrony_allowed": False})
        return _empty_aligned(bundles, f"{basis_name}_uS", unit, membership, base_report), base_report
    grid = np.arange(first, last + 1, dtype=np.int64).astype(float) / target_hz
    if not len(grid):
        base_report.update({"status": "unavailable", "reason": "empty_intersection", "formal_synchrony_allowed": False})
        return _empty_aligned(bundles, f"{basis_name}_uS", unit, membership, base_report), base_report
    estimated_bytes = len(grid) * len(prepared) * 9
    memory_limit = int(config.get("compute", {}).get("memory_limit_mb", 2048)) * 1024 ** 2
    if estimated_bytes > memory_limit:
        fail("sync_memory_budget", "共同同步网格超过配置内存预算；未减少采样率或数据长度。",
             category="resource", stage="synchrony")
    signals, eligible, metadata, reports = {}, {}, {}, {}
    for key, item in prepared.items():
        _cancel(cancel_callback)
        output = np.full(len(grid), np.nan, dtype=float)
        output_mask = np.zeros(len(grid), dtype=bool)
        target_segments = continuous_segments(item["mask"], item["time"], 1.5 / target_hz + 1e-12)
        for start, end in target_segments:
            if end - start < 2:
                continue
            local_t = item["time"][start:end]
            local_y = item["values"][start:end]
            inside = (grid >= local_t[0] - 1e-10) & (grid <= local_t[-1] + 1e-10)
            if not np.any(inside):
                continue
            # This is phase correction at the already anti-aliased analysis rate.
            output[inside] = np.interp(grid[inside], local_t, local_y)
            output_mask[inside] = True
        signals[key], eligible[key] = output, output_mask
        metadata[key] = item["metadata"]
        reports[key] = item["resampling"]
    base_report.update({"status": "aligned", "reason": "", "analysis_rate_hz": target_hz,
                        "grid_start_s": float(grid[0]), "grid_end_s": float(grid[-1]),
                        "grid_samples": int(len(grid)), "common_support_s": float(grid[-1] - grid[0]),
                        "estimated_grid_bytes": int(estimated_bytes), "per_record_resampling": reports,
                        "membership_rows": int(len(membership)), "formal_synchrony_allowed": True})
    resolved_intervals, context_s, interval_report = _analysis_interval_resolution(grid, config)
    base_report.update({"analysis_intervals_s": resolved_intervals,
                        "analysis_context_s": context_s,
                        "analysis_interval_status": interval_report["status"],
                        "analysis_interval_reason": interval_report["reason"],
                        "analysis_interval_diagnostics": interval_report,
                        "analysis_support_policy": "declared_context_retained_full_grid_v1"})
    aligned = {"time_s": grid, "signals": signals, "eligible": eligible, "metadata": metadata,
               "membership": membership.copy(deep=True), "basis": f"{basis_name}_uS", "unit": unit,
               "analysis_intervals_s": copy.deepcopy(resolved_intervals),
               "analysis_context_s": context_s,
               "alignment_report": copy.deepcopy(base_report)}
    return aligned, base_report


def _membership_nodes(aligned):
    membership = aligned.get("membership")
    if not isinstance(membership, pd.DataFrame) or membership.empty:
        return [], {"status": "unavailable", "reason": "group_membership_missing", "missing_columns": ["group_id", "session_id", "subject_id", "recording_id"]}
    required = ("group_id", "session_id", "subject_id", "recording_id")
    missing = [column for column in required if column not in membership.columns]
    if missing:
        return [], {"status": "unavailable", "reason": "group_membership_columns_missing", "missing_columns": missing}
    by_rid = {}
    for key, metadata in aligned.get("metadata", {}).items():
        by_rid.setdefault(str(metadata.get("recording_id", key.split("::", 1)[0])), []).append((key, metadata))
    nodes, diagnostics = [], []
    for row_index, row in membership.iterrows():
        rid = str(row.get("recording_id", ""))
        choices = by_rid.get(rid, [])
        if _present(row.get("channel")):
            choices = [(key, meta) for key, meta in choices if str(meta.get("channel", "eda")) == str(row.get("channel"))]
        elif len(choices) > 1:
            default = [(key, meta) for key, meta in choices if str(meta.get("channel", "eda")) == "eda"]
            choices = default if len(default) == 1 else choices
        if len(choices) != 1:
            diagnostics.append({"row": int(row_index), "recording_id": rid,
                                "group_id": row.get("group_id"), "session_id": row.get("session_id"),
                                "subject_id": row.get("subject_id"), "status": "unavailable",
                                "reason": "recording_channel_membership_ambiguous" if choices else "membership_record_missing"})
            continue
        key, metadata = choices[0]
        values = {column: row.get(column) for column in membership.columns}
        missing_member = [column for column in ("group_id", "session_id", "subject_id")
                          if not _present(row.get(column))]
        if missing_member:
            diagnostics.append({"row": int(row_index), "recording_id": rid,
                                "group_id": row.get("group_id"), "session_id": row.get("session_id"),
                                "subject_id": row.get("subject_id"), "status": "unavailable",
                                "reason": "membership_identity_missing", "missing_fields": missing_member})
            continue
        nodes.append({**values, "record_key": key,
                      "group_id": str(row.get("group_id")), "session_id": str(row.get("session_id")),
                      "subject_id": str(row.get("subject_id")), "recording_id": rid,
                      "channel": str(row.get("channel", metadata.get("channel", "eda")))})
    return nodes, {"status": "available" if nodes and not diagnostics else "partial" if nodes else "unavailable",
                   "reason": "" if nodes else "no_resolvable_membership", "unresolved_rows": diagnostics}


def _fixed_edges(aligned, max_pairs=None):
    nodes, diagnostics = _membership_nodes(aligned)
    groups, edges = {}, []
    invalid_groups = {}
    unresolved = diagnostics.get("unresolved_rows", [])
    unscoped_problem = False
    for item in unresolved:
        if _present(item.get("group_id")) and _present(item.get("session_id")):
            invalid_groups[(str(item["group_id"]), str(item["session_id"]))] = item["reason"]
        else:
            unscoped_problem = True
    for node in nodes:
        groups.setdefault((node["group_id"], node["session_id"]), []).append(node)
    if diagnostics.get("status") == "unavailable" and not unresolved:
        invalid_groups[("", "")] = diagnostics.get("reason", "group_membership_unavailable")
    if unscoped_problem:
        reason = "unscoped_incomplete_group_membership"
        if groups:
            for key in groups:
                invalid_groups.setdefault(key, reason)
        else:
            invalid_groups[("", "")] = reason
    for group_key, members in sorted(groups.items()):
        if group_key in invalid_groups:
            continue
        subject_keys = [row["subject_id"] for row in members]
        if len(set(subject_keys)) != len(subject_keys):
            invalid_groups[group_key] = "duplicate_subject_membership"
            continue
        if len(members) < 2:
            invalid_groups[group_key] = "fewer_than_two_fixed_members"
            continue
    pair_count = sum(len(members) * (len(members) - 1) // 2
                     for key, members in groups.items() if key not in invalid_groups)
    if max_pairs is not None and pair_count > int(max_pairs):
        fail("sync_pair_budget", f"固定成员边数 {pair_count} 超过 compute.max_pairs={max_pairs}；未静默缩减。",
             category="resource", stage="synchrony")
    for group_key, members in sorted(groups.items()):
        if group_key in invalid_groups:
            continue
        for a, b in itertools.combinations(sorted(members, key=lambda x: (x["subject_id"], x["record_key"])), 2):
            if a["subject_id"] == b["subject_id"]:
                continue
            left, right = (a, b) if (a["subject_id"], a["record_key"]) <= (b["subject_id"], b["record_key"]) else (b, a)
            pair_id = f"{group_key[0]}::{group_key[1]}::{left['subject_id']}::{right['subject_id']}"
            edges.append({"group_id": group_key[0], "session_id": group_key[1],
                          "subject_id_a": left["subject_id"], "subject_id_b": right["subject_id"],
                          "record_key_a": left["record_key"], "record_key_b": right["record_key"],
                          "pair_id": pair_id})
    return edges, nodes, {"membership": diagnostics, "invalid_groups": [
        {"group_id": key[0], "session_id": key[1], "reason": reason} for key, reason in invalid_groups.items()]}


def _normalize_pairs(pairs, aligned, max_pairs=None):
    if pairs is None:
        return _fixed_edges(aligned, max_pairs=max_pairs)
    normalized = []
    for index, pair in enumerate(pairs):
        if isinstance(pair, dict):
            a = pair.get("record_key_a", pair.get("a"))
            b = pair.get("record_key_b", pair.get("b"))
            group_id, session_id = str(pair.get("group_id", "")), str(pair.get("session_id", ""))
            subject_a, subject_b = str(pair.get("subject_id_a", a)), str(pair.get("subject_id_b", b))
            pair_id = str(pair.get("pair_id", f"{group_id}::{session_id}::{subject_a}::{subject_b}"))
        else:
            try:
                a, b = pair
            except (TypeError, ValueError):
                fail("sync_pair_contract", f"pairs[{index}] 必须是双元素键或字典。", stage="synchrony")
            group_id, session_id = "", ""
            subject_a, subject_b = str(a), str(b)
            pair_id = f"pair::{min(str(a), str(b))}::{max(str(a), str(b))}"
        if a not in aligned.get("signals", {}) or b not in aligned.get("signals", {}):
            fail("sync_pair_missing_record", f"pairs[{index}] 引用了不存在的record_key。", stage="synchrony")
        if a == b:
            fail("sync_self_pair", "同步边不能把同一记录与自身配对。", stage="synchrony")
        if str(a) > str(b):
            a, b = b, a
            subject_a, subject_b = subject_b, subject_a
        normalized.append({"group_id": group_id, "session_id": session_id,
                           "subject_id_a": subject_a, "subject_id_b": subject_b,
                           "record_key_a": str(a), "record_key_b": str(b), "pair_id": pair_id})
    if max_pairs is not None and len(normalized) > int(max_pairs):
        fail("sync_pair_budget", f"显式pair数 {len(normalized)} 超过 compute.max_pairs={max_pairs}；未静默缩减。",
             category="resource", stage="synchrony")
    return normalized, [], {"membership": {"status": "explicit_pairs"}, "invalid_groups": []}


def _analysis_interval_resolution(time, config):
    """Resolve fixed analysis intervals while retaining the full aligned context grid."""
    sync = config.get("synchrony", {})
    context = float(sync.get("analysis_context_s", 0.0))
    length = float(sync.get("window_s", 30.0))
    step = float(sync.get("step_s", length))
    if length <= 0 or step <= 0:
        fail("sync_window_config", "window_s和step_s必须为正数。", category="configuration", stage="synchrony")
    if context < 0 or not np.isfinite(context):
        fail("sync_context_config", "analysis_context_s必须为非负有限数。", category="configuration", stage="synchrony")
    time = np.asarray(time, dtype=float)
    if not len(time):
        return [], context, {"status": "unavailable", "reason": "analysis_interval_too_short",
                             "intervals_s": [], "skipped_intervals_s": []}
    period = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
    support_start = float(time[0])
    support_end = float(time[-1] + period) if period > 0 else float(time[-1])
    configured = sync.get("analysis_intervals_s")
    if configured is None:
        lo, hi = support_start + context, support_end - context
        intervals = [[lo, hi]] if hi > lo else []
    else:
        intervals = [[float(item[0]), float(item[1])] for item in configured]
    resolved, skipped = [], []
    for lo, hi in intervals:
        if lo < support_start + context - 1e-9 or hi > support_end - context + 1e-9:
            skipped.append({"interval_s": [lo, hi], "reason": "analysis_context_unavailable"})
        elif hi - lo + 1e-12 < length:
            skipped.append({"interval_s": [lo, hi], "reason": "analysis_interval_too_short"})
        else:
            resolved.append([lo, hi])
    if configured is not None and skipped:
        reason = next((item["reason"] for item in skipped
                       if item["reason"] == "analysis_context_unavailable"),
                      "analysis_interval_too_short")
        return [], context, {"status": "unavailable", "reason": reason,
                             "intervals_s": [], "requested_intervals_s": intervals,
                             "skipped_intervals_s": skipped,
                             "common_grid_support_s": [support_start, support_end]}
    if not resolved:
        reason = "analysis_interval_too_short"
        if skipped and all(item["reason"] == "analysis_context_unavailable" for item in skipped):
            reason = "analysis_context_unavailable"
        return [], context, {"status": "unavailable", "reason": reason,
                             "intervals_s": [], "requested_intervals_s": intervals,
                             "skipped_intervals_s": skipped,
                             "common_grid_support_s": [support_start, support_end]}
    return resolved, context, {"status": "available", "reason": "", "intervals_s": resolved,
                               "requested_intervals_s": intervals, "skipped_intervals_s": skipped,
                               "common_grid_support_s": [support_start, support_end]}


def _window_specs(time, config):
    if not len(time):
        return []
    sync = config.get("synchrony", {})
    length = float(sync.get("window_s", 30.0))
    step = float(sync.get("step_s", length))
    intervals, _, resolution = _analysis_interval_resolution(time, config)
    if resolution["status"] != "available":
        return []
    maximum = int(config.get("compute", {}).get("max_windows", 10000))
    projected = sum(max(0, int(math.floor((hi - lo - length + 1e-10) / step)) + 1)
                    for lo, hi in intervals)
    if projected > maximum:
        fail("sync_window_budget", f"同步窗口上界 {projected} 超过 compute.max_windows={maximum}；未静默缩减。",
             category="resource", stage="synchrony")
    rows, ordinal = [], 0
    for lo, hi in intervals:
        count = max(0, int(math.floor((hi - lo - length + 1e-10) / step)) + 1)
        for index in range(count):
            start = float(lo + index * step)
            end = float(start + length)
            if end > hi + 1e-9:
                continue
            rows.append({"window_id": f"sync:{ordinal:05d}", "start_s": start, "end_s": end,
                         "expected_duration_s": length, "analysis_interval_s": [lo, hi]})
            ordinal += 1
    return rows


def _window_resolution(time, config):
    intervals, context, resolution = _analysis_interval_resolution(time, config)
    windows = _window_specs(time, config) if resolution["status"] == "available" else []
    if resolution["status"] == "available" and not windows:
        resolution = {**resolution, "status": "unavailable", "reason": "analysis_interval_too_short"}
    return windows, {**resolution, "analysis_context_s": context}


def _lag_grid(config):
    sync = config.get("synchrony", {})
    max_lag = float(sync.get("max_lag_s", 0.0))
    step = float(sync.get("lag_step_s", 1.0))
    if max_lag < 0 or step <= 0:
        fail("sync_lag_config", "max_lag_s必须非负且lag_step_s必须为正。", category="configuration", stage="synchrony")
    count = int(math.floor(max_lag / step + 1e-12))
    return np.arange(-count, count + 1, dtype=float) * step


def _effective_lag_indices(time, config):
    lags = _lag_grid(config)
    rate = 1.0 / float(np.median(np.diff(time))) if len(time) > 1 else float(
        config.get("synchrony", {}).get("analysis_rate_hz", 1.0))
    lag_indices = np.rint(lags * rate).astype(int)
    max_lag = float(config.get("synchrony", {}).get("max_lag_s", 0.0))
    lag_indices = np.asarray(sorted(set(int(item) for item in lag_indices
                                        if abs(int(item) / rate) <= max_lag + 1e-12)), dtype=int)
    if not np.any(lag_indices == 0):
        lag_indices = np.sort(np.append(lag_indices, 0))
    return lag_indices, lag_indices.astype(float) / rate, rate


def _corr(a, b, metric):
    if len(a) < 2 or len(b) < 2:
        return None, "insufficient_samples"
    if np.all(a == a[0]) or np.all(b == b[0]):
        return None, "constant_series"
    if metric == "spearman":
        a, b = stats.rankdata(a, method="average"), stats.rankdata(b, method="average")
    value = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(value):
        return None, "constant_series"
    return value, ""


def _lag_slices(length, lag_steps):
    if lag_steps > 0:
        return slice(0, length - lag_steps), slice(lag_steps, length)
    if lag_steps < 0:
        return slice(-lag_steps, length), slice(0, length + lag_steps)
    return slice(0, length), slice(0, length)


def _valid_duration(time, mask, max_gap=None):
    if len(time) < 2:
        return 0.0
    dt = np.diff(time)
    good = mask[:-1] & mask[1:] & np.isfinite(dt) & (dt > 0)
    if max_gap is not None:
        good &= dt <= max_gap
    return float(np.sum(dt[good]))


_VALUE_COLUMNS = ["group_id", "session_id", "window_id", "window_start_s", "window_end_s",
                  "pair_id", "subject_id_a", "subject_id_b", "record_key_a", "record_key_b",
                  "metric", "basis", "lag_s", "r", "status", "reason", "valid_n",
                  "duration_s", "coverage"]


def compute_pair_synchrony(aligned, pairs, config, *, cancel_callback=None):
    """Compute signed Pearson/Spearman correlations at zero and configured lags."""
    config = config or {}
    time = np.asarray(aligned.get("time_s", []), dtype=float)
    basis = str(aligned.get("basis", config.get("synchrony", {}).get("basis", "phasic")))
    max_pairs = int(config.get("compute", {}).get("max_pairs", 10000))
    edges, _, edge_diagnostics = _normalize_pairs(pairs, aligned, max_pairs=max_pairs)
    invalid_group_map = {(row["group_id"], row["session_id"]): row["reason"]
                         for row in edge_diagnostics.get("invalid_groups", [])}
    if len(edges) > max_pairs:
        fail("sync_pair_budget", f"固定成员边数 {len(edges)} 超过 compute.max_pairs={max_pairs}；未静默缩减。",
             category="resource", stage="synchrony")
    windows, window_resolution = _window_resolution(time, config)
    max_windows = int(config.get("compute", {}).get("max_windows", 10000))
    if len(windows) > max_windows:
        fail("sync_window_budget", f"同步窗口数 {len(windows)} 超过 compute.max_windows={max_windows}；未静默缩减。",
             category="resource", stage="synchrony")
    requested_metric = str(config.get("synchrony", {}).get("metric", "pearson")).lower()
    metrics = ("pearson", "spearman") if requested_metric in ("both", "pearson,spearman") else (requested_metric,)
    if any(metric not in ("pearson", "spearman") for metric in metrics):
        fail("sync_metric", f"不支持同步指标：{requested_metric}。", category="configuration", stage="synchrony")
    lag_indices, lags_effective, rate = _effective_lag_indices(time, config)
    minimum_n = int(config.get("synchrony", {}).get("min_samples", 8))
    min_coverage = float(config.get("synchrony", {}).get("min_coverage", 0.8))
    signals = aligned.get("signals", {})
    masks = aligned.get("eligible", {})
    rows = []
    if (not windows and len(time)
            and aligned.get("alignment_report", {}).get("formal_synchrony_allowed") is not False):
        reason = window_resolution.get("reason", "analysis_interval_too_short")
        marker = {"window_id": "analysis_interval_unavailable", "start_s": None, "end_s": None}
        for edge in edges:
            for metric in metrics:
                for lag in lags_effective:
                    rows.append({**edge, "window_id": marker["window_id"], "window_start_s": None,
                                 "window_end_s": None, "metric": metric, "basis": basis,
                                 "lag_s": float(lag), "r": np.nan, "status": "unavailable",
                                 "reason": reason, "valid_n": 0, "duration_s": 0.0, "coverage": 0.0})
        for (group_id, session_id), group_reason in sorted(invalid_group_map.items()):
            for metric in metrics:
                for lag in lags_effective:
                    rows.append({"group_id": group_id, "session_id": session_id,
                                 "window_id": marker["window_id"], "window_start_s": None,
                                 "window_end_s": None, "pair_id": f"unavailable::{group_id}::{session_id}",
                                 "subject_id_a": "", "subject_id_b": "", "record_key_a": "", "record_key_b": "",
                                 "metric": metric, "basis": basis, "lag_s": float(lag), "r": np.nan,
                                 "status": "unavailable", "reason": group_reason, "valid_n": 0,
                                 "duration_s": 0.0, "coverage": 0.0})
        return pd.DataFrame(rows, columns=_VALUE_COLUMNS), pd.DataFrame([{
            "status": "unavailable", "reason": reason, "pair_count": len(edges),
            "window_count": 0, "analysis_intervals_s": window_resolution.get("intervals_s", []),
            "analysis_context_s": window_resolution.get("analysis_context_s", 0.0)}])
    if invalid_group_map:
        invalid_windows = windows or [{"window_id": "unavailable", "start_s": None, "end_s": None,
                                      "expected_duration_s": 0.0}]
        for (group_id, session_id), reason in sorted(invalid_group_map.items()):
            for window in invalid_windows:
                for metric in metrics:
                    for lag in lags_effective:
                        rows.append({"group_id": group_id, "session_id": session_id,
                                     "window_id": window["window_id"], "window_start_s": window["start_s"],
                                     "window_end_s": window["end_s"], "pair_id": f"unavailable::{group_id}::{session_id}",
                                     "subject_id_a": "", "subject_id_b": "", "record_key_a": "", "record_key_b": "",
                                     "metric": metric, "basis": basis, "lag_s": float(lag), "r": np.nan,
                                     "status": "unavailable", "reason": reason, "valid_n": 0,
                                     "duration_s": 0.0, "coverage": 0.0})
    if not len(time) or aligned.get("alignment_report", {}).get("formal_synchrony_allowed") is False:
        for edge in edges:
            for window in (windows or [{"window_id": "unverified", "start_s": None, "end_s": None,
                                        "expected_duration_s": 0.0}]) :
                for metric in metrics:
                    for lag in lags_effective:
                        rows.append({**edge, "window_id": window["window_id"], "window_start_s": window["start_s"],
                                     "window_end_s": window["end_s"], "metric": metric, "basis": basis,
                                     "lag_s": float(lag), "r": np.nan, "status": "unverified",
                                     "reason": aligned.get("alignment_report", {}).get("reason", "clock_alignment_unverified"),
                                     "valid_n": 0, "duration_s": 0.0, "coverage": 0.0})
        return pd.DataFrame(rows, columns=_VALUE_COLUMNS), pd.DataFrame([
            {"status": "unverified", "reason": "clock_alignment_unverified", "pair_count": len(edges),
             "window_count": len(windows)}])
    for edge in edges:
        _cancel(cancel_callback)
        group_reason = invalid_group_map.get((edge["group_id"], edge["session_id"]))
        ka, kb = edge["record_key_a"], edge["record_key_b"]
        if ka not in signals or kb not in signals:
            fail("sync_pair_missing_signal", "同步pair在aligned中缺少信号。", stage="synchrony")
        xa, xb = np.asarray(signals[ka], float), np.asarray(signals[kb], float)
        ma, mb = np.asarray(masks[ka], bool), np.asarray(masks[kb], bool)
        if any(array.shape != time.shape for array in (xa, xb, ma, mb)):
            fail("sync_aligned_shape", "共同网格上的信号与mask必须和time_s等长。", stage="synchrony")
        for window in windows:
            selected = np.flatnonzero((time >= window["start_s"] - 1e-12) & (time < window["end_s"] - 1e-12))
            for metric in metrics:
                for lag, lag_steps in zip(lags_effective, lag_indices):
                    a_slice, b_slice = _lag_slices(len(selected), int(lag_steps))
                    ia, ib = selected[a_slice], selected[b_slice]
                    if not len(ia) or not len(ib):
                        valid = np.zeros(0, bool)
                        a, b, sample_times = np.array([]), np.array([]), np.array([])
                    else:
                        valid = ma[ia] & mb[ib] & np.isfinite(xa[ia]) & np.isfinite(xb[ib])
                        a, b, sample_times = xa[ia][valid], xb[ib][valid], time[ia][valid]
                    duration = _valid_duration(sample_times, np.ones(len(sample_times), bool), 1.5 / rate) if len(sample_times) else 0.0
                    expected = float(window["expected_duration_s"])
                    coverage = min(1.0, duration / expected) if expected > 0 else 0.0
                    status, reason, correlation = "valid", "", None
                    if group_reason:
                        status, reason = "unavailable", group_reason
                    elif len(a) < minimum_n:
                        status, reason = "unavailable", "insufficient_valid_samples"
                    elif coverage + 1e-12 < min_coverage:
                        status, reason = "unavailable", "coverage_below_minimum"
                    else:
                        correlation, reason = _corr(a, b, metric)
                        if correlation is None:
                            status = "unavailable"
                    rows.append({**edge, "window_id": window["window_id"],
                                 "window_start_s": float(window["start_s"]), "window_end_s": float(window["end_s"]),
                                 "metric": metric, "basis": basis, "lag_s": float(lag),
                                 "r": np.nan if correlation is None else correlation,
                                 "status": status, "reason": reason, "valid_n": int(len(a)),
                                 "duration_s": duration, "coverage": coverage})
    values = pd.DataFrame(rows, columns=_VALUE_COLUMNS)
    coverage_rows = []
    if not values.empty:
        keys = ["group_id", "session_id", "window_id", "pair_id", "subject_id_a", "subject_id_b",
                "record_key_a", "record_key_b", "metric", "basis"]
        for key, group in values.groupby(keys, dropna=False, sort=True):
            valid = group.loc[group["status"] == "valid"]
            first = group.iloc[0]
            coverage_rows.append({**dict(zip(keys, key)), "status": "valid" if len(valid) else str(first["status"]),
                                  "reason": "" if len(valid) else str(first["reason"]),
                                  "expected_duration_s": float(first["window_end_s"] - first["window_start_s"])
                                  if pd.notna(first["window_end_s"]) and pd.notna(first["window_start_s"]) else 0.0,
                                  "minimum_valid_n": int(group["valid_n"].min()),
                                  "maximum_valid_n": int(group["valid_n"].max()),
                                  "minimum_coverage": float(group["coverage"].min()),
                                  "maximum_coverage": float(group["coverage"].max()),
                                  "lag_count": int(len(group)), "zero_lag_present": bool(np.any(np.isclose(group["lag_s"], 0)))})
    return values, pd.DataFrame(coverage_rows)


def _aggregate_edges(rows, metric):
    if len(rows) == 0 or not np.all(rows["status"].eq("valid")):
        return np.nan, "unavailable", "incomplete_fixed_edge_set"
    values = rows["r"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        return np.nan, "unavailable", "nonfinite_edge_value"
    if metric == "pearson":
        clipped = np.clip(values, -1 + 1e-12, 1 - 1e-12)
        return float(np.tanh(np.mean(np.arctanh(clipped)))), "valid", "fisher_z_mean"
    return float(np.mean(values)), "valid", "mean_spearman_distinct_estimator"


def summarize_group_synchrony(values, aligned, config=None, *, cancel_callback=None):
    """Summarize the fixed complete edge set and fixed-member leave-one-out sets."""
    config = config or {}
    frame = values.copy(deep=True) if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    edges, nodes, diagnostics = _fixed_edges(aligned, max_pairs=int(config.get("compute", {}).get("max_pairs", 10000)))
    invalid_group_map = {(row["group_id"], row["session_id"]): row["reason"]
                         for row in diagnostics.get("invalid_groups", [])}
    if frame.empty:
        return pd.DataFrame(columns=["group_id", "session_id", "window_id", "metric", "basis", "lag_s",
                                     "summary_type", "excluded_subject_id", "r", "status", "reason",
                                     "member_count", "edge_count", "expected_edge_count", "coverage_min"])
    group_members = {}
    for node in nodes:
        group_members.setdefault((node["group_id"], node["session_id"]), []).append(node["subject_id"])
    expected = {(edge["group_id"], edge["session_id"], edge["pair_id"]): edge for edge in edges}
    keys = [column for column in ("group_id", "session_id", "window_id", "metric", "basis", "lag_s") if column in frame]
    templates = frame[keys].drop_duplicates().sort_values(keys, kind="stable")
    out = []
    for _, template in templates.iterrows():
        _cancel(cancel_callback)
        group_id, session_id = str(template["group_id"]), str(template["session_id"])
        members = sorted(set(group_members.get((group_id, session_id), [])))
        local = frame
        for column in keys:
            local = local.loc[local[column].eq(template[column])]
        expected_edges = [edge for edge in edges if edge["group_id"] == group_id and edge["session_id"] == session_id]
        by_pair = local.set_index("pair_id", drop=False) if "pair_id" in local else pd.DataFrame()
        selected = []
        missing = []
        for edge in expected_edges:
            if edge["pair_id"] not in by_pair.index:
                missing.append(edge["pair_id"])
            else:
                row = by_pair.loc[edge["pair_id"]]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                selected.append(row)
        edge_frame = pd.DataFrame(selected)
        metric = str(template["metric"])
        group_invalid = invalid_group_map.get((group_id, session_id))
        if group_invalid:
            r, status, reason = np.nan, "unavailable", group_invalid
        elif missing:
            r, status, reason = np.nan, "unavailable", "missing_fixed_member_edge"
        else:
            r, status, reason = _aggregate_edges(edge_frame, metric)
        coverages = edge_frame["coverage"].to_numpy(dtype=float) if "coverage" in edge_frame else np.array([])
        row_data = {column: template[column] for column in keys}
        out.append({**row_data, "summary_type": "all_fixed_members", "excluded_subject_id": "",
                    "r": r, "status": status, "reason": reason,
                    "member_count": len(members), "edge_count": len(edge_frame),
                    "expected_edge_count": len(expected_edges),
                    "coverage_min": float(np.min(coverages)) if len(coverages) else np.nan,
                    "aggregation": "fisher_z_mean" if metric == "pearson" else "mean_spearman_distinct_estimator"})
        for excluded in ([] if group_invalid else members):
            kept = [member for member in members if member != excluded]
            loo_edges = [edge for edge in expected_edges
                         if excluded not in (edge["subject_id_a"], edge["subject_id_b"])]
            loo_rows, loo_missing = [], []
            for edge in loo_edges:
                if edge["pair_id"] not in by_pair.index:
                    loo_missing.append(edge["pair_id"])
                else:
                    candidate = by_pair.loc[edge["pair_id"]]
                    loo_rows.append(candidate.iloc[0] if isinstance(candidate, pd.DataFrame) else candidate)
            loo_frame = pd.DataFrame(loo_rows)
            if len(kept) < 2:
                loo_r, loo_status, loo_reason = np.nan, "unavailable", "leave_one_out_requires_two_fixed_members"
            elif loo_missing:
                loo_r, loo_status, loo_reason = np.nan, "unavailable", "missing_fixed_member_edge"
            else:
                loo_r, loo_status, loo_reason = _aggregate_edges(loo_frame, metric)
            loo_cov = loo_frame["coverage"].to_numpy(dtype=float) if "coverage" in loo_frame else np.array([])
            out.append({**row_data, "summary_type": "leave_one_out", "excluded_subject_id": excluded,
                        "r": loo_r, "status": loo_status, "reason": loo_reason,
                        "member_count": len(kept), "edge_count": len(loo_frame),
                        "expected_edge_count": len(loo_edges),
                        "coverage_min": float(np.min(loo_cov)) if len(loo_cov) else np.nan,
                        "aggregation": "fisher_z_mean" if metric == "pearson" else "mean_spearman_distinct_estimator"})
    return pd.DataFrame(out)


_GROUP_FEATURE_COLUMNS = FEATURE_COLUMNS
_SYNC_EPISODE_COLUMNS = ["group_id", "session_id", "metric", "basis", "sign", "start_s", "end_s",
                         "observed_duration_s", "left_censored", "right_censored", "member_set",
                         "edge_set", "window_ids", "status", "reason"]
_NODE_STRENGTH_COLUMNS = ["group_id", "session_id", "subject_id", "window_id", "metric", "basis",
                          "signed", "positive", "negative", "absolute", "normalized_signed",
                          "effective_degree", "expected_degree", "coverage", "status", "reason"]
_GROUP_TEMPORAL_DIAGNOSTIC_COLUMNS = ["group_id", "session_id", "metric", "basis", "status", "reason",
                                      "analysis_duration_s", "valid_class_duration_s", "unknown_duration_s",
                                      "positive_duration_s", "negative_duration_s", "neutral_duration_s",
                                      "window_count", "support_segment_count", "member_set", "edge_set"]


def _merge_time_intervals(intervals):
    result = []
    for lo, hi in sorted((float(lo), float(hi)) for lo, hi in intervals if hi > lo):
        if result and lo <= result[-1][1] + 1e-12:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((lo, hi))
    return result


def _time_intersection(intervals, start, end):
    return _merge_time_intervals([(max(lo, start), min(hi, end)) for lo, hi in intervals
                                  if min(hi, end) > max(lo, start)])


def _time_duration(intervals):
    return float(sum(max(0.0, hi - lo) for lo, hi in intervals))


def _aligned_group_support(alignment, group_id, session_id, member_rows):
    """从固定成员共同 eligibility 掩码取连续时间支持区间。"""
    time = np.asarray(alignment.get("time_s", []), dtype=float)
    if time.ndim != 1 or len(time) < 2 or np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0):
        return [], 0
    metadata = alignment.get("metadata", {})
    eligible = alignment.get("eligible", {})
    signals = alignment.get("signals", {})
    keys = []
    for member in member_rows:
        rid = str(member.get("recording_id", ""))
        channel_value = member.get("channel")
        candidates = [key for key, meta in metadata.items()
                      if str(meta.get("recording_id", key.split("::", 1)[0])) == rid]
        if channel_value is not None and not pd.isna(channel_value) and str(channel_value).strip() not in ("", "*"):
            candidates = [key for key in candidates
                          if str(metadata.get(key, {}).get("channel", "eda")) == str(channel_value)]
        elif len(candidates) > 1:
            eda_candidates = [key for key in candidates if str(metadata.get(key, {}).get("channel", "eda")) == "eda"]
            candidates = eda_candidates if len(eda_candidates) == 1 else candidates
        if len(candidates) != 1:
            return [], 0
        keys.append(candidates[0])
    if not keys:
        return [], 0
    common = np.ones(len(time), dtype=bool)
    for key in keys:
        mask = np.asarray(eligible.get(key, []), dtype=bool)
        values = np.asarray(signals.get(key, []), dtype=float)
        if mask.shape != time.shape or values.shape != time.shape:
            return [], 0
        common &= mask & np.isfinite(values)
    rate = alignment.get("alignment_report", {}).get("analysis_rate_hz")
    if rate is None or not np.isfinite(rate) or float(rate) <= 0:
        rate = 1.0 / float(np.median(np.diff(time)))
    max_gap = 1.5 / float(rate) + 1e-12
    segments = continuous_segments(common, time, max_gap)
    intervals = [(float(time[start]), float(time[stop - 1])) for start, stop in segments if stop - start >= 2]
    return _merge_time_intervals(intervals), len(intervals)


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    ordered_values, ordered_weights = np.asarray(values)[order], np.asarray(weights)[order]
    threshold = 0.5 * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _group_feature_row(group_id, session_id, feature_id, value, unit, basis, method, status, reason,
                       n, duration, *, subject_id=None, window_id=None):
    return {"recording_id": str(group_id), "subject_id": subject_id, "session_id": str(session_id),
            "window_id": str(window_id or f"sync:{group_id}:{session_id}:summary"),
            "feature_id": feature_id, "value": value if status == "valid" else np.nan,
            "unit": unit, "basis": basis, "method": method, "status": status,
            "reason": reason or "", "valid_n": int(n), "valid_duration_s": float(duration)}


def _fixed_group_graph(membership_rows, pair_rows, group_id, session_id):
    """Resolve a group's declared complete graph without inferring it from observed edges."""
    declared_members = [row for row in membership_rows
                        if str(row.get("group_id")) == str(group_id)
                        and str(row.get("session_id")) == str(session_id)]
    declared_subjects = [str(row.get("subject_id")) for row in declared_members
                         if _present(row.get("subject_id"))]
    roster_known = (bool(declared_members) and len(declared_subjects) == len(declared_members)
                    and len(set(declared_subjects)) == len(declared_subjects)
                    and len(declared_subjects) >= 2)
    subjects = sorted(set(declared_subjects))
    if roster_known:
        expected_pairs = list(itertools.combinations(subjects, 2))
        expected_edge_ids = [f"{group_id}::{session_id}::{a}::{b}" for a, b in expected_pairs]
    else:
        expected_pairs, expected_edge_ids = [], []
        observed_subjects = set()
        if isinstance(pair_rows, pd.DataFrame) and not pair_rows.empty:
            for row in pair_rows.to_dict("records"):
                for field in ("subject_id_a", "subject_id_b"):
                    if _present(row.get(field)):
                        observed_subjects.add(str(row[field]))
        subjects = sorted(set(subjects) | observed_subjects)
    observed_edge_ids = (sorted(set(pair_rows.get("pair_id", pd.Series(dtype=object)).dropna().astype(str)))
                         if isinstance(pair_rows, pd.DataFrame) and not pair_rows.empty else [])
    return {"known": roster_known, "subjects": subjects, "pairs": expected_pairs,
            "expected_edge_ids": expected_edge_ids,
            "edge_ids": expected_edge_ids if roster_known else observed_edge_ids}


def _aligned_pair_support(alignment, pair_row):
    """返回pair两端实际共同eligible样本的连续时间支持，不按lag重复计窗。"""
    time = np.asarray(alignment.get("time_s", []), dtype=float)
    if time.ndim != 1 or len(time) < 2 or np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0):
        return []
    key_a, key_b = str(pair_row.get("record_key_a", "")), str(pair_row.get("record_key_b", ""))
    masks, signals = alignment.get("eligible", {}), alignment.get("signals", {})
    if key_a not in masks or key_b not in masks or key_a not in signals or key_b not in signals:
        return []
    mask_a, mask_b = np.asarray(masks[key_a], dtype=bool), np.asarray(masks[key_b], dtype=bool)
    values_a, values_b = np.asarray(signals[key_a], dtype=float), np.asarray(signals[key_b], dtype=float)
    if any(array.shape != time.shape for array in (mask_a, mask_b, values_a, values_b)):
        return []
    common = mask_a & mask_b & np.isfinite(values_a) & np.isfinite(values_b)
    rate = 1.0 / float(np.median(np.diff(time)))
    max_gap = 1.5 / float(rate) + 1e-12
    segments = continuous_segments(common, time, max_gap)
    return _merge_time_intervals([(float(time[start]), float(time[stop - 1]))
                                  for start, stop in segments if stop - start >= 2])


def extract_group_temporal_features(pair_values, group_summary, alignment, config):
    """提取固定成员同步的时间占比、截尾片段、滞后稳定度与节点强度。

    时间 cell 由窗口中心中点定义，并裁剪到原窗口、analysis intervals 与共同有效
    成员支持；未观测区间保留为 unknown，不把缺失 r 设为零。episode 只在时间连续、
    同符号且成员/边集相同的 cell 之间延伸。滞后均值、稳定度和人口加权标准差
    (ddof=0)复用相同的非重叠 cell，并按pair两端实际共同eligible连续支持的交集时长加权；
    每pair分别计算后再等权汇总。节点预期度数只从固定membership roster导出，不根据
    当前可见边缩小。返回 (features_long, episodes, node_strength, diagnostics)，
    所有指标仍为实验工程实现。
    """
    config = config or {}
    settings = config.get("features", {}).get("extended", {}).get("group_temporal", {})
    feature_cols = list(_GROUP_FEATURE_COLUMNS)
    empty_features = pd.DataFrame(columns=feature_cols)
    empty_episodes = pd.DataFrame(columns=_SYNC_EPISODE_COLUMNS)
    empty_nodes = pd.DataFrame(columns=_NODE_STRENGTH_COLUMNS)
    empty_diagnostics = pd.DataFrame(columns=_GROUP_TEMPORAL_DIAGNOSTIC_COLUMNS)
    if settings.get("enabled", True) is not True:
        return empty_features, empty_episodes, empty_nodes, empty_diagnostics
    pairs = pair_values.copy(deep=True) if isinstance(pair_values, pd.DataFrame) else pd.DataFrame(pair_values)
    summary = group_summary.copy(deep=True) if isinstance(group_summary, pd.DataFrame) else pd.DataFrame(group_summary)
    if pairs.empty or summary.empty:
        return empty_features, empty_episodes, empty_nodes, empty_diagnostics
    summary = summary.loc[summary.get("summary_type", pd.Series(index=summary.index, dtype=object)).eq("all_fixed_members")].copy()
    if "lag_s" in summary:
        summary = summary.loc[np.isclose(pd.to_numeric(summary["lag_s"], errors="coerce"), 0.0)].copy()
    elif "lag_s" in pairs:
        summary = summary.loc[summary["window_id"].isin(pairs.loc[np.isclose(pairs["lag_s"], 0.0), "window_id"])]
    if summary.empty:
        return empty_features, empty_episodes, empty_nodes, empty_diagnostics
    bounds = (pairs.sort_values([c for c in ("group_id", "session_id", "window_id") if c in pairs], kind="stable")
              .drop_duplicates([c for c in ("group_id", "session_id", "window_id") if c in pairs]))
    bounds_by_key = {(str(row.get("group_id")), str(row.get("session_id")), str(row.get("window_id"))): row
                     for row in bounds.to_dict("records")}
    membership = alignment.get("membership")
    membership_rows = membership.to_dict("records") if isinstance(membership, pd.DataFrame) else []
    threshold = float(settings.get("correlation_threshold", 0.1))
    tolerance = float(settings.get("lag_stability_tolerance_s", 1.0))
    minimum_windows = int(settings.get("min_valid_windows", 3))
    intervals = alignment.get("analysis_intervals_s")
    if not intervals:
        intervals = alignment.get("alignment_report", {}).get("analysis_intervals_s")
    if not intervals:
        return empty_features, empty_episodes, empty_nodes, pd.DataFrame([{
            "group_id": "", "session_id": "", "metric": "", "basis": "", "status": "unavailable",
            "reason": "analysis_intervals_unavailable", "analysis_duration_s": 0.0,
            "valid_class_duration_s": 0.0, "unknown_duration_s": np.nan,
            "positive_duration_s": 0.0, "negative_duration_s": 0.0, "neutral_duration_s": 0.0,
            "window_count": 0, "support_segment_count": 0, "member_set": [], "edge_set": []}],
            columns=_GROUP_TEMPORAL_DIAGNOSTIC_COLUMNS)
    intervals = _merge_time_intervals(intervals)
    analysis_duration = _time_duration(intervals)
    group_keys = [col for col in ("group_id", "session_id", "metric", "basis") if col in summary]
    all_features, all_episodes, all_nodes, all_diags = [], [], [], []
    temporal_cell_bounds = {}
    group_series = summary.groupby(group_keys, dropna=False, sort=True)
    for key, group in group_series:
        key = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(group_keys, key))
        gid, sid = str(meta.get("group_id", "")), str(meta.get("session_id", ""))
        metric, basis = str(meta.get("metric", "pearson")), str(meta.get("basis", ""))
        group_pairs = pairs.loc[pairs["group_id"].astype(str).eq(gid) & pairs["session_id"].astype(str).eq(sid)
                                & pairs["metric"].astype(str).eq(metric) & pairs["basis"].astype(str).eq(basis)]
        group_membership = [row for row in membership_rows
                            if str(row.get("group_id")) == gid and str(row.get("session_id")) == sid]
        fixed_graph = _fixed_group_graph(membership_rows, group_pairs, gid, sid)
        members = fixed_graph["subjects"]
        edge_set = fixed_graph["edge_ids"]
        common_support, support_segment_count = _aligned_group_support(alignment, gid, sid, group_membership)
        cells = []
        rows = group.sort_values("window_id", kind="stable").to_dict("records")
        for row in rows:
            bound_row = bounds_by_key.get((gid, sid, str(row.get("window_id"))), {})
            start, end = bound_row.get("window_start_s"), bound_row.get("window_end_s")
            try:
                start, end = float(start), float(end)
            except (TypeError, ValueError):
                cells.append({"group_id": gid, "session_id": sid, "metric": metric, "basis": basis,
                              "window_id": str(row.get("window_id")), "start": np.nan, "end": np.nan,
                              "r": np.nan, "class": "unknown", "members": tuple(members), "edges": tuple(edge_set),
                              "interval_index": -1, "duration": 0.0, "reason": "window_bounds_unavailable"})
                continue
            center = (start + end) / 2.0
            interval_index = next((idx for idx, (lo, hi) in enumerate(intervals)
                                   if lo - 1e-12 <= center <= hi + 1e-12), -1)
            if interval_index < 0:
                continue
            lo, hi = intervals[interval_index]
            local_centers = [((float(b.get("window_start_s")) + float(b.get("window_end_s"))) / 2.0)
                             for b in bounds.to_dict("records")
                             if str(b.get("group_id")) == gid and str(b.get("session_id")) == sid
                             and float(b.get("window_start_s", np.nan)) < hi
                             and float(b.get("window_end_s", np.nan)) > lo]
            before = [value for value in local_centers if value < center - 1e-12]
            after = [value for value in local_centers if value > center + 1e-12]
            cell_start = max(start, (max(before) + center) / 2.0 if before else lo)
            cell_end = min(end, (min(after) + center) / 2.0 if after else hi)
            cell_start, cell_end = max(cell_start, lo), min(cell_end, hi)
            temporal_cell_bounds[(gid, sid, str(row.get("window_id")))] = (cell_start, cell_end)
            supported = _time_intersection(common_support, cell_start, cell_end)
            duration = _time_duration(supported)
            value = row.get("r")
            valid = row.get("status") == "valid" and value is not None and np.isfinite(value) and duration > 0
            value = float(value) if valid else np.nan
            sign = "positive" if valid and value > threshold else "negative" if valid and value < -threshold else "neutral" if valid else "unknown"
            # Split an individual window cell at every common-support gap. This preserves
            # exposure weighting and prevents an episode from bridging an internal gap.
            for support_start, support_end in supported:
                cells.append({"group_id": gid, "session_id": sid, "metric": metric, "basis": basis,
                              "window_id": str(row.get("window_id")), "start": support_start, "end": support_end,
                              "r": value, "class": sign, "members": tuple(members), "edges": tuple(edge_set),
                              "interval_index": interval_index, "duration": float(support_end - support_start),
                              "reason": "" if valid else str(row.get("reason", "invalid_group_correlation") or "invalid_group_correlation")})
        cells.sort(key=lambda item: (item["interval_index"], item["start"] if np.isfinite(item["start"]) else np.inf))
        positive_duration = sum(cell["duration"] for cell in cells if cell["class"] == "positive")
        negative_duration = sum(cell["duration"] for cell in cells if cell["class"] == "negative")
        neutral_duration = sum(cell["duration"] for cell in cells if cell["class"] == "neutral")
        valid_class_duration = positive_duration + negative_duration + neutral_duration
        unknown_duration = max(0.0, analysis_duration - valid_class_duration)
        for feature_id, value, reason in (
                ("sync_positive_time_proportion", positive_duration / valid_class_duration if valid_class_duration > 0 else None,
                 "zero_valid_class_duration"),
                ("sync_negative_time_proportion", negative_duration / valid_class_duration if valid_class_duration > 0 else None,
                 "zero_valid_class_duration")):
            status = "valid" if value is not None else "unavailable"
            all_features.append(_group_feature_row(gid, sid, feature_id, value, "fraction",
                                                   "lag0_equal_pair_group_r;common_member_support",
                                                   "duration_proportion_of_valid_classified_cells", status,
                                                   "" if status == "valid" else reason, len(cells), valid_class_duration,
                                                   window_id=f"sync:{gid}:{sid}:{metric}:temporal"))

        # An observed neutral/opposite cell completes an episode; unknown/gap/member change right-censors it.
        episodes = []
        active = None

        def close_episode(*, right_censored):
            nonlocal active
            if active is not None:
                episodes.append({"group_id": gid, "session_id": sid, "metric": metric, "basis": basis,
                                 "sign": active["sign"], "start_s": active["start"], "end_s": active["end"],
                                 "observed_duration_s": active["end"] - active["start"],
                                 "left_censored": active["left_censored"], "right_censored": bool(right_censored),
                                 "member_set": list(active["members"]), "edge_set": list(active["edges"]),
                                 "window_ids": list(active["window_ids"]), "status": "censored" if active["left_censored"] or right_censored else "complete",
                                 "reason": "unknown_gap_or_support_boundary" if active["left_censored"] or right_censored else ""})
                active = None

        previous = None
        for cell in cells:
            known = cell["class"] in ("positive", "negative", "neutral") and cell["duration"] > 0
            if active is not None:
                contiguous = (previous is not None and cell["interval_index"] == previous["interval_index"] and
                              abs(cell["start"] - previous["end"]) <= 1e-9)
                same_graph = cell["members"] == active["members"] and cell["edges"] == active["edges"]
                same_episode = known and contiguous and same_graph and cell["class"] == active["sign"]
                if not same_episode:
                    observed_termination = known and contiguous and same_graph
                    close_episode(right_censored=not observed_termination)
            if known and cell["class"] in ("positive", "negative"):
                if active is not None and active["sign"] == cell["class"] and cell["members"] == active["members"] and cell["edges"] == active["edges"] and previous is not None and abs(cell["start"] - previous["end"]) <= 1e-9:
                    active["end"] = cell["end"]
                    active["window_ids"].append(cell["window_id"])
                else:
                    left_censored = (previous is None or previous["class"] == "unknown" or
                                     previous["interval_index"] != cell["interval_index"] or
                                     abs(cell["start"] - previous["end"]) > 1e-9 or
                                     previous["members"] != cell["members"] or previous["edges"] != cell["edges"])
                    active = {"sign": cell["class"], "start": cell["start"], "end": cell["end"],
                              "left_censored": bool(left_censored), "members": cell["members"],
                              "edges": cell["edges"], "window_ids": [cell["window_id"]]}
            previous = cell
        if active is not None:
            close_episode(right_censored=True)
        all_episodes.extend(episodes)
        for sign, feature_id in (("positive", "sync_positive_episode_duration_mean_s"),
                                 ("negative", "sync_negative_episode_duration_mean_s")):
            complete = [row["observed_duration_s"] for row in episodes
                        if row["sign"] == sign and not row["left_censored"] and not row["right_censored"]]
            all_features.append(_group_feature_row(gid, sid, feature_id,
                                                   float(np.mean(complete)) if complete else None,
                                                   "s", "complete_uncensored_group_sync_episodes",
                                                   "mean_complete_episode_duration", "valid" if complete else "unavailable",
                                                   "" if complete else "no_complete_uncensored_episodes", len(complete),
                                                   float(np.sum(complete)),
                                                   window_id=f"sync:{gid}:{sid}:{metric}:episodes"))
        all_diags.append({"group_id": gid, "session_id": sid, "metric": metric, "basis": basis,
                          "status": "valid" if valid_class_duration > 0 else "unavailable",
                          "reason": "" if valid_class_duration > 0 else "no_common_valid_classified_support",
                          "analysis_duration_s": analysis_duration,
                          "valid_class_duration_s": valid_class_duration, "unknown_duration_s": unknown_duration,
                          "positive_duration_s": positive_duration, "negative_duration_s": negative_duration,
                          "neutral_duration_s": neutral_duration, "window_count": len(rows),
                          "support_segment_count": support_segment_count, "member_set": members, "edge_set": edge_set})

    # Select each pair/window lag using the exact original configured tail-search rule.
    selected = _search_statistic_map(pairs, str(config.get("synchrony", {}).get("tail", "two-sided")))
    lag_rows = []
    for key, selected_row in selected.items():
        gid, sid, wid, pair_id, metric, basis = map(str, key)
        pair_rows = pairs.loc[pairs["group_id"].astype(str).eq(gid) & pairs["session_id"].astype(str).eq(sid)
                              & pairs["window_id"].astype(str).eq(wid) & pairs["pair_id"].astype(str).eq(pair_id)
                              & pairs["metric"].astype(str).eq(metric) & pairs["basis"].astype(str).eq(basis)]
        if pair_rows.empty:
            continue
        chosen = pair_rows.loc[np.isclose(pair_rows["lag_s"].astype(float), float(selected_row["selected_lag_s"]))]
        if chosen.empty:
            continue
        row = chosen.iloc[0]
        cell_bounds = temporal_cell_bounds.get((gid, sid, wid))
        if cell_bounds is None or cell_bounds[1] <= cell_bounds[0]:
            continue
        pair_support = _aligned_pair_support(alignment, row)
        duration = _time_duration(_time_intersection(pair_support, cell_bounds[0], cell_bounds[1]))
        if duration > 0:
            lag_rows.append({"group_id": gid, "session_id": sid, "pair_id": pair_id,
                             "metric": metric, "basis": basis, "window_id": wid,
                             "lag_s": float(selected_row["selected_lag_s"]), "duration_s": duration,
                             "start_s": float(cell_bounds[0]), "end_s": float(cell_bounds[1]),
                             "subject_id_a": str(row.get("subject_id_a", "")),
                             "subject_id_b": str(row.get("subject_id_b", ""))})
    lag_frame = pd.DataFrame(lag_rows)
    if not lag_frame.empty:
        grouping = ["group_id", "session_id", "pair_id", "metric", "basis"]
        pair_features = []
        for group_key, local in lag_frame.groupby(grouping, sort=True, dropna=False):
            gid, sid, pair_id, metric, basis = map(str, group_key)
            local = local.sort_values("start_s", kind="stable")
            lags, weights = local["lag_s"].to_numpy(float), local["duration_s"].to_numpy(float)
            if len(local) < minimum_windows or float(np.sum(weights)) <= 0:
                sd = stable = None
                status, reason = "unavailable", "requires_min_valid_windows_and_positive_duration"
            else:
                median = _weighted_median(lags, weights)
                sd = float(np.sqrt(np.average((lags - np.average(lags, weights=weights)) ** 2, weights=weights)))
                stable = float(np.sum(weights[np.abs(lags - median) <= tolerance]) / np.sum(weights))
                status, reason = "valid", ""
            duration = float(np.sum(weights))
            row = {"group_id": gid, "session_id": sid, "pair_id": pair_id, "metric": metric, "basis": basis,
                   "sync_lag_sd_s": sd, "sync_lag_stability_proportion": stable,
                   "status": status, "reason": reason, "valid_windows": len(local), "duration_s": duration,
                   "subject_id_a": str(local.iloc[0]["subject_id_a"]), "subject_id_b": str(local.iloc[0]["subject_id_b"])}
            pair_features.append(row)
            for fid, value, unit, method in (("sync_lag_sd_s", sd, "s", "duration_weighted_population_sd_of_configured_tail_selected_lag"),
                                             ("sync_lag_stability_proportion", stable, "fraction", "duration_near_weighted_median_within_configured_tolerance")):
                all_features.append(_group_feature_row(gid, sid, fid, value, unit, "pair_window_tail_selected_lag",
                                                       method, status, reason, len(local), duration,
                                                       window_id=f"sync_pair:{pair_id}:{metric}:{fid}"))
        pair_feature_frame = pd.DataFrame(pair_features)
        if not pair_feature_frame.empty:
            for group_key, local in pair_feature_frame.groupby(["group_id", "session_id", "metric", "basis"], sort=True):
                gid, sid, metric, basis = map(str, group_key)
                for fid, unit, method in (("sync_lag_sd_s", "s", "equal_pair_mean_of_pair_duration_weighted_lag_sd"),
                                          ("sync_lag_stability_proportion", "fraction", "equal_pair_mean_of_pair_stability_proportions")):
                    valid = local.loc[local["status"].eq("valid"), fid].dropna().to_numpy(float)
                    all_features.append(_group_feature_row(gid, sid, fid, float(np.mean(valid)) if len(valid) else None,
                                                           unit, "equal_pair_group_summary_of_selected_lag", method,
                                                           "valid" if len(valid) else "unavailable",
                                                           "" if len(valid) else "no_pairs_meet_min_valid_windows",
                                                           len(valid), float(local.loc[local["status"].eq("valid"), "duration_s"].sum()),
                                                           window_id=f"sync_group:{gid}:{sid}:{metric}:{fid}"))
                for index, item in local.iterrows():
                    all_diags.append({"group_id": gid, "session_id": sid, "metric": metric, "basis": basis,
                                      "status": item["status"], "reason": item["reason"],
                                      "analysis_duration_s": np.nan, "valid_class_duration_s": item["duration_s"],
                                      "unknown_duration_s": np.nan, "positive_duration_s": np.nan,
                                      "negative_duration_s": np.nan, "neutral_duration_s": np.nan,
                                      "window_count": int(item["valid_windows"]), "support_segment_count": np.nan,
                                      "member_set": [item["subject_id_a"], item["subject_id_b"]], "edge_set": [item["pair_id"]]})

    # Fixed-graph node strengths use only lag-zero rows and never shrink the expected degree.
    zero = pairs.loc[np.isclose(pd.to_numeric(pairs["lag_s"], errors="coerce"), 0.0)].copy()
    for keys, local in zero.groupby(["group_id", "session_id", "window_id", "metric", "basis"], sort=True):
        gid, sid, wid, metric, basis = map(str, keys)
        group_pairs = pairs.loc[pairs["group_id"].astype(str).eq(gid) &
                                pairs["session_id"].astype(str).eq(sid) &
                                pairs["metric"].astype(str).eq(metric) &
                                pairs["basis"].astype(str).eq(basis)]
        fixed_graph = _fixed_group_graph(membership_rows, group_pairs, gid, sid)
        subjects = fixed_graph["subjects"]
        observed_by_endpoints = {}
        for row in local.drop_duplicates("pair_id").to_dict("records"):
            a, b = row.get("subject_id_a"), row.get("subject_id_b")
            if _present(a) and _present(b):
                endpoint_key = tuple(sorted((str(a), str(b))))
                observed_by_endpoints.setdefault(endpoint_key, []).append(row)
        for subject in subjects:
            local_rows = []
            duplicate_incident = False
            if fixed_graph["known"]:
                expected_incident = [edge for edge in fixed_graph["pairs"] if subject in edge]
                for edge in expected_incident:
                    matches = observed_by_endpoints.get(tuple(edge), [])
                    if len(matches) == 1:
                        local_rows.append(matches[0])
                    elif len(matches) > 1:
                        duplicate_incident = True
            else:
                expected_incident = []
                for endpoint, rows in observed_by_endpoints.items():
                    if subject in endpoint:
                        local_rows.extend(rows)
            expected_degree = len(expected_incident) if fixed_graph["known"] else np.nan
            valid = (fixed_graph["known"] and expected_degree > 0 and not duplicate_incident and
                     len(local_rows) == expected_degree and
                     all(row.get("status") == "valid" and np.isfinite(row.get("r", np.nan)) for row in local_rows))
            correlations = np.asarray([float(row["r"]) for row in local_rows], dtype=float) if valid else np.array([])
            signed = float(np.sum(correlations)) if valid else None
            positive = float(np.sum(np.maximum(correlations, 0.0))) if valid else None
            negative = float(np.sum(np.minimum(correlations, 0.0))) if valid else None
            absolute = float(np.sum(np.abs(correlations))) if valid else None
            normalized = signed / expected_degree if valid and expected_degree else None
            coverage_values = [float(row.get("coverage", 0.0)) for row in local_rows]
            coverage = min(coverage_values) if coverage_values else 0.0
            status = "valid" if valid else "unavailable"
            if valid:
                reason = ""
            elif not fixed_graph["known"]:
                reason = "NA_expected_graph_unknown"
            elif duplicate_incident:
                reason = "NA_duplicate_fixed_incident_edge"
            elif len(local_rows) < expected_degree:
                reason = "NA_missing_fixed_incident_edge"
            else:
                reason = "NA_fixed_incident_edge_unavailable"
            node = {"group_id": gid, "session_id": sid, "subject_id": subject, "window_id": wid,
                    "metric": metric, "basis": basis, "signed": signed, "positive": positive,
                    "negative": negative, "absolute": absolute, "normalized_signed": normalized,
                    "effective_degree": len(local_rows), "expected_degree": expected_degree,
                    "coverage": coverage, "status": status, "reason": reason}
            all_nodes.append(node)
            for fid, field in (("node_strength_signed", "signed"), ("node_strength_positive", "positive"),
                               ("node_strength_negative", "negative"), ("node_strength_absolute", "absolute"),
                               ("node_strength_normalized_signed", "normalized_signed")):
                all_features.append(_group_feature_row(gid, sid, fid, node[field], "dimensionless",
                                                       "lag0_fixed_graph", f"fixed_graph_{field}_strength",
                                                       status, reason, node["effective_degree"], 0.0,
                                                       subject_id=subject,
                                                       window_id=f"node:{gid}:{sid}:{wid}:{metric}"))

    features = pd.DataFrame(all_features, columns=feature_cols)
    episodes_df = pd.DataFrame(all_episodes, columns=_SYNC_EPISODE_COLUMNS)
    nodes_df = pd.DataFrame(all_nodes, columns=_NODE_STRENGTH_COLUMNS)
    diagnostics_df = pd.DataFrame(all_diags, columns=_GROUP_TEMPORAL_DIAGNOSTIC_COLUMNS)
    return features, episodes_df, nodes_df, diagnostics_df


def _wilson_interval(extreme, total, z=1.959963984540054):
    if total <= 0:
        return None, None
    p = extreme / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt((p * (1 - p) / total) + z * z / (4 * total * total)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def _circular_shift_bundle(aligned, shifts, max_lag_samples):
    result = {**aligned, "time_s": np.array(aligned["time_s"], copy=True),
              "signals": {}, "eligible": {}, "metadata": copy.deepcopy(aligned["metadata"])}
    n = len(result["time_s"])
    for key, metadata in result["metadata"].items():
        unit = _unit_key(metadata)
        shift = int(shifts.get(unit, 0))
        values = np.asarray(aligned["signals"][key], dtype=float)
        mask = np.asarray(aligned["eligible"][key], dtype=bool)
        moved_values = np.roll(values, shift)
        moved_mask = np.roll(mask, shift)
        seam = shift % n if n else 0
        if n and shift % n:
            for index in range(seam - max_lag_samples - 1, seam + max_lag_samples + 1):
                moved_mask[index % n] = False
        moved_mask &= np.isfinite(moved_values)
        result["signals"][key] = moved_values
        result["eligible"][key] = moved_mask
    return result


def _support_signature(values):
    keys = ["group_id", "session_id", "window_id", "pair_id", "metric", "basis", "lag_s"]
    if values.empty:
        return {}
    signature = {}
    for _, row in values.iterrows():
        key = tuple(row[name] for name in keys)
        signature[key] = (int(row["valid_n"]), float(row["coverage"]), str(row["status"]))
    return signature


def _unit_key(metadata):
    return str(metadata.get("session_id", "")), str(metadata.get("subject_id", ""))


def _required_index_masks(aligned, edges, windows, lag_indices):
    time = np.asarray(aligned.get("time_s", []), dtype=float)
    required = {}
    units = set()
    for edge in edges:
        ka, kb = edge["record_key_a"], edge["record_key_b"]
        meta_a, meta_b = aligned.get("metadata", {}).get(ka, {}), aligned.get("metadata", {}).get(kb, {})
        unit_a, unit_b = _unit_key(meta_a), _unit_key(meta_b)
        if not all(unit_a) or not all(unit_b):
            return {}, set(), "participant_metadata_missing_or_single_member"
        units.update((unit_a, unit_b))
        for window in windows:
            selected = np.flatnonzero((time >= window["start_s"] - 1e-12)
                                      & (time < window["end_s"] - 1e-12))
            for lag in lag_indices:
                a_slice, b_slice = _lag_slices(len(selected), int(lag))
                ia, ib = selected[a_slice], selected[b_slice]
                required.setdefault(unit_a, {}).setdefault(ka, set()).update(map(int, ia))
                required.setdefault(unit_b, {}).setdefault(kb, set()).update(map(int, ib))
    return required, units, ""


def _preserving_shift_candidates(mask, required):
    """Exact cyclic-mask equality at required indices using circular FFT correlation."""
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if not n:
        return np.array([], dtype=int)
    required = np.asarray(required, dtype=bool)
    if required.shape != mask.shape:
        return np.array([], dtype=int)
    if not np.any(required):
        return np.arange(n, dtype=int)
    observed_one = required & mask
    observed_zero = required & ~mask
    mask_fft = np.fft.fft(mask.astype(float))
    moved_one_overlap = np.fft.ifft(np.conj(mask_fft)
                                    * np.fft.fft(observed_one.astype(float))).real
    moved_zero_overlap = np.fft.ifft(np.conj(mask_fft)
                                     * np.fft.fft(observed_zero.astype(float))).real
    mismatch = int(np.count_nonzero(observed_one)) - moved_one_overlap + moved_zero_overlap
    return np.flatnonzero(np.abs(mismatch) < 0.25).astype(int)


def _seam_safe_shifts(required, n, max_lag_samples):
    required = np.asarray(required, dtype=bool)
    if n <= 0:
        return np.array([], dtype=int)
    if not np.any(required):
        return np.arange(n, dtype=int)
    width = 2 * int(max_lag_samples) + 2
    safe = np.ones(n, dtype=bool)
    if width >= n:
        safe[1:] = False
        return np.flatnonzero(safe)
    extended = np.tile(required.astype(np.int64), 3)
    prefix = np.concatenate(([0], np.cumsum(extended)))
    seams = np.arange(n, dtype=np.int64)
    starts = n + seams - int(max_lag_samples) - 1
    counts = prefix[starts + width] - prefix[starts]
    safe &= counts == 0
    safe[0] = True  # A zero shift creates no wrap seam.
    return np.flatnonzero(safe)


def _shift_geometry(aligned, edges, windows, lag_indices, max_lag_samples, cancel_callback=None):
    time = np.asarray(aligned.get("time_s", []), dtype=float)
    n = len(time)
    required, units, reason = _required_index_masks(aligned, edges, windows, lag_indices)
    if reason:
        return {}, units, reason
    candidates, masks = {}, aligned.get("eligible", {})
    signals = aligned.get("signals", {})
    for unit in sorted(units):
        _cancel(cancel_callback)
        keep = np.ones(n, dtype=bool)
        channel_masks = required.get(unit, {})
        unit_required = np.zeros(n, dtype=bool)
        for key, indices in channel_masks.items():
            local_required = np.zeros(n, dtype=bool)
            if indices:
                local_required[np.fromiter(sorted(indices), dtype=int)] = True
            unit_required |= local_required
            channel_mask = np.asarray(masks.get(key, []), dtype=bool)
            channel_values = np.asarray(signals.get(key, []), dtype=float)
            if channel_mask.shape != (n,) or channel_values.shape != (n,):
                return {}, units, "sync_aligned_shape"
            channel_mask &= np.isfinite(channel_values)
            preserving = np.zeros(n, dtype=bool)
            preserving[_preserving_shift_candidates(channel_mask, local_required)] = True
            keep &= preserving
        safe = np.zeros(n, dtype=bool)
        safe[_seam_safe_shifts(unit_required, n, max_lag_samples)] = True
        keep &= safe
        candidates[unit] = np.flatnonzero(keep).astype(int)
        if not len(candidates[unit]):
            return candidates, units, "shift_support_infeasible"
    return candidates, units, ""


def _joint_shift_valid(shifts, constraints, n, minimum_shift):
    for left, right in constraints:
        a, b = int(shifts[left]), int(shifts[right])
        distance = abs(a - b)
        if min(distance, n - distance) < minimum_shift:
            return False
    return True


def _joint_shift_preflight(candidates, edges, aligned, n, minimum_shift, rng, *, exact_limit=100000,
                           search_limit=10000, cancel_callback=None):
    constraints = set()
    for edge in edges:
        a = _unit_key(aligned.get("metadata", {}).get(edge["record_key_a"], {}))
        b = _unit_key(aligned.get("metadata", {}).get(edge["record_key_b"], {}))
        if a == b or a not in candidates or b not in candidates:
            return "joint_shift_infeasible", set()
        constraints.add(tuple(sorted((a, b))))
    units = sorted(candidates)
    space_size = math.prod(len(candidates[unit]) for unit in units)
    if space_size <= exact_limit:
        feasible_count = 0
        for values in itertools.product(*(candidates[unit] for unit in units)):
            _cancel(cancel_callback)
            assignment = dict(zip(units, values))
            if _joint_shift_valid(assignment, constraints, n, minimum_shift):
                feasible_count += 1
        return ("feasible" if feasible_count else "joint_shift_infeasible"), constraints, feasible_count, space_size
    for proposal in range(search_limit):
        _cancel(cancel_callback)
        assignment = {unit: int(rng.choice(candidates[unit])) for unit in units}
        if _joint_shift_valid(assignment, constraints, n, minimum_shift):
            return "feasible", constraints, None, proposal + 1
    return "feasibility_unresolved", constraints, None, search_limit


_NULL_COLUMNS = ["group_id", "session_id", "window_id", "pair_id", "metric", "basis", "method", "tail",
                 "observed_statistic", "null_mean_statistic", "null_sd_statistic", "statistic",
                 "selected_r", "selected_lag_s", "observed_max_abs_r", "null_mean_max_abs_r",
                 "null_sd_max_abs_r", "requested_replicates", "accepted_replicates", "attempted_replicates",
                 "unique_worlds", "finite_support_size", "resolution", "status", "reason",
                 "analysis_intervals_s", "analysis_context_s", "candidate_counts", "provenance_json",
                 "preflight_status", "preflight_proposals", "proposal_count",
                 "sampling_with_replacement", "support_policy_version"]
_TAIL_COLUMNS = ["group_id", "session_id", "window_id", "pair_id", "metric", "basis", "method", "tail",
                 "surrogate_tail_probability", "mc_interval_low", "mc_interval_high", "extreme_count",
                 "accepted_replicates", "requested_replicates", "attempted_replicates", "unique_worlds",
                 "finite_support_size", "resolution", "status", "qualification", "statistic",
                 "observed_statistic", "selected_r", "selected_lag_s", "provenance_json",
                 "preflight_status", "preflight_proposals", "proposal_count",
                 "sampling_with_replacement", "support_policy_version"]


def _null_status(reason, requested, method, *, status="unavailable", **details):
    support_policy = ("complete-block-matched-donor-v1" if method == "pseudodyad"
                      else "participant-required-mask-seam-v1" if method == "circular_shift" else "none-v1")
    row = {"status": status, "reason": reason, "method": method,
           "requested_replicates": int(requested), "accepted_replicates": 0,
           "attempted_replicates": 0, "unique_worlds": 0,
           "finite_support_size": None, "resolution": None,
           "observed_statistic": np.nan, "null_mean_statistic": np.nan,
           "null_sd_statistic": np.nan, "statistic": "", "tail": "",
           "selected_r": np.nan, "selected_lag_s": np.nan,
           "observed_max_abs_r": np.nan, "null_mean_max_abs_r": np.nan,
           "null_sd_max_abs_r": np.nan, "analysis_intervals_s": [],
           "analysis_context_s": np.nan, "candidate_counts": "{}", "provenance_json": "{}",
           "preflight_status": "not_run", "preflight_proposals": 0, "proposal_count": 0,
           "sampling_with_replacement": method != "none" and int(requested) > 0,
           "support_policy_version": support_policy}
    row.update(details)
    return pd.DataFrame([row], columns=_NULL_COLUMNS), pd.DataFrame(columns=_TAIL_COLUMNS)


def _search_statistic_map(values, tail):
    grouping = ["group_id", "session_id", "window_id", "pair_id", "metric", "basis"]
    output = {}
    if values.empty:
        return output
    for key, group in values.groupby(grouping, dropna=False, sort=True):
        valid = group.loc[(group["status"] == "valid") & np.isfinite(group["r"])]
        if valid.empty:
            continue
        valid = valid.sort_values("lag_s", kind="stable")
        correlations = valid["r"].to_numpy(dtype=float)
        if tail == "two-sided":
            index = int(np.argmax(np.abs(correlations)))
            statistic = float(abs(correlations[index]))
        elif tail == "greater":
            index = int(np.argmax(correlations))
            statistic = float(correlations[index])
        elif tail == "less":
            index = int(np.argmin(correlations))
            statistic = float(correlations[index])
        else:
            fail("sync_tail_config", f"不支持tail：{tail}。", category="configuration", stage="synchrony")
        output[key] = {"statistic": statistic, "selected_r": float(correlations[index]),
                       "selected_lag_s": float(valid.iloc[index]["lag_s"])}
    return output


def _shift_provenance(aligned, edges, shifts):
    rows = []
    for edge in edges:
        endpoints = []
        for name in ("record_key_a", "record_key_b"):
            key = edge[name]
            metadata = aligned.get("metadata", {}).get(key, {})
            endpoints.append({"target_record_key": key, "target_subject_id": metadata.get("subject_id"),
                              "shift_samples": int(shifts.get(_unit_key(metadata), 0))})
        rows.append({"pair_id": edge.get("pair_id"), "endpoints": endpoints})
    return rows


def _candidate_summary(candidates):
    return json.dumps({"|".join(unit): int(len(values)) for unit, values in sorted(candidates.items())},
                      sort_keys=True, separators=(",", ":"))


def _finalize_null(accepted, observed, requested, attempted, method, tail,
                   intervals, context, *, finite_support_size=None, candidate_counts="{}",
                   preflight_reason="", preflight_status="feasible",
                   preflight_proposals=0, proposal_count=0,
                   support_policy_version="participant-required-mask-seam-v1"):
    grouping = ["group_id", "session_id", "window_id", "pair_id", "metric", "basis"]
    enough = len(accepted) == requested
    summary_rows, tail_rows = [], []
    unique_worlds = len({world["world_id"] for world in accepted if world.get("world_id")})
    for key in sorted(observed, key=str):
        group_id, session_id, window_id, pair_id, metric, basis = key
        obs = observed[key]
        null_values = np.asarray([rep["statistics"][key]["statistic"] for rep in accepted
                                  if key in rep["statistics"]], dtype=float)
        if len(null_values) != len(accepted):
            return _null_status("surrogate_statistic_unavailable", requested, method,
                                analysis_intervals_s=intervals, analysis_context_s=context,
                                candidate_counts=candidate_counts), pd.DataFrame(columns=_TAIL_COLUMNS)
        if enough:
            if tail == "less":
                extreme = int(np.count_nonzero(null_values <= obs["statistic"] + 1e-12))
            else:
                extreme = int(np.count_nonzero(null_values >= obs["statistic"] - 1e-12))
            probability = float((1 + extreme) / (len(null_values) + 1))
            lo, hi = _wilson_interval(extreme, len(null_values))
            tail_status, reason = "experimental_uncalibrated", "surrogate_tail_probability_not_calibrated"
        else:
            extreme, probability, lo, hi = 0, np.nan, np.nan, np.nan
            tail_status = "insufficient_accepted_replicates"
            reason = preflight_reason or "max_attempts_reached_before_requested_B"
        provenance = []
        for replicate in accepted:
            item = replicate.get("provenance", {}).get(pair_id)
            if item is not None:
                provenance.append(item)
        provenance_json = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        null_mean = float(np.mean(null_values)) if len(null_values) else np.nan
        null_sd = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else np.nan
        common = {"group_id": group_id, "session_id": session_id, "window_id": window_id,
                  "pair_id": pair_id, "metric": metric, "basis": basis, "method": method,
                  "tail": tail, "observed_statistic": obs["statistic"],
                  "null_mean_statistic": null_mean, "null_sd_statistic": null_sd,
                  "statistic": f"{tail}_lag_search", "selected_r": obs["selected_r"],
                  "selected_lag_s": obs["selected_lag_s"],
                  "observed_max_abs_r": obs["statistic"] if tail == "two-sided" else np.nan,
                  "null_mean_max_abs_r": null_mean if tail == "two-sided" else np.nan,
                  "null_sd_max_abs_r": null_sd if tail == "two-sided" else np.nan,
                  "requested_replicates": requested, "accepted_replicates": len(accepted),
                  "attempted_replicates": attempted, "unique_worlds": unique_worlds,
                  "finite_support_size": finite_support_size,
                  "resolution": 1.0 / (len(null_values) + 1) if len(null_values) else np.nan,
                  "status": tail_status, "reason": reason, "analysis_intervals_s": intervals,
                  "analysis_context_s": context, "candidate_counts": candidate_counts,
                  "provenance_json": provenance_json, "preflight_status": preflight_status,
                  "preflight_proposals": preflight_proposals, "proposal_count": proposal_count,
                  "sampling_with_replacement": True,
                  "support_policy_version": support_policy_version}
        summary_rows.append(common)
        tail_rows.append({**{k: common[k] for k in ("group_id", "session_id", "window_id", "pair_id", "metric", "basis",
                                                      "method", "tail", "requested_replicates", "accepted_replicates",
                                                      "attempted_replicates", "unique_worlds", "finite_support_size",
                                                      "resolution", "status", "statistic", "observed_statistic",
                                                      "selected_r", "selected_lag_s", "provenance_json",
                                                      "preflight_status", "preflight_proposals",
                                                      "proposal_count", "sampling_with_replacement",
                                                      "support_policy_version")},
                          "surrogate_tail_probability": probability, "mc_interval_low": lo,
                          "mc_interval_high": hi, "extreme_count": extreme,
                          "qualification": "experimental_uncalibrated"})
    return pd.DataFrame(summary_rows, columns=_NULL_COLUMNS), pd.DataFrame(tail_rows, columns=_TAIL_COLUMNS)


def estimate_sync_null(aligned, pairs, config, seed, *, cancel_callback=None):
    """Dispatch the declared null method and keep all surrogate probabilities experimental."""
    config = config or {}
    sync = config.get("synchrony", {})
    method = str(sync.get("null_method", "circular_shift"))
    requested = int(sync.get("null_replicates", 0))
    max_attempts = int(sync.get("max_attempts", 1))
    tail = str(sync.get("tail", "two-sided"))
    time = np.asarray(aligned.get("time_s", []), dtype=float)
    intervals, context, interval_resolution = _analysis_interval_resolution(time, config)
    if method == "none":
        return _null_status("null_method_none", 0, method, status="not_requested",
                            analysis_intervals_s=intervals, analysis_context_s=context)
    if not len(time) or aligned.get("alignment_report", {}).get("formal_synchrony_allowed") is False:
        return _null_status("clock_alignment_unverified", requested, method,
                            analysis_intervals_s=intervals, analysis_context_s=context)
    if interval_resolution["status"] != "available":
        return _null_status(interval_resolution["reason"], requested, method,
                            analysis_intervals_s=interval_resolution.get("intervals_s", []),
                            analysis_context_s=context)
    edges, _, _ = _normalize_pairs(pairs, aligned, max_pairs=int(config.get("compute", {}).get("max_pairs", 10000)))
    if not requested:
        return _null_status("null_replicates_zero", 0, method, status="not_requested",
                            analysis_intervals_s=intervals, analysis_context_s=context)
    if not edges:
        return _null_status("no_observed_edges", requested, method,
                            analysis_intervals_s=intervals, analysis_context_s=context)
    windows, window_resolution = _window_resolution(time, config)
    if not windows:
        return _null_status(window_resolution["reason"], requested, method,
                            analysis_intervals_s=intervals, analysis_context_s=context)
    lag_indices, _, rate = _effective_lag_indices(time, config)
    metric = str(sync.get("metric", "pearson")).lower()
    metric_count = 2 if metric in ("both", "pearson,spearman") else 1
    eval_per_replicate = len(edges) * len(windows) * len(lag_indices) * metric_count
    eval_budget = int(config.get("compute", {}).get("max_null_evaluations", 10000000))
    evaluation_bound = max_attempts * eval_per_replicate
    rng = np.random.default_rng(int(seed))
    n = len(time)
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
    max_lag_samples = int(math.ceil(float(sync.get("max_lag_s", 0.0)) / dt)) if dt > 0 else 0
    candidate_counts, finite_support_size = "{}", None
    preflight_status, preflight_proposals = "not_applicable", 0
    circular_candidates, circular_units, circular_constraints = None, None, None
    minimum_shift = 0
    if method == "circular_shift":
        minimum_shift = int(math.ceil(float(sync.get("min_shift_s", 0.0)) / dt - 1e-12)) if dt > 0 else n
        if n < 2 or dt <= 0:
            return _null_status("joint_shift_infeasible", requested, method,
                                analysis_intervals_s=intervals, analysis_context_s=context)
        circular_candidates, circular_units, geometry_reason = _shift_geometry(
            aligned, edges, windows, lag_indices, max_lag_samples, cancel_callback=cancel_callback)
        candidate_counts = _candidate_summary(circular_candidates)
        if geometry_reason:
            return _null_status(geometry_reason, requested, method,
                                analysis_intervals_s=intervals, analysis_context_s=context,
                                candidate_counts=candidate_counts,
                                preflight_status=geometry_reason, preflight_proposals=0)
        if minimum_shift > n // 2:
            return _null_status("joint_shift_infeasible", requested, method,
                                analysis_intervals_s=intervals, analysis_context_s=context,
                                candidate_counts=candidate_counts,
                                preflight_status="joint_shift_infeasible", preflight_proposals=0)
        preflight_status, circular_constraints, finite_support_size, preflight_proposals = _joint_shift_preflight(
            circular_candidates, edges, aligned, n, minimum_shift, rng,
            cancel_callback=cancel_callback)
        if preflight_status != "feasible":
            return _null_status(preflight_status, requested, method,
                                analysis_intervals_s=intervals, analysis_context_s=context,
                                candidate_counts=candidate_counts, finite_support_size=finite_support_size,
                                preflight_status=preflight_status, preflight_proposals=preflight_proposals)
    if evaluation_bound > eval_budget:
        return _null_status("max_null_evaluations_exceeded", requested, method,
                            analysis_intervals_s=intervals, analysis_context_s=context,
                            evaluation_budget=eval_budget, evaluation_bound=evaluation_bound,
                            candidate_counts=candidate_counts, finite_support_size=finite_support_size,
                            preflight_status=preflight_status, preflight_proposals=preflight_proposals)
    observed_values, _ = compute_pair_synchrony(aligned, edges, config, cancel_callback=cancel_callback)
    if observed_values.empty or not observed_values["status"].eq("valid").any():
        return _null_status("no_valid_observed_synchrony", requested, method,
                            analysis_intervals_s=intervals, analysis_context_s=context)
    observed_signature = _support_signature(observed_values)
    observed = _search_statistic_map(observed_values, tail)
    if not observed:
        return _null_status("no_valid_observed_search_statistic", requested, method,
                            analysis_intervals_s=intervals, analysis_context_s=context)
    accepted, attempted, proposal_count = [], 0, 0

    if method == "circular_shift":
        ordered_units = sorted(circular_units)
        for _ in range(max_attempts):
            _cancel(cancel_callback)
            if len(accepted) >= requested:
                break
            attempted += 1
            proposal_count += 1
            shifts = {unit: int(rng.choice(circular_candidates[unit])) for unit in ordered_units}
            if not _joint_shift_valid(shifts, circular_constraints, n, minimum_shift):
                continue
            shifted = _circular_shift_bundle(aligned, shifts, max_lag_samples)
            null_values, _ = compute_pair_synchrony(shifted, edges, config, cancel_callback=cancel_callback)
            signature = _support_signature(null_values)
            if signature != observed_signature:
                reason = "surrogate_support_changed_after_preflight"
                if signature.keys() == observed_signature.keys() and all(
                        signature[key][:2] == observed_signature[key][:2] for key in signature):
                    reason = "surrogate_statistic_status_changed_under_null"
                return _null_status(reason, requested, method, attempted_replicates=attempted,
                                    accepted_replicates=len(accepted), unique_worlds=len({x["world_id"] for x in accepted}),
                                    proposal_count=proposal_count, analysis_intervals_s=intervals,
                                    analysis_context_s=context, candidate_counts=candidate_counts,
                                    finite_support_size=finite_support_size)
            replicate_stats = _search_statistic_map(null_values, tail)
            if replicate_stats.keys() != observed.keys():
                return _null_status("surrogate_statistic_unavailable", requested, method,
                                    attempted_replicates=attempted, accepted_replicates=len(accepted),
                                    unique_worlds=len({x["world_id"] for x in accepted}), proposal_count=proposal_count,
                                    analysis_intervals_s=intervals, analysis_context_s=context,
                                    candidate_counts=candidate_counts, finite_support_size=finite_support_size)
            world_id = json.dumps({"|".join(unit): shift for unit, shift in sorted(shifts.items())},
                                  sort_keys=True, separators=(",", ":"))
            provenance = {row["pair_id"]: row["endpoints"] for row in _shift_provenance(aligned, edges, shifts)}
            accepted.append({"statistics": replicate_stats, "world_id": world_id, "provenance": provenance})
    elif method == "pseudodyad":
        if sync.get("pseudodyad_exchangeability") is not True:
            return _null_status("pseudodyad_exchangeability_not_asserted", requested, method,
                                analysis_intervals_s=intervals, analysis_context_s=context)
        # The complete-block implementation below returns only matched donor worlds;
        # it never falls through to circular shifting.
        setup = _pseudodyad_setup(aligned, config, edges, windows, lag_indices, cancel_callback=cancel_callback)
        if setup.get("status") != "available":
            return _null_status(setup.get("reason", "pseudodyad_unavailable"), requested, method,
                                analysis_intervals_s=intervals, analysis_context_s=context,
                                candidate_counts=json.dumps(setup.get("diagnostics", {}), sort_keys=True, default=str))
        world_support = setup["world_support"]
        finite_support_size = setup.get("finite_support_size")
        candidate_counts = json.dumps(setup.get("diagnostics", {}), sort_keys=True, default=str,
                                      separators=(",", ":"))
        if finite_support_size == 0:
            return _null_status("pseudodyad_assignment_infeasible", requested, method,
                                analysis_intervals_s=intervals, analysis_context_s=context,
                                finite_support_size=0, candidate_counts=candidate_counts)
        preflight_status = "feasible"
        if finite_support_size is not None:
            probe = _first_pseudodyad_world(world_support)
            preflight_proposals = 0
        else:
            probe, preflight_proposals = _sample_pseudodyad_world(world_support,
                np.random.default_rng(int(seed) ^ 0x5EED), search_limit=10000,
                cancel_callback=cancel_callback, return_attempts=True)
        if probe is None:
            preflight_status = "feasibility_unresolved"
            return _null_status("pseudodyad_assignment_feasibility_unresolved", requested, method,
                                analysis_intervals_s=intervals, analysis_context_s=context,
                                finite_support_size=finite_support_size, candidate_counts=candidate_counts,
                                preflight_status=preflight_status, preflight_proposals=preflight_proposals,
                                support_policy_version="complete-block-matched-donor-v1")
        for _ in range(max_attempts):
            _cancel(cancel_callback)
            if len(accepted) >= requested:
                break
            attempted += 1
            world, world_proposals = _sample_pseudodyad_world(world_support, rng, search_limit=100,
                                                              cancel_callback=cancel_callback,
                                                              return_attempts=True)
            proposal_count += world_proposals
            if world is None:
                continue
            pseudo_aligned, provenance = _apply_pseudodyad_world(aligned, setup, world)
            null_values, _ = compute_pair_synchrony(pseudo_aligned, edges, config,
                                                    cancel_callback=cancel_callback)
            if _support_signature(null_values) != observed_signature:
                return _null_status("pseudodyad_support_or_status_changed", requested, method,
                                    attempted_replicates=attempted, accepted_replicates=len(accepted),
                                    unique_worlds=len({x["world_id"] for x in accepted}), proposal_count=proposal_count,
                                    analysis_intervals_s=intervals, analysis_context_s=context,
                                    candidate_counts=candidate_counts, finite_support_size=finite_support_size)
            replicate_stats = _search_statistic_map(null_values, tail)
            if replicate_stats.keys() != observed.keys():
                return _null_status("pseudodyad_statistic_unavailable", requested, method,
                                    attempted_replicates=attempted, accepted_replicates=len(accepted),
                                    unique_worlds=len({x["world_id"] for x in accepted}), proposal_count=proposal_count,
                                    analysis_intervals_s=intervals, analysis_context_s=context,
                                    candidate_counts=candidate_counts, finite_support_size=finite_support_size)
            world_id = _pseudodyad_world_id(world)
            accepted.append({"statistics": replicate_stats, "world_id": world_id,
                             "provenance": provenance})
    else:
        return _null_status("unsupported_null_method", requested, method,
                            analysis_intervals_s=intervals, analysis_context_s=context)

    return _finalize_null(accepted, observed, requested, attempted, method, tail,
                          intervals, context, finite_support_size=finite_support_size,
                          candidate_counts=candidate_counts,
                          preflight_reason="max_attempts_reached_before_requested_B",
                          preflight_status=preflight_status, preflight_proposals=preflight_proposals,
                          proposal_count=proposal_count,
                          support_policy_version=("complete-block-matched-donor-v1" if method == "pseudodyad"
                                                  else "participant-required-mask-seam-v1"))


def _json_token(value):
    if _missing(value):
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


def _task_time_declaration(row):
    basis = str(row.get("time_basis", row.get("task_time_basis", ""))).strip().lower()
    declared_relative = row.get("task_relative")
    if (basis in ("task_relative", "task-relative", "task relative")
            or isinstance(declared_relative, (bool, np.bool_)) and bool(declared_relative)):
        return "task_relative", 0.0
    origin = row.get("task_time_origin_s")
    if _present(origin):
        try:
            origin = float(origin)
        except (TypeError, ValueError):
            return "invalid", np.nan
        if np.isfinite(origin):
            return "origin", origin
    return "missing", np.nan


def _pseudodyad_setup(aligned, config, edges, windows, lag_indices, *, cancel_callback=None,
                      exact_world_limit=100000):
    sync = config.get("synchrony", {})
    nodes, member_report = _membership_nodes(aligned)
    fields = list(dict.fromkeys(["task", "stimulus"] + list(sync.get("matching_fields", []))
                                + list(sync.get("pseudodyad_strata", []))))
    timeline_candidates = ("timeline_id", "stimulus_timeline_id", "timeline_hash", "stimulus_schedule_id")
    enriched = []
    for node in nodes:
        row = dict(node)
        for key, value in aligned.get("metadata", {}).get(node["record_key"], {}).items():
            if not _present(row.get(key)) and _present(value):
                row[key] = value
        enriched.append(row)
    available = set().union(*(row.keys() for row in enriched)) if enriched else set()
    timeline_field = next((field for field in timeline_candidates if field in available), None)
    slot_field = "member_slot" if "member_slot" in available else "role" if "role" in available else None
    missing = [field for field in fields if field not in available]
    if timeline_field is None:
        missing.append("timeline_id")
    if slot_field is None:
        missing.append("role_or_member_slot")
    if missing:
        return {"status": "unavailable", "reason": "pseudodyad_matching_metadata_missing",
                "diagnostics": {"missing_fields": sorted(set(missing)),
                                "unresolved_membership": member_report.get("unresolved_rows", [])}}
    if not nodes:
        return {"status": "unavailable", "reason": "pseudodyad_membership_missing",
                "diagnostics": {"unresolved_membership": member_report.get("unresolved_rows", [])}}
    block_edges, _, block_diag = _fixed_edges(aligned, max_pairs=int(config.get("compute", {}).get("max_pairs", 10000)))
    invalid = block_diag.get("invalid_groups", [])
    original_partner_pairs = {frozenset((str(edge["subject_id_a"]), str(edge["subject_id_b"])))
                              for edge in block_edges}
    target_blocks = {(str(edge["group_id"]), str(edge["session_id"])) for edge in block_edges}
    if not block_edges or invalid or any((str(edge["group_id"]), str(edge["session_id"])) not in target_blocks for edge in edges):
        return {"status": "unavailable", "reason": "pseudodyad_target_member_design_incomplete",
                "diagnostics": {"invalid_groups": invalid, "target_block_count": len(target_blocks)}}
    time_modes = {}
    blocks = {}
    strata_fields = fields + [timeline_field]
    duplicate_keys = set()
    for row in enriched:
        _cancel(cancel_callback)
        block = (str(row["group_id"]), str(row["session_id"]))
        values = [_json_token(row.get(field)) for field in strata_fields]
        if any(value is None for value in values):
            return {"status": "unavailable", "reason": "pseudodyad_matching_value_missing",
                    "diagnostics": {"record_key": row["record_key"], "strata_fields": strata_fields}}
        stratum = tuple(values)
        mode, origin = _task_time_declaration(row)
        if mode == "missing" or mode == "invalid":
            return {"status": "unavailable", "reason": "pseudodyad_task_time_basis_missing_or_invalid",
                    "diagnostics": {"record_key": row["record_key"], "task_time_basis": mode}}
        time_modes.setdefault(block, set()).add(mode)
        slot = (str(row.get(slot_field)), str(row.get("channel", "eda")))
        if not slot[0] or slot[0] == "nan":
            return {"status": "unavailable", "reason": "pseudodyad_member_slot_missing",
                    "diagnostics": {"record_key": row["record_key"]}}
        blocks.setdefault(stratum, {}).setdefault(block, {})
        if slot in blocks[stratum][block]:
            duplicate_keys.add((block, slot))
            continue
        blocks[stratum][block][slot] = {**row, "task_time_mode": mode, "task_time_origin_s": origin}
    if duplicate_keys:
        return {"status": "unavailable", "reason": "pseudodyad_member_slot_not_unique",
                "diagnostics": {"duplicates": sorted((list(block), list(slot)) for block, slot in duplicate_keys)}}
    if any(len(modes) != 1 for modes in time_modes.values()):
        return {"status": "unavailable", "reason": "pseudodyad_task_time_basis_inconsistent",
                "diagnostics": {"blocks": {str(key): sorted(value) for key, value in time_modes.items()}}}

    required_by_key = {}
    time = np.asarray(aligned.get("time_s", []), dtype=float)
    for edge in edges:
        for key_name in ("record_key_a", "record_key_b"):
            key = edge[key_name]
            required = required_by_key.setdefault(key, set())
            for window in windows:
                selected = np.flatnonzero((time >= window["start_s"] - 1e-12)
                                          & (time < window["end_s"] - 1e-12))
                for lag in lag_indices:
                    left, right = _lag_slices(len(selected), int(lag))
                    required.update(map(int, selected[left]))
                    required.update(map(int, selected[right]))

    support_list, diagnostics = [], {"timeline_field": timeline_field, "slot_field": slot_field,
                                    "strata_fields": strata_fields, "stratum_block_counts": {},
                                    "stratum_slot_counts": {}, "enumerated_world_counts": {}}
    for stratum, block_map in sorted(blocks.items(), key=lambda item: item[0]):
        block_ids = sorted(block_map)
        if not any(block in target_blocks for block in block_ids):
            continue
        slot_sets = [set(block_map[block]) for block in block_ids]
        if not slot_sets or any(slots != slot_sets[0] for slots in slot_sets[1:]):
            return {"status": "unavailable", "reason": "pseudodyad_incompatible_complete_slot_sets",
                    "diagnostics": {"stratum": stratum, "blocks": [list(x) for x in block_ids]}}
        slots = sorted(slot_sets[0])
        if len(block_ids) < 2 or len(slots) < 2:
            return {"status": "unavailable", "reason": "pseudodyad_independent_blocks_insufficient",
                    "diagnostics": {"stratum": stratum, "block_count": len(block_ids), "slot_count": len(slots)}}
        mode = next(iter(time_modes[block_ids[0]]))
        if any(next(iter(time_modes[block])) != mode for block in block_ids):
            return {"status": "unavailable", "reason": "pseudodyad_task_time_basis_inconsistent",
                    "diagnostics": {"stratum": stratum}}
        allowed_by_slot_target = {slot: {} for slot in slots}
        block_subjects = {block: {str(node["subject_id"]) for node in block_map[block].values()}
                          for block in block_ids}
        for target in block_ids:
            for slot in slots:
                target_node = block_map[target][slot]
                allowed = []
                for donor in block_ids:
                    if donor == target or block_subjects[donor] & block_subjects[target]:
                        continue
                    donor_node = block_map[donor][slot]
                    target_key, donor_key = target_node["record_key"], donor_node["record_key"]
                    if not _pseudodyad_support_compatible(aligned, target_node, donor_node,
                                                           required_by_key.get(target_key, set())):
                        continue
                    allowed.append(donor)
                allowed_by_slot_target[slot][target] = set(allowed)
        factorial_size = math.factorial(len(block_ids))
        enumerate_space = factorial_size ** len(slots) <= exact_world_limit
        permutation_lists = {}
        if enumerate_space:
            for slot in slots:
                candidates = []
                for permutation in itertools.permutations(block_ids):
                    _cancel(cancel_callback)
                    if all(permutation[index] in allowed_by_slot_target[slot][target]
                           for index, target in enumerate(block_ids)):
                        candidates.append(permutation)
                permutation_lists[slot] = candidates
                if not candidates:
                    return {"status": "unavailable", "reason": "pseudodyad_assignment_infeasible",
                            "diagnostics": {"stratum": stratum, "slot": list(slot)}}
        worlds = None
        world_count = None
        if enumerate_space:
            permutation_product = math.prod(len(permutation_lists[slot]) for slot in slots)
            if permutation_product <= exact_world_limit:
                worlds = []
                for permutations in itertools.product(*(permutation_lists[slot] for slot in slots)):
                    _cancel(cancel_callback)
                    assignment = {target: {slot: permutations[slot_index][target_index]
                                           for slot_index, slot in enumerate(slots)}
                                  for target_index, target in enumerate(block_ids)}
                    if _pseudodyad_assignment_valid(assignment, block_map, slots,
                                                    allowed_by_slot_target, original_partner_pairs):
                        worlds.append(assignment)
                world_count = len(worlds)
                if not worlds:
                    return {"status": "unavailable", "reason": "pseudodyad_assignment_infeasible",
                            "diagnostics": {"stratum": stratum, "enumerated_permutation_product": permutation_product}}
            else:
                worlds = None
        support_list.append({"stratum": stratum, "blocks": block_ids, "slots": slots,
                             "block_map": block_map, "allowed": allowed_by_slot_target,
                             "original_partner_pairs": original_partner_pairs,
                             "permutation_lists": permutation_lists, "worlds": worlds,
                             "world_count": world_count, "enumerated": worlds is not None,
                             "task_time_mode": mode})
        diagnostics["stratum_block_counts"][str(stratum)] = len(block_ids)
        diagnostics["stratum_slot_counts"][str(stratum)] = len(slots)
        diagnostics["enumerated_world_counts"][str(stratum)] = world_count
    if not support_list:
        return {"status": "unavailable", "reason": "pseudodyad_no_compatible_independent_blocks",
                "diagnostics": diagnostics}
    finite_counts = [support["world_count"] for support in support_list]
    finite_support_size = math.prod(finite_counts) if all(count is not None for count in finite_counts) else None
    return {"status": "available", "world_support": support_list, "edges": edges,
            "finite_support_size": finite_support_size, "diagnostics": diagnostics,
            "required_by_key": required_by_key}


def _pseudodyad_support_compatible(aligned, target_node, donor_node, required_indices):
    time = np.asarray(aligned.get("time_s", []), dtype=float)
    target_key, donor_key = target_node["record_key"], donor_node["record_key"]
    target_mask = np.asarray(aligned.get("eligible", {}).get(target_key, []), dtype=bool)
    donor_mask = np.asarray(aligned.get("eligible", {}).get(donor_key, []), dtype=bool)
    target_values = np.asarray(aligned.get("signals", {}).get(target_key, []), dtype=float)
    donor_values = np.asarray(aligned.get("signals", {}).get(donor_key, []), dtype=float)
    if (target_mask.shape != time.shape or donor_mask.shape != time.shape
            or target_values.shape != time.shape or donor_values.shape != time.shape):
        return False
    target_mask &= np.isfinite(target_values)
    donor_mask &= np.isfinite(donor_values)
    target_mode = target_node["task_time_mode"]
    donor_mode = donor_node["task_time_mode"]
    if target_mode != donor_mode:
        return False
    if target_mode == "task_relative":
        shift = 0
    else:
        dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
        if dt <= 0:
            return False
        shift_float = (float(donor_node["task_time_origin_s"]) - float(target_node["task_time_origin_s"])) / dt
        shift = int(np.rint(shift_float))
        if abs(shift_float - shift) > 1e-8:
            return False
    indices = np.arange(len(time), dtype=int) + shift
    in_bounds = (indices >= 0) & (indices < len(time))
    moved_mask = np.zeros(len(time), dtype=bool)
    moved_mask[in_bounds] = donor_mask[indices[in_bounds]]
    if required_indices:
        required = np.fromiter(sorted(required_indices), dtype=int)
        if np.any(required < 0) or np.any(required >= len(time)):
            return False
        if not np.array_equal(moved_mask[required], target_mask[required]):
            return False
    return True


def _pseudodyad_assignment_valid(assignment, block_map, slots, allowed_by_slot_target,
                                 original_partner_pairs):
    for target, slot_map in assignment.items():
        if len(set(slot_map.values())) != len(slots):
            return False
        donor_nodes = []
        for slot in slots:
            donor = slot_map[slot]
            if donor not in allowed_by_slot_target[slot][target]:
                return False
            donor_nodes.append(block_map[donor][slot])
        for left, right in itertools.combinations(donor_nodes, 2):
            if str(left["subject_id"]) == str(right["subject_id"]):
                return False
            if frozenset((str(left["subject_id"]), str(right["subject_id"]))) in original_partner_pairs:
                return False
    return True


def _sample_pseudodyad_world(world_support, rng, search_limit=1000, cancel_callback=None,
                             return_attempts=False):
    combined = {}
    proposals = 0
    for support in world_support:
        _cancel(cancel_callback)
        if support["worlds"] is not None:
            proposals += 1
            selected = support["worlds"][int(rng.integers(0, len(support["worlds"]))) ]
        else:
            blocks, slots = support["blocks"], support["slots"]
            selected = None
            for _ in range(search_limit):
                _cancel(cancel_callback)
                proposals += 1
                if support["permutation_lists"]:
                    permutations = [support["permutation_lists"][slot][
                        int(rng.integers(0, len(support["permutation_lists"][slot])))] for slot in slots]
                else:
                    permutations = [tuple(rng.permutation(blocks).tolist()) for _ in slots]
                assignment = {target: {slot: permutations[slot_index][target_index]
                                       for slot_index, slot in enumerate(slots)}
                              for target_index, target in enumerate(blocks)}
                valid = _pseudodyad_assignment_valid(assignment, support["block_map"], slots,
                                                     support["allowed"], support["original_partner_pairs"])
                if valid:
                    selected = assignment
                    break
            if selected is None:
                return (None, proposals) if return_attempts else None
        for target, slot_map in selected.items():
            combined[(support["stratum"], target)] = slot_map
    return (combined, proposals) if return_attempts else combined


def _first_pseudodyad_world(world_support):
    combined = {}
    for support in world_support:
        if support["worlds"] is None or not support["worlds"]:
            return None
        selected = support["worlds"][0]
        for target, slot_map in selected.items():
            combined[(support["stratum"], target)] = slot_map
    return combined


def _pseudodyad_world_id(world):
    normalized = {f"{stratum!r}::{target[0]}::{target[1]}": {
        f"{slot[0]}::{slot[1]}": f"{donor[0]}::{donor[1]}" for slot, donor in sorted(slot_map.items())}
        for (stratum, target), slot_map in sorted(world.items(), key=lambda item: str(item[0]))}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _apply_pseudodyad_world(aligned, setup, world):
    result = {**aligned, "signals": {}, "eligible": {}, "metadata": copy.deepcopy(aligned.get("metadata", {}))}
    donor_by_target, node_lookup = {}, {}
    for support in setup["world_support"]:
        for (stratum, target_block), slot_map in world.items():
            if stratum != support["stratum"] or target_block not in support["block_map"]:
                continue
            for slot, donor_block in slot_map.items():
                target_node = support["block_map"][target_block][slot]
                donor_node = support["block_map"][donor_block][slot]
                target_key, donor_key = target_node["record_key"], donor_node["record_key"]
                time = np.asarray(aligned["time_s"], dtype=float)
                donor_values = np.asarray(aligned["signals"][donor_key], dtype=float)
                donor_mask = np.asarray(aligned["eligible"][donor_key], dtype=bool)
                if target_node["task_time_mode"] == "origin":
                    dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
                    shift = int(np.rint((float(donor_node["task_time_origin_s"])
                                         - float(target_node["task_time_origin_s"])) / dt))
                else:
                    shift = 0
                indices = np.arange(len(time), dtype=int) + shift
                in_bounds = (indices >= 0) & (indices < len(time))
                values = np.full(len(time), np.nan, dtype=float)
                mask = np.zeros(len(time), dtype=bool)
                values[in_bounds] = donor_values[indices[in_bounds]]
                mask[in_bounds] = donor_mask[indices[in_bounds]] & np.isfinite(values[in_bounds])
                result["signals"][target_key] = values
                result["eligible"][target_key] = mask
                donor_by_target[target_key] = donor_node
                node_lookup[target_key] = target_node
    for key in aligned.get("signals", {}):
        if key not in result["signals"]:
            result["signals"][key] = np.asarray(aligned["signals"][key], dtype=float).copy()
            result["eligible"][key] = np.asarray(aligned["eligible"][key], dtype=bool).copy()
    provenance = {}
    for edge in setup.get("edges", []):
        endpoints = []
        for field in ("record_key_a", "record_key_b"):
            target_key = edge[field]
            donor = donor_by_target.get(target_key)
            target = node_lookup.get(target_key)
            if donor is None or target is None:
                continue
            endpoints.append({"target_record_key": target_key,
                              "target_subject_id": target.get("subject_id"),
                              "target_group_id": target.get("group_id"),
                              "target_session_id": target.get("session_id"),
                              "donor_record_key": donor.get("record_key"),
                              "donor_subject_id": donor.get("subject_id"),
                              "donor_group_id": donor.get("group_id"),
                              "donor_session_id": donor.get("session_id")})
        provenance[edge.get("pair_id", "")] = endpoints
    return result, provenance


def matched_pseudodyads(aligned, config=None, *, cancel_callback=None):
    """Build cross-group/session matched pairs; return unavailable when exchangeability keys are absent."""
    config = config or {}
    nodes, membership_report = _membership_nodes(aligned)
    fields = list(dict.fromkeys(["task", "stimulus"] + list(config.get("synchrony", {}).get("matching_fields", []))
                                + list(config.get("synchrony", {}).get("pseudodyad_strata", []))))
    timeline_candidates = ("timeline_id", "stimulus_timeline_id", "timeline_hash", "stimulus_schedule_id")
    # Enrich membership rows from per-record metadata without overwriting declared table values.
    enriched = []
    for node in nodes:
        row = dict(node)
        row.update({key: value for key, value in aligned.get("metadata", {}).get(node["record_key"], {}).items()
                    if row.get(key) in (None, "") and value is not None})
        enriched.append(row)
    available_cols = set().union(*(row.keys() for row in enriched)) if enriched else set()
    slot_field = "member_slot" if "member_slot" in available_cols else "role" if "role" in available_cols else None
    missing = [] if slot_field else ["role_or_member_slot"]
    missing.extend(field for field in fields if field not in available_cols)
    timeline_field = next((field for field in timeline_candidates if field in available_cols), None)
    if timeline_field is None:
        missing.append("timeline_id (or stimulus_timeline_id/timeline_hash/stimulus_schedule_id)")
    diagnostics = {"status": "available", "method": "matched_pseudodyads",
                   "matching_fields": [*fields, slot_field, timeline_field] if timeline_field else [*fields, slot_field],
                   "unresolved_membership": membership_report.get("unresolved_rows", [])}
    if missing:
        diagnostics.update({"status": "unavailable", "reason": "matching_metadata_missing", "missing_fields": sorted(set(missing))})
        return pd.DataFrame(columns=["pseudodyad_id", "group_id_a", "session_id_a", "subject_id_a", "record_key_a",
                                     "group_id_b", "session_id_b", "subject_id_b", "record_key_b", "status", "reason"]), diagnostics
    rows = []
    for left, right in itertools.combinations(enriched, 2):
        _cancel(cancel_callback)
        if left["subject_id"] == right["subject_id"]:
            continue
        left_block = (str(left["group_id"]), str(left["session_id"]))
        right_block = (str(right["group_id"]), str(right["session_id"]))
        if left["group_id"] == right["group_id"] or left_block == right_block:
            continue
        if any(not _present(left.get(field)) or not _present(right.get(field))
               or _json_token(left.get(field)) != _json_token(right.get(field))
               for field in fields + [slot_field, timeline_field]):
            continue
        ordered = sorted((left, right), key=lambda row: (row["group_id"], row["session_id"], row["subject_id"], row["record_key"]))
        a, b = ordered
        identity = f"{a['record_key']}::{b['record_key']}"
        rows.append({"pseudodyad_id": identity, "group_id_a": a["group_id"], "session_id_a": a["session_id"],
                     "subject_id_a": a["subject_id"], "record_key_a": a["record_key"],
                     "group_id_b": b["group_id"], "session_id_b": b["session_id"],
                     "subject_id_b": b["subject_id"], "record_key_b": b["record_key"],
                     **{field: a.get(field) for field in fields + [slot_field, timeline_field]},
                     "status": "candidate", "reason": "matching_metadata_equal_independent_group_and_session"})
    diagnostics.update({"status": "available" if rows else "unavailable",
                        "reason": "" if rows else "no_exchangeable_cross_group_session_pairs",
                        "candidate_count": len(rows), "timeline_field": timeline_field})
    return pd.DataFrame(rows), diagnostics


__all__ = ["apply_clock_alignment", "align_recordings", "compute_pair_synchrony",
           "summarize_group_synchrony", "estimate_sync_null", "matched_pseudodyads"]
