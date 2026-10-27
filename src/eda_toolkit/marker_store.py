"""Versioned, channel-aware event markers with append-only change history."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable

from .project_store import (
    _canonical_json,
    _sha256_text,
    list_audit_actions,
    read_project_snapshot,
    save_project_revision,
)


_EVENT_FIELDS = ("recording_id", "event_id", "onset_s", "duration_s", "event_type", "condition", "channel", "clock_domain")
_CSV_FIELDS = ("recording_id", "event_id", "source_event_id", "source_event_id_original", "marker_id", "onset_s", "duration_s",
              "event_type", "condition", "channel", "clock_domain", "source_path", "source_sha256",
              "source_recording_id", "source_channel_scope", "source_event_id_bound", "marker_revision")
_EDITABLE_FIELDS = {"recording_id", "channel", "source_event_id", "onset_s", "duration_s",
                    "event_type", "condition", "clock_domain"}


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().casefold() in {"na", "nan", "null"}
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def _as_rows(rows: Any) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict") and hasattr(rows, "columns"):
        return [dict(item) for item in rows.to_dict(orient="records")]
    if isinstance(rows, dict):
        return [dict(rows)]
    if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes)):
        values = list(rows)
        if not all(isinstance(item, dict) for item in values):
            raise TypeError("marker_rows_must_be_objects")
        return [dict(item) for item in values]
    raise TypeError("marker_rows_must_be_table_or_object_list")


def _clean_text(value: Any, field: str, *, optional: bool = False) -> str | None:
    if _missing(value):
        if optional:
            return None
        raise ValueError(f"marker_field_required:{field}")
    result = str(value).strip()
    if not result and not optional:
        raise ValueError(f"marker_field_required:{field}")
    return result or None


def _finite_number(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"marker_number_invalid:{field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"marker_number_invalid:{field}") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"marker_number_invalid:{field}")
    return result


def _marker_key(marker: dict[str, Any]) -> tuple[str, str, str]:
    return (str(marker["recording_id"]), str(marker["channel_scope"]), str(marker["source_event_id"]))


def _semantic_payload(marker: dict[str, Any]) -> dict[str, Any]:
    return {key: marker.get(key) for key in (
        "recording_id", "channel_scope", "source_event_id", "onset_s", "duration_s",
        "event_type", "condition", "clock_domain",
    )}


def _internal_event_id(marker_id: str) -> str:
    return "mk_" + uuid.UUID(marker_id).hex


def _source_binding(row: dict[str, Any], *, source_path: str | None, source_sha256: str,
                    recording_id: str, channel_scope: str, source_event_id: str) -> dict[str, Any] | None:
    """Keep the original CSV row identity stable while editable marker fields change."""
    prior = row.get("source_binding")
    if isinstance(prior, dict):
        raw_path = prior.get("source_path")
        raw_hash = prior.get("source_sha256")
        bound_recording = prior.get("recording_id")
        bound_channel = prior.get("channel_scope")
        bound_event = prior.get("source_event_id")
    else:
        raw_path = source_path or row.get("source_path")
        raw_hash = source_sha256 or row.get("source_sha256")
        bound_recording = row.get("source_recording_id", recording_id)
        bound_channel = row.get("source_channel_scope", channel_scope)
        bound_event = row.get("source_event_id_bound") or row.get("source_event_id_original") or source_event_id
    if _missing(raw_path) or _missing(raw_hash):
        return None
    digest = str(raw_hash).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    try:
        resolved_path = str(Path(str(raw_path)).expanduser().resolve())
        bound_recording = _clean_text(bound_recording, "source_recording_id")
        bound_event = _clean_text(bound_event, "source_event_id_bound")
        if _missing(bound_channel) or str(bound_channel).strip() == "*":
            bound_channel = "*"
        else:
            bound_channel = str(bound_channel).strip()
    except (TypeError, ValueError, OSError):
        return None
    return {"source_path": resolved_path, "source_sha256": digest,
            "recording_id": bound_recording, "channel_scope": bound_channel,
            "source_event_id": bound_event}


def list_marker_source_tombstones(project: Any) -> list[dict[str, Any]]:
    """Return durable suppressions for imported source rows removed by the user."""
    snapshot = read_project_snapshot(project)
    tombstones = snapshot.get("marker_source_tombstones", []) if isinstance(snapshot, dict) else []
    if not isinstance(tombstones, list) or not all(isinstance(item, dict) for item in tombstones):
        raise ValueError("marker_source_tombstones_invalid")
    return [dict(item) for item in tombstones]


def merge_marker_event_rows(source_rows: list[dict[str, Any]], source_report: dict[str, Any],
                            managed_rows: list[dict[str, Any]],
                            tombstones: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Materialize editable markers over a read-only source CSV without resurrecting deletions."""
    def channel_scope(value: Any) -> str:
        text = "" if value is None else str(value).strip()
        return "*" if not text or text.casefold() in {"na", "nan", "null"} else text

    replacement_keys: set[tuple[str, str, str]] = set()
    for marker in managed_rows:
        recording_id = str(marker.get("recording_id", ""))
        channel = channel_scope(marker.get("channel", marker.get("channel_scope")))
        for source_id in (marker.get("source_event_id"), marker.get("source_event_id_original")):
            if source_id is not None:
                replacement_keys.add((recording_id, channel, str(source_id)))
        binding = marker.get("source_binding")
        if isinstance(binding, dict):
            replacement_keys.add((str(binding.get("recording_id", "")),
                                  channel_scope(binding.get("channel_scope")),
                                  str(binding.get("source_event_id", ""))))

    source_path = source_report.get("source_path")
    source_sha256 = str(source_report.get("source_sha256", "")).lower()
    try:
        source_path = str(Path(source_path).expanduser().resolve()) if source_path else None
    except (TypeError, OSError):
        source_path = None
    suppressed_keys: set[tuple[str, str, str]] = set()
    for tombstone in tombstones or []:
        try:
            tombstone_path = str(Path(tombstone.get("source_path", "")).expanduser().resolve())
        except (TypeError, OSError):
            continue
        if tombstone_path != source_path or str(tombstone.get("source_sha256", "")).lower() != source_sha256:
            continue
        suppressed_keys.add((str(tombstone.get("recording_id", "")),
                             channel_scope(tombstone.get("channel_scope")),
                             str(tombstone.get("source_event_id", ""))))

    fields = ("recording_id", "event_id", "source_event_id", "source_event_id_original",
              "marker_id", "onset_s", "duration_s", "event_type", "condition", "channel",
              "clock_domain", "source_path", "source_sha256", "marker_revision")
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        key = (str(row.get("recording_id", "")), channel_scope(row.get("channel")),
               str(row.get("source_event_id") or row.get("event_id", "")))
        if key in replacement_keys or key in suppressed_keys:
            continue
        preserved = {field: row.get(field) for field in fields}
        preserved["source_event_id"] = row.get("source_event_id") or row.get("event_id")
        preserved["source_path"] = row.get("source_path") or source_report.get("source_path")
        preserved["source_sha256"] = row.get("source_sha256") or source_report.get("source_sha256")
        rows.append(preserved)
    rows.extend({field: marker.get(field) for field in fields} for marker in managed_rows)
    return rows


def _normalise_row(row: dict[str, Any], *, source_path: str | None, source_sha256: str,
                   default_clock_domain: str | None = None) -> dict[str, Any]:
    recording_id = _clean_text(row.get("recording_id"), "recording_id")
    user_event_id = row.get("source_event_id")
    if _missing(user_event_id):
        user_event_id = row.get("event_id")
    source_event_id = _clean_text(user_event_id, "source_event_id")
    channel_raw = row.get("channel_scope", row.get("channel"))
    channel_scope = "*" if _missing(channel_raw) or str(channel_raw).strip() == "*" else str(channel_raw).strip()
    clock_domain = row.get("clock_domain")
    if _missing(clock_domain):
        clock_domain = default_clock_domain
    clock_domain = _clean_text(clock_domain, "clock_domain")
    marker_id_value = row.get("marker_id")
    marker_id = str(uuid.UUID(str(marker_id_value))) if not _missing(marker_id_value) else str(uuid.uuid4())
    supplied_event_id = row.get("event_id")
    # A user may legitimately label an event "mk_*". Only normalized snapshots
    # carrying an explicit marker_id assert that event_id is the internal UUID form.
    if not _missing(marker_id_value) and supplied_event_id and str(supplied_event_id) != _internal_event_id(marker_id):
        raise ValueError("marker_internal_event_id_mismatch")
    resolved_source_path = (str(Path(source_path).expanduser().resolve()) if source_path else
                            _clean_text(row.get("source_path"), "source_path", optional=True))
    resolved_source_hash = (source_sha256 or
                            _clean_text(row.get("source_sha256"), "source_sha256", optional=True))
    binding = _source_binding(row, source_path=resolved_source_path,
                              source_sha256=resolved_source_hash or "",
                              recording_id=recording_id, channel_scope=channel_scope,
                              source_event_id=source_event_id)
    return {
        "marker_id": marker_id,
        "event_id": _internal_event_id(marker_id),
        "recording_id": recording_id,
        "channel_scope": channel_scope,
        "channel": channel_scope,
        "source_event_id": source_event_id,
        "source_event_id_original": _clean_text(row.get("source_event_id_original"), "source_event_id_original", optional=True),
        "onset_s": _finite_number(row.get("onset_s"), "onset_s"),
        "duration_s": _finite_number(row.get("duration_s"), "duration_s", minimum=0.0),
        "event_type": _clean_text(row.get("event_type"), "event_type"),
        "condition": _clean_text(row.get("condition"), "condition", optional=True),
        "clock_domain": clock_domain,
        "source_path": resolved_source_path,
        "source_sha256": resolved_source_hash,
        "source_binding": binding,
        "source_recording_id": binding.get("recording_id") if binding else None,
        "source_channel_scope": binding.get("channel_scope") if binding else None,
        "source_event_id_bound": binding.get("source_event_id") if binding else None,
        "revision": int(row.get("revision", 0) or 0),
    }


def _source_fingerprint(rows: list[dict[str, Any]], source_path: str | os.PathLike[str] | None,
                        source_sha256: str | None) -> tuple[str | None, str]:
    if source_path is not None:
        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"marker_source_not_file:{path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_sha256 is not None and source_sha256 != digest:
            raise ValueError("marker_source_hash_mismatch")
        return str(path), digest
    if source_sha256 is not None:
        if len(source_sha256) != 64 or any(char not in "0123456789abcdefABCDEF" for char in source_sha256):
            raise ValueError("marker_source_hash_invalid")
        return None, source_sha256.lower()
    safe_rows = []
    for row in rows:
        safe_row = {}
        for key, value in row.items():
            if _missing(value):
                safe_row[key] = None
            elif hasattr(value, "item"):
                safe_row[key] = value.item()
            else:
                safe_row[key] = value
        safe_rows.append(safe_row)
    return None, _sha256_text(_canonical_json(safe_rows))


def read_marker_csv(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read event CSV fields as text first, preserving IDs such as 001 exactly."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"marker_source_not_file:{source}")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("marker_csv_header_required")
        missing = {"recording_id", "event_id", "onset_s", "duration_s", "event_type", "clock_domain"} - set(reader.fieldnames)
        if missing:
            raise ValueError("marker_csv_missing_columns:" + ",".join(sorted(missing)))
        rows = [dict(row) for row in reader]
    return rows, {"source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "row_count": len(rows)}


def get_markers(project: Any) -> list[dict[str, Any]]:
    snapshot = read_project_snapshot(project)
    markers = snapshot.get("markers", []) if isinstance(snapshot, dict) else []
    if not isinstance(markers, list) or not all(isinstance(item, dict) for item in markers):
        raise ValueError("marker_snapshot_invalid")
    return [dict(item) for item in markers]


def marker_snapshot_hash(project: Any) -> str:
    rows = get_markers(project)
    return _sha256_text(_canonical_json(rows))


def _apply_delta(current: list[dict[str, Any]], replacing: list[dict[str, Any] | None] | None,
                 desired: list[dict[str, Any] | None] | None) -> list[dict[str, Any]]:
    result = [dict(item) for item in current]
    for marker in replacing or []:
        if not isinstance(marker, dict):
            continue
        marker_id = marker["marker_id"]
        result = [item for item in result if item.get("marker_id") != marker_id]
    for marker in desired or []:
        if isinstance(marker, dict):
            result = [item for item in result if item.get("marker_id") != marker["marker_id"]]
            result.append(dict(marker))
    return result


def _commit_marker_change(project: dict[str, Any], current: list[dict[str, Any]], updated: list[dict[str, Any]],
                          *, operator: str, operation: str, before: list[dict[str, Any]],
                          after: list[dict[str, Any]], reason: str | None = None,
                          undo_stack: list[str] | None = None, redo_stack: list[str] | None = None) -> dict[str, Any]:
    if _canonical_json(current) == _canonical_json(updated):
        return {"changed": False, "marker_revision": int(project.get("marker_revision", 0)),
                "markers": current, "marker_snapshot_sha256": _sha256_text(_canonical_json(current))}
    next_revision = int(project.get("marker_revision", 0)) + 1
    normalised = []
    for item in updated:
        value = dict(item)
        value["revision"] = next_revision if value.get("marker_id") in {entry.get("marker_id") for entry in after} else int(value.get("revision", 0))
        normalised.append(value)
    action_id = str(uuid.uuid4())
    old_snapshot = read_project_snapshot(project)
    source_tombstones = [dict(item) for item in old_snapshot.get("marker_source_tombstones", [])]
    deleted_markers = before if operation == "delete" else before if operation == "redo" and not after else []
    for marker in deleted_markers:
        binding = marker.get("source_binding") if isinstance(marker.get("source_binding"), dict) else None
        if binding is None:
            binding = _source_binding(marker, source_path=marker.get("source_path"),
                                      source_sha256=str(marker.get("source_sha256") or ""),
                                      recording_id=str(marker.get("recording_id", "")),
                                      channel_scope=str(marker.get("channel_scope", marker.get("channel", "*"))),
                                      source_event_id=str(marker.get("source_event_id_original") or
                                                           marker.get("source_event_id", "")))
        if binding and binding not in source_tombstones:
            source_tombstones.append(binding)
    new_undo = list(undo_stack if undo_stack is not None else old_snapshot.get("marker_undo_stack", []))
    new_redo = list(redo_stack if redo_stack is not None else old_snapshot.get("marker_redo_stack", []))
    if operation not in {"undo", "redo"}:
        new_undo.append(action_id)
        new_redo = []
    payload = {"markers": normalised, "marker_undo_stack": new_undo, "marker_redo_stack": new_redo,
               "marker_source_tombstones": source_tombstones}
    revision = save_project_revision(
        project, payload, int(project.get("project_revision", 0)), kind="marker",
        audit_table="marker_actions",
        audit_action={"action_id": action_id, "operator": operator, "operation": operation,
                      "before": before, "after": after, "reason": reason},
    )
    return {"changed": True, "project_revision": revision,
            "marker_revision": int(project.get("marker_revision", next_revision)),
            "action_id": action_id, "markers": normalised,
            "marker_snapshot_sha256": _sha256_text(_canonical_json(normalised))}


def import_markers(project: dict[str, Any], rows: Any, *, operator: str,
                   source_path: str | os.PathLike[str] | None = None,
                   source_sha256: str | None = None,
                   conflict_policy: str = "reject", reason: str | None = None,
                   default_clock_domain: str | None = None) -> dict[str, Any]:
    """Import marker rows idempotently or resolve same-scope conflicts explicitly."""
    if conflict_policy not in {"reject", "replace", "rename"}:
        raise ValueError("marker_conflict_policy_invalid")
    if not str(operator).strip():
        raise ValueError("marker_operator_required")
    raw = _as_rows(rows)
    resolved_source_path, resolved_source_hash = _source_fingerprint(raw, source_path, source_sha256)
    incoming = [_normalise_row(row, source_path=resolved_source_path,
                               source_sha256=resolved_source_hash,
                               default_clock_domain=default_clock_domain) for row in raw]
    current = get_markers(project)
    updated = [dict(item) for item in current]
    changed_before: list[dict[str, Any]] = []
    changed_after: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    seen_ids: dict[str, tuple[str, str, str]] = {}
    existing_by_id = {str(item["marker_id"]): item for item in current}
    tombstones = list_marker_source_tombstones(project)
    next_revision = int(project.get("marker_revision", 0)) + 1
    for marker in incoming:
        marker["revision"] = next_revision
        key = _marker_key(marker)
        binding = marker.get("source_binding")
        if isinstance(binding, dict):
            if any(binding == item for item in tombstones):
                warnings.append("source_row_previously_deleted")
                continue
            if any(item.get("source_binding") == binding for item in updated):
                # Reimporting the exact same source row must preserve the user's
                # current edits, even when its visible event ID has been renamed.
                continue
        renamed_import = next((item for item in updated
                               if item.get("source_event_id_original") == marker["source_event_id"]
                               and item.get("source_sha256") == marker.get("source_sha256")
                               and {key: value for key, value in _semantic_payload(item).items()
                                    if key != "source_event_id"}
                               == {key: value for key, value in _semantic_payload(marker).items()
                                   if key != "source_event_id"}), None)
        if renamed_import is not None:
            continue
        prior_key = seen_ids.get(marker["marker_id"])
        if prior_key is not None and prior_key != key:
            raise ValueError("marker_id_duplicate_within_import")
        seen_ids[marker["marker_id"]] = key
        existing_id = existing_by_id.get(marker["marker_id"])
        if existing_id is not None and _marker_key(existing_id) != key:
            raise ValueError("marker_id_conflict")
        if key in seen:
            if _semantic_payload(seen[key]) == _semantic_payload(marker):
                continue
            raise ValueError("duplicate_marker_key_within_import")
        seen[key] = marker
        index = next((i for i, item in enumerate(updated) if _marker_key(item) == key), None)
        if index is None:
            updated.append(marker)
            changed_before.append(None)
            changed_after.append(marker)
            continue
        existing = updated[index]
        if _semantic_payload(existing) == _semantic_payload(marker):
            # Same contents are idempotent; keep the first source provenance and UUID.
            continue
        if conflict_policy == "reject":
            raise ValueError("marker_conflict_requires_replace_rename_or_reject")
        if conflict_policy == "replace":
            marker["marker_id"] = existing["marker_id"]
            marker["event_id"] = existing["event_id"]
            marker["source_event_id_original"] = existing.get("source_event_id_original") or marker["source_event_id"]
            updated[index] = marker
            changed_before.append(existing)
            changed_after.append(marker)
            continue
        # A rename is explicit policy selection. Keep the imported label separately,
        # and create a stable, collision-free visible source ID for the additional row.
        original_source_id = marker["source_event_id"]
        suffix = resolved_source_hash[:8]
        renamed = f"{original_source_id}__import_{suffix}"
        occupied = {_marker_key(item) for item in updated}
        serial = 2
        while (marker["recording_id"], marker["channel_scope"], renamed) in occupied:
            renamed = f"{original_source_id}__import_{suffix}_{serial}"
            serial += 1
        marker["source_event_id_original"] = original_source_id
        marker["source_event_id"] = renamed
        marker["marker_id"] = str(uuid.uuid4())
        marker["event_id"] = _internal_event_id(marker["marker_id"])
        updated.append(marker)
        changed_before.append(None)
        changed_after.append(marker)

    if not changed_after:
        return {"changed": False, "idempotent": True,
                "marker_revision": int(project.get("marker_revision", 0)), "markers": current,
                "marker_snapshot_sha256": _sha256_text(_canonical_json(current)), "warnings": warnings}
    result = _commit_marker_change(project, current, updated, operator=operator, operation="import",
                                   before=changed_before, after=changed_after, reason=reason or conflict_policy)
    result.update({"idempotent": False, "imported_count": len(changed_after), "warnings": warnings})
    return result


def update_marker(project: dict[str, Any], marker_id: str, changes: dict[str, Any], *,
                  operator: str, reason: str) -> dict[str, Any]:
    if not isinstance(changes, dict) or not changes:
        raise ValueError("marker_changes_required")
    unknown = set(changes) - _EDITABLE_FIELDS
    if unknown:
        raise ValueError("marker_fields_not_editable:" + ",".join(sorted(unknown)))
    current = get_markers(project)
    index = next((i for i, item in enumerate(current) if item["marker_id"] == str(uuid.UUID(marker_id))), None)
    if index is None:
        raise KeyError("marker_not_found")
    before = current[index]
    candidate = {**before, **changes}
    if "channel" in changes and "channel_scope" not in changes:
        candidate["channel_scope"] = changes["channel"]
    if ("source_event_id" in changes and
            str(changes["source_event_id"]) != str(before.get("source_event_id")) and
            not before.get("source_event_id_original")):
        binding = before.get("source_binding") if isinstance(before.get("source_binding"), dict) else {}
        candidate["source_event_id_original"] = (
            binding.get("source_event_id") or before.get("source_event_id"))
    normalized = _normalise_row(candidate, source_path=candidate.get("source_path"),
                                source_sha256=candidate.get("source_sha256") or "",
                                default_clock_domain=before["clock_domain"])
    normalized["marker_id"] = before["marker_id"]
    normalized["event_id"] = before["event_id"]
    normalized["source_event_id_original"] = (
        before.get("source_event_id_original") or candidate.get("source_event_id_original"))
    if any(i != index and _marker_key(item) == _marker_key(normalized) for i, item in enumerate(current)):
        raise ValueError("marker_conflict_requires_replace_rename_or_reject")
    updated = [dict(item) for item in current]
    updated[index] = normalized
    return _commit_marker_change(project, current, updated, operator=operator, operation="edit",
                                 before=[before], after=[normalized], reason=reason)


def delete_marker(project: dict[str, Any], marker_id: str, *, operator: str, reason: str) -> dict[str, Any]:
    normalized_id = str(uuid.UUID(marker_id))
    current = get_markers(project)
    marker = next((item for item in current if item["marker_id"] == normalized_id), None)
    if marker is None:
        raise KeyError("marker_not_found")
    updated = [dict(item) for item in current if item["marker_id"] != normalized_id]
    return _commit_marker_change(project, current, updated, operator=operator, operation="delete",
                                 before=[marker], after=[], reason=reason)


def _action_map(project: Any) -> dict[str, dict[str, Any]]:
    return {item["action_id"]: item for item in list_audit_actions(project, "marker_actions")}


def undo_marker_action(project: dict[str, Any], *, operator: str, reason: str = "undo") -> dict[str, Any]:
    snapshot = read_project_snapshot(project)
    undo_stack = list(snapshot.get("marker_undo_stack", []))
    redo_stack = list(snapshot.get("marker_redo_stack", []))
    if not undo_stack:
        raise RuntimeError("marker_undo_unavailable")
    target_id = undo_stack.pop()
    action = _action_map(project).get(target_id)
    if action is None:
        raise RuntimeError("marker_undo_action_missing")
    current = get_markers(project)
    updated = _apply_delta(current, action.get("after"), action.get("before"))
    result = _commit_marker_change(project, current, updated, operator=operator, operation="undo",
                                   before=action.get("after") or [], after=action.get("before") or [],
                                   reason=f"target_action_id={target_id};{reason}",
                                   undo_stack=undo_stack, redo_stack=[*redo_stack, target_id])
    return result


def redo_marker_action(project: dict[str, Any], *, operator: str, reason: str = "redo") -> dict[str, Any]:
    snapshot = read_project_snapshot(project)
    undo_stack = list(snapshot.get("marker_undo_stack", []))
    redo_stack = list(snapshot.get("marker_redo_stack", []))
    if not redo_stack:
        raise RuntimeError("marker_redo_unavailable")
    target_id = redo_stack.pop()
    action = _action_map(project).get(target_id)
    if action is None:
        raise RuntimeError("marker_redo_action_missing")
    current = get_markers(project)
    updated = _apply_delta(current, action.get("before"), action.get("after"))
    result = _commit_marker_change(project, current, updated, operator=operator, operation="redo",
                                   before=action.get("before") or [], after=action.get("after") or [],
                                   reason=f"target_action_id={target_id};{reason}",
                                   undo_stack=[*undo_stack, target_id], redo_stack=redo_stack)
    return result


def export_marker_snapshot(project: Any, path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Return normalized events with internal IDs plus original user IDs; optionally write CSV."""
    markers = get_markers(project)
    revision = int(project.get("marker_revision", 0)) if isinstance(project, dict) else 0
    rows = []
    for marker in markers:
        rows.append({
            "recording_id": marker["recording_id"], "event_id": marker["event_id"],
            "source_event_id": marker["source_event_id"],
            "source_event_id_original": marker.get("source_event_id_original"),
            "marker_id": marker["marker_id"],
            "onset_s": marker["onset_s"], "duration_s": marker["duration_s"],
            "event_type": marker["event_type"], "condition": marker.get("condition"),
            "channel": marker["channel_scope"], "clock_domain": marker["clock_domain"],
            "source_path": marker.get("source_path"), "source_sha256": marker.get("source_sha256"),
            "source_recording_id": (marker.get("source_binding") or {}).get("recording_id"),
            "source_channel_scope": (marker.get("source_binding") or {}).get("channel_scope"),
            "source_event_id_bound": (marker.get("source_binding") or {}).get("source_event_id"),
            "source_binding": marker.get("source_binding"),
            "marker_revision": marker.get("revision", revision),
        })
    if path is not None:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8-sig", newline="", dir=target.parent,
                                             prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
                temp_name = stream.name
                writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)
    return rows

