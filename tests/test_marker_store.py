"""Versioned marker import, channel scope, conflict policy, and history."""

from __future__ import annotations

import csv
import uuid
from pathlib import Path

import pytest

from eda_toolkit.marker_store import (
    delete_marker,
    export_marker_snapshot,
    get_markers,
    import_markers,
    marker_snapshot_hash,
    list_marker_source_tombstones,
    merge_marker_event_rows,
    read_marker_csv,
    redo_marker_action,
    undo_marker_action,
    update_marker,
)
from eda_toolkit.project_store import list_audit_actions, open_project


def _row(event_id="001", channel="c1", onset=2.5):
    return {"recording_id": "r1", "event_id": event_id, "onset_s": onset, "duration_s": 0.2,
            "event_type": "stimulus", "condition": "A", "channel": channel,
            "clock_domain": "device"}


def test_channel_scoped_same_source_id_and_broadcast_legacy_row(tmp_path: Path):
    project = open_project(tmp_path / "project")
    result = import_markers(project, [_row(channel="c1", onset=2.5), _row(channel="c2", onset=4.0)],
                            operator="tester")
    assert result["imported_count"] == 2
    markers = get_markers(project)
    assert {item["source_event_id"] for item in markers} == {"001"}
    assert len({item["event_id"] for item in markers}) == 2
    assert all(item["event_id"].startswith("mk_") for item in markers)
    assert {item["channel_scope"] for item in markers} == {"c1", "c2"}

    legacy = import_markers(project, [_row(channel=None, onset=8.0)], operator="tester")
    assert legacy["markers"][-1]["channel_scope"] == "*"
    rows = export_marker_snapshot(project)
    assert len(rows) == 3
    assert {(item["event_id"], item["source_event_id"], item["channel"]) for item in rows}


def test_conflict_policy_is_explicit_and_rename_reimport_is_idempotent(tmp_path: Path):
    project = open_project(tmp_path / "project")
    import_markers(project, [_row(onset=2.5)], operator="tester")
    before_revision = project["marker_revision"]
    with pytest.raises(ValueError, match="marker_conflict_requires_replace_rename_or_reject"):
        import_markers(project, [_row(onset=9.0)], operator="tester", conflict_policy="reject")
    assert project["marker_revision"] == before_revision

    renamed = import_markers(project, [_row(onset=9.0)], operator="tester", conflict_policy="rename")
    assert renamed["imported_count"] == 1
    assert renamed["markers"][-1]["source_event_id"].startswith("001__import_")
    old_revision = project["marker_revision"]
    again = import_markers(project, [_row(onset=9.0)], operator="tester", conflict_policy="rename")
    assert again["idempotent"] is True
    assert project["marker_revision"] == old_revision
    history = list_audit_actions(project)
    assert [item["operation"] for item in history] == ["import", "import"]

    replaced = import_markers(project, [_row(onset=3.0)], operator="tester", conflict_policy="replace")
    assert replaced["changed"] is True
    assert len({item["marker_id"] for item in replaced["markers"]}) == 2


def test_edit_undo_redo_delete_and_reopen_preserve_internal_identity(tmp_path: Path):
    project = open_project(tmp_path / "project")
    imported = import_markers(project, [_row(onset=2.5)], operator="tester")
    marker = imported["markers"][0]
    marker_id = marker["marker_id"]
    internal_id = marker["event_id"]
    original_hash = marker_snapshot_hash(project)
    edited = update_marker(project, marker_id, {"onset_s": 4.0}, operator="tester", reason="move marker")
    assert edited["markers"][0]["marker_id"] == marker_id
    assert edited["markers"][0]["event_id"] == internal_id
    assert marker_snapshot_hash(project) != original_hash

    undone = undo_marker_action(project, operator="tester")
    assert undone["markers"][0]["onset_s"] == 2.5
    redone = redo_marker_action(project, operator="tester")
    assert redone["markers"][0]["onset_s"] == 4.0

    delete_marker(project, marker_id, operator="tester", reason="remove")
    assert get_markers(project) == []
    undo_marker_action(project, operator="tester")
    reopened = open_project(project["path"])
    rows = export_marker_snapshot(reopened)
    assert rows[0]["marker_id"] == marker_id
    assert rows[0]["event_id"] == internal_id
    assert rows[0]["source_event_id"] == "001"


def test_csv_import_preserves_leading_zero_ids_and_exports_provenance(tmp_path: Path):
    source = tmp_path / "events.csv"
    source.write_text(
        "recording_id,event_id,onset_s,duration_s,event_type,channel,clock_domain\n"
        "r1,001,1.25,0.5,stimulus,c1,device\n", encoding="utf-8",
    )
    rows, report = read_marker_csv(source)
    assert rows[0]["event_id"] == "001"
    assert report["row_count"] == 1
    project = open_project(tmp_path / "project")
    import_markers(project, rows, source_path=source, operator="tester")
    output = tmp_path / "markers-normalized.csv"
    exported = export_marker_snapshot(project, output)
    assert exported[0]["source_event_id"] == "001"
    with output.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        saved = next(reader)
    assert saved["event_id"].startswith("mk_")
    assert saved["source_event_id"] == "001"
    assert saved["marker_id"] == str(uuid.UUID(saved["marker_id"]))


def test_source_csv_rows_are_replaced_after_rename_and_do_not_resurrect_after_delete(tmp_path: Path):
    source = tmp_path / "legacy-events.csv"
    source.write_text(
        "recording_id,event_id,onset_s,duration_s,event_type,channel,clock_domain\n"
        "r1,001,1.25,0.5,stimulus,c1,device\n", encoding="utf-8",
    )
    source_bytes = source.read_bytes()
    source_rows, source_report = read_marker_csv(source)
    project = open_project(tmp_path / "project")
    marker = import_markers(project, source_rows, source_path=source,
                             operator="tester")["markers"][0]

    update_marker(project, marker["marker_id"], {"source_event_id": "002"},
                  operator="tester", reason="rename imported source event")
    renamed_rows = merge_marker_event_rows(source_rows, source_report,
                                           export_marker_snapshot(project),
                                           list_marker_source_tombstones(project))
    assert [row["source_event_id"] for row in renamed_rows] == ["002"]
    assert renamed_rows[0]["event_id"] == marker["event_id"]
    assert source.read_bytes() == source_bytes

    delete_marker(project, marker["marker_id"], operator="tester", reason="delete imported source event")
    deleted_rows = merge_marker_event_rows(source_rows, source_report,
                                           export_marker_snapshot(project),
                                           list_marker_source_tombstones(project))
    assert deleted_rows == []
    assert len(list_marker_source_tombstones(project)) == 1
    assert source.read_bytes() == source_bytes

    undo_marker_action(project, operator="tester", reason="restore deleted imported event")
    restored_rows = merge_marker_event_rows(source_rows, source_report,
                                            export_marker_snapshot(project),
                                            list_marker_source_tombstones(project))
    assert len(restored_rows) == 1
    assert restored_rows[0]["source_event_id"] == "002"
    assert restored_rows[0]["event_id"] == marker["event_id"]


def test_reimport_same_source_after_rename_or_delete_does_not_duplicate_or_resurrect(tmp_path: Path):
    source = tmp_path / "legacy-events.csv"
    source.write_text(
        "recording_id,event_id,onset_s,duration_s,event_type,channel,clock_domain\n"
        "r1,001,1.25,0.5,stimulus,c1,device\n", encoding="utf-8",
    )
    source_bytes = source.read_bytes()
    source_rows, _source_report = read_marker_csv(source)
    project = open_project(tmp_path / "project")
    marker = import_markers(project, source_rows, source_path=source,
                             operator="tester")["markers"][0]
    marker_id = marker["marker_id"]
    internal_id = marker["event_id"]

    update_marker(project, marker_id, {"source_event_id": "002"},
                  operator="tester", reason="rename imported source event")
    renamed_revision = project["marker_revision"]
    repeated = import_markers(project, source_rows, source_path=source, operator="tester")
    assert repeated["changed"] is False
    assert project["marker_revision"] == renamed_revision
    assert len(repeated["markers"]) == 1
    assert repeated["markers"][0]["source_event_id"] == "002"
    assert repeated["markers"][0]["source_event_id_original"] == "001"
    assert repeated["markers"][0]["marker_id"] == marker_id
    assert repeated["markers"][0]["event_id"] == internal_id

    delete_marker(project, marker_id, operator="tester", reason="delete imported source event")
    deleted_revision = project["marker_revision"]
    repeated_after_delete = import_markers(project, source_rows, source_path=source, operator="tester")
    assert repeated_after_delete["changed"] is False
    assert "source_row_previously_deleted" in repeated_after_delete["warnings"]
    assert repeated_after_delete["markers"] == []
    assert project["marker_revision"] == deleted_revision
    assert source.read_bytes() == source_bytes


def test_unmanaged_source_row_conflict_is_not_silently_replaced_by_merge(tmp_path: Path):
    source = tmp_path / "legacy-events.csv"
    source.write_text(
        "recording_id,event_id,onset_s,duration_s,event_type,channel,clock_domain\n"
        "r1,001,1.25,0.5,stimulus,c1,device\n", encoding="utf-8",
    )
    source_rows, source_report = read_marker_csv(source)
    project = open_project(tmp_path / "project")
    unrelated = import_markers(project, [_row(event_id="002", channel="c1")], operator="tester")["markers"]
    rows = merge_marker_event_rows(source_rows, source_report, export_marker_snapshot(project),
                                   list_marker_source_tombstones(project))
    assert [row["source_event_id"] for row in rows] == ["001", "002"]


def test_editing_channel_changes_marker_scope_without_changing_identity(tmp_path: Path):
    project = open_project(tmp_path / "project")
    marker = import_markers(project, [_row(channel="c1")], operator="tester")["markers"][0]
    edited = update_marker(project, marker["marker_id"], {"channel": "c3"},
                           operator="tester", reason="change channel scope")
    updated = next(row for row in edited["markers"] if row["marker_id"] == marker["marker_id"])
    assert edited["changed"] is True
    assert updated["channel_scope"] == "c3"
    assert updated["event_id"] == marker["event_id"]
    assert get_markers(project)[0]["channel_scope"] == "c3"


def test_user_event_id_with_mk_prefix_is_not_inferred_as_internal_id(tmp_path: Path):
    project = open_project(tmp_path / "project")
    marker = import_markers(project, [_row(event_id="mk_ordinary_user_label")], operator="tester")["markers"][0]
    assert marker["source_event_id"] == "mk_ordinary_user_label"
    assert marker["event_id"] == "mk_" + uuid.UUID(marker["marker_id"]).hex
    assert marker["event_id"] != marker["source_event_id"]

