import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from eda_toolkit.core_io import (
    export_template,
    list_templates,
    load_recordings,
    preview_csv,
    read_signal_csv,
    read_template,
    validate_table,
)
from eda_toolkit.core_qc import append_annotation, build_quality_mask, redo_annotation, undo_annotation
from eda_toolkit.contracts import QUALITY_FACTS, make_record


def _write_csv(path, text, *, encoding="utf-8-sig"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(encoding))
    return path


def test_chinese_bom_leading_zero_missingness_and_bounded_preview(tmp_path):
    source = _write_csv(
        tmp_path / "中文路径" / "皮电.csv",
        "recording_id;subject_id;session_id;channel;time_s;eda_uS;timezone;note\n"
        "REC_001;001;S01;EDA;0.0;5.5;+08:00;keep me\n"
        "REC_001;001;S01;EDA;0.1;NA;+08:00;raw NA\n"
        "REC_001;001;S01;EDA;0.2;6.0;+08:00;last\n",
    )
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    preview = preview_csv(source, max_rows=1)
    samples, metadata, report = read_signal_csv(source)

    assert preview["total_rows"] == 3
    assert len(preview["preview"]) == 1
    assert preview["delimiter"] == ";"
    assert samples["subject_id"].tolist() == ["001"] * 3
    assert np.isnan(samples.loc[1, "eda_uS"])
    assert samples.loc[0, "extras"]["note"] == "keep me"
    assert samples.loc[0, "note"] == "keep me"
    assert samples["source_time"].tolist() == ["0.0", "0.1", "0.2"]
    assert metadata["timezone"] == "+08:00"
    assert metadata["unit"] == "uS"
    assert metadata["source_sha256"] == before == report["source_sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_custom_amplitude_column_requires_explicit_unit(tmp_path):
    source = _write_csv(tmp_path / "custom.csv", "t;conductance\n0;0.005\n1;0.006\n")
    mapping = {"columns": {"time_s": "t", "eda_uS": "conductance"}}

    with pytest.raises(ValueError, match="必须明确声明 eda_unit"):
        read_signal_csv(source, mapping=mapping, options={"input": {"delimiter": ";"}, "mapping": {"time_unit": "s"}})

    samples, metadata, _ = read_signal_csv(
        source,
        mapping={**mapping, "eda_unit": "mS", "time_unit": "s"},
        options={"input": {"delimiter": ";"}},
    )
    assert samples["eda_uS"].tolist() == pytest.approx([5.0, 6.0])
    assert metadata["source_unit"] == "mS"


@pytest.mark.parametrize("times", ["0,0,1", "0,2,1"])
def test_duplicate_or_reversed_time_is_rejected_without_sorting(tmp_path, times):
    time_values = times.split(",")
    rows = "\n".join(f"REC,001,S,EDA,{t},5" for t in time_values)
    source = _write_csv(tmp_path / f"time-{times.replace(',', '-')}.csv", "recording_id,subject_id,session_id,channel,time_s,eda_uS\n" + rows + "\n")

    with pytest.raises(ValueError, match="严格递增"):
        read_signal_csv(source)


def test_missing_time_requires_rate_and_declared_unit(tmp_path):
    source = _write_csv(tmp_path / "no-time.csv", "recording_id,eda_uS\nREC,5\nREC,6\n")
    with pytest.raises(ValueError, match="sampling_rate_hz"):
        read_signal_csv(source, options={"mapping": {"time_unit": "s"}})

    samples, metadata, _ = read_signal_csv(
        source,
        options={"mapping": {"sampling_rate_hz": 10, "time_unit": "ms"}},
    )
    assert samples["time_s"].tolist() == pytest.approx([0.0, 0.1])
    assert metadata["fs_declared"] == pytest.approx(10)
    assert metadata["time_unit"] == "ms"


def test_relative_time_subtracts_high_precision_origin_before_float_conversion(tmp_path):
    source = _write_csv(
        tmp_path / "high-precision.csv",
        "recording_id,time_s,eda_uS\nREC,1700000000000000.0000001,5\nREC,1700000000000000.0000002,6\n",
    )
    samples, metadata, _ = read_signal_csv(source)

    assert samples["time_s"].tolist() == pytest.approx([0.0, 1e-7], abs=1e-14)
    assert samples["source_time"].tolist() == ["1700000000000000.0000001", "1700000000000000.0000002"]
    assert metadata["time_origin"] == "1700000000000000.0000001"
    assert metadata["time_origin_generated"] is True


def test_resistance_conversion_rejects_zero_and_negative(tmp_path):
    source = _write_csv(tmp_path / "resistance.csv", "recording_id,time_s,R\nREC,0,100000\nREC,1,0\n")
    mapping = {"columns": {"eda_uS": "R"}, "eda_unit": "ohm", "time_unit": "s"}
    with pytest.raises(ValueError, match="正值"):
        read_signal_csv(source, mapping=mapping)


@pytest.mark.parametrize(
    ("unit", "raw", "expected"),
    [("S", 0.00001, 10.0), ("ohm", 100000.0, 10.0), ("kohm", 100.0, 10.0), ("Mohm", 0.1, 10.0)],
)
def test_declared_conductance_and_resistance_units_convert_to_micro_siemens(tmp_path, unit, raw, expected):
    source = _write_csv(tmp_path / f"unit-{unit}.csv", f"t;signal\n0;{raw}\n1;{raw}\n")
    samples, metadata, _ = read_signal_csv(
        source,
        mapping={"columns": {"time_s": "t", "eda_uS": "signal"}, "eda_unit": unit, "time_unit": "s"},
        options={"input": {"delimiter": ";"}},
    )
    assert samples["eda_uS"].tolist() == pytest.approx([expected, expected])
    assert metadata["source_unit"] == unit


def test_templates_have_synthetic_examples_and_schema_dictionary(tmp_path):
    templates = list_templates()
    assert len(templates) == 9
    assert all(item["example_label"] == "SYNTHETIC_ONLY" for item in templates)
    example = read_template("signals", "example")
    assert example.loc[0, "subject_id"] == "SYNTHETIC_SUBJECT_001"
    assert validate_table("signals", example)["valid"]
    schema = read_template("sync_anchors", "schema")
    reference_clock = next(field for field in schema["fields"] if field["name"] == "reference_clock_domain")
    assert reference_clock["required"] is True
    design = read_template("statistical_design", "example")
    assert design.loc[0, "outcome"] == "mean"
    assert json.loads(design.loc[0, "contrast"]) == {
        "conditions": ["SYNTHETIC_CONDITION_A", "SYNTHETIC_CONDITION_B"]
    }
    assert validate_table("statistical_design", design)["valid"]
    exported = export_template("events", tmp_path / "events.csv")
    assert exported.read_text(encoding="utf-8-sig").startswith("recording_id,event_id")


def test_load_recordings_validates_associated_foreign_keys_and_clock_domain(tmp_path):
    signal = _write_csv(
        tmp_path / "signal.csv",
        "recording_id,subject_id,session_id,channel,time_s,eda_uS,clock_domain\n"
        "REC,001,S1,EDA,0,5,local\nREC,001,S1,EDA,1,6,local\n",
    )
    events = pd.DataFrame([{
        "recording_id": "REC", "event_id": "E1", "onset_s": 0.5, "duration_s": 0.2,
        "event_type": "stimulus", "condition": None, "channel": "EDA", "clock_domain": "other",
    }])
    options = {"input": {"signals": [str(signal)], "events": events}, "mapping": {"time_unit": "s"}}

    with pytest.raises(ValueError, match="关系校验失败"):
        load_recordings(options=options)


def test_load_recordings_rejects_conflicting_duplicate_participant_metadata(tmp_path):
    signal = _write_csv(
        tmp_path / "signal.csv",
        "recording_id,subject_id,session_id,channel,time_s,eda_uS\nREC,001,S1,EDA,0,5\nREC,001,S1,EDA,1,6\n",
    )
    participants = pd.DataFrame([
        {"subject_id": "001", "experimental_group": "A"},
        {"subject_id": "001", "experimental_group": "B"},
    ])
    options = {"input": {"signals": [str(signal)], "participants": participants}, "mapping": {"time_unit": "s"}}

    with pytest.raises(ValueError, match="属性记录冲突"):
        load_recordings(options=options)


def test_load_recordings_returns_typed_empty_optional_manual_qc_and_consumed_hashes(tmp_path):
    signal = _write_csv(
        tmp_path / "signal.csv",
        "recording_id,subject_id,session_id,channel,time_s,eda_uS\nREC,001,S1,EDA,0,5\nREC,001,S1,EDA,1,6\n",
    )
    before = hashlib.sha256(signal.read_bytes()).hexdigest()
    options = {
        "input": {"signals": [str(signal)]},
        "mapping": {"time_unit": "s"},
        "qc": {"annotations": []},
    }

    records, tables, report = load_recordings(options=options)

    assert len(records) == 1
    assert records[0]["metadata"]["source_path"] == str(signal.resolve())
    assert records[0]["metadata"]["source_sha256"] == before
    assert tables["manual_qc"].empty
    assert {"annotation_id", "recording_id", "action"}.issubset(tables["manual_qc"].columns)
    assert report["consumed_inputs"][str(signal.resolve())] == before


def test_quality_mask_preserves_missingness_manual_precedence_and_append_only_history():
    record = make_record(
        [0, 1, 2, 3, 4], [1.0, np.nan, 1.0, 3.0, 3.0],
        {"recording_id": "REC", "subject_id": "001", "session_id": "S1", "channel": "EDA", "unit": "uS", "clock_domain": "local"},
    )
    config = {"qc": {"min_uS": 2.0, "flatline_duration_s": 10.0, "exclude_auto": True, "gap_factor": 1.5}}
    history = []
    history = append_annotation(history, {
        "annotation_id": "exclude-1", "recording_id": "REC", "channel": "EDA", "start_s": 2,
        "end_s": 3, "action": "exclude", "reason": "artifact", "operator": "reviewer",
        "timestamp": "2026-10-25T00:00:00+00:00", "clock_domain": "local",
    }, expected_parent_revision=0)
    history = append_annotation(history, {
        "annotation_id": "restore-1", "recording_id": "REC", "channel": "EDA", "start_s": 0,
        "end_s": 1, "action": "restore", "reason": "checked raw trace", "operator": "reviewer",
        "timestamp": "2026-10-25T00:01:00+00:00", "clock_domain": "local",
    }, expected_parent_revision=1)

    qc = build_quality_mask(record, config, history)
    assert set(qc["facts"]) == set(QUALITY_FACTS)
    assert qc["facts"]["original_missing"][1]
    assert qc["facts"]["manual_restore"][0]
    assert qc["facts"]["manual_exclude"][2]
    assert qc["eligible"][0]
    assert not qc["eligible"][1]  # restore never reanimates NaN
    assert not qc["eligible"][2]  # explicit manual exclusion wins
    assert qc["candidate_reasons"]["below_min_uS"][0]

    undone = undo_annotation(history, "exclude-1", expected_parent_revision=2, operator="reviewer", reason="undo review")
    redone = redo_annotation(undone, "exclude-1", expected_parent_revision=3, operator="reviewer", reason="restore decision")
    assert [entry["action"] for entry in redone] == ["exclude", "restore", "undo", "redo"]
    with pytest.raises(ValueError, match="父版本冲突"):
        append_annotation(history, {
            "annotation_id": "stale", "recording_id": "REC", "start_s": 0, "end_s": 1,
            "action": "exclude", "reason": "stale", "operator": "reviewer",
            "timestamp": "2026-10-25T00:02:00+00:00", "clock_domain": "local",
        }, expected_parent_revision=1)


def test_quality_mask_splits_on_large_acquisition_gap_and_tracks_overlapping_reasons():
    record = make_record(
        [0.0, 0.1, 0.2, 1.0, 1.1], [0.0, 0.0, 0.0, 5.0, 5.0],
        {"recording_id": "REC", "subject_id": "001", "session_id": "S1", "channel": "EDA", "unit": "uS", "clock_domain": "local"},
    )
    qc = build_quality_mask(record, {"qc": {
        "min_uS": 1.0, "flatline_duration_s": 0.15, "flatline_tolerance_uS": 0.0,
        "gap_factor": 1.5, "exclude_auto": False,
    }})
    assert qc["segments"] == [(0, 3), (3, 5)]
    assert qc["candidate_reasons"]["below_min_uS"][1]
    assert qc["candidate_reasons"]["flatline"][1]
    assert qc["summary"]["candidate_counts_by_reason"]["below_min_uS"] == 3
    assert len(qc["gap_boundaries"]) == 1
