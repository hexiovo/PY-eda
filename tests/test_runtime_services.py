import json
import uuid
import copy

import pandas as pd

from eda_toolkit.config import get_preset
from eda_toolkit.pipeline import _channel_scoped_associated
from eda_toolkit.runtime_events import estimate_progress, emit_run_event
from eda_toolkit.runtime_logging import create_runtime_session, export_logs, load_eta_history, search_logs
from eda_toolkit.runtime_preflight import preflight_run, validate_preflight_snapshot
import eda_toolkit.runtime_preflight as runtime_preflight
from eda_toolkit.capabilities import get_capabilities


def test_runtime_progress_requires_real_work_and_final_manifest():
    plan = {"revision": 1, "total_units": 1, "eta_min_completed_units": 3,
            "units": [{"work_id": "rec::eda:qc", "stage": "qc", "unit_type": "record_stage",
                       "method": "quality_mask", "rate_hz": 4.0}]}
    base = {"schema_version": "1.0", "event_id": str(uuid.uuid4()), "seq": 1,
            "timestamp_utc": "2026-10-26T00:00:00+00:00", "monotonic_s": 1.0,
            "level": "INFO", "project_id": None, "task_id": None, "attempt_id": None,
            "run_id": "run-1", "record_id": "rec", "units_done": 1,
            "units_total": 1, "unit_type": "record_stage"}
    completed = {**base, "stage": "qc", "kind": "completed", "message": "done",
                 "details": {"work_id": "rec::eda:qc", "method": "quality_mask", "rate_hz": 4.0,
                             "duration_s": 0.2}}
    before_final_manifest = {**base, "seq": 2, "stage": "finished", "kind": "completed",
                             "message": "finished", "units_done": None, "units_total": None,
                             "unit_type": "", "details": {"run_status": "succeeded",
                                                             "manifest_finalized": False}}
    assert estimate_progress([completed, before_final_manifest], plan, [])['percent'] == 99.0
    after_final_manifest = {**before_final_manifest, "seq": 3,
                            "details": {"run_status": "succeeded", "manifest_finalized": True}}
    assert estimate_progress([completed, after_final_manifest], plan, [])['percent'] == 100.0
    trailing_compat_progress = {**base, "seq": 4, "stage": "finished", "kind": "progress",
                                "units_done": 0, "units_total": None, "unit_type": "",
                                "details": {"legacy_pipeline_progress": {"stage": "finished",
                                                                              "completed": 1,
                                                                              "total": 1}}}
    assert estimate_progress([completed, after_final_manifest, trailing_compat_progress],
                             plan, [])['percent'] == 100.0
    unknown_plan = {"revision": 1, "total_units": None, "units": []}
    assert estimate_progress([completed], unknown_plan, [])['percent'] is None
    failed = {**before_final_manifest, "seq": 4, "kind": "failed",
              "details": {"run_status": "failed", "manifest_finalized": True}}
    assert estimate_progress([completed, failed], plan, [])['percent'] == 99.0
    failed_progress = {**trailing_compat_progress, "seq": 5}
    assert estimate_progress([completed, failed, failed_progress], plan, [])['percent'] == 99.0
    cancelled = {**before_final_manifest, "seq": 6, "kind": "cancelled",
                 "details": {"run_status": "cancelled", "manifest_finalized": False}}
    cancelled_progress = {**trailing_compat_progress, "seq": 7}
    assert estimate_progress([completed, cancelled, cancelled_progress],
                             plan, [])['percent'] == 99.0


def test_known_work_plan_ignores_unbound_preflight_completion():
    plan = {"revision": 1, "total_units": 22, "eta_min_completed_units": 3,
            "units": [{"work_id": f"rec-{index}::eda:qc", "stage": "qc",
                       "unit_type": "record_stage", "method": "quality_mask", "rate_hz": 4.0}
                      for index in range(22)]}
    preflight = {"seq": 1, "stage": "preflight", "kind": "completed", "units_done": 1,
                 "units_total": 1, "unit_type": "preflight_step", "details": {}}
    progress = estimate_progress([preflight], plan, [])
    assert progress["global_done"] == 0
    assert progress["percent"] == 0

    actual_work = {**preflight, "seq": 2, "stage": "qc", "unit_type": "record_stage",
                   "details": {"work_id": "rec-0::eda:qc", "duration_s": 0.2}}
    progress = estimate_progress([preflight, actual_work], plan, [])
    assert progress["global_done"] == 1
    assert progress["percent"] == 100 / 22

    unknown_plan = {"revision": 1, "total_units": None, "units": []}
    legacy = estimate_progress([preflight], unknown_plan, [])
    assert legacy["global_done"] == 1
    assert legacy["percent"] is None


def test_runtime_session_orders_persists_searches_and_exports(tmp_path):
    project = tmp_path / "project"
    session = create_runtime_session(project, install_hooks=False)
    seen = []
    session['subscribe'](seen.append)
    try:
        for index in range(3):
            emit_run_event(session['event_sink'], {
                "run_id": "run-a", "stage": "qc", "kind": "completed",
                "message": f"work {index}", "units_done": 1, "units_total": 3,
                "unit_type": "record_stage",
                "details": {"work_id": f"rec-{index}::eda:qc", "method": "quality_mask",
                            "rate_hz": 4.0, "duration_s": 0.1 + index / 100},
            })
        session['flush']()
        assert [row['seq'] for row in seen] == [1, 2, 3]
        assert all(row['project_id'] == session['project_id'] for row in seen)
        rows, cursor = search_logs(project, query="work", run_id="run-a", limit=2)
        assert len(rows) == 2 and cursor == "2"
        rows2, cursor2 = search_logs(project, query="work", run_id="run-a", limit=2, cursor=cursor)
        assert len(rows2) == 1 and cursor2 is None
        history = load_eta_history(project, exclude_run_id="other")
        assert len(history) == 3 and history[0]['unit_type'] == "record_stage"
        destination = tmp_path / "events.jsonl"
        assert export_logs(project, "jsonl", destination, run_id="run-a") == str(destination.resolve())
        assert len(destination.read_text(encoding="utf-8").splitlines()) == 3
    finally:
        session['flush']()
        session['close']()


def test_preflight_rechecks_actual_input_bytes_and_revisions(tmp_path):
    signal = tmp_path / "信号.csv"
    signal.write_text("time_s,eda_uS\n0,1.0\n0.25,1.1\n0.5,1.2\n", encoding="utf-8")
    config = get_preset("single_standard")
    config['input']['base_directory'] = str(tmp_path)
    config['input']['signals'] = [{"file_path": str(signal), "recording_id": "001",
                                   "subject_id": "S001", "session_id": "A",
                                   "sampling_rate_hz": 4, "eda_unit": "uS",
                                   "clock_domain": "recording_local"}]
    config['output']['directory'] = str(tmp_path / "results")
    config['workspace']['project_directory'] = str(tmp_path / "project")
    state = {"project_id": str(uuid.uuid4()), "project_revision": 2,
             "draft_revision": 3, "marker_revision": 4, "qc_revision": 5}
    report = preflight_run(config, None, state)
    assert report['eligible_to_enqueue'], report['errors']
    assert validate_preflight_snapshot(report, config, None, state) == {"valid": True, "stale": False, "errors": []}
    signal.write_text("time_s,eda_uS\n0,1.0\n0.25,1.1\n0.5,1.3\n", encoding="utf-8")
    stale = validate_preflight_snapshot(report, config, None, state)
    assert not stale['valid'] and stale['stale']
    changed_state = {**state, "marker_revision": 5}
    stale_revision = validate_preflight_snapshot(report, config, None, changed_state)
    assert not stale_revision['valid'] and stale_revision['stale']


def test_preflight_rejects_runtime_semantic_drift_but_ignores_volatile_probes(tmp_path, monkeypatch):
    signal = tmp_path / "signal.csv"
    signal.write_text("time_s,eda_uS\n0,1.0\n0.25,1.1\n0.5,1.2\n", encoding="utf-8")
    config = get_preset("single_standard")
    config['input']['base_directory'] = str(tmp_path)
    config['input']['signals'] = [{"file_path": str(signal), "recording_id": "001",
                                   "subject_id": "S001", "session_id": "A",
                                   "sampling_rate_hz": 4, "eda_unit": "uS",
                                   "clock_domain": "recording_local"}]
    config['output']['directory'] = str(tmp_path / "results")
    config['workspace']['project_directory'] = str(tmp_path / "project")
    state = {"project_id": str(uuid.uuid4()), "project_revision": 0,
             "draft_revision": 0, "marker_revision": 0, "qc_revision": 0}
    report = preflight_run(config, None, state)
    assert report['eligible_to_enqueue'], report['errors']
    assert validate_preflight_snapshot(report, config, None, state)['valid']

    real_get_capabilities = runtime_preflight.get_capabilities
    for field, code in (("source_fingerprint", "preflight_source_stale"),
                        ("numerical_environment_fingerprint", "preflight_environment_stale"),
                        ("method", "preflight_capabilities_stale")):
        def changed_capabilities(*args, _field=field, **kwargs):
            current = copy.deepcopy(real_get_capabilities(*args, **kwargs))
            if _field == "method":
                current['methods']['filter_separation']['enabled'] = False
            else:
                current[_field] = "0" * 64
            return current

        monkeypatch.setattr(runtime_preflight, "get_capabilities", changed_capabilities)
        stale = validate_preflight_snapshot(report, config, None, state)
        assert not stale['valid'] and stale['stale']
        assert code in {issue['code'] for issue in stale['errors']}

    def changed_only_volatile_probes(*args, **kwargs):
        current = copy.deepcopy(get_capabilities(*args, **kwargs))
        current['memory']['available_mb'] = 1
        current['gpu']['probe_status'] = "query_unavailable"
        current['gpu']['devices'] = ["transient-device-probe"]
        current['output_directory'] = {"writable": True, "disk_free_bytes": 1}
        return current

    monkeypatch.setattr(runtime_preflight, "get_capabilities", changed_only_volatile_probes)
    assert validate_preflight_snapshot(report, config, None, state) == {
        "valid": True, "stale": False, "errors": []}


def test_channel_scoped_tables_keep_broadcast_and_exact_channel_rows():
    events = pd.DataFrame([
        {"recording_id": "r1", "event_id": "001", "channel": "eda-a", "event_type": "a"},
        {"recording_id": "r1", "event_id": "002", "channel": "eda-b", "event_type": "b"},
        {"recording_id": "r1", "event_id": "003", "channel": "*", "event_type": "all"},
        {"recording_id": "r1", "event_id": "004", "channel": None, "event_type": "legacy"},
    ])
    associated = {"events": events, "conditions": events.copy(), "manual_qc": events.copy()}
    scoped = _channel_scoped_associated(associated, "eda-a")
    assert scoped['events']['event_id'].tolist() == ["001", "003", "004"]
    assert scoped['conditions']['event_id'].tolist() == ["001", "003", "004"]
    assert scoped['manual_qc']['event_id'].tolist() == ["001", "003", "004"]
    assert len(associated['events']) == 4
