import json
import pandas as pd

from eda_toolkit.core_export import atomic_json, export_run, spreadsheet_safe_table, verify_manifest


def test_exports_preserve_numeric_and_detect_mutation(tmp_path):
    table = pd.DataFrame({"identifier": ["001", "=1+1"], "value": [-2.0, float("nan")]})
    safe = spreadsheet_safe_table(table)
    assert safe.iloc[0].value == -2
    assert safe.iloc[1].identifier == "'=1+1"
    config = {"output": {"formats": ["csv", "json", "xlsx", "html"], "spreadsheet_safe": True}}
    manifest = export_run({"summary": {"run_id": "r", "status": "succeeded"}, "tables": {"features": table}},
                          {"config_sha256": "x", "environment": {}}, tmp_path, config)
    assert verify_manifest(tmp_path)["valid"]
    atomic_json(tmp_path / "pipeline_state.json", {"status": "succeeded"})
    assert verify_manifest(tmp_path)["valid"]
    content = next((tmp_path / "tables").glob("*.json")).read_text(encoding="utf-8")
    assert json.loads(content)[1]["value"] is None
    next((tmp_path / "tables").glob("*.csv")).write_text("changed", encoding="utf-8")
    assert not verify_manifest(tmp_path)["valid"]
