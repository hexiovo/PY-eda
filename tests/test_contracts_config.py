import json

import numpy as np
import pytest

from eda_toolkit.config import get_preset, load_config, resolve_config
from eda_toolkit.contracts import continuous_segments, make_record, valid_duration


def test_contract_keeps_missingness_and_copies_original():
    source = np.array([0.0, np.nan, 2.0])
    record = make_record([0, 1, 3], source, {"subject_id": "001"})
    record["eda_uS"][0] = 9
    assert source[0] == 0
    assert np.isnan(record["eda_uS"][1])
    assert valid_duration(record["time_s"], np.isfinite(record["eda_uS"])) == 0
    assert continuous_segments([True, True, True], [0, 1, 7], 2) == [(0, 2), (2, 3)]


@pytest.mark.parametrize("config", [
    {"unknown": 1}, {"compute": {"workers": True}},
    {"qc": {"exclude_auto": "false"}}, {"preprocess": {"filter_order": 0}},
    {"windows": {"event_response_s": [4, 1]}},
    {"statistics": {"designs": [{"formula": "__import__('os')"}]}},
])
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        resolve_config(config)


def test_strict_json_and_roundtrip(tmp_path):
    config_path = tmp_path / "方案.json"
    config_path.write_text('{"compute":{"workers":1,"workers":2}}', encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_config(config_path)
    config_path.write_text('{"mapping":{"sampling_rate_hz":NaN}}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(config_path)
    event = get_preset("single_event")
    config_path.write_text(json.dumps(event), encoding="utf-8")
    assert load_config(config_path) == resolve_config(event)
    assert resolve_config()["mapping"]["eda_unit"] is None
