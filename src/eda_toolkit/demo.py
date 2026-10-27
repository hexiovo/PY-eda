"""显式合成演示数据；不包含真实被试或科学资格结论。"""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import get_preset, save_config


def create_demo(directory, *, duration_s=600, sampling_rate_hz=4, seed=2026):
    """模拟慢变基线、已知Bateman型响应和独立噪声，仅供工程演示。"""
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("演示目录必须为空，避免覆盖用户文件。")
    rng = np.random.default_rng(seed)
    time = np.arange(int(duration_s * sampling_rate_hz) + 1) / sampling_rate_hz
    onsets = np.arange(50, duration_s - 40, 45)
    rows, events, conditions, anchors, members, participants = [], [], [], [], [], []
    for index in range(4):
        subject, recording = f"{index + 1:03d}", f"synthetic_{index + 1:03d}"
        delay = .75 * (index % 2)
        signal = 2 + .12 * np.sin(2 * np.pi * time / 180) + index * .15
        for onset in onsets:
            elapsed = np.maximum(0, time - onset - delay)
            signal += .5 * (np.exp(-elapsed / 3) - np.exp(-elapsed / .6)) * (time >= onset + delay)
        signal += rng.normal(0, .004, len(time))
        pd.DataFrame({"subject_id": subject, "session_id": "demo", "recording_id": recording,
                      "time_s": time, "eda_uS": signal}).to_csv(root / f"{recording}.csv", index=False, encoding="utf-8-sig")
        rows.append({"file_path": f"{recording}.csv", "recording_id": recording, "subject_id": subject,
                     "session_id": "demo", "sampling_rate_hz": sampling_rate_hz, "eda_unit": "uS", "clock_domain": "recording_local"})
        participants.append({"subject_id": subject, "experimental_group": "synthetic", "age": 20 + index, "sex": "synthetic"})
        members.append({"group_id": f"G{index // 2 + 1}", "session_id": "demo", "subject_id": subject,
                        "recording_id": recording, "role": "A" if index % 2 == 0 else "B",
                        "member_slot": "A" if index % 2 == 0 else "B", "task_id": "synthetic_task",
                        "stimulus_timeline_id": "synthetic_identical_onsets", "task_time_origin_s": 0.0})
        for position, onset in enumerate(onsets):
            events.append({"recording_id": recording, "event_id": f"e{position + 1}", "onset_s": float(onset),
                           "duration_s": 0, "event_type": "synthetic_pulse", "condition": "demo", "clock_domain": "recording_local"})
        conditions.append({"recording_id": recording, "interval_id": "demo_interval", "start_s": 0,
                           "end_s": duration_s, "condition": "demo", "clock_domain": "recording_local"})
        for endpoint in (0, duration_s / 2, duration_s):
            anchors.append({"recording_id": recording, "anchor_id": f"a{endpoint}", "local_time_s": endpoint,
                            "reference_time_s": endpoint, "clock_domain": "recording_local",
                            "reference_clock_domain": "shared_reference", "uncertainty_s": 0.0, "quality": "synthetic_exact"})
    for name, values in (("input_manifest", rows), ("participants", participants), ("events", events),
                         ("conditions", conditions), ("group_membership", members), ("sync_anchors", anchors)):
        pd.DataFrame(values).to_csv(root / f"{name}.csv", index=False, encoding="utf-8-sig")
    config = get_preset()
    config["input"].update({"manifest": "input_manifest.csv", "participants": "participants.csv", "events": "events.csv",
                            "conditions": "conditions.csv", "base_directory": str(root)})
    config["mapping"].update({"eda_unit": "uS", "sampling_rate_hz": sampling_rate_hz})
    save_config(config, root / "single.json")
    group = get_preset("group_sync")
    group.update({"input": {**config["input"], "group_membership": "group_membership.csv", "sync_anchors": "sync_anchors.csv"},
                  "mapping": config["mapping"].copy(), "output": config["output"].copy()})
    save_config(group, root / "group.json")
    statistical_conditions = []
    for row in rows:
        for condition, start, end in (("A", 0, duration_s / 2), ("B", duration_s / 2, duration_s)):
            statistical_conditions.append({"recording_id": row["recording_id"], "interval_id": condition,
                                           "start_s": start, "end_s": end, "condition": condition, "clock_domain": "recording_local"})
    pd.DataFrame(statistical_conditions).to_csv(root / "conditions_ab.csv", index=False, encoding="utf-8-sig")
    statistical = get_preset()
    statistical["input"].update({**config["input"], "events": None, "conditions": "conditions_ab.csv"})
    statistical["mapping"] = config["mapping"].copy()
    statistical["windows"].update({"whole_record": False, "fixed_length_s": None})
    statistical["features"]["ids"] = ["mean"]
    statistical["statistics"].update({"enabled": True, "designs": [{"design_id": "synthetic_lmm", "outcome": "mean",
                                         "design_type": "mixed_effects", "conditions": ["A", "B"], "independent_unit": "subject_id",
                                         "random_intercept": True, "aggregate": "mean", "correction_family": "synthetic_lmm_coefficients"}]})
    save_config(statistical, root / "statistics.json")
    (root / "README.txt").write_text("全部数据由EDA Toolkit确定性生成，仅供工程演示，不是真实被试或设备科学验证。\n"
                                    f"seed={seed}; duration={duration_s}s; fs={sampling_rate_hz}Hz\n", encoding="utf-8")
    return {"directory": str(root), "single_config": str(root / "single.json"), "group_config": str(root / "group.json"),
            "statistics_config": str(root / "statistics.json"),
            "provenance": "synthetic_engineering_demo", "seed": seed}
