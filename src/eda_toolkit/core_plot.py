"""Matplotlib 研究图形：以显式 Figure/Canvas 生成独立文件，不依赖 pyplot 状态。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, fontManager


_COLORS = {"ink": "#17324D", "blue": "#47728F", "teal": "#168A8A", "aqua": "#5CB8B2",
           "muted": "#64788A", "grid": "#D9E2EA", "surface": "#F6F9FB", "orange": "#D9822B",
           "red": "#B94A48", "purple": "#7866A8", "light": "#A7C2D2"}


def _slug(value, limit=72):
    value = re.sub(r"[^\w.-]+", "_", str(value), flags=re.UNICODE).strip("._-")
    return (value or "figure")[:limit]


def _font_properties():
    bundled = (Path(__file__).resolve().parent / "resources" / "help" / "fonts" /
               "NotoSansSC-Regular.ttf")
    candidates = [bundled, Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf"),
                  Path(r"C:\Windows\Fonts\msyh.ttf"), Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf")]
    for candidate in candidates:
        if candidate.is_file():
            try:
                fontManager.addfont(str(candidate))
                properties = FontProperties(fname=str(candidate))
                family = properties.get_name()
            except (OSError, RuntimeError, ValueError):
                continue
            return properties, {"family": family, "path": str(candidate.resolve()),
                                "source": "bundled" if candidate == bundled else "system_fallback",
                                "cjk_available": True}
    for entry in fontManager.ttflist:
        if entry.name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP"):
            return FontProperties(fname=entry.fname), {"family": entry.name, "path": entry.fname,
                                                        "cjk_available": True}
    return FontProperties(family="DejaVu Sans"), {"family": "DejaVu Sans", "path": None,
                                                  "cjk_available": False}


def _apply_fonts(fig, font):
    for text in fig.findobj(match=lambda item: hasattr(item, "set_fontproperties")):
        try:
            text.set_fontproperties(font)
        except (AttributeError, TypeError):
            pass


def _finish_figure(fig, path, font):
    path.parent.mkdir(parents=True, exist_ok=True)
    _apply_fonts(fig, font)
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=_COLORS["surface"])
    fig.clear()
    return str(path.resolve())


def _frame(value):
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame(value or [])


def _metadata(bundle):
    metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    return metadata


def _record_figure(result, destination, font):
    bundle = result.get("bundle", {})
    if not isinstance(bundle, dict) or "time_s" not in bundle:
        return None
    time = np.asarray(bundle.get("time_s", []), dtype=float)
    if not len(time):
        return None
    metadata = _metadata(bundle)
    rid = metadata.get("recording_id") or result.get("summary", {}).get("recording_id") or "record"
    max_points = 4000
    stride = max(1, int(math.ceil(len(time) / max_points)))
    selection = np.arange(0, len(time), stride)
    plotted_time = time[selection]
    quality = bundle.get("quality", {}) if isinstance(bundle.get("quality", {}), dict) else {}
    signals = {}
    for name in ("clean_uS", "tonic_uS", "phasic_uS", "residual_uS"):
        value = bundle.get(name)
        if value is not None:
            array = np.asarray(value, dtype=float)
            if array.shape == time.shape:
                signals[name] = array[selection]
    figure = Figure(figsize=(13.5, 8.2), constrained_layout=True)
    ax_signal, ax_phasic, ax_coverage = figure.subplots(3, 1, sharex=True,
                                                        gridspec_kw={"height_ratios": [1.2, 1.0, 0.55]})
    raw = bundle.get("record", {})
    raw_time = np.asarray(raw.get("time_s", []), dtype=float)
    raw_signal = np.asarray(raw.get("eda_uS", []), dtype=float)
    if raw_time.size and raw_signal.shape == raw_time.shape:
        raw_stride = max(1, int(math.ceil(len(raw_time) / max_points)))
        ax_signal.plot(raw_time[::raw_stride], raw_signal[::raw_stride], color=_COLORS["muted"],
                       alpha=0.65, linewidth=0.65, label="原始电导 · μS")
    if "clean_uS" in signals:
        ax_signal.plot(plotted_time, signals["clean_uS"], color=_COLORS["blue"], linewidth=0.85,
                       label="清理电导 · μS")
    if "tonic_uS" in signals:
        ax_signal.plot(plotted_time, signals["tonic_uS"], color=_COLORS["teal"], linewidth=1.0,
                       label="tonic · μS")
    scr = _frame(result.get("tables", {}).get("scr"))
    if not scr.empty and "peak_s" in scr and "amplitude_uS" in scr:
        clean = np.asarray(bundle.get("clean_uS", []), dtype=float)
        if clean.shape == time.shape:
            peak_times, peak_values = [], []
            for peak_time in pd.to_numeric(scr["peak_s"], errors="coerce").to_numpy(dtype=float):
                if not np.isfinite(peak_time):
                    continue
                index = int(np.clip(np.searchsorted(time, peak_time), 0, len(time) - 1))
                if index and abs(time[index - 1] - peak_time) <= abs(time[index] - peak_time):
                    index -= 1
                value = clean[index]
                if np.isfinite(value):
                    peak_times.append(float(peak_time))
                    peak_values.append(float(value))
            if peak_times:
                ax_signal.scatter(peak_times, peak_values, s=22, marker="v", color=_COLORS["orange"],
                                  edgecolors="white", linewidths=0.4, label="SCR 峰", zorder=4)
                ax_signal.text(0.01, 0.02, "峰幅值与恢复指标见 SCR 表", transform=ax_signal.transAxes,
                               color=_COLORS["muted"], fontsize=8, va="bottom")
    windows = _frame(result.get("tables", {}).get("windows"))
    if "event_onset_s" in windows:
        event_times = pd.to_numeric(windows["event_onset_s"], errors="coerce").dropna().unique()
        for index, onset in enumerate(sorted(event_times)[:80]):
            ax_signal.axvline(float(onset), color=_COLORS["muted"], alpha=0.3, linewidth=0.7,
                              linestyle="--", label="事件" if index == 0 else None)
        if len(event_times) > 80:
            ax_signal.text(.01,.1,f"事件标记显示前80/{len(event_times)}；完整记录见窗口表",transform=ax_signal.transAxes,fontsize=7)
    if ax_signal.lines or ax_signal.collections:
        ax_signal.legend(loc="upper right", frameon=False, ncol=3)
    else:
        ax_signal.text(0.5, 0.5, "没有可用电导信号", transform=ax_signal.transAxes, ha="center", va="center")
    ax_signal.set_ylabel("电导 (μS)")
    ax_signal.set_title("皮电信号与分解", loc="left", color=_COLORS["ink"], fontweight="bold")

    if "phasic_uS" in signals:
        ax_phasic.plot(plotted_time, signals["phasic_uS"], color=_COLORS["purple"], linewidth=0.85,
                       label="phasic · μS")
    if "residual_uS" in signals:
        ax_phasic.plot(plotted_time, signals["residual_uS"], color=_COLORS["muted"], linewidth=0.75,
                       alpha=0.85, label="残差 · μS")
    if ax_phasic.lines:
        ax_phasic.legend(loc="upper right", frameon=False, ncol=2)
    else:
        ax_phasic.text(0.5, 0.5, "没有可用 phasic / residual 分量", transform=ax_phasic.transAxes,
                       ha="center", va="center")
    ax_phasic.axhline(0, color=_COLORS["muted"], linewidth=0.7, alpha=0.55)
    ax_phasic.set_ylabel("幅值差 (μS)")
    ax_phasic.set_title("phasic 与残差", loc="left", color=_COLORS["ink"], fontweight="bold")

    masks = (("eligible", "信号有效", _COLORS["teal"]),
             ("detector_eligible", "SCR可检测", _COLORS["blue"]),
             ("interpolated", "插值影响", _COLORS["orange"]),
             ("edge_affected", "滤波/支撑边缘", _COLORS["red"]))
    any_mask = False
    for key, label, color in masks:
        value = quality.get(key, bundle.get(key))
        if value is None:
            continue
        mask = np.asarray(value, dtype=bool)
        if mask.shape != time.shape:
            continue
        ax_coverage.step(plotted_time, mask[selection].astype(float), where="post", linewidth=1.0,
                         color=color, label=label, alpha=0.9)
        any_mask = True
    if any_mask:
        ax_coverage.set_yticks([0, 1], labels=["不可用", "可用/标记"])
        ax_coverage.set_ylim(-0.12, 1.2)
        ax_coverage.legend(loc="upper right", frameon=False, ncol=4, fontsize=8)
    else:
        ax_coverage.text(0.5, 0.5, "无质量标记", transform=ax_coverage.transAxes, ha="center", va="center")
    ax_coverage.set_ylabel("覆盖状态")
    ax_coverage.set_xlabel("时间 (s)")
    ax_coverage.set_title("有效性与处理影响", loc="left", color=_COLORS["ink"], fontweight="bold")

    selected = np.asarray(quality.get("analysis_eligible", quality.get("eligible", [])), dtype=bool)
    if selected.shape == time.shape and np.any(selected):
        valid_range = f"有效时间范围 {time[np.flatnonzero(selected)[0]]:.3f}–{time[np.flatnonzero(selected)[-1]]:.3f} s"
    else:
        valid_range = "没有可用分析区间"
    fs = bundle.get("sampling_rate_hz", metadata.get("fs_hz", metadata.get("fs_reference_hz")))
    clock = metadata.get("clock_domain", "未声明")
    subjects = metadata.get("subject_id", "")
    channel = metadata.get("channel", "eda")
    figure.suptitle(f"{rid}  ·  {subjects}  ·  {channel}  ·  fs={fs if fs is not None else 'NA'} Hz  ·  时钟={clock}\n{valid_range}",
                    color=_COLORS["ink"], fontsize=11, fontweight="bold")
    return _finish_figure(figure, destination / f"signal_{_slug(rid)}.png", font)


def _feature_figures(tables, destination, font):
    features = _frame(tables.get("features"))
    if features.empty or not {"feature_id", "value"} <= set(features.columns):
        return []
    frame = features.copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if "status" in frame:
        frame = frame.loc[frame["status"].astype(str) == "valid"]
    frame = frame.loc[np.isfinite(frame["value"].to_numpy(dtype=float))]
    if frame.empty:
        return []
    if "condition" not in frame:
        frame["condition"] = "全部"
    else:
        frame["condition"] = frame["condition"].fillna("未分组").astype(str)
    if "unit" not in frame:
        frame["unit"] = "未标单位"
    paths = []
    for unit, unit_frame in frame.groupby("unit", dropna=False, sort=True):
        counts = unit_frame.groupby("feature_id", dropna=False)["value"].count().sort_values(ascending=False)
        feature_ids = counts.index[:8].tolist()
        plot_data = unit_frame.loc[unit_frame["feature_id"].isin(feature_ids)]
        groups = list(dict.fromkeys(plot_data["condition"].tolist()))
        feature_ids = [str(item) for item in feature_ids]
        fig = Figure(figsize=(max(9.5, len(feature_ids) * 1.15), 5.6), constrained_layout=True)
        ax = fig.subplots(1, 1)
        centers = np.arange(len(feature_ids), dtype=float)
        width = min(0.78 / max(len(groups), 1), 0.24)
        for group_index, group in enumerate(groups):
            values, errors, labels = [], [], []
            for feature_id in feature_ids:
                selected = plot_data.loc[(plot_data["feature_id"].astype(str) == feature_id) &
                                         (plot_data["condition"] == group), "value"].to_numpy(dtype=float)
                values.append(float(np.mean(selected)) if len(selected) else np.nan)
                errors.append(float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0)
                labels.append(f"n={len(selected)}")
            x = centers + (group_index - (len(groups) - 1) / 2) * width
            color = [_COLORS["teal"], _COLORS["blue"], _COLORS["orange"], _COLORS["purple"], _COLORS["aqua"]][group_index % 5]
            ax.bar(x, values, width=width, yerr=errors, color=color, edgecolor="white", linewidth=0.6,
                   capsize=2, label=str(group), alpha=0.92)
        ax.axhline(0, color=_COLORS["muted"], linewidth=0.75, alpha=0.75)
        ax.set_xticks(centers, feature_ids, rotation=28, ha="right")
        ax.set_ylabel(str(unit))
        ax.set_title("有效特征的条件汇总（均值 ± 样本 SD）", loc="left", color=_COLORS["ink"], fontweight="bold")
        ax.legend(title="条件", frameon=False, ncol=min(len(groups), 5))
        ax.grid(axis="y", color=_COLORS["grid"], linewidth=0.65)
        ax.set_axisbelow(True)
        members = sorted(set(str(value) for value in unit_frame.get("subject_id", pd.Series(dtype=str)).dropna()))
        if not members:
            members = sorted(set(str(value) for value in unit_frame.get("recording_id", pd.Series(dtype=str)).dropna()))
        ax.text(0, -0.34, f"有效记录 {unit_frame['recording_id'].nunique() if 'recording_id' in unit_frame else 'NA'}  ·  实际成员: {', '.join(members) or '未提供'}  ·  仅汇总 status=valid",
                transform=ax.transAxes, color=_COLORS["muted"], fontsize=8, va="top")
        file_path = destination / f"features_{_slug(unit)}.png"
        paths.append(_finish_figure(fig, file_path, font))
    return paths


def _sync_figures(tables, destination, font):
    synchrony = _frame(tables.get("synchrony"))
    if synchrony.empty or not {"r", "lag_s"} <= set(synchrony.columns):
        return []
    frame = synchrony.copy()
    frame["r"] = pd.to_numeric(frame["r"], errors="coerce")
    frame["lag_s"] = pd.to_numeric(frame["lag_s"], errors="coerce")
    valid = frame.loc[(frame.get("status", pd.Series(index=frame.index, dtype=str)).astype(str) == "valid") &
                      np.isfinite(frame["r"].to_numpy(dtype=float)) & np.isfinite(frame["lag_s"].to_numpy(dtype=float))]
    if valid.empty:
        return []
    keys = [column for column in ("group_id", "session_id", "window_id", "metric", "basis") if column in valid]
    paths = []
    for group_key, local in valid.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(keys, group_key))
        # Signed zero-lag pair matrix; members are the actual dyad endpoints in this output table.
        if "subject_id_a" in local and "subject_id_b" in local:
            zero = local.loc[np.isclose(local["lag_s"], 0.0)]
            members = sorted(set(zero["subject_id_a"].dropna().astype(str)) |
                             set(zero["subject_id_b"].dropna().astype(str)))
            if len(members) >= 2:
                matrix = np.full((len(members), len(members)), np.nan)
                np.fill_diagonal(matrix, 1.0)
                for row in zero.to_dict(orient="records"):
                    a, b = str(row["subject_id_a"]), str(row["subject_id_b"])
                    if a in members and b in members:
                        matrix[members.index(a), members.index(b)] = float(row["r"])
                        matrix[members.index(b), members.index(a)] = float(row["r"])
                fig = Figure(figsize=(max(6.5, len(members) * 0.72), max(5.5, len(members) * 0.65)),
                             constrained_layout=True)
                ax = fig.subplots(1, 1)
                image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
                for i in range(len(members)):
                    for j in range(len(members)):
                        if np.isfinite(matrix[i, j]):
                            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                                    color="white" if abs(matrix[i, j]) > 0.55 else _COLORS["ink"], fontsize=8)
                ax.set_xticks(np.arange(len(members)), members, rotation=35, ha="right")
                ax.set_yticks(np.arange(len(members)), members)
                ax.set_title(f"零滞后有符号同步矩阵\n{group_meta}", loc="left", color=_COLORS["ink"], fontweight="bold")
                colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
                colorbar.set_label("r (−1 到 +1)")
                member_text = ", ".join(members)
                ax.text(0, -0.18, f"实际成员: {member_text}", transform=ax.transAxes, color=_COLORS["muted"], fontsize=8)
                paths.append(_finish_figure(fig, destination / f"sync_matrix_{_slug('_'.join(map(str, group_key)))}.png", font))

        # Lag-time rows retain sign; line identity is the actual pair.
        pair_key = "pair_id" if "pair_id" in local else "subject_id_a" if "subject_id_a" in local else None
        fig = Figure(figsize=(10.5, 5.6), constrained_layout=True)
        ax = fig.subplots(1, 1)
        plotted = 0
        if pair_key:
            for index, (pair, pair_rows) in enumerate(local.groupby(pair_key, dropna=False, sort=True)):
                pair_rows = pair_rows.sort_values("lag_s")
                label = str(pair)
                if "subject_id_a" in pair_rows and "subject_id_b" in pair_rows:
                    first = pair_rows.iloc[0]
                    # Keep the data pair_id intact; use a glyph covered by common
                    # UI fallback fonts for the plotted legend display only.
                    label = f"{first['subject_id_a']} / {first['subject_id_b']}"
                ax.plot(pair_rows["lag_s"], pair_rows["r"], marker="o", markersize=3.2, linewidth=1.05,
                        color=font.color if hasattr(font, "color") else None,
                        label=label if index < 12 else None)
                plotted += 1
        if plotted:
            ax.axhline(0, color=_COLORS["muted"], linewidth=0.75)
            ax.axvline(0, color=_COLORS["grid"], linewidth=0.75)
            ax.set_xlabel("滞后 (s；报告的是网格实际表示值)")
            ax.set_ylabel("有符号相关 r")
            ax.set_ylim(-1.05, 1.05)
            ax.grid(color=_COLORS["grid"], linewidth=0.6)
            ax.set_title(f"相关随滞后变化 · {group_meta} · pair数={plotted}", loc="left",
                         color=_COLORS["ink"], fontweight="bold")
            ax.legend(frameon=False, fontsize=8, ncol=2)
            paths.append(_finish_figure(fig, destination / f"sync_lags_{_slug('_'.join(map(str, group_key)))}.png", font))
        else:
            fig.clear()
    return paths


def _null_summary_figure(tables, destination, font):
    null = _frame(tables.get("sync_null_summary"))
    if null.empty or not {"observed_max_abs_r", "null_mean_max_abs_r"} <= set(null.columns):
        return None
    frame = null.copy()
    for column in ("observed_max_abs_r", "null_mean_max_abs_r", "null_sd_max_abs_r"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[np.isfinite(frame["observed_max_abs_r"]) & np.isfinite(frame["null_mean_max_abs_r"])]
    if frame.empty:
        return None
    labels = []
    for row in frame.to_dict(orient="records"):
        labels.append(" · ".join(str(row.get(key)) for key in ("group_id", "session_id", "window_id", "pair_id")
                                 if row.get(key) not in (None, "", np.nan)))
    fig = Figure(figsize=(max(9, len(frame) * 0.58), 5.4), constrained_layout=True)
    ax = fig.subplots(1, 1)
    x = np.arange(len(frame))
    sd = frame["null_sd_max_abs_r"].fillna(0).to_numpy(dtype=float)
    ax.errorbar(x - 0.08, frame["null_mean_max_abs_r"], yerr=sd, fmt="o", color=_COLORS["blue"],
                capsize=3, label="surrogate 均值 ± SD")
    ax.scatter(x + 0.08, frame["observed_max_abs_r"], marker="D", color=_COLORS["orange"], s=32,
               label="观察到的 max |r|")
    ax.set_xticks(x, labels, rotation=36, ha="right")
    ax.set_ylabel("max |r|")
    ax.set_title("循环移位 null 的分布摘要", loc="left", color=_COLORS["ink"], fontweight="bold")
    ax.text(0, -0.38, "只绘制已导出的 null mean/SD/observed summary；当前输出不含逐次 surrogate 样本，未构造直方图或正态分布。",
            transform=ax.transAxes, color=_COLORS["muted"], fontsize=8, va="top")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=_COLORS["grid"], linewidth=0.65)
    return _finish_figure(fig, destination / "sync_null_summary.png", font)


def _clock_figures(records, tables, destination, font):
    rows = []
    for result in records:
        metadata = _metadata(result.get("bundle", {}))
        if not metadata.get("clock_domain") and not metadata.get("clock_alignment_status"):
            continue
        rows.append({"recording_id": metadata.get("recording_id", ""),
                     "subject_id": metadata.get("subject_id", ""),
                     "domain": metadata.get("clock_domain", "unknown"),
                     "status": metadata.get("clock_alignment_status", "not_reported"),
                     "a_s": metadata.get("clock_map_a", metadata.get("time_mapping_a", np.nan)),
                     "b": metadata.get("clock_map_b", metadata.get("time_mapping_b", np.nan)),
                     "fs_local_hz": metadata.get("fs_local_hz", metadata.get("source_fs_hz", np.nan)),
                     "fs_reference_hz": metadata.get("fs_reference_hz", metadata.get("fs_hz", np.nan)),
                     "uncertainty": metadata.get("clock_alignment_uncertainty_status", "not_reported")})
    alignment = _frame(tables.get("clock_alignment"))
    if rows:
        frame = pd.DataFrame(rows)
    elif not alignment.empty:
        frame = alignment.copy()
    else:
        return None
    fig = Figure(figsize=(max(9, len(frame) * 0.8), 4.8), constrained_layout=True)
    ax = fig.subplots(1, 1)
    ax.axis("off")
    columns = [column for column in ("recording_id", "subject_id", "domain", "status", "a_s", "b",
                                     "fs_local_hz", "fs_reference_hz", "uncertainty") if column in frame]
    display = frame[columns].astype(object).fillna("—").astype(str)
    table = ax.table(cellText=display.values, colLabels=columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title("时钟映射诊断（真实record元数据）", loc="left", color=_COLORS["ink"], fontweight="bold")
    return _finish_figure(fig, destination / "clock_diagnostics.png", font)


def _statistics_figures(tables, config, destination, font):
    """只绘制真实统计 estimates 中具备单位关联和有限区间的有效行。"""
    estimates = _frame(tables.get("statistics"))
    diagnostics = _frame(tables.get("statistics_diagnostics"))
    features = _frame(tables.get("features"))
    info = {"status": "skipped", "reason": "", "source_rows": int(len(estimates)),
            "valid_rows": 0, "plotted_rows": 0, "missing_ci_rows": 0,
            "missing_feature_unit_rows": 0, "invalid_ci_rows": 0, "figure_files": []}
    if estimates.empty:
        info["reason"] = "statistics table missing or empty"
        return [], info

    options = config.get("statistics", config) if isinstance(config, dict) else {}
    try:
        alpha = float(options.get("alpha", 0.05))
    except (TypeError, ValueError):
        alpha = math.nan
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        info["reason"] = "statistics.alpha is not a finite value between 0 and 1"
        return [], info
    info.update({"alpha": alpha, "confidence_level": 1.0 - alpha,
                 "confidence_level_percent": 100.0 * (1.0 - alpha)})

    required = {"outcome", "design_id", "estimate", "ci_low", "ci_high", "status"}
    if not required <= set(estimates.columns):
        info["reason"] = f"statistics table lacks required columns: {sorted(required - set(estimates.columns))}"
        return [], info
    if not {"feature_id", "unit"} <= set(features.columns):
        info["reason"] = "features table lacks feature_id/unit for outcome-unit association"
        return [], info

    units_by_outcome = {}
    ambiguous_units = set()
    for outcome, group in features.groupby("feature_id", dropna=False, sort=False):
        if pd.isna(outcome):
            continue
        units = sorted(set(group["unit"].dropna().astype(str)))
        if len(units) == 1:
            units_by_outcome[str(outcome)] = units[0]
        elif len(units) > 1:
            ambiguous_units.add(str(outcome))

    frame = estimates.copy()
    frame["_outcome_key"] = frame["outcome"].astype(str)
    frame["_plot_unit"] = frame["_outcome_key"].map(units_by_outcome)
    status_valid = frame["status"].astype(str).eq("valid")
    info["valid_rows"] = int(status_valid.sum())
    no_unit = status_valid & frame["_plot_unit"].isna()
    info["missing_feature_unit_rows"] = int(no_unit.sum())
    numeric = {}
    for column in ("estimate", "ci_low", "ci_high"):
        numeric[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.logical_and.reduce([np.isfinite(numeric[column].to_numpy(dtype=float))
                                    for column in ("estimate", "ci_low", "ci_high")])
    finite = pd.Series(finite, index=frame.index)
    info["missing_ci_rows"] = int((status_valid & ~finite).sum())
    ordered = (numeric["ci_low"] <= numeric["ci_high"])
    encloses = (numeric["ci_low"] <= numeric["estimate"]) & (numeric["estimate"] <= numeric["ci_high"])
    valid_ci = finite & ordered & encloses
    info["invalid_ci_rows"] = int((status_valid & finite & ~valid_ci).sum())
    plot_frame = frame.loc[status_valid & ~no_unit & valid_ci].copy()
    for column, values in numeric.items():
        plot_frame[column] = values.loc[plot_frame.index].astype(float)
    if plot_frame.empty:
        info["reason"] = "no valid estimates have a finite ordered CI and a unique feature-table unit"
        return [], info

    diagnostic_by_design = {}
    for row in diagnostics.to_dict(orient="records"):
        design_id = str(row.get("design_id", ""))
        details = row.get("details")
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except (TypeError, ValueError):
                details = {}
        if not isinstance(details, dict):
            details = {}
        diagnostic_by_design[design_id] = {
            "status": row.get("status"),
            "ci_convention": details.get("ci_convention", "not_reported"),
            "inference": details.get("inference", "not_reported"),
            "diagnostic_alpha": details.get("alpha"),
        }

    paths = []
    diagnostic_labels = set()
    for unit, unit_frame in plot_frame.groupby("_plot_unit", sort=True, dropna=False):
        unit_frame = unit_frame.reset_index(drop=True)
        unit_diagnostic_labels = set()
        y = np.arange(len(unit_frame), dtype=float)
        estimates_x = unit_frame["estimate"].to_numpy(float)
        lower = unit_frame["ci_low"].to_numpy(float)
        upper = unit_frame["ci_high"].to_numpy(float)
        labels = []
        for row in unit_frame.to_dict(orient="records"):
            design_id = str(row.get("design_id") or "design")
            contrast = row.get("contrast")
            if contrast is None or pd.isna(contrast) or not str(contrast).strip():
                contrast = row.get("hypothesis_id") or "估计值"
            labels.append(f"{row['_outcome_key']} · {design_id} · {contrast}")
            diag = diagnostic_by_design.get(design_id, {})
            label = f"{design_id}: {diag.get('inference', 'not_reported')} / {diag.get('ci_convention', 'not_reported')}"
            diagnostic_labels.add(label)
            unit_diagnostic_labels.add(label)

        fig = Figure(figsize=(11.5, max(4.2, 0.48 * len(unit_frame) + 1.8)), constrained_layout=True)
        ax = fig.subplots(1, 1)
        ax.errorbar(estimates_x, y,
                    xerr=np.vstack([estimates_x - lower, upper - estimates_x]),
                    fmt="o", markersize=5, color=_COLORS["teal"], ecolor=_COLORS["blue"],
                    elinewidth=1.3, capsize=3)
        ax.axvline(0.0, color=_COLORS["muted"], linewidth=0.8, linestyle="--")
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"估计值与 {100.0 * (1.0 - alpha):g}% 两侧置信区间（{unit}）")
        ax.set_title(f"统计估计与置信区间 · outcome 单位 {unit}", loc="left",
                     color=_COLORS["ink"], fontweight="bold")
        ax.grid(axis="x", color=_COLORS["grid"], linewidth=0.65)
        ax.set_axisbelow(True)
        diagnostic_text = "；".join(sorted(unit_diagnostic_labels)) or "诊断未提供 CI/inference 定义"
        ax.text(0.0, -0.12,
                f"配置 α={alpha:g}；CI 水平 {100.0 * (1.0 - alpha):g}% · diagnostics: {diagnostic_text}",
                transform=ax.transAxes, color=_COLORS["muted"], fontsize=8, va="top")
        path = destination / f"statistics_ci_{_slug(unit)}.png"
        paths.append(_finish_figure(fig, path, font))

    info.update({"status": "generated", "reason": "", "plotted_rows": int(len(plot_frame)),
                 "figure_files": [Path(item).name for item in paths],
                 "unit_by_outcome": units_by_outcome,
                 "ambiguous_unit_outcomes": sorted(ambiguous_units),
                 "diagnostics": diagnostic_by_design,
                 "ci_inference_labels": sorted(diagnostic_labels),
                 "confidence_interval_source": "statistics.ci_low/ci_high; confidence level from statistics.alpha"})
    return paths, info


def make_figures(results, config, destination):
    """根据已计算的真实结果生成静态 PNG 图并返回绝对路径列表。

    支持 `results={'summary','tables','records'}`：每记录 bundle 绘信号/质量/SCR，
    features 按 condition 与 unit 汇总，有 signed synchrony 时分别绘 pair 矩阵与
    lag-time 曲线，有 null summary 时画 mean±SD 与 observed，不把 summary 伪造成
    原始 null 样本分布；统计 estimates 只绘 valid 且有有限 CI 的真实估计，按 outcome
    单位分图；可用 metadata 绘时钟映射诊断。每个图标出 μS/s/秒等单位、
    实际成员/有效区间与样本范围。只要对应表或有效数组不存在就不生成该类图。
    所有 Figure 用 `matplotlib.figure.Figure`+`FigureCanvasAgg`，不调用 pyplot、
    不更改 GUI/backend、字体 rcParams 或主线程状态；点数超过 4000 时按固定间隔
    下采样显示，定量计算仍使用完整输入。Windows 使用微软雅黑/黑体；无 CJK 字体时
    记录回退并报告字体检查状态。输出写入目标目录，函数只返回生成文件的绝对路径。
    """
    if not isinstance(results, dict):
        raise ValueError("make_figures 的 results 必须为结果字典。")
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    font, font_report = _font_properties()
    tables = results.get("tables", {}) if isinstance(results.get("tables", {}), dict) else {}
    record_results = results.get("records", []) if isinstance(results.get("records", []), list) else []
    paths = []
    for index, record in enumerate(record_results):
        figure_path = _record_figure(record, destination, font)
        if figure_path:
            paths.append(figure_path)
    feature_paths = _feature_figures(tables, destination, font)
    paths.extend(feature_paths)
    sync_paths = _sync_figures(tables, destination, font)
    paths.extend(sync_paths)
    null_path = _null_summary_figure(tables, destination, font)
    if null_path:
        paths.append(null_path)
    statistic_paths, statistics_state = _statistics_figures(tables, config, destination, font)
    paths.extend(statistic_paths)
    clock_path = _clock_figures(record_results, tables, destination, font)
    if clock_path:
        paths.append(clock_path)
    sync_table = _frame(tables.get("synchrony"))
    null_table = _frame(tables.get("sync_null_summary"))
    clock_table = _frame(tables.get("clock_alignment"))
    feature_table = _frame(tables.get("features"))
    plot_state = {
        "record_signals": {"status": "generated" if any(Path(path).name.startswith("signal_") for path in paths) else "skipped",
                           "reason": "" if any(Path(path).name.startswith("signal_") for path in paths) else
                           "no record bundle with aligned time_s and signal arrays"},
        "feature_conditions": {"status": "generated" if feature_paths else "skipped",
                               "reason": "" if feature_paths else "features table missing/empty or has no finite valid rows"},
        "signed_synchrony": {"status": "generated" if sync_paths else "skipped",
                             "reason": "" if sync_paths else "no finite status=valid synchrony r/lag_s rows",
                             "source_rows": int(len(sync_table))},
        "null_summary": {"status": "generated" if null_path else "skipped",
                         "reason": "" if null_path else
                         "null summary absent/empty; source has no exported individual surrogate values",
                         "source_rows": int(len(null_table)),
                         "interpretation": "mean plus sample SD and observed statistic only; no empirical histogram synthesized"},
        "statistics_ci": statistics_state,
        "clock_diagnostics": {"status": "generated" if clock_path else "skipped",
                              "reason": "" if clock_path else "no clock mapping metadata or clock_alignment table",
                              "source_rows": int(len(clock_table))},
    }
    metadata_path = destination / "plot_metadata.json"
    metadata_path.write_text(json.dumps({"font": font_report, "figure_count": len(paths),
                                        "figure_files": [Path(item).name for item in paths],
                                        "rendering": "FigureCanvasAgg; object-oriented; no pyplot state",
                                        "sections": plot_state,
                                        "input_row_counts": {"features": int(len(feature_table)),
                                                              "synchrony": int(len(sync_table)),
                                                              "sync_null_summary": int(len(null_table)),
                                                              "statistics": int(len(_frame(tables.get("statistics")))),
                                                              "statistics_diagnostics": int(len(_frame(tables.get("statistics_diagnostics"))))}},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


__all__ = ["make_figures"]
