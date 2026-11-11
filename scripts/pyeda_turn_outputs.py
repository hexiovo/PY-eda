"""Charts for the response-linked EDA analysis."""

from pathlib import Path
import numpy as np
import pandas as pd

FOCUS = ["relative_mean", "tonic_mean", "phasic_rms", "relative_peak_rate_per_min",
         "relative_mean_change", "phasic_rms_change"]
LABELS = {
    "relative_mean": "相对信号均值", "tonic_mean": "慢变均值", "phasic_rms": "快变 RMS",
    "relative_peak_rate_per_min": "相对峰频率", "relative_mean_change": "相对信号基线变化",
    "phasic_rms_change": "快变 RMS 基线变化",
}


def write_outputs(root):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "figure.facecolor": "white"})
    root = Path(root)
    figures = root / "figures"
    p = pd.read_csv(root / "features/participant_task_role_features.csv", dtype={"subject_id": str})
    f = pd.read_csv(root / "intake/feature_definitions.csv").feature.tolist()
    c = pd.read_csv(root / "statistics/comparisons.csv")
    r = pd.read_csv(root / "statistics/correlations.csv")
    g = pd.read_csv(root / "statistics/participant_regressions.csv")
    ev = pd.read_csv(root / "statistics/event_ioc_regressions.csv")
    fact = pd.read_csv(root / "statistics/factorial_tests.csv")
    colors = ["#235884", "#a3541d"]
    diagnostics = g.groupby("outcome").agg(
        planned=("status", "size"), estimated=("status", lambda s: int(s.eq("estimated").sum())),
        holm_below_05=("p_holm", lambda s: int(s.lt(.05).sum())),
        high_leverage_above_05=("max_design_leverage", lambda s: int(s.gt(.5).sum())),
        bias_reduced_available=("p_bias_reduced", "count"),
        bias_reduced_holm_below_05=("p_holm_bias_reduced", lambda s: int(s.lt(.05).sum())))
    diagnostics.to_csv(figures / "model_diagnostic_review.csv", encoding="utf-8-sig")
    def save(fig, number):
        fig.savefig(figures / f"figure{number}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 2.6), layout="constrained")
    for i, task in enumerate(("AUT", "PIT")):
        counts = [p[(p.task == task) & (p.window == window) & (p.role == role)].subject_id.nunique()
                  for window in ("turn_interval", "post_1_6s") for role in ("self", "other")]
        bars = ax.bar(np.arange(4)+(i-.5)*.32, counts, .30, color=colors[i], label=task)
        ax.bar_label(bars, padding=3)
    ax.set(xticks=np.arange(4), xticklabels=["间隔窗\n本人", "间隔窗\n对方", "固定窗\n本人", "固定窗\n对方"],
           ylabel="至少有一个有效事件的被试数", ylim=(0, 160))
    ax.legend(frameon=False)
    save(fig, 1)
    fig, axes = plt.subplots(1, 2, figsize=(7, 7.6), sharey=True, layout="constrained")
    for ax, task in zip(axes, ("AUT", "PIT")):
        sub = p[p.task.eq(task)]
        columns = [(window, role) for window in ("turn_interval", "post_1_6s") for role in ("self", "other")]
        counts = np.array([[sub[(sub.window == window) & (sub.role == role)][feature].notna().sum()
                            for window, role in columns] for feature in f])
        image = ax.imshow(counts, aspect="auto", cmap="Blues", vmin=0, vmax=140)
        ax.set(xticks=range(4), xticklabels=["间隔本人", "间隔对方", "固定本人", "固定对方"],
               yticks=range(len(f)), yticklabels=f, title=task)
        ax.tick_params(axis="y", labelsize=6.5, length=0)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    fig.colorbar(image, ax=axes, shrink=.5, label="该情境特征可估计的被试数")
    save(fig, 2)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), sharey=True, layout="constrained")
    for ax, window in zip(axes, ("turn_interval", "post_1_6s")):
        for i, task in enumerate(("AUT", "PIT")):
            for j, feature in enumerate(FOCUS):
                t = c[(c.method == "paired_team") & (c.task == task) & (c.window == window)
                      & (c.feature == feature) & (c.status == "estimated")]
                if len(t):
                    ax.scatter(t.effect_dz, j+(i-.5)*.16, c=colors[i], marker="o" if i == 0 else "^", s=26, label=task if j == 0 else None)
        ax.axvline(0, c="gray", lw=.7)
        ax.set(title="相邻回答间隔" if window == "turn_interval" else "发送后 1–6 秒",
               xlabel="团队内本人−对方的标准化差异 dz", yticks=range(len(FOCUS)),
               yticklabels=[LABELS[x] for x in FOCUS])
        ax.legend(frameon=False)
    axes[0].invert_yaxis()
    save(fig, 3)
    humans = p[p.window.eq("turn_interval") & p.role.eq("self")].drop_duplicates(["subject_id", "task"])
    fig, ax = plt.subplots(figsize=(6.4, 2.6), layout="constrained")
    labels = []
    for j, (task, group) in enumerate([(t, s) for t in ("AUT", "PIT") for s in ("ai", "human")], 1):
        ys = humans.loc[humans.task.eq(task) & humans.group_type.eq(group), "ioc_proportion"].dropna().to_numpy()
        ax.boxplot(ys, positions=[j], widths=.5, showfliers=False,
                   medianprops={"color": colors[(j-1) % 2]})
        ax.scatter(j+np.random.default_rng(j+17).uniform(-.14, .14, len(ys)), ys, s=10,
                   alpha=.5, c=colors[(j-1) % 2])
        labels.append(f"{task}\n{'人机' if group == 'ai' else '人人'}\n(n={len(ys)})")
    ax.set(xticks=range(1, 5), xticklabels=labels, ylabel="个人 IOC 比例", ylim=(-.03, 1.03))
    save(fig, 4)
    outcomes = ["novelty_mean", "usefulness_mean", "fluency_ind", "flexibility_ind", "ioc_proportion"]
    for number, task in enumerate(("AUT", "PIT"), 5):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 7.0), sharey=True, layout="constrained")
        for ax, role in zip(axes, ("self", "other")):
            sub = r[r.method.eq("spearman") & r.window.eq("turn_interval") & r.task.eq(task) & r.role.eq(role)]
            values = sub.pivot(index="feature", columns="outcome", values="estimate").reindex(index=f, columns=outcomes)
            image = ax.imshow(np.ma.masked_invalid(values.to_numpy(float)), cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
            ax.set(yticks=range(len(f)), yticklabels=f, xticks=range(5),
                   xticklabels=["新颖性", "有用性", "流畅性", "灵活性", "IOC"],
                   title=f"{task} · {'本人' if role == 'self' else '对方'}")
            ax.tick_params(axis="y", labelsize=6.8, length=0)
            ax.tick_params(axis="x", labelsize=7, rotation=30)
            for row in sub[sub.p_holm.lt(.05)].itertuples():
                ax.text(outcomes.index(row.outcome), f.index(row.feature), "*", ha="center", va="center", fontsize=10)
        fig.colorbar(image, ax=axes, shrink=.5, label="团队层 Spearman ρ")
        save(fig, number)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.2), sharey=True, layout="constrained")
    for ax, window in zip(axes, ("turn_interval", "post_1_6s")):
        for i, task in enumerate(("AUT", "PIT")):
            for j, feature in enumerate(FOCUS):
                t = g[g.outcome.eq("ioc_proportion") & g.role.eq("other") & g.window.eq(window)
                      & g.task.eq(task) & g.feature.eq(feature) & g.status.eq("estimated")]
                if len(t):
                    v = t.iloc[0]
                    low = v.get("ci_low_bias_reduced", v.ci_low)
                    high = v.get("ci_high_bias_reduced", v.ci_high)
                    if not np.isfinite([low, high]).all():
                        low, high = v.ci_low, v.ci_high
                    ax.errorbar(v.estimate, j+(i-.5)*.18, xerr=[[v.estimate-low],[high-v.estimate]],
                                fmt="o" if i == 0 else "^", color=colors[i], capsize=2)
        ax.axvline(0,c="gray",lw=.7)
        ax.set(title="间隔窗 · 对方回答后" if window == "turn_interval" else "固定窗 · 对方回答后",
               xlabel="个人 IOC 二项 GEE 的 log(OR)", yticks=range(6), yticklabels=[LABELS[x] for x in FOCUS])
    axes[0].invert_yaxis()
    save(fig, 7)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.0), sharey=True, layout="constrained")
    for ax, window in zip(axes, ("turn_interval", "post_1_6s")):
        for j, feature in enumerate(FOCUS):
            t = ev[ev.window.eq(window) & ev.feature.eq(feature) & ev.status.eq("estimated")]
            if len(t):
                v = t.iloc[0]
                low = v.get("ci_low_bias_reduced", v.ci_low)
                high = v.get("ci_high_bias_reduced", v.ci_high)
                if not np.isfinite([low, high]).all():
                    continue
                ax.errorbar(v.estimate, j, xerr=[[v.estimate-low],[high-v.estimate]],
                            fmt="o",color=colors[0],capsize=2)
        ax.axvline(0,c="gray",lw=.7)
        ax.set(title="本人回答后的间隔窗" if window == "turn_interval" else "本人回答后的固定窗",
               xlabel="当前回答 IOC 的 log(OR)，95% CI",yticks=range(6),yticklabels=[LABELS[x] for x in FOCUS])
    axes[0].invert_yaxis()
    save(fig, 8)
