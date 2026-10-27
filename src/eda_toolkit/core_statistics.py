"""按独立单位与结构化研究设计进行统计；绝不执行用户公式文本。"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import DESIGN_KEYS, parse_json_text
from .errors import fail

ESTIMATE_COLUMNS = ["design_id", "hypothesis_id", "outcome", "design_type", "contrast", "estimate", "effect_size",
                    "effect_size_type", "se", "ci_low", "ci_high", "statistic", "df", "p_raw", "p_adjusted",
                    "n_observations", "n_subjects", "n_groups", "n_sessions", "family", "status", "reason"]
DIAGNOSTIC_COLUMNS = ["design_id", "status", "code", "message", "excluded_n", "details"]


def _designs(design, config):
    configured = config.get("statistics", config).get("designs", [])
    if configured:
        rows = configured
    elif design is None:
        rows = []
    elif isinstance(design, pd.DataFrame):
        rows = design.to_dict(orient="records")
    elif isinstance(design, dict):
        rows = [design]
    else:
        rows = list(design)
    output = []
    for original in rows:
        item = {key: value for key, value in dict(original).items()
                if value is not None and not (isinstance(value, float) and math.isnan(value)) and not (isinstance(value, str) and not value.strip())}
        for source, target in (("subject_id_column", "subject_column"), ("session_id_column", "session_column")):
            if source in item:
                if target in item and item[target] != item[source]:
                    fail("design_alias_conflict", "统计设计字段别名存在冲突。", category="configuration")
                item[target] = item.pop(source)
        if set(item) - DESIGN_KEYS:
            fail("unsafe_statistical_design", "统计设计包含未知字段或公式文本；请使用结构化字段。", category="configuration")
        if "contrast" in item:
            contrast = item.pop("contrast")
            if isinstance(contrast, str):
                contrast = parse_json_text(contrast)
            if not isinstance(contrast, dict) or set(contrast) - {"conditions", "groups", "condition_column", "group_column", "reference", "predictor"}:
                fail("structured_contrast", "contrast 必须是符合契约的结构化对象。", category="configuration")
            for key, value in contrast.items():
                if key in item and item[key] != value:
                    fail("contrast_conflict", "contrast 与显式设计字段冲突。", category="configuration")
                item[key] = value
        for key in ("conditions", "groups", "covariates"):
            if isinstance(item.get(key), str):
                try:
                    item[key] = parse_json_text(item[key])
                except json.JSONDecodeError:
                    fail("design_list", f"{key} 必须是JSON数组，不能是可执行表达式。", category="configuration")
            if key in item and (not isinstance(item[key], list) or not all(isinstance(x, str) for x in item[key])):
                fail("design_list", f"{key} 只能是列名或水平字符串数组。", category="configuration")
        output.append(item)
    if len({item.get("design_id", f"design_{i}") for i, item in enumerate(output)}) != len(output):
        fail("duplicate_design", "design_id 不得重复。", category="configuration")
    return output


def _numeric(frame, column):
    if column not in frame:
        fail("stat_column", f"统计设计所需字段不存在：{column}")
    return pd.to_numeric(frame[column], errors="coerce")


def _counts(frame):
    counts = {"n_observations": len(frame), "n_subjects": frame["subject_id"].nunique() if "subject_id" in frame else 0,
            "n_groups": frame["group_id"].nunique() if "group_id" in frame else frame["experimental_group"].nunique() if "experimental_group" in frame else 0,
            "n_sessions": frame["session_id"].nunique() if "session_id" in frame else 0}
    counts.update(frame.attrs.get("source_unit_counts", {}))
    return counts


def _prepare(table, design, options):
    data = table.copy()
    outcome = design.get("outcome")
    if not outcome:
        fail("outcome_missing", "每项统计设计必须明确 outcome。", category="configuration")
    if "feature_id" in data:
        data = data.loc[data["feature_id"].astype(str) == str(outcome)].copy()
        data["_value"] = _numeric(data, "value")
    else:
        data["_value"] = _numeric(data, outcome)
    total = len(data)
    usable = np.isfinite(data["_value"])
    if "status" in data:
        usable &= data["status"].eq("valid")
    if not usable.all() and options.get("missing_policy", "complete_case") == "fail":
        fail("missing_stat_outcomes", "计划结局有缺失/不可用，missing_policy=fail 拒绝删除。")
    data = data.loc[usable].copy()
    if data.empty:
        fail("no_stat_observations", "该计划结局没有可用观测。")
    kind = design.get("design_type", "descriptive")
    condition = design.get("condition_column", "condition")
    group = design.get("group_column", "experimental_group")
    required = []
    if kind in ("paired", "wilcoxon") or kind == "mixed_effects" and design.get("conditions"):
        required.append(condition)
    if kind in ("welch", "mannwhitney"):
        required.append(group)
    numeric_required = list(design.get("covariates", [])) if kind == "mixed_effects" else []
    if kind == "correlation":
        if not design.get("predictor"):
            fail("correlation_predictor", "相关分析必须声明 predictor。")
        numeric_required.append(design["predictor"])
    complete = np.ones(len(data), dtype=bool)
    for column in required + numeric_required:
        if column not in data:
            fail("stat_column", f"统计设计所需字段不存在：{column}")
        complete &= (np.isfinite(pd.to_numeric(data[column], errors="coerce")) if column in numeric_required
                     else data[column].notna() & data[column].astype(str).str.strip().ne(""))
    if not complete.all() and options.get("missing_policy", "complete_case") == "fail":
        fail("missing_design_variable", "条件/组别/协变量存在缺失；missing_policy=fail 禁止删除。")
    data = data.loc[complete].copy()
    if kind in ("paired", "wilcoxon") and design.get("conditions"):
        data = data.loc[data[condition].astype(str).isin(design["conditions"])].copy()
    if kind in ("welch", "mannwhitney") and design.get("groups"):
        data = data.loc[data[group].astype(str).isin(design["groups"])].copy()
    excluded_count = total - len(data)
    if data.empty:
        fail("no_complete_design", "按计划字段和水平筛选后没有完整观测。")
    if design.get("design_type") == "mixed_effects" and design.get("conditions"):
        column = design.get("condition_column", "condition")
        if column in data and not set(data[column].dropna().astype(str)) <= set(design["conditions"]):
            fail("undeclared_model_condition", "混合模型出现未声明的条件水平；必须明确选择设计范围，不能并入参考条件。")
    independent = design.get("independent_unit", "subject_id")
    independent = {"subject": "subject_id", "group": "group_id", "session": "session_id"}.get(independent, independent)
    if independent not in {"subject_id", "group_id", "session_id"} or independent not in data:
        fail("independent_unit", "必须指定真实存在的 subject_id/group_id/session_id 独立单位。")
    if data[independent].isna().any() or data[independent].astype(str).eq("").any():
        fail("missing_independent_unit", "独立单位编号不得为空。")
    if any(name in data and data[name].notna().any() for name in ("pair_id", "dyad_id")):
        fail("unsupported_dyad_hierarchy", "重复配对边需要专门的多成员层级模型；当前设计不支持。")
    if independent == "subject_id" and "group_id" in data and data["group_id"].astype(str).nunique() < data["subject_id"].nunique() and data["group_id"].notna().any():
        # 真正互动群组内的被试相关性不能被独立被试检验忽略。
        if design.get("design_type") not in ("descriptive",):
            fail("clustered_independence", "多个被试属于互动群组；该设计需以group_id为独立单位或使用经验证的群组层级模型。")
    data["_unit"] = data[independent].astype(str)
    kind = design.get("design_type", "descriptive")
    condition = design.get("condition_column", "condition")
    group = design.get("group_column", "experimental_group")
    strata = []
    if kind in ("paired", "wilcoxon", "mixed_effects") and condition in data:
        strata.append(condition)
    if kind in ("welch", "mannwhitney"):
        if group not in data:
            fail("group_column", "独立组比较缺少指定组别字段。")
        if data.groupby("_unit", dropna=False)[group].nunique().gt(1).any():
            fail("overlapping_groups", "同一独立单位不能同时属于两个独立比较组。")
        strata.append(group)
    keys = ["_unit"] + strata
    duplicates = data.duplicated(keys, keep=False)
    source_counts = _counts(data)
    if duplicates.any():
        aggregate = design.get("aggregate")
        if aggregate not in ("mean", "median"):
            fail("pseudoreplication", "同一独立单位存在重复窗口/记录；必须预先明确aggregate=mean或median，不能当作独立样本。")
        ancillary = [key for key in data.columns if key not in keys + ["_value"]]
        for column in set(design.get("covariates", []) + ([design["predictor"]] if design.get("predictor") else [])):
            if column not in data or data.groupby(keys, dropna=False)[column].nunique(dropna=False).gt(1).any():
                fail("covariate_aggregation", "重复窗口内协变量发生变化，不能用first静默聚合。")
        grouped = data.groupby(keys, dropna=False, sort=False)
        values = grouped["_value"].agg(aggregate)
        metadata = grouped[ancillary].first()
        data = metadata.join(values).reset_index()
        data.attrs["source_unit_counts"] = {key: source_counts[key] for key in ("n_subjects", "n_groups", "n_sessions")}
    if kind == "correlation":
        predictor = design.get("predictor")
        if not predictor or predictor not in data:
            fail("correlation_predictor", "相关分析必须指定参与者表中的 predictor 字段。")
    return data, excluded_count


def _t_summary(values, reference, alternative, alpha):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        fail("sample_size", "t检验至少需要两个独立观测。")
    sd = float(np.std(values, ddof=1))
    if sd <= 0:
        fail("zero_variance", "零方差下不报告伪造的有限t统计量或效应量。")
    estimate = float(np.mean(values) - reference)
    se = sd / np.sqrt(len(values))
    test = stats.ttest_1samp(values, reference, alternative=alternative)
    half = stats.t.ppf(1 - alpha / 2, len(values) - 1) * se
    return {"estimate": estimate, "effect_size": estimate / sd, "effect_size_type": "Cohen_d_sample_sd",
            "se": se, "ci_low": estimate - half, "ci_high": estimate + half,
            "statistic": float(test.statistic), "df": len(values) - 1, "p_raw": float(test.pvalue)}


def _single_design(data, design, options):
    kind = design.get("design_type", "descriptive")
    alpha = options.get("alpha", .05)
    alternative = design.get("alternative", "two-sided")
    if alternative not in ("two-sided", "less", "greater"):
        fail("test_alternative", "alternative 必须是 two-sided/less/greater。")
    y = data["_value"].to_numpy(float)
    base = {"design_id": design["design_id"], "hypothesis_id": design["design_id"], "outcome": design["outcome"],
            "design_type": kind, "contrast": "", "family": design.get("correction_family", "default"),
            "status": "valid", "reason": "", **_counts(data)}
    diagnostic = {"warnings": [], "ci_convention": "two_sided", "alpha": alpha}
    if kind == "descriptive":
        sd = float(np.std(y, ddof=1)) if len(y) > 1 else math.nan
        se = sd / np.sqrt(len(y))
        half = stats.t.ppf(1 - alpha / 2, len(y) - 1) * se if len(y) > 1 else math.nan
        result = {"estimate": float(np.mean(y)), "se": se, "ci_low": float(np.mean(y) - half),
                  "ci_high": float(np.mean(y) + half), "effect_size_type": "not_applicable"}
        diagnostic.update({"sample_sd": sd, "median": float(np.median(y)), "iqr": float(np.subtract(*np.percentile(y, [75, 25])))})
    elif kind == "one_sample":
        result = _t_summary(y, float(design.get("reference", 0)), alternative, alpha)
        base["contrast"] = "mean_minus_reference"
    elif kind in ("paired", "wilcoxon"):
        condition = design.get("condition_column", "condition")
        levels = design.get("conditions", [])
        if len(levels) != 2 or condition not in data:
            fail("paired_conditions", "配对检验必须声明两个condition水平及列名。")
        wide = data.pivot(index="_unit", columns=condition, values="_value")
        if not set(levels) <= set(wide.columns):
            fail("missing_planned_condition", "计划的配对条件整体缺失，不改写为剩余条件。")
        paired = wide[levels].dropna()
        diagnostic["excluded_unpaired_units"] = sorted(set(wide.index) - set(paired.index))
        differences = (paired[levels[0]] - paired[levels[1]]).to_numpy(float)
        if len(differences) < 2:
            fail("insufficient_pairs", "完整配对单位不足两个。")
        base["n_observations"] = len(paired) * 2
        if "subject_id" in data and data["_unit"].eq(data["subject_id"].astype(str)).all():
            base["n_subjects"] = len(paired)
        base["contrast"] = f"{levels[0]} - {levels[1]}"
        if kind == "paired":
            result = _t_summary(differences, 0, alternative, alpha)
            result["effect_size_type"] = "Cohen_dz_difference_sample_sd"
        else:
            test = stats.wilcoxon(differences, alternative=alternative, zero_method="wilcox") if np.any(differences) else None
            nonzero = differences[differences != 0]
            ranks = stats.rankdata(abs(nonzero))
            effect = float(np.sum(ranks * np.sign(nonzero)) / np.sum(ranks)) if len(nonzero) else 0.0
            result = {"estimate": float(np.median(differences)), "effect_size": effect,
                      "effect_size_type": "matched_rank_biserial", "statistic": float(test.statistic) if test else 0.,
                      "p_raw": float(test.pvalue) if test else 1.}
            diagnostic["interval_status"] = "nonparametric_location_interval_not_estimated"
    elif kind in ("welch", "mannwhitney"):
        group = design.get("group_column", "experimental_group")
        levels = design.get("groups", [])
        if len(levels) != 2 or not set(levels) <= set(data[group].dropna().astype(str)):
            fail("missing_planned_group", "必须有两个明确且实际存在的计划独立组。")
        a = data.loc[data[group].astype(str).eq(levels[0]), "_value"].to_numpy(float)
        b = data.loc[data[group].astype(str).eq(levels[1]), "_value"].to_numpy(float)
        if min(len(a), len(b)) < 2:
            fail("sample_size", "每个独立组至少需要两个独立单位。")
        base["contrast"] = f"{levels[0]} - {levels[1]}"
        if kind == "welch":
            v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
            variance = v1 / len(a) + v2 / len(b)
            if variance <= 0:
                fail("zero_variance", "两个组无可估计方差。")
            se = np.sqrt(variance)
            df = variance ** 2 / ((v1 / len(a)) ** 2 / (len(a) - 1) + (v2 / len(b)) ** 2 / (len(b) - 1))
            test = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
            estimate = float(np.mean(a) - np.mean(b))
            pooled = np.sqrt(((len(a) - 1) * v1 + (len(b) - 1) * v2) / (len(a) + len(b) - 2))
            g = (1 - 3 / (4 * (len(a) + len(b)) - 9)) * estimate / pooled
            half = stats.t.ppf(1 - alpha / 2, df) * se
            result = {"estimate": estimate, "effect_size": g, "effect_size_type": "Hedges_g_pooled_sample_sd",
                      "se": se, "ci_low": estimate - half, "ci_high": estimate + half,
                      "statistic": float(test.statistic), "df": df, "p_raw": float(test.pvalue)}
        else:
            test = stats.mannwhitneyu(a, b, alternative=alternative, method="auto")
            result = {"estimate": float(np.median(a) - np.median(b)), "statistic": float(test.statistic),
                      "effect_size": 2 * float(test.statistic) / (len(a) * len(b)) - 1,
                      "effect_size_type": "independent_rank_biserial", "p_raw": float(test.pvalue)}
            diagnostic["interval_status"] = "nonparametric_location_interval_not_estimated"
    elif kind == "correlation":
        x = _numeric(data, design["predictor"]).to_numpy(float)
        if not np.isfinite(x).all() or len(y) < 4 or np.std(x) == 0 or np.std(y) == 0:
            fail("correlation_data", "相关需要至少4个完整独立单位、有限预测变量和非零方差。")
        test = stats.pearsonr(x, y, alternative=alternative)
        r = float(test.statistic)
        bounds = np.tanh(np.arctanh(np.clip(r, -1 + 1e-15, 1 - 1e-15)) + np.array([-1, 1]) * stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(y) - 3))
        result = {"estimate": r, "effect_size": r, "effect_size_type": "Pearson_r",
                  "statistic": r, "p_raw": float(test.pvalue), "ci_low": bounds[0], "ci_high": bounds[1], "df": len(y) - 2}
    elif kind == "mixed_effects":
        from statsmodels.regression.mixed_linear_model import MixedLM
        subject = design.get("subject_column", "subject_id")
        if subject not in data or data[subject].nunique() < 3 or not data.groupby(subject).size().gt(1).any():
            fail("mixed_model_groups", "随机截距模型需要至少3个被试组及组内重复观测。")
        if design.get("random_intercept", True) not in (True, "true", "True", "subject_id", subject):
            fail("random_effect_structure", "当前只支持显式被试随机截距；不会静默修改随机结构。")
        matrix = pd.DataFrame({"Intercept": np.ones(len(data))}, index=data.index)
        condition = design.get("condition_column", "condition")
        if condition in data:
            levels = design.get("conditions") or sorted(data[condition].dropna().astype(str).unique())
            if not set(levels) <= set(data[condition].dropna().astype(str)):
                fail("missing_planned_condition", "混合模型的计划条件缺失。")
            for level in levels[1:]:
                matrix[f"{condition}:{level}"] = data[condition].astype(str).eq(level).astype(float)
        for covariate in design.get("covariates", []):
            matrix[covariate] = _numeric(data, covariate)
        if not np.isfinite(matrix.to_numpy(float)).all() or np.linalg.matrix_rank(matrix.to_numpy(float)) < matrix.shape[1] or len(matrix) <= matrix.shape[1]:
            fail("mixed_model_rank", "混合模型设计矩阵非有限、秩不足或残差自由度不足。")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = MixedLM(y, matrix.to_numpy(float), groups=data[subject].astype(str).to_numpy()).fit(reml=True, method="lbfgs", disp=False)
        if not fitted.converged:
            fail("mixed_model_nonconvergence", "混合模型未收敛；未自动简化模型。", category="numerical")
        diagnostic.update({"warnings": [str(item.message) for item in caught], "reml": True,
                           "random_intercept_variance": float(fitted.cov_re[0, 0]), "residual_variance": float(fitted.scale),
                           "fixed_effect_columns": list(matrix.columns), "inference": "asymptotic_normal"})
        bounds = fitted.conf_int(alpha=alpha)
        rows = []
        for index, column in enumerate(matrix.columns):
            rows.append({**base, "hypothesis_id": f"{design['design_id']}:{column}", "contrast": column,
                         "estimate": float(fitted.fe_params[index]), "se": float(fitted.bse_fe[index]),
                         "ci_low": float(bounds[index, 0]), "ci_high": float(bounds[index, 1]),
                         "statistic": float(fitted.tvalues[index]), "p_raw": float(fitted.pvalues[index]),
                         "effect_size_type": "unstandardized_fixed_coefficient"})
        return rows, diagnostic
    else:
        fail("unsupported_statistic", f"未实现的统计设计：{kind}", category="configuration")
    return [{**base, **result}], diagnostic


def run_statistics(feature_table, design, config):
    """返回估计表和诊断表；缺失/失败假设仍保留在计划多重比较家族。

    主要来源：SciPy ttest_1samp/ttest_rel/ttest_ind/pearsonr；statsmodels MixedLM
    与 multipletests 官方接口。本函数不做设备、任务或诊断效度的外推。
    """
    options = config.get("statistics", config)
    designs = _designs(design, config)
    if not designs:
        fail("statistical_design_missing", "统计已启用，但没有结构化设计。", category="configuration")
    rows, diagnostics = [], []
    for index, supplied in enumerate(designs):
        supplied = {"design_id": f"design_{index + 1}", **supplied}
        try:
            data, excluded = _prepare(feature_table, supplied, options)
            estimates, details = _single_design(data, supplied, options)
            rows.extend(estimates)
            diagnostics.append({"design_id": supplied["design_id"], "status": "succeeded", "code": "completed",
                                "message": "结构化设计已完成；检查排除与模型诊断。", "excluded_n": excluded,
                                "details": json.dumps(details, ensure_ascii=False, default=str)})
        except Exception as error:
            identifiers = [supplied["design_id"]]
            if supplied.get("design_type") == "mixed_effects":
                column = supplied.get("condition_column", "condition")
                levels = supplied.get("conditions") or (sorted(feature_table[column].dropna().astype(str).unique()) if column in feature_table else [])
                terms = ["Intercept"] + [f"{column}:{level}" for level in levels[1:]] + list(supplied.get("covariates", []))
                identifiers = [f"{supplied['design_id']}:{term}" for term in terms]
            for identifier in identifiers:
                rows.append({"design_id": supplied["design_id"], "hypothesis_id": identifier,
                             "outcome": supplied.get("outcome"), "design_type": supplied.get("design_type"),
                             "family": supplied.get("correction_family", "default"), "status": "unavailable", "reason": str(error)})
            diagnostics.append({"design_id": supplied["design_id"], "status": "failed", "code": getattr(error, "eda_details", {}).get("code", type(error).__name__),
                                "message": str(error), "excluded_n": 0, "details": "未静默更改研究模型或纳入单位。"})
    estimates = pd.DataFrame(rows).reindex(columns=ESTIMATE_COLUMNS)
    planned = options.get("planned_hypotheses", [])
    for item in planned:
        identifier = item if isinstance(item, str) else item.get("hypothesis_id")
        family = "default" if isinstance(item, str) else item.get("correction_family", "default")
        if identifier and identifier not in set(estimates["hypothesis_id"]):
            estimates.loc[len(estimates)] = {"hypothesis_id": identifier, "family": family, "status": "unavailable", "reason": "planned_hypothesis_not_completed"}
    correction = options.get("correction", "holm")
    for family, indices in estimates.groupby("family", dropna=False).groups.items():
        original = pd.to_numeric(estimates.loc[indices, "p_raw"], errors="coerce")
        mask = original.notna() & np.isfinite(original)
        internal = original.fillna(1).to_numpy(float)
        adjusted = internal if correction == "none" else multipletests(internal, alpha=options.get("alpha", .05), method=correction)[1]
        for offset, index in enumerate(indices):
            if mask.loc[index]:
                estimates.loc[index, "p_adjusted"] = float(adjusted[offset])
    return estimates, pd.DataFrame(diagnostics, columns=DIAGNOSTIC_COLUMNS)
