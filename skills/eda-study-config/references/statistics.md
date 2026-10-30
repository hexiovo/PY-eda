# 从研究问题到最终统计

先确定估计量与独立单位，再选检验。选择条件不等同于完成假设验证。下面的最低样本数仅是软件能否计算的底线，不是功效或科学充分性的证明。

| 设计与目标 | 当前 JSON design_type | 必填/检查 |
| --- | --- | --- |
| 只描述当前数据 | descriptive | 明确汇总层级；无推断性 p 值 |
| 独立单位均值与事先参考值比较 | one_sample | reference；独立性；分布/离群；至少两个有效独立值 |
| 同一单位 A/B，目标为均值差 | paired | conditions、condition_column、完整配对；检查差值而非两组各自的正态性 |
| 同一单位 A/B，已预定秩方法问题 | wilcoxon | 差值对称等条件、零差与 ties；不将它叫作所有情形下的均值差检验 |
| 两独立组，目标为均值差 | welch | groups、group_column；组别互斥；独立性和组内分布；每组至少两个 |
| 两独立组，已预定分布/秩问题 | mannwhitney | 不能普遍解释为中位数检验；位置差解释需额外分布形状条件 |
| 独立单位间线性关联 | correlation | predictor 实际存在；有限、非零方差、至少四个完整独立单位；非因果 |
| 简单被试内多条件连续结局，随机截距适用 | mixed_effects | conditions、subject_column、random_intercept=true、covariates、aggregate、参考水平；至少三个被试且有重复，仍须充分样本与模型诊断 |

二项/计数结局、多因素交互、复杂嵌套、随机斜率、生存分析、组内相依 dyad 边不应强塞进以上设计。尤其 event_response_present、SCR 计数等不应因为“软件支持 t 检验”就自动选择 t 检验；根据目标估计量、汇总尺度和有效设计决定。需要 GLMM、GEE、重复测量 ANOVA、稳健/聚类置换等当前未实现模型时，明确 unsupported，给出可审计外部分析方案；不静默降级成一串两两检验。

## 运行前固定的统计计划

对每个假设保存：ID、主要/次要/探索性、结局 ID、信号与单位、窗 kind、有效性规则、汇总顺序（trial→session→subject/group）、独立单位、条件/组顺序、估计量、检验及理由、单/双侧、协变量、随机结构、排除/缺失规则、家族、alpha、效应定义、区间与诊断。

- 不把 Shapiro p 值作为自动二选一开关；小样本和大样本下都需结合设计、图形、离群、残差及预定稳健性分析。正态性检验无法修复非独立。
- 默认可提议 two-sided、alpha=.05、Holm，但标为 proposed 或工程约定；既定方案优先。FDR 的目标及依赖假设须适合研究问题，不能把它当成所有探索的通用答案。
- 保留所有计划假设的家族，失败/不可用项不从分母中删除。不要通过给每个指标单独建一个家族来规避多重比较。
- 当前 complete_case 按实际可用数据删缺失，并不证明无偏；配对缺失、协变量缺失、各组丢失数需另核对。missing_policy=fail 更严格，但不替代缺失机制说明。无缺失插补模型时不能声称实施多重插补。
- unequal trial counts 下 mean/median 如何加权需预定；直接将所有事件 pooled 后求均值不一定等于先会话、再被试均值。当前内置汇总不能表达目标权重时需独立派生层，不添加虚构 aggregate 选项。
- 不随意对负/零值做 log；若所需变换不能由已验证字段表达，使用独立可复现的派生结局及映射。敏感性分析单独配置并保持家族计划。
- 检查多通道、多会话重复以及固定/事件/条件窗重复，不让 aggregate 把它们静默混合。

## 运行后的验收

1. 逐假设读取 status/reason 和 diagnostics，检查是否有不可用、未收敛、奇异、零方差、秩亏、异常边界方差及未处理警告。
2. 用真实完整单位表复核样本数与配对数；报告不是 CSV 行数。对比符号做人工/合成已知方向交叉核对。
3. 检查主要结局是否只来自预定窗；核对单位、是否重复基线校正、是否错误排除真正零响应。
4. 报告估计量、效应与定义、SE/CI、检验统计量/df、p_raw、p_adjusted、家族大小与方法、实际独立单位数、排除和缺失、诊断、方案偏离。
5. 当前 rank 检验未计算位置区间，须标明未估计；不可从普通均值 CI 冒充秩检验 CI。当前 t 类区间是双侧；多重比较后的 p 与边际区间不是同一校正对象，要明确说明。mixed_effects 使用渐近推断，不能声称小样本精确检验。
6. 描述性关联不作因果或临床效度结论；不显著不等于无效应/等效；同步实验性尾部概率不作校准 p 值。

## 一手接口依据（制作时核对，实际版本另记）

- [SciPy 配对 t 检验：两相关样本均值](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)。
- [SciPy Wilcoxon：差值分布及零差处理](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)。
- [statsmodels 多重校正实现](https://www.statsmodels.org/stable/_modules/statsmodels/stats/multitest.html)。

以上接口依据不替代设备/任务验证或研究设计论证。参数依据优先沿用已预注册方案或设备/方法原始资料；需补文献时阅读原文、记录版本/定位与适用限制，不捏造推荐值。
