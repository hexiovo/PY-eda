# 多人同步方法（P07）

本文描述 `eda_toolkit.core_sync` 当前实现。所有时间量用秒，信号单位用 μS。同步是关联度量，不说明因果、谁带领谁或完整的高阶群体耦合。方法与设备/任务的科学资格仍需独立证据。

## 时钟对齐

每条本地记录以独立同步锚点拟合

\[
t_{ref}=a+b\,t_{local},\qquad b>0,
\]

其中 `a` 是时钟offset、`b` 是时钟速率比例。参考采样率为
`fs_reference_hz = fs_local_hz / b`。锚点按 `recording_id` 匹配；可选 `channel` 用于区分同一记录的不同通道。

- 一个有效锚点固定 `b=1`，只给出offset，结果状态是 `one_anchor_offset_only`。
- 两个不同本地时间锚点用直线拟合，但状态明确为 `two_anchors_no_empirical_uncertainty`；两点本身不能估计残差不确定度。
- 三个或更多有效锚点用普通最小二乘，保留所有有效锚点，不静默删异常值。报告保留锚点数、时间跨度、残差RMS/标准差/最大绝对值、映射范围外推量及锚点SHA-256。
- `clock_domain="reference"` 的记录不二次映射。没有锚点的本地时钟为 `unverified`；仅当配置显式写入 `synchrony.shared_clock_verified=true` 时，才按共同时间基准的identity映射继续。相同采样率或样本长度不会自动构成同步证据。

映射前记录时间保存在 `record["local_time_s"]`。记录metadata保留 `original_clock_domain`、`clock_map_a`、`clock_map_b`、`anchor_sha256`、`time_mapping_version`、`fs_local_hz`、`source_fs_hz`、`fs_hz` 和 `fs_reference_hz`。`events.onset_s/start_s/end_s`、事件 `duration_s`、`conditions` 与 `manual_qc` 的区间端点一起变换；变换前的值及原时钟域另存为 `local_*` 字段。

如果映射缺失或无效，时钟报告为 `unverified`，`align_recordings` 返回空的共同网格，成对计算输出 `unverified` 行，不会产生正式同步值。

## 共同网格和有效样本

`align_recordings` 仅在所有记录处于reference clock、含有效独立anchors，或显式验证了共同钟时生成共同网格。共同范围为所有记录有效支持的时间交集；网格锚在reference时间零点，频率为 `synchrony.analysis_rate_hz`。每条记录根据 `quality.analysis_eligible`、`detector_eligible`、插值/伪影标记和有限信号值生成掩码；除非显式允许provisional数据，受detector排除的样本不参与同步。

每个连续有效段单独调用预处理模块的polyphase抗混叠FIR重采样和支撑传播规则；长缺口、无效区、插值区和滤波边缘不跨接。不同采样相位仅在已经低通到目标网格后的连续支撑内插值到绝对reference网格。不能以直接线性插值代替降采样滤波。超出连续段抖动容差、超出`memory_limit_mb`、没有共同有效支持或网格为空时报告明确错误/不可用，不降低目标采样率以绕过限制。

## 成对同步值

固定窗口由 `synchrony.window_s` 和 `step_s` 决定；只纳入完全位于分析区间内的整窗，不截短尾窗。`analysis_intervals_s`可声明不相交的reference秒区间；省略时从共同网格两端各裁去`analysis_context_s`生成分析区间。全长信号和有效性掩码仍保留context，报告发布解析出的区间与context。区间中没有完整窗口时返回`analysis_interval_too_short`，不会自动缩短窗口。`group_sync`预设使用60秒context，通用预设为0秒。

lag候选来自 `max_lag_s` 与 `lag_step_s`，并报告实际落到分析采样网格的lag。零lag始终保留。配置 `synchrony.metric` 可选 `pearson` 或 `spearman`；两者均输出有符号相关，切换配置可计算另一指标。

lag符号遵循

\[
r_{AB}(\tau)=\operatorname{corr}(A(t), B(t+\tau)).
\]

因此正lag表示A的变化先于B，负lag表示B先于A。每个pair/window/metric/basis/lag结果包含`r`、`status`、`reason`、`valid_n`、有效相邻区间累计`duration_s`和相对窗口预期暴露的`coverage`。相邻有效点间的长缺口不计入暴露。样本少于`min_samples`、覆盖低于`min_coverage`、常数序列或时钟未经验证时输出NaN和明确状态，不转成零相关。

`pairs=None` 时仅从 `group_membership` 在每个固定 `group_id × session_id` 内生成一次性完整无向边集，同一受试者不与自己配对，也不跨session建边。缺失/歧义成员、重复受试者或不足两名成员使对应组不可用。下游组汇总要求全部预设边均有效，不静默删边或缩小组。

组Pearson使用所有预设边的Fisher-z算术均值后反变换；组Spearman使用另行标记的边值算术均值，不冒充Pearson Fisher-z估计。留一成员分析在固定成员表上删除指定成员及其关联边；余下至少两名成员且余下所有边均有效时才给出结果。

资源预检使用 `compute.max_pairs` 和 `compute.max_windows`；超过上限时明确失败，不截短pair或窗口集合。

## Surrogate null方法

`synchrony.null_method`严格选择`none`、`circular_shift`或`pseudodyad`；选择一种不会暗中运行另一种。`none`报告未请求。

`circular_shift`对每个`session_id × subject_id`整体移动信号与掩码，保持受试者各通道、窗口和指标共用同一位移。先为每名受试者联合所有分析窗、全部lag样本和接缝屏蔽邻域，计算admissible shifts：接缝不碰需求样本，且每个需求样本的移动mask与观察mask相同。相对圆周距离须不小于`min_shift_s`；个体绝对位移0可以进入候选集。按参与者独立均匀抽取候选位移，再按固定成员边对联合拒绝。小候选笛卡尔积会穷举可行性；较大空间使用有界几何搜索。空候选报告`shift_support_infeasible`；证明联合约束无解报告`joint_shift_infeasible`；大空间搜索未找到解报告`feasibility_unresolved`，不误称无解。满覆盖且只有0位移满足接缝条件时会在相关计算前完成不可行预检。

每个接受的surrogate必须保留观察分析的固定pair/window/metric/lag和有效样本数、coverage、状态签名。按指定B有放回抽样；达到`max_attempts`后停止，不降低请求的B。报告参与者候选数量、几何预检状态/提案数、实际proposal数、可行有限support大小（能精确穷举时）、独立surrogate worlds数、有放回标志和support policy版本。接受数少于请求B时不报告尾部估计。

搜索统计量由`synchrony.tail`决定。`two-sided`在lag搜索取`max(abs(r))`；`greater`取`max(r)`；`less`取`min(r)`。保留所选lag及有符号相关；并列时按lag升序稳定选取。尾部概率按相应方向计算；所有结果仍是实验性surrogate概率，不是校准p值：

\[
\widehat p=\begin{cases}
\dfrac{1+\#\{T_{null}\ge T_{obs}\}}{B_{accepted}+1},&\text{two-sided或greater},\\[4pt]
\dfrac{1+\#\{T_{null}\le T_{obs}\}}{B_{accepted}+1},&\text{less}.
\end{cases}
\]

通用字段为`observed_statistic`、`null_mean_statistic`、`null_sd_statistic`、`statistic`和`tail`。旧`max_abs_r`摘要字段仅在two-sided时填写。报告请求、接受、尝试、几何预检与实际proposal数、有限support（可精确枚举时）、unique worlds、分辨率及经验尾部比例的Wilson Monte Carlo区间。即使B足够，字段名仍为`surrogate_tail_probability`，状态为`experimental_uncalibrated`；在重复已知正/负耦合、共享刺激、缺失模式和lag搜索校准前，不称为校准p值，也不作科学显著性保证。

## 匹配伪配对

`matched_pseudodyads`直接API返回匹配候选表；当`null_method="pseudodyad"`时，估计器会实际构造donor surrogate world。仅支持完整匹配的独立`(group_id, session_id)`区块：用户必须显式设置`pseudodyad_exchangeability=true`，并提供`task`、`stimulus`、时间线标识、匹配层、每区块唯一`role`或`member_slot`及完整兼容的slot集合。时间须显式声明为`task_relative`，或每条记录提供`task_time_origin_s`；不会由reference clock推断任务相位。每个匹配层/slot独立置换donor区块，同一donor映射跨所有窗口、lag、指标和边保持不变；拒绝自配、同一原互动区块、重复donor区块和支持不匹配。结果保留target身份与每个接受world的donor subject/block/record provenance。缺元数据、独立区块不足、support不匹配或assignment不可行时返回明确`unavailable`，不会回退到循环移位。不同组可使用相同session标签（例如都为`V01`），独立性按`group_id × session_id`区块判断。

## 关键输出和资格限制

- `clock_alignment.json`：每条记录的时钟映射、锚点证据、采样率和外推信息。
- `alignment_report.json`：reference网格范围、频率、逐记录重采样状态与成员表行数，以及解析出的同步分析区间/context。
- `synchrony` / `sync_coverage`：固定边、lag曲面和有效支持。
- `group_synchrony`：固定边集组汇总及留一成员结果。
- `sync_null_summary` / `sync_tail_probabilities`：声明的null method、接受状态、有限support/unique worlds、donor provenance、资源上限和实验性尾部量。
- `matched_pseudodyads`（直接API）：交换性候选表及缺少字段/候选的原因；伪配对零模型会在同一同步接口执行真实donor重分配。

这些软件输出可由小型确定性夹具检查，但真实设备时钟精度、任务/刺激交换性、群体科学解释、零模型拒绝率和统计功效尚未由此实现证明。真实数据校准与许可/发行资格在对应阶段分别记录。
