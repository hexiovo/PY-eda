# EDA Toolkit 契约与已知衔接限制

制作时读取当前工作区 EDA Toolkit v0.3.1 源码；配置 schema 1.1。这是离线参考，运行时以实际解释器/EXE 的契约为准。

## 能力发现命令

源码模式：从项目目录调用已存在的 `.venv/Scripts/python.exe -X utf8 -m eda_toolkit`，便携模式调用其完整目录内的 `eda_cli.exe`。以下为尾部参数，不混用不同安装的执行器：

```text
--help
version --json
capabilities --json
presets --name single_event --json
templates --json
validate --config <完整路径> --json
plan --config <完整路径> --json
preflight --config <完整路径> --json
run --config <完整路径> --json
```

模板导出：`templates --name events --output <新CSV路径>`。实际支持九类表：input_manifest、signals、participants、events、conditions、group_membership、sync_anchors、manual_qc、statistical_design。本包 assets/table-contracts 带空表和字段说明；未提供的数据不复制 `.example.csv` 冒充实际数据。

## 重要字段与差异

- JSON `statistics.designs[].design_type` 接受：descriptive、one_sample、paired、welch、correlation、mixed_effects、wilcoxon、mannwhitney。
- 当前 statistical_design CSV schema 枚举另含 `independent_correlation`、`random_intercept_lmm`，与运行统计的名称不一致。优先用 JSON 内嵌 designs，并令 input.statistical_design=null；不要把这两个名称写到 JSON，也不要声称 CSV 别名可被安全自动映射。
- JSON 用 subject_column/session_column；CSV 支持 subject_id_column/session_id_column 的别名。JSON 编译器必须遵循当前 DESIGN_KEYS。
- independent_unit 可用 subject_id/group_id/session_id（亦有 subject/group/session 别名）。session 仅在研究设计确实支持会话独立时使用；多次会话不自动独立。
- 输入表 `group_id` 是互动组，`experimental_group` 是实验组别。不能互换。
- paired/wilcoxon 显式 conditions=[A,B] 和 condition_column，报告 A−B。welch/mannwhitney 显式 groups=[G1,G2] 和 group_column，报告 G1−G2。mixed_effects 的首个 condition 是参考。
- 重复行需 aggregate=mean/median；当前 LMM 同样会先按独立单位与条件汇总，不能声称它保留全部 trial-level 变化。软件当前仅支持随机截距结构；随机斜率、多成员/交叉层级等需要另经验证的执行器。
- planned_hypotheses 使用 `{hypothesis_id, correction_family}` 对象。普通检验 ID 为 design_id；混合模型各系数 ID 为 `design_id:Intercept`、`design_id:condition:B` 等，按真实 condition_column 和水平组成。全体预定系数及其家族在运行前固定，不能结果出来后重建。
- 完整预设包含 schema_version/pipeline_id/preset_version、input、mapping、qc、preprocess、decomposition、scr、windows、features、synchrony、statistics、compute、output、workspace、runtime。未知字段拒绝。
- features.ids=[] 意味着默认选择，不能解释为“没有特征”；正式配置应显式选择。特征实际定义从 list_feature_definitions(config) 或 plan/capabilities 读取，不能用中文展示名替代 ID。
- mapping.columns 是规范字段→源列名映射。身份常量可来自 mapping.constants/manifest；以原始身份来源为准。time_s 和 eda_uS 可作为标准列名，其他幅值列须声明 eda_unit。
- input.base_directory 以配置所在位置解析；信号/伴随表依该 base 或 manifest 位置解析。避免相对路径重复嵌套，交付时优先明确绝对 base 和 output。

## 统计窗类型混入风险（必须处理）

pipeline 将完整 features 长表交给统计，只按 outcome/feature_id 和条件/组筛选，没有“统计只用某种 kind”的配置字段。aggregate=mean 不会自动替你选择正确窗，可能把 whole、fixed、event、condition 多次计入。

- 条件块比较：windows.whole_record=false、fixed_length_s=null，input.events=null，仅保留所需 conditions。
- 事件反应：windows.whole_record=false、fixed_length_s=null；选择 event 专属 feature ID；conditions 如用于合法标签校验可保留，但确认该 outcome 在非 event 窗没有有效行。
- 全记录独立组比较：whole_record=true、fixed_length_s=null、events=null、conditions=null。
- 需要同时展示多种窗口时另建描述性配置；主推断配置只产生其预定分析层级。需要更复杂筛选或层级权重时先建设并验证外部派生统计输入，不能塞入未经支持的字段。
- 同步结果在独立结果表中；当前 pipeline 内置统计使用 features 表，不会自动把 synchrony pair/group 结果送入统计。同步推断需明确独立统计适配，不得声称 `statistics.enabled=true` 即完成同步显著性检验。

## 输出与成功含义

run 输出目录保存 resolved_config.json、输入/环境与清单及结果文件，具体文件以该次 manifest 为准。统计表含 p_raw/p_adjusted、estimate/effect_size、ci_low/ci_high、status/reason，诊断另表。
CLI 状态常见退出码：0 成功、2 配置/输入错误、3 依赖、4 运行失败、5 部分成功、130 取消；不能只以“有文件生成”判断成功。preflight 和 plan 的内部状态也必须读取。

## 离线文件

assets/default-config.json：完整预设快照，未绑定真实数据；禁止直接作为真实实验执行配置。
assets/contract-snapshot.json：制作时软件版本、源码文件哈希、能力快照；用途是识别差异，不代表未来环境已验证。
scripts/check_bundle.py：依赖 Python 标准库做第一道静态检查；提供 --toolkit-root 时调用当前源码 resolve_run_config 和特征注册表，不修改或运行真实数据。它检查不了实验设计真实性、模型假设、实际 CSV 外键和生理有效性。
