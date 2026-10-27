# CSV 字段字典

版本：schema 1.0。所有模板、字段 schema、示例均随包保存在 `eda_toolkit.resources.templates`，可由 `eda_toolkit.core_io` 的模板函数读取或导出。`.example.csv` 文件及其中所有 subject、recording、group、事件、锚点和统计设计均为 **SYNTHETIC_ONLY**；它们不是用户数据，也不证明方法有效。

## 通用读入规则

- CSV 原始字节只读取一次；同一字节串用于 SHA-256 和解析。读取前后的文件大小、修改/创建时间和文件标识发生变化时拒绝导入。源文件不修改。
- 默认列名按模板标准字段精确匹配，不做模糊猜列；自定义列必须通过 `mapping.columns` 明确对应。可在 `input.encoding`、`input.delimiter` 设置编码和单字符分隔符。UTF-8 BOM、UTF-8 和 GB18030 可识别；预览最多缓存 500 行，但会遍历 CSV 以返回准确总行数。
- 标准来源列 `eda_uS` 自带单位 uS。映射到其它幅值列时必须在 mapping 或该文件的 manifest 项中声明 `eda_unit`；逐行 `eda_unit` 列也可提供显式单位。支持 uS/µS/μS、mS、S、ohm、kohm、Mohm。导入后电导统一为 uS；电阻仅允许正值并按 `1e6 / R_ohm` 换算。缺失幅值保留为 NaN；正负无穷拒绝。
- 有时间列时按声明的 `time_unit`（s、ms、us）换成相对秒，先用 Decimal 减去原点，再转成浮点。未指定原点时，以各 recording/channel 第一条源时间作相对原点并记录原始字符串。缺时间列时必须同时给出正的 `sampling_rate_hz` 和 `time_unit`，按每条记录独立生成时间轴并标明来源。
- recording、channel 内的时间必须按输入原顺序有限且严格递增；重复、倒序、缺失或无穷时间拒绝，不排序、不删行。声明采样率与时间戳稳健估计分别记录，差异只报告不静默改写。不能从来源得到的 recording_id 使用文件名 stem；缺失 subject_id/session_id 使用 `unknown`。仅这些推导值的 `metadata.generated` 标记为 true，不能当作实验身份事实。
- 每个伴随时间表都要声明 `clock_domain`。秒单位相同不代表来源时钟相同。多通道事件、条件和人工 QC 必须填写具体 `channel` 或显式 `*` 广播；单通道记录可省略。
- `validate_table(name, frame, tables=None)` 返回 `{valid, errors, warnings, table, rows}`。字段类型、必填/缺失、允许值、唯一组合及可提供的外键都会检查；`load_recordings` 另外验证实际信号 recording/channel、参与者/会话身份、事件条件、组成员、锚点、人工 QC 和时钟域。发现错误时拒绝接收并保留结构化错误码。

## 模板函数

```python
list_templates() -> list[dict]
get_template_path(name, kind="template") -> Traversable
read_template(name, kind="template") -> pandas.DataFrame
read_template(name, kind="example") -> pandas.DataFrame
read_template(name, kind="schema") -> dict
export_template(name, destination, kind="template") -> pathlib.Path
preview_csv(path, max_rows=50, options=None) -> dict
read_signal_csv(path, mapping=None, options=None) -> (samples, metadata, report)
load_recordings(manifest=None, mapping=None, options=None) -> (records, tables, report)
```

`options` 可传整份 pipeline config，也可传 mapping/input 小节。`manifest` 接受 manifest CSV 路径、DataFrame、单个行字典、行字典列表、信号文件路径列表或 `None`；`None` 优先读 `input.manifest`，否则读 `input.signals`。伴随表由 `input.participants/events/conditions/group_membership/sync_anchors/manual_qc/statistical_design` 给出，路径相对 `input.base_directory` 或清单文件目录。

`samples` 是标准化长表，含 `recording_id, subject_id, session_id, channel, time_s, source_time, eda_uS, sample_index, extras`，并把不与标准列重名的未解释来源列直接保留。`source_time` 保留逐行原始坐标字符串，`time_s` 是减去时间原点后的相对秒；构造 `record.extras` 时也会保留原始时间坐标。`records` 按 recording_id 与 channel 分组，使用共享的 `make_record` 契约，元数据保留 source path/hash、来源/标准单位、声明/估计速率、时间原点、时钟域、映射版本和推导身份标记。`tables` 包含清单及已读伴随表；文件表保留 `attrs.source_path` 和 `attrs.source_sha256`。`report.consumed_inputs` 为实际消费的绝对路径到 SHA-256 的映射，控制器可据此检查输入是否改变。

## 九类模板字段

表中“缺失”说明 CSV 单元格级规则；身份可由 manifest 或映射常量提供的字段，在“必填”列注明条件要求。未知额外列保留为未解释字段，不自动进入分析。

### signals — 信号采样长表

| 字段 | 类型 / 单位 | 必填 / 缺失 | 外键 | 时钟 / 说明 |
| --- | --- | --- | --- | --- |
| recording_id | string / 无 | 必须有来源列或 manifest/映射常量；解析后禁止缺失 | recording 标识 | 缺来源时用文件名 stem 并标记 generated |
| subject_id | string / 无 | 可由 manifest 补；解析后缺失时为 `unknown` | `participants.subject_id`（若提供参与者表） | `001` 等前导零保留 |
| session_id | string / 无 | 可由 manifest 补；缺失时为 `unknown` | — | 推导值标记 generated |
| channel | string / 无 | 可由 manifest 补；缺失时为 `unknown` | recording/channel 组合 | 多通道文件须映射或按清单筛选 |
| time_s | number / 秒 | 有列时不能有缺失；可没有列 | — | `clock_domain`；原顺序、严格递增 |
| eda_uS | number / uS | 必须映射；缺失保留 NaN | — | 非有限值拒绝；仅此标准来源列自带 uS 单位 |
| clock_domain | string / 无 | 可用 mapping 默认补齐 | — | 每条时间坐标的时钟身份 |

### participants — 参与者信息

| 字段 | 类型 / 单位 | 必填 / 缺失 | 外键 | 说明 |
| --- | --- | --- | --- | --- |
| subject_id | string / 无 | 必填且唯一 | 信号与成员表引用此键 | 不裁剪前导零 |
| experimental_group | string / 无 | 可空 | — | 实验条件/组别，不等同互动组 group_id |
| age_years | number / 年 | 可空 | — | 不自动推补或重编码 |
| sex | string / 无 | 可空 | — | 类别定义由研究方案声明 |
| 额外协变量 | 保留来源类型 / 方案声明单位 | 可空 | — | 原样保留，使用前须在设计中明确映射 |

### events — 事件

| 字段 | 类型 / 单位 | 必填 / 缺失 | 外键 | 时钟 / 说明 |
| --- | --- | --- | --- | --- |
| recording_id | string / 无 | 必填 | 已导入 signals | — |
| event_id | string / 无 | 必填；与 recording_id 组合唯一 | — | — |
| onset_s | number / 秒 | 必填 | — | `clock_domain` |
| duration_s | number / 秒 | 必填，有限非负数 | — | `clock_domain` |
| event_type | string / 无 | 必填 | — | 事件名称/类别 |
| condition | string / 无 | 可空 | `conditions.condition`（有条件表时） | 保留原类别 |
| channel | string / 无 | 单通道可空；多通道必填 | signals recording/channel | 填具体通道或显式 `*` 广播 |
| clock_domain | string / 无 | 必填 | — | 必须与对应信号记录一致 |

### conditions — 条件时间区间

| 字段 | 类型 / 单位 | 必填 / 缺失 | 外键 | 时钟 / 说明 |
| --- | --- | --- | --- | --- |
| recording_id | string / 无 | 必填 | 已导入 signals | — |
| interval_id | string / 无 | 必填；与 recording_id 组合唯一 | — | — |
| start_s / end_s | number / 秒 | 必填，且 start < end | — | `clock_domain`；半开区间 `[start,end)` |
| condition | string / 无 | 必填 | 可供 events 引用 | 类别由设计声明 |
| channel | string / 无 | 单通道可空；多通道必填 | signals recording/channel | 具体通道或显式 `*` 广播 |
| clock_domain | string / 无 | 必填 | — | 必须与信号时钟一致 |

### input_manifest — 输入清单

| 字段 | 类型 / 单位 | 必填 / 缺失 | 外键 | 说明 |
| --- | --- | --- | --- | --- |
| file_path | path / 无 | 必填 | — | 相对 `base_directory` 或 manifest 文件目录 |
| recording_id | string / 无 | 可空 | 已导入 signals | 单文件多记录时由来源列提供 |
| subject_id | string / 无 | 可空 | `participants.subject_id`（若表提供） | 显式清单身份优先补缺值，冲突拒绝 |
| session_id | string / 无 | 可空 | — | 显式清单身份优先补缺值，冲突拒绝 |
| sampling_rate_hz | number / Hz | 无时间列时必填 | — | 有时间列时作为独立声明保留 |
| eda_unit | string / 无 | 自定义幅值来源时必填 | — | uS、mS、S、ohm、kohm、Mohm |
| mapping_profile | string / 无 | 可空 | — | 未知 profile 拒绝，不推测列映射 |
| channel | string / 无 | 可空 | signals recording/channel | 多通道筛选范围 |
| clock_domain | string / 无 | 可空 | 信号元数据 | 有来源声明时必须一致 |
| time_origin | string / 来源时间单位 | 可空 | — | 精确来源坐标原点；记录原始表示 |
| time_unit | enum / s、ms、us | 无时间列时必填 | — | 来源时间单位；`sampling_rate_hz` 仍按每秒采样 |

### group_membership — 互动组成员

| 字段 | 类型 / 单位 | 必填 / 缺失 | 外键 | 说明 |
| --- | --- | --- | --- | --- |
| group_id | string / 无 | 必填 | — | 社会互动/配对组，不是实验组别 |
| session_id | string / 无 | 必填 | 信号会话身份 | 与关联 recording 的会话一致 |
| subject_id | string / 无 | 必填 | participants（若提供）及信号元数据 | 与 recording 被试一致 |
| recording_id | string / 无 | 必填 | 已导入 signals | — |
| role | string / 无 | 可空 | — | 研究需要时声明互动角色 |
| channel | string / 无 | 可空 | signals recording/channel | 声明所用通道 |

### sync_anchors — 同步锚点

| 字段 | 类型 / 单位 | 必填 / 缺失 | 外键 | 时钟 / 说明 |
| --- | --- | --- | --- | --- |
| recording_id | string / 无 | 必填 | 已导入 signals | — |
| anchor_id | string / 无 | 必填；与 recording_id/channel 组合唯一 | — | — |
| local_time_s | number / 秒 | 必填 | — | `clock_domain` |
| reference_time_s | number / 秒 | 必填 | — | `reference_clock_domain` |
| channel | string / 无 | 可空 | signals recording/channel | 多通道时建议明确 |
| clock_domain | string / 无 | 必填 | — | 记录本地时钟 |
| reference_clock_domain | string / 无 | 必填 | — | 对齐目标时钟；不由单位推断 |
| uncertainty_s | number / 秒 | 可空 | — | 时间不确定度 |
| quality | string / 无 | 可空 | — | 锚点来源/质量说明 |

### manual_qc — 人工 QC 历史

| 字段 | 类型 / 单位 | 必填 / 缺失 | 外键 | 说明 |
| --- | --- | --- | --- | --- |
| annotation_id | string / 无 | 必填且唯一 | — | 每次操作的唯一 ID |
| recording_id | string / 无 | 必填 | 已导入 signals | — |
| channel | string / 无 | 可空 | signals recording/channel | 具体通道或显式 `*` |
| start_s / end_s | number / 秒 | exclude/restore 必填，undo/redo 可空 | — | 非空半开区间 `[start,end)`，`clock_domain` |
| action | enum / 无 | 必填 | `exclude`、`restore`、`undo`、`redo` | exclude 优先；restore 不复活缺失/不可信样本 |
| reason / operator / timestamp | string / 无 | 必填 | — | 保留操作理由、操作者和实际操作时间 |
| revision / parent_revision | integer / 修订号 | 必填 | — | 全历史单调追加；父版本不匹配拒绝 |
| target_annotation_id | string / 无 | undo/redo 必填 | `manual_qc.annotation_id` | 撤销/重做目标；旧记录不删除 |
| clock_domain | string / 无 | 必填 | — | 与 recording 时间坐标一致 |

### statistical_design — 统计设计

| 字段 | 类型 / 单位 | 必填 / 缺失 | 外键 | 说明 |
| --- | --- | --- | --- | --- |
| design_id | string / 无 | 必填且唯一 | — | 设计版本身份 |
| outcome | string / 无 | 必填 | — | 预先定义的结果变量或特征名 |
| design_type | enum / 无 | 必填 | — | descriptive、one_sample、paired、welch、independent_correlation、random_intercept_lmm |
| independent_unit | enum / 无 | 必填 | — | subject、group 或 session |
| subject_id_column | string / 无 | 可空 | participants 字段 | 字段名，不是参与者取值 |
| group_column | string / 无 | 可空 | — | 实验条件字段，不是社会互动 group_id |
| session_id_column | string / 无 | 可空 | — | 字段名 |
| condition_column | string / 无 | 可空 | conditions 字段 | 字段名 |
| covariates / contrast | JSON / 无 | 可空 | — | 结构化值；不得存可执行公式/代码 |
| correction_family | string / 无 | 可空 | — | 预先声明的多重检验家族 |
