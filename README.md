# EDA Toolkit

v0.3.0 新增 9 项 Driver 指标及非负 Bateman 稀疏分解，完整目录 144 项。选择 `single_driver` 预设启用；方法、单位、参数和边缘处理见 [Driver 说明](docs/driver-method.md)。滤波 single/group_sync 仍适用 124/135 项，Driver single/group 适用 133/144 项。

EDA Toolkit 0.3.0 是用于皮肤电活动（EDA）数据整理、质控、分段预处理、特征提取、多人同步和结果导出的本地工程。核心数据处理通过普通 Python 函数、NumPy 数组与 pandas 表运行；PySide6 只由图形界面层使用。

本版本包含 CSV 映射与九类表格模板、可撤销审计的人工 QC、连续段处理、SciPy 独立滤波分离、SCR 检测、四种窗口、统计接口、CLI、中文 GUI，以及 CSV/JSON/HTML/PNG/XLSX/PDF 输出。每次运行保存配置、输入哈希、环境和结果清单。参数仍是研究示例；软件运行成功不等于设备适用、科学验证或可用于临床判断。

## Windows 便携工程候选版

打开 `dist/v0.3/EDA_Toolkit-v0.3.0/eda_gui.exe`；命令行使用同目录的 `eda_cli.exe`。ZIP 位于 `dist/EDA_Toolkit-0.3.0-Windows-x64.zip`。保留整个便携目录，无需安装 Python、MATLAB 或 CUDA。中文手册在便携目录的 `docs/EDA使用指南.pdf`，也可从软件“帮助”菜单打开。

在便携目录中可创建一个新的合成演示项目（目录必须为空）：

~~~powershell
& '.\eda_cli.exe' demo --directory "$env:USERPROFILE\Documents\EDA演示"
& '.\eda_cli.exe' run --config "$env:USERPROFILE\Documents\EDA演示\single.json" --json
~~~

当前本机冻结验证与干净系统资格分开记录；验证报告明确测试所对应的构建快照。

## 源码环境

项目建议 Python 3.11 及以上。当前开发环境使用项目内的 .venv；记录到的实际解释器为 Python 3.14.8。下列命令显式调用项目解释器，不安装到全局环境：

~~~powershell
& '.\.venv\Scripts\python.exe' -m pip install -e '.[dev]'
~~~

运行依赖包括 NumPy、SciPy、pandas、PySide6、Matplotlib、openpyxl、ReportLab 与 statsmodels；pytest 属于开发依赖，PyInstaller 属于可选构建依赖。实际已解析版本见 [环境与来源记录](docs/provenance.md) 和 requirements.lock.txt。安装记录来自开发机，不是干净 Windows 系统验证。

## 启动

查看 CLI 命令：

~~~powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m eda_toolkit --help
~~~

启动中文图形界面：

~~~powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m eda_toolkit.ui.app
~~~

如果通过 pip 安装了本项目，eda-cli 与 eda-gui 也分别是命令行和图形界面入口。Windows x64 便携目录为 `dist/v0.3/EDA_Toolkit-v0.3.0`，双击其中 `eda_gui.exe` 启动界面；`eda_cli.exe` 为无界面入口。保留整个目录及 `_internal`、`third_party`，不能只复制一个 EXE。开发机受限 PATH 的冻结进程检查见 `artifacts/verification/v0.2/frozen-application-r2.json`；当前归档核验和摘要见源码仓库中的 `artifacts/verification/v0.2/archive-integrity-current.json` 与 `artifacts/verification/v0.2/release-summary-current.json`。便携包内文件清单为 `release_manifest.json`。这些证据均不代表干净 Windows 验收。

## 用合成数据跑通工程路径

仓库附带 examples/current_synthetic_demo，其中的四条记录、事件、条件和同步元数据全部是确定性生成的工程演示数据，不是被试数据。它用于检查软件路径和输出结构，不证明算法对设备或研究任务有效。

校验配置并查看导入诊断：

~~~powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m eda_toolkit validate --config examples/current_synthetic_demo/single.json --json
~~~

查看运行计划：

~~~powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m eda_toolkit plan --config examples/current_synthetic_demo/single.json --json
~~~

执行一次合成演示流程：

~~~powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m eda_toolkit run --config examples/current_synthetic_demo/single.json --json
~~~

配置将运行结果写入 examples/current_synthetic_demo/results 下的新运行目录。不要把演示配置的结果解释为生理或实验效应。CLI 还提供 version、capabilities、presets、templates、demo、resume、retry 与 export 子命令；详细步骤见 [快速入门](docs/快速入门.md)。

## 从自己的 CSV 开始

先检查 [CSV 字段字典](docs/CSV字段字典.md) 和九模板，再用 validate 确认输入、单位、时钟、关联键和样本结构。带时间列时，时间必须有限且严格递增；没有时间列时必须声明采样率和时间单位。自定义 EDA 列必须显式声明单位。缺失幅值保持为缺失，不会填零。原始输入以被读取的同一字节流计算哈希，不会被改写。

建议先复制配置到自己的项目目录，修改 input 与 mapping，运行 validate，再运行 plan。只有在核对单位、时钟和缺失情况后才进行完整分析。各页面实际操作与截图见 [操作指南](docs/操作指南.md)。

## 方法与证据边界

默认分解方法是本地 SciPy 二阶 Butterworth SOS 的独立低通/高通滤波分离，不是 NeuroKit、cvxEDA 或 Ledalab 的兼容实现。SCR 阈值、窗口和上下文都可配置，但默认数值没有按真实设备/任务校准。同步 surrogate 的可达性和统计尾部概率仍属实验路线；相关实现必须标记实验性或不可用，不能称为已校准显著性检验。

截至 2026-10-26，v0.2 源码套件有 166 项通过，0 失败、错误或跳过；测试报告记录 40 条 Qt 弃用警告。Windows x64 冻结双 EXE 的 20 个预期进程场景符合预期，归档核验覆盖 1665 个 bundle 文件和 1666 个 ZIP 成员。以上是开发机工程证据，不代表干净 Windows、真实设备科学适用性、零模型校准、任务栏视觉或正式对外发行许可已经通过。完整边界见 [交付状态](docs/交付状态.md) 与 [验证与限制](docs/验证与限制.md)。

## 文档

- [快速入门](docs/快速入门.md)
- [操作指南](docs/操作指南.md)
- [方法指南](docs/方法指南.md)
- [函数开发指南](docs/函数开发指南.md)
- [特征字典](docs/特征字典.csv)
- [验证与限制](docs/验证与限制.md)
- [CSV 字段字典](docs/CSV字段字典.md)
- GUI 内置中文帮助 PDF：src/eda_toolkit/resources/help/EDA使用指南.pdf

## 构建 v0.3.0

在项目 .venv 安装构建依赖后运行 `scripts/build_windows_current.ps1`，再运行 `scripts/verify_driver_release.py` 和 `scripts/finalize_driver_release.py`。构建脚本拒绝覆盖已有同版本便携目录或 ZIP。源码仓库不包含本地运行数据、构建缓存和 EXE；当前交付验证见 [v0.3.0 验证记录](docs/release-validation-v0.3.0.md)。
