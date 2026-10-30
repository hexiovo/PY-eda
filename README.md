# EDA Toolkit

EDA Toolkit 是用于本地整理和分析皮肤电活动（EDA）数据的软件，涵盖数据导入与质量检查、预处理、特征提取、多人同步、统计分析和结果导出。当前程序版本为 **v0.3.1**，最新分发版本为 **v0.3.1-r1**。

软件运行完成只表示本次流程已执行；这本身不能证明设备适用、参数已校准或研究结论成立。开始分析前，请阅读[验证与限制](docs/验证与限制.md)。

## Windows 便携版

在 [v0.3.1-r1 发布页](https://github.com/hexiovo/PY-eda/releases/tag/v0.3.1-r1)查看版本信息，或直接下载 [EDA_Toolkit-0.3.1-r1-Windows-x64.zip](https://github.com/hexiovo/PY-eda/releases/download/v0.3.1-r1/EDA_Toolkit-0.3.1-r1-Windows-x64.zip)。下载后解压，在解压目录中双击 **启动EDA工作台.cmd** 或 **eda_gui.exe** 打开图形界面；命令行入口是同目录的 **eda_cli.exe**。便携版不需要另装 Python、MATLAB 或 CUDA。

请保留解压目录的完整结构，包括 **_internal**、**third_party**、**docs**、**assets**、**presets** 和 **skills**；不要单独移动 EXE。中文手册 **docs/EDA使用指南.pdf** 也可从软件的“帮助”菜单打开。

在解压目录中打开 PowerShell，可查看实际运行版本：

~~~powershell
& '.\eda_cli.exe' version --json
~~~

### 运行合成演示

演示数据用于熟悉命令和输出，不是实验数据，也不能证明设备或方法适用。以下命令会在当前 Windows 用户的“文档”目录下创建演示包；目标目录必须尚不存在。若目录已存在，请把名称改成新的目录。

~~~powershell
$demoDirectory = Join-Path $env:USERPROFILE 'Documents\EDA演示-v0.3.1'
& '.\eda_cli.exe' demo --directory $demoDirectory
if ($LASTEXITCODE -ne 0) { throw '演示数据生成失败。' }

$configPath = Join-Path $demoDirectory 'single.json'
& '.\eda_cli.exe' validate --config $configPath --json
if ($LASTEXITCODE -ne 0) { throw '演示配置检查失败。' }
& '.\eda_cli.exe' plan --config $configPath --json
if ($LASTEXITCODE -ne 0) { throw '演示计划检查失败。' }
& '.\eda_cli.exe' run --config $configPath --json
if ($LASTEXITCODE -ne 0) { throw '演示运行失败。' }
~~~

演示结果写入配置指定的位置。请勿把演示数据中的参数直接用作设备或研究设计建议。

### 使用自己的数据

开始前请阅读[快速入门](docs/快速入门.md)、[CSV 字段字典](docs/CSV字段字典.md)和[操作指南](docs/操作指南.md)。从适用的预设或模板创建配置，依据实际文件填写路径、列名、单位、采样率或时间列、参与者和实验设计信息。先执行 validate 和 plan，查看导入报告、完整诊断与运行计划，再决定是否开始分析。原始输入保持不变，结果写入新的运行目录。

## 分析方法与限制

- 默认 filter_separation 使用 SciPy 二阶 Butterworth SOS 对信号作独立低通/高通分离；它不是 NeuroKit、cvxEDA 或 Ledalab 的兼容实现。
- bateman_sparse 是本地非负稀疏分解方法。Driver 指标、滤波设置、SCR 阈值和窗口需结合真实设备与任务评估，不能直接视为已校准参数。
- 同步 surrogate/null 路径仍属于实验方法；不可用的结果会标明状态和原因，不应解释为“没有效应”。
- 软件不作压力、情绪或疾病判断，也不替代信号质量评估和研究设计审查。

方法定义见[方法指南](docs/方法指南.md)、[Driver 分解说明](docs/driver-method.md)和[验证与限制](docs/验证与限制.md)。

## 研究配置技能

eda-study-config 技能可根据研究方案整理配置草案、参数记录、数据准备清单和操作卡。可直接下载[eda-study-config-0.3.1.zip](https://github.com/hexiovo/PY-eda/releases/download/v0.3.1-r1/eda-study-config-0.3.1.zip)，其中含安装脚本和使用说明；技能边界见[技能说明](skills/README.md)。在源码目录运行 PowerShell 脚本 `skills/install_skill.ps1`；解压技能压缩包后，在该目录运行 `install_skill.ps1`。技能安装完成后开始新的 Codex 对话；若技能未显示，可重启 Codex 后再开新对话。

该技能不安装 EDA Toolkit，也不会代替真实材料的校验。实际能力以当前运行环境的版本、功能、模板及检查结果为准。

## 从源码运行

源码环境需要 Python 3.11 或更新版本。在项目目录打开 PowerShell：

~~~powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install .
& '.\.venv\Scripts\python.exe' -X utf8 -m eda_toolkit --help
& '.\.venv\Scripts\python.exe' -X utf8 -m eda_toolkit.ui.app
~~~

命令行支持 validate、plan、run、resume、retry、export、presets、templates 和 demo。依赖版本见 [requirements.lock.txt](requirements.lock.txt)，来源记录见[provenance](docs/provenance.md)。

## 文档

- [快速入门](docs/快速入门.md)
- [操作指南](docs/操作指南.md)
- [方法指南](docs/方法指南.md)
- [CSV 字段字典](docs/CSV字段字典.md)
- [特征字典](docs/特征字典-v1.1.csv)
- [扩展指标与结果表](docs/扩展指标与结果表-v1.1.md)
- [验证与限制](docs/验证与限制.md)

## 构建便携版

在已安装项目依赖的 Windows 环境中运行：

~~~powershell
& '.\.venv\Scripts\python.exe' -m pip install -e '.[dev,build]'
& '.\.venv\Scripts\python.exe' -X utf8 scripts/build_help.py
& '.\scripts\build_windows_current.ps1' -Clean
~~~

构建脚本会生成 v0.3.1 双入口便携目录、Windows x64 压缩包、技能包和 SHA256 校验清单。构建输出位于 dist/，PyInstaller 中间文件位于 build/；这些文件不纳入源码提交。
