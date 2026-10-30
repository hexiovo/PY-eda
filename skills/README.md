# EDA Study Config 技能

`eda-study-config` 用于把 EDA 研究方案和数据说明整理为可核验的分析配置、参数记录、数据准备清单和操作卡。它明确区分已知事实、待确认事项与工程参数，不会替研究人员臆测单位、身份关系、事件时间或统计设计。

本技能适配 EDA Toolkit v0.3.1。单独安装技能只会让 Codex 获得配置编制指引，不会安装工具包或代替真实数据验证。需要运行 validate、plan、preflight 或分析任务时，还应准备当前版本的执行环境：

- **Windows 便携版**：解压 EDA Toolkit v0.3.1 Windows x64 包，使用包内的 `eda_gui.exe` 或 `eda_cli.exe`。输入给配置助手的工具目录应指向这次解压出的目录。
- **源码环境**：在项目源码和已准备好的 Python 环境中安装项目（`pip install .`），并使用该环境提供的执行命令。

技能源码位于项目或发布包的 `skills/eda-study-config/`。安装后的技能副本位于当前 Windows 用户的 `%USERPROFILE%\.codex\skills\eda-study-config\`。EDA Toolkit 的便携版或源码环境另行保存在项目使用者选定的位置；技能安装路径不是工具包运行路径，也不应把某台机器上的绝对路径复制到研究配置中。

## Windows 安装

从项目源码目录运行：

```powershell
.\skills\install_skill.ps1
```

如果在已解压的发布包根目录运行，并且包内提供了安装脚本：

```powershell
.\install_skill.ps1
```

安装程序会将 `eda-study-config` 复制到当前用户的 Codex 技能目录。若该技能已经安装，程序会停止且不覆盖现有文件。确认要更新时运行：

```powershell
.\skills\install_skill.ps1 -Force
```

发布包根目录下的安装脚本则使用 `.\install_skill.ps1 -Force`。`-Force` 会覆盖同名文件并合并目录，不会删除目标目录中其他已有内容。安装仅写入当前用户的技能目录，不修改全局环境变量。

也可以手动首次安装：在 PowerShell 中从包含 `skills/` 的源码或发布目录运行：

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.codex\skills" | Out-Null
Copy-Item -LiteralPath ".\skills\eda-study-config" -Destination "$env:USERPROFILE\.codex\skills" -Recurse
```

安装后，新技能不会自动加载到已经打开的对话中。请开始一个新 Codex 对话；如技能仍未显示，退出并重新启动 Codex 后再新开对话。

## 使用示例

在新对话中说明要使用本技能，并提供研究方案及可用的数据说明，例如：

> 请使用 eda-study-config 技能，依据附件中的研究方案和数据列说明，整理配置草案、decision-log、数据准备清单、操作卡及核验步骤。工具包版本为 v0.3.1，当前运行环境是 Windows 便携版。无法确认的事实请列出，不要代填；没有实际执行的校验请明确标注。

请提供能够说明实验条件、事件定义、独立单位、采样率和信号单位的资料；不需要为了回答可从匿名表头或设备说明中确认的问题而上传整份敏感数据。若尚未安装工具包，也可先整理草案，但结果必须标明真实执行器校验未完成。
