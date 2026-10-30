# 来源、版本与构建

本仓库当前源码版本为 EDA Toolkit v0.3.1。发行压缩包内的 `release_manifest.json` 记录便携目录文件的大小与 SHA256，并单独记录 `eda_gui.exe`、`eda_cli.exe` 的哈希。发布附件 `SHA256SUMS.txt` 用于校验便携压缩包和 `eda-study-config` 技能包。

## 依赖与许可证材料

- `requirements.lock.txt` 列出项目构建环境记录的依赖版本；`requirements.lock.sha256.txt` 在有对应 wheel 时附带 SHA256。
- `third_party/sbom-installed.json` 与 `third_party/wheel_inventory.json` 记录本次构建环境中的组件及 wheel 来源信息。
- `third_party/licenses/` 保存构建环境附带的许可证文本副本；`third_party/THIRD_PARTY_NOTICES.md` 汇总组件声明。
- 这些记录用于核对软件制品和来源，不构成法律审查或发行义务已获批准的证明。

## 重新构建

在 Windows x64 环境中准备符合 `pyproject.toml` 要求的 Python 版本和依赖，再运行：

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -e '.[dev,build]'
& '.\.venv\Scripts\python.exe' -X utf8 scripts/build_help.py
& '.\scripts\build_windows_current.ps1' -Clean
```

`build_windows_current.ps1` 使用项目 `.venv` 中的 Python、PyInstaller 与 `build_support/eda.spec`，先生成中文帮助 PDF 和当前界面截图，再生成双入口便携包、技能包及校验清单。构建过程拒绝覆盖已有 v0.3.1 发布文件。详细使用边界见[验证与限制](验证与限制.md)。
