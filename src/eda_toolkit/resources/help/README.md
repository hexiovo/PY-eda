# 内置帮助资源

GUI 的“中文帮助”和 F1 动作打开同目录的 `EDA使用指南.pdf`。程序从 `importlib.resources` 读取资源，冻结版会先复制到临时目录再交给系统 PDF 阅读器，因此不依赖当前工作目录。

`fonts/` 保存生成手册和分析报告使用的 Noto Sans SC 静态字体，以及 OFL 许可证和来源记录。`NotoSansSC-Manual-Regular.ttf` 与 `NotoSansSC-Manual-Bold.ttf` 是手册排版所需字形子集；完整 `NotoSansSC-Regular.ttf` 保留给结果 PDF 中任意中文数据字段使用。

重新生成真实界面截图和帮助 PDF：

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 scripts/build_help.py
```

该脚本使用 Qt 窗口及控件的 `grab()` 捕获截图，运行本地合成演示，并以 ReportLab 生成嵌入中文字体的 PDF。截图、字体和 PDF 均须随实际包资源清单核验。
