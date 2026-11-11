# PYEDA 一站式程序发布说明 v0.3.2

这个发布包包含 EDA 一站式 GUI、AUT/PIT 基础整理、个人回答情境扩展、EDA 特征与统计脚本，以及 `eda-study-config` 技能和脱敏示例。程序继续使用本机固定数据位置 `D:\data\PYEDA`，不会把研究数据打进发布包。

## 启动

双击 `启动PYEDA一站式.vbs`。发布包存在 `PYEDA_一站式分析.exe` 时优先使用该 GUI 可执行文件，否则使用 `pythonw.exe` 隐藏控制台窗口。GUI 会显示阶段进度、总体进度、已用时、预计剩余时间、日志和输出目录。若启动失败，会使用弹窗说明缺失的环境或文件。

GUI 启动前读取 `study-config/eta_baseline.json`。该文件保存最近一次完成运行的耗时基数、原始记录数、信号时长代理、行为工作簿体量和输入指纹。输入未变化时复用 workload cache，输入变化时才重新扫描 GSR ZIP。后续估时按记录数、平均信号时长、行为文件体量和安全系数缩放，阶段出现真实完成事件后切换为当前批次估算。

## 输出

基础结果和个人回答情境扩展分别保存到 `D:\data\PYEDA\study-config` 下的运行目录。扩展目录包含预处理、EDA 特征、响应窗口、IOC、组间比较、相关、回归、因子检验、图件、日志、checkpoint、manifest 和 APA 报告交接材料。`LATEST_RESPONSE_RUN.txt` 指向最后一次完成的扩展结果。

## 技能

`skills/eda-study-config/` 是可单独安装的技能目录。运行 `skills/install_skill.ps1 -Force` 更新当前用户的 Codex 技能副本。技能内容包含输入核验、EDA 全特征、完整统计分析、断点续跑、经验 ETA 和 APA 第 7 版报告规范。

## 运行环境

需要已经安装项目依赖的 Python 环境，或在源码目录使用 `.venv\Scripts\pythonw.exe`。发布包包含主程序源码和技能，未包含 Python 运行时、研究原始数据或个人结果。
