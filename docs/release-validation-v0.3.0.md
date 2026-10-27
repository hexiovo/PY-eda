# v0.3.0 验证记录

2026-10-27，Windows 开发机，项目独立 .venv。

- 全量源码回归：209 passed，40 条既有 Qt 弃用警告，无失败、错误或跳过。
- Driver 定向验证：19 passed；包含三个固定随机种子的算子/梯度性质、独立稠密有界最小二乘对照、已知脉冲恢复及边界/集成测试。
- 默认滤波基线：已有 60 行特征结果逐值完全一致。
- 冻结验证：6 个进程场景通过（3 个预设计划、能力检查、Driver 分析、GUI 启动），外部 PATH 限制为 System32。
- 源码与冻结结果逐值完全一致：408 行特征、2884 行信号、78 行 Driver 脉冲、4 行分解诊断。
- 中文 PDF 共 14 页，逐页渲染检查通过；源码与包内 PDF 的 SHA256 相同。
- ZIP 每个文件按清单复核 SHA256；详细记录随便携包 validation 与 release_manifest.json 提供。

复核命令：`.venv/Scripts/python.exe -m pytest -q`；`scripts/verify_driver_release.py`；`scripts/finalize_driver_release.py`。详细本机证据保存在 `artifacts/verification/driver-v0.3.0/`，不上传输入/运行数据到源码仓库。

方法定义见 [Driver 说明](driver-method.md)。验证范围是指定模型与合成数据、当前 Windows 开发机和离屏 GUI；没有完成干净 Windows 或真实设备/任务资格验证。历史 v0.1/v0.2 文档仅记录相应历史构建。
