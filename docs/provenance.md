# 环境与来源记录

记录日期：2026-10-25（Asia/Shanghai）。项目版本：EDA Toolkit 0.1.0。本文记录当前开发环境和已存在的工程验证，不是 clean Windows 测试或发行法律审查。

## P00 / P01 边界

原 planning 中 P00 先建立需求、方法、许可、数据和验收基线；其原阶段边界不包含软件能力实现或依赖安装。本轮包骨架和独立运行环境属于 P01 工程准备。本文件不把后续 P01–P14 的软件工作追记成 P00 已交付。

## 项目解释器与安装

- 工作目录：D:\program\EDA。
- 项目隔离环境：D:\program\EDA\.venv。
- 实际解释器：Python 3.14.8 x64，来自 E:\conda\conda\python.exe。
- 项目声明范围：Python >=3.11。Python 3.12 备选解释器没有用于本次安装；没有因 wheel 解析失败回退。
- 初始安装命令：& 'E:\conda\conda\python.exe' -m venv .venv；随后 & '.\.venv\Scripts\python.exe' -m pip install -e '.[dev]'。开发机之外没有做系统级 pip 安装。
- 当前关键实际版本：eda-toolkit 0.1.0；numpy 2.5.4；scipy 1.18.1；pandas 3.1.0；PySide6 6.12.0；matplotlib 3.11.2；openpyxl 3.1.5；reportlab 5.1.0；statsmodels 0.15.0；pytest 9.1.1；PyInstaller 6.22.3；fonttools 4.67.0。
- 实际锁定与安装库存：requirements.lock.txt 与 requirements.lock.sha256.txt 各记录 43 个发行包；third_party/sbom-installed.json 现有 44 个唯一组件（43 个锁定依赖 + 项目自身）；third_party/wheel_inventory.json 记录 43 个已下载并散列的 Windows x64/Python 3.14 wheel，`hashed_lock_complete=true`；build/wheelhouse 中有相应 wheel。third_party/licenses/ 中有 136 份发行包许可证原文及 1 份 Python runtime 许可证副本。清单是当前开发/构建输入库存，不是最终 EXE 的组件闭包。
- inventory 只说明本机实际安装组件和抽取到的许可证文本。它不是拟发行依赖闭包，也不是分发许可的法律结论。最终包中的 Qt 插件、动态库、Qt 运行时替换/调试权、notice 和源码提供安排待针对实际构建物审查。
- PyInstaller 已装入开发环境，但 dist 双入口包尚未由此记录确认通过，clean Windows 验证尚未执行。

## 安装与入口复核

| 命令 / 文件 | 实际结果 | 限制 |
| --- | --- | --- |
| .venv\Scripts\python.exe -m pip check | 前次记录退出码 0，No broken requirements found. | 当前记录不把它等同于逐项运行测试。 |
| .venv\Scripts\python.exe -X utf8 -m eda_toolkit --help | CLI 帮助入口为项目模块 eda_toolkit.cli；本轮文档生成时执行成功。 | 帮助输出不验证算法数值。 |
| .venv\Scripts\python.exe -X utf8 -m eda_toolkit.ui.app | PySide6 GUI 模块入口；窗口 smoke 与截图见验证记录。 | 开发机环境，非 clean system。 |
| pyproject.toml | name=eda-toolkit，version=0.1.0，requires-python >=3.11；statsmodels 是必装依赖，pytest 为 dev，PyInstaller 为可选 build。 | 版本下限不代表所有 Python/操作系统组合都跑过。 |

所有安装与执行命令都显式指向 .venv。没有使用 py 默认启动器（当前指向 Python 3.15），没有全局 pip install，也没有安装 PyTorch、cvxpy、cvxEDA、Ledalab 或 GPU 算法依赖。

## 实际验证证据

### 定向测试与差异审阅

- 输入/QC：.venv 中执行 & '.\.venv\Scripts\python.exe' -X utf8 -m pytest -q tests/test_io_qc.py，17 passed in 0.55s。
- 独立回归：artifacts/verification/independent_review.json 记录三轮范围受限的差异审阅。首轮 tests/test_independent_validation.py 为 8 passed / 2.19 s；第二轮新增 3 个生命周期/onset/串并行检查通过，随后两个原反例在第三轮复测通过；null 专项 4 passed / 3.25 s。IR01–IR10 的指定原反例均有 targeted evidence，审阅结论是在这些差异范围内未发现新 P1/P2。该审阅不等同完整测试套件、全面数值资格或科学有效性。
- 配置：父代理记录 tests/test_contracts_config.py 8 passed。此处保留其执行来源，不冒充本记录作者独立运行。
- 特征补项：tests/test_numerics.py::test_feature_registry_selection_and_scr_upstroke_area_summaries 有 owner 报告的手算回归通过；功能选择与三个 SCR 上升面积 ID 在三轮差异审阅后加入，尚未纳入上述独立审阅。

### 合成端到端

artifacts/verification/single_pipeline.json 记录 run_id 20261025-145742-881be8e7，状态 succeeded，4 条 SYNTHETIC_ONLY 记录完成，elapsed_s=3.205886499985354，错误/警告为 0，输出 74 个 CSV/JSON/HTML/PNG/XLSX/NPZ 工件，source_integrity=exact_consumed_bytes_hashed，science_qualification=pending_representative_real_data。

此结果证明该次开发版合成流程能结束并生成工件。此历史 pipeline_state.json 缺少后来新增的 source_fingerprint 字段，因此不将它描述为当前源代码的字节级版本绑定验证，不列出演示特征值作科学结果。新运行会在 provenance / pipeline_state 写入算法源码指纹；resume 校验源码、配置、消耗输入哈希和缓存状态，变更后不沿用旧实现缓存。

更新的多人同步工程演示记录于 artifacts/verification/group_pipeline.json 和 examples/current_synthetic_demo/results/20261025-152805-28185e0f。4 条 4 Hz、600 秒合成记录完成，错误/警告为 0；两组 pair、13 个预设窗口共 26 条 two-sided null-tail 结果，每条请求和接受 99 个 surrogate，均明确为 experimental_uncalibrated。该次 pipeline_state.json 写入 source_fingerprint、config_sha256、输入哈希和 manifest；feature selector、SCR 面积汇总、图表等随后仍有源码修改，因此这也是某一时间点的工程证据，不绑定最终冻结源码。该模拟不验证 task exchangeability、I 型错误率、功效或 p 值校准。

### GUI

GUI owner 报告真实离屏 Qt smoke 成功，artifact 为 artifacts/verification/gui-workbench.png；完整窗口用 create_window 构建。本文另由 scripts/build_help.py 切换 live QStackedWidget 页面并对真实 QWidget.grab() 保存 docs/screenshots/01–08，对真实 QMenu.grab() 保存第 09 张帮助菜单；第 09 张不是第九个导航页。脚本实际调用 core_io 映射预览、按当前输入载入 QC，并通过完整 JSON 编辑器恢复四记录 manifest 后运行批处理。GUI及文档截图都来自开发环境，不能代替逐项 UI 或 clean Windows 验收。

## 帮助字体和 PDF

内置 GUI 帮助资源位于 src/eda_toolkit/resources/help/EDA使用指南.pdf，使用项目 .venv 中 ReportLab 5.1.0 生成；最终 PDF 为 14 页 A4、2,843,181 字节，SHA256 `5ac7ca6d93e3a10d21976f430b4ff7e3f992be6baee23f41097f4e1b4d664bc6`。通过 bundled Poppler `pdfinfo` 检查页数，并用 `pdftoppm` 以 100 dpi 渲染视觉检查封面、快速入门、QC/预处理、SCR/特征、同步/限制和截图页；PDF 中可见四个嵌入 TrueType 字体流，均为 NotoSansSC 子集，未见缺字或裁切。中文字体来自 Google Fonts 官方仓库的 Noto Sans SC，依据随附 OFL.txt 重新分发；完整静态 Regular SHA256 为 `baa4ded72a86948c7425f537528abc81a7f7def6b0a759d3ff63724e6f055a4f`，手册 Regular/Bold 子集分别为 `0870adebbd2b7fa44d20a337fdd54970b4473f7b19a571fa060d35091e24fef6` / `f560d7e0026ce3189bc450f3d67de533c9f196fcd75dd3119a2b0f8df1268ff5`。核心 report.pdf 已改用完整静态 Regular，保留任意中文研究字段覆盖；其 QtPdf 白底视觉证据由父代理记录。源码及 OFL 哈希写在 resources/help/fonts/README.md。

## 第三方清单

来源为本机 .venv 中实际安装并为构建准备的组件。requirements.lock.txt 与带哈希的 requirements.lock.sha256.txt 各有 43 个分发条目；wheel_inventory.json 为 43 个 wheel 提供哈希，并标记锁定完整；SBOM 有 44 个唯一组件，包括项目自身。许可证清单有 136 个发行包文本副本，另附 Python 3.14.8 runtime 的 PSF License Version 2 文本副本。以上不等于最终 frozen artifact 的依赖闭包或法律审查结论。

- third_party/sbom-installed.json：实际组件名、版本、声明许可证标识、许可证文件路径与哈希。
- third_party/licenses/：实际安装包带入的许可证文本副本。
- third_party/THIRD_PARTY_NOTICES.md：初步告知，仍需对最终 frozen artifact 做独立核验。
- requirements.lock.txt、requirements.lock.sha256.txt、third_party/wheel_inventory.json、build/wheelhouse/：实际锁定版本、wheel 哈希及已取得的 Windows x64/Python 3.14 wheel。
- third_party/licenses/Python-LICENSE.txt：开发机 Python runtime 附带许可证文本；最终选择 runtime 与发行义务仍需 review。
- docs/baseline/license_inventory.csv：组件与发行审查状态。

许可证字段中的 SPDX 或上游声明由 inventory 记录，不等于已确认所有传递依赖、字体、Qt 插件与平台 DLL 的义务。distribution review 保持 pending。

## 可复现文档构建

~~~powershell
& '.\.venv\Scripts\python.exe' -X utf8 scripts/build_help.py
~~~

脚本更新真实 GUI 截图、帮助 PDF 和手册静态字体子集。字体源取自官方 Google Fonts 仓库；网络不可用时可保留现有 PDF，但重新构建字体需要按 resources/help/fonts/README.md 取得所记来源文件。

## 最终工程候选更新

源码最终全套92项通过，见artifacts/verification/tests-release.xml；此前89/91项保持各自历史。27个模板动作逐项触发并核验真实内容、输出文件和配置/预览。最终双EXE的14项进程验证均符合预期（包含rc2/3/4/5的拒绝/部分状态），报告绑定双EXE SHA，见frozen_application.json与便携目录release_manifest.json。开发机受限PATH验证范围清晰，cleanWindows/真实设备科学资格/正式对外发行法律结论未通过。详细交付状态见交付状态.md。

## v0.2 帮助文档补充快照

以下记录是 2026-10-26 生成的 v0.2 文档补充快照；上文 v0.1 PDF 的 SHA256、页数和截图位置保留为历史记录，不代表本次文件。

- 当前帮助 PDF：`src/eda_toolkit/resources/help/EDA使用指南.pdf`，14 页 A4，SHA256 `8405dcf45396a0fc9146a8f9ec845c98462f43dd05c148a8bca25be469145b22`。实际字节数和逐页图像哈希见 `artifacts/verification/v0.2/help-pdf-final.json`。
- 使用项目虚拟环境运行 `scripts/build_help.py` 生成本次 PDF 和 Qt 截图；新截图只写入 `artifacts/verification/v0.2/help-screenshots/`，没有覆盖旧 `docs/screenshots/`。
- 使用 bundled Poppler `pdfinfo` 确认 14 页，并以 110 dpi 将 14 页全部渲染到 `artifacts/verification/v0.2/help-pdf-qa-final/`。逐页联系表已检查全部页面，另以原分辨率检查封面、SCR/特征方法页、同步与科学限制页、导出完成截图页及帮助入口页；未见裁切、重叠或缺字。渲染器对 Symbol 和 ArialUnicode 输出 display-font 提示；它们没有造成可见缺字，中文字体仍由 PDF 内嵌字体渲染。
- 当前说明保留工程开发与合成数据边界：开发机验证不等于真实设备科学资格、clean Windows 或正式分发许可审查。
- 当前源代码全套测试证据为 `artifacts/verification/v0.2/pytest-results-final.xml`（166 passed，0 failed/error/skipped；40 Qt deprecation warnings）。该测试结论不替代 PDF 的逐页渲染检查或科学资格。
