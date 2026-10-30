"""采集真实 Qt 页面截图并生成嵌入中文字体的使用指南 PDF。"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from datetime import date
from urllib.request import urlopen
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "artifacts" / "verification" / "v0.3.1" / "help-screenshots"
HELP_DIR = ROOT / "src" / "eda_toolkit" / "resources" / "help"
FONT_DIR = HELP_DIR / "fonts"
PDF_PATH = HELP_DIR / "EDA使用指南.pdf"
VARIABLE_URL = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/"
    "NotoSansSC%5Bwght%5D.ttf"
)
LICENSE_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanssc/OFL.txt"
SOURCE_CACHE = Path(tempfile.gettempdir()) / "eda-toolkit-NotoSansSC-wght.ttf"
VARIABLE_SHA256 = "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da"
PDF_DASH_TRANSLATION = str.maketrans("‐‑‒–—―−", "-------")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_text(value: str) -> str:
    """Use plain ASCII hyphens in rendered PDF text for reliable glyph output."""
    return value.translate(PDF_DASH_TRANSLATION)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)


def _instantiate_font(source: Path, weight: int):
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    variable = TTFont(source)
    axes = variable["fvar"].axes
    if not any(axis.axisTag == "wght" for axis in axes):
        raise RuntimeError("Noto Sans SC variable font has no wght axis.")
    return instantiateVariableFont(
        variable, {"wght": weight}, inplace=False, updateFontNames=True
    )


def _cmap(font) -> set[int]:
    return set().union(*(set(table.cmap) for table in font["cmap"].tables))


def _subset_font(source_font, output: Path, required: set[int]) -> None:
    from fontTools import subset

    missing = required - _cmap(source_font)
    if missing:
        preview = ", ".join(f"U+{code:04X}" for code in sorted(missing)[:20])
        raise RuntimeError(f"Required documentation glyphs are missing: {preview}")
    options = subset.Options()
    options.name_IDs = [0, 1, 2, 3, 4, 5, 6, 13, 14, 16, 17]
    options.name_legacy = True
    options.name_languages = [0x409, 0x804]
    options.notdef_glyph = True
    options.notdef_outline = True
    options.recommended_glyphs = True
    options.layout_features = ["*"]
    worker = subset.Subsetter(options=options)
    worker.populate(unicodes=sorted(required))
    worker.subset(source_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_font.save(output)


def _documentation_text() -> str:
    parts: list[str] = []
    for path in sorted((ROOT / "docs").glob("*.md")):
        parts.append(path.read_text(encoding="utf-8-sig"))
    parts.extend(
        [
            "EDA Toolkit 皮肤电研究工作台 使用指南",
            "快速启动 数据导入 质控 预处理 分解 SCR 窗口 特征 同步 统计 结果 导出 帮助",
            "SYNTHETIC_ONLY pending_representative_real_data experimental_uncalibrated unavailable",
            "src/eda_toolkit/resources/help/EDA使用指南.pdf",
            "analysis_context_s analysis_intervals_s onset_left_context_s onset_right_context_s",
            "公式 L+(1-f)A；参考时钟 t_ref=a+b*t_local；fs_ref=fs_local/b。",
        ]
    )
    return "\n".join(parts)


def _required_glyphs() -> set[int]:
    source_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    script_text = "\n".join(
        node.value
        for node in ast.walk(source_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    return {
        ord(character)
        for character in _documentation_text() + "\n" + script_text
        if ord(character) >= 32
    } | set(range(32, 127))


def _ensure_fonts() -> dict[str, Path]:
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    regular_path = FONT_DIR / "NotoSansSC-Regular.ttf"
    manual_regular_path = FONT_DIR / "NotoSansSC-Manual-Regular.ttf"
    manual_bold_path = FONT_DIR / "NotoSansSC-Manual-Bold.ttf"
    license_path = FONT_DIR / "OFL.txt"
    variable_hash = VARIABLE_SHA256
    if SOURCE_CACHE.is_file():
        variable_hash = _sha256(SOURCE_CACHE)
        if variable_hash != VARIABLE_SHA256:
            raise RuntimeError("Cached Noto Sans SC variable font does not match the recorded SHA256.")

    if not license_path.is_file():
        _download(LICENSE_URL, license_path)
    if not regular_path.is_file() or not manual_regular_path.is_file() or not manual_bold_path.is_file():
        if not SOURCE_CACHE.is_file() or SOURCE_CACHE.stat().st_size < 1_000_000:
            _download(VARIABLE_URL, SOURCE_CACHE)
        variable_hash = _sha256(SOURCE_CACHE)
        if variable_hash != VARIABLE_SHA256:
            raise RuntimeError("Downloaded Noto Sans SC variable font does not match the recorded SHA256.")
        if not regular_path.is_file():
            static_regular = _instantiate_font(SOURCE_CACHE, 400)
            static_regular.save(regular_path)
        if not manual_regular_path.is_file():
            full_regular = _instantiate_font(SOURCE_CACHE, 400)
            required = _required_glyphs()
            _subset_font(full_regular, manual_regular_path, required)
        if not manual_bold_path.is_file():
            static_bold = _instantiate_font(SOURCE_CACHE, 700)
            required = _required_glyphs()
            _subset_font(static_bold, manual_bold_path, required)

    required = _required_glyphs()
    from fontTools.ttLib import TTFont

    for font_path in (manual_regular_path, manual_bold_path):
        font = TTFont(font_path)
        missing = required - _cmap(font)
        if missing:
            font.close()
            if SOURCE_CACHE.is_file():
                if font_path == manual_regular_path:
                    full_regular = TTFont(regular_path)
                    _subset_font(full_regular, manual_regular_path, required)
                    full_regular.close()
                else:
                    static_bold = _instantiate_font(SOURCE_CACHE, 700)
                    _subset_font(static_bold, manual_bold_path, required)
                    static_bold.close()
            else:
                _download(VARIABLE_URL, SOURCE_CACHE)
                variable_hash = _sha256(SOURCE_CACHE)
                if font_path == manual_regular_path:
                    full_regular = TTFont(regular_path)
                    _subset_font(full_regular, manual_regular_path, required)
                    full_regular.close()
                else:
                    static_bold = _instantiate_font(SOURCE_CACHE, 700)
                    _subset_font(static_bold, manual_bold_path, required)
                    static_bold.close()
        else:
            font.close()

    readme = [
        "# Noto Sans SC 文档字体",
        "",
        "来源：Google Fonts 官方仓库 Noto Sans SC；许可证原文为同目录 OFL.txt。",
        f"变量字体来源：{VARIABLE_URL}",
        f"许可证来源：{LICENSE_URL}",
        "固定实例：wght=400 用于完整静态 NotoSansSC-Regular.ttf；wght=400/700 子集用于本手册。",
        "手册子集包括 docs/*.md 与构建脚本静态文本所需字符及 ASCII；ReportLab PDF 内嵌字体子集。",
        "core_export 的 report.pdf 应注册完整 NotoSansSC-Regular.ttf，避免未来未知参与者/条件文本缺字。",
        f"生成日期：{date.today().isoformat()}",
        f"变量源 SHA256：{variable_hash or '本次复用已生成静态字体；来源 hash 见首次构建记录。'}",
        f"完整静态 Regular SHA256：{_sha256(regular_path)}",
        f"手册 Regular 子集 SHA256：{_sha256(manual_regular_path)}",
        f"手册 Bold 子集 SHA256：{_sha256(manual_bold_path)}",
        f"OFL SHA256：{_sha256(license_path)}",
    ]
    (FONT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return {
        "regular": regular_path,
        "manual_regular": manual_regular_path,
        "manual_bold": manual_bold_path,
        "license": license_path,
    }


def _save_grab(widget, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = widget.grab()
    if pixmap.isNull() or not pixmap.save(str(path), "PNG"):
        raise RuntimeError(f"Could not capture a nonempty Qt widget: {path.name}")
    print(f"Qt screenshot: {path.name}", flush=True)


def _wait(app, predicate, timeout_s: float, description: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return
        time.sleep(0.02)
    raise TimeoutError(f"Timed out waiting for {description}.")


def _capture_gui() -> list[tuple[str, str]]:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.setdefault("QT_SCALE_FACTOR", "2")
    sys.path.insert(0, str(ROOT / "src"))
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QLineEdit, QPushButton
    from eda_toolkit.config import resolve_config
    from eda_toolkit.ui.app import create_window

    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    example_dir = ROOT / "examples" / "current_synthetic_demo"
    example_config = json.loads((example_dir / "single.json").read_text(encoding="utf-8"))
    example_config = resolve_config(example_config)
    temporary_root = Path(tempfile.mkdtemp(prefix="eda-help-demo-"))
    expected_temp = Path(tempfile.gettempdir()).resolve()
    temporary_root = temporary_root.resolve()
    if not temporary_root.is_relative_to(expected_temp) or not temporary_root.name.startswith("eda-help-demo-"):
        raise RuntimeError("Temporary GUI run escaped the expected system temp directory.")

    example_config["input"]["base_directory"] = str(example_dir.resolve())
    example_config["output"]["directory"] = str(temporary_root / "analysis")
    example_config["output"]["overwrite"] = False
    example_config.setdefault("workspace", {})["project_directory"] = str(temporary_root / "project")
    config = resolve_config(example_config)
    example_config = config
    window = None
    state = None
    captures = [
        ("01-startup.png", "项目首页与能力状态"),
        ("02-import-mapping.png", "九类模板、真实 CSV 表头和映射预览"),
        ("03-qc-history.png", "载入记录后的 QC 与人工历史控件"),
        ("04-config.png", "项目页完整 JSON 配置编辑器"),
        ("05-preprocess.png", "分段预处理参数"),
        ("06-run.png", "后台批量任务运行状态"),
        ("07-result.png", "真实 pipeline 返回的运行摘要"),
        ("08-export.png", "清单内产物预览与副本导出控件"),
        ("09-help-menu.png", "真实 Qt 帮助菜单与 F1 文档入口"),
    ]
    try:
        window, state = create_window(config)
        app = state["app"]
        # The offscreen Qt platform exposes only a 400×400 logical screen.
        # Use a fixed capture viewport so current screen-fit startup sizing does
        # not crop the documentation screenshots.
        window.resize(1420, 920)
        window.show()
        app.processEvents()
        _save_grab(window, SCREENSHOTS / captures[0][0])

        widgets = state["widgets"]
        navigation = widgets["navigation"]

        # QC capture uses the full original synthetic manifest. Mapping preview
        # intentionally commits its selected single-file mapping to the project
        # configuration, so perform this load before that real UI action.
        navigation.setCurrentRow(2)
        app.processEvents()
        widgets["qc_load"].click()
        if not state.get("records"):
            _wait(app, lambda: bool(state.get("records")), 20, "QC input records")
        qc_page = widgets["pages"]["质控"]["scroll"]
        qc_page.verticalScrollBar().setValue(0)
        app.processEvents()
        _save_grab(window, SCREENSHOTS / captures[2][0])

        navigation.setCurrentRow(1)
        app.processEvents()
        signal_path = example_dir / "synthetic_001.csv"
        mapping_line = next(
            item
            for item in window.findChildren(QLineEdit)
            if item.placeholderText() == "选择一条信号 CSV 做映射预览"
        )
        mapping_line.setText(str(signal_path))
        mapping_line.editingFinished.emit()
        app.processEvents()
        widgets["template_buttons"]["signals"]["view"].setFocus()
        widgets["stack"].setCurrentIndex(1)
        mapping_page = widgets["pages"]["导入与映射"]["scroll"]
        mapping_page.verticalScrollBar().setValue(mapping_page.verticalScrollBar().maximum())
        app.processEvents()
        widgets["stack"].setCurrentIndex(1)
        mapping_button = next(
            item for item in window.findChildren(QPushButton)
            if item.text() == "用 core_io 读取并预览"
        )
        mapping_button.click()
        mapping_job = state.get("mapping_future")
        if mapping_job:
            future = mapping_job[1]
            _wait(app, future.done, 20, "core_io mapping preview")
            time.sleep(0.08)
            app.processEvents()
        _save_grab(window, SCREENSHOTS / captures[1][0])

        navigation.setCurrentRow(0)
        app.processEvents()
        # Restore the original manifest through the actual JSON editor after
        # the mapping trial so the following GUI run still covers all 4 records.
        widgets["config_json"].setPlainText(json.dumps(example_config, ensure_ascii=False, indent=2))
        widgets["apply_json"].click()
        app.processEvents()
        project_page = widgets["pages"]["项目"]["scroll"]
        project_page.verticalScrollBar().setValue(project_page.verticalScrollBar().maximum())
        app.processEvents()
        _save_grab(window, SCREENSHOTS / captures[3][0])

        navigation.setCurrentRow(3)
        app.processEvents()
        preprocess_page = widgets["pages"]["预处理"]["scroll"]
        preprocess_page.verticalScrollBar().setValue(0)
        app.processEvents()
        _save_grab(window, SCREENSHOTS / captures[4][0])

        navigation.setCurrentRow(0)
        app.processEvents()
        widgets["batch"].click()
        _wait(
            app,
            lambda: state.get("preflight_future") is None
            and state.get("preflight_report") is not None
            and not state.get("pending_enqueue")
            and widgets["task_table"].rowCount() > 0,
            30,
            "preflight report and queued task",
        )
        preflight_report = state["preflight_report"]
        if not preflight_report.get("eligible_to_enqueue"):
            raise RuntimeError(
                "Synthetic GUI preflight was not eligible: "
                + json.dumps(preflight_report.get("errors", []), ensure_ascii=False)
            )
        if state.get("run_future") is not None or state.get("busy"):
            raise RuntimeError("A queued GUI task started before the explicit queue action.")
        widgets["start_queue"].click()
        _wait(
            app,
            lambda: bool(state.get("busy")) and state.get("run_future") is not None,
            10,
            "explicit queue task start",
        )
        time.sleep(0.12)
        app.processEvents()
        _save_grab(window, SCREENSHOTS / captures[5][0])
        _wait(
            app,
            lambda: isinstance(state.get("last_summary"), dict) and not state.get("busy"),
            120,
            "pipeline completion and summary",
        )
        summary = state["last_summary"]
        if summary.get("status") != "succeeded":
            raise RuntimeError(f"Synthetic GUI run did not succeed: {summary.get('status')}")

        navigation.setCurrentRow(7)
        app.processEvents()
        result_page = widgets["pages"]["结果"]["scroll"]
        result_page.verticalScrollBar().setValue(0)
        app.processEvents()
        _save_grab(window, SCREENSHOTS / captures[6][0])

        artifact_names = list(state.get("artifact_paths", {}))
        feature_csv = next(
            (name for name in artifact_names if name.startswith("tables/features") and name.endswith(".csv")),
            None,
        )
        if feature_csv:
            combo = widgets["artifact_combo"]
            combo.setCurrentText(feature_csv)
            widgets["preview_artifact"].click()
            app.processEvents()
        result_page.verticalScrollBar().setValue(result_page.verticalScrollBar().maximum())
        app.processEvents()
        _save_grab(window, SCREENSHOTS / captures[7][0])

        help_menu = next(
            action.menu()
            for action in window.menuBar().actions()
            if action.text().replace("&", "") == "帮助"
        )
        if help_menu is None:
            raise RuntimeError("The live main window has no Help menu.")
        help_menu.popup(window.mapToGlobal(QPoint(window.width() - 170, 26)))
        app.processEvents()
        _save_grab(help_menu, SCREENSHOTS / captures[8][0])
        help_menu.close()

        width, height = window.width(), window.height()
        if width < 1000 or height < 700:
            raise RuntimeError(f"Unexpected Qt window size: {width}x{height}")
        return captures
    finally:
        if window is not None:
            window.close()
            if state is not None:
                state["app"].processEvents()
                for name in ("executor", "io_executor"):
                    executor = state.get(name)
                    if executor is not None:
                        executor.shutdown(wait=True, cancel_futures=True)
        resolved = temporary_root.resolve()
        if resolved.is_relative_to(expected_temp) and resolved.name.startswith("eda-help-demo-"):
            shutil.rmtree(resolved, ignore_errors=True)


def _register_fonts(fonts: dict[str, Path]) -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    pdfmetrics.registerFont(TTFont("EDAChinese", str(fonts["manual_regular"])))
    pdfmetrics.registerFont(TTFont("EDAChineseBold", str(fonts["manual_bold"])))
    pdfmetrics.registerFontFamily(
        "EDAChinese",
        normal="EDAChinese",
        bold="EDAChineseBold",
        italic="EDAChinese",
        boldItalic="EDAChineseBold",
    )


def _build_pdf(captures: list[tuple[str, str]], fonts: dict[str, Path]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        NextPageTemplate,
    )

    _register_fonts(fonts)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "EDA_Title",
        parent=styles["Title"],
        fontName="EDAChineseBold",
        fontSize=27,
        leading=35,
        textColor=colors.HexColor("#14384D"),
        alignment=TA_LEFT,
        spaceAfter=9,
        wordWrap="CJK",
    )
    subtitle = ParagraphStyle(
        "EDA_Subtitle",
        parent=styles["Normal"],
        fontName="EDAChinese",
        fontSize=14,
        leading=20,
        textColor=colors.HexColor("#087E83"),
        spaceAfter=12,
        wordWrap="CJK",
    )
    heading = ParagraphStyle(
        "EDA_Heading",
        parent=styles["Heading1"],
        fontName="EDAChineseBold",
        fontSize=16,
        leading=22,
        textColor=colors.HexColor("#14384D"),
        spaceBefore=2,
        spaceAfter=9,
        keepWithNext=True,
        wordWrap="CJK",
    )
    body = ParagraphStyle(
        "EDA_Body",
        parent=styles["BodyText"],
        fontName="EDAChinese",
        fontSize=10.2,
        leading=15.2,
        textColor=colors.HexColor("#293E4D"),
        spaceAfter=7,
        alignment=TA_LEFT,
        wordWrap="CJK",
        splitLongWords=True,
    )
    small = ParagraphStyle(
        "EDA_Small",
        parent=body,
        fontSize=8.6,
        leading=12.2,
        textColor=colors.HexColor("#516879"),
        spaceAfter=5,
    )
    code = ParagraphStyle(
        "EDA_Code",
        parent=body,
        fontName="Courier",
        fontSize=8,
        leading=11,
        leftIndent=8,
        rightIndent=8,
        backColor=colors.HexColor("#EDF4F6"),
        borderColor=colors.HexColor("#D8E6EB"),
        borderWidth=0.5,
        borderPadding=6,
        wordWrap="CJK",
    )
    center = ParagraphStyle(
        "EDA_Center",
        parent=body,
        alignment=TA_CENTER,
        fontSize=9,
    )

    portrait_w, portrait_h = A4
    landscape_size = landscape(A4)
    portrait_frame = Frame(
        17 * mm,
        17 * mm,
        portrait_w - 34 * mm,
        portrait_h - 39 * mm,
        id="portrait_frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    landscape_frame = Frame(
        14 * mm,
        14 * mm,
        landscape_size[0] - 28 * mm,
        landscape_size[1] - 30 * mm,
        id="landscape_frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )

    def page_decor(canvas, document):
        page_w, page_h = canvas._pagesize
        if document.page == 1:
            canvas.setFillColor(colors.HexColor("#F4F8FA"))
            canvas.rect(0, 0, page_w, page_h, fill=1, stroke=0)
        else:
            canvas.setFillColor(colors.HexColor("#14384D"))
            canvas.rect(0, page_h - 8 * mm, page_w, 8 * mm, fill=1, stroke=0)
            canvas.setFillColor(colors.white)
            canvas.setFont("EDAChinese", 8.5)
            canvas.drawString(17 * mm, page_h - 5.5 * mm, "EDA Toolkit 0.3.1  ·  使用指南")
        canvas.setStrokeColor(colors.HexColor("#D3E0E6"))
        canvas.setLineWidth(0.6)
        canvas.line(17 * mm, 12 * mm, page_w - 17 * mm, 12 * mm)
        canvas.setFillColor(colors.HexColor("#5B7180"))
        canvas.setFont("EDAChinese", 7.5)
        canvas.drawString(17 * mm, 7.5 * mm, "v0.3.1  ·  合成数据仅用于软件演示")
        canvas.drawRightString(page_w - 17 * mm, 7.5 * mm, f"{document.page}")

    doc = BaseDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        title="EDA Toolkit 使用指南",
        author="EDA Toolkit",
        subject="中文快速入门、方法说明、限制和真实界面截图",
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=15 * mm,
        bottomMargin=17 * mm,
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="portrait", pagesize=A4, frames=[portrait_frame], onPage=page_decor),
            PageTemplate(
                id="landscape",
                pagesize=landscape_size,
                frames=[landscape_frame],
                onPage=page_decor,
            ),
        ]
    )

    story = [
        Spacer(1, 17 * mm),
        Paragraph("EDA TOOLKIT", ParagraphStyle(
            "EDA_Eyebrow", parent=body, fontName="EDAChineseBold", fontSize=9,
            textColor=colors.HexColor("#087E83"), spaceAfter=12,
        )),
        Paragraph("皮肤电研究工作台", title),
        Paragraph("使用指南", subtitle),
        Spacer(1, 2 * mm),
        Table(
            [[Paragraph(
                f"版本 0.3.1　　文档日期 {date.today().isoformat()}　　Windows 便携版",
                ParagraphStyle("CoverBadge", parent=body, fontName="EDAChineseBold",
                               textColor=colors.white, fontSize=9, leading=13, spaceAfter=0),
            )]],
            colWidths=[160 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#087E83")),
                ("BOX", (0, 0), (-1, -1), 0, colors.HexColor("#087E83")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]),
        ),
        Spacer(1, 7 * mm),
        Paragraph(
            "本指南说明当前软件的实际入口、输入约束、主要方法和结果状态。成功运行只表示当前工程流程完成，不证明设备适用、研究结论成立或参数已经科学校准。",
            body,
        ),
        Paragraph(
            "样例数据由程序确定性生成，明确标记 SYNTHETIC_ONLY。请勿把合成示例当作真实被试或设备记录。",
            body,
        ),
        Spacer(1, 6 * mm),
        Paragraph("开始前", heading),
        Paragraph(
            "Windows 便携包无需另装 Python。使用真实研究数据前，请确认单位、采样时钟、缺失情况和实验单位，并查阅字段字典与验证限制。",
            body,
        ),
        Spacer(1, 3 * mm),
        Paragraph("配套资料：docs/快速入门.md · docs/操作指南.md · docs/方法指南.md · docs/验证与限制.md", small),
        PageBreak(),
    ]

    content_pages = [
        (
            "快速启动",
            [
                "在项目 PowerShell 终端使用本地虚拟环境。查看命令：",
                "python -X utf8 -m eda_toolkit --help",
                "启动图形界面：",
                "python -X utf8 -m eda_toolkit.ui.app",
                "CLI 的 validate 读取并校验输入，plan 预览流程，run 调用共享 pipeline。GUI 首页的“单文件试跑”“批量启动”和“取消运行”也调用同一入口。结果目录保存配置快照、状态、来源和 manifest。",
                "先用 examples/current_synthetic_demo/single.json 运行工程示例。所有示例信号、事件、条件、成员和同步锚点均为 SYNTHETIC_ONLY。新建空目录可用 demo 子命令生成另一份确定性示例。",
            ],
        ),
        (
            "导入数据与时间单位",
            [
                "信号列统一成秒 time_s 与物理微西门子 eda_uS。幅值列名不是 eda_uS 时必须声明 eda_unit。S 乘 10^6、mS 乘 10^3 转换为 uS；正电阻按 10^6/R_ohm 转换。未知单位不猜测。",
                "时间必须有限、严格递增。程序保留原始顺序，对缺失、重复或倒退时间明确报错，不通过排序或删点改变记录。没有时间列时，需声明正采样率和来源 time_unit；生成时间会标记为推导值。",
                "输入文件从同一次读取的原始字节计算哈希并解析。record 保存身份、来源、单位、采样率估计/声明、时间原点、clock_domain 和映射版本。原始 NaN 保持为缺失，额外列进入 extras。",
                "GUI 导入预览是有行数上限的快速预览。接受输入前还应运行 validate，检查表字段、重复被试、事件/条件引用、通道和 clock_domain。见 docs/CSV字段字典.md。",
            ],
        ),
        (
            "质控与人工历史",
            [
                "QC 将缺失、非有限、自动候选、人工排除、人工恢复、插值、边缘影响和不可信时间作为可重叠 facts 保存。candidate 只是规则提示，不等于“已确认伪迹”。flatline 按真实时间长度识别，间隔过大的相邻样本切段。",
                "人工区间使用半开 [start,end)。exclude 优先；restore 只覆盖自动候选，不会将原始 NaN/Inf 或不可信时间重新变为观测。每次操作含 operator、reason、timestamp、revision 与 parent_revision。",
                "Undo 和 redo 通过追加新行动完成，历史不删除。若 parent revision 已变化，追加会拒绝。每次运行冻结当时配置和人工修订，结果与该修订相关联。",
                "汇总报告样本数、原始/可用/插值时长、缺失/排除时长、最长缺口及各原因候选。非均匀采样时，时长比例与行数比例不是同一个量。",
            ],
        ),
        (
            "预处理与分解",
            [
                "短缺口可选择线性或 PCHIP 插值，但需有效左右端点并满足最大跨度；首尾缺口不外推，长缺口切分连续段。插值事实保留，可从 primary feature policy 中排除。",
                "Butterworth SOS 低通在每个连续合格段运行。检查正 fs、cutoff 小于 Nyquist、padding 与段长；前后向运行是离线零相位计算。程序报告估计边缘影响半径并传播 edge_affected。重采样采用分段 polyphase 抗混叠并传播支撑质量。",
                "v0.3.1 提供 bateman_sparse 非负稀疏分解。选择 single_driver 预设启用 9 项 Driver 指标。Driver=脉冲质量/采样间隔，单位 µS/s；积分单位 µS。tonic、phasic 与 residual 重构 clean；优化失败的段保持缺失。该方法是本地研究模型，非 Ledalab/CDA 或 cvxEDA 复现。",
                "Driver 独立拟合连续合格段，排除插值、伪迹与预处理边缘，再从两端裁去完整核支持（默认 4 Hz 下约 16 秒）。短段和覆盖不足的窗口不可用。阈值默认 0.01 µS/s；支持边界删失脉冲不计数。事件积分不减基线，重叠窗口可能共享活动。完整数学定义及参数见 docs/driver-method.md。",
                "filter_separation 用 SciPy Butterworth 二阶 SOS 对 clean_uS 分别作低通与高通，输出 tonic_uS 与 phasic_uS。residual_uS=clean_uS−tonic_uS−phasic_uS，driver=None。独立滤波分量不承诺精确重构。",
                "该方法是本地独立实现，不是 NeuroKit、cvxEDA 或 Ledalab 兼容版。默认截止 0.05 Hz、二阶是研究示例，不是设备推荐参数。",
            ],
        ),
        (
            "SCR 定义与上下文",
            [
                "trough_to_peak 在完整 detector-eligible 连续段搜索 trough、peak 和 recovery。阈值在整条记录候选收集后应用；绝对阈值为 uS，relative reference 在整条记录计算，不随窗口变化。min_interval_s 按起点间隔处理冲突。",
                "恢复阈值为 L+(1−f)A，f 表示幅值返回比例。首次下降穿越用相邻点线性插值；被新响应、坏段、长缺口或记录结尾截断时单独标 recovery censor。",
                "SCR 面积从 onset 到 peak 积分，以 onset level 为零基线，采用有符号梯形法，单位 uS·s。scr_area_sum_uS_s / scr_area_mean_uS_s 汇总合格响应；event_scr_area_sum_uS_s 按唯一 event_scr_links 归属汇总事件响应窗内响应。面积只表示已观察的上升段，不外推到恢复段；恢复删失不影响上升面积。曝光不足、事件支持不足或目录面积缺失时显式返回 unavailable。",
                "onset_left_context_s 默认 0；onset_right_context_s 默认 max_rise_s，显式右值不能更小。起点资格按真实时间和同一个连续段核验。完整信号仍用于 trough/peak/recovery 搜索；边界起点候选保留在 catalogue 并标 unavailable_onset_context。",
                "只有 onset_eligible 的候选进入主计数、rate 和幅值摘要。rate 暴露按合格起点支持的相邻时间区间求和，不跨 gap、不外推末点。无合格曝光时是 unavailable，而不是 count=0。阈值需要按设备与任务校准。",
            ],
        ),
        (
            "窗口与特征状态",
            [
                "窗口类型为 whole、fixed、event 和 condition，均使用半开 [start,end)。事件 baseline、response、peak、recovery 分别配置。SCR 起点在窗口内决定响应归属；重复归属、支持不足、缺少基线或跨坏段会写入 ambiguous/invalid/unavailable 状态和理由。",
                "样本均值、中位数、标准差和四分位距按样本计算，不是时间加权；分位数使用线性 percentile method。原 iqr=Q75-Q25 对一个合格样本定义为 0，最低样本数为 1；其他指标按登记的 min_n 判断。phasic_auc 是裁剪到窗口的有符号梯形积分，单位 uS·s；slope 使用实际秒。质量比例是样本比例。",
                "schema 1.1 完整能力目录含 144 项（含 9 项 Driver）；single 预设当前适用 124 项，group_sync 预设 135 项；single_driver 预设 133 项，Driver 加多人同步 144 项；滤波模式关闭扩展计算时保留原 33 项。features.ids 为空表示当前配置下全选。完整 ID、单位、basis、公式、有效性、参数和输出表见 docs/特征字典-v1.1.csv；原 33 项兼容字典仍见 docs/特征字典.csv。",
                "新增 34 项包括 clean/tonic 最小最大值、P05/P25/P75/P95、偏度、超额峰度、MAD；SCR 和零包含事件幅值的平方根、ln(1+A/1µS)、记录内 Z 值及最大值校正；上升时间与配置/50%/63%恢复时间的中位数和标准差。先逐条变换再汇总，原幅值保留。",
                "幅值参考固定为同一记录、被试、会话、通道的全部合格响应或有效试次，跨条件取参考。Z 值参考标准差使用 ddof=1；最大值校正为 A/max(A)。无响应试次保留零，排除试次缺失，零标准差或零最大值时相应校正不可用。参考统计和变换状态随 trial_table、scr_events_extended 导出。",
                "没有响应但检测支持充分时，计数可为 0；幅值均值/中位数没有响应时不可用。恢复时间被删失时保留删失，不抹除已合格起点幅值和计数。每次运行保存冻结 feature_definitions.json；结果页“查看本次运行定义”按所选历史 run 展示冻结动态 band/lag 参数及 manifest 表头，不用当前草稿重新解释。附属表和 status 见 docs/扩展指标与结果表-v1.1.md。",
            ],
        ),
        (
            "时钟、同步和统计",
            [
                "多人同步以独立 anchors 估计 t_ref=a+b·t_local，要求 b>0。一个锚点只能定偏移；秒单位不代表设备同钟。不能用 EDA 相关性猜时钟映射。关联表中的事件、条件和锚点也要声明 clock_domain。",
                "正 lag 定义为 A 领先 B，即 corr(A(t),B(t+lag))。报告保留 signed r、lag、支持和覆盖。lag SD 与稳定比例以窗口中心划分非重叠 midpoint time cells，以 cell 与该 pair 连续合格支持相交后的实际时长计权；先逐 pair 计算，再对有效 pair 等权汇总。lag SD 为总体加权标准差(ddof=0)，稳定比例使用时长加权中位数。中心为2、4、8秒的窗口对应权重3、3、4秒，lag为0、0、2秒时 SD=sqrt(0.96)=0.9797958971秒，容差1秒的稳定比例为0.6。缺口保留为未知。",
                "Node strength 的 expected_degree 来自冻结成员 roster/预定完整图，不随观察到的边数缩小。三人完全图的 expected_degree 恒为2；BC缺失时B/C强度（含标准化值）为 NA_missing_fixed_incident_edge，A若AB和AC可用仍可计算；没有 roster/预定图时为 NA_expected_graph_unknown。群体固定边汇总仍按缺失固定边规则标记不可用。组均值、配对边和 leave-one-out 不能被误读为独立被试。",
                "surrogate/null 是实验路线。显式分析半开区间优先；否则 context 从共同时间网格两端裁剪。观测和每次 surrogate 重复同一窗口/lag 搜索。位移必须保留所需位置的逐点 mask，并把接缝放在分析支持以外。资源或可达性不足返回 unavailable，不减少请求次数假装完成。",
                "双侧比较 max(abs(r))，greater 比较 max(r)，less 比较 min(r)。尾部概率和 Monte Carlo 不确定性仍标 experimental_uncalibrated，不能称为已校准显著性。Pseudodyad 必须实际重分配满足 exchangeability/task-relative 条件的 donor；元数据不足时不可用，不回退到 circular shift。",
                "统计设计必须声明独立单位、重复测量层级、outcome、contrast 和校正家族。公式/任意代码不执行。不可用假设仍留在预先声明的 correction family。",
            ],
        ),
        (
            "运行证据与限制",
            [
                "软件可导出 CSV、JSON、HTML、PNG、XLSX 等结果，并保存配置、输入摘要和运行 manifest。发布页提供当前发行包和工程验证说明；工程测试不替代真实研究数据验证。",
                "目前缺少有代表性的真实设备/任务数据。候选阈值、filter support、SCR 指标解释、同步交换性与模型假设均未完成真实研究场景验证。",
                "cvxEDA、Ledalab 和 GPU 路径当前不可用，软件不会自动换用其它算法。便携包的工程验证、真实设备/任务科学资格及分发许可审查是不同事项；成功运行不代表后两者已经通过。",
                "软件不推断压力、情绪或疾病，不替代生理信号质量评估和统计设计。报告中的 succeeded 仅代表这次 software pipeline 正常返回。",
            ],
        ),
    ]
    for page_index, (page_title, paragraphs) in enumerate(content_pages):
        story.append(Paragraph(_pdf_text(page_title), heading))
        for paragraph in paragraphs:
            if paragraph.startswith("python "):
                story.append(Paragraph(escape(_pdf_text(paragraph)), code))
            else:
                story.append(Paragraph(escape(_pdf_text(paragraph)), body))
        if page_index in {1, 3, 5}:
            story.append(PageBreak())

    story.extend([NextPageTemplate("landscape"), PageBreak()])
    screenshot_notes = {
        "01-startup.png": "项目页显示输出位置、研究预设、本机能力状态和工程演示入口。合成演示创建在新目录中。",
        "02-import-mapping.png": "真实 CSV 表头经 core_io 预览并映射；模板示例与导入预览均不是完整接受校验。",
        "03-qc-history.png": "真实记录载入后的原始波形、QC 概要和人工标注控件。",
        "04-config.png": "项目页滚动至完整 JSON 配置编辑区；应用前会执行严格配置校验。",
        "05-preprocess.png": "缺口、平滑、低通、时间网格、基线和缩放由完整 preprocess 配置控制。",
        "06-run.png": "真实 GUI 后台 pipeline 任务启动时从主窗口抓取；随后等待该次合成任务完成。",
        "07-result.png": "摘要来自本次 GUI pipeline 返回值，science qualification 仍 pending。",
        "08-export.png": "选择当前运行的特征 CSV 作有界预览，旁边保留实际文件副本导出入口。",
        "09-help-menu.png": "真实 QMenu 中的中文帮助和错误解释动作；F1 打开包内中文 PDF。",
    }
    for index, (filename, caption) in enumerate(captures):
        if not (SCREENSHOTS / filename).is_file():
            raise FileNotFoundError(f"Required screenshot is missing: {filename}")
        story.append(Paragraph(_pdf_text(f"界面截图 {index + 1:02d}  ·  {caption}"), heading))
        story.append(Paragraph(_pdf_text(screenshot_notes[filename]), small))
        image_width = 300 if filename == "09-help-menu.png" else 660
        image = Image(str(SCREENSHOTS / filename), width=image_width, height=430, kind="proportional")
        image.hAlign = "CENTER"
        story.append(Spacer(1, 2 * mm))
        story.append(image)
        if index != len(captures) - 1:
            story.append(PageBreak())

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)


def main() -> int:
    print("Preparing Noto Sans SC fonts...", flush=True)
    fonts = _ensure_fonts()
    print("Capturing live Qt pages and running synthetic GUI pipeline...", flush=True)
    captures = _capture_gui()
    print("Building embedded-font PDF...", flush=True)
    _build_pdf(captures, fonts)
    print(json.dumps(
        {
            "pdf": str(PDF_PATH),
            "pdf_sha256": _sha256(PDF_PATH),
            "font_files": {name: str(path) for name, path in fonts.items()},
            "font_sha256": {name: _sha256(path) for name, path in fonts.items()},
            "screenshots": [
                {
                    "path": str(SCREENSHOTS / name),
                    "label": label,
                    "sha256": _sha256(SCREENSHOTS / name),
                }
                for name, label in captures
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
