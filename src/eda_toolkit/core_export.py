"""结果写入边界：原始文件只读，完整运行清单最后提交。"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import pandas as pd


def json_safe(value):
    """将不可估计数字写为JSON null；科学状态/原因由各结果表保留。"""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    """同一目录内临时写入后原子替换；调用方拥有运行目录的唯一写权限。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)


def spreadsheet_safe_table(table):
    """仅转义可能被表格软件当作公式的文本，不改变真实数值列。"""
    result = table.copy(deep=True)
    for column in result.columns:
        if pd.api.types.is_object_dtype(result[column].dtype) or pd.api.types.is_string_dtype(result[column].dtype):
            result[column] = result[column].map(
                lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    return result


def atomic_csv(path, table, *, spreadsheet_safe=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        output = spreadsheet_safe_table(table) if spreadsheet_safe else table
        output.to_csv(temporary, index=False, encoding="utf-8-sig", na_rep="", float_format="%.17g")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(path)


def safe_name(value):
    text = str(value)
    cleaned = re.sub(r"[^\w\-.]+", "_", text, flags=re.UNICODE).strip(". ")[:70] or "record"
    return cleaned + "_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def verify_manifest(directory, manifest=None):
    """只接受所有已提交文件校验完整的清单；临时文件不会成为缓存。"""
    directory = Path(directory).resolve()
    if manifest is None:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for artifact in manifest.get("artifacts", []):
        candidate = (directory / artifact["path"]).resolve()
        if not candidate.is_relative_to(directory) or not candidate.is_file():
            failures.append(artifact["path"])
        elif sha256_file(candidate) != artifact["sha256"]:
            failures.append(artifact["path"])
    return {"valid": not failures and bool(manifest.get("artifacts")), "failures": failures}


def export_run(results, provenance, destination, config=None, *, progress_callback=None):
    """导出真实结果表及环境/参数/来源；返回带逐文件SHA256的完整清单。

    results={'summary':dict,'tables':{name:DataFrame},'records':list}；导出不执行算法。
    JSON中的数值缺失为null。运行控制器保留生命周期状态，不因导出成功而改写为成功。
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    config = config or provenance.get("resolved_config", {})
    provenance = dict(provenance)
    # 结果必须携带本次实际适用指标的冻结定义。动态 band/lag ID 不能在
    # 日后用用户当前草稿参数重新解释。
    from .core_features import list_feature_definitions
    registry = list_feature_definitions(config)
    selected = set(config.get("features", {}).get("ids", []))
    effective_ids = selected or set(registry)
    feature_definitions = {feature_id: registry[feature_id]
                           for feature_id in sorted(effective_ids) if feature_id in registry}
    provenance["feature_definition_scope"] = "selected_or_all_applicable"
    provenance["feature_definitions"] = feature_definitions
    options = config.get("output", {})
    formats = options.get("formats", ["csv", "json", "html", "png", "xlsx"])
    protect = options.get("spreadsheet_safe", True)
    tables = {str(key): value for key, value in results.get("tables", {}).items() if isinstance(value, pd.DataFrame)}

    def notify_format(name, kind, message, *, duration_s=None):
        if progress_callback is not None:
            progress_callback({"format": name, "kind": kind, "work_id": f"export:{name}",
                               "unit_type": "export_file", "method": name,
                               "duration_s": duration_s, "stage": "export", "message": message})

    atomic_json(root / "resolved_config.json", config)
    atomic_json(root / "provenance.json", provenance)
    atomic_json(root / "feature_definitions.json", feature_definitions)
    atomic_json(root / "summary.json", results.get("summary", {}))
    atomic_json(root / "environment.json", provenance.get("environment", {}))
    table_paths = []
    if "csv" in formats:
        format_started = time.monotonic()
        notify_format("csv", "work_started", "正在写入CSV表格")
        try:
            for key, table in tables.items():
                path = root / "tables" / f"{safe_name(key)}.csv"
                atomic_csv(path, table, spreadsheet_safe=protect)
                table_paths.append((key, path.relative_to(root).as_posix(), len(table)))
        except Exception:
            notify_format("csv", "failed", "CSV表格写入失败", duration_s=time.monotonic() - format_started)
            raise
        notify_format("csv", "completed", "CSV表格已写入", duration_s=time.monotonic() - format_started)
    if "json" in formats:
        format_started = time.monotonic()
        notify_format("json", "work_started", "正在写入JSON表格")
        try:
            for key, table in tables.items():
                atomic_json(root / "tables" / f"{safe_name(key)}.json", table)
        except Exception:
            notify_format("json", "failed", "JSON表格写入失败", duration_s=time.monotonic() - format_started)
            raise
        notify_format("json", "completed", "JSON表格与运行元数据已写入", duration_s=time.monotonic() - format_started)
    if "xlsx" in formats and tables:
        format_started = time.monotonic()
        notify_format("xlsx", "work_started", "正在写入XLSX工作簿")
        target = root / "results.xlsx"
        temporary = root / ".writing-results.xlsx"
        try:
            with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
                for index, (key, table) in enumerate(tables.items()):
                    # Excel工作表名限制31字符，固定索引防止截断重名。
                    sheet = f"{index:02d}_" + re.sub(r"[\\/*?:\[\]]", "_", key)[:27]
                    (spreadsheet_safe_table(table) if protect else table).to_excel(writer, sheet_name=sheet, index=False)
                    worksheet = writer.sheets[sheet]
                    worksheet.freeze_panes = "A2"
                    worksheet.auto_filter.ref = worksheet.dimensions
            os.replace(temporary, target)
        except Exception:
            notify_format("xlsx", "failed", "XLSX工作簿写入失败", duration_s=time.monotonic() - format_started)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        notify_format("xlsx", "completed", "XLSX工作簿已写入", duration_s=time.monotonic() - format_started)
    if "html" in formats:
        format_started = time.monotonic()
        notify_format("html", "work_started", "正在生成HTML报告")
        summary = results.get("summary", {})
        lines = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>EDA 分析报告</title>",
                 "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#162c3c}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd6dc}.note{background:#fff3ce;padding:15px}img{max-width:100%}</style>",
                 "<h1>EDA 分析报告</h1>",
                 f"<p>运行：{html.escape(str(summary.get('run_id','')))} · 状态：{html.escape(str(summary.get('status','')))}</p>",
                 "<p class='note'>研究软件：默认参数为可修改的研究示例。参考/合成数据检查不能替代真实设备和研究任务的科学有效性验证。</p>",
                 "<h2>结果表</h2><ul>"]
        for key, relative, count in table_paths:
            lines.append(f"<li><a href='{html.escape(relative, quote=True)}'>{html.escape(key)}</a>（{count} 行）</li>")
        lines.append("</ul><h2>结果预览</h2>")
        for key, table in tables.items():
            lines.extend([f"<h3>{html.escape(key)}</h3>", table.head(20).to_html(index=False, escape=True)])
        for path in sorted((root / "figures").glob("*.png")) if (root / "figures").exists() else []:
            lines.append(f"<figure><img src='{html.escape(path.relative_to(root).as_posix(), quote=True)}'><figcaption>{html.escape(path.stem)}</figcaption></figure>")
        lines.append("<p>完整方法、配置与证据见 resolved_config.json、provenance.json 及 manifest.json。</p></html>")
        (root / "report.html").write_text("\n".join(lines), encoding="utf-8")
        notify_format("html", "completed", "HTML报告已写入", duration_s=time.monotonic() - format_started)
    if "pdf" in formats:
        format_started = time.monotonic()
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        font_path = Path(__file__).parent / "resources" / "help" / "fonts" / "NotoSansSC-Regular.ttf"
        if not font_path.is_file():
            raise FileNotFoundError("中文报告字体缺失，请重新安装完整发行包。")
        if "EDAChinese" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("EDAChinese", str(font_path)))
        styles = getSampleStyleSheet()
        for style in styles.byName.values():
            style.fontName = "EDAChinese"
            style.wordWrap = "CJK"
        cell_style = ParagraphStyle("EDACell", fontName="EDAChinese", fontSize=6.5, leading=9, wordWrap="CJK")
        story = [Paragraph("EDA 皮电分析运行报告", styles["Title"]), Spacer(1, 12),
                 Paragraph(html.escape(str(results.get("summary", {}).get("run_id", ""))), styles["Normal"])]
        story.append(Paragraph("以下为结果预览，每表最多12行、6列；完整数值、有效状态与原因见同目录CSV/JSON表。研究参数与合成验证不能替代真实设备/任务的科学验证。", styles["Normal"]))
        for key, table in tables.items():
            story.extend([Spacer(1, 12), Paragraph(html.escape(key) + f" · 总行数 {len(table)}", styles["Heading2"])])
            shown = table.iloc[:12, :6].fillna("").astype(str)
            if len(shown.columns):
                cells = [[Paragraph(html.escape(str(value)), cell_style) for value in row]
                         for row in [list(shown.columns)] + shown.values.tolist()]
                grid = Table(cells, repeatRows=1, colWidths=[450 / len(shown.columns)] * len(shown.columns))
                grid.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf4f5")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
                story.append(grid)
        SimpleDocTemplate(str(root / "report.pdf")).build(story)
        notify_format("pdf", "completed", "PDF报告已写入", duration_s=time.monotonic() - format_started)
    manifest_started = time.monotonic()
    if progress_callback is not None:
        progress_callback({"format": "manifest", "kind": "work_started", "work_id": "run:manifest_finalization",
                           "unit_type": "manifest", "method": "manifest_sha256",
                           "stage": "finalize", "message": "正在绑定运行产物完整性清单"})
    artifacts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith(".") and path.name not in {"manifest.json", "pipeline_state.json"}:
            artifacts.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"schema_version": "1.0", "run_id": results.get("summary", {}).get("run_id"),
                "status": results.get("summary", {}).get("status"), "artifacts": artifacts,
                "config_sha256": provenance.get("config_sha256"), "input_hashes": provenance.get("input_hashes", {})}
    atomic_json(root / "manifest.json", manifest)
    return manifest


def refresh_manifest(destination, *, run_id=None, status=None, config_sha256=None, input_hashes=None):
    """在追加运行日志等附属产物后重新绑定全部实物哈希并原子提交清单。"""
    root = Path(destination).resolve()
    previous = {}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    artifacts = []
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.name.startswith(".") or
                path.name in {"manifest.json", "pipeline_state.json"}):
            continue
        artifacts.append({"path": path.relative_to(root).as_posix(),
                          "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": previous.get("schema_version", "1.0"),
        "run_id": run_id if run_id is not None else previous.get("run_id"),
        "status": status if status is not None else previous.get("status"),
        "artifacts": artifacts,
        "config_sha256": config_sha256 if config_sha256 is not None else previous.get("config_sha256"),
        "input_hashes": input_hashes if input_hashes is not None else previous.get("input_hashes", {}),
    }
    atomic_json(manifest_path, manifest)
    return manifest
