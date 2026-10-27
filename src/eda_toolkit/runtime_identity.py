"""Windows应用身份与同源发行图标；不导入Qt。"""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys


APP_USER_MODEL_ID = "EDA.AnalysisWorkbench.0_2"


def icon_path():
    """返回打包/源码运行均可解析的ICO绝对路径。"""
    candidates = []
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        frozen = Path(frozen_root)
        candidates.extend((frozen / "eda_toolkit" / "resources" / "icons" / "app_icon.ico",
                           frozen / "assets" / "app_icon.ico"))
    package_icon = Path(__file__).resolve().parent / "resources" / "icons" / "app_icon.ico"
    candidates.append(package_icon)
    source_root = Path(__file__).resolve().parents[2]
    candidates.append(source_root / "assets" / "app_icon.ico")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError("EDA Workbench app_icon.ico未找到；请检查resources/icons发行数据。")


def configure_app_identity():
    """在创建QApplication前设置固定Windows AppUserModelID。

    非Windows开发环境只报告不适用。该API不判断也不声称任务栏图标已被视觉检查。
    """
    result = {"app_user_model_id": APP_USER_MODEL_ID, "platform": sys.platform,
              "configured": False, "method": None, "error": None}
    if sys.platform != "win32":
        result["method"] = "not_applicable"
        return result
    try:
        setter = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        setter.argtypes = [ctypes.c_wchar_p]
        setter.restype = ctypes.c_long
        code = int(setter(APP_USER_MODEL_ID))
        if code != 0:
            result["error"] = f"HRESULT 0x{code & 0xffffffff:08x}"
            result["method"] = "windows_shell32"
            return result
        result["configured"] = True
        result["method"] = "windows_shell32"
    except (AttributeError, OSError, TypeError) as error:
        result["method"] = "windows_shell32"
        result["error"] = str(error)
    return result


__all__ = ["APP_USER_MODEL_ID", "configure_app_identity", "icon_path"]
