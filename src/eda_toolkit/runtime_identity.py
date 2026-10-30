"""Windows应用身份与同源发行图标；不导入Qt。"""
from __future__ import annotations

import ctypes
from pathlib import Path
import sys


APP_USER_MODEL_ID = "EDA.AnalysisWorkbench.UIRefresh2"
_WINDOW_ICON_HANDLES: dict[int, tuple[int, ...]] = {}


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
    """在 QApplication 创建后、任何应用窗口显示前设置固定 Windows AppUserModelID。

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


def configure_window_icons(window, icon_file: str | None = None):
    """Set and read back the native HWND icons from the bundled ICO once.

    Qt's QIcon is still configured by the UI. This Win32 step ensures Windows
    receives actual HICON handles for the small, small2, and large HWND slots.
    """
    result = {"platform": sys.platform, "configured": False, "verified": False,
              "hwnd": None, "icon_path": None, "wm_geticon": {}, "error": None}
    if sys.platform != "win32":
        result["method"] = "not_applicable"
        return result
    try:
        icon_file = str(Path(icon_file or icon_path()).resolve())
        if not Path(icon_file).is_file():
            raise FileNotFoundError(icon_file)
        hwnd = int(window.winId())
        if not hwnd:
            raise RuntimeError("Qt returned a null native HWND")
        user32 = ctypes.windll.user32
        load_image = user32.LoadImageW
        load_image.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
                               ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        load_image.restype = ctypes.c_void_p
        get_metrics = user32.GetSystemMetrics
        get_metrics.argtypes = [ctypes.c_int]
        get_metrics.restype = ctypes.c_int
        small_size = max(16, int(get_metrics(49) or 16))  # SM_CXSMICON
        big_size = max(32, int(get_metrics(11) or 32))    # SM_CXICON
        load_from_file = 0x0010  # LR_LOADFROMFILE
        small_icon = int(load_image(None, icon_file, 1, small_size, small_size, load_from_file) or 0)
        big_icon = int(load_image(None, icon_file, 1, big_size, big_size, load_from_file) or 0)
        if not small_icon or not big_icon:
            raise OSError("LoadImageW could not load app_icon.ico as HICON")

        send_message = user32.SendMessageW
        send_message.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
        send_message.restype = ctypes.c_ssize_t
        hwnd_arg = ctypes.c_void_p(hwnd)
        wm_seticon, wm_geticon = 0x0080, 0x007F
        icon_small, icon_big, icon_small2 = 0, 1, 2
        send_message(hwnd_arg, wm_seticon, icon_small, small_icon)
        send_message(hwnd_arg, wm_seticon, icon_small2, small_icon)
        send_message(hwnd_arg, wm_seticon, icon_big, big_icon)
        actual = {
            "small": int(send_message(hwnd_arg, wm_geticon, icon_small, 0) or 0),
            "small2": int(send_message(hwnd_arg, wm_geticon, icon_small2, 0) or 0),
            "big": int(send_message(hwnd_arg, wm_geticon, icon_big, 0) or 0),
        }
        expected = {"small": small_icon, "small2": small_icon, "big": big_icon}
        verified = actual == expected
        result.update({"method": "WM_SETICON/WM_GETICON", "configured": True,
                       "verified": verified, "hwnd": hwnd, "icon_path": icon_file,
                       "hicons": expected, "wm_geticon": actual})
        # Keep the loaded handles alive for this window's lifetime.
        _WINDOW_ICON_HANDLES[hwnd] = (small_icon, big_icon)
        return result
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        result["method"] = "WM_SETICON/WM_GETICON"
        result["error"] = str(error)
        return result


__all__ = ["APP_USER_MODEL_ID", "configure_app_identity", "configure_window_icons", "icon_path"]
