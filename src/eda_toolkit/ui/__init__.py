"""EDA Toolkit 桌面研究工作台；导入 app 子模块可避免 CLI 后端加载 Qt。"""

__all__ = ["create_window", "main"]


def __getattr__(name):
    if name in __all__:
        from . import app
        return getattr(app, name)
    raise AttributeError(name)
