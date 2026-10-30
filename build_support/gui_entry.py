"""窗口化冻结入口。"""
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    from eda_toolkit.ui.bootstrap import main
    raise SystemExit(main())
