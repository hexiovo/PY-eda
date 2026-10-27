"""冻结入口；标准库freeze_support防止未来受支持进程入口递归启动。"""
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()
    from eda_toolkit.cli import main
    raise SystemExit(main())
