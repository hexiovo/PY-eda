"""模块与冻结入口共用CLI。"""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
