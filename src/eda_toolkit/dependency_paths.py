"""Small standard-library-only helpers for app-private Python dependencies."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


def app_data_root() -> Path:
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "EDA Toolkit"
        return Path.home() / "AppData" / "Local" / "EDA Toolkit"
    base = os.environ.get("XDG_DATA_HOME")
    return (Path(base) if base else Path.home() / ".local" / "share") / "eda-toolkit"


def gpu_dependency_root() -> Path:
    return app_data_root() / "runtime" / "gpu"


def base_dependency_root() -> Path:
    return app_data_root() / "runtime" / "packages" / f"py{sys.version_info.major}{sys.version_info.minor}"


def activate_application_dependency_paths() -> None:
    """Append isolated dependency folders without shadowing bundled packages."""
    root = base_dependency_root()
    candidates = [root]
    if root.is_dir():
        candidates.extend(sorted((path for path in root.iterdir()
                                  if path.is_dir() and not path.name.startswith(".")),
                                 key=lambda path: path.name))
    for candidate in candidates:
        if candidate.is_dir():
            resolved = str(candidate.resolve())
            current = {str(Path(item).resolve()) for item in sys.path if item}
            if resolved not in current:
                sys.path.append(resolved)
    importlib.invalidate_caches()


__all__ = ["app_data_root", "base_dependency_root", "gpu_dependency_root",
           "activate_application_dependency_paths"]
