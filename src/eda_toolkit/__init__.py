"""EDA Toolkit package; numerical core imports must not pull in Qt."""

from .dependency_paths import activate_application_dependency_paths

activate_application_dependency_paths()
del activate_application_dependency_paths

__version__ = "0.3.1"

__all__ = ["__version__"]
