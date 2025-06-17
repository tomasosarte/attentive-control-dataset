"""Transformation registry.

Any module defining a new ``Transformation`` should use the
``register_transformation`` decorator to register itself. All modules in this
package are automatically imported so that their classes become available in the
``TRANSFORMATIONS`` dictionary.
"""

from __future__ import annotations

import pkgutil
from importlib import import_module
from typing import Callable, Dict, Type

from .transformation import Transformation


# Dictionary mapping transformation names to classes
TRANSFORMATIONS: Dict[str, Type[Transformation]] = {}


def register_transformation(name: str) -> Callable[[Type[Transformation]], Type[Transformation]]:
    """Decorator to register a ``Transformation`` subclass."""

    def decorator(cls: Type[Transformation]) -> Type[Transformation]:
        if not issubclass(cls, Transformation):
            raise TypeError("Registered class must subclass Transformation")
        TRANSFORMATIONS[name] = cls
        return cls

    return decorator


def _import_submodules() -> None:
    """Import all modules in this package so that registrations execute."""
    package_name = __name__
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name in {"transformation", "__init__"}:
            continue
        import_module(f"{package_name}.{module_info.name}")


# Perform automatic discovery when the package is imported
_import_submodules()
