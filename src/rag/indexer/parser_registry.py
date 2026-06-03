"""Parser registry — decorator-based parser registration and file type lookup.

Each parser module registers itself by decorating its entry-point function with
:func:`register_parser`.  The registry collects them in a module-level dict,
making the file-type → parser mapping self-describing and extensible.

To add a new parser:

1. Create ``parser_<type>.py`` in this directory.
2. Define ``parse_<type>(file_path, source_label, source_module) -> list[dict]``.
3. Decorate it with ``@register_parser(file_type="<type>", extensions=[...])``.
4. Import the module in ``indexer/__init__.py`` to trigger registration.

No changes are required in crawler.py or orchestrator.py.
"""

from __future__ import annotations

import os
from typing import Callable

# parser function signature
ParserFunc = Callable[
    [str, str, str], list[dict]
]  # (file_path, source_label, source_module) -> parsed elements

# ---- private state ----------------------------------------------------------

_registry: dict[str, ParserFunc] = {}
_ext_to_type: dict[str, str] = {}


# ---- public API -------------------------------------------------------------


def register_parser(file_type: str, extensions: list[str]):
    """Decorator that registers a function as the parser for *file_type*.

    *extensions* is a list of lowercase file extensions (with leading dot)
    that map to this parser, e.g. ``[".html", ".htm"]``.

    Usage::

        @register_parser(file_type="html", extensions=[".html", ".htm"])
        def parse_html(file_path, source_label, source_module):
            ...
    """

    def _decorator(fn: ParserFunc) -> ParserFunc:
        _registry[file_type] = fn
        for ext in extensions:
            _ext_to_type[ext.lower()] = file_type
        return fn

    return _decorator


def classify_file(path: str) -> str:
    """Classify a file by extension using the registered extension map.

    Returns the *file_type* string registered for the file's extension,
    or ``"unknown"`` when no parser claims it.
    """
    ext = os.path.splitext(path)[1].lower()
    return _ext_to_type.get(ext, "unknown")


def get_parser(file_type: str) -> ParserFunc | None:
    """Return the parser callable for *file_type*, or ``None``."""
    return _registry.get(file_type)


def list_types() -> list[str]:
    """Return all registered file types (useful for diagnostics)."""
    return sorted(_registry.keys())
