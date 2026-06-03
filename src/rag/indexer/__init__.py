"""Indexer pipeline: crawl -> parse -> chunk -> embed -> store.

Importing this package triggers parser registration via the
``@register_parser`` decorator on each parser module.
"""

# Trigger parser registration (side-effect: populates parser_registry._registry)
from rag.indexer import parser_header  # noqa: F401
from rag.indexer import parser_html    # noqa: F401
from rag.indexer import parser_pdf     # noqa: F401
