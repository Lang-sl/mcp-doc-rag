#!/usr/bin/env python3
"""Quickstart example — search your SDK docs from Python.

Prerequisites:
    pip install -e .
    python setup_config.py          # one-time setup
    ollama pull nomic-embed-text     # one-time model download
    python -m rag reindex            # index your docs (first time only)

Then run this script:
    python examples/quickstart.py
"""

from rag.config import load_config
from rag.retriever.hybrid import HybridRetriever
from rag.symbol_index import SymbolIndex


def main():
    config = load_config()

    # 1. Hybrid search
    retriever = HybridRetriever(config)
    results = retriever.search("How to initialize the rendering kernel", top_k=5)

    print("=== Search Results ===")
    for i, r in enumerate(results, 1):
        symbol = r.chunk.symbol_id or "(narrative)"
        src = f"{r.chunk.source_label}.{r.chunk.source_module}"
        print(f"{i}. [{r.chunk.type}] {symbol}")
        print(f"   source: {src} | score: {r.score:.4f}")
        if r.chunk.remarks:
            print(f"   {r.chunk.remarks[:120]}...")
        print()

    # 2. Exact symbol lookup (if you indexed API docs)
    idx = SymbolIndex(config.symbol_index_path)
    entries = list(idx._index.items())
    if entries:
        sample_symbol = entries[0][0]
        result = idx.lookup(sample_symbol)
        print(f"=== Symbol Lookup: {sample_symbol} ===")
        if result:
            for key, val in result.items():
                print(f"  {key}: {val}")
        else:
            print("  Not found")


if __name__ == "__main__":
    main()
