"""CLI entry point for the RAG engine."""
from __future__ import annotations

import argparse
import sys

from rag.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Local RAG engine for C++ SDK documentation retrieval.",
    )
    sub = parser.add_subparsers(dest="command")

    # source add
    p = sub.add_parser("source", help="Source management")
    p_src = p.add_subparsers(dest="source_action")
    p_add = p_src.add_parser("add", help="Add a document source")
    p_add.add_argument("label")
    p_add.add_argument("path")
    p_rm = p_src.add_parser("remove", help="Remove a document source")
    p_rm.add_argument("label")
    p_ls = p_src.add_parser("list", help="List document sources")

    # symbol
    p_sym = sub.add_parser("symbol", help="Exact symbol lookup")
    p_sym.add_argument("symbol_name")

    # reindex
    p_rei = sub.add_parser("reindex", help="Rebuild search index")
    p_rei.add_argument("--source", dest="source_label", default=None)
    p_rei.add_argument("--full", action="store_true", help="Force full rebuild")

    # query
    p_q = sub.add_parser("query", help="Search documents")
    p_q.add_argument("query_text")
    p_q.add_argument("--source", dest="source_label", default=None)
    p_q.add_argument("--module", default=None)
    p_q.add_argument("--top-k", type=int, default=10)

    # context
    p_ctx = sub.add_parser("context", help="Build context block")
    p_ctx.add_argument("query_text")
    p_ctx.add_argument("--source", dest="source_label", default=None)
    p_ctx.add_argument("--top-k", type=int, default=10)
    p_ctx.add_argument("--max-tokens", type=int, default=6000)

    # status
    sub.add_parser("status", help="Show index status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config()

    if args.command == "source":
        from rag.source_manager import add_source, remove_source, list_sources

        if args.source_action == "add":
            result = add_source(config, args.label, args.path)
            print(result)
        elif args.source_action == "remove":
            result = remove_source(config, args.label)
            print(result)
        elif args.source_action == "list":
            for s in list_sources(config):
                print(f"  {s['label']}: {s['path']}")

    elif args.command == "symbol":
        from rag.symbol_index import SymbolIndex

        idx = SymbolIndex(config.symbol_index_path)
        result = idx.lookup(args.symbol_name)
        if result:
            import json
            print(json.dumps(result, indent=2))
        else:
            print(f"Symbol not found: {args.symbol_name}")

    elif args.command == "reindex":
        from rag.indexer.orchestrator import index_all, index_source

        if args.full:
            # Force full rebuild: delete index state
            import os
            if os.path.exists(config.index_state_path):
                os.remove(config.index_state_path)

        if args.source_label:
            result = index_source(config, args.source_label)
        else:
            result = index_all(config)
        import json
        print(json.dumps(result, indent=2))

    elif args.command == "query":
        from rag.retriever.hybrid import HybridRetriever

        retriever = HybridRetriever(config)
        results = retriever.search(
            args.query_text,
            top_k=args.top_k,
            source_label=args.source_label,
            module=args.module,
        )
        for i, r in enumerate(results, 1):
            symbol = r.chunk.symbol_id or "(narrative)"
            print(f"{i}. [{r.chunk.type}] {symbol} (score={r.score:.4f})")
            print(f"   Source: {r.chunk.source_label}.{r.chunk.source_module} — {r.chunk.source_file}")
            if r.chunk.remarks:
                print(f"   {r.chunk.remarks[:200]}...")
            print()

    elif args.command == "context":
        from rag.retriever.hybrid import HybridRetriever
        from rag.context_builder import build_context

        retriever = HybridRetriever(config)
        results = retriever.search(
            args.query_text,
            top_k=args.top_k,
            source_label=args.source_label,
        )
        ctx = build_context(results, args.query_text, max_tokens=args.max_tokens)
        print(ctx)

    elif args.command == "status":
        import json
        import chromadb

        from rag.symbol_index import SymbolIndex

        client = chromadb.PersistentClient(path=config.chroma_dir)
        collections = client.list_collections()
        total_chunks = 0
        per_source: dict[str, int] = {}

        for coll in collections:
            try:
                count = coll.count()
            except Exception:
                count = 0
            total_chunks += count

            source = coll.name.split(".")[0] if "." in coll.name else coll.name
            per_source[source] = per_source.get(source, 0) + count

        symbol_idx = SymbolIndex(config.symbol_index_path)

        result = {
            "total_chunks": total_chunks,
            "total_sources": len(config.doc_sources),
            "total_collections": len(collections),
            "total_symbols": len(symbol_idx),
            "per_source": per_source,
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
