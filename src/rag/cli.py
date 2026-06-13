"""CLI entry point for mcp-doc-rag."""
from __future__ import annotations

import argparse
import sys

from rag.config import load_config


def _print_reindex_timing(label: str, stats: dict) -> None:
    """Print per-phase timing for one source after reindex."""
    if stats.get("error"):
        print(f"\n  [{label}] ERROR: {stats['error']}")
        return

    elapsed = stats.get("elapsed_seconds", 0)

    print(f"\n  [{label}]")
    print(f"    Files: {stats.get('files_total', 0)} total, "
          f"{stats.get('files_indexed', 0)} indexed, "
          f"{stats.get('files_skipped', 0)} skipped")
    print(f"    Chunks stored: {stats.get('chunks', 0)}")
    print(f"    Timing (s):")
    print(f"      crawl  {stats.get('crawl_time', 0):>8.2f}")
    print(f"      parse  {stats.get('parse_time', 0):>8.2f}")
    print(f"      chunk  {stats.get('chunk_time', 0):>8.2f}")
    print(f"      embed  {stats.get('embed_time', 0):>8.2f}")
    print(f"      chroma {stats.get('chroma_time', 0):>8.2f}")
    print(f"      TOTAL  {elapsed:>8.2f}")


def _print_reindex_summary(sources: dict[str, dict], total_chunks: int) -> None:
    """Print cross-source timing summary after index_all."""
    print(f"\n{'=' * 60}")
    print(f"  Done — {total_chunks} chunks across {len(sources)} source(s)")
    print(f"{'=' * 60}")
    for label, stats in sources.items():
        if stats.get("error"):
            print(f"  {label}: ERROR — {stats['error']}")
            continue
        elapsed = stats.get("elapsed_seconds", 0)
        embed = stats.get("embed_time", 0)
        pct = (embed / elapsed * 100) if elapsed > 0 else 0
        print(f"  {label}: {elapsed:.1f}s total, embed={embed:.1f}s ({pct:.0f}%)")


def _print_bad_cases(result) -> None:
    """Print only the bad case analysis."""
    print(f"Bad Cases ({result.num_queries} queries)")
    print(f"  knowledge_gap:      {result.num_knowledge_gap}")
    print(f"  ranking_failure:    {result.num_ranking_failure}")
    print(f"  rewrite_regression: {result.num_rewrite_regression}")
    print(f"  reranker_regression: {result.num_reranker_regression}")
    if result.bad_cases:
        print()
        for bc in result.bad_cases:
            print(f"[{bc['category']}] {bc['query']}")
            print(f"  {bc['detail']}")


def _run_compare_rewrite(config, args) -> None:
    """Compare three rewrite modes: none vs rule vs LLM."""
    from rag.retriever.hybrid import HybridRetriever
    from rag.eval import evaluate

    print("=== Comparison: No Rewrite ===")
    retriever = HybridRetriever(config)
    r0 = evaluate(retriever, args.queries, source_label=args.source_label, enable_rewrite=False)
    print(f"  Recall@5={r0.recall_at_5:.3f}  Recall@10={r0.recall_at_10:.3f}  MRR={r0.mrr:.3f}  Zero-recall={r0.num_zero_recall}")

    print()
    print("=== Comparison: Rule Rewrite ===")
    r1 = evaluate(retriever, args.queries, source_label=args.source_label, enable_rewrite=True)
    print(f"  Recall@5={r1.recall_at_5:.3f}  Recall@10={r1.recall_at_10:.3f}  MRR={r1.mrr:.3f}  Zero-recall={r1.num_zero_recall}")

    print()
    print("--- Summary ---")
    print(f"{'Metric':<18} {'No Rewrite':>11} {'Rule':>11} {'Delta':>11}")
    for metric, attr in [("Recall@5", "recall_at_5"), ("Recall@10", "recall_at_10"), ("MRR", "mrr")]:
        v0 = getattr(r0, attr)
        v1 = getattr(r1, attr)
        print(f"{metric:<18} {v0:>11.3f} {v1:>11.3f} {v1 - v0:>+11.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Fully-local, MCP-native RAG system for C/C++ SDK documentation.",
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

    # eval
    p_eval = sub.add_parser("eval", help="Run retrieval evaluation")
    p_eval.add_argument("--queries", required=True, help="Path to JSONL queries file")
    p_eval.add_argument("--source", dest="source_label", default=None)
    p_eval.add_argument("--enable-rewrite", action="store_true", help="Enable query rewrite during evaluation")
    p_eval.add_argument("--bad-cases-only", action="store_true", help="Output only bad case analysis")
    p_eval.add_argument("--compare-rewrite", action="store_true", help="Compare LLM rewrite vs rule rewrite vs none")

    # gateway
    sub.add_parser("gateway", help="Run the gateway MCP stdio server")

    # adapter
    sub.add_parser("adapter", help="Run daemon-backed gateway MCP stdio adapter")

    # status
    sub.add_parser("status", help="Show index status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "gateway":
        from rag.gateway.server import main as gateway_main

        gateway_main()
        return

    if args.command == "adapter":
        from rag.adapter import main as adapter_main

        adapter_main()
        return

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
                print("  [full rebuild] index state cleared")

        if args.source_label:
            result = index_source(config, args.source_label)
            _print_reindex_timing(args.source_label, result)
        else:
            result = index_all(config)
            sources = result.get("sources", {})
            for label, stats in sources.items():
                _print_reindex_timing(label, stats)
            _print_reindex_summary(sources, result.get("total_chunks", 0))

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

    elif args.command == "eval":
        from rag.eval import evaluate
        from rag.retriever.hybrid import HybridRetriever

        if args.compare_rewrite:
            _run_compare_rewrite(config, args)
            return

        retriever = HybridRetriever(config)

        result = evaluate(
            retriever,
            args.queries,
            source_label=args.source_label,
            enable_rewrite=args.enable_rewrite,
        )

        if args.bad_cases_only:
            _print_bad_cases(result)
            return

        print("=" * 50)
        print("Retrieval Evaluation Results")
        print("=" * 50)
        print(f"Queries:            {result.num_queries}")
        print(f"Recall@1:           {result.recall_at_1:.3f}")
        print(f"Recall@3:           {result.recall_at_3:.3f}")
        print(f"Recall@5:           {result.recall_at_5:.3f}")
        print(f"Recall@10:          {result.recall_at_10:.3f}")
        print(f"MRR:                {result.mrr:.3f}")
        print(f"NDCG@5:             {result.ndcg_at_5:.3f}")
        print(f"NDCG@10:            {result.ndcg_at_10:.3f}")
        print(f"Latency p50:        {result.latency_p50_ms:.1f}ms")
        print(f"Latency p95:        {result.latency_p95_ms:.1f}ms")
        print(f"Zero-recall queries: {result.num_zero_recall}/{result.num_queries}")

        # Per-stage metrics
        if result.stages:
            print()
            print("--- Per-Stage Metrics ---")
            print(f"{'Stage':<12} {'Recall@5':>9} {'Recall@10':>9} {'MRR':>8}")
            for stage_name in ["bm25", "vector", "rrf", "reranker", "final"]:
                s = result.stages.get(stage_name)
                if s and (s.recall_at_5 > 0 or s.recall_at_10 > 0):
                    print(f"{stage_name:<12} {s.recall_at_5:>9.3f} {s.recall_at_10:>9.3f} {s.mrr:>8.3f}")

        # Bad cases
        print()
        print(f"--- Bad Cases ---")
        print(f"knowledge_gap: {result.num_knowledge_gap}    "
              f"ranking_failure: {result.num_ranking_failure}    "
              f"rewrite_regression: {result.num_rewrite_regression}    "
              f"reranker_regression: {result.num_reranker_regression}")
        if result.bad_cases:
            print()
            for bc in result.bad_cases[:10]:
                print(f"[{bc['category']}] {bc['query']}")
                print(f"  {bc['detail']}")

        if result.zero_recall_queries:
            print()
            print("--- All Zero-Recall Queries ---")
            for q in result.zero_recall_queries:
                print(f"  - {q}")
        print("=" * 50)


if __name__ == "__main__":
    main()
