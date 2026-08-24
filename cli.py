"""SearchForge CLI: index, search, serve, eval, crawl, spellcheck demos."""

from __future__ import annotations

import argparse
import json
import sys

from searchforge.index import InvertedIndex
from searchforge.search import Searcher
from searchforge.vector import SemanticVectors


def _load_or_build(args) -> tuple[InvertedIndex, Searcher, SemanticVectors | None]:
    if getattr(args, "index", None):
        index = InvertedIndex.load(args.index)
    else:
        index = InvertedIndex()
        corpus_dir = args.corpus
        for name in sorted(os_listdir(corpus_dir)):
            path = f"{corpus_dir.rstrip('/')}/{name}"
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            doc_type = name.rsplit(".", 1)[-1]
            index.add(name, text, {"type": doc_type})
    semantic = None
    if not getattr(args, "no_semantic", False):
        semantic = SemanticVectors(index)
    return index, Searcher(
        index,
        semantic=semantic,
        enable_typo_tolerance=not getattr(args, "no_typo", False),
        enable_prf=getattr(args, "prf", False),
    ), semantic


def os_listdir(path: str) -> list[str]:
    import os

    try:
        return os.listdir(path)
    except OSError:
        return []


def cmd_index(args) -> int:
    index = InvertedIndex()
    count = 0
    for name in sorted(os_listdir(args.corpus)):
        path = f"{args.corpus.rstrip('/')}/{name}"
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        index.add(name, text, {"type": name.rsplit(".", 1)[-1]})
        count += 1
    index.save(args.out)
    print(f"indexed {count} documents -> {args.out}")
    return 0


def cmd_search(args) -> int:
    _, searcher, _ = _load_or_build(args)
    result = searcher.search(args.query, limit=args.limit, use_hybrid=args.hybrid)
    payload = {
        "query": args.query,
        "corrected": result.corrected,
        "total": result.total_candidates,
        "hits": [
            {"id": h.doc_id, "score": round(h.score, 4), "snippet": h.snippet} for h in result.hits
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_serve(args) -> int:
    from searchforge.api import SearchApiServer

    _, searcher, _ = _load_or_build(args)
    metrics: dict = {}
    server = SearchApiServer(searcher, host=args.host, port=args.port, metrics=metrics)
    server.start()
    print(f"searchforge serving http://127.0.0.1:{server.bound_port}/")
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def cmd_eval(args) -> int:
    from searchforge.eval import evaluate

    qrels = {}
    with open(args.qrels, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            query_id, doc_id, grade = parts
            qrels.setdefault(query_id, {})[doc_id] = int(grade)

    queries = {}
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                queries[parts[0]] = parts[1]

    _, searcher, _ = _load_or_build(args)
    run_results = {}
    for query_id, query_text in queries.items():
        result = searcher.search(query_text, limit=args.k)
        run_results[query_id] = [h.doc_id for h in result.hits]

    report = evaluate(run_results, qrels)
    print(json.dumps(report, indent=2))
    return 0


def cmd_crawl(args) -> int:
    from searchforge.crawler import Crawler

    index = InvertedIndex()
    crawler = Crawler(index, max_pages=args.max_pages, max_depth=args.depth)
    fetched = crawler.crawl(args.url)
    print(f"crawled {len(fetched)} pages")
    if args.index_out:
        index.save(args.index_out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="searchforge")
    sub = root.add_subparsers(dest="cmd", required=True)

    idx = sub.add_parser("index", help="build an index from a directory of files")
    idx.add_argument("--corpus", required=True)
    idx.add_argument("--out", required=True)
    idx.set_defaults(fn=cmd_index)

    search_p = sub.add_parser("search", help="run a query")
    search_p.add_argument("query")
    common = {}
    search_p.add_argument("--index", default=None)
    search_p.add_argument("--corpus", default=None)
    search_p.add_argument("--limit", type=int, default=10)
    search_p.add_argument("--hybrid", action="store_true")
    search_p.add_argument("--prf", action="store_true")
    search_p.add_argument("--no-typo", action="store_true")
    search_p.add_argument("--no-semantic", action="store_true")
    search_p.set_defaults(fn=cmd_search, **common)

    serve = sub.add_parser("serve", help="HTTP API + UI")
    serve.add_argument("--index", default=None)
    serve.add_argument("--corpus", default=None)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8805)
    serve.add_argument("--no-typo", action="store_true")
    serve.add_argument("--no-semantic", action="store_true")
    serve.set_defaults(fn=cmd_serve)

    ev = sub.add_parser("eval", help="evaluate against qrels TSV")
    ev.add_argument("--qrels", required=True)
    ev.add_argument("--queries", required=True)
    ev.add_argument("--index", default=None)
    ev.add_argument("--corpus", default=None)
    ev.add_argument("--k", type=int, default=5)
    ev.add_argument("--no-typo", action="store_true")
    ev.add_argument("--no-semantic", action="store_true")
    ev.set_defaults(fn=cmd_eval)

    crawl = sub.add_parser("crawl", help="polite same-domain crawl into an index")
    crawl.add_argument("--url", required=True)
    crawl.add_argument("--max-pages", type=int, default=10)
    crawl.add_argument("--depth", type=int, default=1)
    crawl.add_argument("--index-out", default=None)
    crawl.set_defaults(fn=cmd_crawl)
    return root


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
