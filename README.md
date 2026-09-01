# SearchForge

A **hybrid search engine from scratch**: positional inverted index → BM25 ranking →
phrase queries, facet filters, typo tolerance, highlighted snippets, autocomplete —
blended with a co-occurrence semantic layer, upgraded by pairwise learning-to-rank,
and scaled across hash-partitioned shards with global-IDF fan-out.

> Pure Python stdlib. Zero dependencies. CLI + HTTP API + server-rendered UI.

```
P@5 / NDCG@5 / MRR | reported per query set via `cli.py eval`
sharded vs single | merged rankings proven identical (global IDF)
VM-free | everything is data structures + math you can whiteboard
```

## Feature surface

| Layer | What it does |
|---|---|
| **Tokenization** | lowercase regex tokens, stop-word list, light suffix stemmer (longest-match wins) |
| **Index** | `term → [(doc, tf, [positions])]`, doc store, lengths, JSON persistence (atomic tmp+rename), remove support |
| **BM25** | Okapi scoring k1=1.5 b=0.75 with length normalisation; candidate union per query |
| **Query language** | `"exact phrases"` · `-exclusions` (terms & phrases) · `type:`/`site:`/`tag:` filters |
| **Phrases** | true *positional* adjacency check over stored offsets; pure-phrase queries restrict, mixed queries boost |
| **Typo tolerance** | bounded (banded) Levenshtein against the live vocabulary; corrections surfaced as "did you mean" |
| **Snippets** | HTML-safe `<mark>` highlights with word-boundary stems, centred windows, ellipses |
| **Facets** | per-query counts for type/site/tag (+ any registered field) |
| **Autocomplete** | prefix search ordered by document frequency |
| **Hybrid semantic** | sliding-window co-occurrence vectors → cosine similarity blended with normalised BM25 (`α` knob); synonym-ish expansion via related terms |
| **PRF** | RM3-lite pseudo-relevance feedback: top-doc terms folded back into the query |
| **Learning to rank** | pairwise RankNet-style logistic model over features (BM25, phrase hits, title hits, coverage, semantic); click/CTR smoothing signal |
| **Sharding** | crc32-hash partitions; coordinator fans out queries using **global IDF** so cross-shard scores stay comparable — merge proven equal to a single index in tests |
| **Crawler** | same-domain BFS with delay/page/depth caps, script/style stripping, title extraction |
| **Evaluation** | Precision@K, Recall@K, MRR, graded NDCG@K over TSV qrels |
| **API + UI** | `/api/search?q=&limit=&hybrid=1` JSON, `/health`, and a server-rendered results page |

## Quick start

```bash
pip install -e .
python -m pytest # 50 tests

python cli.py index --corpus examples/corpus --out idx.json
python cli.py search "flask django" --index idx.json
python cli.py search '"web development"' --index idx.json
python cli.py serve --index idx.json --port 8805 # UI at /
python cli.py eval --queries q.tsv --qrels qrels.tsv --index idx.json
python cli.py conform # engine agreement checks
```

Query syntax examples: `python -snake`, `"web development"`,
`type:doc flask`, `tag:python site:docs.example.com -java`.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) — includes the BM25 formula as implemented,
why global IDF is mandatory when merging shards, how the semantic layer builds
word-context vectors without embeddings libraries, the RankNet gradient step, and
the exact snippet-highlighting pipeline.
