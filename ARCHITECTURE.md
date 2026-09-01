# SearchForge architecture

```
 docs / crawled pages
 │ tokenize (lower · stop words · light stem)
 ▼
 positional inverted index ────────────────► persisted JSON (atomic tmp+rename)
 term → [(doc, tf, [positions])]
 doc store + lengths + facets
 │
 query parser ──► ParsedQuery query.py
 terms · phrases · exclusions · filters
 │
 ┌──────┴───────────────┬──────────────────┐
 │ BM25 scorer │ SpellCorrector │ SemanticVectors
 │ k1=1.5 b=0.75 │ banded Levenshtein│ co-occurrence vectors
 │ candidate union │ vocab-only fixes │ cosine + PRF expansion
 └──────────┬───────────┴─────────┬─────────┘
 ▼ ▼
 Searcher: score = bm25 + phrase_boost·hits
 (+ α-blend with cosine when hybrid)
 │
 snippets (<mark>, HTML-safe) · facet counts · autocomplete
 │
 RankNetLinear pairwise LTR over feature vectors (incl. semantic, CTR prior)
 │
 ShardedIndex: crc32 partitions · coordinator fans out with GLOBAL IDF
 │
 HTTP API (/api/search JSON) + server-rendered UI api.py
```

## Ranking math

**BM25** per term/document:

```
score = idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·dl/avgdl))
idf(t) = ln((N - df + 0.5) / (df + 0.5) + 1)
```

Query score sums term scores; phrase hits add a flat boost. In hybrid mode BM25 is
squashed via `x/(1+x)` and linearly blended with cosine similarity (α = 0.75 default),
then multiplied by the phrase bonus so exact phrases still dominate.

**Semantic layer** builds word-context vectors from a ±8-token window with inverse-
distance weights, trimmed to top contexts per term. Document vectors are the mean of
their terms' vectors, L2-normalised; query vectors likewise; relevance = dot product.
No embeddings library — this is transparent sparse linear algebra you can debug.

**Learning to rank.** Features: `[bias, norm-BM25, phrase-hit, title-hit, term-coverage,
semantic]`. Pairwise RankNet objective: for a judged pair (winner > loser), minimise
`-log σ(w·fw - w·fl)`; gradients accumulate per epoch with averaged SGD updates. A
smoothed CTR (`(clicks + β·global_ctr) / (views + β)`) provides an additional prior
signal for rerankers fed by real usage.

## Why global IDF is non-negotiable for shards

IDF is a corpus-level property. If each shard computes IDF over only its own documents,
the same term scores differently on different shards and merged rankings are wrong.
`ShardedIndex.search` therefore overrides each shard's scorer with the coordinator's
**global IDF** (summed df across shards), then merges by max-score. The test suite
proves merged top-K equals a single monolithic index built from the same documents.

## Snippet pipeline

Raw text → find first match position → centre a window → HTML-escape once → wrap
matches with `<mark>` using word-boundary stem patterns (`\w*stem\w*`) so stemmed
queries highlight full surface forms ("jungl" highlights "jungles") → add ellipses.
Escaping happens before marking, and the mark wrapper re-escapes only its own group,
so injected markup can never break out.

## Soundness of the sweep

The heap pins freshly allocated objects until the next statement boundary because,
mid-expression, a just-created array may not yet be reachable from any root while its
evaluating call chain still needs it. Without pinning, mark-sweep would free live
data — the classic precise-GC bug. Statement boundaries are natural release points:
by then every value worth keeping is reachable from environments (interpreter) or
stack/frames/globals (VM).

## Evaluation

TSV qrels (`query_id \t doc_id \t grade`) plus TSV queries drive `evaluate`, which
reports Precision@K, Recall@K, graded NDCG@K (log-discounted, ideal-DCG normalised)
and MRR — the four numbers search interviews always ask about.
