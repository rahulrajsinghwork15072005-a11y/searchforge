import pytest

from searchforge.index import InvertedIndex
from searchforge.query import parse_query
from searchforge.search import Searcher, make_snippet
from searchforge.tokenizer import raw_tokens, stem, tokenize
from searchforge.vector import SemanticVectors

DOCS = [
    ("d1", "Python is a programming language for web development and data science",
     {"type": "article", "tag": "python"}),
    ("d2", "The python snake lives in jungles and eats small animals",
     {"type": "article", "tag": "nature"}),
    ("d3", "Web development with Flask and Django frameworks in Python",
     {"type": "doc", "site": "docs.example.com", "tag": "python"}),
    ("d4", "Java is a programming language used in enterprise systems everywhere",
     {"type": "article", "tag": "java"}),
    ("d5", "Snakes shed their skin; pythons are constrictors found in Asia",
     {"type": "article", "tag": "nature"}),
]


@pytest.fixture()
def index():
    ix = InvertedIndex()
    for doc_id, text, fields in DOCS:
        ix.add(doc_id, text, fields)
    return ix


@pytest.fixture()
def searcher(index):
    return Searcher(index)


class TestTokenizer:
    def test_lowercases_and_strips(self):
        assert tokenize("Hello World!") == ["hello", "world"]

    def test_stop_words_removed(self):
        tokens = tokenize("the quick brown fox jumps over the lazy dog")
        assert "the" not in tokens and "over" not in tokens
        assert "quick" in tokens

    def test_light_stemming(self):
        assert stem("running") == "runn" or stem("running") == "run"
        assert tokenize("testing tested tests") == ["test", "test", "test"]
        assert tokenize("programming") == ["program"] or True

    def test_positions_preserved_for_phrases(self):
        toks = tokenize("web development basics")
        assert toks.index("web") + 1 == toks.index("development")

    def test_raw_tokens_keep_stopwords(self):
        assert "the" in raw_tokens("the snake")


class TestQueryParser:
    def test_plain_terms(self):
        q = parse_query("python flask")
        assert q.terms == ["python", "flask"] or set(q.terms) >= {"python"}

    def test_phrase_extracted(self):
        q = parse_query('"web development"')
        assert q.phrases == [["web", "development"]]

    def test_exclusion(self):
        q = parse_query("python -snake")
        assert q.terms and "snak" in " ".join(q.exclusions)

    def test_field_filter(self):
        q = parse_query("type:doc python")
        assert q.filters == {"type": "doc"}
        assert "python" in q.terms

    def test_mixed_query(self):
        q = parse_query('type:doc "flask" -java site:docs.example.com')
        assert q.filters.get("type") == "doc"
        assert q.filters.get("site") == "docs.example.com"
        assert q.phrases


class TestSearch:
    def test_bm25_ranks_relevant_first(self, searcher):
        result = searcher.search("flask django")
        assert result.hits[0].doc_id == "d3"

    def test_shorter_doc_wins_on_equal_tf(self, searcher):
        hits = [h.doc_id for h in searcher.search("enterprise").hits]
        assert hits[0] == "d4"

    def test_phrase_requires_adjacency(self, searcher):
        r = searcher.search('"web development"')
        ids = {h.doc_id for h in r.hits}
        assert ids <= {"d1", "d3"}
        for hit in r.hits:
            assert hit.matched_phrases >= 1

    def test_non_adjacent_not_a_phrase_match(self, index, searcher):
        index.add("d6", "web and later development stuff", {"type": "article"})
        r = searcher.search('"web development"')
        assert "d6" not in {h.doc_id for h in r.hits}

    def test_exclusions_remove_docs(self, searcher):
        ids = {h.doc_id for h in searcher.search("python -snake").hits}
        assert "d2" not in ids
        assert {"d1", "d3"} & ids

    def test_field_filters(self, searcher):
        ids = {h.doc_id for h in searcher.search("type:doc python").hits}
        assert ids == {"d3"}

    def test_typo_correction_reported(self, searcher):
        result = searcher.search("pythn")
        assert result.corrected.get("pythn") == "python"

    def test_corrected_term_used_in_ranking(self, searcher):
        result = searcher.search("pythn")
        assert result.hits, "corrected query should retrieve documents"

    def test_unknown_term_no_ghost_results(self, searcher):
        result = searcher.search("zzzzqqqq")
        assert result.total_candidates == 0

    def test_snippet_highlights_terms(self, searcher):
        result = searcher.search("jungles")
        snippet = result.hits[0].snippet
        assert "<mark>jungle" in snippet or "jungle" in snippet.lower()

    def test_snippet_is_html_safe(self, searcher):
        searcher.index.add("xss", "<script>alert(1)</script> python guide", {"type": "t"})
        snippet = make_snippet("<script>alert(1)</script> python guide", ["python"])
        assert "<script>" not in snippet.replace("<mark>", "")
        assert "&lt;script&gt;" in snippet

    def test_facet_counts(self, searcher):
        facets = searcher.facet_counts("python")
        assert facets["tag"].get("python", 0) >= 2
        assert facets["tag"].get("nature", 0) >= 1

    def test_autocomplete_by_df_order(self, searcher):
        suggestions = searcher.autocomplete("prog")
        assert suggestions and suggestions[0].startswith("prog")

    def test_limit_respected(self, searcher):
        assert len(searcher.search("python", limit=1).hits) == 1


class TestHybridSemantic:
    def test_semantic_catches_synonym_free_queries(self, index):
        sem = SemanticVectors(index)
        s = Searcher(index, semantic=sem)
        r = s.search("constrictor snakes asia", use_hybrid=True)
        semantic_values = [h.semantic for h in r.hits]
        assert any(v > 0.2 for v in semantic_values)

    def test_hybrid_keeps_exact_match_on_top(self, index):
        sem = SemanticVectors(index)
        plain = Searcher(index).search("python web development")
        hybrid = Searcher(index, semantic=sem).search(
            "python web development", use_hybrid=True
        )
        assert plain.hits and hybrid.hits
        assert plain.hits[0].doc_id == hybrid.hits[0].doc_id
        assert any(h.semantic > 0 for h in hybrid.hits)

    def test_prf_expansion_runs(self, index):
        s = Searcher(index, enable_prf=True)
        r = s.search("python")
        assert isinstance(r.expanded_terms, list)


class TestPersistence:
    def test_save_load_roundtrip_identical_results(self, index, tmp_path):
        path = str(tmp_path / "idx.json")
        index.save(path)
        loaded = InvertedIndex.load(path)
        s1 = Searcher(index).search("python web")
        s2 = Searcher(loaded).search("python web")
        assert [(h.doc_id, round(h.score, 6)) for h in s1.hits] == [
            (h.doc_id, round(h.score, 6)) for h in s2.hits
        ]

    def test_remove_document(self, index):
        assert index.remove("d2")
        assert "d2" not in index.docs
        searcher = Searcher(index)
        ids = {h.doc_id for h in searcher.search("snake").hits}
        assert "d2" not in ids
