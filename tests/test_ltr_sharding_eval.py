import pytest

from searchforge.eval import (
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from searchforge.index import InvertedIndex
from searchforge.ltr import FeatureExtractor, RankNetLinear
from searchforge.search import Searcher
from searchforge.shard import ShardedIndex, shard_for


def build_index():
    ix = InvertedIndex()
    docs = [
        ("d1", "Python programming language for web development", {"type": "article"}),
        ("d2", "Python snake in the jungle", {"type": "article"}),
        ("d3", "Flask and Django are Python web frameworks", {"type": "doc"}),
        ("d4", "Java enterprise systems programming", {"type": "article"}),
        ("d5", "Data science with Python pandas numpy", {"type": "doc"}),
        ("d6", "Snakes of Asia: pythons and cobras", {"type": "article"}),
    ]
    for d, t, f in docs:
        ix.add(d, t, f)
    return ix


class TestEvalMetrics:
    def test_precision_at_k(self):
        assert precision_at_k(["a", "b", "c"], {"a"}, 3) == pytest.approx(1 / 3)
        assert precision_at_k(["a", "a"], {"a"}, 2) == pytest.approx(1.0)

    def test_recall_at_k(self):
        assert recall_at_k(["a"], {"a", "b", "c"}, 1) == pytest.approx(1 / 3)

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["x", "y", "hit"], {"hit"}) == pytest.approx(1 / 3)

    def test_ndcg_graded(self):
        grades = {"a": 2, "b": 1}
        perfect = ndcg_at_k(["a", "b"], grades, 2)
        swapped = ndcg_at_k(["b", "a"], grades, 2)
        assert perfect == pytest.approx(1.0)
        assert swapped < perfect

    def test_evaluate_report_shape(self):
        qrels = {"q1": {"d1": 2, "d2": 0}, "q2": {"d3": 1}}
        run = {"q1": ["d1", "d2"], "q2": ["d4"]}
        report = evaluate(run, qrels, k_values=(1, 2))
        assert report["queries"] == 2
        assert 0 <= report["P@1"] <= 1 and 0 <= report["NDCG@2"] <= 1


class TestSharding:
    def test_shard_routing_is_deterministic(self):
        assert shard_for("doc-a", 3) == shard_for("doc-a", 3)

    def test_all_docs_accounted(self):
        sharded = ShardedIndex(3)
        for i in range(20):
            sharded.add(f"doc-{i}", f"document number {i} about python")
        assert sharded.doc_count == 20
        total_in_shards = sum(s.doc_count for s in sharded.shards)
        assert total_in_shards == 20

    def test_global_idf_differs_from_per_shard(self):
        sharded = ShardedIndex(2)
        for i in range(10):
            sharded.shards[0].add(f"left-{i}", f"common word left {i}")
            sharded.shards[1].add(f"right-{i}", f"rare unique right {i}")
        per_shard_left = sharded.shards[0].idf("word")
        global_left = sharded.global_idf("word")
        assert global_left != per_shard_left

    def test_merged_results_match_single_index(self):
        single = InvertedIndex()
        sharded = ShardedIndex(3)
        docs = [
            (f"d{i}", f"python document number {i} with content {i * 7}", None)
            for i in range(30)
        ]
        for d, t, f in docs:
            single.add(d, t, f)
            sharded.add(d, t, f)

        from searchforge.bm25 import BM25Scorer

        terms = ["python", "content"]
        expected = sorted(
            ((d, round(BM25Scorer(single).score(d, terms), 6)) for d, _, _ in docs),
            key=lambda p: (-p[1], p[0]),
        )[:5]
        got = [(d, round(score, 6)) for d, score in sharded.search(terms, limit=5)]
        assert [d for d, _ in got] == [d for d, _ in expected]


class TestLTR:
    def test_ranknet_learns_to_prefer_relevant(self):
        index = build_index()
        searcher = Searcher(index)
        extractor = FeatureExtractor(searcher)

        samples = {
            "python web": {"d1": 3, "d3": 2, "d2": 0},
            "snake": {"d2": 3, "d6": 2, "d1": 0},
        }
        model = RankNetLinear(len(extractor.features("x", next(iter(index.docs)))))
        stats = model.train(samples, extractor.features, epochs=150, lr=0.1)
        assert stats["pairs"] > 0 and stats["loss"] < 2.0

        def ranked(query):
            hits = searcher.search(query, limit=10).hits
            scored = [
                (h.doc_id, model.score(extractor.features(query, h.doc_id))) for h in hits
            ]
            scored.sort(key=lambda pair: -pair[1])
            return [d for d, _ in scored]

        order = ranked("python web")
        assert "d2" not in order[:2]
        assert set(order) >= {"d1"}

    def test_ctr_boost_smoothed(self):
        from searchforge.ltr import ctr_boost

        clicks = {"d1": 8}
        impressions = {"d1": 10}
        boosted = ctr_boost(clicks, impressions, prior=5)
        assert 0.5 < boosted <= 1.0
