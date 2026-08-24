import os

import pytest

from searchforge.crawler import extract
from searchforge.eval import evaluate
from searchforge.index import InvertedIndex
from searchforge.search import Searcher
from searchforge.shard import ShardedIndex


class TestCrawlerExtraction:
    def test_extracts_title_text_links(self):
        page = (
            "<html><head><title>Docs Home</title></head><body>"
            "<script>var x = 'ignore me';</script>"
            "<h1>Welcome</h1><p>Search engine internals explained.</p>"
            "<a href='/page2.html'>next</a></body></html>"
        )
        title, text, links = extract(page)
        assert title == "Docs Home"
        assert "ignore me" not in text
        assert "Search engine internals" in text
        assert links == ["/page2.html"]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="no network in CI sandbox")
class TestCrawlerLive:
    def test_local_http_crawl(self, tmp_path):
        import http.server
        import threading

        pages = {
            "/": "<html><title>root</title><body>root page <a href='/a.html'>go</a></body></html>",
            "/a.html": "<html><title>a</title><body>alpha content python</body></html>",
        }

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = pages.get(self.path)
                if body is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                payload = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            from searchforge.crawler import Crawler
            from searchforge.index import InvertedIndex

            index = InvertedIndex()
            crawler = Crawler(index, delay_seconds=0.0, max_pages=5, max_depth=1)
            fetched = crawler.crawl(f"http://127.0.0.1:{server.server_address[1]}/")
            assert len(fetched) == 2
            hits = Searcher(index).search("python").hits
            assert any("a.html" in h.doc_id for h in hits)
        finally:
            server.shutdown()


class TestShardPersistence:
    def test_save_all_load_all_roundtrip(self, tmp_path):
        sharded = ShardedIndex(3)
        for i in range(12):
            sharded.add(f"d{i}", f"document {i} about python searching")
        base = str(tmp_path / "idx")
        sharded.save_all(base)
        loaded = ShardedIndex.load_all(base, 3)
        terms = ["python", "searching"]

        original = sorted(sharded.search(terms, limit=10))
        restored = sorted(loaded.search(terms, limit=10))
        assert original == restored


class TestEvalEndToEnd:
    def test_evaluate_real_rankings(self):
        qrels = {"q1": {"d1": 2, "d3": 1}, "q2": {"d2": 1}}
        queries = {"q1": "python web frameworks", "q2": "snake jungle"}

        index = InvertedIndex()
        docs = [
            ("d1", "Python programming language for web development", None),
            ("d2", "Python snake in the jungle", None),
            ("d3", "Flask and Django are Python web frameworks", None),
        ]
        for d, t, f in docs:
            index.add(d, t, f)
        searcher = Searcher(index)

        run = {
            qid: [h.doc_id for h in searcher.search(text).hits]
            for qid, text in queries.items()
        }
        report = evaluate(run, qrels, k_values=(1, 2))
        assert report["counted_queries"] == 2
        assert report["NDCG@1"] > 0
