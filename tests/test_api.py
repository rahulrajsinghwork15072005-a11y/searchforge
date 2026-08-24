import json
import urllib.request

import pytest

from searchforge.api import SearchApiServer
from searchforge.index import InvertedIndex
from searchforge.search import Searcher

DOCS = [
    (
        "d1",
        "Python programming language for web development",
        {"type": "article", "title": "Python Guide"},
    ),
    ("d2", "Flask and Django are Python web frameworks", {"type": "doc", "title": "Frameworks"}),
]


@pytest.fixture()
def api():
    index = InvertedIndex()
    for d, t, f in DOCS:
        index.add(d, t, f)
    searcher = Searcher(index)
    metrics = {}
    server = SearchApiServer(searcher, port=0, metrics=metrics)
    server.start()
    yield server, metrics
    server.stop()


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode())


def test_health_endpoint(api):
    server, _ = api
    status, body = _get(server.bound_port, "/health")
    assert status == 200 and body["status"] == "ok" and body["docs"] == 2


def test_api_search_returns_hits_and_facets(api):
    server, _ = api
    status, body = _get(server.bound_port, "/api/search?q=python")
    assert status == 200
    assert body["query"] == "python"
    assert len(body["hits"]) >= 1
    assert "tag" in body["facets"]
    hit = body["hits"][0]
    assert {"id", "score", "snippet"} <= set(hit)


def test_api_search_correction_field(api):
    server, _ = api
    _, body = _get(server.bound_port, "/api/search?q=pythn")
    assert body["corrected"].get("pythn") == "python"


def test_api_limit_param(api):
    server, _ = api
    _, body = _get(server.bound_port, "/api/search?q=python&limit=1")
    assert len(body["hits"]) == 1


def test_html_ui_renders_results_with_marks(api):
    server, _ = api
    with urllib.request.urlopen(
        f"http://127.0.0.1:{server.bound_port}/search?q=python", timeout=5
    ) as resp:
        html_text = resp.read().decode()
    assert "<mark>" in html_text
    assert "Python Guide" in html_text


def test_metrics_counter_increments(api):
    server, metrics = api
    before = metrics.get("api_searches", 0)
    _get(server.bound_port, "/api/search?q=python")
    assert metrics.get("api_searches", 0) == before + 1
