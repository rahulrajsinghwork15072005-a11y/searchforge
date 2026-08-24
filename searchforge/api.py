"""HTTP API + minimal server-rendered UI on stdlib http.server."""

from __future__ import annotations

import html
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SearchForge</title>
<style>body{{font-family:Georgia,serif;max-width:760px;margin:2rem auto;padding:0 1rem}}
input{{width:60%;padding:.5rem;font-size:1rem}} button{{padding:.5rem 1rem}}
.hit{{margin:1.2rem 0}} .hit a{{color:#1a4d8f;font-size:1.05rem}}
mark{{background:#ffe58a}} .meta{{color:#777;font-size:.85rem}}</style></head><body>
<h1>SearchForge</h1>
<form method="get" action="/search"><input name="q" value="{query}">
<button>Search</button></form>{results}</body></html>"""


def make_handler(searcher, metrics: dict | None = None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: str, status: int = 200, content_type="text/html; charset=utf-8"):
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            query = (params.get("q") or [""])[0]
            limit = min(50, int((params.get("limit") or ["10"])[0]))
            hybrid = (params.get("hybrid") or ["0"])[0] == "1"

            if parsed.path == "/api/search":
                result = searcher.search(query, limit=limit, use_hybrid=hybrid) if query else None
                if result is None:
                    return self._send(json.dumps({"hits": []}), content_type="application/json")
                facets = searcher.facet_counts(query) if query else {}
                body = json.dumps(
                    {
                        "query": query,
                        "corrected": result.corrected,
                        "total": result.total_candidates,
                        "facets": facets,
                        "hits": [
                            {
                                "id": h.doc_id,
                                "score": round(h.score, 4),
                                "snippet": h.snippet,
                                "facets": h.facets,
                            }
                            for h in result.hits
                        ],
                    },
                    indent=2,
                )
                if metrics is not None:
                    metrics["api_searches"] = metrics.get("api_searches", 0) + 1
                return self._send(body, content_type="application/json")

            if parsed.path == "/health":
                return self._send(
                    json.dumps({"status": "ok", "docs": searcher.index.doc_count}),
                    content_type="application/json",
                )

            if parsed.path in ("/", "/search"):
                results_html = ""
                if query:
                    result = searcher.search(query, limit=limit, use_hybrid=hybrid)
                    if result.corrected:
                        fixed = " ".join(result.corrected.get(t, t) for t in query.split())
                        results_html += f'<p class="meta">did you mean <b>{fixed}</b>?</p>'
                    for rank, hit in enumerate(result.hits, start=1):
                        title = hit.facets.get("title") or hit.doc_id
                        results_html += (
                            f'<div class="hit"><a href="#">{rank}. {title}</a>'
                            f'<div>{hit.snippet}</div>'
                            f'<div class="meta">score {hit.score:.3f} · {hit.facets}</div></div>'
                        )
                    if not result.hits:
                        results_html = "<p>No results.</p>"
                html_page = PAGE.format(query=html.escape(query), results=results_html)
                return self._send(html_page)

            self._send("not found", status=404)

    return Handler


class SearchApiServer:
    def __init__(self, searcher, host: str = "", port: int = 0, metrics: dict | None = None):
        self._server = ThreadingHTTPServer((host, port), make_handler(searcher, metrics))

    @property
    def bound_port(self) -> int:
        return self._server.server_address[1]

    def start(self):
        import threading

        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


_ = urllib.parse
