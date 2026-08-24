"""Polite same-domain web crawler: BFS, delay, page/depth caps, text extraction."""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from .index import InvertedIndex

USER_AGENT = "SearchForgeBot/0.1 (+polite; same-domain only)"
_SKIP_TAGS = {"script", "style", "noscript"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text_parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        for name, value in attrs:
            if tag == "a" and name == "href" and value:
                self.links.append(value)

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._skip_depth == 0 and data.strip():
            self.text_parts.append(data.strip())


def extract(html_text: str) -> tuple[str, str, list[str]]:
    parser = _TextExtractor()
    parser.feed(html_text)
    return parser.title.strip(), " ".join(parser.text_parts), parser.links


class Crawler:
    def __init__(
        self,
        index: InvertedIndex,
        delay_seconds: float = 0.5,
        max_pages: int = 20,
        max_depth: int = 2,
    ) -> None:
        self.index = index
        self.delay = delay_seconds
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.visited: set[str] = set()

    def crawl(self, start_url: str) -> list[str]:
        domain = urllib.parse.urlparse(start_url).netloc
        queue: list[tuple[str, int]] = [(start_url, 0)]
        fetched: list[str] = []
        while queue and len(fetched) < self.max_pages:
            url, depth = queue.pop(0)
            normalized = urllib.parse.urldefrag(url)[0]
            if normalized in self.visited:
                continue
            self.visited.add(normalized)
            parsed = urllib.parse.urlparse(normalized)
            if parsed.netloc != domain:
                continue
            try:
                request = urllib.request.Request(
                    normalized, headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=10) as resp:
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        continue
                    body = resp.read().decode("utf-8", "replace")
            except (OSError, ValueError):
                continue

            title, text, links = extract(body)
            doc_id = normalized
            self.index.add(
                doc_id,
                f"{title} {text}".strip(),
                {"type": "page", "site": domain, "title": title},
            )
            fetched.append(doc_id)

            if depth < self.max_depth:
                for link in links:
                    absolute = urllib.parse.urljoin(normalized, link)
                    same_domain = urllib.parse.urlparse(absolute).netloc == domain
                    if same_domain and absolute not in self.visited:
                        queue.append((absolute, depth + 1))

            if len(fetched) < self.max_pages and queue:
                time.sleep(self.delay)
        return fetched


_ = InvertedIndex
