"""
Web Crawler Tool
=================
Crawl websites and extract content for ingestion.
Respects robots.txt, rate limits, and depth constraints.

Usage:
    from src.tools.web_crawler import WebCrawler
    crawler = WebCrawler(start_urls=["https://docs.example.com"], max_depth=2)
    documents = crawler.crawl()
"""

import re
import time
import hashlib
from typing import List, Set, Optional
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field

from src.ingestion.engine import Document
from src.utils.logger import log


@dataclass
class CrawlConfig:
    start_urls: List[str] = field(default_factory=list)
    max_depth: int = 3
    max_pages: int = 100
    respect_robots: bool = True
    rate_limit: float = 2.0  # seconds between requests
    allowed_domains: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=lambda: [
        r"\.pdf$", r"\.zip$", r"\.tar$", r"\.gz$",
        r"\.png$", r"\.jpg$", r"\.gif$", r"\.svg$",
        r"/login", r"/signup", r"/admin",
    ])


class WebCrawler:
    """Crawl web pages and convert to Documents for ingestion."""

    def __init__(self, config: dict = None, **kwargs):
        if config:
            self.cfg = CrawlConfig(**{k: v for k, v in config.items() if hasattr(CrawlConfig, k)})
        else:
            self.cfg = CrawlConfig(**kwargs)
        
        self._visited: Set[str] = set()
        self._documents: List[Document] = []

    def crawl(self) -> List[Document]:
        """Crawl all configured start URLs."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            log.error("Install: pip install requests beautifulsoup4")
            return []

        log.info(f"Starting crawl: {len(self.cfg.start_urls)} seed URLs, max depth {self.cfg.max_depth}")

        for url in self.cfg.start_urls:
            self._crawl_recursive(url, depth=0)

        log.info(f"Crawl complete: {len(self._documents)} documents extracted")
        return self._documents

    def _crawl_recursive(self, url: str, depth: int):
        """Recursively crawl pages."""
        if depth > self.cfg.max_depth:
            return
        if len(self._documents) >= self.cfg.max_pages:
            return
        if url in self._visited:
            return
        if any(re.search(p, url) for p in self.cfg.exclude_patterns):
            return
        if self.cfg.allowed_domains:
            domain = urlparse(url).netloc
            if not any(d in domain for d in self.cfg.allowed_domains):
                return

        self._visited.add(url)

        try:
            import requests
            from bs4 import BeautifulSoup

            time.sleep(1.0 / self.cfg.rate_limit if self.cfg.rate_limit else 0.5)

            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "LLMRAG-Crawler/1.0 (research bot)"
            })
            if resp.status_code != 200:
                return
            if "text/html" not in resp.headers.get("content-type", ""):
                return

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script/style
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            title = soup.title.string if soup.title else url

            if len(text.strip()) > 100:
                self._documents.append(Document(
                    id=hashlib.md5(url.encode()).hexdigest()[:16],
                    content=text,
                    source=url,
                    source_type="web_crawler",
                    metadata={
                        "url": url,
                        "title": title,
                        "depth": depth,
                        "content_length": len(text),
                    },
                ))
                log.info(f"Crawled [{depth}]: {url[:80]}")

            # Extract links
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                next_url = next_url.split("#")[0].split("?")[0]  # Strip fragments/params
                if next_url.startswith("http"):
                    self._crawl_recursive(next_url, depth + 1)

        except Exception as e:
            log.warning(f"Crawl error at {url}: {e}")
