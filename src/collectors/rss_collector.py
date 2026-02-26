"""
RSS Feed Collector

Collects articles from configured RSS feeds (newsletters, news sites).
Handles feed parsing, duplicate detection, and content extraction.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging
import hashlib
from email.utils import parsedate_to_datetime

import feedparser
from bs4 import BeautifulSoup
import requests

from .base import BaseCollector
from ..models.database import (
    Source,
    Document,
    SourceType,
    get_session,
)


logger = logging.getLogger(__name__)


class RSSCollector(BaseCollector):
    """Collector for RSS feeds"""

    def __init__(self, config: Dict[str, Any], db_path: str = "data/radar.db"):
        super().__init__(config, db_path)
        self.source_type = SourceType.RSS
        self._rate_limit_delay = 1.0

        # Load feed configs
        self.feeds = self._load_feed_config()

    def _load_feed_config(self) -> List[Dict[str, Any]]:
        """Load RSS feed configuration"""
        sources_config = self.config.get("sources", {})
        rss_sources = sources_config.get("rss", {})
        return rss_sources.get("feeds", [])

    def get_or_create_source(self, identifier: str, **kwargs) -> Source:
        """Get or create an RSS source (feed)"""
        session = get_session(self.db_path)
        try:
            source = session.query(Source).filter(
                Source.source_type == SourceType.RSS,
                Source.identifier == identifier
            ).first()

            if source is None:
                source = Source(
                    source_type=SourceType.RSS,
                    identifier=identifier,
                    name=kwargs.get("name", identifier),
                    weight=kwargs.get("weight", 1.0),
                    category=kwargs.get("category", "news"),
                    description=kwargs.get("description", ""),
                    enabled=True,
                )
                session.add(source)
                session.commit()
                session.refresh(source)
                logger.info(f"Created new RSS source: {kwargs.get('name', identifier)}")

            return source
        finally:
            session.close()

    def _generate_external_id(self, entry: Dict) -> str:
        """Generate a unique ID for an RSS entry"""
        # Use entry ID if available, otherwise hash the link + title
        if entry.get("id"):
            return entry["id"]

        content = f"{entry.get('link', '')}{entry.get('title', '')}"
        return hashlib.md5(content.encode()).hexdigest()

    def _parse_date(self, entry: Dict) -> Optional[datetime]:
        """Parse publication date from RSS entry"""
        # Try various date fields
        date_fields = ["published_parsed", "updated_parsed", "created_parsed"]

        for field in date_fields:
            if entry.get(field):
                try:
                    return datetime(*entry[field][:6])
                except (TypeError, ValueError):
                    continue

        # Try parsing string date
        date_str = entry.get("published") or entry.get("updated")
        if date_str:
            try:
                return parsedate_to_datetime(date_str)
            except (TypeError, ValueError):
                pass

        return datetime.now(timezone.utc)

    def _extract_content(self, entry: Dict) -> str:
        """Extract and clean content from RSS entry"""
        # Try content field first
        content = ""

        if entry.get("content"):
            content = entry["content"][0].get("value", "")
        elif entry.get("summary"):
            content = entry["summary"]
        elif entry.get("description"):
            content = entry["description"]

        # Strip HTML tags
        if content:
            soup = BeautifulSoup(content, "html.parser")
            content = soup.get_text(separator=" ", strip=True)

        return content

    def _fetch_full_article(self, url: str, timeout: int = 10) -> Optional[str]:
        """Attempt to fetch full article content from URL"""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script, style, nav, and other non-content elements
            for element in soup(["script", "style", "nav", "header", "footer",
                                 "aside", "iframe", "noscript", "form"]):
                element.decompose()

            # Try common article containers (in priority order)
            selectors = [
                ("article", {}),
                ("div", {"class": "article-content"}),
                ("div", {"class": "article-body"}),
                ("div", {"class": "post-content"}),
                ("div", {"class": "entry-content"}),
                ("div", {"class": "story-body"}),
                ("div", {"class": "content-body"}),
                ("div", {"id": "article-body"}),
                ("div", {"id": "content"}),
                ("main", {}),
            ]

            for tag, attrs in selectors:
                element = soup.find(tag, attrs) if attrs else soup.find(tag)
                if element:
                    text = element.get_text(separator=" ", strip=True)
                    if len(text) > 100:  # Minimum useful content
                        return text[:10000]

            # Fallback: get all paragraph text
            paragraphs = soup.find_all("p")
            if paragraphs:
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
                if len(text) > 100:
                    return text[:10000]

            return None

        except Exception as e:
            logger.debug(f"Could not fetch full article from {url}: {e}")
            return None

    def _process_entry(
        self,
        entry: Dict,
        source: Source,
        fetch_full_content: bool = False
    ) -> Optional[Document]:
        """Process an RSS entry into a Document"""
        try:
            external_id = self._generate_external_id(entry)
            published_at = self._parse_date(entry)
            content = self._extract_content(entry)
            url = entry.get("link", "")

            # Fetch full article content if RSS entry content is thin
            if fetch_full_content and url and len(content) < 200:
                full_content = self._fetch_full_article(url)
                if full_content:
                    content = full_content

            doc = Document(
                source_id=source.id,
                author_id=None,  # RSS typically doesn't have author tracking
                external_id=external_id,
                url=url,
                title=entry.get("title", ""),
                content=content,
                content_type="article",
                published_at=published_at,
                fetched_at=datetime.now(timezone.utc),
                processed=False,
            )

            return doc

        except Exception as e:
            logger.warning(f"Error processing RSS entry: {e}")
            return None

    def collect(
        self,
        limit: int = 50,
        fetch_full_content: bool = True
    ) -> List[Document]:
        """
        Collect articles from configured RSS feeds.

        Args:
            limit: Maximum articles per feed
            fetch_full_content: Whether to fetch full article content

        Returns:
            List of Document objects
        """
        documents = []

        for feed_config in self.feeds:
            feed_url = feed_config.get("url")
            feed_name = feed_config.get("name", feed_url)

            if not feed_url:
                continue

            logger.info(f"Collecting from {feed_name}...")

            # Get or create source
            source = self.get_or_create_source(
                identifier=feed_url,
                name=feed_name,
                weight=feed_config.get("weight", 1.0),
                category=feed_config.get("category", "news"),
                description=feed_config.get("description", ""),
            )

            try:
                # Parse the feed
                feed = feedparser.parse(feed_url)

                if feed.bozo:
                    logger.warning(f"Feed parsing error for {feed_name}: {feed.bozo_exception}")

                for entry in feed.entries[:limit]:
                    doc = self._process_entry(entry, source, fetch_full_content)
                    if doc:
                        documents.append(doc)

                logger.info(f"Collected {len(feed.entries[:limit])} articles from {feed_name}")
                self.rate_limit()

            except Exception as e:
                logger.error(f"Error collecting from {feed_name}: {e}")
                continue

        return documents

    def add_feed(
        self,
        url: str,
        name: str,
        weight: float = 1.0,
        category: str = "news"
    ) -> Source:
        """Add a new RSS feed to track"""
        source = self.get_or_create_source(
            identifier=url,
            name=name,
            weight=weight,
            category=category,
        )
        return source
