"""
Base Collector Interface

Abstract base class for all data collectors.
Provides common functionality for rate limiting, error handling, and data persistence.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import logging
import time

from ..models.database import (
    Source,
    Author,
    Document,
    SourceType,
    get_session,
)


logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base class for data collectors"""

    def __init__(self, config: Dict[str, Any], db_path: str = "data/radar.db"):
        self.config = config
        self.db_path = db_path
        self.source_type: SourceType = None
        self._rate_limit_delay = 1.0  # seconds between requests

    @abstractmethod
    def collect(self, limit: int = 100) -> List[Document]:
        """
        Collect documents from the source.

        Args:
            limit: Maximum number of documents to collect per source

        Returns:
            List of Document objects (not yet committed to DB)
        """
        pass

    @abstractmethod
    def get_or_create_source(self, identifier: str, **kwargs) -> Source:
        """Get existing source or create a new one"""
        pass

    def get_or_create_author(
        self,
        username: str,
        platform: SourceType,
        **kwargs
    ) -> Author:
        """Get existing author or create a new one"""
        session = get_session(self.db_path)
        try:
            author = session.query(Author).filter(
                Author.platform == platform,
                Author.username == username
            ).first()

            if author is None:
                author = Author(
                    platform=platform,
                    username=username,
                    **kwargs
                )
                session.add(author)
                session.commit()
                session.refresh(author)
                logger.debug(f"Created new author: {username}")
            else:
                # Update last seen
                author.last_seen_at = datetime.now(timezone.utc)
                session.commit()

            return author
        finally:
            session.close()

    def save_documents(self, documents: List[Document]) -> int:
        """
        Save documents to database, handling duplicates.

        Returns:
            Number of new documents saved
        """
        session = get_session(self.db_path)
        saved_count = 0

        try:
            for doc in documents:
                # Check if document already exists
                existing = session.query(Document).filter(
                    Document.source_id == doc.source_id,
                    Document.external_id == doc.external_id
                ).first()

                if existing is None:
                    session.add(doc)
                    saved_count += 1
                else:
                    # Update engagement metrics for existing doc
                    existing.upvotes = doc.upvotes
                    existing.score = doc.score
                    existing.comment_count = doc.comment_count
                    existing.view_count = doc.view_count

            session.commit()
            logger.info(f"Saved {saved_count} new documents")
            return saved_count
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving documents: {e}")
            raise
        finally:
            session.close()

    def rate_limit(self):
        """Apply rate limiting between requests"""
        time.sleep(self._rate_limit_delay)

    def run(self, limit: int = 100) -> int:
        """
        Run the collector and save results.

        Returns:
            Number of new documents collected
        """
        logger.info(f"Starting {self.__class__.__name__} collection...")
        try:
            documents = self.collect(limit=limit)
            saved = self.save_documents(documents)
            logger.info(f"Collection complete. {saved} new documents.")
            return saved
        except Exception as e:
            logger.error(f"Collection failed: {e}")
            raise
