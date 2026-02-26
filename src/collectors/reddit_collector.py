"""
Reddit Data Collector

Collects posts and comments from configured subreddits using PRAW.
Handles authentication, rate limiting, and duplicate detection.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging

import praw
from praw.models import Submission, Comment

from .base import BaseCollector
from ..models.database import (
    Source,
    Author,
    Document,
    SourceType,
    get_session,
)


logger = logging.getLogger(__name__)


class RedditCollector(BaseCollector):
    """Collector for Reddit posts and comments"""

    def __init__(self, config: Dict[str, Any], db_path: str = "data/radar.db"):
        super().__init__(config, db_path)
        self.source_type = SourceType.REDDIT
        self._rate_limit_delay = 2.0  # Reddit API is strict

        # Initialize PRAW client
        reddit_config = config.get("reddit", {})
        self.reddit = praw.Reddit(
            client_id=reddit_config.get("client_id", ""),
            client_secret=reddit_config.get("client_secret", ""),
            user_agent=reddit_config.get("user_agent", "RotationRadar/1.0"),
        )

        # Load subreddit configs
        self.subreddits = self._load_subreddit_config()

    def _load_subreddit_config(self) -> List[Dict[str, Any]]:
        """Load subreddit configuration from sources config"""
        sources_config = self.config.get("sources", {})
        reddit_sources = sources_config.get("reddit", {})
        return reddit_sources.get("subreddits", [])

    def get_or_create_source(self, identifier: str, **kwargs) -> Source:
        """Get or create a Reddit source (subreddit)"""
        session = get_session(self.db_path)
        try:
            source = session.query(Source).filter(
                Source.source_type == SourceType.REDDIT,
                Source.identifier == identifier
            ).first()

            if source is None:
                source = Source(
                    source_type=SourceType.REDDIT,
                    identifier=identifier,
                    name=f"r/{identifier}",
                    weight=kwargs.get("weight", 1.0),
                    category=kwargs.get("category", "general"),
                    description=kwargs.get("description", ""),
                    enabled=True,
                )
                session.add(source)
                session.commit()
                session.refresh(source)
                logger.info(f"Created new source: r/{identifier}")

            return source
        finally:
            session.close()

    def _process_submission(
        self,
        submission: Submission,
        source: Source
    ) -> Optional[Document]:
        """Process a Reddit submission into a Document"""
        try:
            # Get or create author
            author = None
            if submission.author:
                author = self.get_or_create_author(
                    username=submission.author.name,
                    platform=SourceType.REDDIT,
                    karma_score=getattr(submission.author, "link_karma", 0) +
                               getattr(submission.author, "comment_karma", 0),
                    account_created_at=datetime.fromtimestamp(
                        submission.author.created_utc, tz=timezone.utc
                    ) if hasattr(submission.author, "created_utc") else None,
                )

            # Create document
            doc = Document(
                source_id=source.id,
                author_id=author.id if author else None,
                external_id=submission.id,
                url=f"https://reddit.com{submission.permalink}",
                title=submission.title,
                content=submission.selftext or "",
                content_type="post",
                published_at=datetime.fromtimestamp(
                    submission.created_utc, tz=timezone.utc
                ),
                fetched_at=datetime.now(timezone.utc),
                upvotes=submission.ups,
                downvotes=submission.downs if hasattr(submission, "downs") else 0,
                score=submission.score,
                comment_count=submission.num_comments,
                processed=False,
            )

            return doc

        except Exception as e:
            logger.warning(f"Error processing submission {submission.id}: {e}")
            return None

    def _process_comment(
        self,
        comment: Comment,
        source: Source,
        post_title: str = ""
    ) -> Optional[Document]:
        """Process a Reddit comment into a Document"""
        try:
            # Skip deleted/removed comments
            if comment.body in ["[deleted]", "[removed]"]:
                return None

            # Get or create author
            author = None
            if comment.author:
                author = self.get_or_create_author(
                    username=comment.author.name,
                    platform=SourceType.REDDIT,
                    karma_score=getattr(comment.author, "comment_karma", 0),
                )

            doc = Document(
                source_id=source.id,
                author_id=author.id if author else None,
                external_id=comment.id,
                url=f"https://reddit.com{comment.permalink}",
                title=f"Re: {post_title[:100]}..." if len(post_title) > 100 else f"Re: {post_title}",
                content=comment.body,
                content_type="comment",
                published_at=datetime.fromtimestamp(
                    comment.created_utc, tz=timezone.utc
                ),
                fetched_at=datetime.now(timezone.utc),
                upvotes=comment.ups,
                score=comment.score,
                processed=False,
            )

            return doc

        except Exception as e:
            logger.warning(f"Error processing comment {comment.id}: {e}")
            return None

    def collect(
        self,
        limit: int = 100,
        include_comments: bool = True,
        comments_per_post: int = 10,
        sort: str = "hot"
    ) -> List[Document]:
        """
        Collect posts and comments from configured subreddits.

        Args:
            limit: Max posts per subreddit
            include_comments: Whether to collect top comments
            comments_per_post: Number of top comments to collect per post
            sort: How to sort posts (hot, new, top, rising)

        Returns:
            List of Document objects
        """
        documents = []

        for sub_config in self.subreddits:
            sub_name = sub_config.get("name")
            if not sub_name:
                continue

            logger.info(f"Collecting from r/{sub_name}...")

            # Get or create source
            source = self.get_or_create_source(
                identifier=sub_name,
                weight=sub_config.get("weight", 1.0),
                category=sub_config.get("category", "general"),
                description=sub_config.get("description", ""),
            )

            try:
                subreddit = self.reddit.subreddit(sub_name)

                # Get posts based on sort method
                if sort == "hot":
                    posts = subreddit.hot(limit=limit)
                elif sort == "new":
                    posts = subreddit.new(limit=limit)
                elif sort == "top":
                    posts = subreddit.top(limit=limit, time_filter="day")
                elif sort == "rising":
                    posts = subreddit.rising(limit=limit)
                else:
                    posts = subreddit.hot(limit=limit)

                for submission in posts:
                    # Process post
                    doc = self._process_submission(submission, source)
                    if doc:
                        documents.append(doc)

                    # Process top comments
                    if include_comments:
                        submission.comments.replace_more(limit=0)  # Skip "load more"
                        for comment in submission.comments[:comments_per_post]:
                            comment_doc = self._process_comment(
                                comment, source, submission.title
                            )
                            if comment_doc:
                                documents.append(comment_doc)

                    self.rate_limit()

                logger.info(f"Collected {len(documents)} documents from r/{sub_name}")

            except Exception as e:
                logger.error(f"Error collecting from r/{sub_name}: {e}")
                continue

        return documents

    def collect_by_keyword(
        self,
        keywords: List[str],
        subreddits: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Document]:
        """
        Search for posts containing specific keywords.

        Args:
            keywords: List of keywords to search for
            subreddits: Optional list of subreddits to search in
            limit: Max results per keyword

        Returns:
            List of Document objects
        """
        documents = []

        # Default to configured subreddits
        if subreddits is None:
            subreddits = [s.get("name") for s in self.subreddits if s.get("name")]

        for keyword in keywords:
            logger.info(f"Searching for: {keyword}")

            for sub_name in subreddits:
                source = self.get_or_create_source(identifier=sub_name)

                try:
                    subreddit = self.reddit.subreddit(sub_name)
                    results = subreddit.search(keyword, limit=limit, time_filter="week")

                    for submission in results:
                        doc = self._process_submission(submission, source)
                        if doc:
                            documents.append(doc)

                    self.rate_limit()

                except Exception as e:
                    logger.error(f"Error searching r/{sub_name} for '{keyword}': {e}")
                    continue

        return documents
