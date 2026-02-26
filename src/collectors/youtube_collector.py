"""
YouTube Collector

Collects video metadata and transcripts from configured YouTube channels.
Uses youtube-transcript-api for transcript extraction.
"""

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging
import re

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from .base import BaseCollector
from ..models.database import (
    Source,
    Document,
    SourceType,
    get_session,
)


logger = logging.getLogger(__name__)


# Optional: Google API for channel/video metadata
try:
    from googleapiclient.discovery import build
    HAS_GOOGLE_API = True
except ImportError:
    HAS_GOOGLE_API = False
    logger.info("Google API client not installed. YouTube collection will be limited.")


class YouTubeCollector(BaseCollector):
    """Collector for YouTube videos and transcripts"""

    def __init__(self, config: Dict[str, Any], db_path: str = "data/radar.db"):
        super().__init__(config, db_path)
        self.source_type = SourceType.YOUTUBE
        self._rate_limit_delay = 2.0

        # YouTube API (optional)
        youtube_config = config.get("youtube", {})
        self.api_key = youtube_config.get("api_key", "")
        self.youtube_client = None

        if HAS_GOOGLE_API and self.api_key:
            self.youtube_client = build("youtube", "v3", developerKey=self.api_key)

        # Load channel configs
        self.channels = self._load_channel_config()

    def _load_channel_config(self) -> List[Dict[str, Any]]:
        """Load YouTube channel configuration"""
        sources_config = self.config.get("sources", {})
        youtube_sources = sources_config.get("youtube", {})
        return youtube_sources.get("channels", [])

    def get_or_create_source(self, identifier: str, **kwargs) -> Source:
        """Get or create a YouTube source (channel)"""
        session = get_session(self.db_path)
        try:
            source = session.query(Source).filter(
                Source.source_type == SourceType.YOUTUBE,
                Source.identifier == identifier
            ).first()

            if source is None:
                source = Source(
                    source_type=SourceType.YOUTUBE,
                    identifier=identifier,
                    name=kwargs.get("name", identifier),
                    weight=kwargs.get("weight", 1.0),
                    category=kwargs.get("category", "youtube"),
                    description=kwargs.get("description", ""),
                    enabled=True,
                )
                session.add(source)
                session.commit()
                session.refresh(source)
                logger.info(f"Created new YouTube source: {kwargs.get('name', identifier)}")

            return source
        finally:
            session.close()

    def _extract_video_id(self, url_or_id: str) -> Optional[str]:
        """Extract video ID from URL or return as-is if already an ID"""
        # Already an ID
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
            return url_or_id

        # Extract from various URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)

        return None

    def _get_transcript(self, video_id: str) -> Optional[str]:
        """Fetch transcript for a video"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            # Combine all text segments
            full_text = " ".join([
                segment.get("text", "")
                for segment in transcript_list
            ])

            return full_text

        except TranscriptsDisabled:
            logger.debug(f"Transcripts disabled for video {video_id}")
        except NoTranscriptFound:
            logger.debug(f"No transcript found for video {video_id}")
        except VideoUnavailable:
            logger.debug(f"Video {video_id} unavailable")
        except Exception as e:
            logger.warning(f"Error fetching transcript for {video_id}: {e}")

        return None

    def _get_video_metadata_api(self, video_id: str) -> Optional[Dict]:
        """Fetch video metadata using YouTube API"""
        if not self.youtube_client:
            return None

        try:
            response = self.youtube_client.videos().list(
                part="snippet,statistics",
                id=video_id
            ).execute()

            if response.get("items"):
                item = response["items"][0]
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})

                return {
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "view_count": int(stats.get("viewCount", 0)),
                    "like_count": int(stats.get("likeCount", 0)),
                    "comment_count": int(stats.get("commentCount", 0)),
                }

        except Exception as e:
            logger.warning(f"Error fetching video metadata via API: {e}")

        return None

    def _get_channel_videos_api(
        self,
        channel_id: str,
        max_results: int = 10
    ) -> List[str]:
        """Fetch recent video IDs from a channel using YouTube API"""
        if not self.youtube_client:
            return []

        try:
            # Get channel's upload playlist
            channel_response = self.youtube_client.channels().list(
                part="contentDetails",
                id=channel_id
            ).execute()

            if not channel_response.get("items"):
                return []

            uploads_playlist = (
                channel_response["items"][0]
                .get("contentDetails", {})
                .get("relatedPlaylists", {})
                .get("uploads")
            )

            if not uploads_playlist:
                return []

            # Get videos from playlist
            playlist_response = self.youtube_client.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist,
                maxResults=max_results
            ).execute()

            video_ids = [
                item.get("contentDetails", {}).get("videoId")
                for item in playlist_response.get("items", [])
                if item.get("contentDetails", {}).get("videoId")
            ]

            return video_ids

        except Exception as e:
            logger.warning(f"Error fetching channel videos via API: {e}")
            return []

    def _process_video(
        self,
        video_id: str,
        source: Source,
        metadata: Optional[Dict] = None
    ) -> Optional[Document]:
        """Process a YouTube video into a Document"""
        try:
            # Get metadata from API if not provided
            if metadata is None and self.youtube_client:
                metadata = self._get_video_metadata_api(video_id)

            # Get transcript
            transcript = self._get_transcript(video_id)

            if not transcript and (not metadata or not metadata.get("description")):
                logger.debug(f"No content available for video {video_id}")
                return None

            # Build content from transcript and/or description
            content = ""
            if transcript:
                content = transcript
            elif metadata and metadata.get("description"):
                content = metadata["description"]

            # Parse published date
            published_at = datetime.now(timezone.utc)
            if metadata and metadata.get("published_at"):
                try:
                    published_at = datetime.fromisoformat(
                        metadata["published_at"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            doc = Document(
                source_id=source.id,
                author_id=None,
                external_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=metadata.get("title", "") if metadata else "",
                content=content[:50000],  # Limit transcript length
                content_type="video",
                published_at=published_at,
                fetched_at=datetime.now(timezone.utc),
                view_count=metadata.get("view_count", 0) if metadata else 0,
                upvotes=metadata.get("like_count", 0) if metadata else 0,
                comment_count=metadata.get("comment_count", 0) if metadata else 0,
                processed=False,
            )

            return doc

        except Exception as e:
            logger.warning(f"Error processing video {video_id}: {e}")
            return None

    def collect(self, limit: int = 10) -> List[Document]:
        """
        Collect videos from configured channels.

        Args:
            limit: Maximum videos per channel

        Returns:
            List of Document objects
        """
        documents = []

        for channel_config in self.channels:
            channel_handle = channel_config.get("handle", "")
            channel_id = channel_config.get("id", "")

            if not channel_handle and not channel_id:
                continue

            identifier = channel_id or channel_handle
            logger.info(f"Collecting from YouTube channel: {identifier}...")

            # Get or create source
            source = self.get_or_create_source(
                identifier=identifier,
                name=channel_handle or channel_id,
                weight=channel_config.get("weight", 1.0),
                category=channel_config.get("category", "youtube"),
                description=channel_config.get("description", ""),
            )

            # Get video IDs
            video_ids = []

            if channel_id and self.youtube_client:
                video_ids = self._get_channel_videos_api(channel_id, max_results=limit)
            elif channel_config.get("video_ids"):
                video_ids = channel_config["video_ids"][:limit]

            if not video_ids:
                logger.warning(
                    f"No videos found for {identifier}. "
                    "YouTube API key may be required for automatic video discovery."
                )
                continue

            for video_id in video_ids:
                doc = self._process_video(video_id, source)
                if doc:
                    documents.append(doc)

                self.rate_limit()

            logger.info(f"Collected {len(video_ids)} videos from {identifier}")

        return documents

    def collect_video(self, video_url_or_id: str) -> Optional[Document]:
        """
        Collect a single video by URL or ID.

        Args:
            video_url_or_id: YouTube video URL or ID

        Returns:
            Document object or None
        """
        video_id = self._extract_video_id(video_url_or_id)
        if not video_id:
            logger.error(f"Could not extract video ID from: {video_url_or_id}")
            return None

        # Create a generic source for one-off videos
        source = self.get_or_create_source(
            identifier="youtube_manual",
            name="Manual YouTube Collection",
            category="youtube",
        )

        return self._process_video(video_id, source)

    def collect_from_urls(self, urls: List[str]) -> List[Document]:
        """
        Collect videos from a list of URLs.

        Args:
            urls: List of YouTube video URLs

        Returns:
            List of Document objects
        """
        documents = []

        source = self.get_or_create_source(
            identifier="youtube_manual",
            name="Manual YouTube Collection",
            category="youtube",
        )

        for url in urls:
            video_id = self._extract_video_id(url)
            if video_id:
                doc = self._process_video(video_id, source)
                if doc:
                    documents.append(doc)
                self.rate_limit()

        return documents
