# Data collectors package
from .base import BaseCollector
from .rss_collector import RSSCollector

# Optional collectors (require additional packages)
try:
    from .reddit_collector import RedditCollector
except ImportError:
    RedditCollector = None

try:
    from .youtube_collector import YouTubeCollector
except ImportError:
    YouTubeCollector = None

__all__ = [
    "BaseCollector",
    "RSSCollector",
    "RedditCollector",
    "YouTubeCollector",
]
