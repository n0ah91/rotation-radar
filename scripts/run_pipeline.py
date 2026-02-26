"""
Pipeline Runner

Runs the full data collection → processing → scoring pipeline.
Can be run manually or scheduled via cron/APScheduler.

Usage:
    python scripts/run_pipeline.py                  # Full pipeline
    python scripts/run_pipeline.py --collect-only    # Just collect data
    python scripts/run_pipeline.py --process-only    # Just process existing data
    python scripts/run_pipeline.py --score-only      # Just score existing signals
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone
import logging
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.database import init_db
from src.collectors.rss_collector import RSSCollector
from src.processing.entity_extraction import EntityExtractor

# Optional collectors (require additional packages)
try:
    from src.collectors.reddit_collector import RedditCollector
except ImportError:
    RedditCollector = None

try:
    from src.collectors.youtube_collector import YouTubeCollector
except ImportError:
    YouTubeCollector = None
from src.processing.sentiment import SentimentAnalyzer
from src.processing.velocity import VelocityEngine
from src.processing.divergence import DivergenceEngine
from src.processing.signal_model import SignalModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config():
    """Load all configuration"""
    config = {}

    # Main config
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        config_path = project_root / "config" / "config.example.yaml"

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # Sources config
    sources_path = project_root / "config" / "sources.yaml"
    if sources_path.exists():
        with open(sources_path) as f:
            config["sources"] = yaml.safe_load(f) or {}

    return config


def run_collection(config: dict, db_path: str):
    """Run all data collectors"""
    logger.info("=== PHASE 1: Data Collection ===")
    total_docs = 0

    # Reddit
    if RedditCollector is not None:
        try:
            reddit_config = config.get("reddit", {})
            if reddit_config.get("client_id") and reddit_config["client_id"] != "YOUR_CLIENT_ID":
                logger.info("Collecting from Reddit...")
                collector = RedditCollector(config, db_path)
                docs = collector.run(limit=50)
                total_docs += docs
                logger.info(f"Reddit: {docs} new documents")
            else:
                logger.warning("Reddit API not configured. Skipping.")
        except Exception as e:
            logger.error(f"Reddit collection failed: {e}")
    else:
        logger.info("Reddit collector not available (praw not installed). Skipping.")

    # RSS
    try:
        logger.info("Collecting from RSS feeds...")
        collector = RSSCollector(config, db_path)
        docs = collector.run(limit=50)
        total_docs += docs
        logger.info(f"RSS: {docs} new documents")
    except Exception as e:
        logger.error(f"RSS collection failed: {e}")

    # YouTube
    if YouTubeCollector is not None:
        try:
            youtube_config = config.get("youtube", {})
            if youtube_config.get("enabled", False) and youtube_config.get("api_key"):
                logger.info("Collecting from YouTube...")
                collector = YouTubeCollector(config, db_path)
                docs = collector.run(limit=10)
                total_docs += docs
                logger.info(f"YouTube: {docs} new documents")
            else:
                logger.info("YouTube API not configured. Skipping.")
        except Exception as e:
            logger.error(f"YouTube collection failed: {e}")
    else:
        logger.info("YouTube collector not available. Skipping.")

    logger.info(f"Collection complete. {total_docs} total new documents.")
    return total_docs


def run_processing(config: dict, db_path: str):
    """Run entity extraction and sentiment analysis"""
    logger.info("=== PHASE 2: Processing ===")

    # Entity extraction
    logger.info("Running entity extraction...")
    taxonomy_path = project_root / "config" / "taxonomy.yaml"
    extractor = EntityExtractor(
        taxonomy_path=str(taxonomy_path),
        db_path=db_path,
    )
    extracted = extractor.process_unprocessed_documents()
    logger.info(f"Entity extraction: processed {extracted} documents")

    # Sentiment analysis
    logger.info("Running sentiment analysis...")
    analyzer = SentimentAnalyzer(db_path=db_path)
    analyzed = analyzer.process_documents()
    logger.info(f"Sentiment analysis: processed {analyzed} documents")


def run_scoring(config: dict, db_path: str):
    """Run velocity, divergence, and signal scoring"""
    logger.info("=== PHASE 3: Scoring ===")

    # Compute velocity and metrics for each window
    velocity_engine = VelocityEngine(db_path=db_path)
    for window in ["6h", "24h", "7d"]:
        logger.info(f"Computing metrics (window={window})...")
        computed = velocity_engine.compute_all_entities(window=window)
        logger.info(f"Computed metrics for {computed} entities (window={window})")

    # Update divergence scores
    logger.info("Computing divergence scores...")
    divergence_engine = DivergenceEngine(db_path=db_path)
    updated = divergence_engine.update_signals_with_divergence(window="24h")
    logger.info(f"Updated divergence for {updated} signals")

    # Score signals and assign labels
    logger.info("Scoring signals...")
    thresholds = config.get("thresholds", {})
    signal_model = SignalModel(thresholds=thresholds, db_path=db_path)

    for window in ["6h", "24h", "7d"]:
        scored = signal_model.score_all_signals(window=window)
        logger.info(f"Scored {scored} signals (window={window})")

    # Generate alerts
    logger.info("Generating alerts...")
    alerts = signal_model.generate_alerts(window="24h")
    logger.info(f"Generated {len(alerts)} alerts")

    # Clear price cache
    divergence_engine.clear_price_cache()


def main():
    parser = argparse.ArgumentParser(description="Rotation Radar Pipeline Runner")
    parser.add_argument("--collect-only", action="store_true", help="Only run data collection")
    parser.add_argument("--process-only", action="store_true", help="Only run processing")
    parser.add_argument("--score-only", action="store_true", help="Only run scoring")
    args = parser.parse_args()

    logger.info(f"=== Rotation Radar Pipeline - {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} ===")

    config = load_config()
    db_path = config.get("database", {}).get("path", "data/radar.db")

    # Ensure DB is initialized
    init_db(db_path)

    start = datetime.now(timezone.utc)

    if args.collect_only:
        run_collection(config, db_path)
    elif args.process_only:
        run_processing(config, db_path)
    elif args.score_only:
        run_scoring(config, db_path)
    else:
        # Full pipeline
        run_collection(config, db_path)
        run_processing(config, db_path)
        run_scoring(config, db_path)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(f"=== Pipeline complete in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
