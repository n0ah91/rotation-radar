"""
Backfill Script

Re-processes existing documents through the pipeline.
Useful after changing taxonomy, updating models, or fixing bugs.

Usage:
    python scripts/backfill.py                    # Re-process everything
    python scripts/backfill.py --entities-only     # Just re-extract entities
    python scripts/backfill.py --sentiment-only    # Just re-score sentiment
    python scripts/backfill.py --signals-only      # Just re-compute signals
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

from src.models.database import init_db, get_session, Document
from src.processing.entity_extraction import EntityExtractor
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
    config = {}
    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        config_path = project_root / "config" / "config.example.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    return config


def reset_processing_flags(db_path: str):
    """Reset processed flags on all documents"""
    session = get_session(db_path)
    try:
        session.query(Document).update({Document.processed: False})
        session.commit()
        count = session.query(Document).count()
        logger.info(f"Reset processing flags on {count} documents")
    finally:
        session.close()


def reset_sentiment(db_path: str):
    """Clear sentiment scores on all documents"""
    session = get_session(db_path)
    try:
        session.query(Document).update({
            Document.sentiment_score: None,
            Document.sentiment_label: None,
            Document.conviction_score: None,
        })
        session.commit()
        logger.info("Cleared sentiment scores")
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Rotation Radar Backfill")
    parser.add_argument("--entities-only", action="store_true")
    parser.add_argument("--sentiment-only", action="store_true")
    parser.add_argument("--signals-only", action="store_true")
    args = parser.parse_args()

    config = load_config()
    db_path = config.get("database", {}).get("path", "data/radar.db")
    init_db(db_path)

    logger.info("=== Rotation Radar Backfill ===")
    start = datetime.now(timezone.utc)

    taxonomy_path = project_root / "config" / "taxonomy.yaml"

    if args.entities_only or not (args.sentiment_only or args.signals_only):
        logger.info("Re-extracting entities...")
        reset_processing_flags(db_path)
        extractor = EntityExtractor(
            taxonomy_path=str(taxonomy_path),
            db_path=db_path,
        )
        count = extractor.process_unprocessed_documents(limit=10000)
        logger.info(f"Entity extraction: processed {count} documents")

    if args.sentiment_only or not (args.entities_only or args.signals_only):
        logger.info("Re-scoring sentiment...")
        reset_sentiment(db_path)
        analyzer = SentimentAnalyzer(db_path=db_path)
        count = analyzer.process_documents(limit=10000)
        logger.info(f"Sentiment analysis: processed {count} documents")

    if args.signals_only or not (args.entities_only or args.sentiment_only):
        logger.info("Re-computing signals...")
        velocity = VelocityEngine(db_path=db_path)
        divergence = DivergenceEngine(db_path=db_path)
        model = SignalModel(db_path=db_path)

        for window in ["6h", "24h", "7d"]:
            velocity.compute_all_entities(window=window)
            divergence.update_signals_with_divergence(window=window)
            model.score_all_signals(window=window)

        model.generate_alerts(window="24h")

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(f"=== Backfill complete in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
