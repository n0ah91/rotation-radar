"""
Backfill Script

Re-processes existing documents through the pipeline.
Useful after changing taxonomy, updating models, or fixing bugs.

Usage:
    python scripts/backfill.py                    # Re-process everything
    python scripts/backfill.py --entities-only     # Just re-extract entities
    python scripts/backfill.py --sentiment-only    # Just re-score sentiment
    python scripts/backfill.py --signals-only      # Just re-compute signals
    python scripts/backfill.py --authors-only      # Just backfill missing authors
    python scripts/backfill.py --snapshots-only    # Just generate daily snapshots
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

from src.models.database import (
    init_db,
    get_session,
    Document,
    Author,
    Source,
    SourceType,
)
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


def backfill_authors(db_path: str):
    """Backfill missing author records for documents without authors.

    Creates Author records using the source name as the author username,
    then links orphan documents to those authors.
    """
    session = get_session(db_path)
    try:
        # Find documents missing authors
        orphan_docs = (
            session.query(Document)
            .filter(Document.author_id == None)
            .all()
        )
        logger.info(f"Found {len(orphan_docs)} documents without authors")

        if not orphan_docs:
            return

        # Group by source_id for efficient author creation
        source_ids = set(d.source_id for d in orphan_docs if d.source_id)
        source_author_map = {}

        for source_id in source_ids:
            source = session.query(Source).filter(Source.id == source_id).first()
            if not source:
                continue

            author_name = source.name or source.identifier or f"Source_{source_id}"
            platform = source.source_type or SourceType.RSS

            # Get or create author
            author = (
                session.query(Author)
                .filter(Author.platform == platform, Author.username == author_name)
                .first()
            )
            if not author:
                author = Author(
                    platform=platform,
                    username=author_name,
                    display_name=author_name,
                )
                session.add(author)
                session.flush()  # Get the ID
                logger.info(f"Created author: {author_name} (platform={platform.value})")

            source_author_map[source_id] = author.id

        # Link documents to authors
        linked = 0
        for doc in orphan_docs:
            if doc.source_id in source_author_map:
                doc.author_id = source_author_map[doc.source_id]
                linked += 1

        session.commit()
        logger.info(f"Linked {linked} documents to authors")

    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Rotation Radar Backfill")
    parser.add_argument("--entities-only", action="store_true")
    parser.add_argument("--sentiment-only", action="store_true")
    parser.add_argument("--signals-only", action="store_true")
    parser.add_argument("--authors-only", action="store_true",
                        help="Backfill missing author records")
    parser.add_argument("--snapshots-only", action="store_true",
                        help="Generate daily snapshots from existing signals")
    args = parser.parse_args()

    config = load_config()
    db_path = config.get("database", {}).get("path", "data/radar.db")
    init_db(db_path)

    logger.info("=== Rotation Radar Backfill ===")
    start = datetime.now(timezone.utc)

    taxonomy_path = project_root / "config" / "taxonomy.yaml"

    # Determine what to run
    specific_mode = any([
        args.entities_only, args.sentiment_only, args.signals_only,
        args.authors_only, args.snapshots_only,
    ])

    if args.authors_only or not specific_mode:
        logger.info("Backfilling authors...")
        backfill_authors(db_path)

    if args.entities_only or not specific_mode:
        logger.info("Re-extracting entities...")
        reset_processing_flags(db_path)
        extractor = EntityExtractor(
            taxonomy_path=str(taxonomy_path),
            db_path=db_path,
        )
        count = extractor.process_unprocessed_documents(limit=10000)
        logger.info(f"Entity extraction: processed {count} documents")

    if args.sentiment_only or not specific_mode:
        logger.info("Re-scoring sentiment...")
        reset_sentiment(db_path)
        analyzer = SentimentAnalyzer(db_path=db_path)
        count = analyzer.process_documents(limit=10000)
        logger.info(f"Sentiment analysis: processed {count} documents")

    if args.signals_only or args.snapshots_only or not specific_mode:
        logger.info("Re-computing signals (includes snapshot generation)...")
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
