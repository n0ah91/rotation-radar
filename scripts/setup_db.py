"""
Database Setup Script

Initializes the database, creates all tables, and loads entities from taxonomy.
Run this once before starting data collection.

Usage:
    python scripts/setup_db.py
"""

import sys
from pathlib import Path
import logging
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.database import init_db, get_session
from src.processing.entity_extraction import EntityExtractor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config():
    """Load config files"""
    config = {}

    config_path = project_root / "config" / "config.yaml"
    if not config_path.exists():
        config_path = project_root / "config" / "config.example.yaml"

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # Load sources config
    sources_path = project_root / "config" / "sources.yaml"
    if sources_path.exists():
        with open(sources_path) as f:
            config["sources"] = yaml.safe_load(f) or {}

    return config


def main():
    config = load_config()
    db_path = config.get("database", {}).get("path", "data/radar.db")

    logger.info("=== Rotation Radar Database Setup ===")

    # 1. Create database and tables
    logger.info(f"Creating database at: {db_path}")
    init_db(db_path)
    logger.info("Database tables created successfully")

    # 2. Initialize entities from taxonomy
    logger.info("Loading entities from taxonomy...")
    taxonomy_path = project_root / "config" / "taxonomy.yaml"

    extractor = EntityExtractor(
        taxonomy_path=str(taxonomy_path),
        db_path=db_path,
    )

    # Count entities
    session = get_session(db_path)
    from src.models.database import Entity
    entity_count = session.query(Entity).count()
    session.close()

    logger.info(f"Loaded {entity_count} entities from taxonomy")
    logger.info(f"Tracking {len(extractor.ticker_set)} tickers")

    # 3. Summary
    logger.info("")
    logger.info("=== Setup Complete ===")
    logger.info(f"Database: {db_path}")
    logger.info(f"Entities: {entity_count}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Copy config/config.example.yaml to config/config.yaml")
    logger.info("  2. Add your Reddit API credentials")
    logger.info("  3. Run: python scripts/run_pipeline.py")
    logger.info("  4. Launch dashboard: streamlit run src/ui/app.py")


if __name__ == "__main__":
    main()
