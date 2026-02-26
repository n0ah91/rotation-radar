"""
Rotation Radar Database Models

SQLAlchemy ORM models for storing:
- Sources (subreddits, YouTube channels, RSS feeds)
- Authors (Reddit users, channel owners)
- Documents (posts, videos, articles)
- Entities (tickers, themes, sub-themes)
- Signals (computed scores and phases)
- Narratives (clustered story threads)
- Alerts and Journal entries
"""

from datetime import datetime, timezone


def _utcnow():
    """Timezone-aware UTC timestamp (replaces deprecated _utcnow)"""
    return datetime.now(timezone.utc)
from typing import Optional
from pathlib import Path

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    Index,
    JSON,
    Enum,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
    Session,
)
import enum


Base = declarative_base()


# Enums for categorical fields
class SourceType(enum.Enum):
    REDDIT = "reddit"
    YOUTUBE = "youtube"
    RSS = "rss"
    TWITTER = "twitter"
    STOCKTWITS = "stocktwits"


class EntityType(enum.Enum):
    TICKER = "ticker"
    THEME = "theme"
    SUBTHEME = "subtheme"
    CATALYST = "catalyst"
    PERSON = "person"
    COMPANY = "company"


class Phase(enum.Enum):
    IGNITION = "ignition"
    ACCELERATION = "acceleration"
    CROWDED = "crowded"
    EXHAUSTION = "exhaustion"
    COOLING = "cooling"
    BASELINE = "baseline"


class DecisionLabel(enum.Enum):
    NOW = "now"
    BUILD = "build"
    WATCH = "watch"
    IGNORE = "ignore"


class AlertType(enum.Enum):
    ROTATION_IGNITION = "rotation_ignition"
    DIVERGENCE_EDGE = "divergence_edge"
    OVERHEAT_LATE = "overheat_late"


# ============================================================
# Core Models
# ============================================================


class Source(Base):
    """A data source (subreddit, YouTube channel, RSS feed, etc.)"""

    __tablename__ = "sources"

    id = Column(Integer, primary_key=True)
    source_type = Column(Enum(SourceType), nullable=False)
    identifier = Column(String(255), nullable=False)  # subreddit name, channel ID, feed URL
    name = Column(String(255))
    weight = Column(Float, default=1.0)  # Quality weight for scoring
    category = Column(String(100))
    description = Column(Text)
    alpha_score = Column(Float, default=0.0)  # Historical predictive accuracy
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationships
    documents = relationship("Document", back_populates="source")

    __table_args__ = (
        Index("ix_sources_type_identifier", "source_type", "identifier", unique=True),
    )


class Author(Base):
    """An author/user from any platform"""

    __tablename__ = "authors"

    id = Column(Integer, primary_key=True)
    platform = Column(Enum(SourceType), nullable=False)
    username = Column(String(255), nullable=False)
    display_name = Column(String(255))
    account_created_at = Column(DateTime)
    credibility_score = Column(Float, default=0.5)  # 0-1 scale
    is_verified = Column(Boolean, default=False)
    is_bot = Column(Boolean, default=False)
    karma_score = Column(Integer)  # Platform-specific reputation
    follower_count = Column(Integer)
    total_posts = Column(Integer, default=0)
    first_seen_at = Column(DateTime, default=_utcnow)
    last_seen_at = Column(DateTime, default=_utcnow)
    extra_data = Column(JSON)  # Platform-specific data

    # Relationships
    documents = relationship("Document", back_populates="author")

    __table_args__ = (
        Index("ix_authors_platform_username", "platform", "username", unique=True),
    )


class Document(Base):
    """A piece of content (Reddit post, YouTube video, RSS article)"""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=False)
    author_id = Column(Integer, ForeignKey("authors.id"))
    external_id = Column(String(255))  # Platform-specific ID
    url = Column(String(2048))
    title = Column(Text)
    content = Column(Text)
    content_type = Column(String(50))  # post, comment, video, article
    published_at = Column(DateTime)
    fetched_at = Column(DateTime, default=_utcnow)

    # Engagement metrics
    upvotes = Column(Integer, default=0)
    downvotes = Column(Integer, default=0)
    score = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    engagement_velocity = Column(Float)  # Rate of engagement growth

    # Sentiment (computed)
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # positive, negative, neutral
    conviction_score = Column(Float)  # Strength of conviction language

    # Processing status
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)

    # Relationships
    source = relationship("Source", back_populates="documents")
    author = relationship("Author", back_populates="documents")
    entities = relationship("DocumentEntity", back_populates="document")
    narratives = relationship("NarrativeDocument", back_populates="document")

    __table_args__ = (
        Index("ix_documents_source_external", "source_id", "external_id", unique=True),
        Index("ix_documents_published_at", "published_at"),
        Index("ix_documents_processed", "processed"),
    )


class Entity(Base):
    """An entity (ticker, theme, sub-theme) that we track"""

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True)
    entity_type = Column(Enum(EntityType), nullable=False)
    symbol = Column(String(50))  # Ticker symbol if applicable
    name = Column(String(255), nullable=False)
    parent_id = Column(Integer, ForeignKey("entities.id"))  # For hierarchy (theme -> subtheme)
    keywords = Column(JSON)  # Keywords for matching
    etfs = Column(JSON)  # Related ETFs
    description = Column(Text)
    created_at = Column(DateTime, default=_utcnow)

    # Relationships
    parent = relationship("Entity", remote_side=[id], backref="children")
    mentions = relationship("DocumentEntity", back_populates="entity")
    signals = relationship("Signal", back_populates="entity")

    __table_args__ = (
        Index("ix_entities_type_symbol", "entity_type", "symbol"),
        Index("ix_entities_name", "name"),
    )


class DocumentEntity(Base):
    """Many-to-many relationship between documents and entities they mention"""

    __tablename__ = "document_entities"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False)
    mention_count = Column(Integer, default=1)
    context_snippet = Column(Text)  # Snippet showing the mention
    is_primary = Column(Boolean, default=False)  # Is this the main topic?
    sentiment_toward = Column(Float)  # Sentiment specifically toward this entity

    # Relationships
    document = relationship("Document", back_populates="entities")
    entity = relationship("Entity", back_populates="mentions")

    __table_args__ = (
        Index("ix_doc_entities_doc_entity", "document_id", "entity_id", unique=True),
    )


class Signal(Base):
    """Computed signal for an entity at a point in time"""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey("entities.id"), nullable=False)
    computed_at = Column(DateTime, default=_utcnow)
    window = Column(String(10))  # 6h, 24h, 7d, 30d

    # Raw metrics
    mention_count = Column(Integer, default=0)
    unique_authors = Column(Integer, default=0)
    total_engagement = Column(Integer, default=0)
    platform_count = Column(Integer, default=0)

    # Velocity metrics
    velocity = Column(Float, default=0)
    acceleration = Column(Float, default=0)
    unique_author_velocity = Column(Float, default=0)

    # Z-scores (normalized)
    z_mentions = Column(Float, default=0)
    z_velocity = Column(Float, default=0)
    z_unique_authors = Column(Float, default=0)
    z_sentiment_delta = Column(Float, default=0)

    # Derived metrics
    concentration_top5 = Column(Float)  # % of mentions from top 5 authors
    source_diversity = Column(Float)  # Entropy across sources
    sentiment_mean = Column(Float)
    sentiment_delta = Column(Float)
    conviction_mean = Column(Float)

    # Cross-platform
    cross_platform_score = Column(Float, default=0)

    # Price divergence
    price_return = Column(Float)
    rs_vs_market = Column(Float)
    divergence_score = Column(Float, default=0)

    # Catalyst
    catalyst_score = Column(Float, default=0)
    catalyst_tags = Column(JSON)

    # Final scores
    heat_score = Column(Float, default=0)  # What's loud (0-100)
    edge_score = Column(Float, default=0)  # What's early (0-100)
    signal_score = Column(Float, default=0)  # Composite (0-100)

    # Phase and decision
    phase = Column(Enum(Phase), default=Phase.BASELINE)
    decision_label = Column(Enum(DecisionLabel), default=DecisionLabel.IGNORE)
    explanation = Column(Text)  # Human-readable explanation

    # Gates
    passed_breadth_gate = Column(Boolean, default=False)
    passed_quality_gate = Column(Boolean, default=False)
    passed_pump_filter = Column(Boolean, default=True)  # True = not a pump

    # Relationships
    entity = relationship("Entity", back_populates="signals")

    __table_args__ = (
        Index("ix_signals_entity_window", "entity_id", "window"),
        Index("ix_signals_computed_at", "computed_at"),
        Index("ix_signals_score", "signal_score"),
    )


class Narrative(Base):
    """A narrative/story thread grouping related discussions"""

    __tablename__ = "narratives"

    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    summary = Column(Text)
    key_claims = Column(JSON)  # List of bullet points
    catalyst_tags = Column(JSON)
    bear_case = Column(Text)  # Counter-arguments
    tripwires = Column(JSON)  # Invalidation conditions
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    is_active = Column(Boolean, default=True)

    # Metrics
    doc_count = Column(Integer, default=0)
    velocity = Column(Float, default=0)
    breadth_score = Column(Float, default=0)

    # Relationships
    documents = relationship("NarrativeDocument", back_populates="narrative")
    related_entities = Column(JSON)  # Entity IDs


class NarrativeDocument(Base):
    """Many-to-many relationship between narratives and documents"""

    __tablename__ = "narrative_documents"

    id = Column(Integer, primary_key=True)
    narrative_id = Column(Integer, ForeignKey("narratives.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    relevance_score = Column(Float, default=1.0)

    # Relationships
    narrative = relationship("Narrative", back_populates="documents")
    document = relationship("Document", back_populates="narratives")

    __table_args__ = (
        Index("ix_narrative_docs", "narrative_id", "document_id", unique=True),
    )


class Alert(Base):
    """System-generated alerts"""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    alert_type = Column(Enum(AlertType), nullable=False)
    entity_id = Column(Integer, ForeignKey("entities.id"))
    signal_id = Column(Integer, ForeignKey("signals.id"))
    title = Column(String(500), nullable=False)
    message = Column(Text)
    severity = Column(String(20))  # high, medium, low
    created_at = Column(DateTime, default=_utcnow)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    action_taken = Column(String(50))  # watch, build, now, dismiss

    __table_args__ = (
        Index("ix_alerts_created", "created_at"),
        Index("ix_alerts_acknowledged", "acknowledged"),
    )


class JournalEntry(Base):
    """User journal entries for tracking decisions"""

    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True)
    entity_id = Column(Integer, ForeignKey("entities.id"))
    signal_id = Column(Integer, ForeignKey("signals.id"))
    entry_type = Column(String(50))  # note, trade, thesis
    title = Column(String(500))
    content = Column(Text)
    tags = Column(JSON)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Trade tracking
    action = Column(String(20))  # buy, sell, watch
    ticker = Column(String(20))
    entry_price = Column(Float)
    exit_price = Column(Float)
    outcome = Column(String(20))  # win, loss, open


# ============================================================
# Database Connection Management
# ============================================================

_engine = None
_SessionLocal = None


def get_engine(db_path: str = "data/radar.db"):
    """Get or create the database engine"""
    global _engine
    if _engine is None:
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
    return _engine


def get_session(db_path: str = "data/radar.db") -> Session:
    """Get a new database session"""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(db_path)
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()


def init_db(db_path: str = "data/radar.db"):
    """Initialize the database (create all tables)"""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine
