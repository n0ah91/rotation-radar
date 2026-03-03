"""
Velocity & Metrics Computation Module

Computes rolling windowed metrics for entities:
- Mention velocity and acceleration
- Unique author counts and velocity
- Concentration metrics (top-k author share)
- Source diversity (entropy)
- Cross-platform confirmation
- Engagement velocity
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math
import logging

import numpy as np
import pandas as pd
from sqlalchemy import func, distinct

from ..models.database import (
    DailySnapshot,
    Document,
    DocumentEntity,
    Entity,
    Signal,
    Source,
    Author,
    SourceType,
    Phase,
    DecisionLabel,
    get_session,
)


logger = logging.getLogger(__name__)


# Rolling window definitions
WINDOWS = {
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}

# Baseline window for z-score computation
BASELINE_WINDOW = timedelta(days=30)


class VelocityEngine:
    """Computes velocity and windowed metrics for all entities"""

    def __init__(self, db_path: str = "data/radar.db"):
        self.db_path = db_path

    def compute_entity_metrics(
        self,
        entity_id: int,
        window: str = "24h",
        now: datetime = None,
    ) -> Dict:
        """
        Compute all metrics for a single entity over a time window.

        Args:
            entity_id: Entity to compute metrics for
            window: Time window key (6h, 24h, 7d, 30d)
            now: Reference time (default: utcnow)

        Returns:
            Dict with all computed metrics
        """
        if now is None:
            now = datetime.now(timezone.utc)

        window_delta = WINDOWS.get(window, WINDOWS["24h"])
        window_start = now - window_delta
        baseline_start = now - BASELINE_WINDOW

        session = get_session(self.db_path)

        try:
            # Get documents in current window
            window_docs = self._get_entity_documents(session, entity_id, window_start, now)

            # Get documents in previous window (for velocity)
            prev_window_start = window_start - window_delta
            prev_docs = self._get_entity_documents(session, entity_id, prev_window_start, window_start)

            # Get baseline documents (for z-scores)
            baseline_docs = self._get_entity_documents(session, entity_id, baseline_start, now)

            # Compute metrics
            metrics = {}

            # Volume
            metrics["mention_count"] = len(window_docs)
            prev_count = len(prev_docs)

            # Velocity (change rate)
            metrics["velocity"] = metrics["mention_count"] - prev_count

            # Acceleration (change of velocity) - need one more window back
            prev_prev_start = prev_window_start - window_delta
            prev_prev_docs = self._get_entity_documents(
                session, entity_id, prev_prev_start, prev_window_start
            )
            prev_velocity = prev_count - len(prev_prev_docs)
            metrics["acceleration"] = metrics["velocity"] - prev_velocity

            # Unique authors
            window_authors = set(d.author_id for d in window_docs if d.author_id)
            prev_authors = set(d.author_id for d in prev_docs if d.author_id)
            metrics["unique_authors"] = len(window_authors)
            metrics["unique_author_velocity"] = len(window_authors) - len(prev_authors)

            # Concentration: top-5 author share
            metrics["concentration_top5"] = self._compute_concentration(window_docs, top_k=5)

            # Source diversity (entropy across sources)
            metrics["source_diversity"] = self._compute_source_diversity(window_docs)

            # Platform count (cross-platform) and source names
            metrics["platform_count"] = self._count_platforms(session, window_docs)
            metrics["source_names"] = self._get_source_names(session, window_docs)

            # Engagement totals
            metrics["total_engagement"] = sum(
                (d.score or 0) + (d.comment_count or 0) + (d.view_count or 0)
                for d in window_docs
            )

            # Sentiment stats
            sentiments = [d.sentiment_score for d in window_docs if d.sentiment_score is not None]
            prev_sentiments = [d.sentiment_score for d in prev_docs if d.sentiment_score is not None]

            if sentiments:
                metrics["sentiment_mean"] = float(np.mean(sentiments))
            else:
                metrics["sentiment_mean"] = 0.0

            if sentiments and prev_sentiments:
                metrics["sentiment_delta"] = float(np.mean(sentiments)) - float(np.mean(prev_sentiments))
            else:
                metrics["sentiment_delta"] = 0.0

            # Conviction
            convictions = [d.conviction_score for d in window_docs if d.conviction_score is not None]
            metrics["conviction_mean"] = float(np.mean(convictions)) if convictions else 0.0

            # Z-scores (computed against baseline)
            baseline_mention_counts = self._get_daily_counts(baseline_docs)
            metrics["z_mentions"] = self._compute_z_score(
                metrics["mention_count"], baseline_mention_counts
            )

            baseline_velocities = self._get_daily_velocities(session, entity_id, baseline_start, now)
            metrics["z_velocity"] = self._compute_z_score(
                metrics["velocity"], baseline_velocities
            )

            baseline_author_counts = self._get_daily_unique_authors(baseline_docs)
            metrics["z_unique_authors"] = self._compute_z_score(
                metrics["unique_authors"], baseline_author_counts
            )

            baseline_sentiment_deltas = self._get_daily_sentiment_deltas(baseline_docs)
            metrics["z_sentiment_delta"] = self._compute_z_score(
                metrics["sentiment_delta"], baseline_sentiment_deltas
            )

            # Cross-platform score
            metrics["cross_platform_score"] = self._compute_cross_platform_score(
                session, window_docs
            )

            return metrics

        finally:
            session.close()

    def _get_entity_documents(
        self,
        session,
        entity_id: int,
        start: datetime,
        end: datetime
    ) -> List[Document]:
        """Get documents that mention an entity in a time window"""
        return (
            session.query(Document)
            .join(DocumentEntity)
            .filter(
                DocumentEntity.entity_id == entity_id,
                Document.published_at >= start,
                Document.published_at < end,
            )
            .all()
        )

    def _compute_concentration(self, docs: List[Document], top_k: int = 5) -> float:
        """Compute concentration ratio (top-k author share of total mentions)"""
        if not docs:
            return 0.0

        author_counts = defaultdict(int)
        for doc in docs:
            if doc.author_id:
                author_counts[doc.author_id] += 1

        if not author_counts:
            return 0.0

        sorted_counts = sorted(author_counts.values(), reverse=True)
        top_k_total = sum(sorted_counts[:top_k])

        return top_k_total / len(docs)

    def _compute_source_diversity(self, docs: List[Document]) -> float:
        """Compute Shannon entropy of source distribution"""
        if not docs:
            return 0.0

        source_counts = defaultdict(int)
        for doc in docs:
            source_counts[doc.source_id] += 1

        total = len(docs)
        entropy = 0.0

        for count in source_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _count_platforms(self, session, docs: List[Document]) -> int:
        """Count unique sources (not source types) represented in documents.

        Each distinct source (SeekingAlpha, Yahoo Finance, etc.) counts as
        its own platform, not grouped by source_type.
        """
        if not docs:
            return 0
        return len(set(d.source_id for d in docs if d.source_id))

    def _get_source_names(self, session, docs: List[Document]) -> List[str]:
        """Get list of source names covering these documents."""
        if not docs:
            return []
        source_ids = set(d.source_id for d in docs if d.source_id)
        if not source_ids:
            return []
        sources = session.query(Source).filter(Source.id.in_(source_ids)).all()
        return sorted(set(s.name for s in sources if s.name))

    def _compute_cross_platform_score(
        self,
        session,
        docs: List[Document]
    ) -> float:
        """
        Compute cross-source confirmation score.

        Each unique source (feed/subreddit/channel) is scored individually
        by its quality weight. Having coverage from multiple distinct sources
        increases the score.
        """
        if not docs:
            return 0.0

        source_ids = set(d.source_id for d in docs if d.source_id)
        if not source_ids:
            return 0.0
        sources = session.query(Source).filter(Source.id.in_(source_ids)).all()

        if len(sources) <= 1:
            return 0.0

        # Each source contributes its weight
        total_weight = sum(s.weight or 1.0 for s in sources)
        return min(1.0, total_weight / 3.0)

    def _compute_z_score(self, value: float, baseline_values: List[float]) -> float:
        """Compute z-score of value against baseline"""
        if not baseline_values or len(baseline_values) < 3:
            return 0.0

        mean = np.mean(baseline_values)
        std = np.std(baseline_values)

        if std == 0:
            return 0.0

        return (value - mean) / std

    def _get_daily_counts(self, docs: List[Document]) -> List[float]:
        """Get daily mention counts from a list of documents"""
        if not docs:
            return [0.0]

        daily = defaultdict(int)
        for doc in docs:
            if doc.published_at:
                day = doc.published_at.date()
                daily[day] += 1

        return list(daily.values()) if daily else [0.0]

    def _get_daily_unique_authors(self, docs: List[Document]) -> List[float]:
        """Get daily unique author counts"""
        if not docs:
            return [0.0]

        daily = defaultdict(set)
        for doc in docs:
            if doc.published_at and doc.author_id:
                day = doc.published_at.date()
                daily[day].add(doc.author_id)

        return [float(len(authors)) for authors in daily.values()] if daily else [0.0]

    def _get_daily_velocities(
        self,
        session,
        entity_id: int,
        start: datetime,
        end: datetime
    ) -> List[float]:
        """Compute daily velocities over a period"""
        # Get daily mention counts
        all_docs = self._get_entity_documents(session, entity_id, start, end)
        daily_counts = defaultdict(int)

        for doc in all_docs:
            if doc.published_at:
                day = doc.published_at.date()
                daily_counts[day] += 1

        # Compute day-over-day changes
        if not daily_counts:
            return [0.0]

        dates = sorted(daily_counts.keys())
        velocities = []

        for i in range(1, len(dates)):
            velocity = daily_counts[dates[i]] - daily_counts[dates[i - 1]]
            velocities.append(float(velocity))

        return velocities if velocities else [0.0]

    def _get_daily_sentiment_deltas(self, docs: List[Document]) -> List[float]:
        """Compute daily sentiment deltas"""
        if not docs:
            return [0.0]

        daily_sentiments = defaultdict(list)
        for doc in docs:
            if doc.published_at and doc.sentiment_score is not None:
                day = doc.published_at.date()
                daily_sentiments[day].append(doc.sentiment_score)

        dates = sorted(daily_sentiments.keys())
        if len(dates) < 2:
            return [0.0]

        deltas = []
        for i in range(1, len(dates)):
            prev_mean = np.mean(daily_sentiments[dates[i - 1]])
            curr_mean = np.mean(daily_sentiments[dates[i]])
            deltas.append(float(curr_mean - prev_mean))

        return deltas if deltas else [0.0]

    def compute_all_entities(self, window: str = "24h") -> int:
        """
        Compute metrics for all active entities.

        Returns:
            Number of signals computed
        """
        session = get_session(self.db_path)
        computed = 0

        try:
            # Get all entities that have at least one mention
            entity_ids = (
                session.query(distinct(DocumentEntity.entity_id))
                .all()
            )
            entity_ids = [eid[0] for eid in entity_ids]

            logger.info(f"Computing metrics for {len(entity_ids)} entities (window={window})")

            for entity_id in entity_ids:
                try:
                    metrics = self.compute_entity_metrics(entity_id, window=window)

                    # Save as Signal record
                    signal = Signal(
                        entity_id=entity_id,
                        computed_at=datetime.now(timezone.utc),
                        window=window,
                        mention_count=metrics["mention_count"],
                        unique_authors=metrics["unique_authors"],
                        total_engagement=metrics["total_engagement"],
                        platform_count=metrics["platform_count"],
                        velocity=metrics["velocity"],
                        acceleration=metrics["acceleration"],
                        unique_author_velocity=metrics["unique_author_velocity"],
                        z_mentions=metrics["z_mentions"],
                        z_velocity=metrics["z_velocity"],
                        z_unique_authors=metrics["z_unique_authors"],
                        z_sentiment_delta=metrics["z_sentiment_delta"],
                        concentration_top5=metrics["concentration_top5"],
                        source_diversity=metrics["source_diversity"],
                        sentiment_mean=metrics["sentiment_mean"],
                        sentiment_delta=metrics["sentiment_delta"],
                        conviction_mean=metrics["conviction_mean"],
                        cross_platform_score=metrics["cross_platform_score"],
                    )
                    session.add(signal)

                    # Upsert daily snapshot for trend tracking
                    today = datetime.now(timezone.utc).date()
                    source_names = metrics.get("source_names", [])

                    # Compute momentum % (Δ vs 30d baseline)
                    baseline_daily_avg = 0.0
                    baseline_counts = self._get_daily_counts(
                        self._get_entity_documents(
                            session, entity_id,
                            datetime.now(timezone.utc) - timedelta(days=30),
                            datetime.now(timezone.utc),
                        )
                    )
                    if baseline_counts and len(baseline_counts) >= 3:
                        baseline_daily_avg = float(np.mean(baseline_counts))
                    window_delta_days = WINDOWS.get(window, timedelta(days=7)).days or 1
                    expected = baseline_daily_avg * max(window_delta_days, 1)
                    if expected > 0:
                        momentum_pct = ((metrics["mention_count"] - expected) / expected) * 100
                    else:
                        momentum_pct = 0.0 if metrics["mention_count"] == 0 else 100.0

                    # Compute confidence score
                    hours_since_latest = 48.0  # default to stale
                    if metrics["mention_count"] > 0:
                        window_delta = WINDOWS.get(window, WINDOWS["24h"])
                        window_start = datetime.now(timezone.utc) - window_delta
                        recent_docs = self._get_entity_documents(
                            session, entity_id, window_start, datetime.now(timezone.utc)
                        )
                        if recent_docs:
                            latest_pub = max(
                                (d.published_at for d in recent_docs if d.published_at),
                                default=None,
                            )
                            if latest_pub:
                                if latest_pub.tzinfo is None:
                                    latest_pub = latest_pub.replace(tzinfo=timezone.utc)
                                age = datetime.now(timezone.utc) - latest_pub
                                hours_since_latest = age.total_seconds() / 3600

                    volume_f = min(1.0, metrics["mention_count"] / 10)
                    breadth_f = min(1.0, metrics["platform_count"] / 5)
                    freshness_f = max(0.0, 1.0 - hours_since_latest / 48)
                    confidence = round(0.40 * volume_f + 0.35 * breadth_f + 0.25 * freshness_f, 2)

                    # Upsert snapshot
                    existing = (
                        session.query(DailySnapshot)
                        .filter_by(entity_id=entity_id, date=today, window=window)
                        .first()
                    )
                    if existing:
                        existing.mentions = metrics["mention_count"]
                        existing.momentum_pct = round(momentum_pct, 1)
                        existing.acceleration = metrics["acceleration"]
                        existing.source_count = metrics["platform_count"]
                        existing.source_names = source_names
                        existing.sentiment_mean = metrics["sentiment_mean"]
                        existing.confidence = confidence
                    else:
                        snap = DailySnapshot(
                            entity_id=entity_id,
                            date=today,
                            window=window,
                            mentions=metrics["mention_count"],
                            momentum_pct=round(momentum_pct, 1),
                            acceleration=metrics["acceleration"],
                            source_count=metrics["platform_count"],
                            source_names=source_names,
                            sentiment_mean=metrics["sentiment_mean"],
                            confidence=confidence,
                        )
                        session.add(snap)

                    computed += 1

                except Exception as e:
                    logger.warning(f"Error computing metrics for entity {entity_id}: {e}")
                    continue

            session.commit()
            logger.info(f"Computed metrics for {computed} entities")

        finally:
            session.close()

        return computed
