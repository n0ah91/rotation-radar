"""
Signal Model

The core scoring engine that computes:
- Signal Score (0-100 composite)
- Heat Score (what's loud)
- Edge Score (what's early)
- Phase classification (Ignition → Acceleration → Crowded → Exhaustion → Cooling)
- Decision labels (NOW / BUILD / WATCH / IGNORE)
- Quality gates (Breadth, Quality, Pump Filter)
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from ..models.database import (
    Signal,
    Entity,
    EntityType,
    Phase,
    DecisionLabel,
    AlertType,
    Alert,
    get_session,
)


logger = logging.getLogger(__name__)


# Default scoring weights (tunable via config)
DEFAULT_WEIGHTS = {
    "velocity_z": 0.30,
    "unique_authors_z": 0.20,
    "sentiment_delta_z": 0.15,
    "cross_platform": 0.15,
    "divergence": 0.10,
    "catalyst": 0.10,
}

# Decision thresholds
DEFAULT_THRESHOLDS = {
    "now_score": 80,
    "build_score": 65,
    "watch_score": 50,
    # Gates
    "breadth_min_unique_authors_z": 0.0,
    "breadth_min_cross_platform": 0.5,
    "quality_min_weighted_score": 0.6,
    "pump_max_concentration": 0.70,
    "pump_max_account_age_days": 30,
}


class SignalModel:
    """
    Core signal scoring and classification engine.

    Produces decision-grade signals from raw metrics.
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        thresholds: Dict[str, float] = None,
        db_path: str = "data/radar.db",
    ):
        self.db_path = db_path
        self.weights = weights or DEFAULT_WEIGHTS
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def compute_signal_score(self, signal: Signal) -> float:
        """
        Compute the composite signal score (0-100).

        Formula:
            signal_raw = 0.30 * z_velocity
                       + 0.20 * z_unique_authors
                       + 0.15 * z_sentiment_delta
                       + 0.15 * cross_platform_score
                       + 0.10 * divergence_score
                       + 0.10 * catalyst_score

            signal_score = clamp(round(50 + 20 * signal_raw), 0, 100)
        """
        raw = (
            self.weights["velocity_z"] * (signal.z_velocity or 0) +
            self.weights["unique_authors_z"] * (signal.z_unique_authors or 0) +
            self.weights["sentiment_delta_z"] * (signal.z_sentiment_delta or 0) +
            self.weights["cross_platform"] * (signal.cross_platform_score or 0) +
            self.weights["divergence"] * (signal.divergence_score or 0) +
            self.weights["catalyst"] * (signal.catalyst_score or 0)
        )

        score = round(50 + 20 * raw)
        return max(0, min(100, score))

    def compute_heat_score(self, signal: Signal) -> float:
        """
        Heat Score: What's loud NOW.
        50% mention count + 30% sentiment + 20% engagement
        """
        # Normalize mention count to 0-100 using z-score
        mention_component = max(0, min(100, 50 + 20 * (signal.z_mentions or 0)))

        # Sentiment (shifted to 0-100 scale)
        sentiment_component = max(0, min(100, 50 + 50 * (signal.sentiment_mean or 0)))

        # Engagement (log-scaled)
        engagement = signal.total_engagement or 0
        engagement_component = min(100, np.log1p(engagement) * 10)

        heat = (
            0.50 * mention_component +
            0.30 * sentiment_component +
            0.20 * engagement_component
        )

        return round(max(0, min(100, heat)), 1)

    def compute_edge_score(self, signal: Signal) -> float:
        """
        Edge Score: What's EARLY.
        35% unique author velocity + 25% divergence + 20% cross-platform accel + 20% breadth
        """
        # Unique author velocity (z-scored)
        author_vel_component = max(0, min(100, 50 + 20 * (signal.z_unique_authors or 0)))

        # Divergence (bullish divergence = high edge)
        divergence = signal.divergence_score or 0
        divergence_component = max(0, min(100, 50 + 50 * divergence))

        # Cross-platform confirmation
        cross_platform_component = (signal.cross_platform_score or 0) * 100

        # Source diversity (breadth)
        diversity = signal.source_diversity or 0
        breadth_component = min(100, diversity * 50)  # Entropy scaled

        edge = (
            0.35 * author_vel_component +
            0.25 * divergence_component +
            0.20 * cross_platform_component +
            0.20 * breadth_component
        )

        return round(max(0, min(100, edge)), 1)

    def check_breadth_gate(self, signal: Signal) -> bool:
        """
        Breadth Gate: Ensure signal has sufficient breadth.
        Pass if: unique_authors_z > 0 OR cross_platform_score >= threshold
        """
        return (
            (signal.z_unique_authors or 0) > self.thresholds["breadth_min_unique_authors_z"] or
            (signal.cross_platform_score or 0) >= self.thresholds["breadth_min_cross_platform"]
        )

    def check_quality_gate(self, signal: Signal) -> bool:
        """
        Quality Gate: Ensure signal comes from quality sources.
        """
        # Source-weighted score threshold
        source_diversity = signal.source_diversity or 0
        return source_diversity >= self.thresholds.get("quality_min_diversity", 0.3)

    def check_pump_filter(self, signal: Signal) -> bool:
        """
        Pump/Bot Filter: Detect potential pump-and-dump or bot activity.
        Flags if concentration > threshold AND other suspicious signals.

        Returns:
            True if signal passes (NOT a pump), False if flagged
        """
        concentration = signal.concentration_top5 or 0
        conviction = signal.conviction_mean or 0

        # High concentration is suspicious
        if concentration > self.thresholds["pump_max_concentration"]:
            # Check for euphoric language + weak breadth
            if conviction > 0.7 and (signal.z_unique_authors or 0) < 0:
                return False  # Likely pump

            # Very high concentration alone is a flag
            if concentration > 0.85:
                return False

        return True

    def classify_phase(self, signal: Signal) -> Phase:
        """
        Classify the current phase of an entity's signal lifecycle.

        Phases:
        - IGNITION: Score rising fast, breadth+confirmation, price NOT confirming
        - ACCELERATION: Score rising + RS turning up
        - CROWDED: Concentration high, euphoric, price extended
        - EXHAUSTION: Score decelerating, sentiment still high, price stalls
        - COOLING: Sentiment delta negative, velocity mean-reverting
        - BASELINE: No significant signal
        """
        velocity = signal.velocity or 0
        acceleration = signal.acceleration or 0
        z_vel = signal.z_velocity or 0
        z_authors = signal.z_unique_authors or 0
        concentration = signal.concentration_top5 or 0
        sentiment_delta = signal.sentiment_delta or 0
        divergence = signal.divergence_score or 0

        # No signal
        if abs(z_vel) < 0.5 and abs(z_authors) < 0.5:
            return Phase.BASELINE

        # COOLING: Everything declining
        if z_vel < -0.5 and sentiment_delta < -0.1:
            return Phase.COOLING

        # EXHAUSTION: High but decelerating
        if z_vel > 0.5 and acceleration < 0 and concentration > 0.5:
            return Phase.EXHAUSTION

        # CROWDED: High concentration + euphoria + extended
        if concentration > 0.6 and z_vel > 1.0 and divergence < -0.2:
            return Phase.CROWDED

        # ACCELERATION: Broadening + price confirming
        if z_vel > 1.0 and z_authors > 0.5 and divergence <= 0:
            return Phase.ACCELERATION

        # IGNITION: Rising fast, price NOT yet confirming
        if z_vel > 0.5 and z_authors > 0 and divergence > 0:
            return Phase.IGNITION

        # Default based on velocity
        if z_vel > 1.0:
            return Phase.ACCELERATION
        elif z_vel > 0.3:
            return Phase.IGNITION
        elif z_vel < -0.5:
            return Phase.COOLING

        return Phase.BASELINE

    def assign_decision_label(
        self,
        signal_score: float,
        phase: Phase,
        passed_breadth: bool,
        passed_pump: bool,
    ) -> DecisionLabel:
        """
        Assign a decision label based on score, phase, and gates.

        Rules:
        - NOW: score >= 80, phase in {Ignition, Acceleration}, passes gates
        - BUILD: score 65-79, phase Ignition/Acceleration
        - WATCH: score 50-64 or uncertain phase
        - IGNORE: score < 50 or Crowded/Exhaustion with weak edge
        """
        # Failed pump filter caps label
        if not passed_pump:
            return DecisionLabel.WATCH

        # Failed breadth gate limits to WATCH
        if not passed_breadth and signal_score >= 65:
            signal_score = min(signal_score, 64)  # Cap to WATCH territory

        # Phase-based exclusions
        crowded_phases = {Phase.CROWDED, Phase.EXHAUSTION}

        if phase in crowded_phases:
            # Crowded/Exhaustion can only be WATCH or IGNORE
            if signal_score >= 50:
                return DecisionLabel.WATCH
            return DecisionLabel.IGNORE

        # Score-based labeling
        if signal_score >= self.thresholds["now_score"]:
            if phase in {Phase.IGNITION, Phase.ACCELERATION}:
                return DecisionLabel.NOW
            return DecisionLabel.BUILD  # High score but wrong phase

        if signal_score >= self.thresholds["build_score"]:
            return DecisionLabel.BUILD

        if signal_score >= self.thresholds["watch_score"]:
            return DecisionLabel.WATCH

        return DecisionLabel.IGNORE

    def generate_explanation(self, signal: Signal) -> str:
        """Generate a human-readable explanation of the signal"""
        parts = []

        # Velocity
        if (signal.z_velocity or 0) > 1.0:
            parts.append(f"Chatter velocity elevated (z={signal.z_velocity:.1f})")
        elif (signal.z_velocity or 0) < -1.0:
            parts.append(f"Chatter velocity declining (z={signal.z_velocity:.1f})")

        # Breadth
        if (signal.z_unique_authors or 0) > 0.5:
            parts.append(f"Broadening participation ({signal.unique_authors} unique authors)")

        # Cross-platform
        if (signal.platform_count or 0) > 1:
            parts.append(f"Confirmed across {signal.platform_count} platforms")

        # Sentiment
        if (signal.sentiment_delta or 0) > 0.1:
            parts.append("Sentiment improving")
        elif (signal.sentiment_delta or 0) < -0.1:
            parts.append("Sentiment deteriorating")

        # Divergence
        if (signal.divergence_score or 0) > 0.3:
            parts.append("Bullish chatter-price divergence (potential early edge)")
        elif (signal.divergence_score or 0) < -0.3:
            parts.append("Price extended beyond chatter (late risk)")

        # Concentration warning
        if (signal.concentration_top5 or 0) > 0.6:
            parts.append(f"High author concentration ({signal.concentration_top5:.0%})")

        return ". ".join(parts) + "." if parts else "Baseline activity."

    def score_signal(self, signal: Signal) -> Signal:
        """
        Score a single signal with all composite metrics.

        Updates the signal object in-place and returns it.
        """
        # Compute scores
        signal.signal_score = self.compute_signal_score(signal)
        signal.heat_score = self.compute_heat_score(signal)
        signal.edge_score = self.compute_edge_score(signal)

        # Check gates
        signal.passed_breadth_gate = self.check_breadth_gate(signal)
        signal.passed_quality_gate = self.check_quality_gate(signal)
        signal.passed_pump_filter = self.check_pump_filter(signal)

        # Apply gate penalties
        if not signal.passed_breadth_gate:
            signal.signal_score = min(signal.signal_score, 30)

        if not signal.passed_pump_filter:
            signal.signal_score = min(signal.signal_score, 50)

        if not signal.passed_quality_gate:
            signal.signal_score = int(signal.signal_score * 0.7)

        # Classify phase
        signal.phase = self.classify_phase(signal)

        # Assign decision label
        signal.decision_label = self.assign_decision_label(
            signal.signal_score,
            signal.phase,
            signal.passed_breadth_gate,
            signal.passed_pump_filter,
        )

        # Generate explanation
        signal.explanation = self.generate_explanation(signal)

        return signal

    def score_all_signals(self, window: str = "24h") -> int:
        """
        Score all unscored signals.

        Returns:
            Number of signals scored
        """
        session = get_session(self.db_path)
        scored = 0

        try:
            # Get signals without scores
            signals = (
                session.query(Signal)
                .filter(
                    Signal.window == window,
                    Signal.signal_score == 0,
                )
                .all()
            )

            for signal in signals:
                self.score_signal(signal)
                scored += 1

            session.commit()
            logger.info(f"Scored {scored} signals")

        finally:
            session.close()

        return scored

    def generate_alerts(self, window: str = "24h") -> List[Alert]:
        """
        Generate alerts for notable signals.

        Alert Types:
        - ROTATION_IGNITION: New entity enters Ignition phase with high edge
        - DIVERGENCE_EDGE: Strong chatter-price divergence detected
        - OVERHEAT_LATE: Entity entering Crowded/Exhaustion with high heat

        Returns:
            List of generated alerts
        """
        session = get_session(self.db_path)
        alerts = []

        try:
            signals = (
                session.query(Signal)
                .filter(Signal.window == window)
                .order_by(Signal.computed_at.desc())
                .all()
            )

            # Deduplicate to latest per entity
            seen = set()
            latest_signals = []
            for s in signals:
                if s.entity_id not in seen:
                    seen.add(s.entity_id)
                    latest_signals.append(s)

            for signal in latest_signals:
                entity = session.query(Entity).get(signal.entity_id)
                name = entity.name if entity else f"Entity#{signal.entity_id}"

                # Rotation Ignition alert
                if (
                    signal.phase == Phase.IGNITION and
                    signal.edge_score >= 70 and
                    signal.decision_label in {DecisionLabel.NOW, DecisionLabel.BUILD}
                ):
                    alert = Alert(
                        alert_type=AlertType.ROTATION_IGNITION,
                        entity_id=signal.entity_id,
                        signal_id=signal.id,
                        title=f"Rotation Ignition: {name}",
                        message=(
                            f"{name} entering Ignition phase. "
                            f"Edge={signal.edge_score}, Signal={signal.signal_score}. "
                            f"{signal.explanation}"
                        ),
                        severity="high" if signal.signal_score >= 80 else "medium",
                    )
                    session.add(alert)
                    alerts.append(alert)

                # Divergence Edge alert
                if (
                    (signal.divergence_score or 0) > 0.5 and
                    signal.signal_score >= 60
                ):
                    alert = Alert(
                        alert_type=AlertType.DIVERGENCE_EDGE,
                        entity_id=signal.entity_id,
                        signal_id=signal.id,
                        title=f"Divergence Edge: {name}",
                        message=(
                            f"{name} showing chatter-price divergence. "
                            f"Divergence={signal.divergence_score:.2f}. "
                            f"{signal.explanation}"
                        ),
                        severity="medium",
                    )
                    session.add(alert)
                    alerts.append(alert)

                # Overheat alert
                if (
                    signal.phase in {Phase.CROWDED, Phase.EXHAUSTION} and
                    signal.heat_score >= 70
                ):
                    alert = Alert(
                        alert_type=AlertType.OVERHEAT_LATE,
                        entity_id=signal.entity_id,
                        signal_id=signal.id,
                        title=f"Overheat Warning: {name}",
                        message=(
                            f"{name} in {signal.phase.value} phase. "
                            f"Heat={signal.heat_score}, Concentration={signal.concentration_top5:.0%}. "
                            f"Consider taking profits or tightening stops."
                        ),
                        severity="high" if signal.phase == Phase.EXHAUSTION else "medium",
                    )
                    session.add(alert)
                    alerts.append(alert)

            session.commit()
            logger.info(f"Generated {len(alerts)} alerts")

        finally:
            session.close()

        return alerts

    def get_ranked_signals(
        self,
        window: str = "24h",
        sort_by: str = "signal_score",
        limit: int = 25,
        entity_type: EntityType = None,
    ) -> List[Signal]:
        """
        Get ranked signals for the dashboard.

        Args:
            window: Time window
            sort_by: Field to sort by (signal_score, heat_score, edge_score)
            limit: Maximum results
            entity_type: Filter by entity type

        Returns:
            List of Signal objects, sorted by score
        """
        session = get_session(self.db_path)

        try:
            query = session.query(Signal).filter(Signal.window == window)

            if entity_type:
                query = query.join(Entity).filter(Entity.entity_type == entity_type)

            # Sort
            sort_field = getattr(Signal, sort_by, Signal.signal_score)
            query = query.order_by(sort_field.desc())

            # Deduplicate to latest per entity
            all_signals = query.all()
            seen = set()
            result = []
            for s in all_signals:
                if s.entity_id not in seen:
                    seen.add(s.entity_id)
                    result.append(s)
                    if len(result) >= limit:
                        break

            return result

        finally:
            session.close()
