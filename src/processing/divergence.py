"""
Chatter-Price Divergence Module

Detects divergences between social chatter and price action.
High chatter + flat/falling price = potential early edge.
Low chatter + rising price = potential late entry risk.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

import numpy as np

from ..models.database import (
    Entity,
    EntityType,
    Signal,
    get_session,
)


logger = logging.getLogger(__name__)

# Import yfinance for price data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinance not installed. Price divergence will be disabled.")


class DivergenceEngine:
    """Computes chatter-price divergence scores"""

    def __init__(self, db_path: str = "data/radar.db"):
        self.db_path = db_path
        self._price_cache: Dict[str, Dict] = {}

    def get_price_data(
        self,
        ticker: str,
        period: str = "3mo",
    ) -> Optional[Dict]:
        """
        Fetch price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: Data period (1mo, 3mo, 6mo, 1y)

        Returns:
            Dict with price data or None
        """
        if not HAS_YFINANCE:
            return None

        cache_key = f"{ticker}_{period}"
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                return None

            data = {
                "ticker": ticker,
                "prices": hist["Close"].to_dict(),
                "returns": hist["Close"].pct_change().to_dict(),
                "volume": hist["Volume"].to_dict(),
                "current_price": float(hist["Close"].iloc[-1]),
                "period_return": float(
                    (hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]
                ),
            }

            self._price_cache[cache_key] = data
            return data

        except Exception as e:
            logger.warning(f"Error fetching price data for {ticker}: {e}")
            return None

    def compute_relative_strength(
        self,
        ticker: str,
        benchmark: str = "SPY",
        window_days: int = 30,
    ) -> Optional[float]:
        """
        Compute relative strength vs benchmark.

        Returns:
            RS ratio (>1 = outperforming, <1 = underperforming)
        """
        ticker_data = self.get_price_data(ticker)
        benchmark_data = self.get_price_data(benchmark)

        if not ticker_data or not benchmark_data:
            return None

        try:
            ticker_return = ticker_data["period_return"]
            benchmark_return = benchmark_data["period_return"]

            if benchmark_return == 0:
                return 1.0

            return (1 + ticker_return) / (1 + benchmark_return)

        except Exception as e:
            logger.warning(f"Error computing RS for {ticker}: {e}")
            return None

    def compute_divergence_score(
        self,
        entity_id: int,
        ticker: str,
        z_velocity: float,
        benchmark: str = "SPY",
    ) -> Dict:
        """
        Compute chatter-price divergence.

        High chatter velocity (z > 1) + flat/down RS = bullish divergence (early edge)
        Low chatter velocity (z < -1) + rising price = bearish divergence (late)

        Args:
            entity_id: Entity ID
            ticker: Stock ticker
            z_velocity: Z-score of chatter velocity
            benchmark: Benchmark ticker for RS calculation

        Returns:
            Dict with divergence metrics
        """
        result = {
            "divergence_score": 0.0,
            "price_return": None,
            "rs_vs_market": None,
            "divergence_type": "none",
            "description": "",
        }

        if not HAS_YFINANCE:
            return result

        price_data = self.get_price_data(ticker)
        if not price_data:
            return result

        result["price_return"] = price_data["period_return"]

        # Compute relative strength
        rs = self.compute_relative_strength(ticker, benchmark)
        if rs is not None:
            result["rs_vs_market"] = rs

        # Divergence logic
        # Bullish divergence: chatter rising, price flat/down
        if z_velocity > 1.0:
            if price_data["period_return"] <= 0.02:  # Flat or down
                result["divergence_score"] = min(1.0, z_velocity * 0.3)
                result["divergence_type"] = "bullish"
                result["description"] = (
                    f"Chatter rising (z={z_velocity:.1f}) but price flat/down "
                    f"({price_data['period_return']:.1%}). Potential early edge."
                )
            elif rs is not None and rs < 0.95:  # Underperforming market
                result["divergence_score"] = min(0.8, z_velocity * 0.2)
                result["divergence_type"] = "bullish_rs"
                result["description"] = (
                    f"Chatter rising (z={z_velocity:.1f}) but underperforming "
                    f"market (RS={rs:.2f}). Possible rotation setup."
                )

        # Bearish divergence: chatter falling, price rising
        elif z_velocity < -1.0:
            if price_data["period_return"] > 0.05:
                result["divergence_score"] = min(1.0, abs(z_velocity) * 0.3) * -1
                result["divergence_type"] = "bearish"
                result["description"] = (
                    f"Chatter declining (z={z_velocity:.1f}) but price up "
                    f"({price_data['period_return']:.1%}). Narrative fading."
                )

        # Overheat: both chatter and price extended
        elif z_velocity > 2.0 and price_data["period_return"] > 0.15:
            result["divergence_score"] = -0.5  # Negative = crowded
            result["divergence_type"] = "overheat"
            result["description"] = (
                f"Both chatter (z={z_velocity:.1f}) and price "
                f"({price_data['period_return']:.1%}) extended. Risk of reversal."
            )

        return result

    def update_signals_with_divergence(self, window: str = "24h") -> int:
        """
        Update existing signals with divergence data.

        Returns:
            Number of signals updated
        """
        session = get_session(self.db_path)
        updated = 0

        try:
            # Get recent signals for ticker entities
            signals = (
                session.query(Signal)
                .join(Entity)
                .filter(
                    Signal.window == window,
                    Entity.entity_type == EntityType.TICKER,
                    Entity.symbol != None,
                )
                .order_by(Signal.computed_at.desc())
                .all()
            )

            seen_entities = set()
            for signal in signals:
                # Only process the latest signal per entity
                if signal.entity_id in seen_entities:
                    continue
                seen_entities.add(signal.entity_id)

                entity = session.query(Entity).get(signal.entity_id)
                if not entity or not entity.symbol:
                    continue

                div = self.compute_divergence_score(
                    entity_id=entity.id,
                    ticker=entity.symbol,
                    z_velocity=signal.z_velocity or 0,
                )

                signal.divergence_score = div["divergence_score"]
                signal.price_return = div["price_return"]
                signal.rs_vs_market = div["rs_vs_market"]
                updated += 1

            session.commit()
            logger.info(f"Updated divergence for {updated} signals")

        finally:
            session.close()

        return updated

    def clear_price_cache(self):
        """Clear the price data cache"""
        self._price_cache.clear()
