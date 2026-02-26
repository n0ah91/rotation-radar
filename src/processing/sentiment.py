"""
Sentiment Analysis Module

Analyzes sentiment in documents using FinBERT or lightweight alternatives.
Focuses on sentiment CHANGE rather than raw polarity.
"""

from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timezone

import numpy as np

from ..models.database import Document, get_session


logger = logging.getLogger(__name__)


# Try to import torch + transformers for FinBERT
HAS_TORCH = False
HAS_TRANSFORMERS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    logger.info("PyTorch not installed. Using rule-based sentiment only.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    logger.info("transformers not installed. Using rule-based sentiment only.")


class SentimentAnalyzer:
    """
    Analyzes sentiment in financial text.

    Uses FinBERT for accurate financial sentiment when available,
    falls back to rule-based analysis otherwise.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        use_gpu: bool = True,
        batch_size: int = 32,
        db_path: str = "data/radar.db",
    ):
        self.db_path = db_path
        self.batch_size = batch_size
        self.model_name = model_name

        # Model components
        self.tokenizer = None
        self.model = None
        self.device = None

        # Initialize model if available
        if HAS_TORCH and HAS_TRANSFORMERS:
            self._init_model(use_gpu)

        # Conviction/action keywords for rule-based backup
        self.conviction_keywords = {
            "strong_bullish": [
                "buying", "loading", "adding", "long", "all in", "moon",
                "rocket", "diamond hands", "hold", "bullish af", "gonna rip",
                "breakout", "accumulating", "bottom is in"
            ],
            "bullish": [
                "bullish", "buy", "calls", "upside", "growth", "opportunity",
                "undervalued", "catalyst", "potential", "promising"
            ],
            "bearish": [
                "bearish", "sell", "puts", "short", "downside", "overvalued",
                "dump", "crash", "avoid", "warning", "risk"
            ],
            "strong_bearish": [
                "selling everything", "get out", "puts printing", "going to zero",
                "ponzi", "scam", "fraud", "collapse", "disaster"
            ],
            "uncertainty": [
                "maybe", "might", "could", "uncertain", "watching", "wait",
                "not sure", "on the fence", "risky"
            ],
        }

    def _init_model(self, use_gpu: bool):
        """Initialize FinBERT model"""
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            logger.info("Skipping model init (torch/transformers not available)")
            return

        try:
            logger.info(f"Loading sentiment model: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Set device
            if use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using GPU for sentiment analysis")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for sentiment analysis")

            self.model.to(self.device)
            self.model.eval()

            logger.info("Sentiment model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            self.model = None

    def _rule_based_sentiment(self, text: str) -> Tuple[float, str, float]:
        """
        Rule-based sentiment analysis fallback.

        Returns:
            Tuple of (score, label, conviction)
        """
        text_lower = text.lower()

        scores = {
            "strong_bullish": 0,
            "bullish": 0,
            "bearish": 0,
            "strong_bearish": 0,
            "uncertainty": 0,
        }

        for category, keywords in self.conviction_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[category] += 1

        # Calculate weighted score
        bullish = scores["bullish"] + scores["strong_bullish"] * 2
        bearish = scores["bearish"] + scores["strong_bearish"] * 2

        total = bullish + bearish + scores["uncertainty"]

        if total == 0:
            return 0.0, "neutral", 0.0

        # Score from -1 to 1
        score = (bullish - bearish) / total

        # Conviction based on total mentions and strong keywords
        conviction = min(1.0, (scores["strong_bullish"] + scores["strong_bearish"]) / 5)

        # Label
        if score > 0.3:
            label = "positive"
        elif score < -0.3:
            label = "negative"
        else:
            label = "neutral"

        return score, label, conviction

    def analyze_text(self, text: str, max_length: int = 512) -> Dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze
            max_length: Maximum token length

        Returns:
            Dict with score, label, and conviction
        """
        if not text or not text.strip():
            return {
                "score": 0.0,
                "label": "neutral",
                "conviction": 0.0,
                "probabilities": {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            }

        # Use FinBERT if available
        if self.model and self.tokenizer:
            try:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                probs = probs.cpu().numpy()[0]

                # FinBERT labels: positive, negative, neutral
                labels = ["positive", "negative", "neutral"]
                label_idx = np.argmax(probs)
                label = labels[label_idx]

                # Convert to -1 to 1 score
                score = probs[0] - probs[1]  # positive - negative

                # Conviction based on confidence
                conviction = max(probs) - 0.33  # Above random

                return {
                    "score": float(score),
                    "label": label,
                    "conviction": float(conviction),
                    "probabilities": {
                        "positive": float(probs[0]),
                        "negative": float(probs[1]),
                        "neutral": float(probs[2]),
                    }
                }

            except Exception as e:
                logger.warning(f"FinBERT analysis failed, using fallback: {e}")

        # Fallback to rule-based
        score, label, conviction = self._rule_based_sentiment(text)
        return {
            "score": score,
            "label": label,
            "conviction": conviction,
            "probabilities": None
        }

    def analyze_batch(self, texts: List[str], max_length: int = 512) -> List[Dict]:
        """
        Analyze sentiment of multiple texts in batch.

        Args:
            texts: List of texts to analyze
            max_length: Maximum token length

        Returns:
            List of sentiment result dicts
        """
        if not texts:
            return []

        results = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            if self.model and self.tokenizer:
                try:
                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                        padding=True,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Get predictions
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=-1)

                    probs = probs.cpu().numpy()

                    for j, p in enumerate(probs):
                        labels = ["positive", "negative", "neutral"]
                        label_idx = np.argmax(p)

                        results.append({
                            "score": float(p[0] - p[1]),
                            "label": labels[label_idx],
                            "conviction": float(max(p) - 0.33),
                            "probabilities": {
                                "positive": float(p[0]),
                                "negative": float(p[1]),
                                "neutral": float(p[2]),
                            }
                        })

                    continue

                except Exception as e:
                    logger.warning(f"Batch analysis failed: {e}")

            # Fallback for this batch
            for text in batch:
                score, label, conviction = self._rule_based_sentiment(text)
                results.append({
                    "score": score,
                    "label": label,
                    "conviction": conviction,
                    "probabilities": None
                })

        return results

    def process_documents(self, limit: int = 1000) -> int:
        """
        Process documents that don't have sentiment scores yet.

        Returns:
            Number of documents processed
        """
        session = get_session(self.db_path)
        processed = 0

        try:
            # Get documents without sentiment scores
            documents = session.query(Document).filter(
                Document.sentiment_score == None
            ).limit(limit).all()

            if not documents:
                logger.info("No documents to process for sentiment")
                return 0

            # Extract texts
            texts = [f"{d.title or ''} {d.content or ''}" for d in documents]

            # Batch analyze
            results = self.analyze_batch(texts)

            # Update documents
            for doc, result in zip(documents, results):
                doc.sentiment_score = result["score"]
                doc.sentiment_label = result["label"]
                doc.conviction_score = result["conviction"]
                processed += 1

            session.commit()
            logger.info(f"Processed sentiment for {processed} documents")

        finally:
            session.close()

        return processed

    def get_sentiment_stats(
        self,
        entity_id: int = None,
        window_hours: int = 24
    ) -> Dict:
        """
        Get aggregate sentiment statistics.

        Args:
            entity_id: Optional entity to filter by
            window_hours: Time window in hours

        Returns:
            Dict with sentiment statistics
        """
        from datetime import timedelta

        session = get_session(self.db_path)

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)

            query = session.query(Document).filter(
                Document.published_at >= cutoff,
                Document.sentiment_score != None
            )

            if entity_id:
                from ..models.database import DocumentEntity
                query = query.join(DocumentEntity).filter(
                    DocumentEntity.entity_id == entity_id
                )

            documents = query.all()

            if not documents:
                return {
                    "count": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "positive_pct": 0.0,
                    "negative_pct": 0.0,
                    "neutral_pct": 0.0,
                    "avg_conviction": 0.0,
                }

            scores = [d.sentiment_score for d in documents]
            convictions = [d.conviction_score or 0 for d in documents]
            labels = [d.sentiment_label for d in documents]

            return {
                "count": len(documents),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "positive_pct": labels.count("positive") / len(labels),
                "negative_pct": labels.count("negative") / len(labels),
                "neutral_pct": labels.count("neutral") / len(labels),
                "avg_conviction": float(np.mean(convictions)),
            }

        finally:
            session.close()
