"""
Entity Extraction Module

Extracts tickers, themes, and other entities from document text.
Uses dictionary matching + regex + optional spaCy NER.
"""

import re
from typing import List, Dict, Set, Tuple, Optional, Any
import logging
from collections import defaultdict

import yaml

from ..models.database import (
    Document,
    Entity,
    DocumentEntity,
    EntityType,
    get_session,
)


logger = logging.getLogger(__name__)


# Try to import spaCy for advanced NER
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logger.info("spaCy not installed. Using dictionary-based extraction only.")


class EntityExtractor:
    """Extracts entities (tickers, themes, catalysts) from text"""

    def __init__(
        self,
        taxonomy_path: str = "config/taxonomy.yaml",
        db_path: str = "data/radar.db",
        use_spacy: bool = True,
    ):
        self.db_path = db_path
        self.taxonomy_path = taxonomy_path

        # Load taxonomy
        self.taxonomy = self._load_taxonomy()

        # Build lookup dictionaries
        self.ticker_set: Set[str] = set()
        self.keyword_to_entity: Dict[str, List[int]] = defaultdict(list)
        self.ticker_to_entity: Dict[str, int] = {}

        # Initialize entities in database and build lookups
        self._init_entities()

        # spaCy NLP model (optional)
        self.nlp = None
        if use_spacy and HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy NLP model")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )

        # Regex patterns
        self._ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
        self._cashtag_pattern = re.compile(r'(?<!\w)\$([A-Z]{1,5})(?!\w)')

    def _load_taxonomy(self) -> Dict:
        """Load taxonomy configuration"""
        try:
            with open(self.taxonomy_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Taxonomy file not found: {self.taxonomy_path}")
            return {}

    def _init_entities(self):
        """Initialize entities in database from taxonomy"""
        session = get_session(self.db_path)

        try:
            macro_themes = self.taxonomy.get("macro_themes", {})

            for macro_name, macro_data in macro_themes.items():
                # Create macro theme entity
                macro_entity = self._get_or_create_entity(
                    session,
                    name=macro_name,
                    entity_type=EntityType.THEME,
                    description=macro_data.get("description", ""),
                )

                # Process themes under this macro
                for key, value in macro_data.items():
                    if key in ["description", "keywords", "catalyst_tags", "risk_tags"]:
                        continue

                    if isinstance(value, dict):
                        # This is a theme
                        theme_entity = self._get_or_create_entity(
                            session,
                            name=key,
                            entity_type=EntityType.THEME,
                            parent_id=macro_entity.id,
                            description=value.get("description", ""),
                            keywords=value.get("keywords", []),
                        )

                        # Index theme keywords
                        for kw in value.get("keywords", []):
                            self.keyword_to_entity[kw.lower()].append(theme_entity.id)

                        # Process sub-themes
                        for subkey, subvalue in value.items():
                            if subkey in ["description", "keywords", "catalyst_tags", "risk_tags"]:
                                continue

                            if isinstance(subvalue, dict):
                                # This is a sub-theme
                                subtheme_entity = self._get_or_create_entity(
                                    session,
                                    name=subkey,
                                    entity_type=EntityType.SUBTHEME,
                                    parent_id=theme_entity.id,
                                    keywords=subvalue.get("keywords", []),
                                    etfs=subvalue.get("etfs", []),
                                )

                                # Index sub-theme keywords
                                for kw in subvalue.get("keywords", []):
                                    self.keyword_to_entity[kw.lower()].append(subtheme_entity.id)

                                # Create ticker entities
                                for ticker in subvalue.get("tickers", []):
                                    ticker_entity = self._get_or_create_entity(
                                        session,
                                        name=ticker,
                                        symbol=ticker,
                                        entity_type=EntityType.TICKER,
                                        parent_id=subtheme_entity.id,
                                    )
                                    self.ticker_set.add(ticker.upper())
                                    self.ticker_to_entity[ticker.upper()] = ticker_entity.id

                                # Also track ETFs
                                for etf in subvalue.get("etfs", []):
                                    etf_entity = self._get_or_create_entity(
                                        session,
                                        name=etf,
                                        symbol=etf,
                                        entity_type=EntityType.TICKER,
                                        parent_id=subtheme_entity.id,
                                    )
                                    self.ticker_set.add(etf.upper())
                                    self.ticker_to_entity[etf.upper()] = etf_entity.id

            session.commit()
            logger.info(
                f"Initialized {len(self.ticker_set)} tickers and "
                f"{len(self.keyword_to_entity)} keyword mappings"
            )

        finally:
            session.close()

    def _get_or_create_entity(
        self,
        session,
        name: str,
        entity_type: EntityType,
        symbol: str = None,
        parent_id: int = None,
        keywords: List[str] = None,
        etfs: List[str] = None,
        description: str = "",
    ) -> Entity:
        """Get or create an entity in the database"""
        query = session.query(Entity).filter(
            Entity.name == name,
            Entity.entity_type == entity_type,
        )

        if parent_id:
            query = query.filter(Entity.parent_id == parent_id)

        entity = query.first()

        if entity is None:
            entity = Entity(
                name=name,
                symbol=symbol,
                entity_type=entity_type,
                parent_id=parent_id,
                keywords=keywords,
                etfs=etfs,
                description=description,
            )
            session.add(entity)
            session.flush()  # Get the ID

        return entity

    # Common English words that look like tickers - exclude from bare word matching
    _COMMON_WORDS = frozenset({
        "A", "I", "S", "U", "AN", "AM", "AS", "AT", "BE", "BY", "DO",
        "GO", "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF",
        "OK", "ON", "OR", "SO", "TO", "UP", "US", "WE", "ALL", "AND",
        "ARE", "BIG", "BUT", "CAN", "DAY", "DID", "FOR", "GET", "GOT",
        "HAS", "HAD", "HER", "HIM", "HIS", "HOW", "ITS", "LET", "MAY",
        "NEW", "NOT", "NOW", "OLD", "ONE", "OUR", "OUT", "OWN", "PUT",
        "RUN", "SAY", "SHE", "THE", "TOP", "TRY", "TWO", "USE", "WAR",
        "WAS", "WAY", "WHO", "WHY", "WIN", "YOU", "ALSO", "BACK",
        "BEEN", "BEST", "BOTH", "COME", "EACH", "EVEN", "FIND", "FIVE",
        "FOUR", "FROM", "GAVE", "GOOD", "GREW", "HALF", "HAVE", "HERE",
        "HIGH", "HOME", "INTO", "JUST", "KEEP", "KNOW", "LAST", "LIKE",
        "LINE", "LONG", "LOOK", "MADE", "MAKE", "MANY", "MORE", "MOST",
        "MUCH", "MUST", "NAME", "NEAR", "NEXT", "ONLY", "OPEN", "OVER",
        "PART", "PAST", "PLAN", "REAL", "SAID", "SAME", "SHOW", "SIDE",
        "SOME", "SUCH", "SURE", "TAKE", "TELL", "THAN", "THAT", "THEM",
        "THEN", "THEY", "THIS", "TOOK", "TURN", "VERY", "WANT", "WELL",
        "WENT", "WERE", "WHAT", "WHEN", "WILL", "WITH", "WORK", "YEAR",
        "YOUR", "ZERO", "RATE", "CALL", "GAIN", "LOSS", "FLAT", "FUND",
        "CASH", "BANK", "GROW", "HOLD", "RISK", "COST", "DEBT", "DEAL",
        "MOVE", "SELL", "RISE", "DROP", "FELL", "SELL", "PUSH", "PULL",
        "BEAR", "BULL", "POST", "SEES", "SAYS", "DATA", "FREE", "FULL",
    })

    def extract_tickers(self, text: str) -> List[Tuple[str, int]]:
        """
        Extract ticker symbols from text.

        Returns:
            List of (ticker, count) tuples
        """
        tickers = defaultdict(int)

        # Match $TICKER cashtag patterns (these are explicit, trust them even for short tickers)
        for match in self._cashtag_pattern.finditer(text):
            ticker = match.group(1).upper()
            if ticker in self.ticker_set:
                tickers[ticker] += 1

        # Match standalone tickers from our known set (bare uppercase words)
        # Only match tickers with 2+ chars, or 1-char if not a common word
        words = re.findall(r'\b([A-Z]{1,5})\b', text.upper() if text else "")
        for word in words:
            if word in self.ticker_set and word not in self._COMMON_WORDS:
                # Extra validation: skip very short tickers unless they appear near
                # financial context (stock, share, ticker, $, NYSE, NASDAQ)
                if len(word) <= 2:
                    # Check for nearby financial context (within 100 chars)
                    idx = text.upper().find(word)
                    if idx >= 0:
                        context_window = text[max(0, idx - 80):idx + len(word) + 80].lower()
                        financial_hints = ["stock", "share", "ticker", "$", "nyse",
                                           "nasdaq", "etf", "buy", "sell", "bullish",
                                           "bearish", "position", "trade"]
                        if any(hint in context_window for hint in financial_hints):
                            tickers[word] += 1
                else:
                    tickers[word] += 1

        return list(tickers.items())

    def extract_keywords(self, text: str) -> List[Tuple[int, int]]:
        """
        Extract theme/subtheme keywords from text.

        Returns:
            List of (entity_id, count) tuples
        """
        entity_counts = defaultdict(int)
        text_lower = text.lower()

        for keyword, entity_ids in self.keyword_to_entity.items():
            keyword_lower = keyword.lower()
            # For short keywords (<=3 chars), require word boundaries to avoid false positives
            if len(keyword_lower) <= 3:
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                matches = re.findall(pattern, text_lower)
                count = len(matches)
            else:
                count = text_lower.count(keyword_lower)

            if count > 0:
                for entity_id in entity_ids:
                    entity_counts[entity_id] += count

        return list(entity_counts.items())

    def extract_catalysts(self, text: str) -> List[str]:
        """
        Extract catalyst mentions from text.

        Returns:
            List of catalyst type strings
        """
        catalysts = []
        text_lower = text.lower()

        catalyst_types = self.taxonomy.get("catalyst_types", [])

        for catalyst_def in catalyst_types:
            if isinstance(catalyst_def, dict):
                for cat_type, keywords in catalyst_def.items():
                    for kw in keywords:
                        if kw.lower() in text_lower:
                            catalysts.append(cat_type)
                            break

        return list(set(catalysts))

    def extract_sentiment_keywords(self, text: str) -> Dict[str, int]:
        """
        Extract sentiment-indicating keywords.

        Returns:
            Dict with counts for bullish, bearish, euphoria, etc.
        """
        counts = {
            "bullish": 0,
            "bearish": 0,
            "euphoria": 0,
            "uncertainty": 0,
        }

        text_lower = text.lower()
        global_keywords = self.taxonomy.get("global_keywords", {})

        for sentiment_type, keywords in global_keywords.items():
            for kw in keywords:
                count = text_lower.count(kw.lower())
                if sentiment_type in counts:
                    counts[sentiment_type] += count

        return counts

    def extract_entities_spacy(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities using spaCy.

        Returns:
            List of (entity_text, entity_label) tuples
        """
        if not self.nlp:
            return []

        doc = self.nlp(text[:100000])  # Limit text length

        entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE", "MONEY", "PERCENT"]:
                entities.append((ent.text, ent.label_))

        return entities

    def process_document(self, document: Document) -> List[DocumentEntity]:
        """
        Process a document and extract all entities.

        Args:
            document: Document to process

        Returns:
            List of DocumentEntity objects
        """
        doc_entities = []
        text = f"{document.title or ''} {document.content or ''}"

        if not text.strip():
            return doc_entities

        session = get_session(self.db_path)

        try:
            # Extract tickers
            tickers = self.extract_tickers(text)
            for ticker, count in tickers:
                entity_id = self.ticker_to_entity.get(ticker)
                if entity_id:
                    # Check if already exists
                    existing = session.query(DocumentEntity).filter(
                        DocumentEntity.document_id == document.id,
                        DocumentEntity.entity_id == entity_id
                    ).first()

                    if not existing:
                        doc_entity = DocumentEntity(
                            document_id=document.id,
                            entity_id=entity_id,
                            mention_count=count,
                            is_primary=(count == max(c for _, c in tickers)) if tickers else False,
                        )
                        doc_entities.append(doc_entity)
                        session.add(doc_entity)

            # Extract theme/subtheme keywords
            keywords = self.extract_keywords(text)
            for entity_id, count in keywords:
                existing = session.query(DocumentEntity).filter(
                    DocumentEntity.document_id == document.id,
                    DocumentEntity.entity_id == entity_id
                ).first()

                if not existing:
                    doc_entity = DocumentEntity(
                        document_id=document.id,
                        entity_id=entity_id,
                        mention_count=count,
                    )
                    doc_entities.append(doc_entity)
                    session.add(doc_entity)

            session.commit()

            # Mark document as processed
            doc = session.query(Document).get(document.id)
            if doc:
                doc.processed = True
                session.commit()

        finally:
            session.close()

        return doc_entities

    def process_unprocessed_documents(self, limit: int = 1000) -> int:
        """
        Process all unprocessed documents.

        Returns:
            Number of documents processed
        """
        session = get_session(self.db_path)
        processed_count = 0

        try:
            documents = session.query(Document).filter(
                Document.processed == False
            ).limit(limit).all()

            for doc in documents:
                self.process_document(doc)
                processed_count += 1

                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} documents...")

            logger.info(f"Entity extraction complete. Processed {processed_count} documents.")

        finally:
            session.close()

        return processed_count
