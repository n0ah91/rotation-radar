"""
Rotation Radar Dashboard

Main Streamlit application with 5 screens:
1. Home Radar - Transparent metrics, top movers, data health
2. Theme Drilldown - Metrics + Sub-themes + Narratives
3. Ticker Page - Price + Chatter divergence + Evidence
4. Source Control - Source management + Alpha tracking
5. Alerts & Journal - Alert inbox + Decision log
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.database import (
    init_db,
    get_session,
    Signal,
    Entity,
    EntityType,
    Document,
    DocumentEntity,
    Source,
    Author,
    Alert,
    JournalEntry,
    DailySnapshot,
    Phase,
    DecisionLabel,
    AlertType,
)


# Page config
st.set_page_config(
    page_title="Rotation Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .stMetric label {
        font-size: 0.75rem !important;
        color: #6c757d !important;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def load_config():
    """Load application config"""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        config_path = Path("config/config.example.yaml")

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def run_pipeline_from_ui():
    """Run the data pipeline from the UI with progress feedback"""
    project_root = Path(__file__).parent.parent.parent

    with st.spinner("Running pipeline..."):
        progress = st.sidebar.progress(0, text="Initializing...")

        try:
            # Load config
            config = load_config()
            sources_path = project_root / "config" / "sources.yaml"
            if sources_path.exists():
                with open(sources_path) as f:
                    config["sources"] = yaml.safe_load(f) or {}

            db_path = config.get("database", {}).get("path", "data/radar.db")

            # Step 1: Collect RSS
            progress.progress(10, text="Collecting RSS feeds...")
            from src.collectors.rss_collector import RSSCollector
            rss = RSSCollector(config, db_path)
            rss_count = rss.run(limit=50)

            # Step 2: Entity extraction
            progress.progress(30, text="Extracting entities...")
            from src.processing.entity_extraction import EntityExtractor
            taxonomy_path = project_root / "config" / "taxonomy.yaml"
            extractor = EntityExtractor(
                taxonomy_path=str(taxonomy_path),
                db_path=db_path,
            )
            extractor.process_unprocessed_documents()

            # Step 3: Sentiment
            progress.progress(50, text="Analyzing sentiment...")
            from src.processing.sentiment import SentimentAnalyzer
            analyzer = SentimentAnalyzer(db_path=db_path)
            analyzer.process_documents()

            # Step 4: Velocity + Divergence + Scoring
            progress.progress(70, text="Computing signals...")
            from src.processing.velocity import VelocityEngine
            from src.processing.divergence import DivergenceEngine
            from src.processing.signal_model import SignalModel

            velocity_engine = VelocityEngine(db_path=db_path)
            for window in ["6h", "24h", "7d"]:
                velocity_engine.compute_all_entities(window=window)

            divergence_engine = DivergenceEngine(db_path=db_path)
            divergence_engine.update_signals_with_divergence(window="24h")

            thresholds = config.get("thresholds", {})
            signal_model = SignalModel(thresholds=thresholds, db_path=db_path)
            for window in ["6h", "24h", "7d"]:
                signal_model.score_all_signals(window=window)

            signal_model.generate_alerts(window="24h")
            divergence_engine.clear_price_cache()

            progress.progress(100, text="Done!")
            st.sidebar.success(f"Pipeline complete! {rss_count} new articles collected.")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Pipeline error: {e}")
            logging.exception("Pipeline failed")


# ============================================================
# Sidebar
# ============================================================

def render_sidebar():
    """Render the sidebar with global controls"""
    st.sidebar.title("📡 Rotation Radar")

    # Pipeline control — at the top so it's always visible
    session_check = get_session()
    doc_count = session_check.query(Document).count()
    signal_count = session_check.query(Signal).count()
    entity_count = session_check.query(Entity).count()
    source_count = session_check.query(Source).filter(Source.enabled == True).count()
    author_count = session_check.query(Author).count()
    session_check.close()

    if doc_count == 0:
        st.sidebar.warning("⚠️ No data yet — click below to populate.")
    if st.sidebar.button("🔄 Run Pipeline", type="primary", use_container_width=True):
        run_pipeline_from_ui()
    st.sidebar.caption(
        f"📊 {doc_count} docs · {entity_count} entities · "
        f"{source_count} sources · {author_count} authors"
    )

    st.sidebar.markdown("---")

    # Time window selector
    window = st.sidebar.selectbox(
        "Time Window",
        ["6h", "24h", "7d", "30d"],
        index=2,  # Default to 7d
    )

    # Entity type filter
    entity_filter = st.sidebar.selectbox(
        "View",
        ["All", "Themes", "Sub-themes", "Tickers"],
    )

    entity_type_map = {
        "All": None,
        "Themes": EntityType.THEME,
        "Sub-themes": EntityType.SUBTHEME,
        "Tickers": EntityType.TICKER,
    }

    # Top N
    top_n = st.sidebar.slider("Top N Items", 5, 50, 25)

    return {
        "window": window,
        "entity_type": entity_type_map[entity_filter],
        "top_n": top_n,
    }


# ============================================================
# Helper functions
# ============================================================

def get_confidence_badge(confidence: float) -> str:
    """Return colored confidence label"""
    if confidence >= 0.7:
        return "🟢 High"
    elif confidence >= 0.4:
        return "🟡 Moderate"
    else:
        return "🔴 Low"


def get_sentiment_badge(sentiment: float) -> str:
    """Return sentiment indicator"""
    if sentiment > 0.15:
        return "↑ Bullish"
    elif sentiment < -0.15:
        return "↓ Bearish"
    else:
        return "→ Neutral"


def format_momentum(pct: float) -> str:
    """Format momentum percentage"""
    if pct > 0:
        return f"+{pct:.0f}%"
    else:
        return f"{pct:.0f}%"


# ============================================================
# Screen 1: Home Radar (REBUILT)
# ============================================================

def render_home_radar(params: dict):
    """Render the Home Radar screen with transparent, audit-ready metrics"""

    session = get_session()

    try:
        # ── Data Health Banner ──────────────────────────────────
        doc_count = session.query(Document).count()
        active_sources = (
            session.query(Source)
            .filter(Source.enabled == True)
            .all()
        )
        source_names_list = sorted(set(s.name for s in active_sources if s.name))
        source_count = len(source_names_list)
        author_count = session.query(Author).count()

        # Latest document age
        latest_doc = (
            session.query(Document)
            .order_by(Document.published_at.desc())
            .first()
        )
        if latest_doc and latest_doc.published_at:
            pub_time = latest_doc.published_at
            if pub_time.tzinfo is None:
                pub_time = pub_time.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - pub_time
            hours_ago = age.total_seconds() / 3600
            if hours_ago < 1:
                age_str = f"{int(age.total_seconds() / 60)}m ago"
            elif hours_ago < 24:
                age_str = f"{int(hours_ago)}h ago"
            else:
                age_str = f"{int(hours_ago / 24)}d ago"
        else:
            age_str = "No data"
            hours_ago = 999

        # Data health banner
        health_col1, health_col2 = st.columns([3, 1])
        with health_col1:
            st.markdown(
                f"📊 **Coverage:** {doc_count} articles from {source_count} sources "
                f"({', '.join(source_names_list[:6])}"
                f"{'...' if source_count > 6 else ''})"
            )
            st.caption(
                f"⏱️ Last updated: {age_str} &nbsp;|&nbsp; "
                f"👤 {author_count} authors tracked &nbsp;|&nbsp; "
                f"🪟 Window: {params['window']}"
            )
        with health_col2:
            pass  # Pipeline button is in sidebar

        # Warnings
        missing_sources = []
        if not any("reddit" in s.name.lower() for s in active_sources):
            missing_sources.append("Reddit")
        if not any("twitter" in s.name.lower() or "x.com" in (s.identifier or "") for s in active_sources):
            missing_sources.append("X/Twitter")
        if not any("youtube" in s.name.lower() for s in active_sources):
            missing_sources.append("YouTube")

        if missing_sources:
            st.info(
                f"📡 **Limited data** — {', '.join(missing_sources)} not connected. "
                f"Confidence scores may be lower than expected."
            )

        st.markdown("---")

        # ── How This Works (collapsed) ──────────────────────────
        with st.expander("ℹ️ How This Works — Column Definitions & Methodology"):
            st.markdown("""
**What does this dashboard show?**

Rotation Radar collects articles from financial news sources (RSS feeds), extracts mentions of tickers, themes, and sectors, then computes velocity and momentum metrics to identify what's heating up or cooling down.

---

**Column Definitions:**

| Column | What It Means | Formula |
|--------|--------------|---------|
| **7d Mentions** | Total times this topic appeared across all sources in the last 7 days | `COUNT(documents mentioning entity WHERE published_at > now - 7d)` |
| **Δ vs 30d** | Momentum — % change vs the 30-day daily average. +100% = double normal volume | `(mentions - avg_daily_30d × window_days) / (avg_daily_30d × window_days) × 100` |
| **Acceleration** | Is momentum speeding up or slowing down? Positive = gaining steam | `velocity_current - velocity_previous` (2nd derivative of mentions) |
| **Sources** | Which data sources mentioned this topic, by name | List of distinct source feeds |
| **Source Count** | Number of distinct sources covering this topic. More = higher confidence | `COUNT(DISTINCT source_id)` |
| **Sentiment** | Average tone of coverage from bullish (+1) to bearish (-1) | `MEAN(document.sentiment_score)` based on keyword analysis |
| **Confidence** | How much data backs this row (0–1). Based on volume, source breadth, and data freshness | `0.40 × min(1, mentions/10) + 0.35 × min(1, sources/5) + 0.25 × max(0, 1 - hours_stale/48)` |
| **Trend** | Sparkline of daily mention counts over recent days | Historical daily snapshots |

---

**What "Run Pipeline" does:**
1. Fetches latest articles from all RSS feeds
2. Extracts entity mentions (tickers, themes, sectors)
3. Analyzes sentiment (bullish/bearish/neutral)
4. Computes velocity, acceleration, and z-scores
5. Saves daily snapshots for trend tracking

**Data sources currently active:** """ + ", ".join(source_names_list) + """

**Not yet connected:** Reddit, X/Twitter, YouTube (require API credentials)

---

**How to interpret momentum vs acceleration:**
- **High momentum, positive acceleration** → Heating up fast. Worth watching closely.
- **High momentum, negative acceleration** → Still popular but losing steam.
- **Low momentum, positive acceleration** → Early signal. Could be emerging.
- **Low momentum, negative acceleration** → Quiet and fading. Probably nothing.

**What confidence means:**
- 🟢 **High (≥70%)**: Multiple sources, recent data, many mentions. Trustworthy signal.
- 🟡 **Moderate (40-69%)**: Some data but gaps. Directionally useful.
- 🔴 **Low (<40%)**: Limited data. Take with a grain of salt.
            """)

        # ── Get Data for Main Table ─────────────────────────────
        # Get latest signals per entity (deduped)
        query = (
            session.query(Signal, Entity)
            .join(Entity, Signal.entity_id == Entity.id)
            .filter(Signal.window == params["window"])
        )

        if params["entity_type"]:
            query = query.filter(Entity.entity_type == params["entity_type"])

        query = query.order_by(Signal.computed_at.desc())
        all_results = query.all()

        # Deduplicate: keep only latest signal per entity
        seen_entities = set()
        results = []
        for signal, entity in all_results:
            if entity.id not in seen_entities:
                seen_entities.add(entity.id)
                results.append((signal, entity))

        if not results:
            st.warning(
                "⚠️ No signals computed yet. Click **Run Pipeline** in the sidebar to collect "
                "articles, extract entities, and compute metrics."
            )
            return

        # Get daily snapshots for sparklines (last 14 days)
        fourteen_days_ago = datetime.now(timezone.utc).date() - timedelta(days=14)
        snapshot_data = {}
        snapshots = (
            session.query(DailySnapshot)
            .filter(
                DailySnapshot.window == params["window"],
                DailySnapshot.date >= fourteen_days_ago,
            )
            .order_by(DailySnapshot.date)
            .all()
        )
        for snap in snapshots:
            if snap.entity_id not in snapshot_data:
                snapshot_data[snap.entity_id] = []
            snapshot_data[snap.entity_id].append({
                "date": snap.date,
                "mentions": snap.mentions or 0,
                "momentum": snap.momentum_pct or 0,
            })

        # Build display dataframe
        rows = []
        for signal, entity in results:
            # Get snapshot for this entity
            snap = (
                session.query(DailySnapshot)
                .filter(
                    DailySnapshot.entity_id == entity.id,
                    DailySnapshot.window == params["window"],
                )
                .order_by(DailySnapshot.date.desc())
                .first()
            )

            momentum_pct = snap.momentum_pct if snap else 0.0
            confidence = snap.confidence if snap else 0.0
            source_names = snap.source_names if snap and snap.source_names else []
            source_ct = snap.source_count if snap else (signal.platform_count or 0)
            acceleration = snap.acceleration if snap else (signal.acceleration or 0)
            sentiment = snap.sentiment_mean if snap else (signal.sentiment_mean or 0)

            # Sparkline data (list of mention counts)
            trend = [s["mentions"] for s in snapshot_data.get(entity.id, [])]
            if not trend:
                trend = [signal.mention_count or 0]

            type_badge = ""
            if entity.entity_type == EntityType.TICKER:
                type_badge = f"🏷️ {entity.symbol or entity.name}"
            elif entity.entity_type == EntityType.THEME:
                type_badge = f"🎯 {entity.name}"
            elif entity.entity_type == EntityType.SUBTHEME:
                type_badge = f"📂 {entity.name}"
            else:
                type_badge = entity.name

            rows.append({
                "Topic": type_badge,
                "7d Mentions": signal.mention_count or 0,
                "Δ vs 30d": momentum_pct,
                "Accel": acceleration,
                "Sources": ", ".join(source_names[:4]) + ("..." if len(source_names) > 4 else "") if source_names else "—",
                "Src #": source_ct,
                "Sentiment": sentiment,
                "Confidence": confidence,
                "Trend": trend,
                # Hidden fields for sorting
                "_entity_id": entity.id,
                "_signal_score": signal.signal_score or 0,
                "_heat_score": signal.heat_score or 0,
                "_edge_score": signal.edge_score or 0,
                "_phase": signal.phase.value if signal.phase else "baseline",
                "_label": signal.decision_label.value if signal.decision_label else "ignore",
            })

        df = pd.DataFrame(rows)

        # Sort by momentum magnitude
        df = df.sort_values("Δ vs 30d", ascending=False, key=abs).head(params["top_n"])

        # ── Top 5 Movers Cards ──────────────────────────────────
        st.subheader("🔥 Top Movers")
        top_movers = df.nlargest(5, "7d Mentions")

        if not top_movers.empty:
            cols = st.columns(min(5, len(top_movers)))
            for i, (_, row) in enumerate(top_movers.iterrows()):
                if i >= 5:
                    break
                with cols[i]:
                    delta_str = format_momentum(row["Δ vs 30d"])
                    delta_color = "normal" if row["Δ vs 30d"] >= 0 else "inverse"
                    st.metric(
                        label=row["Topic"][:20],
                        value=f"{row['7d Mentions']} mentions",
                        delta=delta_str,
                        delta_color=delta_color,
                    )
                    st.caption(
                        f"{get_confidence_badge(row['Confidence'])} · "
                        f"{row['Src #']} sources"
                    )

        st.markdown("---")

        # ── Main Radar Table ────────────────────────────────────
        st.subheader("📡 Rotation Radar")

        # Prepare display dataframe (drop hidden columns)
        display_df = df.drop(columns=[
            c for c in df.columns if c.startswith("_")
        ]).reset_index(drop=True)

        # Configure columns
        column_config = {
            "Topic": st.column_config.TextColumn(
                "Topic",
                help="Entity name with type badge (🏷️ Ticker, 🎯 Theme, 📂 Sub-theme)",
                width="medium",
            ),
            "7d Mentions": st.column_config.NumberColumn(
                "7d Mentions",
                help="Total times this topic appeared across all sources in the selected time window.",
                format="%d",
            ),
            "Δ vs 30d": st.column_config.NumberColumn(
                "Δ vs 30d",
                help="Momentum — % change vs the 30-day daily average. +100% means double normal volume.",
                format="%.0f%%",
            ),
            "Accel": st.column_config.NumberColumn(
                "Accel",
                help="Acceleration — is momentum speeding up or slowing down? Positive = gaining steam.",
                format="%.1f",
            ),
            "Sources": st.column_config.TextColumn(
                "Sources",
                help="Which data sources mentioned this topic.",
                width="medium",
            ),
            "Src #": st.column_config.NumberColumn(
                "Source Count",
                help="Number of distinct sources. More sources = higher confidence signal.",
                format="%d",
            ),
            "Sentiment": st.column_config.NumberColumn(
                "Sentiment",
                help="Average tone: bullish (+1) to bearish (-1). Based on keyword analysis.",
                format="%.2f",
            ),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                help="Data completeness score (0-1). Based on mention volume, source breadth, and freshness.",
                min_value=0,
                max_value=1,
                format="%.0f%%",
            ),
            "Trend": st.column_config.LineChartColumn(
                "Trend",
                help="Mention count over recent pipeline runs. Rising = heating up.",
                width="small",
            ),
        }

        st.dataframe(
            display_df,
            column_config=column_config,
            use_container_width=True,
            height=min(600, 40 + len(display_df) * 35),
            hide_index=True,
        )

        # ── Detail Expanders ────────────────────────────────────
        st.subheader("🔍 Entity Details")
        st.caption("Click any entity to see the full scoring breakdown, phase, and evidence.")

        for _, row in df.head(15).iterrows():
            with st.expander(f"{row['Topic']} — {row['7d Mentions']} mentions, {format_momentum(row['Δ vs 30d'])} momentum"):
                detail_cols = st.columns(4)
                detail_cols[0].metric("Signal Score", f"{row['_signal_score']:.0f}/100")
                detail_cols[1].metric("Heat Score", f"{row['_heat_score']:.0f}/100")
                detail_cols[2].metric("Edge Score", f"{row['_edge_score']:.0f}/100")
                detail_cols[3].metric("Stage", row['_phase'].replace("_", " ").title())

                st.caption(
                    f"**Decision Label:** {row['_label'].upper()} · "
                    f"**Sentiment:** {get_sentiment_badge(row['Sentiment'])} ({row['Sentiment']:.2f}) · "
                    f"**Confidence:** {get_confidence_badge(row['Confidence'])}"
                )

                # Show top documents for this entity
                entity_id = row["_entity_id"]
                top_docs = (
                    session.query(Document)
                    .join(DocumentEntity)
                    .filter(DocumentEntity.entity_id == entity_id)
                    .order_by(Document.published_at.desc())
                    .limit(5)
                    .all()
                )
                if top_docs:
                    st.caption("**Recent evidence:**")
                    for doc in top_docs:
                        source = session.query(Source).filter(Source.id == doc.source_id).first()
                        source_name = source.name if source else "Unknown"
                        sentiment_str = f" · {doc.sentiment_label}" if doc.sentiment_label else ""
                        link = f" [↗]({doc.url})" if doc.url else ""
                        st.markdown(
                            f"- **{doc.title or 'Untitled'}** — {source_name}{sentiment_str}{link}"
                        )

        st.markdown("---")

        # ── Mention Volume Chart ────────────────────────────────
        st.subheader("📊 Mention Volume")
        if not df.empty:
            chart_df = df[df["7d Mentions"] > 0].nlargest(20, "7d Mentions")
            if not chart_df.empty:
                fig = px.bar(
                    chart_df,
                    x="Topic",
                    y="7d Mentions",
                    color="Δ vs 30d",
                    color_continuous_scale=["#e74c3c", "#95a5a6", "#2ecc71"],
                    color_continuous_midpoint=0,
                    title="Top Entities by Mention Count (colored by momentum)",
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

        # ── Recent Headlines Feed ───────────────────────────────
        st.subheader("📰 Recent Headlines")
        recent_docs = (
            session.query(Document, Source)
            .join(Source, Document.source_id == Source.id)
            .filter(Document.title != None, Document.title != "")
            .order_by(Document.published_at.desc())
            .limit(15)
            .all()
        )

        if recent_docs:
            for doc, source in recent_docs:
                # Get linked entities
                doc_ents = (
                    session.query(Entity.name)
                    .join(DocumentEntity)
                    .filter(DocumentEntity.document_id == doc.id)
                    .all()
                )
                entity_tags = ", ".join([e.name for e in doc_ents[:4]]) if doc_ents else ""

                # Sentiment badge
                if doc.sentiment_label == "positive":
                    sent_badge = "🟢"
                elif doc.sentiment_label == "negative":
                    sent_badge = "🔴"
                else:
                    sent_badge = "⚪"

                # Time ago
                if doc.published_at:
                    pub = doc.published_at
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)
                    hours = (datetime.now(timezone.utc) - pub).total_seconds() / 3600
                    if hours < 1:
                        time_str = f"{int(hours * 60)}m ago"
                    elif hours < 24:
                        time_str = f"{int(hours)}h ago"
                    else:
                        time_str = f"{int(hours / 24)}d ago"
                else:
                    time_str = ""

                # Render headline row
                link = f"[↗]({doc.url})" if doc.url else ""
                tag_str = f" · `{entity_tags}`" if entity_tags else ""
                st.markdown(
                    f"{sent_badge} **{doc.title}** — "
                    f"_{source.name}_ · {time_str}{tag_str} {link}"
                )
        else:
            st.info("No headlines yet. Run the pipeline to collect articles.")

    finally:
        session.close()


# ============================================================
# Screen 2: Theme Drilldown
# ============================================================

def render_theme_drilldown(params: dict):
    """Render the Theme Drilldown screen"""
    st.header("🎯 Theme Drilldown")

    session = get_session()

    try:
        # Get all themes
        themes = session.query(Entity).filter(
            Entity.entity_type == EntityType.THEME,
            Entity.parent_id != None,  # Skip macro themes
        ).all()

        theme_names = [t.name for t in themes]

        if not theme_names:
            st.info("No themes loaded. Initialize the database first.")
            return

        selected_theme = st.selectbox("Select Theme", theme_names)

        # Get selected theme entity
        theme = session.query(Entity).filter(
            Entity.entity_type == EntityType.THEME,
            Entity.name == selected_theme,
        ).first()

        if not theme:
            return

        # Get theme signal
        theme_signal = (
            session.query(Signal)
            .filter(
                Signal.entity_id == theme.id,
                Signal.window == params["window"],
            )
            .order_by(Signal.computed_at.desc())
            .first()
        )

        # Get snapshot for transparent metrics
        snap = (
            session.query(DailySnapshot)
            .filter(
                DailySnapshot.entity_id == theme.id,
                DailySnapshot.window == params["window"],
            )
            .order_by(DailySnapshot.date.desc())
            .first()
        )

        # Header metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        if theme_signal:
            mentions = theme_signal.mention_count or 0
            momentum = snap.momentum_pct if snap else 0.0
            confidence = snap.confidence if snap else 0.0

            col1.metric("Mentions", mentions)
            col2.metric("Momentum", format_momentum(momentum))
            col3.metric("Sources", theme_signal.platform_count or 0)
            col4.metric("Sentiment", get_sentiment_badge(theme_signal.sentiment_mean or 0))
            col5.metric("Confidence", get_confidence_badge(confidence))

        # Sub-themes tab
        st.subheader("Sub-themes")
        children = session.query(Entity).filter(Entity.parent_id == theme.id).all()

        if children:
            sub_rows = []
            for child in children:
                child_signal = (
                    session.query(Signal)
                    .filter(
                        Signal.entity_id == child.id,
                        Signal.window == params["window"],
                    )
                    .order_by(Signal.computed_at.desc())
                    .first()
                )

                child_snap = (
                    session.query(DailySnapshot)
                    .filter(
                        DailySnapshot.entity_id == child.id,
                        DailySnapshot.window == params["window"],
                    )
                    .order_by(DailySnapshot.date.desc())
                    .first()
                )

                sub_rows.append({
                    "Sub-theme": child.name,
                    "Type": child.entity_type.value,
                    "Mentions": child_signal.mention_count if child_signal else 0,
                    "Δ vs 30d": f"{child_snap.momentum_pct:.0f}%" if child_snap else "—",
                    "Sources": child_signal.platform_count or 0 if child_signal else 0,
                    "Sentiment": round(child_signal.sentiment_mean or 0, 2) if child_signal else 0,
                    "Confidence": round(child_snap.confidence or 0, 2) if child_snap else 0,
                })

            sub_df = pd.DataFrame(sub_rows)
            st.dataframe(sub_df, use_container_width=True, hide_index=True)

        # Related tickers
        st.subheader("Related Tickers")
        ticker_entities = (
            session.query(Entity)
            .filter(
                Entity.entity_type == EntityType.TICKER,
                Entity.parent_id.in_([c.id for c in children]) if children else False,
            )
            .all()
        )

        if ticker_entities:
            ticker_rows = []
            for te in ticker_entities:
                te_signal = (
                    session.query(Signal)
                    .filter(
                        Signal.entity_id == te.id,
                        Signal.window == params["window"],
                    )
                    .order_by(Signal.computed_at.desc())
                    .first()
                )
                ticker_rows.append({
                    "Ticker": te.symbol or te.name,
                    "Mentions": te_signal.mention_count if te_signal else 0,
                    "Sentiment": round(te_signal.sentiment_mean or 0, 2) if te_signal else 0,
                    "Sources": te_signal.platform_count if te_signal else 0,
                    "Divergence": round(te_signal.divergence_score or 0, 2) if te_signal else 0,
                })

            ticker_df = pd.DataFrame(ticker_rows)
            st.dataframe(ticker_df, use_container_width=True, hide_index=True)

        # Recent mentions
        st.subheader("Recent Mentions")
        recent_docs = (
            session.query(Document)
            .join(DocumentEntity)
            .filter(DocumentEntity.entity_id == theme.id)
            .order_by(Document.published_at.desc())
            .limit(20)
            .all()
        )

        for doc in recent_docs:
            with st.expander(f"{doc.title or 'Untitled'} ({doc.content_type})"):
                st.write(f"**Source:** {doc.url or 'N/A'}")
                st.write(f"**Published:** {doc.published_at}")
                if doc.sentiment_label:
                    st.write(f"**Sentiment:** {doc.sentiment_label} ({doc.sentiment_score:.2f})")
                st.write(doc.content[:500] + "..." if doc.content and len(doc.content) > 500 else doc.content or "")

    finally:
        session.close()


# ============================================================
# Screen 3: Ticker Page
# ============================================================

def render_ticker_page(params: dict):
    """Render the Ticker Page screen"""
    st.header("🏷️ Ticker Analysis")

    session = get_session()

    try:
        # Get all ticker entities
        tickers = session.query(Entity).filter(
            Entity.entity_type == EntityType.TICKER,
            Entity.symbol != None,
        ).all()

        ticker_symbols = sorted(set(t.symbol for t in tickers if t.symbol))

        if not ticker_symbols:
            st.info("No tickers loaded yet.")
            return

        selected = st.selectbox("Select Ticker", ticker_symbols)

        # Get ticker entity
        ticker_entity = session.query(Entity).filter(
            Entity.entity_type == EntityType.TICKER,
            Entity.symbol == selected,
        ).first()

        if not ticker_entity:
            return

        # Get latest signal
        signal = (
            session.query(Signal)
            .filter(
                Signal.entity_id == ticker_entity.id,
                Signal.window == params["window"],
            )
            .order_by(Signal.computed_at.desc())
            .first()
        )

        # Get snapshot
        snap = (
            session.query(DailySnapshot)
            .filter(
                DailySnapshot.entity_id == ticker_entity.id,
                DailySnapshot.window == params["window"],
            )
            .order_by(DailySnapshot.date.desc())
            .first()
        )

        # Header metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        if signal:
            col1.metric("Mentions", signal.mention_count or 0)
            col2.metric("Momentum", format_momentum(snap.momentum_pct if snap else 0))
            col3.metric("Sources", signal.platform_count or 0)
            col4.metric("Sentiment", get_sentiment_badge(signal.sentiment_mean or 0))
            col5.metric("Confidence", get_confidence_badge(snap.confidence if snap else 0))
            col6.metric("Divergence", f"{signal.divergence_score:.2f}" if signal.divergence_score else "N/A")

        # Price chart (if yfinance available)
        try:
            import yfinance as yf
            stock = yf.Ticker(selected)
            hist = stock.history(period="3mo")

            if not hist.empty:
                st.subheader("Price Action")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name=selected,
                ))
                fig.update_layout(
                    height=400,
                    xaxis_rangeslider_visible=False,
                    title=f"{selected} - 3 Month Price",
                )
                st.plotly_chart(fig, use_container_width=True)

        except (ImportError, Exception) as e:
            st.info(f"Price chart unavailable: {e}")

        # Explanation
        if signal and signal.explanation:
            st.subheader("Signal Analysis")
            st.write(signal.explanation)

        # Recent documents mentioning this ticker
        st.subheader("Recent Mentions")
        recent = (
            session.query(Document)
            .join(DocumentEntity)
            .filter(DocumentEntity.entity_id == ticker_entity.id)
            .order_by(Document.published_at.desc())
            .limit(15)
            .all()
        )

        for doc in recent:
            with st.expander(f"{doc.title or 'Untitled'} - {doc.published_at}"):
                if doc.url:
                    st.markdown(f"[Link]({doc.url})")
                st.write(f"**Sentiment:** {doc.sentiment_label or 'N/A'}")
                st.write(doc.content[:300] + "..." if doc.content and len(doc.content) > 300 else doc.content or "")

    finally:
        session.close()


# ============================================================
# Screen 4: Source Control
# ============================================================

def render_source_control():
    """Render the Source Control screen"""
    st.header("📡 Source Control")

    session = get_session()

    try:
        sources = session.query(Source).order_by(Source.source_type, Source.weight.desc()).all()

        if not sources:
            st.info("No sources configured yet.")
            return

        # Group by type
        source_rows = []
        for s in sources:
            doc_count = session.query(Document).filter(Document.source_id == s.id).count()
            source_rows.append({
                "Platform": s.source_type.value if s.source_type else "",
                "Name": s.name or s.identifier,
                "Weight": s.weight or 1.0,
                "Category": s.category or "",
                "Documents": doc_count,
                "Alpha Score": s.alpha_score or 0.0,
                "Enabled": "✅" if s.enabled else "❌",
            })

        df = pd.DataFrame(source_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Source breakdown chart
        st.subheader("Documents by Source")
        if not df.empty and df["Documents"].sum() > 0:
            fig = px.bar(
                df.sort_values("Documents", ascending=False).head(20),
                x="Name",
                y="Documents",
                color="Platform",
                title="Top Sources by Document Count",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    finally:
        session.close()


# ============================================================
# Screen 5: Alerts & Journal
# ============================================================

def render_alerts_journal():
    """Render the Alerts & Journal screen"""
    st.header("🔔 Alerts & Journal")

    tab1, tab2 = st.tabs(["Alerts", "Journal"])

    session = get_session()

    try:
        # Alerts Tab
        with tab1:
            alerts = (
                session.query(Alert)
                .order_by(Alert.created_at.desc())
                .limit(50)
                .all()
            )

            if not alerts:
                st.info("No alerts yet. Alerts are generated after signal computation.")
            else:
                # Filter
                show_ack = st.checkbox("Show acknowledged", value=False)

                for alert in alerts:
                    if not show_ack and alert.acknowledged:
                        continue

                    severity_colors = {
                        "high": "red",
                        "medium": "orange",
                        "low": "blue",
                    }
                    color = severity_colors.get(alert.severity, "gray")

                    with st.expander(
                        f":{color}[{alert.alert_type.value.upper()}] {alert.title} - {alert.created_at:%Y-%m-%d %H:%M}"
                    ):
                        st.write(alert.message)
                        if not alert.acknowledged:
                            col1, col2, col3, col4 = st.columns(4)
                            if col1.button("Watch", key=f"watch_{alert.id}"):
                                alert.acknowledged = True
                                alert.acknowledged_at = datetime.now(timezone.utc)
                                alert.action_taken = "watch"
                                session.commit()
                                st.rerun()
                            if col2.button("Build", key=f"build_{alert.id}"):
                                alert.acknowledged = True
                                alert.acknowledged_at = datetime.now(timezone.utc)
                                alert.action_taken = "build"
                                session.commit()
                                st.rerun()
                            if col3.button("Now", key=f"now_{alert.id}"):
                                alert.acknowledged = True
                                alert.acknowledged_at = datetime.now(timezone.utc)
                                alert.action_taken = "now"
                                session.commit()
                                st.rerun()
                            if col4.button("Dismiss", key=f"dismiss_{alert.id}"):
                                alert.acknowledged = True
                                alert.acknowledged_at = datetime.now(timezone.utc)
                                alert.action_taken = "dismiss"
                                session.commit()
                                st.rerun()

        # Journal Tab
        with tab2:
            st.subheader("Add Entry")
            with st.form("journal_form"):
                title = st.text_input("Title")
                content = st.text_area("Notes")
                entry_type = st.selectbox("Type", ["note", "trade", "thesis"])
                tags_input = st.text_input("Tags (comma-separated)")

                col1, col2, col3 = st.columns(3)
                ticker = col1.text_input("Ticker (optional)")
                action = col2.selectbox("Action", ["", "buy", "sell", "watch"])
                entry_price = col3.number_input("Price", min_value=0.0, value=0.0, step=0.01)

                submitted = st.form_submit_button("Save Entry")

                if submitted and (title or content):
                    tags = [t.strip() for t in tags_input.split(",") if t.strip()]
                    entry = JournalEntry(
                        title=title,
                        content=content,
                        entry_type=entry_type,
                        tags=tags,
                        ticker=ticker if ticker else None,
                        action=action if action else None,
                        entry_price=entry_price if entry_price > 0 else None,
                    )
                    session.add(entry)
                    session.commit()
                    st.success("Entry saved!")

            # Display journal entries
            st.subheader("Recent Entries")
            entries = (
                session.query(JournalEntry)
                .order_by(JournalEntry.created_at.desc())
                .limit(20)
                .all()
            )

            for entry in entries:
                with st.expander(
                    f"{entry.title or 'Untitled'} ({entry.entry_type}) - "
                    f"{entry.created_at:%Y-%m-%d %H:%M}"
                ):
                    if entry.ticker:
                        st.write(f"**Ticker:** {entry.ticker} | **Action:** {entry.action or 'N/A'}")
                    if entry.entry_price:
                        st.write(f"**Entry Price:** ${entry.entry_price:.2f}")
                    st.write(entry.content or "")
                    if entry.tags:
                        st.write(f"**Tags:** {', '.join(entry.tags)}")

    finally:
        session.close()


# ============================================================
# Main App
# ============================================================

def main():
    """Main application entry point"""

    # Initialize database
    init_db()

    # Render sidebar and get params
    params = render_sidebar()

    # Navigation
    page = st.sidebar.radio(
        "Navigate",
        ["Home Radar", "Theme Drilldown", "Ticker Page", "Source Control", "Alerts & Journal"],
    )

    if page == "Home Radar":
        render_home_radar(params)
    elif page == "Theme Drilldown":
        render_theme_drilldown(params)
    elif page == "Ticker Page":
        render_ticker_page(params)
    elif page == "Source Control":
        render_source_control()
    elif page == "Alerts & Journal":
        render_alerts_journal()


if __name__ == "__main__":
    main()
