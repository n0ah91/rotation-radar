"""
Rotation Radar Dashboard

Main Streamlit application with 5 screens:
1. Home Radar - Heatmap + Change Detection + Narrative Map
2. Theme Drilldown - Metrics + Sub-themes + Narratives
3. Ticker Page - Price + Chatter divergence + Evidence
4. Source Control - Source management + Alpha tracking
5. Alerts & Journal - Alert inbox + Decision log
"""

import streamlit as st
import pandas as pd
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
    Alert,
    JournalEntry,
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


def get_phase_color(phase: Phase) -> str:
    """Get color for phase badge"""
    colors = {
        Phase.IGNITION: "#ff6b35",
        Phase.ACCELERATION: "#00b894",
        Phase.CROWDED: "#e17055",
        Phase.EXHAUSTION: "#d63031",
        Phase.COOLING: "#74b9ff",
        Phase.BASELINE: "#636e72",
    }
    return colors.get(phase, "#636e72")


def get_label_color(label: DecisionLabel) -> str:
    """Get color for decision label"""
    colors = {
        DecisionLabel.NOW: "#00b894",
        DecisionLabel.BUILD: "#fdcb6e",
        DecisionLabel.WATCH: "#74b9ff",
        DecisionLabel.IGNORE: "#636e72",
    }
    return colors.get(label, "#636e72")


# ============================================================
# Sidebar
# ============================================================

def render_sidebar():
    """Render the sidebar with global controls"""
    st.sidebar.title("Rotation Radar")
    st.sidebar.markdown("---")

    # Time window selector
    window = st.sidebar.selectbox(
        "Time Window",
        ["6h", "24h", "7d", "30d"],
        index=2,  # Default to 7d until we have multi-day data
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

    # Focus toggle
    focus = st.sidebar.radio(
        "Focus",
        ["Signal Score", "Edge (Early)", "Heat (Loud)"],
    )

    sort_map = {
        "Signal Score": "signal_score",
        "Edge (Early)": "edge_score",
        "Heat (Loud)": "heat_score",
    }

    # Top N
    top_n = st.sidebar.slider("Top N Items", 5, 50, 25)

    st.sidebar.markdown("---")

    # Database info
    session = get_session()
    doc_count = session.query(Document).count()
    entity_count = session.query(Entity).count()
    signal_count = session.query(Signal).count()
    session.close()

    st.sidebar.metric("Documents", doc_count)
    st.sidebar.metric("Entities Tracked", entity_count)
    st.sidebar.metric("Signals Computed", signal_count)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Pipeline Control")
    if st.sidebar.button("Run Pipeline", type="primary", use_container_width=True):
        run_pipeline_from_ui()

    if doc_count == 0:
        st.sidebar.caption("Click 'Run Pipeline' to collect data and compute signals.")

    return {
        "window": window,
        "entity_type": entity_type_map[entity_filter],
        "sort_by": sort_map[focus],
        "top_n": top_n,
    }


# ============================================================
# Screen 1: Home Radar
# ============================================================

def render_home_radar(params: dict):
    """Render the Home Radar screen"""
    st.header("Home Radar")

    session = get_session()

    try:
        # Get ranked signals
        query = (
            session.query(Signal, Entity)
            .join(Entity, Signal.entity_id == Entity.id)
            .filter(Signal.window == params["window"])
        )

        if params["entity_type"]:
            query = query.filter(Entity.entity_type == params["entity_type"])

        sort_field = getattr(Signal, params["sort_by"])
        query = query.order_by(sort_field.desc())

        all_results = query.all()

        # Deduplicate: keep only latest signal per entity
        seen_entities = set()
        results = []
        for signal, entity in all_results:
            if entity.id not in seen_entities:
                seen_entities.add(entity.id)
                results.append((signal, entity))
                if len(results) >= params["top_n"]:
                    break

        if not results:
            st.info(
                "No signals computed yet. Run the pipeline first:\n\n"
                "```python\npython scripts/run_pipeline.py\n```"
            )
            return

        # Panel A: Heatmap
        st.subheader("What's Hot")

        # Build dataframe
        rows = []
        for signal, entity in results:
            rows.append({
                "Entity": entity.name,
                "Type": entity.entity_type.value if entity.entity_type else "",
                "Signal": signal.signal_score or 0,
                "Heat": signal.heat_score or 0,
                "Edge": signal.edge_score or 0,
                "Phase": signal.phase.value if signal.phase else "baseline",
                "Label": signal.decision_label.value if signal.decision_label else "ignore",
                "Velocity Z": round(signal.z_velocity or 0, 1),
                "Authors": signal.unique_authors or 0,
                "Platforms": signal.platform_count or 0,
                "Explanation": signal.explanation or "",
            })

        df = pd.DataFrame(rows)

        # Color-coded signal table
        def color_signal(val):
            if val >= 80:
                return "background-color: #00b894; color: white"
            elif val >= 65:
                return "background-color: #fdcb6e"
            elif val >= 50:
                return "background-color: #74b9ff"
            else:
                return "background-color: #dfe6e9"

        styled = df.style.map(
            color_signal,
            subset=["Signal"]
        )

        st.dataframe(styled, use_container_width=True, height=400)

        # Panel B: What Changed
        st.subheader("What Changed")

        col1, col2 = st.columns(2)

        with col1:
            # Phase distribution
            if not df.empty:
                phase_counts = df["Phase"].value_counts()
                fig_phase = px.pie(
                    values=phase_counts.values,
                    names=phase_counts.index,
                    title="Phase Distribution",
                    color_discrete_map={
                        "ignition": "#ff6b35",
                        "acceleration": "#00b894",
                        "crowded": "#e17055",
                        "exhaustion": "#d63031",
                        "cooling": "#74b9ff",
                        "baseline": "#636e72",
                    }
                )
                fig_phase.update_layout(height=300)
                st.plotly_chart(fig_phase, use_container_width=True)

        with col2:
            # Decision label distribution
            if not df.empty:
                label_counts = df["Label"].value_counts()
                fig_labels = px.bar(
                    x=label_counts.index,
                    y=label_counts.values,
                    title="Decision Labels",
                    color=label_counts.index,
                    color_discrete_map={
                        "now": "#00b894",
                        "build": "#fdcb6e",
                        "watch": "#74b9ff",
                        "ignore": "#636e72",
                    },
                )
                fig_labels.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_labels, use_container_width=True)

        # Top movers by velocity
        st.subheader("Top Movers (Velocity)")
        if not df.empty:
            top_movers = df.nlargest(10, "Velocity Z")
            fig_vel = px.bar(
                top_movers,
                x="Entity",
                y="Velocity Z",
                color="Phase",
                title="Top 10 by Chatter Velocity",
                color_discrete_map={
                    "ignition": "#ff6b35",
                    "acceleration": "#00b894",
                    "crowded": "#e17055",
                    "exhaustion": "#d63031",
                    "cooling": "#74b9ff",
                    "baseline": "#636e72",
                },
            )
            fig_vel.update_layout(height=350)
            st.plotly_chart(fig_vel, use_container_width=True)

        # Mention count chart (useful even with baseline scores)
        st.subheader("Mention Heatmap")
        if not df.empty:
            # Get mention counts from signals
            mention_rows = []
            for signal, entity in results:
                mention_rows.append({
                    "Entity": entity.name,
                    "Type": entity.entity_type.value if entity.entity_type else "",
                    "Mentions": signal.mention_count or 0,
                    "Heat": signal.heat_score or 0,
                })
            mention_df = pd.DataFrame(mention_rows)
            mention_df = mention_df[mention_df["Mentions"] > 0].sort_values("Mentions", ascending=False).head(20)

            if not mention_df.empty:
                fig_mentions = px.bar(
                    mention_df,
                    x="Entity",
                    y="Mentions",
                    color="Type",
                    title="Top Entities by Mention Count",
                )
                fig_mentions.update_layout(height=350)
                st.plotly_chart(fig_mentions, use_container_width=True)

        # Recent documents feed
        st.subheader("Viral Now - Recent Posts & Headlines")
        recent_docs = (
            session.query(Document)
            .filter(Document.content != None, Document.content != "")
            .order_by(Document.published_at.desc())
            .limit(10)
            .all()
        )

        for doc in recent_docs:
            # Get linked entities
            doc_ents = (
                session.query(Entity.name)
                .join(DocumentEntity)
                .filter(DocumentEntity.document_id == doc.id)
                .all()
            )
            entity_tags = ", ".join([e.name for e in doc_ents]) if doc_ents else "—"

            with st.expander(f"{doc.title or 'Untitled'} | {entity_tags}"):
                cols = st.columns([3, 1])
                with cols[0]:
                    if doc.url:
                        st.markdown(f"[Source Link]({doc.url})")
                    st.caption(f"Published: {doc.published_at} | Sentiment: {doc.sentiment_label or 'N/A'}")
                with cols[1]:
                    st.caption(f"Entities: {entity_tags}")
                if doc.content:
                    st.write(doc.content[:400] + "..." if len(doc.content) > 400 else doc.content)

    finally:
        session.close()


# ============================================================
# Screen 2: Theme Drilldown
# ============================================================

def render_theme_drilldown(params: dict):
    """Render the Theme Drilldown screen"""
    st.header("Theme Drilldown")

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

        # Header metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        if theme_signal:
            col1.metric("Signal Score", theme_signal.signal_score or 0)
            col2.metric("Heat", theme_signal.heat_score or 0)
            col3.metric("Edge", theme_signal.edge_score or 0)
            col4.metric("Phase", theme_signal.phase.value if theme_signal.phase else "N/A")
            col5.metric("Label", theme_signal.decision_label.value.upper() if theme_signal.decision_label else "N/A")

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

                sub_rows.append({
                    "Sub-theme": child.name,
                    "Type": child.entity_type.value,
                    "Signal": child_signal.signal_score if child_signal else 0,
                    "Phase": child_signal.phase.value if child_signal and child_signal.phase else "baseline",
                    "Label": child_signal.decision_label.value if child_signal and child_signal.decision_label else "ignore",
                    "Velocity Z": round(child_signal.z_velocity or 0, 1) if child_signal else 0,
                    "Authors": child_signal.unique_authors or 0 if child_signal else 0,
                })

            sub_df = pd.DataFrame(sub_rows)
            st.dataframe(sub_df, use_container_width=True)

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
                    "Signal": te_signal.signal_score if te_signal else 0,
                    "Heat": te_signal.heat_score if te_signal else 0,
                    "Edge": te_signal.edge_score if te_signal else 0,
                    "Phase": te_signal.phase.value if te_signal and te_signal.phase else "baseline",
                    "Divergence": round(te_signal.divergence_score or 0, 2) if te_signal else 0,
                })

            ticker_df = pd.DataFrame(ticker_rows)
            st.dataframe(ticker_df, use_container_width=True)

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
                st.write(f"**Score:** {doc.score} | **Comments:** {doc.comment_count}")
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
    st.header("Ticker Analysis")

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

        # Header metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        if signal:
            col1.metric("Signal", signal.signal_score or 0)
            col2.metric("Heat", signal.heat_score or 0)
            col3.metric("Edge", signal.edge_score or 0)
            col4.metric("Phase", signal.phase.value if signal.phase else "N/A")
            col5.metric("Label", signal.decision_label.value.upper() if signal.decision_label else "N/A")
            col6.metric("Divergence", f"{signal.divergence_score:.2f}" if signal and signal.divergence_score else "N/A")

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
    st.header("Source Control")

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
                "Enabled": s.enabled,
            })

        df = pd.DataFrame(source_rows)
        st.dataframe(df, use_container_width=True)

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
    st.header("Alerts & Journal")

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
