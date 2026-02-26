# Rotation Radar

**Sector Rotation Intelligence System** - Turns cross-platform social chatter into ranked, explainable trading signals.

> "What's moving from fringe to consensus, and is price confirming yet?"

---

## What It Does

Rotation Radar monitors Reddit, RSS feeds, and YouTube for sector rotation signals. It's not a generic sentiment dashboard - it's a **Narrative to Signal to Decision Engine** that distinguishes between what's *loud* (Heat) and what's *early* (Edge).

**Decision Labels:**

| Label | Meaning | Timeframe |
|-------|---------|-----------|
| **NOW** | Strong signal, price not moved, catalyst credible | 0-2 weeks |
| **BUILD** | Rising but needs confirmation | 1-3 months |
| **WATCH** | Early blips, insufficient breadth | Monitor |
| **IGNORE** | Noisy or consensus late | Skip |

---

## Quick Start

```bash
# 1. Install dependencies
cd rotation-radar
pip install -r requirements.txt

# 2. Download spaCy model (for entity extraction)
python -m spacy download en_core_web_sm

# 3. Configure
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your Reddit API credentials

# 4. Initialize database and load taxonomy
python scripts/setup_db.py

# 5. Run the full pipeline (collect -> process -> score)
python scripts/run_pipeline.py

# 6. Launch dashboard
streamlit run src/ui/app.py
```

### Getting Reddit API Credentials

1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app"
3. Select "script" type
4. Note the client ID and client secret
5. Add them to `config/config.yaml`

---

## Architecture

```
Data Sources (Reddit, RSS, YouTube)
        |
        v
  Collectors (scheduled, 4x/day)
        |
        v
  Entity Extraction (tickers, themes, catalysts)
  Sentiment Analysis (FinBERT + rule-based)
        |
        v
  Velocity Engine (rolling window metrics, z-scores)
  Divergence Engine (chatter vs price)
        |
        v
  Signal Model (scoring + gates + phase classification)
        |
        v
  Dashboard (Streamlit, 5 screens)
```

### Signal Score Formula

```
signal_raw = 0.30 * velocity_z
           + 0.20 * unique_authors_z
           + 0.15 * sentiment_delta_z
           + 0.15 * cross_platform_score
           + 0.10 * divergence_score
           + 0.10 * catalyst_score

signal_score = clamp(50 + 20 * signal_raw, 0, 100)
```

### Quality Gates

- **Breadth Gate**: Requires unique_authors_z > 0 OR cross-platform confirmation
- **Quality Gate**: Source-weighted score above threshold
- **Pump Filter**: High author concentration + euphoric language + weak breadth = capped score

### Phase Lifecycle

| Phase | Characteristics |
|-------|----------------|
| **Ignition** | First sustained acceleration, breadth present, price NOT confirming |
| **Acceleration** | Broadening participation, cross-platform, RS turning up |
| **Crowded** | High concentration, euphoric language, price extended |
| **Exhaustion** | Velocity decelerating, sentiment high, price stalling |
| **Cooling** | Narrative fading, volume declining |

---

## Project Structure

```
rotation-radar/
├── config/
│   ├── config.example.yaml      # Template (copy to config.yaml)
│   ├── sources.yaml             # Curated sources with weights
│   └── taxonomy.yaml            # Theme hierarchy + tickers + keywords
├── scripts/
│   ├── setup_db.py              # Initialize database + load taxonomy
│   ├── run_pipeline.py          # Full pipeline runner
│   └── backfill.py              # Re-process historical data
├── src/
│   ├── models/
│   │   └── database.py          # SQLAlchemy ORM models
│   ├── collectors/
│   │   ├── base.py              # Base collector interface
│   │   ├── reddit_collector.py  # Reddit via PRAW
│   │   ├── rss_collector.py     # RSS via feedparser
│   │   └── youtube_collector.py # YouTube transcripts
│   ├── processing/
│   │   ├── entity_extraction.py # Ticker/theme extraction
│   │   ├── sentiment.py         # FinBERT sentiment analysis
│   │   ├── velocity.py          # Rolling metrics + z-scores
│   │   ├── divergence.py        # Chatter-price divergence
│   │   └── signal_model.py      # Scoring, gates, phases, labels
│   └── ui/
│       └── app.py               # Streamlit dashboard (5 screens)
├── data/
│   └── radar.db                 # SQLite database (auto-created)
├── docs/                        # Specifications
└── requirements.txt
```

---

## Dashboard Screens

1. **Home Radar** - Heatmap of top signals, phase distribution, velocity chart
2. **Theme Drilldown** - Deep dive into a theme's sub-themes, tickers, recent mentions
3. **Ticker Page** - Price chart, signal analysis, chatter-price divergence
4. **Source Control** - Manage sources, view weights, document counts
5. **Alerts & Journal** - Alert inbox (3 types) + trade/decision journal

---

## Pipeline Commands

```bash
# Full pipeline
python scripts/run_pipeline.py

# Collection only
python scripts/run_pipeline.py --collect-only

# Processing only (entity extraction + sentiment)
python scripts/run_pipeline.py --process-only

# Scoring only (velocity + divergence + signals)
python scripts/run_pipeline.py --score-only

# Re-process everything (after taxonomy changes)
python scripts/backfill.py

# Re-extract entities only
python scripts/backfill.py --entities-only
```

---

## Tracked Sectors

Pre-configured taxonomy covers 6 macro themes:

- **Energy**: Nuclear (SMR, Fuel Cycle, Utilities), Grid (Storage, Transmission, Switchgear)
- **AI Infrastructure**: Data Centers (Compute, Neocloud, Power/Cooling), Software (Platforms, Cybersecurity)
- **Space & Defense**: Space (Launch, Satellites), Defense (Primes, Drones, Cyber)
- **Fintech/Crypto**: Crypto (Bitcoin, Miners), Fintech (Brokerage, Payments)
- **Healthcare**: Biotech (Large Cap, GLP-1/Obesity)
- **Industrials**: Reshoring (Construction, Automation)

Easily extensible via `config/taxonomy.yaml`.

---

## Tech Stack

- **Python 3.11+**
- **SQLite** (SQLAlchemy ORM)
- **Streamlit** (Dashboard)
- **PRAW** (Reddit API)
- **FinBERT** (Financial sentiment via transformers)
- **yfinance** (Price data)
- **Plotly** (Charts)
- **spaCy** (Optional NER)

---

## License

MIT
