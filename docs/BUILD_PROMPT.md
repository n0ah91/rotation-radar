# Build Prompt for Claude Code / Engineers

Copy and paste this prompt to any AI coding assistant or engineer to build the system.

---

## The Prompt

```
You are my senior engineer + quant PM. Build a single-user "Sector Rotation Radar" dashboard that converts cross-platform social + headline chatter into ranked, explainable narrative signals.

## Goal
- Identify early sector/theme rotations using: mention velocity + acceleration, unique-author velocity, sentiment trajectory (delta), cross-platform confirmation, and chatter–price divergence.
- Output should label each theme/ticker: NOW (0–2w), BUILD (1–3m), WATCH, IGNORE.
- Every signal must be explainable (top drivers + linked sources) and include a suggested tripwire/invalidation condition.

## Constraints
- Single user, local-first, cheap to run.
- MVP must work without X/Twitter (X is optional plugin).
- Start with Reddit + (optional) Stocktwits + RSS news.
- Use Python + SQLite + Streamlit for MVP. Provide a clean repo structure.

## Data Model
Core entities:
- Source: type, identifier, weight, alpha_score
- Author: username, credibility_score, account_age, is_suspected_bot
- Document: content, url, published_at, engagement metrics
- Entity: type (ticker/theme/sub_theme/catalyst), symbol, parent_id (hierarchy)
- Narrative: title, claim_summary, catalyst_tags, bear_case, tripwires, linked documents
- Signal: entity_id, window, all metric z-scores, heat_score, edge_score, phase, decision, explanation
- Alert: type (rotation_ignition/divergence_edge/overheat), entity_id, message

## Signal Model
Compute rolling metrics over 1h/6h/24h/7d windows:
- velocity_z: mention acceleration vs baseline
- unique_authors_z: new unique posters
- sentiment_delta_z: sentiment change vs 7d baseline
- cross_platform_score: platforms confirming together
- divergence_score: chatter_z - price_z
- catalyst_score: weighted relevance of tagged catalysts

Score formula:
```
signal_score = (
    0.30 * velocity_z +
    0.20 * unique_authors_z +
    0.15 * sentiment_delta_z +
    0.15 * cross_platform_score +
    0.10 * divergence_score +
    0.10 * catalyst_score
)
```

Gates:
- Breadth gate: unique_authors_z > 0 OR cross_platform_score > 0.5 (else cap at 30)
- Quality gate: source-weighted score > threshold (else apply 0.7x)
- Pump filter: concentration > 70% AND new accounts AND euphoria → cap at 50, tag "Speculative"

Two composite scores:
- Heat Score: what's loud now (volume + sentiment + engagement)
- Edge Score: what's early (unique author velocity + divergence + low concentration)

Phase classification: ignition → acceleration → crowded → exhaustion → cooling

## UI Requirements

5 screens:

1) **Home Radar**:
   - Scrollable heatmap of Themes/Sub-themes with score + phase badge
   - "What changed since yesterday" feed (new entrants, phase flips, narrative shifts, divergence/overheat)
   - "Top Narratives" list with velocity/breadth/quality indicators
   - "Viral Now" feed: top posts + headlines with theme relevance

2) **Theme Drilldown**:
   - Metric charts: mentions, unique authors, sentiment delta, cross-platform, chatter vs price
   - Sub-theme tiles with scores
   - Narrative cards: claim summary, catalyst, who's pushing, bear case, tripwires, source links
   - Time Machine: first mention date, inflection point, peak velocity, current position
   - ETF vs single-name recommendations

3) **Ticker Page**:
   - Chatter vs price divergence chart
   - Narratives affecting this ticker (ranked)
   - Top sources mentioning ticker
   - Relative strength vs benchmarks

4) **Source Control**:
   - Add/remove sources with weight sliders
   - Source Alpha Leaderboard (which sources spot moves early?)
   - Last active timestamps

5) **Alerts + Journal**:
   - 3 alert types only: Rotation Ignition, Divergence Edge, Overheat/Late
   - Journal for logging trades/passes with signal context
   - Backtest summary: signal accuracy over time

## Deliverables
1. Repo scaffold with clear structure
2. Ingestion scripts (Reddit first via PRAW)
3. Processing pipeline (entity extraction, sentiment, velocity, divergence)
4. Signal computation with all gates
5. Streamlit app with 5 screens (at least screens 1-3 fully working)
6. README with setup steps
7. Config via YAML (sources.yaml, taxonomy.yaml)
8. Environment variables for API keys

## Initial Configuration

taxonomy.yaml should support:
- Macro Theme → Theme → Sub-theme → Tickers/ETFs
- Example: Energy → Nuclear → SMR Builders → [OKLO, SMR, NNE]

sources.yaml should support:
- Reddit subreddits with weights
- YouTube channels (optional)
- RSS feeds (optional)
- Twitter accounts (optional, requires API)

## Technical Notes
- Use FinBERT for sentiment (transformers library)
- Use yfinance for price data
- Schedule collection 4x/day via cron or APScheduler
- Store everything in SQLite (single file, easy backup)
- All dates in UTC

Start by creating the repo structure, then implement the Reddit collector, then the signal pipeline, then the UI screens in order.
```

---

## Configuration Templates to Provide

When asked, provide these starter configs:

### taxonomy.yaml
```yaml
macro_themes:
  Energy:
    Nuclear:
      SMR_Builders:
        tickers: [OKLO, SMR, NNE]
        etfs: []
      Fuel_Cycle:
        tickers: [CCJ, UEC, UUUU, LEU]
        etfs: [URNM, URA]
      Utilities_Nuclear:
        tickers: [CEG, VST, NRG]
        etfs: []
    Grid:
      Storage:
        tickers: [EOSE, FLNC, STEM]
        etfs: []
      Transmission:
        tickers: [NEE, ETN, PWR]
        etfs: []
  
  AI_Infrastructure:
    Data_Centers:
      Compute:
        tickers: [NVDA, AMD, AVGO, MRVL]
        etfs: [SMH, SOXX]
      Neocloud:
        tickers: [IREN, CIFR, CLSK, MARA]
        etfs: []
      Power_Cooling:
        tickers: [VRT, CEG]
        etfs: []
    Software:
      Platforms:
        tickers: [PLTR, SNOW, MDB, DDOG]
        etfs: []
      Cybersecurity:
        tickers: [CRWD, PANW, ZS, FTNT]
        etfs: [BUG, HACK]
  
  Space_Defense:
    Space:
      Launch:
        tickers: [RKLB, LUNR]
        etfs: [UFO]
      Satellites:
        tickers: [ASTS, PL, IRDM]
        etfs: []
    Defense:
      Primes:
        tickers: [LMT, RTX, NOC, GD]
        etfs: [ITA]
      Drones_Autonomy:
        tickers: [AVAV, KTOS]
        etfs: []
  
  Fintech_Crypto:
    Crypto:
      Bitcoin:
        tickers: []
        etfs: [IBIT, GBTC]
      Miners:
        tickers: [MARA, RIOT, CLSK]
        etfs: []
    Fintech:
      Brokerage:
        tickers: [HOOD, IBKR]
        etfs: [IAI]
      Payments:
        tickers: [SQ, PYPL, AFRM]
        etfs: []
```

### sources.yaml
```yaml
reddit:
  subreddits:
    - name: wallstreetbets
      weight: 1.0
      category: retail_sentiment
    - name: investing
      weight: 0.9
      category: general
    - name: stocks
      weight: 0.8
      category: general
    - name: options
      weight: 0.7
      category: derivatives
    - name: uraniumsqueeze
      weight: 1.5
      category: nuclear_focused
    - name: pennystocks
      weight: 0.5
      category: speculative
    - name: spacs
      weight: 0.6
      category: speculative

youtube:
  channels:
    - handle: "@InvestAnswers"
      weight: 1.2
    - handle: "@TheChartGuys"
      weight: 1.0

rss:
  feeds:
    - url: "https://seekingalpha.com/feed.xml"
      weight: 0.8
      name: SeekingAlpha
    - url: "https://www.zerohedge.com/fullrss2.xml"
      weight: 0.6
      name: ZeroHedge

twitter:
  enabled: false  # requires paid API
  lists: []
  accounts: []
```

---

## Questions to Ask Before Building

If the user hasn't specified, ask:

1. **Initial universe**: What 10-20 themes and 50-100 tickers to start with?
2. **Priority sources**: Which 5-10 Reddit subreddits matter most?
3. **Thresholds**: What signal score should trigger NOW vs BUILD vs WATCH?
4. **Time horizon**: Days/weeks (momentum) or months (position building)?
5. **Alert preference**: Few high-quality alerts or more signals to filter yourself?

---

## Implementation Order

1. **Day 1**: Repo structure + Reddit collector + basic entity extraction
2. **Day 2**: Velocity computation + sentiment scoring + signal model
3. **Day 3**: Streamlit Home Radar + Theme Drilldown
4. **Day 4**: Ticker Page + Source Control
5. **Day 5**: Alerts + Journal + polish

Each day should produce working, runnable code that builds on the previous day.
