# Product Specification: Rotation Radar

## 1. Problem Statement

Retail investors and small family offices lack access to the sentiment intelligence tools used by hedge funds. Existing solutions are either:
- **Too basic**: Simple mention counts without signal quality
- **Too noisy**: No bot filtering, author weighting, or cross-platform confirmation
- **Wrong outputs**: Show "what's loud" not "what's early"
- **No decision framework**: Data without actionable labels

**Goal**: Build a single-user "Sector Rotation Radar" that converts cross-platform social + headline chatter into ranked, explainable narrative signals with clear decision outputs.

---

## 2. Core Entities (Data Model)

```
Source
├── id: uuid
├── type: enum(reddit, twitter, youtube, rss, stocktwits)
├── identifier: string (subreddit name, account handle, etc.)
├── weight: float (user-assigned credibility)
├── alpha_score: float (computed: how early does this source spot moves?)
└── last_fetched: timestamp

Author
├── id: uuid
├── source_id: fk
├── username: string
├── account_age_days: int
├── credibility_score: float (behavioral)
├── is_whale: boolean (verified sector expert)
└── is_suspected_bot: boolean

Document
├── id: uuid
├── source_id: fk
├── author_id: fk
├── content: text
├── url: string
├── published_at: timestamp
├── fetched_at: timestamp
├── engagement: jsonb (upvotes, comments, retweets, etc.)
└── raw_metadata: jsonb

Entity
├── id: uuid
├── type: enum(ticker, theme, sub_theme, catalyst, person, company)
├── symbol: string (e.g., "NVDA", "nuclear_energy")
├── display_name: string
├── parent_id: fk (for hierarchical themes)
└── metadata: jsonb

DocumentEntity (junction)
├── document_id: fk
├── entity_id: fk
├── confidence: float
└── sentiment_toward: float (-1 to 1)

Narrative
├── id: uuid
├── title: string (auto-generated or user-edited)
├── claim_summary: text
├── catalyst_tags: array[string]
├── bear_case: text
├── tripwires: array[string]
├── entity_ids: array[uuid]
├── document_ids: array[uuid]
├── created_at: timestamp
└── last_active: timestamp

Signal
├── id: uuid
├── entity_id: fk
├── computed_at: timestamp
├── window: enum(1h, 6h, 24h, 7d)
├── velocity_z: float
├── unique_authors_z: float
├── sentiment_delta_z: float
├── cross_platform_score: float
├── divergence_score: float
├── catalyst_score: float
├── heat_score: float (composite: what's loud)
├── edge_score: float (composite: what's early)
├── phase: enum(ignition, acceleration, crowded, exhaustion, cooling)
├── decision: enum(now, build, watch, ignore)
└── explanation: text

Alert
├── id: uuid
├── type: enum(rotation_ignition, divergence_edge, overheat_late)
├── entity_id: fk
├── signal_id: fk
├── triggered_at: timestamp
├── message: text
├── acknowledged: boolean
└── action_taken: text (user journal entry)
```

---

## 3. Signal Model

### 3.1 Feature Set

For each Theme/Ticker, compute rolling metrics over multiple windows (1h, 6h, 24h, 7d):

**Volume & Breadth**
- `mention_count`: raw count
- `mention_velocity`: Δmentions / Δtime
- `mention_acceleration`: Δvelocity / Δtime
- `unique_authors`: distinct authors in window
- `unique_authors_velocity`: new unique posters vs baseline
- `concentration`: % of mentions from top 5 accounts (penalty if >70%)

**Sentiment & Trajectory**
- `sentiment_raw`: weighted average sentiment (-1 to 1)
- `sentiment_delta`: sentiment change vs 7d baseline
- `conviction_score`: presence of high-conviction language ("breakout", "inevitable", "adding")
- `euphoria_flag`: meme language + extreme bullishness

**Cross-Platform Confirmation**
- `platforms_active`: count of platforms with mentions
- `cross_platform_velocity`: are multiple platforms accelerating together?
- `confirmation_score`: weighted average across platforms

**Chatter-Price Divergence**
- `price_return_window`: % price change over same window
- `divergence_score`: chatter_z - price_z (high positive = opportunity)
- `extension_flag`: price extended beyond 2σ from 20d mean

**Catalyst Tagging**
- `catalyst_tags`: array of detected catalysts
- `catalyst_score`: weighted relevance of active catalysts

### 3.2 Scoring Formula

```python
signal_score = (
    0.30 * velocity_z +
    0.20 * unique_authors_z +
    0.15 * sentiment_delta_z +
    0.15 * cross_platform_score +
    0.10 * divergence_score +
    0.10 * catalyst_score
)
```

### 3.3 Gating Logic

Scores are modified/capped based on gates:

**Gate A: Breadth**
- PASS if: `unique_authors_z > 0` OR `cross_platform_score > 0.5`
- FAIL: cap score at 30

**Gate B: Quality**
- Compute source-weighted score: sum(mention × source.weight) / sum(mention)
- PASS if: weighted_score > threshold
- FAIL: apply 0.7x multiplier

**Gate C: Pump Filter**
- Triggered if: `concentration > 70%` AND `avg_account_age < 30d` AND `euphoria_flag`
- Action: cap score at 50, add "Speculative" tag

### 3.4 Heat vs Edge

```python
heat_score = (
    0.50 * mention_count_z +
    0.30 * sentiment_raw +
    0.20 * engagement_z
)

edge_score = (
    0.35 * unique_authors_velocity_z +
    0.25 * divergence_score +
    0.20 * cross_platform_acceleration +
    0.20 * (1 - concentration)  # reward broad participation
)
```

### 3.5 Phase Classification

```python
def classify_phase(entity_metrics):
    if velocity_z < 0.5 and baseline_mentions < 10:
        return "dormant"
    elif velocity_z > 1.0 and unique_authors_z > 0.5 and cross_platform < 2:
        return "ignition"
    elif velocity_z > 1.5 and cross_platform >= 2 and divergence_score > 0:
        return "acceleration"
    elif velocity_z > 2.0 and euphoria_flag and extension_flag:
        return "crowded"
    elif velocity_acceleration < 0 and sentiment_delta < 0:
        return "exhaustion"
    else:
        return "cooling"
```

### 3.6 Decision Labels

```python
def assign_decision(signal, phase):
    if phase == "acceleration" and signal.divergence_score > 0.5:
        return "NOW"
    elif phase in ["ignition", "acceleration"] and signal.edge_score > 60:
        return "BUILD"
    elif phase == "ignition" or signal.edge_score > 40:
        return "WATCH"
    else:
        return "IGNORE"
```

---

## 4. User Interface Specification

### Screen 1: Radar (Home)

**Purpose**: Snapshot of what's hot + what's changing + what's viral

**Layout**:
```
┌─────────────────────────────────────────────────────────────────┐
│ [Regime Tags: Risk-On ⚡] [Universe: Themes ▼] [Window: 7d ▼]   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WHAT'S HOT (Heatmap - scrollable, not capped at 12)            │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐            │
│  │ Nuclear │ AI Infra│ Space   │ Crypto  │ Defense │ ...        │
│  │ 🔥 87   │ 🔥 82   │ ⚡ 71   │ ⚡ 68   │ 📈 65   │            │
│  │ ACCEL   │ ACCEL   │ IGNIT   │ CROWD   │ BUILD   │            │
│  └─────────┴─────────┴─────────┴─────────┴─────────┘            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  WHAT CHANGED SINCE YESTERDAY                                   │
│  • 🆕 "SMR Builders" entered Top 25 (was #42)                   │
│  • 🔄 "Grid Storage" phase flip: Ignition → Acceleration        │
│  • ⚠️ "Quantum" overheat: concentration 78%, euphoria detected  │
│  • 📊 "Cybersecurity" divergence: chatter ↑35%, price ↓2%       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  TOP NARRATIVES                                                 │
│  1. "AI power bottleneck → grid buildout trade"                 │
│     Velocity: ↑↑ | Breadth: 3 platforms | Quality: High         │
│  2. "Nuclear restarts as data center solution"                  │
│     Velocity: ↑ | Breadth: 2 platforms | Quality: High          │
│  3. "Defense budget expansion under new admin"                  │
│     Velocity: → | Breadth: 2 platforms | Quality: Medium        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  VIRAL NOW                                                      │
│  📱 Reddit: "OKLO DD - why SMRs are the..." (r/investing, 847↑) │
│  📺 YouTube: "Nuclear Renaissance..." (@InvestAnswers, 45k views)│
│  📰 News: "Microsoft signs nuclear deal..." (Reuters)           │
└─────────────────────────────────────────────────────────────────┘
```

**Interactions**:
- Click theme tile → Theme Drilldown
- Click "What Changed" item → expand score breakdown + top posts
- Click narrative → Narrative Card Stack
- Click viral item → open source link

### Screen 2: Theme Drilldown

**Purpose**: Deep dive into a specific theme with trend lines and narrative cards

**Layout**:
```
┌─────────────────────────────────────────────────────────────────┐
│ ← Back to Radar          NUCLEAR ENERGY           [Export PDF]  │
├─────────────────────────────────────────────────────────────────┤
│ Signal: 87 | Phase: ACCELERATION | Decision: NOW                │
│ "Why": Velocity +2.3σ, cross-platform confirmed, divergence +0.6│
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  METRICS (7d view, toggle: 1d/7d/30d/90d)                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ [Mentions Chart]  [Unique Authors]  [Sentiment Delta]      │ │
│  │ [Cross-Platform]  [Chatter vs Price RS]                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  SUB-THEMES                                                     │
│  ┌──────────────┬────────────────┬───────────────┐              │
│  │ SMR Builders │ Fuel Cycle     │ Utilities     │              │
│  │ 🔥 91 NOW    │ 📈 78 BUILD    │ ⚡ 65 WATCH   │              │
│  └──────────────┴────────────────┴───────────────┘              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  NARRATIVE CARDS                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ "Data centers driving nuclear renaissance"                 │ │
│  │ Claim: AI power demand exceeds grid capacity, nuclear      │ │
│  │        is only scalable baseload solution                  │ │
│  │ Catalyst: Microsoft/Amazon/Google deals, DOE loan programs │ │
│  │ Who's pushing: r/uraniumsqueeze, @uraniuminsider, Substack │ │
│  │ Bear case: Regulatory delays, cost overruns, SMR timeline  │ │
│  │ Tripwires: If NRC approval delays >6mo, reassess           │ │
│  │ [View 7 sources]                                           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  TIME MACHINE                                                   │
│  First mention: Mar 2024 (r/investing) | Inflection: Sep 2024   │
│  Peak velocity: Nov 2024 | Now: +45% from inflection            │
│  Who was early: @uraniuminsider (lead time: 4mo), r/uranium     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  TICKERS                                                        │
│  ETFs: URNM (pure), URA (diversified)                           │
│  Singles: OKLO (NOW), SMR (BUILD), CCJ (BUILD), CEG (WATCH)     │
└─────────────────────────────────────────────────────────────────┘
```

### Screen 3: Ticker Page

**Purpose**: Individual ticker view with narratives affecting it and divergence analysis

**Layout**:
```
┌─────────────────────────────────────────────────────────────────┐
│ ← Back                    OKLO                    [Add to List] │
├─────────────────────────────────────────────────────────────────┤
│ Price: $24.50 (+3.2%)  |  52w: $8.10 - $28.40  |  Vol: 2.1M     │
│ Signal: 91  |  Phase: ACCELERATION  |  Decision: NOW            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CHATTER vs PRICE (divergence panel)                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ [Dual-axis chart: Social mentions (bars) vs Price (line)]  │ │
│  │ Divergence Score: +0.8 (Opportunity zone)                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  NARRATIVES AFFECTING THIS TICKER                               │
│  1. "SMRs as data center power solution" (91 signal)            │
│  2. "DOE loan program beneficiaries" (72 signal)                │
│  3. "Nuclear restarts momentum" (68 signal)                     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  TOP SOURCES MENTIONING OKLO                                    │
│  📱 r/wallstreetbets: "OKLO the next..." (324↑, 2h ago)         │
│  📱 r/uraniumsqueeze: "Sam Altman backed..." (189↑, 6h ago)     │
│  📺 @InvestAnswers: "Why OKLO could..." (12k views, 1d ago)     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  RELATIVE STRENGTH                                              │
│  vs SPY: +45% (30d)  |  vs XLU: +62% (30d)  |  vs URNM: +28%    │
└─────────────────────────────────────────────────────────────────┘
```

### Screen 4: Source Control

**Purpose**: Manage your curated sources and see which ones have alpha

**Layout**:
```
┌─────────────────────────────────────────────────────────────────┐
│                      SOURCE CONTROL                             │
├─────────────────────────────────────────────────────────────────┤
│ [+ Add Source]  [Import List]  [Export List]                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  REDDIT SUBREDDITS                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Source              │ Weight │ Alpha Score │ Last Active   │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │ r/wallstreetbets    │ [1.0▼] │ +12% (30d)  │ 2m ago        │ │
│  │ r/uraniumsqueeze    │ [1.5▼] │ +28% (30d)  │ 15m ago       │ │
│  │ r/investing         │ [0.8▼] │ +5% (30d)   │ 5m ago        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  TWITTER/X ACCOUNTS                                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ @uraniuminsider     │ [2.0▼] │ +35% (30d)  │ 1h ago        │ │
│  │ @chartguys          │ [1.2▼] │ +8% (30d)   │ 3h ago        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
│  YOUTUBE CHANNELS                                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ @InvestAnswers      │ [1.5▼] │ +15% (30d)  │ 1d ago        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  SOURCE ALPHA LEADERBOARD (who spots moves early?)              │
│  1. @uraniuminsider - avg lead time: 4.2 weeks                  │
│  2. r/uraniumsqueeze - avg lead time: 3.1 weeks                 │
│  3. @InvestAnswers - avg lead time: 2.8 weeks                   │
└─────────────────────────────────────────────────────────────────┘
```

### Screen 5: Alerts + Journal

**Purpose**: High-signal alerts and trade journaling for learning

**Layout**:
```
┌─────────────────────────────────────────────────────────────────┐
│                    ALERTS + JOURNAL                             │
├─────────────────────────────────────────────────────────────────┤
│  ACTIVE ALERTS (3)                               [Settings ⚙️]  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 🔥 ROTATION IGNITION: "Grid Storage" crossed NOW threshold │ │
│  │    Signal: 78 → 85 | Cross-platform confirmed | 2h ago     │ │
│  │    [View] [Acknowledge] [Journal Entry]                    │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │ 📊 DIVERGENCE EDGE: RKLB chatter ↑45%, price flat          │ │
│  │    Divergence score: +0.9 | Phase: Acceleration | 6h ago   │ │
│  │    [View] [Acknowledge] [Journal Entry]                    │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │ ⚠️ OVERHEAT: "Quantum Computing" showing exhaustion signs   │ │
│  │    Concentration: 82% | Euphoria flag | Extension flag     │ │
│  │    [View] [Acknowledge] [Journal Entry]                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  JOURNAL                                          [+ New Entry] │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Jan 15: Took starter in OKLO @ $22.50                      │ │
│  │   Signal was: 85 / NOW / Acceleration                      │ │
│  │   Thesis: SMR demand from data centers                     │ │
│  │   Tripwire: Exit if NRC delays >6mo                        │ │
│  │   Status: Open | P&L: +8.9%                                │ │
│  ├────────────────────────────────────────────────────────────┤ │
│  │ Jan 10: Passed on IONQ despite signal                      │ │
│  │   Signal was: 72 / BUILD / Ignition                        │ │
│  │   Reason: Euphoria flag, wanted to wait for pullback       │ │
│  │   Outcome: Up 15% since - missed it                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  BACKTEST SUMMARY (last 90 days)                                │
│  NOW signals: 12 triggered | 9 profitable (75%) | Avg: +18%     │
│  BUILD signals: 24 triggered | 15 profitable (63%) | Avg: +11%  │
│  Best source alpha: r/uraniumsqueeze (+35% avg on early calls)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Alert Types

Only 3 alert types (few, high quality):

| Type | Trigger Conditions | Priority |
|------|-------------------|----------|
| **Rotation Ignition** | Theme crosses into Top 25 + breadth gate passes + persistence >24h | High |
| **Divergence Edge** | Chatter z > 1.5 + price z < 0.5 + phase is Ignition/Acceleration | High |
| **Overheat/Late** | Concentration >70% + euphoria flag + price extension >2σ | Medium |

---

## 6. Technical Requirements

### Stack (MVP)
- **Language**: Python 3.11+
- **Database**: SQLite (upgradeable to PostgreSQL)
- **UI**: Streamlit
- **Scheduling**: cron or APScheduler
- **Sentiment**: FinBERT + optional LLM classification

### Data Refresh
- **Collection**: 4x daily (6am, 12pm, 6pm, 12am)
- **Processing**: After each collection
- **Alerts**: Real-time check after processing

### Storage Estimates
- ~1000 documents/day across sources
- ~50MB/month raw storage
- 12 months = ~600MB (trivial for SQLite)

---

## 7. Success Metrics

After 90 days, evaluate:

1. **Signal Quality**: What % of NOW signals led to positive 30d returns?
2. **Edge vs Heat**: Does Edge score outperform Heat score for early detection?
3. **Source Alpha**: Which sources consistently spot moves early?
4. **User Value**: How many times did alerts lead to trades that wouldn't have happened otherwise?

---

## 8. Future Enhancements

- Brokerage integration (Schwab API) for position tracking
- Position sizing recommendations based on signal strength
- Backtesting UI for historical signal evaluation
- Multi-user mode for family/small group
- Mobile app or Telegram bot for alerts
- LLM-generated daily briefings
