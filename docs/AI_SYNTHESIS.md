# AI Conversation Synthesis: What Each Model Contributed

This document captures the key insights from iterating on the Rotation Radar concept across multiple AI assistants.

---

## Claude (Anthropic) - Initial Architecture

**What Claude provided:**
- Basic tech stack recommendation (Python + SQLite + Streamlit)
- GitHub repos to fork (Reddit-Stock-Sentiment-Analyzer, stocksight, etc.)
- Three-phase build plan (MVP → Edge Features → Intelligence)
- Core signal concepts (velocity, sentiment, cross-platform)

**What was weak:**
- Generic "plumbing" without a real signal model
- No decision framework (just data, no NOW/BUILD/WATCH)
- "Streamlit dashboard" as UI spec (too vague)
- No noise defense (bot filtering, author weighting)

---

## ChatGPT (OpenAI) - Object Model & Interface Spec

**Key contributions:**
1. **Object Model**: Defined the missing backbone
   - Source → Author → Document → Entity → Theme → Narrative → Signal → Decision
   - This hierarchy is what turns raw data into actionable insights

2. **Signal Model with Gates**:
   - Explicit scoring formula with weights
   - Three gates: Breadth, Quality, Pump Filter
   - Distinguishes Heat Score (loud) vs Edge Score (early)

3. **Phase Classification**:
   - Ignition → Acceleration → Crowded → Exhaustion → Cooling
   - Maps to decision labels: NOW / BUILD / WATCH / IGNORE

4. **Interface Vision (5 Screens)**:
   - Home Radar with "What Changed" feed
   - Theme Drilldown with Narrative Cards
   - Ticker Page with Divergence panel
   - Source Control with Alpha Leaderboard
   - Alerts + Journal for learning

5. **Time Machine View**:
   - First mention → Inflection → Peak → Now
   - "Who was early" tracking for source alpha

6. **Tripwires**:
   - Every signal needs an invalidation condition
   - Examples: "If NRC delays >6mo...", "If earnings decelerate while capex elevated..."

---

## Gemini (Google) - "Rotation Intelligence" Framing

**Key contributions:**
1. **Reframing**: From "Sentiment Dashboard" to "Rotation Intelligence System"

2. **Velocity of Narrative**: Not just "how many talking" but "how fast spreading to new circles"

3. **Cross-Platform Arbitrage**: 
   - Reddit screaming + X silent = different signal than both aligned
   - Explicit confirmation logic

4. **Divergence as Holy Grail**:
   - High Sentiment + Flat Price = Accumulation Zone
   - High Sentiment + Ripping Price = Late/Euphoria

5. **Z-Score Velocity**: 
   - Track standard deviations from 30-day mean
   - If chatter is >2.5σ above normal, it's a rotation not a fluke

6. **Ensemble Scoring Suggestion**:
   - Layer multiple models: Llama for context + FinBERT for tone + Narrative Classification

---

## Grok (xAI) - Signal Confirmation Logic

**Key contributions:**
1. **Whale vs Retail Weighting**:
   - Verified sector experts should be weighted 10x higher
   - Behavioral credibility scoring for authors

2. **Automated Divergence Heatmap**:
   - Visual grid showing Tickers where Sentiment ↑ but Price ↔
   - Quick scan for opportunities

3. **Catalyst Type Tagging**:
   - Fed Policy, Tech Breakthrough, Supply Chain Shock
   - Helps explain "why" behind sentiment moves

4. **Interface Questions** (for refinement):
   - "Panic" vs "Opportunity" view?
   - Curation depth (show top posts or just numbers)?
   - Time horizon (day trading vs position building)?
   - Divergence notification method?

---

## Synthesis: The Complete Vision

Combining all inputs, the ideal system is:

### Core Principle
**"What's moving from fringe → consensus, and is price confirming yet?"**

### Data Flow
```
Curated Sources (weighted) 
    → Entity Extraction (themes, tickers, catalysts)
    → Velocity + Breadth + Sentiment computation
    → Cross-platform confirmation
    → Price divergence check
    → Gate filtering (breadth, quality, pump)
    → Phase classification
    → Decision label (NOW/BUILD/WATCH/IGNORE)
    → Narrative card + tripwire generation
    → Alert if thresholds crossed
```

### Two Scores That Matter
1. **Heat Score**: What's loud now (for FOMO awareness)
2. **Edge Score**: What's early (for alpha generation)

### Phase Lifecycle
`Ignition → Acceleration → Crowded → Exhaustion → Cooling`

### Output for Each Theme/Ticker
- Signal Score (0-100)
- Phase badge
- Decision label
- "Why" explanation (top 3 drivers)
- Best source links
- Tripwire / invalidation condition

### Unique Features vs Generic Dashboards
1. **"What Changed" feed** - not static view, shows deltas
2. **Time Machine** - when was first mention, who was early
3. **Source Alpha** - track which sources actually predict moves
4. **Tripwires** - every signal has an exit condition
5. **Journal** - learn from your own trades over time

---

## Build Priority Order

Based on the synthesis, the most valuable features to build first:

1. **Reddit ingestion + Entity extraction** - data foundation
2. **Velocity + Unique Authors** - the core "early signal" metrics
3. **Home Radar with "What Changed"** - daily usability
4. **Divergence computation** - the money-maker
5. **Phase classification** - decision framework
6. **Theme Drilldown** - dig into signals
7. **Alerts** - don't miss opportunities
8. **Journal** - continuous improvement

---

## What's NOT Worth Building (Yet)

- Real-time streaming (daily is fine for rotation trading)
- Complex ML models (simple rules + FinBERT is enough)
- Mobile app (web works fine)
- Multi-user auth (single user first)
- Twitter integration (Reddit alone has signal, X is expensive)
