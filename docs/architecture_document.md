# NSE Opportunity Radar — Architecture Document

**ET AI Hackathon 2026 · Problem Statement 6: AI for the Indian Investor**

---

## 1. System Overview

NSE Opportunity Radar is a 4-layer AI pipeline that converts raw NSE market data and corporate filings into ranked, plain-English investment signals for Indian retail investors.

```
NSE/BSE Data Sources (50 stocks · 92,456 filings)
        │
        ▼
┌─────────────────────────────────────────┐
│         Layer 1 — Data Pipeline         │
│  yfinance OHLCV + NSE Announcements API │
│  Output: 50 parquet files + CSV exports │
└────────────────────┬────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────────────┐  ┌─────────────────────────┐
│   Layer 2 — Day 2     │  │  Layer 3 — Day 3         │
│   Opportunity Agents  │  │  Pattern Intelligence    │
│                       │  │                          │
│  • Filing Analyzer    │  │  • Pattern Detector      │
│  • Anomaly Detector   │  │  • Backtester            │
│  • Signal Combiner    │  │  • Pattern Explainer     │
│                       │  │                          │
│  Groq AI ↑            │  │  Groq AI ↑               │
└──────────┬────────────┘  └────────────┬─────────────┘
           │                            │
           └──────────┬─────────────────┘
                      ▼
          ┌───────────────────────┐
          │  Layer 4 — Dashboard  │
          │  Streamlit · 3 Tabs   │
          └───────────────────────┘
```

---

## 2. Layer 1 — Data Pipeline

### 2.1 Stock Data (`utils/data_fetcher.py`)

- **Source:** `yfinance` API with `.NS` suffix for NSE tickers
- **Universe:** 50 NSE stocks
- **History:** 90-day OHLCV (Open, High, Low, Close, Volume)
- **Storage:** One parquet file per ticker at `data/stocks/{ticker}.NS.parquet`
- **Retry logic:** 3 attempts with exponential backoff — 1s → 2s → 4s
- **Constants:** `RETRY_LIMIT=3`, `INITIAL_BACKOFF=1`, `BACKOFF_MULTIPLIER=2`

### 2.2 Filing Scraper (`utils/filing_scraper.py`)

- **Source:** NSE Announcements API (`https://www.nseindia.com/api/corporate-announcements`)
- **Request params:** `index=equities`, `series=EQ`, `corpAction=all`
- **Anti-blocking:** Cookie preflight to `nseindia.com` (10s timeout) before API call (15s timeout)
- **Lookback:** 90 days per symbol
- **Deduplication:** By `(company_name, filing_date, url)`
- **Fallback:** Synthetic rows generated if API returns no data for a symbol
- **Output:** `data/bse_filings.csv` — 92,456 rows, 6 columns

### 2.3 Data Exporter (`utils/data_exporter.py`)

Consolidates all parquet files into two CSV exports:

| Output File | Rows | Description |
|---|---|---|
| `combined_stock_data.csv` | 4,499 | All OHLCV data across 50 stocks |
| `stock_summary_with_filings.csv` | 50 | One row per ticker with 17 summary columns |

**Summary columns:** `latest_close`, `latest_volume`, `avg_close`, `price_change_pct`, `trading_days`, `filing_count`, `latest_filing_type`, `latest_filing_date` + more.

---

## 3. Layer 2 — Opportunity Radar Agents (Day 2)

### 3.1 Filing Analyzer (`agents/filing_analyzer.py`)

Classifies each NSE corporate filing using Groq AI.

**Model config:**
```
Model:       llama-3.1-8b-instant
Temperature: 0.2
Max tokens:  300
Timeout:     20s
Retries:     3 (delay: 1.5s)
Coverage:    max 3 filings/ticker · max 180 total
```

**Output schema per filing:**
```json
{
  "signal":     "bullish | bearish | neutral",
  "confidence": 0.0 to 1.0,
  "reason":     "one sentence plain English",
  "event_type": "earnings | insider_trade | acquisition | management_change | regulatory | other",
  "urgency":    "high | medium | low"
}
```

**Validation:** Signal clamped to `{bullish, bearish, neutral}`. Confidence clamped to `[0.0, 1.0]`. Malformed JSON safely skipped.

**Output:** `data/filing_signals.csv` — 147 rows, 8 columns

---

### 3.2 Anomaly Detector (`agents/anomaly_detector.py`)

Scans OHLCV for unusual market activity using pure pandas.

**Detection conditions (flag if ANY met):**

| Condition | Threshold |
|---|---|
| Volume spike | `volume_ratio > 2.0` |
| Price move | `abs(price_change) > 3.0%` |
| RSI overbought | `RSI(14) > 70` |
| RSI oversold | `RSI(14) < 30` |

**Anomaly strength formula:**
```
volume_component = clamp((volume_ratio - 1.0) / 3.0,      0, 1)
price_component  = clamp(abs(price_change) / 10.0,         0, 1)
rsi_component    = clamp((abs(rsi - 50.0) - 20.0) / 30.0, 0, 1)
anomaly_strength = 0.45 × volume + 0.35 × price + 0.20 × rsi
```

**Output:** `data/anomaly_signals.csv` — 19 rows, 7 columns

---

### 3.3 Signal Combiner (`agents/signal_combiner.py`)

Merges filing and anomaly signals into a single ranked score.

**Scoring formula:**
```
final_score = 0.65 × filing_component
            + 0.10 × urgency_component
            + 0.25 × anomaly_component

filing_component:  (bullish=+1, bearish=-1, neutral=0) × confidence
urgency_component: high=0.30, medium=0.15, low=0.00
anomaly_component: anomaly_strength (0–1) or fallback heuristic
```

**Output:** `data/top_signals.csv` — 50 rows, 6 columns (includes `score_breakdown` for explainability)

---

## 4. Layer 3 — Chart Pattern Intelligence (Day 3)

### 4.1 Pattern Detector (`agents/pattern_detector.py`)

Detects 5 technical patterns across all 50 stocks using pure pandas.

| Pattern | Signal | Logic | Min Bars |
|---|---|---|---|
| Golden Cross | Bullish | SMA50 crosses above SMA200 in last 5 days | 200 |
| Death Cross | Bearish | SMA50 crosses below SMA200 in last 5 days | 200 |
| Volume Breakout | Bullish | Close > 20-day high + volume ≥ 1.5x avg (last 3 days) | 20 |
| RSI Divergence | Bullish | Price lower low + RSI higher low in last 14 days | 14 |
| Support Bounce | Bullish | Price within 2% of 20-day low, closes higher (last 5 days) | 20 |

**Pattern output fields:** `ticker`, `pattern_name`, `signal`, `pattern_date`, `key_values`, `strength (0–1)`, `current_price`

### 4.2 Backtester (`agents/pattern_detector.py → backtest_pattern()`)

For each detected pattern, calculates historical success rate on that specific stock.

```
Lookback:       730 days of history
Success cond:   price up +2% within 15 trading days
Min events:     3 (returns None if fewer)
Volume filter:  exclude events where volume < 0.8x 20-day avg
```

### 4.3 Pattern Explainer (`agents/pattern_explainer.py`)

Generates plain-English explanation per pattern using Groq AI.

```
Model:       llama-3.1-8b-instant
Rate limit:  2s between calls
Format:      exactly 3 sentences · max 80 words
```

### 4.4 Pattern Ranking (`agents/pattern_scanner.py`)

```
# When backtest data is available (≥ 3 occurrences):
rank_score = 0.3 × strength + 0.7 × success_rate

# When backtest unavailable (< 3 occurrences):
rank_score = strength
           + bullish_bonus (0.25) or bearish_bonus (0.05)
           + strength_bonus (0.10 if strength ≥ 0.8 else 0.05)
           capped at 1.0
```

**Outputs:**
- `data/pattern_signals.csv` — 66 rows, 7 columns
- `data/pattern_explanations.csv` — 66 rows, 7 columns

---

## 5. Layer 4 — Streamlit Dashboard

**Entry point:** `streamlit run dashboard/app.py`
**URL:** `http://localhost:8501`

### Tab 1 — Opportunity Radar
- Top ranked opportunities table (from `top_signals.csv`)
- Filing signals with signal/confidence filters (min confidence slider default 0.5)
- Top 10 volume anomaly cards with AI explanations
- Live NSE filings feed (search by company name or ticker)
- Clickable external filing URLs

### Tab 2 — Chart Patterns
- Pattern summary metrics (count by pattern type)
- Pattern cards sorted by success rate / strength
- Plotly candlestick chart with SMA50 + SMA200 overlays
- RSI panel with overbought (70) / oversold (30) zones
- Success rate bar chart across NSE universe

### Tab 3 — Stock Deep Dive
- Search any of 50 NSE tickers
- Current price, 52W high/low, volume vs 30-day avg
- Full 90-day candlestick + RSI chart
- All signals (filing + anomaly + pattern) per stock
- AI-generated outlook summary with key price levels
- OHLCV CSV download

### Dashboard Controls
- **Refresh Data** — runs `python main.py` (timeout: 600s)
- **Clear Cache** — clears `st.cache_data`
- **KPI threshold** — `HIGH_CONFIDENCE_SUCCESS_THRESHOLD = 0.80`

---

## 6. Data Schemas

### `bse_filings.csv`
`ticker · company_name · filing_date · filing_type · description · url`

### `filing_signals.csv`
`ticker · company_name · filing_date · signal · confidence · reason · event_type · urgency`

### `anomaly_signals.csv`
`ticker · anomaly_type · volume_ratio · price_change · rsi · anomaly_strength · explanation`

### `top_signals.csv`
`ticker · final_score · filing_signal · anomaly_type · top_reason · score_breakdown`

### `pattern_signals.csv`
`ticker · pattern_name · signal · pattern_date · key_values · strength · current_price`

### `pattern_explanations.csv`
`ticker · pattern_name · signal · strength · success_rate · current_price · explanation`

---

## 7. Technology Choices

| Layer | Technology | Reason |
|---|---|---|
| Stock data | `yfinance` | Free, real NSE OHLCV |
| Filing data | NSE API + browser headers | Real announcements, not static RSS |
| AI | Groq (`llama-3.1-8b-instant`) | 14,400 free req/day, <1s response |
| Technical analysis | Pure `pandas` | No ta-lib dependency |
| Storage | `.parquet` + CSV | Fast per-ticker queries |
| Dashboard | Streamlit + Plotly | Rapid development, interactive charts |

---

## 8. Security Notes

- `.env` is excluded from git via `.gitignore`
- API key is loaded via `python-dotenv` — never hardcoded
- NSE API uses read-only public endpoints — no authentication required
- No user data is stored or transmitted

---

*NSE Opportunity Radar · ET AI Hackathon 2026 · Problem Statement 6*