# 📡 NSE Opportunity Radar
### AI-Powered Signal Intelligence for Indian Retail Investors

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/AI-Groq%20llama--3.1--8b--instant-F55036?style=flat)](https://console.groq.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

> **ET AI Hackathon 2026 — Problem Statement 6: AI for the Indian Investor**

---

## 🎯 The Problem

India has **14 crore+ demat account holders** — but most retail investors are flying blind.

| Pain Point | Reality |
|---|---|
| Corporate filings | Missed order wins, earnings beats, management changes |
| Technical charts | Cannot read patterns — buy at tops, sell at bottoms |
| Volume anomalies | Miss institutional accumulation signals |
| Information overload | Filings, charts, news — all in separate places |

Retail investors need **one system** that converts filings + market behavior into clear, ranked, actionable opportunities — in near real time.

---

## ✅ Our Solution

NSE Opportunity Radar is a **3-agent AI pipeline** that continuously monitors **50 NSE stocks** and **92,456+ corporate filings**, detects opportunities using ML and technical analysis, and surfaces them as plain-English, ranked signals through an interactive Streamlit dashboard.

**Not a summarizer. A signal-finder.**

---

## 🏗️ Architecture

```
NSE/BSE Data Sources (50 stocks · 92,456 filings)
        │
        ▼
┌─────────────────────────────────────────┐
│         Day 1 — Data Pipeline           │
│  yfinance OHLCV + NSE Announcements API │
│  Output: 50 parquet files + CSV exports │
└────────────────────┬────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────────────┐  ┌─────────────────────────┐
│   Day 2 — Agents      │  │  Day 3 — Patterns        │
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
          │  Streamlit Dashboard  │
          │  • Opportunity Radar  │
          │  • Chart Patterns     │
          │  • Stock Deep Dive    │
          └───────────────────────┘
```

---

## 📦 What Gets Generated

| File | Rows | Description |
|---|---|---|
| `data/bse_filings.csv` | 92,456 | Real NSE corporate announcements |
| `data/filing_signals.csv` | 147 | AI-classified filing signals |
| `data/anomaly_signals.csv` | 19 | Volume / RSI / price anomalies |
| `data/top_signals.csv` | 50 | Final ranked opportunities |
| `data/pattern_signals.csv` | 66 | Detected chart patterns |
| `data/pattern_explanations.csv` | 66 | AI explanations + success rates |
| `data/stocks/*.parquet` | 50 files | 90-day OHLCV per ticker |

---

## 🤖 Agent Details

### Agent 1 — Filing Analyzer
Reads NSE corporate filings and classifies each as bullish/bearish/neutral using Groq AI.

| Parameter | Value |
|---|---|
| Model | `llama-3.1-8b-instant` |
| Max filings per ticker | 3 |
| Max total filings | 180 |
| Temperature | 0.2 |
| Timeout | 20s with 3 retries (1.5s delay) |

Output schema:
```
signal:     bullish | bearish | neutral
confidence: 0.0 – 1.0
reason:     one sentence plain English
event_type: earnings | insider_trade | acquisition | management_change | regulatory | other
urgency:    high | medium | low
```

---

### Agent 2 — Anomaly Detector
Scans OHLCV data for unusual volume, sharp price moves, and RSI extremes.

Flags a stock when **any** of these conditions are met:

| Condition | Threshold |
|---|---|
| Volume spike | `volume_ratio > 2.0` |
| Price move | `abs(price_change) > 3.0%` |
| RSI overbought | `RSI > 70` |
| RSI oversold | `RSI < 30` |

Anomaly strength formula:
```
volume_component = clamp((volume_ratio - 1.0) / 3.0,       0, 1)
price_component  = clamp(abs(price_change) / 10.0,          0, 1)
rsi_component    = clamp((abs(rsi - 50.0) - 20.0) / 30.0,  0, 1)
anomaly_strength = 0.45×volume + 0.35×price + 0.20×rsi
```

---

### Agent 3 — Signal Combiner
Merges filing and anomaly signals into a single ranked score with full explainability.

```
final_score = 0.65 × filing_component
            + 0.10 × urgency_component
            + 0.25 × anomaly_component

filing_component:  bullish=+1, bearish=-1, neutral=0  ×  confidence
urgency_component: high=0.30, medium=0.15, low=0.00
```

---

### Agent 4 — Chart Pattern Intelligence
Detects 5 technical patterns across all 50 stocks with backtested success rates.

| Pattern | Signal | Detection Logic |
|---|---|---|
| Golden Cross | 🟢 Bullish | SMA50 crosses above SMA200 — last 5 days (needs 200 bars) |
| Death Cross | 🔴 Bearish | SMA50 crosses below SMA200 — last 5 days (needs 200 bars) |
| Volume Breakout | 🟢 Bullish | Close > 20-day high + volume ≥ 1.5x average (last 3 days) |
| RSI Divergence | 🟢 Bullish | Price lower low + RSI higher low in last 14 days |
| Support Bounce | 🟢 Bullish | Price within 2% of 20-day low, closes higher (last 5 days) |

**Backtesting logic:**
- Looks back up to 730 days of history per stock
- Success = price up **+2% within 15 trading days** after pattern
- Requires minimum 3 historical occurrences — otherwise returns `None`
- Excludes events where volume < 0.8x 20-day average

**Pattern ranking score:**
```
# When backtest data exists:
rank_score = 0.3 × strength + 0.7 × success_rate

# When backtest unavailable (< 3 occurrences):
rank_score = strength + bullish_bonus(0.25) + strength_bonus(0.10 if ≥0.8 else 0.05)
             capped at 1.0
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nse-opportunity-radar.git
cd nse-opportunity-radar

# 2. Create and activate virtual environment
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
echo GROQ_API_KEY=your_key_here > .env

# 5. Run full pipeline (Day 1 + 2 + 3)
python main.py

# 6. Launch dashboard
streamlit run dashboard/app.py
```

Dashboard opens at **http://localhost:8501**

---

## 📁 Project Structure

```
nse-opportunity-radar/
├── data/
│   ├── stocks/                    # 50 parquet files (90-day OHLCV)
│   ├── bse_filings.csv            # 92,456 NSE corporate announcements
│   ├── filing_signals.csv         # 147 AI-classified filing signals
│   ├── anomaly_signals.csv        # 19 detected anomalies
│   ├── top_signals.csv            # 50 final ranked opportunities
│   ├── pattern_signals.csv        # 66 detected chart patterns
│   └── pattern_explanations.csv   # 66 AI explanations + success rates
│
├── agents/
│   ├── filing_analyzer.py         # Groq AI filing classification
│   ├── anomaly_detector.py        # RSI, volume, price anomaly detection
│   ├── signal_combiner.py         # Weighted signal ranking
│   ├── pattern_detector.py        # 5 chart patterns + backtester
│   ├── pattern_explainer.py       # Groq AI pattern explanations
│   └── pattern_scanner.py         # Master pattern orchestrator
│
├── utils/
│   ├── data_fetcher.py            # yfinance OHLCV (retry: 3x, 1s→2s→4s)
│   ├── filing_scraper.py          # NSE announcements scraper
│   ├── data_exporter.py           # CSV export utilities
│   └── groq_client.py             # Shared Groq client
│
├── dashboard/
│   ├── app.py                     # Streamlit main app
│   └── components.py              # Reusable UI components
│
├── tests/
│   ├── test_day2_agents.py
│   └── test_day3_patterns.py
│
├── main.py                        # Full pipeline orchestrator
├── requirements.txt
└── .env                           # GROQ_API_KEY (never committed)
```

---

## 📈 Tracked Stock Universe (50 Tickers)

```
RELIANCE    INFY        TCS         HDFCBANK    ICICIBANK   LT          WIPRO
MARUTI      SUNPHARMA   BAJAJFINSV  BHARTIARTL  ASIANPAINT  JSWSTEEL    TATASTEEL
BPCL        ONGC        INDIGO      AXISBANK    SBIN        BAJAJ-AUTO  EICHERMOT
HINDALCO    APOLLOHOSP  NTPC        POWERGRID   M&M         HEROMOTOCO  ULTRACEMCO
ADANIPORTS  ADANIENT    ITC         BRITANNIA   YESBANK     UPL         NESTLEIND
GRASIM      HCLTECH     TECHM       TITAN       SBICARD     GODREJCP    AUROPHARMA
TATAMOTORS  KOTAKBANK   HINDUNILVR  CIPLA       DRREDDY     COALINDIA   BEL
SHRIRAMFIN
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Stock Data | `yfinance` — real NSE OHLCV, 90-day history |
| Filing Data | NSE Announcements API with browser-simulated headers |
| AI / NLP | Groq API (`llama-3.1-8b-instant`) — free, <1s response |
| Technical Analysis | Pure `pandas` — manual RSI(14), SMA, rolling windows |
| Storage | Per-ticker `.parquet` files + CSV exports |
| Dashboard | `Streamlit` + `Plotly` (dark theme, interactive charts) |
| Config | `python-dotenv` + `pathlib` (cross-platform) |

---

## 💰 Business Impact Model

```
India demat accounts:         14 crore+   (SEBI / NSDL 2024)
Target users (Year 1):         1 lakh retail investors
Avg deployable capital:       ₹50,000 per investor
Return improvement vs random:  3% annually

  1,00,000 × ₹50,000 × 0.03 = ₹1,50,00,00,000

  ≈ ₹150 crore additional value created annually
```

**Why 3% is a conservative estimate:**
- Average backtested pattern success rate: ~67%
- Random baseline: 50%
- Relative improvement: **34% better decision quality**
- Compounded over 12+ investment decisions per year = measurable alpha for retail investors

---

## 🧪 Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run Day 2 agent tests only
python -m unittest tests/test_day2_agents.py -v

# Run Day 3 pattern tests only
python -m unittest tests/test_day3_patterns.py -v
```

---

## ⚠️ Troubleshooting

| Error | Fix |
|---|---|
| `GROQ_API_KEY invalid` | Check `.env` — no quotes, no spaces around `=` |
| `Rate limit 429` | Daily quota hit — wait until midnight or create new Groq project |
| `No data found for ticker` | Run `python main.py` first to generate parquet files |
| `Missing CSV files` | Run agents individually: `python -m agents.filing_analyzer` |
| `yfinance delisted error` | Normal — some tickers renamed/delisted, safely skipped |
| `NSE API timeout` | NSE blocks repeated requests — filing scraper has cookie preflight built in |

---

## 📌 Operational Notes

- Filing availability depends on NSE API reachability per symbol
- Some ranking rows are anomaly-only when no filing signal exists for that ticker
- Pattern backtest returns `None` for stocks with fewer than 3 historical occurrences — ranking uses technical conviction score instead
- Dashboard Refresh button runs `python main.py` with a 600s timeout
- `requirements.txt` has 294 pinned packages — use a fresh venv to avoid conflicts

---

## 📄 Documentation

- [Architecture Document](docs/architecture_document.md) — full agent design and data schemas
- [Architecture Diagram](docs/architecture_diagram.svg) — system flow visual

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for ET AI Hackathon 2026 · Problem Statement 6: AI for the Indian Investor**

*Powered by Groq AI · Python · Streamlit · Real NSE Data*

</div>