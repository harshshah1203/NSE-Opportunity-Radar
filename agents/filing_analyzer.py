"""
NSE Opportunity Radar - Filing Analyzer Agent
Analyzes NSE corporate filings using Groq API to generate trading signals.
"""

import time
import json
from json import JSONDecodeError
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from utils.groq_client import get_groq_client

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FILING_SIGNALS_CSV = DATA_DIR / "filing_signals.csv"

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_TIMEOUT_SECONDS = 20.0
GROQ_MAX_RETRIES = 3
GROQ_RETRY_DELAY_SECONDS = 1.5
OUTPUT_COLUMNS = [
    "ticker",
    "company_name",
    "filing_date",
    "signal",
    "confidence",
    "reason",
    "event_type",
    "urgency",
]
VALID_SIGNALS = {"bullish", "bearish", "neutral"}
VALID_URGENCY = {"high", "medium", "low"}

client = get_groq_client()


def _resolve_ticker(row: pd.Series) -> str:
    """
    Resolve ticker from filing row; fallback to company-name inference.
    """
    ticker = str(row.get("ticker", "")).strip().upper()
    if ticker and ticker not in {"NAN", "NONE"}:
        return ticker.replace(".NS", "")

    company_name = str(row.get("company_name", "")).strip()
    if not company_name:
        return ""

    try:
        from utils.filing_scraper import infer_ticker_from_company_name
        return infer_ticker_from_company_name(company_name)
    except Exception:
        return ""


def _extract_json_object(text: str) -> Optional[Dict]:
    """
    Parse the first JSON object from model text response.
    """
    cleaned = text.strip()

    # Handle common markdown response wrappers.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def _validate_filing_analysis(result: Dict) -> Optional[Dict]:
    """
    Validate and normalize filing-analysis JSON schema.
    """
    required = ["signal", "confidence", "reason", "event_type", "urgency"]
    if not all(key in result for key in required):
        return None

    signal = str(result["signal"]).strip().lower()
    urgency = str(result["urgency"]).strip().lower()
    event_type = str(result["event_type"]).strip().lower()
    reason = str(result["reason"]).strip()

    try:
        confidence = float(result["confidence"])
    except (ValueError, TypeError):
        return None

    if signal not in VALID_SIGNALS:
        return None
    if urgency not in VALID_URGENCY:
        return None
    if not reason:
        return None

    confidence = max(0.0, min(confidence, 1.0))

    return {
        "signal": signal,
        "confidence": round(confidence, 3),
        "reason": reason,
        "event_type": event_type if event_type else "other",
        "urgency": urgency,
    }


def _analyze_filing_with_groq(company: str, date: str, filing_type: str, description: str) -> Optional[Dict]:
    """
    Analyze a single filing using Groq API.
    """

    prompt = f"""You are a senior equity analyst. Analyze this NSE corporate filing and return ONLY a JSON object with these fields:
signal (bullish/bearish/neutral),
confidence (0.0 to 1.0),
reason (one sentence max, plain English),
event_type (earnings/insider_trade/acquisition/management_change/regulatory/other),
urgency (high/medium/low).

Filing details:
Company: {company}
Date: {date}
Type: {filing_type}
Description: {description}

Return only valid JSON. No markdown. No explanation.
"""

    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            response = client.with_options(timeout=GROQ_TIMEOUT_SECONDS).chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300,
            )

            response_text = (response.choices[0].message.content or "").strip()
            parsed = _extract_json_object(response_text)
            if parsed is None:
                print(f"  [WARN] Groq returned non-JSON output (attempt {attempt}/{GROQ_MAX_RETRIES})")
            else:
                validated = _validate_filing_analysis(parsed)
                if validated is not None:
                    return validated
                print(f"  [WARN] Groq returned invalid schema values (attempt {attempt}/{GROQ_MAX_RETRIES})")

        except Exception as e:
            print(f"  [ERROR] Groq API failed (attempt {attempt}/{GROQ_MAX_RETRIES}): {type(e).__name__}: {str(e)}")

        if attempt < GROQ_MAX_RETRIES:
            time.sleep(GROQ_RETRY_DELAY_SECONDS)

    return None


def _select_filings_for_coverage(
    filings_df: pd.DataFrame,
    max_filings_per_ticker: int,
    max_total_filings: Optional[int],
    n: Optional[int],
) -> pd.DataFrame:
    """
    Build an analysis set with broad ticker coverage.

    Strategy:
    1) Keep latest rows per ticker (coverage first)
    2) Cap global row count for API/runtime control
    """
    if filings_df.empty:
        return filings_df

    work = filings_df.copy()
    work["ticker"] = work.apply(_resolve_ticker, axis=1)
    work["ticker"] = work["ticker"].fillna("").astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
    work = work[work["ticker"] != ""]
    if work.empty:
        return work

    work["filing_date"] = pd.to_datetime(work.get("filing_date"), errors="coerce")
    work = work.sort_values("filing_date", ascending=False, na_position="last")

    per_ticker = max(1, int(max_filings_per_ticker))
    work = work.groupby("ticker", as_index=False, group_keys=False).head(per_ticker)
    work = work.sort_values("filing_date", ascending=False, na_position="last")

    if n is not None:
        work = work.head(max(int(n), 0))
    elif max_total_filings is not None:
        work = work.head(max(int(max_total_filings), 0))

    return work.reset_index(drop=True)


def analyze_filings(
    n: Optional[int] = None,
    max_filings_per_ticker: int = 3,
    max_total_filings: int = 180,
    quarters: int = 2,
) -> pd.DataFrame:
    """
    Analyze corporate filings using Groq API with broad ticker coverage.

    Args:
        n: Optional hard cap on total rows (None uses max_total_filings).
        max_filings_per_ticker: Latest filings analyzed per ticker.
        max_total_filings: Global safety cap to control API usage.
        quarters: Lookback window in calendar quarters.
    """

    from utils.filing_scraper import get_latest_filings

    filings_df = get_latest_filings(n=None, quarters=quarters)
    filings_df = _select_filings_for_coverage(
        filings_df=filings_df,
        max_filings_per_ticker=max_filings_per_ticker,
        max_total_filings=max_total_filings,
        n=n,
    )

    if filings_df.empty:
        print("No filings to analyze.")
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        empty_df.to_csv(FILING_SIGNALS_CSV, index=False)
        print(f"[OK] Saved 0 filing signals to {FILING_SIGNALS_CSV}")
        return empty_df

    results = []

    tickers_covered = filings_df["ticker"].nunique() if "ticker" in filings_df.columns else 0
    print(
        f"Analyzing {len(filings_df)} filings across {tickers_covered} tickers "
        f"(max {max_filings_per_ticker}/ticker)..."
    )

    for idx, (_, row) in enumerate(filings_df.iterrows(), start=1):
        company = row['company_name']
        date = str(row['filing_date'])
        filing_type = row['filing_type']
        description = row['description']
        ticker = str(row.get("ticker", "")).strip().upper()

        print(f"[{idx:2d}/{len(filings_df)}] Analyzing {ticker or 'UNKNOWN'} - {company[:20]} filing...")

        # Call Groq API
        analysis = _analyze_filing_with_groq(
            company,
            date,
            filing_type,
            description
        )

        if analysis:
            result = {
                "ticker": ticker or _resolve_ticker(row),
                "company_name": company,
                "filing_date": date,
                "signal": analysis["signal"],
                "confidence": analysis["confidence"],
                "reason": analysis["reason"],
                "event_type": analysis["event_type"],
                "urgency": analysis["urgency"],
            }

            results.append(result)
            print(f"  Signal: {analysis['signal']} ({analysis['confidence']:.2f})")

        else:
            print("  [SKIP] Failed to analyze")

        # Rate limit
        if idx < len(filings_df):
            time.sleep(1)

    signals_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    signals_df.to_csv(FILING_SIGNALS_CSV, index=False)

    print(f"\n[OK] Saved {len(signals_df)} filing signals to {FILING_SIGNALS_CSV}")

    return signals_df


if __name__ == "__main__":
    print("Testing Filing Analyzer Agent...")
    signals = analyze_filings(5)
    print(f"Analyzed {len(signals)} filings")
