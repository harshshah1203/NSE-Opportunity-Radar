"""
NSE Opportunity Radar - Day 3 Pattern Explainer.

Generates plain-English explanations for detected chart patterns.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from utils.groq_client import get_groq_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PATTERN_EXPLANATIONS_CSV = DATA_DIR / "pattern_explanations.csv"
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_RATE_LIMIT_SECONDS = 2

OUTPUT_COLUMNS = [
    "ticker",
    "pattern_name",
    "signal",
    "strength",
    "success_rate",
    "current_price",
    "explanation",
]


load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
client = get_groq_client()


def _normalize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV columns to flat Open/High/Low/Close/Volume schema.
    """
    if df.empty:
        return df

    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)

    col_lookup = {str(col).strip().lower(): col for col in normalized.columns}
    required = ["open", "high", "low", "close", "volume"]
    if not all(col in col_lookup for col in required):
        return pd.DataFrame()

    rename_map = {col_lookup[name]: name.capitalize() for name in required}
    rename_map[col_lookup["volume"]] = "Volume"
    normalized = normalized.rename(columns=rename_map)
    normalized = normalized.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    normalized = normalized.dropna(subset=["Close"])
    return normalized


def _lookup_current_price(ticker: str) -> float:
    """
    Lookup latest close price from cached stock data.
    """
    try:
        from utils.data_fetcher import get_latest_data

        df = get_latest_data(ticker, days=10)
        df = _normalize_ohlcv_dataframe(df)
        if df.empty:
            return 0.0
        return round(float(df["Close"].iloc[-1]), 2)
    except Exception:
        return 0.0


def _format_success_rate_for_prompt(success_rate: Optional[float]) -> str:
    """
    Format success rate for prompt insertion.
    """
    if success_rate is None or pd.isna(success_rate):
        return "N/A"
    return f"{float(success_rate) * 100:.1f}"


def _truncate_to_word_limit(text: str, max_words: int = 80) -> str:
    """
    Truncate output text to word-count limit while preserving readability.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(".") + "."


def explain_pattern(pattern_dict: Dict, current_price: float, success_rate: Optional[float]) -> str:
    """
    Generate a 3-sentence plain-English explanation for one pattern.
    """
    ticker = str(pattern_dict.get("ticker", "")).strip().upper()
    pattern_name = str(pattern_dict.get("pattern_name", "")).strip()
    strength = float(pattern_dict.get("strength", 0.0))
    success_rate_text = _format_success_rate_for_prompt(success_rate)
    current_price = round(float(current_price), 2)

    prompt = (
        "You are a technical analyst explaining to an Indian retail investor with basic knowledge.\n"
        f"Stock: {ticker}. Pattern detected: {pattern_name}.\n"
        f"Key data: current price \u20B9{current_price}, pattern strength: {strength}/1.0,\n"
        f"backtested success rate on this stock: {success_rate_text}%.\n"
        "Write exactly 3 sentences:\n"
        "1. What this pattern means in simple terms (no jargon)\n"
        "2. What has historically happened on this stock after this pattern\n"
        "3. One specific price level or condition to watch to confirm this pattern\n"
        "Keep it under 80 words total. Write in a friendly, direct tone."
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=160,
        )
        explanation = (response.choices[0].message.content or "").strip()
        explanation = " ".join(explanation.split())
        if not explanation:
            explanation = (
                f"{pattern_name} is active in {ticker}, but the AI explanation is currently unavailable. "
                "Watch follow-through price action and volume before taking a decision."
            )
        return _truncate_to_word_limit(explanation, max_words=80)
    except Exception as e:
        return (
            f"{pattern_name} is active in {ticker}. Historical pattern commentary is currently unavailable "
            f"due to API issue ({type(e).__name__}). Watch price confirmation before acting."
        )


def explain_all_patterns(patterns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AI explanations to all detected patterns and save results CSV.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if patterns_df is None or patterns_df.empty:
        empty_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty_df.to_csv(PATTERN_EXPLANATIONS_CSV, index=False)
        print(f"[OK] Saved 0 pattern explanations to {PATTERN_EXPLANATIONS_CSV}")
        return empty_df

    rows = []
    total = len(patterns_df)

    for idx, (_, row) in enumerate(patterns_df.iterrows(), start=1):
        ticker = str(row.get("ticker", "")).strip().upper()
        pattern_name = str(row.get("pattern_name", "")).strip()
        signal = str(row.get("signal", "")).strip().lower()
        strength = float(row.get("strength", 0.0))

        raw_success_rate = row.get("success_rate", None)
        success_rate = None if pd.isna(raw_success_rate) else float(raw_success_rate)

        raw_price = row.get("current_price", np.nan) if "current_price" in row else np.nan
        current_price = _lookup_current_price(ticker) if pd.isna(raw_price) else round(float(raw_price), 2)

        explanation = explain_pattern(
            pattern_dict=row.to_dict(),
            current_price=current_price,
            success_rate=success_rate,
        )

        rows.append(
            {
                "ticker": ticker,
                "pattern_name": pattern_name,
                "signal": signal,
                "strength": round(strength, 3),
                "success_rate": success_rate,
                "current_price": current_price,
                "explanation": explanation,
            }
        )

        print(f"[{idx}/{total}] Explained {ticker} - {pattern_name}")
        if idx < total:
            time.sleep(GROQ_RATE_LIMIT_SECONDS)

    explained_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    explained_df.to_csv(PATTERN_EXPLANATIONS_CSV, index=False)
    print(f"[OK] Saved {len(explained_df)} pattern explanations to {PATTERN_EXPLANATIONS_CSV}")
    return explained_df


if __name__ == "__main__":
    """
    Standalone run helper for Day 3 pattern explanation.
    """
    sample_path = DATA_DIR / "pattern_signals.csv"
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        explain_all_patterns(sample_df)
    else:
        explain_all_patterns(pd.DataFrame(columns=OUTPUT_COLUMNS))
