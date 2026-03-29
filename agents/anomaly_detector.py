"""
NSE Opportunity Radar - Anomaly Detector Agent
Detects stock price and volume anomalies using technical indicators.
"""

import time
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from utils.groq_client import get_groq_client

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ANOMALY_SIGNALS_CSV = DATA_DIR / "anomaly_signals.csv"

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_TIMEOUT_SECONDS = 15.0
GROQ_MAX_RETRIES = 2
GROQ_RETRY_DELAY_SECONDS = 1.0
ANOMALY_COLUMNS = [
    "ticker",
    "anomaly_type",
    "volume_ratio",
    "price_change",
    "rsi",
    "anomaly_strength",
    "explanation",
]

client = get_groq_client()


def _normalize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance OHLCV frames to flat columns: Open/High/Low/Close/Volume.
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

    return normalized[["Open", "High", "Low", "Close", "Volume"]]

def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate RSI (Relative Strength Index) manually.
    
    Args:
        prices: Series of closing prices
        period: RSI period (default 14)
        
    Returns:
        RSI value (0-100) or NaN if insufficient data
    """
    if len(prices) < period + 1:
        return np.nan
    
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses (simple moving average)
    avg_gain = gains.rolling(window=period).mean().iloc[-1]
    avg_loss = losses.rolling(window=period).mean().iloc[-1]
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def _detect_anomaly_for_stock(ticker: str, df: pd.DataFrame) -> Optional[Dict]:
    """
    Detect anomalies for a single stock.
    
    Args:
        ticker: Stock ticker
        df: DataFrame with OHLCV data
        
    Returns:
        Dict with anomaly details or None if no anomaly
    """
    if df.empty or len(df) < 30:
        return None
    
    # Get latest data
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Calculate metrics
    latest_volume = float(latest["Volume"])
    volume_30_avg = float(df["Volume"].tail(30).mean())
    volume_ratio = latest_volume / volume_30_avg if volume_30_avg > 0 else 0.0
    
    prev_close = float(prev["Close"])
    latest_close = float(latest["Close"])
    price_change = ((latest_close - prev_close) / prev_close) * 100 if prev_close else 0.0

    rsi = _calculate_rsi(df["Close"])
    
    # Check anomaly conditions
    is_anomaly = (
        volume_ratio > 2.0 or
        abs(price_change) > 3.0 or
        (rsi > 70) or (rsi < 30)
    )
    
    if not is_anomaly:
        return None
    
    # Determine anomaly type
    anomaly_type = []
    if volume_ratio > 2.0:
        anomaly_type.append("volume_spike")
    if abs(price_change) > 3.0:
        anomaly_type.append("price_move")
    if rsi > 70:
        anomaly_type.append("overbought")
    elif rsi < 30:
        anomaly_type.append("oversold")
    
    anomaly_type_str = ", ".join(anomaly_type)

    # Normalize anomaly intensity to [0.0, 1.0] for cross-signal ranking.
    volume_component = min(max((volume_ratio - 1.0) / 3.0, 0.0), 1.0)
    price_component = min(abs(price_change) / 10.0, 1.0)
    if np.isnan(rsi):
        rsi_component = 0.0
    else:
        rsi_component = min(max((abs(rsi - 50.0) - 20.0) / 30.0, 0.0), 1.0)
    anomaly_strength = round((0.45 * volume_component) + (0.35 * price_component) + (0.20 * rsi_component), 3)
    
    # Call Groq for explanation
    prompt = f"""Stock: {ticker}. Volume is {volume_ratio:.1f}x above 30-day average. Price moved {price_change:.2f}% last session. RSI is {rsi:.1f}. As an equity analyst, explain in 2 sentences what likely caused this anomaly and whether it is worth investigating. Be specific and concise."""
    
    explanation = "No AI explanation available."
    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            response = client.with_options(timeout=GROQ_TIMEOUT_SECONDS).chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
            )
            explanation = (response.choices[0].message.content or "").strip()
            if explanation:
                break
        except Exception as e:
            explanation = f"Analysis failed: {type(e).__name__}: {str(e)}"
            if attempt < GROQ_MAX_RETRIES:
                time.sleep(GROQ_RETRY_DELAY_SECONDS)

    return {
        "ticker": ticker,
        "anomaly_type": anomaly_type_str,
        "volume_ratio": round(float(volume_ratio), 3),
        "price_change": round(float(price_change), 3),
        "rsi": round(float(rsi), 3),
        "anomaly_strength": anomaly_strength,
        "explanation": explanation,
    }

def detect_anomalies(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Detect anomalies across stocks.
    
    Args:
        tickers: List of tickers to check, or None for all available
        
    Returns:
        DataFrame with anomaly signals
    """
    from utils.data_fetcher import get_top_nse_tickers, get_latest_data
    
    if tickers is None:
        tickers = get_top_nse_tickers()
    
    results = []
    checked_count = 0
    no_data_count = 0
    
    for ticker in tickers:
        checked_count += 1
        df = get_latest_data(ticker, days=90)
        df = _normalize_ohlcv_dataframe(df)
        if df.empty:
            no_data_count += 1
            continue
        
        anomaly = _detect_anomaly_for_stock(ticker, df)
        if anomaly:
            results.append(anomaly)
            print(f"[ANOMALY] {ticker}: {anomaly['anomaly_type']}")
        
        # Rate limiting: 1 second between stocks
        time.sleep(1)
    
    # Create DataFrame and save
    anomalies_df = pd.DataFrame(results, columns=ANOMALY_COLUMNS)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    anomalies_df.to_csv(ANOMALY_SIGNALS_CSV, index=False)
    print(f"\n[OK] Saved {len(anomalies_df)} anomaly signals to {ANOMALY_SIGNALS_CSV}")
    if anomalies_df.empty:
        print(
            f"[INFO] No anomalies met thresholds. Checked={checked_count}, "
            f"no_data={no_data_count}, thresholds=(volume>2x | |price|>3% | RSI outside 30-70)"
        )
    
    return anomalies_df

if __name__ == "__main__":
    print("Testing Anomaly Detector Agent...")
    anomalies = detect_anomalies(["RELIANCE", "INFY"])
    print(f"Detected {len(anomalies)} anomalies")
