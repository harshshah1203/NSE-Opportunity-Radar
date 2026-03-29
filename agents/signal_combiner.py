"""
NSE Opportunity Radar - Signal Combiner Agent
Combines filing and anomaly signals to rank investment opportunities.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FILING_SIGNALS_CSV = DATA_DIR / "filing_signals.csv"
ANOMALY_SIGNALS_CSV = DATA_DIR / "anomaly_signals.csv"
TOP_SIGNALS_CSV = DATA_DIR / "top_signals.csv"

WEIGHT_FILING = 0.65
WEIGHT_URGENCY = 0.10
WEIGHT_ANOMALY = 0.25

OUTPUT_COLUMNS = [
    "ticker",
    "final_score",
    "filing_signal",
    "anomaly_type",
    "top_reason",
    "score_breakdown",
]


def _empty_result_df() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _get_tracked_universe_tickers() -> List[str]:
    """
    Resolve tracked universe tickers (prefer configured top-50 list).
    """
    tickers: List[str] = []
    try:
        from utils.data_fetcher import get_top_nse_tickers
        tickers = [_normalize_ticker(t) for t in get_top_nse_tickers()]
    except Exception:
        tickers = []

    if not tickers:
        stocks_dir = DATA_DIR / "stocks"
        if stocks_dir.exists():
            for parquet_file in stocks_dir.glob("*.parquet"):
                ticker = _normalize_ticker(parquet_file.stem.replace(".NS", ""))
                if ticker:
                    tickers.append(ticker)

    # preserve order while removing duplicates/blanks
    seen = set()
    ordered: List[str] = []
    for ticker in tickers:
        if ticker and ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    return ordered


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return pd.DataFrame()


def _normalize_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    ticker = str(value).strip().upper()
    if ticker.lower() == "nan":
        return ""
    return ticker


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _calculate_signal_score(signal: str, confidence: float) -> float:
    """
    Calculate filing component from signal and confidence.
    """
    signal_map: Dict[str, float] = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    signal_norm = str(signal).strip().lower()
    confidence_norm = max(0.0, min(float(confidence), 1.0))
    return signal_map.get(signal_norm, 0.0) * confidence_norm


def _calculate_urgency_component(urgency: str) -> float:
    urgency_map: Dict[str, float] = {"high": 0.30, "medium": 0.15, "low": 0.00}
    return urgency_map.get(str(urgency).strip().lower(), 0.00)


def _calculate_anomaly_component(row: pd.Series) -> float:
    # Prefer precomputed anomaly_strength when available.
    try:
        strength = float(row.get("anomaly_strength", 0.0))
        if strength >= 0:
            return round(max(0.0, min(strength, 1.0)), 3)
    except (TypeError, ValueError):
        pass

    score = 0.0
    anomaly_type = str(row.get("anomaly_type", "")).lower()
    if "volume_spike" in anomaly_type:
        score += 0.35
    if "price_move" in anomaly_type:
        score += 0.30
    if "overbought" in anomaly_type or "oversold" in anomaly_type:
        score += 0.20

    try:
        volume_ratio = float(row.get("volume_ratio", 0.0))
        if volume_ratio > 1:
            score += min((volume_ratio - 1.0) / 4.0, 0.25)
    except (TypeError, ValueError):
        pass

    try:
        price_change = abs(float(row.get("price_change", 0.0)))
        score += min(price_change / 10.0, 0.25)
    except (TypeError, ValueError):
        pass

    return round(max(0.0, min(score, 1.0)), 3)


def _prepare_filing_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "signal",
                "confidence",
                "urgency",
                "reason",
                "filing_component",
                "urgency_component",
            ]
        )

    required_cols = ["ticker", "signal", "confidence", "urgency", "reason"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    work = df.copy()
    work["ticker"] = work["ticker"].apply(_normalize_ticker)
    work = work[work["ticker"] != ""]
    if work.empty:
        return pd.DataFrame(columns=["ticker", "signal", "confidence", "urgency", "reason", "filing_component", "urgency_component"])

    work["confidence"] = pd.to_numeric(work["confidence"], errors="coerce").fillna(0.0)
    work["filing_component"] = work.apply(
        lambda row: _calculate_signal_score(row.get("signal", ""), row.get("confidence", 0.0)),
        axis=1,
    )
    work["urgency_component"] = work["urgency"].apply(_calculate_urgency_component)
    work["filing_rank"] = work["filing_component"].abs() + work["urgency_component"]
    work = work.sort_values(["filing_rank", "confidence"], ascending=False)
    work = work.groupby("ticker", as_index=False).first()

    return work[["ticker", "signal", "confidence", "urgency", "reason", "filing_component", "urgency_component"]]


def _prepare_anomaly_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["ticker", "anomaly_type", "explanation", "anomaly_component"])

    required_cols = ["ticker", "anomaly_type", "explanation", "anomaly_strength", "volume_ratio", "price_change"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    work = df.copy()
    work["ticker"] = work["ticker"].apply(_normalize_ticker)
    work = work[work["ticker"] != ""]
    if work.empty:
        return pd.DataFrame(columns=["ticker", "anomaly_type", "explanation", "anomaly_component"])

    work["anomaly_component"] = work.apply(_calculate_anomaly_component, axis=1)
    work = work.sort_values("anomaly_component", ascending=False)
    work = work.groupby("ticker", as_index=False).first()

    return work[["ticker", "anomaly_type", "explanation", "anomaly_component"]]


def get_top_signals(n: int = 50) -> pd.DataFrame:
    """
    Combine and rank signals from filings and anomalies.
    """
    filing_df_raw = _safe_read_csv(FILING_SIGNALS_CSV)
    anomaly_df_raw = _safe_read_csv(ANOMALY_SIGNALS_CSV)

    filing_df = _prepare_filing_df(filing_df_raw)
    anomaly_df = _prepare_anomaly_df(anomaly_df_raw)
    universe_tickers = _get_tracked_universe_tickers()

    if universe_tickers:
        base_df = pd.DataFrame({"ticker": universe_tickers})
        merged_df = (
            base_df
            .merge(filing_df, on="ticker", how="left")
            .merge(anomaly_df, on="ticker", how="left")
        )
    else:
        merged_df = pd.merge(filing_df, anomaly_df, on="ticker", how="outer")

    if merged_df.empty:
        result_df = _empty_result_df()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(TOP_SIGNALS_CSV, index=False)
        print("No signals to combine.")
        print(f"[OK] Saved top 0 signals to {TOP_SIGNALS_CSV}")
        return result_df

    merged_df["filing_component"] = pd.to_numeric(merged_df.get("filing_component", 0.0), errors="coerce").fillna(0.0)
    merged_df["urgency_component"] = pd.to_numeric(merged_df.get("urgency_component", 0.0), errors="coerce").fillna(0.0)
    merged_df["anomaly_component"] = pd.to_numeric(merged_df.get("anomaly_component", 0.0), errors="coerce").fillna(0.0)

    merged_df["final_score"] = (
        (WEIGHT_FILING * merged_df["filing_component"]) +
        (WEIGHT_URGENCY * merged_df["urgency_component"]) +
        (WEIGHT_ANOMALY * merged_df["anomaly_component"])
    ).round(4)

    merged_df["score_breakdown"] = merged_df.apply(
        lambda row: (
            f"filing={row['filing_component']:+.2f}*{WEIGHT_FILING:.2f}; "
            f"urgency={row['urgency_component']:.2f}*{WEIGHT_URGENCY:.2f}; "
            f"anomaly={row['anomaly_component']:.2f}*{WEIGHT_ANOMALY:.2f}"
        ),
        axis=1,
    )

    merged_df["top_reason"] = merged_df.apply(
        lambda row: (
            _safe_text(row.get("reason", ""))
            if _safe_text(row.get("reason", ""))
            else _safe_text(row.get("explanation", ""))
        ),
        axis=1,
    )
    merged_df["top_reason"] = merged_df["top_reason"].replace("", "No reason available.")

    merged_df["filing_signal"] = merged_df.get("signal", "").fillna("")
    merged_df["anomaly_type"] = merged_df.get("anomaly_type", "").fillna("")

    merged_df = merged_df.sort_values("final_score", ascending=False)
    if n is not None:
        merged_df = merged_df.head(max(int(n), 0))
    result_df = merged_df.loc[:, OUTPUT_COLUMNS]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(TOP_SIGNALS_CSV, index=False)
    print(f"[OK] Saved top {len(result_df)} signals to {TOP_SIGNALS_CSV}")

    if not result_df.empty:
        preview = result_df[["ticker", "final_score", "score_breakdown"]].head(3)
        print("[INFO] Top score breakdown preview:")
        for _, row in preview.iterrows():
            print(f"  - {row['ticker']}: {row['final_score']:.4f} ({row['score_breakdown']})")

    return result_df


if __name__ == "__main__":
    print("Testing Signal Combiner Agent...")
    top_signals = get_top_signals(5)
    print(top_signals)
