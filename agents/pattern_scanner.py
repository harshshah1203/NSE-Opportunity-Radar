"""
NSE Opportunity Radar - Day 3 Pattern Scanner.

Coordinates pattern detection, backtesting, explanation, and ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from agents.pattern_detector import scan_all_stocks, backtest_pattern
from agents.pattern_explainer import explain_all_patterns, OUTPUT_COLUMNS as EXPLANATION_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PATTERN_EXPLANATIONS_CSV = DATA_DIR / "pattern_explanations.csv"


load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)


def _compute_rank_score(strength: float, success_rate: Optional[float], signal: str = "bullish") -> float:
    """
    Compute rank score using hybrid confidence model:
    
    1. WITH BACKTEST DATA (3+ events): Use proven historical success
       Score = 0.3 * strength + 0.7 * success_rate
       
    2. WITHOUT BACKTEST DATA (<3 events, common with 90-day windows):
       Use technical signal quality. High-strength bullish patterns are inherently
       high-conviction even without historical test data.
       Score = strength + signal_bonus + uptrend_bias
       - Bullish signal: +0.25 (strong conviction)
       - Bearish signal: +0.05 (lower conviction)
       - Technical strength 0.8+: +0.10 bonus
       
    This ensures good patterns aren't penalized for limited historical data.
    Target: 80%+ success rate based on technical quality metrics.
    """
    strength = float(strength)
    
    # WITH SUFFICIENT BACKTEST DATA: Heavily weight proven success
    if success_rate is not None and not pd.isna(success_rate):
        return round((strength * 0.3) + (float(success_rate) * 0.7), 4)
    
    # WITHOUT BACKTEST DATA: Score by technical conviction
    # Base: strength (0.0-1.0)
    base_score = strength
    
    # Signal conviction bonus
    signal_bonus = 0.25 if signal == "bullish" else 0.05
    
    # Additional bonus for very strong patterns
    strength_bonus = 0.10 if strength >= 0.8 else 0.05
    
    # Combined confidence score (cap at 1.0 for consistency)
    confidence = base_score + signal_bonus + strength_bonus
    return round(min(1.0, confidence), 4)


def _format_success_rate(value: Optional[float]) -> str:
    """
    Format optional success-rate into user-friendly text.
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _print_top_patterns_table(df: pd.DataFrame, n: int) -> None:
    """
    Print top-ranked patterns in table format.
    """
    print(f"\nTop {min(n, len(df))} Chart Patterns:")
    if df.empty:
        print("No patterns available.")
        return

    display = df.copy()
    display["strength"] = display["strength"].map(lambda x: f"{float(x):.2f}")
    display["success_rate"] = display["success_rate"].map(_format_success_rate)
    display["rank_score"] = display["rank_score"].map(lambda x: f"{float(x):.3f}")

    cols = ["ticker", "pattern_name", "signal", "strength", "success_rate", "rank_score"]
    print(display.loc[:, cols].to_string(index=False))


def get_top_patterns(n: int = 10) -> pd.DataFrame:
    """
    Run Day 3 chart intelligence and return top N ranked patterns.
    """
    patterns_df = scan_all_stocks()

    if patterns_df.empty:
        explained_df = explain_all_patterns(patterns_df)
        explained_df["rank_score"] = pd.Series(dtype=float)
        explained_df.loc[:, EXPLANATION_COLUMNS].to_csv(PATTERN_EXPLANATIONS_CSV, index=False)
        _print_top_patterns_table(explained_df, n)
        explained_df.attrs["total_patterns_detected"] = 0
        explained_df.attrs["total_stocks_detected"] = 0
        return explained_df

    success_rates = []
    total_patterns = len(patterns_df)
    for idx, (_, row) in enumerate(patterns_df.iterrows(), start=1):
        ticker = str(row["ticker"])
        pattern_name = str(row["pattern_name"])
        success = backtest_pattern(ticker, pattern_name)
        success_rates.append(success)
        print(
            f"  Backtest [{idx}/{total_patterns}] {ticker} {pattern_name}: "
            f"{_format_success_rate(success)}"
        )

    patterns_with_backtest = patterns_df.copy()
    patterns_with_backtest["success_rate"] = success_rates

    explained_df = explain_all_patterns(patterns_with_backtest)
    if explained_df.empty:
        explained_df["rank_score"] = pd.Series(dtype=float)
        explained_df.loc[:, EXPLANATION_COLUMNS].to_csv(PATTERN_EXPLANATIONS_CSV, index=False)
        explained_df.attrs["total_patterns_detected"] = 0
        explained_df.attrs["total_stocks_detected"] = 0
        return explained_df

    explained_df["rank_score"] = explained_df.apply(
        lambda row: _compute_rank_score(row["strength"], row["success_rate"], row.get("signal", "bullish")),
        axis=1,
    )
    ranked_df = explained_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked_df.loc[:, EXPLANATION_COLUMNS].to_csv(PATTERN_EXPLANATIONS_CSV, index=False)

    top_df = ranked_df.head(n).copy()
    _print_top_patterns_table(top_df, n)

    top_df.attrs["total_patterns_detected"] = int(len(ranked_df))
    top_df.attrs["total_stocks_detected"] = int(ranked_df["ticker"].nunique())
    return top_df


if __name__ == "__main__":
    """
    Standalone run helper for Day 3 master scanner.
    """
    get_top_patterns(10)
