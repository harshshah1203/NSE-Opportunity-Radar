"""
NSE Stock Data Fetcher Module
Fetches OHLCV data for NSE tickers using yfinance with exponential backoff retry logic.
Saves data as parquet files for efficient storage and retrieval.
"""

import time
import os
from pathlib import Path
from functools import wraps
from typing import Optional, Dict, List

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for data storage and retry logic
DATA_STOCKS_DIR = Path("data/stocks")
RETRY_LIMIT = 3
INITIAL_BACKOFF = 1  # Initial backoff delay in seconds
BACKOFF_MULTIPLIER = 2  # Exponential backoff multiplier


def _ensure_data_directory():
    """
    Ensure the data/stocks directory exists.
    Creates it if it doesn't already exist.
    """
    DATA_STOCKS_DIR.mkdir(parents=True, exist_ok=True)


def retry_with_backoff(func):
    """
    Decorator that retries a function with exponential backoff on failure.
    
    Retries up to RETRY_LIMIT times with exponential backoff (1s -> 2s -> 4s).
    Catches network errors, rate limit errors, and general exceptions.
    
    Args:
        func: Function to retry
        
    Returns:
        Wrapped function with retry logic
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        delay = INITIAL_BACKOFF
        last_exception = None
        
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < RETRY_LIMIT:
                    # Log detailed error info for debugging
                    error_msg = str(e)
                    if len(error_msg) > 100:
                        error_msg = error_msg[:100] + "..."
                    print(f"    > Attempt {attempt}/{RETRY_LIMIT} failed: {error_msg}")
                    print(f"    * Waiting {delay}s before retry...")
                    time.sleep(delay)
                    delay *= BACKOFF_MULTIPLIER
        
        # All retries exhausted - raise the last exception for caller to handle
        raise last_exception
    
    return wrapper


def get_top_nse_tickers() -> List[str]:
    """
    Return a hardcoded list of 50 popular NSE tickers (without .NS suffix).
    
    This list contains the most actively traded and liquid stocks on NSE
    sorted roughly by market capitalization as of 2025.
    
    Returns:
        List of 50 NSE ticker symbols (without .NS suffix)
    """
    tickers = [
        # Broad liquid NSE universe (50 symbols, without .NS suffix)
        "RELIANCE", "INFY", "TCS", "HDFCBANK", "ICICIBANK", "LT", "WIPRO",
        "MARUTI", "SUNPHARMA", "BAJAJFINSV", "BHARTIARTL", "ASIANPAINT",
        "JSWSTEEL", "TATASTEEL", "BPCL", "ONGC", "INDIGO", "AXISBANK", "SBIN",
        "BAJAJ-AUTO", "EICHERMOT", "HINDALCO", "APOLLOHOSP", "NTPC",
        "POWERGRID", "M&M", "HEROMOTOCO", "ULTRACEMCO", "ADANIPORTS",
        "ADANIENT", "ITC", "BRITANNIA", "YESBANK", "UPL", "NESTLEIND",
        "GRASIM", "HCLTECH", "TECHM", "TITAN", "SBICARD", "GODREJCP",
        "AUROPHARMA", "TATAMOTORS", "KOTAKBANK", "HINDUNILVR", "CIPLA",
        "DRREDDY", "COALINDIA", "BEL", "SHRIRAMFIN"
    ]
    return tickers


@retry_with_backoff
def _download_yfinance(ticker_with_ns: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Internal helper to download data from yfinance with retry decorator.
    
    Args:
        ticker_with_ns: Ticker with .NS suffix (e.g., "RELIANCE.NS")
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        Exception: If download fails (rate limit, invalid ticker, etc.)
    """
    # Default to last 90 calendar days if no explicit date range is provided
    if start_date is None and end_date is None:
        data = yf.download(ticker_with_ns, period="90d", progress=False)
    else:
        data = yf.download(ticker_with_ns, start=start_date, end=end_date, progress=False)

    if data.empty:
        raise ValueError(f"No data found for {ticker_with_ns} - possibly delisted or invalid.")
    
    return data


def fetch_stock_data(ticker: str, start_date: Optional[str] = None, 
                     end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a single NSE ticker using yfinance with exponential backoff.
    
    Automatically appends .NS to the ticker symbol. Saves the data as a parquet file
    in the data/stocks/ directory. Handles delisted stocks, invalid tickers, and
    network errors gracefully.
    
    Args:
        ticker: Ticker symbol without .NS suffix (e.g., "RELIANCE")
        start_date: Optional start date in YYYY-MM-DD format (default: last 1 year)
        end_date: Optional end date in YYYY-MM-DD format (default: today)
        
    Returns:
        DataFrame with OHLCV data if successful, None if failed
        
    Example:
        >>> df = fetch_stock_data("RELIANCE")
        >>> print(df.head())
    """
    _ensure_data_directory()
    ticker_with_ns = f"{ticker}.NS"
    
    try:
        # Download data with retry logic via decorator
        df = _download_yfinance(ticker_with_ns, start_date, end_date)
        
        # Save to parquet file
        parquet_path = DATA_STOCKS_DIR / f"{ticker_with_ns}.parquet"
        df.to_parquet(parquet_path)
        print(f"  [OK] Saved to data/stocks/{ticker_with_ns}.parquet")
        
        return df
    
    except ValueError as e:
        # Delisted or invalid ticker
        print(f"  [FAIL] {ticker}: {str(e)}")
        return None
    
    except Exception as e:
        # Network or other errors (already retried)
        print(f"  [FAIL] {ticker}: Failed after {RETRY_LIMIT} attempts - {str(e)[:60]}")
        return None


def get_latest_data(ticker: str, days: int = 90) -> pd.DataFrame:
    """
    Retrieve the latest N days of OHLCV data for a ticker.
    
    Loads data from the saved parquet file if it exists. Falls back to
    fetching fresh data via yfinance if not cached (with retry logic).
    Filters to the last N trading days.
    
    Args:
        ticker: Ticker symbol without .NS suffix (e.g., "RELIANCE")
        days: Number of recent trading days to return (default: 90)
        
    Returns:
        DataFrame with OHLCV data for the last N days, or empty DataFrame if not found
        
    Example:
        >>> df = get_latest_data("RELIANCE", days=30)
        >>> print(f"Latest {len(df)} days of RELIANCE data")
    """
    _ensure_data_directory()
    ticker_with_ns = f"{ticker}.NS"
    parquet_path = DATA_STOCKS_DIR / f"{ticker_with_ns}.parquet"
    
    try:
        # Try to load from cached parquet file
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            print(f"Loaded {ticker} from cache ({len(df)} total rows)")
        else:
            # Fetch fresh data if not cached
            print(f"Fetching {ticker} (not cached)...")
            df = _download_yfinance(ticker_with_ns)
            if df is None or df.empty:
                return pd.DataFrame()
            # Save for future use
            df.to_parquet(parquet_path)
        
        # Return last N days
        return df.tail(days)
    
    except Exception as e:
        print(f"Warning: Could not retrieve {ticker} data: {str(e)}")
        return pd.DataFrame()


def fetch_all_stocks(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a batch of NSE tickers with progress reporting.
    
    Iterates through the ticker list, fetching data for each with exponential
    backoff retry logic. Prints progress as [X/N] for each ticker.
    
    Args:
        tickers: List of ticker symbols without .NS suffix
        
    Returns:
        Dictionary mapping ticker -> DataFrame (only includes successful fetches)
        
    Example:
        >>> tickers = get_top_nse_tickers()
        >>> results = fetch_all_stocks(tickers)
        >>> print(f"Successfully fetched {len(results)}/{len(tickers)} stocks")
    """
    _ensure_data_directory()
    results = {}
    
    print(f"\n{'='*70}")
    print(f"Fetching OHLCV data for {len(tickers)} NSE tickers...")
    print(f"{'='*70}\n")
    
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx:2d}/{len(tickers)}] Fetching {ticker}...")
        df = fetch_stock_data(ticker)
        
        if df is not None:
            results[ticker] = df
            print(f"  [OK] {ticker} complete ({len(df)} rows)")
        else:
            print(f"  [SKIP] {ticker} skipped")
        
        print()  # Blank line for readability
    
    print(f"{'='*70}")
    print(f"Summary: {len(results)}/{len(tickers)} stocks fetched successfully")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    # Example usage when running this module directly
    print("Testing data_fetcher module...\n")
    
    # Test get_top_nse_tickers
    tickers = get_top_nse_tickers()
    print(f"Found {len(tickers)} NSE tickers: {', '.join(tickers[:5])}...")
    
    # Test fetch_stock_data for a single ticker
    print("\nFetching sample data for RELIANCE...")
    df = fetch_stock_data("RELIANCE")
    if df is not None:
        print(f"Retrieved {len(df)} rows of data")
        print(f"Columns: {list(df.columns)}")
    
    # Test get_latest_data
    print("\nRetrieving latest 30 days of RELIANCE data...")
    df_latest = get_latest_data("RELIANCE", days=30)
    print(f"Latest data shape: {df_latest.shape}")
