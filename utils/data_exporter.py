"""
Data Exporter Module
Combines all fetched stock data and filings into comprehensive CSV files for analysis.
"""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
DATA_STOCKS_DIR = Path("data/stocks")
FILINGS_CSV_PATH = Path("data/bse_filings.csv")
COMBINED_CSV_PATH = Path("data/combined_stock_data.csv")
SUMMARY_CSV_PATH = Path("data/stock_summary_with_filings.csv")


def export_all_stock_data(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Combine all stock OHLCV data from parquet files into a single CSV.
    
    Reads all parquet files from data/stocks/, concatenates them with a ticker column,
    and saves to a combined CSV file. Useful for batch analysis across all stocks.
    
    Args:
        output_path: Optional path to save the combined CSV (default: data/combined_stock_data.csv)
        
    Returns:
        DataFrame containing all stock data combined
        
    Example:
        >>> df = export_all_stock_data()
        >>> print(f"Combined: {len(df)} total rows from {df['ticker'].nunique()} stocks")
    """
    output_path = output_path or COMBINED_CSV_PATH
    
    print(f"Exporting all stock data...")
    all_data = []
    
    if not DATA_STOCKS_DIR.exists():
        print(f"[WARN] No stock data directory found: {DATA_STOCKS_DIR}")
        return pd.DataFrame()
    
    # Read all parquet files
    parquet_files = list(DATA_STOCKS_DIR.glob("*.parquet"))
    
    if not parquet_files:
        print(f"[WARN] No parquet files found in {DATA_STOCKS_DIR}")
        return pd.DataFrame()
    
    print(f"[OK] Found {len(parquet_files)} stock data files")
    
    for parquet_file in sorted(parquet_files):
        try:
            # Extract ticker from filename (e.g., "RELIANCE.NS.parquet" -> "RELIANCE.NS")
            ticker = parquet_file.stem  # Remove .parquet extension
            
            # Read parquet file
            df = pd.read_parquet(parquet_file)
            
            # Flatten multi-level column index if present (from yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                # Take first level (e.g., 'Close', 'High', etc.)
                df.columns = df.columns.get_level_values(0)
            
            # Reset index to make date a column
            df.reset_index(inplace=True)
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Reorder columns: ticker, date, then OHLCV
            cols = ['ticker', 'Date'] + [c for c in df.columns if c not in ['ticker', 'Date']]
            df = df[cols]
            
            all_data.append(df)
            print(f"  [OK] {ticker}: {len(df)} rows")
        
        except Exception as e:
            print(f"  [ERROR] {parquet_file.name}: {str(e)[:60]}")
            continue
    
    if not all_data:
        print(f"[ERROR] No stock data could be loaded")
        return pd.DataFrame()
    
    # Concatenate all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by ticker and date
    combined_df = combined_df.sort_values(['ticker', 'Date']).reset_index(drop=True)
    
    # Rename 'Date' to 'date' for consistency
    combined_df.rename(columns={'Date': 'date'}, inplace=True)
    
    # Standardize column names to lowercase
    combined_df.columns = [col.lower() for col in combined_df.columns]
    
    # Save to CSV
    combined_df.to_csv(output_path, index=False)
    print(f"\n[OK] Exported {len(combined_df)} rows to {output_path}")
    print(f"     Stocks: {combined_df['ticker'].nunique()} unique tickers")
    print(f"     Date range: {combined_df['date'].min()} to {combined_df['date'].max()}\n")
    
    return combined_df


def export_stock_summary_with_filings() -> pd.DataFrame:
    """
    Create a summary combining key stock metrics with filing data.
    
    Aggregates stock data by ticker (latest price, price change, volume)
    and merges with filing information. Useful for identifying stocks with
    recent corporate filings and price activity.
    
    Returns:
        DataFrame with stock summaries and matching filings
        
    Example:
        >>> df = export_stock_summary_with_filings()
        >>> print(df[['ticker', 'latest_close', 'company_name']].head())
    """
    print(f"Creating stock summary with filings...")
    
    # Get combined stock data
    if COMBINED_CSV_PATH.exists():
        combined_df = pd.read_csv(COMBINED_CSV_PATH)
    else:
        combined_df = export_all_stock_data()
    
    if combined_df.empty:
        print(f"[WARN] No stock data available")
        return pd.DataFrame()
    
    # Create summary: latest price and metrics per ticker
    summary = combined_df.groupby('ticker').agg(
        latest_date=('date', 'max'),
        latest_close=('close', 'last'),
        latest_open=('open', 'last'),
        latest_high=('high', 'max'),
        latest_low=('low', 'min'),
        latest_volume=('volume', 'last'),
        avg_close=('close', 'mean'),
        min_close=('close', 'min'),
        max_close=('close', 'max'),
        total_volume=('volume', 'sum'),
        trading_days=('date', 'count')
    ).reset_index()
    
    # Calculate 20-day change if available
    summary['price_change'] = summary['latest_close'] - summary['avg_close']
    summary['price_change_pct'] = (summary['price_change'] / summary['avg_close'] * 100).round(2)
    
    print(f"[OK] Created summary for {len(summary)} tickers")
    
    # Load filings
    if FILINGS_CSV_PATH.exists():
        filings_df = pd.read_csv(FILINGS_CSV_PATH)
        has_ticker_col = "ticker" in filings_df.columns
        if has_ticker_col:
            filings_df["ticker"] = (
                filings_df["ticker"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(".NS", "", regex=False)
            )
        else:
            filings_df["ticker"] = ""
        
        # Create a filing count column
        filings_count = filings_df.groupby('company_name').size().reset_index(name='filing_count')
        
        # Try to match company names to tickers
        # This is a simple heuristic: match if ticker appears in company name
        def find_matching_ticker(company_name, tickers):
            company_upper = company_name.upper()
            for ticker in tickers:
                ticker_name = ticker.replace('.NS', '').upper()
                if ticker_name in company_upper:
                    return ticker
            return None
        
        # Add filing information
        summary['latest_filing_type'] = None
        summary['latest_filing_date'] = None
        summary['filing_count'] = 0
        
        for idx, row in summary.iterrows():
            ticker = row['ticker']
            ticker_norm = str(ticker).replace('.NS', '').upper()
            
            # Find filings for this ticker
            if has_ticker_col:
                matching_filings = filings_df[filings_df['ticker'] == ticker_norm]
            else:
                matching_filings = filings_df[filings_df['company_name'].str.contains(
                    ticker_norm, case=False, na=False)]
            
            if not matching_filings.empty:
                latest_filing = matching_filings.iloc[0]
                summary.at[idx, 'latest_filing_type'] = latest_filing['filing_type']
                summary.at[idx, 'latest_filing_date'] = latest_filing['filing_date']
                summary.at[idx, 'filing_count'] = len(matching_filings)
        
        print(f"[OK] Matched {summary['filing_count'].sum()} filings to stocks\n")
    
    # Save summary
    summary.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"[OK] Saved summary to {SUMMARY_CSV_PATH}\n")
    
    return summary


def export_filings_detailed() -> pd.DataFrame:
    """
    Load and standardize filing data for easy reference.
    
    Returns:
        DataFrame with filing data, already saved to CSV
    """
    print(f"Loading filings data...")
    
    if not FILINGS_CSV_PATH.exists():
        print(f"[WARN] Filings file not found: {FILINGS_CSV_PATH}")
        return pd.DataFrame()
    
    filings_df = pd.read_csv(FILINGS_CSV_PATH)
    print(f"[OK] Loaded {len(filings_df)} filings\n")
    
    return filings_df


def export_all(verbose: bool = True) -> dict:
    """
    Export all data: combined stock data, summaries, and filings.
    
    Convenience function that runs all export operations and returns results.
    
    Args:
        verbose: Print detailed progress information
        
    Returns:
        Dictionary with keys: 'combined_stocks', 'summary', 'filings'
        
    Example:
        >>> results = export_all()
        >>> print(f"Stock data: {len(results['combined_stocks'])} rows")
        >>> print(f"Summary: {len(results['summary'])} stocks")
        >>> print(f"Filings: {len(results['filings'])} entries")
    """
    print(f"{'='*70}")
    print(f"Exporting All Data to CSV")
    print(f"{'='*70}\n")
    
    results = {
        'combined_stocks': export_all_stock_data(),
        'summary': export_stock_summary_with_filings(),
        'filings': export_filings_detailed()
    }
    
    print(f"{'='*70}")
    print(f"Export Summary:")
    print(f"  [OK] Combined stock data: {len(results['combined_stocks'])} rows")
    print(f"  [OK] Stock summary: {len(results['summary'])} tickers")
    print(f"  [OK] Filings: {len(results['filings'])} entries")
    print(f"{'='*70}\n")
    
    print(f"Files created:")
    print(f"  - {COMBINED_CSV_PATH}")
    print(f"  - {SUMMARY_CSV_PATH}")
    print(f"  - {FILINGS_CSV_PATH}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing data_exporter module...\n")
    results = export_all()
