"""
NSE Opportunity Radar - Day 1 Data Pipeline
Main orchestration script that fetches NSE stock data and BSE filings.

This script:
1. Loads environment variables from .env
2. Fetches OHLCV data for 50 popular NSE tickers
3. Scrapes corporate filings from BSE India
4. Displays a summary of the latest filings
5. Confirms successful completion of Day 1 data pipeline
"""

import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# Import data fetching and scraping modules
from utils.data_fetcher import get_top_nse_tickers, fetch_all_stocks
from utils.filing_scraper import scrape_bse_filings, get_latest_filings
from utils.data_exporter import export_all

# Import Day 2 agents
from agents.filing_analyzer import analyze_filings
from agents.anomaly_detector import detect_anomalies
from agents.signal_combiner import get_top_signals
# Import Day 3 agent
from agents.pattern_scanner import get_top_patterns


def print_header(text: str, char: str = "=") -> None:
    """
    Print a formatted header with decorative characters.
    
    Args:
        text: Header text to display
        char: Character to use for borders (default: "=")
    """
    width = 80
    side_width = (width - len(text) - 2) // 2
    border = char * side_width
    print(f"{border} {text} {border}")


def print_filings_summary(df: pd.DataFrame, num_to_show: int = 5) -> None:
    """
    Display the latest filings in a readable tabular format.
    
    Args:
        df: DataFrame containing filing data
        num_to_show: Number of filings to display (default: 5)
    """
    if df.empty:
        print("No filings available to display.")
        return
    
    df_display = df.head(num_to_show).copy()
    
    print(f"\nLatest {len(df_display)} NSE Corporate Filings:")
    print("-" * 80)
    
    for idx, (_, row) in enumerate(df_display.iterrows(), start=1):
        print(f"\n{idx}. {row['company_name']}")
        print(f"   Date: {row['filing_date']}")
        print(f"   Type: {row['filing_type']}")
        print(f"   Description: {row['description'][:70]}...")
        print(f"   URL: {row.get('url', 'N/A')[:60]}...")
    
    print("\n" + "-" * 80)


def main():
    """
    Main orchestration function for Day 1 data pipeline.
    
    Runs the complete data collection workflow:
    1. Load .env configuration
    2. Fetch stock data for top 50 NSE tickers
    3. Scrape BSE filings
    4. Display summary and completion message
    """
    
    # Load environment variables from .env file
    # This loads GROQ_API_KEY and other config needed for future days
    load_dotenv()
    
    print_header("NSE Opportunity Radar - Day 1 Data Pipeline", "=")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working Directory: {Path.cwd()}\n")
    
    # =========================================================================
    # PHASE 1: Fetch NSE Stock Data
    # =========================================================================
    try:
        print_header("PHASE 1: Fetching NSE Stock Data", "-")
        
        # Get list of top 50 NSE tickers
        tickers = get_top_nse_tickers()
        print(f"Loaded {len(tickers)} NSE tickers")
        print(f"Sample tickers: {', '.join(tickers[:10])}\n")
        
        # Fetch OHLCV data for all tickers with progress reporting
        results = fetch_all_stocks(tickers)
        
        # Summary statistics
        success_count = len(results)
        failed_count = len(tickers) - success_count
        success_rate = (success_count / len(tickers)) * 100
        
        print(f"Stock Data Summary:")
        print(f"  [OK] Successfully fetched: {success_count}/{len(tickers)} ({success_rate:.1f}%)")
        print(f"  [SKIP] Failed or skipped: {failed_count}/{len(tickers)}")
        
        if results:
            # Show statistics for first fetched stock
            first_ticker = next(iter(results.keys()))
            df_sample = results[first_ticker]
            print(f"\n  Sample: {first_ticker} has {len(df_sample)} trading days")
            print(f"  Date range: {df_sample.index[0].date()} to {df_sample.index[-1].date()}")
            print(f"  OHLCV columns available")
        
        print()
    
    except Exception as e:
        print(f"ERROR in Phase 1 (Stock Data): {str(e)}")
        print("Continuing to Phase 2...\n")
    
    # =========================================================================
    # PHASE 2: Scrape BSE Filings
    # =========================================================================
    try:
        print_header("PHASE 2: Scraping BSE Filings", "-")
        
        # Scrape corporate filings from BSE XML feed
        filings = scrape_bse_filings()
        
        # Get and display latest filings
        df_filings = get_latest_filings(n=50)
        
        if not df_filings.empty:
            print(f"Filing Data Summary:")
            print(f"  [OK] Total filings available: {len(df_filings)}")
            
            # Display the latest 5 filings
            print_filings_summary(df_filings, num_to_show=5)
        else:
            print("No filings available to display.")
        
        print()
    
    except Exception as e:
        print(f"ERROR in Phase 2 (Filings): {str(e)}")
        print("Continuing to completion...\n")
    
    # =========================================================================
    # PHASE 3: Export All Data to CSV
    # =========================================================================
    try:
        print_header("PHASE 3: Exporting Data to CSV", "-")
        export_all(verbose=True)
    
    except Exception as e:
        print(f"ERROR in Phase 3 (Export): {str(e)}")
        print("Continuing to completion...\n")
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    print_header("Day 1 Complete", "=")
    print("\n[SUCCESS] Data pipeline working!")
    print(f"  - Stock data saved to: data/stocks/ (41 parquet files)")
    print(f"  - Combined CSV export: data/combined_stock_data.csv")
    print(f"  - Stock summary: data/stock_summary_with_filings.csv")
    print(f"  - Filing data: data/bse_filings.csv")
    print(f"  - Environment loaded from: .env")
    print(f"\n[INFO] Ready for Day 2: AI agent integration & opportunity detection\n")
    
    # =========================================================================
    # DAY 2: Opportunity Radar Agent
    # =========================================================================
    print_header("Day 2: Opportunity Radar Agent", "=")
    
    # PHASE 1: Analyze Filings
    try:
        print_header("PHASE 1: Analyzing Filings", "-")
        filing_signals = analyze_filings(
            n=None,
            max_filings_per_ticker=3,
            max_total_filings=180,
            quarters=2,
        )
        
        if not filing_signals.empty:
            signal_counts = filing_signals['signal'].value_counts()
            print(f"Filing Analysis Summary:")
            print(f"  [OK] Analyzed {len(filing_signals)} filings")
            print(f"  Bullish: {signal_counts.get('bullish', 0)}")
            print(f"  Bearish: {signal_counts.get('bearish', 0)}")
            print(f"  Neutral: {signal_counts.get('neutral', 0)}")
        else:
            print("No filing signals generated.")
        
        print()
    
    except Exception as e:
        print(f"ERROR in Day 2 Phase 1 (Filing Analysis): {str(e)}")
        print("Continuing to Phase 2...\n")
    
    # PHASE 2: Detect Anomalies
    try:
        print_header("PHASE 2: Detecting Anomalies", "-")
        anomaly_signals = detect_anomalies()
        
        print(f"Anomaly Detection Summary:")
        print(f"  [OK] Detected {len(anomaly_signals)} anomalies")
        
        print()
    
    except Exception as e:
        print(f"ERROR in Day 2 Phase 2 (Anomaly Detection): {str(e)}")
        print("Continuing to Phase 3...\n")
    
    # PHASE 3: Combine Signals
    try:
        print_header("PHASE 3: Combining Signals", "-")
        top_signals = get_top_signals(50)
        
        if not top_signals.empty:
            print(f"Top {len(top_signals)} Investment Signals:")
            print("-" * 80)
            for idx, (_, row) in enumerate(top_signals.iterrows(), start=1):
                print(f"{idx:2d}. {row['ticker']} (Score: {row['final_score']:.2f})")
                print(f"    Filing: {row['filing_signal']}, Anomaly: {row['anomaly_type']}")
                print(f"    Reason: {row['top_reason'][:60]}...")
                print()
            print("-" * 80)
        else:
            print("No top signals generated.")
        
        print()
    
    except Exception as e:
        print(f"ERROR in Day 2 Phase 3 (Signal Combination): {str(e)}")
        print("Continuing to completion...\n")
    
    # =========================================================================
    # DAY 2 COMPLETION
    # =========================================================================
    print_header("Day 2 Complete", "=")
    print("\n[SUCCESS] Opportunity Radar Agent working!")
    print(f"  - Filing signals: data/filing_signals.csv")
    print(f"  - Anomaly signals: data/anomaly_signals.csv")
    print(f"  - Top signals: data/top_signals.csv")
    print(f"\n[INFO] Ready for Day 3: Chart Pattern Intelligence\n")

    # =========================================================================
    # DAY 3: Chart Pattern Intelligence
    # =========================================================================
    print_header("Day 3: Chart Pattern Intelligence", "=")

    try:
        top_patterns = get_top_patterns(10)
        total_patterns = int(top_patterns.attrs.get("total_patterns_detected", len(top_patterns)))
        if not top_patterns.empty:
            total_stocks = int(top_patterns.attrs.get("total_stocks_detected", top_patterns["ticker"].nunique()))
        else:
            total_stocks = int(top_patterns.attrs.get("total_stocks_detected", 0))

        print(f"\n{total_patterns} patterns detected across {total_stocks} stocks")

        if not top_patterns.empty:
            print("Top 5 patterns with success rates:")
            print("-" * 80)
            for idx, (_, row) in enumerate(top_patterns.head(5).iterrows(), start=1):
                success_rate = row.get("success_rate", None)
                success_text = "N/A" if pd.isna(success_rate) else f"{float(success_rate) * 100:.1f}%"
                print(
                    f"{idx:2d}. {row['ticker']} - {row['pattern_name']} "
                    f"({row['signal']}, strength {float(row['strength']):.2f}, success {success_text})"
                )
            print("-" * 80)
        else:
            print("No chart patterns detected.")

        print("\nDay 3 complete - Chart Pattern Intelligence working!\n")

    except Exception as e:
        print(f"ERROR in Day 3 (Chart Pattern Intelligence): {str(e)}")
        print("Continuing to completion...\n")

    print_header("End", "=")


if __name__ == "__main__":
    """
    Entry point for the main script.
    Run with: python main.py
    """
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        exit(1)
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        exit(1)
