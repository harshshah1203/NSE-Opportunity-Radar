"""
NSE Opportunity Radar - Main Streamlit Dashboard
AI-Powered Stock Signal Intelligence for Indian Retail Investors
"""

import os
import sys
import subprocess
from html import escape
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dashboard.components import (
        signal_badge,
        confidence_bar,
        urgency_chip,
        candlestick_chart,
        rsi_chart,
        pattern_card,
        load_top_signals,
        load_filing_signals,
        load_anomaly_signals,
        load_pattern_explanations,
        load_stock_data,
        load_bse_filings,
        load_pattern_signals,
        load_latest_filings,
        load_available_tickers,
        get_last_update_time,
        check_all_data_exists,
    )
except ImportError:
    # Fallback for relative imports
    from components import (
        signal_badge,
        confidence_bar,
        urgency_chip,
        candlestick_chart,
        rsi_chart,
        pattern_card,
        load_top_signals,
        load_filing_signals,
        load_anomaly_signals,
        load_pattern_explanations,
        load_stock_data,
        load_bse_filings,
        load_pattern_signals,
        load_latest_filings,
        load_available_tickers,
        get_last_update_time,
        check_all_data_exists,
    )


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="NSE Opportunity Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
HIGH_CONFIDENCE_SUCCESS_THRESHOLD = 0.80

print(f"DEBUG: App.py PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DEBUG: App.py DATA_DIR = {DATA_DIR}")
print(f"DEBUG: DATA_DIR exists = {DATA_DIR.exists()}")


# ============================================================================
# CUSTOM CSS
# ============================================================================

custom_css = """
<style>
    :root {
        --bg-dark: #0a0e1a;
        --bg-card: #161c2e;
        --accent: #6378ff;
        --bullish: #3df0c2;
        --bearish: #ff6b6b;
        --neutral: #ffd166;
    }
    
    * {
        color: white;
    }
    
    body {
        background-color: var(--bg-dark) !important;
    }
    
    .main {
        background-color: var(--bg-dark) !important;
        padding: 0px !important;
    }
    
    .stMainBlockContainer {
        background-color: var(--bg-dark) !important;
        padding: 2rem 1rem !important;
    }
    
    .stSidebar {
        background-color: var(--bg-card) !important;
    }
    
    [data-testid="metric-container"] {
        background-color: var(--bg-card) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    .card {
        background-color: var(--bg-card) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        margin-bottom: 16px !important;
    }
    
    /* Streamlit buttons */
    .stButton > button {
        background-color: var(--accent) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
    }
    
    .stButton > button:hover {
        background-color: #537fff !important;
    }
    
    /* Selectbox and inputs */
    .stSelectbox, .stMultiSelect, .stTextInput, .stSlider {
        background-color: var(--bg-card) !important;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background-color: var(--bg-card) !important;
    }
    
    th {
        background-color: #0a0e1a !important;
        color: var(--accent) !important;
        font-weight: bold !important;
    }
    
    td {
        background-color: var(--bg-card) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--bg-card) !important;
        border-radius: 8px !important;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    
    /* Code and monospace */
    code {
        background-color: var(--bg-card) !important;
        color: var(--accent) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render sidebar with navigation and controls."""
    st.sidebar.markdown("## 📡 NSE Opportunity Radar")
    st.sidebar.markdown("---")
    
    # Navigation
    st.session_state.page = st.sidebar.radio(
        "**Navigate:**",
        ["🎯 Opportunity Radar", "📈 Chart Patterns", "🔍 Stock Deep Dive"],
        key="nav_tab"
    )
    
    st.sidebar.markdown("---")
    
    # Refresh button
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            with st.spinner("Running pipeline..."):
                try:
                    result = subprocess.run(
                        ["python", str(PROJECT_ROOT / "main.py")],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    if result.returncode == 0:
                        st.success("✅ Pipeline completed successfully!")
                        # Clear cache to reload data
                        st.cache_data.clear()
                    else:
                        st.error(f"Pipeline failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("Pipeline timed out (10 minutes)")
                except Exception as e:
                    st.error(f"Error running pipeline: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
    
    st.sidebar.markdown("---")
    
    # Data stats
    st.sidebar.subheader("📊 Data Stats")
    
    bse_filings = load_bse_filings()
    filing_count = len(bse_filings)
    tracked_tickers = load_available_tickers()
    
    st.sidebar.metric("Stocks Tracked", str(len(tracked_tickers)))
    st.sidebar.metric("Filings Analyzed", f"{filing_count:,}")
    
    top_signals_path = DATA_DIR / "top_signals.csv"
    if top_signals_path.exists():
        last_update = get_last_update_time(str(top_signals_path))
        st.sidebar.caption(f"Last updated: {last_update}")
    
    # Check data files
    all_exist, missing = check_all_data_exists()
    if all_exist:
        st.sidebar.success("✅ All data loaded")
    else:
        st.sidebar.warning(f"⚠️ Missing files: {', '.join(missing)}")
        st.sidebar.caption("Run `python main.py` to generate data")


# ============================================================================
# PAGE: OPPORTUNITY RADAR
# ============================================================================

def page_opportunity_radar():
    """Render the Opportunity Radar tab."""
    
    st.markdown("## 🎯 Opportunity Radar")
    st.markdown("Real-time AI-powered stock signals and anomalies")
    st.markdown("---")
    
    # SECTION 1: Top Ranked Opportunities
    st.subheader("📊 Top Ranked Opportunities")
    
    top_signals = load_top_signals()
    
    if top_signals.empty:
        st.warning("No opportunity signals available. Run main.py to generate data.")
        return
    
    # Display top signals table with styling
    display_cols = ['ticker', 'final_score', 'filing_signal', 'anomaly_type', 'top_reason']
    available_cols = [col for col in display_cols if col in top_signals.columns]
    
    top_signals_display = top_signals[available_cols].reset_index(drop=True)
    top_signals_display.index = top_signals_display.index + 1
    top_signals_display.index.name = "Rank"
    
    # Convert to styled dataframe
    st.dataframe(
        top_signals_display,
        use_container_width=True,
        height=300,
    )
    
    # Make tickers clickable by storing selection
    selected_ticker = st.selectbox(
        "👇 Select a stock to analyze:",
        top_signals['ticker'].unique(),
        key="opp_ticker_select"
    )
    
    if selected_ticker:
        st.session_state.selected_ticker = selected_ticker
    
    st.markdown("---")
    
    # SECTION 2: Two columns - Filing Signals + Anomalies
    col1, col2 = st.columns(2)
    
    # LEFT COLUMN: Filing Signals
    with col1:
        st.subheader("📋 Filing Signals")
        
        filing_signals = load_filing_signals()
        
        if not filing_signals.empty:
            # Filters
            signal_types = filing_signals['signal'].unique()
            selected_signals = st.multiselect(
                "Filter by signal:",
                signal_types,
                default=signal_types,
                key="filing_signals_filter"
            )
            
            min_confidence = st.slider(
                "Minimum confidence:",
                0.0, 1.0, 0.5, 0.05,
                key="filing_confidence_slider"
            )
            
            # Filter data
            filtered_filings = filing_signals[
                (filing_signals['signal'].isin(selected_signals)) &
                (filing_signals['confidence'] >= min_confidence)
            ].copy()
            
            if not filtered_filings.empty:
                # Prepare display
                display_cols_filing = ['ticker', 'signal', 'confidence', 'event_type', 'urgency', 'reason']
                available_cols_filing = [col for col in display_cols_filing if col in filtered_filings.columns]
                
                st.dataframe(
                    filtered_filings[available_cols_filing].head(10),
                    use_container_width=True,
                    height=300,
                )
            else:
                st.info("No signals match the selected filters.")
        else:
            st.warning("No filing signals available.")
    
    # RIGHT COLUMN: Volume Anomalies
    with col2:
        st.subheader("🚨 Volume Anomalies")
        
        anomaly_signals = load_anomaly_signals()
        
        if not anomaly_signals.empty:
            # Show top 10 anomalies as cards
            for idx, row in anomaly_signals.head(10).iterrows():
                ticker = row.get('ticker', 'N/A')
                anomaly_type = row.get('anomaly_type', 'Unknown')
                key_metric = row.get('key_metric', 'N/A')
                explanation = row.get('explanation', '')
                
                # Determine card color
                is_bullish = 'bullish' in str(explanation).lower()
                border_color = "#3df0c2" if is_bullish else "#ff6b6b"
                
                html = f"""
                <div style="
                    border-left: 4px solid {border_color};
                    background-color: #161c2e;
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                ">
                    <div style="font-family: monospace; font-size: 16px; font-weight: bold; color: white;">
                        {ticker}
                    </div>
                    <div style="color: #aaa; font-size: 12px; margin-top: 4px;">
                        {anomaly_type}
                    </div>
                    <div style="color: #6378ff; font-weight: bold; font-size: 13px; margin-top: 4px;">
                        {key_metric}
                    </div>
                    <div style="color: #ccc; font-size: 12px; margin-top: 6px; line-height: 1.3;">
                        {str(explanation)[:150]}...
                    </div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No anomalies detected.")
    
    st.markdown("---")
    
    # SECTION 3: Latest NSE Filings Feed
    st.subheader("📰 Latest NSE Filings Feed")
    
    all_filings = load_bse_filings()
    
    if not all_filings.empty:
        # Search bar for company name
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_company = st.text_input(
                "🔍 Search company name:",
                placeholder="e.g., HDFC Bank, INFY, TCS, RELIANCE...",
                key="filing_search"
            )
        
        with col2:
            show_count = st.selectbox(
                "Show results:",
                [10, 20, 50, 100],
                index=1,
                key="filing_count"
            )
        
        # Filter filings based on search
        if search_company.strip():
            search_text = search_company.strip()
            company_match = all_filings['company_name'].str.contains(search_text, case=False, na=False)
            ticker_match = (
                all_filings['ticker'].astype(str).str.contains(search_text, case=False, na=False)
                if 'ticker' in all_filings.columns else False
            )
            filtered_filings = all_filings[company_match | ticker_match]
            results_text = f"Found {len(filtered_filings)} filings for '{search_company}'"
        else:
            filtered_filings = all_filings
            results_text = f"Showing latest {show_count} filings"
        
        st.caption(f"📊 {results_text}")
        
        if not filtered_filings.empty:
            # Limit display
            display_filings = filtered_filings.head(show_count if not search_company.strip() else len(filtered_filings))
            
            for idx, row in display_filings.iterrows():
                company = row.get('company_name', 'N/A')
                filing_type = row.get('filing_type', 'Update')
                description = row.get('description', '')
                filing_date = row.get('filing_date', 'N/A')
                filing_url = str(row.get('url', '')).strip()
                has_valid_link = filing_url.startswith("http://") or filing_url.startswith("https://")
                
                html = f"""
                <div style="
                    background-color: #161c2e;
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                    border-left: 4px solid #6378ff;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 6px;">
                        <div style="font-weight: bold; color: white;">
                            {company}
                        </div>
                        <span style="background-color: #6378ff; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">
                            {filing_type}
                        </span>
                    </div>
                    <div style="color: #ccc; font-size: 13px; margin-bottom: 6px;">
                        {str(description)[:120]}...
                    </div>
                    <div style="color: #888; font-size: 11px;">
                        📅 {filing_date}
                    </div>
                </div>
                """
                if has_valid_link:
                    safe_url = escape(filing_url, quote=True)
                    clickable_html = (
                        f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
                        f'style="display:block; text-decoration:none; color:inherit;" '
                        f'title="Open filing on NSE">{html}</a>'
                    )
                    st.markdown(clickable_html, unsafe_allow_html=True)
                else:
                    st.markdown(html, unsafe_allow_html=True)
                    st.caption("NSE link not available for this filing.")
        else:
            st.info(f"No filings found for '{search_company}'")
    else:
        st.warning("No filings data available. Run main.py to generate data.")


# ============================================================================
# PAGE: CHART PATTERNS
# ============================================================================

def page_chart_patterns():
    """Render the Chart Patterns tab."""
    
    st.markdown("## 📈 Chart Patterns")
    st.markdown("Detected technical patterns with AI explanations")
    st.markdown("---")
    
    # SECTION 1: Pattern Summary Bar
    st.subheader("📊 Pattern Detection Summary")
    
    pattern_signals = load_pattern_signals()
    
    if not pattern_signals.empty:
        pattern_counts = pattern_signals['pattern_name'].value_counts()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            ("Golden Cross", pattern_counts.get("Golden Cross", 0)),
            ("Death Cross", pattern_counts.get("Death Cross", 0)),
            ("Breakout", pattern_counts.get("Breakout", 0)),
            ("RSI Divergence", pattern_counts.get("RSI Divergence (Bullish)", 0)),
            ("Support Bounce", pattern_counts.get("Support Bounce", 0)),
        ]
        
        for col, (name, count) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                st.metric(name, count)
    
    st.markdown("---")
    
    # SECTION 2: Two columns - Pattern Cards + Pattern Chart
    col1, col2 = st.columns([1, 1])
    
    # LEFT COLUMN: Pattern Cards
    with col1:
        st.subheader("🎯 Detected Patterns")
        
        pattern_explanations = load_pattern_explanations()
        
        if not pattern_explanations.empty:
            # Filter and sort options
            sort_by = st.selectbox(
                "Sort by:",
                ["Success Rate", "Strength", "Signal"],
                key="pattern_sort"
            )
            
            filter_signal = st.selectbox(
                "Filter by signal:",
                ["All", "Bullish", "Bearish"],
                key="pattern_filter"
            )
            
            # Apply filters
            filtered_patterns = pattern_explanations.copy()
            
            if filter_signal != "All":
                filtered_patterns = filtered_patterns[
                    filtered_patterns['signal'].str.lower() == filter_signal.lower()
                ]
            
            # Sort
            if sort_by == "Success Rate":
                filtered_patterns = filtered_patterns.sort_values(
                    'success_rate',
                    ascending=False,
                    na_position='last'
                )
            elif sort_by == "Strength":
                filtered_patterns = filtered_patterns.sort_values('strength', ascending=False)
            
            # Display pattern cards
            for idx, row in filtered_patterns.head(10).iterrows():
                pattern_card(row)
        else:
            st.info("No patterns detected yet.")
    
    # RIGHT COLUMN: Pattern Chart
    with col2:
        st.subheader("📊 Pattern Chart Analysis")
        
        pattern_signals = load_pattern_signals()
        
        available_tickers = load_available_tickers()

        if available_tickers:
            # Select ticker from full tracked universe
            selected_ticker = st.selectbox(
                "Select stock:",
                available_tickers,
                key="pattern_chart_ticker"
            )
            
            stock_data = load_stock_data(selected_ticker)
            
            if not stock_data.empty:
                # Show candlestick chart
                fig_candle = candlestick_chart(selected_ticker, stock_data)
                if fig_candle:
                    st.plotly_chart(fig_candle, use_container_width=True)
                
                # Show RSI chart
                fig_rsi = rsi_chart(stock_data)
                if fig_rsi:
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                # Show patterns for this ticker
                st.markdown("**Patterns detected for this stock:**")
                
                ticker_patterns = (
                    pattern_signals[pattern_signals['ticker'] == selected_ticker]
                    if not pattern_signals.empty else pd.DataFrame()
                )
                
                if not ticker_patterns.empty:
                    for idx, row in ticker_patterns.iterrows():
                        pattern_name = row.get('pattern_name', 'Unknown')
                        signal = row.get('signal', 'neutral')
                        st.caption(f"• {pattern_name} ({signal})")
                else:
                    st.info(f"No patterns detected for {selected_ticker}")
            else:
                st.error(f"No data found for {selected_ticker}")
        else:
            st.info("No pattern signals available.")
    
    st.markdown("---")
    
    # SECTION 3: Backtesting Summary
    st.subheader("📊 Historical Pattern Success Rates")
    
    pattern_explanations = load_pattern_explanations()
    
    if not pattern_explanations.empty:
        # Group by pattern name and calculate average success rate
        pattern_success = pattern_explanations.groupby('pattern_name')['success_rate'].apply(
            lambda x: (x.dropna().mean() * 100) if len(x.dropna()) > 0 else 0
        ).sort_values(ascending=False)
        
        if not pattern_success.empty:
            # Create bar chart
            colors = []
            for rate in pattern_success.values:
                if rate >= 65:
                    colors.append("#3df0c2")  # Green
                elif rate >= 50:
                    colors.append("#ffd166")  # Yellow
                else:
                    colors.append("#ff6b6b")  # Red
            
            fig = go.Figure(data=[
                go.Bar(
                    x=pattern_success.index,
                    y=pattern_success.values,
                    marker=dict(color=colors),
                    text=[f"{v:.1f}%" for v in pattern_success.values],
                    textposition="auto",
                )
            ])
            
            fig.update_layout(
                title="NSE Universe - Pattern Success Rates (Last 5 Trading Days)",
                xaxis_title="Pattern Type",
                yaxis_title="Success Rate (%)",
                template="plotly_dark",
                height=400,
                plot_bgcolor="#0a0e1a",
                paper_bgcolor="#0a0e1a",
                font=dict(color="white"),
            )
            
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: STOCK DEEP DIVE
# ============================================================================

def page_stock_deep_dive():
    """Render the Stock Deep Dive tab."""
    
    st.markdown("## 🔍 Stock Deep Dive")
    st.markdown("Comprehensive analysis for any stock")
    st.markdown("---")
    
    # SECTION 1: Search
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker_input = st.text_input(
            "Search NSE ticker:",
            placeholder="e.g. RELIANCE, TCS, INFY, HDFCBANK...",
            key="deep_dive_search"
        )
    
    with col2:
        # Auto-complete from full tracked universe
        available_tickers = load_available_tickers()
        if available_tickers:
            auto_complete_tickers = st.selectbox(
                "Or select from tracked stocks:",
                [""] + available_tickers,
                key="deep_dive_autocomplete"
            )
            if auto_complete_tickers and not ticker_input:
                ticker_input = auto_complete_tickers
    
    if not ticker_input:
        st.info("👆 Enter a ticker symbol to begin analysis")
        return
    
    ticker = ticker_input.upper().strip()
    stock_data = load_stock_data(ticker)
    
    if stock_data.empty:
        st.error(f"❌ No data found for {ticker}. Ensure data/stocks/{ticker}.NS.parquet exists.")
        return
    
    st.markdown("---")
    
    # SECTION 2: Stock Header
    st.subheader(f"{ticker}")
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    # Get price info
    current_price = stock_data['Close'].iloc[-1]
    prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close * 100) if prev_close != 0 else 0
    
    high_52w = stock_data['High'].max()
    low_52w = stock_data['Low'].min()
    avg_volume_30d = stock_data['Volume'].tail(30).mean()
    current_volume = stock_data['Volume'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Current Price",
            f"₹{current_price:.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        st.metric("📊 52-Week High", f"₹{high_52w:.2f}")
    
    with col3:
        st.metric("📊 52-Week Low", f"₹{low_52w:.2f}")
    
    with col4:
        st.metric(
            "📈 Volume vs 30d",
            f"{current_volume/avg_volume_30d:.2f}x",
            f"{current_volume:,.0f}"
        )
    
    st.markdown("---")
    
    # SECTION 3: Charts
    fig_candle = candlestick_chart(ticker, stock_data)
    if fig_candle:
        st.plotly_chart(fig_candle, use_container_width=True)
    
    fig_rsi = rsi_chart(stock_data)
    if fig_rsi:
        st.plotly_chart(fig_rsi, use_container_width=True)
    
    st.markdown("---")
    
    # SECTION 4: Two columns - Signals + AI Summary
    col1, col2 = st.columns([1, 1])
    
    # LEFT COLUMN: All Signals
    with col1:
        st.subheader("📋 All Signals for this Stock")
        
        # Filing signals
        filing_signals = load_filing_signals()
        ticker_filings = filing_signals[filing_signals['ticker'] == ticker]
        
        if not ticker_filings.empty:
            with st.expander("📄 Filing Signals", expanded=True):
                for idx, row in ticker_filings.iterrows():
                    signal = row.get('signal', 'neutral')
                    confidence = row.get('confidence', 0.0)
                    reason = row.get('reason', '')
                    
                    st.write(f"**Signal:** {signal.upper()}")
                    st.write(f"**Confidence:** {confidence*100:.1f}%")
                    st.write(f"**Reason:** {reason}")
                    st.divider()

        # Raw filings feed for this ticker (independent of AI analysis)
        all_filings = load_bse_filings()
        if not all_filings.empty and 'ticker' in all_filings.columns:
            filings_work = all_filings.copy()
            filings_work['ticker'] = (
                filings_work['ticker']
                .fillna('')
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace('.NS', '', regex=False)
            )
            ticker_raw_filings = filings_work[filings_work['ticker'] == ticker].head(10)

            if not ticker_raw_filings.empty:
                with st.expander("📰 Latest NSE Filings", expanded=ticker_filings.empty):
                    for idx, row in ticker_raw_filings.iterrows():
                        filing_type = row.get('filing_type', 'Update')
                        filing_date = row.get('filing_date', 'N/A')
                        filing_url = str(row.get('url', '')).strip()
                        if filing_url.startswith("http://") or filing_url.startswith("https://"):
                            st.markdown(f"• **{filing_type}** ({filing_date}) - [Open NSE filing]({filing_url})")
                        else:
                            st.markdown(f"• **{filing_type}** ({filing_date})")
        
        # Pattern detections
        pattern_explanations = load_pattern_explanations()
        ticker_patterns = pattern_explanations[pattern_explanations['ticker'] == ticker]
        
        if not ticker_patterns.empty:
            with st.expander("📈 Pattern Detections", expanded=True):
                for idx, row in ticker_patterns.iterrows():
                    st.write(f"**{row.get('pattern_name', 'Unknown')}**")
                    st.write(f"Signal: {row.get('signal', 'neutral')}")
                    st.write(f"Strength: {row.get('strength', 0):.2f}")
                    st.write(f"Success Rate: {row.get('success_rate', 'N/A')}")
                    st.divider()
        
        # Anomalies
        anomaly_signals = load_anomaly_signals()
        ticker_anomalies = anomaly_signals[anomaly_signals['ticker'] == ticker]
        
        if not ticker_anomalies.empty:
            with st.expander("🚨 Anomalies", expanded=True):
                for idx, row in ticker_anomalies.iterrows():
                    st.write(f"**Type:** {row.get('anomaly_type', 'Unknown')}")
                    st.write(f"**Details:** {row.get('explanation', 'N/A')}")
                    st.divider()
    
    # RIGHT COLUMN: AI Summary
    with col2:
        st.subheader("🤖 AI Outlook")
        
        # Determine overall sentiment
        all_signals = []
        if not ticker_filings.empty:
            all_signals.extend(ticker_filings['signal'].tolist())
        if not ticker_patterns.empty:
            all_signals.extend(ticker_patterns['signal'].tolist())
        
        if all_signals:
            bullish_count = sum(1 for s in all_signals if s.lower() == 'bullish')
            bearish_count = sum(1 for s in all_signals if s.lower() == 'bearish')
            
            if bullish_count > bearish_count:
                sentiment = "📈 Bullish"
                sentiment_color = "#3df0c2"
            elif bearish_count > bullish_count:
                sentiment = "📉 Bearish"
                sentiment_color = "#ff6b6b"
            else:
                sentiment = "➡️ Neutral"
                sentiment_color = "#ffd166"
            
            st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}; font-size: 18px;'>{sentiment}</span>", unsafe_allow_html=True)
        
        # Key reasons
        if not ticker_filings.empty:
            st.markdown("**Key Reasons:**")
            for idx, row in ticker_filings.iterrows():
                st.markdown(f"• {row.get('reason', 'N/A')}")
        
        # Patterns summary
        if not ticker_patterns.empty:
            st.markdown("**Patterns Detected:**")
            for idx, row in ticker_patterns.iterrows():
                pattern = row.get('pattern_name', 'Unknown')
                success = row.get('success_rate', None)
                success_text = f" ({success*100:.1f}% success)" if pd.notna(success) else ""
                st.markdown(f"• {pattern}{success_text}")
        
        # Key levels
        st.markdown("**Key Price Levels:**")
        
        support_20d = stock_data['Low'].tail(20).min()
        resistance_20d = stock_data['High'].tail(20).max()
        sma50 = stock_data['Close'].rolling(50).mean().iloc[-1]
        sma200 = stock_data['Close'].rolling(200).mean().iloc[-1] if len(stock_data) >= 200 else None
        
        st.markdown(f"• **Support (20d):** ₹{support_20d:.2f}")
        st.markdown(f"• **Resistance (20d):** ₹{resistance_20d:.2f}")
        st.markdown(f"• **SMA50:** ₹{sma50:.2f}")
        if sma200:
            st.markdown(f"• **SMA200:** ₹{sma200:.2f}")
        
        st.warning("⚠️ **Disclaimer:** This analysis is for informational purposes. Conduct your own research before investing.")
    
    st.markdown("---")
    
    # SECTION 5: Raw Data
    with st.expander("📊 View Raw OHLCV Data"):
        st.dataframe(stock_data, use_container_width=True, height=400)
        
        csv = stock_data.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"{ticker}_OHLCV.csv",
            mime="text/csv",
        )


# ============================================================================
# HEADER
# ============================================================================

def render_header():
    """Render main header with metrics."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("# 📡 NSE Opportunity Radar")
        st.markdown("**AI-Powered Signal Intelligence for Indian Retail Investors**")
    
    with col2:
        st.markdown("")  # Spacing
    
    st.markdown("---")
    
    # Summary metrics
    filing_signals = load_filing_signals()
    pattern_explanations = load_pattern_explanations()
    
    bullish_count = len(filing_signals[filing_signals['signal'] == 'bullish']) if not filing_signals.empty else 0
    bearish_count = len(filing_signals[filing_signals['signal'] == 'bearish']) if not filing_signals.empty else 0
    patterns_count = len(pattern_explanations) if not pattern_explanations.empty else 0
    
    avg_success_rate = 0
    qualified_pattern_count = 0
    if not pattern_explanations.empty:
        success_rates = pd.to_numeric(pattern_explanations['success_rate'], errors='coerce').dropna()
        if len(success_rates) > 0:
            qualified_success_rates = success_rates[success_rates >= HIGH_CONFIDENCE_SUCCESS_THRESHOLD]
            if len(qualified_success_rates) > 0:
                avg_success_rate = qualified_success_rates.mean() * 100
                qualified_pattern_count = len(qualified_success_rates)
            else:
                avg_success_rate = success_rates.mean() * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Bullish Signals", bullish_count)
    
    with col2:
        st.metric("📉 Bearish Signals", bearish_count)
    
    with col3:
        st.metric("🎯 Patterns Detected", patterns_count)
    
    with col4:
        if qualified_pattern_count > 0:
            st.metric(
                f"✅ Avg Success (≥{int(HIGH_CONFIDENCE_SUCCESS_THRESHOLD * 100)}%)",
                f"{avg_success_rate:.1f}%"
            )
        else:
            st.metric("✅ Avg Success Rate", f"{avg_success_rate:.1f}%")
    
    # Last update time
    top_signals_path = DATA_DIR / "top_signals.csv"
    if top_signals_path.exists():
        last_update = get_last_update_time(str(top_signals_path))
        st.caption(f"🕐 Last data refresh: {last_update}")
    
    st.markdown("---")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "🎯 Opportunity Radar"
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None
    
    # Render sidebar
    render_sidebar()
    
    # Render header
    render_header()
    
    # Check if data exists
    all_exist, missing = check_all_data_exists()
    
    if not all_exist:
        st.error(f"❌ Missing data files: {', '.join(missing)}")
        st.info("Run `python main.py` from the project directory to generate the required data files.")
        return
    
    # Route to selected page
    if st.session_state.page == "🎯 Opportunity Radar":
        page_opportunity_radar()
    elif st.session_state.page == "📈 Chart Patterns":
        page_chart_patterns()
    elif st.session_state.page == "🔍 Stock Deep Dive":
        page_stock_deep_dive()


if __name__ == "__main__":
    main()
