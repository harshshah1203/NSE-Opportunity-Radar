"""
NSE Opportunity Radar - Reusable Streamlit Components
Provides UI components for the dashboard: badges, cards, charts, data loaders.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional, Tuple, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Debug: Print paths for verification
print(f"DEBUG: Components.py PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DEBUG: Components.py DATA_DIR = {DATA_DIR}")
print(f"DEBUG: DATA_DIR exists = {DATA_DIR.exists()}")


# ============================================================================
# UI COMPONENTS
# ============================================================================

def signal_badge(signal: str) -> None:
    """
    Display a colored signal badge (bullish/bearish/neutral).
    
    Args:
        signal: Signal type ('bullish', 'bearish', 'neutral')
    """
    signal = str(signal).lower().strip()
    
    if signal == "bullish":
        color = "#3df0c2"  # Green
        text = "📈 Bullish"
    elif signal == "bearish":
        color = "#ff6b6b"  # Red
        text = "📉 Bearish"
    else:
        color = "#ffd166"  # Yellow
        text = "➡️ Neutral"
    
    html = f'<span style="background-color: {color}; color: #0a0e1a; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 12px;">{text}</span>'
    st.markdown(html, unsafe_allow_html=True)


def confidence_bar(confidence: float) -> None:
    """
    Display a progress bar with confidence percentage.
    
    Args:
        confidence: Confidence value (0.0 to 1.0)
    """
    confidence = float(confidence)
    st.progress(min(confidence, 1.0))
    st.caption(f"Confidence: {confidence*100:.1f}%")


def urgency_chip(urgency: str) -> None:
    """
    Display an urgency chip with color coding.
    
    Args:
        urgency: Urgency level ('high', 'medium', 'low')
    """
    urgency = str(urgency).lower().strip()
    
    if urgency == "high":
        color = "#ff6b6b"  # Red
    elif urgency == "medium":
        color = "#ffa500"  # Orange
    else:
        color = "#4f7bff"  # Blue
    
    html = f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 8px; font-weight: bold; font-size: 11px;">{urgency.upper()}</span>'
    st.markdown(html, unsafe_allow_html=True)


def metric_card(title: str, value: str, delta: Optional[str] = None, color: Optional[str] = None) -> None:
    """
    Display a styled metric card.
    
    Args:
        title: Card title
        value: Main value to display
        delta: Optional delta indicator
        color: Optional accent color
    """
    col = st.container()
    with col:
        st.metric(label=title, value=value, delta=delta)


def candlestick_chart(ticker: str, df: pd.DataFrame) -> go.Figure:
    """
    Create a candlestick chart with SMA lines and volume.
    
    Args:
        ticker: Stock ticker symbol
        df: OHLCV dataframe with columns: Open, High, Low, Close, Volume
    
    Returns:
        Plotly figure object
    """
    if df.empty:
        st.error(f"No data available for {ticker}")
        return None
    
    # Normalize column names if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure we have required columns
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required):
        st.error(f"Missing OHLCV columns in data for {ticker}")
        return None
    
    df = df.copy()
    
    # Calculate SMAs
    df['SMA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['SMA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    
    # Create figure with secondary y-axis for volume
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        yaxis='y1'
    ))
    
    # Add SMA50 (blue dashed)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA50'],
        mode='lines',
        name='SMA50',
        line=dict(color='#4f7bff', dash='dash', width=2),
        yaxis='y1'
    ))
    
    # Add SMA200 (yellow dashed)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA200'],
        mode='lines',
        name='SMA200',
        line=dict(color='#ffd166', dash='dash', width=2),
        yaxis='y1'
    ))
    
    # Add volume bars (secondary y-axis)
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker=dict(color='#6378ff', opacity=0.3),
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title=f'{ticker} - 90 Day Price Chart',
        template='plotly_dark',
        xaxis_title='Date',
        yaxis=dict(title='Price (₹)', side='left', position=0),
        yaxis2=dict(title='Volume', side='right', overlaying='y'),
        height=500,
        hovermode='x unified',
        plot_bgcolor='#0a0e1a',
        paper_bgcolor='#0a0e1a',
        font=dict(color='white'),
    )
    
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create an RSI(14) chart with overbought/oversold zones.
    
    Args:
        df: OHLCV dataframe with Close prices
    
    Returns:
        Plotly figure object
    """
    if df.empty:
        return None
    
    df = df.copy()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        mode='lines',
        name='RSI(14)',
        line=dict(color='#6378ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(99, 120, 255, 0.1)'
    ))
    
    # Add overbought (70) line
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    
    # Add oversold (30) line
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    # Add shaded zones
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below")
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, layer="below")
    
    fig.update_layout(
        title='RSI(14) - Relative Strength Index',
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='RSI Value',
        height=400,
        hovermode='x unified',
        plot_bgcolor='#0a0e1a',
        paper_bgcolor='#0a0e1a',
        font=dict(color='white'),
        yaxis=dict(range=[0, 100]),
    )
    
    return fig


def pattern_card(pattern_row: pd.Series) -> None:
    """
    Render a styled pattern card with details.
    
    Args:
        pattern_row: Row from pattern_explanations dataframe
    """
    ticker = pattern_row.get('ticker', 'N/A')
    pattern_name = pattern_row.get('pattern_name', 'Unknown')
    signal = pattern_row.get('signal', 'neutral')
    strength = float(pattern_row.get('strength', 0.0))
    success_rate = pattern_row.get('success_rate', None)
    explanation = pattern_row.get('explanation', 'No explanation available.')
    current_price = pattern_row.get('current_price', 'N/A')
    
    # Convert success_rate to percentage if not None
    if pd.notna(success_rate):
        success_pct = float(success_rate) * 100
        success_text = f"{success_pct:.1f}%"
        if success_pct >= 65:
            success_color = "#3df0c2"  # Green
        elif success_pct >= 50:
            success_color = "#ffd166"  # Yellow
        else:
            success_color = "#ff6b6b"  # Red
    else:
        success_text = "N/A"
        success_color = "#999999"  # Gray
    
    # Color based on signal
    if signal.lower() == "bullish":
        border_color = "#3df0c2"
        signal_emoji = "📈"
    else:
        border_color = "#ff6b6b"
        signal_emoji = "📉"
    
    # Strength dots (5 dots)
    filled_dots = int(strength * 5)
    dots = "●" * filled_dots + "○" * (5 - filled_dots)
    
    with st.container():
        html = f"""
        <div style="
            border-left: 4px solid {border_color};
            background-color: #161c2e;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 12px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                <div>
                    <h4 style="margin: 0; font-family: monospace; color: white;">{signal_emoji} {ticker}</h4>
                    <p style="margin: 4px 0 0 0; color: #aaa; font-size: 12px;">{pattern_name}</p>
                </div>
                <span style="background-color: {success_color}; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 12px;">{success_text}</span>
            </div>
            <div style="margin-bottom: 8px;">
                <span style="font-size: 14px; color: #6378ff;">{dots}</span>
                <span style="color: #aaa; font-size: 12px; margin-left: 8px;">Strength: {strength:.2f}</span>
            </div>
            <p style="color: #ccc; font-size: 13px; line-height: 1.4; margin: 0;">{explanation}</p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# DATA LOADERS (with caching)
# ============================================================================

@st.cache_data(ttl=3600)
def load_top_signals() -> pd.DataFrame:
    """Load top ranked signals from CSV."""
    try:
        path = DATA_DIR / "top_signals.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Remove spaces from column names
        return df
    except Exception as e:
        st.error(f"Error loading top signals: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_filing_signals() -> pd.DataFrame:
    """Load filing signals from CSV."""
    try:
        path = DATA_DIR / "filing_signals.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Remove spaces from column names

        if 'ticker' not in df.columns:
            df['ticker'] = ''
        else:
            df['ticker'] = (
                df['ticker']
                .fillna('')
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace('.NS', '', regex=False)
            )

        if 'company_name' in df.columns:
            try:
                from utils.filing_scraper import infer_ticker_from_company_name
                missing_ticker = df['ticker'] == ''
                if missing_ticker.any():
                    df.loc[missing_ticker, 'ticker'] = (
                        df.loc[missing_ticker, 'company_name']
                        .fillna('')
                        .astype(str)
                        .apply(infer_ticker_from_company_name)
                    )
            except Exception:
                pass

        return df
    except Exception as e:
        st.error(f"Error loading filing signals: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_anomaly_signals() -> pd.DataFrame:
    """Load anomaly signals from CSV."""
    try:
        path = DATA_DIR / "anomaly_signals.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Remove spaces from column names
        return df
    except Exception as e:
        st.error(f"Error loading anomaly signals: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_pattern_explanations() -> pd.DataFrame:
    """Load pattern explanations from CSV."""
    try:
        path = DATA_DIR / "pattern_explanations.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Remove spaces from column names
        return df
    except Exception as e:
        st.error(f"Error loading pattern explanations: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load OHLCV data for a stock from parquet."""
    try:
        path = DATA_DIR / "stocks" / f"{ticker}.NS.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Error loading stock data for {ticker}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_bse_filings() -> pd.DataFrame:
    """Load BSE filings from CSV."""
    try:
        path = DATA_DIR / "bse_filings.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Remove spaces from column names
        return df
    except Exception as e:
        st.error(f"Error loading BSE filings: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_pattern_signals() -> pd.DataFrame:
    """Load raw pattern signals from CSV."""
    try:
        path = DATA_DIR / "pattern_signals.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()  # Remove spaces from column names
        return df
    except Exception as e:
        st.error(f"Error loading pattern signals: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_available_tickers() -> List[str]:
    """
    Build a unified ticker universe from cached stock files and signal datasets.
    """
    tickers = set()

    stocks_dir = DATA_DIR / "stocks"
    if stocks_dir.exists():
        for parquet_file in stocks_dir.glob("*.parquet"):
            ticker = parquet_file.stem.upper().replace(".NS", "")
            if ticker:
                tickers.add(ticker)

    for file_name in ["top_signals.csv", "filing_signals.csv", "anomaly_signals.csv", "pattern_signals.csv", "bse_filings.csv"]:
        path = DATA_DIR / file_name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, skipinitialspace=True)
            if "ticker" in df.columns:
                values = (
                    df["ticker"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(".NS", "", regex=False)
                )
                tickers.update(v for v in values if v)
        except Exception:
            continue

    if not tickers:
        try:
            from utils.data_fetcher import get_top_nse_tickers
            tickers.update(t.upper().strip() for t in get_top_nse_tickers() if str(t).strip())
        except Exception:
            pass

    return sorted(tickers)


def load_latest_filings(n: int = 20) -> pd.DataFrame:
    """Load latest N filings from BSE CSV."""
    df = load_bse_filings()
    if df.empty:
        return df
    return df.head(n)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_last_update_time(filepath: str) -> str:
    """Get file modification time as readable string."""
    try:
        if os.path.exists(filepath):
            mtime = os.path.getmtime(filepath)
            from datetime import datetime
            dt = datetime.fromtimestamp(mtime)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    return "Unknown"


def check_all_data_exists() -> Tuple[bool, list]:
    """
    Check if all required data files exist.
    
    Returns:
        Tuple of (all_exist: bool, missing_files: list)
    """
    required_files = [
        DATA_DIR / "top_signals.csv",
        DATA_DIR / "filing_signals.csv",
        DATA_DIR / "anomaly_signals.csv",
        DATA_DIR / "pattern_explanations.csv",
        DATA_DIR / "bse_filings.csv",
        DATA_DIR / "pattern_signals.csv",
    ]
    
    missing = [f.name for f in required_files if not f.exists()]
    
    # Debug: Show what we found
    print(f"DEBUG: Checking files in {DATA_DIR}")
    for f in required_files:
        exists = f.exists()
        print(f"  {f.name}: {'✓' if exists else '✗'} ({f})")
    
    return len(missing) == 0, missing
