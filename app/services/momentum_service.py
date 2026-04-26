import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Any, Optional
import os
import json
import requests
import io

logger = logging.getLogger(__name__)

INDEX_URLS = {
    "NIFTY 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY NEXT 50": "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "NIFTY 100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY 200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY 500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "MIDCAP 100": "https://archives.nseindia.com/content/indices/ind_niftymidcap100list.csv",
    "SMALLCAP 100": "https://archives.nseindia.com/content/indices/ind_niftysmallcap100list.csv",
    "NIFTY BANK": "https://archives.nseindia.com/content/indices/ind_niftybanklist.csv",
    "NIFTY IT": "https://archives.nseindia.com/content/indices/ind_niftyitlist.csv",
    "NIFTY PHARMA": "https://archives.nseindia.com/content/indices/ind_niftypharmalist.csv"
}

def fetch_index_symbols(index_name: str) -> List[str]:
    """Fetch constituents of an index from official NSE CSV links."""
    url = INDEX_URLS.get(index_name.upper())
    if not url:
        # Try finding partial match if exact upper fails
        for k, v in INDEX_URLS.items():
            if index_name.upper() in k.upper():
                url = v
                break
    
    if not url:
        logger.error(f"No URL found for index: {index_name}")
        return []

    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        if "Symbol" in df.columns:
            return df["Symbol"].tolist()
        return []
    except Exception as e:
        logger.error(f"Error fetching index constituents for {index_name}: {e}")
        return []

def fetch_daily_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV data for a stock."""
    # Append .NS for NSE stocks
    ticker_sym = f"{symbol}.NS"
    try:
        df = yf.download(ticker_sym, start=start_date, end=end_date, interval="1d", progress=False)
        if df is None or df.empty:
            return None
        
        # Flatten multi-index if present (sometimes happens with yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        # Consolidate column names
        df.columns = [c.upper() for c in df.columns]
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_momentum(df: pd.DataFrame, period: str = "1y") -> Dict[str, Any]:
    """Calculate momentum metrics and score."""
    if df is None or len(df) < 2:
        raise ValueError("Insufficient data")

    # EMA Calculations
    close_prices = df["CLOSE"]
    ema50 = close_prices.ewm(span=50, adjust=False).mean()
    ema200 = close_prices.ewm(span=200, adjust=False).mean()
    
    current_close = float(close_prices.iloc[-1])
    
    # Calculate return for the requested period based on exact calendar date
    period_kwargs = {
        "1m": {"months": 1},
        "3m": {"months": 3},
        "6m": {"months": 6},
        "1y": {"years": 1},
        "2y": {"years": 2},
        "3y": {"years": 3}
    }
    
    kwargs = period_kwargs.get(period.lower())
    if kwargs:
        target_date = df['DATE'].iloc[-1] - pd.DateOffset(**kwargs)
        past_df = df[df['DATE'] <= target_date]
        if not past_df.empty:
            start_close = float(past_df["CLOSE"].iloc[-1])
        else:
            start_close = float(close_prices.iloc[0])
    else:
        start_close = float(close_prices.iloc[0])
    
    # Return % for the SELECTED period (used for score)
    return_pct = ((current_close - start_close) / start_close) * 100
    
    # Volume metrics
    avg_volume = float(df["VOLUME"].mean())
    recent_vol_avg = float(df["VOLUME"].tail(20).mean())
    volume_trend = recent_vol_avg / avg_volume if avg_volume > 0 else 1.0
    
    # EMA Trend
    last_ema50 = float(ema50.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])
    is_above_ema50 = current_close > last_ema50
    is_above_ema200 = current_close > last_ema200
    
    # Momentum Score calculation
    # Normalized volume trend (capped at 2.0 for scoring 0.0-1.0 range-ish)
    vol_norm = min(max(volume_trend / 2.0, 0), 1.0)
    
    # return_pct contribution (normalized - let's say 50% return = 1.0)
    ret_norm = min(max(return_pct / 50.0, 0), 1.0) if return_pct > 0 else 0
    
    # Daily Return % (Today vs Yesterday)
    daily_return = 0.0
    if len(close_prices) >= 2:
        yesterday_close = float(close_prices.iloc[-2])
        daily_return = ((current_close - yesterday_close) / yesterday_close) * 100

    # Calculate multiple returns based on calendar dates
    def get_return(months=0, years=0):
        target_date = df['DATE'].iloc[-1] - pd.DateOffset(months=months, years=years)
        past_df = df[df['DATE'] <= target_date]
        if not past_df.empty:
            past_close = float(past_df["CLOSE"].iloc[-1])
            return round(((current_close - past_close) / past_close) * 100, 2)
        return None

    ret_1m = get_return(months=1)
    ret_3m = get_return(months=3)
    ret_6m = get_return(months=6)
    ret_1y = get_return(years=1)

    score = (ret_norm * 60) + (vol_norm * 20) + (10 if is_above_ema50 else 0) + (10 if is_above_ema200 else 0)
    
    return {
        "current_close": round(current_close, 2),
        "start_close": round(start_close, 2),
        "return_pct": round(return_pct, 2),
        "return_1m": ret_1m,
        "return_3m": ret_3m,
        "return_6m": ret_6m,
        "return_1y": ret_1y,
        "daily_return": round(daily_return, 2),
        "avg_volume": round(avg_volume, 0),
        "volume_trend": round(volume_trend, 2),
        "ema50": round(last_ema50, 2),
        "ema200": round(last_ema200, 2),
        "is_above_ema50": is_above_ema50,
        "is_above_ema200": is_above_ema200,
        "momentum_score": round(score, 1)
    }

def get_master_data():
    path = os.path.join(os.path.dirname(__file__), "../../data/nse_master.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"indices": {}, "stocks": []}
