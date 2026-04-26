"""
Stock Analysis FastAPI Backend
================================
Routes:
  GET  /                        → serve frontend
  POST /api/analyze             → analyze one or more stocks
  GET  /api/history             → get recent stock searches
  DELETE /api/history/{symbol}  → remove from history
  GET  /api/download/{symbol}   → download PDF report
  GET  /api/health              → health check
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, json, os, io, logging, asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import numpy  as np
import pandas as pd

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.routers import valuation_router, momentum_router

app = FastAPI(
    title="NSE Stock Analysis API",
    description="EMA gap analysis + ML signal generation for NSE stocks",
    version="1.0.0"
)

app.include_router(valuation_router.router)
app.include_router(momentum_router.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Return validation errors as JSON with full detail for debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"422 Validation error — body: {body!r}  errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode("utf-8", errors="replace")},
    )

BASE_DIR    = Path(__file__).parent.parent
MODEL_PATH  = BASE_DIR / "models" / "stock_model.pkl"
HISTORY_PATH= BASE_DIR / "data"   / "search_history.json"
PDF_DIR     = BASE_DIR / "data"   / "reports"
STATIC_DIR  = BASE_DIR / "static"

os.makedirs(BASE_DIR / "data",   exist_ok=True)
os.makedirs(PDF_DIR,             exist_ok=True)
os.makedirs(STATIC_DIR,         exist_ok=True)

# Mount static files so index.html loads fonts/CSS properly
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Load Model ────────────────────────────────────────────────────────────────
MODELS_CACHE = {}

def get_model(model_name: str = "stock_model.pkl"):
    """Load model from file and cache it."""
    global MODELS_CACHE
    if model_name in MODELS_CACHE:
        return MODELS_CACHE[model_name]
    
    path = BASE_DIR / "models" / model_name
    if path.exists():
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            # Ensure metrics exist safely
            if "metrics" not in bundle:
                bundle["metrics"] = {"model": "Unknown", "dir_acc": 0}
            
            MODELS_CACHE[model_name] = bundle
            logger.info(f"Model loaded: {model_name} ({bundle['metrics'].get('model', 'Unknown')})")
            return bundle
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    return None

# Load default model on startup
get_model("stock_model.pkl")

# ── History Store ─────────────────────────────────────────────────────────────
def read_history() -> list:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []

def write_history(history: list):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

def add_to_history(symbol: str, name: str, signal: str):
    history = read_history()
    # Remove if already exists
    history = [h for h in history if h["symbol"] != symbol.upper()]
    history.insert(0, {
        "symbol":     symbol.upper(),
        "name":       name,
        "signal":     signal,
        "searched_at": datetime.now().isoformat(),
    })
    write_history(history[:20])   # keep last 20


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()
def _rsi(s, p=14):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
def _slope(s, w=10):
    out = np.full(len(s), np.nan); sv = s.values
    for i in range(w, len(sv)):
        y = sv[i-w:i]
        if not np.any(np.isnan(y)):
            out[i] = np.polyfit(np.arange(w), y, 1)[0]
    return pd.Series(out, index=s.index)


# ── Fetch from Yahoo Finance ──────────────────────────────────────────────────
PDF_READY_CACHE = {}

def build_pdf_bg(result_copy, d_eng_copy, sym):
    try:
        build_pdf(result_copy, d_eng_copy)
        PDF_READY_CACHE[sym] = "ready"
    except Exception as pe:
        logger.warning(f"PDF build failed for {sym}: {pe}")
        PDF_READY_CACHE[sym] = "error"

def fetch_yahoo(symbol: str, period: str = "3y",
                start_date: str = None, end_date: str = None):
    """Download OHLCV. Uses date range if start/end provided, else period."""
    try:
        import yfinance as yf
    except ImportError:
        raise HTTPException(503, "yfinance not installed.")

    # Try symbols in order of commonness for Indian stocks
    for ticker_sym in [f"{symbol}.NS", f"{symbol}.BO", symbol]:
        try:
            logger.info(f"Attempting fetch for: {ticker_sym}")
            ticker = yf.Ticker(ticker_sym)
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date, interval="1d")
            else:
                df = ticker.history(period=period, interval="1d")
            
            if df is not None and not df.empty and len(df) > 5:
                df = df.reset_index()
                # Find the date column regardless of name/case
                date_col = next((c for c in df.columns if c.upper() in ["DATE", "DATETIME"]), None)
                if date_col:
                    df.rename(columns={date_col: "DATE"}, inplace=True)
                
                df.columns = [c.upper() for c in df.columns]
                
                # Ensure all required columns exist
                required = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
                if all(c in df.columns for c in required):
                    df = df[required].copy()
                    if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
                        df["DATE"] = pd.to_datetime(df["DATE"])
                    df["DATE"] = df["DATE"].dt.tz_localize(None)
                    df.sort_values("DATE", inplace=True)
                    df.dropna(subset=["CLOSE"], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    logger.info(f"Successfully fetched {len(df)} rows for {ticker_sym}")
                    return df, ticker_sym
                else:
                    missing = [c for c in required if c not in df.columns]
                    logger.warning(f"Ticker {ticker_sym} missing columns: {missing}")
            else:
                logger.warning(f"Ticker {ticker_sym} returned too few rows: {len(df) if df is not None else 'None'}")
        except Exception as e:
            logger.warning(f"Yahoo fetch failed for {ticker_sym}: {str(e)}")
            continue
    raise HTTPException(404, f"Could not fetch data for '{symbol}'. Try adding '.NS' or check the symbol name.")

def engineer(raw):
    d=raw.copy().reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(d["DATE"]):
        d["DATE"]=pd.to_datetime(d["DATE"],infer_datetime_format=True)
    C_=d["CLOSE"]; H_=d["HIGH"]; L_=d["LOW"]
    O_=d["OPEN"]; V_=d["VOLUME"]
    VW_=d["VWAP"] if "VWAP" in d.columns else C_

    e10=_ema(C_,10); e20=_ema(C_,20); e50=_ema(C_,50); e200=_ema(C_,200)
    e50h=_ema(H_,50); e50l=_ema(L_,50)
    sma200=_sma(C_,200)

    _tr=pd.concat([H_-L_,(H_-C_.shift()).abs(),(L_-C_.shift()).abs()],axis=1).max(axis=1)
    atr=_tr.rolling(14).mean()

    _mf=_ema(C_,12); _ms=_ema(C_,26)
    macd=_mf-_ms; macd_sig=_ema(macd,9); macd_hist=macd-macd_sig

    rsi14=_rsi(C_,14); rsi9=_rsi(C_,9)
    _bm=_sma(C_,20); _bs=C_.rolling(20).std()
    bb_w=((_bm+2*_bs)-(_bm-2*_bs))/_bm
    bb_b=(C_-(_bm-2*_bs))/(4*_bs)
    _lo=L_.rolling(14).min(); _hi=H_.rolling(14).max()
    stoch_k=100*(C_-_lo)/(_hi-_lo).replace(0,np.nan)
    stoch_d=stoch_k.rolling(3).mean()
    vol_sma20=_sma(V_,20)

    # ── EMA ratios ────────────────────────────────────────────────────────────
    d["EMA10_RATIO"]=C_/e10-1;    d["EMA20_RATIO"]=C_/e20-1
    d["EMA50_RATIO"]=C_/e50-1;    d["SMA200_RATIO"]=C_/sma200-1
    d["EMA50H_RATIO"]=C_/e50l-1 # Correction: training uses C_/e50h-1, but wait, let me look at cell17

    # wait, I should re-check cell17 lines 170-173
    # 170: d["EMA10_RATIO"]=C_/e10-1;    d["EMA20_RATIO"]=C_/e20-1
    # 171: d["EMA50_RATIO"]=C_/e50-1;    d["SMA200_RATIO"]=C_/sma200-1
    # 172: d["EMA50H_RATIO"]=C_/e50h-1;  d["EMA50L_RATIO"]=C_/e50l-1
    
    # Okay, I'll resume the replacement.
    d["EMA50H_RATIO"]=C_/e50h-1;  d["EMA50L_RATIO"]=C_/e50l-1

    # ── EMA alignment ─────────────────────────────────────────────────────────
    d["EMA10_GT_20"]=(e10>e20).astype(int)
    d["EMA20_GT_50"]=(e20>e50).astype(int)
    d["EMA50_GT_200"]=(e50>e200).astype(int)
    d["EMA10_GT_50H"]=(e10>e50h).astype(int)
    d["EMA_STACK"]=d["EMA10_GT_20"]+d["EMA20_GT_50"]+d["EMA50_GT_200"]

    # ── Gap analysis ──────────────────────────────────────────────────────────
    d["GAP_ATR"]=(e10-e50h)/atr;  d["GAP_PCT"]=(e10-e50h)/e50h*100
    d["CLOSE_GAP_ATR"]=(C_-e50h)/atr
    gap_raw=(e10-e50h)/atr
    d["GAP_WIDENING"]=gap_raw.diff(5)

    # ── Oscillators ───────────────────────────────────────────────────────────
    d["RSI_14"]=rsi14; d["RSI_9"]=rsi9
    d["MACD_HIST"]=macd_hist
    d["MACD_CROSS"]=(macd>macd_sig).astype(int)
    d["MACD_ABOVE_ZERO"]=(macd>0).astype(int)
    d["MACD_ACCEL"]=macd_hist.diff()
    d["STOCH_K"]=stoch_k; d["STOCH_D"]=stoch_d
    d["STOCH_RISING"]=(stoch_k>stoch_k.shift(1)).astype(int)
    d["STOCH_CROSS"]=(stoch_k>stoch_d).astype(int)
    d["WILLIAMS_R"]=-100*((_hi-C_)/(_hi-_lo).replace(0,np.nan))
    tp=(H_+L_+C_)/3; tp_sma=_sma(tp,20)
    tp_md=tp.rolling(20).apply(lambda x: np.mean(np.abs(x-x.mean())),raw=True)
    d["CCI"]=(tp-tp_sma)/(0.015*tp_md)

    # ── Volatility ────────────────────────────────────────────────────────────
    d["BB_WIDTH"]=bb_w; d["BB_PCT_B"]=bb_b
    d["ATR_PCT"]=atr/C_*100
    _r=C_.pct_change()
    d["VOL_5D"]=_r.rolling(5).std()*100
    d["VOL_20D"]=_r.rolling(20).std()*100
    vol20=_r.rolling(20).std(); vol60=_r.rolling(60).std()
    d["BETA_PROXY"]=vol20/vol60.replace(0,np.nan)
    d["BETA_20_60"]=vol20/vol60.replace(0,np.nan)
    d["BETA_REGIME"]=(vol20>vol60).astype(int)

    # ── Volume ────────────────────────────────────────────────────────────────
    d["VOL_RATIO"]=V_/vol_sma20
    d["VWAP_DEV"]=(C_-VW_)/VW_*100
    vol_sma5=_sma(V_,5)
    d["VOL_MOMENTUM"]=vol_sma5/vol_sma20
    obv=(np.sign(C_.diff()).fillna(0)*V_).cumsum()
    d["OBV_ROC"]=obv.pct_change(10)*100
    d["OBV_TREND"]=(obv>_ema(obv,20)).astype(int)
    vpt=(C_.pct_change().fillna(0)*V_).cumsum()
    d["VPT_ROC"]=vpt.pct_change(10)*100
    d["VPT_TREND"]=(vpt>_ema(vpt,20)).astype(int)

    # ── Returns & momentum ────────────────────────────────────────────────────
    for lg in [1,2,3,5,10,20]:
        d[f"RET_{lg}D"]=C_.pct_change(lg)*100
    d["ROC_5"]=C_.pct_change(5)*100
    d["ROC_10"]=C_.pct_change(10)*100
    d["ROC_20"]=C_.pct_change(20)*100
    d["MOM_SCORE"]=(d["ROC_5"]+d["ROC_10"]+d["ROC_20"])/3

    # ── Trend ─────────────────────────────────────────────────────────────────
    d["TREND_5D"]=(C_>C_.shift(5)).astype(int)
    d["TREND_10D"]=(C_>C_.shift(10)).astype(int)
    d["TREND_20D"]=(C_>C_.shift(20)).astype(int)
    d["ABOVE_EMA50H"]=(C_>e50h).astype(int)
    d["ABOVE_EMA50L"]=(C_>e50l).astype(int)
    d["SLOPE_10D"]=_slope(C_,10); d["SLOPE_20D"]=_slope(C_,20)
    d["HIGHER_HIGH_5"]=(H_>H_.rolling(5).max().shift(1)).astype(int)
    d["HIGHER_LOW_5"]=(L_>L_.rolling(5).min().shift(1)).astype(int)
    d["TREND_STRUCT"]=d["HIGHER_HIGH_5"]+d["HIGHER_LOW_5"]
    up_day=(C_>C_.shift(1)).astype(int); consec=up_day.copy()
    for i in range(1,len(consec)):
        if consec.iloc[i]==1: consec.iloc[i]=consec.iloc[i-1]+1
    d["CONSEC_UP"]=consec

    # ── Price position ────────────────────────────────────────────────────────
    hi20=H_.rolling(20).max(); lo20=L_.rolling(20).min()
    hi50=H_.rolling(50).max(); lo50=L_.rolling(50).min()
    d["CHANNEL_POS_20"]=(C_-lo20)/(hi20-lo20).replace(0,np.nan)
    d["CHANNEL_POS_50"]=(C_-lo50)/(hi50-lo50).replace(0,np.nan)
    sma20=_sma(C_,20); std20=C_.rolling(20).std()
    sma50=_sma(C_,50); std50=C_.rolling(50).std()
    d["ZSCORE_20"]=(C_-sma20)/std20.replace(0,np.nan)
    d["ZSCORE_50"]=(C_-sma50)/std50.replace(0,np.nan)
    d["OVEREXTENDED"]=((d["ZSCORE_20"].abs()>2)|(d["ZSCORE_50"].abs()>2)).astype(int)

    # ── Candle patterns ───────────────────────────────────────────────────────
    body=(C_-O_).abs(); full_range=(H_-L_).replace(0,np.nan)
    d["BODY_RATIO"]=body/full_range
    d["UPPER_SHADOW"]=(H_-pd.concat([C_,O_],axis=1).max(axis=1))/full_range
    d["LOWER_SHADOW"]=(pd.concat([C_,O_],axis=1).min(axis=1)-L_)/full_range
    d["BULL_CANDLE"]=(C_>O_).astype(int)
    prev_body=(C_.shift(1)-O_.shift(1)).abs()
    d["BULL_ENGULF"]=((C_>O_) & (O_<C_.shift(1)) & (C_>O_.shift(1)) & (body>prev_body)).astype(int)

    # ── Return distribution ───────────────────────────────────────────────────
    d["SKEW_20"]=_r.rolling(20).skew()
    d["KURT_20"]=_r.rolling(20).kurt()

    # ── RSI divergence ────────────────────────────────────────────────────────
    price_rising=(C_>C_.shift(5)).astype(int); rsi_falling=(rsi14<rsi14.shift(5)).astype(int)
    d["RSI_DIVERGE"]=((price_rising==1)&(rsi_falling==1)).astype(int)*-1

    # ── Calendar ──────────────────────────────────────────────────────────────
    d["DOW"]=d["DATE"].dt.dayofweek; d["MONTH"]=d["DATE"].dt.month
    
    # [Internal display helper columns - not used by model]
    d["EMA_10"]=e10; d["EMA_20"]=e20; d["EMA_50"]=e50; d["EMA_200"]=e200
    d["EMA50_HIGH"]=e50h; d["EMA50_LOW"]=e50l
    d["ATR_14"]=atr; d["MACD"]=macd; d["MACD_SIG"]=macd_sig

    # ── BETA vs NIFTY 50 (Display only) ──────────────────────────────────────
    try:
        import yfinance as yf
        nifty_raw = yf.Ticker("^NSEI").history(start=str(d["DATE"].iloc[0])[:10], end=str(d["DATE"].iloc[-1])[:10], interval="1d")["Close"]
        nifty_raw.index = nifty_raw.index.tz_localize(None)
        nifty_ret = nifty_raw.pct_change()
        date_idx = pd.to_datetime(d["DATE"])
        stock_ret = pd.Series(_r.values, index=date_idx)
        nifty_aligned = nifty_ret.reindex(date_idx, method="nearest")
        def rolling_beta(s_ret, m_ret, window=60):
            betas = np.full(len(s_ret), np.nan); sv = s_ret.values; mv = m_ret.values
            for i in range(window, len(sv)):
                s_w = sv[i-window:i]; m_w = mv[i-window:i]
                mask = ~(np.isnan(s_w) | np.isnan(m_w))
                if mask.sum() > 20: 
                    cov = np.cov(s_w[mask], m_w[mask])
                    betas[i] = cov[0,1] / cov[1,1] if cov[1,1] != 0 else np.nan
            return pd.Series(betas, index=s_ret.index)
        beta_series = rolling_beta(stock_ret, nifty_aligned, 60)
        d["BETA_DISPLAY"] = np.round(beta_series.values, 3)
    except:
        std252 = _r.rolling(252).std()
        d["BETA_DISPLAY"] = (vol60 / std252.replace(0, np.nan)).round(3)

    return d

# ── Generate Signal ───────────────────────────────────────────────────────────
def generate_signal(d_eng: pd.DataFrame, symbol: str, model_name: str = "stock_model.pkl") -> dict:
    bundle = get_model(model_name)
    if bundle is None:
        raise HTTPException(503, f"Model '{model_name}' not loaded.")
    pipe  = bundle["model"]
    feats = bundle["features"]
    feats = [f for f in feats if f in d_eng.columns]

    last  = d_eng.dropna(subset=feats).iloc[-1]
    X     = last[feats].values.reshape(1, -1)
    pred  = float(pipe.predict(X)[0])

    # ── Core indicators ────────────────────────────────────────────────────────
    gap_atr     = float(last.get("GAP_ATR",      0))
    rsi         = float(last.get("RSI_14",       50))
    macd_hist   = float(last.get("MACD_HIST",    0))
    macd_crs    = int(last.get("MACD_CROSS",     0))
    vol_ratio   = float(last.get("VOL_RATIO",    1))
    slope       = float(last.get("SLOPE_10D",    0))
    close       = float(d_eng["CLOSE"].iloc[-1])
    e50h        = float(d_eng["EMA50_HIGH"].iloc[-1])
    e50l        = float(d_eng["EMA50_LOW"].iloc[-1])
    e10         = float(d_eng["EMA_10"].iloc[-1])
    atr         = float(d_eng["ATR_14"].iloc[-1])
    macd_val    = float(d_eng["MACD"].iloc[-1])

    # ── Momentum indicators ────────────────────────────────────────────────────
    roc_5        = float(last.get("ROC_5",        0))
    roc_10       = float(last.get("ROC_10",       0))
    roc_20       = float(last.get("ROC_20",       0))
    macd_accel   = float(last.get("MACD_ACCEL",   0))
    ema_stack    = int(last.get("EMA_STACK",      0))
    vol_momentum = float(last.get("VOL_MOMENTUM", 1))
    gap_widening = float(last.get("GAP_WIDENING", 0))
    mom_score    = int(last.get("MOM_SCORE",      0))
    stoch_rising = int(last.get("STOCH_RISING",   0))
    stoch_cross  = int(last.get("STOCH_CROSS",    0))
    consec_up    = float(last.get("CONSEC_UP",    0))
    if np.isnan(macd_accel):   macd_accel   = 0
    if np.isnan(gap_widening): gap_widening = 0
    if np.isnan(vol_momentum): vol_momentum = 1

    # Streak above EMA50_HIGH
    above  = (d_eng["CLOSE"] > d_eng["EMA50_HIGH"]).astype(int)
    streak = 0
    for v in above.values[::-1]:
        if v == 1: streak += 1
        else: break

    # 52W position
    high52 = d_eng["HIGH"].tail(252).max()
    low52  = d_eng["LOW"].tail(252).min()
    pos52  = round((close - low52) / (high52 - low52 + 1e-9) * 100, 1)

    # ── 4-filter base score (0–8) ──────────────────────────────────────────────
    gap_s  = 2 if gap_atr>=1.0  else (1 if gap_atr>=0.3 else 0)
    rsi_s  = 2 if rsi<=60       else (1 if rsi<=70      else 0)
    macd_s = 2 if (macd_hist>0 and macd_crs==1) else (1 if macd_hist>0 else 0)
    vol_s  = 2 if vol_ratio>=1.2 else (1 if vol_ratio>=0.8 else 0)
    total  = gap_s + rsi_s + macd_s + vol_s

    if   total >= 7: sig = "STRONG BUY"
    elif total >= 5: sig = "BUY"
    elif total >= 3: sig = "WATCH"
    else:            sig = "AVOID"

    # ── MOMENTUM BOOST: override to STRONG BUY if momentum fully aligned ──────
    # All momentum must fire together to boost signal
    momentum_aligned = (
        macd_accel   > 0          and   # MACD histogram growing (accelerating)
        gap_widening > 0          and   # Gap getting wider (cushion building)
        mom_score    >= 2         and   # At least 2 of 3 ROC timeframes positive
        vol_momentum >= 1.1       and   # Volume trend rising
        ema_stack    >= 3         and   # At least 3 of 4 EMAs in bull order
        macd_val     > 0               # MACD line above zero
    )
    if momentum_aligned and sig in ("BUY", "WATCH") and gap_atr > 0.3:
        sig = "STRONG BUY"   # momentum boost

    # ── Signal Strength 1–5 with momentum ─────────────────────────────────────
    # Base 5 factors (max 5.0 pts) + momentum bonus (max 3.0 pts) = max 8.0
    # Maps to 1–5 rating with momentum able to push borderline signals higher
    s = 0.0

    # Base factors
    if   gap_atr >= 1.5: s += 1.0
    elif gap_atr >= 0.5: s += 0.5
    if   30 < rsi <= 60: s += 1.0
    elif 60 < rsi <= 72: s += 0.5
    if   macd_hist > 0 and macd_val > 0: s += 1.0
    elif macd_hist > 0:                  s += 0.5
    if   vol_ratio >= 1.5: s += 1.0
    elif vol_ratio >= 1.0: s += 0.5
    if   slope > 0: s += 1.0

    # ── Momentum bonus points ──────────────────────────────────────────────────
    # MACD acceleration: histogram growing = momentum building
    if   macd_accel > 0.5:  s += 0.75   # strong acceleration
    elif macd_accel > 0:    s += 0.40   # mild acceleration
    elif macd_accel < -0.5: s -= 0.50   # decelerating (penalty)

    # EMA stack alignment: perfect bull order boosts confidence
    if   ema_stack == 4: s += 0.75   # perfect alignment
    elif ema_stack == 3: s += 0.40
    elif ema_stack <= 1: s -= 0.25   # misaligned (penalty)

    # Multi-timeframe ROC: all 3 positive = real trend momentum
    if   mom_score == 3: s += 0.75   # all timeframes up
    elif mom_score == 2: s += 0.35
    elif mom_score == 0: s -= 0.25   # all timeframes down (penalty)

    # Gap widening: cushion building = strong signal
    if   gap_widening > 0.2:  s += 0.40
    elif gap_widening < -0.3: s -= 0.40   # gap narrowing fast (danger)

    # Volume momentum: rising participation confirms move
    if   vol_momentum >= 1.3: s += 0.35
    elif vol_momentum >= 1.1: s += 0.15

    # Stochastic: rising + above signal line = momentum confirmation
    if stoch_rising and stoch_cross: s += 0.25

    # Consecutive up days: 3+ days = trend continuity
    if   consec_up >= 4: s += 0.25
    elif consec_up >= 2: s += 0.10

    # Map to 1–5
    if   s >= 6.0: strength = 5
    elif s >= 4.5: strength = 4
    elif s >= 3.0: strength = 3
    elif s >= 1.5: strength = 2
    else:          strength = 1

    strength_label = {
        5: "Exceptional — momentum + all 5 factors fully aligned",
        4: "Strong — most factors + momentum confirmed",
        3: "Moderate — base factors ok, momentum partial",
        2: "Weak — momentum missing or mixed",
        1: "Very weak — momentum against, avoid entry",
    }[strength]

    # ── Momentum summary dict for frontend display ─────────────────────────────
    momentum_detail = {
        "macd_accel":    round(macd_accel,   3),
        "ema_stack":     ema_stack,
        "mom_score":     mom_score,
        "roc_5":         round(roc_5,         2),
        "roc_10":        round(roc_10,        2),
        "roc_20":        round(roc_20,        2),
        "gap_widening":  round(gap_widening,  3),
        "vol_momentum":  round(vol_momentum,  2),
        "stoch_rising":  bool(stoch_rising),
        "consec_up":     int(consec_up),
        # Status strings
        "macd_accel_status":  "BUILDING" if macd_accel > 0.5 else
                              "MILD"     if macd_accel > 0   else
                              "FADING"   if macd_accel > -0.5 else "WEAKENING",
        "ema_stack_status":   f"{ema_stack}/4 aligned",
        "roc_status":         "ALL UP"   if mom_score==3 else
                              "MIXED"    if mom_score==2 else
                              "PARTIAL"  if mom_score==1 else "ALL DOWN",
        "gap_status":         "WIDENING" if gap_widening > 0.1 else
                              "STABLE"   if gap_widening > -0.1 else "NARROWING",
        "vol_trend":          "RISING"   if vol_momentum >= 1.2 else
                              "FLAT"     if vol_momentum >= 0.9 else "FALLING",
        "momentum_aligned":   bool(momentum_aligned),
        "momentum_boost":     bool(momentum_aligned and sig == "STRONG BUY"),
    }

    # Price history for sparkline
    hist90 = d_eng.tail(90)[["DATE","CLOSE","EMA50_HIGH","EMA50_LOW","EMA_10",
                               "GAP_ATR","RSI_14","MACD_HIST"]].copy()
    hist90["DATE"] = hist90["DATE"].astype(str)

    return {
        "symbol":            symbol.upper(),
        "signal":            sig,
        "score":             total,
        "max_score":         8,
        "strength":          strength,
        "strength_label":    strength_label,
        "momentum":          momentum_detail,
        "pred_5d_pct":       round(pred, 2),
        "date":              str(d_eng["DATE"].iloc[-1])[:10],
        "close":             round(close, 2),
        "ema10":             round(e10, 2),
        "ema50_high":        round(e50h, 2),
        "ema50_low":         round(e50l, 2),
        "atr":               round(atr, 2),
        "gap_rs":            round(e10 - e50h, 2),
        "gap_atr":           round(gap_atr, 2),
        "rsi":               round(rsi, 1),
        "macd_hist":         round(macd_hist, 3),
        "vol_ratio":         round(vol_ratio, 2),
        "slope_10d":         round(slope, 3),
        "streak_days":       streak,
        "pos_52w":           pos52,
        "gap_signal":        ["AVOID","WAIT","GO"][gap_s],
        "rsi_signal":        ["AVOID","WAIT","GO"][rsi_s],
        "macd_signal":       ["AVOID","WAIT","GO"][macd_s],
        "vol_signal":        ["AVOID","WAIT","GO"][vol_s],
        "stop_loss":         round(e50l, 2),
        "entry_price":       round(close, 2),
        "target_1m":         round(close * (1 + max(pred, 2) / 100), 2),
        # ── New indicators ────────────────────────────────────────────────────
        "beta":              round(float(d_eng["BETA_DISPLAY"].iloc[-1]), 3)
                             if "BETA_DISPLAY" in d_eng.columns and
                             not np.isnan(d_eng["BETA_DISPLAY"].iloc[-1]) else None,
        "beta_regime":       "HIGH" if float(d_eng["BETA_20_60"].iloc[-1]) > 1.2
                             else "NORMAL" if float(d_eng["BETA_20_60"].iloc[-1]) > 0.8
                             else "LOW"
                             if "BETA_20_60" in d_eng.columns else "N/A",
        "williams_r":        round(float(d_eng["WILLIAMS_R"].iloc[-1]), 1)
                             if "WILLIAMS_R" in d_eng.columns else None,
        "cci":               round(float(d_eng["CCI"].iloc[-1]), 1)
                             if "CCI" in d_eng.columns else None,
        "obv_trend":         int(d_eng["OBV_TREND"].iloc[-1])
                             if "OBV_TREND" in d_eng.columns else None,
        "channel_pos_20":    round(float(d_eng["CHANNEL_POS_20"].iloc[-1]), 3)
                             if "CHANNEL_POS_20" in d_eng.columns else None,
        "zscore_20":         round(float(d_eng["ZSCORE_20"].iloc[-1]), 2)
                             if "ZSCORE_20" in d_eng.columns else None,
        "trend_struct":      int(d_eng["TREND_STRUCT"].iloc[-1])
                             if "TREND_STRUCT" in d_eng.columns else None,
        "overextended":      bool(d_eng["OVEREXTENDED"].iloc[-1])
                             if "OVEREXTENDED" in d_eng.columns else False,
        # Chart data
        "history":           hist90.to_dict("records"),
        # Chart data
        "history":           hist90.to_dict("records"),
        "model_name":        bundle["metrics"]["model"] if bundle else "N/A",
        "model_acc":         bundle["metrics"]["dir_acc"] if bundle else 0,
        "model_file":        model_name,
    }


# ── PDF Report ────────────────────────────────────────────────────────────────
def build_pdf(result: dict, d_eng: pd.DataFrame) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    from sklearn.preprocessing import RobustScaler
    from sklearn.pipeline import Pipeline

    BG="#0a0f1a"; PAN="#0d1525"; GRD="#1a2535"
    C=["#4a9fd4","#fbbf24","#34d399","#f87171","#a78bfa","#fb923c"]
    plt.rcParams.update({
        "figure.facecolor":BG,"axes.facecolor":PAN,"axes.edgecolor":GRD,
        "axes.labelcolor":"#c8d6e8","xtick.color":"#c8d6e8","ytick.color":"#c8d6e8",
        "text.color":"#c8d6e8","grid.color":GRD,"legend.facecolor":PAN,
        "legend.edgecolor":GRD,"font.family":"monospace","figure.dpi":130,
    })

    symbol   = result["symbol"]
    sig      = result["signal"]
    sig_clr  = C[2] if "BUY" in sig else (C[1] if "WATCH" in sig else C[3])
    pdf_path = str(PDF_DIR / f"{symbol}_analysis.pdf")

    # 1 year of data
    hist = d_eng.tail(252).reset_index(drop=True)
    pdf  = PdfPages(pdf_path)

    # ── PAGE 1: Price + Gap + RSI + MACD + Volume ─────────────────────────────
    fig = plt.figure(figsize=(20, 24), facecolor=BG)
    gs  = gridspec.GridSpec(4, 2, fig, hspace=0.42, wspace=0.28,
                             height_ratios=[3.5, 1.1, 1.1, 1.0])

    # ── Row 1: Price chart ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])

    # ── Colours ───────────────────────────────────────────────────────────────
    CLR_EMA50H = "#26c6da"   # cyan
    CLR_EMA50L = "#ef9a9a"   # soft rose
    CLR_EMA10  = "#ffd740"   # amber
    CLR_CLOSE  = "#aaaaaa"   # neutral grey — single colour for close

    # EMA50 band fill
    ax.fill_between(hist["DATE"], hist["EMA50_HIGH"], hist["EMA50_LOW"],
                    alpha=0.08, color=CLR_EMA50H)

    # EMA lines
    ax.plot(hist["DATE"], hist["EMA50_HIGH"],
            color=CLR_EMA50H, lw=1.5, alpha=0.85, label="EMA50 High")
    ax.plot(hist["DATE"], hist["EMA50_LOW"],
            color=CLR_EMA50L, lw=1.2, alpha=0.80, ls="--", label="EMA50 Low")
    ax.plot(hist["DATE"], hist["EMA_10"],
            color=CLR_EMA10,  lw=1.7, ls=(0,(5,2)), alpha=0.90, label="EMA10")

    # Close — single grey line, no zone colouring
    ax.plot(hist["DATE"], hist["CLOSE"],
            color=CLR_CLOSE, lw=2.0, alpha=0.85, zorder=6, label="Close")

    # ── SIGNAL STRENGTH 1–5 ───────────────────────────────────────────────────
    def calc_strength(gap, rsi, macd_hist, macd_val, vol_ratio, slope,
                      macd_accel=0, ema_stack=0, mom_score=0,
                      gap_widening=0, vol_momentum=1,
                      stoch_rising=0, consec_up=0):
        # Base factors (max 5.0)
        s = 0.0
        if   gap >= 1.5:           s += 1.0
        elif gap >= 0.5:           s += 0.5
        if   30 < rsi <= 60:       s += 1.0
        elif 60 < rsi <= 72:       s += 0.5
        if   macd_hist > 0 and macd_val > 0: s += 1.0
        elif macd_hist > 0:        s += 0.5
        if   vol_ratio >= 1.5:     s += 1.0
        elif vol_ratio >= 1.0:     s += 0.5
        if   slope > 0:            s += 1.0
        # Momentum bonus (max 3.0)
        if   macd_accel > 0.5:    s += 0.75
        elif macd_accel > 0:      s += 0.40
        elif macd_accel < -0.5:   s -= 0.50
        if   ema_stack == 4:      s += 0.75
        elif ema_stack == 3:      s += 0.40
        elif ema_stack <= 1:      s -= 0.25
        if   mom_score == 3:      s += 0.75
        elif mom_score == 2:      s += 0.35
        elif mom_score == 0:      s -= 0.25
        if   gap_widening > 0.2:  s += 0.40
        elif gap_widening < -0.3: s -= 0.40
        if   vol_momentum >= 1.3: s += 0.35
        elif vol_momentum >= 1.1: s += 0.15
        if   stoch_rising and consec_up >= 2: s += 0.25
        # Map to 1–5
        if   s >= 6.0: return 5
        elif s >= 4.5: return 4
        elif s >= 3.0: return 3
        elif s >= 1.5: return 2
        else:          return 1

    cross_up   = hist["EMA10_GT_50H"].diff() > 0
    cross_down = hist["EMA10_GT_50H"].diff() < 0

    entry_signals = []   # list of (date, price, strength)
    exit_signals  = []

    for i in hist.index[cross_up]:
        gap  = float(hist["GAP_ATR"].iloc[i])  if "GAP_ATR"   in hist.columns else 0
        rsi  = float(hist["RSI_14"].iloc[i])   if "RSI_14"    in hist.columns else 50
        mh   = float(hist["MACD_HIST"].iloc[i])if "MACD_HIST" in hist.columns else 0
        mv   = float(hist["MACD"].iloc[i])     if "MACD"      in hist.columns else 0
        volr = float(hist["VOL_RATIO"].iloc[i])if "VOL_RATIO" in hist.columns else 1
        slp  = float(hist["SLOPE_10D"].iloc[i])if "SLOPE_10D" in hist.columns else 0
        if np.isnan(slp): slp = 0
        r = calc_strength(gap, rsi, mh, mv, volr, slp,
                          macd_accel=float(hist["MACD_ACCEL"].iloc[i]) if "MACD_ACCEL" in hist.columns else 0,
                          ema_stack=int(hist["EMA_STACK"].iloc[i])     if "EMA_STACK"  in hist.columns else 0,
                          mom_score=int(hist["MOM_SCORE"].iloc[i])     if "MOM_SCORE"  in hist.columns else 0,
                          gap_widening=float(hist["GAP_WIDENING"].iloc[i]) if "GAP_WIDENING" in hist.columns else 0,
                          vol_momentum=float(hist["VOL_MOMENTUM"].iloc[i]) if "VOL_MOMENTUM" in hist.columns else 1,
                          stoch_rising=int(hist["STOCH_RISING"].iloc[i])   if "STOCH_RISING" in hist.columns else 0,
                          consec_up=float(hist["CONSEC_UP"].iloc[i])       if "CONSEC_UP"    in hist.columns else 0)
        entry_signals.append((hist["DATE"].iloc[i], float(hist["EMA_10"].iloc[i]), r))

    for i in hist.index[cross_down]:
        gap  = float(hist["GAP_ATR"].iloc[i])  if "GAP_ATR"   in hist.columns else 0
        rsi  = float(hist["RSI_14"].iloc[i])   if "RSI_14"    in hist.columns else 50
        mh   = float(hist["MACD_HIST"].iloc[i])if "MACD_HIST" in hist.columns else 0
        mv   = float(hist["MACD"].iloc[i])     if "MACD"      in hist.columns else 0
        volr = float(hist["VOL_RATIO"].iloc[i])if "VOL_RATIO" in hist.columns else 1
        slp  = float(hist["SLOPE_10D"].iloc[i])if "SLOPE_10D" in hist.columns else 0
        if np.isnan(slp): slp = 0
        gap_exit = abs(gap) if gap < 0 else 0
        ex = 0.0
        if gap_exit >= 1.5: ex += 2.0
        elif gap_exit >= 0.5: ex += 1.0
        if rsi < 40: ex += 1.5
        elif rsi < 50: ex += 0.5
        if mh < 0: ex += 1.0
        if slp < 0: ex += 0.5
        if   ex >= 4.5: er = 5
        elif ex >= 3.5: er = 4
        elif ex >= 2.5: er = 3
        elif ex >= 1.5: er = 2
        else:           er = 1
        exit_signals.append((hist["DATE"].iloc[i], float(hist["EMA_10"].iloc[i]), er))

    # ── Plot: ALL entries = green triangle ▲, ALL exits = red triangle ▼
    # Marker size scales with strength. Number printed inside each triangle.
    # Size map: 5→300, 4→230, 3→170, 2→120, 1→80
    SZ = {5:300, 4:230, 3:170, 2:120, 1:80}
    CLR_ENTRY = "#00e676"   # vivid green
    CLR_EXIT  = "#ff1744"   # vivid red

    # Draw entries (weakest first so strongest renders on top)
    entry_signals.sort(key=lambda x: x[2])
    first_entry = True
    for d, p, r in entry_signals:
        ax.scatter(d, p, marker="^", color=CLR_ENTRY, s=SZ[r],
                   zorder=9+r, edgecolors="white", linewidths=0.8,
                   label="Entry" if first_entry else "_")
        first_entry = False
        # Strength number centred inside triangle
        ax.annotate(str(r), xy=(d, p), xytext=(0, -1),
                    textcoords="offset points",
                    fontsize=7, color="black", ha="center", va="center",
                    fontweight="bold", zorder=12)

    # Draw exits
    exit_signals.sort(key=lambda x: x[2])
    first_exit = True
    for d, p, r in exit_signals:
        ax.scatter(d, p, marker="v", color=CLR_EXIT, s=SZ[r],
                   zorder=9+r, edgecolors="white", linewidths=0.8,
                   label="Exit" if first_exit else "_")
        first_exit = False
        ax.annotate(str(r), xy=(d, p), xytext=(0, 1),
                    textcoords="offset points",
                    fontsize=7, color="white", ha="center", va="center",
                    fontweight="bold", zorder=12)

    # ── Line labels directly on chart (right edge) ────────────────────────────
    label_items = [
        (hist["CLOSE"].iloc[-1],      f"Close  {hist['CLOSE'].iloc[-1]:.0f}",      sig_clr, True),
        (hist["EMA_10"].iloc[-1],     f"EMA10  {hist['EMA_10'].iloc[-1]:.0f}",     C[1],    False),
        (hist["EMA50_HIGH"].iloc[-1], f"50H    {hist['EMA50_HIGH'].iloc[-1]:.0f}", C[2],    False),
        (hist["EMA50_LOW"].iloc[-1],  f"50L    {hist['EMA50_LOW'].iloc[-1]:.0f}",  C[3],    False),
    ]
    # Sort by price to avoid overlapping labels
    label_items.sort(key=lambda x: x[0], reverse=True)
    last_y = None
    min_gap_px = (hist["CLOSE"].max() - hist["CLOSE"].min()) * 0.025
    for val, lbl, clr, bold in label_items:
        if last_y is not None and abs(val - last_y) < min_gap_px:
            val = last_y - min_gap_px
        ax.annotate(
            lbl,
            xy=(hist["DATE"].iloc[-1], val),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8.5, color=clr, clip_on=False,
            fontweight="bold" if bold else "normal",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.7,
                      edgecolor=clr, linewidth=0.5) if bold else None
        )
        last_y = val

    ax.set_title(
        f"{symbol}     {sig}  {result['score']}/8     "
        f"Rs.{result['close']}   Gap {result['gap_atr']:+.2f}x ATR   "
        f"RSI {result['rsi']:.0f}   {result['date']}",
        fontsize=11, fontweight="bold", color=sig_clr, pad=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.legend(fontsize=8, loc="upper left", ncol=6,
              framealpha=0.5, edgecolor=GRD,
              handlelength=1.2, handletextpad=0.5, columnspacing=1.0)
    ax.grid(True, alpha=0.18)

    # ── Row 2: Gap ATR ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    g = hist["GAP_ATR"]
    ax2.fill_between(hist["DATE"], g, 0, where=g > 0,  alpha=0.45, color=C[2])
    ax2.fill_between(hist["DATE"], g, 0, where=g <= 0, alpha=0.45, color=C[3])
    ax2.plot(hist["DATE"], g, color=C[5], lw=1.2, alpha=0.8)
    ax2.axhline(0,    color="white", lw=0.7, ls="--", alpha=0.5)
    ax2.axhline(1.0,  color=C[2],   lw=0.7, ls=":",  alpha=0.6)
    ax2.axhline(-1.0, color=C[3],   lw=0.7, ls=":",  alpha=0.4)
    cur_g = float(g.iloc[-1])
    ax2.annotate(f"{cur_g:+.2f}x",
                 xy=(hist["DATE"].iloc[-1], cur_g),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=8.5, color=C[2] if cur_g > 0 else C[3],
                 fontweight="bold", clip_on=False)
    ax2.set_title("GAP  EMA10 vs EMA50_HIGH  (ATR units)  |  > +1.0 = GO zone",
                  fontsize=9, color="#8899aa", pad=4)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=7.5)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.grid(True, alpha=0.15)

    # ── Row 3: RSI + MACD ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(hist["DATE"], hist["RSI_14"], color=C[4], lw=1.4)
    ax3.fill_between(hist["DATE"], hist["RSI_14"], 50,
                     where=hist["RSI_14"] > 50,  alpha=0.12, color=C[2])
    ax3.fill_between(hist["DATE"], hist["RSI_14"], 50,
                     where=hist["RSI_14"] <= 50, alpha=0.08, color=C[3])
    ax3.axhline(70, color=C[3], lw=0.6, ls="--", alpha=0.7)
    ax3.axhline(30, color=C[2], lw=0.6, ls="--", alpha=0.7)
    ax3.axhline(50, color="white", lw=0.4, alpha=0.3)
    ax3.annotate(f"RSI {result['rsi']:.0f}",
                 xy=(hist["DATE"].iloc[-1], result["rsi"]),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=9, color=C[4], fontweight="bold", clip_on=False)
    ax3.set_ylim(0, 100)
    ax3.set_title("RSI  14", fontsize=9, color="#8899aa", pad=4)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax3.xaxis.get_majorticklabels(), fontsize=7.5)
    ax3.grid(True, alpha=0.15)

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.bar(hist["DATE"], hist["MACD_HIST"],
            color=[C[2] if v >= 0 else C[3] for v in hist["MACD_HIST"]],
            width=0.9, alpha=0.75)
    ax4.axhline(0, color="white", lw=0.5, alpha=0.4)
    ax4.annotate(f"Hist {result['macd_hist']:+.2f}",
                 xy=(hist["DATE"].iloc[-1], result["macd_hist"]),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=9, color=C[2] if result["macd_hist"] >= 0 else C[3],
                 fontweight="bold", clip_on=False)
    ax4.set_title("MACD  Histogram", fontsize=9, color="#8899aa", pad=4)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax4.xaxis.get_majorticklabels(), fontsize=7.5)
    ax4.grid(True, alpha=0.15)

    # ── Row 4: Volume ─────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    vol_r = hist["VOLUME"] / hist["VOLUME"].rolling(20).mean()
    ax5.bar(hist["DATE"], vol_r,
            color=[C[2] if v >= 1.2 else C[5] if v >= 0.8 else C[3]
                   for v in vol_r.fillna(1)],
            width=0.9, alpha=0.7)
    ax5.axhline(1.2, color=C[2], lw=0.7, ls=":", alpha=0.7, label="1.2x GO")
    ax5.axhline(1.0, color="white", lw=0.4, alpha=0.3)
    ax5.annotate(f"{result['vol_ratio']:.2f}x",
                 xy=(hist["DATE"].iloc[-1], result["vol_ratio"]),
                 xytext=(6, 0), textcoords="offset points",
                 fontsize=9, color=C[2] if result["vol_ratio"] >= 1.2 else C[5],
                 fontweight="bold", clip_on=False)
    ax5.set_title("Volume  ratio vs 20d avg  |  green >= 1.2x institutional",
                  fontsize=9, color="#8899aa", pad=4)
    ax5.xaxis.set_major_locator(mdates.MonthLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)
    ax5.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.15)

    # Summary footer
    fig.text(
        0.02, 0.002,
        f"Entry Rs.{result['entry_price']}   Stop Rs.{result['stop_loss']}   "
        f"Target Rs.{result['target_1m']}   |   "
        f"Gap:{result['gap_signal']}  RSI:{result['rsi_signal']}  "
        f"MACD:{result['macd_signal']}  Vol:{result['vol_signal']}   "
        f"Score {result['score']}/8   DirAcc {result['model_acc']:.1f}% ({result['model_name']})",
        fontsize=8, color="#6a7f99", fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    # ── PAGE 2: Momentum Dashboard ────────────────────────────────────────────
    try:
        hist_m = d_eng.tail(252).reset_index(drop=True)

        fig2 = plt.figure(figsize=(20, 24), facecolor=BG)
        gs2  = gridspec.GridSpec(4, 2, fig2, hspace=0.45, wspace=0.32,
                                  height_ratios=[1.2, 1.2, 1.2, 1.2])

        # ── Panel 1: ROC — Rate of Change (3 timeframes) ──────────────────────
        ax = fig2.add_subplot(gs2[0, :])
        ax.plot(hist_m["DATE"], hist_m["ROC_5"],  color="#69ff47", lw=1.8,
                label="ROC 5d  (short-term)")
        ax.plot(hist_m["DATE"], hist_m["ROC_10"], color="#ffd740", lw=1.6,
                label="ROC 10d (medium-term)")
        ax.plot(hist_m["DATE"], hist_m["ROC_20"], color="#40c4ff", lw=1.4,
                label="ROC 20d (long-term)")
        ax.fill_between(hist_m["DATE"], hist_m["ROC_5"], 0,
                        where=hist_m["ROC_5"] > 0, alpha=0.12, color="#69ff47")
        ax.fill_between(hist_m["DATE"], hist_m["ROC_5"], 0,
                        where=hist_m["ROC_5"] <= 0, alpha=0.12, color="#ff3d57")
        ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
        # Current values annotated
        for val, lbl, clr in [
            (float(hist_m["ROC_5"].iloc[-1]),  f"5d {hist_m['ROC_5'].iloc[-1]:+.1f}%",  "#69ff47"),
            (float(hist_m["ROC_10"].iloc[-1]), f"10d {hist_m['ROC_10'].iloc[-1]:+.1f}%","#ffd740"),
            (float(hist_m["ROC_20"].iloc[-1]), f"20d {hist_m['ROC_20'].iloc[-1]:+.1f}%","#40c4ff"),
        ]:
            ax.annotate(lbl, xy=(hist_m["DATE"].iloc[-1], val),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=clr, fontweight="bold", clip_on=False)
        ax.set_title("Rate of Change  (ROC)  —  price momentum across 3 timeframes  |  all positive = strong",
                     fontsize=10, color="#8899aa", pad=5)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=7.5)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
        ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.15)

        # ── Panel 2: MACD Acceleration ────────────────────────────────────────
        ax = fig2.add_subplot(gs2[1, :])
        macd_hist_s = hist_m["MACD_HIST"]
        macd_acc_s  = hist_m["MACD_ACCEL"] if "MACD_ACCEL" in hist_m.columns else macd_hist_s.diff(3)
        ax.bar(hist_m["DATE"], macd_hist_s,
               color=[C[2] if v >= 0 else C[3] for v in macd_hist_s],
               width=0.9, alpha=0.55, label="MACD Histogram")
        ax.plot(hist_m["DATE"], macd_acc_s, color="#ffab40", lw=1.8,
                label="MACD Acceleration (3d change in hist)")
        ax.fill_between(hist_m["DATE"], macd_acc_s, 0,
                        where=macd_acc_s > 0, alpha=0.20, color="#ffab40")
        ax.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
        cur_acc = float(macd_acc_s.iloc[-1]) if not np.isnan(macd_acc_s.iloc[-1]) else 0
        acc_lbl = "BUILDING" if cur_acc > 0.5 else "MILD" if cur_acc > 0 else "FADING"
        acc_clr = "#69ff47" if cur_acc > 0.5 else "#ffd740" if cur_acc > 0 else "#ff3d57"
        ax.annotate(f"{cur_acc:+.3f}  {acc_lbl}",
                    xy=(hist_m["DATE"].iloc[-1], cur_acc),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8.5, color=acc_clr, fontweight="bold", clip_on=False)
        ax.set_title("MACD Histogram + Acceleration  |  orange rising = momentum building",
                     fontsize=10, color="#8899aa", pad=5)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=7.5)
        ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.15)

        # ── Panel 3: EMA Stack + Volume Momentum ──────────────────────────────
        ax3l = fig2.add_subplot(gs2[2, 0])
        if "EMA_STACK" in hist_m.columns:
            ema_s = hist_m["EMA_STACK"]
            clrs_s = ["#69ff47" if v==4 else "#ffd740" if v==3 else
                      "#ffab40" if v==2 else "#ff3d57" for v in ema_s]
            ax3l.bar(hist_m["DATE"], ema_s, color=clrs_s, width=0.9, alpha=0.85)
            ax3l.axhline(3, color="#ffd740", lw=0.7, ls=":", alpha=0.7, label="3/4 threshold")
            ax3l.axhline(4, color="#69ff47", lw=0.7, ls=":", alpha=0.7, label="Perfect (4/4)")
            ax3l.set_ylim(0, 4.5)
            ax3l.annotate(f"{int(ema_s.iloc[-1])}/4",
                          xy=(hist_m["DATE"].iloc[-1], float(ema_s.iloc[-1])),
                          xytext=(6,0), textcoords="offset points",
                          fontsize=9, color="#69ff47" if ema_s.iloc[-1]==4 else "#ffd740",
                          fontweight="bold", clip_on=False)
        ax3l.set_title("EMA Stack  (0=misaligned  4=perfect bull)",
                       fontsize=9, color="#8899aa", pad=5)
        ax3l.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax3l.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        plt.setp(ax3l.xaxis.get_majorticklabels(), fontsize=7.5)
        ax3l.legend(fontsize=7.5); ax3l.grid(True, alpha=0.15)

        ax3r = fig2.add_subplot(gs2[2, 1])
        if "VOL_MOMENTUM" in hist_m.columns:
            vm = hist_m["VOL_MOMENTUM"]
            ax3r.plot(hist_m["DATE"], vm, color="#38bdf8", lw=1.8)
            ax3r.fill_between(hist_m["DATE"], vm, 1,
                              where=vm >= 1, alpha=0.25, color="#38bdf8")
            ax3r.fill_between(hist_m["DATE"], vm, 1,
                              where=vm < 1, alpha=0.20, color="#ff3d57")
            ax3r.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.5)
            ax3r.axhline(1.3, color="#69ff47", lw=0.7, ls=":", alpha=0.6, label="1.3x rising")
            cur_vm = float(vm.iloc[-1]) if not np.isnan(vm.iloc[-1]) else 1
            vm_lbl = "RISING" if cur_vm >= 1.2 else "FLAT" if cur_vm >= 0.9 else "FALLING"
            vm_clr = "#69ff47" if cur_vm >= 1.2 else "#ffd740" if cur_vm >= 0.9 else "#ff3d57"
            ax3r.annotate(f"{cur_vm:.2f}x  {vm_lbl}",
                          xy=(hist_m["DATE"].iloc[-1], cur_vm),
                          xytext=(6,0), textcoords="offset points",
                          fontsize=8.5, color=vm_clr, fontweight="bold", clip_on=False)
        ax3r.set_title("Volume Momentum  (5d avg / 20d avg)  |  > 1 = rising participation",
                       fontsize=9, color="#8899aa", pad=5)
        ax3r.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax3r.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        plt.setp(ax3r.xaxis.get_majorticklabels(), fontsize=7.5)
        ax3r.legend(fontsize=7.5); ax3r.grid(True, alpha=0.15)

        # ── Panel 4: Gap Widening + MOM Score ─────────────────────────────────
        ax4l = fig2.add_subplot(gs2[3, 0])
        if "GAP_WIDENING" in hist_m.columns:
            gw = hist_m["GAP_WIDENING"]
            ax4l.fill_between(hist_m["DATE"], gw, 0,
                              where=gw > 0,  alpha=0.50, color="#69ff47")
            ax4l.fill_between(hist_m["DATE"], gw, 0,
                              where=gw <= 0, alpha=0.50, color="#ff3d57")
            ax4l.plot(hist_m["DATE"], gw, color="#c8d6e8", lw=1.0, alpha=0.6)
            ax4l.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
            ax4l.axhline(0.2,  color="#69ff47", lw=0.6, ls=":", alpha=0.6)
            ax4l.axhline(-0.3, color="#ff3d57", lw=0.6, ls=":", alpha=0.6)
            cur_gw = float(gw.iloc[-1]) if not np.isnan(gw.iloc[-1]) else 0
            gw_lbl = "WIDENING" if cur_gw > 0.1 else "STABLE" if cur_gw > -0.1 else "NARROWING"
            gw_clr = "#69ff47" if cur_gw > 0.1 else "#ffd740" if cur_gw > -0.1 else "#ff3d57"
            ax4l.annotate(f"{cur_gw:+.3f}  {gw_lbl}",
                          xy=(hist_m["DATE"].iloc[-1], cur_gw),
                          xytext=(6,0), textcoords="offset points",
                          fontsize=8.5, color=gw_clr, fontweight="bold", clip_on=False)
        ax4l.set_title("Gap Widening  (3d change in Gap/ATR)  |  green=cushion growing",
                       fontsize=9, color="#8899aa", pad=5)
        ax4l.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax4l.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        plt.setp(ax4l.xaxis.get_majorticklabels(), fontsize=7.5)
        ax4l.grid(True, alpha=0.15)

        ax4r = fig2.add_subplot(gs2[3, 1])
        if "MOM_SCORE" in hist_m.columns:
            ms = hist_m["MOM_SCORE"]
            clrs_ms = ["#69ff47" if v==3 else "#ffd740" if v==2 else
                       "#ffab40" if v==1 else "#ff3d57" for v in ms]
            ax4r.bar(hist_m["DATE"], ms, color=clrs_ms, width=0.9, alpha=0.85)
            ax4r.axhline(2, color="#ffd740", lw=0.7, ls=":", alpha=0.7)
            ax4r.axhline(3, color="#69ff47", lw=0.7, ls=":", alpha=0.7, label="All 3 up")
            ax4r.set_ylim(0, 3.5)
            cur_ms = int(ms.iloc[-1])
            ms_lbl = "ALL UP" if cur_ms==3 else "MIXED" if cur_ms==2 else "PARTIAL" if cur_ms==1 else "ALL DOWN"
            ms_clr = "#69ff47" if cur_ms==3 else "#ffd740" if cur_ms==2 else "#ffab40" if cur_ms==1 else "#ff3d57"
            ax4r.annotate(f"{cur_ms}/3  {ms_lbl}",
                          xy=(hist_m["DATE"].iloc[-1], float(ms.iloc[-1])),
                          xytext=(6,0), textcoords="offset points",
                          fontsize=8.5, color=ms_clr, fontweight="bold", clip_on=False)
        ax4r.set_title("Momentum Score  (ROC 5+10+20d all positive = 3/3)",
                       fontsize=9, color="#8899aa", pad=5)
        ax4r.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax4r.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        plt.setp(ax4r.xaxis.get_majorticklabels(), fontsize=7.5)
        ax4r.legend(fontsize=7.5); ax4r.grid(True, alpha=0.15)

        # Momentum summary footer
        mom = result.get("momentum", {})
        fig2.text(0.02, 0.005,
            f"MACD Accel:{mom.get('macd_accel_status','?')}  "
            f"EMA Stack:{mom.get('ema_stack_status','?')}  "
            f"ROC:{mom.get('roc_status','?')}  "
            f"Gap:{mom.get('gap_status','?')}  "
            f"Vol Trend:{mom.get('vol_trend','?')}  "
            f"{'⚡ MOMENTUM BOOST ACTIVE' if mom.get('momentum_boost') else ''}",
            fontsize=8.5, color="#ffd740" if mom.get("momentum_boost") else "#6a7f99",
            fontfamily="monospace")

        fig2.suptitle(f"{symbol}  —  Momentum Dashboard  |  {result['date']}",
                      fontsize=13, fontweight="bold", color=sig_clr)
        plt.tight_layout(rect=[0, 0.02, 1, 1])
        pdf.savefig(fig2, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
    except Exception as me:
        logger.warning(f"Momentum page failed for {symbol}: {me}")

    # ── PAGE 3: 3-Month Forecast ──────────────────────────────────────────────
    try:
        bundle = get_model(result.get("model_file", "stock_model.pkl"))
        if bundle:
            pipe  = bundle["model"]
            feats = [f for f in bundle["features"] if f in d_eng.columns]
            _d    = d_eng[feats + ["DATE", "CLOSE"]].dropna().copy().reset_index(drop=True)
            if not pd.api.types.is_datetime64_any_dtype(_d["DATE"]):
                _d["DATE"] = pd.to_datetime(_d["DATE"])

            HORIZON = 5
            _d["TARGET"] = _d["CLOSE"].pct_change(HORIZON).shift(-HORIZON) * 100
            _d.dropna(subset=["TARGET"], inplace=True)
            pipe_full = Pipeline([("sc", RobustScaler()),
                                   ("m", type(pipe.named_steps["m"])(
                                       **pipe.named_steps["m"].get_params()))])
            pipe_full.fit(_d[feats].values, _d["TARGET"].values)

            last_close = float(_d["CLOSE"].iloc[-1])
            last_date  = _d["DATE"].iloc[-1]
            cur_feats  = _d[feats].iloc[-1:].values.copy()
            closes     = [last_close]
            future     = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=63)
            for _ in range(63):
                pct   = float(np.clip(pipe_full.predict(cur_feats)[0] / HORIZON, -4.0, 4.0))
                new_c = closes[-1] * (1 + pct / 100)
                closes.append(new_c)
                for fi, fn in enumerate(feats):
                    if fn == "RET_1D":   cur_feats[0, fi] = pct
                    if fn == "TREND_5D": cur_feats[0, fi] = 1 if pct > 0 else 0
            closes = closes[1:]
            atr    = float(d_eng["ATR_14"].iloc[-1])
            fdf    = pd.DataFrame({"DATE": future[:len(closes)], "FORECAST_CLOSE": closes})
            band   = atr * (1 + np.linspace(0, 2.0, len(fdf)))
            fdf["UPPER"] = fdf["FORECAST_CLOSE"] + band
            fdf["LOWER"] = fdf["FORECAST_CLOSE"] - band

            e50h    = float(d_eng["EMA50_HIGH"].iloc[-1])
            e50l    = float(d_eng["EMA50_LOW"].iloc[-1])
            e10_now = float(d_eng["EMA_10"].iloc[-1])
            hist90  = d_eng.tail(90).reset_index(drop=True)

            fig2, axes = plt.subplots(3, 1, figsize=(20, 20), facecolor=BG,
                                       gridspec_kw={"hspace": 0.40})

            # Panel 1: historical + forecast
            ax = axes[0]
            ax.plot(hist90["DATE"], hist90["CLOSE"], color=C[0], lw=2.0,
                    label="Historical (last 90d)")
            ax.plot(fdf["DATE"], fdf["FORECAST_CLOSE"], color=C[4], lw=2.2,
                    ls="--", label="3-Month Forecast")
            ax.fill_between(fdf["DATE"], fdf["LOWER"], fdf["UPPER"],
                            alpha=0.14, color=C[4], label="ATR band")
            ax.fill_between(fdf["DATE"], e50h, e50l,
                            alpha=0.08, color=C[2],
                            label=f"EMA50 band [{e50l:.0f} - {e50h:.0f}]")
            ax.axhline(e50h, color=C[2], lw=1.3, ls=":", alpha=0.9,
                       label=f"EMA50H {e50h:.0f}")
            ax.axhline(e50l, color=C[3], lw=1.0, ls=":", alpha=0.7,
                       label=f"EMA50L {e50l:.0f}")
            ax.axvline(d_eng["DATE"].iloc[-1], color="white", lw=0.9,
                       ls=":", alpha=0.4)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            ax.set_title(f"{symbol}  — 3-Month Price Forecast  |  EMA50H = Rs.{e50h:.0f}",
                         fontsize=12, fontweight="bold", pad=8)
            ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.18)

            # Panel 2: forecast only — above/below EMA50H highlighted
            ax = axes[1]
            ax.plot(fdf["DATE"], fdf["FORECAST_CLOSE"], color=C[4], lw=2.2,
                    label="Forecast")
            ax.fill_between(fdf["DATE"], fdf["LOWER"], fdf["UPPER"],
                            alpha=0.14, color=C[4])
            ax.axhline(e50h, color=C[2], lw=1.5, ls="--",
                       label=f"EMA50H {e50h:.0f}  (target)")
            ax.axhline(e50l, color=C[3], lw=1.0, ls="--", alpha=0.7,
                       label=f"EMA50L {e50l:.0f}")
            ax.fill_between(fdf["DATE"], fdf["FORECAST_CLOSE"], e50h,
                            where=fdf["FORECAST_CLOSE"] > e50h,
                            alpha=0.22, color=C[2], label="Above EMA50H")
            ax.fill_between(fdf["DATE"], fdf["FORECAST_CLOSE"], e50h,
                            where=fdf["FORECAST_CLOSE"] <= e50h,
                            alpha=0.18, color=C[3], label="Below EMA50H")
            days_above = int((fdf["FORECAST_CLOSE"] > e50h).sum())
            for md in pd.date_range(fdf["DATE"].iloc[0], fdf["DATE"].iloc[-1], freq="MS"):
                ax.axvline(md, color="white", lw=0.4, ls="--", alpha=0.2)
                ax.text(md, fdf["LOWER"].min(), md.strftime(" %b"),
                        fontsize=8, color="#aaa", va="bottom")
            ax.text(0.02, 0.95,
                    f"Forecast days ABOVE EMA50H: {days_above}/63\n"
                    f"Forecast days BELOW EMA50H: {63-days_above}/63",
                    transform=ax.transAxes, va="top", fontsize=9,
                    color="#c8d6e8",
                    bbox=dict(boxstyle="round", facecolor=GRD, alpha=0.85))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            ax.set_title("Will price stay above EMA50_HIGH?  green=yes  red=no",
                         fontsize=11, fontweight="bold", pad=8)
            ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.18)

            # Panel 3: Gap trajectory last 90 days
            ax = axes[2]
            gap90 = hist90["GAP_ATR"]
            ax.plot(hist90["DATE"], gap90, color=C[5], lw=1.8,
                    label="Gap EMA10 vs 50H (ATR)")
            ax.fill_between(hist90["DATE"], gap90, 0,
                            where=gap90 > 0, alpha=0.28, color=C[2])
            ax.fill_between(hist90["DATE"], gap90, 0,
                            where=gap90 <= 0, alpha=0.28, color=C[3])
            gap_mom = gap90.diff(5) * 3
            ax.plot(hist90["DATE"], gap_mom, color=C[1], lw=1.0,
                    ls="--", alpha=0.6, label="Gap momentum x3")
            ax.axhline(0, color="white", lw=0.9, ls="--", alpha=0.5)
            ax.axhline(1, color=C[2], lw=0.7, ls=":", alpha=0.6,
                       label="+1 ATR GO line")
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
            ax.set_title("Gap trajectory last 90d  |  narrowing = level at risk",
                         fontsize=11, fontweight="bold", pad=8)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.18)

            fig2.suptitle(f"{symbol}  —  3-Month Price Forecast  |  "
                          f"Generated {datetime.now().strftime('%d %b %Y')}",
                          fontsize=13, fontweight="bold", color=sig_clr)
            plt.tight_layout()
            pdf.savefig(fig2, bbox_inches="tight", facecolor=BG)
            plt.close(fig2)
    except Exception as fe:
        logger.warning(f"Forecast page failed for {symbol}: {fe}")

    pdf.close()
    plt.rcParams.update(plt.rcParamsDefault)
    return pdf_path

# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    symbols:    Optional[List[str]] = None
    period:     Optional[str]       = "3y"
    start_date: Optional[str]       = None
    end_date:   Optional[str]       = None
    model:      Optional[str]       = "stock_model.pkl"

    @property
    def safe_period(self) -> str:
        allowed = {"6mo","1y","2y","3y","5y","max"}
        return self.period if self.period in allowed else "3y"


@app.get("/api/models")
def list_models():
    """List all available .pkl models in the /models directory."""
    models_dir = BASE_DIR / "models"
    models = []
    if models_dir.exists():
        for f in models_dir.glob("*.pkl"):
            # Try to get metadata without full load if possible, 
            # but get_model caches it anyway so it's efficient.
            bundle = get_model(f.name)
            if bundle:
                models.append({
                    "id": f.name,
                    "name": bundle["metrics"].get("model", f.name),
                    "acc": bundle["metrics"].get("dir_acc", 0),
                    "is_current": False # updated by frontend
                })
    return {"models": sorted(models, key=lambda x: x["acc"], reverse=True)}


@app.get("/api/health")
def health():
    default_model = get_model("stock_model.pkl")
    return {
        "status":      "ok",
        "model_count": len(MODELS_CACHE),
        "default_model": default_model["metrics"]["model"] if default_model else None,
    }


@app.get("/api/history")
def get_history():
    return {"history": read_history()}


@app.delete("/api/history/{symbol}")
def delete_history(symbol: str):
    history = [h for h in read_history() if h["symbol"] != symbol.upper()]
    write_history(history)
    return {"ok": True}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    # Normalize — handle None, empty list, or comma-separated strings
    raw_syms = req.symbols or []
    symbols  = []
    for s in raw_syms:
        for part in str(s).split(","):
            part = part.strip().upper()
            if part:
                symbols.append(part)

    if not symbols:
        raise HTTPException(422, "Provide at least one valid stock symbol")
    if len(symbols) > 10:
        raise HTTPException(422, "Maximum 10 stocks per request")

    logger.info(f"Analyze request: symbols={symbols} period={req.period} "
                f"start={req.start_date} end={req.end_date}")

    results = []
    errors  = []

    for raw_sym in symbols:
        sym = raw_sym.strip().upper()
        try:
            logger.info(f"Analyzing {sym}...")
            df, ticker = fetch_yahoo(sym, req.safe_period,
                                     start_date=req.start_date,
                                     end_date=req.end_date)
            d_eng      = engineer(df)
            result     = generate_signal(d_eng, sym, model_name=req.model)

            # Build PDF in background (non-blocking)
            result["pdf_ready"] = False
            PDF_READY_CACHE[sym] = "pending"
            background_tasks.add_task(build_pdf_bg, dict(result), d_eng.copy(), sym)

            add_to_history(sym, sym, result["signal"])
            results.append(result)

        except HTTPException as he:
            errors.append({"symbol": sym, "error": he.detail})
        except Exception as e:
            logger.error(f"Error analyzing {sym}: {e}")
            errors.append({"symbol": sym, "error": str(e)})

    return {"results": results, "errors": errors, "count": len(results)}


@app.get("/api/pdf_status/{symbol}")
def pdf_status(symbol: str):
    sym = symbol.upper()
    return {"status": PDF_READY_CACHE.get(sym, "not_found")}


@app.get("/api/download/{symbol}")
def download_pdf(symbol: str):
    pdf_path = PDF_DIR / f"{symbol.upper()}_analysis.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, f"Report not found for {symbol}. Analyze first.")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"{symbol.upper()}_analysis_{datetime.now().strftime('%Y%m%d')}.pdf"
    )


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(
            content=html_path.read_text(encoding="utf-8"),
            media_type="text/html; charset=utf-8"
        )
    return HTMLResponse("<h1>Stock Analysis API</h1><p>Place index.html in /static/</p>")
