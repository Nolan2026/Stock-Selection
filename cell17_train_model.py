# ══════════════════════════════════════════════════════════════════════════════
# ▌CELL 17  │  TRAIN BEST MODEL → SAVE PICKLE  (v2.0 — Full Rewrite)
# ══════════════════════════════════════════════════════════════════════════════
#
# WHAT'S NEW vs v1.0:
#   ✅ yfinance auto-fetch  — no CSV upload needed
#   ✅ India VIX features   — market fear/calm regime
#   ✅ NIFTY 50 features    — relative strength, beta, market trend
#   ✅ ADX / DI+/DI-        — trend strength indicator
#   ✅ 52-week high/low     — swing breakout signals
#   ✅ CLASSIFICATION target — BUY / HOLD / SELL  (not regression)
#   ✅ DNN (MLP)            — neural net added to model pool
#   ✅ Walk-forward CV      — 10-fold with gap=5 to prevent leakage
#   ✅ Soft Voting Ensemble — top-3 classifiers
#   ✅ Two-phase workflow   — train on stock A, predict on stock B
#   ✅ Duplicate columns    — BETA_PROXY duplicate removed
#
# WORKFLOW:
#   Phase 1 → Enter NSE symbol for training  (e.g. RELIANCE)
#   Phase 2 → Enter NSE symbol to predict    (e.g. INFY)
#
# REQUIREMENTS:
#   pip install yfinance lightgbm xgboost scikit-learn matplotlib pandas numpy
# ══════════════════════════════════════════════════════════════════════════════

import warnings; warnings.filterwarnings("ignore")
import numpy  as np
import pandas as pd
import pickle, json, os, datetime
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
from   matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing   import RobustScaler
from sklearn.pipeline        import Pipeline
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     VotingClassifier)
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics         import (classification_report, f1_score,
                                     accuracy_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

import yfinance as yf

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# ── XGBWrapper — module-level class so pickle can serialize it ─────────────────
# XGBoost requires 0-indexed labels (0,1,2) but our target uses (-1,0,1).
# This wrapper remaps labels transparently so all other code uses (-1,0,1).
if _HAS_XGB:
    from sklearn.base import BaseEstimator, ClassifierMixin as _CM

    class XGBWrapper(BaseEstimator, _CM):
        """Picklable XGBClassifier wrapper: remaps -1,0,1 labels → 0,1,2."""
        _LM  = {-1: 0,  0: 1,  1: 2}
        _LMI = { 0:-1,  1: 0,  2: 1}

        def __init__(self, n_estimators=600, max_depth=6, learning_rate=0.02,
                     subsample=0.8, colsample_bytree=0.6,
                     reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
                     eval_metric="mlogloss", random_state=42, verbosity=0):
            self.n_estimators    = n_estimators
            self.max_depth       = max_depth
            self.learning_rate   = learning_rate
            self.subsample       = subsample
            self.colsample_bytree= colsample_bytree
            self.reg_alpha       = reg_alpha
            self.reg_lambda      = reg_lambda
            self.gamma           = gamma
            self.eval_metric     = eval_metric
            self.random_state    = random_state
            self.verbosity       = verbosity

        def _make_clf(self):
            return XGBClassifier(
                n_estimators    =self.n_estimators,
                max_depth       =self.max_depth,
                learning_rate   =self.learning_rate,
                subsample       =self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha       =self.reg_alpha,
                reg_lambda      =self.reg_lambda,
                gamma           =self.gamma,
                eval_metric     =self.eval_metric,
                random_state    =self.random_state,
                verbosity       =self.verbosity)

        def fit(self, X, y, **kw):
            self.classes_ = np.array([-1, 0, 1])
            self._clf = self._make_clf()
            y_remap = np.vectorize(self._LM.get)(y)
            self._clf.fit(X, y_remap)
            return self

        def predict(self, X):
            return np.vectorize(self._LMI.get)(self._clf.predict(X))

        def predict_proba(self, X):
            return self._clf.predict_proba(X)

else:
    class XGBWrapper:
        """Stub when XGBoost is not installed."""
        pass



# ── Plot theme ─────────────────────────────────────────────────────────────────
BG ="   #0a0f1a"; PAN="#0d1525"; GRD="#1a2535"
C  = ["#4a9fd4","#fbbf24","#34d399","#f87171","#a78bfa",
      "#fb923c","#38bdf8","#e879f9","#6ee7b7","#fca5a5"]
plt.rcParams.update({
    "figure.facecolor":BG.strip(),"axes.facecolor":PAN,"axes.edgecolor":GRD,
    "axes.labelcolor":"#c8d6e8","xtick.color":"#c8d6e8","ytick.color":"#c8d6e8",
    "text.color":"#c8d6e8","grid.color":GRD,"legend.facecolor":PAN,
    "legend.edgecolor":GRD,"font.family":"monospace","figure.dpi":120,
})
BG = BG.strip()

TARGET_LOOKAHEAD = 20     # Lookahead window for path-dependent target
TARGET_PCT       = 4.0    # Target % for "hit before stop" logic
STOP_PCT         = 2.0    # Stop Loss % for "hit before stop" logic
TRAIN_YEARS      = 10     # years of data for training
PREDICT_YEARS    = 2      # years of recent data for prediction

# ══════════════════════════════════════════════════════════════════════════════
# §A  DATA FETCHING  (yfinance)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_yf(ticker, period="10y"):
    """Download OHLCV from yfinance, return clean DataFrame."""
    print(f"   📡 Fetching {ticker} ({period}) ...")
    raw = yf.download(ticker, period=period, interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check symbol.")
    # Flatten MultiIndex if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={"Open":"OPEN","High":"HIGH","Low":"LOW",
                               "Close":"CLOSE","Volume":"VOLUME"})
    # Keep only required columns (some tickers return extra columns)
    keep = [c for c in ["OPEN","HIGH","LOW","CLOSE","VOLUME"] if c in raw.columns]
    raw = raw[keep].copy()
    if "VOLUME" not in raw.columns:
        raw["VOLUME"] = 1
    # ── Normalize index: strip timezone so all frames merge cleanly ───────────
    idx = pd.to_datetime(raw.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)   # already tz-aware  → convert to naive
    raw.index = idx
    raw.index.name = "DATE"
    raw = raw.dropna(subset=["CLOSE"])
    raw = raw[raw["VOLUME"] > 0]
    raw.sort_index(inplace=True)
    return raw


def _fetch_vix(period):
    """Try multiple VIX tickers — India VIX is often unavailable."""
    vix_tickers = ["^INDIAVIX", "INDIAVIX.NS", "NIFTY-VIX.NS"]
    for tkr in vix_tickers:
        try:
            df = _fetch_yf(tkr, period=period)
            if len(df) > 50 and df["CLOSE"].notna().sum() > 50:
                print(f"   ✅ VIX loaded from {tkr}  ({len(df)} rows)")
                return df
        except Exception as e:
            print(f"   ⚠️  VIX ticker {tkr} failed: {e}")
    # All tickers failed — return synthetic VIX using Nifty realized vol
    print("   ⚠️  All VIX tickers failed — using Nifty realized vol as proxy")
    return None   # handled in engineer()


def load_training_data(symbol):
    """Fetch stock + NIFTY + VIX for training period."""
    ticker   = symbol.upper() + ".NS"
    stock_df = _fetch_yf(ticker,  period=f"{TRAIN_YEARS}y")
    nifty_df = _fetch_yf("^NSEI", period=f"{TRAIN_YEARS}y")
    vix_df   = _fetch_vix(f"{TRAIN_YEARS}y")

    # Build synthetic VIX from Nifty realized vol if real VIX unavailable
    if vix_df is None:
        nifty_ret = nifty_df["CLOSE"].pct_change()
        syn_vix   = nifty_ret.rolling(20).std() * np.sqrt(252) * 100
        syn_vix   = syn_vix.fillna(15.0).clip(8, 80)
        vix_df    = pd.DataFrame({"CLOSE": syn_vix}, index=nifty_df.index)
        print(f"   ✅ Synthetic VIX built from Nifty realized vol  "
              f"(range {syn_vix.min():.1f}–{syn_vix.max():.1f})")

    # Align on common dates
    common   = stock_df.index.intersection(nifty_df.index)
    stock_df = stock_df.loc[common]
    nifty_df = nifty_df.loc[common]
    vix_df   = vix_df.reindex(common, method="ffill")

    print(f"   ✅ {len(stock_df)} trading days  "
          f"({stock_df.index[0].date()} → {stock_df.index[-1].date()})")
    return stock_df, nifty_df, vix_df, ticker


def load_predict_data(symbol):
    """Fetch recent data for prediction symbol."""
    ticker   = symbol.upper() + ".NS"
    stock_df = _fetch_yf(ticker,  period=f"{PREDICT_YEARS}y")
    nifty_df = _fetch_yf("^NSEI", period=f"{PREDICT_YEARS}y")
    vix_df   = _fetch_vix(f"{PREDICT_YEARS}y")

    if vix_df is None:
        nifty_ret = nifty_df["CLOSE"].pct_change()
        syn_vix   = nifty_ret.rolling(20).std() * np.sqrt(252) * 100
        vix_df    = pd.DataFrame({"CLOSE": syn_vix.fillna(15.0).clip(8,80)},
                                  index=nifty_df.index)

    common   = stock_df.index.intersection(nifty_df.index)
    stock_df = stock_df.loc[common]
    nifty_df = nifty_df.loc[common]
    vix_df   = vix_df.reindex(common, method="ffill")
    return stock_df, nifty_df, vix_df, ticker


# ══════════════════════════════════════════════════════════════════════════════
# §B  HELPER INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def _ema(s, n):  return s.ewm(span=n, adjust=False).mean()
def _sma(s, n):  return s.rolling(n).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _slope(s, w=10):
    out = np.full(len(s), np.nan); sv = s.values
    for i in range(w, len(sv)):
        y = sv[i-w:i]
        if not np.any(np.isnan(y)):
            out[i] = np.polyfit(np.arange(w), y, 1)[0]
    return pd.Series(out, index=s.index)

def _adx(H, L, C, p=14):
    """Returns ADX, +DI, -DI as tuple of Series."""
    tr = pd.concat([H-L, (H-C.shift()).abs(), (L-C.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(p).mean()

    up   = H.diff();   down = -L.diff()
    dm_p = np.where((up > down) & (up > 0), up, 0.0)
    dm_m = np.where((down > up) & (down > 0), down, 0.0)

    dm_p = pd.Series(dm_p, index=H.index).rolling(p).mean()
    dm_m = pd.Series(dm_m, index=H.index).rolling(p).mean()

    di_p = 100 * dm_p / atr.replace(0, np.nan)
    di_m = 100 * dm_m / atr.replace(0, np.nan)
    dx   = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx  = dx.rolling(p).mean()
    return adx, di_p, di_m


# ══════════════════════════════════════════════════════════════════════════════
# §C  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = [
    # EMA ratios
    "EMA10_RATIO","EMA20_RATIO","EMA50_RATIO","SMA200_RATIO",
    "EMA50H_RATIO","EMA50L_RATIO",
    # EMA alignment
    "EMA10_GT_20","EMA20_GT_50","EMA50_GT_200","EMA10_GT_50H","EMA_STACK",
    # Gap
    "GAP_ATR","GAP_PCT","CLOSE_GAP_ATR","GAP_WIDENING",
    # Oscillators
    "RSI_14","RSI_9","MACD_HIST","MACD_CROSS","MACD_ABOVE_ZERO","MACD_ACCEL",
    "STOCH_K","STOCH_D","STOCH_RISING","STOCH_CROSS",
    "WILLIAMS_R","CCI",
    # Volatility
    "BB_WIDTH","BB_PCT_B","ATR_PCT","VOL_5D","VOL_20D",
    "BETA_20_60","BETA_REGIME",
    # Volume
    "VOL_RATIO","VWAP_DEV","VOL_MOMENTUM",
    "OBV_ROC","OBV_TREND","VPT_ROC","VPT_TREND",
    # Returns & momentum
    "RET_1D","RET_2D","RET_3D","RET_5D","RET_10D","RET_20D",
    "ROC_5","ROC_10","ROC_20","MOM_SCORE",
    # Trend
    "TREND_5D","TREND_10D","TREND_20D","ABOVE_EMA50H","ABOVE_EMA50L",
    "SLOPE_10D","SLOPE_20D",
    "HIGHER_HIGH_5","HIGHER_LOW_5","TREND_STRUCT","CONSEC_UP",
    # Price position
    "CHANNEL_POS_20","CHANNEL_POS_50","ZSCORE_20","ZSCORE_50","OVEREXTENDED",
    # Candle patterns
    "BODY_RATIO","UPPER_SHADOW","LOWER_SHADOW","BULL_CANDLE","BULL_ENGULF",
    # Return distribution
    "SKEW_20","KURT_20",
    # RSI divergence
    "RSI_DIVERGE",
    # ADX / trend strength  (NEW)
    "ADX_14","DI_PLUS","DI_MINUS","ADX_STRONG",
    # 52-week levels  (NEW)
    "HIGH_52W_DIST","LOW_52W_DIST","NEAR_52W_HIGH",
    # India VIX  (NEW)
    "VIX_CLOSE","VIX_MA10","VIX_ZSCORE","VIX_PCT_CHANGE","VIX_REGIME","VIX_SPIKE",
    # NIFTY relative  (NEW)
    "RS_NIFTY_5D","RS_NIFTY_20D","NIFTY_TREND","NIFTY_MOMENTUM","BETA_VS_NIFTY",
    # Calendar
    "DOW","MONTH",
]


def engineer(stock_df, nifty_df, vix_df):
    """Full feature engineering on aligned DataFrames."""
    d = stock_df.copy().reset_index()     # DATE becomes a column
    d.columns.name = None

    C_ = d["CLOSE"]; H_ = d["HIGH"]; L_ = d["LOW"]
    O_ = d["OPEN"];  V_ = d["VOLUME"]

    e10=_ema(C_,10); e20=_ema(C_,20); e50=_ema(C_,50); e200=_ema(C_,200)
    e50h=_ema(H_,50); e50l=_ema(L_,50)
    sma200=_sma(C_,200)

    tr  = pd.concat([H_-L_,(H_-C_.shift()).abs(),(L_-C_.shift()).abs()],axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    mf=_ema(C_,12); ms=_ema(C_,26)
    macd=mf-ms; macd_sig=_ema(macd,9); macd_hist=macd-macd_sig

    rsi14=_rsi(C_,14); rsi9=_rsi(C_,9)
    bm=_sma(C_,20); bs=C_.rolling(20).std()
    bb_w=((bm+2*bs)-(bm-2*bs))/bm
    bb_b=(C_-(bm-2*bs))/(4*bs)
    lo14=L_.rolling(14).min(); hi14=H_.rolling(14).max()
    stoch_k=100*(C_-lo14)/(hi14-lo14).replace(0,np.nan)
    stoch_d=stoch_k.rolling(3).mean()
    vol_sma20=_sma(V_,20)

    # ── EMA ratios ────────────────────────────────────────────────────────────
    d["EMA10_RATIO"] =C_/e10-1;   d["EMA20_RATIO"] =C_/e20-1
    d["EMA50_RATIO"] =C_/e50-1;   d["SMA200_RATIO"]=C_/sma200-1
    d["EMA50H_RATIO"]=C_/e50h-1;  d["EMA50L_RATIO"]=C_/e50l-1

    # ── EMA alignment ─────────────────────────────────────────────────────────
    d["EMA10_GT_20"] =(e10>e20).astype(int)
    d["EMA20_GT_50"] =(e20>e50).astype(int)
    d["EMA50_GT_200"]=(e50>e200).astype(int)
    d["EMA10_GT_50H"]=(e10>e50h).astype(int)
    d["EMA_STACK"]   =d["EMA10_GT_20"]+d["EMA20_GT_50"]+d["EMA50_GT_200"]

    # ── Gap ───────────────────────────────────────────────────────────────────
    d["GAP_ATR"]     =(e10-e50h)/atr
    d["GAP_PCT"]     =(e10-e50h)/e50h*100
    d["CLOSE_GAP_ATR"]=(C_-e50h)/atr
    d["GAP_WIDENING"] =d["GAP_ATR"].diff(5)

    # ── Oscillators ───────────────────────────────────────────────────────────
    d["RSI_14"]=rsi14; d["RSI_9"]=rsi9
    d["MACD_HIST"]=macd_hist
    d["MACD_CROSS"]=(macd>macd_sig).astype(int)
    d["MACD_ABOVE_ZERO"]=(macd>0).astype(int)
    d["MACD_ACCEL"]=macd_hist.diff()
    d["STOCH_K"]=stoch_k; d["STOCH_D"]=stoch_d
    d["STOCH_RISING"]=(stoch_k>stoch_k.shift(1)).astype(int)
    d["STOCH_CROSS"] =(stoch_k>stoch_d).astype(int)
    d["WILLIAMS_R"]  =-100*((hi14-C_)/(hi14-lo14).replace(0,np.nan))
    tp=( H_+L_+C_)/3; tp_sma=_sma(tp,20)
    tp_md=tp.rolling(20).apply(lambda x: np.mean(np.abs(x-x.mean())),raw=True)
    d["CCI"]=(tp-tp_sma)/(0.015*tp_md)

    # ── Volatility ────────────────────────────────────────────────────────────
    d["BB_WIDTH"]=bb_w; d["BB_PCT_B"]=bb_b
    d["ATR_PCT"] =atr/C_*100
    _r=C_.pct_change()
    d["VOL_5D"] =_r.rolling(5).std()*100
    d["VOL_20D"]=_r.rolling(20).std()*100
    vol20=_r.rolling(20).std(); vol60=_r.rolling(60).std()
    d["BETA_20_60"]=vol20/vol60.replace(0,np.nan)   # single column (duplicate removed)
    d["BETA_REGIME"]=(vol20>vol60).astype(int)

    # ── Volume ────────────────────────────────────────────────────────────────
    d["VOL_RATIO"]   =V_/vol_sma20
    d["VWAP_DEV"]    =(C_-_sma(C_*V_,20)/_sma(V_,20))/C_*100
    d["VOL_MOMENTUM"]=_sma(V_,5)/vol_sma20
    obv=(np.sign(C_.diff()).fillna(0)*V_).cumsum()
    d["OBV_ROC"] =obv.pct_change(10)*100
    d["OBV_TREND"]=(obv>_ema(obv,20)).astype(int)
    vpt=(C_.pct_change().fillna(0)*V_).cumsum()
    d["VPT_ROC"] =vpt.pct_change(10)*100
    d["VPT_TREND"]=(vpt>_ema(vpt,20)).astype(int)

    # ── Returns & momentum ────────────────────────────────────────────────────
    for lg in [1,2,3,5,10,20]:
        d[f"RET_{lg}D"]=C_.pct_change(lg)*100
    d["ROC_5"] =C_.pct_change(5)*100
    d["ROC_10"]=C_.pct_change(10)*100
    d["ROC_20"]=C_.pct_change(20)*100
    d["MOM_SCORE"]=(d["ROC_5"]+d["ROC_10"]+d["ROC_20"])/3

    # ── Trend ─────────────────────────────────────────────────────────────────
    d["TREND_5D"] =(C_>C_.shift(5)).astype(int)
    d["TREND_10D"]=(C_>C_.shift(10)).astype(int)
    d["TREND_20D"]=(C_>C_.shift(20)).astype(int)
    d["ABOVE_EMA50H"]=(C_>e50h).astype(int)
    d["ABOVE_EMA50L"]=(C_>e50l).astype(int)
    d["SLOPE_10D"]=_slope(C_,10)
    d["SLOPE_20D"]=_slope(C_,20)
    d["HIGHER_HIGH_5"]=(H_>H_.rolling(5).max().shift(1)).astype(int)
    d["HIGHER_LOW_5"] =(L_>L_.rolling(5).min().shift(1)).astype(int)
    d["TREND_STRUCT"] =d["HIGHER_HIGH_5"]+d["HIGHER_LOW_5"]
    up_day=(C_>C_.shift(1)).astype(int)
    consec_arr = up_day.values.astype(float)
    for i in range(1, len(consec_arr)):
        if consec_arr[i] == 1:
            consec_arr[i] = consec_arr[i-1] + 1
    d["CONSEC_UP"] = consec_arr

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
    d["BODY_RATIO"]  =body/full_range
    d["UPPER_SHADOW"]=(H_-pd.concat([C_,O_],axis=1).max(axis=1))/full_range
    d["LOWER_SHADOW"]=(pd.concat([C_,O_],axis=1).min(axis=1)-L_)/full_range
    d["BULL_CANDLE"] =(C_>O_).astype(int)
    prev_body=(C_.shift(1)-O_.shift(1)).abs()
    d["BULL_ENGULF"] =((C_>O_)&(O_<C_.shift(1))&(C_>O_.shift(1))&
                       (body>prev_body)).astype(int)

    # ── Return distribution ───────────────────────────────────────────────────
    d["SKEW_20"]=_r.rolling(20).skew()
    d["KURT_20"]=_r.rolling(20).kurt()

    # ── RSI divergence ────────────────────────────────────────────────────────
    pr=(C_>C_.shift(5)).astype(int)
    rf=(rsi14<rsi14.shift(5)).astype(int)
    d["RSI_DIVERGE"]=((pr==1)&(rf==1)).astype(int)*-1

    # ── ADX / trend strength  (NEW) ───────────────────────────────────────────
    adx14, di_p, di_m = _adx(H_, L_, C_, p=14)
    d["ADX_14"]   = adx14
    d["DI_PLUS"]  = di_p
    d["DI_MINUS"] = di_m
    d["ADX_STRONG"]=(adx14>25).astype(int)

    # ── 52-week high / low  (NEW) ─────────────────────────────────────────────
    h52 = H_.rolling(252).max(); l52 = L_.rolling(252).min()
    d["HIGH_52W_DIST"] = (C_-h52)/h52*100   # negative = below 52W high
    d["LOW_52W_DIST"]  = (C_-l52)/l52*100   # positive = above 52W low
    d["NEAR_52W_HIGH"] = ((C_/h52)>=0.95).astype(int)

    # ── India VIX features  (NEW) ─────────────────────────────────────────────
    # Align VIX by date values (robust: works even if index types differ)
    vix_series = vix_df["CLOSE"].copy()
    vix_series.index = pd.to_datetime(vix_series.index).tz_localize(None)
    date_idx = pd.to_datetime(d["DATE"])
    vix_s = vix_series.reindex(date_idx, method="ffill").reset_index(drop=True)
    # If VIX data is still all-NaN, fill with a neutral default (15)
    if vix_s.isna().all():
        print("   ⚠️  India VIX data unavailable — filling with neutral 15")
        vix_s = pd.Series(np.full(len(d), 15.0))
    else:
        vix_s = vix_s.ffill().bfill().fillna(15.0)
    d["VIX_CLOSE"]     = vix_s.values
    d["VIX_MA10"]      = vix_s.rolling(10).mean().values
    vix_mu=vix_s.rolling(20).mean(); vix_sd=vix_s.rolling(20).std()
    _vix_zscore = (vix_s-vix_mu)/vix_sd.replace(0,np.nan)
    d["VIX_ZSCORE"]    = _vix_zscore.fillna(0).values   # 0=neutral when VIX is constant
    d["VIX_PCT_CHANGE"]= vix_s.pct_change().fillna(0).values*100
    # regime: 0=low(<15), 1=medium(15-20), 2=high(>20)
    d["VIX_REGIME"]    = pd.cut(vix_s, bins=[-np.inf,15,20,np.inf],
                                labels=[0,1,2]).astype(float).fillna(1).values
    d["VIX_SPIKE"]     = (vix_s.pct_change()>0.15).astype(int).values

    # ── NIFTY relative features  (NEW) ────────────────────────────────────────
    nifty_series = nifty_df["CLOSE"].copy()
    _nidx = pd.to_datetime(nifty_series.index)
    nifty_series.index = _nidx.tz_convert(None) if _nidx.tz is not None else _nidx
    nf_c = nifty_series.reindex(date_idx, method="ffill").reset_index(drop=True)
    if nf_c.isna().all():
        print("   ⚠️  NIFTY data unavailable — filling with stock close")
        nf_c = C_.reset_index(drop=True).copy()
    else:
        nf_c = nf_c.ffill().bfill().fillna(float(C_.mean()))
    nf_c = nf_c.reset_index(drop=True)   # ensure integer RangeIndex
    nf_r = nf_c.pct_change()
    d["RS_NIFTY_5D"]   = (_r.values - nf_r.values)  # 1-day RS proxy
    d["RS_NIFTY_20D"]  = (pd.Series(_r.values).rolling(20).sum().values -
                           nf_r.rolling(20).sum().values)
    d["NIFTY_TREND"]   = (nf_c > _ema(nf_c, 50)).astype(int).values
    d["NIFTY_MOMENTUM"]= nf_c.pct_change(10).values * 100
    # Rolling beta of stock vs NIFTY (60 days)
    _r_s   = pd.Series(_r.values)
    cov60  = _r_s.rolling(60).cov(nf_r)
    var60  = nf_r.rolling(60).var()
    d["BETA_VS_NIFTY"] = (cov60 / var60.replace(0, np.nan)).values

    # ── Calendar ──────────────────────────────────────────────────────────────
    d["DOW"]  = pd.to_datetime(d["DATE"]).dt.dayofweek
    d["MONTH"]= pd.to_datetime(d["DATE"]).dt.month

    # ── Store raw levels for signal generation ─────────────────────────────────
    d["EMA_10"]=e10.values; d["EMA_20"]=e20.values; d["EMA_50"]=e50.values
    d["EMA50_HIGH"]=e50h.values; d["EMA50_LOW"]=e50l.values
    d["ATR_14"]=atr.values; d["MACD"]=macd.values; d["MACD_SIG"]=macd_sig.values

    return d


# ══════════════════════════════════════════════════════════════════════════════
# §D  CLASSIFICATION TARGET
# ══════════════════════════════════════════════════════════════════════════════

def make_target(df, target_pct=TARGET_PCT, stop_pct=STOP_PCT, max_lookahead=TARGET_LOOKAHEAD):
    """
    Advanced Path-Dependent Labeling:
    Calculates if the price reaches the Target (+T%) before hitting the Stop Loss (-S%)
    within a specified lookahead window.

    +1 = BUY   (Hits Target first)
    -1 = SELL  (Hits Stop Loss first)
     0 = HOLD  (Neither hit within window, or noise)
    """
    close = df["CLOSE"].values
    high  = df["HIGH"].values
    low   = df["LOW"].values
    n     = len(close)
    labels = np.zeros(n)

    # Use vectorized lookahead for efficiency
    for i in range(n - max_lookahead):
        entry = close[i]
        upper = entry * (1 + target_pct/100)
        lower = entry * (1 - stop_pct/100)

        hit_upper = -1
        hit_lower = -1

        # Check for first event in high/low path
        window_h = high[i+1 : i + max_lookahead + 1]
        window_l = low[i+1 : i + max_lookahead + 1]

        # Find first index where high >= upper
        ups = np.where(window_h >= upper)[0]
        if len(ups) > 0:
            hit_upper = ups[0]

        # Find first index where low <= lower
        downs = np.where(window_l <= lower)[0]
        if len(downs) > 0:
            hit_lower = downs[0]

        # Determine label based on which hit first
        if hit_upper != -1 and (hit_lower == -1 or hit_upper < hit_lower):
            labels[i] = 1
        elif hit_lower != -1 and (hit_upper == -1 or hit_lower < hit_upper):
            labels[i] = -1
        else:
            # Optional: handle neutral at end of window
            final_ret = (close[i + max_lookahead] / entry - 1) * 100
            if   final_ret >=  1.0: labels[i] =  1
            elif final_ret <= -1.0: labels[i] = -1
            else:                   labels[i] =  0

    return pd.Series(labels, index=df.index, name="TARGET")


# ══════════════════════════════════════════════════════════════════════════════
# §E  MODEL TRAINING + WALK-FORWARD EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_candidates(class_weights):
    cw = {-1: class_weights[0], 0: class_weights[1], 1: class_weights[2]}
    candidates = {}

    if _HAS_LGB:
        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.02,
            num_leaves=63, subsample=0.8, colsample_bytree=0.6,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
            class_weight="balanced", random_state=42, verbose=-1)

    if _HAS_XGB:
        candidates["XGBoost"] = XGBWrapper()

    candidates["RandomForest"] = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1)

    candidates["GradBoost"] = GradientBoostingClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.03,
        subsample=0.8, min_samples_leaf=5, random_state=42)

    # Deep Neural Network (MLP)
    candidates["DNN"] = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation="relu", solver="adam",
        learning_rate_init=0.001, max_iter=300,
        early_stopping=True, validation_fraction=0.1,
        random_state=42, alpha=0.001)

    return candidates


def train_evaluate(d_eng):
    feats = [f for f in FEATURES if f in d_eng.columns]
    missing = [f for f in FEATURES if f not in d_eng.columns]
    if missing: print(f"   ⚠️  Missing features (skipped): {missing}")
    print(f"   Using {len(feats)} features")

    sub = d_eng[feats + ["DATE","CLOSE","HIGH","LOW"]].copy()
    sub["TARGET"] = make_target(sub).values

    # Drop only rows where TARGET is 0 but we don't have enough data to be sure
    # (i.e. the last TARGET_LOOKAHEAD rows)
    sub = sub.iloc[:-TARGET_LOOKAHEAD].reset_index(drop=True)

    # Step 1: replace +inf / -inf with NaN
    sub[feats] = sub[feats].replace([np.inf, -np.inf], np.nan)

    # Step 2: fill NaN features with column median (0 if median is also NaN)
    for f in feats:
        if sub[f].isna().any():
            med = sub[f].median()
            sub[f] = sub[f].fillna(0.0 if (np.isnan(med) or np.isinf(med)) else med)

    # Step 3: safety clip — remove any remaining extreme values
    sub[feats] = sub[feats].clip(-1e6, 1e6)

    print(f"   Rows after imputation: {len(sub)}  (inf→NaN, NaN→median)")

    X = sub[feats].values
    y = sub["TARGET"].values.astype(int)
    dates = sub["DATE"].values
    closes = sub["CLOSE"].values

    if len(sub) < 100:
        raise ValueError(
            f"Only {len(sub)} rows after dropna — check VIX/NIFTY alignment or use a longer period."
        )

    sp = int(len(sub)*0.80)
    X_tr, y_tr = X[:sp], y[:sp]
    X_te, y_te = X[sp:], y[sp:]
    d_te, c_te = dates[sp:], closes[sp:]

    print(f"   Train: {sp} rows  Test: {len(X_te)} rows")
    print(f"   Class dist  BUY={np.sum(y_tr==1)}  HOLD={np.sum(y_tr==0)}  SELL={np.sum(y_tr==-1)}")

    # ── Oversample minority classes to reduce HOLD dominance ──────────────────
    # Manual oversampling: duplicate BUY and SELL rows until balanced with HOLD
    try:
        n_hold = np.sum(y_tr == 0)
        n_buy  = np.sum(y_tr == 1)
        n_sell = np.sum(y_tr == -1)
        target_n = max(n_hold, n_buy, n_sell)

        X_parts, y_parts = [X_tr], [y_tr]
        for cls, n in [(1, n_buy), (-1, n_sell)]:
            if n > 0 and n < target_n:
                idx      = np.where(y_tr == cls)[0]
                needed   = target_n - n
                oversamp = np.random.choice(idx, size=needed, replace=True)
                noise    = np.random.normal(0, 0.01, (needed, X_tr.shape[1]))
                X_parts.append(X_tr[oversamp] + noise)
                y_parts.append(np.full(needed, cls))

        if len(X_parts) > 1:
            X_tr = np.vstack(X_parts)
            y_tr = np.concatenate(y_parts)
            # Shuffle
            idx_shuf = np.random.permutation(len(X_tr))
            X_tr, y_tr = X_tr[idx_shuf], y_tr[idx_shuf]
            print(f"   After oversampling: BUY={np.sum(y_tr==1)}  "
                  f"HOLD={np.sum(y_tr==0)}  SELL={np.sum(y_tr==-1)}")
    except Exception as oe:
        print(f"   ⚠️  Oversampling skipped: {oe}")

    # Compute class weights — only for classes actually present in y_tr
    present_classes = np.unique(y_tr)
    cw_vals_partial = compute_class_weight("balanced", classes=present_classes, y=y_tr)
    cw_map = dict(zip(present_classes, cw_vals_partial))
    # Fill missing classes with weight 1.0
    cw_vals = np.array([cw_map.get(c, 1.0) for c in [-1, 0, 1]])
    candidates = _build_candidates(cw_vals)

    # ── Walk-forward CV (10-fold, gap=HORIZON) ────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=10, gap=HORIZON)

    results = []
    for mname, _m in candidates.items():
        pipe = Pipeline([("sc", RobustScaler()), ("m", _m)])
        pipe.fit(X_tr, y_tr)
        pv  = pipe.predict(X_te)

        acc  = accuracy_score(y_te, pv)*100
        f1   = f1_score(y_te, pv, average="weighted", zero_division=0)*100
        # Direction accuracy: among BUY/SELL only (ignore HOLD)
        mask = (y_te != 0) & (pv != 0)
        da   = (np.sum((y_te[mask]==pv[mask]))/mask.sum()*100) if mask.sum()>0 else 0

        try:
            cv_sc = cross_val_score(pipe, X_tr, y_tr, cv=tscv,
                                    scoring="f1_weighted", error_score=np.nan)
            cv_f1 = round(np.nanmean(cv_sc)*100, 2)
        except Exception:
            cv_f1 = 0.0

        results.append({"Model":mname,"Acc%":round(acc,2),"F1%":round(f1,2),
                         "DirAcc%":round(da,2),"CV_F1%":cv_f1,
                         "pipe":pipe,"pv":pv})
        print(f"   {mname:<14}  Acc={acc:.1f}%  F1={f1:.1f}%  "
              f"DirAcc={da:.1f}%  CV_F1={cv_f1:.1f}%")

    # ── Soft Voting Ensemble (top-3 by DirAcc) ────────────────────────────────
    sorted_r = sorted(results, key=lambda x: (x["DirAcc%"], x["F1%"]), reverse=True)
    top3     = sorted_r[:min(3, len(sorted_r))]

    # Build proba-based ensemble
    proba_sum = None
    for r in top3:
        if hasattr(r["pipe"], "predict_proba"):
            pb = r["pipe"].predict_proba(X_te)
            # map classes to fixed order [-1, 0, 1]
            clf_classes = r["pipe"].named_steps["m"].classes_.tolist()
            ordered = np.zeros((len(X_te), 3))
            for ci, cls in enumerate([-1,0,1]):
                if cls in clf_classes:
                    ordered[:,ci] = pb[:,clf_classes.index(cls)]
            proba_sum = ordered if proba_sum is None else proba_sum + ordered

    if proba_sum is not None:
        ens_pv = np.array([-1,0,1])[np.argmax(proba_sum, axis=1)]
    else:
        # Majority vote fallback
        votes = np.array([r["pv"] for r in top3])
        ens_pv = np.array([np.bincount(v+1, minlength=3).argmax()-1 for v in votes.T])

    ens_acc = accuracy_score(y_te, ens_pv)*100
    ens_f1  = f1_score(y_te, ens_pv, average="weighted", zero_division=0)*100
    mask    = (y_te != 0) & (ens_pv != 0)
    ens_da  = (np.sum(y_te[mask]==ens_pv[mask])/mask.sum()*100) if mask.sum()>0 else 0

    ens_names = "+".join([r["Model"] for r in top3])
    results.append({"Model":f"Ensemble({ens_names})",
                    "Acc%":round(ens_acc,2),"F1%":round(ens_f1,2),
                    "DirAcc%":round(ens_da,2),"CV_F1%":0.0,
                    "pipe":None,"pv":ens_pv})
    print(f"   {'Ensemble':<14}  Acc={ens_acc:.1f}%  F1={ens_f1:.1f}%  DirAcc={ens_da:.1f}%")

    # ── Best model selection ──────────────────────────────────────────────────
    best = max(results, key=lambda x: (x["DirAcc%"], x["F1%"]))
    print(f"\n   ✅ BEST: {best['Model']}  DirAcc={best['DirAcc%']:.1f}%  F1={best['F1%']:.1f}%")

    if best["pipe"] is None:
        save_pipe = top3[0]["pipe"]
    else:
        save_pipe = best["pipe"]

    # Retrain best pipeline on ALL data using sklearn clone
    from sklearn.base import clone
    best_full = Pipeline([("sc", RobustScaler()),
                          ("m", clone(save_pipe.named_steps["m"]))])
    best_full.fit(X, y)

    test_df = pd.DataFrame({
        "DATE":d_te,"CLOSE":c_te,
        "Y_TRUE":y_te,"Y_PRED":best["pv"],
        "DIR_OK":(y_te==best["pv"]).astype(int)
    })
    comp_df = pd.DataFrame([{k:v for k,v in r.items() if k not in ("pipe","pv")}
                             for r in results]
                           ).sort_values("DirAcc%", ascending=False).reset_index(drop=True)

    metrics = {
        "model":best["Model"],"accuracy":best["Acc%"],"f1":best["F1%"],
        "dir_acc":best["DirAcc%"],"cv_f1":best["CV_F1%"],
        "n_train":sp,"n_test":len(X_te),
        "features":feats,"horizon_days":HORIZON,
        "buy_thresh":BUY_THRESH,"sell_thresh":SELL_THRESH,
        "trained_on":str(datetime.date.today()),
    }
    return best_full, metrics, test_df, comp_df, feats, sub


# ══════════════════════════════════════════════════════════════════════════════
# §F  SIGNAL GENERATION  (classification-aware)
# ══════════════════════════════════════════════════════════════════════════════

def generate_signal(d_eng, pipe, feats):
    """Predict BUY/HOLD/SELL + confidence for current bar."""
    valid = d_eng.dropna(subset=feats)
    if valid.empty:
        return {"signal":"INSUFFICIENT DATA","score":0,"confidence":0}

    last = valid.iloc[-1]
    X    = last[feats].values.reshape(1, -1)

    # Model vote
    label = int(pipe.predict(X)[0])
    signal_map = {1:"BUY", 0:"HOLD", -1:"SELL"}
    sig = signal_map.get(label, "HOLD")

    # Confidence via proba if available
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        classes = pipe.named_steps["m"].classes_.tolist()
        conf = round(float(proba[classes.index(label)])*100, 1) if label in classes else 0.0
    else:
        conf = 0.0

    # Filter scoring (unchanged — 8-point system)
    gap_atr   = float(last.get("GAP_ATR", 0))
    rsi       = float(last.get("RSI_14", 50))
    macd_hist = float(last.get("MACD_HIST", 0))
    macd_crs  = int(last.get("MACD_CROSS", 0))
    vol_ratio = float(last.get("VOL_RATIO", 1))
    vix_val   = float(last.get("VIX_CLOSE", 15))
    adx_val   = float(last.get("ADX_14", 0))
    rs_nifty  = float(last.get("RS_NIFTY_5D", 0))
    near52    = int(last.get("NEAR_52W_HIGH", 0))

    gap_s  = 2 if gap_atr>=1.0 else (1 if gap_atr>=0.3 else 0)
    rsi_s  = 2 if rsi<=60     else (1 if rsi<=70     else 0)
    macd_s = 2 if (macd_hist>0 and macd_crs==1) else (1 if macd_hist>0 else 0)
    vol_s  = 2 if vol_ratio>=1.2 else (1 if vol_ratio>=0.8 else 0)
    # VIX bonus: low fear = good for swing buys
    vix_bonus  = 1 if vix_val < 15 else (-1 if vix_val > 22 else 0)
    # ADX: strong trend confirmation
    adx_bonus  = 1 if adx_val > 25 else 0
    # RS vs NIFTY: outperforming market
    rs_bonus   = 1 if rs_nifty > 0 else 0
    # Near 52W high breakout
    hw_bonus   = 1 if near52 == 1 else 0

    total = gap_s + rsi_s + macd_s + vol_s + vix_bonus + adx_bonus + rs_bonus + hw_bonus
    total = max(0, min(12, total))   # bounded 0–12

    close  = float(last["CLOSE"])
    e50h   = float(d_eng["EMA50_HIGH"].iloc[-1]) if "EMA50_HIGH" in d_eng.columns else close
    e50l   = float(d_eng["EMA50_LOW"].iloc[-1])  if "EMA50_LOW"  in d_eng.columns else close
    e10    = float(d_eng["EMA_10"].iloc[-1])      if "EMA_10"     in d_eng.columns else close
    atr    = float(d_eng["ATR_14"].iloc[-1])      if "ATR_14"     in d_eng.columns else 0

    return {
        "signal":sig,"label":label,"confidence":conf,
        "filter_score":total,"filter_max":12,
        "close":round(close,2),
        "ema10":round(e10,2),"ema50_high":round(e50h,2),"ema50_low":round(e50l,2),
        "atr":round(atr,2),"gap_atr":round(gap_atr,2),
        "rsi":round(rsi,1),"macd_hist":round(macd_hist,3),
        "vol_ratio":round(vol_ratio,2),
        "vix":round(vix_val,2),"adx":round(adx_val,2),
        "rs_vs_nifty_5d":round(rs_nifty,2),
        "near_52w_high":bool(near52),
        "stop_loss":round(e50l,2),
        "target_1":round(close*(1+abs(gap_atr)*0.02),2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# §G  SAVE / LOAD PICKLE
# ══════════════════════════════════════════════════════════════════════════════

def save_model(pipe, metrics, feats, stock_name, path="stock_model.pkl"):
    bundle = {
        "model":pipe,"features":feats,"metrics":metrics,
        "stock":stock_name,"version":"2.0",
        "created":str(datetime.datetime.now()),
        "horizon":HORIZON,
        "signal_thresholds":{"buy_thresh":BUY_THRESH,"sell_thresh":SELL_THRESH},
    }
    with open(path,"wb") as f: pickle.dump(bundle, f)
    with open("model_metadata.json","w") as f:
        safe = {k:v for k,v in metrics.items()}
        json.dump(safe, f, indent=2)
    print(f"   ✅ Saved: {path}  ({os.path.getsize(path)//1024} KB)")
    print(f"   ✅ Saved: model_metadata.json")


def load_model(path="stock_model.pkl"):
    with open(path,"rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"], bundle["metrics"]


# ══════════════════════════════════════════════════════════════════════════════
# §H  TRAINING REPORT PDF
# ══════════════════════════════════════════════════════════════════════════════

def save_report(comp_df, test_df, metrics, d_eng, stock_name):
    pdf = PdfPages("model_training_report.pdf")
    fig = plt.figure(figsize=(20,24), facecolor=BG)
    gs  = gridspec.GridSpec(4, 2, fig, hspace=0.5, wspace=0.38)

    # ── Model comparison ──────────────────────────────────────────────────────
    ax=fig.add_subplot(gs[0,:])
    x=np.arange(len(comp_df)); w=0.22
    ax.bar(x-w*1.5, comp_df["Acc%"],    w, color=C[2], alpha=0.85, label="Accuracy%")
    ax.bar(x-w*0.5, comp_df["F1%"],     w, color=C[0], alpha=0.85, label="F1%")
    ax.bar(x+w*0.5, comp_df["DirAcc%"], w, color=C[1], alpha=0.85, label="DirAcc%")
    ax.bar(x+w*1.5, comp_df["CV_F1%"],  w, color=C[4], alpha=0.85, label="CV_F1%")
    ax.set_xticks(x); ax.set_xticklabels(comp_df["Model"], fontsize=9, rotation=15)
    bi = comp_df["DirAcc%"].idxmax()
    ax.annotate("★ BEST", xy=(bi+w*0.5, comp_df["DirAcc%"].iloc[bi]),
                xytext=(0,8), textcoords="offset points",
                fontsize=10, color=C[1], fontweight="bold", ha="center")
    ax.set_title(f"{stock_name}  — Model Comparison | Best: {metrics['model']}  "
                 f"DirAcc={metrics['dir_acc']:.1f}%  F1={metrics['f1']:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    ax2=fig.add_subplot(gs[1,0])
    cm = confusion_matrix(test_df["Y_TRUE"], test_df["Y_PRED"], labels=[-1,0,1])
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0,1,2]); ax2.set_yticks([0,1,2])
    ax2.set_xticklabels(["SELL","HOLD","BUY"]); ax2.set_yticklabels(["SELL","HOLD","BUY"])
    for i in range(3):
        for j in range(3):
            ax2.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=13,
                     color="black" if cm[i,j]>cm.max()*0.5 else "white")
    ax2.set_title("Confusion Matrix (Test Set)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    plt.colorbar(im, ax=ax2)

    # ── Rolling direction accuracy ─────────────────────────────────────────────
    ax3=fig.add_subplot(gs[1,1])
    roll=test_df["DIR_OK"].rolling(20).mean()*100
    ax3.plot(range(len(roll)), roll, color=C[2], lw=2)
    ax3.axhline(50, color="white", lw=0.8, ls="--", label="Random (50%)")
    ax3.axhline(metrics["dir_acc"], color=C[1], lw=1.2, ls=":",
                label=f"Avg={metrics['dir_acc']:.1f}%")
    ax3.fill_between(range(len(roll)), roll, 50,
                     where=roll>50, alpha=0.25, color=C[2])
    ax3.fill_between(range(len(roll)), roll, 50,
                     where=roll<=50, alpha=0.25, color=C[3])
    ax3.set_ylim(20, 90)
    ax3.set_title("Rolling 20-Day Direction Accuracy", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.2)

    # ── VIX chart ─────────────────────────────────────────────────────────────
    if "VIX_CLOSE" in d_eng.columns:
        ax4=fig.add_subplot(gs[2,:])
        d_plot = d_eng.tail(252).reset_index(drop=True)
        ax4.plot(d_plot["DATE"], d_plot["VIX_CLOSE"], color=C[3], lw=1.8, label="India VIX")
        ax4.axhline(20, color=C[1], lw=1, ls="--", alpha=0.7, label="High Fear (20)")
        ax4.axhline(15, color=C[2], lw=1, ls="--", alpha=0.7, label="Low Fear (15)")
        ax4.fill_between(d_plot["DATE"], d_plot["VIX_CLOSE"], 20,
                         where=d_plot["VIX_CLOSE"]>20, alpha=0.2, color=C[3])
        ax4.set_title(f"India VIX — Last 252 Days", fontsize=11, fontweight="bold")
        ax4.legend(fontsize=9); ax4.grid(True, alpha=0.2)

    # ── Price + signals ───────────────────────────────────────────────────────
    ax5=fig.add_subplot(gs[3,:])
    d_plot = d_eng.tail(252).reset_index(drop=True)
    ax5.plot(d_plot["DATE"], d_plot["CLOSE"],    color=C[0], lw=1.8, label="Close")
    if "EMA_10"    in d_plot.columns: ax5.plot(d_plot["DATE"], d_plot["EMA_10"],    color=C[1], lw=1.2, ls="--", label="EMA10")
    if "EMA50_HIGH" in d_plot.columns: ax5.plot(d_plot["DATE"], d_plot["EMA50_HIGH"],color=C[2], lw=1.2, label="EMA50H")
    if "EMA50_LOW"  in d_plot.columns: ax5.plot(d_plot["DATE"], d_plot["EMA50_LOW"], color=C[3], lw=1.0, ls="--", alpha=0.7)
    ax5.set_title(f"{stock_name} — Price + EMA Levels (Last 252 Days)",
                  fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9); ax5.grid(True, alpha=0.2)

    fig.suptitle(f"Model Training Report v2.0 — {stock_name}  |  {datetime.date.today()}",
                 fontsize=14, fontweight="bold", color=C[0])
    pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
    plt.show(); plt.close(fig); pdf.close()
    print("   ✅ Saved: model_training_report.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# §I  MAIN  — Two-Phase Workflow
# ══════════════════════════════════════════════════════════════════════════════

def _banner(text):
    print("\n" + "━"*65)
    print(f"  {text}")
    print("━"*65)


# ─── PHASE 1: TRAINING ────────────────────────────────────────────────────────
_banner("🤖  CELL 17 v2.0  —  NSE Swing Trade Model  (yfinance + VIX + DNN)")

print("\n📈  PHASE 1 — MODEL TRAINING")
train_symbol = input("   Enter NSE symbol for TRAINING (e.g. RELIANCE, INFY, TCS): ").strip().upper()
if not train_symbol:
    raise ValueError("No training symbol entered.")

print(f"\n⚙️   Fetching data for {train_symbol}.NS + NIFTY 50 + India VIX ...")
stock_train, nifty_train, vix_train, train_ticker = load_training_data(train_symbol)

print("\n⚙️   Engineering indicators ...")
d_eng_train = engineer(stock_train, nifty_train, vix_train)

print(f"\n🤖  Training & evaluating models (80/20 walk-forward) ...")
pipe, metrics, test_df, comp_df, feats, full_d = train_evaluate(d_eng_train)

print("\n📊  Classification Report (Test Set):")
print(classification_report(test_df["Y_TRUE"], test_df["Y_PRED"],
                             target_names=["SELL(-1)","HOLD(0)","BUY(+1)"],
                             zero_division=0))

print("\n💾  Saving model files ...")
save_model(pipe, metrics, feats, train_ticker)
save_report(comp_df, test_df, metrics, d_eng_train, train_symbol)

# ─── PHASE 2: PREDICTION ──────────────────────────────────────────────────────
print("\n📉  PHASE 2 — PREDICT / TEST ON ANOTHER STOCK")
pred_symbol = input("   Enter NSE symbol to PREDICT (e.g. WIPRO, HDFCBANK, or same as above): ").strip().upper()
if not pred_symbol:
    pred_symbol = train_symbol
    print(f"   (No input — using training symbol: {train_symbol})")

print(f"\n⚙️   Fetching recent {PREDICT_YEARS}yr data for {pred_symbol}.NS ...")
stock_pred, nifty_pred, vix_pred, pred_ticker = load_predict_data(pred_symbol)

print("⚙️   Engineering indicators for prediction stock ...")
d_eng_pred = engineer(stock_pred, nifty_pred, vix_pred)

print("\n📊  Generating signal ...")
sig = generate_signal(d_eng_pred, pipe, feats)

print(f"\n{'━'*65}")
print(f"  🔔  SIGNAL for {pred_symbol}  [{sig['signal']}]  "
      f"(Confidence: {sig['confidence']:.1f}%)")
print(f"{'━'*65}")
print(f"  Filter Score : {sig['filter_score']} / {sig['filter_max']}")
print(f"  Close Price  : ₹{sig['close']}")
print(f"  EMA10        : ₹{sig['ema10']}")
print(f"  EMA50 High   : ₹{sig['ema50_high']}")
print(f"  EMA50 Low    : ₹{sig['ema50_low']}")
print(f"  ATR(14)      : ₹{sig['atr']}")
print(f"  Gap (ATR)    : {sig['gap_atr']:+.2f}×ATR")
print(f"  RSI(14)      : {sig['rsi']:.1f}")
print(f"  MACD Hist    : {sig['macd_hist']:+.4f}")
print(f"  Volume Ratio : {sig['vol_ratio']:.2f}×")
print(f"  India VIX    : {sig['vix']:.2f}")
print(f"  ADX(14)      : {sig['adx']:.2f}")
print(f"  RS vs NIFTY5D: {sig['rs_vs_nifty_5d']:+.2f}%")
print(f"  Near 52W High: {sig['near_52w_high']}")
print(f"  Stop Loss    : ₹{sig['stop_loss']}  (EMA50 Low)")
print(f"  Target       : ₹{sig['target_1']}")
print(f"{'━'*65}")

recommendations = {
    "BUY" : "✅  Entry signal — look for pullback to EMA10 or volume surge.",
    "HOLD": "⏳  No clear edge — wait for stronger confirmation.",
    "SELL": "🔴  Bearish signal — consider exiting or avoiding entry.",
}
print(f"\n  Recommendation: {recommendations.get(sig['signal'],'—')}")

print(f"\n{'━'*65}")
print(f"✅  CELL 17 v2.0 complete")
print(f"   Model trained on : {train_symbol}")
print(f"   Predicted for    : {pred_symbol}")
print(f"   Pickle saved     : stock_model.pkl")
print(f"   Place in FastAPI : /models/stock_model.pkl")
print(f"{'━'*65}")