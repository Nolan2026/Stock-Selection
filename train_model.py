#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_model.py
Upgrade ML model accuracy by:
1. Fetching 5 years of daily data for 20 diversified NSE stocks (large/mid-cap) to create a ~25,000 row training dataset.
2. Running identical feature engineering to the FastAPI app (42 features).
3. Target: 5-day forward return percentage.
4. Model tournament (LGBMRegressor, XGBRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, HuberRegressor, Ridge).
5. Chronological train/test split (80/20) + 5-fold TimeSeriesSplit with 5-day gap for validation.
6. Selecting the best model by directional accuracy on the test set.
7. Fitting the best model on the complete historical dataset.
8. Saving the model pipeline in the identical pickle format expected by the app.
"""

import os
import sys
import pickle
import datetime
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import yfinance as yf

# Check for LightGBM and XGBoost
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: lightgbm is not installed. LGBMRegressor will be skipped.")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost is not installed. XGBRegressor will be skipped.")

# ──────────────────────────────────────────────────────────────────────────────
# §1 DEFINES & PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "SUNPHARMA.NS", "MARUTI.NS", "TATAMOTORS.NS", "AXISBANK.NS", "ONGC.NS",
    "HCLTECH.NS", "NTPC.NS", "TATASTEEL.NS", "COALINDIA.NS", "KOTAKBANK.NS"
]

HORIZON = 5  # 5-day forward return target

FEATURES = [
    'EMA10_RATIO', 'EMA20_RATIO', 'EMA50_RATIO', 'SMA200_RATIO', 'EMA50H_RATIO', 'EMA50L_RATIO',
    'EMA10_GT_20', 'EMA20_GT_50', 'EMA50_GT_200', 'EMA10_GT_50H',
    'GAP_ATR', 'GAP_PCT', 'CLOSE_GAP_ATR',
    'RSI_14', 'RSI_9',
    'MACD_HIST', 'MACD_CROSS', 'MACD_ABOVE_ZERO',
    'STOCH_K', 'STOCH_D',
    'BB_WIDTH', 'BB_PCT_B', 'ATR_PCT',
    'VOL_5D', 'VOL_20D', 'VOL_RATIO', 'VWAP_DEV',
    'RET_1D', 'RET_2D', 'RET_3D', 'RET_5D', 'RET_10D', 'RET_20D',
    'TREND_5D', 'TREND_10D', 'TREND_20D',
    'ABOVE_EMA50H', 'ABOVE_EMA50L',
    'SLOPE_10D', 'SLOPE_20D',
    'DOW', 'MONTH'
]

# ──────────────────────────────────────────────────────────────────────────────
# §2 IDENTICAL FEATURE ENGINEERING LOGIC
# ──────────────────────────────────────────────────────────────────────────────

def _ema(s, n): 
    return s.ewm(span=n, adjust=False).mean()

def _sma(s, n): 
    return s.rolling(n).mean()

def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _slope(s, w=10):
    out = np.full(len(s), np.nan)
    sv = s.values
    for i in range(w, len(sv)):
        y = sv[i-w:i]
        if not np.any(np.isnan(y)):
            out[i] = np.polyfit(np.arange(w), y, 1)[0]
    return pd.Series(out, index=s.index)

def engineer(raw):
    """Identical to main.py engineer() function."""
    d = raw.copy().reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(d["DATE"]):
        d["DATE"] = pd.to_datetime(d["DATE"])
    C_ = d["CLOSE"]; H_ = d["HIGH"]; L_ = d["LOW"]
    O_ = d["OPEN"]; V_ = d["VOLUME"]
    VW_ = d["VWAP"] if "VWAP" in d.columns else C_

    e10 = _ema(C_, 10); e20 = _ema(C_, 20); e50 = _ema(C_, 50); e200 = _ema(C_, 200)
    e50h = _ema(H_, 50); e50l = _ema(L_, 50)
    sma200 = _sma(C_, 200)

    _tr = pd.concat([H_ - L_, (H_ - C_.shift()).abs(), (L_ - C_.shift()).abs()], axis=1).max(axis=1)
    atr = _tr.rolling(14).mean()

    _mf = _ema(C_, 12); _ms = _ema(C_, 26)
    macd = _mf - _ms; macd_sig = _ema(macd, 9); macd_hist = macd - macd_sig

    rsi14 = _rsi(C_, 14); rsi9 = _rsi(C_, 9)
    _bm = _sma(C_, 20); _bs = C_.rolling(20).std()
    bb_w = ((_bm + 2 * _bs) - (_bm - 2 * _bs)) / _bm
    bb_b = (C_ - (_bm - 2 * _bs)) / (4 * _bs)
    _lo = L_.rolling(14).min(); _hi = H_.rolling(14).max()
    stoch_k = 100 * (C_ - _lo) / (_hi - _lo).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()
    vol_sma20 = _sma(V_, 20)

    # ── EMA ratios ────────────────────────────────────────────────────────────
    d["EMA10_RATIO"] = C_ / e10 - 1;    d["EMA20_RATIO"] = C_ / e20 - 1
    d["EMA50_RATIO"] = C_ / e50 - 1;    d["SMA200_RATIO"] = C_ / sma200 - 1
    d["EMA50H_RATIO"] = C_ / e50h - 1;  d["EMA50L_RATIO"] = C_ / e50l - 1

    # ── EMA alignment ─────────────────────────────────────────────────────────
    d["EMA10_GT_20"] = (e10 > e20).astype(int)
    d["EMA20_GT_50"] = (e20 > e50).astype(int)
    d["EMA50_GT_200"] = (e50 > e200).astype(int)
    d["EMA10_GT_50H"] = (e10 > e50h).astype(int)
    d["EMA_STACK"] = d["EMA10_GT_20"] + d["EMA20_GT_50"] + d["EMA50_GT_200"]

    # ── Gap analysis ──────────────────────────────────────────────────────────
    d["GAP_ATR"] = (e10 - e50h) / atr.replace(0, np.nan)
    d["GAP_PCT"] = (e10 - e50h) / e50h.replace(0, np.nan) * 100
    d["CLOSE_GAP_ATR"] = (C_ - e50h) / atr.replace(0, np.nan)
    gap_raw = d["GAP_ATR"]
    d["GAP_WIDENING"] = gap_raw.diff(5)

    # ── Oscillators ───────────────────────────────────────────────────────────
    d["RSI_14"] = rsi14; d["RSI_9"] = rsi9
    d["MACD_HIST"] = macd_hist
    d["MACD_CROSS"] = (macd > macd_sig).astype(int)
    d["MACD_ABOVE_ZERO"] = (macd > 0).astype(int)
    d["MACD_ACCEL"] = macd_hist.diff()
    d["STOCH_K"] = stoch_k; d["STOCH_D"] = stoch_d
    d["STOCH_RISING"] = (stoch_k > stoch_k.shift(1)).astype(int)
    d["STOCH_CROSS"] = (stoch_k > stoch_d).astype(int)
    d["WILLIAMS_R"] = -100 * ((_hi - C_) / (_hi - _lo).replace(0, np.nan))
    tp = (H_ + L_ + C_) / 3; tp_sma = _sma(tp, 20)
    tp_md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["CCI"] = (tp - tp_sma) / (0.015 * tp_md)

    # ── Volatility ────────────────────────────────────────────────────────────
    d["BB_WIDTH"] = bb_w; d["BB_PCT_B"] = bb_b
    d["ATR_PCT"] = atr / C_ * 100
    _r = C_.pct_change()
    d["VOL_5D"] = _r.rolling(5).std() * 100
    d["VOL_20D"] = _r.rolling(20).std() * 100
    vol20 = _r.rolling(20).std(); vol60 = _r.rolling(60).std()
    d["BETA_PROXY"] = vol20 / vol60.replace(0, np.nan)
    d["BETA_20_60"] = vol20 / vol60.replace(0, np.nan)
    d["BETA_REGIME"] = (vol20 > vol60).astype(int)

    # ── Volume ────────────────────────────────────────────────────────────────
    d["VOL_RATIO"] = V_ / vol_sma20
    d["VWAP_DEV"] = (C_ - VW_) / VW_ * 100
    vol_sma5 = _sma(V_, 5)
    d["VOL_MOMENTUM"] = vol_sma5 / vol_sma20
    obv = (np.sign(C_.diff()).fillna(0) * V_).cumsum()
    d["OBV_ROC"] = obv.pct_change(10) * 100
    d["OBV_TREND"] = (obv > _ema(obv, 20)).astype(int)
    vpt = (C_.pct_change().fillna(0) * V_).cumsum()
    d["VPT_ROC"] = vpt.pct_change(10) * 100
    d["VPT_TREND"] = (vpt > _ema(vpt, 20)).astype(int)

    # ── Returns & momentum ────────────────────────────────────────────────────
    for lg in [1, 2, 3, 5, 10, 20]:
        d[f"RET_{lg}D"] = C_.pct_change(lg) * 100
    d["ROC_5"] = C_.pct_change(5) * 100
    d["ROC_10"] = C_.pct_change(10) * 100
    d["ROC_20"] = C_.pct_change(20) * 100
    d["MOM_SCORE"] = (d["ROC_5"] + d["ROC_10"] + d["ROC_20"]) / 3

    # ── Trend ─────────────────────────────────────────────────────────────────
    d["TREND_5D"] = (C_ > C_.shift(5)).astype(int)
    d["TREND_10D"] = (C_ > C_.shift(10)).astype(int)
    d["TREND_20D"] = (C_ > C_.shift(20)).astype(int)
    d["ABOVE_EMA50H"] = (C_ > e50h).astype(int)
    d["ABOVE_EMA50L"] = (C_ > e50l).astype(int)
    d["SLOPE_10D"] = _slope(C_, 10); d["SLOPE_20D"] = _slope(C_, 20)
    d["HIGHER_HIGH_5"] = (H_ > H_.rolling(5).max().shift(1)).astype(int)
    d["HIGHER_LOW_5"] = (L_ > L_.rolling(5).min().shift(1)).astype(int)
    d["TREND_STRUCT"] = d["HIGHER_HIGH_5"] + d["HIGHER_LOW_5"]
    up_day = (C_ > C_.shift(1)).astype(int); consec = up_day.copy()
    for i in range(1, len(consec)):
        if consec.iloc[i] == 1: 
            consec.iloc[i] = consec.iloc[i-1] + 1
    d["CONSEC_UP"] = consec

    # ── Price position ────────────────────────────────────────────────────────
    hi20 = H_.rolling(20).max(); lo20 = L_.rolling(20).min()
    hi50 = H_.rolling(50).max(); lo50 = L_.rolling(50).min()
    d["CHANNEL_POS_20"] = (C_ - lo20) / (hi20 - lo20).replace(0, np.nan)
    d["CHANNEL_POS_50"] = (C_ - lo50) / (hi50 - lo50).replace(0, np.nan)
    sma20 = _sma(C_, 20); std20 = C_.rolling(20).std()
    sma50 = _sma(C_, 50); std50 = C_.rolling(50).std()
    d["ZSCORE_20"] = (C_ - sma20) / std20.replace(0, np.nan)
    d["ZSCORE_50"] = (C_ - sma50) / std50.replace(0, np.nan)
    d["OVEREXTENDED"] = ((d["ZSCORE_20"].abs() > 2) | (d["ZSCORE_50"].abs() > 2)).astype(int)

    # ── Candle patterns ───────────────────────────────────────────────────────
    body = (C_ - O_).abs(); full_range = (H_ - L_).replace(0, np.nan)
    d["BODY_RATIO"] = body / full_range
    d["UPPER_SHADOW"] = (H_ - pd.concat([C_, O_], axis=1).max(axis=1)) / full_range
    d["LOWER_SHADOW"] = (pd.concat([C_, O_], axis=1).min(axis=1) - L_) / full_range
    d["BULL_CANDLE"] = (C_ > O_).astype(int)
    prev_body = (C_.shift(1) - O_.shift(1)).abs()
    d["BULL_ENGULF"] = ((C_ > O_) & (O_ < C_.shift(1)) & (C_ > O_.shift(1)) & (body > prev_body)).astype(int)

    # ── Return distribution ───────────────────────────────────────────────────
    d["SKEW_20"] = _r.rolling(20).skew()
    d["KURT_20"] = _r.rolling(20).kurt()

    # ── RSI divergence ────────────────────────────────────────────────────────
    price_rising = (C_ > C_.shift(5)).astype(int)
    rsi_falling = (rsi14 < rsi14.shift(5)).astype(int)
    d["RSI_DIVERGE"] = ((price_rising == 1) & (rsi_falling == 1)).astype(int) * -1

    # ── DOW & MONTH ───────────────────────────────────────────────────────────
    d["DOW"] = d["DATE"].dt.dayofweek
    d["MONTH"] = d["DATE"].dt.month

    # Final cleanup to ensure JSON/Imputation compliance
    num_cols = d.select_dtypes(include=[np.number]).columns
    d[num_cols] = d[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return d

# ──────────────────────────────────────────────────────────────────────────────
# §3 DATA RETRIEVAL & CONCATENATION
# ──────────────────────────────────────────────────────────────────────────────

def fetch_and_prepare_data(tickers, period="5y"):
    """Fetch ticker data, engineer features, create target, and concatenate."""
    all_dfs = []
    print(f"\n[+] Fetching {period} daily data for {len(tickers)} stocks...")
    
    for ticker_sym in tickers:
        try:
            print(f"   Fetching {ticker_sym}...", end="", flush=True)
            ticker = yf.Ticker(ticker_sym)
            df = ticker.history(period=period, interval="1d")
            
            if df.empty or len(df) < 300:
                print(" [x] (Too little data, skipping)")
                continue
            
            df = df.reset_index()
            # Find the date column regardless of case
            date_col = next((c for c in df.columns if c.upper() in ["DATE", "DATETIME"]), None)
            if date_col:
                df.rename(columns={date_col: "DATE"}, inplace=True)
            
            df.columns = [c.upper() for c in df.columns]
            required = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
            
            if not all(c in df.columns for c in required):
                print(" [x] (Missing required columns, skipping)")
                continue
                
            df = df[required].copy()
            df["DATE"] = pd.to_datetime(df["DATE"]).dt.tz_localize(None)
            df.sort_values("DATE", inplace=True)
            df.dropna(subset=["CLOSE"], inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            # Feature engineering
            df_eng = engineer(df)
            
            # Create target: 5-day forward return percentage
            # Shifted back by 5 days, so at index t, target is return from t to t+5
            df_eng["TARGET"] = df_eng["CLOSE"].pct_change(HORIZON).shift(-HORIZON) * 100
            
            # Drop last 5 rows because target is NaN (forward return)
            df_eng = df_eng.dropna(subset=["TARGET"]).reset_index(drop=True)
            
            # Safety drop first 200 rows as they have NaNs in rolling features
            df_eng = df_eng.iloc[200:].reset_index(drop=True)
            
            if len(df_eng) > 100:
                all_dfs.append(df_eng)
                print(f" [ok] ({len(df_eng)} rows)")
            else:
                print(" [x] (Too few rows after filtering)")
                
        except Exception as e:
            print(f" [x] (Error: {e})")
            
    if not all_dfs:
        raise ValueError("No tickers successfully fetched and prepared.")
        
    # Combine all stocks
    combined_df = pd.concat(all_dfs, ignore_index=True)
    # Sort chronologically by date to make splits clean
    combined_df.sort_values("DATE", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    
    print(f"\n[+] Total dataset size: {len(combined_df)} rows from {len(all_dfs)} stocks.")
    return combined_df

# ──────────────────────────────────────────────────────────────────────────────
# §4 WALK-FORWARD CROSS-VALIDATION & EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model_tournament(df):
    """Train and evaluate multiple regression models using chronological split."""
    # Split chronologically: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    X_train = train_df[FEATURES].values
    y_train = train_df["TARGET"].values
    
    X_test = test_df[FEATURES].values
    y_test = test_df["TARGET"].values
    
    print(f"   Train set: {len(X_train)} samples ({train_df['DATE'].min().date()} to {train_df['DATE'].max().date()})")
    print(f"   Test set:  {len(X_test)} samples ({test_df['DATE'].min().date()} to {test_df['DATE'].max().date()})")
    
    # 5-fold TimeSeriesSplit with gap to prevent overlap leakage
    tscv = TimeSeriesSplit(n_splits=5, gap=HORIZON)
    
    # Define models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "GradBoost": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, min_samples_leaf=5, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "Huber": HuberRegressor(max_iter=300),
        "Ridge": Ridge(alpha=10.0),
    }
    
    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        
    results = {}
    print("\n[+] Running Model Tournament (Walk-Forward CV on Train, Evaluate on Test)...")
    
    for name, model in models.items():
        print(f"   Training {name}...", end="", flush=True)
        pipeline = Pipeline([("sc", RobustScaler()), ("m", model)])
        
        # Cross-validation on train set
        cv_r2_scores = []
        cv_dir_accs = []
        for train_cv_idx, val_cv_idx in tscv.split(X_train):
            X_tr_cv, y_tr_cv = X_train[train_cv_idx], y_train[train_cv_idx]
            X_val_cv, y_val_cv = X_train[val_cv_idx], y_train[val_cv_idx]
            
            pipeline.fit(X_tr_cv, y_tr_cv)
            val_preds = pipeline.predict(X_val_cv)
            
            cv_r2_scores.append(r2_score(y_val_cv, val_preds))
            
            # Directional accuracy
            da = np.mean(np.sign(val_preds) == np.sign(y_val_cv)) * 100
            cv_dir_accs.append(da)
            
        mean_cv_r2 = np.mean(cv_r2_scores)
        mean_cv_da = np.mean(cv_dir_accs)
        
        # Fit on full training set and evaluate on test set
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        test_mae = mean_absolute_error(y_test, preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        test_r2 = r2_score(y_test, preds)
        test_da = np.mean(np.sign(preds) == np.sign(y_test)) * 100
        
        results[name] = {
            "model_obj": model,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_da": test_da,
            "cv_r2": mean_cv_r2,
            "cv_da": mean_cv_da,
            "preds": preds
        }
        print(f" Finished. Test Dir. Acc: {test_da:.1f}%, Test R2: {test_r2:.3f}")
        
    # Print comparison table
    print("\n[+] Model Performance Comparison:")
    print(f"   {'Model':<14} | {'Test Dir.Acc':<12} | {'Test R2':<8} | {'Test MAE':<8} | {'CV Dir.Acc':<12} | {'CV R2':<8}")
    print(f"   {'-'*14}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*8}")
    for name, res in results.items():
        print(f"   {name:<14} | {res['test_da']:10.1f}% | {res['test_r2']:8.3f} | {res['test_mae']:8.3f} | {res['cv_da']:10.1f}% | {res['cv_r2']:8.3f}")
        
    best_model_name = max(results, key=lambda k: results[k]["test_da"])
    print(f"\n[+] Best Model: {best_model_name} with {results[best_model_name]['test_da']:.1f}% Directional Accuracy.")
    
    return best_model_name, results[best_model_name], X_train, y_train, X_test, y_test

# ──────────────────────────────────────────────────────────────────────────────
# §5 EXPORT & SAVING
# ──────────────────────────────────────────────────────────────────────────────

def save_best_model(best_name, best_res, full_df):
    """Re-fit the best model on all available data and save it in the same pickle format."""
    print(f"\n[+] Fitting best model ({best_name}) on ALL available data...")
    X_all = full_df[FEATURES].values
    y_all = full_df["TARGET"].values
    
    # Instantiate clean model with parameters from best run
    best_model_obj = best_res["model_obj"]
    # We clone it safely
    import copy
    final_model = copy.deepcopy(best_model_obj)
    
    final_pipeline = Pipeline([
        ("sc", RobustScaler()),
        ("m", final_model)
    ])
    
    final_pipeline.fit(X_all, y_all)
    print("   Fitting complete.")
    
    # Prepare model bundle
    bundle = {
        "model": final_pipeline,
        "features": FEATURES,
        "metrics": {
            "model": best_name,
            "mae": float(best_res["test_mae"]),
            "rmse": float(best_res["test_rmse"]),
            "r2": float(best_res["test_r2"]),
            "dir_acc": float(best_res["test_da"]),
            "cv_r2": float(best_res["cv_r2"]),
            "n_train": len(X_all),
            "n_test": len(best_res["preds"]),
            "trained_on": datetime.date.today().isoformat()
        }
    }
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "stock_model.pkl")
    backup_path = os.path.join(models_dir, "stock_model_backup.pkl")
    
    # Backup existing model if it exists
    if os.path.exists(model_path):
        print(f"   Backing up existing model to models/stock_model_backup.pkl...")
        shutil.copyfile(model_path, backup_path)
        
    print(f"   Saving new model to models/stock_model.pkl...")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("[+] Model successfully upgraded and saved.")
    
    # Print relative metrics
    if os.path.exists(backup_path):
        try:
            with open(backup_path, "rb") as f:
                old_bundle = pickle.load(f)
            old_metrics = old_bundle.get("metrics", {})
            print("\n[+] Comparison vs Old Model:")
            print(f"   Metric         | Old Model ({old_metrics.get('model', 'Unknown')}) | New Model ({best_name})")
            print(f"   ---------------|-----------------------------|---------------------------")
            print(f"   Dir. Accuracy  | {old_metrics.get('dir_acc', 0.0):25.1f}% | {best_res['test_da']:23.1f}%")
            print(f"   R2 Score       | {old_metrics.get('r2', 0.0):26.4f} | {best_res['test_r2']:24.4f}")
            print(f"   MAE            | {old_metrics.get('mae', 0.0):26.4f} | {best_res['test_mae']:24.4f}")
            print(f"   Train samples  | {old_metrics.get('n_train', 0):26d} | {len(X_all):24d}")
        except Exception as e:
            print(f"   (Failed to print comparison: {e})")

# ──────────────────────────────────────────────────────────────────────────────
# §6 MAIN FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("[+] UPGRADING STOCK SELECTION MODEL ACCURACY (REGRESSION)")
    print("="*80)
    
    try:
        # Step 1: Fetch and prepare data
        df = fetch_and_prepare_data(TICKERS, period="5y")
        
        # Step 2: Evaluate models
        best_name, best_res, X_train, y_train, X_test, y_test = evaluate_model_tournament(df)
        
        # Step 3: Save the best model
        save_best_model(best_name, best_res, df)
        
    except Exception as e:
        print(f"\n[x] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
