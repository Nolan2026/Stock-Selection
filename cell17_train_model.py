# ══════════════════════════════════════════════════════════════════════════════
# ▌CELL 17  │  TRAIN BEST MODEL → SAVE PICKLE FOR FASTAPI DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE:
#   Trains our full indicator + gap analysis model on your NSE stock data.
#   Runs 5-model comparison on proper train/test split.
#   Saves the BEST model as a pickle with all metadata.
#
# OUTPUT FILES (auto-downloaded):
#   stock_model.pkl          — trained model + scaler pipeline
#   model_metadata.json      — accuracy scores, feature list, thresholds
#   model_training_report.pdf— visual report of training results
#
# The pickle is then placed in the FastAPI project's /models/ folder.
# ══════════════════════════════════════════════════════════════════════════════

import warnings; warnings.filterwarnings("ignore")
import numpy  as np
import pandas as pd
import pickle, json, os, io, datetime
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
from   matplotlib.backends.backend_pdf import PdfPages
from   sklearn.preprocessing  import RobustScaler
from   sklearn.pipeline       import Pipeline
from   sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from   sklearn.linear_model   import HuberRegressor
from   sklearn.model_selection import TimeSeriesSplit, cross_val_score
from   sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score
from   google.colab           import files as _DL

BG="#0a0f1a"; PAN="#0d1525"; GRD="#1a2535"
C=["#4a9fd4","#fbbf24","#34d399","#f87171","#a78bfa","#fb923c","#38bdf8","#e879f9"]
plt.rcParams.update({
    "figure.facecolor":BG,"axes.facecolor":PAN,"axes.edgecolor":GRD,
    "axes.labelcolor":"#c8d6e8","xtick.color":"#c8d6e8","ytick.color":"#c8d6e8",
    "text.color":"#c8d6e8","grid.color":GRD,"legend.facecolor":PAN,
    "legend.edgecolor":GRD,"font.family":"monospace","figure.dpi":130,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _sma(s,n): return s.rolling(n).mean()
def _rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def _slope(s,w=10):
    out=np.full(len(s),np.nan); sv=s.values
    for i in range(w,len(sv)):
        y=sv[i-w:i]
        if not np.any(np.isnan(y)): out[i]=np.polyfit(np.arange(w),y,1)[0]
    return pd.Series(out,index=s.index)


# ══════════════════════════════════════════════════════════════════════════════
# §A  LOAD + CLEAN CSV
# ══════════════════════════════════════════════════════════════════════════════

def load_csv():
    print("📂  Upload NSE stock CSV (3 years recommended)...")
    up  = _DL.upload()
    csv = [k for k in up if k.endswith(".csv")]
    if not csv: raise ValueError("No CSV found")
    r = pd.read_csv(io.BytesIO(up[csv[0]]))
    r.columns = r.columns.str.strip().str.upper()
    r.rename(columns={"LTP":"CLOSE","PREV. CLOSE":"PREV_CLOSE"}, inplace=True)
    r = r.loc[:,~r.columns.duplicated()]
    r["DATE"] = r["DATE"].astype(str).str.strip()
    for fmt in ("%d-%b-%Y","%d-%b-%y","%d/%m/%Y","%Y-%m-%d","%d-%m-%Y"):
        try: r["DATE"]=pd.to_datetime(r["DATE"],format=fmt); break
        except: continue
    r.sort_values("DATE",inplace=True); r.reset_index(drop=True,inplace=True)
    for col in ["OPEN","HIGH","LOW","CLOSE","VOLUME"]:
        if col not in r.columns: continue
        s=r[col]
        if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
        r[col]=pd.to_numeric(s.astype(str).str.replace(",","").str.strip(),errors="coerce")
    r.dropna(subset=["CLOSE"],inplace=True)
    pct=r["CLOSE"].pct_change()
    for si in r.index[pct<-0.35].tolist():
        if si==0: continue
        rat=r.loc[si-1,"CLOSE"]/r.loc[si,"CLOSE"]
        for c in ["OPEN","HIGH","LOW","CLOSE"]:
            if c in r.columns: r.loc[:si-1,c]/=rat
    return r.reset_index(drop=True), csv[0]


# ══════════════════════════════════════════════════════════════════════════════
# §B  FULL INDICATOR ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

# All features the FastAPI model will need — exactly this list
FEATURES = [
    # EMA ratios
    "EMA10_RATIO","EMA20_RATIO","EMA50_RATIO","SMA200_RATIO",
    "EMA50H_RATIO","EMA50L_RATIO",
    # EMA alignment
    "EMA10_GT_20","EMA20_GT_50","EMA50_GT_200","EMA10_GT_50H",
    "EMA_STACK",
    # Gap
    "GAP_ATR","GAP_PCT","CLOSE_GAP_ATR","GAP_WIDENING",
    # Oscillators
    "RSI_14","RSI_9","MACD_HIST","MACD_CROSS","MACD_ABOVE_ZERO","MACD_ACCEL",
    "STOCH_K","STOCH_D","STOCH_RISING","STOCH_CROSS",
    "WILLIAMS_R","CCI",
    # Volatility
    "BB_WIDTH","BB_PCT_B",
    "ATR_PCT","VOL_5D","VOL_20D",
    "BETA_PROXY","BETA_20_60","BETA_REGIME",
    # Volume
    "VOL_RATIO","VWAP_DEV","VOL_MOMENTUM",
    "OBV_ROC","OBV_TREND","VPT_ROC","VPT_TREND",
    # Returns & momentum
    "RET_1D","RET_2D","RET_3D","RET_5D","RET_10D","RET_20D",
    "ROC_5","ROC_10","ROC_20","MOM_SCORE",
    # Trend
    "TREND_5D","TREND_10D","TREND_20D",
    "ABOVE_EMA50H","ABOVE_EMA50L",
    "SLOPE_10D","SLOPE_20D",
    "HIGHER_HIGH_5","HIGHER_LOW_5","TREND_STRUCT",
    "CONSEC_UP",
    # Price position
    "CHANNEL_POS_20","CHANNEL_POS_50",
    "ZSCORE_20","ZSCORE_50","OVEREXTENDED",
    # Candle patterns
    "BODY_RATIO","UPPER_SHADOW","LOWER_SHADOW",
    "BULL_CANDLE","BULL_ENGULF",
    # Return distribution
    "SKEW_20","KURT_20",
    # RSI divergence
    "RSI_DIVERGE",
    # Calendar
    "DOW","MONTH",
]

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

    d["EMA10_RATIO"]=C_/e10-1;    d["EMA20_RATIO"]=C_/e20-1
    d["EMA50_RATIO"]=C_/e50-1;    d["SMA200_RATIO"]=C_/sma200-1
    d["EMA50H_RATIO"]=C_/e50h-1;  d["EMA50L_RATIO"]=C_/e50l-1
    d["EMA10_GT_20"]=(e10>e20).astype(int)
    d["EMA20_GT_50"]=(e20>e50).astype(int)
    d["EMA50_GT_200"]=(e50>e200).astype(int)
    d["EMA10_GT_50H"]=(e10>e50h).astype(int)

    d["GAP_ATR"]=(e10-e50h)/atr;  d["GAP_PCT"]=(e10-e50h)/e50h*100
    d["CLOSE_GAP_ATR"]=(C_-e50h)/atr

    d["RSI_14"]=rsi14; d["RSI_9"]=rsi9
    d["MACD_HIST"]=macd_hist
    d["MACD_CROSS"]=(macd>macd_sig).astype(int)
    d["MACD_ABOVE_ZERO"]=(macd>0).astype(int)
    d["STOCH_K"]=stoch_k; d["STOCH_D"]=stoch_d
    d["BB_WIDTH"]=bb_w; d["BB_PCT_B"]=bb_b
    d["ATR_PCT"]=atr/C_*100
    d["VOL_RATIO"]=V_/vol_sma20
    d["VWAP_DEV"]=(C_-VW_)/VW_*100
    for lg in [1,2,3,5,10,20]:
        d[f"RET_{lg}D"]=C_.pct_change(lg)*100
    _r=C_.pct_change()
    d["VOL_5D"]=_r.rolling(5).std()*100
    d["VOL_20D"]=_r.rolling(20).std()*100
    d["TREND_5D"]=(C_>C_.shift(5)).astype(int)
    d["TREND_10D"]=(C_>C_.shift(10)).astype(int)
    d["TREND_20D"]=(C_>C_.shift(20)).astype(int)
    d["ABOVE_EMA50H"]=(C_>e50h).astype(int)
    d["ABOVE_EMA50L"]=(C_>e50l).astype(int)
    d["SLOPE_10D"]=_slope(C_,10)
    d["SLOPE_20D"]=_slope(C_,20)
    d["DOW"]=d["DATE"].dt.dayofweek
    d["MONTH"]=d["DATE"].dt.month

    # Store key levels for signal generation
    d["EMA_10"]=e10; d["EMA_20"]=e20; d["EMA_50"]=e50
    d["EMA50_HIGH"]=e50h; d["EMA50_LOW"]=e50l
    d["ATR_14"]=atr; d["MACD"]=macd; d["MACD_SIG"]=macd_sig
    return d


# ══════════════════════════════════════════════════════════════════════════════
# §C  TRAIN / TEST SPLIT + MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def train_evaluate(d):
    feats = [f for f in FEATURES if f in d.columns]
    _d = d[feats+["DATE","CLOSE"]].dropna().copy().reset_index(drop=True)
    HORIZON = 5
    _d["TARGET"] = _d["CLOSE"].pct_change(HORIZON).shift(-HORIZON)*100
    _d.dropna(subset=["TARGET"],inplace=True)
    _d.reset_index(drop=True,inplace=True)

    sp = int(len(_d)*0.80)
    X_tr=_d[feats].values[:sp]; y_tr=_d["TARGET"].values[:sp]
    X_te=_d[feats].values[sp:]; y_te=_d["TARGET"].values[sp:]
    d_te=_d["DATE"].values[sp:]; c_te=_d["CLOSE"].values[sp:]

    print(f"   Train: {sp} rows  ({_d['DATE'].iloc[0].date()} → {_d['DATE'].iloc[sp-1].date()})")
    print(f"   Test : {len(X_te)} rows  ({_d['DATE'].iloc[sp].date()} → {_d['DATE'].iloc[-1].date()})")

    # Models
    candidates={}
    try:
        from lightgbm import LGBMRegressor
        candidates["LightGBM"]=LGBMRegressor(n_estimators=600,max_depth=4,
            learning_rate=0.02,num_leaves=31,subsample=0.8,colsample_bytree=0.6,
            min_child_samples=10,reg_alpha=0.1,reg_lambda=1.0,random_state=42,verbose=-1)
    except ImportError: pass
    try:
        from xgboost import XGBRegressor
        candidates["XGBoost"]=XGBRegressor(n_estimators=500,max_depth=4,
            learning_rate=0.02,subsample=0.8,colsample_bytree=0.6,
            reg_alpha=0.1,random_state=42,verbosity=0)
    except ImportError: pass
    candidates["RandomForest"]=RandomForestRegressor(n_estimators=400,max_depth=5,
        min_samples_leaf=5,random_state=42)
    candidates["GradBoost"]=GradientBoostingRegressor(n_estimators=400,max_depth=3,
        learning_rate=0.03,subsample=0.8,random_state=42)
    candidates["HuberRidge"]=HuberRegressor(epsilon=1.35,max_iter=300)

    results=[]
    for mname,_m in candidates.items():
        pipe=Pipeline([("sc",RobustScaler()),("m",_m)])
        pipe.fit(X_tr,y_tr); pv=pipe.predict(X_te)
        mae=mean_absolute_error(y_te,pv)
        rmse=np.sqrt(mean_squared_error(y_te,pv))
        r2=r2_score(y_te,pv)
        da=np.mean(np.sign(y_te)==np.sign(pv))*100
        nz=y_te!=0
        mape=np.mean(np.abs((y_te[nz]-pv[nz])/y_te[nz]))*100 if nz.sum()>0 else np.nan
        # CV on train set
        tscv=TimeSeriesSplit(n_splits=5)
        cv=cross_val_score(pipe,X_tr,y_tr,cv=tscv,scoring="r2",error_score=np.nan)
        results.append({"Model":mname,"MAE":round(mae,4),"RMSE":round(rmse,4),
                        "R2":round(r2,4),"DirAcc%":round(da,2),"MAPE%":round(mape,2),
                        "CV_R2":round(np.nanmean(cv),3),"pipe":pipe,"pv":pv})
        print(f"   {mname:<14}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
              f"R²={r2:.3f}  DirAcc={da:.1f}%  CV_R2={np.nanmean(cv):.3f}")

    # Rank: primary=DirAcc, tiebreak=MAE
    best=max(results,key=lambda x:(x["DirAcc%"],-x["MAE"]))
    print(f"\n   ✅ BEST: {best['Model']}  DirAcc={best['DirAcc%']:.1f}%  MAE={best['MAE']:.4f}")

    # Retrain best on ALL data
    best_full=Pipeline([("sc",RobustScaler()),
                         ("m",type(best["pipe"].named_steps["m"])(
                             **best["pipe"].named_steps["m"].get_params()))])
    best_full.fit(_d[feats].values,_d["TARGET"].values)

    test_df=pd.DataFrame({"DATE":d_te,"CLOSE":c_te,
                          "Y_TRUE":y_te,"Y_PRED":best["pv"],
                          "DIR_OK":(np.sign(y_te)==np.sign(best["pv"])).astype(int)})
    comp_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ("pipe","pv")}
                           for r in results]).sort_values("DirAcc%",ascending=False
                          ).reset_index(drop=True)
    metrics={
        "model":best["Model"],"mae":best["MAE"],"rmse":best["RMSE"],
        "r2":best["R2"],"dir_acc":best["DirAcc%"],"mape":best["MAPE%"],
        "cv_r2":best["CV_R2"],"n_train":sp,"n_test":len(X_te),
        "features":feats,"horizon_days":HORIZON,
        "trained_on":str(datetime.date.today()),
    }
    return best_full,metrics,test_df,comp_df,feats,_d


# ══════════════════════════════════════════════════════════════════════════════
# §D  GENERATE SIGNAL FROM MODEL
# ══════════════════════════════════════════════════════════════════════════════

def generate_signal(d_eng, pipe, feats):
    """Generate GO/WAIT/AVOID signal for current state."""
    last = d_eng.dropna(subset=feats).iloc[-1]
    X    = last[feats].values.reshape(1,-1)
    pred_pct = float(pipe.predict(X)[0]) / 5   # daily equivalent

    gap_atr   = float(last["GAP_ATR"])
    rsi       = float(last["RSI_14"])
    macd_hist = float(last["MACD_HIST"])
    macd_crs  = int(last["MACD_CROSS"])
    vol_ratio = float(last["VOL_RATIO"])
    close     = float(last["CLOSE"])
    e50h      = float(d_eng["EMA50_HIGH"].iloc[-1])
    e50l      = float(d_eng["EMA50_LOW"].iloc[-1])
    e10       = float(d_eng["EMA_10"].iloc[-1])
    atr       = float(d_eng["ATR_14"].iloc[-1])

    # 4-filter scoring
    gap_s  = 2 if gap_atr>=1.0 else (1 if gap_atr>=0.3 else 0)
    rsi_s  = 2 if rsi<=60      else (1 if rsi<=70      else 0)
    macd_s = 2 if (macd_hist>0 and macd_crs==1) else (1 if macd_hist>0 else 0)
    vol_s  = 2 if vol_ratio>=1.2 else (1 if vol_ratio>=0.8 else 0)
    total  = gap_s+rsi_s+macd_s+vol_s

    if   total>=7: sig="STRONG BUY"
    elif total>=5: sig="BUY"
    elif total>=3: sig="WATCH"
    else:          sig="AVOID"

    return {
        "signal":         sig,
        "score":          total,
        "max_score":      8,
        "pred_5d_pct":    round(pred_pct*5, 2),
        "close":          round(close, 2),
        "ema10":          round(e10, 2),
        "ema50_high":     round(e50h, 2),
        "ema50_low":      round(e50l, 2),
        "atr":            round(atr, 2),
        "gap_atr":        round(gap_atr, 2),
        "rsi":            round(rsi, 1),
        "macd_hist":      round(macd_hist, 3),
        "vol_ratio":      round(vol_ratio, 2),
        "gap_signal":     ["AVOID","WAIT","GO"][gap_s],
        "rsi_signal":     ["AVOID","WAIT","GO"][rsi_s],
        "macd_signal":    ["AVOID","WAIT","GO"][macd_s],
        "vol_signal":     ["AVOID","WAIT","GO"][vol_s],
        "stop_loss":      round(e50l, 2),
        "target_1m":      round(close*(1+abs(pred_pct)*20/100), 2),
        "entry_price":    round(close, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# §E  SAVE PICKLE + METADATA
# ══════════════════════════════════════════════════════════════════════════════

def save_model(pipe, metrics, feats, stock_name):
    bundle = {
        "model":     pipe,
        "features":  feats,
        "metrics":   metrics,
        "stock":     stock_name,
        "version":   "1.0",
        "created":   str(datetime.datetime.now()),
        "horizon":   5,
        "signal_thresholds": {
            "gap_go":   1.0,
            "gap_wait": 0.3,
            "rsi_go":   60,
            "rsi_warn": 70,
            "vol_go":   1.2,
        },
    }
    with open("stock_model.pkl","wb") as f:
        pickle.dump(bundle,f)
    with open("model_metadata.json","w") as f:
        safe_metrics = {k:v for k,v in metrics.items() if k!="features"}
        safe_metrics["features"] = feats
        json.dump(safe_metrics, f, indent=2)
    print(f"   ✅ Saved: stock_model.pkl  ({os.path.getsize('stock_model.pkl')//1024} KB)")
    print(f"   ✅ Saved: model_metadata.json")


# ══════════════════════════════════════════════════════════════════════════════
# §F  TRAINING REPORT PDF
# ══════════════════════════════════════════════════════════════════════════════

def save_report(comp_df, test_df, metrics, d_eng, stock_name):
    pdf = PdfPages("model_training_report.pdf")

    fig = plt.figure(figsize=(20,22),facecolor=BG)
    gs  = gridspec.GridSpec(3,2,fig,hspace=0.45,wspace=0.35)

    # Model comparison
    ax=fig.add_subplot(gs[0,:])
    x=np.arange(len(comp_df)); w=0.22
    ax.bar(x-w*1.5, comp_df["DirAcc%"],  w, color=C[2], alpha=0.85, label="DirAcc%")
    ax.bar(x-w*0.5, comp_df["MAE"],      w, color=C[0], alpha=0.85, label="MAE")
    ax.bar(x+w*0.5, comp_df["RMSE"],     w, color=C[1], alpha=0.85, label="RMSE")
    ax.bar(x+w*1.5, comp_df["R2"]+1,     w, color=C[4], alpha=0.85, label="R²+1")
    ax.set_xticks(x); ax.set_xticklabels(comp_df["Model"],fontsize=10)
    best_idx=comp_df["DirAcc%"].idxmax()
    ax.annotate("★ BEST",xy=(best_idx-w*1.5,comp_df["DirAcc%"].iloc[best_idx]),
                xytext=(0,8),textcoords="offset points",fontsize=10,
                color=C[2],fontweight="bold",ha="center")
    ax.set_title(f"{stock_name} — Model Comparison | Best: {metrics['model']} "
                 f"DirAcc={metrics['dir_acc']:.1f}% MAE={metrics['mae']:.4f}",
                 fontsize=12,fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.2)

    # Predicted vs actual
    ax2=fig.add_subplot(gs[1,0])
    ax2.scatter(test_df["Y_TRUE"],test_df["Y_PRED"],
                c=test_df["DIR_OK"],cmap="RdYlGn",s=30,alpha=0.6,
                edgecolors="none")
    mx=max(abs(test_df["Y_TRUE"].max()),abs(test_df["Y_PRED"].max()))
    ax2.plot([-mx,mx],[-mx,mx],color="white",lw=0.8,ls="--")
    ax2.axhline(0,color="white",lw=0.5); ax2.axvline(0,color="white",lw=0.5)
    ax2.set_title(f"Pred vs Actual (test set)\nR²={metrics['r2']:.3f}  DirAcc={metrics['dir_acc']:.1f}%",
                  fontsize=11,fontweight="bold")
    ax2.set_xlabel("Actual 5d return %"); ax2.set_ylabel("Predicted 5d return %")
    ax2.grid(True,alpha=0.2)

    # Direction accuracy over time
    ax3=fig.add_subplot(gs[1,1])
    roll=test_df["DIR_OK"].rolling(20).mean()*100
    ax3.plot(range(len(roll)),roll,color=C[2],lw=2)
    ax3.axhline(50,color="white",lw=0.8,ls="--",label="Random (50%)")
    ax3.axhline(metrics["dir_acc"],color=C[1],lw=1.2,ls=":",
                label=f"Avg={metrics['dir_acc']:.1f}%")
    ax3.fill_between(range(len(roll)),roll,50,
                     where=roll>50,alpha=0.25,color=C[2])
    ax3.fill_between(range(len(roll)),roll,50,
                     where=roll<=50,alpha=0.25,color=C[3])
    ax3.set_ylim(20,80); ax3.set_title("Rolling 20-day Direction Accuracy",
                                        fontsize=11,fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(True,alpha=0.2)

    # Price + signals
    ax4=fig.add_subplot(gs[2,:])
    d_plot=d_eng.tail(252).reset_index(drop=True)
    C_=d_plot["CLOSE"]
    ax4.plot(d_plot["DATE"],C_,color=C[0],lw=1.8,label="Close")
    ax4.plot(d_plot["DATE"],d_plot["EMA_10"],color=C[1],lw=1.2,ls="--",label="EMA10")
    ax4.plot(d_plot["DATE"],d_plot["EMA50_HIGH"],color=C[2],lw=1.2,label="EMA50H")
    ax4.plot(d_plot["DATE"],d_plot["EMA50_LOW"],color=C[3],lw=1.0,ls="--",alpha=0.7)
    cross_up=(d_plot["EMA10_GT_50H"].diff()>0)
    ax4.scatter(d_plot["DATE"][cross_up],d_plot["EMA_10"][cross_up],
                color=C[2],marker="^",s=100,zorder=5,label="▲ EMA10 cross above 50H")
    ax4.set_title(f"{stock_name} — Last 252 Days with Entry Signals",
                  fontsize=11,fontweight="bold")
    ax4.legend(fontsize=9); ax4.grid(True,alpha=0.2)

    fig.suptitle(f"Model Training Report — {stock_name}  |  {datetime.date.today()}",
                 fontsize=14,fontweight="bold",color=C[0])
    pdf.savefig(fig,bbox_inches="tight",facecolor=BG)
    plt.show(); plt.close(fig)
    pdf.close()
    print("   ✅ Saved: model_training_report.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# §G  MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("━"*65)
print("🤖  CELL 17  —  Train Best Model + Save Pickle for FastAPI")
print("━"*65)

raw, filename = load_csv()
stock_name    = os.path.splitext(filename)[0].replace("_"," ").replace("-"," ").title()
print(f"\n✅  {stock_name}  |  {len(raw)} rows  |  "
      f"{raw['DATE'].min().date()} → {raw['DATE'].max().date()}")

print("\n⚙️   Engineering indicators...")
d_eng = engineer(raw)

print("\n🤖  Training & evaluating 5 models (80/20 chronological split)...")
pipe, metrics, test_df, comp_df, feats, full_d = train_evaluate(d_eng)

print("\n📊  Current signal:")
sig = generate_signal(d_eng, pipe, feats)
print(f"   Signal     : {sig['signal']}  (score {sig['score']}/8)")
print(f"   Close      : Rs.{sig['close']}")
print(f"   Gap(ATR)   : {sig['gap_atr']:+.2f}×ATR  ({sig['gap_signal']})")
print(f"   RSI        : {sig['rsi']:.0f}  ({sig['rsi_signal']})")
print(f"   MACD Hist  : {sig['macd_hist']:+.3f}  ({sig['macd_signal']})")
print(f"   Vol Ratio  : {sig['vol_ratio']:.2f}×  ({sig['vol_signal']})")
print(f"   Stop Loss  : Rs.{sig['stop_loss']}  (EMA50_LOW)")

print("\n💾  Saving model files...")
save_model(pipe, metrics, feats, stock_name)
save_report(comp_df, test_df, metrics, d_eng, stock_name)

print(f"\n{'━'*65}")
print("📦  Downloading model files...")
_DL.download("stock_model.pkl")
_DL.download("model_metadata.json")
_DL.download("model_training_report.pdf")
print(f"\n✅  CELL 17 complete")
print(f"   → Place stock_model.pkl in your FastAPI project's /models/ folder")
print(f"   → Then run the FastAPI app with: uvicorn app.main:app --reload")
