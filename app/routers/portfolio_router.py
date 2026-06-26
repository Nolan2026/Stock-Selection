"""
Portfolio Router
================
Endpoints:
  GET  /api/portfolio              → get all holdings
  POST /api/portfolio              → add / update a holding
  DELETE /api/portfolio/{symbol}   → remove a holding
  POST /api/portfolio/analyze      → re-analyze all holdings & compute portfolio metrics
  GET  /api/portfolio/rebalance    → rebalancing suggestions
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import json, os, logging, io
import pandas as pd
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

BASE_DIR       = Path(__file__).parent.parent.parent
PORTFOLIO_PATH = BASE_DIR / "data" / "portfolio.json"
os.makedirs(BASE_DIR / "data", exist_ok=True)


# ── Pydantic models ────────────────────────────────────────────────────────────
class FavoritesSync(BaseModel):
    favorites: List[str]


class HoldingIn(BaseModel):
    symbol:         str
    qty:            float = Field(..., gt=0, description="Number of shares held")
    avg_cost:       float = Field(..., gt=0, description="Average purchase price per share")
    sector:         Optional[str] = "Unknown"
    notes:          Optional[str] = ""


class HoldingUpdate(BaseModel):
    qty:            Optional[float] = None
    avg_cost:       Optional[float] = None
    sector:         Optional[str]   = None
    notes:          Optional[str]   = None


# ── Persistence helpers ────────────────────────────────────────────────────────
def _read() -> dict:
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH) as f:
            return json.load(f)
    return {"holdings": {}, "updated_at": None}


def _write(data: dict):
    data["updated_at"] = datetime.now().isoformat()
    with open(PORTFOLIO_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── CRUD ───────────────────────────────────────────────────────────────────────
@router.get("")
def get_portfolio():
    """Return all holdings from the portfolio store."""
    return _read()


@router.post("")
def upsert_holding(h: HoldingIn):
    """Add or update a stock holding."""
    data = _read()
    sym  = h.symbol.strip().upper()
    data["holdings"][sym] = {
        "symbol":    sym,
        "qty":       h.qty,
        "avg_cost":  h.avg_cost,
        "sector":    h.sector or "Unknown",
        "notes":     h.notes or "",
        "favorite":  data["holdings"].get(sym, {}).get("favorite", False),
        "added_at":  data["holdings"].get(sym, {}).get("added_at", datetime.now().isoformat()),
    }
    _write(data)
    return {"ok": True, "holding": data["holdings"][sym]}


@router.post("/{symbol}/toggle_favorite")
def toggle_favorite(symbol: str):
    """Toggle the favorite status of a stock holding."""
    data = _read()
    sym  = symbol.strip().upper()
    if sym not in data["holdings"]:
        raise HTTPException(404, f"{sym} not in portfolio")
    fav = data["holdings"][sym].get("favorite", False)
    data["holdings"][sym]["favorite"] = not fav
    _write(data)
    return {"ok": True, "symbol": sym, "favorite": not fav}


@router.delete("/clear")
def clear_portfolio():
    """Clear all holdings from the portfolio."""
    data = _read()
    data["holdings"] = {}
    _write(data)
    return {"ok": True, "message": "All holdings cleared"}


@router.delete("/{symbol}")
def remove_holding(symbol: str):
    """Remove a holding from the portfolio."""
    data = _read()
    sym  = symbol.strip().upper()
    if sym not in data["holdings"]:
        raise HTTPException(404, f"{sym} not in portfolio")
    del data["holdings"][sym]
    _write(data)
    return {"ok": True, "removed": sym}


@router.post("/upload_excel")
async def upload_portfolio_excel(file: UploadFile = File(...)):
    """
    Import holdings from an Excel or CSV file.
    Expected columns: Symbol, Qty, Entry Price (or Avg Cost), [Sector]
    """
    content = await file.read()
    try:
        # 1. Read based on extension
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            # Requires openpyxl for .xlsx
            df = pd.read_excel(io.BytesIO(content))
        
        if df.empty:
            raise HTTPException(400, "The uploaded file is empty.")

        # 2. Normalize columns (lowercase & stripped)
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # 3. Flexible column mapping
        col_map = {
            'symbol': next((c for c in df.columns if 'symbol' in c or 'ticker' in c or 'stock' in c), None),
            'qty': next((c for c in df.columns if 'qty' in c or 'quantity' in c or 'shares' in c or 'vol' in c), None),
            'avg_cost': next((c for c in df.columns if 'cost' in c or 'price' in c or 'avg' in c or 'buy' in c), None),
            'sector': next((c for c in df.columns if 'sector' in c or 'indus' in c), None)
        }
        
        if not col_map['symbol'] or not col_map['qty'] or not col_map['avg_cost']:
            missing = []
            if not col_map['symbol']: missing.append("Symbol")
            if not col_map['qty']: missing.append("Qty")
            if not col_map['avg_cost']: missing.append("Entry Price")
            raise HTTPException(400, f"Missing required columns: {', '.join(missing)}")
        
        # 4. Process rows
        portfolio_data = _read()
        portfolio_data["holdings"] = {}  # Clear existing holdings for clean import
        import_count = 0
        
        for _, row in df.iterrows():
            try:
                sym = str(row[col_map['symbol']]).strip().upper()
                if not sym or sym == 'NAN' or sym == 'NONE': continue
                
                qty = float(row[col_map['qty']])
                avg_cost = float(row[col_map['avg_cost']])
                
                # Basic validation
                if qty <= 0 or avg_cost <= 0: continue
                
                sector = "Unknown"
                if col_map['sector'] and not pd.isna(row[col_map['sector']]):
                    sector = str(row[col_map['sector']]).strip()
                
                portfolio_data["holdings"][sym] = {
                    "symbol":    sym,
                    "qty":       qty,
                    "avg_cost":  avg_cost,
                    "sector":    sector,
                    "notes":     f"Imported from {file.filename} on {datetime.now().strftime('%Y-%m-%d')}",
                    "favorite":  portfolio_data["holdings"].get(sym, {}).get("favorite", False),
                    "added_at":  portfolio_data["holdings"].get(sym, {}).get("added_at", datetime.now().isoformat()),
                }
                import_count += 1
            except (ValueError, TypeError):
                continue # Skip invalid rows
        
        if import_count == 0:
            raise HTTPException(400, "No valid holdings were found in the file.")
            
        # 5. Persist
        portfolio_data["name"] = f"Imported: {file.filename}"
        _write(portfolio_data)
        
        return {
            "ok": True, 
            "count": import_count, 
            "message": f"Successfully imported {import_count} holdings from {file.filename}"
        }
        
    except Exception as e:
        logger.error(f"Excel import failed: {str(e)}")
        if isinstance(e, HTTPException): raise e
        raise HTTPException(400, "Failed to process file. Check structure and content.")


# ── Reusable Portfolio analysis helper ─────────────────────────────────────────
def run_holdings_analysis(holdings: dict):
    from app.main import fetch_yahoo, engineer, generate_signal, get_model

    if not holdings:
        return {"results": [], "portfolio_metrics": {}, "rebalance_flags": []}

    results, errors = [], []
    sector_exposure = {}   # sector → current_value
    total_value     = 0.0
    betas           = []

    for sym, h in holdings.items():
        try:
            df, _ticker  = fetch_yahoo(sym, period="3y")
            d_eng        = engineer(df)
            sig_result   = generate_signal(d_eng, sym)

            # Fetch the actual unadjusted current closing price from yfinance fast_info/info
            import yfinance as yf
            ticker_obj = yf.Ticker(_ticker)
            current_price = None
            try:
                current_price = float(ticker_obj.fast_info['last_price'])
            except Exception as fe:
                logger.warning(f"Failed to fetch fast_info['last_price'] for {sym}: {fe}")
                try:
                    info = ticker_obj.info
                    current_price = float(info.get('currentPrice') or info.get('regularMarketPrice'))
                except Exception as ie:
                    logger.warning(f"Failed to fetch info price for {sym}: {ie}")

            if current_price is None or pd.isna(current_price) or current_price <= 0:
                current_price = sig_result["close"]

            cost_basis    = h["avg_cost"]
            qty           = h["qty"]
            current_val   = current_price * qty
            pnl_pct       = round((current_price - cost_basis) / cost_basis * 100, 2)
            pnl_abs       = round((current_price - cost_basis) * qty, 2)

            # Today's P&L calculation using official closing price from history
            close_yesterday = df["CLOSE"].iloc[-2] if len(df) >= 2 else current_price

            today_change = current_price - close_yesterday
            today_pnl_abs = round(today_change * qty, 2)
            today_pnl_pct = round((today_change / close_yesterday * 100), 2) if close_yesterday > 0 else 0.0

            # Beta from signal result
            beta_val = sig_result.get("beta") or 1.0

            sector = h.get("sector", "Unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + current_val
            total_value += current_val
            betas.append((beta_val, current_val))

            # Margin of safety (how far above stop loss)
            stop   = sig_result["stop_loss"]
            mos    = round((current_price - stop) / current_price * 100, 2) if current_price > 0 else 0

            results.append({
                **sig_result,
                # Portfolio-specific fields
                "qty":           qty,
                "avg_cost":      cost_basis,
                "current_price": current_price,
                "current_value": current_val,
                "pnl_pct":       pnl_pct,
                "pnl_abs":       pnl_abs,
                "today_pnl_abs": today_pnl_abs,
                "today_pnl_pct": today_pnl_pct,
                "margin_of_safety": mos,
                "sector":        sector,
                "notes":         h.get("notes", ""),
                "favorite":      h.get("favorite", False),
                # Rebalance flag
                "rebalance_flag": _rebalance_flag(sig_result["signal"]),
            })
        except Exception as e:
            logger.error(f"Portfolio analyze error for {sym}: {e}")
            errors.append({"symbol": sym, "error": str(e)})

    # ── Portfolio-level metrics ────────────────────────────────────────────────
    # Weighted Beta
    weighted_beta = 0.0
    if total_value > 0:
        weighted_beta = sum(b * v for b, v in betas) / total_value

    # Sector weights (%)
    sector_weights = {k: round(v / total_value * 100, 1)
                      for k, v in sector_exposure.items()} if total_value > 0 else {}

    # Aggregate margin of safety (value-weighted)
    agg_mos = 0.0
    if total_value > 0:
        for r in results:
            w = r["current_value"] / total_value
            agg_mos += w * r["margin_of_safety"]

    total_today_pnl_abs = sum(r["today_pnl_abs"] for r in results)
    prev_total_value = total_value - total_today_pnl_abs
    total_today_pnl_pct = round(total_today_pnl_abs / max(prev_total_value, 1) * 100, 2) if prev_total_value > 0 else 0.0

    portfolio_metrics = {
        "total_value":      round(total_value, 2),
        "total_pnl_abs":    round(sum(r["pnl_abs"] for r in results), 2),
        "total_cost_basis": round(sum(r["avg_cost"] * r["qty"] for r in results), 2),
        "overall_pnl_pct":  round(
            sum(r["pnl_abs"] for r in results) /
            max(sum(r["avg_cost"] * r["qty"] for r in results), 1) * 100, 2
        ),
        "total_today_pnl_abs": round(total_today_pnl_abs, 2),
        "total_today_pnl_pct": total_today_pnl_pct,
        "weighted_beta":    round(weighted_beta, 3),
        "beta_regime":      "HIGH" if weighted_beta > 1.2 else
                            "NORMAL" if weighted_beta > 0.8 else "LOW",
        "sector_exposure":  sector_weights,
        "agg_margin_of_safety": round(agg_mos, 2),
        "holdings_count":   len(results),
        "analyzed_at":      datetime.now().isoformat(),
    }

    # ── Rebalancing suggestions ────────────────────────────────────────────────
    rebalance_flags = []
    avoid_stocks    = [r for r in results if r["signal"] == "AVOID"]
    buy_stocks      = [r for r in results if r["signal"] in ("STRONG BUY", "DEEP VALUE", "BUY")
                       and r["pnl_pct"] > -5]   # not already in deep loss

    for av in avoid_stocks:
        suggestion = {
            "action":    "ROTATE OUT",
            "symbol":    av["symbol"],
            "reason":    f"Signal dropped to AVOID (score {av['score']}/8)",
            "pnl_pct":   av["pnl_pct"],
            "rotate_into": [b["symbol"] for b in buy_stocks[:3]],
        }
        rebalance_flags.append(suggestion)

    return {
        "results":           results,
        "portfolio_metrics": portfolio_metrics,
        "rebalance_flags":   rebalance_flags,
        "errors":            errors,
    }


# ── Portfolio analysis (re-uses main.py helpers via import) ───────────────────
@router.post("/analyze")
async def analyze_portfolio():
    """
    Re-fetch live data for every holding, run the signal engine,
    and return enriched metrics + portfolio-level risk summary.
    """
    data = _read()
    res = run_holdings_analysis(data["holdings"])
    res["name"] = data.get("name")
    return res


@router.post("/analyze_custom")
async def analyze_custom_portfolio(holdings: dict):
    """
    Analyze a custom portfolio payload (e.g. from local storage saved portfolios)
    without affecting the active portfolio holdings list.
    """
    return run_holdings_analysis(holdings)


@router.post("/download_pdf")
async def download_portfolio_pdf(background_tasks: BackgroundTasks):
    """Generate and download the portfolio report as a colored PDF."""
    from app.services.portfolio_report_service import create_portfolio_report
    
    # 1. Get fresh analysis
    try:
        data = await analyze_portfolio()
        if not data["results"]:
            raise HTTPException(400, "Portfolio is empty. Add holdings first.")
        
        # 2. Create PDF
        pdf_path = create_portfolio_report(data, format="pdf")
        
        # 3. Clean up later
        background_tasks.add_task(lambda p: os.remove(p) if os.path.exists(p) else None, pdf_path)
        
        return FileResponse(
            path=pdf_path,
            filename=f"NSE_Portfolio_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            media_type="application/pdf"
        )
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise HTTPException(500, "Internal server error during PDF generation.")


@router.post("/download_jpg")
async def download_portfolio_jpg(background_tasks: BackgroundTasks):
    """Generate and download the portfolio report as a JPG image."""
    from app.services.portfolio_report_service import create_portfolio_report
    
    # 1. Get fresh analysis
    try:
        data = await analyze_portfolio()
        if not data["results"]:
            raise HTTPException(400, "Portfolio is empty. Add holdings first.")
        
        # 2. Create JPG
        jpg_path = create_portfolio_report(data, format="jpg")
        
        # 3. Clean up later
        background_tasks.add_task(lambda p: os.remove(p) if os.path.exists(p) else None, jpg_path)
        
        return FileResponse(
            path=jpg_path,
            filename=f"NSE_Portfolio_Report_{datetime.now().strftime('%Y%m%d')}.jpg",
            media_type="image/jpeg"
        )
    except Exception as e:
        logger.error(f"JPG download failed: {str(e)}")
        raise HTTPException(500, "Internal server error during JPG generation.")


@router.post("/load")
def load_portfolio(payload: dict):
    """Overwrite the active portfolio holdings with the provided ones."""
    if "holdings" in payload and isinstance(payload["holdings"], dict):
        name = payload.get("name")
        holdings = payload["holdings"]
    else:
        name = None
        holdings = payload
    data = {"name": name, "holdings": holdings, "updated_at": datetime.now().isoformat()}
    _write(data)
    return {"ok": True}


@router.post("/sync_favorites")
def sync_favorites(payload: FavoritesSync):
    """Synchronize favorite status with a list of favorite symbols."""
    data = _read()
    fav_set = {sym.strip().upper() for sym in payload.favorites}
    for sym, h in data["holdings"].items():
        h["favorite"] = (sym in fav_set)
    _write(data)
    return {"ok": True, "count": len(fav_set)}


def _rebalance_flag(signal: str) -> str:
    if signal == "AVOID":
        return "ROTATE_OUT"
    if signal in ("STRONG BUY",):
        return "HOLD_OR_ADD"
    if signal == "BUY":
        return "HOLD"
    if signal == "WATCH":
        return "MONITOR"
    return "NEUTRAL"
