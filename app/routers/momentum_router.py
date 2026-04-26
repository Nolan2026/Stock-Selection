from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from app.services import momentum_service
import asyncio
import logging
import os
import json
from datetime import datetime

router = APIRouter(prefix="/api/momentum", tags=["Momentum Scanner"])
logger = logging.getLogger(__name__)

class MomentumRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    period: Optional[str] = "1y"

class MomentumResult(BaseModel):
    symbol: str
    sector: str
    current_close: float
    start_close: float
    return_pct: float
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    return_6m: Optional[float] = None
    return_1y: Optional[float] = None
    daily_return: float
    avg_volume: float
    volume_trend: float
    momentum_score: float
    is_above_ema50: bool
    is_above_ema200: bool
    error: Optional[str] = None

@router.get("/master")
async def get_momentum_master():
    """Retrieve index and stock master data."""
    # Use dynamic indices from service
    from app.services.momentum_service import INDEX_URLS
    indices = {k: [] for k in INDEX_URLS.keys()}
    
    master_path = os.path.join("data", "nse_master.json")
    stocks = []
    if os.path.exists(master_path):
        with open(master_path, "r") as f:
            data = json.load(f)
            stocks = data.get("stocks", [])
            
    return {"indices": indices, "stocks": stocks}

@router.get("/index/{index_name}")
async def get_index_constituents(index_name: str):
    """Fetch index constituents dynamically."""
    from app.services.momentum_service import fetch_index_symbols
    symbols = fetch_index_symbols(index_name)
    if not symbols:
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found or empty")
    return {"index": index_name, "symbols": symbols}

@router.post("/scan", response_model=List[MomentumResult])
async def scan_momentum(req: MomentumRequest):
    master = momentum_service.get_master_data()
    stocks_meta = {s["symbol"]: s["sector"] for s in master["stocks"]}
    
    results = []
    
    async def process_stock(symbol):
        symbol = symbol.strip().upper()
        try:
            df = momentum_service.fetch_daily_data(symbol, req.start_date, req.end_date)
            if df is None:
                return MomentumResult(
                    symbol=symbol, sector=stocks_meta.get(symbol, "Unknown"),
                    current_close=0, start_close=0, return_pct=0,
                    avg_volume=0, volume_trend=0, momentum_score=0,
                    is_above_ema50=False, is_above_ema200=False,
                    error="Data fetch failed"
                )
            
            metrics = momentum_service.calculate_momentum(df, period=req.period)
            
            sector = stocks_meta.get(symbol, "Unknown")
            if sector == "Unknown":
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(f"{symbol}.NS")
                    sector = ticker.info.get("sector", "Unknown")
                except:
                    pass

            return MomentumResult(
                symbol=symbol,
                sector=sector,
                **metrics
            )
        except Exception as e:
            return MomentumResult(
                symbol=symbol, sector=stocks_meta.get(symbol, "Unknown"),
                current_close=0, start_close=0, return_pct=0,
                avg_volume=0, volume_trend=0, momentum_score=0,
                is_above_ema50=False, is_above_ema200=False,
                error=str(e)
            )

    # Process in chunks or concurrently with limits
    # For now, simple gather
    results = await asyncio.gather(*[process_stock(s) for s in req.symbols])
    
    return sorted(results, key=lambda x: x.momentum_score, reverse=True)
