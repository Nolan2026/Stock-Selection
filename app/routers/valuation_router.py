"""
Valuation Router
================
FastAPI routes for the Stock Valuation Analysis feature.
Provides endpoints for single-stock analysis and multi-stock comparison.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from datetime import datetime
import logging
import os
import tempfile
from typing import List, Dict, Any

from app.models.valuation_models import (
    ValuationRequest, 
    ValuationResponse, 
    CompareRequest, 
    CompareResponse,
    OverallVerdict,
    MetricSummary,
    DataAvailability
)
from app.services import valuation_service

router = APIRouter(prefix="/api/valuation", tags=["Valuation Analysis"])
logger = logging.getLogger(__name__)

def _perform_valuation(ticker_sym: str) -> Dict[str, Any]:
    """Helper to perform the full valuation logic for a single ticker."""
    try:
        # 1. Fetch raw data
        info = valuation_service.fetch_stock_data(ticker_sym)
        
        # 2. Calculate metrics
        metrics = [
            valuation_service.calculate_pe(info),
            valuation_service.calculate_pb(info),
            valuation_service.calculate_ps(info),
            valuation_service.calculate_ev_ebitda(info),
            valuation_service.calculate_peg(info),
            valuation_service.calculate_dividend_yield(info),
            valuation_service.calculate_roe(info),
            valuation_service.calculate_fcf_yield(info),
            valuation_service.calculate_debt_equity(info),
        ]
        
        sector = info.get("sector")
        industry = info.get("industry")
        
        # 3. Calculate composite score
        score = valuation_service.calculate_composite_score(metrics, sector, industry)
        label = valuation_service.score_label(score)
        summary = valuation_service.generate_summary(score, metrics, info.get("longName", ticker_sym), sector)
        
        # 4. Generate flags and factors
        risk_flags = valuation_service.generate_risk_flags(metrics)
        positive_factors = valuation_service.generate_positive_factors(metrics)
        
        # 5. Build response structure
        return {
            "ticker": ticker_sym,
            "company_name": info.get("longName", ticker_sym),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "sector": sector,
            "industry": industry,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "overall_verdict": {
                "valuation_score": score,
                "label": label,
                "summary": summary
            },
            "metrics": metrics,
            "metric_summary": valuation_service.build_metric_summary(metrics),
            "risk_flags": risk_flags,
            "positive_factors": positive_factors,
            "data_availability": valuation_service.build_data_availability(metrics)
        }
    except ValueError as e:
        logger.warning(f"Ticker not found: {ticker_sym}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing valuation for {ticker_sym}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed for {ticker_sym}: {str(e)}")

@router.post("", response_model=List[ValuationResponse])
async def get_valuation(req: ValuationRequest):
    """
    Perform a metric-by-metric valuation analysis for one or more stock tickers.
    Supports comma-separated strings (e.g. 'TCS, RELIANCE, HDFCBANK').
    """
    results = []
    # Split by comma and clean up
    tickers = [t.strip().upper() for t in req.ticker.split(",") if t.strip()]
    
    if not tickers:
        raise HTTPException(status_code=400, detail="No valid tickers provided.")

    for ticker_sym in tickers:
        try:
            result = _perform_valuation(ticker_sym)
            results.append(result)
        except HTTPException as e:
            # For multiple stocks, skip failures instead of crashing the whole batch
            if len(tickers) == 1:
                raise e
            logger.warning(f"Skipping failed ticker '{ticker_sym}': {e.detail}")
            continue

    if not results:
        raise HTTPException(status_code=400, detail="Could not analyze any of the provided tickers.")

    return results

@router.post("/compare", response_model=CompareResponse)
async def compare_valuations(req: CompareRequest):
    """
    Perform side-by-side valuation comparison for multiple stock tickers.
    """
    results = []
    for ticker in req.tickers:
        ticker_sym = ticker.strip().upper()
        try:
            res = _perform_valuation(ticker_sym)
            results.append(res)
        except HTTPException as e:
            # For comparison, we might want to skip failing tickers or return partial data
            # Here we'll just log and continue if one fails, but you could decide otherwise
            logger.error(f"Comparison: failed to fetch {ticker_sym}: {e.detail}")
            continue
    
    if not results:
        raise HTTPException(status_code=400, detail="No valid tickers found for comparison.")
        
    return {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "stocks": results,
        "comparison_summary": valuation_service.generate_comparison_summary(results)
    }

@router.post("/download_pdf")
async def download_valuation_pdf(req: ValuationRequest):
    """
    Generate and download a PDF report for the provided tickers.
    """
    tickers = [t.strip().upper() for t in req.ticker.split(",") if t.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided.")

    all_data = []
    for ticker_sym in tickers:
        try:
            # We don't use _perform_valuation here because it raises HTTPException
            # which would stop the whole process. We want to skip single failures.
            info = valuation_service.fetch_stock_data(ticker_sym)
            metrics = [
                valuation_service.calculate_pe(info),
                valuation_service.calculate_pb(info),
                valuation_service.calculate_ps(info),
                valuation_service.calculate_ev_ebitda(info),
                valuation_service.calculate_peg(info),
                valuation_service.calculate_dividend_yield(info),
                valuation_service.calculate_roe(info),
                valuation_service.calculate_fcf_yield(info),
                valuation_service.calculate_debt_equity(info),
            ]
            score = valuation_service.calculate_composite_score(metrics, info.get("sector"), info.get("industry"))
            all_data.append({
                "ticker": ticker_sym,
                "company_name": info.get("longName", ticker_sym),
                "sector": info.get("sector") or "N/A",
                "industry": info.get("industry") or "N/A",
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "overall_verdict": {
                    "valuation_score": score,
                    "label": valuation_service.score_label(score),
                    "summary": valuation_service.generate_summary(score, metrics, info.get("longName", ticker_sym), info.get("sector"))
                },
                "metrics": metrics,
                "positive_factors": valuation_service.generate_positive_factors(metrics),
                "risk_flags": valuation_service.generate_risk_flags(metrics)
            })
        except Exception as e:
            logger.warning(f"PDF Gen: Skipping {ticker_sym}: {e}")
            continue

    if not all_data:
        raise HTTPException(status_code=400, detail="No valid data to generate PDF.")

    # Create temp file
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="valuation_report_")
    os.close(fd)

    try:
        pdf_path = valuation_service.create_valuation_pdf(all_data, path)
        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="PDF generation failed.")
        
        return FileResponse(
            path=pdf_path,
            filename=f"Valuation_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            media_type="application/pdf"
        )
    except Exception as e:
        if os.path.exists(path): os.remove(path)
        logger.error(f"PDF Endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
