"""
Valuation Models
================
Pydantic models for the Stock Valuation Analysis feature.
Defines request/response schemas for single-stock and comparison endpoints.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────


class ValuationRequest(BaseModel):
    """Request body for single-stock valuation analysis."""
    ticker: str = Field(
        ...,
        description="Stock ticker symbol(s), e.g. 'RELIANCE' or comma-separated 'TCS, RELIANCE, HDFCBANK'",
        min_length=1,
        max_length=200,
    )


class CompareRequest(BaseModel):
    """Request body for side-by-side stock comparison."""
    tickers: List[str] = Field(
        ...,
        description="List of 2–5 ticker symbols to compare",
        min_length=2,
        max_length=5,
    )


class MasterReportRequest(BaseModel):
    """Request body for comprehensive master report generation."""
    tech_data: List[Dict[str, Any]]
    momentum_data: List[Dict[str, Any]]
    valuation_data: List[Dict[str, Any]]



# ── Metric Result ────────────────────────────────────────────────────────────


class MetricResult(BaseModel):
    """
    Result for a single valuation metric.

    Each metric carries its computed value, a human-readable verdict,
    colour coding for UI display, and contextual explanation.
    """
    metric_name: str = Field(..., description="Human-readable name, e.g. 'PE Ratio'")
    metric_code: str = Field(..., description="Machine key, e.g. 'pe_ratio'")
    value: Optional[float] = Field(None, description="Computed metric value (null if unavailable)")
    unit: str = Field("x", description="Display unit — 'x', '%', or 'ratio'")
    verdict: str = Field(..., description="Qualitative label, e.g. 'Undervalued'")
    verdict_color: str = Field("grey", description="UI colour: dark_green/green/yellow/orange/red/grey")
    threshold_context: str = Field("", description="Explanation of what the value means in context")
    raw_data_used: str = Field("", description="Which yfinance fields were consumed")


# ── Overall Verdict ──────────────────────────────────────────────────────────


class OverallVerdict(BaseModel):
    """Composite valuation score and qualitative summary."""
    valuation_score: float = Field(..., ge=0, le=100)
    label: str
    summary: str


# ── Metric Summary Counts ────────────────────────────────────────────────────


class MetricSummary(BaseModel):
    """Quick tally of how many metrics fall into each bucket."""
    undervalued_count: int = 0
    fairly_valued_count: int = 0
    expensive_count: int = 0
    highly_overvalued_count: int = 0
    not_applicable_count: int = 0


# ── Data Availability ────────────────────────────────────────────────────────


class DataAvailability(BaseModel):
    """Flags indicating which metrics had valid source data."""
    pe_ratio: bool = False
    pb_ratio: bool = False
    ps_ratio: bool = False
    ev_ebitda: bool = False
    peg_ratio: bool = False
    dividend_yield: bool = False
    roe: bool = False
    fcf_yield: bool = False
    debt_to_equity: bool = False


# ── Full Response ─────────────────────────────────────────────────────────────


class ValuationResponse(BaseModel):
    """
    Complete valuation analysis response for a single stock.

    Contains the composite score, per-metric breakdowns,
    risk flags, positive factors, and data-availability map.
    """
    ticker: str
    company_name: str
    current_price: Optional[float]
    sector: Optional[str]
    industry: Optional[str] = None
    analysis_date: str

    overall_verdict: OverallVerdict
    metrics: List[MetricResult]
    metric_summary: MetricSummary
    risk_flags: List[str]
    positive_factors: List[str]
    data_availability: DataAvailability


# ── Compare Response ─────────────────────────────────────────────────────────


class CompareResponse(BaseModel):
    """Side-by-side valuation comparison for multiple stocks."""
    analysis_date: str
    stocks: List[ValuationResponse]
    comparison_summary: str = ""
