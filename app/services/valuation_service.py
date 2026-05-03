"""
Valuation Service
=================
Core business logic for Stock Valuation Analysis.
"""

from __future__ import annotations
import logging
import time
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import yfinance as yf

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker_sym: str) -> Dict[str, Any]:
    """Fetch raw data from yfinance."""
    ticker_sym = ticker_sym.strip().upper()
    if not ticker_sym.endswith(".NS") and not ticker_sym.endswith(".BO"):
        # Default to NSE for Indian stocks if no suffix
        ticker = yf.Ticker(f"{ticker_sym}.NS")
    else:
        ticker = yf.Ticker(ticker_sym)
    
    info = ticker.info
    if not info or "regularMarketPrice" not in info and "currentPrice" not in info:
        # Try fallback or raise
        if not ticker_sym.endswith(".NS"):
             ticker = yf.Ticker(f"{ticker_sym}.NS")
             info = ticker.info
    
    if not info or ("regularMarketPrice" not in info and "currentPrice" not in info):
        raise ValueError(f"Ticker '{ticker_sym}' not found or no data available.")
        
    return info

# ─────────────────────────────────────────────────────────────────────────────
# Metric Calculations
# ─────────────────────────────────────────────────────────────────────────────

def _r2(v: Any) -> Optional[float]:
    if v is None: return None
    try:
        fv = float(v)
        if not math.isfinite(fv): return None
        return round(fv, 2)
    except: return None

def _metric(name: str, code: str, value: Any, unit: str, verdict: str, ctx: str, raw: Any) -> Dict[str, Any]:
    def _colour(v):
        v = v.lower()
        if "excellent" in v or "undervalued" in v or "good" in v or "low debt" in v: return "green"
        if "expensive" in v or "overvalued" in v or "high" in v: return "red"
        if "fair" in v or "neutral" in v: return "yellow"
        return "grey"

    return {
        "metric_name": name,
        "metric_code": code,
        "value": _r2(value),
        "unit": unit,
        "verdict": verdict,
        "verdict_color": _colour(verdict),
        "threshold_context": ctx,
        "raw_data_used": str(raw),
    }

def calculate_pe(info: Dict[str, Any]) -> Dict[str, Any]:
    val = info.get("trailingPE") or info.get("forwardPE")
    if val is None: return _unavailable("PE Ratio", "pe_ratio", "x")
    
    verdict = "Fair Value"
    ctx = "PE is within normal historical ranges (15-25x)."
    if val > 60:
        verdict = "Highly Overvalued"
        ctx = "Above 60x indicates market is pricing in very high growth. High risk."
    elif val > 35:
        verdict = "Expensive"
        ctx = "Trading at a significant premium to historical averages."
    elif val < 15:
        verdict = "Undervalued"
        ctx = "Low PE suggests the stock might be overlooked or facing headwinds."
        
    return _metric("PE Ratio", "pe_ratio", val, "x", verdict, ctx, val)

def calculate_pb(info: Dict[str, Any]) -> Dict[str, Any]:
    val = info.get("priceToBook")
    if val is None:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        bv = info.get("bookValue")
        if price and bv and bv != 0:
            val = price / bv
    if val is None: return _unavailable("PB Ratio", "pb_ratio", "x")
    
    if val > 10:
        verdict, ctx = "Highly Overvalued", "Paying >10x book value is extremely risky unless ROE is >30%."
    elif val > 5:
        verdict, ctx = "Expensive", "High PB suggests asset-heavy premium."
    elif val < 2:
        verdict, ctx = "Good Value", "Low PB suggests margin of safety on assets."
    else:
        verdict, ctx = "Fair Value", "PB is in the reasonable 2-5x range."
        
    return _metric("PB Ratio", "pb_ratio", val, "x", verdict, ctx, val)

def calculate_ps(info: Dict[str, Any]) -> Dict[str, Any]:
    val = info.get("priceToSalesTrailing12Months")
    if val is None: return _unavailable("PS Ratio", "ps_ratio", "x")
    
    if val > 8:
        verdict, ctx = "Expensive", "Paying >8x revenue requires very high margins to justify."
    elif val < 1.5:
        verdict, ctx = "Undervalued", "Low PS suggests deep value or low margins."
    else:
        verdict, ctx = "Fair Value", "Market standard PS pricing."
    return _metric("PS Ratio", "ps_ratio", val, "x", verdict, ctx, val)

def calculate_ev_ebitda(info: Dict[str, Any]) -> Dict[str, Any]:
    val = info.get("enterpriseToEbitda")
    if val is None: return _unavailable("EV/EBITDA", "ev_ebitda", "x")
    
    if val > 25:
        verdict, ctx = "Highly Overvalued", "Extremely expensive relative to operating cash flow."
    elif val > 15:
        verdict, ctx = "Expensive", "Premium pricing on EBITDA."
    elif val < 8:
        verdict, ctx = "Good Value", "Attractive operating valuation."
    else:
        verdict, ctx = "Fair Value", "Standard 8-15x range."
    return _metric("EV/EBITDA", "ev_ebitda", val, "x", verdict, ctx, val)

def calculate_peg(info: Dict[str, Any]) -> Dict[str, Any]:
    val = info.get("pegRatio") or info.get("trailingPegRatio")
    if val is None:
        pe = info.get("trailingPE") or info.get("forwardPE")
        growth = info.get("earningsGrowth")
        if pe and growth and growth > 0:
            val = pe / (growth * 100)
    if val is None: return _unavailable("PEG Ratio", "peg_ratio", "x")
    
    if val > 2:
        verdict, ctx = "Overvalued", "PE far exceeds growth rate (PEG > 2)."
    elif val < 1:
        verdict, ctx = "Undervalued", "Growth is being bought at a discount (PEG < 1)."
    else:
        verdict, ctx = "Fair Value", "Growth and valuation are in sync."
    return _metric("PEG Ratio", "peg_ratio", val, "x", verdict, ctx, val)

def calculate_dividend_yield(info: Dict[str, Any]) -> Dict[str, Any]:
    val = info.get("dividendYield")
    if val is not None:
        # yfinance dividendYield: values like 0.0041 mean 0.41% (true ratio → multiply by 100)
        # But some versions return 0.41 meaning 0.41% directly.
        # Cross-check: dividendRate / currentPrice gives the true ratio.
        # If val > 0.20 (i.e. >20% as ratio = unrealistic), it's already a percentage.
        # If val < 0.20, it could be a true ratio → multiply by 100.
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        div_rate = info.get("dividendRate")
        if price and div_rate and price > 0:
            # Use actual calculation as ground truth
            val = (div_rate / price) * 100
        elif val < 0.20:
            # Likely a true ratio, convert to percentage
            val = val * 100
        # else: already a percentage, use as-is
    else:
        val = 0.0
    
    if val > 3:
        verdict, ctx = "High Yield", "Strong passive income component (>3%)."
    elif val > 1:
        verdict, ctx = "Moderate Yield", "Healthy dividend payout."
    elif val > 0:
        verdict, ctx = "Low Yield", "Dividends exist but are not the primary return driver."
    else:
        verdict, ctx = "No Yield", "Company does not pay dividends."
    return _metric("Dividend Yield", "dividend_yield", val, "%", verdict, ctx, val)

def calculate_roe(info: Dict[str, Any]) -> Dict[str, Any]:
    val = info.get("returnOnEquity")
    if val is None:
        eps = info.get("trailingEps")
        bv = info.get("bookValue")
        if eps is not None and bv and bv != 0:
            val = eps / bv
    if val is not None: val = val * 100
    if val is None: return _unavailable("ROE", "roe", "%")
    
    if val > 20:
        verdict, ctx = "Excellent", "Outstanding capital efficiency (>20%)."
    elif val > 15:
        verdict, ctx = "Good", "Strong returns on shareholder equity."
    elif val < 10:
        verdict, ctx = "Poor", "Capital efficiency is below cost of capital."
    else:
        verdict, ctx = "Average", "Standard 10-15% ROE."
    return _metric("ROE", "roe", val, "%", verdict, ctx, val)

def calculate_fcf_yield(info: Dict[str, Any]) -> Dict[str, Any]:
    # Fallback Strategy: Operating Cash Flow / Enterprise Value
    fcf = info.get("freeCashflow")
    ocf = info.get("operatingCashflow")
    ev = info.get("enterpriseValue") or info.get("marketCap")
    
    val = None
    if fcf and ev: val = (fcf / ev) * 100
    elif ocf and ev: val = (ocf / ev) * 100
    
    if val is None: return _unavailable("FCF Yield", "fcf_yield", "%")
    
    if val > 8:
        verdict, ctx = "Excellent", "Generates massive cash relative to its size."
    elif val > 4:
        verdict, ctx = "Strong", "Healthy cash generation."
    elif val < 1:
        verdict, ctx = "Very Expensive", "Price far exceeds cash generation ability."
    else:
        verdict, ctx = "Moderate", "Fair cash-to-price ratio."
    return _metric("FCF Yield", "fcf_yield", val, "%", verdict, ctx, val)

def calculate_debt_equity(info: Dict[str, Any]) -> Dict[str, Any]:
    val = info.get("debtToEquity")
    if val is None:
        td = info.get("totalDebt")
        shares = info.get("sharesOutstanding")
        bv = info.get("bookValue")
        if td is not None and shares and bv:
            total_equity = shares * bv
            if total_equity > 0:
                val = (td / total_equity) * 100
    if val is not None: val = val / 100 # yfinance usually gives 0-100+ for D/E
    if val is None: return _unavailable("Debt to Equity", "debt_to_equity", "ratio")
    
    if val > 1.5:
        verdict, ctx = "High Debt", "Significant leverage risk (>1.5)."
    elif val < 0.5:
        verdict, ctx = "Low Debt", "Strong balance sheet, low leverage."
    else:
        verdict, ctx = "Moderate", "Average leverage."
    return _metric("Debt to Equity", "debt_to_equity", val, "ratio", verdict, ctx, val)

def _unavailable(name: str, code: str, unit: str) -> Dict[str, Any]:
    return _metric(name, code, None, unit, "N/A", "Data not available.", "N/A")

# ─────────────────────────────────────────────────────────────────────────────
# Scoring & Logic
# ─────────────────────────────────────────────────────────────────────────────

def calculate_composite_score(metrics: List[Dict[str, Any]], sector: str, industry: str) -> float:
    # Weights based on metric importance
    weights = {
        "pe_ratio": 20,
        "pb_ratio": 10,
        "ps_ratio": 10,
        "ev_ebitda": 15,
        "peg_ratio": 15,
        "roe": 15,
        "fcf_yield": 10,
        "debt_to_equity": 5
    }
    
    total_score = 0
    total_weight = 0
    
    for m in metrics:
        code = m["metric_code"]
        if code not in weights: continue
        
        # Calculate individual metric score (0 to 100)
        # This is a simplified version of the logic
        m_score = 50
        v = m["verdict"].lower()
        if "high" in v and "overvalued" in v: m_score = 0
        elif "excellent" in v or "undervalued" in v: m_score = 100
        elif "expensive" in v or "overvalued" in v: m_score = 25
        elif "good" in v or "strong" in v: m_score = 80
        elif "fair" in v or "average" in v: m_score = 50
        elif "no yield" in v or "poor" in v: m_score = 20
        
        total_score += m_score * weights[code]
        total_weight += weights[code]
        
    if total_weight == 0: return 0
    return round(total_score / total_weight)

def score_label(score: float) -> str:
    if score > 80: return "Deep Value / Strong Buy"
    if score > 60: return "Good Value"
    if score > 40: return "Fair Value"
    if score > 20: return "Expensive / Wait for Correction"
    return "Significantly Overvalued"

def generate_summary(score: float, metrics: List[Dict[str, Any]], name: str, sector: str) -> str:
    label = score_label(score)
    # Find key metrics with safe fallbacks
    pe_val = next((m["value"] for m in metrics if m["metric_code"] == "pe_ratio"), None)
    roe_val = next((m["value"] for m in metrics if m["metric_code"] == "roe"), None)
    pe = f"{pe_val}" if pe_val is not None else "N/A"
    roe = f"{roe_val}" if roe_val is not None else "N/A"
    
    summary = f"{name} scores {score}/100 — \"{label}\". "
    if score < 40:
        summary += f"Trading at {pe}x PE with {roe}% ROE, the stock is currently expensive. "
    else:
        summary += f"Supported by a {roe}% ROE and {pe}x PE, the valuation looks attractive. "
    return summary

def generate_risk_flags(metrics: List[Dict[str, Any]]) -> List[str]:
    flags = []
    for m in metrics:
        if m["verdict_color"] == "red":
            flags.append(f"{m['metric_name']} of {m['value']}{m['unit']} — {m['verdict'].lower()}")
    return flags

def generate_positive_factors(metrics: List[Dict[str, Any]]) -> List[str]:
    factors = []
    for m in metrics:
        if m["verdict_color"] == "green":
            factors.append(f"{m['metric_name']} of {m['value']}{m['unit']} — {m['verdict'].lower()}")
    return factors

def build_metric_summary(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Count how many metrics fall into each verdict category."""
    counts = {
        "undervalued_count": 0,
        "fairly_valued_count": 0,
        "expensive_count": 0,
        "highly_overvalued_count": 0,
        "not_applicable_count": 0,
    }
    for m in metrics:
        v = m["verdict"].lower()
        if m["value"] is None or v == "n/a":
            counts["not_applicable_count"] += 1
        elif "undervalued" in v or "excellent" in v or "good" in v or "strong" in v or "low debt" in v or "high yield" in v:
            counts["undervalued_count"] += 1
        elif "fair" in v or "average" in v or "moderate" in v or "no yield" in v or "low yield" in v:
            counts["fairly_valued_count"] += 1
        elif "highly" in v or "significantly" in v:
            counts["highly_overvalued_count"] += 1
        elif "expensive" in v or "overvalued" in v or "poor" in v or "very expensive" in v or "high debt" in v:
            counts["expensive_count"] += 1
        else:
            counts["fairly_valued_count"] += 1
    return counts

def build_data_availability(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {m["metric_code"]: m["value"] is not None for m in metrics}

def generate_comparison_summary(results: List[Dict[str, Any]]) -> str:
    best = max(results, key=lambda x: x["overall_verdict"]["valuation_score"])
    return f"Based on composite scoring, {best['ticker']} shows the best valuation margin of safety."

# ─────────────────────────────────────────────────────────────────────────────
# PDF Generation with ReportLab
# ─────────────────────────────────────────────────────────────────────────────

def create_valuation_pdf(stocks: List[Dict[str, Any]], filename: str) -> str:
    """Generate a multipage PDF."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, KeepTogether
        from reportlab.lib.units import inch
    except ImportError:
        logger.error("ReportLab not found.")
        return ""

    try:
        # Standard margins for A4 - Large bottom margin to avoid footer conflicts
        doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=70)
        styles = getSampleStyleSheet()
    
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, spaceAfter=4, textColor=colors.HexColor("#FF6700"))
        sub_style = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=12)
        score_style = ParagraphStyle('Score', parent=styles['Italic'], fontSize=14, fontWeight='bold', spaceAfter=8)
        summary_style = ParagraphStyle('Summary', parent=styles['Normal'], fontSize=9, leading=13, spaceAfter=12, textColor=colors.HexColor("#6D7B8D"))
        metric_ctx_style = ParagraphStyle('MetricCtx', parent=styles['Normal'], fontSize=9, leading=11, textColor=colors.HexColor("#25383C"))
    
        elements = []
    
        for i, stock in enumerate(stocks):
            elements.append(Paragraph(f"{stock['ticker']} Valuation Analysis", title_style))
            elements.append(Paragraph(f"{stock['company_name']} • {stock.get('sector', 'N/A')} • {stock.get('industry', 'N/A')}", sub_style))
            
            score = stock['overall_verdict']['valuation_score']
            verdict_label = stock['overall_verdict']['label']
            score_color_hex = "#1a6644" if score > 70 else ("#FF8C00" if score > 40 else "#DC143C")
            
            elements.append(Paragraph(f"Composite Score: <font color='{score_color_hex}'>{score}/100</font> &nbsp;&nbsp;&nbsp; <b>Verdict: <font color='{score_color_hex}'>{verdict_label}</font></b>", score_style))
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(stock['overall_verdict']['summary'], summary_style))
    
            data = [[
                Paragraph("<b>Metric</b>", ParagraphStyle('H', parent=metric_ctx_style, textColor=colors.HexColor("#007BA7"))),
                Paragraph("<b>Value</b>", ParagraphStyle('H', parent=metric_ctx_style, textColor=colors.HexColor("#007BA7"))),
                Paragraph("<b>Verdict</b>", ParagraphStyle('H', parent=metric_ctx_style, textColor=colors.HexColor("#007BA7"))),
                Paragraph("<b>Threshold Context</b>", ParagraphStyle('H', parent=metric_ctx_style, textColor=colors.HexColor("#007BA7")))
            ]]
            
            for m in stock['metrics']:
                val_str = f"{m['value']}{m['unit']}" if m['value'] is not None else "N/A"
                v_color = m.get('verdict_color', 'grey')
                hex_color = "#1a6644" if v_color == 'green' else ("#fbbf24" if v_color == 'yellow' else ("#f87171" if v_color == 'red' else "#64748b"))
                
                data.append([
                    Paragraph(m['metric_name'], metric_ctx_style),
                    Paragraph(f"<b>{val_str}</b>", ParagraphStyle('VVal', parent=metric_ctx_style, textColor=colors.HexColor(hex_color))),
                    Paragraph(f"<b>{m['verdict']}</b>", ParagraphStyle('VVer', parent=metric_ctx_style, textColor=colors.HexColor(hex_color), fontSize=8.5)),
                    Paragraph(m['threshold_context'], metric_ctx_style)
                ])
    
            t = Table(data, colWidths=[1.1*inch, 0.7*inch, 1.5*inch, 3.6*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EBF4FA")),
                ('ALIGN', (0,0), (-1,0), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#334155")), 
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('TOPPADDING', (0,1), (-1,-1), 4),
                ('BOTTOMPADDING', (0,1), (-1,-1), 4),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 0.05*inch))
    
            pos_title = Paragraph("<font color='#1a6644'><b>Positive Factors:</b></font>", ParagraphStyle('P', parent=styles['Normal'], fontSize=10))
            risk_title = Paragraph("<font color='#6b1a1a'><b>Risk Factors:</b></font>", ParagraphStyle('R', parent=styles['Normal'], fontSize=10))
            pos_txt = "<br/>".join([f"• {f}" for f in stock['positive_factors']])
            risk_txt = "<br/>".join([f"• {f}" for f in stock['risk_flags']])
            fact_style = ParagraphStyle('Fact', parent=styles['Normal'], fontSize=9, leading=11, textColor=colors.HexColor("#2C3E50"))
            
            factor_data = [[pos_title, risk_title], [Paragraph(pos_txt, fact_style), Paragraph(risk_txt, fact_style)]]
            ft = Table(factor_data, colWidths=[3.4*inch, 3.4*inch])
            ft.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,0), 0), ('TOPPADDING', (0,1), (-1,1), 4)]))
            elements.append(ft)
            
            if i < len(stocks) - 1:
                elements.append(PageBreak())
    
        def draw_header_footer(canvas, doc):
            canvas.saveState()
            footer_text = f"Analysis generated on {datetime.now().strftime('%Y-%m-%d')} by NSE Signal Terminal."
            canvas.setFont('Helvetica-Oblique', 8)
            canvas.setFillColor(colors.grey)
            # Move footer up to avoid bottomMargin conflict
            canvas.drawString(40, 45, footer_text)
            canvas.setFont('Helvetica-Bold', 8)
            canvas.setFillColor(colors.HexColor("#4a9fd4"))
            canvas.drawString(40, A4[1] - 30, "NSE SIGNAL TERMINAL — FUNDAMENTAL RESEARCH")
            canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
            canvas.setLineWidth(0.5)
            canvas.line(40, A4[1] - 35, A4[0] - 40, A4[1] - 35)
            canvas.restoreState()
    
        doc.build(elements, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
    except Exception as e:
        logger.error(f"Error building PDF: {e}")
        return ""
    
    return filename

def create_master_pdf(tech_data, mom_data, val_data, filename: str) -> str:
    """Generate a comprehensive master report PDF."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.units import inch
        from xml.sax.saxutils import escape
    except ImportError:
        return ""

    try:
        doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        
        main_title = ParagraphStyle('MainTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=20, textColor=colors.HexColor("#4a9fd4"), alignment=1)
        sec_title = ParagraphStyle('SecTitle', parent=styles['Heading2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#334155"))
        table_header = ParagraphStyle('THeader', parent=styles['Normal'], fontSize=8, fontWeight='bold', textColor=colors.white)
        table_cell = ParagraphStyle('TCell', parent=styles['Normal'], fontSize=8)
        
        elements = []
        
        # Cover Page
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("COMPREHENSIVE STOCK ANALYSIS REPORT", main_title))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", ParagraphStyle('Date', parent=styles['Normal'], alignment=1, fontSize=12, textColor=colors.grey)))
        elements.append(Spacer(1, 4*inch))
        elements.append(Paragraph("NSE SIGNAL TERMINAL", ParagraphStyle('Brand', parent=styles['Normal'], alignment=1, fontSize=10, textColor=colors.HexColor("#4a9fd4"))))
        elements.append(PageBreak())
        
        # 1. Technical Analysis Rankings
        elements.append(Paragraph("1. Technical Analysis Rankings", sec_title))
        if not tech_data:
            elements.append(Paragraph("No technical analysis data available.", styles['Normal']))
        else:
            # Sort by Signal Tier -> Strength -> Score
            def tech_sort_key(r):
                sig = r.get('signal', 'AVOID')
                tier = 4 if 'STRONG BUY' in sig else (3 if 'BUY' in sig else (2 if 'WATCH' in sig else 1))
                return (tier, r.get('strength', 0), r.get('score', 0))
            
            tech_data_sorted = sorted(tech_data, key=tech_sort_key, reverse=True)
            
            t_data = [[
                Paragraph("<b>Rank</b>", table_header),
                Paragraph("<b>Stock</b>", table_header),
                Paragraph("<b>Score</b>", table_header),
                Paragraph("<b>Price</b>", table_header),
                Paragraph("<b>Beta</b>", table_header),
                Paragraph("<b>Target</b>", table_header),
                Paragraph("<b>Stop Loss</b>", table_header),
                Paragraph("<b>Signal</b>", table_header)
            ]]
            for i, row in enumerate(tech_data_sorted[:20]): # Limit to top 20
                t_data.append([
                    Paragraph(str(i+1), table_cell),
                    Paragraph(row.get('symbol', 'N/A'), table_cell),
                    Paragraph(str(row.get('score', 'N/A')), table_cell),
                    Paragraph(str(row.get('price', 'N/A')), table_cell),
                    Paragraph(str(row.get('beta', 'N/A')), table_cell),
                    Paragraph(str(row.get('target', 'N/A')), table_cell),
                    Paragraph(str(row.get('stop_loss', 'N/A')), table_cell),
                    Paragraph(row.get('signal', 'N/A'), table_cell)
                ])
            
            t = Table(t_data, colWidths=[0.5*inch, 1.0*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.9*inch, 0.9*inch, 1.2*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(t)
        
        elements.append(PageBreak())
        
        # 2. Momentum Scanner Rankings
        elements.append(Paragraph("2. Momentum Scanner Rankings", sec_title))
        if not mom_data:
            elements.append(Paragraph("No momentum data available.", styles['Normal']))
        else:
            # Sort by Momentum Score
            mom_data_sorted = sorted(mom_data, key=lambda x: x.get('momentum_score', 0), reverse=True)
            
            m_data = [[
                Paragraph("<b>Rank</b>", table_header),
                Paragraph("<b>Stock</b>", table_header),
                Paragraph("<b>Score</b>", table_header),
                Paragraph("<b>Price</b>", table_header),
                Paragraph("<b>1M Ret%</b>", table_header),
                Paragraph("<b>3M Ret%</b>", table_header),
                Paragraph("<b>Vol Trend</b>", table_header)
            ]]
            for i, row in enumerate(mom_data_sorted[:20]):
                m_data.append([
                    Paragraph(str(i+1), table_cell),
                    Paragraph(row.get('symbol', 'N/A'), table_cell),
                    Paragraph(str(row.get('momentum_score', 'N/A')), table_cell),
                    Paragraph(str(row.get('current_close', 'N/A')), table_cell),
                    Paragraph(f"{row.get('return_1m', 'N/A')}%", table_cell),
                    Paragraph(f"{row.get('return_3m', 'N/A')}%", table_cell),
                    Paragraph(f"{row.get('volume_trend', 'N/A')}x", table_cell)
                ])
            
            t = Table(m_data, colWidths=[0.6*inch, 1.2*inch, 0.8*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f172a")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(t)
            
        elements.append(PageBreak())
        
        # 3. Fundamental Valuation Rankings
        elements.append(Paragraph("3. Fundamental Valuation Rankings", sec_title))
        if not val_data:
            elements.append(Paragraph("No valuation data available.", styles['Normal']))
        else:
            # Sort by Valuation Score
            val_data_sorted = sorted(val_data, key=lambda x: x.get('overall_verdict', {}).get('valuation_score', 0), reverse=True)
            
            v_data = [[
                Paragraph("<b>Rank</b>", table_header),
                Paragraph("<b>Symbol</b>", table_header),
                Paragraph("<b>Verdict</b>", table_header),
                Paragraph("<b>Score</b>", table_header),
                Paragraph("<b>PE</b>", table_header),
                Paragraph("<b>ROE%</b>", table_header),
                Paragraph("<b>D/E</b>", table_header)
            ]]
            for i, row in enumerate(val_data_sorted[:20]):
                metrics = row.get('metrics', [])
                pe = next((m.get('value') for m in metrics if m.get('metric_code') == 'pe_ratio'), 'N/A')
                roe = next((m.get('value') for m in metrics if m.get('metric_code') == 'roe'), 'N/A')
                de = next((m.get('value') for m in metrics if m.get('metric_code') == 'debt_to_equity'), 'N/A')
                
                v_data.append([
                    Paragraph(str(i+1), table_cell),
                    Paragraph(row.get('ticker', 'N/A'), table_cell),
                    Paragraph(row.get('overall_verdict', {}).get('label', 'N/A'), table_cell),
                    Paragraph(str(row.get('overall_verdict', {}).get('valuation_score', 'N/A')), table_cell),
                    Paragraph(str(pe), table_cell),
                    Paragraph(str(roe), table_cell),
                    Paragraph(str(de), table_cell)
                ])
            
            t = Table(v_data, colWidths=[0.6*inch, 1*inch, 2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            elements.append(t)
            
        # 4. Detailed Valuation Sheets (Full Detail - One per page)
        if val_data:
            elements.append(PageBreak())
            elements.append(Paragraph("4. Detailed Valuation Analysis", sec_title))
            elements.append(Spacer(1, 0.2*inch))
            
            # Use same styles as single report for consistency
            v_title_style = ParagraphStyle('VTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=4, textColor=colors.HexColor("#FF6700"))
            v_sub_style = ParagraphStyle('VSub', parent=styles['Normal'], fontSize=9, textColor=colors.grey, spaceAfter=12)
            v_score_style = ParagraphStyle('VScore', parent=styles['Italic'], fontSize=14, fontWeight='bold', spaceAfter=8)
            v_summary_style = ParagraphStyle('VSummary', parent=styles['Normal'], fontSize=9, leading=13, spaceAfter=12, textColor=colors.HexColor("#6D7B8D"))
            v_metric_ctx_style = ParagraphStyle('VMetricCtx', parent=styles['Normal'], fontSize=9, leading=11, textColor=colors.HexColor("#25383C"))
            v_fact_style = ParagraphStyle('VFact', parent=styles['Normal'], fontSize=9, leading=11, textColor=colors.HexColor("#2C3E50"))
            
            for i, stock in enumerate(val_data):
                try:
                    # Page break before each stock detailed sheet (except the first one if it follows the section header)
                    if i > 0:
                        elements.append(PageBreak())
                    
                    ticker = stock.get('ticker', 'N/A')
                    name = stock.get('company_name', 'N/A')
                    sector = stock.get('sector', 'N/A')
                    industry = stock.get('industry', 'N/A')
                    overall = stock.get('overall_verdict', {})
                    score = overall.get('valuation_score', 0)
                    label = escape(overall.get('label', 'N/A'))
                    summary = escape(overall.get('summary', 'No valuation summary available.'))
                    
                    elements.append(Paragraph(f"{ticker} Valuation Analysis", v_title_style))
                    elements.append(Paragraph(f"{escape(name)} • {escape(sector)} • {escape(industry)}", v_sub_style))
                    
                    score_color_hex = "#1a6644" if score > 70 else ("#FF8C00" if score > 40 else "#DC143C")
                    elements.append(Paragraph(f"Composite Score: <font color='{score_color_hex}'>{score}/100</font> &nbsp;&nbsp;&nbsp; <b>Verdict: <font color='{score_color_hex}'>{label}</font></b>", v_score_style))
                    elements.append(Paragraph(summary, v_summary_style))
                    
                    # Metrics Table
                    det_data = [[
                        Paragraph("<b>Metric</b>", ParagraphStyle('H', parent=v_metric_ctx_style, textColor=colors.HexColor("#007BA7"))),
                        Paragraph("<b>Value</b>", ParagraphStyle('H', parent=v_metric_ctx_style, textColor=colors.HexColor("#007BA7"))),
                        Paragraph("<b>Verdict</b>", ParagraphStyle('H', parent=v_metric_ctx_style, textColor=colors.HexColor("#007BA7"))),
                        Paragraph("<b>Threshold Context</b>", ParagraphStyle('H', parent=v_metric_ctx_style, textColor=colors.HexColor("#007BA7")))
                    ]]
                    
                    for m in stock.get('metrics', []):
                        val_str = f"{m.get('value', 'N/A')}{m.get('unit', '')}" if m.get('value') is not None else "N/A"
                        v_color = m.get('verdict_color', 'grey')
                        hex_color = "#1a6644" if v_color == 'green' else ("#fbbf24" if v_color == 'yellow' else ("#f87171" if v_color == 'red' else "#64748b"))
                        
                        det_data.append([
                            Paragraph(m.get('metric_name', 'N/A'), v_metric_ctx_style),
                            Paragraph(f"<b>{val_str}</b>", ParagraphStyle('VVal', parent=v_metric_ctx_style, textColor=colors.HexColor(hex_color))),
                            Paragraph(f"<b>{m.get('verdict', 'N/A')}</b>", ParagraphStyle('VVer', parent=v_metric_ctx_style, textColor=colors.HexColor(hex_color), fontSize=8.5)),
                            Paragraph(m.get('threshold_context', 'N/A'), v_metric_ctx_style)
                        ])
                    
                    dt = Table(det_data, colWidths=[1.1*inch, 0.7*inch, 1.5*inch, 3.6*inch])
                    dt.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EBF4FA")),
                        ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#334155")), 
                        ('VALIGN', (0,0), (-1,-1), 'TOP'),
                        ('TOPPADDING', (0,1), (-1,-1), 4),
                        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
                    ]))
                    elements.append(dt)
                    elements.append(Spacer(1, 0.15*inch))
                    
                    # Positive and Risk Factors
                    pos_title = Paragraph("<font color='#1a6644'><b>Positive Factors:</b></font>", ParagraphStyle('P', parent=styles['Normal'], fontSize=10))
                    risk_title = Paragraph("<font color='#6b1a1a'><b>Risk Factors:</b></font>", ParagraphStyle('R', parent=styles['Normal'], fontSize=10))
                    pos_txt = "<br/>".join([f"• {escape(f)}" for f in stock.get('positive_factors', [])])
                    risk_txt = "<br/>".join([f"• {escape(f)}" for f in stock.get('risk_flags', [])])
                    
                    factor_data = [[pos_title, risk_title], [Paragraph(pos_txt, v_fact_style), Paragraph(risk_txt, v_fact_style)]]
                    ft = Table(factor_data, colWidths=[3.4*inch, 3.4*inch])
                    ft.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('LEFTPADDING', (0,0), (-1,-1), 0), ('BOTTOMPADDING', (0,0), (-1,0), 0), ('TOPPADDING', (0,1), (-1,1), 4)]))
                    elements.append(ft)

                except Exception as e:
                    logger.warning(f"Skipping detailed sheet for {stock.get('ticker')}: {e}")
                    continue
                
        def draw_header_footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica-Bold', 8)
            canvas.setFillColor(colors.HexColor("#4a9fd4"))
            canvas.drawString(40, A4[1] - 30, "NSE SIGNAL TERMINAL — MASTER REPORT")
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.grey)
            canvas.drawString(40, 30, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            canvas.drawRightString(A4[0] - 40, 30, f"Page {canvas.getPageNumber()}")
            canvas.restoreState()
            
        doc.build(elements, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)
        time.sleep(0.2) # Small buffer for Windows file handles
        return filename
    except Exception as e:
        logger.error(f"Error building Master PDF: {e}")
        return ""

