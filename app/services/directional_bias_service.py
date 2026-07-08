"""
Directional Bias Service
========================
Computes a directional-bias score and recommendation for each stock that
has already been enriched by the FNO pipeline (technicals + option_chain).

Key design decisions:
  - **Reuses FNO pipeline data** — no redundant yfinance calls.
  - **EMA bias** uses the same EMA10 / EMA50H / EMA50L signals the user
    sees in the FNO table (not EMA20/50/200 which are a different indicator).
  - **OI signal** comes from the option-chain data already on the stock dict,
    with explicit "OI_DATA_MISSING" when no OI data is available.
  - **Debug mode** logs raw inputs → final score for every ticker.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Module-level debug flag (toggled by the router) ──────────────────────────
DEBUG_MODE: bool = False


# ═════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def classify_ema_bias(tech: dict) -> tuple[str, int]:
    """Classify EMA bias from the FNO pipeline's EMA10/EMA50H/EMA50L flags.

    Returns:
        (bias_label, ema_pass_count)
        bias_label  : "Bullish" | "Partial Bullish" | "Neutral" | "Bearish"
        ema_pass_count : 0-3  (how many of the 3 EMA conditions pass)
    """
    ema10_pass = bool(tech.get("ema10_pass", False))
    ema50h_pass = bool(tech.get("ema50h_pass", False))
    ema50l_pass = bool(tech.get("ema50l_pass", False))

    pass_count = int(ema10_pass) + int(ema50h_pass) + int(ema50l_pass)

    if pass_count == 3:
        return "Bullish", pass_count
    elif pass_count == 2:
        return "Partial Bullish", pass_count
    elif pass_count == 1:
        return "Neutral", pass_count
    else:
        return "Bearish", pass_count


def classify_oi_signal(
    price_change_pct: float,
    oi_change_pct: float,
    oi_data_available: bool,
) -> tuple[str, str]:
    """Classify OI signal from price change and OI change.

    Returns:
        (oi_signal, oi_data_source)
        oi_signal      : "Long Buildup" | "Short Buildup" | "Short Covering"
                         | "Long Unwinding" | "Neutral" | "OI_DATA_MISSING"
        oi_data_source : "PROXY" | "MISSING"
    """
    if not oi_data_available:
        return "OI_DATA_MISSING", "MISSING"

    # All current OI data comes from the yfinance volume proxy
    data_source = "PROXY"

    if price_change_pct > 0 and oi_change_pct > 0:
        return "Long Buildup", data_source
    elif price_change_pct < 0 and oi_change_pct > 0:
        return "Short Buildup", data_source
    elif price_change_pct > 0 and oi_change_pct < 0:
        return "Short Covering", data_source
    elif price_change_pct < 0 and oi_change_pct < 0:
        return "Long Unwinding", data_source
    return "Neutral", data_source


# ═════════════════════════════════════════════════════════════════════════════
#  SCORE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_directional_score(
    ema_bias: str,
    ema_pass_count: int,
    oi_signal: str,
    momentum_confirm: bool,
    momentum_score_pct: float,
    price_change_pct: float,
) -> int:
    """Compute a composite directional score (0-100).

    Weighting:
      - EMA bias:  up to ±40 points (scaled by pass count)
      - OI signal: up to ±35 points
      - Momentum:  up to ±25 points

    Returns directional_score in [0, 100].
    """
    # ── EMA component (max ±40) ──────────────────────────────────────────────
    if ema_bias == "Bullish":
        ema_score = 40
    elif ema_bias == "Partial Bullish":
        ema_score = 25
    elif ema_bias == "Neutral":
        ema_score = 0
    elif ema_bias == "Bearish":
        ema_score = -40
    else:
        ema_score = 0

    # ── OI component (max ±35) ───────────────────────────────────────────────
    oi_score = 0
    if oi_signal == "Long Buildup":
        oi_score = 35
    elif oi_signal == "Short Covering":
        oi_score = 20
    elif oi_signal == "Short Buildup":
        oi_score = -35
    elif oi_signal == "Long Unwinding":
        oi_score = -20
    elif oi_signal == "OI_DATA_MISSING":
        oi_score = 0  # Don't penalise or reward — data is absent

    # ── Momentum component (max ±25) ─────────────────────────────────────────
    mom_score = 0
    if momentum_confirm:
        if ema_bias in ("Bullish", "Partial Bullish"):
            mom_score = 25
        elif ema_bias == "Bearish":
            mom_score = -25
    else:
        # Partial credit/penalty based on momentum score magnitude
        if momentum_score_pct > 10:
            mom_score = 15
        elif momentum_score_pct > 5:
            mom_score = 8
        elif momentum_score_pct < -10:
            mom_score = -15
        elif momentum_score_pct < -5:
            mom_score = -8

    raw_score = ema_score + oi_score + mom_score
    # Map from [-100, +100] → [0, 100]
    directional_score = round((raw_score + 100) / 2)
    # Clamp
    directional_score = max(0, min(100, directional_score))

    return directional_score


def _sanity_check_score(
    ticker: str,
    ema_bias: str,
    oi_signal: str,
    momentum_confirm: bool,
    directional_score: int,
) -> None:
    """Assert score consistency with the underlying signals.

    Raises AssertionError if impossible combinations are detected.
    These are logged as errors but do NOT crash the service.
    """
    try:
        # Strong bullish: all signals agree → score must be high
        if (
            ema_bias == "Bullish"
            and oi_signal == "Long Buildup"
            and momentum_confirm
            and directional_score <= 80
        ):
            logger.error(
                f"SCORE SANITY FAIL [{ticker}]: Bullish + Long Buildup + "
                f"Momentum Confirm but score={directional_score} (expected >80)"
            )

        # Strong bearish: all signals agree → score must be low
        if (
            ema_bias == "Bearish"
            and oi_signal == "Short Buildup"
            and momentum_confirm
            and directional_score >= 20
        ):
            logger.error(
                f"SCORE SANITY FAIL [{ticker}]: Bearish + Short Buildup + "
                f"Momentum Confirm but score={directional_score} (expected <20)"
            )

        # Bearish EMA should never produce BUY CALL score
        if ema_bias == "Bearish" and directional_score > 65:
            logger.error(
                f"SCORE SANITY FAIL [{ticker}]: Bearish EMA but "
                f"score={directional_score} (>65, would trigger BUY CALL)"
            )

        # Bullish EMA should never produce BUY PUT score
        if ema_bias == "Bullish" and directional_score < 35:
            logger.error(
                f"SCORE SANITY FAIL [{ticker}]: Bullish EMA but "
                f"score={directional_score} (<35, would trigger BUY PUT)"
            )

    except Exception as e:
        logger.warning(f"Sanity check error for {ticker}: {e}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def compute_directional_bias(
    fno_stocks: list[dict],
    debug: bool = False,
) -> list[dict]:
    """Compute directional bias for stocks already enriched by the FNO pipeline.

    Args:
        fno_stocks: List of stock dicts from the FNO pipeline output.
                    Each dict is expected to have 'symbol', 'technicals',
                    'option_chain', 'pChange', etc.
        debug:      If True, log raw inputs → final score for every ticker.

    Returns:
        List of bias result dicts, sorted by Directional_Score descending.
    """
    global DEBUG_MODE
    if debug:
        DEBUG_MODE = True

    results: list[dict] = []

    for stock in fno_stocks:
        symbol = stock.get("symbol", "UNKNOWN")

        # ── 1. Extract technicals (already computed by FNO pipeline) ─────────
        tech = stock.get("technicals")
        if not tech:
            if DEBUG_MODE:
                logger.info(f"[BIAS DEBUG] {symbol}: SKIPPED — no technicals data")
            continue

        close = tech.get("close", 0)
        momentum_score_pct = tech.get("momentum_score", 0)

        # ── 2. EMA Bias — from FNO pipeline's EMA10/EMA50H/EMA50L ───────────
        ema_bias, ema_pass_count = classify_ema_bias(tech)

        # ── 3. OI Signal — from FNO pipeline's option chain data ─────────────
        oc: dict | None = stock.get("option_chain")
        oi_data_available = oc is not None and bool(oc)

        price_change_pct = stock.get("pChange", 0) or 0
        oi_change_pct = 0.0
        ce_oi = 0
        pe_oi = 0
        ce_oi_change = 0
        pe_oi_change = 0

        if oi_data_available:
            atm_ce = oc.get("atm_ce") or {}
            atm_pe = oc.get("atm_pe") or {}

            ce_oi = atm_ce.get("openInterest", 0)
            pe_oi = atm_pe.get("openInterest", 0)
            ce_oi_change = atm_ce.get("changeinOpenInterest", 0)
            pe_oi_change = atm_pe.get("changeinOpenInterest", 0)

            total_oi = max(ce_oi + pe_oi, 1)
            total_oi_change = ce_oi_change + pe_oi_change
            oi_change_pct = (total_oi_change / total_oi) * 100

        oi_signal, oi_data_source = classify_oi_signal(
            price_change_pct, oi_change_pct, oi_data_available
        )

        # ── 4. Momentum Confirm ──────────────────────────────────────────────
        # Momentum confirms when the trend direction and momentum magnitude
        # agree with the EMA bias.
        momentum_confirm = False
        if ema_bias in ("Bullish", "Partial Bullish") and momentum_score_pct > 5:
            momentum_confirm = True
        elif ema_bias == "Bearish" and momentum_score_pct < -5:
            momentum_confirm = True

        # ── 5. Composite Score ───────────────────────────────────────────────
        directional_score = compute_directional_score(
            ema_bias=ema_bias,
            ema_pass_count=ema_pass_count,
            oi_signal=oi_signal,
            momentum_confirm=momentum_confirm,
            momentum_score_pct=momentum_score_pct,
            price_change_pct=price_change_pct,
        )

        # ── 6. Sanity check ─────────────────────────────────────────────────
        _sanity_check_score(
            symbol, ema_bias, oi_signal, momentum_confirm, directional_score
        )

        # ── 7. Recommendation ───────────────────────────────────────────────
        if directional_score > 65 and ema_bias in ("Bullish", "Partial Bullish"):
            recommendation = "BUY CALL"
        elif directional_score < 35 and ema_bias == "Bearish":
            recommendation = "BUY PUT"
        else:
            recommendation = "NO CLEAR BIAS - SKIP"

        # ── 8. Confidence ───────────────────────────────────────────────────
        confidence = "Low"
        if recommendation != "NO CLEAR BIAS - SKIP":
            is_bull = (recommendation == "BUY CALL")
            agree_count = 0

            if ema_bias == ("Bullish" if is_bull else "Bearish"):
                agree_count += 1
            if is_bull and oi_signal in ("Long Buildup", "Short Covering"):
                agree_count += 1
            elif not is_bull and oi_signal in ("Short Buildup", "Long Unwinding"):
                agree_count += 1
            if momentum_confirm:
                agree_count += 1

            if agree_count == 3:
                confidence = "High"
            elif agree_count == 2:
                confidence = "Medium"
            else:
                confidence = "Low"

        # ── 9. PCR from option chain ─────────────────────────────────────────
        pcr = round(oc.get("pcr", 1.0), 2) if oi_data_available else 0.0

        # ── Build result ─────────────────────────────────────────────────────
        result = {
            "Ticker": symbol,
            "LTP": round(close, 2),
            "Chg_Pct": round(price_change_pct, 2),
            "EMA_Bias": ema_bias,
            "EMA_Pass_Count": ema_pass_count,
            "EMA10_Pass": bool(tech.get("ema10_pass", False)),
            "EMA50H_Pass": bool(tech.get("ema50h_pass", False)),
            "EMA50L_Pass": bool(tech.get("ema50l_pass", False)),
            "OI_Signal": oi_signal,
            "OI_Data_Source": oi_data_source,
            "PCR": pcr,
            "Momentum_Score_Pct": round(momentum_score_pct, 2),
            "Momentum_Confirm": momentum_confirm,
            "Directional_Score": directional_score,
            "Recommendation": recommendation,
            "Confidence": confidence,
        }
        results.append(result)

        # ── Debug logging ────────────────────────────────────────────────────
        if DEBUG_MODE:
            logger.info(
                f"[BIAS DEBUG] {symbol}: "
                f"Close={close:.2f} | Chg%={price_change_pct:.2f} | "
                f"EMA10_Pass={tech.get('ema10_pass')} "
                f"EMA50H_Pass={tech.get('ema50h_pass')} "
                f"EMA50L_Pass={tech.get('ema50l_pass')} | "
                f"EMA_Bias={ema_bias} ({ema_pass_count}/3) | "
                f"OI_Signal={oi_signal} (src={oi_data_source}) "
                f"CE_OI_Chg={ce_oi_change} PE_OI_Chg={pe_oi_change} "
                f"OI_Chg%={oi_change_pct:.2f} | "
                f"Momentum={momentum_score_pct:.2f}% "
                f"Confirm={momentum_confirm} | "
                f"Score={directional_score} → {recommendation} "
                f"({confidence})"
            )

    # Sort by score descending
    results.sort(key=lambda x: x["Directional_Score"], reverse=True)

    if DEBUG_MODE:
        logger.info(
            f"[BIAS DEBUG] Processed {len(results)} stocks. "
            f"Top: {results[0]['Ticker']}={results[0]['Directional_Score']} "
            f"Bot: {results[-1]['Ticker']}={results[-1]['Directional_Score']}"
            if results else "[BIAS DEBUG] No results."
        )

    # Reset debug mode after run
    DEBUG_MODE = False

    return results
