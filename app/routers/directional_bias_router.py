from fastapi import APIRouter, HTTPException
from typing import Any
import logging
import time

from app.services.fno_service import get_full_fno_terminal
from app.services.directional_bias_service import compute_directional_bias

router = APIRouter(prefix="/api/fno", tags=["Directional Bias"])
logger = logging.getLogger(__name__)

# ── Accumulated bias results across batches ──────────────────────────────────
# Keyed by ticker symbol so new batches merge (upsert) into the existing set.
_accumulated_bias: dict[str, dict] = {}
_accumulated_ts: float = 0.0        # timestamp of first batch
_ACCUMULATE_TTL = 7200               # 2 hours — matches FNO cache TTL


def _is_accumulation_stale() -> bool:
    """Check if accumulated data is too old and should be reset."""
    if _accumulated_ts == 0.0:
        return False  # nothing accumulated yet
    return (time.time() - _accumulated_ts) > _ACCUMULATE_TTL


@router.get("/directional-bias")
async def get_directional_bias(
    version: str = "v2",
    start_idx: int = 0,
    end_idx: int = 30,
    debug: bool = False,
    reset: bool = False,
) -> list[dict[str, Any]]:
    """
    Computes and returns the Directional Bias Table for the filtered FNO stocks.

    Results accumulate across batches — requesting batch 0-30 then 30-60
    returns the merged set of all 60 stocks, sorted by score.

    Query params:
        debug:      If true, logs raw inputs → final score for every ticker.
        reset:      If true, clears accumulated results before this batch.
    """
    global _accumulated_bias, _accumulated_ts

    # Reset if explicitly requested or if stale
    if reset or _is_accumulation_stale():
        _accumulated_bias.clear()
        _accumulated_ts = 0.0
        if debug:
            logger.info("[BIAS] Accumulated results cleared")

    try:
        # First, run the existing FNO filter pipeline to get the universe
        fno_result = await get_full_fno_terminal(
            version=version, start_idx=start_idx, end_idx=end_idx
        )

        # get_full_fno_terminal returns a dict with a 'stocks' key.
        # Each stock dict already contains 'technicals', 'option_chain',
        # 'pChange', etc. from the pipeline enrichment.
        #
        # IMPORTANT: We need the *enriched internal dicts* (with 'technicals'
        # and 'option_chain' keys), NOT the flattened output dicts.
        # The pipeline stores intermediate enriched dicts before building
        # the output list.  However, the public API returns flattened output.
        #
        # To get the enriched data, we access the internal pipeline state.
        # If the result only has flattened output, we fall back to re-running
        # the pipeline for the specific symbols.
        fno_stocks = []

        if isinstance(fno_result, dict):
            # Check for enriched_stocks (internal enriched dicts with
            # 'technicals' and 'option_chain' sub-dicts)
            enriched = fno_result.get("enriched_stocks")
            if enriched:
                fno_stocks = enriched
            else:
                # Fallback: use the flattened output stocks.
                # Reconstruct minimal technicals from the flattened fields.
                output_stocks = fno_result.get("stocks", [])
                for s in output_stocks:
                    fno_stocks.append({
                        "symbol": s.get("symbol", ""),
                        "pChange": s.get("change_pct", 0),
                        "technicals": {
                            "close": s.get("cmp", 0),
                            "ema10_pass": s.get("ema_status", {}).get("ema10", False),
                            "ema50h_pass": s.get("ema_status", {}).get("ema50h", False),
                            "ema50l_pass": s.get("ema_status", {}).get("ema50l", False),
                            "ema10": s.get("ema10", 0),
                            "ema50h": s.get("ema50h", 0),
                            "ema50l": s.get("ema50l", 0),
                            "momentum_score": s.get("momentum_score", 0),
                        },
                        "option_chain": {
                            "pcr": s.get("pcr", 0),
                            "atm_ce": {
                                "openInterest": 0,
                                "changeinOpenInterest": s.get("ce_oi_change", 0),
                            },
                            "atm_pe": {
                                "openInterest": 0,
                                "changeinOpenInterest": s.get("pe_oi_change", 0),
                            },
                        } if s.get("pcr", 0) > 0 else None,
                    })

        if not fno_stocks:
            # No new stocks — return whatever is already accumulated
            if _accumulated_bias:
                combined = sorted(
                    _accumulated_bias.values(),
                    key=lambda x: x["Directional_Score"],
                    reverse=True,
                )
                return combined
            return []

        # Compute bias for this batch
        bias_table = await compute_directional_bias(fno_stocks, debug=debug)

        # ── Accumulate: merge new results into existing ──────────────────────
        # Set timestamp on first batch
        if _accumulated_ts == 0.0:
            _accumulated_ts = time.time()

        # Upsert by ticker — new data for same ticker overwrites old
        for entry in bias_table:
            ticker = entry["Ticker"]
            _accumulated_bias[ticker] = entry

        # Return the full accumulated set, sorted by score descending
        combined = sorted(
            _accumulated_bias.values(),
            key=lambda x: x["Directional_Score"],
            reverse=True,
        )

        logger.info(
            f"Directional bias: batch {start_idx}-{end_idx} added "
            f"{len(bias_table)} stocks, total accumulated: {len(combined)}"
        )

        return combined
    except Exception as e:
        logger.error(f"Error computing directional bias: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/directional-bias")
async def clear_directional_bias() -> dict[str, str]:
    """Clear accumulated directional bias results."""
    global _accumulated_bias, _accumulated_ts
    count = len(_accumulated_bias)
    _accumulated_bias.clear()
    _accumulated_ts = 0.0
    return {"status": "cleared", "removed": str(count)}
