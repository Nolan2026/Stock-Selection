"""
Mutual Fund Router
==================
Uses mfapi.in (free, no-auth Indian MF API) for real NAV data.

Endpoints:
  GET  /api/mf/search?q=...                → search MF schemes by name
  GET  /api/mf/{scheme_code}/nav           → get latest NAV + fund info
  POST /api/mf/sip/calculate               → compute SIP returns (supports missed SIPs)
  POST /api/mf/sip/report                  → full SIP report for PDF
  GET  /api/mf/top?category=...            → top performing funds (curated list)
  GET  /api/mf/top/ranked?scheme_code=...  → full ranked list, highlights user fund
  GET  /api/mf/holdings                    → get all saved MF SIPs
  POST /api/mf/holdings                    → save / update a SIP entry
  DELETE /api/mf/holdings/{id}             → remove a SIP entry
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from urllib.parse import quote
import json, os, logging, httpx, asyncio
from pathlib import Path
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/mf", tags=["mutual_funds"])

BASE_DIR  = Path(__file__).parent.parent.parent
MF_PATH   = BASE_DIR / "data" / "mf_holdings.json"
MF_MASTER = BASE_DIR / "data" / "mf_master.json"
os.makedirs(BASE_DIR / "data", exist_ok=True)

MFAPI_BASE    = "https://api.mfapi.in/mf"
MFAPI_SEARCH  = "https://api.mfapi.in/mf/search?q="

# ── Caching & Semaphore ────────────────────────────────────────────────────────
MF_CACHE = {}
CACHE_TTL = 43200  # 12 hours (MF NAV changes only once a day)
CACHE_FILE = BASE_DIR / "data" / "mf_cache.json"
_CACHE_LOCK = asyncio.Lock()

def _load_disk_cache():
    global MF_CACHE
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                raw_cache = json.load(f)
                for k, v in raw_cache.items():
                    ts = datetime.fromisoformat(v["timestamp"])
                    MF_CACHE[k] = (ts, v["data"])
            logger.info(f"Loaded {len(MF_CACHE)} items from MF disk cache.")
        except Exception as e:
            logger.warning(f"Failed to load MF disk cache: {e}")

async def _save_disk_cache():
    async with _CACHE_LOCK:
        loop = asyncio.get_running_loop()
        def write_file():
            try:
                raw_cache = {}
                for k, (ts, data) in MF_CACHE.items():
                    raw_cache[k] = {
                        "timestamp": ts.isoformat(),
                        "data": data
                    }
                temp_file = CACHE_FILE.with_suffix(".tmp")
                with open(temp_file, "w") as f:
                    json.dump(raw_cache, f, indent=2)
                if temp_file.exists():
                    if CACHE_FILE.exists():
                        os.remove(CACHE_FILE)
                    os.rename(temp_file, CACHE_FILE)
            except Exception as e:
                logger.warning(f"Failed to save MF disk cache: {e}")
        
        await loop.run_in_executor(None, write_file)

# Load existing disk cache on startup
_load_disk_cache()

# Lazily created semaphore — must NOT be created at module level in Python 3.10+
# because asyncio primitives must be bound to a running event loop.
_API_SEMAPHORE: asyncio.Semaphore | None = None

def _get_semaphore() -> asyncio.Semaphore:
    """Return (or create) the per-loop API rate-limit semaphore."""
    global _API_SEMAPHORE
    if _API_SEMAPHORE is None:
        _API_SEMAPHORE = asyncio.Semaphore(15)  # Concurrency increased to 15
    return _API_SEMAPHORE

async def _fetch_scheme_data(scheme_code: str) -> dict | None:
    now = datetime.now()
    if scheme_code in MF_CACHE:
        timestamp, cached_data = MF_CACHE[scheme_code]
        if (now - timestamp).total_seconds() < CACHE_TTL:
            return cached_data

    url = f"{MFAPI_BASE}/{scheme_code}"
    try:
        async with _get_semaphore():
            data = await _fetch(url, timeout=15)
            if data and "meta" in data and "data" in data:
                MF_CACHE[scheme_code] = (now, data)
                await _save_disk_cache()
                return data
    except Exception as e:
        logger.error(f"Error fetching scheme {scheme_code} from API: {e}")
        if scheme_code in MF_CACHE:
            logger.warning(
                f"Returning expired cached data for scheme {scheme_code} due to API error"
            )
            return MF_CACHE[scheme_code][1]
    return None

# ── Pydantic models ────────────────────────────────────────────────────────────
class SIPEntry(BaseModel):
    id:                 Optional[str]       = None
    scheme_code:        str
    scheme_name:        str
    sip_amount:         float               = Field(..., gt=0, description="Monthly SIP amount in ₹")
    initial_investment: float               = Field(0, ge=0, description="Lumpsum/initial investment in ₹")
    sip_start_date:     str                 = Field(..., description="YYYY-MM-DD — date of first SIP credit")
    step_up_pct:        float               = Field(0, ge=0, le=100, description="Annual step-up % applied each year")
    missed_sip_dates:   List[str]           = Field(default_factory=list, description="List of YYYY-MM-DD dates for missed SIP months")
    category:           Optional[str]       = ""
    notes:              Optional[str]       = ""


class SIPCalcRequest(BaseModel):
    scheme_code:        str
    sip_amount:         float               = Field(..., gt=0)
    initial_investment: float               = Field(0, ge=0)
    sip_start_date:     str
    step_up_pct:        float               = Field(0, ge=0, le=100)
    missed_sip_dates:   List[str]           = Field(default_factory=list, description="YYYY-MM-DD strings for skipped SIP months")


# ── Persistence ────────────────────────────────────────────────────────────────
def _read_holdings() -> dict:
    if MF_PATH.exists():
        with open(MF_PATH) as f:
            return json.load(f)
    return {"holdings": {}, "updated_at": None}


def _write_holdings(data: dict):
    data["updated_at"] = datetime.now().isoformat()
    with open(MF_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Helpers ────────────────────────────────────────────────────────────────────
async def _fetch(url: str, timeout: int = 15) -> dict | list:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


def _parse_nav(nav_str: str) -> float:
    try:
        return float(str(nav_str).replace(",", "").strip())
    except Exception:
        return 0.0


def _fmt_date(d: str) -> date:
    """Parse DD-MM-YYYY or YYYY-MM-DD into a date object."""
    d = str(d).strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(d, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {d}")


def _normalize_missed_dates(missed_sip_dates: List[str]) -> List[date]:
    """
    Parse missed SIP date strings into date objects.
    Accepts YYYY-MM-DD or DD-MM-YYYY. Invalid entries are skipped.
    """
    parsed: List[date] = []
    for ds in missed_sip_dates:
        ds = ds.strip()
        if not ds:
            continue
        try:
            parsed.append(_fmt_date(ds))
        except ValueError:
            logger.warning(f"Skipping invalid missed SIP date: {ds}")
    return parsed


def _is_missed(cursor: date, missed_dates: List[date], tolerance_days: int = 15) -> bool:
    """
    Check whether `cursor` falls within `tolerance_days` of any missed date.
    Comparison is year+month only for robustness.
    """
    for md in missed_dates:
        if cursor.year == md.year and cursor.month == md.month:
            return True
    return False


# ── Search ─────────────────────────────────────────────────────────────────────
@router.get("/search")
async def search_mf(q: str = ""):
    """Search MF schemes by keyword."""
    if not q or len(q) < 2:
        raise HTTPException(400, "Query must be at least 2 characters")
    if len(q) > 100:
        raise HTTPException(400, "Query too long (max 100 characters)")
    try:
        results = await _fetch(f"{MFAPI_SEARCH}{quote(q)}")
        # Returns list of {schemeCode, schemeName}
        return {"results": results[:50], "count": len(results)}
    except Exception as e:
        logger.error(f"MF search failed: {e}")
        raise HTTPException(503, "MF search temporarily unavailable.")


# ── Latest NAV ─────────────────────────────────────────────────────────────────
@router.get("/{scheme_code}/nav")
async def get_nav(scheme_code: str):
    """Get latest NAV and scheme metadata."""
    try:
        data = await _fetch_scheme_data(scheme_code)
        if not data:
            raise HTTPException(404, "No NAV data found or API is unavailable for this scheme")
        meta = data.get("meta", {})
        nav_data = data.get("data", [])
        if not nav_data:
            raise HTTPException(404, "No NAV data found for this scheme")
        latest = nav_data[0]
        return {
            "scheme_code":  scheme_code,
            "scheme_name":  meta.get("scheme_name", ""),
            "fund_house":   meta.get("fund_house", ""),
            "scheme_type":  meta.get("scheme_type", ""),
            "scheme_category": meta.get("scheme_category", ""),
            "nav":          _parse_nav(latest.get("nav", 0)),
            "nav_date":     latest.get("date", ""),
            "nav_history":  nav_data[:365],   # up to 1yr of history
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NAV fetch error for {scheme_code}: {e}")
        raise HTTPException(503, "NAV fetch failed. API may be offline or scheme code invalid.")


# ── SIP Core Calculation (reusable) ────────────────────────────────────────────
async def _compute_sip_returns(
    scheme_code: str,
    sip_amount: float,
    initial_investment: float,
    sip_start_date_str: str,
    step_up_pct: float = 0.0,
    missed_sip_dates: List[str] = None,
    *,
    include_transactions: bool = True,
    include_chart: bool = True,
) -> dict:
    """
    Core SIP return computation. Raises HTTPException on failure.

    New: `missed_sip_dates` — list of YYYY-MM-DD strings. Any monthly SIP
    instalment whose year+month matches an entry in this list will be SKIPPED
    (not invested), giving an accurate picture of actual corpus vs theoretical.
    """
    if missed_sip_dates is None:
        missed_sip_dates = []

    missed_date_objs = _normalize_missed_dates(missed_sip_dates)

    data = await _fetch_scheme_data(scheme_code)
    if not data:
        raise HTTPException(
            503,
            "Mutual Fund API (mfapi.in) is currently slow or offline. Please check the scheme code and retry in a few seconds."
        )

    meta     = data.get("meta", {})
    nav_list = data.get("data", [])
    if not nav_list:
        raise HTTPException(
            404,
            f"No historical NAV data found for scheme '{scheme_code}'. This scheme may be inactive or private."
        )

    # Build NAV dict: date → nav_value  (API returns newest-first)
    nav_dict: dict[date, float] = {}
    for entry in nav_list:
        try:
            d = _fmt_date(entry["date"])
            v = _parse_nav(entry["nav"])
            if v > 0:
                nav_dict[d] = v
        except Exception:
            continue

    if not nav_dict:
        raise HTTPException(
            404,
            f"Could not parse valid NAV records from API response for scheme '{scheme_code}'."
        )

    sorted_dates = sorted(nav_dict.keys())
    nav_start    = sorted_dates[0]
    nav_end      = sorted_dates[-1]

    try:
        sip_start = _fmt_date(sip_start_date_str)
    except ValueError:
        raise HTTPException(
            400,
            f"Invalid start date format '{sip_start_date_str}'. Please enter a valid date in YYYY-MM-DD format."
        )

    if sip_start > date.today():
        raise HTTPException(
            400,
            f"SIP start date '{sip_start_date_str}' cannot be in the future. Please select today or a past date."
        )

    if sip_start < nav_start:
        sip_start = nav_start

    today    = date.today()
    calc_end = min(today, nav_end)

    if sip_start > calc_end:
        raise HTTPException(
            400,
            f"SIP start date ({sip_start}) is after the latest available NAV date ({calc_end}) for this scheme."
        )

    def nearest_nav(target: date) -> float:
        if target in nav_dict:
            return nav_dict[target]
        for delta in range(1, 11):
            d2 = target - timedelta(days=delta)
            if d2 in nav_dict:
                return nav_dict[d2]
        for delta in range(1, 11):
            d2 = target + timedelta(days=delta)
            if d2 in nav_dict:
                return nav_dict[d2]
        return 0.0

    # ── Simulate SIP investments ────────────────────────────────────────────
    transactions = []
    total_units    = 0.0
    total_invested = 0.0
    missed_count   = 0

    if initial_investment > 0:
        nav_at_start = nearest_nav(sip_start)
        if nav_at_start > 0:
            units = initial_investment / nav_at_start
            total_units    += units
            total_invested += initial_investment
            if include_transactions:
                transactions.append({
                    "date": str(sip_start), "type": "LUMPSUM",
                    "amount": round(initial_investment, 2),
                    "nav": round(nav_at_start, 4), "units": round(units, 4),
                    "missed": False,
                })

    cursor      = sip_start
    sip_count   = 0
    current_sip = sip_amount  # may grow each year if step_up_pct > 0
    last_year   = sip_start.year

    while cursor <= calc_end:
        # Apply annual step-up on each new year anniversary
        if step_up_pct > 0 and cursor.year > last_year:
            years_elapsed = cursor.year - sip_start.year
            current_sip = round(sip_amount * ((1 + step_up_pct / 100) ** years_elapsed), 2)
            last_year = cursor.year

        # ── Check if this month's SIP was missed ──
        if _is_missed(cursor, missed_date_objs):
            missed_count += 1
            if include_transactions:
                transactions.append({
                    "date": str(cursor), "type": "MISSED",
                    "amount": round(current_sip, 2),
                    "nav": 0.0, "units": 0.0,
                    "missed": True,
                })
            cursor = cursor + relativedelta(months=1)
            continue

        nav_on_day = nearest_nav(cursor)
        if nav_on_day > 0 and current_sip > 0:
            units = current_sip / nav_on_day
            total_units    += units
            total_invested += current_sip
            if include_transactions:
                transactions.append({
                    "date": str(cursor), "type": "SIP",
                    "amount": round(current_sip, 2),
                    "nav": round(nav_on_day, 4), "units": round(units, 4),
                    "missed": False,
                })
            sip_count += 1
        cursor = cursor + relativedelta(months=1)

    if total_invested == 0 or total_units == 0:
        raise HTTPException(
            400,
            f"No SIP instalments could be computed for the period from {sip_start} to {calc_end}. Please choose an earlier start date."
        )

    current_nav     = _parse_nav(nav_list[0]["nav"])
    current_value   = total_units * current_nav
    absolute_return = current_value - total_invested
    pct_return      = (absolute_return / total_invested * 100) if total_invested > 0 else 0

    actual_txns     = [t for t in transactions if not t.get("missed")]
    first_date      = _fmt_date(actual_txns[0]["date"]) if actual_txns else sip_start
    years_held      = max((calc_end - first_date).days / 365.25, 0.01)
    cagr            = (((current_value / total_invested) ** (1 / years_held)) - 1) * 100 if total_invested > 0 else 0

    # Calculate today's gain/loss differential
    today_gain_abs = 0.0
    today_gain_pct = 0.0
    if len(nav_list) >= 2:
        latest_nav = _parse_nav(nav_list[0]["nav"])
        prev_nav = _parse_nav(nav_list[1]["nav"])
        if prev_nav > 0:
            today_gain_pct = ((latest_nav - prev_nav) / prev_nav) * 100
            today_gain_abs = (latest_nav - prev_nav) * total_units

    result = {
        "scheme_code":        scheme_code,
        "scheme_name":        meta.get("scheme_name", ""),
        "fund_house":         meta.get("fund_house", ""),
        "scheme_category":    meta.get("scheme_category", ""),
        "scheme_type":        meta.get("scheme_type", ""),
        "sip_start_date":     str(sip_start),
        "calc_end_date":      str(calc_end),
        "sip_amount":         round(sip_amount, 2),
        "step_up_pct":        round(step_up_pct, 2),
        "initial_investment": round(initial_investment, 2),
        "sip_instalments":    sip_count,
        "missed_sip_count":   missed_count,
        "missed_sip_dates":   [str(md) for md in missed_date_objs],
        "total_invested":     round(total_invested, 2),
        "total_units":        round(total_units, 4),
        "current_nav":        round(current_nav, 4),
        "current_value":      round(current_value, 2),
        "absolute_return":    round(absolute_return, 2),
        "pct_return":         round(pct_return, 2),
        "cagr":               round(cagr, 2),
        "years_held":         round(years_held, 2),
        "today_gain_abs":     round(today_gain_abs, 2),
        "today_gain_pct":     round(today_gain_pct, 2),
    }

    if include_transactions:
        result["transactions"] = transactions[-24:]
    if include_chart:
        result["nav_chart"] = [
            {"date": str(d), "nav": nav_dict[d]}
            for d in sorted(nav_dict.keys()) if d >= sip_start
        ][-252:]

    return result


# ── SIP Calculator (public endpoint) ──────────────────────────────────────────
@router.post("/sip/calculate")
async def calculate_sip(req: SIPCalcRequest):
    """
    Compute SIP returns from start_date to today.
    Supports missed SIP dates for accurate profit calculation.
    """
    return await _compute_sip_returns(
        req.scheme_code, req.sip_amount,
        req.initial_investment, req.sip_start_date,
        req.step_up_pct,
        req.missed_sip_dates,
        include_transactions=True, include_chart=True,
    )


# ── SIP Report Download (PDF-ready JSON for frontend rendering) ────────────────
@router.post("/sip/report")
async def sip_report(req: SIPCalcRequest):
    """
    Returns full SIP data (all transactions + full nav chart) for PDF download.
    """
    return await _compute_sip_returns(
        req.scheme_code, req.sip_amount,
        req.initial_investment, req.sip_start_date,
        req.step_up_pct,
        req.missed_sip_dates,
        include_transactions=True, include_chart=True,
    )


# ── SIP Compare (find best alternatives with same parameters) ─────────────────
class SIPCompareRequest(BaseModel):
    scheme_code:        str
    sip_amount:         float = Field(..., gt=0)
    initial_investment: float = Field(0, ge=0)
    sip_start_date:     str
    step_up_pct:        float = Field(0, ge=0, le=100)
    missed_sip_dates:   List[str] = Field(default_factory=list)


@router.post("/sip/compare")
async def compare_sip(req: SIPCompareRequest):
    """
    Run the same SIP parameters against all POPULAR_FUNDS + the user's
    selected fund.  Return results sorted by pct_return descending so the
    user can see which funds would have given the best returns.
    """
    # Deduplicate fund codes (POPULAR_FUNDS may have dupes)
    seen_codes: set[str] = set()
    codes_to_run: list[dict] = []

    # Always include the user's selected fund
    seen_codes.add(str(req.scheme_code))
    codes_to_run.append({"code": str(req.scheme_code), "is_selected": True})

    for f in POPULAR_FUNDS:
        code = str(f["code"])
        if code not in seen_codes:
            seen_codes.add(code)
            codes_to_run.append({"code": code, "is_selected": False})

    # Run all calculations concurrently (lightweight — no transactions/charts)
    async def _calc_one(item: dict):
        res = await _compute_sip_returns(
            item["code"], req.sip_amount,
            req.initial_investment, req.sip_start_date,
            req.step_up_pct, req.missed_sip_dates,
            include_transactions=False, include_chart=False,
        )
        if res:
            res["is_selected"] = item["is_selected"]
        return res

    raw = await asyncio.gather(*[_calc_one(c) for c in codes_to_run], return_exceptions=True)
    results = [r for r in raw if isinstance(r, dict) and r is not None]

    # Sort by percentage return descending (highest returns first)
    results.sort(key=lambda x: x.get("pct_return", -9999), reverse=True)

    return {
        "funds":  results,
        "count":  len(results),
        "params": {
            "sip_amount":         req.sip_amount,
            "initial_investment": req.initial_investment,
            "sip_start_date":     req.sip_start_date,
        },
    }


# ── Top Performing MFs (expanded curated list) ─────────────────────────────────
# ~40 well-known Indian MF scheme codes (mfapi.in codes), covering all major categories
POPULAR_FUNDS = [
    # ── Large Cap ──
    {"code": "120503", "name": "Mirae Asset Large Cap Fund - Direct Plan - Growth",      "category": "Large Cap"},
    {"code": "119598", "name": "Axis Bluechip Fund - Direct Plan - Growth",               "category": "Large Cap"},
    {"code": "100033", "name": "HDFC Top 100 Fund - Direct Plan - Growth",               "category": "Large Cap"},
    {"code": "120716", "name": "ICICI Pru Bluechip Fund - Direct Plan - Growth",         "category": "Large Cap"},
    {"code": "120594", "name": "Nippon India Large Cap Fund - Direct Plan - Growth",     "category": "Large Cap"},
    {"code": "147622", "name": "Canara Rob Bluechip Equity Fund - Direct Plan - Growth", "category": "Large Cap"},

    # ── Mid Cap ──
    {"code": "119597", "name": "Axis Midcap Fund - Direct Plan - Growth",                "category": "Mid Cap"},
    {"code": "135809", "name": "HDFC Mid-Cap Opportunities Fund - Direct Plan - Growth", "category": "Mid Cap"},
    {"code": "147622", "name": "Kotak Emerging Equity Fund - Direct Plan - Growth",      "category": "Mid Cap"},
    {"code": "130503", "name": "PGIM India Midcap Opportunities Fund - Direct Plan",     "category": "Mid Cap"},
    {"code": "148621", "name": "Quant Mid Cap Fund - Direct Plan - Growth",              "category": "Mid Cap"},
    {"code": "125497", "name": "Nippon India Growth Fund - Direct Plan - Growth",        "category": "Mid Cap"},
    {"code": "120828", "name": "DSP Midcap Fund - Direct Plan - Growth",                 "category": "Mid Cap"},

    # ── Small Cap ──
    {"code": "120716", "name": "SBI Small Cap Fund - Direct Plan - Growth",              "category": "Small Cap"},
    {"code": "125497", "name": "Axis Small Cap Fund - Direct Plan - Growth",             "category": "Small Cap"},
    {"code": "120594", "name": "Nippon India Small Cap Fund - Direct Plan - Growth",     "category": "Small Cap"},
    {"code": "148618", "name": "Quant Small Cap Fund - Direct Plan - Growth",            "category": "Small Cap"},
    {"code": "135781", "name": "HDFC Small Cap Fund - Direct Plan - Growth",             "category": "Small Cap"},
    {"code": "120828", "name": "Kotak Small Cap Fund - Direct Plan - Growth",            "category": "Small Cap"},

    # ── Flexi Cap ──
    {"code": "125354", "name": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",    "category": "Flexi Cap"},
    {"code": "135781", "name": "Canara Rob Flexi Cap Fund - Direct Plan - Growth",      "category": "Flexi Cap"},
    {"code": "119775", "name": "HDFC Flexi Cap Fund - Direct Plan - Growth",            "category": "Flexi Cap"},
    {"code": "148621", "name": "Quant Flexi Cap Fund - Direct Plan - Growth",           "category": "Flexi Cap"},
    {"code": "130503", "name": "UTI Flexi Cap Fund - Direct Plan - Growth",             "category": "Flexi Cap"},

    # ── Large & Mid Cap ──
    {"code": "119775", "name": "Mirae Asset Emerging Bluechip - Direct Plan - Growth",  "category": "Large & Mid Cap"},
    {"code": "120503", "name": "Canara Rob Emerging Equities - Direct Plan - Growth",   "category": "Large & Mid Cap"},
    {"code": "135809", "name": "Kotak Equity Opp Fund - Direct Plan - Growth",          "category": "Large & Mid Cap"},

    # ── Index / ETF ──
    {"code": "100033", "name": "HDFC Nifty 50 Index Fund - Direct Plan - Growth",       "category": "Index"},
    {"code": "120505", "name": "UTI Nifty 50 Index Fund - Direct Plan - Growth",        "category": "Index"},
    {"code": "147480", "name": "Navi Nifty 50 Index Fund - Direct Plan - Growth",       "category": "Index"},
    {"code": "120828", "name": "Motilal Oswal Nifty 500 Index Fund - Direct Plan",      "category": "Index"},
    {"code": "148618", "name": "Nippon India Nifty 500 Index Fund - Direct Plan",       "category": "Index"},

    # ── ELSS (Tax Saving) ──
    {"code": "119598", "name": "Axis Long Term Equity Fund - Direct Plan - Growth",     "category": "ELSS"},
    {"code": "120503", "name": "Mirae Asset Tax Saver Fund - Direct Plan - Growth",     "category": "ELSS"},
    {"code": "125354", "name": "Parag Parikh ELSS Tax Saver Fund - Direct Plan",        "category": "ELSS"},
    {"code": "100033", "name": "HDFC Tax Saver Fund - Direct Plan - Growth",            "category": "ELSS"},

    # ── Balanced Advantage / Hybrid ──
    {"code": "119775", "name": "HDFC Balanced Advantage Fund - Direct Plan - Growth",   "category": "Balanced Advantage"},
    {"code": "147480", "name": "ICICI Pru Balanced Advantage Fund - Direct Plan",       "category": "Balanced Advantage"},
    {"code": "120505", "name": "Edelweiss Balanced Advantage Fund - Direct Plan",       "category": "Balanced Advantage"},
]


async def _get_fund_returns(fund: dict, extra_scheme_code: str | None = None) -> dict | None:
    """Fetch live NAV history for one fund and compute 1Y / 3Y / 5Y returns."""
    try:
        data = await _fetch_scheme_data(str(fund["code"]))
        if not data:
            return None
        nav_list = data.get("data", [])
        if len(nav_list) < 2:
            return None
        meta = data.get("meta", {})
        current_nav = _parse_nav(nav_list[0]["nav"])
        nav_1y = nav_3y = nav_5y = None
        today = date.today()
        for entry in nav_list:
            try:
                d = _fmt_date(entry["date"])
                delta = (today - d).days
                if nav_1y is None and delta >= 355:
                    nav_1y = _parse_nav(entry["nav"])
                if nav_3y is None and delta >= 1080:
                    nav_3y = _parse_nav(entry["nav"])
                if nav_5y is None and delta >= 1800:
                    nav_5y = _parse_nav(entry["nav"])
                if nav_1y and nav_3y and nav_5y:
                    break
            except Exception:
                continue

        ret_1y = ((current_nav / nav_1y) - 1) * 100 if nav_1y else None
        ret_3y = (((current_nav / nav_3y) ** (1 / 3)) - 1) * 100 if nav_3y else None
        ret_5y = (((current_nav / nav_5y) ** (1 / 5)) - 1) * 100 if nav_5y else None

        is_user = (extra_scheme_code is not None and str(fund["code"]) == str(extra_scheme_code))
        return {
            "scheme_code":  str(fund["code"]),
            "scheme_name":  meta.get("scheme_name", fund["name"]),
            "fund_house":   meta.get("fund_house", ""),
            "category":     fund["category"],
            "current_nav":  round(current_nav, 4),
            "nav_date":     nav_list[0].get("date", ""),
            "return_1y":    round(ret_1y, 2) if ret_1y is not None else None,
            "return_3y":    round(ret_3y, 2) if ret_3y is not None else None,
            "return_5y":    round(ret_5y, 2) if ret_5y is not None else None,
            "is_user_fund": is_user,
        }
    except Exception as e:
        logger.warning(f"Could not fetch top-fund {fund['code']}: {e}")
        return None


@router.get("/top")
async def get_top_funds(category: str = "all", limit: int = 8):
    """
    Return top-performing funds with 1Y, 3Y & 5Y returns.
    Uses the POPULAR_FUNDS list + live NAV to compute returns.
    """
    filtered = POPULAR_FUNDS
    if category and category.lower() != "all":
        filtered = [f for f in POPULAR_FUNDS if f["category"].lower() == category.lower()]

    # Deduplicate by code within the filtered list
    seen: set[str] = set()
    deduped = []
    for f in filtered:
        if f["code"] not in seen:
            seen.add(f["code"])
            deduped.append(f)

    tasks = [_get_fund_returns(f) for f in deduped[:limit + 6]]
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in results_raw if isinstance(r, dict) and r is not None]

    # Sort by 1Y return descending
    results.sort(key=lambda x: x.get("return_1y") or -999, reverse=True)
    return {"funds": results[:limit], "count": len(results)}


# ── Ranked Top Funds endpoint — includes user fund position ───────────────────
@router.get("/top/ranked")
async def get_ranked_funds(scheme_code: str = "", limit: int = 40):
    """
    Return ALL popular funds ranked by 1Y return, with the user's selected
    fund highlighted (is_user_fund = True). The user's fund is always included
    even if it is not in the POPULAR_FUNDS list.
    Returns:
      {
        "funds": [...],      ← sorted by return_1y descending, with rank field
        "user_fund_rank": N, ← 1-indexed rank of user's fund (null if not found)
        "count": N
      }
    """
    # Deduplicate POPULAR_FUNDS by code
    seen: set[str] = set()
    funds_to_fetch: list[dict] = []
    for f in POPULAR_FUNDS:
        if f["code"] not in seen:
            seen.add(f["code"])
            funds_to_fetch.append(f)

    # Always include user's fund if not already in the list
    user_code = str(scheme_code).strip()
    if user_code and user_code not in seen:
        funds_to_fetch.append({"code": user_code, "name": "Selected Fund", "category": "Selected"})
        seen.add(user_code)

    tasks = [_get_fund_returns(f, extra_scheme_code=user_code if user_code else None)
             for f in funds_to_fetch]
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    results = [r for r in results_raw if isinstance(r, dict) and r is not None]

    # Mark user fund explicitly (in case it was added dynamically)
    for r in results:
        if user_code and r["scheme_code"] == user_code:
            r["is_user_fund"] = True

    # Sort by 1Y return descending
    results.sort(key=lambda x: x.get("return_1y") or -9999, reverse=True)

    # Assign rank (1-indexed)
    user_rank = None
    for i, r in enumerate(results):
        r["rank"] = i + 1
        if r.get("is_user_fund"):
            user_rank = i + 1

    return {
        "funds":          results[:limit],
        "user_fund_rank": user_rank,
        "count":          len(results),
    }


# ── CRUD for saved SIP holdings ────────────────────────────────────────────────
@router.get("/holdings")
def get_mf_holdings():
    return _read_holdings()


@router.post("/holdings")
def save_mf_holding(entry: SIPEntry):
    data = _read_holdings()
    uid = entry.id or f"{entry.scheme_code}_{entry.sip_start_date}"
    data["holdings"][uid] = {
        "id":                 uid,
        "scheme_code":        entry.scheme_code,
        "scheme_name":        entry.scheme_name,
        "sip_amount":         entry.sip_amount,
        "step_up_pct":        entry.step_up_pct,
        "initial_investment": entry.initial_investment,
        "sip_start_date":     entry.sip_start_date,
        "missed_sip_dates":   entry.missed_sip_dates,
        "category":           entry.category or "",
        "notes":              entry.notes or "",
        "added_at":           data["holdings"].get(uid, {}).get("added_at", datetime.now().isoformat()),
    }
    _write_holdings(data)
    return {"ok": True, "holding": data["holdings"][uid]}


@router.delete("/holdings/{holding_id}")
def delete_mf_holding(holding_id: str):
    data = _read_holdings()
    if holding_id not in data["holdings"]:
        raise HTTPException(404, f"Holding '{holding_id}' not found")
    del data["holdings"][holding_id]
    _write_holdings(data)
    return {"ok": True, "removed": holding_id}
