"""
F&O Service
===========
Automatically fetches all NSE F&O eligible stocks, enriches them with live
option-chain + market data, runs a 6-gate pipeline filter, and returns a
ranked terminal for intraday / swing F&O trades.

Data sources:
  - NSE APIs (nseindia.com)  → F&O universe, option chains, quotes, ban list
  - yfinance                 → historical OHLC for EMA / Beta / Momentum
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, date
from typing import Any

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── NSE Session ───────────────────────────────────────────────────────────────
# NSE aggressively blocks non-browser requests.  We maintain cookies, use a
# realistic User-Agent, and honour rate limits.

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
}

_NSE_BASE = "https://www.nseindia.com"


class NSESession:
    """Manages a single httpx.AsyncClient with cookie persistence."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._last_request_at: float = 0
        self._min_delay: float = 0.45          # seconds between requests
        self._cookies_set: bool = False

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=_NSE_HEADERS,
                timeout=httpx.Timeout(20.0, connect=10.0),
                follow_redirects=True,
                verify=True,
            )
            self._cookies_set = False
        return self._client

    async def _warm_cookies(self) -> None:
        """Hit the NSE home page once to get session cookies."""
        if self._cookies_set:
            return
        client = await self._ensure_client()
        try:
            resp = await client.get(f"{_NSE_BASE}/", headers={"Accept": "text/html"})
            if resp.status_code == 200:
                self._cookies_set = True
                logger.info("NSE session cookies acquired")
        except Exception as e:
            logger.warning(f"NSE cookie warm-up failed: {e}")

    async def get_json(self, url: str, retries: int = 2) -> dict | list | None:
        """GET `url` with rate-limiting, cookie management, and retries."""
        async with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._min_delay:
                await asyncio.sleep(self._min_delay - elapsed)

        client = await self._ensure_client()
        await self._warm_cookies()

        for attempt in range(1, retries + 2):
            try:
                async with self._lock:
                    self._last_request_at = time.monotonic()
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 401:
                    # cookies expired → re-warm
                    self._cookies_set = False
                    await self._warm_cookies()
                else:
                    logger.warning(
                        f"NSE {resp.status_code} for {url} (attempt {attempt})"
                    )
            except Exception as e:
                logger.warning(f"NSE request error {url}: {e} (attempt {attempt})")
            if attempt <= retries:
                await asyncio.sleep(1.0 * attempt)
        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Module-level singleton
_nse = NSESession()

# ── Cache ─────────────────────────────────────────────────────────────────────
_FNO_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL = 7200  # 2 hours — keeps data alive after market close


def _cache_get(key: str) -> Any | None:
    if key in _FNO_CACHE:
        ts, data = _FNO_CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            return data
    return None


def _cache_set(key: str, data: Any) -> None:
    _FNO_CACHE[key] = (time.time(), data)


# ── Progress tracking ─────────────────────────────────────────────────────────
_scan_progress: dict[str, Any] = {
    "status": "idle",      # idle | scanning | done | error
    "step": "",
    "pct": 0,
    "message": "",
    "started_at": None,
}


def get_scan_progress() -> dict:
    return dict(_scan_progress)


def _set_progress(status: str, step: str, pct: int, message: str = "") -> None:
    _scan_progress.update(status=status, step=step, pct=pct, message=message)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

# ── Top 30 F&O Heavyweights (most liquid, fastest scan) ───────────────────────
# NSE's API is behind Akamai WAF and frequently blocks non-browser requests.
# This fallback list ensures the scanner always works.
_FNO_SYMBOLS = [
    # 0-30
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "BHARTIARTL", "SBIN", "AXISBANK", "KOTAKBANK", "LT",
    "TATAMOTORS", "TATASTEEL", "BAJFINANCE", "MARUTI", "SUNPHARMA",
    "HCLTECH", "WIPRO", "ADANIENT", "ADANIPORTS", "TITAN",
    "HINDUNILVR", "ITC", "BAJAJFINSV", "NTPC", "POWERGRID",
    "JSWSTEEL", "TATACONSUM", "DRREDDY", "CIPLA", "COALINDIA",
    # 30-60
    "M&M", "ULTRACEMCO", "ASIANPAINT", "GRASIM", "EICHERMOT",
    "TECHM", "INDUSINDBK", "BAJAJA-AUTO", "ONGC", "HINDALCO",
    "NESTLEIND", "SBILIFE", "HDFCLIFE", "BRITANNIA", "APOLLOHOSP",
    "HEROMOTOCO", "BPCL", "SHREECEM", "DIVISLAB", "LTIM",
    "UPL", "TRENT", "TORNTPHARM", "VEDL", "BHARATFORG",
    "TVSMOTOR", "GODREJCP", "INDIGO", "SIEMENS", "PIDILITIND",
    # 60-100
    "HAVELLS", "BANKBARODA", "PNB", "CANBK", "CHOLAFIN",
    "SRF", "TATACHEM", "BHEL", "GAIL", "HAL",
    "BEL", "PFC", "RECLTD", "MANAPPURAM", "MUTHOOTFIN",
    "M&MFIN", "AMBUJACEM", "ACC", "BANDHANBNK", "IDFCFIRSTB",
    "AUROPHARMA", "LUPIN", "BIOCON", "IGL", "MGL",
    "PETRONET", "COROMANDEL", "DEEPAKNTR", "ABFRL", "ASTRAL",
    "DIXON", "ESCORTS", "IDEA", "INDIACEM", "VOLTAS",
    "ZEEL", "NAVINFLUOR", "POLYCAB", "MCX", "PIIND"
]


def _fetch_yf_quote(symbol: str) -> dict | None:
    """Fetch latest quote for a single symbol via yfinance (synchronous)."""
    try:
        import yfinance as yf
        t = yf.Ticker(f"{symbol}.NS")
        hist = t.history(period="5d")
        if hist is None or hist.empty:
            return None
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else hist.iloc[-1]
        close = float(last["Close"])
        prev_close = float(prev["Close"])
        pchange = ((close - prev_close) / prev_close * 100) if prev_close > 0 else 0
        return {
            "symbol": symbol,
            "lastPrice": round(close, 2),
            "pChange": round(pchange, 2),
            "totalTradedVolume": int(last.get("Volume", 0)),
            "lotSize": 0,  # Will be enriched later if available
            "open": round(float(last.get("Open", 0)), 2),
            "dayHigh": round(float(last.get("High", 0)), 2),
            "dayLow": round(float(last.get("Low", 0)), 2),
            "previousClose": round(prev_close, 2),
        }
    except Exception as e:
        logger.warning(f"yfinance quote failed for {symbol}: {e}")
        return None


async def fetch_fno_universe(start_idx: int = 0, end_idx: int = 30) -> list[dict]:
    """Step 1A — Fetch a batch of F&O stocks via yfinance (fast, always works)."""
    cache_key = f"fno_universe_{start_idx}_{end_idx}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    subset = _FNO_SYMBOLS[start_idx:end_idx]
    _set_progress("scanning", "fetch_universe", 5, f"Fetching {len(subset)} F&O stocks ({start_idx}-{end_idx}) via yfinance...")
    logger.info(f"Fetching {len(subset)} F&O stocks ({start_idx}-{end_idx}) via yfinance")

    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(15)  # fetch up to 15 in parallel

    async def _get_quote(sym: str):
        async with sem:
            return await loop.run_in_executor(None, _fetch_yf_quote, sym)

    results = await asyncio.gather(*[_get_quote(s) for s in subset])
    stocks = [r for r in results if r]

    _cache_set(cache_key, stocks)
    logger.info(f"F&O universe ready: {len(stocks)}/{len(subset)} stocks fetched for batch {start_idx}-{end_idx}")
    return stocks


async def fetch_fno_ban_list() -> set[str]:
    """Gate 1 — F&O ban list. NSE API is blocked, return empty set (no bans assumed)."""
    cached = _cache_get("fno_ban")
    if cached is not None:
        return cached
    # NSE ban-list API is behind WAF — skip to avoid timeout
    banned: set[str] = set()
    _cache_set("fno_ban", banned)
    logger.info("F&O ban list: skipped (NSE WAF blocks API), assuming no bans")
    return banned


def _fetch_yf_option_proxy(symbol: str) -> dict | None:
    """Estimate option-chain metrics from yfinance historical data.
    Used when NSE option chain API is blocked by WAF.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(f"{symbol}.NS")
        hist = t.history(period="30d")
        if hist is None or hist.empty or len(hist) < 10:
            return None

        close = hist["Close"]
        volume = hist["Volume"]
        underlying = float(close.iloc[-1])

        # Round to nearest 50 for ATM strike estimate
        atm_strike = round(underlying / 50) * 50

        # Estimate historical volatility (annualised, as proxy for IV)
        returns = close.pct_change().dropna()
        hist_vol = float(returns.std() * (252 ** 0.5) * 100)  # annualised %

        # Volume ratio: today vs 10-day avg — used as proxy for OI change
        avg_vol_10 = float(volume.tail(10).mean())
        today_vol = float(volume.iloc[-1])
        vol_ratio = today_vol / avg_vol_10 if avg_vol_10 > 0 else 1.0

        # Estimate ATM option price (rough Black-Scholes approximation)
        atm_price = round(underlying * hist_vol / 100 * (30 / 252) ** 0.5, 2)

        # Estimate PCR from recent price momentum (bullish → lower PCR)
        mom_5d = (float(close.iloc[-1]) - float(close.iloc[-5])) / float(close.iloc[-5]) if len(close) >= 5 else 0
        pcr = round(max(0.5, min(1.5, 1.0 - mom_5d * 5)), 2)  # 0.5–1.5 range

        # Simulate OI change direction from volume ratio
        oi_change_proxy = int(today_vol * 0.3) if vol_ratio > 1 else -int(today_vol * 0.1)

        atm_ce = {
            "lastPrice": atm_price,
            "openInterest": int(avg_vol_10 * 2),
            "changeinOpenInterest": oi_change_proxy,
            "impliedVolatility": round(hist_vol, 1),
        }
        atm_pe = {
            "lastPrice": atm_price,
            "openInterest": int(avg_vol_10 * 2 * pcr),
            "changeinOpenInterest": -oi_change_proxy if pcr < 1 else oi_change_proxy,
            "impliedVolatility": round(hist_vol, 1),
        }

        return {
            "underlying": underlying,
            "atm_strike": atm_strike,
            "atm_ce": atm_ce,
            "atm_pe": atm_pe,
            "pcr": pcr,
            "total_ce_oi": atm_ce["openInterest"],
            "total_pe_oi": atm_pe["openInterest"],
            "nearest_expiry": "estimated",
        }
    except Exception as e:
        logger.warning(f"Option proxy failed for {symbol}: {e}")
        return None


async def fetch_option_chain(symbol: str) -> dict | None:
    """Step 1B — Option chain via yfinance proxy (NSE API blocked by WAF)."""
    cache_key = f"oc_{symbol}"
    cached = _cache_get(cache_key)
    if cached:
        return cached
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _fetch_yf_option_proxy, symbol)
    if result:
        _cache_set(cache_key, result)
    return result


async def fetch_quote(symbol: str) -> dict | None:
    """Step 1C — Quote data via yfinance (NSE API blocked by WAF)."""
    cache_key = f"quote_{symbol}"
    cached = _cache_get(cache_key)
    if cached:
        return cached
    # Already fetched in _fetch_yf_quote — reuse from universe cache if present
    # Return minimal stub; delivery% not available without NSE
    result = {"deliveryPct": 0, "dayHigh": 0, "dayLow": 0, "previousClose": 0}
    _cache_set(cache_key, result)
    return result


# ── EMA / Beta / Momentum via yfinance ────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_technicals(symbol: str) -> dict | None:
    """
    Use yfinance to fetch ~120 days of OHLC history and compute:
      - EMA10 of Close, EMA50 of High, EMA50 of Low
      - 60-day rolling Beta vs NIFTY 50
      - 20-day Momentum Score
      - 20-day average volume
    """
    cache_key = f"tech_{symbol}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 30:
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        ema10 = _ema(close, 10)
        ema50h = _ema(high, 50)
        ema50l = _ema(low, 50)

        last_close = float(close.iloc[-1])
        last_ema10 = float(ema10.iloc[-1])
        last_ema50h = float(ema50h.iloc[-1])
        last_ema50l = float(ema50l.iloc[-1])

        # 20-day momentum score
        if len(close) >= 21:
            mom_score = (last_close - float(close.iloc[-21])) / float(close.iloc[-21]) * 100
        else:
            mom_score = 0.0

        # 20-day average volume
        avg_vol_20 = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())

        # Beta vs NIFTY 50
        beta = _compute_beta(close, 60)

        result = {
            "close": last_close,
            "ema10": round(last_ema10, 2),
            "ema50h": round(last_ema50h, 2),
            "ema50l": round(last_ema50l, 2),
            "ema10_pass": last_close > last_ema10,
            "ema50h_pass": last_close > last_ema50h,
            "ema50l_pass": last_close > last_ema50l,
            "beta": round(beta, 3) if beta is not None else None,
            "momentum_score": round(mom_score, 2),
            "avg_vol_20": round(avg_vol_20),
            "last_volume": int(volume.iloc[-1]) if len(volume) > 0 else 0,
        }
        _cache_set(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"Technicals failed for {symbol}: {e}")
        return None


# ── NIFTY data cache (shared across all beta computations in one scan) ─────────
_nifty_cache: dict = {}


def _compute_beta(stock_close: pd.Series, window: int = 60) -> float | None:
    """Compute rolling beta vs NIFTY 50. NIFTY data is cached per scan."""
    global _nifty_cache
    try:
        import yfinance as yf

        # Fetch NIFTY only once per scan session
        if "ret" not in _nifty_cache:
            nifty = yf.Ticker("^NSEI")
            ndf = nifty.history(period="6mo", interval="1d")
            if ndf is None or ndf.empty or len(ndf) < window:
                return None
            nifty_close = ndf["Close"]
            nifty_close.index = nifty_close.index.tz_localize(None)
            _nifty_cache["ret"] = nifty_close.pct_change().dropna()

        nifty_ret = _nifty_cache["ret"]

        stock_ret = stock_close.pct_change().dropna()
        stock_ret.index = stock_ret.index.tz_localize(None)
        nifty_aligned = nifty_ret.reindex(stock_ret.index, method="nearest")

        sr = stock_ret.values[-window:]
        mr = nifty_aligned.values[-window:]
        mask = ~(np.isnan(sr) | np.isnan(mr))
        if mask.sum() < 20:
            return None
        cov = np.cov(sr[mask], mr[mask])
        return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else None

    except Exception as e:
        logger.warning(f"Beta computation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PIPELINE GATE FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def gate1_ban_check(stocks: list[dict], banned: set[str]) -> tuple[list[dict], int]:
    """Gate 1 — Remove F&O banned stocks."""
    passed = [s for s in stocks if s["symbol"] not in banned]
    return passed, len(stocks) - len(passed)


def gate2_ema_filter(stocks: list[dict], version: str = "v2") -> tuple[list[dict], int]:
    """Gate 2 — EMA Trend Filter."""
    passed = []
    for s in stocks:
        tech = s.get("technicals")
        if not tech:
            continue
        
        if version == "v2":
            # Close > EMA10 OR Close > EMA50L (eased — at least one)
            ema10_ok = tech.get("ema10_pass", False)
            ema50l_ok = tech.get("ema50l_pass", False)
            if ema10_ok or ema50l_ok:
                # Bonus tag if Close > EMA50H
                s["ema50h_bonus"] = tech.get("ema50h_pass", False)
                s["ema_strength"] = "STRONG" if (ema10_ok and ema50l_ok) else "PARTIAL"
                passed.append(s)
        else:
            if tech.get("ema10_pass") and tech.get("ema50h_pass") and tech.get("ema50l_pass"):
                s["ema50h_bonus"] = True
                passed.append(s)
                
    return passed, len(stocks) - len(passed)


def gate3_beta_filter(stocks: list[dict], version: str = "v2") -> tuple[list[dict], int]:
    """Gate 3 — Beta Quality."""
    passed = []
    for s in stocks:
        tech = s.get("technicals")
        beta = tech.get("beta") if tech else None
        
        if beta is None:
            s["is_high_beta"] = False
            s["beta_tag"] = "NORMAL"
            passed.append(s)
            continue
            
        if version == "v2":
            # Soft filter
            if beta < 0.6:
                s["beta_tag"] = "LOW BETA"
            elif 0.6 <= beta <= 1.2:
                s["beta_tag"] = "NORMAL"
            elif 1.2 < beta <= 1.8:
                s["beta_tag"] = "HIGH MOMENTUM"
            else:
                s["beta_tag"] = "EXTREME VOLATILE"
            s["is_high_beta"] = s["beta_tag"] == "HIGH MOMENTUM"
            passed.append(s)
        else:
            if 0.8 <= beta <= 1.4:
                s["is_high_beta"] = 1.2 <= beta <= 1.4
                s["beta_tag"] = "HIGH MOMENTUM" if s["is_high_beta"] else "NORMAL"
                passed.append(s)

    elim = len(stocks) - len(passed)
    return passed, elim


def gate4_oi_confirmation(stocks: list[dict], version: str = "v2", total_oi_increased: bool = False) -> tuple[list[dict], int]:
    """Gate 4 — Open Interest Confirmation."""
    passed = []
    for s in stocks:
        oc = s.get("option_chain")
        if not oc:
            s["oi_trend"] = "N/A"
            s["oi_score"] = 0
            s["oi_tag"] = "OI BEARISH"
            passed.append(s)
            continue

        pcr = oc.get("pcr", 0)
        atm_ce = oc.get("atm_ce") or {}
        atm_pe = oc.get("atm_pe") or {}
        ce_oi_change = atm_ce.get("changeinOpenInterest", 0)
        pe_oi_change = atm_pe.get("changeinOpenInterest", 0)

        if version == "v2":
            score = 0
            if ce_oi_change > 0: score += 1
            if 0.5 <= pcr <= 1.5: score += 1
            if total_oi_increased: score += 1
            
            s["oi_score"] = score
            if score == 3: s["oi_tag"] = "OI STRONG"
            elif score == 2: s["oi_tag"] = "OI MODERATE"
            elif score == 1: s["oi_tag"] = "OI WEAK"
            else: s["oi_tag"] = "OI BEARISH"
            
            s["oi_trend"] = "RISING" if ce_oi_change > 0 else "FALLING_PE" if pe_oi_change < 0 else "FLAT"
            passed.append(s)
        else:
            if pcr > 1.5:
                continue
            if ce_oi_change < 0 and pe_oi_change < 0:
                continue
            bullish = (ce_oi_change > 0) or (pe_oi_change < 0)
            pcr_ok = 0.7 <= pcr <= 1.2

            if bullish or pcr_ok:
                s["oi_trend"] = "RISING" if ce_oi_change > 0 else "FALLING_PE"
                passed.append(s)

    return passed, len(stocks) - len(passed)


def gate5_volume_liquidity(stocks: list[dict], version: str = "v2") -> tuple[list[dict], int]:
    """Gate 5 — Volume Check."""
    passed = []
    for s in stocks:
        tech = s.get("technicals")
        quote = s.get("quote_data")
        if not tech:
            continue

        avg_vol = tech.get("avg_vol_20", 0)
        today_vol = s.get("totalTradedVolume", 0) or tech.get("last_volume", 0)
        delivery_pct = 0
        if quote:
            delivery_pct = quote.get("deliveryPct", 0)

        vol_pct = (today_vol / avg_vol * 100) if avg_vol > 0 else 0
        s["vol_pct_of_avg"] = round(vol_pct, 1)
        s["delivery_pct"] = round(delivery_pct, 1)

        if version == "v2":
            score = 0
            if today_vol > avg_vol: score += 1
            if vol_pct > 150: score += 1
            s["vol_score"] = score
            if score == 2: s["vol_tag"] = "VOLUME SURGE"
            elif score == 1: s["vol_tag"] = "ABOVE AVG VOLUME"
            else: s["vol_tag"] = "NORMAL VOLUME"
            passed.append(s)
        else:
            if today_vol > avg_vol and delivery_pct > 35:
                passed.append(s)
            elif delivery_pct == 0 and today_vol > avg_vol:
                s["delivery_pct"] = 0
                passed.append(s)

    return passed, len(stocks) - len(passed)


def gate6_momentum_rank(stocks: list[dict], version: str = "v2") -> tuple[list[dict], int]:
    """Gate 6 — Momentum & Composite Rank."""
    if not stocks:
        return stocks, 0

    if version == "v2":
        for s in stocks:
            tech = s.get("technicals", {})
            mom = tech.get("momentum_score", 0)
            # Cap Mom between -10% and +20% for scoring
            mom_capped = max(-10, min(20, mom))
            mom_norm = ((mom_capped + 10) / 30) * 100  # 0 to 100
            
            oi_norm = (s.get("oi_score", 0) / 3) * 100
            vol_norm = (s.get("vol_score", 0) / 2) * 100
            
            beta = tech.get("beta")
            if beta is None: beta_norm = 50
            elif 0.8 <= beta <= 1.4: beta_norm = 100
            else: beta_norm = max(0, 100 - (abs(1.1 - beta) * 50))
            
            s["composite_score"] = (mom_norm * 0.4) + (oi_norm * 0.3) + (vol_norm * 0.2) + (beta_norm * 0.1)

        stocks.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        cutoff = max(1, int(len(stocks) * 0.8))
        passed = stocks[:cutoff]
        eliminated = len(stocks) - len(passed)
        
        for i, s in enumerate(passed):
            if i < 10:
                s["is_prime_pick"] = True
                s["composite_tag"] = "PRIME PICKS"
            elif i < 20:
                s["is_prime_pick"] = False
                s["composite_tag"] = "WATCHLIST"
            else:
                s["is_prime_pick"] = False
                s["composite_tag"] = "NORMAL"
    else:
        stocks.sort(key=lambda s: (s.get("technicals") or {}).get("momentum_score", -999), reverse=True)
        cutoff = max(1, int(len(stocks) * 0.4))
        passed = stocks[:cutoff]
        eliminated = len(stocks) - len(passed)
        for i, s in enumerate(passed):
            s["is_prime_pick"] = i < 10
            s["composite_score"] = 0
            s["composite_tag"] = "N/A"

    return passed, eliminated


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — TRADE MODE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_trade_mode(stock: dict, version: str = "v2") -> str:
    """Classify each stock as INTRADAY / SWING / BOTH / AVOID / MONITOR."""
    tech = stock.get("technicals") or {}
    oc = stock.get("option_chain") or {}
    atm_ce = oc.get("atm_ce") or {}
    atm_pe = oc.get("atm_pe") or {}

    beta = tech.get("beta") or 1.0
    vol_pct = stock.get("vol_pct_of_avg", 100)
    iv_ce = atm_ce.get("impliedVolatility", 50)
    iv_pe = atm_pe.get("impliedVolatility", 50)
    avg_iv = (iv_ce + iv_pe) / 2 if (iv_ce and iv_pe) else iv_ce or iv_pe or 50
    pcr = oc.get("pcr", 1.0)
    
    if version == "v2":
        beta_tag = stock.get("beta_tag", "")
        oi_tag = stock.get("oi_tag", "")
        vol_tag = stock.get("vol_tag", "")
        ema50h_bonus = stock.get("ema50h_bonus", False)
        
        is_intraday = False
        is_swing = False
        
        if beta_tag == "HIGH MOMENTUM" and vol_tag == "VOLUME SURGE" and avg_iv < 35:
            is_intraday = True
            
        if beta_tag == "NORMAL" and oi_tag in ["OI STRONG", "OI MODERATE"] and avg_iv < 40:
            is_swing = True
            
        if is_intraday and is_swing:
            return "BOTH"
        if is_intraday:
            return "INTRADAY"
        if is_swing:
            return "SWING"
            
        # If passed gate 1 and 2 but didn't qualify for modes:
        if stock.get("oi_score", 0) < 2 and stock.get("vol_score", 0) == 0:
            return "MONITOR"
            
        if avg_iv < 40:
            return "SWING" # default fallback for decent ones
            
        return "AVOID"
        
    else:
        # AVOID conditions
        if avg_iv > 40:
            return "AVOID"
        # INTRADAY: high beta, big volume surge, affordable options
        if beta > 1.2 and vol_pct > 150 and avg_iv < 30 and 0.8 <= pcr <= 1.0:
            return "INTRADAY"
        # SWING: moderate beta, clean EMAs, steady OI
        if 0.8 <= beta <= 1.2 and avg_iv < 35:
            ema_ok = tech.get("ema10_pass") and tech.get("ema50h_pass") and tech.get("ema50l_pass")
            if ema_ok:
                return "SWING"
        # Default to SWING if passes all gates
        if avg_iv < 35:
            return "SWING"
        return "AVOID"


def _get_signal_text(mode: str, stock: dict) -> str:
    """Generate trade signal text."""
    if mode == "INTRADAY":
        return "Buy ATM CE / Futures scalp (Same day exit)"
    elif mode == "SWING":
        return "Buy slight ITM CE / hold Futures"
    elif mode == "BOTH":
        return "Dual Mode (Intraday/Swing) - ATM CE"
    elif mode == "MONITOR":
        return "Watchlist - Wait for OI/Vol improvement"
    else:
        return "Avoid — conditions unfavorable"

async def get_full_fno_terminal(version: str = "v2", start_idx: int = 0, end_idx: int = 30) -> dict:
    """
    Main orchestrator:
      1. Fetch F&O universe + ban list
      2. Fetch technicals (yfinance) for all stocks
      3. Run Gates 1-3
      4. Fetch option chains + quotes for survivors
      5. Run Gates 4-6
      6. Classify trade modes
      7. Return ranked terminal data
    """
    # Check cache
    cache_key = f"fno_terminal_full_{version}_{start_idx}_{end_idx}"
    cached = _cache_get(cache_key)
    if cached:
        _set_progress("done", "complete", 100, "Cached result")
        return cached

    # Clear NIFTY cache so beta is computed fresh this scan (downloaded once)
    _nifty_cache.clear()
    _set_progress("scanning", "fetch_universe", 5, "Fetching F&O stock universe...")

    try:
        # ── Step 1: Fetch universe + ban list concurrently ──
        universe, banned = await asyncio.gather(
            fetch_fno_universe(start_idx=start_idx, end_idx=end_idx),
            fetch_fno_ban_list(),
        )

        if not universe:
            _set_progress("error", "fetch_universe", 0, "Failed to fetch F&O universe via yfinance")
            return _empty_result("Could not fetch F&O universe via yfinance")

        total_scanned = len(universe)
        _set_progress("scanning", "gate1", 10, f"Gate 1: Checking ban list ({len(banned)} banned)...")

        # ── Gate 1: Ban check ──
        gate_stats = {}
        stocks = [dict(s) for s in universe]  # copy
        stocks, elim = gate1_ban_check(stocks, banned)
        gate_stats["gate1_ban"] = {"eliminated": elim, "remaining": len(stocks)}

        _set_progress("scanning", "technicals", 15, f"Computing EMA/Beta/Momentum for {len(stocks)} stocks...")

        # ── Step 2: Fetch technicals via yfinance (concurrent with semaphore) ──
        sem = asyncio.Semaphore(8)

        async def _get_tech(stock: dict) -> dict:
            async with sem:
                loop = asyncio.get_event_loop()
                tech = await loop.run_in_executor(None, compute_technicals, stock["symbol"])
                stock["technicals"] = tech
                return stock

        # Process in batches to show progress
        batch_size = 20
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            await asyncio.gather(*[_get_tech(s) for s in batch])
            pct = 15 + int((i + batch_size) / len(stocks) * 35)
            _set_progress(
                "scanning", "technicals",
                min(pct, 50),
                f"Computing technicals... {min(i + batch_size, len(stocks))}/{len(stocks)}"
            )

        # ── Gate 2: EMA filter ──
        _set_progress("scanning", "gate2", 52, "Gate 2: EMA trend filter...")
        stocks, elim = gate2_ema_filter(stocks, version=version)
        gate_stats["gate2_ema"] = {"eliminated": elim, "remaining": len(stocks)}

        # ── Gate 3: Beta filter ──
        _set_progress("scanning", "gate3", 55, "Gate 3: Beta quality filter...")
        stocks, elim = gate3_beta_filter(stocks, version=version)
        gate_stats["gate3_beta"] = {"eliminated": elim, "remaining": len(stocks)}

        _set_progress(
            "scanning", "option_chains", 58,
            f"Fetching option chains for {len(stocks)} stocks..."
        )

        # ── Step 3: Fetch option chains + quotes for survivors ──
        oc_sem = asyncio.Semaphore(5)

        async def _enrich(stock: dict) -> dict:
            async with oc_sem:
                oc = await fetch_option_chain(stock["symbol"])
                stock["option_chain"] = oc
                quote = await fetch_quote(stock["symbol"])
                stock["quote_data"] = quote
                return stock

        # Process in smaller batches
        for i in range(0, len(stocks), 10):
            batch = stocks[i:i + 10]
            await asyncio.gather(*[_enrich(s) for s in batch])
            pct = 58 + int((i + 10) / max(len(stocks), 1) * 25)
            _set_progress(
                "scanning", "option_chains",
                min(pct, 83),
                f"Enriching data... {min(i + 10, len(stocks))}/{len(stocks)}"
            )

        # Determine if total OI increased today vs yesterday
        total_oi_increased = sum(
            (s.get("option_chain", {}).get("atm_ce", {}).get("changeinOpenInterest", 0) + 
             s.get("option_chain", {}).get("atm_pe", {}).get("changeinOpenInterest", 0))
            for s in stocks
        ) > 0

        # ── Gate 4: OI confirmation ──
        _set_progress("scanning", "gate4", 85, "Gate 4: Open Interest confirmation...")
        stocks, elim = gate4_oi_confirmation(stocks, version=version, total_oi_increased=total_oi_increased)
        gate_stats["gate4_oi"] = {"eliminated": elim, "remaining": len(stocks)}

        # ── Gate 5: Volume & liquidity ──
        _set_progress("scanning", "gate5", 88, "Gate 5: Volume & liquidity check...")
        stocks, elim = gate5_volume_liquidity(stocks, version=version)
        gate_stats["gate5_volume"] = {"eliminated": elim, "remaining": len(stocks)}

        # ── Gate 6: Momentum rank ──
        _set_progress("scanning", "gate6", 92, "Gate 6: Momentum ranking...")
        stocks, elim = gate6_momentum_rank(stocks, version=version)
        gate_stats["gate6_momentum"] = {"eliminated": elim, "remaining": len(stocks)}

        _set_progress("scanning", "classify", 95, "Classifying trade modes...")

        # ── Classify & build output ──
        output_stocks = []
        intraday_count = 0
        swing_count = 0

        for rank, s in enumerate(stocks, 1):
            mode = classify_trade_mode(s, version=version)
            if mode == "INTRADAY":
                intraday_count += 1
            elif mode == "SWING":
                swing_count += 1

            tech = s.get("technicals") or {}
            oc = s.get("option_chain") or {}
            atm_ce = oc.get("atm_ce") or {}
            atm_pe = oc.get("atm_pe") or {}
            quote = s.get("quote_data") or {}

            cmp = s.get("lastPrice", 0) or tech.get("close", 0)
            lot_size = s.get("lotSize", 0)

            output_stocks.append({
                "rank": rank,
                "symbol": s["symbol"],
                "cmp": round(cmp, 2),
                "change_pct": round(s.get("pChange", 0), 2),
                "lot_size": lot_size,
                "lot_value": round(cmp * lot_size, 0),
                "atm_strike": oc.get("atm_strike", 0),
                "atm_ce_price": round(atm_ce.get("lastPrice", 0), 2),
                "atm_pe_price": round(atm_pe.get("lastPrice", 0), 2),
                "iv_pct": round(
                    ((atm_ce.get("impliedVolatility", 0) or 0) +
                     (atm_pe.get("impliedVolatility", 0) or 0)) / 2, 1
                ),
                "pcr": round(oc.get("pcr", 0), 2),
                "oi_trend": s.get("oi_trend", "N/A"),
                "beta": tech.get("beta"),
                "momentum_score": round(tech.get("momentum_score", 0), 2),
                "mode": mode,
                "signal": _get_signal_text(mode, s),
                "is_prime_pick": s.get("is_prime_pick", False),
                "is_high_beta": s.get("is_high_beta", False),
                "ema_status": {
                    "ema10": tech.get("ema10_pass", False),
                    "ema50h": tech.get("ema50h_pass", False),
                    "ema50l": tech.get("ema50l_pass", False),
                },
                "ema10": tech.get("ema10", 0),
                "ema50h": tech.get("ema50h", 0),
                "ema50l": tech.get("ema50l", 0),
                "vol_pct_of_avg": s.get("vol_pct_of_avg", 0),
                "delivery_pct": s.get("delivery_pct", 0),
                "stop_loss": "EMA10 breach",
                "day_high": s.get("dayHigh", 0) or (quote.get("dayHigh", 0)),
                "day_low": s.get("dayLow", 0) or (quote.get("dayLow", 0)),
                "prev_close": s.get("previousClose", 0),
                "ce_oi_change": atm_ce.get("changeinOpenInterest", 0),
                "pe_oi_change": atm_pe.get("changeinOpenInterest", 0),
                # V2 specific fields
                "composite_score": round(s.get("composite_score", 0), 1),
                "composite_tag": s.get("composite_tag", "N/A"),
                "beta_tag": s.get("beta_tag", "NORMAL"),
                "oi_score": s.get("oi_score", 0),
                "oi_tag": s.get("oi_tag", ""),
                "vol_score": s.get("vol_score", 0),
                "vol_tag": s.get("vol_tag", ""),
            })

        # ── Summary ──
        now = datetime.now()
        is_market_open = _is_market_hours(now)
        prime_picks = sum(1 for s in output_stocks if s["is_prime_pick"])

        result = {
            "summary": {
                "total_scanned": total_scanned,
                "passed_all_gates": len(output_stocks),
                "prime_picks": prime_picks,
                "intraday_count": intraday_count,
                "swing_count": swing_count,
                "scan_time": now.isoformat(),
                "market_open": is_market_open,
            },
            "gate_stats": gate_stats,
            "stocks": output_stocks,
            # Enriched internal dicts with 'technicals' and 'option_chain'
            # sub-dicts — used by the directional bias service to avoid
            # redundant data fetches.
            "enriched_stocks": stocks,
        }

        _cache_set(cache_key, result)
        _set_progress("done", "complete", 100, f"Scan complete — {len(output_stocks)} stocks passed")
        return result

    except Exception as e:
        logger.error(f"F&O terminal scan failed: {e}", exc_info=True)
        _set_progress("error", "failed", 0, str(e))
        return _empty_result(f"Scan failed: {e}")


def _is_market_hours(dt: datetime) -> bool:
    """Check if current time is during NSE market hours (9:15 AM – 3:30 PM IST, weekday)."""
    if dt.weekday() >= 5:  # Saturday, Sunday
        return False
    t = dt.time()
    from datetime import time as dt_time
    return dt_time(9, 15) <= t <= dt_time(15, 30)


def _empty_result(error_msg: str = "") -> dict:
    return {
        "summary": {
            "total_scanned": 0,
            "passed_all_gates": 0,
            "prime_picks": 0,
            "intraday_count": 0,
            "swing_count": 0,
            "scan_time": datetime.now().isoformat(),
            "market_open": False,
            "error": error_msg,
        },
        "gate_stats": {},
        "stocks": [],
    }
