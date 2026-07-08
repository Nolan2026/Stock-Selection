"""
F&O Router
==========
REST API endpoints for the F&O trading terminal.

Endpoints:
  GET  /api/fno/scan       → full scan — returns ranked terminal data
  GET  /api/fno/progress   → scan progress (polling endpoint)
  GET  /api/fno/status     → quick status (last scan time, stock count)
"""

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
import logging

from app.services.fno_service import (
    get_full_fno_terminal,
    get_scan_progress,
    _cache_get,
    _FNO_CACHE,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/fno", tags=["fno"])

# Track the background scan task so we don't double-launch
_scan_task: asyncio.Task | None = None


@router.get("/scan")
async def fno_scan(version: str = "v2", start_idx: int = 0, end_idx: int = 30):
    """
    Trigger a full F&O universe scan.
    Returns the ranked terminal data directly.
    """
    global _scan_task

    # If there's already a cached result, return immediately
    cache_key = f"fno_terminal_full_{version}_{start_idx}_{end_idx}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    # If a scan is currently running in the background, wait for it
    if _scan_task and not _scan_task.done():
        try:
            await _scan_task
        except asyncio.CancelledError:
            logger.warning("F&O scan was cancelled by the user")
            return JSONResponse(
                status_code=500,
                content={"error": "Scan cancelled", "summary": {"total_scanned": 0, "passed_all_gates": 0}},
            )
        except Exception as e:
            pass # the exception is logged elsewhere, we'll try to get cache or run again

    # Check cache again just in case the task succeeded
    cached = _cache_get(cache_key)
    if cached:
        return cached

    # Run scan if it wasn't running
    try:
        result = await get_full_fno_terminal(version=version, start_idx=start_idx, end_idx=end_idx)
        return result
    except asyncio.CancelledError:
        logger.warning("F&O scan was cancelled by the user")
        return JSONResponse(
            status_code=500,
            content={"error": "Scan cancelled", "summary": {"total_scanned": 0, "passed_all_gates": 0}},
        )
    except Exception as e:
        logger.error(f"F&O scan failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "summary": {"total_scanned": 0, "passed_all_gates": 0}},
        )


@router.get("/scan/start")
async def fno_scan_start(background_tasks: BackgroundTasks, version: str = "v2", force: bool = False, start_idx: int = 0, end_idx: int = 30):
    """
    Start a scan in the background.  The frontend can poll /progress
    to track progress, then call /scan to get the final result.
    If force=true, clears all cache before scanning.
    """
    global _scan_task

    # If a scan is already running, don't start another
    progress = get_scan_progress()
    if progress["status"] == "scanning":
        return {"status": "already_scanning", "progress": progress}

    # Clear cache if forced
    if force:
        _FNO_CACHE.clear()

    # Launch in background
    async def _run_scan():
        await get_full_fno_terminal(version=version, start_idx=start_idx, end_idx=end_idx)

    _scan_task = asyncio.create_task(_run_scan())
    return {"status": "started", "message": "Scan started in background"}


@router.get("/scan/stop")
async def fno_scan_stop():
    """Stop a currently running scan."""
    global _scan_task
    if _scan_task and not _scan_task.done():
        _scan_task.cancel()
        from app.services.fno_service import _set_progress
        _set_progress("error", "cancelled", 0, "Scan cancelled by user")
        return {"status": "stopped", "message": "Scan stopped"}
    return {"status": "not_running", "message": "No scan is currently running"}


@router.get("/progress")
async def fno_progress():
    """Get current scan progress (for polling)."""
    return get_scan_progress()


@router.get("/status")
async def fno_status(version: str = "v2"):
    """Quick status: was there a recent scan? How many stocks?"""
    cache_key = f"fno_terminal_full_{version}"
    cached = _cache_get(cache_key)
    if cached:
        return {
            "has_data": True,
            "summary": cached.get("summary", {}),
            "stock_count": len(cached.get("stocks", [])),
        }
    return {
        "has_data": False,
        "summary": {},
        "stock_count": 0,
    }
