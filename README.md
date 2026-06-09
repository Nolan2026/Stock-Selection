# NSE Signal — Stock Analysis FastAPI App

EMA gap analysis + ML signal generation for NSE stocks.
Fetches live data from Yahoo Finance, runs your trained model, and serves PDF reports.

---

## Project Structure

```
Swing_Trade/
├── app/
│   ├── main.py                         ← FastAPI backend (core routes + analysis logic)
│   ├── routers/
│   │   ├── valuation_router.py         ← Stock valuation analysis endpoints
│   │   ├── momentum_router.py          ← Momentum scanner endpoints
│   │   ├── portfolio_router.py         ← Portfolio management endpoints
│   │   └── mf_router.py               ← Mutual fund SIP tracker endpoints
│   ├── services/
│   │   ├── valuation_service.py        ← Valuation calculations & PDF generation
│   │   ├── momentum_service.py         ← Momentum scanning logic
│   │   └── portfolio_report_service.py ← Portfolio report generation
│   └── models/
│       └── valuation_models.py         ← Pydantic models for valuation
├── models/
│   └── stock_model.pkl                 ← YOUR TRAINED MODEL (not tracked in git)
├── static/
│   └── index.html                      ← Frontend UI (single-page app)
├── pine/
│   ├── swing_trade_signal.pine         ← TradingView Pine Script (signal)
│   └── swing_trade_projection.pine     ← TradingView Pine Script (projection)
├── data/                               ← Auto-created runtime data (gitignored)
│   ├── search_history.json
│   ├── portfolio.json
│   ├── mf_holdings.json
│   ├── nse_master.json
│   └── reports/                        ← Generated PDF reports
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Step 1 — Train your model (Google Colab)

1. Open Google Colab
2. Run the Cell 17 training script
3. Upload your NSE 3-year stock CSV when prompted
4. Download the generated `stock_model.pkl`

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Place your model

```bash
cp stock_model.pkl models/stock_model.pkl
```

### Step 4 — Run the app

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5 — Open in browser

```
http://localhost:8000
```

---

## API Endpoints

### Core Analysis

| Method | URL | Description |
|--------|-----|-------------|
| GET  | `/` | Frontend UI |
| GET  | `/api/health` | Check if model is loaded |
| POST | `/api/analyze` | Analyze one or more stocks |
| GET  | `/api/history` | Get recent search history |
| DELETE | `/api/history/{symbol}` | Remove from history |
| GET  | `/api/download/{symbol}` | Download PDF report |
| GET  | `/api/models` | List available ML models |

### Valuation Analysis

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/valuation` | Metric-by-metric valuation |
| POST | `/api/valuation/compare` | Side-by-side comparison |
| POST | `/api/valuation/download_pdf` | Download valuation PDF |
| POST | `/api/valuation/download_master_pdf` | Full master report PDF |

### Momentum Scanner

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/momentum/scan` | Scan stocks for momentum signals |

### Portfolio

| Method | URL | Description |
|--------|-----|-------------|
| GET  | `/api/portfolio/holdings` | Get portfolio holdings |
| POST | `/api/portfolio/holdings` | Add/update holdings |
| POST | `/api/portfolio/report` | Generate portfolio report |

### Mutual Fund SIP Tracker

| Method | URL | Description |
|--------|-----|-------------|
| GET  | `/api/mf/search?q=...` | Search MF schemes |
| GET  | `/api/mf/{code}/nav` | Get latest NAV |
| POST | `/api/mf/sip/calculate` | Compute SIP returns |
| POST | `/api/mf/sip/compare` | Compare SIP across funds |
| GET  | `/api/mf/holdings` | Get saved MF SIPs |

### POST /api/analyze — Example

```json
{
  "symbols": ["ASHOKLEY", "HPCL", "TCS"],
  "period": "3y"
}
```

Response includes for each stock:
- Signal: STRONG BUY / BUY / WATCH / AVOID
- Score: 0–8 (gap + RSI + MACD + volume)
- Entry price, stop loss, 1-month target
- 90-day price history for sparkline
- PDF download ready flag

---

## How to Use the UI

1. **Type stock symbols** in the search bar (comma-separated)
   - NSE symbols: `ASHOKLEY`, `HPCL`, `MOTHERSON`, `TCS`, `INFY`
2. **Click Analyze** — data fetched live from Yahoo Finance
3. **Read the signal card** — score, entry/stop/target, 4 filter breakdown
4. **Click Download PDF** — full analysis report
5. **Recent stocks** saved below search bar — click to re-analyze

---

## Signal Logic

```
Signal = GO/WAIT/AVOID per filter:

  GAP_FILTER:  EMA10 - EMA50_HIGH (in ATR units)
               ≥ +1.0 ATR → GO  |  +0.3–1.0 → WAIT  |  < 0.3 → AVOID

  RSI_FILTER:  RSI(14)
               ≤ 60 → GO  |  60–70 → WAIT  |  > 70 → AVOID

  MACD_FILTER: MACD Histogram
               > 0 + cross → GO  |  > 0 → WAIT  |  ≤ 0 → AVOID

  VOL_FILTER:  Volume / 20d avg
               ≥ 1.2× → GO  |  0.8–1.2× → WAIT  |  < 0.8× → AVOID

  TOTAL SCORE: 0–8
               7–8 = STRONG BUY  |  5–6 = BUY  |  3–4 = WATCH  |  0–2 = AVOID
```

---

## Updating the Model

To retrain with new data or a different stock:
1. Run the Cell 17 training script again with updated CSV
2. Replace `models/stock_model.pkl`
3. Restart uvicorn — model reloads automatically

---

## Notes

- Yahoo Finance `.NS` suffix is added automatically for NSE stocks
- Search history is saved in `data/search_history.json` (last 20 searches)
- PDF reports saved in `data/reports/` — one per symbol, overwritten on re-analyze
- All data files in `data/` are gitignored — they contain runtime/personal data
- Model files in `models/` are gitignored — place your trained `.pkl` manually
