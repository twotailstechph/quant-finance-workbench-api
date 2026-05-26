from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
from datetime import datetime

app = FastAPI(
    title="Quant Finance Workbench API",
    version="1.0.0",
    description="Backend API for Quant & Finance Architecture GPT Action."
)

API_KEY = os.getenv("API_KEY", "")


def verify_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Quant Finance Workbench API",
        "message": "Backend is live."
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Quant Finance Workbench API",
        "timestamp": datetime.utcnow().isoformat()
    }


class MarketSnapshotRequest(BaseModel):
    asset_class: str
    symbol: str
    timeframe: str
    lookback_bars: int = 300
    indicators: Optional[List[str]] = ["ema_200", "macd_12_26_9", "rsi_14", "atr_14"]
    include_news_context: bool = False
    timezone: str = "Asia/Manila"


@app.post("/market/snapshot")
def market_snapshot(payload: MarketSnapshotRequest, x_api_key: Optional[str] = Header(None)):
    verify_api_key(x_api_key)

    return {
        "symbol": payload.symbol,
        "timeframe": payload.timeframe,
        "latest_price": None,
        "trend_state": "not_connected_to_market_data_yet",
        "indicator_summary": {
            "note": "Market data engine not yet connected. This endpoint is working as a backend test."
        },
        "volatility_state": "unknown",
        "spread_state": "unknown",
        "bias": "unclear",
        "warnings": [
            "This is a placeholder response. Connect real market data before using for trading decisions."
        ],
        "evidence": [
            "API endpoint is live.",
            "Request body was received successfully."
        ]
    }
