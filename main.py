from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

app = FastAPI(
    title="Quant Finance Workbench API",
    version="1.0.1",
    description="Backend API for Quant & Finance Architecture GPT Action."
)


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


@app.get("/market/test-snapshot")
def test_market_snapshot(
    asset_class: str = "forex",
    symbol: str = "EURUSD",
    timeframe: str = "M5"
):
    return {
        "status": "ok",
        "symbol": symbol,
        "asset_class": asset_class,
        "timeframe": timeframe,
        "trend_state": "not_connected_to_market_data_yet",
        "bias": "unclear",
        "message": "Action connection works. Live market data is not connected yet."
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
def market_snapshot(payload: MarketSnapshotRequest):
    return {
        "status": "ok",
        "symbol": payload.symbol,
        "asset_class": payload.asset_class,
        "timeframe": payload.timeframe,
        "latest_price": "not_available",
        "trend_state": "not_connected_to_market_data_yet",
        "indicator_summary": "Market data engine not yet connected. This endpoint is working as a backend test.",
        "volatility_state": "unknown",
        "spread_state": "unknown",
        "bias": "unclear",
        "warnings": [
            "This is a placeholder response.",
            "Connect real market data before using for trading decisions."
        ],
        "evidence": [
            "API endpoint is live.",
            "Request body was received successfully."
        ]
    }
