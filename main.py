from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

app = FastAPI(
    title="Quant Finance Workbench API",
    version="1.0.2",
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


@app.get("/market/snapshot")
def get_market_snapshot(
    asset_class: str = "forex",
    symbol: str = "EURUSD",
    timeframe: str = "M5",
    lookback_bars: int = 300,
    indicators: str = "ema_200,macd_12_26_9,rsi_14,atr_14",
    include_news_context: bool = False,
    timezone: str = "Asia/Manila"
):
    indicator_list = [item.strip() for item in indicators.split(",") if item.strip()]

    return {
        "status": "ok",
        "connection_status": "backend_reachable",
        "live_market_data": "not_connected_yet",
        "symbol": symbol,
        "asset_class": asset_class,
        "timeframe": timeframe,
        "lookback_bars": lookback_bars,
        "requested_indicators": indicator_list,
        "include_news_context": include_news_context,
        "timezone": timezone,
        "latest_price": "not_available",
        "trend_state": "not_connected_to_market_data_yet",
        "ema_200_state": "not_available",
        "macd_state": "not_available",
        "rsi_14_state": "not_available",
        "atr_14_state": "not_available",
        "volatility_state": "unknown",
        "spread_state": "unknown",
        "bias": "unclear",
        "action": "HOLD",
        "warnings": [
            "Backend route is working.",
            "Live market data is not connected yet.",
            "Do not use this output for real trading decisions yet."
        ],
        "evidence": [
            "GPT Action successfully reached the Render FastAPI backend.",
            "Market snapshot parameters were received correctly.",
            "Indicator engine still needs to be connected."
        ],
        "next_backend_upgrade": "Connect live/historical candle data and calculate EMA 200, MACD, RSI 14, and ATR 14."
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
def post_market_snapshot(payload: MarketSnapshotRequest):
    return {
        "status": "ok",
        "connection_status": "backend_reachable",
        "live_market_data": "not_connected_yet",
        "symbol": payload.symbol,
        "asset_class": payload.asset_class,
        "timeframe": payload.timeframe,
        "lookback_bars": payload.lookback_bars,
        "requested_indicators": payload.indicators,
        "include_news_context": payload.include_news_context,
        "timezone": payload.timezone,
        "latest_price": "not_available",
        "trend_state": "not_connected_to_market_data_yet",
        "ema_200_state": "not_available",
        "macd_state": "not_available",
        "rsi_14_state": "not_available",
        "atr_14_state": "not_available",
        "volatility_state": "unknown",
        "spread_state": "unknown",
        "bias": "unclear",
        "action": "HOLD",
        "warnings": [
            "POST endpoint is reachable.",
            "Live market data is not connected yet.",
            "Do not use this output for real trading decisions yet."
        ],
        "evidence": [
            "Request body was received successfully.",
            "Backend route is live.",
            "Indicator engine still needs to be connected."
        ]
    }
