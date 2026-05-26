from fastapi import FastAPI, Query
from datetime import datetime
import os
import requests
import pandas as pd
from typing import Optional


app = FastAPI(
    title="Quant Finance Workbench API",
    version="1.1.0",
    description="Market Snapshot v1 for Quant & Finance Architecture GPT backend."
)


TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Quant Finance Workbench API",
        "version": "1.1.0",
        "message": "Backend is live."
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Quant Finance Workbench API",
        "version": "1.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }


def normalize_symbol(symbol: str) -> str:
    """
    Converts EURUSD to EUR/USD for Twelve Data.
    Keeps already formatted symbols like EUR/USD unchanged.
    """
    symbol = symbol.upper().replace(" ", "")

    if "/" in symbol:
        return symbol

    if len(symbol) == 6:
        return f"{symbol[:3]}/{symbol[3:]}"

    return symbol


def normalize_timeframe(timeframe: str) -> str:
    """
    Converts trading shorthand like M5 to Twelve Data interval format.
    """
    tf = timeframe.upper().replace(" ", "")

    mapping = {
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
        "D1": "1day"
    }

    return mapping.get(tf, timeframe)


def fetch_twelve_data_candles(symbol: str, interval: str, outputsize: int = 500) -> pd.DataFrame:
    if not TWELVE_DATA_API_KEY:
        raise ValueError("Missing TWELVE_DATA_API_KEY in Render environment variables.")

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }

    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if "status" in data and data.get("status") == "error":
        raise ValueError(data.get("message", "Twelve Data returned an error."))

    if "values" not in data:
        raise ValueError(f"Unexpected Twelve Data response: {data}")

    values = data["values"]

    df = pd.DataFrame(values)

    required_cols = ["datetime", "open", "high", "low", "close"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required candle column: {col}")

    df["datetime"] = pd.to_datetime(df["datetime"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds EMA 200, MACD 12/26/9, RSI 14, and ATR 14.
    """
    df = df.copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA 200
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()

    # MACD 12/26/9
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()

    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    # RSI 14 using Wilder-style smoothing
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ATR 14
    prev_close = close.shift(1)

    tr_1 = high - low
    tr_2 = (high - prev_close).abs()
    tr_3 = (low - prev_close).abs()

    true_range = pd.concat([tr_1, tr_2, tr_3], axis=1).max(axis=1)
    df["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    return df


def round_float(value, digits: int = 5):
    if pd.isna(value):
        return None
    return round(float(value), digits)


def build_signal(df: pd.DataFrame):
    """
    Conservative signal logic:
    BUY only when price > EMA 200, fresh bullish MACD cross, and RSI healthy.
    SELL only when price < EMA 200, fresh bearish MACD cross, and RSI healthy.
    Otherwise HOLD.
    """
    if len(df) < 210:
        return {
            "action": "HOLD",
            "reason": "Not enough candles to reliably calculate EMA 200 and momentum state.",
            "warnings": ["Need at least 210 clean candles for this v1 logic."]
        }

    current = df.iloc[-1]
    previous = df.iloc[-2]

    close = current["close"]
    ema_200 = current["ema_200"]
    rsi_14 = current["rsi_14"]

    macd_line = current["macd_line"]
    macd_signal = current["macd_signal"]

    prev_macd_line = previous["macd_line"]
    prev_macd_signal = previous["macd_signal"]

    price_above_ema_200 = close > ema_200
    price_below_ema_200 = close < ema_200

    macd_bullish_cross = prev_macd_line <= prev_macd_signal and macd_line > macd_signal
    macd_bearish_cross = prev_macd_line >= prev_macd_signal and macd_line < macd_signal

    rsi_healthy_buy = 40 <= rsi_14 <= 70
    rsi_healthy_sell = 30 <= rsi_14 <= 60
    rsi_overbought = rsi_14 > 70
    rsi_oversold = rsi_14 < 30

    if price_above_ema_200:
        trend_state = "bullish_above_ema_200"
    elif price_below_ema_200:
        trend_state = "bearish_below_ema_200"
    else:
        trend_state = "neutral_at_ema_200"

    if macd_bullish_cross:
        macd_state = "fresh_bullish_cross"
    elif macd_bearish_cross:
        macd_state = "fresh_bearish_cross"
    elif macd_line > macd_signal:
        macd_state = "bullish_but_no_fresh_cross"
    elif macd_line < macd_signal:
        macd_state = "bearish_but_no_fresh_cross"
    else:
        macd_state = "neutral"

    if rsi_overbought:
        rsi_state = "overbought"
    elif rsi_oversold:
        rsi_state = "oversold"
    else:
        rsi_state = "healthy"

    action = "HOLD"
    reason = "Conditions are mixed or no fresh confirmation."

    if price_above_ema_200 and macd_bullish_cross and rsi_healthy_buy:
        action = "BUY"
        reason = "Price is above EMA 200, MACD has a fresh bullish cross, and RSI 14 is in a healthy buy momentum zone."

    elif price_below_ema_200 and macd_bearish_cross and rsi_healthy_sell:
        action = "SELL"
        reason = "Price is below EMA 200, MACD has a fresh bearish cross, and RSI 14 is in a healthy sell momentum zone."

    warnings = []

    if rsi_overbought:
        warnings.append("RSI 14 is overbought. Avoid late BUY entries.")
    if rsi_oversold:
        warnings.append("RSI 14 is oversold. Avoid late SELL entries.")
    if not macd_bullish_cross and not macd_bearish_cross:
        warnings.append("No fresh MACD cross detected on the latest candle.")
    if abs(close - ema_200) <= current["atr_14"] * 0.15:
        warnings.append("Price is very close to EMA 200. Possible chop/compression zone.")

    return {
        "action": action,
        "reason": reason,
        "trend_state": trend_state,
        "macd_state": macd_state,
        "rsi_state": rsi_state,
        "checks": {
            "price_above_ema_200": bool(price_above_ema_200),
            "price_below_ema_200": bool(price_below_ema_200),
            "macd_bullish_cross": bool(macd_bullish_cross),
            "macd_bearish_cross": bool(macd_bearish_cross),
            "rsi_healthy_buy": bool(rsi_healthy_buy),
            "rsi_healthy_sell": bool(rsi_healthy_sell),
            "rsi_overbought": bool(rsi_overbought),
            "rsi_oversold": bool(rsi_oversold)
        },
        "warnings": warnings
    }


@app.get("/market/snapshot-v1")
def market_snapshot_v1(
    asset_class: str = Query(default="forex"),
    symbol: str = Query(default="EURUSD"),
    timeframe: str = Query(default="M5"),
    lookback_bars: int = Query(default=300, ge=220, le=5000),
    timezone: Optional[str] = Query(default="Asia/Manila")
):
    """
    Market Snapshot v1:
    Fetches candles from Twelve Data, calculates indicators,
    and returns a conservative BUY / SELL / HOLD result.
    """
    try:
        normalized_symbol = normalize_symbol(symbol)
        interval = normalize_timeframe(timeframe)

        fetch_size = max(lookback_bars, 300)

        candles = fetch_twelve_data_candles(
            symbol=normalized_symbol,
            interval=interval,
            outputsize=fetch_size
        )

        if len(candles) < 220:
            return {
                "status": "error",
                "symbol": symbol,
                "normalized_symbol": normalized_symbol,
                "timeframe": timeframe,
                "interval": interval,
                "action": "HOLD",
                "reason": "Not enough candle data returned from provider.",
                "candles_returned": len(candles),
                "minimum_required": 220
            }

        candles = candles.tail(lookback_bars).reset_index(drop=True)
        candles = calculate_indicators(candles)

        signal = build_signal(candles)

        latest = candles.iloc[-1]
        previous = candles.iloc[-2]

        return {
            "status": "ok",
            "provider": "Twelve Data",
            "asset_class": asset_class,
            "symbol": symbol.upper(),
            "normalized_symbol": normalized_symbol,
            "timeframe": timeframe.upper(),
            "provider_interval": interval,
            "lookback_bars": lookback_bars,
            "candles_used": len(candles),
            "latest_candle_time": str(latest["datetime"]),
            "previous_candle_time": str(previous["datetime"]),
            "latest": {
                "open": round_float(latest["open"]),
                "high": round_float(latest["high"]),
                "low": round_float(latest["low"]),
                "close": round_float(latest["close"]),
                "ema_200": round_float(latest["ema_200"]),
                "macd_line": round_float(latest["macd_line"], 7),
                "macd_signal": round_float(latest["macd_signal"], 7),
                "macd_histogram": round_float(latest["macd_histogram"], 7),
                "rsi_14": round_float(latest["rsi_14"], 2),
                "atr_14": round_float(latest["atr_14"], 7)
            },
            "previous": {
                "close": round_float(previous["close"]),
                "macd_line": round_float(previous["macd_line"], 7),
                "macd_signal": round_float(previous["macd_signal"], 7),
                "rsi_14": round_float(previous["rsi_14"], 2)
            },
            "trend_state": signal.get("trend_state"),
            "macd_state": signal.get("macd_state"),
            "rsi_state": signal.get("rsi_state"),
            "checks": signal.get("checks"),
            "action": signal.get("action"),
            "reason": signal.get("reason"),
            "warnings": signal.get("warnings"),
            "integrity_note": "This is an analytical signal only. It is not financial advice and should not execute trades without broker-side risk controls.",
            "next_upgrade": "Add spread filter, session filter, news filter, and backtest validation."
        }

    except Exception as e:
        return {
            "status": "error",
            "provider": "Twelve Data",
            "symbol": symbol,
            "timeframe": timeframe,
            "action": "HOLD",
            "error": str(e),
            "reason": "Market Snapshot v1 failed safely. Defaulting to HOLD."
        }
