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
    "format": "JSON",
    "timezone": "UTC"
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


def timeframe_to_minutes(timeframe: str) -> int:
    tf = timeframe.upper().replace(" ", "")

    mapping = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440
    }

    return mapping.get(tf, 5)


def classify_trading_session(latest_dt: pd.Timestamp):
    """
    Session classification based on UTC hour.
    This is approximate but useful for filtering market behavior.
    """
    hour = latest_dt.hour

    sessions = []

    if 0 <= hour < 8:
        sessions.append("asian_session")

    if 7 <= hour < 16:
        sessions.append("london_session")

    if 13 <= hour < 22:
        sessions.append("new_york_session")

    if 7 <= hour < 8:
        sessions.append("asia_london_overlap")

    if 13 <= hour < 16:
        sessions.append("london_new_york_overlap")

    if not sessions:
        sessions.append("off_peak_fx_liquidity")

    return sessions


def check_candle_freshness(latest_dt: pd.Timestamp, timeframe: str):
    """
    Checks whether the latest provider candle appears fresh.
    For M5, anything older than about 15 minutes is suspicious.
    """
    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    latest_clean = pd.Timestamp(latest_dt).tz_localize(None)

    age_minutes = (now_utc - latest_clean).total_seconds() / 60
    tf_minutes = timeframe_to_minutes(timeframe)

    max_allowed_age = tf_minutes * 3

    return {
        "latest_candle_utc": str(latest_clean),
        "now_utc": str(now_utc),
        "age_minutes": round(age_minutes, 2),
        "max_allowed_age_minutes": max_allowed_age,
        "is_fresh": age_minutes <= max_allowed_age
    }


def calculate_confidence_score(signal: dict, latest: pd.Series):
    """
    Simple confluence-based score.
    This is not a profitability score. It only measures setup alignment.
    """
    checks = signal.get("checks", {})
    action = signal.get("action", "HOLD")

    score = 0
    reasons = []

    if action == "BUY":
        if checks.get("price_above_ema_200"):
            score += 35
            reasons.append("Price above EMA 200.")
        if checks.get("macd_bullish_cross"):
            score += 35
            reasons.append("Fresh bullish MACD cross.")
        if checks.get("rsi_healthy_buy"):
            score += 20
            reasons.append("RSI 14 is healthy for bullish momentum.")

    elif action == "SELL":
        if checks.get("price_below_ema_200"):
            score += 35
            reasons.append("Price below EMA 200.")
        if checks.get("macd_bearish_cross"):
            score += 35
            reasons.append("Fresh bearish MACD cross.")
        if checks.get("rsi_healthy_sell"):
            score += 20
            reasons.append("RSI 14 is healthy for bearish momentum.")

    else:
        score = 0
        reasons.append("No valid BUY or SELL confirmation.")

    close = latest["close"]
    ema_200 = latest["ema_200"]
    atr_14 = latest["atr_14"]

    if pd.notna(close) and pd.notna(ema_200) and pd.notna(atr_14) and atr_14 > 0:
        ema_distance_atr = abs(close - ema_200) / atr_14

        if ema_distance_atr > 0.25:
            score += 10
            reasons.append("Price is not tightly compressed around EMA 200.")
        else:
            reasons.append("Price is close to EMA 200; chop risk present.")

    score = min(score, 100)

    if score >= 80:
        confidence_label = "high"
    elif score >= 60:
        confidence_label = "medium"
    elif score >= 40:
        confidence_label = "low"
    else:
        confidence_label = "weak_or_no_trade"

    return {
        "score": score,
        "label": confidence_label,
        "reasons": reasons
    }
