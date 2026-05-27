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

def make_json_safe(value):
    """
    Converts pandas/numpy objects into normal JSON-safe Python values.
    Prevents FastAPI Internal Server Error from numpy.bool_, numpy.float64,
    pandas Timestamp, NaN, etc.
    """
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(v) for v in value]

    if isinstance(value, pd.Timestamp):
        return str(value)

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, (int, float, str)) or value is None:
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return value

    if hasattr(value, "item"):
        try:
            return make_json_safe(value.item())
        except Exception:
            pass

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)

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
        "is_fresh": bool(age_minutes <= max_allowed_age)
    }

def select_confirmed_candles(df: pd.DataFrame, timeframe: str):
    """
    No-repaint confirmed-candle selector.
    If the latest provider candle is still forming, ignore it and use the last fully closed candle.
    Assumes provider candle timestamps are candle open times in UTC.
    """
    working = df.sort_values("datetime").reset_index(drop=True).copy()

    tf_minutes = timeframe_to_minutes(timeframe)
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)

    provider_latest_open = pd.Timestamp(working.iloc[-1]["datetime"]).to_pydatetime().replace(tzinfo=None)
    provider_latest_close = provider_latest_open + pd.Timedelta(minutes=tf_minutes)

    ignored_forming_candle = now_utc < provider_latest_close

    if ignored_forming_candle:
        confirmed = working.iloc[:-1].copy().reset_index(drop=True)
    else:
        confirmed = working.copy().reset_index(drop=True)

    if len(confirmed) == 0:
        decision_open = None
        decision_close = None
    else:
        decision_open = pd.Timestamp(confirmed.iloc[-1]["datetime"]).to_pydatetime().replace(tzinfo=None)
        decision_close = decision_open + pd.Timedelta(minutes=tf_minutes)

    return {
        "candles": confirmed,
        "mode": "confirmed_candle_only",
        "timeframe": timeframe.upper(),
        "now_utc": str(now_utc),
        "provider_latest_candle_open_utc": str(provider_latest_open),
        "provider_latest_candle_close_utc": str(provider_latest_close),
        "ignored_forming_candle": bool(ignored_forming_candle),
        "decision_candle_open_utc": str(decision_open) if decision_open is not None else None,
        "decision_candle_close_utc": str(decision_close) if decision_close is not None else None,
        "signal_delay_bars": 1 if ignored_forming_candle else 0,
        "confirmed_candles_available": int(len(confirmed))
    }

def evaluate_spread_filter(current_spread_pips: Optional[float], max_allowed_spread_pips: float = 1.5):
    """
    Spread filter / spread status gate.
    Always returns a dictionary.
    In research mode, unavailable spread does not block.
    In live execution mode, broker-side spread should be required.
    """
    try:
        max_spread = float(max_allowed_spread_pips)

        if current_spread_pips is None:
            return {
                "spread_status": "unavailable",
                "trade_allowed": True,
                "current_spread_pips": None,
                "max_allowed_spread_pips": max_spread,
                "reason": "Spread not provided. Not blocking in research mode, but live execution should require broker spread."
            }

        spread_value = float(current_spread_pips)

        if spread_value < 0:
            return {
                "spread_status": "invalid",
                "trade_allowed": False,
                "current_spread_pips": spread_value,
                "max_allowed_spread_pips": max_spread,
                "reason": "Spread cannot be negative. Forced HOLD."
            }

        if spread_value <= max_spread:
            return {
                "spread_status": "normal",
                "trade_allowed": True,
                "current_spread_pips": spread_value,
                "max_allowed_spread_pips": max_spread,
                "reason": "Spread is within allowed threshold."
            }

        return {
            "spread_status": "elevated",
            "trade_allowed": False,
            "current_spread_pips": spread_value,
            "max_allowed_spread_pips": max_spread,
            "reason": "Spread is above allowed threshold. Trade should be blocked."
        }

    except Exception as e:
        return {
            "spread_status": "error",
            "trade_allowed": False,
            "current_spread_pips": current_spread_pips,
            "max_allowed_spread_pips": max_allowed_spread_pips,
            "reason": "Spread filter calculation failed. Forced HOLD.",
            "error": str(e)
        }


def evaluate_low_capital_risk_gate(
    account_balance: Optional[float],
    risk_percent: float = 1.0,
    stop_loss_pips: Optional[float] = None,
    pip_value_per_standard_lot: float = 10.0,
    broker_min_lot_size: float = 0.01,
    broker_lot_step: float = 0.01
):
    """
    Low-capital risk / lot-size gate.
    Always returns a dictionary.
    Forces HOLD when safe position sizing is impossible.
    """
    try:
        if account_balance is None:
            return {
                "risk_gate_status": "unavailable",
                "risk_gate_passed": True,
                "reason": "Account balance not provided. Not blocking in research mode, but live mode should require account balance.",
                "account_balance": None,
                "risk_percent": risk_percent,
                "stop_loss_pips": stop_loss_pips
            }

        balance = float(account_balance)
        risk_pct = float(risk_percent)

        if balance <= 0:
            return {
                "risk_gate_status": "invalid_account_balance",
                "risk_gate_passed": False,
                "reason": "Account balance must be greater than zero. Forced HOLD.",
                "account_balance": balance,
                "risk_percent": risk_pct,
                "stop_loss_pips": stop_loss_pips
            }

        if stop_loss_pips is None or float(stop_loss_pips) <= 0:
            return {
                "risk_gate_status": "missing_stop_loss",
                "risk_gate_passed": False,
                "reason": "Stop-loss pips not provided or invalid. Cannot calculate safe lot size. Forced HOLD.",
                "account_balance": balance,
                "risk_percent": risk_pct,
                "stop_loss_pips": stop_loss_pips
            }

        sl_pips = float(stop_loss_pips)
        pip_value = float(pip_value_per_standard_lot)
        min_lot = float(broker_min_lot_size)
        lot_step = float(broker_lot_step)

        if pip_value <= 0 or min_lot <= 0 or lot_step <= 0:
            return {
                "risk_gate_status": "invalid_broker_settings",
                "risk_gate_passed": False,
                "reason": "Pip value, broker minimum lot size, and lot step must be greater than zero. Forced HOLD.",
                "account_balance": balance,
                "risk_percent": risk_pct,
                "stop_loss_pips": sl_pips,
                "pip_value_per_standard_lot": pip_value,
                "broker_min_lot_size": min_lot,
                "broker_lot_step": lot_step
            }

        max_risk_amount = balance * (risk_pct / 100.0)
        raw_lot_size = max_risk_amount / (sl_pips * pip_value)

        rounded_lot_size = int(raw_lot_size / lot_step) * lot_step
        rounded_lot_size = round(rounded_lot_size, 5)

        min_lot_risk_amount = sl_pips * pip_value * min_lot
        min_lot_actual_risk_percent = (min_lot_risk_amount / balance) * 100.0

        if rounded_lot_size < min_lot:
            return {
                "risk_gate_status": "blocked_min_lot_too_large",
                "risk_gate_passed": False,
                "reason": "Broker minimum lot size would exceed allowed risk. Forced HOLD.",
                "account_balance": balance,
                "risk_percent": risk_pct,
                "max_risk_amount": round(max_risk_amount, 2),
                "stop_loss_pips": sl_pips,
                "pip_value_per_standard_lot": pip_value,
                "raw_lot_size": round(raw_lot_size, 5),
                "rounded_lot_size": rounded_lot_size,
                "broker_min_lot_size": min_lot,
                "broker_lot_step": lot_step,
                "min_lot_risk_amount": round(min_lot_risk_amount, 2),
                "min_lot_actual_risk_percent": round(min_lot_actual_risk_percent, 2)
            }

        actual_risk_amount = sl_pips * pip_value * rounded_lot_size
        actual_risk_percent = (actual_risk_amount / balance) * 100.0

        return {
            "risk_gate_status": "passed",
            "risk_gate_passed": True,
            "reason": "Safe position size is possible within risk settings.",
            "account_balance": balance,
            "risk_percent": risk_pct,
            "max_risk_amount": round(max_risk_amount, 2),
            "stop_loss_pips": sl_pips,
            "pip_value_per_standard_lot": pip_value,
            "raw_lot_size": round(raw_lot_size, 5),
            "rounded_lot_size": rounded_lot_size,
            "broker_min_lot_size": min_lot,
            "broker_lot_step": lot_step,
            "actual_risk_amount": round(actual_risk_amount, 2),
            "actual_risk_percent": round(actual_risk_percent, 2)
        }

    except Exception as e:
        return {
            "risk_gate_status": "error",
            "risk_gate_passed": False,
            "reason": "Risk gate calculation failed. Forced HOLD.",
            "error": str(e),
            "account_balance": account_balance,
            "risk_percent": risk_percent,
            "stop_loss_pips": stop_loss_pips
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

def build_signal(df: pd.DataFrame):
    """
    Conservative M5 signal logic:
    BUY only when price > EMA 200, fresh bullish MACD cross, and RSI healthy.
    SELL only when price < EMA 200, fresh bearish MACD cross, and RSI healthy.
    Otherwise HOLD.
    """
    if len(df) < 210:
        return {
            "action": "HOLD",
            "reason": "Not enough candles to reliably calculate EMA 200 and momentum state.",
            "trend_state": "unknown",
            "macd_state": "unknown",
            "rsi_state": "unknown",
            "checks": {},
            "warnings": ["Need at least 210 clean candles for this signal logic."]
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

    if pd.notna(current["atr_14"]) and current["atr_14"] > 0:
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

def build_signal(df: pd.DataFrame):
    """
    Conservative M5 signal logic:
    BUY only when price > EMA 200, fresh bullish MACD cross, and RSI healthy.
    SELL only when price < EMA 200, fresh bearish MACD cross, and RSI healthy.
    Otherwise HOLD.
    """
    if len(df) < 210:
        return {
            "action": "HOLD",
            "reason": "Not enough candles to reliably calculate EMA 200 and momentum state.",
            "trend_state": "unknown",
            "macd_state": "unknown",
            "rsi_state": "unknown",
            "checks": {},
            "warnings": ["Need at least 210 clean candles for this signal logic."]
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

    if pd.notna(current["atr_14"]) and current["atr_14"] > 0:
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

@app.get("/market/snapshot-v1-1")
def market_snapshot_v1_1(
    asset_class: str = Query(default="forex"),
    symbol: str = Query(default="EURUSD"),
    timeframe: str = Query(default="M5"),
    lookback_bars: int = Query(default=300, ge=220, le=5000),
    timezone: Optional[str] = Query(default="Asia/Manila")
):
    """
    Market Snapshot v1.1:
    Adds candle freshness, session label, EMA compression filter,
    confidence score, and safer HOLD rules.
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
                "final_action": "HOLD",
                "reason": "Not enough candle data returned from provider.",
                "candles_returned": len(candles),
                "minimum_required": 220
            }

        candles = candles.tail(lookback_bars).reset_index(drop=True)
        candles = calculate_indicators(candles)

        signal = build_signal(candles)

        latest = candles.iloc[-1]
        previous = candles.iloc[-2]

        freshness = check_candle_freshness(latest["datetime"], timeframe)
        sessions = classify_trading_session(latest["datetime"])
        confidence = calculate_confidence_score(signal, latest)

        close = latest["close"]
        ema_200 = latest["ema_200"]
        atr_14 = latest["atr_14"]

        ema_distance = abs(close - ema_200)
        ema_distance_atr = ema_distance / atr_14 if atr_14 and atr_14 > 0 else None

        compression_zone = False
        if ema_distance_atr is not None:
            compression_zone = bool(ema_distance_atr <= 0.25)

        final_action = signal.get("action")
        final_reason = signal.get("reason")
        hard_filters = []

        if not freshness["is_fresh"]:
            final_action = "HOLD"
            hard_filters.append("Latest candle appears stale. Forced HOLD.")

        if compression_zone:
            final_action = "HOLD"
            hard_filters.append("Price is too close to EMA 200. Possible chop/compression. Forced HOLD.")

        if confidence.get("score", 0) < 60 and final_action in ["BUY", "SELL"]:
            final_action = "HOLD"
            hard_filters.append("Confidence score is below 60. Forced HOLD.")

        warnings = signal.get("warnings", [])

        if hard_filters:
            warnings = warnings + hard_filters
            final_reason = "One or more safety filters blocked the trade."

        return {
            "status": "ok",
            "provider": "Twelve Data",    
            "engine_version": "market_snapshot_v1_1",
            "asset_class": asset_class,
            "symbol": symbol.upper(),
            "normalized_symbol": normalized_symbol,
            "timeframe": timeframe.upper(),
            "provider_interval": interval,
            "lookback_bars": lookback_bars,
            "candles_used": len(candles),
            "latest_candle_time": str(latest["datetime"]),
            "previous_candle_time": str(previous["datetime"]),
            "session_context": make_json_safe(sessions),
            "freshness": make_json_safe(freshness),
            "latest": {
                "open": round_float(latest["open"]),
                "high": round_float(latest["high"]),
                "low": round_float(latest["low"]),
                "close": round_float(latest["close"]),
                "ema_200": round_float(latest["ema_200"]),
                "ema_distance": round_float(ema_distance, 7),
                "ema_distance_atr": round_float(ema_distance_atr, 3) if ema_distance_atr is not None else None,
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
            "compression_zone": bool(compression_zone),
            "raw_signal_action": signal.get("action"),
            "final_action": final_action,
            "reason": final_reason,
            "confidence": make_json_safe(confidence),
            "checks": make_json_safe(signal.get("checks")),
            "hard_filters": make_json_safe(hard_filters),
            "warnings": make_json_safe(warnings),
            "integrity_note": "Analytical signal only. Not financial advice. Do not execute without broker-side risk controls.",
            "next_upgrade": "Add spread filter, news filter, multi-timeframe confirmation, and backtest validation."
        }

    except Exception as e:
        return {
            "status": "error",
            "provider": "Twelve Data",
            "engine_version": "market_snapshot_v1_1",
            "symbol": symbol,
            "timeframe": timeframe,
            "final_action": "HOLD",
            "error": str(e),
            "reason": "Market Snapshot v1.1 failed safely. Defaulting to HOLD."
        }

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

def build_higher_timeframe_bias(df: pd.DataFrame, label: str = "M15"):
    """
    Higher timeframe bias filter.
    Uses close vs EMA 200 and MACD line vs signal line.
    This does not create entries. It only confirms or blocks M5 entries.
    """
    if len(df) < 210:
        return {
            "timeframe": label,
            "bias": "unknown",
            "reason": "Not enough candles for higher-timeframe EMA 200 confirmation.",
            "checks": {}
        }

    latest = df.iloc[-1]

    close = latest["close"]
    ema_200 = latest["ema_200"]
    macd_line = latest["macd_line"]
    macd_signal = latest["macd_signal"]
    rsi_14 = latest["rsi_14"]

    price_above_ema_200 = close > ema_200
    price_below_ema_200 = close < ema_200
    macd_bullish = macd_line > macd_signal
    macd_bearish = macd_line < macd_signal

    if price_above_ema_200 and macd_bullish:
        bias = "bullish"
        reason = f"{label} confirms bullish bias: price is above EMA 200 and MACD is bullish."
    elif price_below_ema_200 and macd_bearish:
        bias = "bearish"
        reason = f"{label} confirms bearish bias: price is below EMA 200 and MACD is bearish."
    else:
        bias = "mixed"
        reason = f"{label} bias is mixed or unclear."

    return {
        "timeframe": label,
        "bias": bias,
        "reason": reason,
        "latest": {
            "close": round_float(close),
            "ema_200": round_float(ema_200),
            "macd_line": round_float(macd_line, 7),
            "macd_signal": round_float(macd_signal, 7),
            "rsi_14": round_float(rsi_14, 2)
        },
        "checks": {
            "price_above_ema_200": bool(price_above_ema_200),
            "price_below_ema_200": bool(price_below_ema_200),
            "macd_bullish": bool(macd_bullish),
            "macd_bearish": bool(macd_bearish)
        }
    }

@app.get("/quant/forex-decision-stack-v1")
def quant_forex_decision_stack_v1(
    asset_class: str = Query(default="forex"),
    symbol: str = Query(default="EURUSD"),
    entry_timeframe: str = Query(default="M5"),
    confirm_timeframe: str = Query(default="M15"),
    lookback_bars: int = Query(default=300, ge=220, le=5000),
    timezone: Optional[str] = Query(default="Asia/Manila"),
    current_spread_pips: Optional[float] = Query(default=None, ge=0),
    max_allowed_spread_pips: float = Query(default=1.5, ge=0.1),
    account_balance: Optional[float] = Query(default=None, ge=0),
    risk_percent: float = Query(default=1.0, ge=0.1, le=5.0),
    stop_loss_pips: Optional[float] = Query(default=None, ge=0.1),
    pip_value_per_standard_lot: float = Query(default=10.0, ge=0.01),
    broker_min_lot_size: float = Query(default=0.01, ge=0.0001),
    broker_lot_step: float = Query(default=0.01, ge=0.0001)
):
    """
    Quant Forex Decision Stack v1:
    M5 entry signal + M15 trend confirmation + freshness/compression/confidence filters.
    """
    try:
        normalized_symbol = normalize_symbol(symbol)

        entry_interval = normalize_timeframe(entry_timeframe)
        confirm_interval = normalize_timeframe(confirm_timeframe)

        fetch_size = max(lookback_bars + 5, 305)

        entry_candles = fetch_twelve_data_candles(
            symbol=normalized_symbol,
            interval=entry_interval,
            outputsize=fetch_size
        )

        confirm_candles = fetch_twelve_data_candles(
            symbol=normalized_symbol,
            interval=confirm_interval,
            outputsize=fetch_size
        )

        if len(entry_candles) < 220 or len(confirm_candles) < 220:
            return make_json_safe({
                "status": "error",
                "engine_version": "quant_forex_decision_stack_v1",
                "symbol": symbol,
                "final_action": "HOLD",
                "reason": "Not enough candle data returned for entry or confirmation timeframe.",
                "entry_candles_returned": len(entry_candles),
                "confirm_candles_returned": len(confirm_candles),
                "minimum_required": 220
            })

        entry_confirmed_info = select_confirmed_candles(entry_candles, entry_timeframe)
        confirm_confirmed_info = select_confirmed_candles(confirm_candles, confirm_timeframe)

        entry_confirmed_candles = entry_confirmed_info["candles"]
        confirm_confirmed_candles = confirm_confirmed_info["candles"]

        if len(entry_confirmed_candles) < 220 or len(confirm_confirmed_candles) < 220:
             return make_json_safe({
                "status": "error",
                "engine_version": "quant_forex_decision_stack_v1",
                "symbol": symbol,
                "final_action": "HOLD",
                "reason": "Not enough confirmed candle data after removing forming candles.",
                "entry_confirmed_candles": len(entry_confirmed_candles),
                "confirm_confirmed_candles": len(confirm_confirmed_candles),
                "minimum_required": 220,
                "entry_confirmed_candle_mode": {k: v for k, v in entry_confirmed_info.items() if k != "candles"},
                "confirm_confirmed_candle_mode": {k: v for k, v in confirm_confirmed_info.items() if k != "candles"}
            })

        entry_candles = entry_confirmed_candles.tail(lookback_bars).reset_index(drop=True)
        confirm_candles = confirm_confirmed_candles.tail(lookback_bars).reset_index(drop=True)

        entry_candles = calculate_indicators(entry_candles)
        confirm_candles = calculate_indicators(confirm_candles)

        entry_signal = build_signal(entry_candles)
        confirm_bias = build_higher_timeframe_bias(confirm_candles, label=confirm_timeframe.upper())

        latest_entry = entry_candles.iloc[-1]
        previous_entry = entry_candles.iloc[-2]

        freshness = check_candle_freshness(latest_entry["datetime"], entry_timeframe)
        sessions = classify_trading_session(latest_entry["datetime"])
        confidence = calculate_confidence_score(entry_signal, latest_entry)

        close = latest_entry["close"]
        ema_200 = latest_entry["ema_200"]
        atr_14 = latest_entry["atr_14"]

        ema_distance = abs(close - ema_200)
        ema_distance_atr = ema_distance / atr_14 if atr_14 and atr_14 > 0 else None

        compression_zone = False
        if ema_distance_atr is not None:
            compression_zone = bool(ema_distance_atr <= 0.25)

        raw_action = entry_signal.get("action", "HOLD")
        final_action = raw_action
        final_reason = entry_signal.get("reason", "No valid entry reason returned.")
        hard_filters = []
        
        spread_filter = evaluate_spread_filter(current_spread_pips, max_allowed_spread_pips)
        risk_gate = evaluate_low_capital_risk_gate(
            account_balance=account_balance,
            risk_percent=risk_percent,
            stop_loss_pips=stop_loss_pips,
            pip_value_per_standard_lot=pip_value_per_standard_lot,
            broker_min_lot_size=broker_min_lot_size,
            broker_lot_step=broker_lot_step
        )

        if risk_gate is None:
            risk_gate = {
                "risk_gate_status": "error",
                "risk_gate_passed": False,
                "reason": "Risk gate returned None. Forced HOLD."
            }

        # SAFE NONE NORMALIZATION BLOCK
        # Prevents any helper returning None from crashing the endpoint.
        if entry_signal is None:
            entry_signal = {
                "action": "HOLD",
                "reason": "Entry signal returned None. Forced HOLD.",
                "trend_state": "unknown",
                "macd_state": "unknown",
                "rsi_state": "unknown",
                "checks": {},
                "warnings": ["Entry signal returned None."]
            }

        if confirm_bias is None:
            confirm_bias = {
                "timeframe": confirm_timeframe.upper(),
                "bias": "unknown",
                "reason": "Higher-timeframe confirmation returned None. Forced HOLD.",
                "checks": {}
            }

        if freshness is None:
            freshness = {
                "is_fresh": False,
                "reason": "Freshness check returned None. Forced HOLD."
            }

        if confidence is None:
            confidence = {
                "score": 0,
                "label": "weak_or_no_trade",
                "reasons": ["Confidence calculation returned None."]
            }

        if spread_filter is None:
            spread_filter = {
                "spread_status": "error",
                "trade_allowed": False,
                "reason": "Spread filter returned None. Forced HOLD."
            }

        if risk_gate is None:
            risk_gate = {
                "risk_gate_status": "error",
                "risk_gate_passed": False,
                "reason": "Risk gate returned None. Forced HOLD."
            }

        if raw_action == "BUY" and confirm_bias.get("bias") != "bullish":
            final_action = "HOLD"
            hard_filters.append("M15 confirmation does not support BUY. Forced HOLD.")

        if raw_action == "SELL" and confirm_bias.get("bias") != "bearish":
            final_action = "HOLD"
            hard_filters.append("M15 confirmation does not support SELL. Forced HOLD.")

        if not freshness.get("is_fresh", False):
            final_action = "HOLD"
            hard_filters.append("Latest M5 candle appears stale. Forced HOLD.")

        if compression_zone:
            final_action = "HOLD"
            hard_filters.append("Price is too close to M5 EMA 200. Possible chop/compression. Forced HOLD.")

        if confidence.get("score", 0) < 60 and final_action in ["BUY", "SELL"]:
            final_action = "HOLD"
            hard_filters.append("Confidence score is below 60. Forced HOLD.")

        if not spread_filter.get("trade_allowed", True):
            final_action = "HOLD"
            hard_filters.append("Spread is above allowed threshold. Forced HOLD.")

        if not risk_gate.get("risk_gate_passed", True):
            final_action = "HOLD"
            hard_filters.append("Low-capital risk gate failed. Safe position sizing is impossible. Forced HOLD.")
        
        warnings = entry_signal.get("warnings", [])

        if hard_filters:
            warnings = warnings + hard_filters
            final_reason = "One or more decision-stack filters blocked the trade."

        response = {
            "status": "ok",
            "provider": "Twelve Data",
            "engine_version": "quant_forex_decision_stack_v1",
            "asset_class": asset_class,
            "symbol": symbol.upper(),
            "normalized_symbol": normalized_symbol,
            "entry_timeframe": entry_timeframe.upper(),
            "confirm_timeframe": confirm_timeframe.upper(),
            "entry_provider_interval": entry_interval,
            "confirm_provider_interval": confirm_interval,
            "lookback_bars": lookback_bars,
            "entry_candles_used": len(entry_candles),
            "confirm_candles_used": len(confirm_candles),
            "latest_entry_candle_time": latest_entry["datetime"],
            "previous_entry_candle_time": previous_entry["datetime"],
            "confirmed_candle_mode": {
            "enabled": True,
            "entry": {k: v for k, v in entry_confirmed_info.items() if k != "candles"},
            "confirmation": {k: v for k, v in confirm_confirmed_info.items() if k != "candles"}
},
"session_context": sessions,
            "freshness": freshness,
            "entry_snapshot": {
                "open": round_float(latest_entry["open"]),
                "high": round_float(latest_entry["high"]),
                "low": round_float(latest_entry["low"]),
                "close": round_float(latest_entry["close"]),
                "ema_200": round_float(latest_entry["ema_200"]),
                "ema_distance": round_float(ema_distance, 7),
                "ema_distance_atr": round_float(ema_distance_atr, 3) if ema_distance_atr is not None else None,
                "macd_line": round_float(latest_entry["macd_line"], 7),
                "macd_signal": round_float(latest_entry["macd_signal"], 7),
                "macd_histogram": round_float(latest_entry["macd_histogram"], 7),
                "rsi_14": round_float(latest_entry["rsi_14"], 2),
                "atr_14": round_float(latest_entry["atr_14"], 7)
            },
            "entry_signal": {
                "trend_state": entry_signal.get("trend_state"),
                "macd_state": entry_signal.get("macd_state"),
                "rsi_state": entry_signal.get("rsi_state"),
                "checks": entry_signal.get("checks"),
                "raw_action": raw_action,
                "reason": entry_signal.get("reason")
            },
            "higher_timeframe_confirmation": confirm_bias,
            "compression_zone": compression_zone,
            "confidence": confidence,
            "spread_filter": spread_filter,
            "risk_gate": risk_gate,
            "final_action": final_action,
            "final_reason": final_reason,
            "hard_filters": hard_filters,
            "warnings": warnings,
            "integrity_note": "Analytical signal only. Not financial advice. No live execution permission.",
            "next_upgrade": "Add backtest validation, session performance breakdown, and news filter."
        }

        return make_json_safe(response)

    except Exception as e:
        return make_json_safe({
            "status": "error",
            "provider": "Twelve Data",
            "engine_version": "quant_forex_decision_stack_v1",
            "symbol": symbol,
            "entry_timeframe": entry_timeframe,
            "confirm_timeframe": confirm_timeframe,
            "final_action": "HOLD",
            "error": str(e),
            "reason": "Quant Forex Decision Stack v1 failed safely. Defaulting to HOLD."
        })


@app.get("/quant/backtest-decision-stack-v1")
def quant_backtest_decision_stack_v1(
    asset_class: str = Query(default="forex"),
    symbol: str = Query(default="EURUSD"),
    entry_timeframe: str = Query(default="M5"),
    confirm_timeframe: str = Query(default="M15"),
    lookback_bars: int = Query(default=1000, ge=300, le=5000),
    spread_pips: float = Query(default=0.8, ge=0),
    max_allowed_spread_pips: float = Query(default=1.5, ge=0.1),
    account_balance: float = Query(default=50.0, ge=1),
    risk_percent: float = Query(default=1.0, ge=0.1, le=5.0),
    stop_loss_pips: float = Query(default=20.0, ge=1),
    reward_risk: float = Query(default=1.5, ge=0.5, le=5.0),
    max_hold_bars: int = Query(default=12, ge=1, le=100),
    pip_size: float = Query(default=0.0001, ge=0.00001),
    pip_value_per_standard_lot: float = Query(default=10.0, ge=0.01),
    broker_min_lot_size: float = Query(default=0.001, ge=0.0001),
    broker_lot_step: float = Query(default=0.001, ge=0.0001)
):
    """
    Backtest Decision Stack v1:
    Tests the current M5 entry + M15 confirmation stack with spread and low-capital risk gate.
    This is a rough validation engine, not proof of future profitability.
    """
    try:
        normalized_symbol = normalize_symbol(symbol)
        entry_interval = normalize_timeframe(entry_timeframe)
        confirm_interval = normalize_timeframe(confirm_timeframe)

        fetch_size = max(lookback_bars + 100, 400)

        entry_raw = fetch_twelve_data_candles(
            symbol=normalized_symbol,
            interval=entry_interval,
            outputsize=fetch_size
        )

        confirm_raw = fetch_twelve_data_candles(
            symbol=normalized_symbol,
            interval=confirm_interval,
            outputsize=fetch_size
        )

        entry_confirmed_info = select_confirmed_candles(entry_raw, entry_timeframe)
        confirm_confirmed_info = select_confirmed_candles(confirm_raw, confirm_timeframe)

        entry_df = entry_confirmed_info["candles"].tail(lookback_bars).reset_index(drop=True)
        confirm_df = confirm_confirmed_info["candles"].tail(lookback_bars).reset_index(drop=True)

        if len(entry_df) < 250 or len(confirm_df) < 220:
            return make_json_safe({
                "status": "error",
                "engine_version": "quant_backtest_decision_stack_v1",
                "symbol": symbol,
                "reason": "Not enough confirmed candles for backtest.",
                "entry_candles": len(entry_df),
                "confirm_candles": len(confirm_df),
                "minimum_entry_required": 250,
                "minimum_confirm_required": 220
            })

        entry_df = calculate_indicators(entry_df)
        confirm_df = calculate_indicators(confirm_df)

        spread_filter_template = evaluate_spread_filter(spread_pips, max_allowed_spread_pips)
        risk_gate_template = evaluate_low_capital_risk_gate(
            account_balance=account_balance,
            risk_percent=risk_percent,
            stop_loss_pips=stop_loss_pips,
            pip_value_per_standard_lot=pip_value_per_standard_lot,
            broker_min_lot_size=broker_min_lot_size,
            broker_lot_step=broker_lot_step
        )

        trades = []
        block_counts = {}
        raw_signal_counts = {
            "BUY": 0,
            "SELL": 0,
            "HOLD": 0
        }

        start_index = 220
        end_index = len(entry_df) - max_hold_bars - 1

        if end_index <= start_index:
            return make_json_safe({
                "status": "error",
                "engine_version": "quant_backtest_decision_stack_v1",
                "symbol": symbol,
                "reason": "Not enough candles after indicator warmup and max_hold_bars.",
                "entry_candles": len(entry_df),
                "start_index": start_index,
                "end_index": end_index
            })

        def add_block(reason: str):
            block_counts[reason] = block_counts.get(reason, 0) + 1

        for i in range(start_index, end_index):
            entry_slice = entry_df.iloc[:i + 1].reset_index(drop=True)
            signal = build_signal(entry_slice)

            if signal is None:
                signal = {
                    "action": "HOLD",
                    "reason": "Entry signal returned None.",
                    "checks": {},
                    "warnings": []
                }

            raw_action = signal.get("action", "HOLD")
            raw_signal_counts[raw_action] = raw_signal_counts.get(raw_action, 0) + 1

            if raw_action not in ["BUY", "SELL"]:
                continue

            current = entry_slice.iloc[-1]
            decision_time = current["datetime"]

            confirm_slice = confirm_df[confirm_df["datetime"] <= decision_time].reset_index(drop=True)

            if len(confirm_slice) < 210:
                add_block("Not enough M15 confirmation candles.")
                continue

            confirm_bias = build_higher_timeframe_bias(confirm_slice, label=confirm_timeframe.upper())

            if confirm_bias is None:
                confirm_bias = {
                    "bias": "unknown",
                    "reason": "Higher-timeframe confirmation returned None."
                }

            confidence = calculate_confidence_score(signal, current)

            if confidence is None:
                confidence = {
                    "score": 0,
                    "label": "weak_or_no_trade",
                    "reasons": ["Confidence returned None."]
                }

            spread_filter = evaluate_spread_filter(spread_pips, max_allowed_spread_pips)

            if spread_filter is None:
                spread_filter = {
                    "spread_status": "error",
                    "trade_allowed": False,
                    "reason": "Spread filter returned None."
                }

            risk_gate = evaluate_low_capital_risk_gate(
                account_balance=account_balance,
                risk_percent=risk_percent,
                stop_loss_pips=stop_loss_pips,
                pip_value_per_standard_lot=pip_value_per_standard_lot,
                broker_min_lot_size=broker_min_lot_size,
                broker_lot_step=broker_lot_step
            )

            if risk_gate is None:
                risk_gate = {
                    "risk_gate_status": "error",
                    "risk_gate_passed": False,
                    "reason": "Risk gate returned None."
                }

            close = current["close"]
            ema_200 = current["ema_200"]
            atr_14 = current["atr_14"]

            ema_distance_atr = None
            compression_zone = False

            if atr_14 and atr_14 > 0:
                ema_distance_atr = abs(close - ema_200) / atr_14
                compression_zone = bool(ema_distance_atr <= 0.25)

            if raw_action == "BUY" and confirm_bias.get("bias") != "bullish":
                add_block("M15 confirmation blocked BUY.")
                continue

            if raw_action == "SELL" and confirm_bias.get("bias") != "bearish":
                add_block("M15 confirmation blocked SELL.")
                continue

            if compression_zone:
                add_block("Compression zone blocked trade.")
                continue

            if confidence.get("score", 0) < 60:
                add_block("Confidence score below 60.")
                continue

            if not spread_filter.get("trade_allowed", True):
                add_block("Spread filter blocked trade.")
                continue

            if not risk_gate.get("risk_gate_passed", True):
                add_block("Risk gate blocked trade.")
                continue

            # Execute on next candle open to reduce lookahead bias.
            execution_bar = entry_df.iloc[i + 1]
            entry_price = execution_bar["open"]
            entry_time = execution_bar["datetime"]

            sl_distance = stop_loss_pips * pip_size
            tp_distance = stop_loss_pips * reward_risk * pip_size

            if raw_action == "BUY":
                stop_price = entry_price - sl_distance
                target_price = entry_price + tp_distance
            else:
                stop_price = entry_price + sl_distance
                target_price = entry_price - tp_distance

            outcome = "timeout"
            r_result = 0.0
            exit_price = None
            exit_time = None

            final_j = min(i + 1 + max_hold_bars, len(entry_df) - 1)

            for j in range(i + 1, final_j + 1):
                bar = entry_df.iloc[j]
                high = bar["high"]
                low = bar["low"]

                if raw_action == "BUY":
                    hit_stop = low <= stop_price
                    hit_target = high >= target_price

                    # Conservative assumption: if both hit in same candle, count loss.
                    if hit_stop and hit_target:
                        outcome = "loss"
                        r_result = -1.0
                        exit_price = stop_price
                        exit_time = bar["datetime"]
                        break

                    if hit_stop:
                        outcome = "loss"
                        r_result = -1.0
                        exit_price = stop_price
                        exit_time = bar["datetime"]
                        break

                    if hit_target:
                        outcome = "win"
                        r_result = reward_risk
                        exit_price = target_price
                        exit_time = bar["datetime"]
                        break

                if raw_action == "SELL":
                    hit_stop = high >= stop_price
                    hit_target = low <= target_price

                    # Conservative assumption: if both hit in same candle, count loss.
                    if hit_stop and hit_target:
                        outcome = "loss"
                        r_result = -1.0
                        exit_price = stop_price
                        exit_time = bar["datetime"]
                        break

                    if hit_stop:
                        outcome = "loss"
                        r_result = -1.0
                        exit_price = stop_price
                        exit_time = bar["datetime"]
                        break

                    if hit_target:
                        outcome = "win"
                        r_result = reward_risk
                        exit_price = target_price
                        exit_time = bar["datetime"]
                        break

            if outcome == "timeout":
                timeout_bar = entry_df.iloc[final_j]
                exit_price = timeout_bar["close"]
                exit_time = timeout_bar["datetime"]

                if raw_action == "BUY":
                    pnl_pips = (exit_price - entry_price) / pip_size
                else:
                    pnl_pips = (entry_price - exit_price) / pip_size

                r_result = round(pnl_pips / stop_loss_pips, 3)

                if r_result > 0:
                    outcome = "timeout_win"
                elif r_result < 0:
                    outcome = "timeout_loss"
                else:
                    outcome = "breakeven"

            session_context = classify_trading_session(pd.Timestamp(entry_time))

            trades.append({
                "decision_time": decision_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "action": raw_action,
                "entry_price": round_float(entry_price, 5),
                "exit_price": round_float(exit_price, 5),
                "stop_price": round_float(stop_price, 5),
                "target_price": round_float(target_price, 5),
                "outcome": outcome,
                "r_result": round(r_result, 3),
                "confidence_score": confidence.get("score", 0),
                "m15_bias": confirm_bias.get("bias"),
                "session_context": session_context,
                "spread_status": spread_filter.get("spread_status"),
                "risk_gate_status": risk_gate.get("risk_gate_status"),
                "rounded_lot_size": risk_gate.get("rounded_lot_size")
            })

        total_trades = len(trades)
        wins = [t for t in trades if t["r_result"] > 0]
        losses = [t for t in trades if t["r_result"] < 0]
        breakeven = [t for t in trades if t["r_result"] == 0]

        gross_win_r = sum(t["r_result"] for t in wins)
        gross_loss_r = abs(sum(t["r_result"] for t in losses))

        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        avg_r = (sum(t["r_result"] for t in trades) / total_trades) if total_trades > 0 else 0
        total_r = sum(t["r_result"] for t in trades)
        profit_factor = (gross_win_r / gross_loss_r) if gross_loss_r > 0 else None

        max_losing_streak = 0
        current_losing_streak = 0

        for t in trades:
            if t["r_result"] < 0:
                current_losing_streak += 1
                max_losing_streak = max(max_losing_streak, current_losing_streak)
            else:
                current_losing_streak = 0

        session_breakdown = {}

        for t in trades:
            for session in t.get("session_context", []):
                if session not in session_breakdown:
                    session_breakdown[session] = {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "total_r": 0.0
                    }

                session_breakdown[session]["trades"] += 1
                session_breakdown[session]["total_r"] += t["r_result"]

                if t["r_result"] > 0:
                    session_breakdown[session]["wins"] += 1
                elif t["r_result"] < 0:
                    session_breakdown[session]["losses"] += 1

        for session, stats in session_breakdown.items():
            trades_count = stats["trades"]
            stats["win_rate"] = round((stats["wins"] / trades_count) * 100, 2) if trades_count else 0
            stats["avg_r"] = round(stats["total_r"] / trades_count, 3) if trades_count else 0
            stats["total_r"] = round(stats["total_r"], 3)

        response = {
            "status": "ok",
            "provider": "Twelve Data",
            "engine_version": "quant_backtest_decision_stack_v1",
            "asset_class": asset_class,
            "symbol": symbol.upper(),
            "normalized_symbol": normalized_symbol,
            "entry_timeframe": entry_timeframe.upper(),
            "confirm_timeframe": confirm_timeframe.upper(),
            "lookback_bars": lookback_bars,
            "entry_candles_used": len(entry_df),
            "confirm_candles_used": len(confirm_df),
            "backtest_assumptions": {
                "execution": "next_candle_open_after_confirmed_signal",
                "confirmed_candle_mode": True,
                "same_candle_tp_sl_conflict": "counted_as_loss_conservative",
                "spread_pips": spread_pips,
                "max_allowed_spread_pips": max_allowed_spread_pips,
                "stop_loss_pips": stop_loss_pips,
                "reward_risk": reward_risk,
                "max_hold_bars": max_hold_bars,
                "account_balance": account_balance,
                "risk_percent": risk_percent,
                "broker_min_lot_size": broker_min_lot_size,
                "broker_lot_step": broker_lot_step
            },
            "gate_templates": {
                "spread_filter": spread_filter_template,
                "risk_gate": risk_gate_template
            },
            "raw_signal_counts": raw_signal_counts,
            "block_counts": block_counts,
            "performance": {
                "trades_taken": total_trades,
                "wins": len(wins),
                "losses": len(losses),
                "breakeven": len(breakeven),
                "win_rate_percent": round(win_rate, 2),
                "total_r": round(total_r, 3),
                "average_r": round(avg_r, 3),
                "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
                "max_losing_streak": max_losing_streak
            },
            "session_breakdown": session_breakdown,
            "recent_trades_sample": trades[-10:],
            "integrity_note": "Backtest is an approximation for research only. It is not financial advice and does not guarantee future results.",
            "next_upgrade": "Add H1 regime filter and walk-forward validation."
        }

        return make_json_safe(response)

    except Exception as e:
        return make_json_safe({
            "status": "error",
            "provider": "Twelve Data",
            "engine_version": "quant_backtest_decision_stack_v1",
            "symbol": symbol,
            "entry_timeframe": entry_timeframe,
            "confirm_timeframe": confirm_timeframe,
            "error": str(e),
            "reason": "Backtest failed safely."
        })


@app.get("/quant/backtest-sweep-v1")
def quant_backtest_sweep_v1(
    asset_class: str = Query(default="forex"),
    symbol: str = Query(default="EURUSD"),
    entry_timeframe: str = Query(default="M5"),
    confirm_timeframe: str = Query(default="M15"),
    lookback_bars: int = Query(default=1000, ge=300, le=5000),
    spread_pips: float = Query(default=0.8, ge=0),
    max_allowed_spread_pips: float = Query(default=1.5, ge=0.1),
    account_balance: float = Query(default=50.0, ge=1),
    risk_percent: float = Query(default=1.0, ge=0.1, le=5.0),
    pip_size: float = Query(default=0.0001, ge=0.00001),
    pip_value_per_standard_lot: float = Query(default=10.0, ge=0.01),
    broker_min_lot_size: float = Query(default=0.001, ge=0.0001),
    broker_lot_step: float = Query(default=0.001, ge=0.0001),
    min_trades_required: int = Query(default=5, ge=1, le=100)
):
    """
    Backtest Sweep v1:
    Runs multiple parameter combinations against the existing backtest endpoint.
    This is for diagnostics and parameter sensitivity only, not proof of profitability.
    """
    try:
        stop_loss_tests = [8, 10, 12, 15, 20]
        reward_risk_tests = [1.0, 1.2, 1.5]
        max_hold_tests = [12, 24, 36]

        results = []

        for sl in stop_loss_tests:
            for rr in reward_risk_tests:
                for hold in max_hold_tests:
                    bt = quant_backtest_decision_stack_v1(
                        asset_class=asset_class,
                        symbol=symbol,
                        entry_timeframe=entry_timeframe,
                        confirm_timeframe=confirm_timeframe,
                        lookback_bars=lookback_bars,
                        spread_pips=spread_pips,
                        max_allowed_spread_pips=max_allowed_spread_pips,
                        account_balance=account_balance,
                        risk_percent=risk_percent,
                        stop_loss_pips=float(sl),
                        reward_risk=float(rr),
                        max_hold_bars=int(hold),
                        pip_size=pip_size,
                        pip_value_per_standard_lot=pip_value_per_standard_lot,
                        broker_min_lot_size=broker_min_lot_size,
                        broker_lot_step=broker_lot_step
                    )

                    if not isinstance(bt, dict):
                        results.append({
                            "status": "error",
                            "stop_loss_pips": sl,
                            "reward_risk": rr,
                            "max_hold_bars": hold,
                            "reason": "Backtest returned non-dict response."
                        })
                        continue

                    if bt.get("status") != "ok":
                        results.append({
                            "status": "error",
                            "stop_loss_pips": sl,
                            "reward_risk": rr,
                            "max_hold_bars": hold,
                            "reason": bt.get("reason", "Backtest failed."),
                            "error": bt.get("error")
                        })
                        continue

                    perf = bt.get("performance", {})
                    raw_counts = bt.get("raw_signal_counts", {})
                    blocks = bt.get("block_counts", {})

                    trades_taken = perf.get("trades_taken", 0)
                    profit_factor = perf.get("profit_factor")
                    total_r = perf.get("total_r", 0)
                    avg_r = perf.get("average_r", 0)
                    win_rate = perf.get("win_rate_percent", 0)
                    max_losing_streak = perf.get("max_losing_streak", 0)

                    # Ranking score: conservative and simple.
                    # Penalize too few trades and long losing streaks.
                    if trades_taken < min_trades_required:
                        rank_score = -999
                    else:
                        pf_component = profit_factor if profit_factor is not None else 0
                        rank_score = (
                            (float(total_r) * 10)
                            + (float(avg_r) * 100)
                            + (float(pf_component) * 5)
                            - (float(max_losing_streak) * 2)
                        )

                    results.append({
                        "status": "ok",
                        "stop_loss_pips": sl,
                        "reward_risk": rr,
                        "max_hold_bars": hold,
                        "rank_score": round(rank_score, 3),
                        "trades_taken": trades_taken,
                        "wins": perf.get("wins", 0),
                        "losses": perf.get("losses", 0),
                        "win_rate_percent": win_rate,
                        "total_r": total_r,
                        "average_r": avg_r,
                        "profit_factor": profit_factor,
                        "max_losing_streak": max_losing_streak,
                        "raw_signal_counts": raw_counts,
                        "block_counts": blocks
                    })

        valid_results = [r for r in results if r.get("status") == "ok"]
        error_results = [r for r in results if r.get("status") != "ok"]

        ranked = sorted(
            valid_results,
            key=lambda x: (
                x.get("rank_score", -999),
                x.get("total_r", -999),
                x.get("profit_factor") or 0,
                -x.get("max_losing_streak", 999)
            ),
            reverse=True
        )

        profitable = [
            r for r in valid_results
            if r.get("trades_taken", 0) >= min_trades_required
            and r.get("total_r", 0) > 0
            and (r.get("profit_factor") or 0) > 1
        ]

        negative = [
            r for r in valid_results
            if r.get("trades_taken", 0) >= min_trades_required
            and r.get("total_r", 0) <= 0
        ]

        best = ranked[0] if ranked else None

        verdict = "not_enough_data"

        if best:
            if best.get("trades_taken", 0) < min_trades_required:
                verdict = "not_enough_trades_to_trust"
            elif best.get("total_r", 0) > 0 and (best.get("profit_factor") or 0) > 1:
                verdict = "some_parameter_sets_show_potential_but_need_out_of_sample_validation"
            else:
                verdict = "current_strategy_stack_not_validated"

        response = {
            "status": "ok",
            "engine_version": "quant_backtest_sweep_v1",
            "asset_class": asset_class,
            "symbol": symbol.upper(),
            "entry_timeframe": entry_timeframe.upper(),
            "confirm_timeframe": confirm_timeframe.upper(),
            "lookback_bars": lookback_bars,
            "sweep_space": {
                "stop_loss_pips": stop_loss_tests,
                "reward_risk": reward_risk_tests,
                "max_hold_bars": max_hold_tests,
                "combinations_tested": len(results),
                "min_trades_required": min_trades_required
            },
            "shared_assumptions": {
                "spread_pips": spread_pips,
                "max_allowed_spread_pips": max_allowed_spread_pips,
                "account_balance": account_balance,
                "risk_percent": risk_percent,
                "broker_min_lot_size": broker_min_lot_size,
                "broker_lot_step": broker_lot_step,
                "pip_size": pip_size,
                "pip_value_per_standard_lot": pip_value_per_standard_lot
            },
            "verdict": verdict,
            "best_result": best,
            "top_10_results": ranked[:10],
            "profitable_result_count": len(profitable),
            "negative_result_count": len(negative),
            "error_result_count": len(error_results),
            "integrity_note": "Parameter sweep is diagnostic only. It can reveal sensitivity and overfitting risk, but it does not prove future profitability.",
            "next_upgrade": "If no stable positive parameter zone appears, improve entry logic before adding execution."
        }

        return make_json_safe(response)

    except Exception as e:
        return make_json_safe({
            "status": "error",
            "engine_version": "quant_backtest_sweep_v1",
            "symbol": symbol,
            "error": str(e),
            "reason": "Backtest sweep failed safely."
        })

