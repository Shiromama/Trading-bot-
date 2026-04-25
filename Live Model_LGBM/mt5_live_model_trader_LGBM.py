from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ==========================================================
# LIVE TRADING SETTINGS
# ==========================================================
SYMBOL = "BTCUSDm"
TIMEFRAME = mt5.TIMEFRAME_M1
POLL_SECONDS = 2
HOLD_BARS = 30
MAGIC_NUMBER = 20260423
ORDER_COMMENT = "lgbm-live"
ALLOW_LONG = True
ALLOW_SHORT = True
ONE_POSITION_PER_SYMBOL = True
ONLY_TRADE_SIGNALS = True
CONFIDENCE_THRESHOLD = 0.55
MAX_SPREAD_POINTS = None          # e.g. 500, or None to disable
DEVIATION = 50                    # max slippage in points
DRY_RUN = False                   # True = log signals only, no real orders
STRICT_FEATURE_MATCH = True       # safer: abort if live features don't match training features

# Risk / sizing
LOT_SIZE = 0.01
STOP_LOSS_ACCOUNT_FRACTION = 0.15
RISK_REWARD_RATIO = 2.0
USE_TAKE_PROFIT = True
MIN_BALANCE_TO_TRADE = 1.0

# MT5 login
# Safer approach: leave these blank and use an already logged-in MT5 terminal
LOGIN = 145679563
PASSWORD = "Wintermelon@25"
SERVER = "Exness-MT5Real17"
TERMINAL_PATH = None

# Files
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "lgbm_model.pkl"
FEATURES_PATH = SCRIPT_DIR / "lgbm_features.pkl"
STATE_PATH = SCRIPT_DIR / "mt5_live_state.json"
LOG_PATH = SCRIPT_DIR / "mt5_live_trade_log.csv"

EPS = 1e-9

# ==========================================================
# HELPERS
# ==========================================================
@dataclass
class Signal:
    bar_time: int
    predicted_class: int
    confidence: float
    features: pd.DataFrame
    raw_probabilities: Dict[str, float]


TIMEFRAME_TO_SECONDS = {
    mt5.TIMEFRAME_M1: 60,
    mt5.TIMEFRAME_M2: 120,
    mt5.TIMEFRAME_M3: 180,
    mt5.TIMEFRAME_M4: 240,
    mt5.TIMEFRAME_M5: 300,
    mt5.TIMEFRAME_M6: 360,
    mt5.TIMEFRAME_M10: 600,
    mt5.TIMEFRAME_M12: 720,
    mt5.TIMEFRAME_M15: 900,
    mt5.TIMEFRAME_M20: 1200,
    mt5.TIMEFRAME_M30: 1800,
    mt5.TIMEFRAME_H1: 3600,
    mt5.TIMEFRAME_H2: 7200,
    mt5.TIMEFRAME_H3: 10800,
    mt5.TIMEFRAME_H4: 14400,
    mt5.TIMEFRAME_H6: 21600,
    mt5.TIMEFRAME_H8: 28800,
    mt5.TIMEFRAME_H12: 43200,
    mt5.TIMEFRAME_D1: 86400,
}


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_signal_bar_time": None, "positions": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def log_message(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}")


def ensure_mt5() -> None:
    kwargs = {}
    if TERMINAL_PATH:
        kwargs["path"] = TERMINAL_PATH

    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    if LOGIN and PASSWORD and SERVER:
        if not mt5.login(login=LOGIN, password=PASSWORD, server=SERVER):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")

    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"Could not read account info: {mt5.last_error()}")

    log_message(
        f"Connected | login={info.login} server={info.server} "
        f"balance={info.balance:.2f} equity={info.equity:.2f}"
    )


def ensure_symbol(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol not found: {symbol}")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol: {symbol}")
    return mt5.symbol_info(symbol)


def get_latest_rates(symbol: str, timeframe: int, count: int = 400) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def mt5_rates_to_price_df(rates: pd.DataFrame) -> pd.DataFrame:
    df = rates.copy()
    rename_map = {"tick_volume": "volume", "spread": "spread_points"}
    df = df.rename(columns=rename_map)
    keep = ["time", "open", "high", "low", "close", "volume", "spread_points"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()
    df["datetime"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("datetime")
    return df


# ==========================================================
# FEATURE ENGINEERING (must match training builder)
# ==========================================================
def create_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()

    # Basic candle features
    df[f"{prefix}_return"] = df["close"].pct_change()
    df[f"{prefix}_log_return"] = np.log(df["close"] / df["close"].shift(1))

    df[f"{prefix}_range"] = df["high"] - df["low"]
    df[f"{prefix}_body"] = df["close"] - df["open"]

    df[f"{prefix}_upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df[f"{prefix}_lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    range_safe = df[f"{prefix}_range"].replace(0, np.nan)

    # Candle ratios
    df[f"{prefix}_body_ratio"] = df[f"{prefix}_body"] / (range_safe + EPS)
    df[f"{prefix}_upper_wick_ratio"] = df[f"{prefix}_upper_wick"] / (range_safe + EPS)
    df[f"{prefix}_lower_wick_ratio"] = df[f"{prefix}_lower_wick"] / (range_safe + EPS)

    df[f"{prefix}_close_pos_in_range"] = (
        (df["close"] - df["low"]) / (range_safe + EPS)
    )
    df[f"{prefix}_open_pos_in_range"] = (
        (df["open"] - df["low"]) / (range_safe + EPS)
    )

    # Candle direction
    df[f"{prefix}_is_bull"] = (df["close"] > df["open"]).astype(int)
    df[f"{prefix}_is_bear"] = (df["close"] < df["open"]).astype(int)

    # Moving averages
    df[f"{prefix}_ma10"] = df["close"].rolling(10).mean()
    df[f"{prefix}_ma20"] = df["close"].rolling(20).mean()

    df[f"{prefix}_ma10_dist"] = (
        (df["close"] - df[f"{prefix}_ma10"]) / (df[f"{prefix}_ma10"] + EPS)
    )
    df[f"{prefix}_ma20_dist"] = (
        (df["close"] - df[f"{prefix}_ma20"]) / (df[f"{prefix}_ma20"] + EPS)
    )

    # MA slopes
    df[f"{prefix}_ma10_slope"] = df[f"{prefix}_ma10"].pct_change()
    df[f"{prefix}_ma20_slope"] = df[f"{prefix}_ma20"].pct_change()

    # Trend state
    df[f"{prefix}_close_above_ma10"] = (df["close"] > df[f"{prefix}_ma10"]).astype(int)
    df[f"{prefix}_close_above_ma20"] = (df["close"] > df[f"{prefix}_ma20"]).astype(int)
    df[f"{prefix}_ma10_above_ma20"] = (df[f"{prefix}_ma10"] > df[f"{prefix}_ma20"]).astype(int)

    # Volatility
    df[f"{prefix}_volatility10"] = df["close"].rolling(10).std()
    df[f"{prefix}_range_mean5"] = df[f"{prefix}_range"].rolling(5).mean()
    df[f"{prefix}_range_mean10"] = df[f"{prefix}_range"].rolling(10).mean()
    df[f"{prefix}_range_std10"] = df[f"{prefix}_range"].rolling(10).std()

    # Expansion / compression
    df[f"{prefix}_range_vs_mean5"] = (
        df[f"{prefix}_range"] / (df[f"{prefix}_range_mean5"] + EPS)
    )
    df[f"{prefix}_range_vs_mean10"] = (
        df[f"{prefix}_range"] / (df[f"{prefix}_range_mean10"] + EPS)
    )
    df[f"{prefix}_vol_compression10"] = (
        df[f"{prefix}_range_std10"] / (df[f"{prefix}_range_mean10"] + EPS)
    )

    # Candle-to-candle changes
    df[f"{prefix}_range_change1"] = (
        df[f"{prefix}_range"] / (df[f"{prefix}_range"].shift(1) + EPS)
    )
    df[f"{prefix}_body_change1"] = (
        df[f"{prefix}_body"].abs() / (df[f"{prefix}_body"].shift(1).abs() + EPS)
    )

    # Volume
    df[f"{prefix}_volume_change"] = df["volume"].pct_change()
    df[f"{prefix}_volume_mean10"] = df["volume"].rolling(10).mean()
    df[f"{prefix}_volume_vs_mean10"] = (
        df["volume"] / (df[f"{prefix}_volume_mean10"] + EPS)
    )

    # Momentum
    df[f"{prefix}_momentum3"] = df["close"] / df["close"].shift(3) - 1
    df[f"{prefix}_momentum5"] = df["close"] / df["close"].shift(5) - 1
    df[f"{prefix}_momentum10"] = df["close"] / df["close"].shift(10) - 1

    # Momentum acceleration
    df[f"{prefix}_momentum3_accel"] = (
        df[f"{prefix}_momentum3"] - df[f"{prefix}_momentum3"].shift(1)
    )
    df[f"{prefix}_momentum5_accel"] = (
        df[f"{prefix}_momentum5"] - df[f"{prefix}_momentum5"].shift(1)
    )
    df[f"{prefix}_momentum10_accel"] = (
        df[f"{prefix}_momentum10"] - df[f"{prefix}_momentum10"].shift(1)
    )

    # Smoothed return context
    df[f"{prefix}_return_mean3"] = df[f"{prefix}_return"].rolling(3).mean()
    df[f"{prefix}_return_mean5"] = df[f"{prefix}_return"].rolling(5).mean()
    df[f"{prefix}_return_std5"] = df[f"{prefix}_return"].rolling(5).std()
    df[f"{prefix}_return_std10"] = df[f"{prefix}_return"].rolling(10).std()

    # Scalp context
    df[f"{prefix}_body_vs_range_mean10"] = (
        df[f"{prefix}_body"].abs() / (df[f"{prefix}_range_mean10"] + EPS)
    )
    df[f"{prefix}_wick_imbalance"] = (
        df[f"{prefix}_lower_wick"] - df[f"{prefix}_upper_wick"]
    ) / (range_safe + EPS)

    return df


def fetch_aligned_feature_frames(symbol: str, m1_count: int = 400) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rates_1m = get_latest_rates(symbol, mt5.TIMEFRAME_M1, count=m1_count)
    rates_5m = get_latest_rates(symbol, mt5.TIMEFRAME_M5, count=max(150, math.ceil(m1_count / 5) + 80))
    rates_15m = get_latest_rates(symbol, mt5.TIMEFRAME_M15, count=max(150, math.ceil(m1_count / 15) + 80))

    df_1m = mt5_rates_to_price_df(rates_1m)
    df_5m = mt5_rates_to_price_df(rates_5m)
    df_15m = mt5_rates_to_price_df(rates_15m)

    df_5m = df_5m.reindex(df_1m.index, method="ffill")
    df_15m = df_15m.reindex(df_1m.index, method="ffill")

    return df_1m, df_5m, df_15m


def make_feature_frame(symbol: str) -> pd.DataFrame:
    df_1m_raw, df_5m_raw, df_15m_raw = fetch_aligned_feature_frames(symbol, m1_count=400)

    df_1m = create_features(df_1m_raw, "1m")
    df_5m = create_features(df_5m_raw, "5m")
    df_15m = create_features(df_15m_raw, "15m")

    feat = pd.DataFrame(index=df_1m.index)

    for src in (df_1m, df_5m, df_15m):
        prefixed_cols = [c for c in src.columns if c.startswith(("1m_", "5m_", "15m_"))]
        if prefixed_cols:
            feat = pd.concat([feat, src[prefixed_cols]], axis=1)

    # Cross-timeframe features
    feat["align_1m_5m_ma10"] = (
        (df_1m["1m_ma10_above_ma20"] == df_5m["5m_ma10_above_ma20"]).astype(int)
    )
    feat["align_1m_15m_ma10"] = (
        (df_1m["1m_ma10_above_ma20"] == df_15m["15m_ma10_above_ma20"]).astype(int)
    )
    feat["align_5m_15m_ma10"] = (
        (df_5m["5m_ma10_above_ma20"] == df_15m["15m_ma10_above_ma20"]).astype(int)
    )

    feat["triple_trend_alignment"] = (
        (df_1m["1m_ma10_above_ma20"] == df_5m["5m_ma10_above_ma20"]) &
        (df_5m["5m_ma10_above_ma20"] == df_15m["15m_ma10_above_ma20"])
    ).astype(int)

    feat["triple_bull_alignment"] = (
        (df_1m["1m_close_above_ma20"] == 1) &
        (df_5m["5m_close_above_ma20"] == 1) &
        (df_15m["15m_close_above_ma20"] == 1)
    ).astype(int)

    feat["triple_bear_alignment"] = (
        (df_1m["1m_close_above_ma20"] == 0) &
        (df_5m["5m_close_above_ma20"] == 0) &
        (df_15m["15m_close_above_ma20"] == 0)
    ).astype(int)

    feat["momentum_alignment_score"] = (
        np.sign(df_1m["1m_momentum5"]).fillna(0) +
        np.sign(df_5m["5m_momentum5"]).fillna(0) +
        np.sign(df_15m["15m_momentum5"]).fillna(0)
    )

    feat["1m_vs_5m_range_ratio"] = df_1m["1m_range_mean10"] / (df_5m["5m_range_mean10"] + EPS)
    feat["1m_vs_15m_range_ratio"] = df_1m["1m_range_mean10"] / (df_15m["15m_range_mean10"] + EPS)

    feat["1m_body_vs_5m_range"] = df_1m["1m_body"].abs() / (df_5m["5m_range_mean10"] + EPS)
    feat["1m_body_vs_15m_range"] = df_1m["1m_body"].abs() / (df_15m["15m_range_mean10"] + EPS)

    feat["entry_price"] = df_1m_raw["close"]
    feat["spread_points"] = df_1m_raw["spread_points"]

    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


def build_live_feature_row(feature_list: list[str], symbol: str) -> tuple[pd.DataFrame, int]:
    feat_df = make_feature_frame(symbol)

    # Use last fully closed 1m candle
    row = feat_df.iloc[-2].copy()
    bar_time = int(feat_df.index[-2].timestamp())

    missing = [f for f in feature_list if f not in row.index or pd.isna(row[f])]
    if missing and STRICT_FEATURE_MATCH:
        raise RuntimeError(f"Missing {len(missing)} feature(s) in live builder: {missing}")

    data = {}
    for feature in feature_list:
        value = row[feature] if feature in row.index else np.nan
        if pd.isna(value):
            value = 0.0
        if isinstance(value, (np.floating, np.integer, float, int)):
            value = float(value)
        data[feature] = value

    live_row = pd.DataFrame([data], columns=feature_list)
    live_row = live_row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return live_row, bar_time


def predict_signal(model, feature_list: list[str], symbol: str) -> Signal:
    feature_row, bar_time = build_live_feature_row(feature_list, symbol)
    y_pred = int(model.predict(feature_row)[0])
    y_prob = model.predict_proba(feature_row)[0]
    class_order = list(model.classes_)
    prob_map = {str(cls): float(prob) for cls, prob in zip(class_order, y_prob)}
    confidence = float(np.max(y_prob))

    return Signal(
        bar_time=bar_time,
        predicted_class=y_pred,
        confidence=confidence,
        features=feature_row,
        raw_probabilities=prob_map,
    )


# ==========================================================
# EXECUTION HELPERS
# ==========================================================
def get_symbol_tick(symbol: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}: {mt5.last_error()}")
    return tick


def current_spread_points(symbol_info, tick) -> float:
    if symbol_info.point <= 0:
        return 0.0
    return (tick.ask - tick.bid) / symbol_info.point


def get_open_positions(symbol: str, magic: Optional[int] = None):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    if magic is None:
        return list(positions)
    return [p for p in positions if p.magic == magic]


def order_type_from_signal(signal_class: int) -> int:
    if signal_class == 1:
        return mt5.ORDER_TYPE_BUY
    if signal_class == -1:
        return mt5.ORDER_TYPE_SELL
    raise ValueError(f"Unsupported signal class: {signal_class}")


def normalize_price(symbol_info, price: float) -> float:
    return round(float(price), int(symbol_info.digits))


def calc_target_price_by_pnl(symbol: str, order_type: int, volume: float, entry_price: float, target_pnl: float) -> float:
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    if point <= 0:
        raise RuntimeError(f"Invalid point size for {symbol}")

    is_buy = order_type == mt5.ORDER_TYPE_BUY

    if target_pnl < 0:
        near = entry_price
        far = entry_price - 1000 * point if is_buy else entry_price + 1000 * point
        step_dir = -1 if is_buy else 1
    else:
        near = entry_price
        far = entry_price + 1000 * point if is_buy else entry_price - 1000 * point
        step_dir = 1 if is_buy else -1

    for _ in range(30):
        pnl_far = mt5.order_calc_profit(order_type, symbol, volume, entry_price, far)
        if pnl_far is None:
            raise RuntimeError(f"order_calc_profit failed: {mt5.last_error()}")
        if (target_pnl < 0 and pnl_far <= target_pnl) or (target_pnl > 0 and pnl_far >= target_pnl):
            break
        far += step_dir * abs(far - near)
    else:
        raise RuntimeError(f"Could not bracket target pnl={target_pnl} for {symbol}")

    lo = min(near, far)
    hi = max(near, far)

    for _ in range(60):
        mid = (lo + hi) / 2.0
        pnl_mid = mt5.order_calc_profit(order_type, symbol, volume, entry_price, mid)
        if pnl_mid is None:
            raise RuntimeError(f"order_calc_profit failed mid-search: {mt5.last_error()}")

        if pnl_mid < target_pnl:
            if is_buy:
                lo = mid
            else:
                hi = mid
        else:
            if is_buy:
                hi = mid
            else:
                lo = mid

    return normalize_price(symbol_info, (lo + hi) / 2.0)


def compute_live_sl_tp(symbol: str, order_type: int, volume: float, entry_price: float, balance: float) -> tuple[float, Optional[float], float, float]:
    risk_usd = balance * STOP_LOSS_ACCOUNT_FRACTION
    reward_usd = risk_usd * RISK_REWARD_RATIO

    sl_price = calc_target_price_by_pnl(symbol, order_type, volume, entry_price, -risk_usd)
    tp_price = None
    if USE_TAKE_PROFIT:
        tp_price = calc_target_price_by_pnl(symbol, order_type, volume, entry_price, reward_usd)

    return sl_price, tp_price, risk_usd, reward_usd


def place_market_order(symbol: str, signal: Signal) -> Optional[dict]:
    account = mt5.account_info()
    symbol_info = ensure_symbol(symbol)
    tick = get_symbol_tick(symbol)

    if account is None:
        raise RuntimeError("No account info available")
    if account.balance < MIN_BALANCE_TO_TRADE:
        log_message(f"Balance too low to trade: {account.balance:.2f}")
        return None

    spread_pts = current_spread_points(symbol_info, tick)
    if MAX_SPREAD_POINTS is not None and spread_pts > MAX_SPREAD_POINTS:
        log_message(f"Spread too high: {spread_pts:.2f} > {MAX_SPREAD_POINTS}")
        return None

    order_type = order_type_from_signal(signal.predicted_class)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    price = normalize_price(symbol_info, price)

    volume = float(LOT_SIZE)
    sl_price, tp_price, risk_usd, reward_usd = compute_live_sl_tp(
        symbol=symbol,
        order_type=order_type,
        volume=volume,
        entry_price=price,
        balance=float(account.balance),
    )

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price if tp_price is not None else 0.0,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": ORDER_COMMENT,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    log_message(
        f"SIGNAL -> class={signal.predicted_class} conf={signal.confidence:.4f} "
        f"price={price} sl={sl_price} tp={tp_price} "
        f"risk=${risk_usd:.2f} reward=${reward_usd:.2f} spread_pts={spread_pts:.2f}"
    )

    if DRY_RUN:
        return {
            "retcode": "DRY_RUN",
            "request": request,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "volume": volume,
        }

    check = mt5.order_check(request)
    if check is None:
        raise RuntimeError(f"order_check failed: {mt5.last_error()}")
    if check.retcode not in (0, mt5.TRADE_RETCODE_DONE):
        log_message(f"order_check retcode={check.retcode} comment={getattr(check, 'comment', '')}")

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send failed: {mt5.last_error()}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log_message(f"Order rejected: retcode={result.retcode} comment={result.comment}")
        return None

    log_message(f"ORDER OPENED -> order={result.order} deal={result.deal} retcode={result.retcode}")
    return result._asdict()


def close_position(position) -> Optional[dict]:
    symbol_info = ensure_symbol(position.symbol)
    tick = get_symbol_tick(position.symbol)

    if position.type == mt5.POSITION_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = normalize_price(symbol_info, tick.bid)
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = normalize_price(symbol_info, tick.ask)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": close_type,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC_NUMBER,
        "comment": f"{ORDER_COMMENT}-timeout",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if DRY_RUN:
        log_message(f"DRY_RUN CLOSE -> ticket={position.ticket} price={price}")
        return {"retcode": "DRY_RUN_CLOSE", "request": request}

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"close order_send failed: {mt5.last_error()}")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log_message(f"Close rejected: retcode={result.retcode} comment={result.comment}")
        return None

    log_message(f"POSITION CLOSED -> ticket={position.ticket} deal={result.deal}")
    return result._asdict()


def append_trade_log(row: dict) -> None:
    df = pd.DataFrame([row])
    header = not LOG_PATH.exists()
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)


def manage_timeouts(symbol: str, timeframe: int, state: dict) -> None:
    if DRY_RUN:
        return

    rates = get_latest_rates(symbol, timeframe, count=5)
    latest_closed_bar_time = int(rates.iloc[-2]["time"].timestamp())
    tf_seconds = TIMEFRAME_TO_SECONDS.get(timeframe)
    if tf_seconds is None:
        raise RuntimeError(f"Unsupported timeframe for timeout logic: {timeframe}")

    positions = get_open_positions(symbol, MAGIC_NUMBER)
    live_tickets = {str(p.ticket) for p in positions}

    state["positions"] = {
        k: v for k, v in state.get("positions", {}).items()
        if k in live_tickets
    }

    for pos in positions:
        ticket_key = str(pos.ticket)
        meta = state.get("positions", {}).get(ticket_key)
        if not meta:
            continue

        entry_bar_time = int(meta["entry_bar_time"])
        bars_open = max(0, round((latest_closed_bar_time - entry_bar_time) / tf_seconds))

        if bars_open >= HOLD_BARS:
            log_message(f"TIMEOUT EXIT -> ticket={pos.ticket} bars_open={bars_open}")
            result = close_position(pos)
            if result is not None:
                append_trade_log({
                    "timestamp": datetime.now().isoformat(),
                    "event": "timeout_close",
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "bars_open": bars_open,
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "profit": pos.profit,
                })
                state["positions"].pop(ticket_key, None)


def should_skip_signal(signal: Signal) -> bool:
    if signal.confidence < CONFIDENCE_THRESHOLD:
        return True
    if ONLY_TRADE_SIGNALS and signal.predicted_class == 0:
        return True
    if signal.predicted_class == 1 and not ALLOW_LONG:
        return True
    if signal.predicted_class == -1 and not ALLOW_SHORT:
        return True
    return False


# ==========================================================
# MAIN
# ==========================================================
def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_list = list(joblib.load(FEATURES_PATH))

    ensure_mt5()
    ensure_symbol(SYMBOL)
    state = load_state()

    log_message(f"Loaded model and {len(feature_list)} features")
    log_message(
        f"Starting loop | symbol={SYMBOL} timeframe={TIMEFRAME} "
        f"hold_bars={HOLD_BARS} dry_run={DRY_RUN}"
    )

    try:
        while True:
            manage_timeouts(SYMBOL, TIMEFRAME, state)

            signal = predict_signal(model, feature_list, SYMBOL)

            if state.get("last_signal_bar_time") == signal.bar_time:
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue

            state["last_signal_bar_time"] = signal.bar_time

            log_message(
                f"NEW BAR -> time={datetime.fromtimestamp(signal.bar_time, tz=timezone.utc).isoformat()} "
                f"pred={signal.predicted_class} conf={signal.confidence:.4f} "
                f"probs={signal.raw_probabilities}"
            )

            if should_skip_signal(signal):
                append_trade_log({
                    "timestamp": datetime.now().isoformat(),
                    "event": "signal_skipped",
                    "bar_time": signal.bar_time,
                    "predicted_class": signal.predicted_class,
                    "confidence": signal.confidence,
                    "probabilities": json.dumps(signal.raw_probabilities),
                    "reason": "threshold_or_filter",
                })
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue

            if ONE_POSITION_PER_SYMBOL and mt5.positions_get(symbol=SYMBOL):
                log_message("Signal ignored because a managed position is already open")
                append_trade_log({
                    "timestamp": datetime.now().isoformat(),
                    "event": "signal_skipped",
                    "bar_time": signal.bar_time,
                    "predicted_class": signal.predicted_class,
                    "confidence": signal.confidence,
                    "probabilities": json.dumps(signal.raw_probabilities),
                    "reason": "position_already_open",
                })
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue

            result = place_market_order(SYMBOL, signal)
            if result is not None:
                ticket = str(result.get("order") or result.get("deal") or f"dryrun-{signal.bar_time}")
                state.setdefault("positions", {})[ticket] = {
                    "entry_bar_time": signal.bar_time,
                    "predicted_class": signal.predicted_class,
                    "confidence": signal.confidence,
                }
                append_trade_log({
                    "timestamp": datetime.now().isoformat(),
                    "event": "order_opened" if not DRY_RUN else "dry_run_order",
                    "bar_time": signal.bar_time,
                    "ticket": ticket,
                    "symbol": SYMBOL,
                    "predicted_class": signal.predicted_class,
                    "confidence": signal.confidence,
                    "probabilities": json.dumps(signal.raw_probabilities),
                    "result": json.dumps(result, default=str),
                })

            save_state(state)
            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        log_message("Stopped by user")
    finally:
        save_state(state)
        mt5.shutdown()


if __name__ == "__main__":
    main()