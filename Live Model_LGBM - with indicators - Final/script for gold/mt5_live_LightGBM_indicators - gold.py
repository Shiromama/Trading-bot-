from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import joblib
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ==========================================================
# LIVE TRADING SETTINGS
# ==========================================================
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M1
POLL_SECONDS = 2
HOLD_BARS = 360  # closer to dataset FUTURE_SHIFT=30
MAGIC_NUMBER = 2000
ORDER_COMMENT = "lgbm-entry-style"
ALLOW_LONG = True
ALLOW_SHORT = True
ONE_POSITION_PER_SYMBOL = True
INVERT_SIGNALS = False

ONLY_TRADE_SIGNALS = True
CONFIDENCE_THRESHOLD = 0.55
MAX_SPREAD_POINTS = None
DEVIATION = 50
DRY_RUN = False
STRICT_FEATURE_MATCH = True

# New 7-class entry-style output from the updated training script.
ENTRY_STYLE_NAMES = {
    -3: "SELL_WAIT_OB",
    -2: "SELL_WAIT_FVG",
    -1: "SELL_NOW",
     0: "NO_TRADE",
     1: "BUY_NOW",
     2: "BUY_WAIT_FVG",
     3: "BUY_WAIT_OB",
}

USE_PENDING_FOR_WAIT_ENTRIES = True
PENDING_EXPIRATION_MINUTES = 10

LOT_SIZE = 0.01
STOP_LOSS_ACCOUNT_FRACTION = 0.10
RISK_REWARD_RATIO = 2.0
USE_TAKE_PROFIT = True
MIN_BALANCE_TO_TRADE = 1.0
MAX_TOTAL_POSITIONS = 2
MAX_SAME_SIDE_POSITIONS = 5

# Signal flip settings
# If an opposite trade signal appears while positions are open, close the
# opposite position(s), then allow the new signal to enter the other side.
CLOSE_OPPOSITE_ON_SIGNAL = True
ENTER_AFTER_FLIP = True
FLIP_CLOSE_DELAY_SECONDS = 0.5
CANCEL_PENDING_ON_FLIP = True

LOGIN = None
PASSWORD = ""
SERVER = ""
TERMINAL_PATH = None

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "lgbm_model.pkl"
FEATURES_PATH = SCRIPT_DIR / "lgbm_features.pkl"
STATE_PATH = SCRIPT_DIR / "mt5_live_state.json"
LOG_PATH = SCRIPT_DIR / "mt5_live_trade_log.csv"

EPS = 1e-9
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

# Order Block / Breaker Block settings - must match dataset builder
OB_SWING_LOOKBACK = 10
OB_USE_BODY = False
OB_RECENT_WINDOW = 30
OB_MAX_ACTIVE_ZONES = 30
OB_NO_ZONE_DISTANCE = 10.0
OB_NO_ZONE_AGE = 9999

# Fair Value Gap settings - must match dataset builder
FVG_THRESHOLD_PCT = 0.0
FVG_AUTO_THRESHOLD = True
FVG_RECENT_WINDOW = 30
FVG_MAX_ACTIVE_ZONES = 50
FVG_NO_ZONE_DISTANCE = 10.0
FVG_NO_ZONE_AGE = 9999

# Fractal BOS / CHoCH market structure settings - must match dataset builder
STRUCTURE_FRACTAL_LENGTH = 5
STRUCTURE_RECENT_WINDOW = 5
STRUCTURE_NO_BREAK_AGE = 9999


@dataclass
class Signal:
    bar_time: int
    predicted_class: int
    confidence: float
    features: pd.DataFrame
    raw_probabilities: Dict[str, float]


TIMEFRAME_TO_SECONDS = {
    mt5.TIMEFRAME_M1: 60,
    mt5.TIMEFRAME_M5: 300,
    mt5.TIMEFRAME_M15: 900,
    mt5.TIMEFRAME_M30: 1800,
    mt5.TIMEFRAME_H1: 3600,
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def fmt_value(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def print_table(title: str, rows: list[tuple[str, object]], value_width: int = 18) -> None:
    """
    Compact terminal table printer.

    Short tables keep the old single-column layout. Long debug tables are
    split in half and printed side by side so they fit better in VS Code.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] {title}")

    def clean_text(value, width: int) -> str:
        value_text = fmt_value(value)
        if len(value_text) > width:
            value_text = value_text[: max(0, width - 3)] + "..."
        return value_text

    def calc_widths(part_rows: list[tuple[str, object]]) -> tuple[int, int]:
        key_w = max(10, min(22, max((len(str(k)) for k, _ in part_rows), default=10)))
        val_w = max(10, min(value_width, max((len(fmt_value(v)) for _, v in part_rows), default=10)))
        return key_w, val_w

    if len(rows) <= 18:
        key_width, val_width = calc_widths(rows)
        border = "+" + "-" * (key_width + 2) + "+" + "-" * (val_width + 2) + "+"
        print(border)
        print(f"| {'Field':<{key_width}} | {'Value':<{val_width}} |")
        print(border)
        for key, value in rows:
            value_text = clean_text(value, val_width)
            print(f"| {str(key):<{key_width}} | {value_text:<{val_width}} |")
        print(border)
        return

    half = (len(rows) + 1) // 2
    left_rows = rows[:half]
    right_rows = rows[half:]

    left_key_w, left_val_w = calc_widths(left_rows)
    right_key_w, right_val_w = calc_widths(right_rows)

    border = (
        "+" + "-" * (left_key_w + 2) + "+" + "-" * (left_val_w + 2) + "+"
        + "  "
        + "+" + "-" * (right_key_w + 2) + "+" + "-" * (right_val_w + 2) + "+"
    )

    print(border)
    print(
        f"| {'Field':<{left_key_w}} | {'Value':<{left_val_w}} |  "
        f"| {'Field':<{right_key_w}} | {'Value':<{right_val_w}} |"
    )
    print(border)

    max_len = max(len(left_rows), len(right_rows))
    for i in range(max_len):
        if i < len(left_rows):
            lk, lv = left_rows[i]
            left_line = f"| {str(lk):<{left_key_w}} | {clean_text(lv, left_val_w):<{left_val_w}} |"
        else:
            left_line = f"| {'':<{left_key_w}} | {'':<{left_val_w}} |"

        if i < len(right_rows):
            rk, rv = right_rows[i]
            right_line = f"| {str(rk):<{right_key_w}} | {clean_text(rv, right_val_w):<{right_val_w}} |"
        else:
            right_line = f"| {'':<{right_key_w}} | {'':<{right_val_w}} |"

        print(left_line + "  " + right_line)

    print(border)
def signal_name(predicted_class: int) -> str:
    return ENTRY_STYLE_NAMES.get(int(predicted_class), f"UNKNOWN_{predicted_class}")


def signal_direction(predicted_class: int) -> int:
    if int(predicted_class) > 0:
        return 1
    if int(predicted_class) < 0:
        return -1
    return 0


def is_wait_entry(predicted_class: int) -> bool:
    return abs(int(predicted_class)) in (2, 3)


def position_side(position) -> int:
    if position.type == mt5.POSITION_TYPE_BUY:
        return 1
    if position.type == mt5.POSITION_TYPE_SELL:
        return -1
    return 0


def print_signal_table(signal: Signal) -> None:
    debug = getattr(signal, "debug", {}) or {}
    probs = signal.raw_probabilities
    rows = [
        ("bar_time", datetime.fromtimestamp(signal.bar_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("prediction", signal_name(signal.predicted_class)),
        ("confidence", signal.confidence),
        ("prob_sell_ob", probs.get("-3")),
        ("prob_sell_fvg", probs.get("-2")),
        ("prob_sell_now", probs.get("-1")),
        ("prob_no_trade", probs.get("0")),
        ("prob_buy_now", probs.get("1")),
        ("prob_buy_fvg", probs.get("2")),
        ("prob_buy_ob", probs.get("3")),
        ("entry_price", debug.get("entry_price")),
        ("wait_entry_price", debug.get("wait_entry_price")),
        ("spread_points", debug.get("spread_points")),
        ("1m_rsi14", debug.get("1m_rsi14")),
        ("1m_adx14", debug.get("1m_adx14")),
        ("5m_adx14", debug.get("5m_adx14")),
        ("15m_adx14", debug.get("15m_adx14")),
        ("market_trending", debug.get("market_is_trending")),
        ("market_choppy", debug.get("market_is_choppy")),
        ("1m_bos_up", debug.get("1m_bos_up")),
        ("1m_bos_down", debug.get("1m_bos_down")),
        ("1m_choch_up", debug.get("1m_choch_up")),
        ("1m_choch_down", debug.get("1m_choch_down")),
        ("5m_struct_dir", debug.get("5m_structure_direction")),
        ("15m_struct_dir", debug.get("15m_structure_direction")),
        ("struct_bull_score", debug.get("structure_bull_context_score")),
        ("struct_bear_score", debug.get("structure_bear_context_score")),
        ("struct_reversal", debug.get("structure_reversal_warning")),
        ("struct_align", debug.get("structure_direction_alignment")),
        ("15m_bull_ob", debug.get("15m_inside_bull_ob")),
        ("15m_bear_ob", debug.get("15m_inside_bear_ob")),
        ("1m_bull_ob", debug.get("1m_inside_bull_ob")),
        ("1m_bear_ob", debug.get("1m_inside_bear_ob")),
        ("bull_ob_align", debug.get("htf_ltf_bull_ob_alignment")),
        ("bear_ob_align", debug.get("htf_ltf_bear_ob_alignment")),
        ("ob_bull_score", debug.get("ob_bull_context_score")),
        ("ob_bear_score", debug.get("ob_bear_context_score")),
        ("15m_bull_fvg", debug.get("15m_inside_bull_fvg")),
        ("15m_bear_fvg", debug.get("15m_inside_bear_fvg")),
        ("1m_bull_fvg", debug.get("1m_inside_bull_fvg")),
        ("1m_bear_fvg", debug.get("1m_inside_bear_fvg")),
        ("bull_fvg_align", debug.get("htf_ltf_bull_fvg_alignment")),
        ("bear_fvg_align", debug.get("htf_ltf_bear_fvg_alignment")),
        ("fvg_bull_score", debug.get("fvg_bull_context_score")),
        ("fvg_bear_score", debug.get("fvg_bear_context_score")),
        ("bull_ob_fvg", debug.get("ob_fvg_bull_confluence")),
        ("bear_ob_fvg", debug.get("ob_fvg_bear_confluence")),
    ]
    print_table("NEW CLOSED BAR / MODEL SIGNAL", rows)


def append_trade_log(row: dict) -> None:
    header = not LOG_PATH.exists()
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=header, index=False)


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
    print_table("MT5 CONNECTION", [("login", info.login), ("server", info.server), ("balance", info.balance), ("equity", info.equity)])


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
    df = rates.copy().rename(columns={"tick_volume": "volume", "spread": "spread_points"})
    keep = ["time", "open", "high", "low", "close", "volume", "spread_points"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["datetime"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("datetime")


# ========== INDICATORS ==========
def add_rsi(df: pd.DataFrame, prefix: str, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    df[f"{prefix}_rsi{period}"] = 100 - (100 / (1 + rs))
    df[f"{prefix}_rsi_above_50"] = (df[f"{prefix}_rsi{period}"] > 50).astype(int)
    df[f"{prefix}_rsi_overbought"] = (df[f"{prefix}_rsi{period}"] > 70).astype(int)
    df[f"{prefix}_rsi_oversold"] = (df[f"{prefix}_rsi{period}"] < 30).astype(int)
    return df


def add_atr(df: pd.DataFrame, prefix: str, period: int = ATR_PERIOD) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df[f"{prefix}_tr"] = tr
    df[f"{prefix}_atr{period}"] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    df[f"{prefix}_atr_pct{period}"] = df[f"{prefix}_atr{period}"] / (df["close"] + EPS)
    df[f"{prefix}_range_vs_atr{period}"] = df[f"{prefix}_range"] / (df[f"{prefix}_atr{period}"] + EPS)
    return df


def add_adx(df: pd.DataFrame, prefix: str, period: int = ADX_PERIOD) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr + EPS)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    df[f"{prefix}_plus_di{period}"] = plus_di
    df[f"{prefix}_minus_di{period}"] = minus_di
    df[f"{prefix}_adx{period}"] = adx
    df[f"{prefix}_di_direction"] = np.sign(plus_di - minus_di)
    df[f"{prefix}_adx_trending"] = (adx > 25).astype(int)
    df[f"{prefix}_adx_choppy"] = (adx < 20).astype(int)
    return df


def create_features(df: pd.DataFrame, prefix: str, use_rsi: bool = False, use_atr: bool = False, use_adx: bool = False) -> pd.DataFrame:
    df = df.copy()
    df[f"{prefix}_return"] = df["close"].pct_change()
    df[f"{prefix}_log_return"] = np.log(df["close"] / df["close"].shift(1))
    df[f"{prefix}_range"] = df["high"] - df["low"]
    df[f"{prefix}_body"] = df["close"] - df["open"]
    df[f"{prefix}_upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df[f"{prefix}_lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    range_safe = df[f"{prefix}_range"].replace(0, np.nan)
    df[f"{prefix}_body_ratio"] = df[f"{prefix}_body"] / (range_safe + EPS)
    df[f"{prefix}_upper_wick_ratio"] = df[f"{prefix}_upper_wick"] / (range_safe + EPS)
    df[f"{prefix}_lower_wick_ratio"] = df[f"{prefix}_lower_wick"] / (range_safe + EPS)
    df[f"{prefix}_close_pos_in_range"] = (df["close"] - df["low"]) / (range_safe + EPS)
    df[f"{prefix}_open_pos_in_range"] = (df["open"] - df["low"]) / (range_safe + EPS)
    df[f"{prefix}_is_bull"] = (df["close"] > df["open"]).astype(int)
    df[f"{prefix}_is_bear"] = (df["close"] < df["open"]).astype(int)
    df[f"{prefix}_ma10"] = df["close"].rolling(10).mean()
    df[f"{prefix}_ma20"] = df["close"].rolling(20).mean()
    df[f"{prefix}_ma10_dist"] = (df["close"] - df[f"{prefix}_ma10"]) / (df[f"{prefix}_ma10"] + EPS)
    df[f"{prefix}_ma20_dist"] = (df["close"] - df[f"{prefix}_ma20"]) / (df[f"{prefix}_ma20"] + EPS)
    df[f"{prefix}_ma10_slope"] = df[f"{prefix}_ma10"].pct_change()
    df[f"{prefix}_ma20_slope"] = df[f"{prefix}_ma20"].pct_change()
    df[f"{prefix}_close_above_ma10"] = (df["close"] > df[f"{prefix}_ma10"]).astype(int)
    df[f"{prefix}_close_above_ma20"] = (df["close"] > df[f"{prefix}_ma20"]).astype(int)
    df[f"{prefix}_ma10_above_ma20"] = (df[f"{prefix}_ma10"] > df[f"{prefix}_ma20"]).astype(int)
    df[f"{prefix}_volatility10"] = df["close"].rolling(10).std()
    df[f"{prefix}_range_mean5"] = df[f"{prefix}_range"].rolling(5).mean()
    df[f"{prefix}_range_mean10"] = df[f"{prefix}_range"].rolling(10).mean()
    df[f"{prefix}_range_std10"] = df[f"{prefix}_range"].rolling(10).std()
    df[f"{prefix}_range_vs_mean5"] = df[f"{prefix}_range"] / (df[f"{prefix}_range_mean5"] + EPS)
    df[f"{prefix}_range_vs_mean10"] = df[f"{prefix}_range"] / (df[f"{prefix}_range_mean10"] + EPS)
    df[f"{prefix}_vol_compression10"] = df[f"{prefix}_range_std10"] / (df[f"{prefix}_range_mean10"] + EPS)
    df[f"{prefix}_range_change1"] = df[f"{prefix}_range"] / (df[f"{prefix}_range"].shift(1) + EPS)
    df[f"{prefix}_body_change1"] = df[f"{prefix}_body"].abs() / (df[f"{prefix}_body"].shift(1).abs() + EPS)
    df[f"{prefix}_volume_change"] = df["volume"].pct_change()
    df[f"{prefix}_volume_mean10"] = df["volume"].rolling(10).mean()
    df[f"{prefix}_volume_vs_mean10"] = df["volume"] / (df[f"{prefix}_volume_mean10"] + EPS)
    df[f"{prefix}_momentum3"] = df["close"] / df["close"].shift(3) - 1
    df[f"{prefix}_momentum5"] = df["close"] / df["close"].shift(5) - 1
    df[f"{prefix}_momentum10"] = df["close"] / df["close"].shift(10) - 1
    df[f"{prefix}_momentum3_accel"] = df[f"{prefix}_momentum3"] - df[f"{prefix}_momentum3"].shift(1)
    df[f"{prefix}_momentum5_accel"] = df[f"{prefix}_momentum5"] - df[f"{prefix}_momentum5"].shift(1)
    df[f"{prefix}_momentum10_accel"] = df[f"{prefix}_momentum10"] - df[f"{prefix}_momentum10"].shift(1)
    df[f"{prefix}_return_mean3"] = df[f"{prefix}_return"].rolling(3).mean()
    df[f"{prefix}_return_mean5"] = df[f"{prefix}_return"].rolling(5).mean()
    df[f"{prefix}_return_std5"] = df[f"{prefix}_return"].rolling(5).std()
    df[f"{prefix}_return_std10"] = df[f"{prefix}_return"].rolling(10).std()
    df[f"{prefix}_body_vs_range_mean10"] = df[f"{prefix}_body"].abs() / (df[f"{prefix}_range_mean10"] + EPS)
    df[f"{prefix}_wick_imbalance"] = (df[f"{prefix}_lower_wick"] - df[f"{prefix}_upper_wick"]) / (range_safe + EPS)
    if use_rsi:
        df = add_rsi(df, prefix, RSI_PERIOD)
    if use_atr:
        df = add_atr(df, prefix, ATR_PERIOD)
    if use_adx:
        df = add_adx(df, prefix, ADX_PERIOD)
    return df



# ========== FRACTAL BOS / CHoCH MARKET STRUCTURE FEATURES ==========
def add_fractal_structure_features(
    df: pd.DataFrame,
    prefix: str,
    length: int = STRUCTURE_FRACTAL_LENGTH,
    recent_window: int = STRUCTURE_RECENT_WINDOW,
) -> pd.DataFrame:
    """
    Pine Script-inspired fractal BOS/CHoCH features with no future leakage.

    A bullish fractal high / bearish fractal low is confirmed only after
    p=int(length/2) candles pass. Breaks above/below those confirmed fractals
    become BOS or CHoCH depending on the previous structure direction.
    """
    out = df.copy()
    if length < 3:
        raise ValueError("STRUCTURE_FRACTAL_LENGTH must be at least 3.")

    p = int(length / 2)
    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    close_arr = out["close"].to_numpy(dtype=float)
    n = len(out)

    bull_fractal = np.zeros(n, dtype=int)
    bear_fractal = np.zeros(n, dtype=int)
    bos_up = np.zeros(n, dtype=int)
    bos_down = np.zeros(n, dtype=int)
    choch_up = np.zeros(n, dtype=int)
    choch_down = np.zeros(n, dtype=int)
    structure_direction = np.zeros(n, dtype=int)
    last_break_direction = np.zeros(n, dtype=int)

    bars_since_bos_up = np.full(n, STRUCTURE_NO_BREAK_AGE, dtype=float)
    bars_since_bos_down = np.full(n, STRUCTURE_NO_BREAK_AGE, dtype=float)
    bars_since_choch_up = np.full(n, STRUCTURE_NO_BREAK_AGE, dtype=float)
    bars_since_choch_down = np.full(n, STRUCTURE_NO_BREAK_AGE, dtype=float)
    last_fractal_high_value = np.full(n, np.nan, dtype=float)
    last_fractal_low_value = np.full(n, np.nan, dtype=float)
    dist_to_fractal_high = np.full(n, 10.0, dtype=float)
    dist_to_fractal_low = np.full(n, 10.0, dtype=float)

    os = 0
    upper = None
    lower = None
    last_bu = last_bd = last_cu = last_cd = None

    for i in range(n):
        j = i - p
        if j >= p and (j + p) < n:
            win_start = j - p
            win_end = j + p + 1

            left_high_signs = np.sign(np.diff(high_arr[win_start:j + 1]))
            right_high_signs = np.sign(np.diff(high_arr[j:win_end]))
            left_low_signs = np.sign(np.diff(low_arr[win_start:j + 1]))
            right_low_signs = np.sign(np.diff(low_arr[j:win_end]))

            is_bull_fractal = (
                np.nansum(left_high_signs) == p
                and np.nansum(right_high_signs) == -p
                and high_arr[j] == np.nanmax(high_arr[win_start:win_end])
            )
            is_bear_fractal = (
                np.nansum(left_low_signs) == -p
                and np.nansum(right_low_signs) == p
                and low_arr[j] == np.nanmin(low_arr[win_start:win_end])
            )

            if is_bull_fractal:
                bull_fractal[i] = 1
                upper = {"value": float(high_arr[j]), "loc": int(j), "crossed": False}
            if is_bear_fractal:
                bear_fractal[i] = 1
                lower = {"value": float(low_arr[j]), "loc": int(j), "crossed": False}

        if upper is not None:
            last_fractal_high_value[i] = upper["value"]
            dist_to_fractal_high[i] = (upper["value"] - close_arr[i]) / (close_arr[i] + EPS)
        if lower is not None:
            last_fractal_low_value[i] = lower["value"]
            dist_to_fractal_low[i] = (close_arr[i] - lower["value"]) / (close_arr[i] + EPS)

        prev_close = close_arr[i - 1] if i > 0 else np.nan
        if upper is not None and not upper["crossed"] and i > 0:
            if prev_close <= upper["value"] and close_arr[i] > upper["value"]:
                if os == -1:
                    choch_up[i] = 1
                    last_cu = i
                else:
                    bos_up[i] = 1
                    last_bu = i
                upper["crossed"] = True
                os = 1
                last_break_direction[i] = 1

        if lower is not None and not lower["crossed"] and i > 0:
            if prev_close >= lower["value"] and close_arr[i] < lower["value"]:
                if os == 1:
                    choch_down[i] = 1
                    last_cd = i
                else:
                    bos_down[i] = 1
                    last_bd = i
                lower["crossed"] = True
                os = -1
                last_break_direction[i] = -1

        structure_direction[i] = os
        if last_bu is not None:
            bars_since_bos_up[i] = i - last_bu
        if last_bd is not None:
            bars_since_bos_down[i] = i - last_bd
        if last_cu is not None:
            bars_since_choch_up[i] = i - last_cu
        if last_cd is not None:
            bars_since_choch_down[i] = i - last_cd

    out[f"{prefix}_bull_fractal_confirmed"] = bull_fractal
    out[f"{prefix}_bear_fractal_confirmed"] = bear_fractal
    out[f"{prefix}_bos_up"] = bos_up
    out[f"{prefix}_bos_down"] = bos_down
    out[f"{prefix}_choch_up"] = choch_up
    out[f"{prefix}_choch_down"] = choch_down
    out[f"{prefix}_structure_direction"] = structure_direction
    out[f"{prefix}_last_break_direction"] = last_break_direction
    out[f"{prefix}_recent_bos_up"] = (bars_since_bos_up <= recent_window).astype(int)
    out[f"{prefix}_recent_bos_down"] = (bars_since_bos_down <= recent_window).astype(int)
    out[f"{prefix}_recent_choch_up"] = (bars_since_choch_up <= recent_window).astype(int)
    out[f"{prefix}_recent_choch_down"] = (bars_since_choch_down <= recent_window).astype(int)
    out[f"{prefix}_bars_since_bos_up"] = bars_since_bos_up
    out[f"{prefix}_bars_since_bos_down"] = bars_since_bos_down
    out[f"{prefix}_bars_since_choch_up"] = bars_since_choch_up
    out[f"{prefix}_bars_since_choch_down"] = bars_since_choch_down
    out[f"{prefix}_last_fractal_high"] = last_fractal_high_value
    out[f"{prefix}_last_fractal_low"] = last_fractal_low_value
    out[f"{prefix}_dist_to_fractal_high"] = dist_to_fractal_high
    out[f"{prefix}_dist_to_fractal_low"] = dist_to_fractal_low
    return out



def add_order_block_features(df: pd.DataFrame, prefix: str, swing_lookback: int = OB_SWING_LOOKBACK, use_body: bool = OB_USE_BODY, recent_window: int = OB_RECENT_WINDOW, max_active_zones: int = OB_MAX_ACTIVE_ZONES) -> pd.DataFrame:
    out = df.copy()
    open_arr = out["open"].to_numpy(dtype=float)
    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    close_arr = out["close"].to_numpy(dtype=float)
    n = len(out)
    zone_high_arr = np.maximum(open_arr, close_arr) if use_body else high_arr
    zone_low_arr = np.minimum(open_arr, close_arr) if use_body else low_arr
    inside_bull_ob = np.zeros(n, dtype=int); inside_bear_ob = np.zeros(n, dtype=int)
    inside_bull_breaker = np.zeros(n, dtype=int); inside_bear_breaker = np.zeros(n, dtype=int)
    recent_bull_ob = np.zeros(n, dtype=int); recent_bear_ob = np.zeros(n, dtype=int)
    recent_bull_breaker = np.zeros(n, dtype=int); recent_bear_breaker = np.zeros(n, dtype=int)
    dist_to_bull_ob = np.full(n, OB_NO_ZONE_DISTANCE, dtype=float); dist_to_bear_ob = np.full(n, OB_NO_ZONE_DISTANCE, dtype=float)
    dist_to_bull_breaker = np.full(n, OB_NO_ZONE_DISTANCE, dtype=float); dist_to_bear_breaker = np.full(n, OB_NO_ZONE_DISTANCE, dtype=float)
    bull_ob_age = np.full(n, OB_NO_ZONE_AGE, dtype=float); bear_ob_age = np.full(n, OB_NO_ZONE_AGE, dtype=float)
    bull_breaker_age = np.full(n, OB_NO_ZONE_AGE, dtype=float); bear_breaker_age = np.full(n, OB_NO_ZONE_AGE, dtype=float)
    bull_ob_width_pct = np.zeros(n, dtype=float); bear_ob_width_pct = np.zeros(n, dtype=float)
    bull_breaker_width_pct = np.zeros(n, dtype=float); bear_breaker_width_pct = np.zeros(n, dtype=float)
    active_bull, active_bear = [], []
    last_swing_high = None; last_swing_low = None

    def nearest_zone(zones, price: float, want_breaker: bool):
        candidates = [z for z in zones if z["breaker"] == want_breaker]
        if not candidates:
            return None
        def zone_distance(z):
            if z["btm"] <= price <= z["top"]:
                return 0.0
            return min(abs(price - z["top"]), abs(price - z["btm"])) / (price + EPS)
        return min(candidates, key=zone_distance)

    for i in range(n):
        price = close_arr[i]
        body_low = min(open_arr[i], close_arr[i])
        body_high = max(open_arr[i], close_arr[i])
        j = i - swing_lookback
        if j >= swing_lookback:
            left = j - swing_lookback
            right = min(n, j + swing_lookback + 1)
            if high_arr[j] >= np.max(high_arr[left:right]):
                last_swing_high = {"price": high_arr[j], "idx": j, "crossed": False}
            if low_arr[j] <= np.min(low_arr[left:right]):
                last_swing_low = {"price": low_arr[j], "idx": j, "crossed": False}
        if last_swing_high is not None and not last_swing_high["crossed"] and close_arr[i] > last_swing_high["price"]:
            start, end = last_swing_high["idx"] + 1, i
            if end > start:
                ob_idx = start + int(np.argmin(zone_low_arr[start:end]))
                active_bull.insert(0, {"top": float(zone_high_arr[ob_idx]), "btm": float(zone_low_arr[ob_idx]), "loc": int(ob_idx), "breaker": False, "break_loc": None})
                active_bull = active_bull[:max_active_zones]
            last_swing_high["crossed"] = True
        if last_swing_low is not None and not last_swing_low["crossed"] and close_arr[i] < last_swing_low["price"]:
            start, end = last_swing_low["idx"] + 1, i
            if end > start:
                ob_idx = start + int(np.argmax(zone_high_arr[start:end]))
                active_bear.insert(0, {"top": float(zone_high_arr[ob_idx]), "btm": float(zone_low_arr[ob_idx]), "loc": int(ob_idx), "breaker": False, "break_loc": None})
                active_bear = active_bear[:max_active_zones]
            last_swing_low["crossed"] = True
        kept_bull = []
        for z in active_bull:
            if not z["breaker"] and body_low < z["btm"]:
                z["breaker"] = True; z["break_loc"] = i
            if not z["breaker"] or close_arr[i] <= z["top"]:
                kept_bull.append(z)
        active_bull = kept_bull[:max_active_zones]
        kept_bear = []
        for z in active_bear:
            if not z["breaker"] and body_high > z["top"]:
                z["breaker"] = True; z["break_loc"] = i
            if not z["breaker"] or close_arr[i] >= z["btm"]:
                kept_bear.append(z)
        active_bear = kept_bear[:max_active_zones]
        bull = nearest_zone(active_bull, price, False); bear = nearest_zone(active_bear, price, False)
        bull_br = nearest_zone(active_bull, price, True); bear_br = nearest_zone(active_bear, price, True)
        if bull is not None:
            inside_bull_ob[i] = int(bull["btm"] <= price <= bull["top"])
            dist_to_bull_ob[i] = 0.0 if inside_bull_ob[i] else min(abs(price - bull["top"]), abs(price - bull["btm"])) / (price + EPS)
            bull_ob_age[i] = i - bull["loc"]; bull_ob_width_pct[i] = (bull["top"] - bull["btm"]) / (price + EPS)
            recent_bull_ob[i] = int(bull_ob_age[i] <= recent_window)
        if bear is not None:
            inside_bear_ob[i] = int(bear["btm"] <= price <= bear["top"])
            dist_to_bear_ob[i] = 0.0 if inside_bear_ob[i] else min(abs(price - bear["top"]), abs(price - bear["btm"])) / (price + EPS)
            bear_ob_age[i] = i - bear["loc"]; bear_ob_width_pct[i] = (bear["top"] - bear["btm"]) / (price + EPS)
            recent_bear_ob[i] = int(bear_ob_age[i] <= recent_window)
        if bull_br is not None:
            inside_bull_breaker[i] = int(bull_br["btm"] <= price <= bull_br["top"])
            dist_to_bull_breaker[i] = 0.0 if inside_bull_breaker[i] else min(abs(price - bull_br["top"]), abs(price - bull_br["btm"])) / (price + EPS)
            base_loc = bull_br["break_loc"] if bull_br["break_loc"] is not None else bull_br["loc"]
            bull_breaker_age[i] = i - base_loc; bull_breaker_width_pct[i] = (bull_br["top"] - bull_br["btm"]) / (price + EPS)
            recent_bull_breaker[i] = int(bull_breaker_age[i] <= recent_window)
        if bear_br is not None:
            inside_bear_breaker[i] = int(bear_br["btm"] <= price <= bear_br["top"])
            dist_to_bear_breaker[i] = 0.0 if inside_bear_breaker[i] else min(abs(price - bear_br["top"]), abs(price - bear_br["btm"])) / (price + EPS)
            base_loc = bear_br["break_loc"] if bear_br["break_loc"] is not None else bear_br["loc"]
            bear_breaker_age[i] = i - base_loc; bear_breaker_width_pct[i] = (bear_br["top"] - bear_br["btm"]) / (price + EPS)
            recent_bear_breaker[i] = int(bear_breaker_age[i] <= recent_window)
    out[f"{prefix}_inside_bull_ob"] = inside_bull_ob; out[f"{prefix}_inside_bear_ob"] = inside_bear_ob
    out[f"{prefix}_inside_bull_breaker"] = inside_bull_breaker; out[f"{prefix}_inside_bear_breaker"] = inside_bear_breaker
    out[f"{prefix}_recent_bull_ob"] = recent_bull_ob; out[f"{prefix}_recent_bear_ob"] = recent_bear_ob
    out[f"{prefix}_recent_bull_breaker"] = recent_bull_breaker; out[f"{prefix}_recent_bear_breaker"] = recent_bear_breaker
    out[f"{prefix}_dist_to_bull_ob"] = dist_to_bull_ob; out[f"{prefix}_dist_to_bear_ob"] = dist_to_bear_ob
    out[f"{prefix}_dist_to_bull_breaker"] = dist_to_bull_breaker; out[f"{prefix}_dist_to_bear_breaker"] = dist_to_bear_breaker
    out[f"{prefix}_bull_ob_age"] = bull_ob_age; out[f"{prefix}_bear_ob_age"] = bear_ob_age
    out[f"{prefix}_bull_breaker_age"] = bull_breaker_age; out[f"{prefix}_bear_breaker_age"] = bear_breaker_age
    out[f"{prefix}_bull_ob_width_pct"] = bull_ob_width_pct; out[f"{prefix}_bear_ob_width_pct"] = bear_ob_width_pct
    out[f"{prefix}_bull_breaker_width_pct"] = bull_breaker_width_pct; out[f"{prefix}_bear_breaker_width_pct"] = bear_breaker_width_pct
    return out


def add_fvg_features(df: pd.DataFrame, prefix: str, threshold_pct: float = FVG_THRESHOLD_PCT, auto_threshold: bool = FVG_AUTO_THRESHOLD, recent_window: int = FVG_RECENT_WINDOW, max_active_zones: int = FVG_MAX_ACTIVE_ZONES) -> pd.DataFrame:
    out = df.copy()
    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    close_arr = out["close"].to_numpy(dtype=float)
    n = len(out)
    active_bull, active_bear = [], []
    bull_fvg_detected = np.zeros(n, dtype=int); bear_fvg_detected = np.zeros(n, dtype=int)
    inside_bull_fvg = np.zeros(n, dtype=int); inside_bear_fvg = np.zeros(n, dtype=int)
    recent_bull_fvg = np.zeros(n, dtype=int); recent_bear_fvg = np.zeros(n, dtype=int)
    dist_to_bull_fvg = np.full(n, FVG_NO_ZONE_DISTANCE, dtype=float); dist_to_bear_fvg = np.full(n, FVG_NO_ZONE_DISTANCE, dtype=float)
    bull_fvg_age = np.full(n, FVG_NO_ZONE_AGE, dtype=float); bear_fvg_age = np.full(n, FVG_NO_ZONE_AGE, dtype=float)
    bull_fvg_width_pct = np.zeros(n, dtype=float); bear_fvg_width_pct = np.zeros(n, dtype=float)
    bull_fvg_count_active = np.zeros(n, dtype=int); bear_fvg_count_active = np.zeros(n, dtype=int)
    bull_fvg_mitigated = np.zeros(n, dtype=int); bear_fvg_mitigated = np.zeros(n, dtype=int)
    cumulative_range_ratio = 0.0

    def zone_distance(z, price: float) -> float:
        if z["btm"] <= price <= z["top"]:
            return 0.0
        return min(abs(price - z["top"]), abs(price - z["btm"])) / (price + EPS)

    def nearest_zone(zones, price: float):
        if not zones:
            return None
        return min(zones, key=lambda z: zone_distance(z, price))

    for i in range(n):
        price = close_arr[i]
        if i > 0:
            cumulative_range_ratio += (high_arr[i] - low_arr[i]) / (low_arr[i] + EPS)
        threshold = cumulative_range_ratio / max(i, 1) if auto_threshold else threshold_pct
        if i >= 2:
            bull_gap_pct = (low_arr[i] - high_arr[i - 2]) / (high_arr[i - 2] + EPS)
            bear_gap_pct = (low_arr[i - 2] - high_arr[i]) / (high_arr[i] + EPS)
            is_bull_fvg = low_arr[i] > high_arr[i - 2] and close_arr[i - 1] > high_arr[i - 2] and bull_gap_pct > threshold
            is_bear_fvg = high_arr[i] < low_arr[i - 2] and close_arr[i - 1] < low_arr[i - 2] and bear_gap_pct > threshold
            if is_bull_fvg:
                active_bull.insert(0, {"top": float(low_arr[i]), "btm": float(high_arr[i - 2]), "loc": int(i)})
                active_bull = active_bull[:max_active_zones]
                bull_fvg_detected[i] = 1
            if is_bear_fvg:
                active_bear.insert(0, {"top": float(low_arr[i - 2]), "btm": float(high_arr[i]), "loc": int(i)})
                active_bear = active_bear[:max_active_zones]
                bear_fvg_detected[i] = 1
        kept_bull, bull_removed_now = [], 0
        for z in active_bull:
            if close_arr[i] < z["btm"]:
                bull_removed_now += 1
            else:
                kept_bull.append(z)
        active_bull = kept_bull[:max_active_zones]
        kept_bear, bear_removed_now = [], 0
        for z in active_bear:
            if close_arr[i] > z["top"]:
                bear_removed_now += 1
            else:
                kept_bear.append(z)
        active_bear = kept_bear[:max_active_zones]
        bull_fvg_mitigated[i] = bull_removed_now; bear_fvg_mitigated[i] = bear_removed_now
        bull = nearest_zone(active_bull, price); bear = nearest_zone(active_bear, price)
        if bull is not None:
            inside_bull_fvg[i] = int(bull["btm"] <= price <= bull["top"])
            dist_to_bull_fvg[i] = zone_distance(bull, price)
            bull_fvg_age[i] = i - bull["loc"]
            bull_fvg_width_pct[i] = (bull["top"] - bull["btm"]) / (price + EPS)
            recent_bull_fvg[i] = int(bull_fvg_age[i] <= recent_window)
        if bear is not None:
            inside_bear_fvg[i] = int(bear["btm"] <= price <= bear["top"])
            dist_to_bear_fvg[i] = zone_distance(bear, price)
            bear_fvg_age[i] = i - bear["loc"]
            bear_fvg_width_pct[i] = (bear["top"] - bear["btm"]) / (price + EPS)
            recent_bear_fvg[i] = int(bear_fvg_age[i] <= recent_window)
        bull_fvg_count_active[i] = len(active_bull); bear_fvg_count_active[i] = len(active_bear)
    out[f"{prefix}_bull_fvg_detected"] = bull_fvg_detected; out[f"{prefix}_bear_fvg_detected"] = bear_fvg_detected
    out[f"{prefix}_inside_bull_fvg"] = inside_bull_fvg; out[f"{prefix}_inside_bear_fvg"] = inside_bear_fvg
    out[f"{prefix}_recent_bull_fvg"] = recent_bull_fvg; out[f"{prefix}_recent_bear_fvg"] = recent_bear_fvg
    out[f"{prefix}_dist_to_bull_fvg"] = dist_to_bull_fvg; out[f"{prefix}_dist_to_bear_fvg"] = dist_to_bear_fvg
    out[f"{prefix}_bull_fvg_age"] = bull_fvg_age; out[f"{prefix}_bear_fvg_age"] = bear_fvg_age
    out[f"{prefix}_bull_fvg_width_pct"] = bull_fvg_width_pct; out[f"{prefix}_bear_fvg_width_pct"] = bear_fvg_width_pct
    out[f"{prefix}_bull_fvg_count_active"] = bull_fvg_count_active; out[f"{prefix}_bear_fvg_count_active"] = bear_fvg_count_active
    out[f"{prefix}_bull_fvg_mitigated"] = bull_fvg_mitigated; out[f"{prefix}_bear_fvg_mitigated"] = bear_fvg_mitigated
    return out



def add_entry_zone_price_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Adds nearest active 1m OB/FVG zone boundaries, matching the dataset builder.
    These are live features used by the entry-style model and pending-order logic.
    """
    out = df.copy()
    n = len(out)
    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    close_arr = out["close"].to_numpy(dtype=float)
    open_arr = out["open"].to_numpy(dtype=float)

    cols = [
        f"{prefix}_nearest_bull_fvg_top", f"{prefix}_nearest_bull_fvg_btm",
        f"{prefix}_nearest_bear_fvg_top", f"{prefix}_nearest_bear_fvg_btm",
        f"{prefix}_nearest_bull_ob_top", f"{prefix}_nearest_bull_ob_btm",
        f"{prefix}_nearest_bear_ob_top", f"{prefix}_nearest_bear_ob_btm",
    ]
    for c in cols:
        out[c] = np.nan

    def zone_distance(z, price):
        if z["btm"] <= price <= z["top"]:
            return 0.0
        return min(abs(price - z["top"]), abs(price - z["btm"])) / (price + EPS)

    def nearest(zones, price):
        return min(zones, key=lambda z: zone_distance(z, price)) if zones else None

    active_bull_fvg, active_bear_fvg = [], []
    for i in range(n):
        if i >= 2:
            bull_gap_pct = (low_arr[i] - high_arr[i - 2]) / (high_arr[i - 2] + EPS)
            bear_gap_pct = (low_arr[i - 2] - high_arr[i]) / (high_arr[i] + EPS)
            if low_arr[i] > high_arr[i - 2] and close_arr[i - 1] > high_arr[i - 2] and bull_gap_pct > FVG_THRESHOLD_PCT:
                active_bull_fvg.insert(0, {"top": float(low_arr[i]), "btm": float(high_arr[i - 2]), "loc": i})
                active_bull_fvg = active_bull_fvg[:FVG_MAX_ACTIVE_ZONES]
            if high_arr[i] < low_arr[i - 2] and close_arr[i - 1] < low_arr[i - 2] and bear_gap_pct > FVG_THRESHOLD_PCT:
                active_bear_fvg.insert(0, {"top": float(low_arr[i - 2]), "btm": float(high_arr[i]), "loc": i})
                active_bear_fvg = active_bear_fvg[:FVG_MAX_ACTIVE_ZONES]

        active_bull_fvg = [z for z in active_bull_fvg if close_arr[i] >= z["btm"]][:FVG_MAX_ACTIVE_ZONES]
        active_bear_fvg = [z for z in active_bear_fvg if close_arr[i] <= z["top"]][:FVG_MAX_ACTIVE_ZONES]

        bull = nearest(active_bull_fvg, close_arr[i])
        bear = nearest(active_bear_fvg, close_arr[i])
        if bull is not None:
            out.iat[i, out.columns.get_loc(f"{prefix}_nearest_bull_fvg_top")] = bull["top"]
            out.iat[i, out.columns.get_loc(f"{prefix}_nearest_bull_fvg_btm")] = bull["btm"]
        if bear is not None:
            out.iat[i, out.columns.get_loc(f"{prefix}_nearest_bear_fvg_top")] = bear["top"]
            out.iat[i, out.columns.get_loc(f"{prefix}_nearest_bear_fvg_btm")] = bear["btm"]

    zone_high_arr = np.maximum(open_arr, close_arr) if OB_USE_BODY else high_arr
    zone_low_arr = np.minimum(open_arr, close_arr) if OB_USE_BODY else low_arr
    active_bull_ob, active_bear_ob = [], []
    last_swing_high = last_swing_low = None

    for i in range(n):
        j = i - OB_SWING_LOOKBACK
        if j >= OB_SWING_LOOKBACK:
            left = j - OB_SWING_LOOKBACK
            right = min(n, j + OB_SWING_LOOKBACK + 1)
            if high_arr[j] >= np.max(high_arr[left:right]):
                last_swing_high = {"price": high_arr[j], "idx": j, "crossed": False}
            if low_arr[j] <= np.min(low_arr[left:right]):
                last_swing_low = {"price": low_arr[j], "idx": j, "crossed": False}

        if last_swing_high is not None and not last_swing_high["crossed"] and close_arr[i] > last_swing_high["price"]:
            start, end = last_swing_high["idx"] + 1, i
            if end > start:
                ob_idx = start + int(np.argmin(zone_low_arr[start:end]))
                active_bull_ob.insert(0, {"top": float(zone_high_arr[ob_idx]), "btm": float(zone_low_arr[ob_idx]), "loc": ob_idx})
                active_bull_ob = active_bull_ob[:OB_MAX_ACTIVE_ZONES]
            last_swing_high["crossed"] = True

        if last_swing_low is not None and not last_swing_low["crossed"] and close_arr[i] < last_swing_low["price"]:
            start, end = last_swing_low["idx"] + 1, i
            if end > start:
                ob_idx = start + int(np.argmax(zone_high_arr[start:end]))
                active_bear_ob.insert(0, {"top": float(zone_high_arr[ob_idx]), "btm": float(zone_low_arr[ob_idx]), "loc": ob_idx})
                active_bear_ob = active_bear_ob[:OB_MAX_ACTIVE_ZONES]
            last_swing_low["crossed"] = True

        body_low = min(open_arr[i], close_arr[i])
        body_high = max(open_arr[i], close_arr[i])
        active_bull_ob = [z for z in active_bull_ob if body_low >= z["btm"]][:OB_MAX_ACTIVE_ZONES]
        active_bear_ob = [z for z in active_bear_ob if body_high <= z["top"]][:OB_MAX_ACTIVE_ZONES]

        bull = nearest(active_bull_ob, close_arr[i])
        bear = nearest(active_bear_ob, close_arr[i])
        if bull is not None:
            out.iat[i, out.columns.get_loc(f"{prefix}_nearest_bull_ob_top")] = bull["top"]
            out.iat[i, out.columns.get_loc(f"{prefix}_nearest_bull_ob_btm")] = bull["btm"]
        if bear is not None:
            out.iat[i, out.columns.get_loc(f"{prefix}_nearest_bear_ob_top")] = bear["top"]
            out.iat[i, out.columns.get_loc(f"{prefix}_nearest_bear_ob_btm")] = bear["btm"]

    return out


def fetch_raw_timeframes(symbol: str, m1_count: int = 1200) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rates_1m = get_latest_rates(symbol, mt5.TIMEFRAME_M1, count=m1_count)
    rates_5m = get_latest_rates(symbol, mt5.TIMEFRAME_M5, count=max(500, math.ceil(m1_count / 5) + 240))
    rates_15m = get_latest_rates(symbol, mt5.TIMEFRAME_M15, count=max(500, math.ceil(m1_count / 15) + 240))
    return mt5_rates_to_price_df(rates_1m).sort_index(), mt5_rates_to_price_df(rates_5m).sort_index(), mt5_rates_to_price_df(rates_15m).sort_index()


def prefixed_features_only(df_source: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df_source[[c for c in df_source.columns if c.startswith(f"{prefix}_")]].copy()


def make_feature_frame(symbol: str) -> pd.DataFrame:
    df_1m_raw, df_5m_raw, df_15m_raw = fetch_raw_timeframes(symbol, m1_count=1200)
    df_1m = add_fvg_features(add_order_block_features(add_fractal_structure_features(create_features(df_1m_raw, "1m", use_rsi=True, use_atr=True, use_adx=True), "1m"), "1m"), "1m")
    df_1m = add_entry_zone_price_features(df_1m, "1m")
    df_5m = add_fvg_features(add_order_block_features(add_fractal_structure_features(create_features(df_5m_raw, "5m", use_rsi=True, use_atr=False, use_adx=True), "5m"), "5m"), "5m")
    df_15m = add_fvg_features(add_order_block_features(add_fractal_structure_features(create_features(df_15m_raw, "15m", use_rsi=False, use_atr=False, use_adx=True), "15m"), "15m"), "15m")
    df = prefixed_features_only(df_1m, "1m")
    df = pd.merge_asof(df.sort_index(), prefixed_features_only(df_5m, "5m").sort_index(), left_index=True, right_index=True, direction="backward")
    df = pd.merge_asof(df.sort_index(), prefixed_features_only(df_15m, "15m").sort_index(), left_index=True, right_index=True, direction="backward")
    cross = pd.DataFrame(index=df.index)
    cross["align_1m_5m_ma10"] = (df["1m_ma10_above_ma20"] == df["5m_ma10_above_ma20"]).astype(int)
    cross["align_1m_15m_ma10"] = (df["1m_ma10_above_ma20"] == df["15m_ma10_above_ma20"]).astype(int)
    cross["align_5m_15m_ma10"] = (df["5m_ma10_above_ma20"] == df["15m_ma10_above_ma20"]).astype(int)
    cross["triple_trend_alignment"] = ((df["1m_ma10_above_ma20"] == df["5m_ma10_above_ma20"]) & (df["5m_ma10_above_ma20"] == df["15m_ma10_above_ma20"])).astype(int)
    cross["triple_bull_alignment"] = ((df["1m_close_above_ma20"] == 1) & (df["5m_close_above_ma20"] == 1) & (df["15m_close_above_ma20"] == 1)).astype(int)
    cross["triple_bear_alignment"] = ((df["1m_close_above_ma20"] == 0) & (df["5m_close_above_ma20"] == 0) & (df["15m_close_above_ma20"] == 0)).astype(int)
    cross["momentum_alignment_score"] = np.sign(df["1m_momentum5"]).fillna(0) + np.sign(df["5m_momentum5"]).fillna(0) + np.sign(df["15m_momentum5"]).fillna(0)
    cross["1m_vs_5m_range_ratio"] = df["1m_range_mean10"] / (df["5m_range_mean10"] + EPS)
    cross["1m_vs_15m_range_ratio"] = df["1m_range_mean10"] / (df["15m_range_mean10"] + EPS)
    cross["1m_body_vs_5m_range"] = df["1m_body"].abs() / (df["5m_range_mean10"] + EPS)
    cross["1m_body_vs_15m_range"] = df["1m_body"].abs() / (df["15m_range_mean10"] + EPS)
    cross["rsi_1m_5m_alignment"] = df["1m_rsi_above_50"] + df["5m_rsi_above_50"]
    cross["rsi_1m_5m_bullish"] = (cross["rsi_1m_5m_alignment"] == 2).astype(int)
    cross["rsi_1m_5m_bearish"] = (cross["rsi_1m_5m_alignment"] == 0).astype(int)
    cross["adx_trend_alignment_score"] = df["1m_adx_trending"] + df["5m_adx_trending"] + df["15m_adx_trending"]
    cross["adx_choppy_alignment_score"] = df["1m_adx_choppy"] + df["5m_adx_choppy"] + df["15m_adx_choppy"]
    cross["market_is_trending"] = (df["15m_adx14"] > 25).astype(int)
    cross["market_is_choppy"] = (df["15m_adx14"] < 20).astype(int)
    cross["entry_trend_confirmed"] = ((df["1m_adx14"] > 20) & (df["5m_adx14"] > 20) & (df["15m_adx14"] > 20)).astype(int)
    cross["di_direction_alignment_score"] = df["1m_di_direction"].fillna(0) + df["5m_di_direction"].fillna(0) + df["15m_di_direction"].fillna(0)
    cross["rsi_adx_bull_context"] = ((cross["rsi_1m_5m_bullish"] == 1) & (cross["di_direction_alignment_score"] > 0) & (cross["market_is_trending"] == 1)).astype(int)
    cross["rsi_adx_bear_context"] = ((cross["rsi_1m_5m_bearish"] == 1) & (cross["di_direction_alignment_score"] < 0) & (cross["market_is_trending"] == 1)).astype(int)
    cross["structure_bull_context_score"] = (df["15m_structure_direction"].clip(lower=0) + df["5m_structure_direction"].clip(lower=0) + df["1m_structure_direction"].clip(lower=0) + df["15m_recent_bos_up"] + df["5m_recent_bos_up"] + df["1m_recent_bos_up"] + df["15m_recent_choch_up"] + df["5m_recent_choch_up"])
    cross["structure_bear_context_score"] = ((-df["15m_structure_direction"].clip(upper=0)) + (-df["5m_structure_direction"].clip(upper=0)) + (-df["1m_structure_direction"].clip(upper=0)) + df["15m_recent_bos_down"] + df["5m_recent_bos_down"] + df["1m_recent_bos_down"] + df["15m_recent_choch_down"] + df["5m_recent_choch_down"])
    cross["structure_reversal_warning"] = (((df["15m_structure_direction"] == 1) & ((df["1m_recent_choch_down"] == 1) | (df["5m_recent_choch_down"] == 1))) | ((df["15m_structure_direction"] == -1) & ((df["1m_recent_choch_up"] == 1) | (df["5m_recent_choch_up"] == 1)))).astype(int)
    cross["structure_direction_alignment"] = df["1m_structure_direction"].fillna(0) + df["5m_structure_direction"].fillna(0) + df["15m_structure_direction"].fillna(0)
    cross["structure_triple_bull_alignment"] = (cross["structure_direction_alignment"] == 3).astype(int)
    cross["structure_triple_bear_alignment"] = (cross["structure_direction_alignment"] == -3).astype(int)
    cross["structure_recent_break_score"] = (df["1m_recent_bos_up"] - df["1m_recent_bos_down"] + df["5m_recent_bos_up"] - df["5m_recent_bos_down"] + df["15m_recent_bos_up"] - df["15m_recent_bos_down"] + df["1m_recent_choch_up"] - df["1m_recent_choch_down"] + df["5m_recent_choch_up"] - df["5m_recent_choch_down"] + df["15m_recent_choch_up"] - df["15m_recent_choch_down"])
    cross["htf_ltf_bull_ob_alignment"] = ((df["15m_inside_bull_ob"] == 1) & (df["1m_inside_bull_ob"] == 1)).astype(int)
    cross["htf_ltf_bear_ob_alignment"] = ((df["15m_inside_bear_ob"] == 1) & (df["1m_inside_bear_ob"] == 1)).astype(int)
    cross["htf_ltf_bull_breaker_alignment"] = ((df["15m_inside_bear_breaker"] == 1) & (df["1m_inside_bull_ob"] == 1)).astype(int)
    cross["htf_ltf_bear_breaker_alignment"] = ((df["15m_inside_bull_breaker"] == 1) & (df["1m_inside_bear_ob"] == 1)).astype(int)
    cross["ob_bull_context_score"] = df["15m_inside_bull_ob"] + df["5m_inside_bull_ob"] + df["1m_inside_bull_ob"] + df["15m_inside_bear_breaker"] + df["5m_inside_bear_breaker"]
    cross["ob_bear_context_score"] = df["15m_inside_bear_ob"] + df["5m_inside_bear_ob"] + df["1m_inside_bear_ob"] + df["15m_inside_bull_breaker"] + df["5m_inside_bull_breaker"]
    cross["near_15m_bull_ob"] = (df["15m_dist_to_bull_ob"] <= 0.0015).astype(int)
    cross["near_15m_bear_ob"] = (df["15m_dist_to_bear_ob"] <= 0.0015).astype(int)
    cross["near_1m_bull_ob"] = (df["1m_dist_to_bull_ob"] <= 0.0007).astype(int)
    cross["near_1m_bear_ob"] = (df["1m_dist_to_bear_ob"] <= 0.0007).astype(int)
    cross["htf_ltf_bull_fvg_alignment"] = ((df["15m_inside_bull_fvg"] == 1) & (df["1m_inside_bull_fvg"] == 1)).astype(int)
    cross["htf_ltf_bear_fvg_alignment"] = ((df["15m_inside_bear_fvg"] == 1) & (df["1m_inside_bear_fvg"] == 1)).astype(int)
    cross["fvg_bull_context_score"] = df["15m_inside_bull_fvg"] + df["5m_inside_bull_fvg"] + df["1m_inside_bull_fvg"] + df["15m_recent_bull_fvg"] + df["5m_recent_bull_fvg"]
    cross["fvg_bear_context_score"] = df["15m_inside_bear_fvg"] + df["5m_inside_bear_fvg"] + df["1m_inside_bear_fvg"] + df["15m_recent_bear_fvg"] + df["5m_recent_bear_fvg"]
    cross["near_15m_bull_fvg"] = (df["15m_dist_to_bull_fvg"] <= 0.0015).astype(int)
    cross["near_15m_bear_fvg"] = (df["15m_dist_to_bear_fvg"] <= 0.0015).astype(int)
    cross["near_1m_bull_fvg"] = (df["1m_dist_to_bull_fvg"] <= 0.0007).astype(int)
    cross["near_1m_bear_fvg"] = (df["1m_dist_to_bear_fvg"] <= 0.0007).astype(int)
    cross["ob_fvg_bull_confluence"] = ((cross["ob_bull_context_score"] > 0) & (cross["fvg_bull_context_score"] > 0)).astype(int)
    cross["ob_fvg_bear_confluence"] = ((cross["ob_bear_context_score"] > 0) & (cross["fvg_bear_context_score"] > 0)).astype(int)
    cross["htf_adx_strength_mean"] = (df["1m_adx14"] + df["5m_adx14"] + df["15m_adx14"]) / 3
    cross["htf_adx_strength_max"] = pd.concat([df["1m_adx14"], df["5m_adx14"], df["15m_adx14"]], axis=1).max(axis=1)
    cross["htf_adx_strength_min"] = pd.concat([df["1m_adx14"], df["5m_adx14"], df["15m_adx14"]], axis=1).min(axis=1)
    cross["1m_directional_adx"] = df["1m_di_direction"].fillna(0) * df["1m_adx14"]
    cross["5m_directional_adx"] = df["5m_di_direction"].fillna(0) * df["5m_adx14"]
    cross["15m_directional_adx"] = df["15m_di_direction"].fillna(0) * df["15m_adx14"]
    cross["htf_directional_adx_score"] = (cross["1m_directional_adx"] + cross["5m_directional_adx"] + cross["15m_directional_adx"]) / 3
    cross["ma_bull_alignment_score"] = df["1m_ma10_above_ma20"] + df["5m_ma10_above_ma20"] + df["15m_ma10_above_ma20"]
    cross["ma_directional_bias_score"] = (df["1m_ma10_above_ma20"] * 2 - 1) + (df["5m_ma10_above_ma20"] * 2 - 1) + (df["15m_ma10_above_ma20"] * 2 - 1)
    cross["ma20_slope_strength_mean"] = (df["1m_ma20_slope"].abs() + df["5m_ma20_slope"].abs() + df["15m_ma20_slope"].abs()) / 3
    cross["ma20_directional_slope_score"] = np.sign(df["1m_ma20_slope"]).fillna(0) + np.sign(df["5m_ma20_slope"]).fillna(0) + np.sign(df["15m_ma20_slope"]).fillna(0)
    cross["bull_bias_strength"] = (cross["ma_directional_bias_score"].clip(lower=0) / 3) * (cross["htf_adx_strength_mean"] / 50) * (cross["ma20_slope_strength_mean"] * 1000 + 1)
    cross["bear_bias_strength"] = ((-cross["ma_directional_bias_score"].clip(upper=0)) / 3) * (cross["htf_adx_strength_mean"] / 50) * (cross["ma20_slope_strength_mean"] * 1000 + 1)
    cross["mixed_or_weak_trend"] = ((cross["adx_trend_alignment_score"] <= 1) | (cross["ma_bull_alignment_score"].between(1, 2))).astype(int)
    meta = pd.DataFrame({"entry_price": df_1m_raw["close"], "spread_points": df_1m_raw["spread_points"]}, index=df_1m_raw.index)
    out = pd.concat([df, cross, meta], axis=1).replace([np.inf, -np.inf], np.nan)
    nearest_cols = [c for c in out.columns if c.startswith("1m_nearest_")]
    if nearest_cols:
        out[nearest_cols] = out[nearest_cols].fillna(0.0)
    return out


def build_live_feature_row(feature_list: list[str], symbol: str) -> tuple[pd.DataFrame, int, dict]:
    feat_df = make_feature_frame(symbol)
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
        data[feature] = float(value) if isinstance(value, (np.floating, np.integer, float, int)) else value
    live_row = pd.DataFrame([data], columns=feature_list).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    debug_cols = [
        "entry_price", "spread_points",
        "1m_nearest_bull_fvg_top", "1m_nearest_bull_fvg_btm", "1m_nearest_bear_fvg_top", "1m_nearest_bear_fvg_btm",
        "1m_nearest_bull_ob_top", "1m_nearest_bull_ob_btm", "1m_nearest_bear_ob_top", "1m_nearest_bear_ob_btm",
        "1m_rsi14", "5m_rsi14", "1m_atr14", "1m_adx14", "5m_adx14", "15m_adx14",
        "market_is_trending", "market_is_choppy", "adx_trend_alignment_score", "di_direction_alignment_score", "rsi_adx_bull_context", "rsi_adx_bear_context",
        "1m_bos_up", "1m_bos_down", "1m_choch_up", "1m_choch_down", "1m_structure_direction", "1m_recent_bos_up", "1m_recent_bos_down", "1m_recent_choch_up", "1m_recent_choch_down",
        "5m_bos_up", "5m_bos_down", "5m_choch_up", "5m_choch_down", "5m_structure_direction", "5m_recent_bos_up", "5m_recent_bos_down", "5m_recent_choch_up", "5m_recent_choch_down",
        "15m_bos_up", "15m_bos_down", "15m_choch_up", "15m_choch_down", "15m_structure_direction", "15m_recent_bos_up", "15m_recent_bos_down", "15m_recent_choch_up", "15m_recent_choch_down",
        "structure_bull_context_score", "structure_bear_context_score", "structure_reversal_warning", "structure_direction_alignment", "structure_triple_bull_alignment", "structure_triple_bear_alignment", "structure_recent_break_score",
        "15m_inside_bull_ob", "15m_inside_bear_ob", "1m_inside_bull_ob", "1m_inside_bear_ob", "htf_ltf_bull_ob_alignment", "htf_ltf_bear_ob_alignment", "ob_bull_context_score", "ob_bear_context_score",
        "15m_inside_bull_fvg", "15m_inside_bear_fvg", "1m_inside_bull_fvg", "1m_inside_bear_fvg", "htf_ltf_bull_fvg_alignment", "htf_ltf_bear_fvg_alignment", "fvg_bull_context_score", "fvg_bear_context_score", "ob_fvg_bull_confluence", "ob_fvg_bear_confluence",
    ]
    debug = {c: (None if c not in row.index or pd.isna(row[c]) else float(row[c])) for c in debug_cols}
    return live_row, bar_time, debug


def predict_signal(model, feature_list: list[str], symbol: str) -> Signal:
    feature_row, bar_time, debug = build_live_feature_row(feature_list, symbol)
    y_pred = int(model.predict(feature_row)[0])
    y_prob = model.predict_proba(feature_row)[0]
    class_order = list(model.classes_)
    prob_map = {str(cls): float(prob) for cls, prob in zip(class_order, y_prob)}
    confidence = float(np.max(y_prob))
    signal = Signal(bar_time=bar_time, predicted_class=y_pred, confidence=confidence, features=feature_row, raw_probabilities=prob_map)
    signal.debug = debug  # type: ignore[attr-defined]
    return signal


def get_symbol_tick(symbol: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}: {mt5.last_error()}")
    return tick


def current_spread_points(symbol_info, tick) -> float:
    return 0.0 if symbol_info.point <= 0 else (tick.ask - tick.bid) / symbol_info.point


def get_open_positions(symbol: str, magic: Optional[int] = None):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return list(positions) if magic is None else [p for p in positions if p.magic == magic]


def get_pending_orders(symbol: str, magic: Optional[int] = None):
    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        return []
    return list(orders) if magic is None else [o for o in orders if o.magic == magic]


def pending_order_side(order) -> int:
    if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_STOP_LIMIT):
        return 1
    if order.type in (mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP, mt5.ORDER_TYPE_SELL_STOP_LIMIT):
        return -1
    return 0


def cancel_pending_order(order, reason: str = "signal_flip") -> Optional[dict]:
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": order.ticket,
        "symbol": order.symbol,
        "magic": MAGIC_NUMBER,
        "comment": f"{ORDER_COMMENT}-{reason}"[:31],
    }

    if DRY_RUN:
        return {"retcode": "DRY_RUN_CANCEL", "request": request}

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"cancel order_send failed: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log_message(f"Cancel rejected: retcode={result.retcode} comment={result.comment}")
        return None
    return result._asdict()


def cancel_opposite_pending_orders(symbol: str, signal: Signal) -> int:
    signal_dir = signal_direction(signal.predicted_class)
    if signal_dir == 0:
        return 0

    cancelled = 0
    for order in get_pending_orders(symbol, MAGIC_NUMBER):
        order_dir = pending_order_side(order)
        if order_dir == 0:
            continue
        if order_dir != signal_dir:
            log_message(
                f"Signal flip: cancelling opposite pending order {order.ticket} "
                f"dir={order_dir} due to signal_dir={signal_dir}"
            )
            result = cancel_pending_order(order, reason="signal_flip")
            if result is not None:
                cancelled += 1
                append_trade_log({
                    "timestamp": datetime.now().isoformat(),
                    "event": "signal_flip_cancel_pending",
                    "order": order.ticket,
                    "symbol": order.symbol,
                    "predicted_class": signal.predicted_class,
                    "confidence": signal.confidence,
                    "result": json.dumps(result, default=str),
                })
    return cancelled


def close_opposite_positions(symbol: str, signal: Signal, state: Optional[dict] = None) -> int:
    """
    Closes all open positions that are opposite to the current trade signal.
    Returns the number of positions successfully closed.
    """
    signal_dir = signal_direction(signal.predicted_class)
    if signal_dir == 0:
        return 0

    closed = 0
    for pos in get_open_positions(symbol, MAGIC_NUMBER):
        pos_dir = position_side(pos)
        if pos_dir == 0:
            continue

        if pos_dir != signal_dir:
            log_message(
                f"Signal flip: closing position {pos.ticket} "
                f"dir={pos_dir} profit={pos.profit:.2f} due to signal_dir={signal_dir}"
            )
            result = close_position(pos, reason="signal_flip")
            if result is not None:
                closed += 1
                if state is not None:
                    state.get("positions", {}).pop(str(pos.ticket), None)
                append_trade_log({
                    "timestamp": datetime.now().isoformat(),
                    "event": "signal_flip_close",
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "position_side": pos_dir,
                    "signal_side": signal_dir,
                    "position_profit": pos.profit,
                    "predicted_class": signal.predicted_class,
                    "confidence": signal.confidence,
                    "result": json.dumps(result, default=str),
                })
    return closed


def order_type_from_signal(signal_class: int) -> int:
    direction = signal_direction(signal_class)
    if INVERT_SIGNALS:
        direction = -direction
    if direction == 1:
        return mt5.ORDER_TYPE_BUY
    if direction == -1:
        return mt5.ORDER_TYPE_SELL
    raise ValueError(f"Unsupported signal class: {signal_class}")


def pending_order_type_from_signal(signal_class: int) -> int:
    direction = signal_direction(signal_class)
    if INVERT_SIGNALS:
        direction = -direction
    if direction == 1:
        return mt5.ORDER_TYPE_BUY_LIMIT
    if direction == -1:
        return mt5.ORDER_TYPE_SELL_LIMIT
    raise ValueError(f"Unsupported pending signal class: {signal_class}")


def wait_entry_price_from_debug(signal: Signal) -> Optional[float]:
    debug = getattr(signal, "debug", {}) or {}
    cls = int(signal.predicted_class)
    if cls == 2:
        return debug.get("1m_nearest_bull_fvg_top")
    if cls == 3:
        return debug.get("1m_nearest_bull_ob_top")
    if cls == -2:
        return debug.get("1m_nearest_bear_fvg_btm")
    if cls == -3:
        return debug.get("1m_nearest_bear_ob_btm")
    return None


def normalize_price(symbol_info, price: float) -> float:
    return round(float(price), int(symbol_info.digits))


def profit_calc_order_type(order_type: int) -> int:
    """
    mt5.order_calc_profit() only accepts market BUY/SELL types.
    Pending limit order types must be converted to their equivalent side
    before calculating SL/TP prices.
    """
    if order_type in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_BUY_STOP_LIMIT):
        return mt5.ORDER_TYPE_BUY
    if order_type in (mt5.ORDER_TYPE_SELL, mt5.ORDER_TYPE_SELL_LIMIT, mt5.ORDER_TYPE_SELL_STOP, mt5.ORDER_TYPE_SELL_STOP_LIMIT):
        return mt5.ORDER_TYPE_SELL
    raise ValueError(f"Unsupported order type for profit calculation: {order_type}")


def calc_target_price_by_pnl(symbol: str, order_type: int, volume: float, entry_price: float, target_pnl: float) -> float:
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError(f"Symbol not found while calculating SL/TP: {symbol}")

    point = symbol_info.point
    if point <= 0:
        raise RuntimeError(f"Invalid point size for {symbol}")

    calc_type = profit_calc_order_type(order_type)
    is_buy = calc_type == mt5.ORDER_TYPE_BUY
    near = entry_price

    # For buys: profit target is above entry, loss target is below entry.
    # For sells: profit target is below entry, loss target is above entry.
    far = entry_price + (1000 * point if (target_pnl > 0) == is_buy else -1000 * point)

    for _ in range(30):
        far = normalize_price(symbol_info, far)
        pnl_far = mt5.order_calc_profit(calc_type, symbol, volume, entry_price, far)
        if pnl_far is None:
            raise RuntimeError(
                f"order_calc_profit failed: {mt5.last_error()} | "
                f"calc_type={calc_type}, original_type={order_type}, volume={volume}, "
                f"entry={entry_price}, target_price={far}"
            )
        if (target_pnl < 0 and pnl_far <= target_pnl) or (target_pnl > 0 and pnl_far >= target_pnl):
            break
        far = entry_price + 2 * (far - entry_price)

    lo, hi = min(near, far), max(near, far)
    for _ in range(60):
        mid = normalize_price(symbol_info, (lo + hi) / 2)
        pnl_mid = mt5.order_calc_profit(calc_type, symbol, volume, entry_price, mid)
        if pnl_mid is None:
            raise RuntimeError(
                f"order_calc_profit failed mid-search: {mt5.last_error()} | "
                f"calc_type={calc_type}, original_type={order_type}, volume={volume}, "
                f"entry={entry_price}, target_price={mid}"
            )

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

    return normalize_price(symbol_info, (lo + hi) / 2)


def compute_live_sl_tp(symbol: str, order_type: int, volume: float, entry_price: float, balance: float) -> tuple[float, Optional[float], float, float]:
    risk_usd = balance * STOP_LOSS_ACCOUNT_FRACTION
    reward_usd = risk_usd * RISK_REWARD_RATIO
    sl_price = calc_target_price_by_pnl(symbol, order_type, volume, entry_price, -risk_usd)
    tp_price = calc_target_price_by_pnl(symbol, order_type, volume, entry_price, reward_usd) if USE_TAKE_PROFIT else None
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

    volume = float(LOT_SIZE)
    wait_price = wait_entry_price_from_debug(signal)
    use_pending = USE_PENDING_FOR_WAIT_ENTRIES and is_wait_entry(signal.predicted_class)

    if use_pending:
        if wait_price is None or not np.isfinite(float(wait_price)) or float(wait_price) <= 0:
            log_message(f"WAIT signal skipped: no valid active zone price for {signal_name(signal.predicted_class)}")
            return None
        order_type = pending_order_type_from_signal(signal.predicted_class)
        price = normalize_price(symbol_info, float(wait_price))

        if order_type == mt5.ORDER_TYPE_BUY_LIMIT and price >= tick.ask:
            log_message(f"WAIT BUY skipped: limit price {price} is not below ask {tick.ask}")
            return None
        if order_type == mt5.ORDER_TYPE_SELL_LIMIT and price <= tick.bid:
            log_message(f"WAIT SELL skipped: limit price {price} is not above bid {tick.bid}")
            return None
    else:
        order_type = order_type_from_signal(signal.predicted_class)
        price = normalize_price(symbol_info, tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid)

    sl_price, tp_price, risk_usd, reward_usd = compute_live_sl_tp(symbol, order_type, volume, price, float(account.balance))

    debug = getattr(signal, "debug", {}) or {}
    debug["wait_entry_price"] = wait_price

    print_table("ORDER REQUEST", [
        ("action", "PENDING_LIMIT" if use_pending else "MARKET"),
        ("entry_style", signal_name(signal.predicted_class)),
        ("confidence", signal.confidence),
        ("volume", volume),
        ("entry_price", price),
        ("stop_loss", sl_price),
        ("take_profit", tp_price),
        ("risk_usd", risk_usd),
        ("reward_usd", reward_usd),
        ("spread_points", spread_pts),
        ("dry_run", DRY_RUN),
    ])

    if use_pending:
        expiration = datetime.now() + timedelta(minutes=PENDING_EXPIRATION_MINUTES)
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price if tp_price is not None else 0.0,
            "deviation": DEVIATION,
            "magic": MAGIC_NUMBER,
            "comment": ORDER_COMMENT,
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": int(expiration.timestamp()),
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
    else:
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

    if DRY_RUN:
        return {"retcode": "DRY_RUN", "request": request, "price": price, "sl": sl_price, "tp": tp_price, "volume": volume, "pending": use_pending}

    check = mt5.order_check(request)
    if check is None:
        raise RuntimeError(f"order_check failed: {mt5.last_error()}")

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send failed: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log_message(f"Order rejected: retcode={result.retcode} comment={result.comment}")
        return None

    log_message(f"ORDER SENT -> order={result.order} deal={result.deal} pending={use_pending}")
    return result._asdict()

def close_position(position, reason: str = "timeout") -> Optional[dict]:
    symbol_info = ensure_symbol(position.symbol)
    tick = get_symbol_tick(position.symbol)
    if position.type == mt5.POSITION_TYPE_BUY:
        close_type, price = mt5.ORDER_TYPE_SELL, normalize_price(symbol_info, tick.bid)
    else:
        close_type, price = mt5.ORDER_TYPE_BUY, normalize_price(symbol_info, tick.ask)
    request = {"action": mt5.TRADE_ACTION_DEAL, "position": position.ticket, "symbol": position.symbol, "volume": position.volume, "type": close_type, "price": price, "deviation": DEVIATION, "magic": MAGIC_NUMBER, "comment": f"{ORDER_COMMENT}-{reason}"[:31], "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
    if DRY_RUN:
        return {"retcode": "DRY_RUN_CLOSE", "request": request}
    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"close order_send failed: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log_message(f"Close rejected: retcode={result.retcode} comment={result.comment}")
        return None
    return result._asdict()


def manage_timeouts(symbol: str, timeframe: int, state: dict) -> None:
    if DRY_RUN:
        return
    rates = get_latest_rates(symbol, timeframe, count=5)
    latest_closed_bar_time = int(rates.iloc[-2]["time"].timestamp())
    tf_seconds = TIMEFRAME_TO_SECONDS.get(timeframe, 60)
    positions = get_open_positions(symbol, MAGIC_NUMBER)
    live_tickets = {str(p.ticket) for p in positions}
    state["positions"] = {k: v for k, v in state.get("positions", {}).items() if k in live_tickets}
    for pos in positions:
        meta = state.get("positions", {}).get(str(pos.ticket))
        if not meta:
            continue
        bars_open = max(0, round((latest_closed_bar_time - int(meta["entry_bar_time"])) / tf_seconds))
        if bars_open >= HOLD_BARS:
            result = close_position(pos, reason="timeout")
            if result is not None:
                append_trade_log({"timestamp": datetime.now().isoformat(), "event": "timeout_close", "ticket": pos.ticket, "symbol": pos.symbol, "bars_open": bars_open, "profit": pos.profit})
                state["positions"].pop(str(pos.ticket), None)


def should_skip_signal(signal: Signal) -> Optional[str]:
    if signal.confidence < CONFIDENCE_THRESHOLD:
        return "low_confidence"
    if ONLY_TRADE_SIGNALS and signal.predicted_class == 0:
        return "neutral_signal"
    if signal.predicted_class == 1 and not ALLOW_LONG:
        return "long_disabled"
    if signal.predicted_class == -1 and not ALLOW_SHORT:
        return "short_disabled"
    return None


def can_open_new_position(symbol: str, signal: Signal) -> tuple[bool, str]:
    positions = get_open_positions(symbol, MAGIC_NUMBER)
    signal_dir = signal_direction(signal.predicted_class)

    if signal_dir == 0:
        return False, "neutral_signal"

    if not positions:
        return True, "ok"

    same_side = [p for p in positions if position_side(p) == signal_dir]
    opposite_side = [p for p in positions if position_side(p) not in (0, signal_dir)]

    # If a same-direction position already exists, keep the previous behavior:
    # do not stack more same-side trades when one-position mode is enabled.
    if ONE_POSITION_PER_SYMBOL and same_side:
        return False, "same_direction_position_exists"

    if len(same_side) >= MAX_SAME_SIDE_POSITIONS:
        return False, "max_same_side_positions"

    # Opposite positions are allowed through because the main loop will close
    # them first, then enter the new opposite signal.
    if CLOSE_OPPOSITE_ON_SIGNAL and opposite_side:
        return True, "flip_allowed"

    if ONE_POSITION_PER_SYMBOL and positions:
        return False, "position_already_open"

    if len(positions) >= MAX_TOTAL_POSITIONS:
        return False, "max_total_positions"

    return True, "ok"


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
    print_table("BOT SETTINGS", [("symbol", SYMBOL), ("features", len(feature_list)), ("hold_bars", HOLD_BARS), ("confidence", CONFIDENCE_THRESHOLD), ("lot_size", LOT_SIZE), ("risk_reward", RISK_REWARD_RATIO), ("pending_wait", USE_PENDING_FOR_WAIT_ENTRIES), ("flip_enabled", CLOSE_OPPOSITE_ON_SIGNAL), ("enter_after_flip", ENTER_AFTER_FLIP), ("dry_run", DRY_RUN)])
    try:
        while True:
            manage_timeouts(SYMBOL, TIMEFRAME, state)
            signal = predict_signal(model, feature_list, SYMBOL)
            if state.get("last_signal_bar_time") == signal.bar_time:
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue
            state["last_signal_bar_time"] = signal.bar_time
            print_signal_table(signal)
            skip_reason = should_skip_signal(signal)
            if skip_reason:
                append_trade_log({"timestamp": datetime.now().isoformat(), "event": "signal_skipped", "bar_time": signal.bar_time, "predicted_class": signal.predicted_class, "confidence": signal.confidence, "reason": skip_reason, "probabilities": json.dumps(signal.raw_probabilities), "debug": json.dumps(getattr(signal, "debug", {}))})
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue
            allowed, reason = can_open_new_position(SYMBOL, signal)
            if not allowed:
                log_message(f"Signal ignored: {reason}")
                append_trade_log({"timestamp": datetime.now().isoformat(), "event": "signal_skipped", "bar_time": signal.bar_time, "predicted_class": signal.predicted_class, "confidence": signal.confidence, "reason": reason, "debug": json.dumps(getattr(signal, "debug", {}))})
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue

            if CLOSE_OPPOSITE_ON_SIGNAL:
                cancelled_pending = 0
                if CANCEL_PENDING_ON_FLIP:
                    cancelled_pending = cancel_opposite_pending_orders(SYMBOL, signal)

                closed_positions = close_opposite_positions(SYMBOL, signal, state)
                if closed_positions > 0 or cancelled_pending > 0:
                    log_message(
                        f"Signal flip handled: closed_positions={closed_positions}, "
                        f"cancelled_pending={cancelled_pending}"
                    )
                    save_state(state)
                    time.sleep(FLIP_CLOSE_DELAY_SECONDS)

                    if not ENTER_AFTER_FLIP:
                        append_trade_log({
                            "timestamp": datetime.now().isoformat(),
                            "event": "signal_flip_exit_only",
                            "bar_time": signal.bar_time,
                            "predicted_class": signal.predicted_class,
                            "confidence": signal.confidence,
                            "closed_positions": closed_positions,
                            "cancelled_pending": cancelled_pending,
                            "debug": json.dumps(getattr(signal, "debug", {})),
                        })
                        time.sleep(POLL_SECONDS)
                        continue

            result = place_market_order(SYMBOL, signal)
            if result is not None:
                ticket = str(result.get("order") or result.get("deal") or f"dryrun-{signal.bar_time}")
                state.setdefault("positions", {})[ticket] = {"entry_bar_time": signal.bar_time, "predicted_class": signal.predicted_class, "confidence": signal.confidence}
                append_trade_log({"timestamp": datetime.now().isoformat(), "event": "order_sent" if not DRY_RUN else "dry_run_order", "bar_time": signal.bar_time, "ticket": ticket, "symbol": SYMBOL, "predicted_class": signal.predicted_class, "confidence": signal.confidence, "probabilities": json.dumps(signal.raw_probabilities), "debug": json.dumps(getattr(signal, "debug", {})), "result": json.dumps(result, default=str)})
            save_state(state)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        log_message("Stopped by user")
    finally:
        save_state(state)
        mt5.shutdown()


if __name__ == "__main__":
    main()
