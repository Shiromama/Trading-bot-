from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

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
CONFIDENCE_THRESHOLD = 0.60
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
RISK_REWARD_RATIO = 1.5
USE_TAKE_PROFIT = True
MIN_BALANCE_TO_TRADE = 1.0
MAX_TOTAL_POSITIONS = 4
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
OB_RECENT_WINDOW = 5
OB_MAX_ACTIVE_ZONES = 30
OB_NO_ZONE_DISTANCE = 10.0
OB_NO_ZONE_AGE = 9999

# Fair Value Gap settings - must match dataset builder
FVG_THRESHOLD_PCT = 0.0
FVG_AUTO_THRESHOLD = False
FVG_RECENT_WINDOW = 5
FVG_MAX_ACTIVE_ZONES = 50
FVG_NO_ZONE_DISTANCE = 10.0
FVG_NO_ZONE_AGE = 9999

# Fractal BOS / CHoCH market structure settings - must match dataset builder
STRUCTURE_FRACTAL_LENGTH = 5
STRUCTURE_RECENT_WINDOW = 5
STRUCTURE_NO_BREAK_AGE = 9999

# Liquidity Sweep / Rejection settings - must match dataset builder
SWEEP_LOOKBACK = 20
SWEEP_REJECTION_BODY_CLOSE = True
SWEEP_RECENT_WINDOW = 5
SWEEP_NO_SWEEP_AGE = 9999
SWEEP_STRONG_WICK_RATIO = 0.45
SWEEP_STRONG_CLOSE_BEYOND_MIDPOINT = True

# True session-based HTF liquidity settings - must match dataset builder
HTF_LIQUIDITY_NEAR_ATR_MULTIPLIER = 0.35
HTF_LIQUIDITY_NEAR_PCT_FALLBACK = 0.0015
HTF_LIQUIDITY_RECENT_WINDOW = 10
HTF_LIQUIDITY_NO_SWEEP_AGE = 9999
LIVE_M1_HISTORY_BARS = 10000
USE_DIRECT_HTF_LEVELS = True
DIRECT_HTF_FALLBACK_TO_M1 = True

# ICT Session Liquidity settings - must match dataset builder
SESSION_LIQUIDITY_RECENT_WINDOW = 10
SESSION_LIQUIDITY_NEAR_ATR_MULTIPLIER = 0.35
SESSION_LIQUIDITY_NEAR_PCT_FALLBACK = 0.0015
SESSION_LIQUIDITY_NO_SWEEP_AGE = 9999
SESSION_WINDOWS = {
    "asia": (0, 8),
    "london": (8, 16),
    "ny": (13, 21),
}
KILLZONE_WINDOWS = {
    "asia_killzone": (0, 3),
    "london_killzone": (7, 10),
    "ny_killzone": (13, 16),
    "london_ny_overlap": (13, 16),
}

# Smart execution layer thresholds
STRONG_CONTEXT_CONFIDENCE_THRESHOLD = 0.50
WEAK_CONTEXT_CONFIDENCE_THRESHOLD = 0.70
REQUIRE_STRONG_CONTEXT_FOR_ENTRY = True
CONTEXT_STRENGTH_MIN_SCORE = 4.0
FLIP_ONLY_ON_STRONG_CONTEXT = True


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


def print_table(title: str, rows: list[tuple[str, object]], value_width: int = 16, max_cols: int = 3):
    """
    Smart grid table:
    - Automatically spreads rows across multiple columns
    - Prevents vertical overflow
    - Keeps everything visible in terminal
    """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] {title}")

    if not rows:
        print("(no data)")
        return

    # --- Clean values ---
    def clean(value):
        v = fmt_value(value)
        return v[:value_width-3] + "..." if len(v) > value_width else v

    # --- Split into chunks ---
    total = len(rows)
    cols = min(max_cols, max(1, total // 15 + 1))  # auto scale
    chunk_size = (total + cols - 1) // cols

    chunks = [rows[i*chunk_size:(i+1)*chunk_size] for i in range(cols)]

    # --- Column widths ---
    key_w = 18
    val_w = value_width

    # --- Build header ---
    header = ""
    border = ""

    for _ in range(cols):
        border += "+" + "-"*(key_w+2) + "+" + "-"*(val_w+2) + "+  "
        header += f"| {'Field':<{key_w}} | {'Value':<{val_w}} |  "

    print(border)
    print(header)
    print(border)

    # --- Print rows ---
    for i in range(chunk_size):
        line = ""
        for c in range(cols):
            if i < len(chunks[c]):
                k, v = chunks[c][i]
                line += f"| {str(k):<{key_w}} | {clean(v):<{val_w}} |  "
            else:
                line += f"| {'':<{key_w}} | {'':<{val_w}} |  "
        print(line)

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
        ("prev_D_hi", debug.get("prev_daily_high")),
        ("prev_D_lo", debug.get("prev_daily_low")),
        ("prev_W_hi", debug.get("prev_weekly_high")),
        ("prev_W_lo", debug.get("prev_weekly_low")),
        ("near_D_hi", debug.get("near_prev_daily_high")),
        ("near_D_lo", debug.get("near_prev_daily_low")),
        ("swept_D_hi", debug.get("swept_prev_daily_high")),
        ("swept_D_lo", debug.get("swept_prev_daily_low")),
        ("reject_D_hi", debug.get("reject_prev_daily_high")),
        ("reject_D_lo", debug.get("reject_prev_daily_low")),
        ("htf_near_hi", debug.get("htf_liquidity_near_high_score")),
        ("htf_near_lo", debug.get("htf_liquidity_near_low_score")),
        ("htf_reject_hi", debug.get("htf_liquidity_reject_high_score")),
        ("htf_reject_lo", debug.get("htf_liquidity_reject_low_score")),
        ("htf_rev_bias", debug.get("htf_liquidity_reversal_bias")),
        ("liq_bull_score", debug.get("liquidity_interaction_bull_score")),
        ("liq_bear_score", debug.get("liquidity_interaction_bear_score")),
        ("liq_score_diff", debug.get("liquidity_interaction_score_diff")),
        ("sess_rej_decay", debug.get("session_reject_decay_bias")),
        ("sess_sweep_decay", debug.get("session_sweep_decay_bias")),
        ("adv_liq_bias", debug.get("advanced_liquidity_pressure_bias")),
        ("entry_fusion", debug.get("entry_zone_liquidity_fusion_bias")),
        ("struct_norm", debug.get("structure_bias_normalized")),
        ("15m_disp_press", debug.get("15m_displacement_pressure")),
        ("5m_disp_press", debug.get("5m_displacement_pressure")),
        ("1m_disp_press", debug.get("1m_displacement_pressure")),
        ("seq_disp_bias", debug.get("seq_displacement_bias_50")),
        ("seq_struct_bias", debug.get("seq_structure_break_bias_50")),
        ("seq_sweep_bias", debug.get("seq_sweep_pressure_bias_50")),
        ("seq_comp_50", debug.get("seq_compression_count_50")),
        ("seq_comp_break", debug.get("seq_breakout_after_compression_count_50")),
        ("dnt_uncertainty", debug.get("dnt_uncertainty_score")),
        ("dnt_conflict", debug.get("dnt_context_conflict_score")),
        ("dnt_low_quality", debug.get("dnt_low_quality_trade_environment")),
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



# ========== LIQUIDITY SWEEP / REJECTION FEATURES ==========
def add_liquidity_sweep_features(
    df: pd.DataFrame,
    prefix: str,
    lookback: int = SWEEP_LOOKBACK,
    recent_window: int = SWEEP_RECENT_WINDOW,
) -> pd.DataFrame:
    """
    Adds liquidity sweep, rejection, strength, wick-quality, and ATR-normalized features.

    No future leakage:
      - prev_high/prev_low use shift(1) before rolling.
      - current candle is only compared against already completed candles.

    Core idea:
      - sweep_high / sweep_low = price took recent high/low liquidity.
      - sweep_reject_high / sweep_reject_low = price took liquidity then closed back inside.
      - *_strength = how far price swept beyond the level.
      - *_atr_strength = sweep size normalized by volatility.
      - *_wick_rejection_strength = how much of the candle rejected from the swept level.
    """
    out = df.copy()

    prev_high = out["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    prev_low = out["low"].shift(1).rolling(lookback, min_periods=lookback).min()

    out[f"{prefix}_prev_high_{lookback}"] = prev_high
    out[f"{prefix}_prev_low_{lookback}"] = prev_low

    out[f"{prefix}_sweep_high"] = (out["high"] > prev_high).astype(int)
    out[f"{prefix}_sweep_low"] = (out["low"] < prev_low).astype(int)

    if SWEEP_REJECTION_BODY_CLOSE:
        out[f"{prefix}_sweep_reject_high"] = (
            (out[f"{prefix}_sweep_high"] == 1) &
            (out["close"] < prev_high)
        ).astype(int)
        out[f"{prefix}_sweep_reject_low"] = (
            (out[f"{prefix}_sweep_low"] == 1) &
            (out["close"] > prev_low)
        ).astype(int)
    else:
        out[f"{prefix}_sweep_reject_high"] = (
            (out[f"{prefix}_sweep_high"] == 1) &
            (out[["open", "close"]].max(axis=1) < prev_high)
        ).astype(int)
        out[f"{prefix}_sweep_reject_low"] = (
            (out[f"{prefix}_sweep_low"] == 1) &
            (out[["open", "close"]].min(axis=1) > prev_low)
        ).astype(int)

    # ----- Sweep depth / strength -----
    raw_high_depth = (out["high"] - prev_high).clip(lower=0)
    raw_low_depth = (prev_low - out["low"]).clip(lower=0)

    out[f"{prefix}_sweep_high_depth"] = np.where(out[f"{prefix}_sweep_high"] == 1, raw_high_depth, 0.0)
    out[f"{prefix}_sweep_low_depth"] = np.where(out[f"{prefix}_sweep_low"] == 1, raw_low_depth, 0.0)

    out[f"{prefix}_sweep_high_strength"] = out[f"{prefix}_sweep_high_depth"] / (out["close"] + EPS)
    out[f"{prefix}_sweep_low_strength"] = out[f"{prefix}_sweep_low_depth"] / (out["close"] + EPS)

    atr_col = f"{prefix}_atr14"
    if atr_col in out.columns:
        atr_safe = out[atr_col].replace(0, np.nan)
    else:
        atr_safe = out[f"{prefix}_range"].rolling(lookback, min_periods=max(3, lookback // 2)).mean().replace(0, np.nan)

    out[f"{prefix}_sweep_high_atr_strength"] = out[f"{prefix}_sweep_high_depth"] / (atr_safe + EPS)
    out[f"{prefix}_sweep_low_atr_strength"] = out[f"{prefix}_sweep_low_depth"] / (atr_safe + EPS)

    # ----- Wick rejection quality -----
    range_safe = out[f"{prefix}_range"].replace(0, np.nan) if f"{prefix}_range" in out.columns else (out["high"] - out["low"]).replace(0, np.nan)
    upper_wick_ratio = out.get(f"{prefix}_upper_wick_ratio", (out["high"] - out[["open", "close"]].max(axis=1)) / (range_safe + EPS))
    lower_wick_ratio = out.get(f"{prefix}_lower_wick_ratio", (out[["open", "close"]].min(axis=1) - out["low"]) / (range_safe + EPS))
    close_pos = out.get(f"{prefix}_close_pos_in_range", (out["close"] - out["low"]) / (range_safe + EPS))

    out[f"{prefix}_sweep_high_wick_ratio"] = np.where(out[f"{prefix}_sweep_high"] == 1, upper_wick_ratio, 0.0)
    out[f"{prefix}_sweep_low_wick_ratio"] = np.where(out[f"{prefix}_sweep_low"] == 1, lower_wick_ratio, 0.0)

    # How much price rejected back from the swept extreme into the old range.
    out[f"{prefix}_sweep_high_rejection_depth"] = np.where(
        out[f"{prefix}_sweep_reject_high"] == 1,
        (out["high"] - out["close"]).clip(lower=0),
        0.0
    )
    out[f"{prefix}_sweep_low_rejection_depth"] = np.where(
        out[f"{prefix}_sweep_reject_low"] == 1,
        (out["close"] - out["low"]).clip(lower=0),
        0.0
    )

    out[f"{prefix}_sweep_high_rejection_atr_strength"] = out[f"{prefix}_sweep_high_rejection_depth"] / (atr_safe + EPS)
    out[f"{prefix}_sweep_low_rejection_atr_strength"] = out[f"{prefix}_sweep_low_rejection_depth"] / (atr_safe + EPS)

    out[f"{prefix}_sweep_high_wick_rejection_strength"] = (
        out[f"{prefix}_sweep_reject_high"] * upper_wick_ratio * out[f"{prefix}_sweep_high_atr_strength"]
    )
    out[f"{prefix}_sweep_low_wick_rejection_strength"] = (
        out[f"{prefix}_sweep_reject_low"] * lower_wick_ratio * out[f"{prefix}_sweep_low_atr_strength"]
    )

    strong_high_extra = close_pos < 0.5 if SWEEP_STRONG_CLOSE_BEYOND_MIDPOINT else True
    strong_low_extra = close_pos > 0.5 if SWEEP_STRONG_CLOSE_BEYOND_MIDPOINT else True

    out[f"{prefix}_strong_sweep_reject_high"] = (
        (out[f"{prefix}_sweep_reject_high"] == 1) &
        (upper_wick_ratio >= SWEEP_STRONG_WICK_RATIO) &
        strong_high_extra
    ).astype(int)

    out[f"{prefix}_strong_sweep_reject_low"] = (
        (out[f"{prefix}_sweep_reject_low"] == 1) &
        (lower_wick_ratio >= SWEEP_STRONG_WICK_RATIO) &
        strong_low_extra
    ).astype(int)

    out[f"{prefix}_dist_to_prev_high_{lookback}"] = (prev_high - out["close"]) / (out["close"] + EPS)
    out[f"{prefix}_dist_to_prev_low_{lookback}"] = (out["close"] - prev_low) / (out["close"] + EPS)

    # ----- Recent/age features -----
    def bars_since(signal: pd.Series) -> pd.Series:
        last_idx = pd.Series(
            np.where(signal.astype(bool), np.arange(len(out)), np.nan),
            index=out.index
        ).ffill()
        bar_index = pd.Series(np.arange(len(out)), index=out.index)
        return (bar_index - last_idx).fillna(SWEEP_NO_SWEEP_AGE)

    out[f"{prefix}_bars_since_sweep_high"] = bars_since(out[f"{prefix}_sweep_high"])
    out[f"{prefix}_bars_since_sweep_low"] = bars_since(out[f"{prefix}_sweep_low"])
    out[f"{prefix}_bars_since_sweep_reject_high"] = bars_since(out[f"{prefix}_sweep_reject_high"])
    out[f"{prefix}_bars_since_sweep_reject_low"] = bars_since(out[f"{prefix}_sweep_reject_low"])
    out[f"{prefix}_bars_since_strong_sweep_reject_high"] = bars_since(out[f"{prefix}_strong_sweep_reject_high"])
    out[f"{prefix}_bars_since_strong_sweep_reject_low"] = bars_since(out[f"{prefix}_strong_sweep_reject_low"])

    out[f"{prefix}_recent_sweep_high"] = (out[f"{prefix}_bars_since_sweep_high"] <= recent_window).astype(int)
    out[f"{prefix}_recent_sweep_low"] = (out[f"{prefix}_bars_since_sweep_low"] <= recent_window).astype(int)
    out[f"{prefix}_recent_sweep_reject_high"] = (out[f"{prefix}_bars_since_sweep_reject_high"] <= recent_window).astype(int)
    out[f"{prefix}_recent_sweep_reject_low"] = (out[f"{prefix}_bars_since_sweep_reject_low"] <= recent_window).astype(int)
    out[f"{prefix}_recent_strong_sweep_reject_high"] = (out[f"{prefix}_bars_since_strong_sweep_reject_high"] <= recent_window).astype(int)
    out[f"{prefix}_recent_strong_sweep_reject_low"] = (out[f"{prefix}_bars_since_strong_sweep_reject_low"] <= recent_window).astype(int)

    out[f"{prefix}_sweep_reversal_bias"] = (
        out[f"{prefix}_recent_sweep_reject_low"] - out[f"{prefix}_recent_sweep_reject_high"]
    )
    out[f"{prefix}_strong_sweep_reversal_bias"] = (
        out[f"{prefix}_recent_strong_sweep_reject_low"] - out[f"{prefix}_recent_strong_sweep_reject_high"]
    )
    out[f"{prefix}_sweep_continuation_bias"] = (
        out[f"{prefix}_recent_sweep_high"] - out[f"{prefix}_recent_sweep_low"]
    )

    return out


def add_htf_liquidity_level_features(df: pd.DataFrame, raw_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Adds previous Daily / Weekly / Monthly high-low liquidity features.

    No future leakage:
      - Daily/weekly/monthly levels are built from completed periods only.
      - The current day uses the previous day's high/low.
      - The current week uses the previous week's high/low.
      - The current month uses the previous month's high/low.

    Main feature groups:
      - raw previous levels: prev_daily_high, prev_weekly_low, etc.
      - distance to each level
      - near level flags
      - swept / rejected / reclaimed level flags
      - bars since sweep
      - previous range position and premium/discount context
      - combined HTF liquidity bias scores
    """
    out = df.copy()
    raw = raw_1m.reindex(out.index).copy()

    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("raw_1m must have a DatetimeIndex before adding HTF liquidity features.")

    high = raw["high"]
    low = raw["low"]
    close = raw["close"]

    # ATR-based near-threshold if available; percent fallback if ATR is missing.
    if "1m_atr14" in out.columns:
        near_distance = (out["1m_atr14"].abs() * HTF_LIQUIDITY_NEAR_ATR_MULTIPLIER).fillna(close * HTF_LIQUIDITY_NEAR_PCT_FALLBACK)
    else:
        near_distance = close * HTF_LIQUIDITY_NEAR_PCT_FALLBACK

    def previous_period_levels(period_code: str) -> tuple[pd.Series, pd.Series]:
        period = raw.index.to_period(period_code)
        period_frame = pd.DataFrame({"period": period, "high": high, "low": low}, index=raw.index)
        levels = period_frame.groupby("period").agg(period_high=("high", "max"), period_low=("low", "min"))
        prev_high_by_period = levels["period_high"].shift(1)
        prev_low_by_period = levels["period_low"].shift(1)
        prev_high = pd.Series(period.map(prev_high_by_period), index=raw.index, dtype="float64")
        prev_low = pd.Series(period.map(prev_low_by_period), index=raw.index, dtype="float64")
        return prev_high, prev_low

    def bars_since(signal: pd.Series) -> pd.Series:
        last_idx = pd.Series(np.where(signal.astype(bool), np.arange(len(out)), np.nan), index=out.index).ffill()
        bar_idx = pd.Series(np.arange(len(out)), index=out.index)
        return (bar_idx - last_idx).fillna(HTF_LIQUIDITY_NO_SWEEP_AGE)

    level_specs = [
        ("daily", "D"),
        ("weekly", "W"),
        ("monthly", "M"),
    ]

    for name, period_code in level_specs:
        prev_high, prev_low = previous_period_levels(period_code)
        prev_mid = (prev_high + prev_low) / 2.0
        prev_range = (prev_high - prev_low).replace(0, np.nan)

        out[f"prev_{name}_high"] = prev_high
        out[f"prev_{name}_low"] = prev_low
        out[f"prev_{name}_mid"] = prev_mid
        out[f"prev_{name}_range"] = prev_range

        out[f"dist_to_prev_{name}_high"] = (prev_high - close) / (close + EPS)
        out[f"dist_to_prev_{name}_low"] = (close - prev_low) / (close + EPS)
        out[f"abs_dist_to_prev_{name}_high"] = (prev_high - close).abs() / (close + EPS)
        out[f"abs_dist_to_prev_{name}_low"] = (close - prev_low).abs() / (close + EPS)

        out[f"near_prev_{name}_high"] = ((prev_high - close).abs() <= near_distance).astype(int)
        out[f"near_prev_{name}_low"] = ((close - prev_low).abs() <= near_distance).astype(int)

        out[f"closed_above_prev_{name}_high"] = (close > prev_high).astype(int)
        out[f"closed_below_prev_{name}_low"] = (close < prev_low).astype(int)
        out[f"inside_prev_{name}_range"] = ((close <= prev_high) & (close >= prev_low)).astype(int)

        out[f"swept_prev_{name}_high"] = (high > prev_high).astype(int)
        out[f"swept_prev_{name}_low"] = (low < prev_low).astype(int)

        # Rejection = takes liquidity, then closes back inside the previous range.
        out[f"reject_prev_{name}_high"] = ((out[f"swept_prev_{name}_high"] == 1) & (close < prev_high)).astype(int)
        out[f"reject_prev_{name}_low"] = ((out[f"swept_prev_{name}_low"] == 1) & (close > prev_low)).astype(int)

        # Reclaim / breakout confirmation = closes beyond the swept level.
        out[f"reclaim_prev_{name}_high"] = ((out[f"swept_prev_{name}_high"] == 1) & (close > prev_high)).astype(int)
        out[f"reclaim_prev_{name}_low"] = ((out[f"swept_prev_{name}_low"] == 1) & (close < prev_low)).astype(int)

        out[f"prev_{name}_range_position"] = ((close - prev_low) / (prev_range + EPS)).clip(lower=-1, upper=2)
        out[f"above_prev_{name}_mid"] = (close > prev_mid).astype(int)
        out[f"below_prev_{name}_mid"] = (close < prev_mid).astype(int)
        out[f"prev_{name}_premium_discount"] = np.where(close > prev_mid, 1, np.where(close < prev_mid, -1, 0))

        out[f"prev_{name}_high_sweep_depth"] = np.where(out[f"swept_prev_{name}_high"] == 1, (high - prev_high).clip(lower=0), 0.0)
        out[f"prev_{name}_low_sweep_depth"] = np.where(out[f"swept_prev_{name}_low"] == 1, (prev_low - low).clip(lower=0), 0.0)
        out[f"prev_{name}_high_sweep_depth_pct"] = out[f"prev_{name}_high_sweep_depth"] / (close + EPS)
        out[f"prev_{name}_low_sweep_depth_pct"] = out[f"prev_{name}_low_sweep_depth"] / (close + EPS)

        out[f"bars_since_swept_prev_{name}_high"] = bars_since(out[f"swept_prev_{name}_high"])
        out[f"bars_since_swept_prev_{name}_low"] = bars_since(out[f"swept_prev_{name}_low"])
        out[f"bars_since_reject_prev_{name}_high"] = bars_since(out[f"reject_prev_{name}_high"])
        out[f"bars_since_reject_prev_{name}_low"] = bars_since(out[f"reject_prev_{name}_low"])

        out[f"recent_swept_prev_{name}_high"] = (out[f"bars_since_swept_prev_{name}_high"] <= HTF_LIQUIDITY_RECENT_WINDOW).astype(int)
        out[f"recent_swept_prev_{name}_low"] = (out[f"bars_since_swept_prev_{name}_low"] <= HTF_LIQUIDITY_RECENT_WINDOW).astype(int)
        out[f"recent_reject_prev_{name}_high"] = (out[f"bars_since_reject_prev_{name}_high"] <= HTF_LIQUIDITY_RECENT_WINDOW).astype(int)
        out[f"recent_reject_prev_{name}_low"] = (out[f"bars_since_reject_prev_{name}_low"] <= HTF_LIQUIDITY_RECENT_WINDOW).astype(int)

    # Combined liquidity context scores.
    out["htf_liquidity_near_high_score"] = out[["near_prev_daily_high", "near_prev_weekly_high", "near_prev_monthly_high"]].sum(axis=1)
    out["htf_liquidity_near_low_score"] = out[["near_prev_daily_low", "near_prev_weekly_low", "near_prev_monthly_low"]].sum(axis=1)

    out["htf_liquidity_sweep_high_score"] = out[["recent_swept_prev_daily_high", "recent_swept_prev_weekly_high", "recent_swept_prev_monthly_high"]].sum(axis=1)
    out["htf_liquidity_sweep_low_score"] = out[["recent_swept_prev_daily_low", "recent_swept_prev_weekly_low", "recent_swept_prev_monthly_low"]].sum(axis=1)

    out["htf_liquidity_reject_high_score"] = out[["recent_reject_prev_daily_high", "recent_reject_prev_weekly_high", "recent_reject_prev_monthly_high"]].sum(axis=1)
    out["htf_liquidity_reject_low_score"] = out[["recent_reject_prev_daily_low", "recent_reject_prev_weekly_low", "recent_reject_prev_monthly_low"]].sum(axis=1)

    # Positive = bullish liquidity reaction; negative = bearish liquidity reaction.
    out["htf_liquidity_reversal_bias"] = out["htf_liquidity_reject_low_score"] - out["htf_liquidity_reject_high_score"]
    out["htf_liquidity_continuation_bias"] = out["htf_liquidity_sweep_high_score"] - out["htf_liquidity_sweep_low_score"]

    out["daily_weekly_liquidity_confluence_high"] = ((out["near_prev_daily_high"] == 1) & (out["near_prev_weekly_high"] == 1)).astype(int)
    out["daily_weekly_liquidity_confluence_low"] = ((out["near_prev_daily_low"] == 1) & (out["near_prev_weekly_low"] == 1)).astype(int)

    return out


# ========== ICT SESSION LIQUIDITY FEATURES ==========
def add_session_liquidity_features(df: pd.DataFrame, raw_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ICT-style session liquidity features on top of the existing feature set.

    No future leakage:
      - current session high/low uses expanding values up to the current row only
      - previous session levels are taken from the most recently completed same session
      - London/NY sweep interaction features use already-known Asia/London levels
    """
    out = df.copy()
    raw = raw_1m.reindex(out.index).copy()

    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("raw_1m must have a DatetimeIndex before adding session liquidity features.")

    high = raw["high"].astype(float)
    low = raw["low"].astype(float)
    close = raw["close"].astype(float)
    hour = out.index.hour

    if "1m_atr14" in out.columns:
        near_distance = (out["1m_atr14"].abs() * SESSION_LIQUIDITY_NEAR_ATR_MULTIPLIER).fillna(
            close * SESSION_LIQUIDITY_NEAR_PCT_FALLBACK
        )
    else:
        near_distance = close * SESSION_LIQUIDITY_NEAR_PCT_FALLBACK

    def in_hour_window(start_hour: int, end_hour: int) -> np.ndarray:
        if start_hour < end_hour:
            return (hour >= start_hour) & (hour < end_hour)
        return (hour >= start_hour) | (hour < end_hour)

    def bars_since(signal: pd.Series) -> pd.Series:
        last_idx = pd.Series(np.where(signal.astype(bool), np.arange(len(out)), np.nan), index=out.index).ffill()
        bar_idx = pd.Series(np.arange(len(out)), index=out.index)
        return (bar_idx - last_idx).fillna(SESSION_LIQUIDITY_NO_SWEEP_AGE)

    for kz_name, (start_h, end_h) in KILLZONE_WINDOWS.items():
        out[f"in_{kz_name}"] = in_hour_window(start_h, end_h).astype(int)

    previous_level_names = []

    for session_name, (start_h, end_h) in SESSION_WINDOWS.items():
        mask = pd.Series(in_hour_window(start_h, end_h), index=out.index)
        out[f"in_{session_name}_session_liquidity_window"] = mask.astype(int)

        session_date = pd.Series(out.index.normalize(), index=out.index)
        if start_h > end_h:
            session_date.loc[hour < end_h] = session_date.loc[hour < end_h] - pd.Timedelta(days=1)

        session_key = pd.Series(pd.NA, index=out.index, dtype="object")
        session_key.loc[mask] = session_date.loc[mask].astype(str) + f"_{session_name}"

        session_high = high.where(mask).groupby(session_key).cummax()
        session_low = low.where(mask).groupby(session_key).cummin()

        completed_sessions = pd.DataFrame({
            "session_key": session_key[mask],
            "session_high": high[mask],
            "session_low": low[mask],
        }).groupby("session_key").agg(
            session_high=("session_high", "max"),
            session_low=("session_low", "min"),
        )

        prev_high_map = completed_sessions["session_high"].shift(1)
        prev_low_map = completed_sessions["session_low"].shift(1)
        prev_session_high = pd.Series(session_key.map(prev_high_map), index=out.index, dtype="float64").ffill()
        prev_session_low = pd.Series(session_key.map(prev_low_map), index=out.index, dtype="float64").ffill()

        out[f"current_{session_name}_high"] = session_high.ffill()
        out[f"current_{session_name}_low"] = session_low.ffill()
        out[f"prev_{session_name}_session_high"] = prev_session_high
        out[f"prev_{session_name}_session_low"] = prev_session_low
        out[f"prev_{session_name}_session_mid"] = (prev_session_high + prev_session_low) / 2.0
        out[f"prev_{session_name}_session_range"] = (prev_session_high - prev_session_low).replace(0, np.nan)

        previous_level_names.append(session_name)

        ph = out[f"prev_{session_name}_session_high"]
        pl = out[f"prev_{session_name}_session_low"]
        pm = out[f"prev_{session_name}_session_mid"]
        pr = out[f"prev_{session_name}_session_range"]

        out[f"dist_to_prev_{session_name}_session_high"] = (ph - close) / (close + EPS)
        out[f"dist_to_prev_{session_name}_session_low"] = (close - pl) / (close + EPS)
        out[f"abs_dist_to_prev_{session_name}_session_high"] = (ph - close).abs() / (close + EPS)
        out[f"abs_dist_to_prev_{session_name}_session_low"] = (close - pl).abs() / (close + EPS)

        out[f"near_prev_{session_name}_session_high"] = ((ph - close).abs() <= near_distance).astype(int)
        out[f"near_prev_{session_name}_session_low"] = ((close - pl).abs() <= near_distance).astype(int)

        out[f"swept_prev_{session_name}_session_high"] = (high > ph).astype(int)
        out[f"swept_prev_{session_name}_session_low"] = (low < pl).astype(int)
        out[f"reject_prev_{session_name}_session_high"] = ((out[f"swept_prev_{session_name}_session_high"] == 1) & (close < ph)).astype(int)
        out[f"reject_prev_{session_name}_session_low"] = ((out[f"swept_prev_{session_name}_session_low"] == 1) & (close > pl)).astype(int)
        out[f"reclaim_prev_{session_name}_session_high"] = ((out[f"swept_prev_{session_name}_session_high"] == 1) & (close > ph)).astype(int)
        out[f"reclaim_prev_{session_name}_session_low"] = ((out[f"swept_prev_{session_name}_session_low"] == 1) & (close < pl)).astype(int)

        out[f"prev_{session_name}_session_range_position"] = ((close - pl) / (pr + EPS)).clip(lower=-1, upper=2)
        out[f"prev_{session_name}_session_premium_discount"] = np.where(close > pm, 1, np.where(close < pm, -1, 0))

        out[f"bars_since_swept_prev_{session_name}_session_high"] = bars_since(out[f"swept_prev_{session_name}_session_high"])
        out[f"bars_since_swept_prev_{session_name}_session_low"] = bars_since(out[f"swept_prev_{session_name}_session_low"])
        out[f"bars_since_reject_prev_{session_name}_session_high"] = bars_since(out[f"reject_prev_{session_name}_session_high"])
        out[f"bars_since_reject_prev_{session_name}_session_low"] = bars_since(out[f"reject_prev_{session_name}_session_low"])

        out[f"recent_swept_prev_{session_name}_session_high"] = (out[f"bars_since_swept_prev_{session_name}_session_high"] <= SESSION_LIQUIDITY_RECENT_WINDOW).astype(int)
        out[f"recent_swept_prev_{session_name}_session_low"] = (out[f"bars_since_swept_prev_{session_name}_session_low"] <= SESSION_LIQUIDITY_RECENT_WINDOW).astype(int)
        out[f"recent_reject_prev_{session_name}_session_high"] = (out[f"bars_since_reject_prev_{session_name}_session_high"] <= SESSION_LIQUIDITY_RECENT_WINDOW).astype(int)
        out[f"recent_reject_prev_{session_name}_session_low"] = (out[f"bars_since_reject_prev_{session_name}_session_low"] <= SESSION_LIQUIDITY_RECENT_WINDOW).astype(int)

    out["london_swept_asia_high"] = ((out.get("session_london", 0) == 1) & (high > out["current_asia_high"])).astype(int)
    out["london_swept_asia_low"] = ((out.get("session_london", 0) == 1) & (low < out["current_asia_low"])).astype(int)
    out["london_reject_asia_high"] = ((out["london_swept_asia_high"] == 1) & (close < out["current_asia_high"])).astype(int)
    out["london_reject_asia_low"] = ((out["london_swept_asia_low"] == 1) & (close > out["current_asia_low"])).astype(int)

    out["ny_swept_london_high"] = ((out.get("session_ny", 0) == 1) & (high > out["current_london_high"])).astype(int)
    out["ny_swept_london_low"] = ((out.get("session_ny", 0) == 1) & (low < out["current_london_low"])).astype(int)
    out["ny_reject_london_high"] = ((out["ny_swept_london_high"] == 1) & (close < out["current_london_high"])).astype(int)
    out["ny_reject_london_low"] = ((out["ny_swept_london_low"] == 1) & (close > out["current_london_low"])).astype(int)

    out["ny_swept_asia_high"] = ((out.get("session_ny", 0) == 1) & (high > out["current_asia_high"])).astype(int)
    out["ny_swept_asia_low"] = ((out.get("session_ny", 0) == 1) & (low < out["current_asia_low"])).astype(int)
    out["ny_reject_asia_high"] = ((out["ny_swept_asia_high"] == 1) & (close < out["current_asia_high"])).astype(int)
    out["ny_reject_asia_low"] = ((out["ny_swept_asia_low"] == 1) & (close > out["current_asia_low"])).astype(int)

    high_near_cols = [f"near_prev_{name}_session_high" for name in previous_level_names]
    low_near_cols = [f"near_prev_{name}_session_low" for name in previous_level_names]
    high_sweep_cols = [f"recent_swept_prev_{name}_session_high" for name in previous_level_names]
    low_sweep_cols = [f"recent_swept_prev_{name}_session_low" for name in previous_level_names]
    high_reject_cols = [f"recent_reject_prev_{name}_session_high" for name in previous_level_names]
    low_reject_cols = [f"recent_reject_prev_{name}_session_low" for name in previous_level_names]

    out["session_liquidity_near_high_score"] = out[high_near_cols].sum(axis=1)
    out["session_liquidity_near_low_score"] = out[low_near_cols].sum(axis=1)
    out["session_liquidity_sweep_high_score"] = out[high_sweep_cols].sum(axis=1)
    out["session_liquidity_sweep_low_score"] = out[low_sweep_cols].sum(axis=1)
    out["session_liquidity_reject_high_score"] = out[high_reject_cols].sum(axis=1)
    out["session_liquidity_reject_low_score"] = out[low_reject_cols].sum(axis=1)

    out["session_liquidity_reversal_bias"] = out["session_liquidity_reject_low_score"] - out["session_liquidity_reject_high_score"]
    out["session_liquidity_continuation_bias"] = out["session_liquidity_sweep_high_score"] - out["session_liquidity_sweep_low_score"]

    out["london_asia_sweep_reversal_bias"] = out["london_reject_asia_low"] - out["london_reject_asia_high"]
    out["ny_london_sweep_reversal_bias"] = out["ny_reject_london_low"] - out["ny_reject_london_high"]
    out["ny_asia_sweep_reversal_bias"] = out["ny_reject_asia_low"] - out["ny_reject_asia_high"]

    out["killzone_session_sweep_score"] = (
        out["in_london_killzone"] * (out["london_swept_asia_high"] + out["london_swept_asia_low"]) +
        out["in_ny_killzone"] * (out["ny_swept_london_high"] + out["ny_swept_london_low"] + out["ny_swept_asia_high"] + out["ny_swept_asia_low"])
    )

    return out

# ========== ADVANCED LIQUIDITY INTERACTION FEATURES ==========
def add_advanced_liquidity_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds strength-based ICT liquidity interaction features without removing anything.

    This layer turns rare binary events into more learnable continuous signals:
      - time-decayed session sweep/rejection signals
      - weighted London/NY killzone liquidity behavior
      - session rejection strength using wick quality + close position
      - OB/FVG proximity fused with prior liquidity reactions
      - bullish/bearish continuous liquidity pressure scores

    No future leakage:
      - uses only current-row and historical bars_since/session features already built earlier
      - does not use future_return, targets, or candidate label returns
    """
    out = df.copy()

    def c(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return out[name].replace([np.inf, -np.inf], np.nan).fillna(default)
        return pd.Series(default, index=out.index, dtype="float64")

    def bool_c(name: str) -> pd.Series:
        return (c(name, 0.0) == 1)

    def decay_from_bars(name: str, half_life: float = 10.0, max_age: float = 9998.0) -> pd.Series:
        bars = c(name, max_age).clip(lower=0, upper=max_age)
        decayed = np.exp(-bars / max(half_life, EPS))
        return pd.Series(np.where(bars >= max_age, 0.0, decayed), index=out.index)

    def proximity_from_dist(name: str, scale: float = 1000.0) -> pd.Series:
        # dist columns are usually pct/relative distances. Higher = closer to liquidity/zone.
        dist = c(name, 10.0).abs().clip(lower=0)
        return (1.0 / (1.0 + dist * scale)).clip(lower=0, upper=1)

    # ----- Ensure basic interaction flags exist too -----
    session_high_near = bool_c("near_prev_asia_session_high") | bool_c("near_prev_london_session_high") | bool_c("near_prev_ny_session_high")
    session_low_near = bool_c("near_prev_asia_session_low") | bool_c("near_prev_london_session_low") | bool_c("near_prev_ny_session_low")
    session_high_reject_now = bool_c("reject_prev_asia_session_high") | bool_c("reject_prev_london_session_high") | bool_c("reject_prev_ny_session_high")
    session_low_reject_now = bool_c("reject_prev_asia_session_low") | bool_c("reject_prev_london_session_low") | bool_c("reject_prev_ny_session_low")

    out["session_high_near_any"] = session_high_near.astype(int)
    out["session_low_near_any"] = session_low_near.astype(int)
    out["session_high_reject_now_any"] = session_high_reject_now.astype(int)
    out["session_low_reject_now_any"] = session_low_reject_now.astype(int)

    if "hour" in out.columns:
        out["ict_london_killzone"] = ((out["hour"] >= 7) & (out["hour"] <= 10)).astype(int)
        out["ict_ny_killzone"] = ((out["hour"] >= 13) & (out["hour"] <= 16)).astype(int)
    else:
        out["ict_london_killzone"] = c("in_london_killzone").astype(int)
        out["ict_ny_killzone"] = c("in_ny_killzone").astype(int)

    out["session_high_bearish_structure"] = (
        session_high_near & (c("structure_bear_context_score") > c("structure_bull_context_score"))
    ).astype(int)
    out["session_low_bullish_structure"] = (
        session_low_near & (c("structure_bull_context_score") > c("structure_bear_context_score"))
    ).astype(int)

    # ----- Time-decayed session sweep/rejection memory -----
    high_reject_decays = [
        decay_from_bars("bars_since_reject_prev_asia_session_high"),
        decay_from_bars("bars_since_reject_prev_london_session_high"),
        decay_from_bars("bars_since_reject_prev_ny_session_high"),
    ]
    low_reject_decays = [
        decay_from_bars("bars_since_reject_prev_asia_session_low"),
        decay_from_bars("bars_since_reject_prev_london_session_low"),
        decay_from_bars("bars_since_reject_prev_ny_session_low"),
    ]
    high_sweep_decays = [
        decay_from_bars("bars_since_swept_prev_asia_session_high"),
        decay_from_bars("bars_since_swept_prev_london_session_high"),
        decay_from_bars("bars_since_swept_prev_ny_session_high"),
    ]
    low_sweep_decays = [
        decay_from_bars("bars_since_swept_prev_asia_session_low"),
        decay_from_bars("bars_since_swept_prev_london_session_low"),
        decay_from_bars("bars_since_swept_prev_ny_session_low"),
    ]

    out["session_high_reject_decay_max"] = pd.concat(high_reject_decays, axis=1).max(axis=1)
    out["session_low_reject_decay_max"] = pd.concat(low_reject_decays, axis=1).max(axis=1)
    out["session_high_sweep_decay_max"] = pd.concat(high_sweep_decays, axis=1).max(axis=1)
    out["session_low_sweep_decay_max"] = pd.concat(low_sweep_decays, axis=1).max(axis=1)
    out["session_reject_decay_bias"] = out["session_low_reject_decay_max"] - out["session_high_reject_decay_max"]
    out["session_sweep_decay_bias"] = out["session_high_sweep_decay_max"] - out["session_low_sweep_decay_max"]

    # ----- Continuous proximity to session liquidity -----
    out["session_high_proximity_max"] = pd.concat([
        proximity_from_dist("abs_dist_to_prev_asia_session_high"),
        proximity_from_dist("abs_dist_to_prev_london_session_high"),
        proximity_from_dist("abs_dist_to_prev_ny_session_high"),
    ], axis=1).max(axis=1)
    out["session_low_proximity_max"] = pd.concat([
        proximity_from_dist("abs_dist_to_prev_asia_session_low"),
        proximity_from_dist("abs_dist_to_prev_london_session_low"),
        proximity_from_dist("abs_dist_to_prev_ny_session_low"),
    ], axis=1).max(axis=1)

    # ----- Wick/close-position based rejection strength -----
    close_pos = c("1m_close_pos_in_range", 0.5).clip(lower=0, upper=1)
    upper_wick = c("1m_upper_wick_ratio").clip(lower=0, upper=1)
    lower_wick = c("1m_lower_wick_ratio").clip(lower=0, upper=1)

    # High rejection is bearish when upper wick is large and close is lower in candle.
    out["session_high_rejection_strength_cont"] = (
        out["session_high_reject_decay_max"] * upper_wick * (1.0 - close_pos)
    )
    # Low rejection is bullish when lower wick is large and close is higher in candle.
    out["session_low_rejection_strength_cont"] = (
        out["session_low_reject_decay_max"] * lower_wick * close_pos
    )
    out["session_rejection_strength_bias_cont"] = (
        out["session_low_rejection_strength_cont"] - out["session_high_rejection_strength_cont"]
    )

    # ----- Weighted session logic: London/NY sweeps matter more than generic all-day signals -----
    london_weight = 1.0 + 0.50 * c("ict_london_killzone") + 0.25 * c("session_london")
    ny_weight = 1.0 + 0.50 * c("ict_ny_killzone") + 0.25 * c("session_ny")

    out["weighted_london_asia_high_sweep_strength"] = london_weight * (
        c("london_swept_asia_high") + 2.0 * c("london_reject_asia_high")
    ) * (1.0 + upper_wick)
    out["weighted_london_asia_low_sweep_strength"] = london_weight * (
        c("london_swept_asia_low") + 2.0 * c("london_reject_asia_low")
    ) * (1.0 + lower_wick)
    out["weighted_ny_london_high_sweep_strength"] = ny_weight * (
        c("ny_swept_london_high") + 2.0 * c("ny_reject_london_high")
    ) * (1.0 + upper_wick)
    out["weighted_ny_london_low_sweep_strength"] = ny_weight * (
        c("ny_swept_london_low") + 2.0 * c("ny_reject_london_low")
    ) * (1.0 + lower_wick)
    out["weighted_ny_asia_high_sweep_strength"] = ny_weight * (
        c("ny_swept_asia_high") + 2.0 * c("ny_reject_asia_high")
    ) * (1.0 + upper_wick)
    out["weighted_ny_asia_low_sweep_strength"] = ny_weight * (
        c("ny_swept_asia_low") + 2.0 * c("ny_reject_asia_low")
    ) * (1.0 + lower_wick)

    out["weighted_session_bullish_reversal_strength"] = (
        out["weighted_london_asia_low_sweep_strength"] +
        out["weighted_ny_london_low_sweep_strength"] +
        out["weighted_ny_asia_low_sweep_strength"] +
        out["session_low_rejection_strength_cont"]
    )
    out["weighted_session_bearish_reversal_strength"] = (
        out["weighted_london_asia_high_sweep_strength"] +
        out["weighted_ny_london_high_sweep_strength"] +
        out["weighted_ny_asia_high_sweep_strength"] +
        out["session_high_rejection_strength_cont"]
    )
    out["weighted_session_reversal_strength_bias"] = (
        out["weighted_session_bullish_reversal_strength"] - out["weighted_session_bearish_reversal_strength"]
    )

    # ----- OB/FVG + liquidity fusion: continuous instead of rare binary only -----
    bull_ob_prox = proximity_from_dist("1m_dist_to_bull_ob")
    bear_ob_prox = proximity_from_dist("1m_dist_to_bear_ob")
    bull_fvg_prox = proximity_from_dist("1m_dist_to_bull_fvg")
    bear_fvg_prox = proximity_from_dist("1m_dist_to_bear_fvg")

    out["bull_ob_liquidity_fusion_strength"] = bull_ob_prox * out["session_low_rejection_strength_cont"]
    out["bear_ob_liquidity_fusion_strength"] = bear_ob_prox * out["session_high_rejection_strength_cont"]
    out["bull_fvg_liquidity_fusion_strength"] = bull_fvg_prox * out["session_low_rejection_strength_cont"]
    out["bear_fvg_liquidity_fusion_strength"] = bear_fvg_prox * out["session_high_rejection_strength_cont"]

    out["bull_entry_zone_liquidity_fusion_strength"] = (
        out["bull_ob_liquidity_fusion_strength"] + out["bull_fvg_liquidity_fusion_strength"]
    )
    out["bear_entry_zone_liquidity_fusion_strength"] = (
        out["bear_ob_liquidity_fusion_strength"] + out["bear_fvg_liquidity_fusion_strength"]
    )
    out["entry_zone_liquidity_fusion_bias"] = (
        out["bull_entry_zone_liquidity_fusion_strength"] - out["bear_entry_zone_liquidity_fusion_strength"]
    )

    # ----- Structure-weighted liquidity pressure -----
    bull_struct = c("structure_bull_context_score")
    bear_struct = c("structure_bear_context_score")
    struct_total = (bull_struct + bear_struct).replace(0, np.nan)
    out["structure_bias_normalized"] = ((bull_struct - bear_struct) / (struct_total + EPS)).fillna(0.0).clip(-1, 1)

    out["advanced_liquidity_bull_pressure"] = (
        out["weighted_session_bullish_reversal_strength"] +
        out["bull_entry_zone_liquidity_fusion_strength"] +
        out["session_low_proximity_max"] * out["session_low_reject_decay_max"]
    ) * (1.0 + out["structure_bias_normalized"].clip(lower=0))

    out["advanced_liquidity_bear_pressure"] = (
        out["weighted_session_bearish_reversal_strength"] +
        out["bear_entry_zone_liquidity_fusion_strength"] +
        out["session_high_proximity_max"] * out["session_high_reject_decay_max"]
    ) * (1.0 + (-out["structure_bias_normalized"].clip(upper=0)))

    out["advanced_liquidity_pressure_bias"] = (
        out["advanced_liquidity_bull_pressure"] - out["advanced_liquidity_bear_pressure"]
    )
    out["advanced_liquidity_pressure_abs"] = out["advanced_liquidity_pressure_bias"].abs()

    # ----- Legacy-style simple scores, kept for compatibility if older training reports expect them -----
    out["liquidity_interaction_bull_score"] = (
        out["session_low_bullish_structure"] +
        (out["session_low_reject_now_any"] == 1).astype(int) +
        (out["bull_entry_zone_liquidity_fusion_strength"] > 0).astype(int)
    )
    out["liquidity_interaction_bear_score"] = (
        out["session_high_bearish_structure"] +
        (out["session_high_reject_now_any"] == 1).astype(int) +
        (out["bear_entry_zone_liquidity_fusion_strength"] > 0).astype(int)
    )
    out["liquidity_interaction_score_diff"] = out["liquidity_interaction_bull_score"] - out["liquidity_interaction_bear_score"]

    return out

# ========== MUST-ADD CONTEXT FEATURES: TIME + DAILY POSITION + VOLATILITY REGIME ==========
def add_must_have_context_features(df: pd.DataFrame, raw_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the first context upgrade only, without changing/removing existing features.

    Features added:
      - cyclical time: hour_sin/hour_cos, dow_sin/dow_cos
      - session flags: Asia, London, NY, London open, NY open
      - daily range context: daily_position and distance to rolling daily high/low
      - volatility regime context: ATR rolling mean and ATR z-score

    Notes:
      - Uses the existing datetime index from MT5 data.
      - Rolling daily high/low uses 1440 one-minute candles, so it stays historical only.
      - ATR z-score uses existing 1m_atr_pct14, which is already calculated before this function runs.
    """
    out = df.copy()

    # ----- Time cycle features -----
    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    # ----- Session / activity-window flags -----
    # These use the timestamp as provided by your MT5 CSV/server time.
    out["session_asia"] = ((out["hour"] >= 0) & (out["hour"] < 8)).astype(int)
    out["session_london"] = ((out["hour"] >= 8) & (out["hour"] < 16)).astype(int)
    out["session_ny"] = ((out["hour"] >= 13) & (out["hour"] < 21)).astype(int)
    out["london_open"] = ((out["hour"] >= 7) & (out["hour"] <= 10)).astype(int)
    out["ny_open"] = ((out["hour"] >= 13) & (out["hour"] <= 16)).astype(int)

    # ----- Daily price-location context -----
    raw_aligned = raw_1m.reindex(out.index)
    daily_high = raw_aligned["high"].rolling(1440, min_periods=60).max()
    daily_low = raw_aligned["low"].rolling(1440, min_periods=60).min()
    daily_range = daily_high - daily_low

    out["daily_high"] = daily_high
    out["daily_low"] = daily_low
    out["daily_range"] = daily_range
    out["daily_position"] = (raw_aligned["close"] - daily_low) / (daily_range + EPS)
    out["dist_to_daily_high"] = (daily_high - raw_aligned["close"]) / (raw_aligned["close"] + EPS)
    out["dist_to_daily_low"] = (raw_aligned["close"] - daily_low) / (raw_aligned["close"] + EPS)

    # Keep daily_position clean even during startup rows or flat-range moments.
    out["daily_position"] = out["daily_position"].clip(lower=0, upper=1)

    # ----- Volatility regime context -----
    if "1m_atr_pct14" not in out.columns:
        raise ValueError("1m_atr_pct14 is required before adding volatility context features.")

    out["atr_mean_100"] = out["1m_atr_pct14"].rolling(100, min_periods=20).mean()
    out["atr_std_100"] = out["1m_atr_pct14"].rolling(100, min_periods=20).std()
    out["volatility_zscore"] = (out["1m_atr_pct14"] - out["atr_mean_100"]) / (out["atr_std_100"] + EPS)

    return out


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



# ========== BEHAVIOR FLOW UPGRADE SETTINGS ==========
# Added on top of the existing feature stack. These settings do not alter labels.
SEQUENCE_WINDOWS = [5, 10, 20, 50]
DISPLACEMENT_BODY_ATR_THRESHOLD = 1.20
DISPLACEMENT_EFFICIENCY_THRESHOLD = 0.55
EXPANSION_RANGE_MEAN_THRESHOLD = 1.40
COMPRESSION_SHORT_WINDOW = 20
COMPRESSION_LONG_WINDOW = 100
COMPRESSION_THRESHOLD = 0.75
CHOP_FLIP_WINDOW = 20
DO_NOT_TRADE_CONFLICT_THRESHOLD = 2.0


# ========== DISPLACEMENT / EXPANSION DETECTION ==========
def add_displacement_expansion_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Adds displacement and expansion features without changing existing logic.

    Purpose:
      - separates weak BOS/drift moves from aggressive displacement candles
      - uses only current and historical candle data
      - works even when ATR is not available on 5m/15m by falling back to range mean
    """
    out = df.copy()

    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return out[name].replace([np.inf, -np.inf], np.nan).fillna(default)
        return pd.Series(default, index=out.index, dtype="float64")

    range_safe = col(f"{prefix}_range", 0.0).replace(0, np.nan)
    body = col(f"{prefix}_body", 0.0)
    body_abs = body.abs()

    atr_col = f"{prefix}_atr14"
    if atr_col in out.columns:
        atr_safe = out[atr_col].replace([np.inf, -np.inf], np.nan).replace(0, np.nan)
    else:
        atr_safe = col(f"{prefix}_range", 0.0).rolling(14, min_periods=5).mean().replace(0, np.nan)

    range_mean10 = col(f"{prefix}_range_mean10", 0.0).replace(0, np.nan)
    volume_vs_mean = col(f"{prefix}_volume_vs_mean10", 1.0)
    close_pos = col(f"{prefix}_close_pos_in_range", 0.5).clip(0, 1)

    out[f"{prefix}_body_atr_ratio"] = body_abs / (atr_safe + EPS)
    out[f"{prefix}_signed_body_atr_ratio"] = body / (atr_safe + EPS)
    out[f"{prefix}_range_atr_ratio"] = col(f"{prefix}_range", 0.0) / (atr_safe + EPS)
    out[f"{prefix}_candle_efficiency"] = body_abs / (range_safe + EPS)
    out[f"{prefix}_signed_candle_efficiency"] = np.sign(body) * out[f"{prefix}_candle_efficiency"]
    out[f"{prefix}_range_expansion_vs_mean10"] = col(f"{prefix}_range", 0.0) / (range_mean10 + EPS)
    out[f"{prefix}_volume_expansion_vs_mean10"] = volume_vs_mean

    out[f"{prefix}_bull_displacement"] = (
        (body > 0) &
        (out[f"{prefix}_body_atr_ratio"] >= DISPLACEMENT_BODY_ATR_THRESHOLD) &
        (out[f"{prefix}_candle_efficiency"] >= DISPLACEMENT_EFFICIENCY_THRESHOLD) &
        (close_pos >= 0.60)
    ).astype(int)

    out[f"{prefix}_bear_displacement"] = (
        (body < 0) &
        (out[f"{prefix}_body_atr_ratio"] >= DISPLACEMENT_BODY_ATR_THRESHOLD) &
        (out[f"{prefix}_candle_efficiency"] >= DISPLACEMENT_EFFICIENCY_THRESHOLD) &
        (close_pos <= 0.40)
    ).astype(int)

    out[f"{prefix}_expansion_bar"] = (
        (out[f"{prefix}_range_expansion_vs_mean10"] >= EXPANSION_RANGE_MEAN_THRESHOLD) &
        (out[f"{prefix}_body_atr_ratio"] >= 0.80)
    ).astype(int)

    out[f"{prefix}_bull_expansion_bar"] = ((out[f"{prefix}_expansion_bar"] == 1) & (body > 0)).astype(int)
    out[f"{prefix}_bear_expansion_bar"] = ((out[f"{prefix}_expansion_bar"] == 1) & (body < 0)).astype(int)

    # Multi-candle impulse: signed body pressure normalized by current volatility.
    for win in [3, 5, 10]:
        out[f"{prefix}_impulse_body_atr_{win}"] = body.rolling(win, min_periods=max(2, win // 2)).sum() / (atr_safe + EPS)
        out[f"{prefix}_bull_displacement_count_{win}"] = out[f"{prefix}_bull_displacement"].rolling(win, min_periods=1).sum()
        out[f"{prefix}_bear_displacement_count_{win}"] = out[f"{prefix}_bear_displacement"].rolling(win, min_periods=1).sum()
        out[f"{prefix}_expansion_count_{win}"] = out[f"{prefix}_expansion_bar"].rolling(win, min_periods=1).sum()

    out[f"{prefix}_displacement_pressure"] = (
        out[f"{prefix}_signed_body_atr_ratio"] *
        out[f"{prefix}_candle_efficiency"] *
        out[f"{prefix}_range_expansion_vs_mean10"].clip(lower=0, upper=5)
    )

    out[f"{prefix}_displacement_pressure_5"] = out[f"{prefix}_displacement_pressure"].rolling(5, min_periods=1).sum()
    out[f"{prefix}_displacement_pressure_10"] = out[f"{prefix}_displacement_pressure"].rolling(10, min_periods=1).sum()

    return out


# ========== MARKET COMPRESSION -> EXPANSION DETECTION ==========
def add_compression_expansion_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Adds compression, squeeze, and breakout-from-compression features.

    Purpose:
      - lets the model recognize coiled/low-energy markets
      - identifies when expansion happens immediately after compression
      - uses only rolling historical windows
    """
    out = df.copy()

    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return out[name].replace([np.inf, -np.inf], np.nan).fillna(default)
        return pd.Series(default, index=out.index, dtype="float64")

    rng = col(f"{prefix}_range", 0.0)
    close = out["close"] if "close" in out.columns else pd.Series(np.nan, index=out.index)
    high = out["high"] if "high" in out.columns else pd.Series(np.nan, index=out.index)
    low = out["low"] if "low" in out.columns else pd.Series(np.nan, index=out.index)

    range_short = rng.rolling(COMPRESSION_SHORT_WINDOW, min_periods=5).mean()
    range_long = rng.rolling(COMPRESSION_LONG_WINDOW, min_periods=20).mean()
    out[f"{prefix}_range_compression_ratio_20_100"] = range_short / (range_long + EPS)

    if f"{prefix}_atr_pct14" in out.columns:
        atr_pct = col(f"{prefix}_atr_pct14", 0.0)
    else:
        atr_pct = rng / (close + EPS)

    atr_short = atr_pct.rolling(COMPRESSION_SHORT_WINDOW, min_periods=5).mean()
    atr_long = atr_pct.rolling(COMPRESSION_LONG_WINDOW, min_periods=20).mean()
    out[f"{prefix}_atr_compression_ratio_20_100"] = atr_short / (atr_long + EPS)

    roll_high = high.rolling(COMPRESSION_SHORT_WINDOW, min_periods=5).max()
    roll_low = low.rolling(COMPRESSION_SHORT_WINDOW, min_periods=5).min()
    out[f"{prefix}_box_range_pct_20"] = (roll_high - roll_low) / (close + EPS)
    out[f"{prefix}_box_position_20"] = ((close - roll_low) / ((roll_high - roll_low) + EPS)).clip(0, 1)

    # Bollinger-style width using existing price only; no external indicator dependency.
    ma20 = close.rolling(20, min_periods=10).mean()
    std20 = close.rolling(20, min_periods=10).std()
    bb_width = (4.0 * std20) / (ma20 + EPS)
    out[f"{prefix}_bb_width_20"] = bb_width
    out[f"{prefix}_bb_width_compression_ratio"] = bb_width / (bb_width.rolling(100, min_periods=20).mean() + EPS)

    out[f"{prefix}_is_compressed"] = (
        (out[f"{prefix}_range_compression_ratio_20_100"] <= COMPRESSION_THRESHOLD) |
        (out[f"{prefix}_atr_compression_ratio_20_100"] <= COMPRESSION_THRESHOLD) |
        (out[f"{prefix}_bb_width_compression_ratio"] <= COMPRESSION_THRESHOLD)
    ).astype(int)

    out[f"{prefix}_compression_count_20"] = out[f"{prefix}_is_compressed"].rolling(20, min_periods=1).sum()
    out[f"{prefix}_compression_count_50"] = out[f"{prefix}_is_compressed"].rolling(50, min_periods=1).sum()

    expansion = col(f"{prefix}_expansion_bar", 0.0)
    prev_compression = out[f"{prefix}_is_compressed"].shift(1).rolling(20, min_periods=1).max().fillna(0)
    out[f"{prefix}_breakout_after_compression"] = ((expansion == 1) & (prev_compression == 1)).astype(int)

    bull_disp = col(f"{prefix}_bull_displacement", 0.0)
    bear_disp = col(f"{prefix}_bear_displacement", 0.0)
    out[f"{prefix}_bull_breakout_after_compression"] = ((out[f"{prefix}_breakout_after_compression"] == 1) & (bull_disp == 1)).astype(int)
    out[f"{prefix}_bear_breakout_after_compression"] = ((out[f"{prefix}_breakout_after_compression"] == 1) & (bear_disp == 1)).astype(int)

    return out


# ========== MULTI-TIMEFRAME SEQUENCE AWARENESS ==========
def add_multitimeframe_sequence_awareness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling memory of structure, sweeps, displacement, and compression.

    Purpose:
      - converts isolated events into behavior flow
      - helps the model learn persistence, repeated pressure, and failed moves
      - no future leakage: all rolling windows use current/past rows only
    """
    out = df.copy()

    def c(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return out[name].replace([np.inf, -np.inf], np.nan).fillna(default)
        return pd.Series(default, index=out.index, dtype="float64")

    for win in SEQUENCE_WINDOWS:
        # Structure memory
        out[f"seq_bos_up_count_{win}"] = (c("1m_bos_up") + c("5m_bos_up") + c("15m_bos_up")).rolling(win, min_periods=1).sum()
        out[f"seq_bos_down_count_{win}"] = (c("1m_bos_down") + c("5m_bos_down") + c("15m_bos_down")).rolling(win, min_periods=1).sum()
        out[f"seq_choch_up_count_{win}"] = (c("1m_choch_up") + c("5m_choch_up") + c("15m_choch_up")).rolling(win, min_periods=1).sum()
        out[f"seq_choch_down_count_{win}"] = (c("1m_choch_down") + c("5m_choch_down") + c("15m_choch_down")).rolling(win, min_periods=1).sum()
        out[f"seq_structure_break_bias_{win}"] = (
            out[f"seq_bos_up_count_{win}"] + out[f"seq_choch_up_count_{win}"] -
            out[f"seq_bos_down_count_{win}"] - out[f"seq_choch_down_count_{win}"]
        )
        out[f"seq_structure_instability_{win}"] = (
            out[f"seq_choch_up_count_{win}"] + out[f"seq_choch_down_count_{win}"]
        )

        # Liquidity memory
        out[f"seq_sweep_high_count_{win}"] = (c("1m_sweep_high") + c("5m_sweep_high") + c("15m_sweep_high")).rolling(win, min_periods=1).sum()
        out[f"seq_sweep_low_count_{win}"] = (c("1m_sweep_low") + c("5m_sweep_low") + c("15m_sweep_low")).rolling(win, min_periods=1).sum()
        out[f"seq_reject_high_count_{win}"] = (c("1m_sweep_reject_high") + c("5m_sweep_reject_high") + c("15m_sweep_reject_high")).rolling(win, min_periods=1).sum()
        out[f"seq_reject_low_count_{win}"] = (c("1m_sweep_reject_low") + c("5m_sweep_reject_low") + c("15m_sweep_reject_low")).rolling(win, min_periods=1).sum()
        out[f"seq_sweep_pressure_bias_{win}"] = out[f"seq_sweep_high_count_{win}"] - out[f"seq_sweep_low_count_{win}"]
        out[f"seq_rejection_reversal_bias_{win}"] = out[f"seq_reject_low_count_{win}"] - out[f"seq_reject_high_count_{win}"]
        out[f"seq_both_sides_swept_{win}"] = ((out[f"seq_sweep_high_count_{win}"] > 0) & (out[f"seq_sweep_low_count_{win}"] > 0)).astype(int)

        # Displacement memory
        out[f"seq_bull_displacement_count_{win}"] = (c("1m_bull_displacement") + c("5m_bull_displacement") + c("15m_bull_displacement")).rolling(win, min_periods=1).sum()
        out[f"seq_bear_displacement_count_{win}"] = (c("1m_bear_displacement") + c("5m_bear_displacement") + c("15m_bear_displacement")).rolling(win, min_periods=1).sum()
        out[f"seq_displacement_bias_{win}"] = out[f"seq_bull_displacement_count_{win}"] - out[f"seq_bear_displacement_count_{win}"]
        out[f"seq_expansion_count_{win}"] = (c("1m_expansion_bar") + c("5m_expansion_bar") + c("15m_expansion_bar")).rolling(win, min_periods=1).sum()

        # Compression memory
        out[f"seq_compression_count_{win}"] = (c("1m_is_compressed") + c("5m_is_compressed") + c("15m_is_compressed")).rolling(win, min_periods=1).sum()
        out[f"seq_breakout_after_compression_count_{win}"] = (
            c("1m_breakout_after_compression") + c("5m_breakout_after_compression") + c("15m_breakout_after_compression")
        ).rolling(win, min_periods=1).sum()

    out["seq_market_pressure_bias_20"] = (
        c("seq_structure_break_bias_20") +
        c("seq_displacement_bias_20") +
        c("seq_rejection_reversal_bias_20")
    )

    out["seq_market_activity_score_20"] = (
        c("seq_expansion_count_20") +
        c("seq_sweep_high_count_20") + c("seq_sweep_low_count_20") +
        c("seq_bos_up_count_20") + c("seq_bos_down_count_20")
    )

    return out


# ========== DO NOT TRADE INTELLIGENCE ==========
def add_do_not_trade_intelligence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds uncertainty/chop/conflict features. These are model inputs, not hard filters.

    Purpose:
      - lets LightGBM learn when market conditions are low-quality
      - improves NO_TRADE recognition without removing existing trade labels
    """
    out = df.copy()

    def c(name: str, default: float = 0.0) -> pd.Series:
        if name in out.columns:
            return out[name].replace([np.inf, -np.inf], np.nan).fillna(default)
        return pd.Series(default, index=out.index, dtype="float64")

    bull_context = (
        c("structure_bull_context_score") +
        c("ob_bull_context_score") +
        c("fvg_bull_context_score") +
        c("liquidity_interaction_bull_score") +
        c("seq_bull_displacement_count_20") +
        c("htf_liquidity_reject_low_score") +
        c("session_liquidity_reject_low_score")
    )
    bear_context = (
        c("structure_bear_context_score") +
        c("ob_bear_context_score") +
        c("fvg_bear_context_score") +
        c("liquidity_interaction_bear_score") +
        c("seq_bear_displacement_count_20") +
        c("htf_liquidity_reject_high_score") +
        c("session_liquidity_reject_high_score")
    )

    out["dnt_bull_context_total"] = bull_context
    out["dnt_bear_context_total"] = bear_context
    out["dnt_context_abs_diff"] = (bull_context - bear_context).abs()
    out["dnt_context_conflict_score"] = (np.minimum(bull_context, bear_context) / (np.maximum(bull_context, bear_context) + EPS)).clip(0, 1)

    out["dnt_structure_flip_count_20"] = c("seq_structure_instability_20")
    out["dnt_both_sides_swept_20"] = c("seq_both_sides_swept_20")
    out["dnt_both_sides_swept_50"] = c("seq_both_sides_swept_50")
    out["dnt_weak_or_mixed_trend"] = c("mixed_or_weak_trend")
    out["dnt_choppy_alignment"] = c("adx_choppy_alignment_score")

    # Weak displacement while signals exist = often poor continuation quality.
    out["dnt_weak_displacement_environment"] = (
        (c("seq_bull_displacement_count_20") + c("seq_bear_displacement_count_20") <= 1) &
        (c("seq_market_activity_score_20") > 3)
    ).astype(int)

    # Compression with no breakout = wait condition, not necessarily directional edge.
    out["dnt_unresolved_compression"] = (
        (c("seq_compression_count_20") >= 10) &
        (c("seq_breakout_after_compression_count_20") == 0)
    ).astype(int)

    out["dnt_conflicting_structure_liquidity"] = (
        ((c("structure_bull_context_score") > c("structure_bear_context_score")) & (c("advanced_liquidity_pressure_bias") < 0)) |
        ((c("structure_bear_context_score") > c("structure_bull_context_score")) & (c("advanced_liquidity_pressure_bias") > 0))
    ).astype(int)

    out["dnt_uncertainty_score"] = (
        out["dnt_context_conflict_score"] * 3.0 +
        (out["dnt_structure_flip_count_20"] / 5.0).clip(0, 3) +
        out["dnt_both_sides_swept_20"] * 1.5 +
        out["dnt_weak_or_mixed_trend"] * 1.0 +
        (out["dnt_choppy_alignment"] / 3.0).clip(0, 1.5) +
        out["dnt_weak_displacement_environment"] * 1.0 +
        out["dnt_unresolved_compression"] * 1.5 +
        out["dnt_conflicting_structure_liquidity"] * 1.5
    )

    out["dnt_high_uncertainty_regime"] = (out["dnt_uncertainty_score"] >= 4.0).astype(int)
    out["dnt_low_quality_trade_environment"] = (
        (out["dnt_high_uncertainty_regime"] == 1) |
        ((out["dnt_context_abs_diff"] <= DO_NOT_TRADE_CONFLICT_THRESHOLD) & (out["dnt_context_conflict_score"] > 0.55))
    ).astype(int)

    return out




def get_previous_completed_htf_level(symbol: str, timeframe: int, label: str) -> Optional[dict]:
    """
    Fetch the previous completed D1/W1/MN1 candle directly from MT5.
    This preserves the meaning of prev_daily/weekly/monthly levels without
    requiring 60k+ M1 candles in the live feature builder.
    """
    try:
        rates = get_latest_rates(symbol, timeframe, count=3).sort_values("time")
        if len(rates) < 2:
            return None
        prev = rates.iloc[-2]  # last row is current forming HTF candle; -2 is completed
        high = float(prev["high"])
        low = float(prev["low"])
        if not np.isfinite(high) or not np.isfinite(low) or high <= 0 or low <= 0:
            return None
        return {
            "label": label,
            "time": prev["time"],
            "high": high,
            "low": low,
            "mid": (high + low) / 2.0,
            "range": max(high - low, np.nan),
        }
    except Exception as exc:
        log_message(f"Direct HTF fetch failed for {label}: {exc}")
        return None


def fetch_direct_htf_levels(symbol: str) -> Optional[dict]:
    levels = {
        "daily": get_previous_completed_htf_level(symbol, mt5.TIMEFRAME_D1, "daily"),
        "weekly": get_previous_completed_htf_level(symbol, mt5.TIMEFRAME_W1, "weekly"),
        "monthly": get_previous_completed_htf_level(symbol, mt5.TIMEFRAME_MN1, "monthly"),
    }
    if any(v is None for v in levels.values()):
        return None
    return levels


def add_htf_liquidity_level_features_live(df: pd.DataFrame, raw_1m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Live-optimized HTF liquidity builder.

    Feature names and formulas are kept compatible with training. The only
    change is where prev_daily/weekly/monthly high-low levels come from:
      - Direct MT5 D1/W1/MN1 candles for the previous completed HTF levels.
      - Recent M1 candles are still used for sweep/reject/reclaim detection.

    If direct HTF fetch is disabled or unavailable, it falls back to the old
    M1-derived method.
    """
    if not USE_DIRECT_HTF_LEVELS:
        return add_htf_liquidity_level_features(df, raw_1m)

    levels = fetch_direct_htf_levels(symbol)
    if levels is None:
        if DIRECT_HTF_FALLBACK_TO_M1:
            log_message("Direct HTF levels unavailable; falling back to old M1-derived HTF logic.")
            return add_htf_liquidity_level_features(df, raw_1m)
        raise RuntimeError("Direct HTF levels unavailable and fallback is disabled.")

    out = df.copy()
    raw = raw_1m.reindex(out.index).copy()
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("raw_1m must have a DatetimeIndex before adding HTF liquidity features.")

    high = raw["high"].astype(float)
    low = raw["low"].astype(float)
    close = raw["close"].astype(float)

    if "1m_atr14" in out.columns:
        near_distance = (out["1m_atr14"].abs() * HTF_LIQUIDITY_NEAR_ATR_MULTIPLIER).fillna(
            close * HTF_LIQUIDITY_NEAR_PCT_FALLBACK
        )
    else:
        near_distance = close * HTF_LIQUIDITY_NEAR_PCT_FALLBACK

    def bars_since(signal: pd.Series) -> pd.Series:
        last_idx = pd.Series(np.where(signal.astype(bool), np.arange(len(out)), np.nan), index=out.index).ffill()
        bar_idx = pd.Series(np.arange(len(out)), index=out.index)
        return (bar_idx - last_idx).fillna(HTF_LIQUIDITY_NO_SWEEP_AGE)

    for name in ("daily", "weekly", "monthly"):
        lv = levels[name]
        prev_high = pd.Series(float(lv["high"]), index=out.index, dtype="float64")
        prev_low = pd.Series(float(lv["low"]), index=out.index, dtype="float64")
        prev_mid = pd.Series(float(lv["mid"]), index=out.index, dtype="float64")
        prev_range = (prev_high - prev_low).replace(0, np.nan)

        out[f"prev_{name}_high"] = prev_high
        out[f"prev_{name}_low"] = prev_low
        out[f"prev_{name}_mid"] = prev_mid
        out[f"prev_{name}_range"] = prev_range

        out[f"dist_to_prev_{name}_high"] = (prev_high - close) / (close + EPS)
        out[f"dist_to_prev_{name}_low"] = (close - prev_low) / (close + EPS)
        out[f"abs_dist_to_prev_{name}_high"] = (prev_high - close).abs() / (close + EPS)
        out[f"abs_dist_to_prev_{name}_low"] = (close - prev_low).abs() / (close + EPS)

        out[f"near_prev_{name}_high"] = ((prev_high - close).abs() <= near_distance).astype(int)
        out[f"near_prev_{name}_low"] = ((close - prev_low).abs() <= near_distance).astype(int)

        out[f"closed_above_prev_{name}_high"] = (close > prev_high).astype(int)
        out[f"closed_below_prev_{name}_low"] = (close < prev_low).astype(int)
        out[f"inside_prev_{name}_range"] = ((close <= prev_high) & (close >= prev_low)).astype(int)

        out[f"swept_prev_{name}_high"] = (high > prev_high).astype(int)
        out[f"swept_prev_{name}_low"] = (low < prev_low).astype(int)
        out[f"reject_prev_{name}_high"] = ((out[f"swept_prev_{name}_high"] == 1) & (close < prev_high)).astype(int)
        out[f"reject_prev_{name}_low"] = ((out[f"swept_prev_{name}_low"] == 1) & (close > prev_low)).astype(int)
        out[f"reclaim_prev_{name}_high"] = ((out[f"swept_prev_{name}_high"] == 1) & (close > prev_high)).astype(int)
        out[f"reclaim_prev_{name}_low"] = ((out[f"swept_prev_{name}_low"] == 1) & (close < prev_low)).astype(int)

        out[f"prev_{name}_range_position"] = ((close - prev_low) / (prev_range + EPS)).clip(lower=-1, upper=2)
        out[f"above_prev_{name}_mid"] = (close > prev_mid).astype(int)
        out[f"below_prev_{name}_mid"] = (close < prev_mid).astype(int)
        out[f"prev_{name}_premium_discount"] = np.where(close > prev_mid, 1, np.where(close < prev_mid, -1, 0))

        out[f"prev_{name}_high_sweep_depth"] = np.where(out[f"swept_prev_{name}_high"] == 1, (high - prev_high).clip(lower=0), 0.0)
        out[f"prev_{name}_low_sweep_depth"] = np.where(out[f"swept_prev_{name}_low"] == 1, (prev_low - low).clip(lower=0), 0.0)
        out[f"prev_{name}_high_sweep_depth_pct"] = out[f"prev_{name}_high_sweep_depth"] / (close + EPS)
        out[f"prev_{name}_low_sweep_depth_pct"] = out[f"prev_{name}_low_sweep_depth"] / (close + EPS)

        out[f"bars_since_swept_prev_{name}_high"] = bars_since(out[f"swept_prev_{name}_high"])
        out[f"bars_since_swept_prev_{name}_low"] = bars_since(out[f"swept_prev_{name}_low"])
        out[f"bars_since_reject_prev_{name}_high"] = bars_since(out[f"reject_prev_{name}_high"])
        out[f"bars_since_reject_prev_{name}_low"] = bars_since(out[f"reject_prev_{name}_low"])

        out[f"recent_swept_prev_{name}_high"] = (out[f"bars_since_swept_prev_{name}_high"] <= HTF_LIQUIDITY_RECENT_WINDOW).astype(int)
        out[f"recent_swept_prev_{name}_low"] = (out[f"bars_since_swept_prev_{name}_low"] <= HTF_LIQUIDITY_RECENT_WINDOW).astype(int)
        out[f"recent_reject_prev_{name}_high"] = (out[f"bars_since_reject_prev_{name}_high"] <= HTF_LIQUIDITY_RECENT_WINDOW).astype(int)
        out[f"recent_reject_prev_{name}_low"] = (out[f"bars_since_reject_prev_{name}_low"] <= HTF_LIQUIDITY_RECENT_WINDOW).astype(int)

    out["htf_liquidity_near_high_score"] = out[["near_prev_daily_high", "near_prev_weekly_high", "near_prev_monthly_high"]].sum(axis=1)
    out["htf_liquidity_near_low_score"] = out[["near_prev_daily_low", "near_prev_weekly_low", "near_prev_monthly_low"]].sum(axis=1)
    out["htf_liquidity_sweep_high_score"] = out[["recent_swept_prev_daily_high", "recent_swept_prev_weekly_high", "recent_swept_prev_monthly_high"]].sum(axis=1)
    out["htf_liquidity_sweep_low_score"] = out[["recent_swept_prev_daily_low", "recent_swept_prev_weekly_low", "recent_swept_prev_monthly_low"]].sum(axis=1)
    out["htf_liquidity_reject_high_score"] = out[["recent_reject_prev_daily_high", "recent_reject_prev_weekly_high", "recent_reject_prev_monthly_high"]].sum(axis=1)
    out["htf_liquidity_reject_low_score"] = out[["recent_reject_prev_daily_low", "recent_reject_prev_weekly_low", "recent_reject_prev_monthly_low"]].sum(axis=1)
    out["htf_liquidity_reversal_bias"] = out["htf_liquidity_reject_low_score"] - out["htf_liquidity_reject_high_score"]
    out["htf_liquidity_continuation_bias"] = out["htf_liquidity_sweep_high_score"] - out["htf_liquidity_sweep_low_score"]
    out["daily_weekly_liquidity_confluence_high"] = ((out["near_prev_daily_high"] == 1) & (out["near_prev_weekly_high"] == 1)).astype(int)
    out["daily_weekly_liquidity_confluence_low"] = ((out["near_prev_daily_low"] == 1) & (out["near_prev_weekly_low"] == 1)).astype(int)

    out["direct_htf_levels_used"] = 1
    return out

def fetch_raw_timeframes(symbol: str, m1_count: int = LIVE_M1_HISTORY_BARS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rates_1m = get_latest_rates(symbol, mt5.TIMEFRAME_M1, count=m1_count)
    rates_5m = get_latest_rates(symbol, mt5.TIMEFRAME_M5, count=max(500, math.ceil(m1_count / 5) + 240))
    rates_15m = get_latest_rates(symbol, mt5.TIMEFRAME_M15, count=max(500, math.ceil(m1_count / 15) + 240))
    return mt5_rates_to_price_df(rates_1m).sort_index(), mt5_rates_to_price_df(rates_5m).sort_index(), mt5_rates_to_price_df(rates_15m).sort_index()


def prefixed_features_only(df_source: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df_source[[c for c in df_source.columns if c.startswith(f"{prefix}_")]].copy()


def make_feature_frame(symbol: str) -> pd.DataFrame:
    df_1m_raw, df_5m_raw, df_15m_raw = fetch_raw_timeframes(symbol, m1_count=LIVE_M1_HISTORY_BARS)
    df_1m = create_features(df_1m_raw, "1m", use_rsi=True, use_atr=True, use_adx=True)
    df_5m = create_features(df_5m_raw, "5m", use_rsi=True, use_atr=False, use_adx=True)
    df_15m = create_features(df_15m_raw, "15m", use_rsi=False, use_atr=False, use_adx=True)

    # Behavior-flow upgrade features must be calculated on each native timeframe
    # before safe backward merging, matching the upgraded dataset builder.
    df_1m = add_displacement_expansion_features(df_1m, "1m")
    df_5m = add_displacement_expansion_features(df_5m, "5m")
    df_15m = add_displacement_expansion_features(df_15m, "15m")

    df_1m = add_compression_expansion_features(df_1m, "1m")
    df_5m = add_compression_expansion_features(df_5m, "5m")
    df_15m = add_compression_expansion_features(df_15m, "15m")

    # Match dataset builder feature order: displacement/compression -> sweep -> structure -> OB -> FVG -> entry-zone boundaries.
    df_1m = add_liquidity_sweep_features(df_1m, "1m")
    df_5m = add_liquidity_sweep_features(df_5m, "5m")
    df_15m = add_liquidity_sweep_features(df_15m, "15m")

    df_1m = add_fractal_structure_features(df_1m, "1m")
    df_5m = add_fractal_structure_features(df_5m, "5m")
    df_15m = add_fractal_structure_features(df_15m, "15m")

    df_1m = add_order_block_features(df_1m, "1m")
    df_5m = add_order_block_features(df_5m, "5m")
    df_15m = add_order_block_features(df_15m, "15m")

    df_1m = add_fvg_features(df_1m, "1m")
    df_5m = add_fvg_features(df_5m, "5m")
    df_15m = add_fvg_features(df_15m, "15m")

    df_1m = add_entry_zone_price_features(df_1m, "1m")
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

# ========== LIQUIDITY SWEEP / REJECTION ALIGNMENT FEATURES ==========
    cross["sweep_high_context_score"] = (
        df["1m_recent_sweep_high"] +
        df["5m_recent_sweep_high"] +
        df["15m_recent_sweep_high"]
    )
    
    cross["sweep_low_context_score"] = (
        df["1m_recent_sweep_low"] +
        df["5m_recent_sweep_low"] +
        df["15m_recent_sweep_low"]
    )
    
    cross["sweep_reject_high_context_score"] = (
        df["1m_recent_sweep_reject_high"] +
        df["5m_recent_sweep_reject_high"] +
        df["15m_recent_sweep_reject_high"]
    )
    
    cross["sweep_reject_low_context_score"] = (
        df["1m_recent_sweep_reject_low"] +
        df["5m_recent_sweep_reject_low"] +
        df["15m_recent_sweep_reject_low"]
    )
    
    cross["strong_sweep_reject_high_context_score"] = (
        df["1m_recent_strong_sweep_reject_high"] +
        df["5m_recent_strong_sweep_reject_high"] +
        df["15m_recent_strong_sweep_reject_high"]
    )
    
    cross["strong_sweep_reject_low_context_score"] = (
        df["1m_recent_strong_sweep_reject_low"] +
        df["5m_recent_strong_sweep_reject_low"] +
        df["15m_recent_strong_sweep_reject_low"]
    )
    
    cross["sweep_reversal_context_score"] = (
        df["1m_sweep_reversal_bias"] +
        df["5m_sweep_reversal_bias"] +
        df["15m_sweep_reversal_bias"]
    )
    
    cross["strong_sweep_reversal_context_score"] = (
        df["1m_strong_sweep_reversal_bias"] +
        df["5m_strong_sweep_reversal_bias"] +
        df["15m_strong_sweep_reversal_bias"]
    )
    
    cross["sweep_continuation_context_score"] = (
        df["1m_sweep_continuation_bias"] +
        df["5m_sweep_continuation_bias"] +
        df["15m_sweep_continuation_bias"]
    )
    
    cross["sweep_high_atr_strength_sum"] = (
        df["1m_sweep_high_atr_strength"] +
        df["5m_sweep_high_atr_strength"] +
        df["15m_sweep_high_atr_strength"]
    )
    
    cross["sweep_low_atr_strength_sum"] = (
        df["1m_sweep_low_atr_strength"] +
        df["5m_sweep_low_atr_strength"] +
        df["15m_sweep_low_atr_strength"]
    )
    
    cross["sweep_high_wick_rejection_strength_sum"] = (
        df["1m_sweep_high_wick_rejection_strength"] +
        df["5m_sweep_high_wick_rejection_strength"] +
        df["15m_sweep_high_wick_rejection_strength"]
    )
    
    cross["sweep_low_wick_rejection_strength_sum"] = (
        df["1m_sweep_low_wick_rejection_strength"] +
        df["5m_sweep_low_wick_rejection_strength"] +
        df["15m_sweep_low_wick_rejection_strength"]
    )
    
    cross["htf_sweep_high_ltf_reject"] = (
        ((df["15m_recent_sweep_high"] == 1) | (df["5m_recent_sweep_high"] == 1)) &
        (df["1m_recent_sweep_reject_high"] == 1)
    ).astype(int)
    
    cross["htf_sweep_low_ltf_reject"] = (
        ((df["15m_recent_sweep_low"] == 1) | (df["5m_recent_sweep_low"] == 1)) &
        (df["1m_recent_sweep_reject_low"] == 1)
    ).astype(int)
    
    cross["htf_sweep_high_ltf_strong_reject"] = (
        ((df["15m_recent_sweep_high"] == 1) | (df["5m_recent_sweep_high"] == 1)) &
        (df["1m_recent_strong_sweep_reject_high"] == 1)
    ).astype(int)
    
    cross["htf_sweep_low_ltf_strong_reject"] = (
        ((df["15m_recent_sweep_low"] == 1) | (df["5m_recent_sweep_low"] == 1)) &
        (df["1m_recent_strong_sweep_reject_low"] == 1)
    ).astype(int)
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

    # Match dataset builder context/session/liquidity feature order.
    out = add_must_have_context_features(out, df_1m_raw)
    out = add_htf_liquidity_level_features_live(out, df_1m_raw, symbol)
    out = add_session_liquidity_features(out, df_1m_raw)
    out = add_advanced_liquidity_interaction_features(out)

    # Final behavior-flow layers that need merged multi-timeframe/context columns.
    out = add_multitimeframe_sequence_awareness_features(out)
    out = add_do_not_trade_intelligence_features(out)

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
        "prev_daily_high", "prev_daily_low", "prev_weekly_high", "prev_weekly_low", "prev_monthly_high", "prev_monthly_low",
        "dist_to_prev_daily_high", "dist_to_prev_daily_low", "dist_to_prev_weekly_high", "dist_to_prev_weekly_low",
        "near_prev_daily_high", "near_prev_daily_low", "near_prev_weekly_high", "near_prev_weekly_low", "near_prev_monthly_high", "near_prev_monthly_low",
        "swept_prev_daily_high", "swept_prev_daily_low", "reject_prev_daily_high", "reject_prev_daily_low",
        "htf_liquidity_near_high_score", "htf_liquidity_near_low_score", "htf_liquidity_sweep_high_score", "htf_liquidity_sweep_low_score", "htf_liquidity_reject_high_score", "htf_liquidity_reject_low_score", "htf_liquidity_reversal_bias",
        "prev_daily_range_position", "prev_weekly_range_position", "prev_monthly_range_position",
        "session_high_near_any", "session_low_near_any",
        "session_high_reject_recent_any", "session_low_reject_recent_any",
        "session_high_bearish_structure", "session_low_bullish_structure",
        "session_high_reject_bearish_structure", "session_low_reject_bullish_structure",
        "liquidity_interaction_reversal_bias", "killzone_liquidity_reversal_bias",
        "liquidity_interaction_bull_score", "liquidity_interaction_bear_score", "liquidity_interaction_score_diff",
        "session_sweep_decay_bias", "session_reject_decay_bias",
        "weighted_session_reversal_strength_bias", "entry_zone_liquidity_fusion_bias",
        "advanced_liquidity_pressure_bias", "structure_bias_normalized",
        "session_high_proximity_max", "session_low_proximity_max",
        "session_rejection_strength_bias_cont",
        "15m_displacement_pressure", "5m_displacement_pressure", "1m_displacement_pressure",
        "15m_signed_body_atr_ratio", "5m_signed_body_atr_ratio", "1m_signed_body_atr_ratio",
        "15m_candle_efficiency", "5m_candle_efficiency", "1m_candle_efficiency",
        "seq_displacement_bias_50", "seq_structure_break_bias_50", "seq_sweep_pressure_bias_50",
        "seq_market_pressure_bias_20", "seq_market_activity_score_20",
        "seq_compression_count_50", "seq_breakout_after_compression_count_50",
        "dnt_uncertainty_score", "dnt_context_conflict_score", "dnt_context_abs_diff",
        "dnt_high_uncertainty_regime", "dnt_low_quality_trade_environment",
        "dnt_weak_displacement_environment", "dnt_unresolved_compression",
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


def safe_debug_value(debug: dict, key: str, default: float = 0.0) -> float:
    value = debug.get(key, default)
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def evaluate_context_strength(signal: Signal) -> dict:
    """
    Smart execution layer.

    The model still decides the class, but this checks whether the current
    market context supports that class before sending orders.
    """
    debug = getattr(signal, "debug", {}) or {}

    # Prefer the advanced v3 features when available.
    advanced_bull = safe_debug_value(debug, "liquidity_interaction_bull_score")
    advanced_bear = safe_debug_value(debug, "liquidity_interaction_bear_score")
    advanced_diff = safe_debug_value(debug, "liquidity_interaction_score_diff")
    advanced_pressure = safe_debug_value(debug, "advanced_liquidity_pressure_bias")
    weighted_reversal = safe_debug_value(debug, "weighted_session_reversal_strength_bias")
    entry_fusion = safe_debug_value(debug, "entry_zone_liquidity_fusion_bias")
    structure_norm = safe_debug_value(debug, "structure_bias_normalized")

    # Fallback context from existing raw scores.
    structure_bull = safe_debug_value(debug, "structure_bull_context_score")
    structure_bear = safe_debug_value(debug, "structure_bear_context_score")
    ob_bull = safe_debug_value(debug, "ob_bull_context_score")
    ob_bear = safe_debug_value(debug, "ob_bear_context_score")
    fvg_bull = safe_debug_value(debug, "fvg_bull_context_score")
    fvg_bear = safe_debug_value(debug, "fvg_bear_context_score")
    htf_reversal = safe_debug_value(debug, "htf_liquidity_reversal_bias")
    session_reject_decay = safe_debug_value(debug, "session_reject_decay_bias")
    session_sweep_decay = safe_debug_value(debug, "session_sweep_decay_bias")

    bull_strength = (
        structure_bull
        + ob_bull
        + fvg_bull
        + max(htf_reversal, 0)
        + max(advanced_diff, 0)
        + max(advanced_pressure, 0)
        + max(weighted_reversal, 0)
        + max(entry_fusion, 0)
        + max(structure_norm, 0)
        + max(session_reject_decay, 0)
        + max(session_sweep_decay, 0)
        + advanced_bull
    )

    bear_strength = (
        structure_bear
        + ob_bear
        + fvg_bear
        + max(-htf_reversal, 0)
        + max(-advanced_diff, 0)
        + max(-advanced_pressure, 0)
        + max(-weighted_reversal, 0)
        + max(-entry_fusion, 0)
        + max(-structure_norm, 0)
        + max(-session_reject_decay, 0)
        + max(-session_sweep_decay, 0)
        + advanced_bear
    )

    dnt_uncertainty = safe_debug_value(debug, "dnt_uncertainty_score")
    dnt_low_quality = safe_debug_value(debug, "dnt_low_quality_trade_environment")
    dnt_conflict = safe_debug_value(debug, "dnt_context_conflict_score")

    direction = signal_direction(signal.predicted_class)
    strong_bull = bull_strength >= CONTEXT_STRENGTH_MIN_SCORE
    strong_bear = bear_strength >= CONTEXT_STRENGTH_MIN_SCORE
    strong_for_signal = (direction == 1 and strong_bull) or (direction == -1 and strong_bear)

    # Behavior-flow DNT features do not override the model by themselves;
    # they raise caution only when the environment is genuinely messy.
    if dnt_low_quality >= 1 or dnt_uncertainty >= 4.0:
        strong_for_signal = False

    if strong_for_signal:
        threshold = STRONG_CONTEXT_CONFIDENCE_THRESHOLD
    else:
        threshold = WEAK_CONTEXT_CONFIDENCE_THRESHOLD

    return {
        "bull_strength": bull_strength,
        "bear_strength": bear_strength,
        "strong_bull": strong_bull,
        "strong_bear": strong_bear,
        "strong_for_signal": strong_for_signal,
        "dynamic_threshold": threshold,
        "dnt_uncertainty_score": dnt_uncertainty,
        "dnt_context_conflict_score": dnt_conflict,
        "dnt_low_quality_trade_environment": dnt_low_quality,
    }


def should_skip_signal(signal: Signal) -> Optional[str]:
    if ONLY_TRADE_SIGNALS and signal.predicted_class == 0:
        return "neutral_signal"

    direction = signal_direction(signal.predicted_class)
    if direction == 1 and not ALLOW_LONG:
        return "long_disabled"
    if direction == -1 and not ALLOW_SHORT:
        return "short_disabled"

    ctx = evaluate_context_strength(signal)
    threshold = ctx["dynamic_threshold"]

    # Dynamic threshold from validation behavior:
    # strong context can use lower threshold, weak context must be much stricter.
    if signal.confidence < threshold:
        return f"low_confidence_dynamic({threshold:.2f})"

    # Optional hard block: prevents the live bot from taking high-confidence but
    # context-weak trades, which were the main source of overtrading.
    if REQUIRE_STRONG_CONTEXT_FOR_ENTRY and not ctx["strong_for_signal"]:
        if direction == 1:
            return "weak_bull_context"
        if direction == -1:
            return "weak_bear_context"

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
    print_table("BOT SETTINGS", [("symbol", SYMBOL), ("features", len(feature_list)), ("hold_bars", HOLD_BARS), ("confidence", CONFIDENCE_THRESHOLD), ("lot_size", LOT_SIZE), ("risk_reward", RISK_REWARD_RATIO), ("pending_wait", USE_PENDING_FOR_WAIT_ENTRIES), ("flip_enabled", CLOSE_OPPOSITE_ON_SIGNAL), ("enter_after_flip", ENTER_AFTER_FLIP), ("dry_run", DRY_RUN), ("m1_history", LIVE_M1_HISTORY_BARS), ("direct_htf", USE_DIRECT_HTF_LEVELS)])
    try:
        while True:
            loop_t0 = time.perf_counter()
            manage_timeouts(SYMBOL, TIMEFRAME, state)

            candle_t0 = time.perf_counter()
            recent_rates = get_latest_rates(SYMBOL, TIMEFRAME, count=5)
            latest_closed_bar_time = int(recent_rates.iloc[-2]["time"].timestamp())
            candle_check_seconds = time.perf_counter() - candle_t0

            if state.get("last_signal_bar_time") == latest_closed_bar_time:
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue

            predict_t0 = time.perf_counter()
            signal = predict_signal(model, feature_list, SYMBOL)
            predict_seconds = time.perf_counter() - predict_t0

            if signal.bar_time != latest_closed_bar_time:
                log_message(
                    f"Warning: cheap candle check time {latest_closed_bar_time} != "
                    f"feature row time {signal.bar_time}; using feature row time."
                )

            state["last_signal_bar_time"] = signal.bar_time
            log_message(
                f"Timing: candle_check={candle_check_seconds:.2f}s | "
                f"predict_signal={predict_seconds:.2f}s | "
                f"loop_before_print={time.perf_counter() - loop_t0:.2f}s"
            )
            print_signal_table(signal)
            context_strength = evaluate_context_strength(signal)
            print_table("SMART EXECUTION CONTEXT", [
                ("bull_strength", context_strength["bull_strength"]),
                ("bear_strength", context_strength["bear_strength"]),
                ("strong_bull", context_strength["strong_bull"]),
                ("strong_bear", context_strength["strong_bear"]),
                ("strong_for_signal", context_strength["strong_for_signal"]),
                ("dynamic_threshold", context_strength["dynamic_threshold"]),
                ("dnt_uncertainty", context_strength.get("dnt_uncertainty_score")),
                ("dnt_conflict", context_strength.get("dnt_context_conflict_score")),
                ("dnt_low_quality", context_strength.get("dnt_low_quality_trade_environment")),
            ])
            skip_reason = should_skip_signal(signal)
            if skip_reason:
                append_trade_log({"timestamp": datetime.now().isoformat(), "event": "signal_skipped", "bar_time": signal.bar_time, "predicted_class": signal.predicted_class, "confidence": signal.confidence, "reason": skip_reason, "context_strength": json.dumps(context_strength), "probabilities": json.dumps(signal.raw_probabilities), "debug": json.dumps(getattr(signal, "debug", {}))})
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

            strong_signal_for_flip = context_strength["strong_for_signal"] or not FLIP_ONLY_ON_STRONG_CONTEXT

            if CLOSE_OPPOSITE_ON_SIGNAL and strong_signal_for_flip:
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
