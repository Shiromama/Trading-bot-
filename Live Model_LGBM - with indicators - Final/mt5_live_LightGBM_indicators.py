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
HOLD_BARS = 60
MAGIC_NUMBER = 20260423
ORDER_COMMENT = "lgbm-live"
ALLOW_LONG = True
ALLOW_SHORT = True
ONE_POSITION_PER_SYMBOL = False
INVERT_SIGNALS = False   #Flip BUY ↔ SELL from model output kahlil's idea

# Position manager / selective profit-lock settings
MAX_TOTAL_POSITIONS = 10              # max total managed positions for this symbol/magic
MAX_SAME_SIDE_POSITIONS = 5         # max BUYs or max SELLs at the same time
CLOSE_PROFITABLE_OPPOSITE_ON_FLIP = True
CLOSE_OPPOSITE_MIN_CONFIDENCE = 0.60 # only profit-lock old side if new signal is strong
MIN_PROFIT_TO_CLOSE_OPPOSITE = 1.0  # close opposite winners once profit >= this USD
MARK_LEFTOVER_OPPOSITE_FOR_RECOVERY = True
RECOVERY_CLOSE_PROFIT_USD = 1.0     # close leftover old-side trade once it recovers to this profit
RECOVERY_MAX_BARS = 60               # after this many bars, recovery tag expires; SL/timeout still applies
ALLOW_NEW_ENTRY_AFTER_PROFIT_LOCK = True
ONLY_TRADE_SIGNALS = True
CONFIDENCE_THRESHOLD = 0.60
MAX_SPREAD_POINTS = None          # e.g. 500, or None to disable
DEVIATION = 50                    # max slippage in points
DRY_RUN = False                   # True = log signals only, no real orders
STRICT_FEATURE_MATCH = True       # safer: abort if live features don't match training features

# Risk / sizing
LOT_SIZE = 0.02
STOP_LOSS_ACCOUNT_FRACTION = 0.10
RISK_REWARD_RATIO = 2.0
USE_TAKE_PROFIT = True
MIN_BALANCE_TO_TRADE = 1.0

# MT5 login
# Safer approach: leave these blank and use an already logged-in MT5 terminal
LOGIN = None
PASSWORD = ""
SERVER = ""
TERMINAL_PATH = None

# Files
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "lgbm_model.pkl"
FEATURES_PATH = SCRIPT_DIR / "lgbm_features.pkl"
STATE_PATH = SCRIPT_DIR / "mt5_live_state.json"
LOG_PATH = SCRIPT_DIR / "mt5_live_trade_log.csv"

EPS = 1e-9
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

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
    return {"last_signal_bar_time": None, "positions": {}, "recovery_positions": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def log_message(msg: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}")


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
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    key_width = max(10, min(28, max((len(str(k)) for k, _ in rows), default=10)))
    value_width = max(value_width, min(40, max((len(fmt_value(v)) for _, v in rows), default=value_width)))
    border = "+" + "-" * (key_width + 2) + "+" + "-" * (value_width + 2) + "+"
    print(f"\n[{stamp}] {title}")
    print(border)
    print(f"| {'Field':<{key_width}} | {'Value':<{value_width}} |")
    print(border)
    for key, value in rows:
        value_text = fmt_value(value)
        if len(value_text) > value_width:
            value_text = value_text[: value_width - 3] + "..."
        print(f"| {str(key):<{key_width}} | {value_text:<{value_width}} |")
    print(border)


def signal_name(predicted_class: int) -> str:
    if predicted_class == 1:
        return "BUY"
    if predicted_class == -1:
        return "SELL"
    return "NEUTRAL"


def print_signal_table(signal: Signal) -> None:
    debug = getattr(signal, "debug", {}) or {}
    probs = signal.raw_probabilities
    rows = [
        ("bar_time", datetime.fromtimestamp(signal.bar_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("prediction", signal_name(signal.predicted_class)),
        ("class", signal.predicted_class),
        ("confidence", signal.confidence),
        ("prob_sell", probs.get("-1")),
        ("prob_neutral", probs.get("0")),
        ("prob_buy", probs.get("1")),
        ("entry_price", debug.get("entry_price")),
        ("spread_points", debug.get("spread_points")),
        ("1m_rsi14", debug.get("1m_rsi14")),
        ("5m_rsi14", debug.get("5m_rsi14")),
        ("1m_adx14", debug.get("1m_adx14")),
        ("5m_adx14", debug.get("5m_adx14")),
        ("15m_adx14", debug.get("15m_adx14")),
        ("market_trending", debug.get("market_is_trending")),
        ("market_choppy", debug.get("market_is_choppy")),
        ("adx_trend_score", debug.get("adx_trend_alignment_score")),
        ("rsi_align_score", debug.get("rsi_1m_5m_alignment")),
        ("di_align_score", debug.get("di_direction_alignment_score")),
        ("bull_context", debug.get("rsi_adx_bull_context")),
        ("bear_context", debug.get("rsi_adx_bear_context")),
    ]
    print_table("NEW CLOSED BAR / MODEL SIGNAL", rows, value_width=24)


def print_order_table(signal: Signal, price: float, sl_price: float, tp_price: Optional[float], risk_usd: float, reward_usd: float, spread_pts: float, volume: float) -> None:
    rows = [
        ("action", signal_name(signal.predicted_class)),
        ("confidence", signal.confidence),
        ("volume", volume),
        ("entry_price", price),
        ("stop_loss", sl_price),
        ("take_profit", tp_price),
        ("risk_usd", risk_usd),
        ("reward_usd", reward_usd),
        ("spread_points", spread_pts),
        ("dry_run", DRY_RUN),
    ]
    print_table("ORDER REQUEST", rows, value_width=18)


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

    print_table("MT5 CONNECTION", [
        ("login", info.login),
        ("server", info.server),
        ("balance", info.balance),
        ("equity", info.equity),
    ], value_width=20)


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
# FEATURE ENGINEERING (must match dataset_builder_LightGBM_indicators.py)
# ==========================================================
def add_rsi(df: pd.DataFrame, prefix: str, period: int = RSI_PERIOD) -> pd.DataFrame:
    df = df.copy()
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
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df[f"{prefix}_tr"] = tr
    df[f"{prefix}_atr{period}"] = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    df[f"{prefix}_atr_pct{period}"] = df[f"{prefix}_atr{period}"] / (df["close"] + EPS)
    df[f"{prefix}_range_vs_atr{period}"] = df[f"{prefix}_range"] / (df[f"{prefix}_atr{period}"] + EPS)
    return df


def add_adx(df: pd.DataFrame, prefix: str, period: int = ADX_PERIOD) -> pd.DataFrame:
    df = df.copy()
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


def fetch_aligned_feature_frames(symbol: str, m1_count: int = 600) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rates_1m = get_latest_rates(symbol, mt5.TIMEFRAME_M1, count=m1_count)
    rates_5m = get_latest_rates(symbol, mt5.TIMEFRAME_M5, count=max(220, math.ceil(m1_count / 5) + 120))
    rates_15m = get_latest_rates(symbol, mt5.TIMEFRAME_M15, count=max(220, math.ceil(m1_count / 15) + 120))
    df_1m = mt5_rates_to_price_df(rates_1m)
    df_5m = mt5_rates_to_price_df(rates_5m)
    df_15m = mt5_rates_to_price_df(rates_15m)
    df_5m = df_5m.reindex(df_1m.index, method="ffill")
    df_15m = df_15m.reindex(df_1m.index, method="ffill")
    return df_1m, df_5m, df_15m


def make_feature_frame(symbol: str) -> pd.DataFrame:
    """
    Build the live feature table in one concat step instead of repeatedly inserting
    columns. This keeps pandas from creating a fragmented DataFrame and removes
    the PerformanceWarning spam in the terminal.
    """
    df_1m_raw, df_5m_raw, df_15m_raw = fetch_aligned_feature_frames(symbol, m1_count=600)

    df_1m = create_features(df_1m_raw, "1m", use_rsi=True, use_atr=True, use_adx=True)
    df_5m = create_features(df_5m_raw, "5m", use_rsi=True, use_atr=False, use_adx=True)
    df_15m = create_features(df_15m_raw, "15m", use_rsi=False, use_atr=False, use_adx=True)

    base_parts = [
        df_1m.filter(regex=r"^1m_"),
        df_5m.filter(regex=r"^5m_"),
        df_15m.filter(regex=r"^15m_"),
    ]

    cross = pd.DataFrame(index=df_1m.index)

    cross["align_1m_5m_ma10"] = (df_1m["1m_ma10_above_ma20"] == df_5m["5m_ma10_above_ma20"]).astype(int)
    cross["align_1m_15m_ma10"] = (df_1m["1m_ma10_above_ma20"] == df_15m["15m_ma10_above_ma20"]).astype(int)
    cross["align_5m_15m_ma10"] = (df_5m["5m_ma10_above_ma20"] == df_15m["15m_ma10_above_ma20"]).astype(int)

    cross["triple_trend_alignment"] = (
        (df_1m["1m_ma10_above_ma20"] == df_5m["5m_ma10_above_ma20"]) &
        (df_5m["5m_ma10_above_ma20"] == df_15m["15m_ma10_above_ma20"])
    ).astype(int)

    cross["triple_bull_alignment"] = (
        (df_1m["1m_close_above_ma20"] == 1) &
        (df_5m["5m_close_above_ma20"] == 1) &
        (df_15m["15m_close_above_ma20"] == 1)
    ).astype(int)

    cross["triple_bear_alignment"] = (
        (df_1m["1m_close_above_ma20"] == 0) &
        (df_5m["5m_close_above_ma20"] == 0) &
        (df_15m["15m_close_above_ma20"] == 0)
    ).astype(int)

    cross["momentum_alignment_score"] = (
        np.sign(df_1m["1m_momentum5"]).fillna(0) +
        np.sign(df_5m["5m_momentum5"]).fillna(0) +
        np.sign(df_15m["15m_momentum5"]).fillna(0)
    )

    cross["1m_vs_5m_range_ratio"] = df_1m["1m_range_mean10"] / (df_5m["5m_range_mean10"] + EPS)
    cross["1m_vs_15m_range_ratio"] = df_1m["1m_range_mean10"] / (df_15m["15m_range_mean10"] + EPS)
    cross["1m_body_vs_5m_range"] = df_1m["1m_body"].abs() / (df_5m["5m_range_mean10"] + EPS)
    cross["1m_body_vs_15m_range"] = df_1m["1m_body"].abs() / (df_15m["15m_range_mean10"] + EPS)

    cross["rsi_1m_5m_alignment"] = df_1m["1m_rsi_above_50"] + df_5m["5m_rsi_above_50"]
    cross["rsi_1m_5m_bullish"] = (cross["rsi_1m_5m_alignment"] == 2).astype(int)
    cross["rsi_1m_5m_bearish"] = (cross["rsi_1m_5m_alignment"] == 0).astype(int)

    cross["adx_trend_alignment_score"] = (
        df_1m["1m_adx_trending"] +
        df_5m["5m_adx_trending"] +
        df_15m["15m_adx_trending"]
    )
    cross["adx_choppy_alignment_score"] = (
        df_1m["1m_adx_choppy"] +
        df_5m["5m_adx_choppy"] +
        df_15m["15m_adx_choppy"]
    )

    cross["market_is_trending"] = (df_15m["15m_adx14"] > 25).astype(int)
    cross["market_is_choppy"] = (df_15m["15m_adx14"] < 20).astype(int)
    cross["entry_trend_confirmed"] = (
        (df_1m["1m_adx14"] > 20) &
        (df_5m["5m_adx14"] > 20) &
        (df_15m["15m_adx14"] > 20)
    ).astype(int)

    cross["di_direction_alignment_score"] = (
        df_1m["1m_di_direction"].fillna(0) +
        df_5m["5m_di_direction"].fillna(0) +
        df_15m["15m_di_direction"].fillna(0)
    )

    cross["rsi_adx_bull_context"] = (
        (cross["rsi_1m_5m_bullish"] == 1) &
        (cross["di_direction_alignment_score"] > 0) &
        (cross["market_is_trending"] == 1)
    ).astype(int)

    cross["rsi_adx_bear_context"] = (
        (cross["rsi_1m_5m_bearish"] == 1) &
        (cross["di_direction_alignment_score"] < 0) &
        (cross["market_is_trending"] == 1)
    ).astype(int)

    meta = pd.DataFrame({
        "entry_price": df_1m_raw["close"],
        "spread_points": df_1m_raw["spread_points"],
    }, index=df_1m.index)

    feat = pd.concat(base_parts + [cross, meta], axis=1).copy()
    return feat.replace([np.inf, -np.inf], np.nan)


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
    debug_cols = ["entry_price", "spread_points", "1m_rsi14", "5m_rsi14", "1m_atr14", "1m_adx14", "5m_adx14", "15m_adx14", "market_is_trending", "market_is_choppy", "adx_trend_alignment_score", "adx_choppy_alignment_score", "rsi_1m_5m_alignment", "di_direction_alignment_score", "rsi_adx_bull_context", "rsi_adx_bear_context"]
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
    # 🔥 Invert signal if enabled
    if INVERT_SIGNALS:
        signal_class = -signal_class

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

    print_order_table(
        signal=signal,
        price=price,
        sl_price=sl_price,
        tp_price=tp_price,
        risk_usd=risk_usd,
        reward_usd=reward_usd,
        spread_pts=spread_pts,
        volume=volume,
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


def close_position(position, reason: str = "manual_close") -> Optional[dict]:
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
        "comment": f"{ORDER_COMMENT}-{reason}"[:31],
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



def position_side(position) -> int:
    """Return 1 for BUY, -1 for SELL."""
    if position.type == mt5.POSITION_TYPE_BUY:
        return 1
    if position.type == mt5.POSITION_TYPE_SELL:
        return -1
    return 0


def position_side_name(position) -> str:
    side = position_side(position)
    return "BUY" if side == 1 else "SELL" if side == -1 else "UNKNOWN"


def bars_since_entry(current_bar_time: int, entry_bar_time: int, timeframe: int) -> int:
    tf_seconds = TIMEFRAME_TO_SECONDS.get(timeframe)
    if tf_seconds is None:
        return 0
    return max(0, round((current_bar_time - entry_bar_time) / tf_seconds))


def print_positions_table(symbol: str, state: dict) -> None:
    positions = get_open_positions(symbol, MAGIC_NUMBER)
    if not positions:
        print_table("OPEN MANAGED POSITIONS", [("status", "none")], value_width=18)
        return

    total_profit = sum(float(p.profit) for p in positions)
    buy_count = sum(1 for p in positions if p.type == mt5.POSITION_TYPE_BUY)
    sell_count = sum(1 for p in positions if p.type == mt5.POSITION_TYPE_SELL)
    recovery_count = len(state.get("recovery_positions", {}))
    best = max(positions, key=lambda p: p.profit)
    worst = min(positions, key=lambda p: p.profit)

    rows = [
        ("total_positions", len(positions)),
        ("buy_positions", buy_count),
        ("sell_positions", sell_count),
        ("recovery_marked", recovery_count),
        ("total_profit", total_profit),
        ("best_ticket", best.ticket),
        ("best_side", position_side_name(best)),
        ("best_profit", best.profit),
        ("worst_ticket", worst.ticket),
        ("worst_side", position_side_name(worst)),
        ("worst_profit", worst.profit),
    ]
    print_table("OPEN MANAGED POSITIONS", rows, value_width=20)


def sync_state_with_live_positions(symbol: str, state: dict) -> None:
    positions = get_open_positions(symbol, MAGIC_NUMBER)
    live_tickets = {str(p.ticket) for p in positions}
    state["positions"] = {
        k: v for k, v in state.get("positions", {}).items()
        if k in live_tickets
    }
    state["recovery_positions"] = {
        k: v for k, v in state.get("recovery_positions", {}).items()
        if k in live_tickets
    }


def manage_recovery_positions(symbol: str, timeframe: int, state: dict) -> None:
    sync_state_with_live_positions(symbol, state)
    if DRY_RUN:
        return

    recovery = state.setdefault("recovery_positions", {})
    if not recovery:
        return

    rates = get_latest_rates(symbol, timeframe, count=5)
    latest_closed_bar_time = int(rates.iloc[-2]["time"].timestamp())
    positions = {str(p.ticket): p for p in get_open_positions(symbol, MAGIC_NUMBER)}

    for ticket_key, meta in list(recovery.items()):
        pos = positions.get(ticket_key)
        if pos is None:
            recovery.pop(ticket_key, None)
            continue

        marked_bar_time = int(meta.get("marked_bar_time", latest_closed_bar_time))
        bars_marked = bars_since_entry(latest_closed_bar_time, marked_bar_time, timeframe)

        if float(pos.profit) >= RECOVERY_CLOSE_PROFIT_USD:
            log_message(
                f"RECOVERY PROFIT CLOSE -> ticket={pos.ticket} side={position_side_name(pos)} "
                f"profit={pos.profit:.2f} target={RECOVERY_CLOSE_PROFIT_USD:.2f}"
            )
            result = close_position(pos, reason="recovery")
            if result is not None:
                append_trade_log({
                    "timestamp": datetime.now().isoformat(),
                    "event": "recovery_profit_close",
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "side": position_side_name(pos),
                    "profit": pos.profit,
                    "target_profit": RECOVERY_CLOSE_PROFIT_USD,
                    "bars_marked": bars_marked,
                    "result": json.dumps(result, default=str),
                })
                recovery.pop(ticket_key, None)
                state.get("positions", {}).pop(ticket_key, None)
            continue

        if bars_marked >= RECOVERY_MAX_BARS:
            log_message(
                f"RECOVERY TAG EXPIRED -> ticket={pos.ticket} side={position_side_name(pos)} "
                f"profit={pos.profit:.2f} bars_marked={bars_marked}"
            )
            append_trade_log({
                "timestamp": datetime.now().isoformat(),
                "event": "recovery_tag_expired",
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "side": position_side_name(pos),
                "profit": pos.profit,
                "bars_marked": bars_marked,
            })
            recovery.pop(ticket_key, None)


def manage_profit_lock_on_signal(symbol: str, signal: Signal, state: dict) -> bool:
    sync_state_with_live_positions(symbol, state)
    positions = get_open_positions(symbol, MAGIC_NUMBER)

    if signal.predicted_class not in (1, -1):
        return False

    signal_side = int(signal.predicted_class)
    same_side = [p for p in positions if position_side(p) == signal_side]
    opposite_side = [p for p in positions if position_side(p) == -signal_side]

    closed_count = 0
    marked_count = 0
    blocked_reasons = []

    if CLOSE_PROFITABLE_OPPOSITE_ON_FLIP and signal.confidence >= CLOSE_OPPOSITE_MIN_CONFIDENCE:
        for pos in opposite_side:
            ticket_key = str(pos.ticket)
            if float(pos.profit) >= MIN_PROFIT_TO_CLOSE_OPPOSITE:
                log_message(
                    f"PROFIT LOCK CLOSE -> new_signal={signal_name(signal.predicted_class)} "
                    f"closing {position_side_name(pos)} ticket={pos.ticket} profit={pos.profit:.2f}"
                )
                result = close_position(pos, reason="profitlock")
                if result is not None:
                    closed_count += 1
                    state.get("positions", {}).pop(ticket_key, None)
                    state.get("recovery_positions", {}).pop(ticket_key, None)
                    append_trade_log({
                        "timestamp": datetime.now().isoformat(),
                        "event": "profit_lock_close",
                        "bar_time": signal.bar_time,
                        "ticket": pos.ticket,
                        "symbol": pos.symbol,
                        "closed_side": position_side_name(pos),
                        "new_signal": signal_name(signal.predicted_class),
                        "confidence": signal.confidence,
                        "profit": pos.profit,
                        "min_profit_required": MIN_PROFIT_TO_CLOSE_OPPOSITE,
                        "result": json.dumps(result, default=str),
                    })
            elif MARK_LEFTOVER_OPPOSITE_FOR_RECOVERY:
                state.setdefault("recovery_positions", {})[ticket_key] = {
                    "marked_bar_time": signal.bar_time,
                    "old_side": position_side(pos),
                    "new_signal": signal.predicted_class,
                    "profit_when_marked": float(pos.profit),
                    "confidence_when_marked": float(signal.confidence),
                }
                marked_count += 1
                log_message(
                    f"RECOVERY MARK -> ticket={pos.ticket} side={position_side_name(pos)} "
                    f"profit={pos.profit:.2f}; close later if profit >= {RECOVERY_CLOSE_PROFIT_USD:.2f}"
                )
                append_trade_log({
                    "timestamp": datetime.now().isoformat(),
                    "event": "recovery_marked",
                    "bar_time": signal.bar_time,
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "side": position_side_name(pos),
                    "new_signal": signal_name(signal.predicted_class),
                    "confidence": signal.confidence,
                    "profit_when_marked": pos.profit,
                    "recovery_target_profit": RECOVERY_CLOSE_PROFIT_USD,
                })
    elif opposite_side:
        blocked_reasons.append("signal_not_strong_enough_for_profit_lock")

    sync_state_with_live_positions(symbol, state)
    positions = get_open_positions(symbol, MAGIC_NUMBER)
    same_side = [p for p in positions if position_side(p) == signal_side]

    if len(positions) >= MAX_TOTAL_POSITIONS:
        blocked_reasons.append("max_total_positions")
    if len(same_side) >= MAX_SAME_SIDE_POSITIONS:
        blocked_reasons.append("max_same_side_positions")
    if not ALLOW_NEW_ENTRY_AFTER_PROFIT_LOCK:
        blocked_reasons.append("new_entry_disabled_after_management")

    rows = [
        ("new_signal", signal_name(signal.predicted_class)),
        ("confidence", signal.confidence),
        ("opposite_before", len(opposite_side)),
        ("profit_locked", closed_count),
        ("recovery_marked", marked_count),
        ("open_positions", len(positions)),
        ("same_side_open", len(same_side)),
        ("allow_new_entry", not blocked_reasons),
        ("block_reason", ",".join(blocked_reasons) if blocked_reasons else "-"),
    ]
    print_table("POSITION MANAGER", rows, value_width=32)

    if blocked_reasons:
        append_trade_log({
            "timestamp": datetime.now().isoformat(),
            "event": "entry_blocked_by_position_manager",
            "bar_time": signal.bar_time,
            "symbol": symbol,
            "predicted_class": signal.predicted_class,
            "confidence": signal.confidence,
            "reason": ",".join(blocked_reasons),
            "open_positions": len(positions),
            "same_side_open": len(same_side),
            "profit_locked": closed_count,
            "recovery_marked": marked_count,
        })
        return False

    return True

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
            result = close_position(pos, reason="timeout")
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
                state.get("recovery_positions", {}).pop(ticket_key, None)
                state.get("recovery_positions", {}).pop(ticket_key, None)


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

    print_table("BOT SETTINGS", [
        ("symbol", SYMBOL),
        ("timeframe", "M1" if TIMEFRAME == mt5.TIMEFRAME_M1 else TIMEFRAME),
        ("features", len(feature_list)),
        ("hold_bars", HOLD_BARS),
        ("confidence", CONFIDENCE_THRESHOLD),
        ("lot_size", LOT_SIZE),
        ("sl_balance_frac", STOP_LOSS_ACCOUNT_FRACTION),
        ("risk_reward", RISK_REWARD_RATIO),
        ("max_total_pos", MAX_TOTAL_POSITIONS),
        ("max_same_side", MAX_SAME_SIDE_POSITIONS),
        ("profit_lock_min", MIN_PROFIT_TO_CLOSE_OPPOSITE),
        ("recovery_target", RECOVERY_CLOSE_PROFIT_USD),
        ("recovery_max_bars", RECOVERY_MAX_BARS),
        ("dry_run", DRY_RUN),
    ], value_width=18)

    try:
        while True:
            manage_recovery_positions(SYMBOL, TIMEFRAME, state)
            manage_timeouts(SYMBOL, TIMEFRAME, state)

            signal = predict_signal(model, feature_list, SYMBOL)

            if state.get("last_signal_bar_time") == signal.bar_time:
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue

            state["last_signal_bar_time"] = signal.bar_time

            print_signal_table(signal)
            print_positions_table(SYMBOL, state)

            if should_skip_signal(signal):
                append_trade_log({
                    "timestamp": datetime.now().isoformat(),
                    "event": "signal_skipped",
                    "bar_time": signal.bar_time,
                    "predicted_class": signal.predicted_class,
                    "confidence": signal.confidence,
                    "probabilities": json.dumps(signal.raw_probabilities),
                    "debug": json.dumps(getattr(signal, "debug", {})),
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
                    "debug": json.dumps(getattr(signal, "debug", {})),
                    "reason": "position_already_open",
                })
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue

            allowed_to_open = manage_profit_lock_on_signal(SYMBOL, signal, state)
            if not allowed_to_open:
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
                    "debug": json.dumps(getattr(signal, "debug", {})),
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
