import pandas as pd
import numpy as np
from pathlib import Path

# ========== STEP 0: SCRIPT DIRECTORY ==========
SCRIPT_DIR = Path(__file__).resolve().parent
print(f"Script directory: {SCRIPT_DIR}")

# XAUUSDm usually has Digits = 2, so 1 point = 0.01.
# If you switch symbols, update POINT_VALUE and the profit conversion settings below.
POINT_VALUE = 0.01
EPS = 1e-9

# ========== LIVE-STYLE SL/TP LABELING SETTINGS ==========
# These make the dataset label trades closer to how the live bot places SL/TP.
# Live bot logic:
#   risk_usd   = balance * STOP_LOSS_ACCOUNT_FRACTION
#   reward_usd = risk_usd * RISK_REWARD_RATIO
#   SL/TP price is calculated from that money risk and lot size.
#
# Dataset cannot know your future live balance per trade, so SIM_BALANCE is the
# fixed balance assumption used while labeling historical rows. Set this close
# to the balance you expect to trade with.
USE_LIVE_STYLE_SL_FOR_LABELS = True
SIM_BALANCE = 40
SIM_LOT_SIZE = 0.01
STOP_LOSS_ACCOUNT_FRACTION = 0.10

# XAUUSD assumption: 1.00 price move at 1.00 lot is about $100.
# Therefore at 0.01 lot, 1.00 price move is about $1.
# If your broker contract is different, adjust this value.
USD_PER_PRICE_MOVE_PER_1_LOT = 100.0

# Optional safety clamp for dataset labels. Live MT5 may also reject orders that
# violate broker stop-level rules, but this keeps labels from becoming absurd.
USE_LIVE_STYLE_SL_CLAMP = True

# Label settings
FUTURE_SHIFT = 30
RETURN_THRESHOLD = 0.0009

# Indicator settings
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

# Dynamic SL/TP planner settings
BASE_ATR_SL_MULTIPLIER = 1.5
MIN_ATR_SL_MULTIPLIER = 1.0
MAX_ATR_SL_MULTIPLIER = 2.5
RISK_REWARD_RATIO = 2.0

# Minimum practical SL distance to avoid tiny SLs
MIN_SL_POINTS = 1500
MAX_SL_POINTS = 20000

# Order Block / Breaker Block settings
OB_SWING_LOOKBACK = 10
OB_USE_BODY = False
OB_RECENT_WINDOW = 5
OB_MAX_ACTIVE_ZONES = 30
OB_NO_ZONE_DISTANCE = 10.0
OB_NO_ZONE_AGE = 9999

# Fair Value Gap settings
FVG_THRESHOLD_PCT = 0.0
FVG_AUTO_THRESHOLD = False
FVG_RECENT_WINDOW = 5
FVG_MAX_ACTIVE_ZONES = 50
FVG_NO_ZONE_DISTANCE = 10.0
FVG_NO_ZONE_AGE = 9999

# Fractal BOS / CHoCH market structure settings
STRUCTURE_FRACTAL_LENGTH = 5
STRUCTURE_RECENT_WINDOW = 5
STRUCTURE_NO_BREAK_AGE = 9999


# Liquidity Sweep / Rejection settings
# These detect fake breakouts around recent highs/lows using only historical candles.
SWEEP_LOOKBACK = 20
SWEEP_REJECTION_BODY_CLOSE = True
SWEEP_RECENT_WINDOW = 5
SWEEP_NO_SWEEP_AGE = 9999
SWEEP_STRONG_WICK_RATIO = 0.45
SWEEP_STRONG_CLOSE_BEYOND_MIDPOINT = True

# ========== HTF LIQUIDITY LEVEL SETTINGS ==========
HTF_LIQUIDITY_NEAR_ATR_MULTIPLIER = 0.35
HTF_LIQUIDITY_NEAR_PCT_FALLBACK = 0.0015
HTF_LIQUIDITY_RECENT_WINDOW = 10
HTF_LIQUIDITY_NO_SWEEP_AGE = 9999

# ========== ICT SESSION LIQUIDITY SETTINGS ==========
# Uses the MT5/server timestamp already present in your CSVs.
# Keep these windows consistent in the live script later.
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



# ========== STEP 1: AUTO-DETECT FILES ==========
def find_mt5_file(timeframe: str) -> Path:
    matches = sorted(SCRIPT_DIR.glob(f"*_{timeframe}_*.csv"))

    if not matches:
        raise FileNotFoundError(
            f"No CSV file found for timeframe '{timeframe}' in:\n{SCRIPT_DIR}"
        )

    if len(matches) > 1:
        print(f"[INFO] Multiple files found for {timeframe}. Using: {matches[0].name}")

    return matches[0]


# ========== STEP 2: LOAD DATA ==========
def load_mt5_file(path: Path) -> pd.DataFrame:
    print(f"[LOADING] {path.name}")

    df = pd.read_csv(path, delimiter="\t")
    df.columns = df.columns.str.strip()

    required_cols = [
        "<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>",
        "<TICKVOL>", "<SPREAD>"
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}\n"
            f"Detected columns: {list(df.columns)}"
        )

    df["datetime"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"])
    df.set_index("datetime", inplace=True)

    df.rename(columns={
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "volume",
        "<SPREAD>": "spread_points"
    }, inplace=True)

    return df[["open", "high", "low", "close", "volume", "spread_points"]].copy()


# ========== STEP 3: INDICATORS ==========
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
    df[f"{prefix}_atr{period}"] = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    df[f"{prefix}_atr_pct{period}"] = df[f"{prefix}_atr{period}"] / (df["close"] + EPS)
    df[f"{prefix}_range_vs_atr{period}"] = df[f"{prefix}_range"] / (df[f"{prefix}_atr{period}"] + EPS)

    return df


def add_adx(df: pd.DataFrame, prefix: str, period: int = ADX_PERIOD) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean() / (atr + EPS)

    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean() / (atr + EPS)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    df[f"{prefix}_plus_di{period}"] = plus_di
    df[f"{prefix}_minus_di{period}"] = minus_di
    df[f"{prefix}_adx{period}"] = adx
    df[f"{prefix}_di_direction"] = np.sign(plus_di - minus_di)
    df[f"{prefix}_adx_trending"] = (adx > 25).astype(int)
    df[f"{prefix}_adx_choppy"] = (adx < 20).astype(int)

    return df


# ========== STEP 4: FEATURE ENGINEERING ==========
def create_features(
    df: pd.DataFrame,
    prefix: str,
    use_rsi: bool = False,
    use_atr: bool = False,
    use_adx: bool = False
) -> pd.DataFrame:
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
    df[f"{prefix}_wick_imbalance"] = (
        df[f"{prefix}_lower_wick"] - df[f"{prefix}_upper_wick"]
    ) / (range_safe + EPS)

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


# ========== ORDER BLOCK / BREAKER BLOCK FEATURES ==========
def add_order_block_features(
    df: pd.DataFrame,
    prefix: str,
    swing_lookback: int = OB_SWING_LOOKBACK,
    use_body: bool = OB_USE_BODY,
    recent_window: int = OB_RECENT_WINDOW,
    max_active_zones: int = OB_MAX_ACTIVE_ZONES,
) -> pd.DataFrame:
    """
    LuxAlgo-style OB/Breaker features with no future leakage:
    confirmed swing -> structure break -> zone from extreme candle between swing and break.
    """
    out = df.copy()

    open_arr = out["open"].to_numpy(dtype=float)
    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    close_arr = out["close"].to_numpy(dtype=float)
    n = len(out)

    if use_body:
        zone_high_arr = np.maximum(open_arr, close_arr)
        zone_low_arr = np.minimum(open_arr, close_arr)
    else:
        zone_high_arr = high_arr
        zone_low_arr = low_arr

    inside_bull_ob = np.zeros(n, dtype=int)
    inside_bear_ob = np.zeros(n, dtype=int)
    inside_bull_breaker = np.zeros(n, dtype=int)
    inside_bear_breaker = np.zeros(n, dtype=int)

    recent_bull_ob = np.zeros(n, dtype=int)
    recent_bear_ob = np.zeros(n, dtype=int)
    recent_bull_breaker = np.zeros(n, dtype=int)
    recent_bear_breaker = np.zeros(n, dtype=int)

    dist_to_bull_ob = np.full(n, OB_NO_ZONE_DISTANCE, dtype=float)
    dist_to_bear_ob = np.full(n, OB_NO_ZONE_DISTANCE, dtype=float)
    dist_to_bull_breaker = np.full(n, OB_NO_ZONE_DISTANCE, dtype=float)
    dist_to_bear_breaker = np.full(n, OB_NO_ZONE_DISTANCE, dtype=float)

    bull_ob_age = np.full(n, OB_NO_ZONE_AGE, dtype=float)
    bear_ob_age = np.full(n, OB_NO_ZONE_AGE, dtype=float)
    bull_breaker_age = np.full(n, OB_NO_ZONE_AGE, dtype=float)
    bear_breaker_age = np.full(n, OB_NO_ZONE_AGE, dtype=float)

    bull_ob_width_pct = np.zeros(n, dtype=float)
    bear_ob_width_pct = np.zeros(n, dtype=float)
    bull_breaker_width_pct = np.zeros(n, dtype=float)
    bear_breaker_width_pct = np.zeros(n, dtype=float)

    active_bull = []
    active_bear = []
    last_swing_high = None
    last_swing_low = None

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

        # Confirm swing only after swing_lookback candles have passed.
        j = i - swing_lookback
        if j >= swing_lookback:
            left = j - swing_lookback
            right = min(n, j + swing_lookback + 1)

            if high_arr[j] >= np.max(high_arr[left:right]):
                last_swing_high = {"price": high_arr[j], "idx": j, "crossed": False}

            if low_arr[j] <= np.min(low_arr[left:right]):
                last_swing_low = {"price": low_arr[j], "idx": j, "crossed": False}

        # Bullish OB: close breaks above confirmed swing high.
        if last_swing_high is not None and not last_swing_high["crossed"]:
            if close_arr[i] > last_swing_high["price"]:
                start = last_swing_high["idx"] + 1
                end = i
                if end > start:
                    ob_idx = start + int(np.argmin(zone_low_arr[start:end]))
                    active_bull.insert(0, {
                        "top": float(zone_high_arr[ob_idx]),
                        "btm": float(zone_low_arr[ob_idx]),
                        "loc": int(ob_idx),
                        "breaker": False,
                        "break_loc": None,
                    })
                    active_bull = active_bull[:max_active_zones]
                last_swing_high["crossed"] = True

        # Bearish OB: close breaks below confirmed swing low.
        if last_swing_low is not None and not last_swing_low["crossed"]:
            if close_arr[i] < last_swing_low["price"]:
                start = last_swing_low["idx"] + 1
                end = i
                if end > start:
                    ob_idx = start + int(np.argmax(zone_high_arr[start:end]))
                    active_bear.insert(0, {
                        "top": float(zone_high_arr[ob_idx]),
                        "btm": float(zone_low_arr[ob_idx]),
                        "loc": int(ob_idx),
                        "breaker": False,
                        "break_loc": None,
                    })
                    active_bear = active_bear[:max_active_zones]
                last_swing_low["crossed"] = True

        # Bullish OB becomes a breaker if violated below.
        kept_bull = []
        for z in active_bull:
            if not z["breaker"]:
                if body_low < z["btm"]:
                    z["breaker"] = True
                    z["break_loc"] = i
                kept_bull.append(z)
            else:
                # Remove invalidated breaker after price closes back above top.
                if close_arr[i] <= z["top"]:
                    kept_bull.append(z)
        active_bull = kept_bull[:max_active_zones]

        # Bearish OB becomes a breaker if violated above.
        kept_bear = []
        for z in active_bear:
            if not z["breaker"]:
                if body_high > z["top"]:
                    z["breaker"] = True
                    z["break_loc"] = i
                kept_bear.append(z)
            else:
                # Remove invalidated breaker after price closes back below bottom.
                if close_arr[i] >= z["btm"]:
                    kept_bear.append(z)
        active_bear = kept_bear[:max_active_zones]

        bull = nearest_zone(active_bull, price, want_breaker=False)
        bear = nearest_zone(active_bear, price, want_breaker=False)
        bull_br = nearest_zone(active_bull, price, want_breaker=True)
        bear_br = nearest_zone(active_bear, price, want_breaker=True)

        if bull is not None:
            inside_bull_ob[i] = int(bull["btm"] <= price <= bull["top"])
            dist_to_bull_ob[i] = 0.0 if inside_bull_ob[i] else min(abs(price - bull["top"]), abs(price - bull["btm"])) / (price + EPS)
            bull_ob_age[i] = i - bull["loc"]
            bull_ob_width_pct[i] = (bull["top"] - bull["btm"]) / (price + EPS)
            recent_bull_ob[i] = int(bull_ob_age[i] <= recent_window)

        if bear is not None:
            inside_bear_ob[i] = int(bear["btm"] <= price <= bear["top"])
            dist_to_bear_ob[i] = 0.0 if inside_bear_ob[i] else min(abs(price - bear["top"]), abs(price - bear["btm"])) / (price + EPS)
            bear_ob_age[i] = i - bear["loc"]
            bear_ob_width_pct[i] = (bear["top"] - bear["btm"]) / (price + EPS)
            recent_bear_ob[i] = int(bear_ob_age[i] <= recent_window)

        if bull_br is not None:
            inside_bull_breaker[i] = int(bull_br["btm"] <= price <= bull_br["top"])
            dist_to_bull_breaker[i] = 0.0 if inside_bull_breaker[i] else min(abs(price - bull_br["top"]), abs(price - bull_br["btm"])) / (price + EPS)
            base_loc = bull_br["break_loc"] if bull_br["break_loc"] is not None else bull_br["loc"]
            bull_breaker_age[i] = i - base_loc
            bull_breaker_width_pct[i] = (bull_br["top"] - bull_br["btm"]) / (price + EPS)
            recent_bull_breaker[i] = int(bull_breaker_age[i] <= recent_window)

        if bear_br is not None:
            inside_bear_breaker[i] = int(bear_br["btm"] <= price <= bear_br["top"])
            dist_to_bear_breaker[i] = 0.0 if inside_bear_breaker[i] else min(abs(price - bear_br["top"]), abs(price - bear_br["btm"])) / (price + EPS)
            base_loc = bear_br["break_loc"] if bear_br["break_loc"] is not None else bear_br["loc"]
            bear_breaker_age[i] = i - base_loc
            bear_breaker_width_pct[i] = (bear_br["top"] - bear_br["btm"]) / (price + EPS)
            recent_bear_breaker[i] = int(bear_breaker_age[i] <= recent_window)

    out[f"{prefix}_inside_bull_ob"] = inside_bull_ob
    out[f"{prefix}_inside_bear_ob"] = inside_bear_ob
    out[f"{prefix}_inside_bull_breaker"] = inside_bull_breaker
    out[f"{prefix}_inside_bear_breaker"] = inside_bear_breaker

    out[f"{prefix}_recent_bull_ob"] = recent_bull_ob
    out[f"{prefix}_recent_bear_ob"] = recent_bear_ob
    out[f"{prefix}_recent_bull_breaker"] = recent_bull_breaker
    out[f"{prefix}_recent_bear_breaker"] = recent_bear_breaker

    out[f"{prefix}_dist_to_bull_ob"] = dist_to_bull_ob
    out[f"{prefix}_dist_to_bear_ob"] = dist_to_bear_ob
    out[f"{prefix}_dist_to_bull_breaker"] = dist_to_bull_breaker
    out[f"{prefix}_dist_to_bear_breaker"] = dist_to_bear_breaker

    out[f"{prefix}_bull_ob_age"] = bull_ob_age
    out[f"{prefix}_bear_ob_age"] = bear_ob_age
    out[f"{prefix}_bull_breaker_age"] = bull_breaker_age
    out[f"{prefix}_bear_breaker_age"] = bear_breaker_age

    out[f"{prefix}_bull_ob_width_pct"] = bull_ob_width_pct
    out[f"{prefix}_bear_ob_width_pct"] = bear_ob_width_pct
    out[f"{prefix}_bull_breaker_width_pct"] = bull_breaker_width_pct
    out[f"{prefix}_bear_breaker_width_pct"] = bear_breaker_width_pct

    return out



# ========== FAIR VALUE GAP FEATURES ==========
def add_fvg_features(
    df: pd.DataFrame,
    prefix: str,
    threshold_pct: float = FVG_THRESHOLD_PCT,
    auto_threshold: bool = FVG_AUTO_THRESHOLD,
    recent_window: int = FVG_RECENT_WINDOW,
    max_active_zones: int = FVG_MAX_ACTIVE_ZONES,
) -> pd.DataFrame:
    """
    LuxAlgo/ICT-style Fair Value Gap features with no future leakage.

    Bullish FVG:
        low[i] > high[i-2] and close[i-1] > high[i-2]
        zone top = low[i], zone bottom = high[i-2]

    Bearish FVG:
        high[i] < low[i-2] and close[i-1] < low[i-2]
        zone top = low[i-2], zone bottom = high[i]

    Mitigation/removal:
        bullish FVG removed if close < zone bottom
        bearish FVG removed if close > zone top
    """
    out = df.copy()

    high_arr = out["high"].to_numpy(dtype=float)
    low_arr = out["low"].to_numpy(dtype=float)
    close_arr = out["close"].to_numpy(dtype=float)
    n = len(out)

    active_bull = []
    active_bear = []

    bull_fvg_detected = np.zeros(n, dtype=int)
    bear_fvg_detected = np.zeros(n, dtype=int)

    inside_bull_fvg = np.zeros(n, dtype=int)
    inside_bear_fvg = np.zeros(n, dtype=int)

    recent_bull_fvg = np.zeros(n, dtype=int)
    recent_bear_fvg = np.zeros(n, dtype=int)

    dist_to_bull_fvg = np.full(n, FVG_NO_ZONE_DISTANCE, dtype=float)
    dist_to_bear_fvg = np.full(n, FVG_NO_ZONE_DISTANCE, dtype=float)

    bull_fvg_age = np.full(n, FVG_NO_ZONE_AGE, dtype=float)
    bear_fvg_age = np.full(n, FVG_NO_ZONE_AGE, dtype=float)

    bull_fvg_width_pct = np.zeros(n, dtype=float)
    bear_fvg_width_pct = np.zeros(n, dtype=float)

    bull_fvg_count_active = np.zeros(n, dtype=int)
    bear_fvg_count_active = np.zeros(n, dtype=int)

    bull_fvg_mitigated = np.zeros(n, dtype=int)
    bear_fvg_mitigated = np.zeros(n, dtype=int)

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

        threshold = (
            cumulative_range_ratio / max(i, 1)
            if auto_threshold
            else threshold_pct
        )

        if i >= 2:
            bull_gap_pct = (low_arr[i] - high_arr[i - 2]) / (high_arr[i - 2] + EPS)
            bear_gap_pct = (low_arr[i - 2] - high_arr[i]) / (high_arr[i] + EPS)

            is_bull_fvg = (
                low_arr[i] > high_arr[i - 2]
                and close_arr[i - 1] > high_arr[i - 2]
                and bull_gap_pct > threshold
            )

            is_bear_fvg = (
                high_arr[i] < low_arr[i - 2]
                and close_arr[i - 1] < low_arr[i - 2]
                and bear_gap_pct > threshold
            )

            if is_bull_fvg:
                active_bull.insert(0, {
                    "top": float(low_arr[i]),
                    "btm": float(high_arr[i - 2]),
                    "loc": int(i),
                })
                active_bull = active_bull[:max_active_zones]
                bull_fvg_detected[i] = 1

            if is_bear_fvg:
                active_bear.insert(0, {
                    "top": float(low_arr[i - 2]),
                    "btm": float(high_arr[i]),
                    "loc": int(i),
                })
                active_bear = active_bear[:max_active_zones]
                bear_fvg_detected[i] = 1

        kept_bull = []
        bull_removed_now = 0
        for z in active_bull:
            if close_arr[i] < z["btm"]:
                bull_removed_now += 1
            else:
                kept_bull.append(z)
        active_bull = kept_bull[:max_active_zones]

        kept_bear = []
        bear_removed_now = 0
        for z in active_bear:
            if close_arr[i] > z["top"]:
                bear_removed_now += 1
            else:
                kept_bear.append(z)
        active_bear = kept_bear[:max_active_zones]

        bull_fvg_mitigated[i] = bull_removed_now
        bear_fvg_mitigated[i] = bear_removed_now

        bull = nearest_zone(active_bull, price)
        bear = nearest_zone(active_bear, price)

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

        bull_fvg_count_active[i] = len(active_bull)
        bear_fvg_count_active[i] = len(active_bear)

    out[f"{prefix}_bull_fvg_detected"] = bull_fvg_detected
    out[f"{prefix}_bear_fvg_detected"] = bear_fvg_detected

    out[f"{prefix}_inside_bull_fvg"] = inside_bull_fvg
    out[f"{prefix}_inside_bear_fvg"] = inside_bear_fvg

    out[f"{prefix}_recent_bull_fvg"] = recent_bull_fvg
    out[f"{prefix}_recent_bear_fvg"] = recent_bear_fvg

    out[f"{prefix}_dist_to_bull_fvg"] = dist_to_bull_fvg
    out[f"{prefix}_dist_to_bear_fvg"] = dist_to_bear_fvg

    out[f"{prefix}_bull_fvg_age"] = bull_fvg_age
    out[f"{prefix}_bear_fvg_age"] = bear_fvg_age

    out[f"{prefix}_bull_fvg_width_pct"] = bull_fvg_width_pct
    out[f"{prefix}_bear_fvg_width_pct"] = bear_fvg_width_pct

    out[f"{prefix}_bull_fvg_count_active"] = bull_fvg_count_active
    out[f"{prefix}_bear_fvg_count_active"] = bear_fvg_count_active

    out[f"{prefix}_bull_fvg_mitigated"] = bull_fvg_mitigated
    out[f"{prefix}_bear_fvg_mitigated"] = bear_fvg_mitigated

    return out


# ========== ENTRY ZONE PRICE FEATURES ==========
def add_entry_zone_price_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Adds nearest active 1m OB/FVG zone boundaries so the labeler can test
    wait-for-pullback entries. This does not use future candles for the current row.
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


# ========== ENTRY STYLE LABELER ==========
ENTRY_WAIT_BARS = 10
ENTRY_EVAL_BARS = FUTURE_SHIFT
MIN_ENTRY_STYLE_RETURN = 0.00025  # require real edge before labeling a trade
MIN_EDGE_OVER_SECOND_BEST = 0.00005  # avoid ambiguous best entries becoming trades

ENTRY_STYLE_NAMES = {
    -3: "SELL_WAIT_OB",
    -2: "SELL_WAIT_FVG",
    -1: "SELL_NOW",
     0: "NO_TRADE",
     1: "BUY_NOW",
     2: "BUY_WAIT_FVG",
     3: "BUY_WAIT_OB",
}

def add_entry_style_labels(df: pd.DataFrame, raw_1m: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw = raw_1m.reindex(out.index)

    close = raw["close"].to_numpy(float)
    high = raw["high"].to_numpy(float)
    low = raw["low"].to_numpy(float)
    spread_ret = out["spread_return"].to_numpy(float)

    if USE_LIVE_STYLE_SL_FOR_LABELS and "live_style_sl_price_distance" in out.columns:
        sl_dist = out["live_style_sl_price_distance"].to_numpy(float)
        tp_dist = out["live_style_tp_price_distance"].to_numpy(float)
    else:
        sl_dist = out["dynamic_sl_price_distance"].to_numpy(float)
        tp_dist = out["dynamic_tp_price_distance"].to_numpy(float)

    n = len(out)

    candidate_cols = {
        1: "ret_buy_now", 2: "ret_buy_wait_fvg", 3: "ret_buy_wait_ob",
        -1: "ret_sell_now", -2: "ret_sell_wait_fvg", -3: "ret_sell_wait_ob",
    }
    for col in candidate_cols.values():
        out[col] = np.nan

    target_entry_style = np.zeros(n, dtype=int)
    best_return = np.zeros(n, dtype=float)

    def trade_result(i, direction, entry_price, start_j):
        if (
            start_j is None
            or not np.isfinite(entry_price)
            or not np.isfinite(sl_dist[i])
            or not np.isfinite(tp_dist[i])
            or sl_dist[i] <= 0
            or tp_dist[i] <= 0
        ):
            return np.nan
        sl = sl_dist[i]
        tp = tp_dist[i]
        end_j = min(n - 1, i + ENTRY_EVAL_BARS)
        if start_j > end_j:
            return np.nan
        if direction == 1:
            sl_price = entry_price - sl
            tp_price = entry_price + tp
            for j in range(start_j, end_j + 1):
                if low[j] <= sl_price:
                    return (-sl / entry_price) - spread_ret[i]
                if high[j] >= tp_price:
                    return (tp / entry_price) - spread_ret[i]
            return ((close[end_j] - entry_price) / entry_price) - spread_ret[i]
        if direction == -1:
            sl_price = entry_price + sl
            tp_price = entry_price - tp
            for j in range(start_j, end_j + 1):
                if high[j] >= sl_price:
                    return (-sl / entry_price) - spread_ret[i]
                if low[j] <= tp_price:
                    return (tp / entry_price) - spread_ret[i]
            return ((entry_price - close[end_j]) / entry_price) - spread_ret[i]
        return np.nan

    def wait_entry(i, direction, zone_top, zone_btm):
        if not np.isfinite(zone_top) or not np.isfinite(zone_btm):
            return np.nan, None
        end_wait = min(n - 1, i + ENTRY_WAIT_BARS)
        for j in range(i + 1, end_wait + 1):
            if direction == 1 and low[j] <= zone_top:
                return zone_top, j
            if direction == -1 and high[j] >= zone_btm:
                return zone_btm, j
        return np.nan, None

    z = {name: out[name].to_numpy(float) for name in [
        "1m_nearest_bull_fvg_top", "1m_nearest_bull_fvg_btm",
        "1m_nearest_bear_fvg_top", "1m_nearest_bear_fvg_btm",
        "1m_nearest_bull_ob_top", "1m_nearest_bull_ob_btm",
        "1m_nearest_bear_ob_top", "1m_nearest_bear_ob_btm",
    ]}

    for i in range(n - ENTRY_EVAL_BARS):
        candidates = {
            1: trade_result(i, 1, close[i], i + 1),
            -1: trade_result(i, -1, close[i], i + 1),
        }
        entry, start = wait_entry(i, 1, z["1m_nearest_bull_fvg_top"][i], z["1m_nearest_bull_fvg_btm"][i])
        candidates[2] = trade_result(i, 1, entry, start)
        entry, start = wait_entry(i, -1, z["1m_nearest_bear_fvg_top"][i], z["1m_nearest_bear_fvg_btm"][i])
        candidates[-2] = trade_result(i, -1, entry, start)
        entry, start = wait_entry(i, 1, z["1m_nearest_bull_ob_top"][i], z["1m_nearest_bull_ob_btm"][i])
        candidates[3] = trade_result(i, 1, entry, start)
        entry, start = wait_entry(i, -1, z["1m_nearest_bear_ob_top"][i], z["1m_nearest_bear_ob_btm"][i])
        candidates[-3] = trade_result(i, -1, entry, start)

        for cls, value in candidates.items():
            out.iat[i, out.columns.get_loc(candidate_cols[cls])] = value

        valid = {k: v for k, v in candidates.items() if np.isfinite(v)}
        if valid:
            # Sort candidate entry styles from best to worst simulated return.
            # A row becomes a trade label only if:
            #   1) the best candidate has enough net return after spread, and
            #   2) the best candidate is clearly better than the second-best candidate.
            # Otherwise it stays 0 = NO_TRADE.
            ranked = sorted(valid.items(), key=lambda item: item[1], reverse=True)
            best_cls, best_val = ranked[0]
            second_val = ranked[1][1] if len(ranked) > 1 else -np.inf
            edge_over_second = best_val - second_val

            if (best_val >= MIN_ENTRY_STYLE_RETURN) and (edge_over_second >= MIN_EDGE_OVER_SECOND_BEST):
                target_entry_style[i] = best_cls
                best_return[i] = best_val

    out["target_entry_style"] = target_entry_style
    out["entry_style_return"] = best_return
    out["entry_style_name"] = pd.Series(target_entry_style, index=out.index).map(ENTRY_STYLE_NAMES)
    out["target_direction"] = np.sign(target_entry_style).astype(int)
    out["target"] = out["target_entry_style"]
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


# ========== STEP 5: SL/TP PLANNER ==========
def add_dynamic_sl_tp_plan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the old ATR-based SL/TP columns for comparison/debugging, then adds
    live-style SL/TP columns used by the entry-style labeler.

    The live-style columns approximate the live bot's MT5 order_calc_profit()
    behavior without needing an MT5 connection during dataset building.
    """
    df = df.copy()

    # ----- Old ATR-based SL/TP plan kept for comparison -----
    df["sl_atr_multiplier"] = BASE_ATR_SL_MULTIPLIER

    df.loc[df["market_is_choppy"] == 1, "sl_atr_multiplier"] += 0.3
    df.loc[df["entry_trend_confirmed"] == 1, "sl_atr_multiplier"] -= 0.2
    df.loc[df["1m_range_vs_atr14"] > 1.5, "sl_atr_multiplier"] += 0.2

    df["sl_atr_multiplier"] = df["sl_atr_multiplier"].clip(
        lower=MIN_ATR_SL_MULTIPLIER,
        upper=MAX_ATR_SL_MULTIPLIER
    )

    df["dynamic_sl_price_distance"] = df["1m_atr14"] * df["sl_atr_multiplier"]
    df["dynamic_sl_points"] = df["dynamic_sl_price_distance"] / POINT_VALUE
    df["dynamic_sl_points"] = df["dynamic_sl_points"].clip(
        lower=MIN_SL_POINTS,
        upper=MAX_SL_POINTS
    )
    df["dynamic_sl_price_distance"] = df["dynamic_sl_points"] * POINT_VALUE

    df["risk_reward_ratio"] = RISK_REWARD_RATIO
    df["dynamic_tp_price_distance"] = df["dynamic_sl_price_distance"] * RISK_REWARD_RATIO
    df["dynamic_tp_points"] = df["dynamic_sl_points"] * RISK_REWARD_RATIO

    df["sl_distance_pct"] = df["dynamic_sl_price_distance"] / (df["1m_ma20"] + EPS)
    df["tp_distance_pct"] = df["dynamic_tp_price_distance"] / (df["1m_ma20"] + EPS)

    # ----- New live-style SL/TP plan used by labels -----
    live_risk_usd = SIM_BALANCE * STOP_LOSS_ACCOUNT_FRACTION
    live_reward_usd = live_risk_usd * RISK_REWARD_RATIO

    usd_per_price_move = SIM_LOT_SIZE * USD_PER_PRICE_MOVE_PER_1_LOT
    if usd_per_price_move <= 0:
        raise ValueError("Invalid live-style profit conversion settings.")

    live_sl_price_distance = live_risk_usd / usd_per_price_move
    live_tp_price_distance = live_reward_usd / usd_per_price_move

    df["live_style_risk_usd"] = live_risk_usd
    df["live_style_reward_usd"] = live_reward_usd
    df["live_style_lot_size"] = SIM_LOT_SIZE
    df["live_style_sim_balance"] = SIM_BALANCE
    df["live_style_usd_per_price_move"] = usd_per_price_move

    df["live_style_sl_price_distance"] = live_sl_price_distance
    df["live_style_tp_price_distance"] = live_tp_price_distance
    df["live_style_sl_points"] = df["live_style_sl_price_distance"] / POINT_VALUE
    df["live_style_tp_points"] = df["live_style_tp_price_distance"] / POINT_VALUE

    if USE_LIVE_STYLE_SL_CLAMP:
        df["live_style_sl_points"] = df["live_style_sl_points"].clip(
            lower=MIN_SL_POINTS,
            upper=MAX_SL_POINTS
        )
        df["live_style_sl_price_distance"] = df["live_style_sl_points"] * POINT_VALUE
        df["live_style_tp_price_distance"] = df["live_style_sl_price_distance"] * RISK_REWARD_RATIO
        df["live_style_tp_points"] = df["live_style_sl_points"] * RISK_REWARD_RATIO

    df["label_sl_price_distance"] = np.where(
        USE_LIVE_STYLE_SL_FOR_LABELS,
        df["live_style_sl_price_distance"],
        df["dynamic_sl_price_distance"]
    )
    df["label_tp_price_distance"] = np.where(
        USE_LIVE_STYLE_SL_FOR_LABELS,
        df["live_style_tp_price_distance"],
        df["dynamic_tp_price_distance"]
    )
    df["label_sl_points"] = df["label_sl_price_distance"] / POINT_VALUE
    df["label_tp_points"] = df["label_tp_price_distance"] / POINT_VALUE

    return df


# ========== STEP 6: LOAD ORIGINAL TIMEFRAMES ==========
file_1m = find_mt5_file("M1")
file_5m = find_mt5_file("M5")
file_15m = find_mt5_file("M15")

print(f"[FOUND] M1  -> {file_1m.name}")
print(f"[FOUND] M5  -> {file_5m.name}")
print(f"[FOUND] M15 -> {file_15m.name}")

df_1m_raw = load_mt5_file(file_1m).sort_index()
df_5m_raw = load_mt5_file(file_5m).sort_index()
df_15m_raw = load_mt5_file(file_15m).sort_index()

print("\n[RAW DATE RANGES]")
print(f"1m : {df_1m_raw.index.min()} -> {df_1m_raw.index.max()} | rows={len(df_1m_raw)}")
print(f"5m : {df_5m_raw.index.min()} -> {df_5m_raw.index.max()} | rows={len(df_5m_raw)}")
print(f"15m: {df_15m_raw.index.min()} -> {df_15m_raw.index.max()} | rows={len(df_15m_raw)}")

print("\n[FEATURES] Calculating indicators on each ORIGINAL timeframe first")
print("[FEATURES] 1m: candle features + RSI + ATR + ADX + Fractal BOS/CHoCH + OB/Breaker Blocks + FVG + Sweeps/Rejections")
print("[FEATURES] 5m: candle features + RSI + ADX + Fractal BOS/CHoCH + OB/Breaker Blocks + FVG + Sweeps/Rejections")
print("[FEATURES] 15m: candle features + ADX + Fractal BOS/CHoCH + OB/Breaker Blocks + FVG + Sweeps/Rejections")

# IMPORTANT:
# Do NOT reindex 5m/15m raw candles to the 1m index before calculating indicators.
# Indicators must be calculated on their real timeframe candles first.
df_1m_features = create_features(df_1m_raw, "1m", use_rsi=True, use_atr=True, use_adx=True)
df_5m_features = create_features(df_5m_raw, "5m", use_rsi=True, use_atr=False, use_adx=True)
df_15m_features = create_features(df_15m_raw, "15m", use_rsi=False, use_atr=False, use_adx=True)

print("[FEATURES] Adding displacement / expansion features")
df_1m_features = add_displacement_expansion_features(df_1m_features, "1m")
df_5m_features = add_displacement_expansion_features(df_5m_features, "5m")
df_15m_features = add_displacement_expansion_features(df_15m_features, "15m")

print("[FEATURES] Adding compression -> expansion features")
df_1m_features = add_compression_expansion_features(df_1m_features, "1m")
df_5m_features = add_compression_expansion_features(df_5m_features, "5m")
df_15m_features = add_compression_expansion_features(df_15m_features, "15m")

print("[FEATURES] Adding liquidity sweep / rejection features")
df_1m_features = add_liquidity_sweep_features(df_1m_features, "1m")
df_5m_features = add_liquidity_sweep_features(df_5m_features, "5m")
df_15m_features = add_liquidity_sweep_features(df_15m_features, "15m")

print("[FEATURES] Adding Pine/LuxAlgo-style fractal BOS/CHoCH market structure features")
df_1m_features = add_fractal_structure_features(df_1m_features, "1m")
df_5m_features = add_fractal_structure_features(df_5m_features, "5m")
df_15m_features = add_fractal_structure_features(df_15m_features, "15m")

print("[FEATURES] Adding LuxAlgo-style order block / breaker block features")
df_1m_features = add_order_block_features(df_1m_features, "1m")
df_5m_features = add_order_block_features(df_5m_features, "5m")
df_15m_features = add_order_block_features(df_15m_features, "15m")

print("[FEATURES] Adding LuxAlgo/ICT-style fair value gap features")
df_1m_features = add_fvg_features(df_1m_features, "1m")
df_5m_features = add_fvg_features(df_5m_features, "5m")
df_15m_features = add_fvg_features(df_15m_features, "15m")

print("[FEATURES] Adding nearest 1m OB/FVG zone boundaries for entry-style labeling")
df_1m_features = add_entry_zone_price_features(df_1m_features, "1m")


# ========== STEP 7: SAFE BACKWARD MERGE INTO 1M ROWS ==========
def prefixed_features_only(df_source: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [col for col in df_source.columns if col.startswith(f"{prefix}_")]
    return df_source[cols].copy()


# Base rows are 1m rows only. Labels and trade entries are still based on 1m.
df = prefixed_features_only(df_1m_features, "1m").copy()

# merge_asof with direction="backward" means:
# for each 1m candle at time T, use only the latest known 5m/15m feature row at or before T.
# This prevents future leakage.
df = pd.merge_asof(
    df.sort_index(),
    prefixed_features_only(df_5m_features, "5m").sort_index(),
    left_index=True,
    right_index=True,
    direction="backward"
)

df = pd.merge_asof(
    df.sort_index(),
    prefixed_features_only(df_15m_features, "15m").sort_index(),
    left_index=True,
    right_index=True,
    direction="backward"
)

print("\n[MERGE CHECK]")
print(f"Merged rows before cleaning: {len(df)}")
print(f"Merged date range: {df.index.min()} -> {df.index.max()}")
print("Used backward merge only: no future 5m/15m candles are allowed.")


# ========== STEP 8: CROSS-TIMEFRAME FEATURES ==========
# After the backward merge, all features below are aligned on the 1m index.
df["align_1m_5m_ma10"] = (
    df["1m_ma10_above_ma20"] == df["5m_ma10_above_ma20"]
).astype(int)

df["align_1m_15m_ma10"] = (
    df["1m_ma10_above_ma20"] == df["15m_ma10_above_ma20"]
).astype(int)

df["align_5m_15m_ma10"] = (
    df["5m_ma10_above_ma20"] == df["15m_ma10_above_ma20"]
).astype(int)

df["triple_trend_alignment"] = (
    (df["1m_ma10_above_ma20"] == df["5m_ma10_above_ma20"]) &
    (df["5m_ma10_above_ma20"] == df["15m_ma10_above_ma20"])
).astype(int)

df["triple_bull_alignment"] = (
    (df["1m_close_above_ma20"] == 1) &
    (df["5m_close_above_ma20"] == 1) &
    (df["15m_close_above_ma20"] == 1)
).astype(int)

df["triple_bear_alignment"] = (
    (df["1m_close_above_ma20"] == 0) &
    (df["5m_close_above_ma20"] == 0) &
    (df["15m_close_above_ma20"] == 0)
).astype(int)

df["momentum_alignment_score"] = (
    np.sign(df["1m_momentum5"]).fillna(0) +
    np.sign(df["5m_momentum5"]).fillna(0) +
    np.sign(df["15m_momentum5"]).fillna(0)
)

df["1m_vs_5m_range_ratio"] = df["1m_range_mean10"] / (df["5m_range_mean10"] + EPS)
df["1m_vs_15m_range_ratio"] = df["1m_range_mean10"] / (df["15m_range_mean10"] + EPS)

df["1m_body_vs_5m_range"] = df["1m_body"].abs() / (df["5m_range_mean10"] + EPS)
df["1m_body_vs_15m_range"] = df["1m_body"].abs() / (df["15m_range_mean10"] + EPS)

df["rsi_1m_5m_alignment"] = (
    df["1m_rsi_above_50"] + df["5m_rsi_above_50"]
)

df["rsi_1m_5m_bullish"] = (df["rsi_1m_5m_alignment"] == 2).astype(int)
df["rsi_1m_5m_bearish"] = (df["rsi_1m_5m_alignment"] == 0).astype(int)

df["adx_trend_alignment_score"] = (
    df["1m_adx_trending"] +
    df["5m_adx_trending"] +
    df["15m_adx_trending"]
)

df["adx_choppy_alignment_score"] = (
    df["1m_adx_choppy"] +
    df["5m_adx_choppy"] +
    df["15m_adx_choppy"]
)

df["market_is_trending"] = (df["15m_adx14"] > 25).astype(int)
df["market_is_choppy"] = (df["15m_adx14"] < 20).astype(int)

df["entry_trend_confirmed"] = (
    (df["1m_adx14"] > 20) &
    (df["5m_adx14"] > 20) &
    (df["15m_adx14"] > 20)
).astype(int)

df["di_direction_alignment_score"] = (
    df["1m_di_direction"].fillna(0) +
    df["5m_di_direction"].fillna(0) +
    df["15m_di_direction"].fillna(0)
)

df["rsi_adx_bull_context"] = (
    (df["rsi_1m_5m_bullish"] == 1) &
    (df["di_direction_alignment_score"] > 0) &
    (df["market_is_trending"] == 1)
).astype(int)

df["rsi_adx_bear_context"] = (
    (df["rsi_1m_5m_bearish"] == 1) &
    (df["di_direction_alignment_score"] < 0) &
    (df["market_is_trending"] == 1)
).astype(int)


# ========== FRACTAL BOS / CHoCH STRUCTURE ALIGNMENT FEATURES ==========
df["structure_bull_context_score"] = (
    df["15m_structure_direction"].clip(lower=0) +
    df["5m_structure_direction"].clip(lower=0) +
    df["1m_structure_direction"].clip(lower=0) +
    df["15m_recent_bos_up"] + df["5m_recent_bos_up"] + df["1m_recent_bos_up"] +
    df["15m_recent_choch_up"] + df["5m_recent_choch_up"]
)

df["structure_bear_context_score"] = (
    (-df["15m_structure_direction"].clip(upper=0)) +
    (-df["5m_structure_direction"].clip(upper=0)) +
    (-df["1m_structure_direction"].clip(upper=0)) +
    df["15m_recent_bos_down"] + df["5m_recent_bos_down"] + df["1m_recent_bos_down"] +
    df["15m_recent_choch_down"] + df["5m_recent_choch_down"]
)

df["structure_reversal_warning"] = (
    ((df["15m_structure_direction"] == 1) & ((df["1m_recent_choch_down"] == 1) | (df["5m_recent_choch_down"] == 1))) |
    ((df["15m_structure_direction"] == -1) & ((df["1m_recent_choch_up"] == 1) | (df["5m_recent_choch_up"] == 1)))
).astype(int)

df["structure_direction_alignment"] = (
    df["1m_structure_direction"].fillna(0) +
    df["5m_structure_direction"].fillna(0) +
    df["15m_structure_direction"].fillna(0)
)

df["structure_triple_bull_alignment"] = (df["structure_direction_alignment"] == 3).astype(int)
df["structure_triple_bear_alignment"] = (df["structure_direction_alignment"] == -3).astype(int)

df["structure_recent_break_score"] = (
    df["1m_recent_bos_up"] - df["1m_recent_bos_down"] +
    df["5m_recent_bos_up"] - df["5m_recent_bos_down"] +
    df["15m_recent_bos_up"] - df["15m_recent_bos_down"] +
    df["1m_recent_choch_up"] - df["1m_recent_choch_down"] +
    df["5m_recent_choch_up"] - df["5m_recent_choch_down"] +
    df["15m_recent_choch_up"] - df["15m_recent_choch_down"]
)



# ========== LIQUIDITY SWEEP / REJECTION ALIGNMENT FEATURES ==========
df["sweep_high_context_score"] = (
    df["1m_recent_sweep_high"] +
    df["5m_recent_sweep_high"] +
    df["15m_recent_sweep_high"]
)

df["sweep_low_context_score"] = (
    df["1m_recent_sweep_low"] +
    df["5m_recent_sweep_low"] +
    df["15m_recent_sweep_low"]
)

df["sweep_reject_high_context_score"] = (
    df["1m_recent_sweep_reject_high"] +
    df["5m_recent_sweep_reject_high"] +
    df["15m_recent_sweep_reject_high"]
)

df["sweep_reject_low_context_score"] = (
    df["1m_recent_sweep_reject_low"] +
    df["5m_recent_sweep_reject_low"] +
    df["15m_recent_sweep_reject_low"]
)

df["strong_sweep_reject_high_context_score"] = (
    df["1m_recent_strong_sweep_reject_high"] +
    df["5m_recent_strong_sweep_reject_high"] +
    df["15m_recent_strong_sweep_reject_high"]
)

df["strong_sweep_reject_low_context_score"] = (
    df["1m_recent_strong_sweep_reject_low"] +
    df["5m_recent_strong_sweep_reject_low"] +
    df["15m_recent_strong_sweep_reject_low"]
)

df["sweep_reversal_context_score"] = (
    df["1m_sweep_reversal_bias"] +
    df["5m_sweep_reversal_bias"] +
    df["15m_sweep_reversal_bias"]
)

df["strong_sweep_reversal_context_score"] = (
    df["1m_strong_sweep_reversal_bias"] +
    df["5m_strong_sweep_reversal_bias"] +
    df["15m_strong_sweep_reversal_bias"]
)

df["sweep_continuation_context_score"] = (
    df["1m_sweep_continuation_bias"] +
    df["5m_sweep_continuation_bias"] +
    df["15m_sweep_continuation_bias"]
)

df["sweep_high_atr_strength_sum"] = (
    df["1m_sweep_high_atr_strength"] +
    df["5m_sweep_high_atr_strength"] +
    df["15m_sweep_high_atr_strength"]
)

df["sweep_low_atr_strength_sum"] = (
    df["1m_sweep_low_atr_strength"] +
    df["5m_sweep_low_atr_strength"] +
    df["15m_sweep_low_atr_strength"]
)

df["sweep_high_wick_rejection_strength_sum"] = (
    df["1m_sweep_high_wick_rejection_strength"] +
    df["5m_sweep_high_wick_rejection_strength"] +
    df["15m_sweep_high_wick_rejection_strength"]
)

df["sweep_low_wick_rejection_strength_sum"] = (
    df["1m_sweep_low_wick_rejection_strength"] +
    df["5m_sweep_low_wick_rejection_strength"] +
    df["15m_sweep_low_wick_rejection_strength"]
)

df["htf_sweep_high_ltf_reject"] = (
    ((df["15m_recent_sweep_high"] == 1) | (df["5m_recent_sweep_high"] == 1)) &
    (df["1m_recent_sweep_reject_high"] == 1)
).astype(int)

df["htf_sweep_low_ltf_reject"] = (
    ((df["15m_recent_sweep_low"] == 1) | (df["5m_recent_sweep_low"] == 1)) &
    (df["1m_recent_sweep_reject_low"] == 1)
).astype(int)

df["htf_sweep_high_ltf_strong_reject"] = (
    ((df["15m_recent_sweep_high"] == 1) | (df["5m_recent_sweep_high"] == 1)) &
    (df["1m_recent_strong_sweep_reject_high"] == 1)
).astype(int)

df["htf_sweep_low_ltf_strong_reject"] = (
    ((df["15m_recent_sweep_low"] == 1) | (df["5m_recent_sweep_low"] == 1)) &
    (df["1m_recent_strong_sweep_reject_low"] == 1)
).astype(int)


# ========== ORDER BLOCK / BREAKER BLOCK ALIGNMENT FEATURES ==========
df["htf_ltf_bull_ob_alignment"] = (
    (df["15m_inside_bull_ob"] == 1) &
    (df["1m_inside_bull_ob"] == 1)
).astype(int)

df["htf_ltf_bear_ob_alignment"] = (
    (df["15m_inside_bear_ob"] == 1) &
    (df["1m_inside_bear_ob"] == 1)
).astype(int)

df["htf_ltf_bull_breaker_alignment"] = (
    (df["15m_inside_bear_breaker"] == 1) &
    (df["1m_inside_bull_ob"] == 1)
).astype(int)

df["htf_ltf_bear_breaker_alignment"] = (
    (df["15m_inside_bull_breaker"] == 1) &
    (df["1m_inside_bear_ob"] == 1)
).astype(int)

df["ob_bull_context_score"] = (
    df["15m_inside_bull_ob"] +
    df["5m_inside_bull_ob"] +
    df["1m_inside_bull_ob"] +
    df["15m_inside_bear_breaker"] +
    df["5m_inside_bear_breaker"]
)

df["ob_bear_context_score"] = (
    df["15m_inside_bear_ob"] +
    df["5m_inside_bear_ob"] +
    df["1m_inside_bear_ob"] +
    df["15m_inside_bull_breaker"] +
    df["5m_inside_bull_breaker"]
)

df["near_15m_bull_ob"] = (df["15m_dist_to_bull_ob"] <= 0.0015).astype(int)
df["near_15m_bear_ob"] = (df["15m_dist_to_bear_ob"] <= 0.0015).astype(int)
df["near_1m_bull_ob"] = (df["1m_dist_to_bull_ob"] <= 0.0007).astype(int)
df["near_1m_bear_ob"] = (df["1m_dist_to_bear_ob"] <= 0.0007).astype(int)


# ========== FAIR VALUE GAP ALIGNMENT FEATURES ==========
df["htf_ltf_bull_fvg_alignment"] = (
    (df["15m_inside_bull_fvg"] == 1) &
    (df["1m_inside_bull_fvg"] == 1)
).astype(int)

df["htf_ltf_bear_fvg_alignment"] = (
    (df["15m_inside_bear_fvg"] == 1) &
    (df["1m_inside_bear_fvg"] == 1)
).astype(int)

df["fvg_bull_context_score"] = (
    df["15m_inside_bull_fvg"] +
    df["5m_inside_bull_fvg"] +
    df["1m_inside_bull_fvg"] +
    df["15m_recent_bull_fvg"] +
    df["5m_recent_bull_fvg"]
)

df["fvg_bear_context_score"] = (
    df["15m_inside_bear_fvg"] +
    df["5m_inside_bear_fvg"] +
    df["1m_inside_bear_fvg"] +
    df["15m_recent_bear_fvg"] +
    df["5m_recent_bear_fvg"]
)

df["near_15m_bull_fvg"] = (df["15m_dist_to_bull_fvg"] <= 0.0015).astype(int)
df["near_15m_bear_fvg"] = (df["15m_dist_to_bear_fvg"] <= 0.0015).astype(int)
df["near_1m_bull_fvg"] = (df["1m_dist_to_bull_fvg"] <= 0.0007).astype(int)
df["near_1m_bear_fvg"] = (df["1m_dist_to_bear_fvg"] <= 0.0007).astype(int)

df["ob_fvg_bull_confluence"] = (
    (df["ob_bull_context_score"] > 0) &
    (df["fvg_bull_context_score"] > 0)
).astype(int)

df["ob_fvg_bear_confluence"] = (
    (df["ob_bear_context_score"] > 0) &
    (df["fvg_bear_context_score"] > 0)
).astype(int)


# ========== HIGHER TIMEFRAME BIAS / TREND STRENGTH FEATURES ==========
df["htf_adx_strength_mean"] = (
    df["1m_adx14"] +
    df["5m_adx14"] +
    df["15m_adx14"]
) / 3

df["htf_adx_strength_max"] = pd.concat([
    df["1m_adx14"],
    df["5m_adx14"],
    df["15m_adx14"]
], axis=1).max(axis=1)

df["htf_adx_strength_min"] = pd.concat([
    df["1m_adx14"],
    df["5m_adx14"],
    df["15m_adx14"]
], axis=1).min(axis=1)

df["1m_directional_adx"] = df["1m_di_direction"].fillna(0) * df["1m_adx14"]
df["5m_directional_adx"] = df["5m_di_direction"].fillna(0) * df["5m_adx14"]
df["15m_directional_adx"] = df["15m_di_direction"].fillna(0) * df["15m_adx14"]

df["htf_directional_adx_score"] = (
    df["1m_directional_adx"] +
    df["5m_directional_adx"] +
    df["15m_directional_adx"]
) / 3

df["ma_bull_alignment_score"] = (
    df["1m_ma10_above_ma20"] +
    df["5m_ma10_above_ma20"] +
    df["15m_ma10_above_ma20"]
)

df["ma_directional_bias_score"] = (
    (df["1m_ma10_above_ma20"] * 2 - 1) +
    (df["5m_ma10_above_ma20"] * 2 - 1) +
    (df["15m_ma10_above_ma20"] * 2 - 1)
)

df["ma20_slope_strength_mean"] = (
    df["1m_ma20_slope"].abs() +
    df["5m_ma20_slope"].abs() +
    df["15m_ma20_slope"].abs()
) / 3

df["ma20_directional_slope_score"] = (
    np.sign(df["1m_ma20_slope"]).fillna(0) +
    np.sign(df["5m_ma20_slope"]).fillna(0) +
    np.sign(df["15m_ma20_slope"]).fillna(0)
)

df["bull_bias_strength"] = (
    (df["ma_directional_bias_score"].clip(lower=0) / 3) *
    (df["htf_adx_strength_mean"] / 50) *
    (df["ma20_slope_strength_mean"] * 1000 + 1)
)

df["bear_bias_strength"] = (
    ((-df["ma_directional_bias_score"].clip(upper=0)) / 3) *
    (df["htf_adx_strength_mean"] / 50) *
    (df["ma20_slope_strength_mean"] * 1000 + 1)
)

df["mixed_or_weak_trend"] = (
    (df["adx_trend_alignment_score"] <= 1) |
    (df["ma_bull_alignment_score"].between(1, 2))
).astype(int)




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

# ========== MUST-ADD CONTEXT FEATURES ==========
print("[FEATURES] Adding must-have context features: cyclical time + sessions + daily position + volatility z-score")
df = add_must_have_context_features(df, df_1m_raw)

# ========== TRUE SESSION-BASED HTF LIQUIDITY FEATURES ==========
# Keeps the existing rolling daily context above, then adds exact previous
# daily/weekly/monthly high-low levels like the Pine Script D/W/M logic.
print("[FEATURES] Adding true session-based previous daily/weekly/monthly liquidity levels")
df = add_htf_liquidity_level_features(df, df_1m_raw)

# ========== ICT SESSION LIQUIDITY FEATURES ==========
print("[FEATURES] Adding ICT session liquidity features: session highs/lows + sweeps + killzone interactions")
df = add_session_liquidity_features(df, df_1m_raw)

print("[FEATURES] Adding advanced liquidity interactions: strength + decay + weighted session + OB/FVG fusion")
df = add_advanced_liquidity_interaction_features(df)

print("[FEATURES] Adding multi-timeframe sequence awareness: rolling structure + sweep + displacement memory")
df = add_multitimeframe_sequence_awareness_features(df)

print("[FEATURES] Adding do-not-trade intelligence: conflict + chop + uncertainty features")
df = add_do_not_trade_intelligence_features(df)
# ========== STEP 9: SPREAD INFO ==========
df["spread_points"] = df_1m_raw["spread_points"]
df["spread_price"] = df["spread_points"] * POINT_VALUE
df["spread_return"] = df["spread_price"] / df_1m_raw["close"]


# ========== STEP 10: DYNAMIC SL/TP PLAN ==========
df = add_dynamic_sl_tp_plan(df)


# ========== STEP 11: CREATE DIRECTION + ENTRY STYLE LABELS ==========
# Old directional return is still saved for comparison/debugging.
df["future_return"] = df_1m_raw["close"].shift(-FUTURE_SHIFT) / df_1m_raw["close"] - 1


def label_direction(x: float) -> int:
    if x > RETURN_THRESHOLD:
        return 1
    elif x < -RETURN_THRESHOLD:
        return -1
    return 0


df["old_direction_target"] = df["future_return"].apply(label_direction)

print(f"[LABELS] Creating entry-style target using {'LIVE-STYLE' if USE_LIVE_STYLE_SL_FOR_LABELS else 'ATR-DYNAMIC'} SL/TP: NOW vs WAIT_FVG vs WAIT_OB vs NO_TRADE")
df = add_entry_style_labels(df, df_1m_raw)


# ========== STEP 12: CLEAN ==========
rows_before_clean = len(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# IMPORTANT FIX:
# Do NOT use global df.dropna() here.
# The entry-style upgrade creates zone boundary columns such as:
#   1m_nearest_bull_fvg_top, 1m_nearest_bear_ob_btm, etc.
# These are naturally NaN when there is no active zone nearby.
# If we drop every row with any NaN, the dataset collapses massively.

# 1) Zone boundary NaNs mean "no active zone", so keep the row and encode as 0.
zone_boundary_cols = [
    col for col in df.columns
    if col.startswith("1m_nearest_")
]
if zone_boundary_cols:
    df[zone_boundary_cols] = df[zone_boundary_cols].fillna(0.0)

# 2) Candidate return NaNs mean that specific entry type did not trigger.
# Keep the row, but encode the missing candidate as 0.
# These columns should be excluded from training later anyway because they are label/debug columns.
candidate_return_cols = [
    "ret_buy_now",
    "ret_buy_wait_fvg",
    "ret_buy_wait_ob",
    "ret_sell_now",
    "ret_sell_wait_fvg",
    "ret_sell_wait_ob",
]
candidate_return_cols = [col for col in candidate_return_cols if col in df.columns]
if candidate_return_cols:
    df[candidate_return_cols] = df[candidate_return_cols].fillna(0.0)

# 3) Text labels should not cause row drops.
if "entry_style_name" in df.columns:
    df["entry_style_name"] = df["entry_style_name"].fillna("NO_TRADE")

# 4) Drop only rows that are missing truly critical values.
# These are required for model inputs and/or labels.
critical_cols = [
    "1m_atr14",
    "5m_adx14",
    "15m_adx14",
    "spread_return",
    "dynamic_sl_price_distance",
    "dynamic_tp_price_distance",
    "live_style_sl_price_distance",
    "live_style_tp_price_distance",
    "label_sl_price_distance",
    "label_tp_price_distance",
    "target_entry_style",
    "target_direction",
    "target",
]
critical_cols = [col for col in critical_cols if col in df.columns]
df.dropna(subset=critical_cols, inplace=True)

# 5) For remaining numeric feature NaNs caused by rolling indicators or early candles,
# forward-fill first, then fill any remaining startup gaps with 0.
# This preserves rows instead of deleting almost the whole dataset.
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].ffill().fillna(0.0)

# 6) Final safety cleanup.
df.replace([np.inf, -np.inf], 0.0, inplace=True)
df.sort_index(inplace=True)
rows_after_clean = len(df)


# ========== STEP 13: SAVE ==========
output_path = SCRIPT_DIR / "training_dataset.csv"
df.to_csv(output_path)

print("\nDataset created successfully!")
print(f"Saved to: {output_path}")
print(f"Shape: {df.shape}")
print(f"Rows dropped during cleaning: {rows_before_clean - rows_after_clean}")

print("\nFinal dataset date range:")
print(f"{df.index.min()} -> {df.index.max()}")

print("\nEntry-style target distribution:")
print(df["target_entry_style"].value_counts().sort_index())

print("\nEntry-style target distribution (%):")
print((df["target_entry_style"].value_counts(normalize=True).sort_index() * 100).round(2))

print("\nEntry-style names:")
print(df["entry_style_name"].value_counts())

print("\nOld direction-only target distribution:")
print(df["old_direction_target"].value_counts().sort_index())

print("\nFractal BOS / CHoCH market structure feature summary:")
structure_summary_cols = [
    "1m_bos_up", "1m_bos_down", "1m_choch_up", "1m_choch_down",
    "5m_bos_up", "5m_bos_down", "5m_choch_up", "5m_choch_down",
    "15m_bos_up", "15m_bos_down", "15m_choch_up", "15m_choch_down",
    "1m_bull_fractal_confirmed", "1m_bear_fractal_confirmed",
    "5m_bull_fractal_confirmed", "5m_bear_fractal_confirmed",
    "15m_bull_fractal_confirmed", "15m_bear_fractal_confirmed",
    "structure_bull_context_score", "structure_bear_context_score",
    "structure_reversal_warning", "structure_triple_bull_alignment", "structure_triple_bear_alignment"
]
print(df[structure_summary_cols].sum().sort_values(ascending=False))

print("\nOrder Block / Breaker Block feature summary:")
ob_summary_cols = [
    "1m_inside_bull_ob", "1m_inside_bear_ob",
    "5m_inside_bull_ob", "5m_inside_bear_ob",
    "15m_inside_bull_ob", "15m_inside_bear_ob",
    "1m_inside_bull_breaker", "1m_inside_bear_breaker",
    "15m_inside_bull_breaker", "15m_inside_bear_breaker",
    "htf_ltf_bull_ob_alignment", "htf_ltf_bear_ob_alignment",
    "ob_bull_context_score", "ob_bear_context_score"
]
print(df[ob_summary_cols].sum().sort_values(ascending=False))

print("\nFair Value Gap feature summary:")
fvg_summary_cols = [
    "1m_bull_fvg_detected", "1m_bear_fvg_detected",
    "5m_bull_fvg_detected", "5m_bear_fvg_detected",
    "15m_bull_fvg_detected", "15m_bear_fvg_detected",
    "1m_inside_bull_fvg", "1m_inside_bear_fvg",
    "5m_inside_bull_fvg", "5m_inside_bear_fvg",
    "15m_inside_bull_fvg", "15m_inside_bear_fvg",
    "htf_ltf_bull_fvg_alignment", "htf_ltf_bear_fvg_alignment",
    "fvg_bull_context_score", "fvg_bear_context_score",
    "ob_fvg_bull_confluence", "ob_fvg_bear_confluence"
]
print(df[fvg_summary_cols].sum().sort_values(ascending=False))


print("\nLiquidity Sweep / Rejection feature summary:")
sweep_summary_cols = [
    "1m_sweep_high", "1m_sweep_low", "1m_sweep_reject_high", "1m_sweep_reject_low",
    "5m_sweep_high", "5m_sweep_low", "5m_sweep_reject_high", "5m_sweep_reject_low",
    "15m_sweep_high", "15m_sweep_low", "15m_sweep_reject_high", "15m_sweep_reject_low",
    "1m_recent_sweep_high", "1m_recent_sweep_low",
    "1m_recent_sweep_reject_high", "1m_recent_sweep_reject_low",
    "sweep_high_context_score", "sweep_low_context_score",
    "sweep_reject_high_context_score", "sweep_reject_low_context_score",
    "htf_sweep_high_ltf_reject", "htf_sweep_low_ltf_reject",
    "1m_strong_sweep_reject_high", "1m_strong_sweep_reject_low",
    "5m_strong_sweep_reject_high", "5m_strong_sweep_reject_low",
    "15m_strong_sweep_reject_high", "15m_strong_sweep_reject_low",
    "strong_sweep_reject_high_context_score", "strong_sweep_reject_low_context_score",
    "sweep_high_atr_strength_sum", "sweep_low_atr_strength_sum",
    "sweep_high_wick_rejection_strength_sum", "sweep_low_wick_rejection_strength_sum",
    "htf_sweep_high_ltf_strong_reject", "htf_sweep_low_ltf_strong_reject"
]
sweep_summary_cols = [col for col in sweep_summary_cols if col in df.columns]
print(df[sweep_summary_cols].sum().sort_values(ascending=False))

print("\nTrue Session-Based HTF Liquidity feature summary:")
htf_liquidity_summary_cols = [
    "near_prev_daily_high", "near_prev_daily_low",
    "near_prev_weekly_high", "near_prev_weekly_low",
    "near_prev_monthly_high", "near_prev_monthly_low",
    "swept_prev_daily_high", "swept_prev_daily_low",
    "swept_prev_weekly_high", "swept_prev_weekly_low",
    "swept_prev_monthly_high", "swept_prev_monthly_low",
    "reject_prev_daily_high", "reject_prev_daily_low",
    "reject_prev_weekly_high", "reject_prev_weekly_low",
    "reject_prev_monthly_high", "reject_prev_monthly_low",
    "reclaim_prev_daily_high", "reclaim_prev_daily_low",
    "reclaim_prev_weekly_high", "reclaim_prev_weekly_low",
    "reclaim_prev_monthly_high", "reclaim_prev_monthly_low",
    "htf_liquidity_near_high_score", "htf_liquidity_near_low_score",
    "htf_liquidity_sweep_high_score", "htf_liquidity_sweep_low_score",
    "htf_liquidity_reject_high_score", "htf_liquidity_reject_low_score",
    "htf_liquidity_reversal_bias", "htf_liquidity_continuation_bias",
    "daily_weekly_liquidity_confluence_high", "daily_weekly_liquidity_confluence_low",
]
htf_liquidity_summary_cols = [col for col in htf_liquidity_summary_cols if col in df.columns]
print(df[htf_liquidity_summary_cols].sum().sort_values(ascending=False))

print("\nICT Session Liquidity feature summary:")
session_liquidity_summary_cols = [
    "in_asia_session_liquidity_window", "in_london_session_liquidity_window", "in_ny_session_liquidity_window",
    "in_asia_killzone", "in_london_killzone", "in_ny_killzone", "in_london_ny_overlap",
    "near_prev_asia_session_high", "near_prev_asia_session_low",
    "near_prev_london_session_high", "near_prev_london_session_low",
    "near_prev_ny_session_high", "near_prev_ny_session_low",
    "swept_prev_asia_session_high", "swept_prev_asia_session_low",
    "swept_prev_london_session_high", "swept_prev_london_session_low",
    "swept_prev_ny_session_high", "swept_prev_ny_session_low",
    "reject_prev_asia_session_high", "reject_prev_asia_session_low",
    "reject_prev_london_session_high", "reject_prev_london_session_low",
    "reject_prev_ny_session_high", "reject_prev_ny_session_low",
    "london_swept_asia_high", "london_swept_asia_low",
    "london_reject_asia_high", "london_reject_asia_low",
    "ny_swept_london_high", "ny_swept_london_low",
    "ny_reject_london_high", "ny_reject_london_low",
    "ny_swept_asia_high", "ny_swept_asia_low",
    "ny_reject_asia_high", "ny_reject_asia_low",
    "session_liquidity_near_high_score", "session_liquidity_near_low_score",
    "session_liquidity_sweep_high_score", "session_liquidity_sweep_low_score",
    "session_liquidity_reject_high_score", "session_liquidity_reject_low_score",
    "session_liquidity_reversal_bias", "session_liquidity_continuation_bias",
    "london_asia_sweep_reversal_bias", "ny_london_sweep_reversal_bias", "ny_asia_sweep_reversal_bias",
    "killzone_session_sweep_score",
]
session_liquidity_summary_cols = [col for col in session_liquidity_summary_cols if col in df.columns]
print(df[session_liquidity_summary_cols].sum().sort_values(ascending=False))

print("\nDynamic ATR SL/TP summary:")
print(df[[
    "sl_atr_multiplier",
    "dynamic_sl_points",
    "dynamic_tp_points",
    "dynamic_sl_price_distance",
    "dynamic_tp_price_distance",
    "sl_distance_pct",
    "tp_distance_pct"
]].describe())

print("\nLive-style label SL/TP summary:")
print(df[[
    "live_style_sim_balance",
    "live_style_lot_size",
    "live_style_risk_usd",
    "live_style_reward_usd",
    "live_style_sl_points",
    "live_style_tp_points",
    "live_style_sl_price_distance",
    "live_style_tp_price_distance",
    "label_sl_points",
    "label_tp_points"
]].describe())
print(f"\nLabels used live-style SL/TP: {USE_LIVE_STYLE_SL_FOR_LABELS}")

print("\nSpread summary:")
print(df[["spread_points", "spread_price", "spread_return"]].describe())

print("\nPreview:")
print(df.head())
