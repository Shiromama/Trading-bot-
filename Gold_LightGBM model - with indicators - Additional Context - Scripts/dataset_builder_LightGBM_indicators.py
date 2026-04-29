import pandas as pd
import numpy as np
from pathlib import Path

# ========== STEP 0: SCRIPT DIRECTORY ==========
SCRIPT_DIR = Path(__file__).resolve().parent
print(f"Script directory: {SCRIPT_DIR}")

# BTCUSDm has Digits = 2, so 1 point = 0.01
POINT_VALUE = 0.01
EPS = 1e-9

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


# ========== STEP 5: DYNAMIC SL/TP PLANNER ==========
def add_dynamic_sl_tp_plan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Base SL from 1m ATR
    base_sl_price_distance = df["1m_atr14"] * BASE_ATR_SL_MULTIPLIER

    # If market is choppy, widen SL a bit
    # If trend is confirmed, keep SL closer
    df["sl_atr_multiplier"] = BASE_ATR_SL_MULTIPLIER

    df.loc[df["market_is_choppy"] == 1, "sl_atr_multiplier"] += 0.3
    df.loc[df["entry_trend_confirmed"] == 1, "sl_atr_multiplier"] -= 0.2

    # If current 1m range is unusually large, widen SL
    df.loc[df["1m_range_vs_atr14"] > 1.5, "sl_atr_multiplier"] += 0.2

    df["sl_atr_multiplier"] = df["sl_atr_multiplier"].clip(
        lower=MIN_ATR_SL_MULTIPLIER,
        upper=MAX_ATR_SL_MULTIPLIER
    )

    df["dynamic_sl_price_distance"] = df["1m_atr14"] * df["sl_atr_multiplier"]

    # Convert to points and clamp
    df["dynamic_sl_points"] = df["dynamic_sl_price_distance"] / POINT_VALUE
    df["dynamic_sl_points"] = df["dynamic_sl_points"].clip(
        lower=MIN_SL_POINTS,
        upper=MAX_SL_POINTS
    )

    # Recalculate price distance after clipping
    df["dynamic_sl_price_distance"] = df["dynamic_sl_points"] * POINT_VALUE

    # TP is always 2x SL
    df["risk_reward_ratio"] = RISK_REWARD_RATIO
    df["dynamic_tp_price_distance"] = df["dynamic_sl_price_distance"] * RISK_REWARD_RATIO
    df["dynamic_tp_points"] = df["dynamic_sl_points"] * RISK_REWARD_RATIO

    # Useful extra columns for future sim/live checks
    df["sl_distance_pct"] = df["dynamic_sl_price_distance"] / (df["1m_ma20"] + EPS)
    df["tp_distance_pct"] = df["dynamic_tp_price_distance"] / (df["1m_ma20"] + EPS)

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
print("[FEATURES] 1m: candle features + RSI + ATR + ADX")
print("[FEATURES] 5m: candle features + RSI + ADX")
print("[FEATURES] 15m: candle features + ADX only")

# IMPORTANT:
# Do NOT reindex 5m/15m raw candles to the 1m index before calculating indicators.
# Indicators must be calculated on their real timeframe candles first.
df_1m_features = create_features(df_1m_raw, "1m", use_rsi=True, use_atr=True, use_adx=True)
df_5m_features = create_features(df_5m_raw, "5m", use_rsi=True, use_atr=False, use_adx=True)
df_15m_features = create_features(df_15m_raw, "15m", use_rsi=False, use_atr=False, use_adx=True)


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


# ========== STEP 9: SPREAD INFO ==========
df["spread_points"] = df_1m_raw["spread_points"]
df["spread_price"] = df["spread_points"] * POINT_VALUE
df["spread_return"] = df["spread_price"] / df_1m_raw["close"]


# ========== STEP 10: DYNAMIC SL/TP PLAN ==========
df = add_dynamic_sl_tp_plan(df)


# ========== STEP 11: CREATE DIRECTION LABELS ==========
df["future_return"] = df_1m_raw["close"].shift(-FUTURE_SHIFT) / df_1m_raw["close"] - 1


def label(x: float) -> int:
    if x > RETURN_THRESHOLD:
        return 1
    elif x < -RETURN_THRESHOLD:
        return -1
    return 0


df["target"] = df["future_return"].apply(label)


# ========== STEP 12: CLEAN ==========
rows_before_clean = len(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
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

print("\nTarget distribution:")
print(df["target"].value_counts().sort_index())

print("\nTarget distribution (%):")
print((df["target"].value_counts(normalize=True).sort_index() * 100).round(2))

print("\nDynamic SL/TP summary:")
print(df[[
    "sl_atr_multiplier",
    "dynamic_sl_points",
    "dynamic_tp_points",
    "dynamic_sl_price_distance",
    "dynamic_tp_price_distance",
    "sl_distance_pct",
    "tp_distance_pct"
]].describe())

print("\nSpread summary:")
print(df[["spread_points", "spread_price", "spread_return"]].describe())

print("\nPreview:")
print(df.head())
