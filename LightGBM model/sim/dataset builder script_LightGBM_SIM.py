import pandas as pd
import numpy as np
from pathlib import Path

# ========== STEP 0: SCRIPT DIRECTORY ==========
SCRIPT_DIR = Path(__file__).resolve().parent
print(f"Script directory: {SCRIPT_DIR}")

# BTCUSDm has Digits = 2, so 1 point = 0.01
POINT_VALUE = 0.01
EPS = 1e-9

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


file_1m = find_mt5_file("M1")
file_5m = find_mt5_file("M5")
file_15m = find_mt5_file("M15")

print(f"[FOUND] M1  -> {file_1m.name}")
print(f"[FOUND] M5  -> {file_5m.name}")
print(f"[FOUND] M15 -> {file_15m.name}")

df_1m = load_mt5_file(file_1m)
df_5m = load_mt5_file(file_5m)
df_15m = load_mt5_file(file_15m)

# ========== STEP 3: ALIGN TIMEFRAMES ==========
df_5m = df_5m.reindex(df_1m.index, method="ffill")
df_15m = df_15m.reindex(df_1m.index, method="ffill")


# ========== STEP 4: FEATURE ENGINEERING ==========
def create_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()

    # ---------- Basic candle features ----------
    df[f"{prefix}_return"] = df["close"].pct_change()
    df[f"{prefix}_log_return"] = np.log(df["close"] / df["close"].shift(1))

    df[f"{prefix}_range"] = df["high"] - df["low"]
    df[f"{prefix}_body"] = df["close"] - df["open"]

    df[f"{prefix}_upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df[f"{prefix}_lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Safe denominator
    range_safe = df[f"{prefix}_range"].replace(0, np.nan)

    # ---------- Candle ratios ----------
    df[f"{prefix}_body_ratio"] = df[f"{prefix}_body"] / (range_safe + EPS)
    df[f"{prefix}_upper_wick_ratio"] = df[f"{prefix}_upper_wick"] / (range_safe + EPS)
    df[f"{prefix}_lower_wick_ratio"] = df[f"{prefix}_lower_wick"] / (range_safe + EPS)

    df[f"{prefix}_close_pos_in_range"] = (
        (df["close"] - df["low"]) / (range_safe + EPS)
    )
    df[f"{prefix}_open_pos_in_range"] = (
        (df["open"] - df["low"]) / (range_safe + EPS)
    )

    # ---------- Candle direction ----------
    df[f"{prefix}_is_bull"] = (df["close"] > df["open"]).astype(int)
    df[f"{prefix}_is_bear"] = (df["close"] < df["open"]).astype(int)

    # ---------- Moving averages ----------
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

    # ---------- Volatility ----------
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

    # Candle-to-candle change
    df[f"{prefix}_range_change1"] = (
        df[f"{prefix}_range"] / (df[f"{prefix}_range"].shift(1) + EPS)
    )
    df[f"{prefix}_body_change1"] = (
        df[f"{prefix}_body"].abs() / (df[f"{prefix}_body"].shift(1).abs() + EPS)
    )

    # ---------- Volume ----------
    df[f"{prefix}_volume_change"] = df["volume"].pct_change()
    df[f"{prefix}_volume_mean10"] = df["volume"].rolling(10).mean()
    df[f"{prefix}_volume_vs_mean10"] = (
        df["volume"] / (df[f"{prefix}_volume_mean10"] + EPS)
    )

    # ---------- Momentum ----------
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

    # ---------- Smoothed return context ----------
    df[f"{prefix}_return_mean3"] = df[f"{prefix}_return"].rolling(3).mean()
    df[f"{prefix}_return_mean5"] = df[f"{prefix}_return"].rolling(5).mean()
    df[f"{prefix}_return_std5"] = df[f"{prefix}_return"].rolling(5).std()
    df[f"{prefix}_return_std10"] = df[f"{prefix}_return"].rolling(10).std()

    # ---------- Scalp context ----------
    df[f"{prefix}_body_vs_range_mean10"] = (
        df[f"{prefix}_body"].abs() / (df[f"{prefix}_range_mean10"] + EPS)
    )
    df[f"{prefix}_wick_imbalance"] = (
        df[f"{prefix}_lower_wick"] - df[f"{prefix}_upper_wick"]
    ) / (range_safe + EPS)

    return df


df_1m = create_features(df_1m, "1m")
df_5m = create_features(df_5m, "5m")
df_15m = create_features(df_15m, "15m")


# ========== STEP 5: MERGE FEATURES ==========
df = pd.DataFrame(index=df_1m.index)

for col in df_1m.columns:
    if col.startswith("1m_"):
        df[col] = df_1m[col]

for col in df_5m.columns:
    if col.startswith("5m_"):
        df[col] = df_5m[col]

for col in df_15m.columns:
    if col.startswith("15m_"):
        df[col] = df_15m[col]


# ========== STEP 6: CROSS-TIMEFRAME FEATURES ==========
# Trend alignment
df["align_1m_5m_ma10"] = (
    (df_1m["1m_ma10_above_ma20"] == df_5m["5m_ma10_above_ma20"]).astype(int)
)
df["align_1m_15m_ma10"] = (
    (df_1m["1m_ma10_above_ma20"] == df_15m["15m_ma10_above_ma20"]).astype(int)
)
df["align_5m_15m_ma10"] = (
    (df_5m["5m_ma10_above_ma20"] == df_15m["15m_ma10_above_ma20"]).astype(int)
)

df["triple_trend_alignment"] = (
    (df_1m["1m_ma10_above_ma20"] == df_5m["5m_ma10_above_ma20"]) &
    (df_5m["5m_ma10_above_ma20"] == df_15m["15m_ma10_above_ma20"])
).astype(int)

# Multi-timeframe directional agreement
df["triple_bull_alignment"] = (
    (df_1m["1m_close_above_ma20"] == 1) &
    (df_5m["5m_close_above_ma20"] == 1) &
    (df_15m["15m_close_above_ma20"] == 1)
).astype(int)

df["triple_bear_alignment"] = (
    (df_1m["1m_close_above_ma20"] == 0) &
    (df_5m["5m_close_above_ma20"] == 0) &
    (df_15m["15m_close_above_ma20"] == 0)
).astype(int)

# Momentum agreement
df["momentum_alignment_score"] = (
    np.sign(df_1m["1m_momentum5"]).fillna(0) +
    np.sign(df_5m["5m_momentum5"]).fillna(0) +
    np.sign(df_15m["15m_momentum5"]).fillna(0)
)

# Volatility regime relation
df["1m_vs_5m_range_ratio"] = df_1m["1m_range_mean10"] / (df_5m["5m_range_mean10"] + EPS)
df["1m_vs_15m_range_ratio"] = df_1m["1m_range_mean10"] / (df_15m["15m_range_mean10"] + EPS)

# Short-term impulse vs higher timeframe context
df["1m_body_vs_5m_range"] = df_1m["1m_body"].abs() / (df_5m["5m_range_mean10"] + EPS)
df["1m_body_vs_15m_range"] = df_1m["1m_body"].abs() / (df_15m["15m_range_mean10"] + EPS)


# ========== STEP 7: ADD SPREAD INFO ==========
# Keep spread only for evaluation, not as a model feature
df["spread_points"] = df_1m["spread_points"]
df["spread_price"] = df["spread_points"] * POINT_VALUE
df["spread_return"] = df["spread_price"] / df_1m["close"]


# ========== STEP 8: CREATE LABELS ==========
future_shift = 15
threshold = 0.0015  # 0.15%

df["future_return"] = df_1m["close"].shift(-future_shift) / df_1m["close"] - 1

def label(x: float) -> int:
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0

df["target"] = df["future_return"].apply(label)


# ========== STEP 9: CLEAN ==========
df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
df.dropna(inplace=True)


# ========== STEP 10: SAVE ==========
output_path = SCRIPT_DIR / "training_dataset.csv"
df.to_csv(output_path)

print("\nDataset created successfully!")
print(f"Saved to: {output_path}")
print(f"Shape: {df.shape}")

print("\nTarget distribution:")
print(df["target"].value_counts())

print("\nSpread summary:")
print(df[["spread_points", "spread_price", "spread_return"]].describe())

print("\nPreview:")
print(df.head())