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
    sl_dist = out["dynamic_sl_price_distance"].to_numpy(float)
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
        if start_j is None or not np.isfinite(entry_price) or not np.isfinite(sl_dist[i]) or sl_dist[i] <= 0:
            return np.nan
        sl = sl_dist[i]
        tp = sl * RISK_REWARD_RATIO
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
print("[FEATURES] 1m: candle features + RSI + ATR + ADX + Fractal BOS/CHoCH + OB/Breaker Blocks + FVG")
print("[FEATURES] 5m: candle features + RSI + ADX + Fractal BOS/CHoCH + OB/Breaker Blocks + FVG")
print("[FEATURES] 15m: candle features + ADX + Fractal BOS/CHoCH + OB/Breaker Blocks + FVG")

# IMPORTANT:
# Do NOT reindex 5m/15m raw candles to the 1m index before calculating indicators.
# Indicators must be calculated on their real timeframe candles first.
df_1m_features = create_features(df_1m_raw, "1m", use_rsi=True, use_atr=True, use_adx=True)
df_5m_features = create_features(df_5m_raw, "5m", use_rsi=True, use_atr=False, use_adx=True)
df_15m_features = create_features(df_15m_raw, "15m", use_rsi=False, use_atr=False, use_adx=True)

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

print("[LABELS] Creating entry-style target: NOW vs WAIT_FVG vs WAIT_OB vs NO_TRADE")
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
