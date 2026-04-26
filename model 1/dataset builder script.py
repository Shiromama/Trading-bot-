import pandas as pd
from pathlib import Path

# ========== STEP 0: SCRIPT DIRECTORY ==========
SCRIPT_DIR = Path(__file__).resolve().parent
print(f"Script directory: {SCRIPT_DIR}")

# BTCUSDm has Digits = 2, so 1 point = 0.01
POINT_VALUE = 0.01

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
        '<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>',
        '<TICKVOL>', '<SPREAD>'
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}\n"
            f"Detected columns: {list(df.columns)}"
        )

    df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    df.set_index('datetime', inplace=True)

    df.rename(columns={
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'volume',
        '<SPREAD>': 'spread_points'
    }, inplace=True)

    return df[['open', 'high', 'low', 'close', 'volume', 'spread_points']].copy()

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
df_5m = df_5m.reindex(df_1m.index, method='ffill')
df_15m = df_15m.reindex(df_1m.index, method='ffill')

# ========== STEP 4: FEATURE ENGINEERING ==========
def create_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()

    # Basic price-action features
    df[f'{prefix}_return'] = df['close'].pct_change()
    df[f'{prefix}_range'] = df['high'] - df['low']
    df[f'{prefix}_body'] = df['close'] - df['open']

    # Wick features
    df[f'{prefix}_upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df[f'{prefix}_lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # Trend context
    df[f'{prefix}_ma10'] = df['close'].rolling(10).mean()
    df[f'{prefix}_ma20'] = df['close'].rolling(20).mean()
    df[f'{prefix}_ma10_dist'] = (df['close'] - df[f'{prefix}_ma10']) / df[f'{prefix}_ma10']
    df[f'{prefix}_ma20_dist'] = (df['close'] - df[f'{prefix}_ma20']) / df[f'{prefix}_ma20']

    # Volatility
    df[f'{prefix}_volatility10'] = df['close'].rolling(10).std()

    # Volume context
    df[f'{prefix}_volume_change'] = df['volume'].pct_change()

    return df

df_1m = create_features(df_1m, '1m')
df_5m = create_features(df_5m, '5m')
df_15m = create_features(df_15m, '15m')

# ========== STEP 5: MERGE FEATURES ==========
df = pd.DataFrame(index=df_1m.index)

for col in df_1m.columns:
    if col.startswith('1m_'):
        df[col] = df_1m[col]

for col in df_5m.columns:
    if col.startswith('5m_'):
        df[col] = df_5m[col]

for col in df_15m.columns:
    if col.startswith('15m_'):
        df[col] = df_15m[col]

# ========== STEP 6: ADD SPREAD INFO ==========
# Keep spread only for evaluation, not as a model feature for now
df['spread_points'] = df_1m['spread_points']
df['spread_price'] = df['spread_points'] * POINT_VALUE
df['spread_return'] = df['spread_price'] / df_1m['close']

# ========== STEP 7: CREATE LABELS ==========
future_shift = 15
threshold = 0.0015  # 0.15%

df['future_return'] = df_1m['close'].shift(-future_shift) / df_1m['close'] - 1

def label(x: float) -> int:
    if x > threshold:
        return 1
    elif x < -threshold:
        return -1
    else:
        return 0

df['target'] = df['future_return'].apply(label)

# ========== STEP 8: CLEAN ==========
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# ========== STEP 9: SAVE ==========
output_path = SCRIPT_DIR / "training_dataset.csv"
df.to_csv(output_path)

print("\nDataset created successfully!")
print(f"Saved to: {output_path}")
print(f"Shape: {df.shape}")

print("\nTarget distribution:")
print(df['target'].value_counts())

print("\nSpread summary:")
print(df[['spread_points', 'spread_price', 'spread_return']].describe())

print("\nPreview:")
print(df.head())