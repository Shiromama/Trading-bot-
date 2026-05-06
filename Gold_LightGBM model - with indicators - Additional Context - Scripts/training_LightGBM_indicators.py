import re
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, confusion_matrix

# ========== SETTINGS ==========
ONLY_TRADE_SIGNALS = True
THRESHOLD_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
ENTRY_STYLE_LABELS = [-3, -2, -1, 0, 1, 2, 3]
ENTRY_STYLE_NAMES = {
    -3: "SELL_WAIT_OB",
    -2: "SELL_WAIT_FVG",
    -1: "SELL_NOW",
     0: "NO_TRADE",
     1: "BUY_NOW",
     2: "BUY_WAIT_FVG",
     3: "BUY_WAIT_OB",
}

# Columns that may exist in the dataset but must NOT be used for training.
# These are either labels, future outcomes, candidate trade results, or execution/evaluation helpers.
MANUAL_EXCLUDE_COLS = [
    # Main targets / label helpers
    "target",
    "target_entry_style",
    "target_direction",
    "old_direction_target",
    "entry_style_name",
    "entry_style_return",
    "future_return",

    # Candidate entry-style returns produced by the labeler
    "ret_buy_now",
    "ret_buy_wait_fvg",
    "ret_buy_wait_ob",
    "ret_sell_now",
    "ret_sell_wait_fvg",
    "ret_sell_wait_ob",

    # Spread / evaluation columns
    "spread_points",
    "spread_price",
    "spread_return",

    # Dynamic SL/TP planner columns; kept for live/sim use, not model input
    "sl_atr_multiplier",
    "dynamic_sl_price_distance",
    "dynamic_tp_price_distance",
    "dynamic_sl_points",
    "dynamic_tp_points",
    "risk_reward_ratio",
    "sl_distance_pct",
    "tp_distance_pct",

    # Live-style SL/TP label planner columns from dataset_builder_live_style_sltp.py.
    # These describe how labels were simulated, so they must not leak into X.
    "live_style_risk_usd",
    "live_style_reward_usd",
    "live_style_lot_size",
    "live_style_sim_balance",
    "live_style_usd_per_price_move",
    "live_style_sl_price_distance",
    "live_style_tp_price_distance",
    "live_style_sl_points",
    "live_style_tp_points",
    "label_sl_price_distance",
    "label_tp_price_distance",
    "label_sl_points",
    "label_tp_points",
]

# Extra safety net: if future dataset builders add similar result/label columns,
# these patterns keep them out of X automatically.
LEAKAGE_PATTERNS = [
    r"^target",
    r"^old_direction_target$",
    r"^future_",
    r"^entry_style",
    r"^ret_",
    r"^live_style_",
    r"^label_",
    r"_return$",              # catches target/result return columns; 1m_return etc handled below by allowlist exception
]

# These are legitimate historical feature names ending in _return.
ALLOW_RETURN_FEATURE_PATTERNS = [
    r"^1m_return$", r"^5m_return$", r"^15m_return$",
    r"^1m_log_return$", r"^5m_log_return$", r"^15m_log_return$",
]

# ========== LOAD DATASET ==========
SCRIPT_DIR = Path(__file__).resolve().parent
dataset_path = SCRIPT_DIR / "training_dataset.csv"

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Could not find {dataset_path}. Run the fixed dataset builder first."
    )

df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
df.sort_index(inplace=True)

print(f"Dataset date range: {df.index.min()} -> {df.index.max()}")
print(f"Dataset raw shape: {df.shape}")

live_style_cols_present = [
    c for c in [
        "live_style_sl_price_distance",
        "live_style_tp_price_distance",
        "label_sl_price_distance",
        "label_tp_price_distance",
    ]
    if c in df.columns
]
if live_style_cols_present:
    print("Live-style SL/TP columns detected:", live_style_cols_present)
else:
    print("WARNING: No live-style SL/TP columns detected. Run dataset_builder_live_style_sltp.py first if intended.")

# ========== TARGET ==========
if "target_entry_style" in df.columns:
    target_col = "target_entry_style"
elif "target" in df.columns:
    target_col = "target"
else:
    raise ValueError("Dataset has no target_entry_style or target column.")

y = df[target_col].astype(int)

# ========== FEATURE SELECTION / LEAKAGE GUARD ==========
def matches_any(name: str, patterns: list[str]) -> bool:
    return any(re.search(p, name) for p in patterns)

def is_allowed_return_feature(name: str) -> bool:
    return matches_any(name, ALLOW_RETURN_FEATURE_PATTERNS)

auto_exclude_cols = []
for col in df.columns:
    if col in MANUAL_EXCLUDE_COLS:
        auto_exclude_cols.append(col)
        continue
    if matches_any(col, LEAKAGE_PATTERNS) and not is_allowed_return_feature(col):
        auto_exclude_cols.append(col)

actual_exclude_cols = sorted(set(c for c in auto_exclude_cols if c in df.columns))
X = df.drop(columns=actual_exclude_cols)

# Keep numeric features only. This also drops entry_style_name if it somehow was not excluded.
X = X.select_dtypes(include=[np.number]).copy()

# Clean bad numeric values.
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Keep rows where the model inputs and target are valid.
valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]
df = df.loc[valid_rows]

# Final paranoia check: fail loudly if known leakage columns reached X.
leakage_left = [
    c for c in X.columns
    if (matches_any(c, LEAKAGE_PATTERNS) and not is_allowed_return_feature(c))
]
if leakage_left:
    raise ValueError(
        "Potential leakage columns are still in training features: " + str(leakage_left[:50])
    )

print(f"Loaded dataset: {dataset_path}")
print(f"Training target column: {target_col}")
print(f"Rows after cleaning: {len(df)}")
print(f"Feature count: {X.shape[1]}")

print("\nExcluded from training:")
for col in actual_exclude_cols:
    print(f"- {col}")

# ========== TIME-SERIES SPLIT ==========
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("\nTrain shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)

print("\nTrain target distribution:")
print(y_train.value_counts().sort_index())

print("\nVal target distribution:")
print(y_val.value_counts().sort_index())

print("\nTest target distribution:")
print(y_test.value_counts().sort_index())

# ========== MODEL ==========
model = LGBMClassifier(
    objective="multiclass",
    n_estimators=1000,
    learning_rate=0.01,
    num_leaves=47,
    max_depth=7,
    min_child_samples=70,
    subsample=0.8,
    colsample_bytree=0.75,
    reg_alpha=0.8,
    reg_lambda=1.5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_logloss",
    callbacks=[
        early_stopping(stopping_rounds=120),
        log_evaluation(period=50)
    ]
)

print(f"\nBest iteration: {model.best_iteration_}")
print("Model classes:", list(model.classes_))

# ========== HELPERS ==========
def make_prob_df(X_part: pd.DataFrame, y_part: pd.Series, source_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    pred = model.predict(X_part)
    prob = model.predict_proba(X_part)
    class_order = model.classes_

    out = pd.DataFrame(
        prob,
        columns=[f"prob_{c}" for c in class_order],
        index=X_part.index
    )

    out["predicted_class"] = pred
    out["true_class"] = y_part.values
    out["max_confidence"] = prob.max(axis=1)
    out["predicted_name"] = pd.Series(pred, index=X_part.index).map(ENTRY_STYLE_NAMES)
    out["true_name"] = pd.Series(y_part.values, index=X_part.index).map(ENTRY_STYLE_NAMES)

    # Keep these only for evaluation/reporting; they were excluded from training above.
    eval_cols = [
        "ret_buy_now",
        "ret_buy_wait_fvg",
        "ret_buy_wait_ob",
        "ret_sell_now",
        "ret_sell_wait_fvg",
        "ret_sell_wait_ob",
        "entry_style_return",
        "future_return",
        "spread_return",
        "live_style_sl_price_distance",
        "live_style_tp_price_distance",
        "live_style_sl_points",
        "live_style_tp_points",
        "label_sl_price_distance",
        "label_tp_price_distance",
        "label_sl_points",
        "label_tp_points",
    ]
    for col in eval_cols:
        if col in source_df.columns:
            out[col] = source_df.loc[X_part.index, col].values

    return out, pred


def predicted_entry_return(frame: pd.DataFrame) -> pd.Series:
    """Return the simulated candidate return corresponding to each predicted entry-style class."""
    ret = pd.Series(0.0, index=frame.index)
    mapping = {
        1: "ret_buy_now",
        2: "ret_buy_wait_fvg",
        3: "ret_buy_wait_ob",
        -1: "ret_sell_now",
        -2: "ret_sell_wait_fvg",
        -3: "ret_sell_wait_ob",
    }
    for cls, col in mapping.items():
        if col in frame.columns:
            mask = frame["predicted_class"] == cls
            ret.loc[mask] = frame.loc[mask, col].fillna(-999.0)
    return ret


def summarize_trade_results(frame: pd.DataFrame, title: str) -> dict:
    if len(frame) == 0:
        print(f"\n========== {title} ==========")
        print("No trades passed the filter.")
        return {"trades": 0, "win_rate": 0.0, "avg_return": 0.0, "total_return": 0.0}

    frame["strategy_return"] = predicted_entry_return(frame)
    frame["win"] = frame["strategy_return"] > 0

    total_trades = len(frame)
    win_rate = float(frame["win"].mean())
    avg_return = float(frame["strategy_return"].mean())
    total_return = float(frame["strategy_return"].sum())

    print(f"\n========== {title} ==========")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.4f}")
    print(f"Average return per trade: {avg_return:.6f}")
    print(f"Total raw return sum: {total_return:.6f}")

    return {
        "trades": total_trades,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_return": total_return,
    }

# ========== VALIDATION ==========
val_prob_df, val_pred = make_prob_df(X_val, y_val, df)

labels_for_report = [c for c in ENTRY_STYLE_LABELS if c in sorted(set(y_val.unique()) | set(val_pred))]

print("\n========== RAW VALIDATION RESULTS ==========")
print("\nClassification Report:")
print(classification_report(y_val, val_pred, labels=labels_for_report, digits=4, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_pred, labels=labels_for_report))
print("Labels:", labels_for_report)

# ========== THRESHOLD SWEEP ==========
print("\n========== VALIDATION THRESHOLD SWEEP ==========")
threshold_results = []

for threshold in THRESHOLD_GRID:
    temp = val_prob_df[val_prob_df["max_confidence"] >= threshold].copy()

    if ONLY_TRADE_SIGNALS:
        temp = temp[temp["predicted_class"] != 0].copy()

    if len(temp) == 0:
        threshold_results.append({
            "threshold": threshold,
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return": 0.0,
        })
        continue

    temp["strategy_return"] = predicted_entry_return(temp)
    temp["win"] = temp["strategy_return"] > 0

    threshold_results.append({
        "threshold": threshold,
        "trades": len(temp),
        "win_rate": temp["win"].mean(),
        "avg_return": temp["strategy_return"].mean(),
        "total_return": temp["strategy_return"].sum(),
    })

threshold_df = pd.DataFrame(threshold_results)
print(threshold_df)

best_row = threshold_df.sort_values(
    ["total_return", "avg_return", "win_rate"],
    ascending=False
).iloc[0]

BEST_THRESHOLD = float(best_row["threshold"])
print(f"\nBest validation threshold selected: {BEST_THRESHOLD}")

# ========== TEST ==========
test_prob_df, test_pred = make_prob_df(X_test, y_test, df)
labels_for_test_report = [c for c in ENTRY_STYLE_LABELS if c in sorted(set(y_test.unique()) | set(test_pred))]

print("\n========== RAW TEST RESULTS ==========")
print("\nClassification Report:")
print(classification_report(y_test, test_pred, labels=labels_for_test_report, digits=4, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred, labels=labels_for_test_report))
print("Labels:", labels_for_test_report)

filtered_test = test_prob_df[test_prob_df["max_confidence"] >= BEST_THRESHOLD].copy()

if ONLY_TRADE_SIGNALS:
    filtered_test = filtered_test[filtered_test["predicted_class"] != 0].copy()

print("\n========== FILTERED TEST RESULTS ==========")
print(f"Threshold used: {BEST_THRESHOLD}")
print(f"Filtered rows kept: {len(filtered_test)} / {len(test_prob_df)}")

if len(filtered_test) == 0:
    print("\nNo predictions passed the filter.")
else:
    print("\nFiltered Prediction Distribution:")
    print(filtered_test["predicted_class"].value_counts().sort_index())
    print("\nFiltered Prediction Names:")
    print(filtered_test["predicted_name"].value_counts())

    filtered_labels = [
        c for c in ENTRY_STYLE_LABELS
        if c in sorted(set(filtered_test["true_class"].unique()) | set(filtered_test["predicted_class"].unique()))
    ]

    print("\nFiltered Classification Report:")
    print(classification_report(
        filtered_test["true_class"],
        filtered_test["predicted_class"],
        labels=filtered_labels,
        digits=4,
        zero_division=0
    ))

    print("\nFiltered Confusion Matrix:")
    print(confusion_matrix(
        filtered_test["true_class"],
        filtered_test["predicted_class"],
        labels=filtered_labels
    ))
    print("Labels:", filtered_labels)

    summarize_trade_results(filtered_test, "FINAL TEST TRADE EVALUATION")

    print("\nTop 10 filtered predictions:")
    cols_to_show = [
        "predicted_class",
        "predicted_name",
        "true_class",
        "true_name",
        "max_confidence",
        "strategy_return",
        "ret_buy_now",
        "ret_buy_wait_fvg",
        "ret_buy_wait_ob",
        "ret_sell_now",
        "ret_sell_wait_fvg",
        "ret_sell_wait_ob",
        "label_sl_points",
        "label_tp_points",
        "label_sl_price_distance",
        "label_tp_price_distance",
    ]
    cols_to_show = [c for c in cols_to_show if c in filtered_test.columns]
    print(filtered_test[cols_to_show].sort_values("max_confidence", ascending=False).head(10))

# ========== FEATURE IMPORTANCE ==========
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTop 40 Features:")
print(feature_importance.head(40))

indicator_importance = feature_importance[
    feature_importance.index.str.contains(
        "rsi|atr|adx|di_direction|market_is|alignment_score|bias|slope_strength|mixed_or_weak|htf_",
        case=False,
        regex=True
    )
]

ob_importance = feature_importance[
    feature_importance.index.str.contains(
        "ob|breaker|dist_to_bull|dist_to_bear|near_15m|near_1m",
        case=False,
        regex=True
    )
]

fvg_importance = feature_importance[
    feature_importance.index.str.contains(
        "fvg|dist_to_bull_fvg|dist_to_bear_fvg|near_15m_bull_fvg|near_15m_bear_fvg|near_1m_bull_fvg|near_1m_bear_fvg",
        case=False,
        regex=True
    )
]

structure_importance = feature_importance[
    feature_importance.index.str.contains(
        "bos|choch|fractal|structure_",
        case=False,
        regex=True
    )
]

context_importance = feature_importance[
    feature_importance.index.str.contains(
        "hour_|dow_|day_of_week|session_|london_open|ny_open|daily_|dist_to_daily|atr_mean_100|atr_std_100|volatility_zscore",
        case=False,
        regex=True
    )
]

sweep_importance = feature_importance[
    feature_importance.index.str.contains(
        "sweep|prev_high_20|prev_low_20|dist_to_prev_high|dist_to_prev_low|wick_rejection|atr_strength|strong_sweep",
        case=False,
        regex=True
    )
]

htf_liquidity_importance = feature_importance[
    feature_importance.index.str.contains(
        "htf_liquidity|prev_daily|prev_weekly|prev_monthly|near_prev|swept_prev|reject_prev|reclaim_prev|bars_since_swept_prev|bars_since_reject_prev|premium_discount|range_position|daily_weekly_liquidity_confluence",
        case=False,
        regex=True
    )
]

session_liquidity_importance = feature_importance[
    feature_importance.index.str.contains(
        "session_liquidity|current_asia|current_london|current_ny|prev_asia_session|prev_london_session|prev_ny_session|in_asia_killzone|in_london_killzone|in_ny_killzone|in_london_ny_overlap|in_asia_session_liquidity_window|in_london_session_liquidity_window|in_ny_session_liquidity_window|london_swept_asia|london_reject_asia|ny_swept_london|ny_reject_london|ny_swept_asia|ny_reject_asia|london_asia_sweep|ny_london_sweep|ny_asia_sweep|killzone_session_sweep",
        case=False,
        regex=True
    )
]

advanced_liquidity_importance = feature_importance[
    feature_importance.index.str.contains(
        "advanced_liquidity|weighted_session|weighted_london|weighted_ny|session_high_reject_decay_max|session_low_reject_decay_max|session_high_sweep_decay_max|session_low_sweep_decay_max|session_reject_decay_bias|session_sweep_decay_bias|session_high_proximity_max|session_low_proximity_max|session_high_rejection_strength_cont|session_low_rejection_strength_cont|session_rejection_strength_bias_cont|liquidity_fusion|entry_zone_liquidity_fusion|structure_bias_normalized|liquidity_interaction|session_high_near_any|session_low_near_any|session_high_bearish_structure|session_low_bullish_structure",
        case=False,
        regex=True
    )
]

behavior_flow_importance = feature_importance[
    feature_importance.index.str.contains(
        "body_atr_ratio|signed_body_atr_ratio|range_atr_ratio|candle_efficiency|displacement|expansion_bar|impulse_body_atr|range_expansion|volume_expansion|compression|breakout_after_compression|box_range|box_position|bb_width|seq_|dnt_",
        case=False,
        regex=True
    )
]

displacement_importance = feature_importance[
    feature_importance.index.str.contains(
        "body_atr_ratio|signed_body_atr_ratio|range_atr_ratio|candle_efficiency|displacement|expansion_bar|impulse_body_atr|range_expansion|volume_expansion",
        case=False,
        regex=True
    )
]

compression_importance = feature_importance[
    feature_importance.index.str.contains(
        "compression|breakout_after_compression|box_range|box_position|bb_width",
        case=False,
        regex=True
    )
]

sequence_importance = feature_importance[
    feature_importance.index.str.contains(
        "^seq_",
        case=False,
        regex=True
    )
]

do_not_trade_importance = feature_importance[
    feature_importance.index.str.contains(
        "^dnt_",
        case=False,
        regex=True
    )
]

print("\nTop Indicator / Regime Features:")
print(indicator_importance.head(30))

print("\nTop Order Block / Breaker Block Features:")
print(ob_importance.head(30))

print("\nTop Fair Value Gap Features:")
print(fvg_importance.head(30))

print("\nTop Must-Add Context Features:")
print(context_importance.head(30))

print("\nTop Fractal BOS / CHoCH Structure Features:")
print(structure_importance.head(30))

print("\nTop Liquidity Sweep / Rejection Features:")
print(sweep_importance.head(30))

print("\nTop True Session-Based HTF Liquidity Features:")
print(htf_liquidity_importance.head(40))

print("\nTop ICT Session Liquidity Features:")
print(session_liquidity_importance.head(40))


print("\nTop Behavior Flow Upgrade Features:")
print(behavior_flow_importance.head(50))

print("\nTop Displacement / Expansion Features:")
print(displacement_importance.head(40))

print("\nTop Compression -> Expansion Features:")
print(compression_importance.head(40))

print("\nTop Multi-Timeframe Sequence Awareness Features:")
print(sequence_importance.head(40))

print("\nTop Do-Not-Trade Intelligence Features:")
print(do_not_trade_importance.head(40))

# ========== SAVE OUTPUTS ==========
model_path = SCRIPT_DIR / "lgbm_model.pkl"
features_path = SCRIPT_DIR / "lgbm_features.pkl"
validation_thresholds_path = SCRIPT_DIR / "lgbm_validation_threshold_sweep.csv"
test_results_path = SCRIPT_DIR / "lgbm_filtered_test_predictions.csv"
feature_importance_path = SCRIPT_DIR / "lgbm_feature_importance.csv"
indicator_importance_path = SCRIPT_DIR / "lgbm_indicator_feature_importance.csv"
ob_importance_path = SCRIPT_DIR / "lgbm_ob_feature_importance.csv"
fvg_importance_path = SCRIPT_DIR / "lgbm_fvg_feature_importance.csv"
structure_importance_path = SCRIPT_DIR / "lgbm_structure_feature_importance.csv"
context_importance_path = SCRIPT_DIR / "lgbm_context_feature_importance.csv"
sweep_importance_path = SCRIPT_DIR / "lgbm_sweep_feature_importance.csv"
htf_liquidity_importance_path = SCRIPT_DIR / "lgbm_htf_liquidity_feature_importance.csv"
session_liquidity_importance_path = SCRIPT_DIR / "lgbm_session_liquidity_feature_importance.csv"
advanced_liquidity_importance_path = SCRIPT_DIR / "lgbm_advanced_liquidity_feature_importance.csv"
behavior_flow_importance_path = SCRIPT_DIR / "lgbm_behavior_flow_feature_importance.csv"
displacement_importance_path = SCRIPT_DIR / "lgbm_displacement_feature_importance.csv"
compression_importance_path = SCRIPT_DIR / "lgbm_compression_feature_importance.csv"
sequence_importance_path = SCRIPT_DIR / "lgbm_sequence_feature_importance.csv"
do_not_trade_importance_path = SCRIPT_DIR / "lgbm_do_not_trade_feature_importance.csv"
metadata_path = SCRIPT_DIR / "lgbm_training_metadata.pkl"

joblib.dump(model, model_path)
joblib.dump(list(X.columns), features_path)
threshold_df.to_csv(validation_thresholds_path, index=False)
feature_importance.to_csv(feature_importance_path, header=["importance"])
indicator_importance.to_csv(indicator_importance_path, header=["importance"])
ob_importance.to_csv(ob_importance_path, header=["importance"])
fvg_importance.to_csv(fvg_importance_path, header=["importance"])
structure_importance.to_csv(structure_importance_path, header=["importance"])
context_importance.to_csv(context_importance_path, header=["importance"])
sweep_importance.to_csv(sweep_importance_path, header=["importance"])
htf_liquidity_importance.to_csv(htf_liquidity_importance_path, header=["importance"])
session_liquidity_importance.to_csv(session_liquidity_importance_path, header=["importance"])
advanced_liquidity_importance.to_csv(advanced_liquidity_importance_path, header=["importance"])
behavior_flow_importance.to_csv(behavior_flow_importance_path, header=["importance"])
displacement_importance.to_csv(displacement_importance_path, header=["importance"])
compression_importance.to_csv(compression_importance_path, header=["importance"])
sequence_importance.to_csv(sequence_importance_path, header=["importance"])
do_not_trade_importance.to_csv(do_not_trade_importance_path, header=["importance"])

metadata = {
    "best_threshold": BEST_THRESHOLD,
    "feature_count": X.shape[1],
    "classes": list(model.classes_),
    "entry_style_names": ENTRY_STYLE_NAMES,
    "target_col": target_col,
    "only_trade_signals": ONLY_TRADE_SIGNALS,
    "threshold_grid": THRESHOLD_GRID,
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "excluded_columns": actual_exclude_cols,
    "uses_dynamic_sl_tp_dataset": True,
    "uses_live_style_sl_tp_labeling": True,
    "uses_htf_bias_strength_features": True,
    "uses_htf_safe_backward_merge_dataset": True,
    "uses_order_block_breaker_features": True,
    "uses_fair_value_gap_features": True,
    "uses_fractal_bos_choch_structure_features": True,
    "uses_entry_style_target": True,
    "uses_must_add_context_features": True,
    "uses_liquidity_sweep_rejection_features": True,
    "uses_advanced_sweep_strength_features": True,
    "uses_wick_rejection_strength_features": True,
    "uses_multitimeframe_sweep_alignment_features": True,
    "uses_true_session_htf_liquidity_features": True,
    "uses_ict_session_liquidity_features": True,
    "uses_advanced_liquidity_strength_features": True,
    "uses_time_decayed_liquidity_features": True,
    "uses_weighted_session_liquidity_features": True,
    "uses_ob_fvg_liquidity_fusion_features": True,
    "uses_behavior_flow_upgrade_features": True,
    "uses_displacement_expansion_features": True,
    "uses_compression_expansion_features": True,
    "uses_multitimeframe_sequence_awareness_features": True,
    "uses_do_not_trade_intelligence_features": True,
    "keeps_existing_rolling_daily_context_features": True,
    "true_session_htf_liquidity_features": [
        "prev_daily_high", "prev_daily_low", "prev_daily_mid", "prev_daily_range",
        "prev_weekly_high", "prev_weekly_low", "prev_weekly_mid", "prev_weekly_range",
        "prev_monthly_high", "prev_monthly_low", "prev_monthly_mid", "prev_monthly_range",
        "dist_to_prev_daily_high", "dist_to_prev_daily_low",
        "dist_to_prev_weekly_high", "dist_to_prev_weekly_low",
        "dist_to_prev_monthly_high", "dist_to_prev_monthly_low",
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
        "prev_daily_range_position", "prev_weekly_range_position", "prev_monthly_range_position",
        "prev_daily_premium_discount", "prev_weekly_premium_discount", "prev_monthly_premium_discount",
        "htf_liquidity_near_high_score", "htf_liquidity_near_low_score",
        "htf_liquidity_sweep_high_score", "htf_liquidity_sweep_low_score",
        "htf_liquidity_reject_high_score", "htf_liquidity_reject_low_score",
        "htf_liquidity_reversal_bias", "htf_liquidity_continuation_bias",
        "daily_weekly_liquidity_confluence_high", "daily_weekly_liquidity_confluence_low"
    ],
    "must_add_context_features": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "session_asia", "session_london", "session_ny", "london_open", "ny_open", "daily_position", "dist_to_daily_high", "dist_to_daily_low", "volatility_zscore"],
    "liquidity_sweep_features": [
        "prev_high_20", "prev_low_20",
        "sweep_high", "sweep_low",
        "sweep_reject_high", "sweep_reject_low",
        "sweep_high_depth", "sweep_low_depth",
        "sweep_high_strength", "sweep_low_strength",
        "sweep_high_atr_strength", "sweep_low_atr_strength",
        "sweep_high_wick_ratio", "sweep_low_wick_ratio",
        "sweep_high_wick_rejection_strength", "sweep_low_wick_rejection_strength",
        "strong_sweep_reject_high", "strong_sweep_reject_low",
        "recent_sweep_high", "recent_sweep_low",
        "recent_sweep_reject_high", "recent_sweep_reject_low",
        "recent_strong_sweep_reject_high", "recent_strong_sweep_reject_low",
        "sweep_high_context_score", "sweep_low_context_score",
        "sweep_reject_high_context_score", "sweep_reject_low_context_score",
        "strong_sweep_reject_high_context_score", "strong_sweep_reject_low_context_score",
        "sweep_high_atr_strength_sum", "sweep_low_atr_strength_sum",
        "sweep_high_wick_rejection_strength_sum", "sweep_low_wick_rejection_strength_sum"
    ],
    "ict_session_liquidity_features": [
        "current_asia_high", "current_asia_low", "current_london_high", "current_london_low", "current_ny_high", "current_ny_low",
        "prev_asia_session_high", "prev_asia_session_low", "prev_london_session_high", "prev_london_session_low", "prev_ny_session_high", "prev_ny_session_low",
        "near_prev_asia_session_high", "near_prev_asia_session_low", "near_prev_london_session_high", "near_prev_london_session_low", "near_prev_ny_session_high", "near_prev_ny_session_low",
        "swept_prev_asia_session_high", "swept_prev_asia_session_low", "swept_prev_london_session_high", "swept_prev_london_session_low", "swept_prev_ny_session_high", "swept_prev_ny_session_low",
        "reject_prev_asia_session_high", "reject_prev_asia_session_low", "reject_prev_london_session_high", "reject_prev_london_session_low", "reject_prev_ny_session_high", "reject_prev_ny_session_low",
        "london_swept_asia_high", "london_swept_asia_low", "london_reject_asia_high", "london_reject_asia_low",
        "ny_swept_london_high", "ny_swept_london_low", "ny_reject_london_high", "ny_reject_london_low",
        "ny_swept_asia_high", "ny_swept_asia_low", "ny_reject_asia_high", "ny_reject_asia_low",
        "session_liquidity_near_high_score", "session_liquidity_near_low_score",
        "session_liquidity_sweep_high_score", "session_liquidity_sweep_low_score",
        "session_liquidity_reject_high_score", "session_liquidity_reject_low_score",
        "session_liquidity_reversal_bias", "session_liquidity_continuation_bias",
        "london_asia_sweep_reversal_bias", "ny_london_sweep_reversal_bias", "ny_asia_sweep_reversal_bias",
        "killzone_session_sweep_score"
    ],
    "behavior_flow_upgrade_feature_groups": {
        "displacement_expansion": [
            "body_atr_ratio", "signed_body_atr_ratio", "range_atr_ratio",
            "candle_efficiency", "bull_displacement", "bear_displacement",
            "expansion_bar", "impulse_body_atr", "displacement_pressure"
        ],
        "compression_expansion": [
            "range_compression_ratio_20_100", "atr_compression_ratio_20_100",
            "box_range_pct_20", "box_position_20", "bb_width_20",
            "is_compressed", "breakout_after_compression"
        ],
        "sequence_awareness": [
            "seq_bos_up_count", "seq_bos_down_count", "seq_choch_up_count",
            "seq_choch_down_count", "seq_sweep_pressure_bias",
            "seq_rejection_reversal_bias", "seq_displacement_bias",
            "seq_market_pressure_bias_20", "seq_market_activity_score_20"
        ],
        "do_not_trade_intelligence": [
            "dnt_context_conflict_score", "dnt_structure_flip_count_20",
            "dnt_both_sides_swept_20", "dnt_weak_displacement_environment",
            "dnt_unresolved_compression", "dnt_conflicting_structure_liquidity",
            "dnt_uncertainty_score", "dnt_low_quality_trade_environment"
        ]
    },
    "note": "Live-style SL/TP training script: excludes leakage/result columns including live_style_* and label_* helper columns, evaluates all 7 entry-style classes, computes returns using the predicted class candidate return, and reports feature importance groups including behavior-flow upgrades."
}
joblib.dump(metadata, metadata_path)

if len(filtered_test) > 0:
    filtered_test.to_csv(test_results_path)

print(f"\nModel saved to: {model_path}")
print(f"Feature list saved to: {features_path}")
print(f"Validation threshold sweep saved to: {validation_thresholds_path}")
print(f"Feature importance saved to: {feature_importance_path}")
print(f"Indicator feature importance saved to: {indicator_importance_path}")
print(f"Order block feature importance saved to: {ob_importance_path}")
print(f"Fair value gap feature importance saved to: {fvg_importance_path}")
print(f"Context feature importance saved to: {context_importance_path}")
print(f"Liquidity sweep feature importance saved to: {sweep_importance_path}")
print(f"HTF liquidity feature importance saved to: {htf_liquidity_importance_path}")
print(f"ICT session liquidity feature importance saved to: {session_liquidity_importance_path}")
print(f"Advanced liquidity feature importance saved to: {advanced_liquidity_importance_path}")
print(f"Behavior-flow feature importance saved to: {behavior_flow_importance_path}")
print(f"Displacement feature importance saved to: {displacement_importance_path}")
print(f"Compression feature importance saved to: {compression_importance_path}")
print(f"Sequence-awareness feature importance saved to: {sequence_importance_path}")
print(f"Do-not-trade feature importance saved to: {do_not_trade_importance_path}")
print(f"Training metadata saved to: {metadata_path}")

if len(filtered_test) > 0:
    print(f"Filtered test predictions saved to: {test_results_path}")