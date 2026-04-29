import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, confusion_matrix

# ========== SETTINGS ==========
ONLY_TRADE_SIGNALS = True
THRESHOLD_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

# These columns are kept in the dataset for evaluation/simulation/live planning,
# but are NOT used as training features.
EXCLUDE_COLS = [
    "target",
    "future_return",

    # Spread/evaluation columns
    "spread_points",
    "spread_price",
    "spread_return",

    # Dynamic SL/TP planner columns
    "sl_atr_multiplier",
    "dynamic_sl_price_distance",
    "dynamic_tp_price_distance",
    "dynamic_sl_points",
    "dynamic_tp_points",
    "risk_reward_ratio",
    "sl_distance_pct",
    "tp_distance_pct",
]

# ========== LOAD DATASET ==========
SCRIPT_DIR = Path(__file__).resolve().parent
dataset_path = SCRIPT_DIR / "training_dataset.csv"

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Could not find {dataset_path}. Run the updated dataset builder first."
    )

df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
df.sort_index(inplace=True)

print(f"Dataset date range: {df.index.min()} -> {df.index.max()}")
print(f"Dataset raw shape: {df.shape}")

# ========== FEATURES / TARGET ==========
missing_exclude = [col for col in EXCLUDE_COLS if col not in df.columns]

if missing_exclude:
    print("[WARNING] Some excluded columns are missing:")
    print(missing_exclude)
    print("Continuing anyway...")

actual_exclude_cols = [col for col in EXCLUDE_COLS if col in df.columns]

X = df.drop(columns=actual_exclude_cols)
y = df["target"]

# Keep numeric only.
X = X.select_dtypes(include=[np.number]).copy()

# Clean bad values.
X.replace([np.inf, -np.inf], np.nan, inplace=True)

valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]
df = df.loc[valid_rows]

print(f"Loaded dataset: {dataset_path}")
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

val_future_return = df.iloc[train_end:val_end]["future_return"].values
val_spread_return = df.iloc[train_end:val_end]["spread_return"].values

test_future_return = df.iloc[val_end:]["future_return"].values
test_spread_return = df.iloc[val_end:]["spread_return"].values

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
    num_class=3,
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

# ========== HELPER ==========
def make_prob_df(X_part, y_part, future_return, spread_return):
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
    out["future_return"] = future_return
    out["spread_return"] = spread_return

    return out, pred

# ========== VALIDATION ==========
val_prob_df, val_pred = make_prob_df(
    X_val,
    y_val,
    val_future_return,
    val_spread_return
)

print("\n========== RAW VALIDATION RESULTS ==========")
print("\nClassification Report:")
print(classification_report(y_val, val_pred, digits=4, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, val_pred, labels=[-1, 0, 1]))

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
            "total_return": 0.0
        })
        continue

    temp["strategy_return"] = 0.0

    temp.loc[temp["predicted_class"] == 1, "strategy_return"] = (
        temp["future_return"] - temp["spread_return"]
    )

    temp.loc[temp["predicted_class"] == -1, "strategy_return"] = (
        -temp["future_return"] - temp["spread_return"]
    )

    temp["win"] = temp["strategy_return"] > 0

    threshold_results.append({
        "threshold": threshold,
        "trades": len(temp),
        "win_rate": temp["win"].mean(),
        "avg_return": temp["strategy_return"].mean(),
        "total_return": temp["strategy_return"].sum()
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
test_prob_df, test_pred = make_prob_df(
    X_test,
    y_test,
    test_future_return,
    test_spread_return
)

print("\n========== RAW TEST RESULTS ==========")
print("\nClassification Report:")
print(classification_report(y_test, test_pred, digits=4, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred, labels=[-1, 0, 1]))

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

    print("\nFiltered Classification Report:")
    print(classification_report(
        filtered_test["true_class"],
        filtered_test["predicted_class"],
        digits=4,
        zero_division=0
    ))

    print("\nFiltered Confusion Matrix:")
    print(confusion_matrix(
        filtered_test["true_class"],
        filtered_test["predicted_class"],
        labels=[-1, 0, 1]
    ))

    filtered_test["strategy_return"] = 0.0

    filtered_test.loc[filtered_test["predicted_class"] == 1, "strategy_return"] = (
        filtered_test["future_return"] - filtered_test["spread_return"]
    )

    filtered_test.loc[filtered_test["predicted_class"] == -1, "strategy_return"] = (
        -filtered_test["future_return"] - filtered_test["spread_return"]
    )

    filtered_test["win"] = filtered_test["strategy_return"] > 0

    total_trades = len(filtered_test)
    win_rate = filtered_test["win"].mean()
    avg_return = filtered_test["strategy_return"].mean()
    total_return = filtered_test["strategy_return"].sum()

    print("\n========== FINAL TEST TRADE EVALUATION ==========")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.4f}")
    print(f"Average return per trade: {avg_return:.6f}")
    print(f"Total raw return sum: {total_return:.6f}")

    print("\nTop 10 filtered predictions:")
    print(filtered_test[[
        "predicted_class",
        "true_class",
        "max_confidence",
        "future_return",
        "spread_return",
        "strategy_return"
    ]].sort_values("max_confidence", ascending=False).head(10))

# ========== FEATURE IMPORTANCE ==========
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nTop 40 Features:")
print(feature_importance.head(40))

indicator_importance = feature_importance[
    feature_importance.index.str.contains(
        "rsi|atr|adx|di_direction|market_is|alignment_score|bias|slope_strength|mixed_or_weak|htf_",
        case=False,
        regex=True
    )
]

print("\nTop Indicator / Regime Features:")
print(indicator_importance.head(30))

# ========== SAVE OUTPUTS ==========
model_path = SCRIPT_DIR / "lgbm_model.pkl"
features_path = SCRIPT_DIR / "lgbm_features.pkl"
validation_thresholds_path = SCRIPT_DIR / "lgbm_validation_threshold_sweep.csv"
test_results_path = SCRIPT_DIR / "lgbm_filtered_test_predictions.csv"
feature_importance_path = SCRIPT_DIR / "lgbm_feature_importance.csv"
indicator_importance_path = SCRIPT_DIR / "lgbm_indicator_feature_importance.csv"
metadata_path = SCRIPT_DIR / "lgbm_training_metadata.pkl"

joblib.dump(model, model_path)
joblib.dump(list(X.columns), features_path)

threshold_df.to_csv(validation_thresholds_path, index=False)
feature_importance.to_csv(feature_importance_path, header=["importance"])
indicator_importance.to_csv(indicator_importance_path, header=["importance"])

metadata = {
    "best_threshold": BEST_THRESHOLD,
    "feature_count": X.shape[1],
    "classes": list(model.classes_),
    "only_trade_signals": ONLY_TRADE_SIGNALS,
    "threshold_grid": THRESHOLD_GRID,
    "train_rows": len(X_train),
    "val_rows": len(X_val),
    "test_rows": len(X_test),
    "excluded_columns": actual_exclude_cols,
    "uses_dynamic_sl_tp_dataset": True,
    "uses_htf_bias_strength_features": True,
    "uses_htf_safe_backward_merge_dataset": True,
    "note": "Dynamic SL/TP columns are saved in the dataset but excluded from model training. Higher timeframe features are expected to be calculated on original 5m/15m candles, then backward-merged into 1m rows to avoid leakage."
}

joblib.dump(metadata, metadata_path)

if len(filtered_test) > 0:
    filtered_test.to_csv(test_results_path)

print(f"\nModel saved to: {model_path}")
print(f"Feature list saved to: {features_path}")
print(f"Validation threshold sweep saved to: {validation_thresholds_path}")
print(f"Feature importance saved to: {feature_importance_path}")
print(f"Indicator feature importance saved to: {indicator_importance_path}")
print(f"Training metadata saved to: {metadata_path}")

if len(filtered_test) > 0:
    print(f"Filtered test predictions saved to: {test_results_path}")