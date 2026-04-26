import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ========== SETTINGS ==========
CONFIDENCE_THRESHOLD = 0.55   # try 0.55, 0.60, 0.65, 0.70, 0.75
ONLY_TRADE_SIGNALS = True     # True = only BUY/SELL, ignore HOLD for trade eval

# ========== LOAD DATASET ==========
SCRIPT_DIR = Path(__file__).resolve().parent
dataset_path = SCRIPT_DIR / "training_dataset.csv"

df = pd.read_csv(dataset_path, index_col=0)

# ========== FEATURES / TARGET ==========
# Exclude spread columns from model features for now
exclude_cols = ["target", "future_return", "spread_points", "spread_price", "spread_return"]
X = df.drop(columns=exclude_cols)
y = df["target"]

# ========== TIME-SERIES SPLIT ==========
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTrain target distribution:")
print(y_train.value_counts())

print("\nTest target distribution:")
print(y_test.value_counts())

# ========== MODEL ==========
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ========== RAW PREDICTIONS ==========
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
class_order = model.classes_

prob_df = pd.DataFrame(
    y_prob,
    columns=[f"prob_{c}" for c in class_order],
    index=X_test.index
)

prob_df["predicted_class"] = y_pred
prob_df["true_class"] = y_test.values
prob_df["max_confidence"] = y_prob.max(axis=1)

# Add evaluation columns from original dataset
prob_df["future_return"] = df.iloc[split:]["future_return"].values
prob_df["spread_return"] = df.iloc[split:]["spread_return"].values

print("\n========== RAW MODEL RESULTS ==========")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[-1, 0, 1]))

print("\nPrediction Distribution:")
print(pd.Series(y_pred).value_counts())

print("\nPrediction Confidence Summary:")
print(prob_df["max_confidence"].describe())

# ========== FILTER BY CONFIDENCE ==========
filtered_df = prob_df[prob_df["max_confidence"] >= CONFIDENCE_THRESHOLD].copy()

if ONLY_TRADE_SIGNALS:
    filtered_df = filtered_df[filtered_df["predicted_class"] != 0].copy()

print("\n========== FILTERED RESULTS ==========")
print(f"\nConfidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"Only trade signals: {ONLY_TRADE_SIGNALS}")
print(f"Filtered rows kept: {len(filtered_df)} / {len(prob_df)}")

if len(filtered_df) == 0:
    print("\nNo predictions passed the filter. Lower the threshold.")
else:
    print("\nFiltered Prediction Distribution:")
    print(filtered_df["predicted_class"].value_counts())

    print("\nFiltered Classification Report:")
    print(classification_report(
        filtered_df["true_class"],
        filtered_df["predicted_class"],
        digits=4,
        zero_division=0
    ))

    print("\nFiltered Confusion Matrix:")
    print(confusion_matrix(
        filtered_df["true_class"],
        filtered_df["predicted_class"],
        labels=[-1, 0, 1]
    ))

    # ========== SIMPLE TRADE EVALUATION WITH SPREAD ==========
    filtered_df["strategy_return"] = 0.0

    # BUY: profit must overcome spread
    filtered_df.loc[filtered_df["predicted_class"] == 1, "strategy_return"] = (
        filtered_df["future_return"] - filtered_df["spread_return"]
    )

    # SELL: same spread penalty
    filtered_df.loc[filtered_df["predicted_class"] == -1, "strategy_return"] = (
        -filtered_df["future_return"] - filtered_df["spread_return"]
    )

    filtered_df["win"] = filtered_df["strategy_return"] > 0

    total_trades = len(filtered_df)
    win_rate = filtered_df["win"].mean() if total_trades > 0 else 0.0
    avg_return = filtered_df["strategy_return"].mean() if total_trades > 0 else 0.0
    total_return = filtered_df["strategy_return"].sum() if total_trades > 0 else 0.0

    print("\n========== SIMPLE TRADE EVALUATION (WITH SPREAD) ==========")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.4f}")
    print(f"Average return per trade: {avg_return:.6f}")
    print(f"Total raw return sum: {total_return:.6f}")

    print("\nTop 10 filtered predictions:")
    print(filtered_df[[
        "predicted_class", "true_class", "max_confidence",
        "future_return", "spread_return", "strategy_return"
    ]].sort_values("max_confidence", ascending=False).head(10))

# ========== THRESHOLD SWEEP ==========
print("\n========== THRESHOLD SWEEP ==========")

threshold_results = []

for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    temp = prob_df[prob_df["max_confidence"] >= threshold].copy()

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

# ========== FEATURE IMPORTANCE ==========
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nTop 15 Features:")
print(feature_importance.head(15))

# ========== SAVE MODEL + FEATURE LIST ==========
model_path = SCRIPT_DIR / "rf_model.pkl"
features_path = SCRIPT_DIR / "rf_features.pkl"
results_path = SCRIPT_DIR / "rf_filtered_predictions.csv"
thresholds_path = SCRIPT_DIR / "rf_threshold_sweep.csv"

joblib.dump(model, model_path)
joblib.dump(list(X.columns), features_path)

if len(filtered_df) > 0:
    filtered_df.to_csv(results_path)

threshold_df.to_csv(thresholds_path, index=False)

print(f"\nModel saved to: {model_path}")
print(f"Feature list saved to: {features_path}")
if len(filtered_df) > 0:
    print(f"Filtered predictions saved to: {results_path}")
print(f"Threshold sweep saved to: {thresholds_path}")