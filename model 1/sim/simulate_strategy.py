import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# SIMULATION SETTINGS
# ==========================================================
STARTING_BALANCE = 18.44
ACCOUNT_LEVERAGE = 2000

# Manual-style trading settings
LOT_SIZE = 0.01
CONTRACT_SIZE = 100000          # standard forex contract size
REFERENCE_PRICE = 1.10          # used if no price column is found

# Risk model
STOP_LOSS_ACCOUNT_FRACTION = 0.06   # risk 6% of current balance
RISK_REWARD_RATIO = 2.0             # 1:2 means TP is 2x SL in money terms
USE_TAKE_PROFIT = True

CONFIDENCE_THRESHOLD = 0.55
ONLY_TRADE_SIGNALS = True
TRAIN_TEST_SPLIT = 0.80
BROKER_FEE_PER_TRADE = 0.0      # flat USD fee per closed trade

# Thresholds to compare
THRESHOLD_VALUES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Stop simulation if the balance gets too small
STOP_IF_BALANCE_BELOW = 0.01

# ==========================================================
# PATHS
# ==========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "training_dataset.csv"
MODEL_PATH = SCRIPT_DIR / "rf_model.pkl"
FEATURES_PATH = SCRIPT_DIR / "rf_features.pkl"

TRADE_LOG_PATH = SCRIPT_DIR / "simulation_trade_log.csv"
SUMMARY_PATH = SCRIPT_DIR / "simulation_summary.txt"
EQUITY_PLOT_PATH = SCRIPT_DIR / "simulation_equity_curve.png"
DRAWDOWN_PLOT_PATH = SCRIPT_DIR / "simulation_drawdown_curve.png"
THRESHOLD_SWEEP_PATH = SCRIPT_DIR / "simulation_threshold_sweep.csv"


# ==========================================================
# HELPERS
# ==========================================================
def compute_max_drawdown_pct(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    running_peak = equity_series.cummax()
    drawdown = (equity_series - running_peak) / running_peak
    return float(drawdown.min() * 100.0)


def longest_streak(values: pd.Series, target: bool) -> int:
    best = 0
    current = 0
    for v in values.astype(bool):
        if v == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def infer_price_series(df: pd.DataFrame, split_idx: int) -> pd.Series:
    """
    Try to find a usable entry/close price column for notional sizing.
    Falls back to REFERENCE_PRICE if none is found.
    """
    candidate_cols = [
        "entry_price", "close", "Close", "open", "Open",
        "price", "Price", "bid_close", "ask_close"
    ]
    for col in candidate_cols:
        if col in df.columns:
            series = pd.to_numeric(df.iloc[split_idx:][col], errors="coerce")
            if series.notna().any():
                return series.ffill().bfill().fillna(REFERENCE_PRICE)

    return pd.Series(REFERENCE_PRICE, index=df.iloc[split_idx:].index, dtype=float)


def prepare_prediction_frame(df: pd.DataFrame, model, feature_list, split_idx: int) -> pd.DataFrame:
    exclude_cols = ["target", "future_return", "spread_points", "spread_price", "spread_return"]
    X = df.drop(columns=exclude_cols)
    X = X[feature_list]

    X_test = X.iloc[split_idx:]
    y_test = df.iloc[split_idx:]["target"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    class_order = model.classes_

    prob_df = pd.DataFrame(
        y_prob,
        columns=[f"prob_{c}" for c in class_order],
        index=X_test.index,
    )

    prob_df["predicted_class"] = y_pred
    prob_df["true_class"] = y_test.values
    prob_df["max_confidence"] = y_prob.max(axis=1)
    prob_df["future_return"] = df.iloc[split_idx:]["future_return"].values
    prob_df["spread_return"] = df.iloc[split_idx:]["spread_return"].values
    prob_df["entry_price"] = infer_price_series(df, split_idx).values
    return prob_df


def filter_predictions(prob_df: pd.DataFrame, confidence_threshold: float, only_trade_signals: bool) -> pd.DataFrame:
    filtered = prob_df[prob_df["max_confidence"] >= confidence_threshold].copy()
    if only_trade_signals:
        filtered = filtered[filtered["predicted_class"] != 0].copy()
    return filtered


def add_trade_returns(filtered_df: pd.DataFrame) -> pd.DataFrame:
    df_out = filtered_df.copy()
    df_out["strategy_return"] = 0.0

    # BUY
    df_out.loc[df_out["predicted_class"] == 1, "strategy_return"] = (
        df_out["future_return"] - df_out["spread_return"]
    )

    # SELL
    df_out.loc[df_out["predicted_class"] == -1, "strategy_return"] = (
        -df_out["future_return"] - df_out["spread_return"]
    )

    df_out["win_raw"] = df_out["strategy_return"] > 0
    return df_out


def position_notional_usd(entry_price: float) -> float:
    return LOT_SIZE * CONTRACT_SIZE * float(entry_price)


def required_margin_usd(entry_price: float) -> float:
    return position_notional_usd(entry_price) / ACCOUNT_LEVERAGE


def simulate_account(trades_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if trades_df.empty:
        return trades_df.copy(), {
            "starting_balance": STARTING_BALANCE,
            "ending_balance": STARTING_BALANCE,
            "net_profit": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
            "blown_account": False,
            "skipped_for_margin": 0,
            "stop_loss_hits": 0,
            "take_profit_hits": 0,
            "avg_margin_required": 0.0,
            "avg_position_notional": 0.0,
            "avg_sl_amount_usd": 0.0,
            "avg_tp_amount_usd": 0.0,
        }

    sim = trades_df.copy()
    sim.insert(0, "row_id", sim.index.to_numpy())
    sim = sim.reset_index(drop=True)

    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    blown_account = False
    skipped_for_margin = 0
    records = []

    for trade_number, row in enumerate(sim.itertuples(index=False), start=1):
        if balance <= STOP_IF_BALANCE_BELOW:
            blown_account = True
            break

        entry_price = float(row.entry_price)
        notional_usd = position_notional_usd(entry_price)
        margin_required = required_margin_usd(entry_price)

        if margin_required > balance:
            skipped_for_margin += 1
            continue

        balance_before = balance
        raw_trade_return = float(row.strategy_return)

        sl_amount_usd = balance_before * STOP_LOSS_ACCOUNT_FRACTION
        tp_amount_usd = sl_amount_usd * RISK_REWARD_RATIO

        sl_return_dynamic = sl_amount_usd / notional_usd if notional_usd > 0 else float("inf")
        tp_return_dynamic = tp_amount_usd / notional_usd if notional_usd > 0 else float("inf")

        exit_reason = "final_horizon"
        effective_raw_return = raw_trade_return

        # Apply SL/TP using money-based risk model.
        # Check SL first as a conservative rule when only final-horizon return is available.
        if raw_trade_return <= -sl_return_dynamic:
            effective_raw_return = -sl_return_dynamic
            exit_reason = "stop_loss"
        elif USE_TAKE_PROFIT and raw_trade_return >= tp_return_dynamic:
            effective_raw_return = tp_return_dynamic
            exit_reason = "take_profit"

        pnl = notional_usd * effective_raw_return - BROKER_FEE_PER_TRADE
        balance = max(balance + pnl, 0.0)
        peak_balance = max(peak_balance, balance)
        drawdown_pct = ((balance - peak_balance) / peak_balance * 100.0) if peak_balance > 0 else 0.0

        records.append({
            "trade_number": trade_number,
            "row_id": row.row_id,
            "predicted_class": row.predicted_class,
            "true_class": row.true_class,
            "max_confidence": row.max_confidence,
            "entry_price": entry_price,
            "future_return": row.future_return,
            "spread_return": row.spread_return,
            "raw_trade_return": raw_trade_return,
            "effective_raw_return": effective_raw_return,
            "exit_reason": exit_reason,
            "position_notional_usd": notional_usd,
            "margin_required_usd": margin_required,
            "dynamic_sl_amount_usd": sl_amount_usd,
            "dynamic_tp_amount_usd": tp_amount_usd,
            "dynamic_sl_return": sl_return_dynamic,
            "dynamic_tp_return": tp_return_dynamic,
            "fee_cost": BROKER_FEE_PER_TRADE,
            "pnl": pnl,
            "balance_before": balance_before,
            "balance_after": balance,
            "peak_balance": peak_balance,
            "drawdown_pct": drawdown_pct,
            "win_after_exit": pnl > 0,
        })

    result_df = pd.DataFrame(records)

    if result_df.empty:
        summary = {
            "starting_balance": STARTING_BALANCE,
            "ending_balance": STARTING_BALANCE,
            "net_profit": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
            "blown_account": blown_account,
            "skipped_for_margin": skipped_for_margin,
            "stop_loss_hits": 0,
            "take_profit_hits": 0,
            "avg_margin_required": 0.0,
            "avg_position_notional": 0.0,
            "avg_sl_amount_usd": 0.0,
            "avg_tp_amount_usd": 0.0,
        }
        return result_df, summary

    equity = result_df["balance_after"]
    summary = {
        "starting_balance": STARTING_BALANCE,
        "ending_balance": float(result_df["balance_after"].iloc[-1]),
        "net_profit": float(result_df["balance_after"].iloc[-1] - STARTING_BALANCE),
        "return_pct": float((result_df["balance_after"].iloc[-1] / STARTING_BALANCE - 1.0) * 100.0),
        "max_drawdown_pct": compute_max_drawdown_pct(equity),
        "total_trades": int(len(result_df)),
        "win_rate": float(result_df["win_after_exit"].mean() * 100.0),
        "longest_win_streak": int(longest_streak(result_df["win_after_exit"], True)),
        "longest_loss_streak": int(longest_streak(result_df["win_after_exit"], False)),
        "blown_account": bool(blown_account or result_df["balance_after"].iloc[-1] <= STOP_IF_BALANCE_BELOW),
        "skipped_for_margin": int(skipped_for_margin),
        "stop_loss_hits": int((result_df["exit_reason"] == "stop_loss").sum()),
        "take_profit_hits": int((result_df["exit_reason"] == "take_profit").sum()),
        "avg_margin_required": float(result_df["margin_required_usd"].mean()),
        "avg_position_notional": float(result_df["position_notional_usd"].mean()),
        "avg_sl_amount_usd": float(result_df["dynamic_sl_amount_usd"].mean()),
        "avg_tp_amount_usd": float(result_df["dynamic_tp_amount_usd"].mean()),
    }
    return result_df, summary


def make_plots(trade_log: pd.DataFrame):
    if trade_log.empty:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(trade_log["trade_number"], trade_log["balance_after"])
    plt.xlabel("Trade Number")
    plt.ylabel("Balance (USD)")
    plt.title("Simulation Equity Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EQUITY_PLOT_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(trade_log["trade_number"], trade_log["drawdown_pct"])
    plt.xlabel("Trade Number")
    plt.ylabel("Drawdown %")
    plt.title("Simulation Drawdown Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DRAWDOWN_PLOT_PATH, dpi=150)
    plt.close()


def run_threshold_sweep(prob_df: pd.DataFrame):
    results = []
    for threshold in THRESHOLD_VALUES:
        filtered = filter_predictions(prob_df, threshold, ONLY_TRADE_SIGNALS)
        filtered = add_trade_returns(filtered)
        _, summary = simulate_account(filtered)
        results.append({
            "threshold": threshold,
            "trades": summary["total_trades"],
            "ending_balance": summary["ending_balance"],
            "return_pct": summary["return_pct"],
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "win_rate_pct": summary["win_rate"],
            "stop_loss_hits": summary["stop_loss_hits"],
            "take_profit_hits": summary["take_profit_hits"],
            "skipped_for_margin": summary["skipped_for_margin"],
            "blown_account": summary["blown_account"],
        })

    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(THRESHOLD_SWEEP_PATH, index=False)
    return sweep_df


# ==========================================================
# MAIN
# ==========================================================
def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature list: {FEATURES_PATH}")

    df = pd.read_csv(DATASET_PATH, index_col=0)
    model = joblib.load(MODEL_PATH)
    feature_list = joblib.load(FEATURES_PATH)

    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    prob_df = prepare_prediction_frame(df, model, feature_list, split_idx)
    filtered = filter_predictions(prob_df, CONFIDENCE_THRESHOLD, ONLY_TRADE_SIGNALS)
    filtered = add_trade_returns(filtered)

    trade_log, summary = simulate_account(filtered)

    if not trade_log.empty:
        trade_log.to_csv(TRADE_LOG_PATH, index=False)
        make_plots(trade_log)

    threshold_sweep_df = run_threshold_sweep(prob_df)

    lines = [
        "========== SIMULATION SUMMARY ==========",
        f"Starting balance: ${summary['starting_balance']:.2f}",
        f"Ending balance:   ${summary['ending_balance']:.2f}",
        f"Net profit:       ${summary['net_profit']:.2f}",
        f"Return:           {summary['return_pct']:.2f}%",
        f"Max drawdown:     {summary['max_drawdown_pct']:.2f}%",
        f"Total trades:     {summary['total_trades']}",
        f"Win rate:         {summary['win_rate']:.2f}%",
        f"Longest win streak:  {summary['longest_win_streak']}",
        f"Longest loss streak: {summary['longest_loss_streak']}",
        f"Stop-loss hits:   {summary['stop_loss_hits']}",
        f"Take-profit hits: {summary['take_profit_hits']}",
        f"Skipped for margin: {summary['skipped_for_margin']}",
        f"Blown account:    {summary['blown_account']}",
        "",
        "========== ASSUMPTIONS ==========",
        f"Account leverage: {ACCOUNT_LEVERAGE}x",
        f"Lot size: {LOT_SIZE}",
        f"Contract size: {CONTRACT_SIZE}",
        f"Reference price fallback: {REFERENCE_PRICE}",
        f"Stop loss = {STOP_LOSS_ACCOUNT_FRACTION:.2%} of current balance",
        f"Risk reward ratio = 1:{RISK_REWARD_RATIO:.2f}",
        f"Use take profit: {USE_TAKE_PROFIT}",
        f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}",
        f"Only trade signals: {ONLY_TRADE_SIGNALS}",
        f"Flat broker fee per trade: ${BROKER_FEE_PER_TRADE:.2f}",
        f"Average position notional: ${summary['avg_position_notional']:.2f}",
        f"Average margin required:   ${summary['avg_margin_required']:.4f}",
        f"Average SL amount:         ${summary['avg_sl_amount_usd']:.2f}",
        f"Average TP amount:         ${summary['avg_tp_amount_usd']:.2f}",
        "",
        "Note: This version uses money-based SL and TP:",
        "- fixed 0.01 lot size",
        "- stop loss = a chosen fraction of current balance",
        "- take profit = stop loss amount x risk-reward ratio",
        "- both are dynamic as balance changes",
        "",
        "========== THRESHOLD SWEEP ==========",
    ]

    print("\n".join(lines))
    print(threshold_sweep_df.to_string(index=False))

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
        f.write(threshold_sweep_df.to_string(index=False))
        f.write("\n")

    print(f"\nTrade log saved to: {TRADE_LOG_PATH}")
    print(f"Summary saved to:   {SUMMARY_PATH}")
    if not trade_log.empty:
        print(f"Equity curve saved to:   {EQUITY_PLOT_PATH}")
        print(f"Drawdown curve saved to: {DRAWDOWN_PLOT_PATH}")
    print(f"Threshold sweep saved to: {THRESHOLD_SWEEP_PATH}")


if __name__ == "__main__":
    main()
