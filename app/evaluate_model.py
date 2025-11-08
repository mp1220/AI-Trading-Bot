"""Evaluate latest trained model on unseen market data."""

import pandas as pd
import joblib
import datetime as dt
from pathlib import Path
from app.storage import DB
from app.features import ema, rsi, realized_vol
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_latest_model(model_dir: str = "models/daily") -> Path:
    """Find newest model by timestamp."""
    path = Path(model_dir)
    pkl_files = sorted(path.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not pkl_files:
        raise FileNotFoundError("No models found in models/daily/")
    return pkl_files[0]


def build_features(db: DB, start: str, end: str) -> tuple[pd.DataFrame, pd.Series]:
    """Extract market data and build features for evaluation window."""
    sql = """
        SELECT timestamp AS ts, symbol AS ticker, close, volume
        FROM market_bars
        WHERE timestamp BETWEEN %(start)s AND %(end)s
        ORDER BY symbol, timestamp
    """
    bars = db.to_df(sql, {"start": start, "end": end})
    if bars.empty:
        raise ValueError("No bars found in this date range.")

    frames = []
    for _, group in bars.groupby("ticker"):
        group = group.sort_values("ts").copy()
        group["ema_fast"] = ema(group["close"], 12)
        group["ema_slow"] = ema(group["close"], 26)
        group["rsi"] = rsi(group["close"], 14)
        group["vol"] = realized_vol(group["close"], 20)
        group["return_fwd"] = group["close"].shift(-3) / group["close"] - 1
        frames.append(group)

    dataset = pd.concat(frames, ignore_index=True).dropna()
    y = (dataset["return_fwd"] > 0).astype(int)
    X = dataset[["ema_fast", "ema_slow", "rsi", "vol", "volume"]]
    return X, y


def evaluate():
    db = DB()
    try:
        model_path = load_latest_model()
        print(f"✅ Loaded model: {model_path.name}")

        # Use the day after training for unseen validation
        start = dt.datetime(2025, 9, 25)
        end = dt.datetime(2025, 10, 7)
        X, y = build_features(db, start, end)

        model = joblib.load(model_path)
        preds = model.predict(X)

        acc = accuracy_score(y, preds)
        print(f"\n📊 Accuracy: {acc:.3f}")
        print("\n🔍 Classification report:")
        print(classification_report(y, preds))
        print("\n📉 Confusion matrix:")
        print(confusion_matrix(y, preds))

        print(f"\n✅ Evaluated {len(y)} samples from {start.date()} to {end.date()}")
    finally:
        db.close()


if __name__ == "__main__":
    evaluate()
