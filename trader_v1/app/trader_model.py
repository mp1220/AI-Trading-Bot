# app/trader_model.py
import pandas as pd
import numpy as np
from .features import ema, rsi, realized_vol

class TraderModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        model_cfg = cfg.get("model", cfg)
        self.fast = model_cfg.get("fast_ema", 12)
        self.slow = model_cfg.get("slow_ema", 26)
        self.rsi_lo = model_cfg.get("rsi_oversold", 35)
        self.rsi_hi = model_cfg.get("rsi_overbought", 65)
        self.vol_cap = model_cfg.get("vol_cap", 0.6)  # cap effect of vola

    def _score_row(self, row) -> tuple[float,str,dict]:
        # Convert features to signals
        trend = np.tanh(row["ema_fast"] - row["ema_slow"])
        mom = np.tanh((50 - abs(row["rsi"] - 50)) / 10)  # closer to 50 = stronger continuation
        sent = np.tanh(row.get("sentiment", 0.0))
        vol_raw = row.get("vol", 0.0)
        try:
            vol_val = float(vol_raw)
        except (TypeError, ValueError):
            vol_val = 0.0
        if not np.isfinite(vol_val):
            vol_val = 0.0
        vol = min(self.vol_cap, max(0.0, vol_val))

        # Combine (weights can be tuned)
        raw = 0.45*trend + 0.25*mom + 0.30*sent
        # Volatility penalty (reduce confidence in stormy markets)
        raw = raw * (1 - 0.5*vol)

        prob_up = 0.5 + 0.5*raw
        if prob_up > 0.58:
            action = "BUY"
        elif prob_up < 0.42:
            action = "SELL"
        else:
            action = "HOLD"

        return float(prob_up), action, {"trend":float(trend),"mom":float(mom),"sent":float(sent),"vol":float(vol)}

    def predict_many(self, bars: pd.DataFrame, sentiments: dict[str,float]) -> list[dict]:
        recs = []
        for ticker, df in bars.groupby("ticker"):
            df = df.sort_values("ts").copy()
            df["ema_fast"] = ema(df["close"], self.fast)
            df["ema_slow"] = ema(df["close"], self.slow)
            df["rsi"] = rsi(df["close"], 14)
            df["vol"] = realized_vol(df["close"], 20)
            df["sentiment"] = sentiments.get(ticker, 0.0)
            last = df.iloc[-1]
            prob, action, details = self._score_row(last)
            recs.append({"ticker": ticker, "ts": last["ts"], "prob_up": prob, "sentiment": float(df["sentiment"].iloc[-1]), "action": action, "details": details})
        return recs
