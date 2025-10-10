from dataclasses import dataclass

@dataclass
class Decision:
    decision: str
    reason: str

class Reasoner:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def combine(self, prob_up: float, sentiment: float, volatility: float) -> Decision:
        model_cfg = self.cfg.get("model") or {}
        th_buy = float(model_cfg.get("threshold_buy", 0.58))
        th_sell = float(model_cfg.get("threshold_sell", 0.42))
        sentiment_cfg = self.cfg.get("sentiment") or {}
        s_weight = float(sentiment_cfg.get("weight", 0.25))
        score = prob_up + s_weight * ((sentiment + 1) / 2 - 0.5)
        # simple volatility filter
        if volatility and volatility > 0.03:
            return Decision("HOLD", f"High vol {volatility:.2%}, deferring")
        if score >= th_buy:
            return Decision("BUY", f"score={score:.2f}, prob={prob_up:.2f}, sent={sentiment:.2f}")
        if score <= th_sell:
            return Decision("SELL", f"score={score:.2f}, prob={prob_up:.2f}, sent={sentiment:.2f}")
        return Decision("HOLD", f"score={score:.2f}, prob={prob_up:.2f}, sent={sentiment:.2f}")
