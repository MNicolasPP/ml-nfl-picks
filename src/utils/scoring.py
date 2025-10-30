from __future__ import annotations
import numpy as np

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def edge(prob_over: float, line_price: float = -110) -> float:
    if line_price < 0:
        implied = (-line_price) / ((-line_price) + 100)
        payout = 100 / (-line_price)
    else:
        implied = 100 / (line_price + 100)
        payout = line_price / 100
    ev = prob_over * payout - (1 - prob_over)
    return ev
