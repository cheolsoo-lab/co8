"""앱용: 거시 방향(일봉) 실시간 계산.  검증은 macro.py (BTC 11년 · ETH 10년)."""
from __future__ import annotations
import numpy as np, pandas as pd

NAMES = ("200일선 위/아래", "200일선 기울기", "50일선 vs 200일선", "1년 수익")


def state(dd: pd.DataFrame) -> dict | None:
    """dd: 마감된 일봉 (최소 566개). G1·G2·G3·G5 평균 ≥ +0.5 상승장, ≤ −0.5 하락장, 사이 중립."""
    c = dd["close"].astype(float)
    if len(c) < 566: return None
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    g = [float(np.sign(c.iloc[-1] - s200.iloc[-1])), float(np.sign(s200.iloc[-1] - s200.iloc[-21])),
         float(np.sign(s50.iloc[-1] - s200.iloc[-1])), float(np.sign(np.log(c.iloc[-1] / c.iloc[-366])))]
    m = float(np.mean(g))
    return {"parts": dict(zip(NAMES, g)), "mval": m, "label": "상승장" if m >= 0.5 else ("하락장" if m <= -0.5 else "중립"),
            "dd1y": float(c.iloc[-1] / c.iloc[-365:].max() - 1), "dist200": float(c.iloc[-1] / s200.iloc[-1] - 1),
            "s200": float(s200.iloc[-1]), "date": str((dd.index[-1] + pd.Timedelta(days=1)).date())}
