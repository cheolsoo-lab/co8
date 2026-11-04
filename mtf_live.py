"""흐름 정렬 역가중 — 실시간 계산.  주봉·일봉 방향이 4H 포지션과 몇 개 같은가 → 크기 배수.
모두 같음(늙은 추세) 0.5배 · 하나만 같음 1.0배 · 모두 반대(새 흐름 태동) 1.5배.  검증: mtf.py · mtf2.py (5년 샤프 1.04 → 1.36, 낙폭 −19% → −13%)."""
import numpy as np, pandas as pd
import indicators as ta, mtf as MT

W = (1.5, 1.0, 0.5)


def state(df4: pd.DataFrame) -> dict:
    idx = df4.index
    dD, _ = MT.tf_signal(ta.resample_ohlcv(df4, "1D"), (5, 10, 20, 40, 60), (2, 3, 4), (3, 5, 10))
    dW, _ = MT.tf_signal(ta.resample_ohlcv(df4, "W-MON"), (4, 8, 13, 26), (2, 3), (2, 4, 6))
    d_ = float(MT.to_4h(dD, idx, pd.Timedelta(days=1))[-1]); w_ = float(MT.to_4h(dW, idx, pd.Timedelta(days=7))[-1])
    return {"dD": d_, "dW": w_}


def factor(sign_pos: float, st: dict) -> tuple:
    if sign_pos == 0: return 1.0, None
    ag = int(np.sign(st["dD"]) == sign_pos) + int(np.sign(st["dW"]) == sign_pos)
    return W[ag], ag
