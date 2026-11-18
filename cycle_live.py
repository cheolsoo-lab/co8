"""반감기 사이클 위치 (11-18 ETF 전후 분석, research/etf2.py · etf3.py · etf4.py).
국면은 미리 아는 날짜(반감기 뒤 일수)로만 나눔 — 고점은 지나고 나서야 알기 때문.
  ① 반감기~365일 · ② 365~540일 (역사적 고점 526·547·534일) · ③ 540~900일 (고점 뒤 하락장, 저점 889·924일) · ④ 900일~다음 반감기
앱(배율 1) 국면별 성적 BTC 2015-07~2026-11: ① 샤프 1.73 · ② 2.47 · ③ 0.90 · ④ 1.99 — ③에서 가장 약함(휩쏘), 그냥 보유는 ③ 연 −69%.
시험 크기 [1.25, 1.25, 0.75, 1]: ETF 전·후, 사이클 3개 모두 +0.05~+0.15였지만 교차 검토에서 기각 —
  540·900일 경계가 같은 고점·저점에 맞춘 값이라 표본 안 결과이고, 경계를 450/810·480/900·620/1000일로 옮기면 ETF 뒤 성적이 −로 뒤집힘. BTC 단독 ETF 뒤 −0.06.
  → 크기 조절에는 쓰지 않고 '위험 달력' 표시로만 사용."""
from __future__ import annotations
import numpy as np, pandas as pd

HALV = [pd.Timestamp(x, tz="UTC") for x in ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20", "2028-04-15", "2032-04-10")]   # 2028·2032년은 블록 속도로 본 추정
EDGES = (365, 540, 900)
NAMES = ("① 반감기~1년", "② 1년~540일 (역사적 고점 구간)", "③ 540~900일 (고점 뒤 하락장)", "④ 900일~다음 반감기 (바닥·준비)")
MULT = (1.25, 1.25, 0.75, 1.0)
SHARPE = (1.73, 2.47, 0.90, 1.99)


def _days(ts: pd.Timestamp) -> tuple[int, int]:
    ts = pd.Timestamp(ts); ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    i = max(k for k in range(len(HALV)) if HALV[k] <= ts); return i, (ts - HALV[i]).days


def phase(ts) -> int:
    _, d = _days(ts); return int(np.searchsorted(EDGES, d, side="right"))


def mult_series(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.array([MULT[phase(t)] for t in idx])


def state(ts=None) -> dict:
    ts = pd.Timestamp.now(tz="UTC") if ts is None else ts; i, d = _days(ts); ph = phase(ts); h = HALV[i]
    cal = []
    for k in range(i, min(i + 2, len(HALV) - 1)):
        hv = HALV[k]; nxt = HALV[k + 1]; bounds = [hv] + [hv + pd.Timedelta(days=e) for e in EDGES] + [nxt]
        for j in range(4):
            if bounds[j + 1] > ts: cal.append({"국면": NAMES[j], "시작": bounds[j].strftime("%Y-%m"), "끝": bounds[j + 1].strftime("%Y-%m"), "앱 샤프(역사)": SHARPE[j], "시험 크기": MULT[j]})
    return {"days": d, "phase": ph, "name": NAMES[ph], "mult": MULT[ph], "halving": h.strftime("%Y-%m-%d"), "next": HALV[i + 1].strftime("%Y-%m"), "calendar": cal}
