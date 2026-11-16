"""거래소 순유출 (11-17 재검토): 코인이 거래소에서 빠져나가나(보관·매집) 들어오나(팔 준비) — 가격·프리미엄과 다른 온체인 재료.
  값 = 최근 30일 (거래소 입금 − 출금) ÷ 유통량 → 지난 250일 기준 z, 부호 뒤집음 (+ = 순유출 = 롱 쪽). 자료: 코인메트릭스 커뮤니티(하루 단위, 이틀 늦게 반영).
검증(research/newlab.py · newsleeve.py · newsleeve2.py · newwf.py):
  1주 뒤 수익과 순위상관 −0.07~−0.09 (순입금 기준, BTC·ETH·앞뒤 절반 같은 부호, 시간 어긋낸 우주 p≤0.03), 과거 수익을 빼도 유지.
  주머니 ×0.5: 바이낸스 2018-06~2022-07 샤프 1.50→1.92 (p=0.00) · 비트겟 2022-08~ (흐름+추적 위에) 2.15→2.30, 연 +47%→+55%, 낙폭 같음 (p=0.03).
  다만 창 14~90일 이웃은 비트겟에서 −0.02~+0.04로 작고 BTC 단독은 1.93→1.88 → 엄격 기준 미달. 과거 자료는 거래소 주소가 나중에 보정된 값이라 실시간보다 좋게 보일 수 있음.
  → 기본은 표시만, 설정에서 켜면 ×0.25 (바이낸스 +0.21 · 비트겟 +0.08)."""
from __future__ import annotations
import numpy as np, pandas as pd, requests

CM = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
W = 0.25; WIN = 30; ZN = 250; LAG = 2


def _net(asset: str) -> pd.Series:
    st = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=ZN + WIN + 120)).strftime("%Y-%m-%d")
    r = requests.get(CM, params={"assets": asset, "metrics": "FlowInExNtv,FlowOutExNtv,SplyCur", "frequency": "1d", "page_size": 1000, "start_time": st},
                     timeout=20, headers={"User-Agent": "btceth-app"})
    r.raise_for_status(); d = pd.DataFrame(r.json()["data"])
    d.index = pd.to_datetime(d["time"], utc=True); d = d[["FlowInExNtv", "FlowOutExNtv", "SplyCur"]].apply(pd.to_numeric, errors="coerce").sort_index()
    return ((d["FlowInExNtv"] - d["FlowOutExNtv"]) / d["SplyCur"].ffill()).dropna()


def state() -> dict:
    out = {}; last = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=LAG)      # 그날 값은 이틀 뒤부터 사용 (검증과 같은 지연)
    for coin in ("BTC", "ETH"):
        nf = _net(coin.lower()); s30 = nf.rolling(WIN).sum(); z = -(s30 - s30.rolling(ZN, min_periods=50).mean()) / s30.rolling(ZN, min_periods=50).std()
        z = z[z.index <= last].dropna(); zz = float(z.iloc[-1])
        out[coin] = {"z": zz, "s": float(np.clip(zz / 2, -1, 1)), "pct30": float(s30.reindex(z.index).iloc[-1] * 100), "t": str(z.index[-1])[:10], "series": z}
    return out


def label(z: float) -> str:
    return "순유출 (롱 쪽)" if z >= 0.5 else ("순입금 (숏 쪽)" if z <= -0.5 else "보통")
