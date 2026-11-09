"""돈의 흐름 (11-09 재분석): 가격 모양 대신 '실제로 돈이 들어오나'를 보는 두 값.
  테더 프리미엄     = 코인베이스 USDT-USD − 1  (코인을 사려고 달러→테더로 바꾸는 수요가 많으면 + )
  코인베이스 프리미엄 = 코인베이스 BTC-USD ÷ 비트겟 현물 BTCUSDT − 1, 3일(6시간 봉 12개) 평균  (미국 현물 매수세)
  각각 지난 250일(6시간 봉 1000개) 기준 z → 두 z 평균 = 흐름 점수 → 주머니 = clip(점수/2, −1, 1) × 변동성 목표 크기 × 0.5 를 앱 포지션에 더함.
검증(research/flowrule.py · wf.py · flow_bitget.py · flow_final.py):
  1주 뒤 수익과 순위상관 0.08~0.14 (BTC·ETH, 앞·뒤 절반 모두 +, 시간 어긋낸 우주 p<0.01), 과거 수익을 빼도 그대로 → 추세와 다른 정보
  비트겟 무기한 2022-08~ (실시간 재료 그대로): 샤프 1.77 → 1.95, 연 +36% → +54%, 낙폭 −11% → −16%, 시간 어긋낸 우주 p=0.027
  매 분기 그때까지 자료로만 재료를 고른 걷기 검증에서도 +0.05~0.15. 2022·2025년은 주머니 단독 손실."""
from __future__ import annotations
import time
import numpy as np, pandas as pd, requests

CB = "https://api.exchange.coinbase.com"; BG = "https://api.bitget.com"; OKX = "https://www.okx.com"
W = 0.5                      # 주머니 비중 (×0.25·×0.75 이웃은 엄격 판정 한두 칸 미달)
N6 = 1000                    # z 기준 기간 (6시간 봉 1000개 ≈ 250일)


def _cb6(prod: str, n: int = N6 + 20) -> pd.Series:
    t1 = pd.Timestamp.now(tz="UTC").floor("6h") + pd.Timedelta(hours=6); cur = t1 - pd.Timedelta(hours=6 * n); rows = []
    while cur < t1:
        end = min(cur + pd.Timedelta(hours=6 * 300), t1)
        for k in range(3):
            r = requests.get(f"{CB}/products/{prod}/candles", params={"granularity": 21600, "start": cur.isoformat(), "end": end.isoformat()},
                             timeout=15, headers={"User-Agent": "btceth-app"})
            if r.status_code == 200: rows += r.json(); break
            time.sleep(0.8 * (k + 1))
        else:
            raise RuntimeError(f"코인베이스 {prod} 실패 {r.status_code}")
        cur = end; time.sleep(0.12)
    d = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"]).astype(float)
    s = pd.Series(d["close"].to_numpy(), index=pd.to_datetime(d["ts"].astype("int64"), unit="s", utc=True))
    return s[~s.index.duplicated()].sort_index()


def _spot6(sym: str) -> pd.Series:
    """비트겟 현물 6시간(UTC 0·6·12·18시) 종가 1000개, 막히면 OKX 현물."""
    try:
        r = requests.get(f"{BG}/api/v2/spot/market/candles", params={"symbol": sym, "granularity": "6Hutc", "limit": 1000}, timeout=15).json()
        if r.get("code") != "00000" or not r.get("data"): raise RuntimeError(str(r)[:120])
        x = r["data"]
    except Exception:
        x, after = [], None
        while len(x) < N6:
            p = {"instId": sym.replace("USDT", "-USDT"), "bar": "6Hutc", "limit": 100}
            if after: p["after"] = after
            d = requests.get(f"{OKX}/api/v5/market/history-candles", params=p, timeout=15).json().get("data") or []
            if not d: break
            x += d; after = d[-1][0]; time.sleep(0.12)
    s = pd.Series([float(r[4]) for r in x], index=pd.to_datetime([int(r[0]) for r in x], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index()


def _z(s: pd.Series) -> pd.Series:
    return (s - s.rolling(N6, min_periods=200).mean()) / s.rolling(N6, min_periods=200).std()


def state() -> dict:
    now = pd.Timestamp.now(tz="UTC"); done = lambda s: s[s.index + pd.Timedelta(hours=6) <= now]    # 끝난 6시간 봉만
    ut = done(_cb6("USDT-USD")); uz = _z(ut - 1); out = {"usdt": float(ut.iloc[-1]), "usdt_z": float(uz.iloc[-1]), "t": str((ut.index[-1] + pd.Timedelta(hours=6)).tz_convert("Asia/Seoul"))[:16]}
    for coin in ("BTC", "ETH"):
        cb = done(_cb6(f"{coin}-USD")); sp = done(_spot6(f"{coin}USDT"))
        prem = (cb / sp.reindex(cb.index)).dropna() - 1; p3 = prem.rolling(12).mean(); pz = _z(p3)
        u = float(uz.reindex(pz.index).ffill().iloc[-1]); score = float(np.nanmean([pz.iloc[-1], u]))
        out[coin] = {"prem3": float(p3.iloc[-1]), "prem_z": float(pz.iloc[-1]), "usdt_z": u, "score": score, "s": float(np.clip(score / 2, -1, 1))}
    return out


def label(score: float) -> str:
    return "매수 우위" if score >= 0.5 else ("매도 우위" if score <= -0.5 else "중립")
