"""앱용: 테더 도미넌스·BTC 도미넌스·TOTAL3 실시간 (코인마켓캡 차트 + DefiLlama). 검증은 coinwise.py — 매매에는 쓰지 않음(참고)."""
from __future__ import annotations
import numpy as np, pandas as pd, requests

H = {"User-Agent": "Mozilla/5.0"}
CMC = "https://api.coinmarketcap.com/data-api/v3/global-metrics/quotes/historical"


def _trend(s: pd.Series) -> float:
    l = np.log(s.dropna()); return float(np.mean([np.sign(l.iloc[-1] - l.iloc[-1 - L]) for L in (20, 50, 100)]))


def _z(s: pd.Series, h: int) -> float:
    """h일 로그 변화 ÷ 지난 1년 같은 길이 변화의 표준편차 (strength.py와 같은 정의)."""
    l = np.log(s.dropna()); ch = l - l.shift(h); sd = ch.iloc[-366:-1].std()
    return float(ch.iloc[-1] / sd) if sd > 0 else 0.0


def _lvl(s: pd.Series) -> float:
    w = s.dropna().iloc[-365:]; return float((w.iloc[-1] - w.min()) / (w.max() - w.min()))


def label(z: float) -> str:
    return "강한 하락" if z <= -1.5 else ("하락" if z <= -0.5 else ("보합" if z < 0.5 else ("상승" if z < 1.5 else "강한 상승")))


def state(days: int = 430) -> dict:
    t1 = pd.Timestamp.now(tz="UTC"); t0 = t1 - pd.Timedelta(days=days)
    j = requests.get(CMC, params={"format": "chart_crypto_details", "interval": "1d", "timeStart": int(t0.timestamp()), "timeEnd": int(t1.timestamp())}, headers=H, timeout=15).json()
    D = pd.DataFrame([{"time": pd.Timestamp(q["timestamp"]).tz_convert("UTC").normalize(), "total": q["quote"][0]["totalMarketCap"], "btc_d": q["btcDominance"], "eth_d": q["ethDominance"]}
                      for q in j["data"]["quotes"]]).drop_duplicates("time").set_index("time").sort_index()
    def llama(i):
        x = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", params={"stablecoin": i}, headers=H, timeout=20).json()
        return pd.Series({pd.to_datetime(int(r["date"]), unit="s", utc=True).normalize(): float(r["totalCirculatingUSD"]["peggedUSD"]) for r in x[-days - 5:]})
    D["usdt"] = llama(1).reindex(D.index).ffill(); D["usdc"] = llama(2).reindex(D.index).ffill()
    D["usdt_d"] = D["usdt"] / D["total"] * 100; D["total3"] = D["total"] * (1 - (D["btc_d"] + D["eth_d"]) / 100)
    D["total3_ex"] = D["total3"] - D["usdt"] - D["usdc"]; D["alt"] = 100 - D["btc_d"]; D["eva"] = D["eth_d"] / (100 - D["btc_d"] - D["eth_d"])
    return {"usdt_d": float(D["usdt_d"].iloc[-1]), "usdt_d_tr": _trend(D["usdt_d"]), "usdt_30": float(D["usdt"].iloc[-1] / D["usdt"].iloc[-31] - 1),
            "btc_d": float(D["btc_d"].iloc[-1]), "btc_d_tr": _trend(D["btc_d"]), "eth_d": float(D["eth_d"].iloc[-1]),
            "t3_tr": _trend(D["total3_ex"]), "t3_30": float(D["total3_ex"].iloc[-1] / D["total3_ex"].iloc[-31] - 1),
            "alt_tr": _trend(D["alt"]), "eva_tr": _trend(D["eva"]), "date": str(D.index[-1].date()),
            "u_z7": _z(D["usdt_d"], 7), "u_z30": _z(D["usdt_d"], 30), "u_lvl": _lvl(D["usdt_d"]), "t3_z7": _z(D["total3_ex"], 7), "t3_z30": _z(D["total3_ex"], 30)}


def arrow(x: float) -> str:
    return "▲" if x > 0.3 else ("▼" if x < -0.3 else "―")
