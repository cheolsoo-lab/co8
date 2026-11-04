"""'고르지 않고 평균' 방식의 현재 상태.  python robust_live.py
규칙(전부 미리 고정, 코인별 튜닝 없음):
  방향 = 0.5 × TSMOM(7·14·30·60·90일 수익 부호 평균) + 0.5 × 호흡(같은 방향 4H 봉 k=3~6개 뒤 h=12·24·48봉 추종 평균)
  크기 = 방향 × min(연변동성 40% 타깃 / 현재 변동성, 3배) → 작은 변화는 무시(0.15)
  24개 코인 동일 비중(1/24). 코인 하나가 포트폴리오에서 차지하는 명목 = 자본 × 배율 × 크기 / 24."""
import numpy as np, pandas as pd
import data, novel as N, robust as RB

UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
            "HYPEUSDT", "RENDERUSDT", "TAOUSDT", "FILUSDT", "GRASSUSDT", "TIAUSDT", "STXUSDT", "NEARUSDT", "ARBUSDT", "APTUSDT", "ONDOUSDT", "ZROUSDT"]
BARS = 1000          # TSMOM 최장 540봉 + 변동성 안정화 여유
DEAD = 0.15          # 판정 경계: |합성 방향| < 0.15 이면 관망


def one(df: pd.DataFrame) -> dict:
    c = df["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))])
    sg = N.sigma(r); cap = np.minimum(N.TV / sg, N.LMAX)
    d1, d2 = RB.tsmom_dir(r), RB.breath_dir(r)
    d = 0.5 * d1 + 0.5 * d2
    pos = N.band(d * cap, 0.15)
    return {"close": float(c[-1]), "tsmom": float(d1[-1]), "breath": float(d2[-1]), "dir": float(d[-1]),
            "pos": float(pos[-1]), "ann": float(sg[-1] * np.sqrt(6 * 365)), "n": len(c),
            "time": str(df.index[-1])}


def triggers(df: pd.DataFrame) -> list:
    """다음 4H 봉 마감가에 따라 포지션이 어떻게 되는지 — 정확한 전환 가격.
    점수는 다음 마감가 p에 대해 계단 함수: 추세는 p가 L봉 전 종가(5개) 위/아래인지, 호흡은 p가 직전 종가 위/아래인지로만 바뀜.
    → 그 6개 가격이 경계. 각 구간 대표값으로 실제 계산(완충 포함)해 같은 결과끼리 합침."""
    c = df["close"].to_numpy(float); last = float(c[-1])
    pts = sorted(set([float(c[-L]) for L in RB.LS] + [last]))
    edges = [-np.inf] + pts + [np.inf]
    nxt = df.index[-1] + (df.index[-1] - df.index[-2])
    base = df[["open", "high", "low", "close", "volume"]]
    vol = float(df["volume"].iloc[-20:].mean())
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # 크기는 '현재가에 가까운 쪽 경계 바로 안쪽'에서 계산 (먼 가격일수록 변동성이 커져 크기가 줄어드는 효과는 구간마다 다름)
        p = hi * 0.999 if hi <= last else lo * 1.001
        new = pd.DataFrame({"open": [last], "high": [max(last, p)], "low": [min(last, p)], "close": [p], "volume": [vol]}, index=[nxt])
        v = one(pd.concat([base, new]))
        rows.append({"lo": lo, "hi": hi, "pos": v["pos"], "dir": v["dir"]})
    merged = []
    for r in rows:
        if merged and verdict(merged[-1]["dir"]) == verdict(r["dir"]) and abs(merged[-1]["pos"] - r["pos"]) < 0.15:
            m = merged[-1]; m["hi"] = r["hi"]
            if abs(r["lo"] - last) < abs(m["lo"] - last) or not np.isfinite(m["lo"]): m["pos"], m["dir"] = (r["pos"], r["dir"]) if r["lo"] >= last else (m["pos"], m["dir"])
        else:
            merged.append(dict(r))
    return merged


def live_all(symbols=UNIVERSE, bars=BARS) -> dict:
    out = {}
    for s in symbols:
        try:
            df = data.drop_unclosed(data.fetch_candles(s, "4H", bars), "4H")
            out[s] = one(df)
        except Exception as e:
            out[s] = {"err": str(e)}
    return out


def verdict(d: float) -> str:
    if abs(d) < DEAD: return "관망"
    side = "롱" if d > 0 else "숏"
    return f"{side} (강)" if abs(d) >= 0.6 else f"{side} (약)"


if __name__ == "__main__":
    res = live_all()
    for s, v in res.items():
        if "err" in v: print(s, "ERR", v["err"]); continue
        print(f"{s:10s} {verdict(v['dir']):8s} 합성 {v['dir']:+.2f} (추세 {v['tsmom']:+.2f} · 호흡 {v['breath']:+.2f}) 크기 {v['pos']:+.2f} 연변동성 {v['ann']:.0%}")
    ok = [v for v in res.values() if "err" not in v]
    print(f"\n순노출 {sum(v['pos'] for v in ok)/len(UNIVERSE):+.2f}배 · 총노출 {sum(abs(v['pos']) for v in ok)/len(UNIVERSE):.2f}배 (자본 대비, 배율 1.0 기준)")
