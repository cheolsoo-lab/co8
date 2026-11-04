"""스윙(길게) 모드 현재 상태: 롱만 규칙 12개 묶음 평균.  python swing_live.py
12개 = 반전확인 10·15·20·25·30% / 돌파 20·50·100일 / 장기방향 50·100·200일 / 방향합의(20·50·100·200일 중 3개 이상)
각 규칙이 '롱'이면 1, 아니면 0 → 평균 = 들고 있을 비중(0~100%). 일봉 마감 기준, 다음 날 진입."""
import numpy as np, pandas as pd
import data, swing as W

RULES = {}
for d in (0.10, 0.15, 0.20, 0.25, 0.30): RULES[f"반전확인 {d:.0%}"] = lambda df, d=d: W.rule_reversal2(df["close"].to_numpy(float), d)
for n in (20, 50, 100): RULES[f"돌파 {n}일"] = lambda df, n=n: W.rule_breakout(df, n)
for L in (50, 100, 200): RULES[f"장기방향 {L}일"] = lambda df, L=L: W.rule_tsmom(df["close"].to_numpy(float), L)
RULES["방향합의"] = lambda df: W.rule_consensus(df["close"].to_numpy(float))


def level(w):
    """3단계로 단순화: 롱 규칙 25% 미만 → 0%, 75% 미만 → 50%, 그 이상 → 100%. (8년 검증: 성적 같고 조정 연 65→16회)"""
    return 0.0 if w < 0.25 else (0.5 if w < 0.75 else 1.0)


def state(df: pd.DataFrame) -> dict:
    votes = {k: int(fn(df)[-1] > 0) for k, fn in RULES.items()}
    w = level(float(np.mean(list(votes.values()))))
    raw = np.mean([np.where(fn(df) > 0, 1.0, 0.0) for fn in RULES.values()], axis=0)
    hist = np.array([level(x) for x in raw])
    chg = np.flatnonzero(np.abs(np.diff(hist)) > 1e-9)
    last = df.index[chg[-1] + 1] if len(chg) else None
    c = df["close"].to_numpy(float)
    return {"weight": w, "votes": votes, "close": float(c[-1]), "time": str(df.index[-1]), "last_change": str(last) if last is not None else "",
            "prev_weight": float(hist[chg[-1]]) if len(chg) else w,
            "hi50": float(df["high"].iloc[-51:-1].max()), "lo50": float(df["low"].iloc[-51:-1].min()),
            "hi20": float(df["high"].iloc[-21:-1].max()), "ret50": float(c[-1] / c[-51] - 1), "ret200": float(c[-1] / c[-201] - 1) if len(c) > 200 else float("nan")}


def live(symbols=("BTCUSDT", "ETHUSDT")):
    out = {}
    for s in symbols:
        df = data.drop_unclosed(data.fetch_candles(s, "1D", 500), "1D")
        out[s] = state(df)
    return out


if __name__ == "__main__":
    for s, v in live().items():
        print(s, f"비중 {v['weight']:.0%} (직전 {v['prev_weight']:.0%}, 변경 {v['last_change'][:10]})", f"50일 {v['ret50']:+.0%} 200일 {v['ret200']:+.0%}", f"20일 최고 {v['hi20']:,.1f} 50일 최고 {v['hi50']:,.1f}")
        print("   ", {k: v_ for k, v_ in v["votes"].items()})
