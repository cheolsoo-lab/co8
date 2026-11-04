"""위에서 아래로: 주봉 큰 흐름 → 일봉 → 4시간 상태 지도.  python topdown.py
상태(8) = 주봉 방향(상승/하락) × 일봉이 주봉과 같은가 × 4시간이 주봉과 같은가.
  상승 주봉: 순풍(일·4H 같음) / 단기 눌림(일 같음·4H 반대) / 조정 후 재개 시도(일 반대·4H 같음) / 깊은 조정(일·4H 반대). 하락 주봉은 거울.
측정(그 시점 정보만, 이후 결과): 주봉 방향으로의 1주·24시간 수익, 4주 안에 주봉 방향이 뒤집힌 비율, 앞/뒤 절반.
행동표(앞 절반으로만 결정): 상태별 1주 수익(주봉 방향 기준)의 부호로 롱/숏, |t|<1이면 관망 → 뒤 절반에서 확인. 현재 방식(역정렬 가중)과 비교."""
import json, math, warnings
import numpy as np, pandas as pd
import novel as N, precision as PR, fakes as FK, mtf as MT

warnings.filterwarnings("ignore")
COST = 0.0003
NAMES = {(1, 1): "순풍", (1, 0): "단기 눌림", (0, 1): "조정 후 재개 시도", (0, 0): "깊은 조정 (전환 의심)"}


def states(F):
    w = np.sign(F["dW"]); dsame = (np.sign(F["dD"]) == w).astype(int); hsame = (np.sign(F["d"]) == w).astype(int)
    code = np.where(w == 0, -1, (w > 0).astype(int) * 4 + dsame * 2 + hsame)   # 0~7, w>0이면 4~7
    return code, w


def label(code):
    if code < 0: return "주봉 중립"
    up = code >= 4; ds, hs = (code % 4) // 2, code % 2
    return f"{'상승' if up else '하락'} 주봉 · {NAMES[(ds, hs)]}"


def main():
    D, FR = MT.build(); out = {}
    rows = []
    for s, F in FR.items():
        code, w = states(F); r = F["r"]; n = len(r); cs = np.cumsum(r)
        f42 = np.full(n, np.nan); f42[:-42] = cs[42:] - cs[:-42]; f6 = np.full(n, np.nan); f6[:-6] = cs[6:] - cs[:-6]
        wsign = w.copy(); flip = np.full(n, np.nan)
        for t in range(n - 168):
            if wsign[t] != 0: flip[t] = float((wsign[t + 1:t + 169] == -wsign[t]).any())
        half = (np.arange(n) >= n // 2).astype(int) + 1
        ok = np.arange(n) > 1300   # 주봉 26주 추세 준비
        for t in np.flatnonzero(ok & (code >= 0)):
            rows.append((s, t, code[t], w[t], w[t] * f42[t], w[t] * f6[t], flip[t], half[t]))
    S = pd.DataFrame(rows, columns=["sym", "t", "code", "w", "r1w", "r24", "flip4w", "half"])
    base_flip = S["flip4w"].mean()
    print(f"══ 1) 상태 지도 — BTC·ETH 4H 5년 (주봉 방향 기준 수익 · 4주 안 주봉 전환 확률, 기준 {base_flip:.0%}) ══")
    print(f"{'상태':32s} {'비중':>5s} | {'1주 수익(주봉 방향)':>18s} {'적중':>5s} | {'24시간':>7s} | {'4주 내 주봉 전환':>12s} | 앞/뒤 1주 수익 · 앞/뒤 전환")
    table = {}
    for c in range(8):
        g = S[S["code"] == c]
        if len(g) < 50: continue
        a, b = g[g["half"] == 1], g[g["half"] == 2]
        neff = len(g) / 42; t1 = g["r1w"].mean() / (g["r1w"].std() / math.sqrt(max(neff, 1)))
        table[c] = {"name": label(c), "share": len(g) / len(S), "r1w": float(g["r1w"].mean()), "hit": float((g["r1w"] > 0).mean()), "r24": float(g["r24"].mean()),
                    "flip": float(g["flip4w"].mean()), "r1w_a": float(a["r1w"].mean()), "r1w_b": float(b["r1w"].mean()),
                    "flip_a": float(a["flip4w"].mean()), "flip_b": float(b["flip4w"].mean()), "t": float(t1)}
        T = table[c]
        print(f"{T['name']:32s} {T['share']:5.0%} | {T['r1w']:+17.2%} {T['hit']:5.0%} | {T['r24']:+7.2%} | {T['flip']:11.0%} | {T['r1w_a']:+.2%}/{T['r1w_b']:+.2%} · {T['flip_a']:.0%}/{T['flip_b']:.0%}")
    out["table"] = {str(k): v for k, v in table.items()}; out["base_flip"] = float(base_flip)
    # 2) 전환 예측력: 일·4H 상태로 4주 내 주봉 전환을 가려내나
    from sklearn.metrics import roc_auc_score
    S2 = S.dropna(subset=["flip4w"])
    deep = ((S2["code"] % 4) == 0).astype(float)                    # 깊은 조정
    score = 2 * ((S2["code"] % 4) == 0) + 1 * ((S2["code"] % 4) == 1) + 1 * ((S2["code"] % 4) == 2)   # 반대 개수
    for h in (1, 2):
        m = S2["half"] == h
        print(f"  {'앞' if h == 1 else '뒤'} 절반: '반대 개수'로 4주 내 주봉 전환 구별력 AUC {roc_auc_score(S2[m]['flip4w'], score[m]):.2f} · 깊은 조정일 때 전환 {S2[m & (deep > 0)]['flip4w'].mean():.0%} vs 순풍일 때 {S2[m & ((S2['code'] % 4) == 3)]['flip4w'].mean():.0%}")
    # 3) 상태별 행동표 — 앞 절반으로 결정, 뒤 절반 확인
    act = {}
    for c in range(8):
        g = S[(S["code"] == c) & (S["half"] == 1)]
        if len(g) < 50: act[c] = 0; continue
        t1 = g["r1w"].mean() / (g["r1w"].std() / math.sqrt(max(len(g) / 42, 1)))
        wdir = 1 if c >= 4 else -1
        act[c] = 0 if abs(t1) < 1 else int(np.sign(g["r1w"].mean()) * wdir)
    print("\n══ 2) 상태별 행동표 (앞 절반으로 결정) ══")
    for c in range(8): print(f"  {label(c):32s} → {({1: '롱', -1: '숏', 0: '관망'})[act[c]]}")
    out["actions"] = {label(c): act[c] for c in range(8)}
    P_tab, P_base, P_inv = {}, {}, {}
    for s, F in FR.items():
        code, w = states(F)
        dirn = np.array([act.get(int(c), 0) if c >= 0 else 0 for c in code], float)
        P_tab[s] = N.band(dirn * F["cap"], 0.15)
        P_base[s] = F["base"]
        ag = (np.sign(F["dD"]) == np.sign(F["base"])).astype(int) + (np.sign(F["dW"]) == np.sign(F["base"])).astype(int)
        P_inv[s] = F["base"] * np.array([1.5, 1.0, 0.5])[ag]
    n = len(FR["BTCUSDT"]["r"]); sl = slice(n // 2, n)
    print(f"\n══ 3) 뒤 절반 확인 ({FR['BTCUSDT']['idx'][n//2].date()} ~) ══")
    rb = FK.judge(FR, P_base, None, sl); FK.show("현재 앱 (이전 방식)", rb)
    ri = FK.judge(FR, P_inv, None, sl); FK.show("역정렬 가중 (지금 기본)", ri)
    rt = FK.judge(FR, P_tab, ri, sl); FK.show("상태별 행동표 (단독)", rt)
    P_mix = {s: P_inv[s] + 0.5 * P_tab[s] for s in FR}
    rm = FK.judge(FR, P_mix, ri, sl); FK.show("역정렬 + 행동표 ½ 더하기", rm)
    out["test"] = {"base": rb, "inverse": ri, "table": rt, "mix": rm}
    now = {s: label(int(states(F)[0][-1])) for s, F in FR.items()}; print("\n지금:", now); out["now"] = now
    with open("state/topdown_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
