"""과최적을 피한 수익 극대화: 고르지 않고 '전부 평균'낸다.  python robust.py
원칙
 1) 파라미터를 하나로 고르지 않는다 — 미리 정한 여러 설정을 동일비중으로 섞는다(분산). 점 최적화가 없으니 맞춰진 값이 없다.
 2) 코인별 튜닝 없음 — 24개 코인에 같은 규칙, 같은 위험.
 3) 전 기간(5년)·연도별·앞뒤로 모두 보고, 설정 격자 전체(고원)를 공개한다.
 4) 크기는 '추정 엣지를 절반으로 깎은' 세계에서 성장률이 최대인 배율로 정한다(켈리의 불확실성 보정).
구성: TSMOM(시계열 모멘텀) 앙상블 L∈{42,84,180,360,540}봉(7~90일) + 호흡 앙상블 k∈{3,4,5,6}×h∈{12,24,48} 추종."""
import json, math
import numpy as np, pandas as pd
import data, indicators as ta, novel as N, nature as NT
from nature_all import OLD, NEW

LS = (42, 84, 180, 360, 540)
KS = (3, 4, 5, 6)
HS = (12, 24, 48)


def load_all():
    hb = ta.resample_ohlcv(data.load_history("BTCUSDT", "1H", 1825, refresh=False), "4h")
    he = ta.resample_ohlcv(data.load_history("ETHUSDT", "1H", 1825, refresh=False), "4h")
    coins = {"BTCUSDT": hb, "ETHUSDT": he}
    for s in OLD + NEW:
        df = pd.read_csv(data.CACHE / f"{s}_4H.csv", index_col=0, parse_dates=True, encoding="utf-8"); df.index = pd.to_datetime(df.index, utc=True)
        if len(df) >= 3000: coins[s] = df
    return coins, hb.index


def tsmom_dir(r, Ls=LS):
    cs = np.cumsum(r); outs = []
    for L in Ls:
        m = np.full(len(r), np.nan); m[L:] = np.sign(cs[L:] - cs[:-L]); outs.append(np.nan_to_num(m))
    return np.mean(outs, axis=0)


def breath_dir(r, ks=KS, hs=HS):
    sg = np.sign(r); n = len(r); streak = np.zeros(n)
    for t in range(1, n):
        streak[t] = streak[t - 1] + 1 if (sg[t] != 0 and sg[t] == sg[t - 1] and streak[t - 1] > 0) else (1 if sg[t] != 0 else 0)
    outs = []
    for k in ks:
        d = np.where(streak >= k, sg, 0.0)
        for h in hs: outs.append(NT.hold_machine(d, h))
    return np.mean(outs, axis=0)


def coin_streams(coins, cost):
    N.COST = cost
    S = {"TSMOM": {}, "호흡": {}, "합성": {}}; DIR = {"TSMOM": {}, "호흡": {}}
    for s, df in coins.items():
        c = df["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))]); sg = N.sigma(r)
        cap = np.minimum(N.TV / sg, N.LMAX)
        d1, d2 = tsmom_dir(r), breath_dir(r)
        DIR["TSMOM"][s], DIR["호흡"][s] = d1, d2
        for nm, d in (("TSMOM", d1), ("호흡", d2), ("합성", 0.5 * d1 + 0.5 * d2)):
            pos = N.band(d * cap, 0.15)
            S[nm][s] = pd.Series(N.pnl_from_pos(pos, r), index=df.index)
    return S, DIR


def port(streams):
    return pd.concat(streams, axis=1).mean(axis=1, skipna=True).fillna(0.0)


def summ(p, lab, mult=1.0):
    x = p.to_numpy() * mult; st = N.stats(x)
    return f"{lab:24s} 샤프 {st['sharpe']:+.2f} | 연 {st['ann']:+.0%} | 5년 {st['total']:+.0%} | 최대낙폭 {st['mdd']:.0%}", st


def main():
    coins, idx = load_all(); out = {}
    print(f"코인 {len(coins)}개, 4H 5년 · 모든 설정 미리 고정 · 선택/튜닝 없음\n")
    for cost in (0.0008, 0.0015):
        S, DIR = coin_streams(coins, cost)
        print(f"════ 비용 편도 {cost:.2%} ════")
        P = {k: port(v) for k, v in S.items()}
        for k, p in P.items():
            line, st = summ(p, f"{k} 앙상블 (24코인 동일위험)")
            print(line); out.setdefault(f"{cost}", {})[k] = st
        # 연도별
        yrs = sorted(set(P["합성"].index.year))
        print("  연도별 샤프(합성):", "  ".join(f"{y}:{(lambda x: x.mean()/x.std()*math.sqrt(N.BARS_Y) if x.std()>0 else 0)(P['합성'][P['합성'].index.year==y]):+.2f}" for y in yrs))
        print("  연도별 샤프(TSMOM):", "  ".join(f"{y}:{(lambda x: x.mean()/x.std()*math.sqrt(N.BARS_Y) if x.std()>0 else 0)(P['TSMOM'][P['TSMOM'].index.year==y]):+.2f}" for y in yrs))
        print("  연도별 샤프(호흡):", "  ".join(f"{y}:{(lambda x: x.mean()/x.std()*math.sqrt(N.BARS_Y) if x.std()>0 else 0)(P['호흡'][P['호흡'].index.year==y]):+.2f}" for y in yrs))
        h = len(P["합성"]) // 2
        for k in P:
            a, b = N.stats(P[k].to_numpy()[:h]), N.stats(P[k].to_numpy()[h:])
            print(f"  {k}: 전반 샤프 {a['sharpe']:+.2f} / 후반 {b['sharpe']:+.2f}")
        pos_coins = {k: sum(1 for s in S[k] if N.stats(S[k][s].to_numpy()) and N.stats(S[k][s].to_numpy())["sharpe"] > 0) for k in S}
        print("  코인별 양수 개수:", pos_coins, f"/ {len(coins)}\n")
        if cost == 0.0008:
            keep_S, keep_P = S, P
    # 벤치마크
    bh = port({s: pd.Series(np.diff(np.log(df["close"].to_numpy()), prepend=np.log(df["close"].iloc[0])), index=df.index) for s, df in coins.items()})
    print(summ(bh, "참고: 24코인 단순 보유")[0])
    btc = pd.Series(np.diff(np.log(coins["BTCUSDT"]["close"].to_numpy()), prepend=np.log(coins["BTCUSDT"]["close"].iloc[0])), index=coins["BTCUSDT"].index)
    print(summ(btc, "참고: BTC 단순 보유")[0], "\n")

    # 고원: 호흡 설정 격자 / TSMOM 설정 격자 각각 포트폴리오 샤프 (비용 0.08%)
    N.COST = 0.0008
    print("══ 설정 고원 확인 (24코인 포트폴리오 샤프, 비용 0.08%) — 어느 한 점이 아니라 전체가 양수인가 ══")
    cache = {}
    for s, df in coins.items():
        c = df["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))]); cache[s] = (r, np.minimum(N.TV / N.sigma(r), N.LMAX), df.index)
    print("TSMOM 단일 L(봉):", end=" ")
    for L in (42, 84, 180, 360, 540, 1080):
        p = port({s: pd.Series(N.pnl_from_pos(N.band(tsmom_dir(r, (L,)) * cap, 0.15), r), index=ix) for s, (r, cap, ix) in cache.items()})
        print(f"L={L}: {N.stats(p.to_numpy())['sharpe']:+.2f}", end="  ")
    print("\n호흡 단일 (k,h):")
    for k in (3, 4, 5, 6):
        row = []
        for h in (6, 12, 24, 48, 96):
            p = port({s: pd.Series(N.pnl_from_pos(N.band(breath_dir(r, (k,), (h,)) * cap, 0.15), r), index=ix) for s, (r, cap, ix) in cache.items()})
            row.append(f"h={h}: {N.stats(p.to_numpy())['sharpe']:+.2f}")
        print(f"  k={k}  " + "  ".join(row))

    # 배율 (합성, 비용 0.08%)
    print("\n══ 크기(배율) 결정 — 합성 포트폴리오, 비용 0.08% ══")
    x = keep_P["합성"].to_numpy()
    st0 = N.stats(x)
    print(f"기본(코인당 연변동성 40% 타깃) 포트폴리오: 연 {st0['ann']:+.0%}, 변동성 {x.std()*math.sqrt(N.BARS_Y):.0%}, 낙폭 {st0['mdd']:.0%}")
    rng = np.random.default_rng(5); BL = 180; T = 6 * 365 * 3
    res = {}
    print(f"{'배율':>5s} {'5년 실제수익':>11s} {'5년 낙폭':>8s} {'월별 최악':>8s} | 3년 부트스트랩(엣지 100% / 50%) 중앙값 · 낙폭≤-50% 확률")
    for m in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        y = x * m; eq = np.cumprod(1 + y); mdd = (eq / np.maximum.accumulate(eq) - 1).min()
        mon = pd.Series(y, index=keep_P["합성"].index).groupby(keep_P["合成".replace("合成", "합성")].index.to_period("M")).apply(lambda z: (1 + z).prod() - 1)
        row = []
        for k in (1.0, 0.5):
            xx = x - (1 - k) * x.mean(); nb = math.ceil(T / BL)
            st_ = rng.integers(0, len(xx) - BL, size=(2000, nb)); ix = (st_[:, :, None] + np.arange(BL)[None, None, :]).reshape(2000, -1)[:, :T]
            paths = np.cumprod(1 + m * xx[ix], axis=1); fin = paths[:, -1]; dd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
            row.append((float(np.median(fin)), float((dd <= -0.5).mean()), float(np.mean(np.log(np.maximum(fin, 1e-9))))))
        res[m] = {"실제": float(eq[-1] - 1), "mdd": float(mdd), "월최악": float(mon.min()), "부트": row}
        print(f"{m:5.1f} {eq[-1]-1:+11.0%} {mdd:8.0%} {mon.min():8.0%} | {row[0][0]:.2f}배·{row[0][1]:.0%}  /  {row[1][0]:.2f}배·{row[1][1]:.0%}  (평균 로그성장 {row[0][2]:+.2f} / {row[1][2]:+.2f})")
    out["배율"] = res
    with open("state/robust_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    pd.DataFrame({k: v for k, v in keep_P.items()}).to_csv("state/robust_pnl.csv", encoding="utf-8")


if __name__ == "__main__":
    main()
