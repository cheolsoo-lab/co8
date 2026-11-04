"""상황에 따라 바꿔 타는 분석법 — 방법 대회 + 상황별 성적표 + 워크포워드 전환.  python adaptive.py
방법(모두 같은 크기 규칙: 방향 × 연변동성 40% 기준 × 완충 0.15, 지정가 0.03%, 롱 펀딩 연 10%):
  기존 7: 이평선 교차(EMA20/100), 돈치안 돌파(20 진입/10 청산), MACD, RSI 역추세(30/70→50), 볼린저 역추세(밴드 밖→중심), 일목 구름, 슈퍼트렌드(10,3)
  자체 4: 추세(TSMOM 7~90일), 호흡, 현재 앱(추세+호흡), 베어트랩
  새 관점 4: 기억 전환(최근 자기상관 +면 추종 / −면 역행), 깨끗한 길(효율비 상위 30%일 때만 그 방향),
            가속(EMA20 기울기와 그 변화가 같은 방향일 때만), 합의의 폭(1·3·7·14·30·60일 방향 6개 중 5개 이상 일치할 때만)
상황 6가지(그 시점까지 정보만): 30일 수익/변동성 z > 0.5 상승, < −0.5 하락, 그 사이 횡보 × 변동성이 지난 1년 중 위/아래 절반
전환 메타(매달 재선택, 과거만): 상황별 1등 / 상황별 상위 3 평균 / 상황별 샤프 가중."""
import json, math, warnings
import numpy as np, pandas as pd
import data, indicators as ta, novel as N, precision as PR, beartrap as BT

warnings.filterwarnings("ignore")
COST = 0.0003
REG = ["상승·변동성↓", "상승·변동성↑", "횡보·변동성↓", "횡보·변동성↑", "하락·변동성↓", "하락·변동성↑"]


def state_machine(enter_long, exit_long, enter_short, exit_short):
    n = len(enter_long); pos = np.zeros(n); s = 0
    for t in range(n):
        if s == 1 and exit_long[t]: s = 0
        if s == -1 and exit_short[t]: s = 0
        if s == 0:
            if enter_long[t]: s = 1
            elif enter_short[t]: s = -1
        pos[t] = s
    return pos


def supertrend(df, n=10, m=3.0):
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close")); atr = ta.atr(df, n).to_numpy(); mid = (h + l) / 2
    ub, lb = mid + m * atr, mid - m * atr; fu, fl = ub.copy(), lb.copy(); d = np.ones(len(c))
    for t in range(1, len(c)):
        fu[t] = ub[t] if (ub[t] < fu[t - 1] or c[t - 1] > fu[t - 1]) else fu[t - 1]
        fl[t] = lb[t] if (lb[t] > fl[t - 1] or c[t - 1] < fl[t - 1]) else fl[t - 1]
        d[t] = 1 if c[t] > fu[t - 1] else (-1 if c[t] < fl[t - 1] else d[t - 1])
    return d


def methods(df, F):
    c = df["close"]; cv = c.to_numpy(float); r = F["r"]; M = {}
    M["이평선 교차"] = np.sign((ta.ema(c, 20) - ta.ema(c, 100)).to_numpy())
    hh, ll = df["high"].rolling(20).max().shift(1).to_numpy(), df["low"].rolling(20).min().shift(1).to_numpy()
    xh, xl = df["high"].rolling(10).max().shift(1).to_numpy(), df["low"].rolling(10).min().shift(1).to_numpy()
    M["돈치안 돌파"] = state_machine(cv > hh, cv < xl, cv < ll, cv > xh)
    M["MACD"] = np.sign(ta.macd_hist(c).to_numpy())
    rs = ta.rsi(c).to_numpy(); M["RSI 역추세"] = state_machine(rs < 30, rs > 50, rs > 70, rs < 50)
    pb, _ = ta.bollinger(c); pb = pb.to_numpy(); M["볼린저 역추세"] = state_machine(pb < 0, pb > 0.5, pb > 1, pb < 0.5)
    _, _, sa, sb = ta.ichimoku(df); top, bot = np.fmax(sa, sb).to_numpy(), np.fmin(sa, sb).to_numpy()
    M["일목 구름"] = np.where(cv > top, 1.0, np.where(cv < bot, -1.0, 0.0))
    M["슈퍼트렌드"] = supertrend(df)
    M["추세(TSMOM)"] = F["d1"]; M["호흡"] = F["d2"]; M["현재 앱(추세+호흡)"] = F["d"]
    M["베어트랩"] = BT.signal(df)[0]
    # 새 관점
    rr = pd.Series(r); ac = rr.rolling(120).corr(rr.shift(1)).to_numpy(); m6 = np.sign(rr.rolling(6).sum().to_numpy())
    M["기억 전환"] = np.where(ac > 0.05, m6, np.where(ac < -0.05, -m6, 0.0))
    net = rr.rolling(42).sum(); path = rr.abs().rolling(42).sum(); er = (net.abs() / path).to_numpy()
    erp = pd.Series(er).rolling(2190, min_periods=500).apply(lambda a: (a[-1] >= a).mean(), raw=True).to_numpy()
    M["깨끗한 길"] = np.where(erp > 0.7, np.sign(net.to_numpy()), 0.0)
    e20 = ta.ema(c, 20); sl = (e20 - e20.shift(6)).to_numpy(); acc = sl - np.concatenate([np.full(6, np.nan), sl[:-6]])
    M["가속"] = np.where(np.sign(sl) == np.sign(acc), np.sign(sl), 0.0)
    cs = np.cumsum(r); agree = np.mean([np.sign(np.concatenate([np.zeros(L), cs[L:] - cs[:-L]])) for L in (6, 18, 42, 84, 180, 360)], axis=0)
    M["합의의 폭"] = np.where(np.abs(agree) >= 4 / 6 - 1e-9, np.sign(agree), 0.0)
    return {k: N.band(np.nan_to_num(v) * F["cap"], 0.15) for k, v in M.items()}


def regimes(F):
    r = pd.Series(F["r"]); sg = N.sigma(F["r"]); z = (r.rolling(180).sum() / (sg * math.sqrt(180))).to_numpy()
    vp = pd.Series(sg).rolling(2190, min_periods=500).apply(lambda a: (a[-1] >= a).mean(), raw=True).to_numpy()
    tr = np.where(z > 0.5, 0, np.where(z < -0.5, 2, 1)); vv = np.where(vp > 0.5, 1, 0)
    reg = tr * 2 + vv; reg = np.where(np.isfinite(z) & np.isfinite(vp), reg, -1)
    return reg


def sh(x):
    return float(x.mean() / x.std() * math.sqrt(PR.B4)) if len(x) > 50 and x.std() > 0 else np.nan


def main():
    D, FR, P, X, RG = {}, {}, {}, {}, {}
    for s in ("BTCUSDT", "ETHUSDT"):
        D[s] = ta.resample_ohlcv(data.load_history(s, "1H", 1825, refresh=False), "4h").dropna(); FR[s] = PR.coin_frame(D[s])
        P[s] = methods(D[s], FR[s]); X[s] = {k: PR.pnl(v, FR[s], COST) for k, v in P[s].items()}; RG[s] = regimes(FR[s])
    names = list(P["BTCUSDT"]); n = len(FR["BTCUSDT"]["r"]); h = n // 2; idx = FR["BTCUSDT"]["idx"]
    out = {"names": names, "regimes": REG}
    # 1) 대회
    print("══ 1) 방법 대회 — BTC·ETH 반반, 5년 (지정가·펀딩) ══")
    rows = []
    for k in names:
        p = 0.5 * (X["BTCUSDT"][k] + X["ETHUSDT"][k]); e = np.cumprod(1 + p)
        rows.append((k, sh(p), sh(p[:h]), sh(p[h:]), sh(X["BTCUSDT"][k]), sh(X["ETHUSDT"][k]), float(e[-1] ** (PR.B4 / n) - 1), float((e / np.maximum.accumulate(e) - 1).min())))
    grp = lambda k: "새 관점" if k in ("기억 전환", "깨끗한 길", "가속", "합의의 폭") else ("자체" if k in ("추세(TSMOM)", "호흡", "현재 앱(추세+호흡)", "베어트랩") else "기존")
    for k, a, b, c_, bb, ee, ann, mdd in sorted(rows, key=lambda z: -z[1]):
        print(f"  {k:18s} [{grp(k)}] 샤프 {a:+.2f} (앞/뒤 {b:+.2f}/{c_:+.2f}) · BTC {bb:+.2f} · ETH {ee:+.2f} · 연 {ann:+.0%} · 낙폭 {mdd:.0%}")
    out["tournament"] = rows
    # 2) 상황별 성적표
    print("\n══ 2) 상황별 성적표 — 각 상황에 있었던 봉에서의 샤프 (앞 절반 | 뒤 절반) ══")
    cnt = {g: int(sum((RG[s] == g).sum() for s in RG)) for g in range(6)}
    print("  상황별 봉 수:", {REG[g]: cnt[g] for g in range(6)})
    tab = {}
    for k in names:
        for half, sl in (("앞", slice(0, h)), ("뒤", slice(h, n)), ("전체", slice(0, n))):
            for g in range(6):
                xs = []
                for s in RG:
                    reg = np.concatenate([[-1], RG[s][:-1]])[sl]; x = X[s][k][sl]; xs.append(x[reg == g])
                tab[(k, half, g)] = sh(np.concatenate(xs))
    hdr = "  " + f"{'방법':18s}" + "".join(f"{REG[g]:>16s}" for g in range(6))
    print(hdr)
    for k in names:
        print("  " + f"{k:18s}" + "".join(f"{tab[(k, '앞', g)]:+6.2f}|{tab[(k, '뒤', g)]:+6.2f}  " for g in range(6)))
    a = np.array([tab[(k, "앞", g)] for k in names for g in range(6)]); b = np.array([tab[(k, "뒤", g)] for k in names for g in range(6)])
    ok = np.isfinite(a) & np.isfinite(b)
    from scipy.stats import spearmanr
    rho_all = spearmanr(a[ok], b[ok]).correlation
    # 상황 안에서 방법 순위가 유지되나 (방법 간 비교만)
    within = []
    for g in range(6):
        aa = np.array([tab[(k, "앞", g)] for k in names]); bb_ = np.array([tab[(k, "뒤", g)] for k in names]); m = np.isfinite(aa) & np.isfinite(bb_)
        within.append(spearmanr(aa[m], bb_[m]).correlation)
        best = names[int(np.nanargmax(aa))]; rank_b = int((bb_[m] > tab[(best, "뒤", g)]).sum()) + 1
        print(f"  {REG[g]:12s}: 앞 기간 1등 '{best}' (앞 {tab[(best, '앞', g)]:+.2f}) → 뒤 기간 {tab[(best, '뒤', g)]:+.2f}, 순위 {rank_b}/{m.sum()} · 방법 순위 상관 {within[-1]:+.2f}")
    print(f"  전체 칸 앞/뒤 상관 {rho_all:+.2f} · 상황 안 방법 순위 상관 평균 {np.nanmean(within):+.2f}")
    out["table"] = {f"{k}|{hf}|{REG[g]}": v for (k, hf, g), v in tab.items()}; out["rho_all"] = float(rho_all); out["within"] = [float(w) for w in within]
    # 3) 워크포워드 전환
    print("\n══ 3) 상황 맞춤 전환 — 매달 그때까지의 성적만으로 상황별 방법 선택 (첫 1년은 학습) ══")
    months = pd.Series(idx).dt.to_period("M").to_numpy()
    starts = [i for i in range(1, n) if months[i] != months[i - 1] and i >= 2190]
    metas = {"상황별 1등": {}, "상황별 상위 3 평균": {}, "상황별 샤프 가중": {}}
    for s in RG:
        for m in metas: metas[m][s] = np.zeros(n)
    choice_log = []
    for j, st in enumerate(starts):
        en = starts[j + 1] if j + 1 < len(starts) else n
        # 과거 성적: 상황별·방법별 샤프 (두 코인 합침, st 이전 봉만)
        perf = np.full((len(names), 6), np.nan)
        for gi in range(6):
            for ki, k in enumerate(names):
                xs = []
                for s in RG:
                    reg = np.concatenate([[-1], RG[s][:-1]])[:st]; xs.append(X[s][k][:st][reg == gi])
                x = np.concatenate(xs); perf[ki, gi] = sh(x) if len(x) >= 300 else np.nan
        for s in RG:
            for t in range(st, en):
                g = RG[s][t]
                if g < 0 or not np.isfinite(perf[:, g]).any():
                    for m in metas: metas[m][s][t] = P[s]["현재 앱(추세+호흡)"][t]
                    continue
                col = np.nan_to_num(perf[:, g], nan=-9); order = np.argsort(-col)
                metas["상황별 1등"][s][t] = P[s][names[order[0]]][t]
                metas["상황별 상위 3 평균"][s][t] = np.mean([P[s][names[i]][t] for i in order[:3]])
                w = np.clip(col, 0, None); metas["상황별 샤프 가중"][s][t] = (np.dot(w, [P[s][k][t] for k in names]) / w.sum()) if w.sum() > 0 else 0.0
        choice_log.append((str(idx[st].date()), {REG[g]: names[int(np.nanargmax(np.nan_to_num(perf[:, g], nan=-9)))] for g in range(6)}))
    t0 = starts[0]
    def summ(xb, xe):
        p = 0.5 * (xb + xe)[t0:]; hh = len(p) // 2; e = np.cumprod(1 + p)
        return {"sh": sh(p), "h1": sh(p[:hh]), "h2": sh(p[hh:]), "BTC": sh(xb[t0:]), "ETH": sh(xe[t0:]), "ann": float(e[-1] ** (PR.B4 / len(p)) - 1), "mdd": float((e / np.maximum.accumulate(e) - 1).min())}
    base = summ(X["BTCUSDT"]["현재 앱(추세+호흡)"], X["ETHUSDT"]["현재 앱(추세+호흡)"])
    print(f"  평가 기간 {idx[t0].date()} ~ (전환은 매달 초 재선택)")
    print(f"  {'현재 앱 (같은 기간)':20s} 샤프 {base['sh']:+.2f} (앞/뒤 {base['h1']:+.2f}/{base['h2']:+.2f}) · BTC {base['BTC']:+.2f} · ETH {base['ETH']:+.2f} · 연 {base['ann']:+.0%} · 낙폭 {base['mdd']:.0%}")
    out["meta"] = {"base": base}
    for m in metas:
        xb, xe = PR.pnl(N.band(metas[m]["BTCUSDT"], 0.0), FR["BTCUSDT"], COST), PR.pnl(N.band(metas[m]["ETHUSDT"], 0.0), FR["ETHUSDT"], COST)
        r_ = summ(xb, xe)
        r_["verdict"] = "채택" if (r_["sh"] - base["sh"] >= 0.10 and r_["BTC"] >= base["BTC"] and r_["ETH"] >= base["ETH"] and r_["h1"] >= base["h1"] and r_["h2"] >= base["h2"]) else "기각"
        out["meta"][m] = r_
        print(f"  {m:20s} 샤프 {r_['sh']:+.2f} (앞/뒤 {r_['h1']:+.2f}/{r_['h2']:+.2f}) · BTC {r_['BTC']:+.2f} · ETH {r_['ETH']:+.2f} · 연 {r_['ann']:+.0%} · 낙폭 {r_['mdd']:.0%} → {r_['verdict']}")
    print("  마지막 달 상황별 선택:", choice_log[-1])
    out["choices"] = choice_log
    for s in RG: print(f"  지금 {s[:3]} 상황: {REG[RG[s][-1]] if RG[s][-1] >= 0 else '-'}")
    out["now"] = {s: int(RG[s][-1]) for s in RG}
    with open("state/adaptive_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
