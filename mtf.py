"""주봉·일봉·4시간봉 흐름 — 정렬 진단 + 다중 시간대 매매법 5종.  python mtf.py
시간대별 방향(같은 원리, 미리 고정):
  4H = 현재 앱(추세 7~90일 + 4H 호흡)
  일봉 = 추세(5·10·20·40·60일 부호 평균) + 일봉 호흡(같은 방향 2~4일 연속 → 3·5·10일 유지)
  주봉 = 추세(4·8·13·26주 부호 평균) + 주봉 호흡(같은 방향 2~3주 연속 → 2·4·6주 유지)
  일봉·주봉 값은 그 봉이 '마감된 다음 4H 봉'부터 사용(미래참조 없음).
매매법(미리 고정): M1 보강 간섭 (4H+일+주)/3 · M2 큰 물줄기(주봉과 같은 방향만) · M3 정렬도 크기(상위 동의 0/1/2개 → 0.5/0.75/1.0배)
  M4 프랙탈 호흡(세 시간대 모두 같은 방향 연속봉이면 1.5배) · M5 역류 확인(주봉 반대 4H 전환은 일봉 동의할 때만)."""
import json, math, warnings
import numpy as np, pandas as pd
import data, indicators as ta, novel as N, precision as PR, robust as RB, fakes as FK

warnings.filterwarnings("ignore")
COST = 0.0003


def tf_signal(bars, Ls, ks, hs):
    c = bars["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))])
    d = 0.5 * RB.tsmom_dir(r, Ls) + 0.5 * RB.breath_dir(r, ks, hs)
    sg = np.sign(r); streak = np.zeros(len(r))
    for t in range(1, len(r)): streak[t] = streak[t - 1] + 1 if (sg[t] != 0 and sg[t] == sg[t - 1]) else 1
    return pd.Series(d, index=bars.index), pd.Series(streak * sg, index=bars.index)


def to_4h(s, idx4, shift):
    return pd.Series(s.to_numpy(), index=s.index + shift).reindex(idx4, method="ffill").fillna(0).to_numpy()


def build():
    D, FR = FK.load()
    for s in FR:
        df = D[s]; idx = FR[s]["idx"]
        dd = ta.resample_ohlcv(df, "1D"); ww = ta.resample_ohlcv(df, "W-MON")
        dD, sD = tf_signal(dd, (5, 10, 20, 40, 60), (2, 3, 4), (3, 5, 10))
        dW, sW = tf_signal(ww, (4, 8, 13, 26), (2, 3), (2, 4, 6))
        FR[s]["dD"] = to_4h(dD, idx, pd.Timedelta(days=1)); FR[s]["sD"] = to_4h(sD, idx, pd.Timedelta(days=1))
        FR[s]["dW"] = to_4h(dW, idx, pd.Timedelta(days=7)); FR[s]["sW"] = to_4h(sW, idx, pd.Timedelta(days=7))
        sg = np.sign(FR[s]["r"]); st = np.zeros(len(sg))
        for t in range(1, len(sg)): st[t] = st[t - 1] + 1 if (sg[t] != 0 and sg[t] == sg[t - 1]) else 1
        FR[s]["s4"] = st * sg
    return D, FR


def main():
    D, FR = build(); out = {}
    base = FK.judge(FR, {s: F["base"] for s, F in FR.items()}); out["base"] = base
    # ── 진단: 정렬 상태별 4H 포지션 성과 / 24시간·1주 방향 적중 / 가짜 전환 비율
    print("══ 1) 정렬 상태별 — 현재 앱 포지션의 성과 (앞 | 뒤) ══")
    stats = {}
    for s, F in FR.items():
        x = PR.pnl(F["base"], F, COST); n = len(x); p = np.concatenate([[0.0], F["base"][:-1]]); sgp = np.sign(p)
        agD = np.concatenate([[0], (np.sign(F["dD"]) == np.sign(F["base"]))[:-1]]); agW = np.concatenate([[0], (np.sign(F["dW"]) == np.sign(F["base"]))[:-1]])
        state = np.where(sgp == 0, -1, agD.astype(int) + agW.astype(int))
        for lab, m in (("세 시간대 모두 같은 방향", state == 2), ("하나만 같은 방향", state == 1), ("일봉·주봉 모두 반대", state == 0)):
            for half, sl in (("앞", slice(0, n // 2)), ("뒤", slice(n // 2, n))):
                stats.setdefault((lab, half), []).append(x[sl][m[sl]])
    for lab in ("세 시간대 모두 같은 방향", "하나만 같은 방향", "일봉·주봉 모두 반대"):
        a, b = np.concatenate(stats[(lab, "앞")]), np.concatenate(stats[(lab, "뒤")])
        share = (len(a) + len(b)) / sum(len(np.concatenate(stats[(k, h)])) for k in ("세 시간대 모두 같은 방향", "하나만 같은 방향", "일봉·주봉 모두 반대") for h in ("앞", "뒤"))
        print(f"  {lab:20s} 보유 시간 {share:.0%} · 샤프 앞 {PR.sh(a):+.2f} | 뒤 {PR.sh(b):+.2f} · 4시간당 평균 {a.mean()*1e4:+.1f}bp | {b.mean()*1e4:+.1f}bp")
        out[f"diag|{lab}"] = {"share": share, "sh_a": PR.sh(a), "sh_b": PR.sh(b)}
    # 방향 적중 (24시간·1주): 정렬 vs 비정렬
    for H, hl in ((6, "24시간"), (42, "1주")):
        res = {}
        for lab in ("정렬(3개 같음)", "비정렬"):
            hits = []; neff = 0
            for s, F in FR.items():
                fwd = pd.Series(F["r"]).rolling(H).sum().shift(-H).to_numpy(); d4 = np.sign(F["d"])
                al = (np.sign(F["dD"]) == d4) & (np.sign(F["dW"]) == d4)
                m = (np.abs(F["d"]) >= 0.15) & np.isfinite(fwd) & (al if lab.startswith("정렬") else ~al) & (np.arange(len(fwd)) > 600)
                hits.append(np.sign(fwd[m]) == d4[m]); neff += m.sum() / H
            hh = np.concatenate(hits); res[lab] = (hh.mean(), 2 * math.sqrt(hh.mean() * (1 - hh.mean()) / neff))
        print(f"  {hl} 방향 적중: 정렬 {res['정렬(3개 같음)'][0]:.1%}±{res['정렬(3개 같음)'][1]:.1%} vs 비정렬 {res['비정렬'][0]:.1%}±{res['비정렬'][1]:.1%}")
        out[f"hit|{hl}"] = {k: v[0] for k, v in res.items()}
    # 가짜 전환 비율: 주봉과 같은 방향 전환 vs 반대 방향 전환
    rows = []
    for s, F in FR.items():
        x = PR.pnl(F["base"], F, COST); n = len(x)
        for d, a, b in FK.segments(F["base"]):
            if a < 600 or b >= n: continue
            ret = float(np.prod(1 + x[a + 1:b + 1]) - 1); rows.append((np.sign(F["dW"][a]) == d, np.sign(F["dD"][a]) == d, ret))
    R = pd.DataFrame(rows, columns=["withW", "withD", "ret"])
    for lab, m in (("주봉과 같은 방향 전환", R["withW"]), ("주봉 반대 방향 전환", ~R["withW"]), ("일봉·주봉 모두 같은 방향", R["withW"] & R["withD"]), ("일봉·주봉 모두 반대", ~R["withW"] & ~R["withD"])):
        g = R[m]; print(f"  {lab:18s} {len(g):4d}번 · 진짜 {(g['ret'] > 0).mean():.0%} · 평균 {g['ret'].mean():+.2%} · 합계 {g['ret'].sum():+.0%}")
        out[f"flip|{lab}"] = {"n": len(g), "real": float((g["ret"] > 0).mean()), "avg": float(g["ret"].mean())}
    # ── 매매법
    print("\n══ 2) 다중 시간대 매매법 (엄격 판정) ══"); FK.show("기준 (현재 앱)", base)
    M = {}
    M["M1 보강 간섭 (4H+일+주 평균)"] = {s: N.band((F["d"] + F["dD"] + F["dW"]) / 3 * F["cap"], 0.15) for s, F in FR.items()}
    M["M2 큰 물줄기 (주봉과 같은 방향만)"] = {s: np.where(np.sign(F["base"]) == np.sign(F["dW"]), F["base"], 0.0) for s, F in FR.items()}
    def m3(F):
        ag = (np.sign(F["dD"]) == np.sign(F["base"])).astype(int) + (np.sign(F["dW"]) == np.sign(F["base"])).astype(int)
        return F["base"] * np.array([0.5, 0.75, 1.0])[ag]
    M["M3 정렬도 크기 (0.5/0.75/1.0)"] = {s: m3(F) for s, F in FR.items()}
    M["M4 프랙탈 호흡 (세 시간대 연속봉 일치 1.5배)"] = {s: F["base"] * np.where((np.sign(F["s4"]) == np.sign(F["base"])) & (np.sign(F["sD"]) == np.sign(F["base"])) & (np.sign(F["sW"]) == np.sign(F["base"])) & (np.abs(F["s4"]) >= 2) & (np.abs(F["sD"]) >= 2) & (np.abs(F["sW"]) >= 2), 1.5, 1.0) for s, F in FR.items()}
    M["M5 역류 확인 (주봉 반대 전환은 일봉 동의 시)"] = {s: FK.confirm_filter(F["base"], (lambda F_: (lambda t, d: np.sign(F_["dW"][t]) == d or np.sign(F_["dD"][t]) == d))(F)) for s, F in FR.items()}
    for nm, P in M.items():
        r = FK.judge(FR, P, base); out[nm] = r; FK.show(nm, r)
    # 단독: 일봉·주봉 방향만으로
    for nm, key in (("참고: 일봉 흐름 단독", "dD"), ("참고: 주봉 흐름 단독", "dW")):
        r = FK.judge(FR, {s: N.band(F[key] * F["cap"], 0.15) for s, F in FR.items()}); out[nm] = r; FK.show(nm, r)
    now = {s: {"4H": float(F["d"][-1]), "D": float(F["dD"][-1]), "W": float(F["dW"][-1])} for s, F in FR.items()}
    print("\n지금:", {s[:3]: {k: round(v, 2) for k, v in d.items()} for s, d in now.items()}); out["now"] = now
    with open("state/mtf_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
