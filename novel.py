"""새 관점의 분석·매매 틀 (지표를 쓰지 않고 수익률의 '구조'만 본다).  python novel.py

철학: 차트는 가격이 아니라 '기억'과 '질서'의 흐름이다.
 1) 기억력  — 분산비(VR): 과거 변화가 미래 변화에 남기는 흔적. VR>1 지속(추세), VR<1 복원(되돌림), ≈1 무기억(랜덤워크)
 2) 질서도  — 순열 엔트로피(PE): 최근 움직임의 순서 패턴이 얼마나 치우쳤는가
 3) 내재시간 — 방향전환(DC): 시계 대신 '일정 비율 변화'를 시간 단위로 삼는 지그재그 추종
 4) 적응    — Hedge(지수가중) 앙상블: 여러 시간척도의 추세/반전 전문가 중 최근 잘 맞은 쪽에 가중, 전부 틀리면 현금
 5) 크기    — 변동성 타깃 + 켈리식 크기 (엣지가 약할수록 작게)
모든 포지션은 봉 마감 정보로 정하고 다음 봉 수익을 얻는다(미래참조 없음). 비용: 포지션 변화량 × 편도 0.08%."""
from __future__ import annotations
import json, math
import numpy as np, pandas as pd
import data, indicators as ta

BARS_Y = 6 * 365
COST = 0.0008
TV = 0.40 / math.sqrt(BARS_Y)          # 4H 봉당 목표 변동성 (연 40%)
LMAX = 3.0
NS = (3, 6, 12, 24, 48, 96, 192)       # 전문가 시간척도 (4H 봉: 12시간 ~ 32일)


def load_returns():
    out = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        h = ta.resample_ohlcv(data.load_history(s, "1H", 1825, refresh=False), "4h")
        out[s] = pd.Series(np.log(h["close"]).diff().to_numpy(), index=h.index).iloc[1:]
    return out


def sigma(r: np.ndarray, alpha=0.03):
    return np.sqrt(pd.Series(r ** 2).ewm(alpha=alpha, adjust=False).mean().to_numpy())


def pnl_from_pos(pos: np.ndarray, r: np.ndarray) -> np.ndarray:
    """pos[t]는 t 마감 때 정한 포지션 → r[t+1]을 번다."""
    p = np.nan_to_num(pos)
    gross = np.zeros_like(r); gross[1:] = p[:-1] * r[1:]
    turn = np.abs(np.diff(p, prepend=0.0)); tc = np.zeros_like(r); tc[1:] = COST * turn[:-1]
    return gross - tc


def band(pos: np.ndarray, width=0.25):
    """작은 변화는 무시 (회전율 억제)."""
    out = np.zeros_like(pos); cur = 0.0
    for i, p in enumerate(pos):
        if not np.isfinite(p): out[i] = cur; continue
        if abs(p - cur) > width: cur = p
        out[i] = cur
    return out


def mom_sign(r: np.ndarray, n: int):
    cs = np.cumsum(r); s = np.full_like(r, np.nan); s[n:] = cs[n:] - cs[:-n]
    return np.sign(s)


# ───────── 1) 기억력: 분산비 ─────────
def variance_ratio(r: np.ndarray, q=6, W=180):
    n = len(r); vr = np.full(n, np.nan); z = np.full(n, np.nan)
    cs = np.concatenate([[0.0], np.cumsum(r)])
    rq = cs[q:] - cs[:-q]                                  # 겹치는 q봉 합
    for t in range(W, n):
        a = r[t - W + 1:t + 1]; b = rq[t - W + 1 - q + 1 + 0:t - q + 2] if False else None
        v1 = a.var(ddof=1)
        seg = cs[t + 1] - cs[t + 1 - W:t + 1 - q + 1]       # 길이 W-q+1 인 q봉 합
        seg = cs[t - W + q + 1:t + 2] - cs[t - W + 1:t - q + 2]
        vq = seg.var(ddof=1)
        vr[t] = vq / (q * v1) if v1 > 0 else np.nan
        se = math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * W))
        z[t] = (vr[t] - 1) / se
    return vr, z


def strat_vr(r, q=6, W=180, mom_n=18, zth=1.0):
    vr, z = variance_ratio(r, q, W)
    reg = np.where(z > zth, 1.0, np.where(z < -zth, -1.0, 0.0))
    d = mom_sign(r, mom_n)
    sg = sigma(r)
    return band(np.nan_to_num(reg * d * np.minimum(TV / sg, LMAX))), vr, z


# ───────── 2) 질서도: 순열 엔트로피 ─────────
def perm_entropy(c: np.ndarray, m=3, W=120):
    n = len(c); pat = np.full(n, -1)
    for t in range(m - 1, n):
        w = c[t - m + 1:t + 1]
        pat[t] = int("".join(map(str, np.argsort(np.argsort(w)))), m)
    pe = np.full(n, np.nan); L = math.log(math.factorial(m))
    for t in range(W + m, n):
        cnt = np.bincount(pat[t - W + 1:t + 1], minlength=m ** m + 1)
        p = cnt[cnt > 0] / W
        pe[t] = -(p * np.log(p)).sum() / L
    return pe


def strat_pe(r, mom_n=18, lowq=0.25, mode="follow"):
    c = np.cumsum(r); pe = perm_entropy(c)
    s = pd.Series(pe)
    lo = s.rolling(1000, min_periods=300).quantile(lowq).to_numpy()
    hi = s.rolling(1000, min_periods=300).quantile(1 - lowq).to_numpy()
    d = mom_sign(r, mom_n); sg = sigma(r)
    if mode == "follow":      # 질서 높음(엔트로피 낮음) → 추종, 그 외 현금
        reg = np.where(pe < lo, 1.0, 0.0)
    else:                     # 무질서(엔트로피 높음) → 되돌림
        reg = np.where(pe > hi, -1.0, 0.0)
    return band(np.nan_to_num(reg * d * np.minimum(TV / sg, LMAX))), pe


# ───────── 3) 내재시간: 방향전환(DC) 지그재그 ─────────
def strat_dc(c: np.ndarray, delta: float, sgn=1.0):
    """고점에서 delta 하락 확인 → 숏 전환, 저점에서 delta 반등 확인 → 롱 전환 (sgn=-1이면 반대=역추세)."""
    n = len(c); pos = np.zeros(n); mode = 0; ext = c[0]
    sg = sigma(np.diff(np.log(c), prepend=np.log(c[0])))
    for t in range(1, n):
        if mode >= 0:
            if c[t] > ext: ext = c[t]
            if c[t] <= ext * (1 - delta): mode, ext = -1, c[t]
        if mode <= 0:
            if c[t] < ext: ext = c[t]
            if c[t] >= ext * (1 + delta): mode, ext = 1, c[t]
        pos[t] = sgn * mode * min(TV / sg[t], LMAX)
    return band(pos)


# ───────── 4) 적응 앙상블 (Hedge) ─────────
def strat_hedge(r, eta=0.15, lam=0.99, ns=NS, hmax=None):
    n = len(r); sg = sigma(r); K = len(ns)
    M = np.vstack([mom_sign(r, k) for k in ns])           # 추세 전문가
    M = np.vstack([M, -M]); M = np.nan_to_num(M)          # + 반전 전문가
    S = np.zeros(2 * K + 1)                                # 마지막 = 현금
    sig = np.zeros(n); W = np.zeros((n, 2 * K + 1))
    for t in range(n - 1):
        e = np.exp(eta * (S - S.max())); w = e / e.sum()
        W[t] = w; sig[t] = (w[:-1] * M[:, t]).sum()
        g = np.append(M[:, t] * r[t + 1] / max(sg[t], 1e-9), 0.0)
        S = lam * S + g
    pos = sig * np.minimum(TV / sg, LMAX)
    return band(pos), sig, W


# ───────── 5) 순서 패턴 온라인 확률 예측 ─────────
def strat_ordinal(r, m=4, thr=0.04, nmin=40):
    c = np.cumsum(r); n = len(r)
    pat = np.full(n, -1)
    for t in range(m, n):
        w = c[t - m:t + 1]; code = 0
        for x in np.argsort(np.argsort(w)): code = code * (m + 1) + int(x)
        pat[t] = code
    up, tot = {}, {}; pos = np.zeros(n); sg = sigma(r); hits = [];
    for t in range(m, n - 1):
        p = pat[t - 1]                                      # t-1의 패턴 다음에 r[t]가 나옴 → 지금 학습
        if t >= m + 1:
            tot[p] = tot.get(p, 0) + 1; up[p] = up.get(p, 0) + (1 if r[t] > 0 else 0)
        q = pat[t]; nn = tot.get(q, 0)
        if nn >= nmin:
            ph = (up.get(q, 0) + 1) / (nn + 2)
            if abs(ph - 0.5) > thr:
                pos[t] = np.sign(ph - 0.5) * min(TV / sg[t], LMAX)
                hits.append((t, np.sign(ph - 0.5) == np.sign(r[t + 1])))
    return band(pos), hits


# ───────── 평가 ─────────
def stats(pnl: np.ndarray, pos: np.ndarray | None = None):
    x = pnl[np.isfinite(pnl)]
    if len(x) < 100 or x.std() == 0: return None
    eq = np.cumprod(1 + x)
    out = {"sharpe": float(x.mean() / x.std() * math.sqrt(BARS_Y)), "ann": float(x.mean() * BARS_Y),
           "total": float(eq[-1] - 1), "mdd": float((eq / np.maximum.accumulate(eq) - 1).min())}
    if pos is not None:
        out["turn_yr"] = float(np.abs(np.diff(np.nan_to_num(pos), prepend=0)).mean() * BARS_Y)
    return out


def evaluate(name, fn, R, split, keep=None):
    """fn(r)->pos. 두 코인 동일비중."""
    pn, ps = {}, {}
    for s, r in R.items():
        pos = fn(r.to_numpy()); pn[s] = pd.Series(pnl_from_pos(pos, r.to_numpy()), index=r.index); ps[s] = pd.Series(np.nan_to_num(pos), index=r.index)
    P = pd.concat(pn, axis=1).mean(axis=1); Q = pd.concat(ps, axis=1).mean(axis=1)
    res = {"전체": stats(P.to_numpy(), Q.to_numpy()), "앞": stats(P[P.index < split].to_numpy()), "뒤": stats(P[P.index >= split].to_numpy())}
    for s in R: res[s] = stats(pn[s].to_numpy())
    return res, P


def show(name, res):
    a, b, c = res["전체"], res["앞"], res["뒤"]
    if not (a and b and c): print(f"{name:34s} (계산 불가)"); return
    print(f"{name:34s} 샤프 {a['sharpe']:+.2f} (앞 {b['sharpe']:+.2f} / 뒤 {c['sharpe']:+.2f}) | 연 {a['ann']:+.0%} 5년 {a['total']:+.0%} 낙폭 {a['mdd']:.0%} | 회전 연 {a.get('turn_yr', 0):.0f}배 | BTC {(res['BTCUSDT'] or {}).get('sharpe', float('nan')):+.2f} ETH {(res['ETHUSDT'] or {}).get('sharpe', float('nan')):+.2f}")


def main():
    R = load_returns(); idx = R["BTCUSDT"].index; split = idx[int(len(idx) * 0.6)]
    print("분할:", split.date(), "| 비용 편도 0.08% · 목표변동성 연 40% · 최대 3배 · BTC+ETH 동일비중\n")
    out = {}; curves = {}
    bh = lambda r: np.ones_like(r) * 1.0
    res, curves["보유"] = evaluate("BTC·ETH 단순 보유", bh, R, split); show("기준: 단순 보유(1배)", res); out["보유"] = res
    print("\n[1] 기억력(VR) — 지속이면 추종, 복원이면 역추세, 무기억이면 현금")
    for q, W in ((6, 180), (12, 360), (4, 120)):
        res, curves[f"VR{q}/{W}"] = evaluate("", lambda r: strat_vr(r, q, W)[0], R, split); show(f"VR(q={q}, 창={W})", res); out[f"VR_{q}_{W}"] = res
    print("\n[2] 질서도(PE) — 질서 높을 때만 추종 / 무질서일 때 역추세")
    for mode in ("follow", "revert"):
        res, curves[f"PE{mode}"] = evaluate("", lambda r: strat_pe(r, 18, 0.25, mode)[0], R, split); show(f"엔트로피 {mode}", res); out[f"PE_{mode}"] = res
    print("\n[3] 내재시간(DC) — 고점·저점에서 일정 비율 반전 시 방향 전환")
    for d in (0.03, 0.05, 0.08):
        for sgn, lab in ((1.0, "추종"), (-1.0, "역추세")):
            res, curves[f"DC{d}{lab}"] = evaluate("", lambda r, d=d, sgn=sgn: strat_dc(np.exp(np.cumsum(r.astype(float))), d, sgn), R, split)
            show(f"DC δ={d:.0%} {lab}", res); out[f"DC_{d}_{lab}"] = res
    print("\n[4] 적응 앙상블(Hedge) — 14개 전문가 + 현금, 최근 성과로 가중")
    for eta, lam in ((0.15, 0.99), (0.30, 0.99), (0.15, 0.995), (0.05, 0.99)):
        res, curves[f"H{eta}/{lam}"] = evaluate("", lambda r, e=eta, l=lam: strat_hedge(r, e, l)[0], R, split)
        show(f"Hedge η={eta} λ={lam}", res); out[f"Hedge_{eta}_{lam}"] = res
    print("\n[5] 순서 패턴 온라인 확률 예측")
    res, curves["ORD"] = evaluate("", lambda r: strat_ordinal(r)[0], R, split); show("순서패턴(m=4)", res); out["ORD"] = res
    for s, r in R.items():
        hits = strat_ordinal(r.to_numpy())[1]
        if hits:
            acc = float(np.mean([h for _, h in hits])); n = len(hits)
            print(f"   {s} 4시간 방향 적중 {acc:.1%} (n={n}, 유의 기준 ±{1.96*0.5/math.sqrt(n):.1%})")
    with open("state/novel_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    pd.DataFrame(curves).to_csv("state/novel_pnl.csv", encoding="utf-8")


if __name__ == "__main__":
    main()
