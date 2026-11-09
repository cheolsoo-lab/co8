"""앱용 파동 국면: 중파동(4H ATR×4) 꼭짓점을 엘리어트 규칙으로 판정 → '2파 완성(높은 저점)'·'4파 완성(0-4 유효)' 뒤 새 파동이 진행 중이면
앱 포지션이 그 방향과 같을 때 1.5배, 반대일 때 ½ (그 꼭짓점을 종가로 다시 넘으면 해제, 다음 꼭짓점 확인 때 끝).
검증: elliott2.py · elliott_robust.py (코인베이스 11년 +0.15 · 비트겟 5년 +0.19, BTC·ETH 모두 어긋난 우주 p<0.03)."""
from __future__ import annotations
import numpy as np, pandas as pd
import indicators as ta, patterns2 as P2

K = 4.0
GOOD = {"2파 완성 (높은 저점)", "4파 완성 (0-4 유효)"}
UP, DN = 1.5, 0.5


def _imp(v, s):
    w = [s * (v[i + 1] - v[i]) for i in range(5)]
    return s * (v[2] - v[0]) > 0 and s * (v[3] - v[1]) > 0 and s * (v[4] - v[1]) > 0 and s * (v[5] - v[3]) > 0 and w[2] >= min(w[0], w[4])


def _first4(v, s):
    w = [s * (v[i + 1] - v[i]) for i in range(4)]
    return s * (v[2] - v[0]) > 0 and s * (v[3] - v[1]) > 0 and s * (v[4] - v[1]) > 0 and w[2] > w[0]


def _corr(v5, a, b, c, s):
    A = s * (v5 - a)
    if A <= 0 or s * (b - a) <= 0: return "기타 조정"
    bret = s * (b - a) / A; clen = s * (b - c) / A; beyond = s * (a - c) > 0
    if bret <= 0.618 and beyond: return "지그재그"
    if 0.8 <= bret <= 1.05 and 0.8 <= clen <= 1.2: return "정규 플랫"
    if bret > 1.05: return "확장 플랫" if beyond else "러닝 플랫"
    return "기타 조정"


def label_at(v, t, j):
    """꼭짓점 j가 확인된 순간의 국면 (elliott2.label_at과 같은 판정)."""
    imp_end = lambda i: i >= 5 and _imp(v[i - 5:i + 1], t[i])
    if imp_end(j): return "5파 완성"
    if j >= 3 and imp_end(j - 3): return "C 완성·" + _corr(v[j - 3], v[j - 2], v[j - 1], v[j], t[j - 3])
    if j >= 5 and imp_end(j - 5):
        L = [abs(v[j - 4 + i] - v[j - 5 + i]) for i in range(5)]
        if all(L[i + 1] < L[i] for i in range(4)): return "삼각형 E 완성"
    if j >= 1 and imp_end(j - 1): return "A파 완성"
    if j >= 2 and imp_end(j - 2): return "B파 완성"
    if j >= 4 and t[j - 4] == t[j] and _first4(v[j - 4:j + 1], -t[j]): return "4파 완성 (0-4 유효)"
    if j >= 3 and t[j] * (v[j - 1] - v[j - 3]) > 0 and t[j] * (v[j] - v[j - 2]) > 0: return "3파 완성 (0-3)"
    if j >= 2 and -t[j] * (v[j] - v[j - 2]) > 0: return "2파 완성 (높은 저점)"
    return "기타"


def state(df: pd.DataFrame) -> dict | None:
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close")); atr = ta.atr(df).to_numpy()
    piv = {k: P2.zigzag(h, l, atr, k) for k in (2.0, 4.0, 8.0)}
    dirs = {k: (-p[-1][3] if p else 0) for k, p in piv.items()}
    pv = piv[K]
    if len(pv) < 6: return {"dirs": dirs}
    v = [p[2] for p in pv]; t = [p[3] for p in pv]; j = len(pv) - 1
    lab = label_at(v, t, j); d = -t[j]; t0 = pv[j][0]; stop = v[j]
    broken = bool(np.any(d * (c[t0 + 1:] - stop) < 0)) if t0 + 1 < len(c) else False
    active = lab in GOOD and not broken
    return {"dirs": dirs, "label": lab, "dir": d, "stop": float(stop), "active": active, "since": str(df.index[pv[j][1]]),
            "conf": str(df.index[t0]), "broken": broken}


C53 = (0.021, 0.662)        # 사실은 3파일 확률 = 거리 비율의 로지스틱 보정 (wave53.py, 11년 442번: 시험 AUC 0.69, 다른 단서는 모두 정보 없음)


def _atr_c(x):
    r = np.diff(x, prepend=x[0]); return pd.Series(np.abs(r)).ewm(alpha=1 / 14, adjust=False).mean().to_numpy() * 1.25


def fate53(df: pd.DataFrame, k: float, recent: int = 42) -> dict | None:
    """종가 지그재그(차수 k)의 가장 최근 '5파 완성' 판별 (wave53.py · wave53_more.py, BTC·ETH 11년).
    미결: 사실은 3파일 확률(거리 비율 보정, 다른 단서는 정보 없음) / 3파 확정: 5파 끝 돌파(뒤 7일 같은 쪽 평균 +5%, 55%)
    대기: 4파 끝 이탈로 5파 끝 판정(따라가면 7일 평균 −1%) 뒤 5파 끝·시작점 사이 / 추세 재개: 판정 뒤 5파 끝 재돌파(+7%, 59%)
    추세 전환: 시작점 이탈(+6%, 62%).  3파 확정·재개·전환은 최근 recent봉 안일 때만."""
    x = np.log(df["close"].to_numpy(float)); n = len(x); pv = P2.zigzag(x, x, _atr_c(x), k); v = [p[2] for p in pv]; t = [p[3] for p in pv]
    for j in range(len(pv) - 1, 4, -1):
        if _imp(v[j - 5:j + 1], t[j]): break
    else:
        return None
    s = t[j]; t0 = pv[j][0] + 1
    first = lambda m: int(np.flatnonzero(m)[0]) if m.any() else 10 ** 9
    out = {"k": k, "s": int(s), "v5": float(np.exp(v[j])), "v4": float(np.exp(v[j - 1])), "v0": float(np.exp(v[j - 5])), "d5": str(df.index[pv[j][1]])[:10]}
    ia, ib = first(s * (x[t0:] - v[j]) > 0), first(s * (x[t0:] - v[j - 1]) < 0)
    if ia == ib == 10 ** 9:
        d5 = s * (v[j] - x[-1]); d4 = s * (x[-1] - v[j - 1]); prw = min(max(d4 / (d4 + d5), 0.02), 0.98)
        return {**out, "state": "미결", "p3": float(1 / (1 + np.exp(-(C53[0] + C53[1] * np.log(prw / (1 - prw))))))}
    if ia < ib:
        te, st = t0 + ia, "3파 확정"
    else:
        tb = t0 + ib; i5, i0 = first(s * (x[tb:] - v[j]) > 0), first(s * (x[tb:] - v[j - 5]) < 0)
        if i5 == i0 == 10 ** 9: return {**out, "state": "대기", "ago": int(n - 1 - tb)}
        te, st = (tb + i5, "추세 재개") if i5 < i0 else (tb + i0, "추세 전환")
    ago = n - 1 - te
    return {**out, "state": st, "ago": int(ago)} if ago < recent else None


def split_rows(rows: list[dict], st: dict | None) -> list[dict]:
    """전환 가격표를 파동 해제 가격에서 나눔 (각 칸 안에서 크기 배수가 같아짐)."""
    if not st or not st.get("active"): return rows
    x = st["stop"]; out = []
    for r in rows:
        lo, hi = r["lo"], r["hi"]
        if (not np.isfinite(lo) or x > lo) and (not np.isfinite(hi) or x < hi): out += [{**r, "hi": x}, {**r, "lo": x}]
        else: out.append(r)
    return out


def factor(st: dict | None, sign_pos: float, price: float | None = None) -> float:
    if not st or not st.get("active") or sign_pos == 0: return 1.0
    if price is not None and st["dir"] * (price - st["stop"]) < 0: return 1.0       # 다음 마감이 그 꼭짓점을 넘어 되돌아가면 해제
    return UP if np.sign(sign_pos) == st["dir"] else DN
