"""고전 차트 패턴·캔들 패턴 자동 탐지 + 검증 (BTC·ETH, 4H 5년 / 일봉 8년).  python patterns2.py
피벗: 지그재그(ATR 2배 되돌림으로 확정, 확정 시점부터 사용 → 미래참조 없음).
차트 패턴(포스터 정의를 수치로): 신호는 '돌파 마감' 봉에서 확정, 다음 봉 시가 진입.
  고갈→반전: 이중천장/바닥, 헤드앤숄더/역, 상승쐐기(하락)/하락쐐기(상승), 원형 천장/바닥, 가틀리·사이퍼(하모닉)
  응축→방출: 상승/하락/대칭 삼각형, 박스(직사각형), 깃발·페넌트(직전 깃대 방향)
  함정→역행: 불트랩(돌파 후 3봉 안에 복귀 → 숏)/베어트랩
캔들 패턴 약 30종: 직전 추세 맥락(하락 중 망치 등) 포함.
평가: 패턴 방향 기준 적중률·평균 수익(겹침 제거), 앞/뒤 절반, 포스터식 손절·목표 매매(R)."""
import json, math, warnings
import numpy as np, pandas as pd
from scipy import stats as SS
import data, indicators as ta, swing as W

warnings.filterwarnings("ignore")
COST = 0.0003
FAM = {"이중천장": "고갈→반전", "이중바닥": "고갈→반전", "헤드앤숄더": "고갈→반전", "역헤드앤숄더": "고갈→반전", "상승쐐기": "고갈→반전", "하락쐐기": "고갈→반전",
       "원형천장": "고갈→반전", "원형바닥": "고갈→반전", "가틀리(상승)": "고갈→반전", "가틀리(하락)": "고갈→반전", "사이퍼(상승)": "고갈→반전", "사이퍼(하락)": "고갈→반전",
       "상승삼각형": "응축→방출", "하락삼각형": "응축→방출", "대칭삼각형↑": "응축→방출", "대칭삼각형↓": "응축→방출", "박스 상단돌파": "응축→방출", "박스 하단돌파": "응축→방출",
       "상승깃발·페넌트": "응축→방출", "하락깃발·페넌트": "응축→방출", "불트랩": "함정→역행", "베어트랩": "함정→역행"}


def load(tf):
    out = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        df = ta.resample_ohlcv(data.load_history(s, "1H", 1825, refresh=False), "4h").dropna() if tf == "4H" else W.load(s)
        out[s] = df
    return out


def zigzag(h, l, atr, k=2.0):
    n = len(h); piv = []; trend = 0; hi = lo = 0
    for t in range(1, n):
        a = atr[t]
        if not np.isfinite(a): hi = lo = t; continue
        if trend >= 0 and h[t] >= h[hi]: hi = t
        if trend <= 0 and l[t] <= l[lo]: lo = t
        if trend == 0:
            if hi < t and h[hi] - l[t] >= k * a: piv.append((t, hi, h[hi], 1)); trend, lo = -1, t
            elif lo < t and h[t] - l[lo] >= k * a: piv.append((t, lo, l[lo], -1)); trend, hi = 1, t
        elif trend == 1 and l[t] <= h[hi] - k * a:
            piv.append((t, hi, h[hi], 1)); trend, lo = -1, t
        elif trend == -1 and h[t] >= l[lo] + k * a:
            piv.append((t, lo, l[lo], -1)); trend, hi = 1, t
    return piv


def chart_events(df, k=2.0, maxw=90, round_w=60):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    atr = ta.atr(df).to_numpy(); n = len(c)
    piv = zigzag(h, l, atr, k); pv_by_t = {}
    for p in piv: pv_by_t.setdefault(p[0], []).append(p)
    P = []; armed = {}; ev = []; last_round = -999; bo = None  # bo: 돌파 기록 (함정용)

    def arm(name, side, kind, line, stop, target, expire, key):
        armed[key] = dict(name=name, side=side, kind=kind, line=line, stop=stop, target=target, expire=expire)

    for t in range(n):
        a = atr[t]
        for p in pv_by_t.get(t, []):
            P.append(p)
            if len(P) >= 4 and np.isfinite(a):
                (_, i0, p0, t0), (_, i1, p1, t1), (_, i2, p2, t2), (_, i3, p3, t3) = P[-4:]
                # 이중천장: L0 H1 L H2
                if t3 == 1:
                    hgt = (p1 + p3) / 2 - p2
                    if hgt >= 2.5 * a and abs(p1 - p3) <= 0.25 * hgt and p0 < p2:
                        arm("이중천장", -1, "below", lambda tt, v=p2: v, max(p1, p3), p2 - hgt, t + max(10, i3 - i1), ("DT", i1))
                else:
                    hgt = p2 - (p1 + p3) / 2
                    if hgt >= 2.5 * a and abs(p1 - p3) <= 0.25 * hgt and p0 > p2:
                        arm("이중바닥", 1, "above", lambda tt, v=p2: v, min(p1, p3), p2 + hgt, t + max(10, i3 - i1), ("DB", i1))
            if len(P) >= 6 and np.isfinite(a):
                q = P[-6:]; ty = q[-1][3]
                (_, iL0, L0, _), (_, iS1, S1, _), (_, iN1, N1, _), (_, iH, Hd, _), (_, iN2, N2, _), (_, iS2, S2, _) = q
                if ty == 1 and Hd > max(S1, S2) + 0.5 * a and abs(S1 - S2) <= 0.35 * (Hd - (N1 + N2) / 2) and L0 < min(N1, N2):
                    sl_ = (N2 - N1) / max(iN2 - iN1, 1); neck = lambda tt, b=N1, s_=sl_, i_=iN1: b + s_ * (tt - i_)
                    arm("헤드앤숄더", -1, "below", neck, S2, None, t + max(10, iS2 - iS1), ("HS", iS1)); armed[("HS", iS1)]["height"] = Hd - neck(iH)
                if ty == -1 and Hd < min(S1, S2) - 0.5 * a and abs(S1 - S2) <= 0.35 * ((N1 + N2) / 2 - Hd) and L0 > max(N1, N2):
                    sl_ = (N2 - N1) / max(iN2 - iN1, 1); neck = lambda tt, b=N1, s_=sl_, i_=iN1: b + s_ * (tt - i_)
                    arm("역헤드앤숄더", 1, "above", neck, S2, None, t + max(10, iS2 - iS1), ("IHS", iS1)); armed[("IHS", iS1)]["height"] = neck(iH) - Hd
            if len(P) >= 5 and np.isfinite(a):
                q = P[-5:]
                Hs = [(i, v) for (_, i, v, ty) in q if ty == 1][-2:]; Ls = [(i, v) for (_, i, v, ty) in q if ty == -1][-2:]
                if len(Hs) == 2 and len(Ls) == 2 and (q[-1][1] - q[0][1]) <= maxw:
                    (ha_i, ha), (hb_i, hb) = Hs; (la_i, la), (lb_i, lb) = Ls
                    du, dl = hb - ha, lb - la; su, sl_ = du / max(hb_i - ha_i, 1), dl / max(lb_i - la_i, 1)
                    up = lambda tt, b=hb, s_=su, i_=hb_i: b + s_ * (tt - i_); dn = lambda tt, b=lb, s_=sl_, i_=lb_i: b + s_ * (tt - i_)
                    flat_u, flat_l = abs(du) < 0.6 * a, abs(dl) < 0.6 * a
                    height = max(ha, hb) - min(la, lb); start = q[0][1]; exp = t + max(10, q[-1][1] - start)
                    pole = abs(q[0][2] - P[-6][2]) if len(P) >= 6 else 0; pole_dir = np.sign(q[0][2] - P[-6][2]) if len(P) >= 6 else 0
                    key = ("TRI", start)
                    if flat_u and dl >= 0.6 * a: arm("상승삼각형", 1, "above", up, lb, None, exp, key)
                    elif du <= -0.6 * a and flat_l: arm("하락삼각형", -1, "below", dn, hb, None, exp, key)
                    elif du <= -0.6 * a and dl >= 0.6 * a:
                        if pole >= 2 * height and pole_dir != 0:
                            arm("상승깃발·페넌트" if pole_dir > 0 else "하락깃발·페넌트", int(pole_dir), "above" if pole_dir > 0 else "below", up if pole_dir > 0 else dn,
                                lb if pole_dir > 0 else hb, None, exp, key)
                        else:
                            arm("대칭삼각형↑", 1, "above", up, lb, None, exp, key); arm("대칭삼각형↓", -1, "below", dn, hb, None, exp, ("TRI2", start))
                    elif du >= 0.6 * a and dl >= 0.6 * a:
                        if dl > du: arm("상승쐐기", -1, "below", dn, hb, None, exp, key)
                        elif pole >= 2 * height and pole_dir < 0: arm("하락깃발·페넌트", -1, "below", dn, hb, None, exp, key)
                    elif du <= -0.6 * a and dl <= -0.6 * a:
                        if du < dl: arm("하락쐐기", 1, "above", up, lb, None, exp, key)
                        elif pole >= 2 * height and pole_dir > 0: arm("상승깃발·페넌트", 1, "above", up, lb, None, exp, key)
                    elif flat_u and flat_l:
                        arm("박스 상단돌파", 1, "above", up, lb, None, exp, key); arm("박스 하단돌파", -1, "below", dn, hb, None, exp, ("BOX2", start))
                    for kk in (key, ("TRI2", start), ("BOX2", start)):
                        if kk in armed: armed[kk]["height"] = height
                # 하모닉 X A B C D (D = 방금 확정)
                (_, iX, X, tX), (_, iA, A, _), (_, iB, B, _), (_, iC, C, _), (_, iD, Dp, tD) = q
                if tD == -1:   # 상승형: X저 A고 B저 C고 D저
                    xa = A - X
                    if xa > 0:
                        ab, bc, ad = (A - B) / xa, (C - B) / max(A - B, 1e-9), (A - Dp) / xa
                        if 0.55 <= ab <= 0.70 and 0.382 <= bc <= 0.886 and 0.70 <= ad <= 0.85 and Dp > X:
                            ev.append(dict(t=t, side=1, name="가틀리(상승)", stop=X, target=C))
                        xc = (C - X) / xa; dd_ = (C - Dp) / max(C - X, 1e-9)
                        if 1.13 <= xc <= 1.414 and 0.382 <= ab <= 0.618 and 0.70 <= dd_ <= 0.85 and Dp > X:
                            ev.append(dict(t=t, side=1, name="사이퍼(상승)", stop=X, target=C))
                else:          # 하락형
                    xa = X - A
                    if xa > 0:
                        ab, bc, ad = (B - A) / xa, (B - C) / max(B - A, 1e-9), (Dp - A) / xa
                        if 0.55 <= ab <= 0.70 and 0.382 <= bc <= 0.886 and 0.70 <= ad <= 0.85 and Dp < X:
                            ev.append(dict(t=t, side=-1, name="가틀리(하락)", stop=X, target=C))
                        xc = (X - C) / xa; dd_ = (Dp - C) / max(X - C, 1e-9)
                        if 1.13 <= xc <= 1.414 and 0.382 <= ab <= 0.618 and 0.70 <= dd_ <= 0.85 and Dp < X:
                            ev.append(dict(t=t, side=-1, name="사이퍼(하락)", stop=X, target=C))
        # 돌파 확인
        for key in list(armed):
            if key not in armed: continue
            A_ = armed[key]
            if t > A_["expire"]: del armed[key]; continue
            lv = A_["line"](t)
            hit = (A_["kind"] == "above" and c[t] > lv) or (A_["kind"] == "below" and c[t] < lv)
            if hit:
                hgt = A_.get("height") or abs(lv - A_["stop"])
                tgt = A_["target"] if A_["target"] is not None else lv + A_["side"] * hgt
                ev.append(dict(t=t, side=A_["side"], name=A_["name"], stop=A_["stop"], target=tgt))
                # 같은 구조에서 나온 반대편 무장 해제
                del armed[key]
                for kk in [x for x in armed if x[1] == key[1]]: del armed[kk]
        # 함정: 직전 확정 고점 돌파 후 3봉 안 복귀
        lastH = next((p for p in reversed(P) if p[3] == 1), None); lastL = next((p for p in reversed(P) if p[3] == -1), None)
        if t > 0 and lastH is not None and c[t] > lastH[2] >= c[t - 1]: bo = ("up", lastH[2], t)
        if t > 0 and lastL is not None and c[t] < lastL[2] <= c[t - 1]: bo = ("dn", lastL[2], t)
        if bo is not None and t - bo[2] <= 3 and t > bo[2]:
            if bo[0] == "up" and c[t] < bo[1]:
                stp = h[bo[2]:t + 1].max(); ev.append(dict(t=t, side=-1, name="불트랩", stop=stp, target=c[t] - 2 * (stp - c[t]))); bo = None
            elif bo[0] == "dn" and c[t] > bo[1]:
                stp = l[bo[2]:t + 1].min(); ev.append(dict(t=t, side=1, name="베어트랩", stop=stp, target=c[t] + 2 * (c[t] - stp))); bo = None
        elif bo is not None and t - bo[2] > 3: bo = None
        # 원형 천장/바닥 (2차 곡선 적합)
        if t >= round_w and t - last_round > round_w // 2 and np.isfinite(a):
            y = c[t - round_w + 1:t + 1]; x = np.arange(round_w); cf = np.polyfit(x, y, 2); fit = np.polyval(cf, x)
            r2 = 1 - ((y - fit) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12); vx = -cf[1] / (2 * cf[0]) if cf[0] != 0 else -1
            if r2 >= 0.8 and 0.3 * round_w <= vx <= 0.7 * round_w:
                rim = y[: max(3, round_w // 7)]
                if cf[0] > 0 and c[t] > rim.max() and c[t - 1] <= rim.max() and rim.max() - y.min() >= 3 * a:
                    ev.append(dict(t=t, side=1, name="원형바닥", stop=y.min(), target=rim.max() + (rim.max() - y.min()))); last_round = t
                if cf[0] < 0 and c[t] < rim.min() and c[t - 1] >= rim.min() and y.max() - rim.min() >= 3 * a:
                    ev.append(dict(t=t, side=-1, name="원형천장", stop=y.max(), target=rim.min() - (y.max() - rim.min()))); last_round = t
    return ev


def candle_events(df):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(c); body = np.abs(c - o); rng = np.maximum(h - l, 1e-12); up = h - np.maximum(o, c); lo = np.minimum(o, c) - l
    bull, bear = c > o, c < o; sma = pd.Series(c).rolling(20).mean().to_numpy()
    sh = lambda x, k: np.concatenate([np.full(k, np.nan if x.dtype.kind == "f" else False), x[:-k]])
    c1, o1, h1, l1, b1, r1 = sh(c, 1), sh(o, 1), sh(h, 1), sh(l, 1), sh(body, 1), sh(rng, 1)
    c2, o2, b2, r2 = sh(c, 2), sh(o, 2), sh(body, 2), sh(rng, 2)
    bull1, bear1, bull2, bear2 = sh(bull, 1).astype(bool), sh(bear, 1).astype(bool), sh(bull, 2).astype(bool), sh(bear, 2).astype(bool)
    down = (sh(c, 1) < sh(c, 6)) & (sh(c, 1) < sh(sma, 1)); upc = (sh(c, 1) > sh(c, 6)) & (sh(c, 1) > sh(sma, 1))
    down3 = (sh(c, 3) < sh(c, 8)) & (sh(c, 3) < sh(sma, 3)); up3 = (sh(c, 3) > sh(c, 8)) & (sh(c, 3) > sh(sma, 3))
    doji = body <= 0.1 * rng; doji1 = sh(doji, 1).astype(bool)
    hammer_shape = (lo >= 2 * body) & (up <= np.maximum(body, 0.1 * rng)) & (lo >= 0.6 * rng)
    inv_shape = (up >= 2 * body) & (lo <= np.maximum(body, 0.1 * rng)) & (up >= 0.6 * rng)
    big1 = b1 >= 0.6 * r1; big2 = b2 >= 0.6 * r2
    P = {
        ("망치형", 1): hammer_shape & down, ("교수형", -1): hammer_shape & upc,
        ("역망치형", 1): inv_shape & down, ("유성형", -1): inv_shape & upc,
        ("잠자리 도지", 1): doji & (lo >= 0.7 * rng) & down, ("비석 도지", -1): doji & (up >= 0.7 * rng) & upc,
        ("상승 장악형", 1): bear1 & bull & (c >= o1) & (o <= c1) & (body > b1) & down, ("하락 장악형", -1): bull1 & bear & (c <= o1) & (o >= c1) & (body > b1) & upc,
        ("관통형", 1): bear1 & big1 & bull & (o <= c1) & (c > (o1 + c1) / 2) & (c < o1) & down, ("먹구름형", -1): bull1 & big1 & bear & (o >= c1) & (c < (o1 + c1) / 2) & (c > o1) & upc,
        ("상승 잉태형", 1): bear1 & big1 & bull & (np.maximum(o, c) <= o1) & (np.minimum(o, c) >= c1) & (body <= 0.5 * b1) & down,
        ("하락 잉태형", -1): bull1 & big1 & bear & (np.maximum(o, c) <= c1) & (np.minimum(o, c) >= o1) & (body <= 0.5 * b1) & upc,
        ("상승 잉태 십자", 1): bear1 & big1 & doji & (np.maximum(o, c) <= o1) & (np.minimum(o, c) >= c1) & down,
        ("하락 잉태 십자", -1): bull1 & big1 & doji & (np.maximum(o, c) <= c1) & (np.minimum(o, c) >= o1) & upc,
        ("샛별형", 1): bear2 & big2 & (b1 <= 0.3 * b2) & bull & (c > (o2 + c2) / 2) & down3 | False,
        ("석별형", -1): bull2 & big2 & (b1 <= 0.3 * b2) & bear & (c < (o2 + c2) / 2) & up3 | False,
        ("샛별 도지", 1): bear2 & big2 & doji1 & bull & (c > (o2 + c2) / 2) & down3, ("석별 도지", -1): bull2 & big2 & doji1 & bear & (c < (o2 + c2) / 2) & up3,
        ("적삼병", 1): bull & bull1 & bull2 & (c > c1) & (c1 > c2) & (body >= 0.5 * rng) & (b1 >= 0.5 * r1) & (b2 >= 0.5 * r2) & (o > o1) & (o1 > o2) & down3,
        ("흑삼병", -1): bear & bear1 & bear2 & (c < c1) & (c1 < c2) & (body >= 0.5 * rng) & (b1 >= 0.5 * r1) & (b2 >= 0.5 * r2) & (o < o1) & (o1 < o2) & up3,
        ("상승 쓰리 인사이드", 1): bear2 & big2 & bull1 & (np.maximum(o1, c1) <= o2) & (np.minimum(o1, c1) >= c2) & bull & (c > o2) & down3,
        ("하락 쓰리 인사이드", -1): bull2 & big2 & bear1 & (np.maximum(o1, c1) <= c2) & (np.minimum(o1, c1) >= o2) & bear & (c < o2) & up3,
        ("상승 쓰리 아웃사이드", 1): bear2 & bull1 & (c1 >= o2) & (o1 <= c2) & bull & (c > c1) & down3,
        ("하락 쓰리 아웃사이드", -1): bull2 & bear1 & (c1 <= o2) & (o1 >= c2) & bear & (c < c1) & up3,
        ("집게 바닥", 1): bear1 & bull & (np.abs(l - l1) <= 0.1 * rng) & down, ("집게 천장", -1): bull1 & bear & (np.abs(h - h1) <= 0.1 * rng) & upc,
        ("상승 샅바형", 1): bull & (lo <= 0.02 * rng) & (body >= 0.6 * rng) & down, ("하락 샅바형", -1): bear & (up <= 0.02 * rng) & (body >= 0.6 * rng) & upc,
        ("상승 장대양봉(마루보즈)", 1): bull & (body >= 0.9 * rng), ("하락 장대음봉(마루보즈)", -1): bear & (body >= 0.9 * rng),
    }
    ev = []
    for (nm, side), m in P.items():
        for t in np.flatnonzero(np.nan_to_num(m.astype(float)) > 0):
            ev.append(dict(t=int(t), side=side, name=nm, stop=None, target=None))
    return ev


def evaluate(ev, df, hs, T_exit):
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close")); n = len(c); atr = ta.atr(df).to_numpy()
    lc = np.log(c); mid = df.index[n // 2]
    rows = []
    for e in ev:
        t = e["t"]
        if t + 1 >= n: continue
        ent = o[t + 1]; r = {"name": e["name"], "side": e["side"], "t": t, "time": df.index[t], "half": 1 if df.index[t] < mid else 2}
        for hh in hs:
            r[f"f{hh}"] = e["side"] * (lc[t + hh] - math.log(ent)) if t + hh < n else np.nan
        if e.get("stop") is not None and np.isfinite(e["stop"]):
            risk = e["side"] * (ent - e["stop"])
            if risk > 0.2 * atr[t]:
                ex = None
                for j in range(t + 1, min(n, t + 1 + T_exit)):
                    if (e["side"] > 0 and l[j] <= e["stop"]) or (e["side"] < 0 and h[j] >= e["stop"]): ex = e["stop"]; break
                    if (e["side"] > 0 and h[j] >= e["target"]) or (e["side"] < 0 and l[j] <= e["target"]): ex = e["target"]; break
                if ex is None: ex = c[min(n - 1, t + T_exit)]
                r["R"] = (e["side"] * (ex - ent) - COST * 2 * ent) / risk
        rows.append(r)
    return pd.DataFrame(rows)


def dedupe(df_, h):
    """같은 패턴·코인 신호가 h봉 안에 겹치면 첫 신호만 (통계용)."""
    keep, last = [], {}
    for i, r in df_.sort_values("t").iterrows():
        k = (r["name"], r["sym"])
        if k in last and r["t"] - last[k] < h: continue
        last[k] = r["t"]; keep.append(i)
    return df_.loc[keep]


def bh(p, q=0.10):
    p = np.asarray(p); m = len(p); o = np.argsort(p); thr = q * (np.arange(1, m + 1) / m); passed = np.zeros(m, bool)
    ok = np.flatnonzero(p[o] <= thr)
    if len(ok): passed[o[: ok.max() + 1]] = True
    return passed


def table(E, h, label, min_n=15):
    D_ = dedupe(E, h); rows = []
    base = {}
    for nm, g in D_.groupby("name"):
        x = g[f"f{h}"].dropna()
        if len(x) < min_n: continue
        tstat, p2 = SS.ttest_1samp(x, 0.0) if len(x) > 2 else (np.nan, np.nan)
        h1, h2 = g[g["half"] == 1][f"f{h}"].dropna(), g[g["half"] == 2][f"f{h}"].dropna()
        rr = g["R"].dropna() if "R" in g else pd.Series(dtype=float)
        rows.append({"패턴": nm, "가족": FAM.get(nm, "캔들"), "방향": "롱" if g["side"].iloc[0] > 0 else "숏", "n": len(x), "적중": float((x > 0).mean()), "평균": float(x.mean()),
                     "t": float(tstat), "p": float(p2), "앞": float(h1.mean()) if len(h1) else np.nan, "뒤": float(h2.mean()) if len(h2) else np.nan,
                     "앞적중": float((h1 > 0).mean()) if len(h1) else np.nan, "뒤적중": float((h2 > 0).mean()) if len(h2) else np.nan,
                     "R_n": len(rr), "R평균": float(rr.mean()) if len(rr) else np.nan, "R승률": float((rr > 0).mean()) if len(rr) else np.nan})
    T = pd.DataFrame(rows)
    if len(T): T["BH통과"] = bh(T["p"].fillna(1).to_numpy())
    return T


def main():
    out = {}
    for tf, hs, h0, T_exit in (("4H", (1, 6, 18), 6, 60), ("1D", (1, 5, 10), 5, 30)):
        D = load(tf); CE, KE = [], []
        for s, df in D.items():
            ce = evaluate(chart_events(df, maxw=90 if tf == "4H" else 120, round_w=60 if tf == "4H" else 40), df, hs, T_exit); ce["sym"] = s; CE.append(ce)
            ke = evaluate(candle_events(df), df, hs, T_exit); ke["sym"] = s; KE.append(ke)
        CE, KE = pd.concat(CE, ignore_index=True), pd.concat(KE, ignore_index=True)
        CE.to_csv(f"state/pattern_events_{tf}.csv", encoding="utf-8"); KE.to_csv(f"state/candle_events_{tf}.csv", encoding="utf-8")
        # 기준: 무조건 수익(드리프트)
        drift = {s: float(np.nanmean(np.diff(np.log(D[s]["close"].to_numpy()))) * h0) for s in D}
        print(f"\n════ {tf} — 기준 드리프트(아무 때나 롱 {h0}봉): BTC {drift['BTCUSDT']:+.2%} ETH {drift['ETHUSDT']:+.2%} ════")
        for nm_, E in (("차트 패턴", CE), ("캔들 패턴", KE)):
            T = table(E, h0, nm_)
            out[f"{tf}|{nm_}"] = T.to_dict("records")
            print(f"\n[{nm_}] {h0}봉 뒤 패턴 방향 기준 (겹침 제거) — BH 10% 다중검정")
            print(f"{'패턴':22s} {'가족':7s} {'방향':3s} {'n':>5s} {'적중':>6s} {'평균':>7s} {'p':>6s} {'앞/뒤 평균':>15s} {'BH':>3s} | 포스터식 매매 R(n, 승률)")
            for _, r in T.sort_values("평균", ascending=False).iterrows():
                rtxt = f"{r['R평균']:+.2f}R ({r['R_n']}, {r['R승률']:.0%})" if r["R_n"] else ""
                print(f"{r['패턴']:22s} {r['가족']:7s} {r['방향']:3s} {r['n']:5d} {r['적중']:6.1%} {r['평균']:+7.2%} {r['p']:6.3f} {r['앞']:+7.2%}/{r['뒤']:+7.2%} {'O' if r['BH통과'] else '-':>3s} | {rtxt}")
    with open("state/patterns2_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
