"""구조 지도 (11-26, research/atlas*.py · entry_struct.py · struct_sleeve.py, 교차 검토 반영).
레벨 22종을 '직전 4H 파동 안 같은 위치의 가짜 레벨'·가짜 시장 40개와 비교해 확인한 것만 지금 차트에 표시.
- 위치가 거의 다 정함: 직전 파동을 90% 넘게 되돌린 자리(시작점 근처)는 아무 가격이나 잘 뚫림(58%), 30~70% 되돌림은 살짝 버팀(53%, 가짜 시장 51~52%).
- 같은 위치에서 더 뚫리는 선: 전주 고저(+3.7%p) · 4H 꼭짓점(+2%p) · 같은 고점·저점 · 짧은 추세선 · 라운드. 피보 0.618은 특별하지 않음.
방향은 앱이 정함. 구조 신호를 크기에 더하면 비트겟 2022-08~에서 나빠져서 크기엔 안 씀 (표시만)."""
import numpy as np, pandas as pd

# 실제로 닿았을 때의 과거 비율 (코인베이스 BTC·ETH 2015-07~2026-11 1시간 체결, 레벨에서 1.5ATR 먼저 가는 쪽). 위치 효과 포함한 절대 비율.
P_ORIGIN = 0.58      # 직전 파동 시작점까지 되돌림 → 뚫림 (앱 반대쪽 레벨 60% · 앱 쪽 55% · 알트 53%; 받침 구역보다 −11.8%p, 가짜 시장 ±5.7%p 안)
P_CUSHION = 0.53     # 30~70% 되돌림 구역 → 버팀 (앱 쪽 0.5~0.7 55% · 가짜 시장 51~52% → 약함)
P_EXTREME = 0.54     # 되돌림 뒤 직전 끝(고점·저점) 재시험 → 뚫림
BRK = {"전주 고저": 0.57, "같은 고점·저점": 0.56, "4H 꼭짓점": 0.54, "라운드 넘버": 0.53, "짧은 추세선": 0.54}


def _atr(df, n=14):
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close")); pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc))); return pd.Series(tr).rolling(n).mean().to_numpy()


def zigzag(h, l, atr, k):
    """확정 꼭짓점만: (꼭짓점 봉, 가격, +1 고점/−1 저점, 확정 봉). 확정 봉 종가부터 알 수 있음."""
    n = len(h); ok = np.flatnonzero(np.isfinite(atr))
    if not len(ok): return []
    i0 = int(ok[0]); piv = []; dirn = 1; hi, hi_i, lo, lo_i = h[i0], i0, l[i0], i0
    for i in range(i0 + 1, n):
        if dirn == 1:
            if h[i] > hi: hi, hi_i = h[i], i
            elif l[i] <= hi - k * atr[i]: piv.append((hi_i, hi, 1, i)); dirn = -1; lo, lo_i = l[i], i
        else:
            if l[i] < lo: lo, lo_i = l[i], i
            elif h[i] >= lo + k * atr[i]: piv.append((lo_i, lo, -1, i)); dirn = 1; hi, hi_i = h[i], i
    return piv


def swing(df, k, atr):
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close")); P = zigzag(h, l, atr, k)
    if len(P) < 2: return None
    a, b = P[-2], P[-1]; ty = b[2]; rng = b[1] - a[1]
    r = (b[1] - c[-1]) / rng if rng else np.nan
    ext = (h[b[3]:].max() if ty < 0 else l[b[3]:].min())            # 지금 진행 중인 되돌림의 끝
    return {"k": k, "dir": "상승" if ty > 0 else "하락", "ty": int(ty), "start": float(a[1]), "end": float(b[1]), "t_start": df.index[a[0]], "t_end": df.index[b[0]],
            "r": float(r), "zone": (float(b[1] - 0.7 * rng), float(b[1] - 0.3 * rng)), "now_ext": float(ext), "piv": P}


def state(df):
    """df: 닫힌 4H 봉. 앱 방향과 맞춘 설명은 view(st, 앱 부호)에서."""
    if len(df) < 200: return None
    atr = _atr(df); A = float(atr[-1]); c = float(df["close"].iloc[-1]); h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    S3, S6 = swing(df, 3.0, atr), swing(df, 6.0, atr)
    if S3 is None: return None
    L = []
    def add(px, what, role, note, kind):
        if np.isfinite(px) and abs(px - c) <= 8 * A: L.append({"price": float(px), "what": what, "role": role, "note": note, "kind": kind, "dist": (px - c) / A})
    for S, tag in ((S3, "4H 파동"), (S6, "큰 파동")):
        if S is None: continue
        add(S["start"], f"{tag} 시작점 ({S['dir']} {S['start']:,.0f}→{S['end']:,.0f})", "방아쇠", f"닿으면 뚫림 {P_ORIGIN:.0%}", "origin")
        if 0.0 <= S["r"] <= 1.0:
            add(S["end"], f"{tag} 끝 ({'고점' if S['ty'] > 0 else '저점'})", "방아쇠", f"다시 닿으면 뚫림 {P_EXTREME:.0%}", "extreme")
        z0, z1 = S["zone"]
        if tag == "4H 파동" and S["r"] < 0.7:
            add((z0 + z1) / 2, f"{tag} 받침 구역 {min(z0, z1):,.0f}~{max(z0, z1):,.0f} (30~70% 되돌림)", "받침", f"닿으면 버팀 {P_CUSHION:.0%}", "cushion")
    # 전주 고저 (지난 완성 주, 월요일 시작 UTC)
    wk = df.resample("W-MON", label="left", closed="left").agg({"high": "max", "low": "min"})
    if len(wk) >= 2:
        add(float(wk["high"].iloc[-2]), "전주 고가", "방아쇠", f"닿으면 뚫림 {BRK['전주 고저']:.0%}", "pw")
        add(float(wk["low"].iloc[-2]), "전주 저가", "방아쇠", f"닿으면 뚫림 {BRK['전주 고저']:.0%}", "pw")
    # 같은 고점·저점 · 아직 안 깨진 4H 꼭짓점 (×3, 최근 14개)
    P = S3["piv"][-14:]; seen = set()
    for j in range(len(P)):
        i4, pr, ty, cf = P[j]
        after = (h[cf + 1:].max() if ty > 0 else l[cf + 1:].min()) if cf + 1 < len(h) else np.nan
        if np.isfinite(after) and ((ty > 0 and after > pr) or (ty < 0 and after < pr)): continue      # 이미 뚫림
        eq = None
        for m in range(j - 2, -1, -2):
            q = P[m]; mid = [x[1] for x in P[m + 2:j:2]]
            if mid and ((ty > 0 and max(mid) > max(pr, q[1])) or (ty < 0 and min(mid) < min(pr, q[1]))): break
            if abs(q[1] - pr) <= 0.3 * atr[cf]: eq = (max if ty > 0 else min)(pr, q[1]); break
        if eq is not None: add(eq, f"같은 {'고점' if ty > 0 else '저점'} 두 번", "방아쇠", f"닿으면 뚫림 {BRK['같은 고점·저점']:.0%}", "eq"); seen.add(round(eq, 6))
        elif j >= len(P) - 6: add(pr, f"4H {'고점' if ty > 0 else '저점'} ({df.index[i4].tz_convert('Asia/Seoul'):%m-%d})", "방아쇠", f"닿으면 뚫림 {BRK['4H 꼭짓점']:.0%}", "piv")
    # 짧은 추세선: 마지막 두 같은 종류 꼭짓점 (로그 직선, 지금 봉까지 연장)
    tl = []
    for typ in (1, -1):
        q = [p for p in S3["piv"] if p[2] == typ][-2:]
        if len(q) == 2 and q[1][0] > q[0][0]:
            x0, y0, x1, y1 = q[0][0], np.log(q[0][1]), q[1][0], np.log(q[1][1]); n = len(df) - 1
            now = float(np.exp(y0 + (y1 - y0) / (x1 - x0) * (n - x0)))
            add(now, f"짧은 추세선 ({'고점' if typ > 0 else '저점'} 두 개 연결)", "방아쇠", f"닿으면 뚫림 {BRK['짧은 추세선']:.0%}", "tl")
            tl.append({"x0": df.index[x0], "y0": float(q[0][1]), "x1": df.index[n], "y1": now})
    # 라운드 넘버 (BTC 5,000 · ETH 500 단위)
    st_ = 10 ** np.floor(np.log10(c)) / 2
    for m in (np.floor(c / st_), np.ceil(c / st_)):
        add(m * st_, f"라운드 넘버 {m * st_:,.0f}", "방아쇠", f"닿으면 뚫림 {BRK['라운드 넘버']:.0%}", "round")
    # 0.25ATR 안 겹친 선 합치기 (방아쇠 우선)
    PRI = {"origin": 6, "pw": 5, "eq": 4, "extreme": 3, "piv": 2, "tl": 1, "round": 0}
    L.sort(key=lambda x: x["price"]); M = []
    for x in L:
        if M and abs(x["price"] - M[-1]["price"]) <= 0.25 * A and x["kind"] != "cushion" and M[-1]["kind"] != "cushion":
            y = M[-1]; y["n"] = y.get("n", 1) + 1
            if PRI[x["kind"]] > PRI[y["kind"]]: y.update(what=x["what"] + " · " + y["what"], kind=x["kind"], note=x["note"], price=x["price"], dist=x["dist"])
            else: y["what"] += " · " + x["what"]
        else: M.append(dict(x))
    return {"close": c, "atr": A, "s3": {k: v for k, v in S3.items() if k != "piv"}, "s6": ({k: v for k, v in S6.items() if k != "piv"} if S6 else None),
            "levels": M, "trend_lines": tl}


def view(st, app_side=0):
    """앱 방향(+1 롱 · −1 숏 · 0)에 맞춘 표·한 줄. 시작점 확률: 앱이 그쪽을 향하면(레벨이 앱 반대 역할) 60%, 앱 쪽을 받치면 55%."""
    c, A = st["close"], st["atr"]; rows = []
    for x in st["levels"]:
        note = x["note"]
        if x["kind"] == "origin" and app_side:
            toward = (x["price"] > c) == (app_side > 0); note = f"닿으면 뚫림 {P_ORIGIN + (0.02 if toward else -0.03):.0%}"
        rows.append({"가격": x["price"], "거리": f"{x['dist']:+.1f} ATR", "성격": x["role"], "닿으면 (과거)": note, "무엇": x["what"]})
    strong = lambda x: x["role"] == "방아쇠" and x["kind"] in ("origin", "pw", "eq", "extreme")
    up = [x for x in st["levels"] if strong(x) and x["price"] > c]; dn = [x for x in st["levels"] if strong(x) and x["price"] < c]
    S3 = st["s3"]; r = S3["r"]
    pos = ("받침 구역 안" if 0.3 <= r <= 0.7 else ("시작점 앞" if 0.9 <= r <= 1.0 else ("시작점 뚫음" if r > 1 else ("끝 넘어 진행" if r < 0 else ("얕은 되돌림" if r < 0.3 else "깊은 되돌림")))))
    chip = f"4H {S3['dir']} 파동 {pos}({r:.0%})"
    pf = 0 if c >= 1000 else 2
    if dn: chip += f" · 아래 방아쇠 {dn[-1]['price']:,.{pf}f}"
    if up: chip += f" · 위 방아쇠 {up[0]['price']:,.{pf}f}"
    return {"rows": rows, "chip": chip, "text": summary(st["s3"], st["s6"], c, A, app_side)}


def summary(S3, S6, c, A, app_side):
    def one(S, tag):
        r = S["r"]; mv = S["dir"]
        if r < 0: pos = f"{mv} 파동 끝을 넘어 진행 중"
        elif r < 0.3: pos = f"얕은 되돌림 {r:.0%}"
        elif r <= 0.7: pos = f"받침 구역 안 ({r:.0%} 되돌림)"
        elif r < 0.9: pos = f"깊은 되돌림 {r:.0%} — 시작점 가까움"
        elif r <= 1.0: pos = f"시작점 바로 앞 ({r:.0%}) — 뚫릴 쪽이 우세"
        else: pos = f"시작점을 뚫음 ({r:.0%}) — {('하락' if S['ty'] > 0 else '상승')} 쪽으로 넘어감"
        return f"{tag} {mv} {S['start']:,.0f}→{S['end']:,.0f} · {pos}"
    t = one(S3, "4H 파동")
    if S6: t += " │ " + one(S6, "큰 파동")
    if app_side:
        back = (S3["ty"] > 0) == (app_side > 0)                   # 직전 4H 파동 방향 = 앱 방향 → 받침 구역이 앱 쪽
        t += " │ 앱 " + ("롱" if app_side > 0 else "숏") + (f" = 직전 4H 파동 쪽: 받침 구역이 앱 쪽을 살짝 받침, 시작점 {S3['start']:,.0f}이 앱 반대 방아쇠" if back
                                                          else f" ≠ 직전 4H 파동: 그 파동의 시작점 {S3['start']:,.0f}이 앱 쪽 방아쇠 (닿으면 뚫림 60%)")
    return t
