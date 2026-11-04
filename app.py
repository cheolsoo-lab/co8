"""BTC·ETH 최종 분석 (단일 화면).  실행: streamlit run app.py   ·  예전 전체 화면: streamlit run app_full.py
방향 = 추세(7~90일) + 호흡(4H 연속봉) 앙상블, 크기 = 변동성 맞춤 × 배율, 지지·저항 = 참고 표시(검증상 필터로 쓰면 손해)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data, indicators as ta, robust_live as RL, final as FN

if not hasattr(RL, "triggers"):   # 파일 일부만 바뀐 경우 안내
    st.error("robust_live.py가 예전 버전입니다. 최신 zip의 파일 전체를 폴더에 덮어쓰고, Streamlit을 껐다가(Ctrl+C) 다시 실행하세요.")
    st.stop()

st.set_page_config(page_title="△", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:3rem;max-width:1000px}"
            "[data-testid=stMetricValue]{font-size:1.3rem}</style>", unsafe_allow_html=True)

COINS = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
STATE = data.STATE
JOURNAL = STATE / "journal.csv"
JCOLS = ["date", "symbol", "side", "entry", "exit", "qty", "pnl_usdt", "note"]
MIN_NOTIONAL = 5.0
LEV_SET = 3   # 거래소 레버리지 설정값(격리). 실제 위험은 '명목'이 결정, 레버리지는 증거금만 줄임

with st.expander("△ 설정", expanded=False):
    c1, c2 = st.columns(2)
    equity = c1.number_input("선물 계좌 자본 (USDT)", 10.0, 10_000_000.0, 80.0, 10.0, key="equity")
    mult = c2.select_slider("배율 (1.5 권장 · 2 초과 비권장)", options=[0.5, 1.0, 1.5, 2.0, 3.0], value=1.5, key="mult")
    order = st.radio("주문 방식", ["지정가 (Post-only, 메이커) — 권장", "시장가 (테이커)"], horizontal=True, key="order")
    mtf_on = st.checkbox("흐름 정렬 역가중 (권장·기본) — 주봉·일봉·4시간이 모두 같으면 ½, 4시간이 둘 다 거스르면 1.5배", True, key="mtf_on")
    shape_on = st.checkbox("두 달 모양 반반 운용 (실험) — 4시간 방식 절반 + 일봉 60일 모양 절반을 한 수량으로 합침", False, key="shape_on")
    trap_on = st.checkbox("베어트랩 추가 (엄격 기준 통과 · 설정값 민감) — 직전 저점을 깼다가 3봉 안에 되돌아오면 24시간 롱 추가", False, key="trap_on")
    mode = st.radio("매매 방식 (하나만 고르기 — 둘은 보는 기간이 달라 서로 다른 말을 할 수 있음)",
                    ["4시간 추세 — 롱·숏 모두 (기본·권장, 낙폭 작음)", "스윙 — 길게 먹기 (롱만, 최근 5년 성적 약함, 낙폭 큼)"], key="mode")
SWING = mode.startswith("스윙")
MAKER = order.startswith("지정가")
MTF = bool(mtf_on)
SHAPE = bool(shape_on) and not SWING
TRAP = bool(trap_on) and not SWING


@st.cache_data(ttl=600, show_spinner="4H 봉 읽는 중…")
def load(symbol: str):
    df = data.drop_unclosed(data.fetch_candles(symbol, "4H", 1600), "4H")   # 주봉 26주 추세용 여유
    v = RL.one(df)
    v["pos_prev"] = RL.one(df.iloc[:-1])["pos"]          # 직전 마감 기준 목표 (방금 바뀌었는지 표시용)
    try:
        import mtf_live as ML
        v["mtf"] = ML.state(df)
    except Exception:
        v["mtf"] = None
    F = FN.frame(df)
    v.update({"atr": float(F["atr"][-1]), "res": float(F["res"][-1]), "sup": float(F["sup"][-1]), "trig": RL.triggers(df)})
    try:
        import beartrap as BT, novel as N_
        sg, _, stt = BT.signal(df)
        r_ = np.concatenate([[0.0], np.diff(np.log(df["close"].to_numpy(float)))])
        stt.update({"active": float(sg[-1]), "cap": float(min(N_.TV / N_.sigma(r_)[-1], N_.LMAX))})
        v["trap"] = stt
    except Exception:
        v["trap"] = None
    try:
        import adaptive as AD, precision as PR_, novel as N2, fakes as FK
        F2 = PR_.coin_frame(df)
        g = int(AD.regimes(F2)[-1]); v["regime"] = AD.REG[g] if g >= 0 else None
        base = N2.band(F2["d"] * F2["cap"], 0.15); x = PR_.pnl(base, F2, 0.0003); n_ = len(x)
        segs = FK.segments(base); open_ = segs and segs[-1][2] == n_ and np.sign(base[-1]) != 0
        done = segs[:-1] if open_ else segs
        rets = [float(np.prod(1 + x[a + 1:b + 1]) - 1) for _, a, b in done]
        streak = 0
        for r_ in reversed(rets):
            if r_ <= 0: streak += 1
            else: break
        v["fake_streak"] = streak
        v["open_ret"] = float(np.prod(1 + x[segs[-1][1] + 1:]) - 1) if open_ else None
    except Exception:
        v["regime"] = v.get("regime"); v["fake_streak"] = None; v["open_ret"] = None
    return df.iloc[-120:], v


def load_shape(symbol: str):
    return _load_shape(symbol)


@st.cache_data(ttl=1800, show_spinner="일봉 모양 읽는 중…")
def _load_shape(symbol: str):
    import shape_model as SM
    df = data.drop_unclosed(data.fetch_candles(symbol, "1D", 400), "1D")
    return SM.state(df)


@st.cache_data(ttl=3600, show_spinner=False)
def fear_greed():
    try:
        import requests
        d = requests.get("https://api.alternative.me/fng/", params={"limit": 2, "format": "json"}, timeout=10).json()["data"]
        return int(d[0]["value"]), d[0]["value_classification"], int(d[1]["value"])
    except Exception:
        return None


@st.cache_data(ttl=20, show_spinner=False)
def live_px(symbol: str):
    try:
        t = data.ticker(symbol); return float(t["lastPr"]), float(t["bidPr"]), float(t["askPr"])
    except Exception:
        return None


STEP = {"BTCUSDT": 0.0001, "ETHUSDT": 0.01}   # 비트겟 최소 주문 단위(수량)


def qty(notional: float, price: float, step: float) -> float:
    """주문 가능한 수량(부호 포함): 거래소 단위로 반올림, 5 USDT 미만이면 0."""
    q = round(abs(notional) / price / step) * step
    return 0.0 if q * price < MIN_NOTIONAL else float(np.sign(notional) * q)


def todo(cur: float, tgt: float, price: float, step: float, nm: str) -> str:
    """현재 수량 → 목표 수량: 할 일 (수량은 거래소 단위로 반올림된 값)."""
    cur, tgt = qty(cur, price, step), qty(tgt, price, step)
    side = lambda x: "롱" if x > 0 else "숏"
    f = lambda x: f"{abs(x):.{max(0, -int(np.floor(np.log10(step))))}f} {nm}"
    if cur == 0 and tgt == 0: return "없음 (관망)"
    if cur == 0: return f"{side(tgt)} 진입 {f(tgt)}"
    if tgt == 0: return f"{side(cur)} 청산"
    if np.sign(cur) != np.sign(tgt): return f"{side(cur)} 청산 → {side(tgt)} {f(tgt)} 진입"
    if abs(tgt - cur) < step / 2: return "유지"
    return f"{side(cur)} {'늘리기' if abs(tgt) > abs(cur) else '줄이기'} → {f(tgt)}"


def chart(df, v):
    fig = go.Figure(go.Candlestick(x=df.index.tz_convert("Asia/Seoul"), open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                   increasing_line_color="#2e9e5b", decreasing_line_color="#d64545", name=""))
    for lv, nm, col in ((v["res"], "저항", "#d64545"), (v["sup"], "지지", "#2e9e5b")):
        if np.isfinite(lv):
            fig.add_hline(y=lv, line_dash="dot", line_color=col, annotation_text=f"{nm} {lv:,.1f}", annotation_position="top left")
    fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False, showlegend=False)
    return fig


@st.cache_data(ttl=1800, show_spinner="일봉 읽는 중…")
def load_swing(symbol: str):
    import swing_live as SL
    df = data.drop_unclosed(data.fetch_candles(symbol, "1D", 500), "1D")
    return df.iloc[-240:], SL.state(df)


def chart_swing(df, v):
    fig = go.Figure(go.Candlestick(x=df.index.tz_convert("Asia/Seoul"), open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                   increasing_line_color="#2e9e5b", decreasing_line_color="#d64545", name=""))
    fig.add_hline(y=v["hi20"], line_dash="dot", line_color="#2e9e5b", annotation_text=f"20일 최고 {v['hi20']:,.1f} (넘으면 돌파 롱)", annotation_position="top left")
    fig.add_hline(y=v["lo50"], line_dash="dot", line_color="#d64545", annotation_text=f"50일 최저 {v['lo50']:,.1f}", annotation_position="bottom left")
    fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False, showlegend=False)
    return fig


def render_journal():
    with st.expander("△ 매매 일지", expanded=False):
        j = pd.read_csv(JOURNAL, encoding="utf-8") if JOURNAL.exists() else pd.DataFrame(columns=JCOLS)
        with st.form("jform", clear_on_submit=True):
            f1, f2, f3, f4 = st.columns(4)
            s_ = f1.selectbox("코인", list(COINS.values())); sd = f2.selectbox("방향", ["롱", "숏", "모의 롱", "모의 숏"])
            en = f3.number_input("진입가", 0.0, step=1.0); ex = f4.number_input("청산가", 0.0, step=1.0)
            g1, g2 = st.columns(2)
            q = g1.number_input("수량(코인)", 0.0, format="%.4f"); note = g2.text_input("메모")
            if st.form_submit_button("기록"):
                sign = 1 if "롱" in sd else -1
                pnl = sign * (ex - en) * q if ex > 0 else 0.0
                j = pd.concat([j, pd.DataFrame([[pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M"), s_, sd, en, ex, q, round(pnl, 4), note]], columns=JCOLS)])
                j.to_csv(JOURNAL, index=False, encoding="utf-8")
        if len(j):
            st.dataframe(j.tail(20), hide_index=True, width="stretch")
            st.caption(f"누적 손익 {j['pnl_usdt'].sum():+,.2f} USDT · 기록 {len(j)}건")


st.title("△")
if SWING:
    st.caption("스윙 모드: 하루 한 번(일봉 마감 새벽 01시 KST, 아침에 확인하면 됨) → 비중이 바뀌었을 때만 조정. 숏은 하지 않음. 수익을 보장하지 않습니다.")
    for sym, nm in COINS.items():
        try:
            dfs, sv = load_swing(sym)
        except Exception as e:
            st.warning(f"{nm} 일봉 오류: {e}"); continue
        w = sv["weight"]; nt = equity * w / len(COINS)
        n_long = sum(sv["votes"].values())
        st.subheader(f"△ {nm}  ${sv['close']:,.1f}")
        a, b, c_ = st.columns(3)
        a.metric("판정", "관망 (현금)" if w == 0 else f"롱 {w:.0%}", f"12개 규칙 중 {n_long}개 롱 → 0·50·100% 중 {w:.0%}", delta_color="off")
        b.metric("지금 들고 있을 포지션", (f"롱 {nt:,.0f} USDT" if nt >= MIN_NOTIONAL else ("없음" if nt == 0 else "5 USDT 미만 → 없음")),
                 f"레버리지 1배 기준 (최대 2배)", delta_color="off")
        c_.metric("50일 / 200일 수익", f"{sv['ret50']:+.0%} / {sv['ret200']:+.0%}", f"마지막 조정 {sv['last_change'][:10]} ({sv['prev_weight']:.0%}→{w:.0%})", delta_color="off")
        st.caption("롱 규칙: " + ", ".join(k for k, x in sv["votes"].items() if x) + " · 현금 규칙: " + (", ".join(k for k, x in sv["votes"].items() if not x) or "없음"))
        st.plotly_chart(chart_swing(dfs, sv), width="stretch", key=f"sw_{sym}")
    with st.expander("△ 스윙 규칙과 검증 (8년, 2018-07~2026-11)", expanded=False):
        st.markdown("""
- **비중** = 아래 12개 규칙 중 '롱'인 비율 → 3단계(25% 미만 0%, 75% 미만 50%, 그 이상 100%). 하나를 고르지 않고 평균 (과최적 방지). BTC·ETH에 자본 반씩. 조정은 코인당 연 16회 정도.
  반전확인 10·15·20·25·30% (저점에서 그만큼 오르면 롱, 고점에서 그만큼 내리면 현금) · 신고가 돌파 20·50·100일 · 장기방향 50·100·200일 · 방향합의
- **숏은 안 함**: 스윙 기간(몇 주~몇 달)으로 숏을 하면 8년 누적 숏 수익 −42%. 코인은 급락 뒤 급반등이 커서, 느린 규칙이 하락을 확인할 즈음엔 이미 많이 빠진 뒤였음. 만장일치 약세일 때만 숏해도 −12%. **숏은 4시간 모드에서** (5년 숏 부분 연 +4%, 2022년 +7%·2026년 +11%).
- **주의 — 최근 5년(2021-10~)만 보면 연 −3%, 낙폭 −58%** (8년 성적은 2019~2021 상승장 덕분). 같은 5년 4시간 모드는 연 +15%, 낙폭 −20% (펀딩비 반영).

| 8년 검증 (BTC·ETH 반반, 1배) | 연 수익 | 최대 낙폭 | 샤프 (전반/후반) |
|---|---|---|---|
| **스윙 롱만 3단계** | **+42%** | **−55%** | 1.01 (1.26 / 0.68) |
| 같은 것, 펀딩비 연 10% 반영 | +34% | −58% | 0.87 (1.14 / 0.49) |
| 그냥 보유 (선물, 펀딩 반영) | +17% | −78% | 0.57 |
| 그냥 보유 (현물) | +29% | −76% | 0.72 |
| 스윙 ×2 레버리지 | 약 +65% | 약 −81% | 같음 |

- 이득의 대부분은 큰 하락장을 피한 것(2018년 −4% vs 보유 −64%, 2022년 −36% vs −66%). 후반 4년만 보면 그냥 보유와 비슷함.
- 고점·저점은 지나고 나서야 확인됨: 20% 되돌림 확인이면 BTC 상승 스윙 평균 +97% 중 실제 +31%, 스윙의 59%는 손실로 끝남 (큰 스윙 몇 개가 전체 수익).
- 한 번 들어간 뒤 최대 33~45%까지 역행한 적 있음 → 레버리지 2.2배를 넘으면 청산됐을 거래가 있었음. **1배 권장, 최대 2배.**
- 롱만 하는 방식이라 **현물로 해도 됨** → 펀딩비 없음, 청산 없음. 선물로 하면 펀딩비만큼(연 약 8%p) 깎임.
""")
    render_journal()
    st.stop()

st.caption("4시간 봉 마감(01·05·09·13·17·21시 KST) 직후 확인 → '지금 들고 있을 포지션'에 내 포지션을 맞추기. 장중에는 가격표만 보고 기다림. 수익을 보장하지 않습니다.")

rows = {}
for sym, nm in COINS.items():
    try:
        rows[sym] = load(sym)
    except Exception as e:
        st.warning(f"{nm} 데이터 오류: {e}")

_fg = fear_greed()
if _fg:
    _kr = {"Extreme Fear": "극단 공포", "Fear": "공포", "Neutral": "중립", "Greed": "탐욕", "Extreme Greed": "극단 탐욕"}.get(_fg[1], _fg[1])
    st.caption(f"시장 심리 · 공포·탐욕 지수 {_fg[0]} ({_kr}, 어제 {_fg[2]}) — 참고용: 8년 검증에서 '공포에 사기'는 효과 없었고, 지수만으로 매매하면 최근 5년 샤프 −0.04")
# 배지 색: (배경, 테두리, 왼쪽 강조선·글자색, 표시) — 은은한 바탕 + 진한 글자
BADGE = {"롱": ("#f0f8f3", "#d3eadb", "#1f7a45", "롱 ▲"), "숏": ("#fcf2f2", "#f1d4d4", "#b23b3b", "숏 ▼"), "관망": ("#f5f6f8", "#e3e6ea", "#5b6270", "관망 ―")}
badge_cols = st.columns(len(COINS))


try:
    with open(STATE / "fake_runs.json", encoding="utf-8") as _fh:
        FAKE_RUNS = json.load(_fh)
except Exception:
    FAKE_RUNS = {}
try:
    with open(STATE / "topdown_result.json", encoding="utf-8") as _fh:
        TOPDOWN = json.load(_fh)["table"]
except Exception:
    TOPDOWN = {}


def badge(nm, price, word, strength, qty_line, change, flow, next_t, mood=""):
    bg, bd, ac, label = BADGE[word]
    chg = (f'<div style="margin-top:8px;background:#fff8dc;border:1px solid #f0e2a8;color:#6b5400;border-radius:6px;padding:3px 8px;font-size:0.82rem;font-weight:600">{change}</div>' if change else "")
    return (f'<div style="border-radius:12px;padding:12px 14px 10px 14px;margin-bottom:12px;background:{bg};border:1px solid {bd};border-left:5px solid {ac};color:#374151;line-height:1.45">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'<span style="font-size:1.45rem;font-weight:700;color:{ac}">{label}<span style="font-size:0.85rem;font-weight:500;color:#6b7280"> {strength}</span></span>'
            f'<span style="font-size:0.85rem;color:#6b7280">{nm} · {price}</span></div>'
            f'<div style="font-size:0.95rem;font-weight:600;color:#1f2937">{qty_line}</div>{chg}'
            f'<div style="margin-top:6px;font-size:0.82rem;color:#4b5563">{flow}</div>'
            + (f'<div style="margin-top:6px;padding-top:6px;border-top:1px dashed {bd};font-size:0.78rem;color:#6b7280">{mood}</div>' if mood else "") +
            f'<div style="margin-top:4px;font-size:0.75rem;color:#9ca3af">다음 판단 {next_t:%H:%M} 마감</div></div>')


def desc(q, nm, dq):
    return "없음" if q == 0 else f"{'롱' if q > 0 else '숏'} {abs(q):.{dq}f} {nm}"


for i_c, (sym, nm) in enumerate(COINS.items()):
    if sym not in rows:
        continue
    df, v = rows[sym]
    k = equity * mult / len(COINS)
    sv = None
    if SHAPE:
        try:
            sv = load_shape(sym)
        except Exception as e:
            st.warning(f"{nm} 일봉 모양 오류: {e} — 4시간 방식만 사용")
    sp = 0.5 * sv["pos"] if sv else 0.0
    wt = 0.5 if sv else 1.0
    tr_ = v.get("trap") or {}
    tp = tr_.get("active", 0.0) * tr_.get("cap", 0.0) if TRAP else 0.0                 # 지금 베어트랩 몫
    tp_next = tr_.get("cap", 0.0) if (TRAP and tr_.get("active_left", 0) >= 2) else 0.0   # 다음 마감 뒤에도 남는 몫
    import mtf_live as ML
    mstate = v.get("mtf") if (MTF and not SWING) else None
    fx = (lambda p: ML.factor(float(np.sign(p)), mstate)[0]) if mstate else (lambda p: 1.0)
    notional = k * (wt * v["pos"] * fx(v["pos"]) + sp + tp)
    px = live_px(sym); last = px[0] if px else v["close"]
    stp = STEP[sym]; dq = max(0, -int(np.floor(np.log10(stp))))
    q_now = qty(notional, last, stp); has = q_now != 0
    side = ("숏" if q_now < 0 else "롱") if has else ""
    close_t = (pd.Timestamp(v["time"]) + pd.Timedelta(hours=4)).tz_convert("Asia/Seoul")
    next_t = close_t + pd.Timedelta(hours=4)
    st.subheader(f"△ {nm}  ${last:,.1f}")
    a, b, c_ = st.columns(3)
    a.metric("지금 들고 있을 포지션", f"{side} {abs(q_now):.{dq}f} {nm}" if has else "없음 (관망)",
             (f"약 {abs(q_now)*last:,.0f} USDT (목표 {abs(notional):,.0f}) · {LEV_SET}배 격리 증거금 {abs(q_now)*last/LEV_SET:,.0f}") if has else
             (f"목표 {abs(notional):.0f} USDT → 최소 단위 {stp} {nm}(약 {stp*last:,.0f} USDT) 미만" if abs(notional) > 0.5 else f"점수 {v['dir']:+.2f}"), delta_color="off")
    if has and px:
        ent = px[2] if q_now < 0 else px[1]
        b.metric("진입 가격 (지금)", f"{ent:,.{2 if sym == 'ETHUSDT' else 1}f}", f"{'매도1호가' if q_now < 0 else '매수1호가'}에 지정가" if MAKER else "시장가", delta_color="off")
    else:
        b.metric("진입 가격 (지금)", "-", "포지션 없음", delta_color="off")
    # 다음 마감 기준 가장 가까운 '할 일이 생기는' 가격 (위·아래)
    T = v["trig"]; up = dn = None
    for r in T:
        act = todo(notional, k * (wt * r["pos"] * fx(r["pos"]) + sp + tp_next), last, stp, nm)
        if act == "유지" or act.startswith("없음"): continue
        if r["lo"] >= v["close"] - 1e-9 and (up is None or r["lo"] < up[0]): up = (r["lo"], act)
        if r["hi"] <= v["close"] + 1e-9 and (dn is None or r["hi"] > dn[0]): dn = (r["hi"], act)
    lines = []
    if up and np.isfinite(up[0]): lines.append(f"▲ {up[0]:,.1f} 위 마감 → {up[1]}")
    if dn and np.isfinite(dn[0]): lines.append(f"▼ {dn[0]:,.1f} 아래 마감 → {dn[1]}")
    # 상단 한눈 배지
    word = "관망" if not has else side
    vd = RL.verdict(v["dir"]); strength = ""
    if has:
        strength = "(강)" if "(강)" in vd else "(약)"
        if not vd.startswith(side): strength = "(신호 약해짐 · 유지)"
    q_prev = qty(k * (wt * v["pos_prev"] * fx(v["pos_prev"]) + sp + tp), last, stp)
    change = (f"방금 바뀜 ({close_t:%H:%M} 마감): {desc(q_prev, nm, dq)} → {desc(q_now, nm, dq)}" if q_prev != q_now else "")
    qline = (f"{abs(q_now):.{dq}f} {nm} · 약 {abs(q_now)*last:,.0f} USDT" if has else "포지션 없음")
    flow = "<br>".join(lines) if lines else "다음 마감까지 어느 가격이든 유지"
    mood = ""
    if v.get("fake_streak") is not None:
        mx = FAKE_RUNS.get(sym, {}).get("max_run")
        if v.get("open_ret") is not None and has:
            mood = f"지금 방향 {'+' if v['open_ret'] >= 0 else ''}{v['open_ret']:.1%} · "
        mood += f"직전까지 가짜 전환 연속 {v['fake_streak']}번" + (f" (5년 최장 {mx}번)" if mx else "")
        if change: mood += " · 새 방향의 74%는 가짜(평균 −0.6%, 2.4일), 26%가 진짜(평균 +2.6%, 8.4일) — 가짜는 진짜를 잡는 입장료"
    if v.get("mtf"):
        arw = lambda x: "▲" if x > 0 else ("▼" if x < 0 else "―")
        f_, ag_ = ML.factor(float(np.sign(v["pos"])), v["mtf"])
        tag = {2: "모두 같음 = 늙은 추세 → 크기 ½", 1: "하나만 같음 → 크기 그대로", 0: "4시간이 큰 흐름을 거스름 = 새 흐름 태동 → 1.5배"}.get(ag_, "포지션 없음")
        mood = f"흐름 · 주봉 {arw(v['mtf']['dW'])} 일봉 {arw(v['mtf']['dD'])} 4시간 {arw(v['pos'])} — {tag}" + ("" if MTF else " (설정에서 꺼짐)") + ("<br>" + mood if mood else "")
        wsg = float(np.sign(v["mtf"]["dW"]))
        if wsg != 0:
            import topdown as TD
            code = int(wsg > 0) * 4 + int(np.sign(v["mtf"]["dD"]) == wsg) * 2 + int(np.sign(v["dir"]) == wsg)
            tdi = TOPDOWN.get(str(code))
            if tdi:
                mood = f"상태 · {TD.label(code)} — 4주 안 주봉 흐름 전환 확률 {tdi['flip']:.0%} (5년 평균 47%)<br>" + mood
    with badge_cols[i_c]:
        st.markdown(badge(nm, f"${last:,.{2 if sym == 'ETHUSDT' else 1}f}", word, strength, qline, change, flow, next_t, mood), unsafe_allow_html=True)
    c_.metric(f"전환 가격 ({next_t:%H:%M} 마감 기준)", lines[0].split(" → ")[0] if lines else "변화 없음",
              " / ".join(l.split(" → ")[1] for l in lines[:1]) if lines else "어느 가격에 마감해도 유지", delta_color="off")
    tb = []
    for r in reversed(T):
        lo = "" if not np.isfinite(r["lo"]) else f"{r['lo']:,.1f}"; hi = "" if not np.isfinite(r["hi"]) else f"{r['hi']:,.1f}"
        rng = f"{lo} 위" if not hi else (f"{hi} 아래" if not lo else f"{lo} ~ {hi}")
        tgt = qty(k * (wt * r["pos"] * fx(r["pos"]) + sp + tp_next), last, stp); here = r["lo"] <= last < r["hi"]
        tb.append({"다음 4H 마감가": rng + ("  ← 지금" if here else ""), "그때 포지션": (f"{'숏' if tgt < 0 else '롱'} {abs(tgt):.{dq}f} {nm} (약 {abs(tgt)*last:,.0f} USDT)" if tgt else "없음"),
                   "할 일": todo(notional, k * (wt * r["pos"] * fx(r["pos"]) + sp + tp_next), last, stp, nm)})
    st.dataframe(pd.DataFrame(tb), hide_index=True, width="stretch")
    sr = []
    if np.isfinite(v["res"]): sr.append(f"저항 {v['res']:,.1f}")
    if np.isfinite(v["sup"]): sr.append(f"지지 {v['sup']:,.1f}")
    st.caption(f"{RL.verdict(v['dir'])} · 점수 {v['dir']:+.2f} (추세 {v['tsmom']:+.2f} / 호흡 {v['breath']:+.2f}) · " + " · ".join(sr)
               + f" · 기준: {close_t:%m-%d %H:%M} 마감 봉 · 장중에 전환 가격을 찍었다가 돌아오면 무시 (마감가만 봄)"
               + (" · Post-only 지정가, 30분 안에 안 채워지면 시장가" if MAKER else ""))
    if v.get("regime"):
        st.caption(f"지금 상황 · {v['regime']} (30일 흐름 × 최근 변동성) — 상황마다 방법을 바꾸면 5년 검증에서 크게 나빠져서(샤프 1.37 → 0.04~0.56) 방법은 그대로, 크기만 변동성에 맞춰 자동 조절")
    if tr_:
        if tr_.get("active", 0) > 0:
            ttxt = f"발생 ({pd.Timestamp(tr_['last_event']).tz_convert('Asia/Seoul'):%m-%d %H:%M} 마감) → 롱 유지 남은 {tr_['active_left']}봉"
        elif tr_.get("armed_level"):
            ttxt = f"대기 — 직전 저점 {tr_['armed_level']:,.1f} 하향 이탈. {tr_['armed_left']}봉 안에 그 위로 마감하면 베어트랩 롱" + (f" (+{k*tr_['cap']:,.0f} USDT 추가)" if TRAP else "")
        else:
            ttxt = f"없음 · 감시 중인 직전 저점 {tr_['last_low']:,.1f} (종가로 깨졌다가 3봉 안에 복귀하면 신호)" if tr_.get("last_low") else "없음"
        st.caption(f"패턴 감시 · 베어트랩: {ttxt}" + ("" if TRAP else " · 설정에서 '베어트랩 추가'를 켜야 수량에 반영"))
    if sv:
        side_s = "롱" if sv["side"] > 0 else "숏"; held = "롱" if sv["held_side"] > 0 else ("숏" if sv["held_side"] < 0 else "없음")
        mte = f"{sv['m_te']:+.1%}" if sv["m_te"] is not None else "-"
        st.caption(f"두 달 모양: 지금 모양은 '{side_s} 모양' (모양 {sv['cluster']}번 · 학습기간 다음 10일 평균 {sv['m_tr']:+.1%}, 확인기간 {mte}) · "
                   f"모양 몫은 {sv['since']}부터 {held} 유지 중 (남은 {sv['days_left']}일, 10일 단위로만 바꿈) · 모양 몫 {k*sp:+,.0f} + 4시간 몫 {k*wt*v['pos']:+,.0f} = 합계 {notional:+,.0f} USDT (+롱 / −숏)")
        with st.expander(f"△ {nm} 두 달 모양 보기", expanded=False):
            fg = go.Figure()
            fg.add_trace(go.Scatter(y=sv["shape"], mode="lines", name="지금 60일 모양", line=dict(color="#1f4fbf", width=3)))
            fg.add_trace(go.Scatter(y=sv["centroid"], mode="lines", name=f"모양 {sv['cluster']}번 대표 모양", line=dict(color="#9a9a9a", width=2, dash="dot")))
            fg.update_layout(height=240, margin=dict(l=5, r=5, t=10, b=5), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), legend=dict(orientation="h"))
            st.plotly_chart(fg, width="stretch", key=f"shp_{sym}")
    st.plotly_chart(chart(df, v), width="stretch", key=f"ch_{sym}")

try:
    with open(STATE / ("certainty_mtf.json" if MTF else "certainty.json"), encoding="utf-8") as _fh:
        _cert = json.load(_fh)
    _tb = _cert["table"].get(f"{'maker' if MAKER else 'taker'}|{float(mult)}")
    if _tb:
        with st.expander(f"△ 확실성 — 지금 설정(배율 {mult}, {'지정가' if MAKER else '시장가'})으로 하면", expanded=True):
            _rows = [{"기간": k, "수익 날 확률": f"{v['p']:.0%}", "중앙값": f"{v['med']:+.0%}", "나쁜 10%": f"{v['p10']:+.0%}", "좋은 10%": f"{v['p90']:+.0%}",
                      "중간에 −30% 낙폭 겪을 확률": f"{v['dd30']:.0%}"} for k, v in _tb.items()]
            st.dataframe(pd.DataFrame(_rows), hide_index=True, width="stretch")
            _a = _cert.get(f"asym|{'메이커' if MAKER else '테이커'}", {})
            if _a:
                st.caption(f"5년 BTC·ETH 실제 손익을 30일 단위로 섞어 4천 번 다시 뽑은 결과(펀딩비 반영). 한 달은 동전 던지기에 가깝고 시간이 갈수록 확실해짐. "
                           f"방향 적중률은 약 50%지만 맞을 때 더 크게 먹어서 수익이 남 (이긴 달 {_a['mon_win']:.0%}, 평균 {_a['mon_avg_w']:+.1%} / 진 달 평균 {_a['mon_avg_l']:+.1%}). "
                           "한두 달 성적으로 방법을 바꾸지 마세요.")
except Exception:
    pass

with st.expander("△ 4원칙 — 이 매매법이 따르는 원리 (5년 검증)", expanded=False):
    st.markdown("""
**① 손실은 짧게, 이익은 길게 — 참. 이 방식의 본질**
- 거래 900개 중 이긴 건 26%뿐. 하지만 평균 이익 +2.5%(8.6일 보유) vs 평균 손실 −0.6%(2.4일) → 손익비 4.4.
- 상위 10% 거래가 순이익 전체(254%)를 만듦. → **작은 손실이 연달아 나도 신호를 건너뛰지 말 것.** 큰 추세 하나를 놓치면 1년 수익이 사라짐.
- 출구를 손봐도(2ATR 조기 손절, 3ATR 추적 연장) 더 좋아지지 않음 — 이미 '틀리면 금방 방향이 바뀌어 작게, 맞으면 길게' 구조.

**② 힘이 다하면 꺾인다 — 위쪽은 거짓, 아래쪽은 반쪽**
- 위로 과열(EMA20보다 3ATR 위)되면 24시간 뒤 더 오른 비율 54~58%. **과열이라고 숏 치면 손해** (과열 역매매 샤프 −1.12).
- 아래로 과열되면 58~59%가 반등하지만, 틀릴 때 훨씬 크게 빠져 9가지 변형 모두 손해. **떨어지는 칼은 잡지 않음.**
- 포스터의 반전 패턴(쌍봉·H&S·쐐기·원형·하모닉) 전체도 손해(샤프 −0.69). 추세는 생각보다 쉽게 안 꺾임.

**③ 모였다가 터진다 — 변동성은 참, 방향은 거짓**
- 변동성이 눌린 뒤(볼린저 폭 하위 20%) 다음 4일 변동성이 1.2배로 커짐.
- 하지만 처음 터진 방향은 46~48%로 오히려 반대가 많음. **첫 돌파를 쫓지 말고, 추세 신호가 확인된 뒤 따라감** (이 방식이 자동으로 그렇게 함).

**⑨ 위에서 아래로 — 큰 흐름의 '전환'은 일봉·4시간으로 미리 보이지만, 다음 주 '방향'은 안 보인다**
- 주봉 방향 × 일봉·4시간이 따르는지 거스르는지 = 8가지 상태. 일봉·4시간이 모두 주봉을 거스르는 '깊은 조정'이면 4주 안에 주봉 흐름이 뒤집힌 비율 70~80%, 세 시간대가 같은 '순풍'이면 30%대 (앞뒤 일관, 구별력 0.65~0.67).
- 하지만 다음 1주 가격 방향은 8개 상태 모두 통계적으로 의미 없음 → 상태만 보고 롱·숏을 정하면 안 됨. 배지의 '상태'는 큰 흐름이 바뀔 가능성을 가늠하는 참고 정보.
- 약한 경향(두 기간 같은 방향, 확정은 아님): 상승 주봉 속 깊은 조정은 다시 오르는 편(+0.9%/주), 하락 주봉 속 짧은 반등 뒤엔 다시 내리는 편.

**⑧ 흐름 정렬의 역설 — 주봉·일봉·4시간이 다 맞을 때가 가장 나쁘고, 4시간이 큰 흐름을 거스를 때가 가장 좋다** (기본 적용)
- 4시간 포지션 성과(5년): 세 시간대 모두 같은 방향(보유 54%) 샤프 0.33|0.60 · 하나만 같음 1.47|1.80 · 일봉·주봉 모두 반대(11%) 1.85|2.66 (앞|뒤).
- 해석: 모두 같은 방향 = 모두가 아는 늙은 추세. 새 흐름은 언제나 낡은 큰 흐름을 거슬러 시작함. 주봉을 따르는 엘더식 필터는 오히려 손해(0.50).
- 그래서 크기를 거꾸로: 모두 같음 ½ · 하나만 같음 1 · 모두 반대 1.5배 → 샤프 1.04 → 1.36, 낙폭 −19% → −13%. 일봉·주봉 정의 5가지로 바꿔도 모두 개선(1.16~1.36).
- 평균 노출이 0.72배로 줄어서 같은 수익을 원하면 배율을 올림: 지정가 기준 배율 1.5 → 연 +31%·낙폭 −17%, 2.0 → 연 +42%·낙폭 −22%.

**⑦ 차트 속 사람 심리 — 남는 건 '쏠림'과 '손절', 반대매매 심리는 안 통함** (일봉 2018~2026 + 공포·탐욕 지수)
- 쏠림·과소반응은 참: 하루 +2σ 급등 뒤 5일 +2.4%(60% 상승), −2σ 급락 뒤 −1.2% — 두 기간 모두, 추세 통제 후 t=3.8. 이 방식의 추세·호흡이 바로 이 심리를 따라감 (더해도 개선 없음, 상관 0.38).
- '공포에 사고 탐욕에 팔기'는 거짓: 극단 공포 뒤 20일 수익은 평소와 같음(+1.7%), 탐욕일수록 오히려 더 오름(최근 4년은 효과 사라짐).
- 처분 효과(평균단가 위·아래), 52주 고점 기준점: 추세를 빼면 효과 없음.
- 손절 사냥(베어트랩)만 역행이 통함 — 모두의 손절이 몰린 직전 저점.

**⑥ 가짜 전환에 속지 않는다 — 들어가기 전엔 못 가린다. 대신 '입장료'로 작게 낸다**
- 5년 전환 860번 중 74%가 가짜(평균 −0.6%, 2.4일 만에 끝), 26%가 진짜(평균 +2.6%, 8.4일). 가짜 손실 합계 −361% vs 진짜 이익 합계 +577%.
- 전환 순간 단서 12가지(기간 일치도·연속봉·거래량·변동 확대·고점/저점 돌파·일봉 흐름·BTC/ETH 동반·종가 위치·신호 세기·혼조·변동성·경로 곧음)를 다 합친 모델도 뒤 기간 구별력 AUC 0.54 (동전 수준).
- '확인되면 진입' 필터 5종(0.60~1.05), 분할 진입 6종(0.89~1.04) 모두 기준(1.04)보다 낫지 않음 — 가짜를 거르면 진짜의 앞부분도 같이 놓침.
- 그래서 속지 않는 법 = **가짜가 연달아 와도 계속 따라가기.** 연속 가짜 평균 4번, 10번 중 1번은 7~9번, 5년 최장 BTC 18번(그 코인 몫 −12.6%)·ETH 15번. 배지에 지금 연속 횟수를 표시.

**⑤ 상황마다 방법을 바꾼다 — 거짓. 바꾸는 건 '방법'이 아니라 '크기와 반응 속도'**
- 기존 7종(이평선·돈치안·MACD·RSI·볼린저·일목·슈퍼트렌드) + 자체 + 새 관점 4종, 15가지 대회에서 이 방식이 1등 (샤프 1.04, 2등 호흡 0.90, 기존 최고 MACD 0.66, RSI·볼린저 역추세 −1.2).
- 상승·하락·횡보 × 변동성 6가지 상황별 1등이 다음 기간엔 이어지지 않음 (앞·뒤 상관 −0.03). 매달 상황별 최고 방법으로 갈아타면 샤프 0.04~0.56 (그대로 유지 1.37).
- 이 방식은 이미 상황에 맞춰 연속적으로 변함: 변동성 커지면 크기 자동 축소, 호흡(4H 연속봉)이 새 흐름에 빨리 반응, 추세는 7~90일을 섞어 빠름·느림을 같이 봄.

**④ 모두가 아는 자리는 잘 안 온다 — 둥근 숫자는 거짓, '손절 자리'는 참**
- BTC 80,000·ETH 2,500 같은 둥근 숫자도 다른 가격대와 똑같이 93~96% 도달함.
- 진짜는 **모두가 손절을 걸어둔 직전 저점 아래를 쓸고 3봉 안에 되돌아오는 베어트랩** → 롱 (적중 58%, 앱과 합산 샤프 1.04 → 1.26). 반대(불트랩)는 안 통함. 설정에서 '베어트랩 추가'로 켬 (설정값을 바꾸면 약해지는 점은 주의).
""")

with st.expander("△ 규칙 (한 번 읽어두기)", expanded=False):
    st.markdown("""
- **방향**: 추세(7·14·30·60·90일 수익 방향 평균) 50% + 호흡(같은 방향 4H 봉이 3~6개 이어지면 따라가기) 50%. 점수 ±0.15 안이면 관망.
- **크기**: 변동성이 크면 작게, 작으면 크게(코인 연변동성 40% 기준) × 배율. 표시된 금액이 '지금 들고 있어야 할 포지션'.
- **매매 순서**: ① 4H 마감 직후 앱 확인 → ② '지금 들고 있을 포지션'과 내 포지션이 다르면 맞춤 (진입 가격 = 그때 현재가, 지정가) → ③ 다음 마감까지는 '전환 가격'표만 보고 대기.
- **'판정' 글자 말고 '포지션 금액'을 따를 것**: 롱/숏 글자만 보고 크로스로 뒤집으면 5년 샤프 1.08 → 0.47~0.64, 방향 전환 연 92회 → 222회로 크게 나빠짐. 앱 금액은 작은 흔들림엔 그대로 있도록 계산돼 있음.
- **바꾸는 때**: 4H 봉 마감 때만. 앱 금액이 바뀌지 않았으면 그대로 둠 — 잦은 매매는 수수료로 수익을 깎음.
- **지지·저항**: 차트 참고용. "벽 앞에서 기다리기"는 5년 검증에서 수익을 거의 0으로 만들어서 판단에 쓰지 않음.
- **손절 = 전환 가격**: 따로 두지 않음 — 다음 마감가가 전환 가격을 넘으면 그때 정리. 2ATR·지지저항 손절은 검증에서 오히려 나빴음. 대신 크기 자체가 작게 잡혀 있음. 전환 가격은 마감 때마다 새로 계산됨.
- **레버리지**: 거래소 설정은 3배 격리면 충분. 실제 위험은 레버리지가 아니라 '명목 금액'이 정함.
- **롱·숏 모두**: 5년 검증에서 롱 부분 연 +11%, 숏 부분 연 +4%. 숏은 하락장에서 계좌를 지켜줌 (2022년 롱 −17% / 숏 +7% → 합계 −11%).
- **적중률의 진실**: 강한 신호도 다음 4시간~1주 방향 적중률은 48~54% (동전 수준). 돈이 되는 이유는 '맞을 때 크게, 틀릴 때 작게' — 추세가 길게 이어질 때 크게 먹고, 아니면 금방 방향이 바뀌어 작게 잃음.
- **주문은 지정가(Post-only)**: 수수료가 편도 0.08% → 약 0.03%로 줄어 5년 샤프 0.79 → 1.04. 검증한 개선안 중 BTC·ETH와 알트 22종 모두에서 확실히 좋아진 건 이것 하나.
- **바꾸지 않은 것**: 추세·호흡 일치 필터, 완충 확대, 숏 절반, 확신 가중 — BTC·ETH에선 좋아 보였지만 알트 22종에선 나빠져서(과최적 신호) 기각.
- **추가로 기각한 정보** (11-03): 시장 폭(다른 23개 코인 방향), 일봉 스윙 투표, BTC 선행, 머신러닝 메타 라벨(BTC·ETH 1.04→1.16이지만 알트 1.03→0.98), 코인베이스 프리미엄 — 크기 가중 적중률은 모두 50% 근처 그대로.
""")

with st.expander("△ 검증 성적 (5년, 과장 없이)", expanded=False):
    st.markdown("""
| 방식 (BTC·ETH) | 연 수익 | 최대 낙폭 | 비고 |
|---|---|---|---|
| **이 앱 방식 (배율 1)** | **+18%** | **−19%** | 수수료 0.15%면 +11% / −26%, 펀딩비 연 10% 반영 시 +15% / −20% |
| **이 앱 방식 + 지정가 주문 (배율 1, 펀딩 반영)** | **+21%** | **−19%** | 샤프 1.04, 1년 수익 확률 82% · 배율 1.5면 +31% / −27% |
| **+ 흐름 정렬 역가중 (기본, 지정가·펀딩, 배율 1)** | **+20%** | **−11%** | 샤프 1.36, 1년 수익 확률 89% · 배율 1.5면 +31%/−17%, 2.0이면 +42%/−22% |
| 스윙 롱만 (같은 5년) | −3% | −58% | 최근 5년 약함 |
| 앱 + 베어트랩 (5년, 같은 크기 합산) | +27% | −19% | 샤프 1.04 → 1.26, 그러나 뒤 절반은 1.25 → 1.25 (최근엔 보탬 없음), 설정을 바꾸면 약해짐 (실험 옵션) |
| 포스터 반전 패턴 전체 (쌍봉·H&S·쐐기·원형·하모닉) | −12% | −55% | 샤프 −0.69 — 4H에서 포스터 방향대로 매매하면 손실 |
| 캔들 패턴 전체 (약 30종) | −42% | −96% | 샤프 −1.46 |
| 두 달 모양 반반 (2023-07~ 확인 구간) | +40% | −19% | 샤프 1.47 → 1.72, BTC 1.15 → 1.35 / ETH 1.45 → 1.42 · 2025년 약함 (실험 옵션) |
| 이 앱 방식 (배율 1.5) | 약 +27% | 약 −28% | 권장 |
| 지지저항 벽 앞 대기 추가 | +1% | −19% | 버림 |
| 1회 2% 손실 한도 + 2ATR 손절 | +40% | −61% | 낙폭 너무 큼, 최장 17연패 |
| 1회 2% + 지지저항 뒤 손절 | +59% | −70% | 수수료 0.15%면 BTC 연 +3%, 낙폭 −75% |
| 150배 · 증거금 2% | 파산 | — | 0.21% 역행에 청산, 청산율 77~94% |
""")
    st.caption("한계: 펀딩비는 맨 위 줄 비고만 반영, 미래에도 같다는 보장 없음, 하루 1% 복리 수준과는 거리가 멂. 소액일 때는 실제 주문 전 일지에 모의로 먼저 기록 권장.")

render_journal()
