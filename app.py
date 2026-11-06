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
    mult = c2.select_slider("배율 (1.5 권장 · 4~5는 성장 최대지만 낙폭 −40~−50%)", options=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], value=1.5, key="mult")
    order = st.radio("주문 방식", ["지정가 (Post-only, 메이커) — 권장", "시장가 (테이커)"], horizontal=True, key="order")
    mtf_on = st.checkbox("흐름 정렬 역가중 (기본) — 주봉·일봉·4시간이 모두 같으면 ½, 4시간이 둘 다 거스르면 1.5배 · 11년 확인: 최근 5년은 좋았고(샤프 0.63→0.96) 2015~2021엔 나빴음(2.31→2.02), 낙폭은 줄어듦", True, key="mtf_on")
    shape_on = st.checkbox("두 달 모양 반반 운용 (실험) — 4시간 방식 절반 + 일봉 60일 모양 절반을 한 수량으로 합침", False, key="shape_on")
    trap_on = st.checkbox("베어트랩 추가 (엄격 기준 통과 · 설정값 민감) — 직전 저점을 깼다가 3봉 안에 되돌아오면 24시간 롱 추가", False, key="trap_on")
    mode = st.radio("매매 방식 (하나만 고르기 — 둘은 보는 기간이 달라 서로 다른 말을 할 수 있음)",
                    ["4시간 추세 — 롱·숏 모두 (기본·권장, 낙폭 작음)", "스윙 — 길게 먹기 (롱만, 최근 5년 성적 약함, 낙폭 큼)"], key="mode")
SWING = mode.startswith("스윙")
MAKER = order.startswith("지정가")
MTF = bool(mtf_on)
SHAPE = bool(shape_on) and not SWING
TRAP = bool(trap_on) and not SWING
WP = False   # 횡보면 ½: 11년 확인(2015~2021 처음 보는 기간)에서 나빠져 옵션 삭제 (cosmos.py · macro.py)


@st.cache_data(ttl=600, show_spinner="4H 봉 읽는 중…")
def load(symbol: str):
    df = data.drop_unclosed(data.fetch_candles(symbol, "4H", 1600), "4H")   # 주봉 26주 추세용 여유
    v = RL.one(df)
    v["src"] = data.SOURCE.get((symbol, "4H"), "Bitget")
    try:
        import cosmos_live as CL
        v["er"] = CL.er_state(df); v["er_prev"] = CL.er_state(df.iloc[:-1])
    except Exception:
        v["er"] = v["er_prev"] = None
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

st.caption("4시간 마감(01·05·09·13·17·21시) 직후 확인 · 마감가 기준 · 수익 보장 없음")

def render_diag(expanded=False):
    """접속 진단: 어느 시세 서버가 되는지, 어떤 버전으로 돌고 있는지."""
    with st.expander("△ 접속 진단", expanded=expanded):
        if st.button("지금 다시 검사", key="diag_btn"):
            st.cache_data.clear()
        import sys, platform, time as _t, requests as _rq
        tests = [("비트겟 (기본)", "https://api.bitget.com/api/v2/mix/market/ticker", {"symbol": "BTCUSDT", "productType": "USDT-FUTURES"}),
                 ("OKX (1차 대체)", "https://www.okx.com/api/v5/market/ticker", {"instId": "BTC-USDT-SWAP"}),
                 ("코인베이스 (2차 대체)", "https://api.exchange.coinbase.com/products/BTC-USD/ticker", None),
                 ("공포·탐욕 지수", "https://api.alternative.me/fng/", {"limit": 1})]
        out = []
        for nm_, url, p in tests:
            t0 = _t.time()
            try:
                r = _rq.get(url, params=p, timeout=8, headers={"User-Agent": "btceth-app"})
                out.append((nm_, "되는 중" if r.status_code == 200 else f"막힘 (HTTP {r.status_code})", f"{(_t.time() - t0) * 1000:.0f} ms", r.text[:80].replace("\n", " ")))
            except Exception as e:
                out.append((nm_, "접속 실패", "-", str(e)[:80]))
        st.dataframe(pd.DataFrame(out, columns=["시세 서버", "상태", "응답", "내용 앞부분"]), hide_index=True, width="stretch")
        used = {COINS[s]: (rows[s][1].get("src", "?") if s in rows else "불러오기 실패") for s in COINS}
        st.caption("지금 화면에 쓴 시세: " + " · ".join(f"{k} {v_}" for k, v_ in used.items()))
        import importlib.metadata as _md
        vers = []
        for pk in ("streamlit", "pandas", "numpy", "plotly", "scikit-learn", "scipy", "requests"):
            try: vers.append(f"{pk} {_md.version(pk)}")
            except Exception: vers.append(f"{pk} 없음")
        st.caption(f"파이썬 {platform.python_version()} · " + " · ".join(vers))
        st.caption("셋 다 '막힘'이면 그 서버에서 시세를 못 받는 것 → 내 PC에서 실행. 하나라도 '되는 중'이면 자동으로 그걸 씀.")


rows = {}
errs = {}
for sym, nm in COINS.items():
    try:
        rows[sym] = load(sym)
    except Exception as e:
        errs[nm] = str(e)
        st.warning(f"{nm} 데이터 오류: {e}")
if errs:
    render_diag(expanded=True)
_alt = {COINS[s]: rows[s][1].get("src", "") for s in rows if not str(rows[s][1].get("src", "Bitget")).startswith("Bitget")}
if _alt:
    st.caption("시세 대체 · " + " · ".join(f"{k} {v_.split(' (')[0]}" for k, v_ in _alt.items()) + " (비트겟 막힘)")

@st.cache_data(ttl=3600, show_spinner="일봉(거시) 읽는 중…")
def load_macro(symbol: str):
    import macro_live as MCL
    dd = data.drop_unclosed(data.fetch_candles(symbol, "1D", 700), "1D")
    return MCL.state(dd)


MAC = {}
for _s, _n in COINS.items():
    try:
        MAC[_n] = load_macro(_s)
    except Exception:
        MAC[_n] = None
@st.cache_data(ttl=3600, show_spinner=False)
def load_dom():
    import dominance_live as DL
    return DL.state()


try:
    DOM = load_dom()
except Exception:
    DOM = None
_fg = fear_greed()
_top = []
if any(MAC.values()):
    _top.append("거시 " + " · ".join(f"{n} {m['label']}" for n, m in MAC.items() if m))
if _fg:
    _kr = {"Extreme Fear": "극단 공포", "Fear": "공포", "Neutral": "중립", "Greed": "탐욕", "Extreme Greed": "극단 탐욕"}.get(_fg[1], _fg[1])
    _top.append(f"공포·탐욕 {_fg[0]} {_kr}")
if _top:
    st.caption(" │ ".join(_top) + " — 참고 (매매엔 안 씀)")
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
    import cosmos_live as CL
    ers = v.get("er") if WP else None
    wf = CL.factor(ers)                                                   # 지금 봉 기준 직진 1 / 횡보 ½
    wfx = (lambda r: CL.factor(ers, CL.mid(r))) if ers else (lambda r: 1.0)
    notional = k * (wt * v["pos"] * fx(v["pos"]) * wf + sp + tp)
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
    T = CL.split_rows(v["trig"], ers) if ers else v["trig"]; up = dn = None
    for r in T:
        act = todo(notional, k * (wt * r["pos"] * fx(r["pos"]) * wfx(r) + sp + tp_next), last, stp, nm)
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
    q_prev = qty(k * (wt * v["pos_prev"] * fx(v["pos_prev"]) * CL.factor(v.get("er_prev") if WP else None) + sp + tp), last, stp)
    change = (f"방금 바뀜 ({close_t:%H:%M} 마감): {desc(q_prev, nm, dq)} → {desc(q_now, nm, dq)}" if q_prev != q_now else "")
    qline = (f"{abs(q_now):.{dq}f} {nm} · 약 {abs(q_now)*last:,.0f} USDT" if has else "포지션 없음")
    flow = "<br>".join(lines) if lines else "다음 마감까지 어느 가격이든 유지"
    mood = ""
    if v.get("fake_streak") is not None:
        mx = FAKE_RUNS.get(sym, {}).get("max_run")
        if v.get("open_ret") is not None and has:
            mood = f"이번 방향 {'+' if v['open_ret'] >= 0 else ''}{v['open_ret']:.1%} · "
        mood += f"가짜 전환 연속 {v['fake_streak']}번" + (f" (최장 {mx})" if mx else "")
        if change: mood += " · 새 방향 4번 중 3번은 가짜"
    if v.get("mtf"):
        arw = lambda x: "▲" if x > 0 else ("▼" if x < 0 else "―")
        f_, ag_ = ML.factor(float(np.sign(v["pos"])), v["mtf"])
        tag = {2: "크기 ½", 1: "크기 그대로", 0: "크기 1.5배"}.get(ag_, "")
        mood = f"주 {arw(v['mtf']['dW'])} 일 {arw(v['mtf']['dD'])} 4H {arw(v['pos'])}" + (f" · {tag}" if (tag and MTF) else "") + ("<br>" + mood if mood else "")
        wsg = float(np.sign(v["mtf"]["dW"]))
        if wsg != 0:
            import topdown as TD
            code = int(wsg > 0) * 4 + int(np.sign(v["mtf"]["dD"]) == wsg) * 2 + int(np.sign(v["dir"]) == wsg)
            tdi = TOPDOWN.get(str(code))
            if tdi:
                mood = f"{TD.label(code).replace(' · ', '·')} · 4주 내 전환 {tdi['flip']:.0%}<br>" + mood
    if DOM:
        import dominance_live as DL
        mood += "<br>" if mood else ""
        if sym == "BTCUSDT":
            mood += (f"USDT.D {DOM['usdt_d']:.2f}% · 7일 {DL.label(DOM['u_z7'])}({DOM['u_z7']:+.1f}σ) · 1년 범위 {DOM['u_lvl']:.0%} · BTC.D {DOM['btc_d']:.1f}% {DL.arrow(DOM['btc_d_tr'])}"
                     if "u_z7" in DOM else f"USDT.D {DOM['usdt_d']:.2f}% {DL.arrow(DOM['usdt_d_tr'])} · BTC.D {DOM['btc_d']:.1f}% {DL.arrow(DOM['btc_d_tr'])}")
        else:
            mood += (f"TOTAL3 7일 {DL.label(DOM['t3_z7'])}({DOM['t3_z7']:+.1f}σ) · 30일 {DL.label(DOM['t3_z30'])}({DOM['t3_z30']:+.1f}σ) · 알트 비중 {DL.arrow(DOM['alt_tr'])}"
                     if "t3_z7" in DOM else f"TOTAL3 {DL.arrow(DOM['t3_tr'])} · 알트 비중 {DL.arrow(DOM['alt_tr'])}")
    with badge_cols[i_c]:
        st.markdown(badge(nm, f"${last:,.{2 if sym == 'ETHUSDT' else 1}f}", word, strength, qline, change, flow, next_t, mood), unsafe_allow_html=True)
    c_.metric(f"전환 가격 ({next_t:%H:%M} 마감 기준)", lines[0].split(" → ")[0] if lines else "변화 없음",
              " / ".join(l.split(" → ")[1] for l in lines[:1]) if lines else "어느 가격에 마감해도 유지", delta_color="off")
    tb = []; seg = []
    for r in reversed(T):                                   # 위 가격대 → 아래, 같은 결과가 이어지면 한 줄로
        tgt = qty(k * (wt * r["pos"] * fx(r["pos"]) * wfx(r) + sp + tp_next), last, stp)
        act = todo(notional, k * (wt * r["pos"] * fx(r["pos"]) * wfx(r) + sp + tp_next), last, stp, nm)
        if seg and seg[-1][2] == tgt and seg[-1][3] == act: seg[-1][0] = r["lo"]
        else: seg.append([r["lo"], r["hi"], tgt, act])
    for lo_, hi_, tgt, act in seg:
        lo = "" if not np.isfinite(lo_) else f"{lo_:,.1f}"; hi = "" if not np.isfinite(hi_) else f"{hi_:,.1f}"
        rng = f"{lo} 위" if not hi else (f"{hi} 아래" if not lo else f"{lo} ~ {hi}"); here = lo_ <= last < hi_
        tb.append({"다음 4H 마감가": rng + ("  ← 지금" if here else ""), "그때 포지션": (f"{'숏' if tgt < 0 else '롱'} {abs(tgt):.{dq}f} {nm} (약 {abs(tgt)*last:,.0f} USDT)" if tgt else "없음"),
                   "할 일": act})
    st.dataframe(pd.DataFrame(tb), hide_index=True, width="stretch")
    sr = []
    if np.isfinite(v["res"]): sr.append(f"저항 {v['res']:,.1f}")
    if np.isfinite(v["sup"]): sr.append(f"지지 {v['sup']:,.1f}")
    st.caption(f"{RL.verdict(v['dir'])} · 점수 {v['dir']:+.2f} (추세 {v['tsmom']:+.2f} / 호흡 {v['breath']:+.2f}) · " + " · ".join(sr)
               + f" · {close_t:%m-%d %H:%M} 마감 기준" + (" · 지정가(30분 미체결 시 시장가)" if MAKER else ""))
    if v.get("er"):
        e_ = v["er"]
        st.caption(f"결 · {e_['mode']} (효율비 {e_['er']:.2f}) — " + ("수익이 나는 구간" if e_["mode"] == "직진" else "본전 구간, 그대로 따르기"))
    if v.get("regime"):
        st.caption(f"상황 · {v['regime']} — 방법은 고정, 크기만 변동성에 맞춤")
    if tr_:
        if tr_.get("active", 0) > 0:
            ttxt = f"발생 ({pd.Timestamp(tr_['last_event']).tz_convert('Asia/Seoul'):%m-%d %H:%M} 마감) → 롱 유지 남은 {tr_['active_left']}봉"
        elif tr_.get("armed_level"):
            ttxt = f"대기 — {tr_['armed_level']:,.1f} 위로 {tr_['armed_left']}봉 안에 마감하면 롱" + (f" (+{k*tr_['cap']:,.0f} USDT)" if TRAP else "")
        else:
            ttxt = f"없음 · 감시 저점 {tr_['last_low']:,.1f}" if tr_.get("last_low") else "없음"
        st.caption(f"베어트랩 · {ttxt}" + ("" if TRAP else " (설정에서 켜야 반영)"))
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

with st.expander("△ 하루 1% 기하 복리 — 배율별 성장 지도 (11-06)", expanded=False):
    st.markdown("""
**목표**: 수익일·손실일이 섞인 실제 결과를 복리로 쌓아 하루 평균 기하 성장 1% = 1년 37.8배.
**조건**: 로그 성장의 최대치는 샤프²/2 → 연 샤프 **2.70 이상**(켈리 배율, 낙폭 매우 큼) 또는 3.11(켈리 ½)이 몇 년간 유지돼야 함.

| 배율 | 2022~2026 하루 / 1년 | 최근 2년 하루 / 1년 | 최대 낙폭 |
|---|---|---|---|
| 1 | 0.05% / 1.2배 | 0.06% / 1.2배 | −12% |
| 1.5 (기본) | 0.07% / 1.27배 | 0.08% / 1.36배 | −17% |
| 2 | 0.09% / 1.37배 | 0.11% / 1.49배 | −23% |
| 3 | 0.12% / 1.55배 | 0.16% / 1.77배 | −32% |
| 4 | 0.15% / 1.73배 | 0.20% / 2.07배 | −41% |
| 5 | 0.17% / 1.89배 | 0.24% / 2.37배 | −49% |
| 10~15 (한계) | 0.22% | 0.37% | −80~−90% |
| 20 이상 | 파산 (2019-02·2022-11 FTX 같은 4시간 급락 한 번) | | |

- 지금 앱 그대로, 비용·롱 펀딩·봉 안 최저가 청산까지 반영한 실제 복리 (BTC·ETH 코인베이스 4H).
- 가장 좋았던 2016~2018년(샤프 2.87)도 실제 최고는 하루 0.77% (배율 10, 낙폭 −66%) — 이론(1.14%)보다 낮은 건 급락 꼬리 때문.
- 지금 시대 샤프는 1.2~1.6 → 현실적 최대 하루 0.1~0.25% (1년 1.4~2.4배). 베어트랩을 더해도 −0.01~+0.09%p.
- 지난 1년 성적으로 배율을 정하는 방식(켈리 ½, 최대 10)도 2022~ 하루 0.12%, 최근 2년 0.21% (낙폭 −54%).
- 1%로 가려면 서로 독립적인 수익원(샤프 1짜리 7~8개)이 필요한데, BTC·ETH(상관 0.9)의 추세 추종은 사실상 하나.
- 표의 배율은 지나고 나서 본 값 — 앞으로도 같다는 보장 없음. 4~5는 자본이 절반까지 줄어드는 구간을 견딜 수 있을 때만.
""")

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

with st.expander("△ 코인별 독립 분석 — BTC × 테더 도미넌스 · ETH × TOTAL2·3 · 강도별 흐름 (11-05)", expanded=False):
    if DOM:
        import dominance_live as DL
        st.caption(f"지금 ({DOM['date']}) · USDT.D {DOM['usdt_d']:.2f}% {DL.arrow(DOM['usdt_d_tr'])} · USDT 발행량 30일 {DOM['usdt_30']:+.1%} · BTC.D {DOM['btc_d']:.1f}% {DL.arrow(DOM['btc_d_tr'])} · ETH.D {DOM['eth_d']:.1f}%"
                   f" · TOTAL3(스테이블 제외) {DL.arrow(DOM['t3_tr'])} 30일 {DOM['t3_30']:+.0%} · 알트 비중 {DL.arrow(DOM['alt_tr'])} · ETH/알트 {DL.arrow(DOM['eva_tr'])} (화살표 = 20·50·100일 방향)")
    st.markdown("""
**같은 규칙, 각자의 데이터** — BTC와 ETH는 각자 가격으로 신호·크기·전환 가격·흐름 정렬을 따로 계산합니다.
- ETH 변동이 BTC의 1.33배 → 같은 금액이면 ETH 크기가 자동으로 작아짐. 일별 상관 0.71 (2016년 0.26 → 2026년 0.91로 점점 같이 움직임).
- 같은 시각 앱 방향이 같았던 시간 74%, 반대 25%.
- 코인마다 설정을 따로 골라 보니(18가지 중 앞 절반 1등) 뒤 절반에서 BTC 18위(0.62 vs 지금 0.95), ETH 13위(0.95 vs 1.11) → 따로 맞추면 과거에만 맞음. **규칙은 공유, 데이터는 독립**이 가장 좋았음.

**BTC × 테더 도미넌스(USDT.D, 2020~)**
- USDT.D 방향은 BTC 자기 추세와 79% 같음(상관 0.77) — BTC 가격이 분모에 들어 있어 거의 같은 정보.
- 'USDT.D가 30일 오르면(공포) 다음 30일 BTC가 더 오름' 경향(t=3.0)은 앞 절반에만 있고 뒤 절반엔 없음. USDT 발행 속도는 관계없음.
- 규칙 4가지(USDT.D 반대로 매매, 발행 속도로 매매, 어긋나면 ½ 두 가지) 모두 기각.

**ETH × 알트 시장(TOTAL2·TOTAL3)**
- TOTAL3 방향도 ETH 자기 추세와 79% 같음, 앞서거나 뒤서지 않음(같은 날 함께 움직임).
- 규칙 7가지(TOTAL3·TOTAL2 추세대로, 어긋나면 ½, 알트 비중, ETH/알트 강약, 스테이블 제외) 모두 기각 — 가장 나은 'TOTAL2와 어긋나면 ½'도 +0.08로 기준 미달.
- ETH는 최근 1년 앱 성적이 마이너스(샤프 −0.43, 코인베이스 기준)였고 알트 지표로 걸러도 나아지지 않음.

**강도별 흐름 (7일 변화 ÷ 지난 1년 보통 크기 = σ, 5단계)**

| 테더 도미 7일 | 시간 | 그 주 BTC / ETH | 다음 30일 BTC / ETH (오른 비율) | 다음 주 흔들림 |
|---|---|---|---|---|
| 강한 하락 (≤ −1.5σ) | 5% | +17% / +20% | +7.5% (68%) / +11.3% (63%) | 보통 |
| 하락 | 22% | +8% / +11% | +4.0% / +6.0% | 보통 |
| 보합 | 47% | +1% / +1% | +4.4% / +5.8% | 보통 |
| 상승 | 20% | −6% / −8% | +4.5% / +3.3% | 보통 |
| 강한 상승 (≥ +1.5σ) | 6% | −14% / −17% | −0.4% (47%) / −2.1% (39%) | **1.4배** |

| TOTAL3 7일 (2021-05~) | 그 주 ETH / BTC | 다음 30일 ETH / BTC (오른 비율) |
|---|---|---|
| 강한 하락 | −18% / −13% | −3.7% (38%) / −0.3% (52%) |
| 보합 | 0% / 0% | +2.5% / +2.2% |
| 강한 상승 | +16% / +12% | +2.0% / +2.2% |

- **같은 주**: 테더 도미와 코인은 거울처럼 반대, TOTAL3와 코인은 같은 방향. ETH가 BTC보다 1.15~1.4배 크게 움직임 (TOTAL3 1% → ETH 0.93% · BTC 0.67%).
- **다음 주**: 강도 5단계 모두 평소와 차이 없음 → 예측 안 됨.
- **다음 30일**: 센 주간의 방향이 조금 더 이어짐 (테더 도미 강한 하락 → 더 오름, 강한 상승 → 덜 오름, 알트 강한 하락 → 약함, 앞·뒤 절반 모두). 단 이것을 규칙(센 주간 뒤 30일 반대 포지션 ½)으로 만들면 BTC +0.01·ETH +0.05로 개선 없음 — 앱의 추세가 이미 같은 성질을 잡고 있음.
- **관찰 중**: 'TOTAL3 강한 주간 뒤 30일 반대 ½'이 BTC에서 +0.25 (우연 범위 밖)였지만 최근 1년은 나빠졌고 원래 대상인 ETH에선 −0.07 → 채택 안 함.
- **둘이 따로 놀 때**: 테더 도미만 오르고 TOTAL3는 버티는 주 → 다음 주 BTC −1.5%·ETH −2.2% (앞·뒤 절반 같은 방향), 단 겹치지 않는 표본 13주라 참고만.

→ 두 지표는 **카드 아래 참고 줄**로만 표시 (매매·크기엔 안 씀).
""")

with st.expander("△ 거시 → 세부 → 경우의 수 (BTC 11년 · ETH 10년, 11-05)", expanded=False):
    for _n, _m in MAC.items():
        if _m:
            arw = lambda x: "▲" if x > 0 else ("▼" if x < 0 else "―")
            st.caption(f"{_n} 거시 {_m['label']} (합의 {_m['mval']:+.2f}) · " + " · ".join(f"{k} {arw(v)}" for k, v in _m["parts"].items())
                       + f" · 200일선 {_m['s200']:,.0f} (지금 {_m['dist200']:+.0%}) · 1년 고점 대비 {_m['dd1y']:+.0%} · 기준 {_m['date']} 마감")
    st.markdown("""
**1. 거시 방향은 판단할 수 있지만 늘 늦다 (불확정성)**
- 6가지 거시 정의를 BTC 11년에 시험: 거시 상승장일 때 90일 뒤 평균 +14~21%, 하락장일 때 −2~+5%. 차이는 있지만 **방향 적중 57~63%로 '그냥 늘 롱'(59%)과 비슷**하고, 뒤 절반에선 차이가 줄어듦.
- 사이클 고점(2017-12·2021-11·2025-10)·바닥(2018-12·2022-11)을 알아채는 데: 가장 빠른 200일선 위/아래도 고점 뒤 평균 32일(이미 −15~−65%), 대신 1년에 6.8번 뒤집힘. 헛신호를 줄인 200일선 기울기는 연 1.6번이지만 평균 112일 늦음. **빨리 알면 헛신호, 확실하면 늦음** — 둘 다 가질 수 없음.

**2. 세부(4시간)는 거시와 상관없이 양쪽에서 번다**
- 지금 앱을 거시 상태 × 롱·숏 6칸으로 쪼개면 BTC·ETH 모두 6칸 전부 플러스 (상승장 속 숏 샤프 +0.8~1.1, 하락장 속 롱 +0.7~2.1).
- 그래서 '거시와 같은 방향만', '반대는 절반', '같은 쪽 1.5배', '중립이면 절반' × 거시 정의 3가지 = 12가지 모두 기각 (가장 좋은 것도 +0.07, 우연 범위).
- 4시간 추세는 이미 거시를 품고 있음: 큰 흐름이 이어지면 같은 방향으로 오래 머물고, 큰 흐름 안의 반대 파도(상승장 속 급락, 하락장 속 반등)도 따로 먹음.

**3. 경우의 수 — 다음 1주는 5가지 중 하나 (변동성 기준, 11년 빈도)**

| 다음 1주 | 빈도 | 그때 지금 앱의 1주 손익 (배율 1) |
|---|---|---|
| 곧장 상승 | 약 22% | **+2.5%** |
| 눌림 후 상승 | 약 2% | +1.3% |
| 횡보 | **약 58%** | −0.4% |
| 반등 후 하락 | 약 2% | +0.6% |
| 곧장 하락 | 약 16% | +0.8% |

- 가장 흔한 횡보에서 조금 잃고, 드문 직진에서 크게 범 → 1주 기대값 약 +0.5%. **예측이 아니라 경우의 수에 대한 대응**으로 돈이 남.
- 거시·주봉·일봉·4시간·직진도·변동성·200일선 거리 7가지로 '어느 경우가 올지' 학습 → 뒤 5.5년에서 평소 비율보다 0.2~0.4% 나아짐(사실상 없음). 오를지 내릴지 구별 0.51~0.52, 횡보일지 구별 0.59.

**4. 예측이 되는 것과 안 되는 것**
- 안 됨: 다음 4시간·1주 방향 (0.49~0.52), 가짜 전환 구별 (0.54).
- 됨: 다음 하루 움직임 크기 (0.75 — 그래서 앱이 변동성에 맞춰 크기를 줄이고 늘림), 4주 안 주봉 흐름 전환 가능성 (0.66 — 카드의 '상태' 줄).

**5. 11년으로 다시 보니**
- 추세 추종의 힘이 시대가 갈수록 약해짐: BTC 4시간 기본 샤프 2015~2021년 2.31 → 2021~2026년 0.63. 시장이 성숙하면서 쉬운 추세가 줄어듦.
- 지난번 채택한 흐름 정렬 역가중은 처음 보는 2015~2021년에서 2.31 → 2.02로 오히려 나빴고, 최근 5년에서만 0.63 → 0.96으로 좋았음. 11년 전체 샤프는 비슷(1.95 vs 1.89), 최대 낙폭은 −21% → −14%로 줄어듦 → 기본으로 두되 '수익 개선'은 확실하지 않음.
""")

with st.expander("△ 우주·물리 렌즈 17개 — 평행우주 검증 (11-05)", expanded=False):
    try:
        import cosmos_live as CL
        _ph, _pn = CL.moon(); st.caption(f"오늘 · 달 {_pn} (위상 {_ph:.2f}) · 2024-04-20 반감기 뒤 {CL.halving_days()}일 — 둘 다 판단에 쓰지 않음 (아래 ⑤·⑦)")
    except Exception:
        pass
    st.markdown("""
**새 검증법 '평행우주'** — 실제 BTC 성적을 수천 개의 가짜 우주와 비교: ① 포지션을 시간축에서 통째로 민 우주 1000개(롱 편향·매매 횟수는 그대로, 타이밍만 무작위) ② 매 봉 움직임 크기는 그대로 두고 방향만 무작위로 뒤집은 '거울 우주' 200개 ③ 시간을 거꾸로 돌린 우주 ④ ETH 쌍둥이 ⑤ 마지막 1년 ⑥ 설정값을 바꾼 이웃 우주. 실제가 가짜들의 상위 5% 밖이면 '우연'.

| 렌즈 | 관점 | BTC 5년 | 결과 |
|---|---|---|---|
| 현재 앱 (추세+호흡) | 기준 | 샤프 +0.87 · 가짜 우주 상위 1% | **통과** — 새 방식으로도 우연 아님 확인 |
| 오를 때 롱·내릴 때 숏 (5일 효율비 ≥ 0.3일 때) | 양자: 입자 | +1.15 · ETH +1.02 | **통과** — 단 뒤 절반 +0.29로 약해짐, 앱에 더하면 개선 +0.09로 기준(+0.10) 미달 |
| 횡보 때 박스 하단 롱·상단 숏 | 양자: 파동 | −0.21 · ETH −0.41 | 탈락 — 이걸 붙이면 위 규칙이 1.15 → 0.58로 나빠짐 |
| 수축 뒤 첫 이탈 따라가기 | 양자: 측정 붕괴 | −0.15 | 탈락 |
| 변동의 양으로 잰 시간(고유시간) 추세 | 상대성 | +0.31 | 탈락 (우연 범위) |
| 광속 근접 시 크기 축소 | 상대성: 로렌츠 인자 | −0.13 | 탈락 |
| 닮은 과거 궤적 다수결 | 혼돈: 로렌츠 아날로그 | −0.96 | 탈락 |
| 칼만 입자 추적 | 물리: 상태추정 | +0.16 | 탈락 |
| 티핑 포인트 조기경보(자기상관·분산 상승) | 열역학·생태학 | −0.03 · 전환 구별력 0.45~0.54 | 탈락 |
| BTC·ETH 상관 붕괴 시 ½ | 양자: 얽힘 | +0.13 | 탈락 (보정 후 우연 범위, 최근 1년 나빠짐) |
| 렌즈 합의만큼 키움 | 노자: 무위 | −0.02 | 탈락 |
| 최근 1년 잘 맞은 쪽에 가중 | 헤라클레이토스 | +0.51 | 탈락 (앱 단독보다 못함) |
| 횡보면 ½ | 상황 적용 | 5년 개선 +0.22, 앱 1.36→1.53 | 거울 우주(p=0.11)·이웃 설정(67%) 미통과, 11년 확인 때 처음 보는 6년(2015~2021)에서 나빠짐 → 탈락·옵션 삭제 |
| 달: 신월 롱·보름 숏 (11년 일봉) | 우주 | +0.18 (뒤 절반 −0.16) | 탈락 |
| 지구 자기장 폭풍 뒤 6일 | 우주 | 오히려 폭풍 뒤가 더 오름 | 탈락 |
| 태양 활동 / 계절(월) | 우주 | +0.36 / +0.19 (뒤 절반 무너짐) | 탈락 |
| 황금비 0.382·0.5·0.618·0.786 | 자연의 수 | 0.45·0.55·0.70과 똑같음 | 특별하지 않음 |

- **시간의 화살**: 가격을 거꾸로 돌리면 같은 앱 방법이 손실(BTC +1.13 → −0.59). 그런데 '선형 기억'(자기상관)은 0.006으로 정·역방향이 같음 → 앱이 먹는 것은 반복 패턴이 아니라 **되돌릴 수 없는 흐름**(사람들이 늦게 알아채고 따라붙는 과정).
- **되돌림 진입**: 어느 깊이(0.3~0.8)에서 추세 방향으로 들어가도 무작위보다 스윙 크기의 +5% 안팎 유리 — 추세가 이어지는 성질 때문이고 황금비 자리만의 힘은 없음.
- **반감기**: 세 번 모두 반감기 뒤 525~546일에 고점, 이후 약 1년 −53~−84%. 지금 929일째. 표본 3개 → 통계 검증 불가, 참고만.
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
| 11년 재확인 (코인베이스 2015-07~, 지정가·펀딩) | — | — | 4H 기본 BTC 샤프 2.31 (2015~2021) → 0.63 (2021~) · 흐름 정렬 역가중 2.02 → 0.96 · 추세의 힘이 시대가 갈수록 약해짐, 역가중은 최근 시대에서만 보탬 |
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
if not errs:
    render_diag(expanded=False)
