"""목표 계획 (11-17): 지금 앱 기본(흐름+추적 부분익절)의 비트겟 2022-08~ 4시간 손익을 한 달 덩어리로 다시 뽑아(블록 부트스트랩) 목표 시점까지.
세계 3가지 — 과거 성적 그대로 / 평균 수익이 절반만 / 우위 없음(평균 0) — 에서 배율별 중앙 자산·목표 확률·도중 낙폭(월말 기준).
한 봉에 계좌의 95% 이상 잃는 움직임이 있으면 그 달 자산 0(청산)으로 처리. state/plan_base.json = 배율 1 기준 봉별 손익 g와 봉 안 최악 w."""
from __future__ import annotations
import json, numpy as np

WORLDS = (("그대로", 0.0), ("절반", 0.5), ("없음", 1.0)); B = 180


def table(path, eq0: float, monthly: float, months: int, mults, goal: float, n: int = 3000, seed: int = 7) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        J = json.load(fh)
    g, w = np.array(J["g"]), np.array(J["w"]); L = len(g); months = int(max(months, 1))
    st = np.random.default_rng(seed).integers(0, L - B, (n, months)); rows = []
    for m in mults:
        row = {"배율": m}
        for nm, hair in WORLDS:
            gh = g - hair * g.mean(); cs = np.concatenate([[0.0], np.cumsum(np.log(np.maximum(1 + m * gh, 1e-12)))]); F = np.exp(cs[B:] - cs[:-B])
            cd = np.concatenate([[0], np.cumsum(m * w <= -0.95)]); F[(cd[B:] - cd[:-B]) > 0] = 0.0
            eq = np.full(n, float(eq0)); peak = eq.copy(); mdd = np.zeros(n)
            for k in range(months):
                eq = eq * F[st[:, k]]; mdd = np.minimum(mdd, eq / np.maximum(peak, 1e-9) - 1); eq = eq + monthly; peak = np.maximum(peak, eq)
            row[nm] = {"med": float(np.median(eq)), "p10": float(np.quantile(eq, 0.1)), "p": float((eq >= goal).mean()), "dd": float(np.median(mdd)), "dd10": float(np.quantile(mdd, 0.1))}
        rows.append(row)
    return rows
