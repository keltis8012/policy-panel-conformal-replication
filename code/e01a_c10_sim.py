#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e01a_c10_sim.py —— C10 可分离性关卡（定理 2 加性分解合法化）
版本：v1.0 ｜ 2026-08-28 ｜ 判据 C10-v0.4（预注册冻结：C10_prereg.md，红线 #6/#10/#11）
run_status: exploratory（判据首次应用）

修正设计（第六轮裁定）：配对差分 Δε(s) = ε_t(s|高|β|组) − ε_t(s|低|β|组)（同实现配对，
主效应消去）；D(s_beta) = max_s Δε − min_s Δε（组间差的 s 漂移 = 交互载体）；
δ_C10 = 种子噪声标尺（s_beta=0 双种子 ε_t 差，max(标尺, 0.02)）；三预案路由。
联合角点锚：(s_scale=4, s_beta=0.4) 整体缺口 vs ε_t(4) + ε_g(0.4) 加性预测。
扫描：s_scale {1,2,4} × s_beta {0,0.2,0.4,0.8} = 12 格，B=500
用法：python e01a_c10_sim.py [--B 500]
输出：04_output/experiments/E01a/c10_<ts>/ {c10_grid.csv, delta_eps.csv, adjudication.json}
"""
import argparse, json, os, sys, csv, datetime, zlib
import numpy as np

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "03_program", "experiments", "E01_three_arm_coverage"))
from e01a_sim import rolling_pseudo_residuals, fit_arms, summarize
from e01a_f6_sim import gen_panel_f6

S_SCALE = [2.0, 3.0, 4.0]   # 1.0 档无块内定义（sig 全 1.0 → bf 恒 False → 条件覆盖 NaN），排除
S_BETA = [0.0, 0.2, 0.4, 0.8]
ARMS = ("naive", "country_block", "time_block")
EPS_G_F6_FIXED = 0.053        # F6 相对增量（f6_20260828_105508：0.190−0.137，s_scale=2.0 档），角点锚用


def run_cell(s_scale, s_beta, N, T, alpha, B, seed, arm_sel="naive"):
    """一格：B 次重复，整体/高|β|组/低|β|组块内覆盖（目标年入块实现）。"""
    rng = np.random.default_rng(seed)
    K = 5
    params = {"s_mu": 0.5, "s_lam": 0.5, "lam_ar": 0.0, "phi": 0.4, "s_scale": s_scale,
              "rho": 0.5, "s_beta": s_beta}
    agg = {k: [] for k in ("all", "hi", "lo")}
    for b in range(B):
        y, bf, beta = gen_panel_f6(N, T, **params, rng=rng, block_anchor="tail")
        y_train = y[:, :T]; y_true = y[:, T]
        q, center, rr, _ = fit_arms(y_train, alpha, K)
        if not bf[T]:
            continue
        qa = q[arm_sel]
        cov_all = np.mean(np.abs(y_true - center) <= qa)
        ab = np.abs(beta)
        hi = ab >= np.quantile(ab, 0.75); lo = ab <= np.quantile(ab, 0.25)
        agg["all"].append(cov_all)
        agg["hi"].append(np.mean(np.abs(y_true[hi] - center[hi]) <= qa))
        agg["lo"].append(np.mean(np.abs(y_true[lo] - center[lo]) <= qa))
    s_all = summarize(agg["all"]); s_hi = summarize(agg["hi"]); s_lo = summarize(agg["lo"])
    return {"all": s_all, "hi": s_hi, "lo": s_lo, "n_in": len(agg["all"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    args = ap.parse_args()
    N, T, alpha = 38, 24, 0.10
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "04_output", "experiments", "E01a", f"c10_{ts}")
    os.makedirs(outdir, exist_ok=True)

    # ---------- 12 格 ----------
    grid = {}
    for s in S_SCALE:
        for sb in S_BETA:
            cname = f"C10_s={s}_b={sb}"
            seed = 20260828 + (zlib.crc32(cname.encode()) % 10 ** 6)
            res = run_cell(s, sb, N, T, alpha, args.B, seed)
            grid[(s, sb)] = res
            print(f"[C10] s={s} s_beta={sb}: all_in={res['all']['median']:.3f} "
                  f"hi={res['hi']['median']:.3f} lo={res['lo']['median']:.3f} n={res['n_in']}")

    # ---------- Δε(s) 与 D 漂移 ----------
    delta = {}   # (s, sb) -> 高组缺口 − 低组缺口（缺口 = 1−α − 覆盖）
    for s in S_SCALE:
        for sb in S_BETA:
            r = grid[(s, sb)]
            eps_hi = (1 - alpha) - r["hi"]["median"]
            eps_lo = (1 - alpha) - r["lo"]["median"]
            delta[(s, sb)] = eps_hi - eps_lo
    D = {}
    for sb in S_BETA:
        vals = [delta[(s, sb)] for s in S_SCALE]
        D[sb] = max(vals) - min(vals)

    # ---------- δ_C10 种子噪声标尺（s_beta=0 双种子） ----------
    noise = []
    for s in S_SCALE:
        cname = f"C10_s={s}_b=0"
        seed_a = 20260828 + (zlib.crc32((cname + "_a").encode()) % 10 ** 6)
        seed_b = 20260828 + (zlib.crc32((cname + "_b").encode()) % 10 ** 6)
        ea = (1 - alpha) - run_cell(s, 0.0, N, T, alpha, args.B, seed_a)["all"]["median"]
        eb = (1 - alpha) - run_cell(s, 0.0, N, T, alpha, args.B, seed_b)["all"]["median"]
        noise.append(abs(ea - eb))
    delta_thr = max(max(noise), 0.02)

    # ---------- 判据路由 ----------
    over = {sb: round(D[sb], 4) for sb in S_BETA if sb > 0 and D[sb] > delta_thr}
    n_over = len(over)
    if n_over == 0:
        route = "PASS_ADDITIVE"
    elif n_over == 1:
        route = "EDGE_UPPERBOUND"
    else:
        route = "REJECT_1D"
    # 联合角点锚：(s=4, b=0.4) 整体缺口 vs ε_t(4) + ε_g(0.4)，ε_g 用 F6 固定增量（非本格自算——防恒等）
    s4 = 4.0; sb4 = 0.4
    eps_join = (1 - alpha) - grid[(s4, sb4)]["all"]["median"]
    eps_t4 = (1 - alpha) - grid[(s4, 0.0)]["all"]["median"]
    eps_g_rel = EPS_G_F6_FIXED
    add_pred = eps_t4 + eps_g_rel
    corner = {"eps_join": round(eps_join, 4), "eps_t4": round(eps_t4, 4),
              "eps_g_rel(F6 固定)": round(eps_g_rel, 4), "add_pred": round(add_pred, 4),
              "check": round(abs(eps_join - add_pred), 4),
              "within_delta": abs(eps_join - add_pred) <= delta_thr}

    # ---------- 落盘 ----------
    with open(os.path.join(outdir, "c10_grid.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["s_scale", "s_beta", "all_in_med", "hi_in_med", "lo_in_med", "n_in"])
        for (s, sb), r in sorted(grid.items()):
            w.writerow([s, sb, round(r["all"]["median"], 4), round(r["hi"]["median"], 4),
                        round(r["lo"]["median"], 4), r["n_in"]])
    with open(os.path.join(outdir, "delta_eps.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["s_scale", "s_beta", "delta_eps"])
        for (s, sb), d in sorted(delta.items()):
            w.writerow([s, sb, round(d, 4)])
    adj = {"run_status": "exploratory", "protocol_version": "v1.0",
           "criterion_version": "C10-v0.4（C10_prereg.md，跑前冻结）",
           "N": N, "T": T, "alpha": alpha, "B": args.B,
           "delta_eps_by_sbeta": {str(sb): {str(s): round(delta[(s, sb)], 4) for s in S_SCALE}
                                  for sb in S_BETA},
           "D_drift": {str(sb): round(D[sb], 4) for sb in S_BETA},
           "seed_noise_scale": [round(x, 4) for x in noise],
           "delta_thr": round(delta_thr, 4),
           "over_threshold_cells": over,
           "route": route,
           "corner_anchor": corner,
           "verdict": {"PASS_ADDITIVE": "C10 通过：组间差无 s 依赖 → 二维加性主陈述（现计划）",
                       "EDGE_UPPERBOUND": "C10 边缘：1 档超 δ → 主陈述加交互上界 ε_tg_max",
                       "REJECT_1D": "C10 拒绝：≥2 档超 δ → 定理 2 降级一维主定理 + ε_g 实证修正"}[route],
           "note": "配对差分（同实现高/低|β|组）使 ε_t/ε_g 主效应消去、交互进入 Δε 的 s 漂移；δ 标尺 = s_beta=0 双种子 ε_t 差（无交互设定标定检验噪声）。"}
    json.dump(adj, open(os.path.join(outdir, "adjudication.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[C10] 种子噪声标尺={[round(x,4) for x in noise]} → δ_C10={delta_thr:.4f}")
    print(f"[C10] D 漂移: { {str(sb): round(D[sb],4) for sb in S_BETA} }")
    print(f"[C10] 超阈档: {over} → 路由: {route}")
    print(f"[C10] 联合角点锚: {corner}")
    print(f"[out] {outdir}")


if __name__ == "__main__":
    main()
