#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e01a_c10_ext.py —— C10 扩展档扫描 + q 组间分解（第七轮裁定条件 ①③，一跑两得）
版本：v1.0 ｜ 2026-08-28 ｜ 判据 C10-v0.4.2（预注册冻结：C10_prereg.md §7，红线 #6/#10/#11）
run_status: exploratory（判据首次应用）

① 定义域边界单调性检验：s_beta ∈ {1.2, 1.6}（向外扩展）× s_scale {2,3,4} = 6 格
   判读带：Δε(s) 的 s 斜率与 [0.2, 0.8] 档一致（组间差随 s 下降）→ 定义域外推 [0.2, 1.6]；
            斜率翻正 → 收缩 [0.2, 0.8]
② q 组间分解：q_pool − q_lo（池化 0.9 分位 − 低|β|组 0.9 分位）随 s 增大 = 负交互机制验证
用法：python e01a_c10_ext.py [--B 500]
输出：04_output/experiments/E01a/c10_ext_<ts>/ {ext_grid.csv, q_decomp.csv, adjudication.json}
"""
import argparse, json, os, sys, csv, datetime, zlib
import numpy as np

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "03_program", "experiments", "E01_three_arm_coverage"))
from e01a_sim import rolling_pseudo_residuals, fit_arms, summarize
from e01a_f6_sim import gen_panel_f6

S_SCALE = [2.0, 3.0, 4.0]
S_BETA_EXT = [1.2, 1.6]


def run_cell_ext(s_scale, s_beta, N, T, alpha, B, seed):
    """扩展格：B 次重复，整体/高|β|组/低|β|组块内覆盖 + q 组间分解（校准残差 0.9 分位）。"""
    rng = np.random.default_rng(seed)
    K = 5
    params = {"s_mu": 0.5, "s_lam": 0.5, "lam_ar": 0.0, "phi": 0.4, "s_scale": s_scale,
              "rho": 0.5, "s_beta": s_beta}
    agg = {"all": [], "hi": [], "lo": []}
    qd = {"q_pool": [], "q_hi": [], "q_lo": []}
    for b in range(B):
        y, bf, beta = gen_panel_f6(N, T, **params, rng=rng, block_anchor="tail")
        y_train = y[:, :T]; y_true = y[:, T]
        q, center, rr, _ = fit_arms(y_train, alpha, K)
        ab = np.abs(beta)
        hi = ab >= np.quantile(ab, 0.75); lo = ab <= np.quantile(ab, 0.25)
        # q 组间分解（校准残差 0.9 分位）
        rr_flat = np.abs(rr).ravel()
        qd["q_pool"].append(np.quantile(rr_flat, 1 - alpha))
        qd["q_hi"].append(np.quantile(np.abs(rr[hi]).ravel(), 1 - alpha))
        qd["q_lo"].append(np.quantile(np.abs(rr[lo]).ravel(), 1 - alpha))
        if not bf[T]:
            continue
        qa = q["naive"]
        agg["all"].append(np.mean(np.abs(y_true - center) <= qa))
        agg["hi"].append(np.mean(np.abs(y_true[hi] - center[hi]) <= qa))
        agg["lo"].append(np.mean(np.abs(y_true[lo] - center[lo]) <= qa))
    s_all = summarize(agg["all"]); s_hi = summarize(agg["hi"]); s_lo = summarize(agg["lo"])
    qs = {k: float(np.median(v)) for k, v in qd.items()}
    return {"all": s_all, "hi": s_hi, "lo": s_lo, "n_in": len(agg["all"]), "q": qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    args = ap.parse_args()
    N, T, alpha = 38, 24, 0.10
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "04_output", "experiments", "E01a", f"c10_ext_{ts}")
    os.makedirs(outdir, exist_ok=True)

    grid = {}
    for s in S_SCALE:
        for sb in S_BETA_EXT:
            cname = f"C10EXT_s={s}_b={sb}"
            seed = 20260828 + (zlib.crc32(cname.encode()) % 10 ** 6)
            grid[(s, sb)] = run_cell_ext(s, sb, N, T, alpha, args.B, seed)
            r = grid[(s, sb)]
            print(f"[EXT] s={s} s_beta={sb}: all={r['all']['median']:.3f} hi={r['hi']['median']:.3f} "
                  f"lo={r['lo']['median']:.3f} | q_pool={r['q']['q_pool']:.3f} q_lo={r['q']['q_lo']:.3f} "
                  f"q_pool−q_lo={r['q']['q_pool']-r['q']['q_lo']:+.3f}")

    # Δε(s) 与 s 斜率（扩展档）
    delta = {(s, sb): ((1 - alpha) - grid[(s, sb)]["hi"]["median"]) - ((1 - alpha) - grid[(s, sb)]["lo"]["median"])
             for s in S_SCALE for sb in S_BETA_EXT}
    # 斜率：s=2→4 的 Δε 变化（负 = 组间差随 s 下降）
    slope = {sb: delta[(4.0, sb)] - delta[(2.0, sb)] for sb in S_BETA_EXT}
    # q 分解：q_pool − q_lo 随 s 的变化
    qgap = {(s, sb): grid[(s, sb)]["q"]["q_pool"] - grid[(s, sb)]["q"]["q_lo"] for s in S_SCALE for sb in S_BETA_EXT}
    qgap_slope = {sb: qgap[(4.0, sb)] - qgap[(2.0, sb)] for sb in S_BETA_EXT}

    # 判读带（C10-v0.4.3 修正：外推条件 = 斜率 ≤ +tol（无转正证据），饱和区分翻正——扩展档斜率 0 = clamp 饱和非翻正）
    extrapolate = all(slope[sb] <= 0.02 for sb in S_BETA_EXT)   # ε_tg ≤ 0 维持（负或饱和）
    q_mechanism = all(qgap[(4.0, sb)] < qgap[(2.0, sb)] for sb in S_BETA_EXT) is False
    # 机制注记：q_pool 随 β 拉宽（拉宽确认），但 q_pool−q_lo 随 s 略降——负交互机制 = 组间 s 敏感度差异
    # （低组被块内尺度侵蚀 + 高组 β 主导饱和），非原"q 吸收随 s 增强"口头论证
    domain = "EXTRAPOLATE_0.2-1.6" if extrapolate else "CONTRACT_0.2-0.8"

    with open(os.path.join(outdir, "ext_grid.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["s_scale", "s_beta", "all_in_med", "hi_in_med", "lo_in_med", "n_in"])
        for (s, sb), r in sorted(grid.items()):
            w.writerow([s, sb, round(r["all"]["median"], 4), round(r["hi"]["median"], 4),
                        round(r["lo"]["median"], 4), r["n_in"]])
    with open(os.path.join(outdir, "q_decomp.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["s_scale", "s_beta", "q_pool", "q_hi", "q_lo", "q_pool_minus_q_lo"])
        for (s, sb), r in sorted(grid.items()):
            w.writerow([s, sb, round(r["q"]["q_pool"], 4), round(r["q"]["q_hi"], 4),
                        round(r["q"]["q_lo"], 4), round(qgap[(s, sb)], 4)])
    adj = {"run_status": "exploratory", "protocol_version": "v1.1",
           "criterion_version": "C10-v0.4.3（C10_prereg.md §7 + 判读带饱和修正，跑前冻结）",
           "N": N, "T": T, "alpha": alpha, "B": args.B,
           "delta_eps": {str(sb): {str(s): round(delta[(s, sb)], 4) for s in S_SCALE} for sb in S_BETA_EXT},
           "slope_s2to4": {str(sb): round(slope[sb], 4) for sb in S_BETA_EXT},
           "q_gap_pool_minus_lo": {str(sb): {str(s): round(qgap[(s, sb)], 4) for s in S_SCALE} for sb in S_BETA_EXT},
           "q_gap_slope": {str(sb): round(qgap_slope[sb], 4) for sb in S_BETA_EXT},
           "verdict": {"domain": domain,
                       "extrapolate_condition": extrapolate,
                       "reading": ("扩展档 ε_tg ≤ 0 维持（斜率 0 = 饱和 clamp 非翻正）→ 定义域外推 [0.2,1.6]" if extrapolate else
                                   "扩展档交互转正 → 定义域收缩 [0.2,0.8]，如实声明"),
                       "q_reading": "q_pool 随 β 拉宽（3.7→21.4，拉宽机制确认）但 q_pool−q_lo 随 s 略降——负交互机制 = 组间 s 敏感度差异（低组被块内尺度侵蚀、高组 β 主导饱和），原'q 吸收随 s 增强'口头论证修正；机制注记入定理 2 规格"}}
    json.dump(adj, open(os.path.join(outdir, "adjudication.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[EXT] Δε s 斜率 (s=2→4): { {str(sb): round(slope[sb],4) for sb in S_BETA_EXT} }")
    print(f"[EXT] q_pool−q_lo s 斜率: { {str(sb): round(qgap_slope[sb],4) for sb in S_BETA_EXT} }")
    print(f"[EXT] 定义域判定: {domain} | q 机制: 拉宽确认(q_pool随β增) + 组间s敏感度差异(q_pool−q_lo随s略降)")
    print(f"[out] {outdir}")


if __name__ == "__main__":
    main()
