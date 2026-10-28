#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e01a_f6_ci.py —— F6 配对增量 bootstrap（第十一轮：保证版 ε_g 噪声带）
配对设计：同 DGP 实现（mu/lam/u/块位置共享）+ 各 s_beta 档 β 独立采样
  ⟹ 配对增量 δ_b(sb) = in(0,b) − in(sb,b)（共同噪声与主效应消去，与 C10 配对差分同构）
输出：估计版（δ 中位数）与保证版（δ 90% 上分位）的凹上确界参数（min(a·w, emax)）+ CI
run_status: exploratory（数值诊断，非判据 run——红线 #10 注明）
"""
import argparse, json, os, sys, datetime
import numpy as np

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "03_program", "experiments", "E01_three_arm_coverage"))
from e01a_sim import fit_arms

SBETA_GRID = [0.0, 0.1, 0.2, 0.4, 0.8]


def gen_pair_f6(N, T, params, s_beta_list, rng, block_anchor="tail"):
    """共享 mu/lam/f/u/块位置，各档 β 独立——严格配对。返回 {sb: y} 与块标志。"""
    s_mu, s_lam, lam_ar, phi, s_scale, rho = params
    mu = rng.normal(0.0, s_mu, N)
    lam = np.zeros(T + 1)
    for t in range(T + 1):
        lam[t] = lam_ar * (lam[t - 1] if t > 0 else 0.0) + rng.normal(0.0, s_lam)
    f = rng.normal(0.0, 1.0, (T + 1,))
    sig = np.ones(T + 1)
    blk = max(2, (T + 1) // 4)
    if block_anchor == "tail":
        lo = max(0, T - 2 * blk + 2)
        st = int(rng.integers(lo, T + 1))
    else:
        st = int(rng.integers(0, T + 1 - blk + 1))
    sig[st:st + blk] = s_scale
    u = np.zeros((N, T + 1))
    for t in range(T + 1):
        if t == 0:
            u[:, t] = sig[t] * (rho * f[t] + np.sqrt(1 - rho ** 2) * rng.normal(0.0, 1.0, N))
        else:
            u[:, t] = phi * u[:, t - 1] + sig[t] * (rho * f[t] + np.sqrt(1 - rho ** 2) * rng.normal(0.0, 1.0, N))
    tm = np.arange(T + 1) - T / 2.0
    out = {}
    for sb in s_beta_list:
        beta = rng.normal(0.0, sb, N)
        y = mu[None, :].T + lam[None, :] + (beta[:, None] * tm[None, :]) + u
        out[sb] = y
    return out, sig > 1.0


def naive_in_block(y, T, alpha):
    """naive 目标年块内条件覆盖（单重复，bfT 已由调用方判定）"""
    y_train = y[:, :T]
    q, center, rr, _ = fit_arms(y_train, alpha)
    y_true = y[:, T]
    return float(np.mean(np.abs(y_true - center) <= q["naive"]))


def envelope(w, y):
    """凹上确界（有界线性族内）：min(a·w, emax)；a 由 (0,0)→(w*, y(w*)) 弦（w* = max 首现档），
    emax = max y——保证弦从上方穿过全部数据点（凹包络）。"""
    iw = int(np.argmax(y))
    wstar = w[iw]
    ystar = y[iw]
    a = ystar / wstar
    emax = float(np.max(y))
    return a, emax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--N", type=int, default=38)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    params = (0.5, 0.5, 0.0, 0.4, 2.0, 0.5)  # s_mu, s_lam, lam_ar, phi, s_scale, rho
    rng = np.random.default_rng(args.seed)
    in_vals = {sb: [] for sb in SBETA_GRID}
    for b in range(args.B):
        panels, bf = gen_pair_f6(args.N, args.T, params, SBETA_GRID, rng, block_anchor="tail")
        if not bf[args.T]:
            continue
        for sb in SBETA_GRID:
            in_vals[sb].append(naive_in_block(panels[sb], args.T, args.alpha))
    n = min(len(v) for v in in_vals.values())
    for sb in SBETA_GRID:
        in_vals[sb] = in_vals[sb][:n]
    A = np.array([in_vals[sb] for sb in SBETA_GRID])          # (5, n)：行序 = SBETA_GRID
    wg = np.array([0.1, 0.2, 0.4, 0.8])
    n = A.shape[1]

    # ── 诊断：配对增量（v1.0 首跑暴露：中位数趋零——β 净效应 ~0.16 被单重复块内离散 ~0.4 淹没，
    #    配对差分适合 C10 组间比较（β 对比 0.4 vs 0.8，效应大）不适合 ε_g 标定增量（β=0 vs 0.2，效应小））
    delta_pair = A[0] - A[1:]                                    # (4, n)
    pair_med = np.median(delta_pair, axis=1)
    pair_hi90 = np.quantile(delta_pair, 0.90, axis=1)

    # ── 正确工具：跨实现中位数差的 bootstrap（官方 F6 用跨实现中位数 = 稳健统计量）
    #    对配对样本（每样本 = 5 档覆盖）重采样 B_boot 次，算 med(sb) − med(0)
    rng2 = np.random.default_rng(20260828 + 1)
    B_boot = 2000
    boot_delta = np.empty((B_boot, 4))
    idx = rng2.integers(0, n, size=(B_boot, n))
    for bb in range(B_boot):
        sample = A[:, idx[bb]]                                  # (5, n) 重采样
        meds = np.median(sample, axis=1)
        boot_delta[bb] = meds[1:] - meds[0]                     # med(sb) − med(0)
    med_boot = np.median(boot_delta, axis=0)                    # 估计版（应 ≈ 官方 0.026/0.053/0.053/0.053）
    hi_boot = np.quantile(boot_delta, 0.90, axis=0)             # 保证版上带点
    lo_boot = np.quantile(boot_delta, 0.10, axis=0)

    def envelope(w, y):
        """凹上确界（有界线性族内）：min(a·w, emax)；a 由 (0,0)→(w*, y(w*)) 弦（w* = max 首现档），
        emax = max y——保证弦从上方穿过全部数据点（凹包络）。"""
        iw = int(np.argmax(y))
        wstar = w[iw]
        ystar = y[iw]
        a = ystar / wstar
        emax = float(np.max(y))
        return a, emax

    a_est, e_est = envelope(wg, med_boot)
    a_ub, e_ub = envelope(wg, hi_boot)
    over_est = all(med_boot[i] <= min(a_est * wg[i], e_est) + 1e-12 for i in range(len(wg)))
    over_ub = all(hi_boot[i] <= min(a_ub * wg[i], e_ub) + 1e-12 for i in range(len(wg)))

    out = {
        "B": args.B, "n_paired": n, "B_boot": B_boot, "seed": args.seed, "alpha": args.alpha,
        "wg": list(wg),
        "diagnostic_pair_delta": {
            "median": [round(x, 4) for x in pair_med],
            "hi90": [round(x, 4) for x in pair_hi90],
            "note": "配对差分中位数趋零 = β 净效应（1 期增量）被单重复块内覆盖离散（~0.4）淹没——配对差分适合 C10 组间比较（效应大），不适合 ε_g 标定增量（效应小）；ε_g 标定用跨实现中位数（官方 F6 同）",
        },
        "boot_median_delta": [round(x, 4) for x in med_boot],
        "boot_lo10_delta": [round(x, 4) for x in lo_boot],
        "boot_hi90_delta": [round(x, 4) for x in hi_boot],
        "estimate_version": {"a": round(a_est, 4), "emax": round(e_est, 4), "envelope_ok": bool(over_est)},
        "guarantee_version": {"a": round(a_ub, 4), "emax": round(e_ub, 4), "envelope_ok": bool(over_ub)},
        "note": "保证版 = 跨实现中位数差 bootstrap 90% 上分位的凹上确界（更陡、平台更高，被减项上界）；估计版 = bootstrap 中位数。",
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "04_output", "experiments", "E01a", f"f6_ci_{ts}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "adjudication.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
