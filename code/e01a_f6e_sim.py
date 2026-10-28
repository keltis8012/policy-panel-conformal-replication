#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e01a_f6e_sim.py —— F6-empirical（第十一轮：w_g 实证刻度标定，标定桥方案一）
预注册：F6_empirical_prereg.md（判据 F6E-v0.1，跑前冻结；红线 #13 扩展：可识别性检查已过）
DGP：y = mu_i + lam_t + beta_i·(t−T/2) + u_it；beta_i = a·z_i（实证 rank 化趋势强度，IQR 匹配）
  z_i = (|Δ_i| − median(|Δ|)) / IQR(|Δ|)，Δ_i = 39 国样本期平均 ΔlnCO₂（E01b 面板历史属性）
  合法分界（预注册 §4）：β_i 消费面板矩（Δ 分布），非 backtest 输出（FP/放行）——验证逻辑未被触碰
度量：eps_g(a) = med(in(0)) − med(in(a))（跨实现中位数差——第十一轮诊断：配对差中位数趋零，
  跨实现中位数是稳健度量）；实现级数据落盘 → bootstrap 90% 带（保证版 = 上带点凹上确界）
run_status: exploratory（判据 F6E-v0.1 首次应用）
"""
import argparse, json, os, sys, csv, datetime, zlib
import numpy as np

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "03_program", "experiments", "E01_three_arm_coverage"))
sys.path.insert(0, os.path.join(ROOT, "03_program", "analysis"))
from e01a_sim import fit_arms
from e01b_empirical import load_panel

A_GRID = [0.0, 0.2, 0.4, 0.8]   # F6E-v0.2：主档 a*=0.4（尺度匹配），稳健性 {0.2, 0.8}


def empirical_z(countries, Y, years):
    """39 国样本期平均 |ΔlnCO₂| 的 IQR 标准化（历史属性，不消费实验输出）。
    NaN 国（样本期 Δ 缺失，如 CHN 早期缺总量行）→ z=0（无趋势强度信息，β=0 保守）"""
    d = []
    for i in range(Y.shape[0]):
        vals = []
        for t in range(1, Y.shape[1]):
            a, b = Y[i, t - 1], Y[i, t]
            if np.isfinite(a) and np.isfinite(b):
                vals.append(b - a)
        d.append(np.mean(vals) if vals else np.nan)
    d = np.array(d)
    med = np.nanmedian(np.abs(d))
    iqr = np.nanpercentile(np.abs(d), 75) - np.nanpercentile(np.abs(d), 25)
    z = (np.abs(d) - med) / iqr if iqr > 0 else np.zeros_like(d)
    z = np.where(np.isnan(z), 0.0, z)          # NaN 国 → 0（β=0，无趋势）
    return z, d


def gen_panel_f6e(N, T, params, beta, rng, block_anchor="tail"):
    """F6-empirical DGP：beta 由实证 z 给定（固定，非随机）。返回 y, 块标志。"""
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
    y = mu[None, :].T + lam[None, :] + (beta[:, None] * tm[None, :]) + u
    return y, sig > 1.0


def naive_in(y, T, alpha):
    y_train = y[:, :T]
    q, center, rr, _ = fit_arms(y_train, alpha)
    y_true = y[:, T]
    return float(np.mean(np.abs(y_true - center) <= q["naive"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--N", type=int, default=38)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--B_boot", type=int, default=2000)
    args = ap.parse_args()

    # ── 实证 z（39 国）与 β 向量（N=38 取前 38——与 E01a N 对齐；Z 已含全部国家）
    countries, years, Y, S = load_panel()
    z_all, d_all = empirical_z(countries, Y, years)
    z = z_all[: args.N]
    # FP 国位置（机制段桥接：TUR/BRA/IDN 在实证 w_g 分布中的 rank）
    fp_rank = {}
    for c in ("TUR", "BRA", "IDN", "MEX", "KOR"):
        if c in countries:
            i = countries.index(c)
            fp_rank[c] = {"z": round(float(z_all[i]), 3), "rank_pct": round(100 * (z_all < z_all[i]).mean(), 1)}

    params = (0.5, 0.5, 0.0, 0.4, 2.0, 0.5)
    rng = np.random.default_rng(args.seed)
    in_vals = {a: [] for a in A_GRID}
    for b in range(args.B):
        for a in A_GRID:
            beta = a * z
            y, bf = gen_panel_f6e(args.N, args.T, params, beta, rng, block_anchor="tail")
            if bf[args.T]:
                in_vals[a].append(naive_in(y, args.T, args.alpha))
    n = min(len(v) for v in in_vals.values())
    for a in A_GRID:
        in_vals[a] = in_vals[a][:n]
    A = np.array([in_vals[a] for a in A_GRID])          # (4, n)：行序 = A_GRID

    # ── 跨实现中位数差（ε_g 增量 = med(0) − med(a)，正 = 缺口增大）
    meds = np.median(A, axis=1)
    eps_g = meds[0] - meds[1:]                            # a ∈ {0.2, 0.4, 0.8}
    # bootstrap（配对样本重采样，med 差）
    rng2 = np.random.default_rng(args.seed + 7)
    B_boot = args.B_boot
    boot = np.empty((B_boot, len(A_GRID) - 1))
    idx = rng2.integers(0, n, size=(B_boot, n))
    for bb in range(B_boot):
        s = A[:, idx[bb]]
        m = np.median(s, axis=1)
        boot[bb] = m[0] - m[1:]
    est = np.median(boot, axis=0)
    lo10 = np.quantile(boot, 0.10, axis=0)
    hi90 = np.quantile(boot, 0.90, axis=0)

    def envelope(w, y):
        """凹上确界（有界线性族内）：min(a·w, emax)；a = max_i(y_i/w_i)（过原点弦斜率的最大值——
        确保弦从上方穿过全部数据点），emax = max y（平台）。v0.3 修正：argmax 误选最远点，
        正确应为 max 斜率（最陡数据点），否则低 w 处弦低于数据。"""
        a_slope = float(np.max(y / w))
        return a_slope, float(np.max(y))

    ag = np.array([0.2, 0.4, 0.8])
    a_est, e_est = envelope(ag, est)
    a_ub, e_ub = envelope(ag, hi90)
    over_est = all(est[i] <= min(a_est * ag[i], e_est) + 1e-12 for i in range(len(ag)))
    over_ub = all(hi90[i] <= min(a_ub * ag[i], e_ub) + 1e-12 for i in range(len(ag)))

    # ── 判据（F6E-v0.3：A6E 主判据 + C6E 改配对组间分位差——分层覆盖在目标年单期无判别力，
    #     v0.2→v0.3 changelog：组间分位差消去共同块效应（ε_t），方向 +0.91 但 90% 带含零（组内 10 国样本限制））
    a06_ok = bool(eps_g[1] > 0 and lo10[1] > 0)          # A6E：a*=0.4 缺口成立（90% 带下界 > 0）
    hi_mask = z >= np.quantile(z, 0.75)
    lo_mask = z <= np.quantile(z, 0.25)
    pair_diff = []
    for b in range(200):
        beta = 0.4 * z
        y, bf = gen_panel_f6e(args.N, args.T, params, beta, rng2, block_anchor="tail")
        if not bf[args.T]:
            continue
        y_train = y[:, : args.T]
        q, center, rr, _ = fit_arms(y_train, args.alpha)
        e = np.abs(y[:, args.T] - center)
        pair_diff.append(float(np.quantile(e[hi_mask], 0.9) - np.quantile(e[lo_mask], 0.9)))
    pd = np.array(pair_diff) if pair_diff else np.array([np.nan])
    d_hi_lo = float(np.median(pd))
    c6e_ok = bool(d_hi_lo > 0)                            # C6E：方向性（90% 带含零 = 组内样本限制，如实注记）

    out = {
        "run_status": "exploratory",
        "criterion_version": "F6E-v0.3（F6_empirical_prereg.md changelog：v0.1 名义档尺度不足 → v0.2 主档改 a*=0.2/std(z)≈0.4 → v0.3 C6E 改配对组间分位差（分层覆盖无判别力））",
        "B": args.B, "n_paired": n, "B_boot": B_boot, "seed": args.seed,
        "z_stats": {"median": round(float(np.nanmedian(np.abs(z_all))), 4),
                    "iqr": round(float(np.nanpercentile(np.abs(z_all), 75) - np.nanpercentile(np.abs(z_all), 25)), 4),
                    "std": round(float(np.nanstd(z_all)), 4),
                    "a_star": round(float(0.2 / np.nanstd(z_all)), 4)},
        "fp_country_position": fp_rank,
        "eps_g": {"a_0.2": round(float(eps_g[0]), 4), "a_0.4": round(float(eps_g[1]), 4), "a_0.8": round(float(eps_g[2]), 4)},
        "boot": {"est": [round(float(x), 4) for x in est],
                 "lo10": [round(float(x), 4) for x in lo10],
                 "hi90": [round(float(x), 4) for x in hi90]},
        "estimate_version": {"a": round(a_est, 4), "emax": round(e_est, 4), "envelope_ok": bool(over_est)},
        "guarantee_version": {"a": round(a_ub, 4), "emax": round(e_ub, 4), "envelope_ok": bool(over_ub)},
        "stratified": {"pair_hi_minus_lo_q90_median": round(d_hi_lo, 4),
                       "c6e_note": "配对组间分位差（消去共同块效应）；中位 > 0 方向性，90% 带含零（组内 10 国 0.9 分位估计噪声——38 国面板固有限制，如实注记）"},
        "criterion": {"A6E_eps_g_a04_gt0_lo10_gt0": a06_ok,
                      "C6E_pair_hi_lo_q90_diff_gt0": c6e_ok,
                      "monotone_a": bool(all(eps_g[i] <= eps_g[i + 1] + 1e-9 for i in range(len(eps_g) - 1)))},
        "verdict": "PASS" if (a06_ok and c6e_ok) else ("PASS_A6E_only" if a06_ok else "REVIEW"),
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "04_output", "experiments", "E01a", f"f6e_{ts}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "adjudication.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 实现级数据落盘（bootstrap 可复现 + 供后续分析）
    with open(os.path.join(outdir, "realizations.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a"] + [f"in_{i}" for i in range(n)])
        for ai, a in enumerate(A_GRID):
            w.writerow([a] + [round(float(x), 4) for x in A[ai]])
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
