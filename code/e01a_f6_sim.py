#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e01a_f6_sim.py —— F6 群组趋势分化因子 v1.1（定理 2 前置：ε_g 频率标定）
版本：v1.1 ｜ 2026-08-28 ｜ 判据 F6-v0.2（预注册冻结：F6_prereg.md v0.2，红线 #6/#10/#11）
run_status: exploratory（判据首次应用；本跑确认后由评审升格 confirmatory）

DGP：y_it = mu_i + lam_t + beta_i·(t − T/2) + u_it
  beta_i ~ N(0, s_beta)（国家级，连续分布——连续权重 w_g 的仿真对应；
  趋势国 = 尾部 ~5%，对应 E01b 5/39 FP 国占比）
v1.1 变更（F6-v0.2）：beta 国家级替代群组级；s_beta 档位 {0,0.1,0.2,0.4,0.8}（β 须达 q 量级——
  线性趋势被双因素自校准吸收，净效应 = 1 期增量 beta_i，v0.1 物理缺陷入预注册 changelog）；
  新增分层报告（beta_i >= P75 高趋势国 vs <= P25 低趋势国块内条件覆盖）
度量：eps_g(s_beta, arm) = 1−alpha − 块内条件覆盖（in_median），缺口 >= 0（与 eps_t 同构）
判据（F6-v0.2）：A6 naive eps_g 单调 + 0.4 档 90% 带下界>0；B6 至少一臂块缓解；C6 臂序 + 分层差
用法：python e01a_f6_sim.py [--B 500] [--seed 20260828]
输出：04_output/experiments/E01a/f6_<ts>/{response_surface.csv, conditional_coverage.csv, stratified.csv, adjudication.json}
"""
import argparse, json, os, sys, csv, datetime, zlib
import numpy as np

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "03_program", "experiments", "E01_three_arm_coverage"))
from e01a_sim import rolling_pseudo_residuals, fit_arms, coverage_width, summarize

SBETA_GRID = [0.0, 0.1, 0.2, 0.4, 0.8]
ARMS = ("naive", "country_block", "time_block")


def gen_panel_f6(N, T, s_mu, s_lam, lam_ar, phi, s_scale, rho, s_beta, rng, block_anchor="tail"):
    """F6-v0.2 DGP：y = mu_i + lam_t + beta_i·(t−T/2) + u_it；beta_i 国家级连续分布。
    返回 y, block_flags, beta（该轮 beta 实现，供分层）。"""
    mu = rng.normal(0.0, s_mu, N)
    lam = np.zeros(T + 1)
    for t in range(T + 1):
        lam[t] = lam_ar * (lam[t - 1] if t > 0 else 0.0) + rng.normal(0.0, s_lam)
    beta = rng.normal(0.0, s_beta, N)                     # 国家级趋势率
    f = rng.normal(0.0, 1.0, (T + 1,))
    g = np.arange(N) % 1                                  # 占位（国家级 beta 无需区域因子）
    sig = np.ones(T + 1)
    blk = max(2, (T + 1) // 4)
    if block_anchor == "tail":
        lo = max(0, T - 2 * blk + 2)
        st = int(rng.integers(lo, T + 1))
    else:
        st = int(rng.integers(0, T + 1 - blk + 1))
    sig[st:st + blk] = s_scale
    tm = np.arange(T + 1) - T / 2.0
    trend = (beta[:, None] * tm[None, :])
    y = np.zeros((N, T + 1)); u = np.zeros((N, T + 1))
    for t in range(T + 1):
        if t == 0:
            u[:, t] = sig[t] * (rho * f[t] + np.sqrt(1 - rho ** 2) * rng.normal(0.0, 1.0, N))
        else:
            u[:, t] = phi * u[:, t - 1] + sig[t] * (rho * f[t] + np.sqrt(1 - rho ** 2) * rng.normal(0.0, 1.0, N))
        y[:, t] = mu + lam[t] + trend[:, t] + u[:, t]
    return y, sig > 1.0, beta


def run_f6(params, N, T, alpha, B, base_seed, block_anchor="tail"):
    """B 次重复：三臂饱和覆盖 + 块内/块外 + 分层（高/低趋势国）块内覆盖。"""
    rng = np.random.default_rng(base_seed)
    K = 5
    out = {a: {"cov": [], "w": [], "in": [], "out": [], "in_hi": [], "in_lo": []} for a in ARMS}
    for b in range(B):
        y, bf, beta = gen_panel_f6(N, T, **params, rng=rng, block_anchor=block_anchor)
        y_train = y[:, :T]; y_true = y[:, T]
        q, center, rr, _ = fit_arms(y_train, alpha, K)
        hi = np.abs(beta) >= np.quantile(np.abs(beta), 0.75)
        lo = np.abs(beta) <= np.quantile(np.abs(beta), 0.25)
        for a in ARMS:
            c, w = coverage_width(y_true, center, q[a])
            out[a]["cov"].append(c); out[a]["w"].append(w)
            if bf[T]:
                out[a]["in"].append(c)
                out[a]["in_hi"].append(np.mean(np.abs(y_true[hi] - center[hi]) <= q[a]))
                out[a]["in_lo"].append(np.mean(np.abs(y_true[lo] - center[lo]) <= q[a]))
            else:
                out[a]["out"].append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()
    N, T, alpha = 38, 24, 0.10
    base = {"s_mu": 0.5, "s_lam": 0.5, "lam_ar": 0.0, "phi": 0.4, "s_scale": 2.0,
            "rho": 0.5}
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "04_output", "experiments", "E01a", f"f6_{ts}")
    os.makedirs(outdir, exist_ok=True)

    rows, cond_rows, strat_rows = [], [], []
    eps_g = {}
    for sb in SBETA_GRID:
        p = dict(base); p["s_beta"] = sb
        cname = f"F6_beta={sb}"
        seed = args.seed + (zlib.crc32(cname.encode()) % 10 ** 6)
        out = run_f6(p, N, T, alpha, args.B, seed, "tail")
        for a in ARMS:
            o = out[a]
            rows.append({"config": cname, "arm": a,
                         **{f"sat_cov_{k}": v for k, v in summarize(o["cov"]).items()},
                         **{f"sat_w_{k}": v for k, v in summarize(o["w"]).items()}})
            cond_rows.append({"config": cname, "arm": a,
                              **{f"in_{k}": v for k, v in summarize(o["in"]).items()},
                              **{f"out_{k}": v for k, v in summarize(o["out"]).items()},
                              "n_in": len(o["in"]), "n_out": len(o["out"])})
            strat_rows.append({"config": cname, "arm": a,
                               "in_hi_median": round(summarize(o["in_hi"])["median"], 4),
                               "in_lo_median": round(summarize(o["in_lo"])["median"], 4),
                               "hi_lo_diff": round(summarize(o["in_hi"])["median"] - summarize(o["in_lo"])["median"], 4)})
            sm = summarize(o["in"])
            eps_g[(sb, a)] = {"eps_g": round(max(0.0, (1 - alpha) - sm["median"]), 4),
                              "in_median": round(sm["median"], 4),
                              "in_lo90": round(sm["lo90"], 4), "in_hi90": round(sm["hi90"], 4)}

    with open(os.path.join(outdir, "response_surface.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(outdir, "conditional_coverage.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cond_rows[0].keys())); w.writeheader(); w.writerows(cond_rows)
    with open(os.path.join(outdir, "stratified.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(strat_rows[0].keys())); w.writeheader(); w.writerows(strat_rows)

    # ---- 判据 F6-v0.2 ----
    def eg(sb, arm):
        return eps_g[(sb, arm)]
    naive_eg = [eg(sb, "naive") for sb in SBETA_GRID]
    eps_vals = [e["eps_g"] for e in naive_eg]
    monotone = all(eps_vals[i] <= eps_vals[i + 1] + 1e-9 for i in range(len(eps_vals) - 1))
    a6_positive = (1 - alpha) - eg(0.4, "naive")["in_hi90"] > 0.0
    A6 = monotone and a6_positive
    B6_arm = None
    for arm in ("time_block", "country_block"):
        hit = any(eg(sb, arm)["eps_g"] < eg(sb, "naive")["eps_g"] - 1e-9 for sb in SBETA_GRID[1:])
        if hit:
            B6_arm = arm; break
    B6 = B6_arm is not None
    hi_lo = {}
    for sb in SBETA_GRID[1:]:
        r = next(r for r in strat_rows if r["config"] == f"F6_beta={sb}" and r["arm"] == "naive")
        hi_lo[str(sb)] = {"diff": r["hi_lo_diff"], "in_hi": r["in_hi_median"], "in_lo": r["in_lo_median"]}
    C6_strat_monotone = all(abs(hi_lo[str(SBETA_GRID[i])]["diff"]) <= abs(hi_lo[str(SBETA_GRID[i + 1])]["diff"]) + 1e-9
                            for i in range(1, len(SBETA_GRID) - 1))
    order = {str(sb): sorted(ARMS, key=lambda a: -eg(sb, a)["eps_g"]) for sb in SBETA_GRID[1:]}
    adj = {"run_status": "exploratory", "protocol_version": "v1.1",
           "criterion_version": "F6-v0.2（F6_prereg.md v0.2，跑前冻结）",
           "N": N, "T": T, "alpha": alpha, "B": args.B, "seed": args.seed,
           "eps_g_by_sbeta": {str(sb): {a: eps_g[(sb, a)] for a in ARMS} for sb in SBETA_GRID},
           "stratified_naive": hi_lo,
           "criterion": {"A6_monotone": monotone, "A6_eps04_positive": a6_positive, "A6_pass": A6,
                         "B6_relief_arm": B6_arm, "B6_pass": B6,
                         "C6_arm_order": order, "C6_hi_lo_diff": hi_lo,
                         "C6_strat_monotone": C6_strat_monotone,
                         "A6_eps_series": eps_vals},
           "verdict": ("A6∧B6 全过" if (A6 and B6) else
                       ("A6 过 B6 未过" if A6 else ("A6 未过 B6 过" if B6 else "双未过 → indeterminate，转查 DGP 设计"))),
           "note": "ε_g 频率标定（仿真可算覆盖）；E01b FP 画像为机制证据（方向/结构），不入频率标定（循环依赖分工表，spec v0.3）。"}
    json.dump(adj, open(os.path.join(outdir, "adjudication.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"[F6-v0.2] run: {outdir}  status=exploratory（判据 F6-v0.2 首次应用）")
    print("ε_g(s_beta, arm) = 1−α − 块内条件覆盖:")
    print(f"{'s_beta':>6} | " + " | ".join(f"{a:>14}" for a in ARMS))
    for sb in SBETA_GRID:
        print(f"{sb:>6.2f} | " + " | ".join(f"{eg(sb, a)['eps_g']:.4f} ({eg(sb, a)['in_median']:.3f})" for a in ARMS))
    print(f"判据 A6: 单调={monotone} ε(.4)带下界>0={a6_positive} → {'PASS' if A6 else 'FAIL'}")
    print(f"判据 B6: 缓解臂={B6_arm} → {'PASS' if B6 else 'FAIL'}")
    print(f"判据 C6: 分层差(hi−lo)={hi_lo} 单调={C6_strat_monotone}")
    print(f"臂序: {order}")


if __name__ == "__main__":
    main()
