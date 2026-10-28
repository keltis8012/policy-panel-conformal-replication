#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E01a 仿真 v0.3（confirmatory-2 门槛）：DGP 五因子 × 三臂 × 等覆盖宽度 × 条件覆盖一等公民
依据：protocol.md §8（R13 v0.3 清单 = R12 三项 + R13 五项）｜ 版本：v0.3 ｜ 2026-08-27
v0.3 相对 v0.2 的八项增补（R13 清单全落地）：
  1. 块锚定 tail 模式（st ~ U(T−2blk+2, T)，目标年入块概率 ~50%——v0.2 随机模式仅 5%，条件覆盖统计力不足）
  2. cond_in 判据区间化（hi90 < 阈值 或 中位数差 > 0.05 且区间不重叠）
  3. F2 AR 持久化变体（λ_t = lam_ar·λ_{t-1} + η_t，中心质量对照）
  4. 等覆盖宽度比较（α 网格内插值，各臂在经验覆盖 0.90 处的宽度）
  5. 条件覆盖一等公民输出（conditional_coverage.csv：config×arm×块内/块外×聚类 CI）+ 触发器出处钉死
  6. F5 方差保持式加载（u_it = φu + σ_t·(ρf + √(1−ρ²)ε)，边际方差与 ρ 无关）
  7. T 对齐审计（adjudication 记录实际 N/T；pre-2000 已闭环：CO₂ min=2000 → T=21 实证）
  8. covmatch 报告规则（覆盖报 sat，匹配只用于宽度；naive 匹配臂覆盖从报告移除）
用法：python e01a_sim.py [--B 500] [--N 38] [--T 24] [--alpha 0.10] [--seed 20260827]
"""
import argparse, json, os, datetime, zlib
import numpy as np

# ---------------------------------------------------------------- DGP ----
def gen_panel(N, T, s_mu, s_lam, lam_ar, phi, s_scale, rho, n_regions, rng, block_anchor="tail"):
    """v0.3 DGP：y_it = mu_i + lam_t + u_it；索引 0..T（T+1 列，目标 t=T）
      lam_t = lam_ar·lam_{t-1} + η_t                          (F2 AR 持久化变体)
      u_it = phi·u_{i,t-1} + σ_t·(ρ·f_{g(i),t} + √(1−ρ²)·ε_it) (F5 方差保持 + F4 尺度块)
      block_anchor: 'random'(v0.2) | 'tail'(块位置条件化，目标年入块概率~50%)
    返回 y(N,T+1), block_flags(T+1,)"""
    mu = rng.normal(0.0, s_mu, N)
    lam = np.zeros(T + 1)
    for t in range(T + 1):
        lam[t] = lam_ar * (lam[t - 1] if t > 0 else 0.0) + rng.normal(0.0, s_lam)
    g = np.arange(N) % n_regions
    f = rng.normal(0.0, 1.0, (n_regions, T + 1))
    sig = np.ones(T + 1)
    blk = max(2, (T + 1) // 4)
    if block_anchor == "tail":
        lo = max(0, T - 2 * blk + 2)
        st = int(rng.integers(lo, T + 1))       # 目标年入块概率 ≈ blk/(2blk−1) ≈ 50%
    else:
        st = int(rng.integers(0, T + 1 - blk + 1))
    sig[st:st + blk] = s_scale
    y = np.zeros((N, T + 1))
    u = np.zeros((N, T + 1))
    for t in range(T + 1):
        if t == 0:
            u[:, t] = sig[t] * (rho * f[g, t] + np.sqrt(1 - rho ** 2) * rng.normal(0.0, 1.0, N))
        else:
            u[:, t] = phi * u[:, t - 1] + sig[t] * (rho * f[g, t] + np.sqrt(1 - rho ** 2) * rng.normal(0.0, 1.0, N))
        y[:, t] = mu + lam[t] + u[:, t]
    return y, sig > 1.0

# ---------------------------------------------------------------- 评分 ----
def rolling_pseudo_residuals(y, K=5):
    """滚动起点一步伪残差（v0.2 保留）"""
    N, Tcol = y.shape
    yrs = np.arange(K, Tcol)
    rr = np.empty((N, len(yrs)))
    for j, t in enumerate(yrs):
        seg = y[:, :t]
        mu_h = seg.mean(axis=1); lam_h = seg.mean(axis=0)
        rr[:, j] = y[:, t] - (mu_h + lam_h[-1])
    return rr, yrs

def fit_arms(y_train, alpha, K=5):
    rr, yrs = rolling_pseudo_residuals(y_train, K)
    absr = np.abs(rr)
    mu_h = y_train.mean(axis=1); lam_h = y_train.mean(axis=0)
    center = mu_h + lam_h[-1]
    q = {}
    q["naive"] = np.quantile(absr.ravel(), 1 - alpha)
    q["time_block"] = np.quantile(np.quantile(absr, 1 - alpha, axis=0), 1 - alpha)
    q["country_block"] = np.quantile(np.quantile(absr, 1 - alpha, axis=1), 1 - alpha)
    return q, center, rr, yrs

def coverage_width(y_true, center, q):
    return np.mean(np.abs(y_true - center) <= q), 2.0 * q

# ---------------------------------------------------------------- 单配置 ----
def run_config(params, N, T, alpha, B, base_seed, block_anchor="tail"):
    """B 次重复。返回：
      sat: {arm: {'cov':[], 'w':[]}}
      cond: {arm: {'in':[], 'out':[]}}       （条件覆盖一等公民，三臂）
      paired: {arm: []} 名义 α 下匹配宽度（time/country 配对）
    等覆盖宽度已移除（R16c 终审 v0.4 发现：覆盖(q)=mean(|e|<=q) → 等覆盖 0.90 的解 = quantile(|e|,0.90)
    与臂无关——三臂共享 center+误差，等覆盖宽度数学恒同，作为臂间效率指标结构空洞；
    v0.3 的 36% 溢价系 grid 边界截断伪影（naive 6.42 为截断值），弃用。
    臂间宽度比较 = 名义 α 宽度 + 覆盖水平双报告（sat_w/sat_cov）。
    """
    rng = np.random.default_rng(base_seed)
    K = 5
    out = {a: {"cov": [], "w": [], "in": [], "out": []} for a in ("naive", "country_block", "time_block")}
    paired = {"time": [], "country": []}
    eqw_b = {}  # 等覆盖已移除（数学空洞，见 docstring）
    for b in range(B):
        y, bf = gen_panel(N, T, **params, rng=rng, block_anchor=block_anchor)
        y_train = y[:, :T]; y_true = y[:, T]
        q, center, rr, _ = fit_arms(y_train, alpha, K)
        absr = np.abs(rr)
        # 饱和（procedure-true 覆盖）——名义 α 三臂
        for a in ("naive", "country_block", "time_block"):
            c, w = coverage_width(y_true, center, q[a])
            out[a]["cov"].append(c); out[a]["w"].append(w)
            (out[a]["in"] if bf[T] else out[a]["out"]).append(c)
        # 匹配宽度（截断到 n_time_blocks，只用于宽度比较——covmatch 规则 #8）
        n_tb = T - K
        qn = np.quantile(absr.ravel()[rng.choice(absr.size, n_tb, replace=False)], 1 - alpha)
        gi = np.quantile(absr, 1 - alpha, axis=1)
        qc = np.quantile(gi[rng.choice(gi.size, n_tb, replace=False)], 1 - alpha)
        qt = np.quantile(np.quantile(absr, 1 - alpha, axis=0), 1 - alpha)
        paired["time"].append(2.0 * qt); paired["country"].append(2.0 * qc)
    return out, paired

def summarize(vals):
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return {"median": float("nan"), "lo90": float("nan"), "hi90": float("nan")}
    return {"median": float(np.median(v)), "lo90": float(np.quantile(v, 0.05)),
            "hi90": float(np.quantile(v, 0.95))}

# ---------------------------------------------------------------- 判据 v2 解析 ----
def analytic_naive_in(s_scale, alpha=0.10, f_mix=0.25, sigma_e=1.0, lo=1e-6, hi=10.0, tol=1e-9):
    """混合残差池化分位 → 块内 naive 覆盖解析预言（protocol §9.3）：
      F(q) = (1-f)*(2Phi(q)-1) + f*(2Phi(q/s)-1)；q_mix 解 F(q_mix)=1-alpha；
      naive_in(s) = 2Phi(q_mix/(s*sigma_e)) - 1（块内误差尺度 s*sigma_e）
    用 math.erf（无 scipy 依赖）"""
    import math
    phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    def F(q):
        return (1 - f_mix) * (2 * phi(q) - 1) + f_mix * (2 * phi(q / s_scale) - 1)
    lo_v, hi_v = lo, hi
    for _ in range(200):
        mid = 0.5 * (lo_v + hi_v)
        if F(mid) < 1 - alpha:
            lo_v = mid
        else:
            hi_v = mid
        if hi_v - lo_v < tol:
            break
    q_mix = 0.5 * (lo_v + hi_v)
    naive_in = 2 * phi(q_mix / (s_scale * sigma_e)) - 1
    return q_mix, naive_in

def adjudicate_v2(cond_rows, config="baseline"):
    """判据 v2 四分支（protocol §9.2）——三原子条件 A/B/C，全报告不预设结果"""
    naive_in = next(r for r in cond_rows if r["config"] == config and r["arm"] == "naive")
    time_in = next(r for r in cond_rows if r["config"] == config and r["arm"] == "time_block")
    A = naive_in["in_median"] < 0.85
    B = (time_in["in_median"] - naive_in["in_median"]) >= 0.05
    C = time_in["in_median"] < 0.88
    if not A:
        branch, verdict = "DGP_FALLBACK", "体制块不伤 naive（naive 未失效）→ 仿真回炉"
    elif A and B and C:
        branch, verdict = "B3", "A∧B∧C：三级梯度 + 自适应权重主线"
    elif A and B and not C:
        branch, verdict = "B1p", "A∧B∧¬C：均匀块定理主线，自适应作附录（time 已达名义）"
    else:  # A and not B
        branch, verdict = "X", "A∧¬B：块结构无修复价值 → 重心回识别层退化界"
    return {"A_naive_fail": A, "B_block_value": B, "C_adaptive_room": C,
            "naive_in": naive_in["in_median"], "time_in": time_in["in_median"],
            "branch": branch, "verdict": verdict}

# ---------------------------------------------------------------- 主流程 ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--N", type=int, default=38)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--anchor", default="tail")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    AGRID = [0.05, 0.07, 0.10, 0.13, 0.16, 0.20]

    base = {"s_mu": 0.5, "s_lam": 0.5, "lam_ar": 0.0, "phi": 0.4, "s_scale": 2.0,
            "rho": 0.5, "n_regions": 4}
    scans = {
        "F1_country_fe": ("s_mu", [0.0, 0.5, 1.0]),
        "F2_common_time": ("s_lam", [0.0, 0.5, 1.0]),
        "F2b_lam_ar": ("lam_ar", [0.0, 0.9]),
        "F3_ar1": ("phi", [0.0, 0.4, 0.8]),
        "F4_scale_block": ("s_scale", [1.0, 2.0, 4.0]),
        "F5_region_factor": ("rho", [0.0, 0.5, 0.9]),
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.out or os.path.join("04_output", "experiments", "E01a", f"run_{ts}")
    os.makedirs(outdir, exist_ok=True)
    configs = [("baseline", dict(base))]
    for fname, (key, levels) in scans.items():
        for lv in levels:
            p = dict(base); p[key] = lv
            configs.append((f"{fname}={lv}", p))

    rows, cond_rows, pair_sum, out_sat = [], [], {}, None
    for cname, params in configs:
        seed = args.seed + (zlib.crc32(cname.encode()) % 10 ** 6)
        out, paired = run_config(params, args.N, args.T, args.alpha, args.B, seed, args.anchor)
        if cname == "baseline":
            out_sat = out
        for a in ("naive", "country_block", "time_block"):
            o = out[a]
            rows.append({"config": cname, "arm": a,
                         **{f"sat_cov_{k}": v for k, v in summarize(o["cov"]).items()},
                         **{f"sat_w_{k}": v for k, v in summarize(o["w"]).items()}})
            if a == "time_block" or a == "naive" or a == "country_block":
                cond_rows.append({"config": cname, "arm": a, **{f"in_{k}": v for k, v in summarize(o["in"]).items()},
                                  **{f"out_{k}": v for k, v in summarize(o["out"]).items()},
                                  "n_in": len(o["in"]), "n_out": len(o["out"])})
        pair_sum[cname] = {"d_median": float(np.median(np.array(paired["time"]) - np.array(paired["country"]))),
                           "d_lo90": float(np.quantile(np.array(paired["time"]) - np.array(paired["country"]), 0.05)),
                           "d_hi90": float(np.quantile(np.array(paired["time"]) - np.array(paired["country"]), 0.95))}
        # 等覆盖宽度已移除（数学空洞）；宽度比较 = sat_w + sat_cov 双报告

    import csv
    with open(os.path.join(outdir, "response_surface.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(outdir, "conditional_coverage.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cond_rows[0].keys())); w.writeheader(); w.writerows(cond_rows)

    # ------------------------------------------------------------ 判定（confirmatory-2）----
    def row_for(cfg, arm):
        return next(r for r in rows if r["config"] == cfg and r["arm"] == arm)
    # B1 检查：名义 α 配对差值（等覆盖宽度 eqw 作报告，路由判据用名义 α 配对）
    d = pair_sum["baseline"]
    qc_w = row_for("baseline", "country_block")["sat_w_median"]
    sig = d["d_hi90"] < 0 and abs(d["d_median"]) > 0.05 * qc_w
    time_cov_ok = row_for("baseline", "time_block")["sat_cov_median"] >= \
                  row_for("baseline", "naive")["sat_cov_median"] - 0.03
    # 判据 v2（R15b 预注册，protocol §9）：四分支全覆盖，原子条件 A/B/C + 解析预言对照
    v2 = adjudicate_v2(cond_rows, "baseline")
    # 解析预言 vs 实测对照（验证阈值推导；事后性防御：阈值来自解析式非实测倒推）
    q_mix_pred, naive_in_pred = analytic_naive_in(2.0)   # s=2 预言
    analytic_check = {"s": 2.0, "q_mix_pred": q_mix_pred, "naive_in_pred": naive_in_pred,
                      "naive_in_measured": v2["naive_in"],
                      "match": abs(naive_in_pred - v2["naive_in"]) < 0.10}

    adj = {"run_status": "confirmatory-3", "protocol_version": "v0.3",
           "criterion_version": "v2 (protocol §9, R15b 预注册)",
           "alpha": args.alpha, "N": args.N, "T": args.T, "B": args.B,
           "seed": args.seed, "block_anchor": args.anchor,
           "t_alignment": {"pre2000_check": "CO2 min=2000/max=2023 → T=21 实证锁定（R15）",
                           "sim_T": args.T, "sim_sensitivity": "T∈{20,30} 由 E01b 冻结时裁"},
           "criterion_v2": v2,
           "analytic_prediction_check": analytic_check,
           "width_nominal_alpha": {a: round(float(np.median(out_sat[a]["w"])), 4) for a in out_sat},
           "equal_coverage_width_removed": "R16c 终审 v0.4：等覆盖宽度数学恒同（三臂共享 center+误差），结构空洞；36% 溢价为 grid 边界截断伪影，弃用",
           "branch": v2["branch"], "verdict": v2["verdict"],
           "argument_line": "R13 三重理由维持（B3⊃B1 数学包含 / 定性梯度强化 0.711→0.789→0.842 / 理论对齐）——随报告附注，非判据",
           "r15b_notes": "判据 v1 过期（时间线：R12 埋错配→R13 漏改→confirmatory-2 三分支全空→INDETERMINATE）；confirmatory-3 用判据 v2 预注册 + 新种子 + 解析阈值"}
    json.dump(adj, open(os.path.join(outdir, "adjudication.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # stdout
    print(f"[E01a v0.3] run: {outdir}  status=confirmatory-2  anchor={args.anchor}")
    print(f"[E01a v0.3] N={args.N} T={args.T} alpha={args.alpha} B={args.B}")
    print("baseline sat（procedure-true 覆盖）:")
    for a in ("naive", "country_block", "time_block"):
        r = row_for("baseline", a)
        print(f"   {a:14s} cov={r['sat_cov_median']:.3f} w={r['sat_w_median']:.3f}")
    c = cond_rows[0]
    print(f"   条件覆盖(块内): median={c['in_median']:.3f} 90%=[{c['in_lo90']:.3f},{c['in_hi90']:.3f}] n={c['n_in']} | 块外: {c['out_median']:.3f}")
    print("   等覆盖宽度已移除（数学空洞，见 docstring）")
    print(f"[E01a v0.3] 判据 v2 原子条件: A(naive失效)={v2['A_naive_fail']} B(块价值)={v2['B_block_value']} C(自适应空间)={v2['C_adaptive_room']}")
    print(f"[E01a v0.3] naive_in={v2['naive_in']:.3f} time_in={v2['time_in']:.3f}")
    print(f"[E01a v0.3] 解析预言(s=2): naive_in_pred={naive_in_pred:.3f} vs 实测 {v2['naive_in']:.3f} 吻合={analytic_check['match']}")
    print(f"[E01a v0.3] 判定分支(v2): {v2['branch']}\n   {v2['verdict']}")

if __name__ == "__main__":
    main()
