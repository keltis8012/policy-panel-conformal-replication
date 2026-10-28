#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e01a_c10e_sim.py —— C10-empirical（C11 桥接段升实证级：实证趋势尺度上的交互符号直接测量）
版本：v1.0 ｜ 2026-08-28 ｜ 判据 C10E-v0.1（预注册冻结：C10_prereg.md §8，红线 #6/#10/#11/#13）
run_status: exploratory（判据首次应用）；确认版 = --seed-base 20260829 + 判据零改动

设计：DGP = gen_panel_f6e（β = a·z 实证趋势向量 + F4 尺度块 + F5 ρ=0.5），
z = F6E 冻结 39 国向量（f6e_20260828_123830 同源同序，不重算）。
网格：(a, s) ∈ {0, 0.2, 0.4} × {2, 4}，B=500，naive 臂块内条件覆盖（bf[T] 实现筛选）。
分解：ε^tg(a,s) = [D(a,s) − D(0,s)] − ε_grp^F6E(a)，ε_grp 输入 = F6E 冻结估计版实测点
  （a=0.2 → 0.0000，a=0.4 → 0.0263）——防同格自算恒等（C10 §5 同构）。
判读：T1 符号（a=0.4, s=4，四格预案）+ T2 端点消失（a=0.2）+ T3 机制方向（非门控）
  + 实现守卫（s=2 标定一致性格）+ δ_C10E 种子噪声标尺（a=0 双种子）。
用法：python e01a_c10e_sim.py [--B 500] [--seed-base 20260828]
输出：04_output/experiments/E01a/c10e_<ts>/ {c10e_grid.csv, adjudication.json}
"""
import argparse, json, os, sys, csv, datetime, zlib
import numpy as np

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "03_program", "experiments", "E01_three_arm_coverage"))
sys.path.insert(0, os.path.join(ROOT, "03_program", "analysis"))
from e01a_sim import fit_arms, summarize
from e01a_f6e_sim import empirical_z, gen_panel_f6e
from e01b_empirical import load_panel

A_GRID = [0.0, 0.2, 0.4]
S_GRID = [2.0, 4.0]
# F6E 冻结输入（f6e_20260828_123830 "eps_g" 字段，估计版实测点；溯源表 1b-11）
EPS_G_F6E = {0.2: 0.0000, 0.4: 0.0263}


def run_cell(a, s, z, hi_mask, lo_mask, N, T, alpha, B, seed):
    """一格：B 次重复，整体/高|z|组/低|z|组块内覆盖（目标年入块实现筛选）。
    β = a·z 固定（非随机）→ hi/lo 组为固定国家集合，跨重复不变。"""
    rng = np.random.default_rng(seed)
    beta = a * z
    params = (0.5, 0.5, 0.0, 0.4, s, 0.5)   # s_mu, s_lam, lam_ar, phi, s_scale, rho
    agg = {k: [] for k in ("all", "hi", "lo")}
    for _ in range(B):
        y, bf = gen_panel_f6e(N, T, params, beta, rng, block_anchor="tail")
        if not bf[T]:
            continue
        y_train = y[:, :T]
        q, center, rr, _ = fit_arms(y_train, alpha)
        e = np.abs(y[:, T] - center)
        agg["all"].append(float(np.mean(e <= q["naive"])))
        agg["hi"].append(float(np.mean(e[hi_mask] <= q["naive"])))
        agg["lo"].append(float(np.mean(e[lo_mask] <= q["naive"])))
    return {"all": summarize(agg["all"]), "hi": summarize(agg["hi"]),
            "lo": summarize(agg["lo"]), "n_in": len(agg["all"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--seed-base", type=int, default=20260828)
    ap.add_argument("--run-status", type=str, default="exploratory")
    args = ap.parse_args()
    N, T, alpha = 38, 24, 0.10
    base = args.seed_base
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "04_output", "experiments", "E01a", f"c10e_{ts}")
    os.makedirs(outdir, exist_ok=True)

    # ── F6E 冻结 z 向量（39 国 → 前 38 与 N 对齐）+ 固定 hi/lo 组
    countries, years, Y, S = load_panel()
    z_all, _ = empirical_z(countries, Y, years)
    z = z_all[:N]
    hi_mask = z >= np.quantile(z, 0.75)
    lo_mask = z <= np.quantile(z, 0.25)

    # ── 主网格 6 格
    grid = {}
    for s in S_GRID:
        for a in A_GRID:
            cname = f"C10E_a={a}_s={s}"
            seed = base + (zlib.crc32(cname.encode()) % 10 ** 6)
            res = run_cell(a, s, z, hi_mask, lo_mask, N, T, alpha, args.B, seed)
            grid[(a, s)] = res
            print(f"[C10E] a={a} s={s}: all_in={res['all']['median']:.3f} "
                  f"hi={res['hi']['median']:.3f} lo={res['lo']['median']:.3f} n={res['n_in']}")

    # ── δ_C10E 种子噪声标尺（a=0 双种子，同 C10 §3 构造）
    noise = {}
    for s in S_GRID:
        ca, cb = f"C10E_a=0.0_s={s}_a", f"C10E_a=0.0_s={s}_b"
        da = 1 - alpha - run_cell(0.0, s, z, hi_mask, lo_mask, N, T, alpha, args.B,
                                  base + (zlib.crc32(ca.encode()) % 10 ** 6))["all"]["median"]
        db = 1 - alpha - run_cell(0.0, s, z, hi_mask, lo_mask, N, T, alpha, args.B,
                                  base + (zlib.crc32(cb.encode()) % 10 ** 6))["all"]["median"]
        noise[s] = abs(da - db)
    delta = max(max(noise.values()), 0.02)

    # ── 分解与判读（C10E-v0.1 冻结带）
    def D(a, s):
        return (1 - alpha) - grid[(a, s)]["all"]["median"]

    def Deps(a, s):   # 高|z|组缺口 − 低|z|组缺口（机制方向量）
        return ((1 - alpha) - grid[(a, s)]["hi"]["median"]) - ((1 - alpha) - grid[(a, s)]["lo"]["median"])

    etg = {}
    for a in (0.2, 0.4):
        for s in S_GRID:
            etg[(a, s)] = (D(a, s) - D(0.0, s)) - EPS_G_F6E[a]

    # ── 判读（C10E-v0.1.1 映射：ε^tg ≤ 0 = 负交互 = 加性保守 = 定理有利；changelog 见 prereg §8.1）
    t1 = etg[(0.4, 4.0)]
    if t1 <= -delta:
        t1_route = "PASS_CONSERVATIVE"
    elif t1 <= 0:
        t1_route = "PASS_WITHIN_NOISE"
    elif t1 <= delta:
        t1_route = "INDETERMINATE"
    else:
        t1_route = "FLAG_POSITIVE"
    # ── T2 判读域 = 检验格 s=4（v0.1.2：守卫格对符号零判别力——期望信号 ≈ 冻结输入噪声，prereg §8.2）
    t2 = {s: etg[(0.2, s)] for s in S_GRID}
    v4 = t2[4.0]
    if abs(v4) <= delta:
        t2_route = "VANISHING_CONFIRMED"
    elif v4 <= -delta:
        t2_route = "NEGATIVE_NOT_VANISHING"
    else:
        t2_route = "POSITIVE_ENDPOINT_FLAG"
    t3_val = (Deps(0.4, 4.0), Deps(0.4, 2.0))
    t3_route = "CONSISTENT" if t3_val[0] <= t3_val[1] + delta else "INCONSISTENT"
    guard = {a: round((D(a, 2.0) - D(0.0, 2.0)) - EPS_G_F6E[a], 4) for a in (0.2, 0.4)}
    guard_route = "GUARD_WITHIN_2STEPS" if all(abs(v) <= 2 * delta for v in guard.values()) else "GUARD_ALERT"

    if t1_route in ("PASS_CONSERVATIVE", "PASS_WITHIN_NOISE") and t2_route != "POSITIVE_ENDPOINT_FLAG":
        overall = "SIGN_SUPPORTED"
    elif t1_route != "FLAG_POSITIVE" and t2_route != "POSITIVE_ENDPOINT_FLAG":
        overall = "PARTIAL"
    else:
        overall = "REWORK"

    # ── 落盘
    with open(os.path.join(outdir, "c10e_grid.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "s_scale", "all_in_med", "hi_in_med", "lo_in_med", "n_in",
                    "D_all", "eps_tg"])
        for (a, s), r in sorted(grid.items()):
            e = round(etg[(a, s)], 4) if a > 0 else ""
            w.writerow([a, s, round(r["all"]["median"], 4), round(r["hi"]["median"], 4),
                        round(r["lo"]["median"], 4), r["n_in"], round(D(a, s), 4), e])
    adj = {
        "run_status": args.run_status,
        "criterion_version": "C10E-v0.1.2（C10_prereg.md §8 + §8.1/§8.2 changelog：v0.1.1 方向映射反转修正 + v0.1.2 T2 判读域限定检验格 s=4；统计量冻结零改动）",
        "seed_base": base, "N": N, "T": T, "alpha": alpha, "B": args.B,
        "eps_g_f6e_frozen": EPS_G_F6E,
        "seed_noise_scale": {str(s): round(v, 4) for s, v in noise.items()},
        "delta_C10E": round(delta, 4),
        "D_grid": {f"a={a}_s={s}": round(D(a, s), 4) for (a, s) in sorted(grid)},
        "eps_tg": {f"a={a}_s={s}": round(etg[(a, s)], 4) for (a, s) in sorted(etg)},
        "T1_sign_a0.4_s4": {"eps_tg": round(t1, 4), "delta": round(delta, 4), "route": t1_route},
        "T2_endpoint_a0.2": {str(s): round(v, 4) for s, v in t2.items()} | {"route": t2_route},
        "T3_mechanism": {"deps_s4": round(t3_val[0], 4), "deps_s2": round(t3_val[1], 4),
                         "route": t3_route,
                         "note": "非门控方向检查；饱和 clamp 允许（v0.4.3 饱和区分教训）"},
        "guard_calibration_s2": {f"a={a}": round(v, 4) for a, v in guard.items()} | {"route": guard_route},
        "overall": overall,
        "verdict_map": {"SIGN_SUPPORTED": "桥接段承重主张（面板趋势尺度附近 ε^tg ≤ 0，加性保守）实证锚定；端点消失精化如实降回论证级",
                        "PARTIAL": "如实分层：过腿实证级、未过腿维持论证级",
                        "REWORK": "桥接段重构；应用句退幅值版 |ε_tg| ≤ 2·ε_grp ≈ 0.053 兜底"},
        "boundary_note": "只测 a ∈ {0.2, 0.4} + 端点，不证明 ε_tg ≤ 0 于整个 (0, 0.4]；夹逼仍需单调/结构论证（C6E 单调 + C10 域内负号），本跑提供夹逼两端的测量锚",
    }
    json.dump(adj, open(os.path.join(outdir, "adjudication.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[C10E] δ_C10E={delta:.4f}（noise={ {str(s): round(v,4) for s,v in noise.items()} }）")
    print(f"[C10E] ε^tg: { {f'a={a},s={s}': round(etg[(a,s)],4) for (a,s) in sorted(etg)} }")
    print(f"[C10E] T1={t1_route}  T2={t2_route}  T3={t3_route}  guard={guard_route} → {overall}")
    print(f"[out] {outdir}")


if __name__ == "__main__":
    main()
