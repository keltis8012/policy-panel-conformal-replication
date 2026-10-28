#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e01b_empirical.py —— E01b 实测管线（观测数据，红线 #7：禁称覆盖率）
版本：v1.0 ｜ 2026-08-28 ｜ 依据：protocol.md §2.1/§2.6/§2.7 + config.yaml e01b_decision_rule

管线：processed CO₂ 面板 + EPS raw → 39×21 面板（N=39, T=21 冻结）
     → 滚动起点三臂区间（naive/country_block/time_block，复用 E01a v0.3 校准器）
     → 指标：区间宽度（配对）/ 重叠度 / α-时点决策差异（提前·推迟）

决策规则（冻结，config e01b_decision_rule）：
  触发：|ΔEPS_{i,t}| ≥ τ，τ = 0.222（校准期 P75 实证）
  时间基：逐年差分 + 12 个月冷却期
  动作：{hold, adjust}
  区间调制：adjust ⇔ 触发 ∧ (e 区间下界 > 0)，e = -ΔlnCO2（减排量；记号钉死 §2.6）
  损失：主设定对称 λ=1；延迟成本报分布（主图第二指标）

用法：python e01b_empirical.py
输出：04_output/experiments/E01b/run_YYYYMMDD_HHMMSS/{panel.csv, arms_width.csv, overlap.csv, decision_timing.csv, e01b_report.json}
"""
import csv, os, sys, json, datetime, hashlib, warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore", "Mean of empty slice")  # nanmean 全 NaN 行/列的已知无害警告（valid 已过滤）

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "E01_three_arm_coverage"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC = os.path.join(ROOT, "02_datasets", "processed", "ghg_co2_panel.csv")
EPS_RAW = os.path.join(ROOT, "02_datasets", "raw", "oecd_eps_1990_2020.csv")
OUT = os.path.join(ROOT, "04_output", "experiments", "E01b")

# ---------- 冻结参数（config.yaml e01b_decision_rule v1.1 同步，勿改） ----------
ALPHAS = [0.05, 0.10, 0.20]   # α 扫描（骨架 #2 α-时点曲线，R16c 裁定）
TAU = 0.222          # 全样本 |ΔEPS| P75（外生政策显著性阈值；§2.8 文档化：39 国交集 × 2001-2020 差分
                     # 共 840 值；EPS 在 1/36 网格上取值，τ=0.222 恰压网格点，"≥"含边界为写死规则）
COOLDOWN = 2         # 触发年间隔 ≥ 2（24 个月冷却；R16c 裁定——"12 个月"在年度评估下结构性空转，
                     # 相邻评估天然相隔整 12 个月；first-come-first-served，keep-max 需未来信息不可实现）
Y0, Y1 = 2000, 2020
MIN_K = 5            # 滚动起点最小校准长度（与 E01a 一致）

def load_panel():
    """processed CO₂（ln 值矩阵）+ EPS（水平值矩阵），39 国交集，2000-2020"""
    co2 = {}
    with open(PROC, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            co2[(r["REF_AREA"], int(r["TIME_PERIOD"]))] = float(r["OBS_VALUE"])
    eps = {}
    with open(EPS_RAW, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["MEASURE"] == "POL_STRINGENCY" and r["OBS_VALUE"]:
                eps[(r["REF_AREA"], int(r["TIME_PERIOD"]))] = float(r["OBS_VALUE"])
    countries = sorted({c for (c, y) in co2 if Y0 <= y <= Y1} & {c for (c, y) in eps if Y0 <= y <= Y1})
    years = list(range(Y0, Y1 + 1))
    Y = np.full((len(countries), len(years)), np.nan)   # ln CO2
    S = np.full((len(countries), len(years)), np.nan)   # EPS
    for i, c in enumerate(countries):
        for j, y in enumerate(years):
            if (c, y) in co2:
                Y[i, j] = np.log(co2[(c, y)])
            if (c, y) in eps:
                S[i, j] = eps[(c, y)]
    return countries, years, Y, S

def rolling_pseudo_residuals_na(y, K=5):
    """E01b 专用 nan-aware 滚动起点一步伪残差。
    双因素去均值（two-way demeaning）标准形式：r = y − μ̂_i − λ̂_{t-1} + ĝ，
    其中 μ̂_i = 行均值（含水平）、λ̂_{t-1} = 最近校准年列均值（含水平）、ĝ = 校准窗全局均值。
    加回 ĝ 使残差为中心化偏差（R16b 修复：真实数据全局水平 ~12，缺失该加回项时残差被
    −(μ̄+λ̄) 污染至 −12；E01a 仿真全局均值≈0 故原式恰好成立）。
    返回 rr(N, n_years), yrs；缺失单元为 NaN，调用方过滤。"""
    N, Tcol = y.shape
    yrs = np.arange(K, Tcol)
    rr = np.full((N, len(yrs)), np.nan)
    for j, t in enumerate(yrs):
        seg = y[:, :t]
        with np.errstate(all="ignore"):
            mu_h = np.nanmean(seg, axis=1)
            lam_h = np.nanmean(seg, axis=0)
            g = np.nanmean(seg)
        valid = ~np.isnan(y[:, t]) & ~np.isnan(mu_h) & ~np.isnan(lam_h[-1])
        rr[valid, j] = y[valid, t] - (mu_h[valid] + lam_h[-1] - g)
    return rr, yrs

def three_arm_intervals(y_train, alpha, one_sided=False):
    """三臂校准区间（nan-aware；q 仅基于有效残差）。
    one_sided=True（v0.2.2 单边变体）：q = 单边分数 s 的分位（rr 本身，非 |rr|）——
    q_os ≤ q_sym 恒成立，同预算下放行集扩大（E01a 实测 q_os/q_sym=0.60-0.83）。"""
    rr, yrs = rolling_pseudo_residuals_na(y_train, MIN_K)
    valid_r = ~np.isnan(rr)
    with np.errstate(all="ignore"):
        mu_h = np.nanmean(y_train, axis=1)
        lam_h = np.nanmean(y_train, axis=0)
        g = np.nanmean(y_train)
    center = mu_h + lam_h[-1] - g   # 双因素预测中心（去水平：≈ μ_i + λ_{t-1}；缺 −g 时 ≈2G 水平污染）
    vv = rr[valid_r]
    q = {}
    if vv.size == 0:
        q = {"naive": np.nan, "time_block": np.nan, "country_block": np.nan}
        return q, center
    # 单边（v0.2.2）：q = 单边分数 s 的 (1−α) 分位（rr 本身，非 |rr|）——P(s>q) ≤ α 精确到 α；
    # 对称：|rr| 分位（P(s>q_sym) ≤ α 带 ~α/2 松弛，保守）
    base = vv if one_sided else np.abs(vv)
    q["naive"] = float(np.quantile(base, 1 - alpha))
    # 时间块：逐年跨国家分位数（列，有效单元）
    col_q = [np.quantile((rr[:, j] if one_sided else np.abs(rr[:, j]))[~np.isnan(rr[:, j])], 1 - alpha)
             for j in range(rr.shape[1]) if (~np.isnan(rr[:, j])).sum() > 0]
    q["time_block"] = float(np.quantile(col_q, 1 - alpha)) if col_q else np.nan
    # 国家块：逐国跨年分位数（行，有效单元）
    row_q = [np.quantile((rr[i, :] if one_sided else np.abs(rr[i, :]))[~np.isnan(rr[i, :])], 1 - alpha)
             for i in range(rr.shape[0]) if (~np.isnan(rr[i, :])).sum() > 0]
    q["country_block"] = float(np.quantile(row_q, 1 - alpha)) if row_q else np.nan
    return q, center

def main(baseline_mode="peer_med", one_sided=False):
    countries, years, Y, S = load_panel()
    N, T = len(countries), len(years)
    t0 = datetime.datetime.now()
    suf = "_ownhist" if baseline_mode == "own_hist" else ""
    if one_sided: suf += "_onesided"
    outdir = os.path.join(OUT, t0.strftime("%Y%m%d_%H%M%S") + suf)
    os.makedirs(outdir, exist_ok=True)

    # ---------- 面板完整性 ----------
    y_missing = int(np.isnan(Y).sum()); s_missing = int(np.isnan(S).sum())
    # 缺失单元：滚动起点跳过（缺单元不进残差集）；S 侧缺单元不参与触发
    valid_mask = ~np.isnan(Y)

    # ================================================================ v1.1 方案 B
    # 决策时点 = t 年初，全部输入移位到已实现信息（R16c 审查修复 v1.0 时序矛盾）：
    #   触发：|ΔEPS_{t−1}| = |S_{i,t−1} − S_{i,t−2}| ≥ τ（上年末已公布政策盘点）
    #   预测对象：ΔlnCO₂_t（校准窗 [2000, t−1]——现有管线零改动）
    #   目标线：med_{t−1} = 上年（t−1）真实横截面 ΔlnCO₂ 中位数（"以去年同伴表现为基准"）
    #   冷却期：触发年（t−1）与上次触发年差 ≥ COOLDOWN
    #   决策年标签 = t
    # α 扫描外层循环：每 α 独立输出，主图 α-时点曲线一次跑齐
    # ==========================================================================
    for alpha in ALPHAS:
        sub = os.path.join(outdir, f"alpha_{int(alpha*100):03d}")
        os.makedirs(sub, exist_ok=True)
        width = defaultdict(dict)     # year -> arm -> 宽度
        overlap = defaultdict(dict)   # year -> (a,b) -> 重叠度
        dec = []                      # 决策事件记录
        last_trigger_year = defaultdict(int)  # 冷却期：country -> 上次触发年（政策变动年）
        arms = ["naive", "time_block", "country_block"]

        for tj in range(MIN_K + 1, T):   # 决策年 t = 2006..2020
            y = Y0 + tj
            y_train = Y[:, :tj]       # [0, tj) 校准（t 年初可得）
            q, center = three_arm_intervals(y_train, alpha, one_sided=one_sided)
            prev = Y[:, tj - 1]; cur = Y[:, tj]
            valid = valid_mask[:, tj] & valid_mask[:, tj - 1]
            dtrue = cur - prev        # Δln CO2_t（真实，仅用于报告；med_{t−1} 由上一轮差给出）
            center_d = center - prev  # 中心预测的 Δ_t
            # 宽度（横截面配对）
            for a in arms:
                width[y][a] = 2.0 * q[a]
            # 重叠度（臂间，同一年）
            for ai in range(3):
                for aj in range(ai + 1, 3):
                    a, b = arms[ai], arms[aj]
                    lo = np.maximum(center_d - q[a], center_d - q[b])
                    hi = np.minimum(center_d + q[a], center_d + q[b])
                    ov = np.clip(hi - lo, 0, None) / min(2 * q[a], 2 * q[b])
                    overlap[y][f"{a}~{b}"] = float(np.nanmean(ov[valid]))
            # ---- v1.1 决策（方案 B）：触发基于 t−1，目标线 med_{t−1} ----
            # 信息时点断言（制度第 7 条可执行化）：触发只用 S[t−1],S[t−2]；med 只用 t−1 列——任何
            # 未来信息引用将在此被 assert 拦截（时点一致性从文本规格升级为可执行规格）
            assert tj - 2 >= 0, f"决策年 {y}: 触发需 S[t-2]，tj-2<0"
            # 目标线（R16c 验收变体化）：peer_med = 上年横截面中位数（冻结主设定，回溯框架声明 §2.8）；
            # own_hist = 自身历史中位（校准窗内该国 ΔlnCO₂ 中位数）——稳健性变体，检验'同伴基准
            # 分布效应'（放行集中新兴经济体）：自身基准下新兴国家应失去相对优势
            d_prev = Y[:, tj - 1] - Y[:, tj - 2]
            if baseline_mode == "peer_med":
                m_valid = d_prev[valid_mask[:, tj - 1] & valid_mask[:, tj - 2]]
                med_prev = float(np.nanmedian(m_valid)) if m_valid.size else 0.0
            else:  # own_hist：每国自身历史中位（校准窗内该国 Δ 的中位数，NaN 跳过）
                med_prev = 0.0
                for i, c in enumerate(countries):
                    hist = d_prev[valid_mask[:, tj - 1] & valid_mask[:, tj - 2]]  # 占位，下方逐国覆盖
                    break
                med_prev = None  # 逐国在决策循环内计算（见 e_lo 段）
            for i, c in enumerate(countries):
                # 触发：|ΔEPS_{t−1}| = |S_{i,t−1} − S_{i,t−2}| ≥ τ（t 年初可得；assert 保证不用 S_t）
                if np.isnan(S[i, tj - 1]) or np.isnan(S[i, tj - 2]):
                    continue
                deps = S[i, tj - 1] - S[i, tj - 2]
                if abs(deps) < TAU:
                    continue
                trig_year = y - 1  # 政策变动年 = t−1
                if trig_year - last_trigger_year[c] < COOLDOWN:  # 触发年间隔 ≥ 2（24 个月冷却）
                    continue
                last_trigger_year[c] = trig_year
                # 三臂各自判定：e = med_{t−1} − ΔlnCO₂_t；调整 ⇔ e 区间下界 > 0
                # 代数（§2.8 方案 B 展开）：e_lo = med_prev − (center_d[i] + q[a])
                #   → adjust ⇔ center_d + q < med_prev（整个预测区间低于上年中位基准）
                # own_hist：med = 该国自身历史 ΔlnCO₂ 中位数（校准窗 [0,tj-1] 内）
                if baseline_mode == "own_hist":
                    seg_d = Y[i, :tj] - np.roll(Y[i, :tj], 1)
                    seg_d = seg_d[1:][~np.isnan(seg_d[1:])]
                    med_i = float(np.median(seg_d)) if seg_d.size else 0.0
                else:
                    med_i = med_prev
                for a in arms:
                    e_lo = med_i - (center_d[i] + q[a])
                    adj = bool(e_lo > 0.0) if valid[i] else None
                    dec.append({
                        "country": c, "trigger_year": y, "deps": round(deps, 3),
                        "arm": a, "adjust": adj,
                        "e_lo": round(float(e_lo), 4) if valid[i] else None,
                        "med_prev": round(med_i, 4),
                    })

        # ---------- 决策时点（延迟成本：决策年 t 即放行 = 0）----------
        dec_by_country_year = {}
        for d in dec:
            dec_by_country_year[(d["country"], d["trigger_year"], d["arm"])] = d
        delay = {a: [] for a in arms}
        for (c, ty, a), d in dec_by_country_year.items():
            if d["adjust"] is None:
                continue
            if d["adjust"]:
                delay[a].append(0)   # v1.1 语义：0 = 决策年 t 即放行
            else:
                found = None
                for fy in range(ty + 1, min(ty + 4, Y1 + 1)):
                    k = (c, fy, a)
                    if k in dec_by_country_year and dec_by_country_year[k]["adjust"]:
                        found = fy; break
                delay[a].append((found - ty) if found else 99)  # 99 = 后续 3 年内无新触发转正
        # 早期退化检测
        qdiff_years = []
        for y in sorted(width):
            wvals = [width[y][a] for a in arms]
            if max(wvals) - min(wvals) < 0.01 * np.mean(wvals):
                qdiff_years.append(y)
        n_adj_by_year = {str(y): {a: sum(1 for d in dec if d["trigger_year"] == y and d["arm"] == a and d["adjust"] is True)
                                  for a in arms} for y in sorted(set(d["trigger_year"] for d in dec))}
        # ---------- 落盘（每 α 子目录）----------
        with open(os.path.join(sub, "panel.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["country"] + [str(y) for y in years])
            for i, c in enumerate(countries):
                w.writerow([c] + ["" if np.isnan(Y[i, j]) else round(float(Y[i, j]), 4) for j in range(T)])
        with open(os.path.join(sub, "arms_width.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["year", "arm", "width"])
            for y in width:
                for a, v in width[y].items():
                    w.writerow([y, a, round(v, 4)])
        with open(os.path.join(sub, "overlap.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["year", "pair", "overlap"])
            for y in overlap:
                for p, v in overlap[y].items():
                    w.writerow([y, p, round(v, 4)])
        with open(os.path.join(sub, "decision_timing.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["country", "trigger_year", "deps", "arm", "adjust", "e_lo", "med_prev"])
            w.writeheader()
            for d in sorted(dec, key=lambda x: (x["trigger_year"], x["country"], x["arm"])):
                w.writerow(d)
        # 报告
        w_med = {a: float(np.median([width[y][a] for y in width])) for a in arms}
        ov_tn = float(np.nanmean([overlap[y]["naive~time_block"] for y in overlap])) if overlap else float("nan")
        rep = {
            "date": t0.strftime("%Y%m%d_%H%M%S"), "run_status": "confirmatory",
            "protocol_version": "v1.1（protocol §2.8 方案 B）",
            "alpha": alpha, "tau": TAU, "N": N, "T": T,
            "panel_completeness": {"co2_missing": y_missing, "eps_missing": s_missing,
                                   "valid_cells": int(valid_mask.sum())},
            "arms_median_width": w_med,
            "width_ratio": {"time_vs_naive": round(w_med["time_block"] / w_med["naive"], 3),
                            "country_vs_naive": round(w_med["country_block"] / w_med["naive"], 3)},
            "overlap": {"time~naive_mean": round(ov_tn, 3)},
            "decision": {a: {"n_trigger_events": sum(1 for d in dec if d["arm"] == a),
                             "n_na": sum(1 for d in dec if d["arm"] == a and d["adjust"] is None),
                             "n_valid": sum(1 for d in dec if d["arm"] == a and d["adjust"] is not None),
                             "n_adjust": sum(1 for d in dec if d["arm"] == a and d["adjust"] is True),
                             "n_hold": sum(1 for d in dec if d["arm"] == a and d["adjust"] is False),
                             "release_rate": round(sum(1 for d in dec if d["arm"] == a and d["adjust"] is True)
                                                   / sum(1 for d in dec if d["arm"] == a and d["adjust"] is not None), 4)
                             if sum(1 for d in dec if d["arm"] == a and d["adjust"] is not None) else None,
                             "delay_dist": {k: int(v) for k, v in
                                            sorted(defaultdict(int, {str(x): sum(1 for z in delay[a] if z == x)
                                            for x in set(delay[a])}).items())}}
                         for a in arms},
            "note": ("观测数据：无覆盖字样（红线 #7）；v1.1 方案 B 时序自洽（触发 t−1 / 预测 t / 目标线 "
                     "med_{t−1}，全部 t 年初可得，信息时点断言在代码层强制）；冷却期 gap≥2（24 个月，R16c 裁定）；"
                     "右缘删失：2020 年政策变化永不评估（决策年 2021 出面板）——因果时序规则的固有代价；"
                     "α-时点曲线语义：α 只进区间调制、不进触发（同一事件集，仅放行判定随 α 变）"),
            "right_censoring": {"note": "2020 年 ΔEPS 触发无决策年 2021 评估（面板止于 2020）——右缘删失，如实注记"},
            "early_years_degration": {"years_q_converged": qdiff_years,
                                      "explanation": "滚动起点设计：预测年 tj 的校准残差列数 = tj − MIN_K；早期年残差仅 1-3 列，三臂 q 趋同退化——属信息退化非 bug，报告分层呈现"},
            "decision_by_year": n_adj_by_year,
        }
        json.dump(rep, open(os.path.join(sub, "e01b_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[E01b v1.1 α={alpha}] 三臂中位宽度: naive={w_med['naive']:.4f} time={w_med['time_block']:.4f} country={w_med['country_block']:.4f} (time/naive={rep['width_ratio']['time_vs_naive']})")
        for a in arms:
            d = rep["decision"][a]
            print(f"[E01b v1.1 α={alpha}] {a:12s} 触发={d['n_trigger_events']} 有效={d['n_valid']} NA={d['n_na']} 调整={d['n_adjust']} 持有={d['n_hold']} 放行率={d['release_rate']} 延迟={d['delay_dist']}")
    print(f"[E01b] 输出: {outdir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--own-hist", action="store_true", help="自身历史基准线变体（R16c 验收：同伴基准分布效应检验）")
    ap.add_argument("--one-sided", action="store_true", help="单边校准变体（v0.2.2 升格必跑：q_os 效率红利定量，预期放行率上升/臂序不变）")
    a = ap.parse_args()
    main(baseline_mode="own_hist" if a.own_hist else "peer_med", one_sided=a.one_sided)
