#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aux_run.py —— 辅助佐证 M1（安慰剂）/ M2（子指数触发）/ M7s（分层），预注册 PREREG_v0.1.md
首跑 exploratory。统计纪律：二项 CI、2σ 带、Fisher 精确。
"""
import csv, io, os, sys, json, datetime
import numpy as np

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROG = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROG, "analysis"))
sys.path.insert(0, os.path.join(PROG, "experiments", "E01_three_arm_coverage"))
from e01b_empirical import load_panel, three_arm_intervals, rolling_pseudo_residuals_na, TAU, COOLDOWN, MIN_K, Y0
from scipy.stats import fisher_exact

RAW_GHG = os.path.join(ROOT, "02_datasets", "raw", "oecd_ghg_co2_2000_2023.csv")
RAW_EPS = os.path.join(ROOT, "02_datasets", "raw", "oecd_eps_1990_2020.csv")
OUT = os.path.join(ROOT, "04_output", "experiments", "SUPPORT")
ALPHA = 0.10
PARTNERS = {"BRA", "CHN", "IND", "IDN", "ZAF"}


def binom_ci(k, n):
    if n == 0:
        return None, None
    se = ((k / n) * (1 - k / n) / n) ** 0.5
    from scipy.stats import norm
    z = norm.ppf(0.975)
    return max(0.0, k / n - z * se), min(1.0, k / n + z * se)


def scan_ghg(pollutant):
    out = {}
    with io.open(RAW_GHG, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if (r["POLLUTANT"] == pollutant and r["MEASURE"] == "_T"
                    and r["UNIT_MEASURE"] == "T_CO2E" and r["OBS_VALUE"]):
                try:
                    y = int(r["TIME_PERIOD"])
                except ValueError:
                    continue
                if Y0 <= y <= 2020:
                    out.setdefault(r["REF_AREA"], {})[y] = float(r["OBS_VALUE"])
    return out


def build_panel(data, countries, years):
    Y = np.full((len(countries), len(years)), np.nan)
    for i, c in enumerate(countries):
        for j, y in enumerate(years):
            v = data.get(c, {}).get(y)
            if v:
                Y[i, j] = np.log(v)
    return Y


def run_rule(Y, S, countries, years, tau=TAU, alpha=ALPHA):
    """E01b v1.1 方案 B 同构；dec 携带两种 r：
    r_main = 校准窗内滚动伪残差末列（rolling_pseudo_residuals_na(Y[:,:tj]) 最后一列 = 年 tj−1，
             v0.2.3 口径 A 冻结配方——与主文 α_eff=0.106 可比）；r_oos = 决策年 tj 的中心化残差
             （样本外诊断量，非冻结指标，仅标签 oos 报告）"""
    T = len(years)
    valid_mask = ~np.isnan(Y)
    dec, last_trig = [], {c: -99 for c in countries}
    for tj in range(MIN_K + 1, T):
        y_train = Y[:, :tj]
        q, center = three_arm_intervals(y_train, alpha)
        rr_cal, _ = rolling_pseudo_residuals_na(y_train, MIN_K)
        r_main_col = rr_cal[:, -1] if rr_cal.shape[1] else np.full(len(countries), np.nan)
        prev, cur = Y[:, tj - 1], Y[:, tj]
        valid = valid_mask[:, tj] & valid_mask[:, tj - 1]
        center_d = center - prev
        if tj >= 2:
            d_prev = Y[:, tj - 1] - Y[:, tj - 2]
            mv = d_prev[valid_mask[:, tj - 1] & valid_mask[:, tj - 2]]
            med_prev = float(np.nanmedian(mv)) if np.any(np.isfinite(mv)) else 0.0
        else:
            med_prev = 0.0
        for i, c in enumerate(countries):
            if np.isnan(S[i, tj - 1]) or np.isnan(S[i, tj - 2]):
                continue
            deps = S[i, tj - 1] - S[i, tj - 2]
            if abs(deps) < tau:
                continue
            trig_year = years[tj] - 1
            if trig_year - last_trig[c] < COOLDOWN:
                continue
            last_trig[c] = trig_year
            for a in ("naive", "time_block", "country_block"):
                e_lo = med_prev - (center_d[i] + q[a])
                r_main = float(r_main_col[i]) if np.isfinite(r_main_col[i]) else None
                r_oos = (float(cur[i] - center[i])
                         if (valid[i] and np.isfinite(cur[i]) and np.isfinite(center[i])) else None)
                dec.append({"country": c, "trigger_year": years[tj], "arm": a,
                            "adjust": (bool(e_lo > 0) if valid[i] else None),
                            "e_lo": float(e_lo), "med_prev": med_prev,
                            "r": r_main, "r_oos": r_oos,
                            "q_naive_used": float(q["naive"])})
    return dec


def fps_from_levels(dec, levels, Ylog=None, countries=None, years=None):
    """FP = naive 放行 ∧ Δln ≥ med_prev（**log 尺度**，与 m4_fp_profiles 主配方一致）；
    Ylog 优先（主面板），否则由 levels 水平取 log"""
    yidx = {c: i for i, c in enumerate(countries)} if countries else {}
    yj = {y: j for j, y in enumerate(years)} if years else {}
    for d in dec:
        if d["arm"] != "naive":
            continue
        if d["adjust"] is not True:
            d["fp"] = None
            continue
        c, ty = d["country"], d["trigger_year"]
        if Ylog is not None and c in yidx and ty in yj and ty - 1 in yj:
            i, j, jm = yidx[c], yj[ty], yj[ty - 1]
            d["fp"] = (bool(Ylog[i, j] - Ylog[i, jm] >= d["med_prev"])
                       if np.isfinite(Ylog[i, j]) and np.isfinite(Ylog[i, jm]) else None)
        else:
            seq = levels.get(c, {})
            d["fp"] = (bool(np.log(seq[ty]) - np.log(seq[ty - 1]) >= d["med_prev"])
                       if ty in seq and ty - 1 in seq and seq[ty] > 0 and seq[ty - 1] > 0 else None)
    return dec


def band(dec, arm="naive"):
    """α_eff = P(r > q_rule | 触发)（**带符号上尾超越**，v0.2.3 口径 A 权威配方复现，
    与主文 α_eff=0.106 可比）+ 放行带。changelog：PREREG 原文"|r| 超越率"系未对照
    m4_backtest_normalize.py 的失准措辞，首跑废案（CO₂ 对照未能复现 0.106），
    unblinding 前修正为带符号配方，红线 #11 changelog 记录。"""
    n_ex = n_tot = 0
    for d in dec:
        if d["arm"] == arm and d["r"] is not None:
            n_tot += 1
            if d["r"] > d["q_naive_used"]:
                n_ex += 1
    a_eff = n_ex / n_tot if n_tot else float("nan")
    lo, hi = binom_ci(n_ex, n_tot)
    n_rel = sum(1 for d in dec if d["arm"] == arm and d["adjust"] is True and d.get("fp") is not None)
    n_fp = sum(1 for d in dec if d["arm"] == arm and d["adjust"] is True and d.get("fp") is True)
    exp_fp = n_rel * a_eff if n_rel and np.isfinite(a_eff) else float("nan")
    resid = n_fp - exp_fp if np.isfinite(exp_fp) else float("nan")
    return {"alpha_eff": None if not np.isfinite(a_eff) else round(a_eff, 4),
            "alpha_eff_ci": [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)],
            "n_exceed": n_ex, "n_trig_resid": n_tot,
            "in_band": (bool(abs(a_eff - ALPHA) <= 2 * (ALPHA * (1 - ALPHA) / n_tot) ** 0.5)
                        if n_tot and np.isfinite(a_eff) else None),
            "n_rel": n_rel, "n_fp": n_fp,
            "exp_fp": None if not np.isfinite(exp_fp) else round(exp_fp, 3),
            "resid": None if not np.isfinite(resid) else round(resid, 3),
            "flag": (bool(resid > 1) if np.isfinite(resid) else None),
            "fp_rate_rel": round(n_fp / n_rel, 4) if n_rel else None}


def trend_map(countries):
    _, _, Yc, _ = load_panel()
    idx = {c: i for i, c in enumerate(countries)}
    d = []
    for c in countries:
        i = idx[c]
        vals = [Yc[i, t] - Yc[i, t - 1] for t in range(1, Yc.shape[1])
                if np.isfinite(Yc[i, t]) and np.isfinite(Yc[i, t - 1])]
        d.append(np.mean(vals) if vals else np.nan)
    d = np.array(d)
    med = np.nanmedian(np.abs(d))
    iqr = np.nanpercentile(np.abs(d), 75) - np.nanpercentile(np.abs(d), 25)
    z = (np.abs(d) - med) / iqr if iqr > 0 else np.zeros_like(d)
    return {c: (float("nan") if np.isnan(z[k]) else float(z[k])) for k, c in enumerate(countries)}


ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join(OUT, ts)
os.makedirs(outdir, exist_ok=True)
A = {"date": ts, "run_status": "exploratory", "prereg": "PREREG_v0.1.md", "sections": {}}
countries, years, Yco2, S = load_panel()
zmap = trend_map(countries)
co2_levels = {c: {y: float(np.exp(Yco2[i, j])) for j, y in enumerate(years) if np.isfinite(Yco2[i, j])}
              for i, c in enumerate(countries)}

# ============ M1 安慰剂 ============
print("=== M1 placebo ===")
m1 = {}
# CO2 对照（主设定同口径）
dec = run_rule(Yco2, S, countries, years)
dec = fps_from_levels(dec, co2_levels, Ylog=Yco2, countries=countries, years=years)
b = band(dec)
fps = [d for d in dec if d["arm"] == "naive" and d.get("fp") is True]
m1["CO2_ref"] = {"band": b, "fp_list": [f"{d['country']}{str(d['trigger_year'])[2:]}" for d in fps],
                 "fp_mean_z": round(float(np.nanmean([zmap[d["country"]] for d in fps])), 3) if fps else None}
print(f"  CO2_ref: rel={b['n_rel']} fp={b['n_fp']} a_eff={b['alpha_eff']} flag={b['flag']} fp_z={m1['CO2_ref']['fp_mean_z']}")
for pol in ("CH4", "N2O", "GHG"):
    data = scan_ghg(pol)
    Yp = build_panel(data, countries, years)
    dec = fps_from_levels(run_rule(Yp, S, countries, years), data, Ylog=Yp)
    b = band(dec)
    fps = [d for d in dec if d["arm"] == "naive" and d.get("fp") is True]
    topq = np.nanquantile([v for v in zmap.values() if np.isfinite(v)], 0.75)
    sec = {"band": b, "fp_list": [f"{d['country']}{str(d['trigger_year'])[2:]}" for d in fps],
           "fp_mean_z": round(float(np.nanmean([zmap[d["country"]] for d in fps])), 3) if fps else None,
           "fp_topQ_share": (round(sum(1 for d in fps if zmap.get(d["country"], -9) >= topq) / len(fps), 3)
                             if fps else None),
           "n_trigger_events": sum(1 for d in dec if d["arm"] == "naive")}
    m1[pol] = sec
    print(f"  {pol}: events={sec['n_trigger_events']} rel={b['n_rel']} fp={b['n_fp']} "
          f"a_eff={b['alpha_eff']} flag={b['flag']} fp_z={sec['fp_mean_z']}")
# 特异性判定（冻结判据）
if m1["CH4"]["band"]["flag"] is not None:
    m1["specificity_verdict"] = {
        "co2_flag": m1["CO2_ref"]["band"]["flag"],
        "placebo_flags": {p: m1[p]["band"]["flag"] for p in ("CH4", "N2O", "GHG")},
        "co2_fp_z": m1["CO2_ref"]["fp_mean_z"],
        "placebo_fp_z": {p: m1[p]["fp_mean_z"] for p in ("CH4", "N2O", "GHG")},
    }
A["sections"]["M1"] = m1

# ============ M2 子指数触发 ============
print("=== M2 subindex ===")
eps_levels = {}
with io.open(RAW_EPS, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("OBS_VALUE") and r.get("CLIM_POL") in ("EPS_MKT", "EPS_NMKT", "TAXCO2"):
            eps_levels.setdefault(r["CLIM_POL"], {}).setdefault(r["REF_AREA"], {})[int(r["TIME_PERIOD"])] = float(r["OBS_VALUE"])
m2 = {}
for sub in ("EPS_MKT", "EPS_NMKT", "TAXCO2"):
    S2 = np.full_like(S, np.nan)
    for i, c in enumerate(countries):
        for j, y in enumerate(years):
            v = eps_levels.get(sub, {}).get(c, {}).get(y)
            if v is not None:
                S2[i, j] = v
    diffs = np.abs(np.diff(S2, axis=1))
    diffs = diffs[np.isfinite(diffs)]
    tau_sub = round(float(np.quantile(diffs, 0.75)), 3)
    entry = {"tau_p75": tau_sub}
    if tau_sub == 0:
        # 预注册 P75 规则退化（>75% 差分为零：稀疏工具全样本不动）——红线 #13 如实记录，
        # P90 变体补充（changelog：规则退化非结果适配）
        tau90 = round(float(np.quantile(diffs, 0.90)), 3)
        dec = run_rule(Yco2, S2, countries, years, tau=tau90)
        dec = fps_from_levels(dec, co2_levels, Ylog=Yco2, countries=countries, years=years)
        b = band(dec)
        entry.update({"degenerate_p75": True, "tau_p90_used": tau90,
                      "n_trigger_events": sum(1 for d in dec if d["arm"] == "naive"), "band": b,
                      "note": "P75=0（稀疏工具，全触发退化）→ P90 变体，exploratory 级"})
        print(f"  {sub}: P75=0 DEGENERATE → P90={tau90} events={entry['n_trigger_events']} "
              f"rel={b['n_rel']} fp={b['n_fp']} a_eff={b['alpha_eff']}")
    else:
        dec = run_rule(Yco2, S2, countries, years, tau=tau_sub)
        dec = fps_from_levels(dec, co2_levels, Ylog=Yco2, countries=countries, years=years)
        b = band(dec)
        entry.update({"n_trigger_events": sum(1 for d in dec if d["arm"] == "naive"), "band": b})
        print(f"  {sub}: tau={tau_sub} events={entry['n_trigger_events']} rel={b['n_rel']} "
              f"fp={b['n_fp']} a_eff={b['alpha_eff']}")
    m2[sub] = entry
A["sections"]["M2"] = m2

# ============ M7s 分层 ============
print("=== M7s stratified ===")
m7 = {}
for name, sel in (("OECD", lambda c: c not in PARTNERS), ("PARTNER", lambda c: c in PARTNERS)):
    keep = [k for k, c in enumerate(countries) if sel(c)]
    cs = [countries[k] for k in keep]
    dec = fps_from_levels(run_rule(Yco2[keep, :], S[keep, :], cs, years),
                          co2_levels, Ylog=Yco2, countries=countries, years=years)
    b = band(dec)
    fps = [d["country"] + str(d["trigger_year"])[2:] for d in dec
           if d["arm"] == "naive" and d.get("fp") is True]
    m7[name] = {"n_countries": len(cs), "band": b, "fp_list": fps[:15]}
    print(f"  {name}: n={len(cs)} rel={b['n_rel']} fp={b['n_fp']} a_eff={b['alpha_eff']} flag={b['flag']}")
A["sections"]["M7s"] = m7

json.dump(A, io.open(os.path.join(outdir, "adjudication.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
print("[out]", outdir)
