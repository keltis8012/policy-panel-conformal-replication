# -*- coding: utf-8 -*-
"""安慰剂前向口径：截取 aux_run 的函数定义，不执行其主块。"""
import re, numpy as np
from scipy.stats import binom, norm

SRC = r"D:/Paper/policy-panel-conformal/03_program/experiments/aux_corroboration/aux_run.py"
CORE = r"D:/Paper/policy-panel-conformal/03_program/analysis/_aux_core.py"
lines = open(SRC, encoding="utf-8").read().split("\n")
cut = next(i for i, l in enumerate(lines) if l.startswith("outdir = os.path.join(OUT, ts)"))
open(CORE, "w", encoding="utf-8").write("\n".join(lines[:cut]))

import sys
sys.path.insert(0, r"D:/Paper/policy-panel-conformal/03_program/analysis")
import os, _aux_core as A
A.ROOT=r"D:/Paper/policy-panel-conformal"
A.RAW_GHG=os.path.join(A.ROOT,"02_datasets","raw","oecd_ghg_co2_2000_2023.csv")
A.RAW_EPS=os.path.join(A.ROOT,"02_datasets","raw","oecd_eps_1990_2020.csv")

countries, years, Yco2, S = A.load_panel()
co2_levels = {c: {y: float(np.exp(Yco2[i, j])) for j, y in enumerate(years) if np.isfinite(Yco2[i, j])}
              for i, c in enumerate(countries)}

def forward(dec, arm="naive"):
    rows = [d for d in dec if d["arm"] == arm and d.get("adjust") is not None]
    n = len(rows)
    ex = sum(1 for d in rows if d["r_oos"] > d["q_naive_used"])
    rel = sum(1 for d in rows if d["adjust"])
    fp = sum(1 for d in rows if d.get("fp") is True and d["adjust"])
    ae = ex / n if n else float("nan")
    exp = rel * ae
    p = binom.sf(fp - 1, rel, ae) if rel and ae > 0 else float("nan")
    se = (ae * (1 - ae) / n) ** 0.5
    ci = (ae - norm.ppf(.975) * se, ae + norm.ppf(.975) * se)
    return n, ex, ae, ci, rel, fp, exp, fp - exp, p

print(f"{'gas':<5}{'n':>5}{'ex':>4}{'aeff':>7}{'CIlo':>7}{'CIhi':>7}{'rel':>4}{'fp':>3}{'exp':>6}{'resid':>7}{'p':>9}")
dec = A.run_rule(Yco2, S, countries, years)
dec = A.fps_from_levels(dec, co2_levels, Ylog=Yco2, countries=countries, years=years)
n, ex, ae, ci, rel, fp, exp, res, p = forward(dec)
print(f"{'CO2':<5}{n:>5}{ex:>4}{ae:>7.4f}{ci[0]:>7.3f}{ci[1]:>7.3f}{rel:>4}{fp:>3}{exp:>6.3f}{res:>+7.2f}{p:>9.5f}")
for pol in ("CH4", "N2O", "GHG"):
    data = A.scan_ghg(pol)
    Yp = A.build_panel(data, countries, years)
    d2 = A.run_rule(Yp, S, countries, years)
    d2 = A.fps_from_levels(d2, data, Ylog=Yp, countries=countries, years=years)
    n, ex, ae, ci, rel, fp, exp, res, p = forward(d2)
    print(f"{pol:<5}{n:>5}{ex:>4}{ae:>7.4f}{ci[0]:>7.3f}{ci[1]:>7.3f}{rel:>4}{fp:>3}{exp:>6.3f}{res:>+7.2f}{p:>9.5f}")
