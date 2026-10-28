# -*- coding: utf-8 -*-
"""前向口径重算：alpha_eff / expected FP / residual / binom p（Table 4 口径）。"""
import sys, numpy as np
from scipy.stats import binom
sys.path.insert(0, r"D:/Paper/policy-panel-conformal/03_program/analysis")
import e01b_empirical as E

c, yrs, Y, S = E.load_panel()
arms = ["naive", "time_block", "country_block"]
res = {}
for alpha in [0.05, 0.10, 0.20]:
    last = {}
    ev = []
    for tj in range(E.MIN_K + 1, len(yrs)):
        y = E.Y0 + tj
        q, center = E.three_arm_intervals(Y[:, :tj], alpha)
        prev, cur = Y[:, tj - 1], Y[:, tj]
        cd = center - prev
        d_prev = Y[:, tj - 1] - Y[:, tj - 2]
        mv = d_prev[~np.isnan(d_prev)]
        med = float(np.nanmedian(mv)) if mv.size else 0.0
        for i, cc in enumerate(c):
            if np.isnan(S[i, tj - 1]) or np.isnan(S[i, tj - 2]):
                continue
            if abs(S[i, tj - 1] - S[i, tj - 2]) < E.TAU:
                continue
            if y - 1 - last.get(cc, 0) < E.COOLDOWN:
                continue
            last[cc] = y - 1
            if np.isnan(cur[i]) or np.isnan(prev[i]):
                continue
            ev.append((i, y, cur[i] - prev[i], cd[i], med))
    n = len(ev)
    for a in arms:
        q, center = None, None
        rel = fp = ex = 0
        # 重新取 q：按年
        last2 = {}
        k = 0
        for tj in range(E.MIN_K + 1, len(yrs)):
            y = E.Y0 + tj
            qq, cc_ = E.three_arm_intervals(Y[:, :tj], alpha)
            prev, cur = Y[:, tj - 1], Y[:, tj]
            cdz = cc_ - prev
            d_prev = Y[:, tj - 1] - Y[:, tj - 2]
            mv = d_prev[~np.isnan(d_prev)]
            med = float(np.nanmedian(mv)) if mv.size else 0.0
            for i, cn in enumerate(c):
                if np.isnan(S[i, tj - 1]) or np.isnan(S[i, tj - 2]):
                    continue
                if abs(S[i, tj - 1] - S[i, tj - 2]) < E.TAU:
                    continue
                if y - 1 - last2.get(cn, 0) < E.COOLDOWN:
                    continue
                last2[cn] = y - 1
                if np.isnan(cur[i]) or np.isnan(prev[i]):
                    continue
                s = (cur[i] - prev[i]) - cdz[i]
                ex += int(s > qq[a])
                k += 1
                if med - (cdz[i] + qq[a]) > 0:
                    rel += 1
                    fp += int(cur[i] - prev[i] >= med)
        ae = ex / k
        exp = rel * ae
        p = binom.sf(fp - 1, rel, ae) if rel else float("nan")
        res[(alpha, a)] = (rel, fp, ae, exp, fp - exp, p)
        print(f"a={alpha:.2f} {a:13s} n={k} ex={ex} aeff={ae:.4f} rel={rel} fp={fp} "
              f"exp={exp:.3f} resid={fp-exp:+.2f} p={p:.4f}")
