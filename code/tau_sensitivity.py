#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tau_sensitivity.py -- P1-7：τ 敏感性扫描（P70/P75/P80/P90 完整重跑）
忠实重放：monkey-patch e01b_empirical.TAU 后调用 main()（冻结管线本体），
每 τ 产出独立 run 目录（provenance），解析 alpha_010/naive 的决策与误报。"""
import os, sys, csv, json, math, time
import numpy as np

ROOT = os.environ.get("REPL_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根（由脚本位置推导，包副本内自洽）
sys.path.insert(0, os.path.join(ROOT, '03_program', 'analysis'))
import e01b_empirical as E

# 1) 面板 |ΔEPS| 分位数（与冻结口径同：39 国交集 × 2001-2020 差分的有限值）
countries, years, Y, S = E.load_panel()
Y0 = years[0]
dS = []
for i in range(len(countries)):
    for t in range(1, len(years)):
        a, b = S[i, t - 1], S[i, t]
        if not (np.isnan(a) or np.isnan(b)):
            dS.append(abs(b - a))
dS = np.array(dS)
pcts = {p: float(np.percentile(dS, p)) for p in (70, 75, 80, 90)}
print('样本 |ΔEPS| 差分值:', len(dS), '| 分位数:', {k: round(v, 4) for k, v in pcts.items()})
assert abs(pcts[75] - 0.222) < 0.01, f'P75={pcts[75]:.4f} 应≈0.222（冻结口径核验）'

# 2) 四次完整重跑（TAU monkey-patch）
runs = {}
for p in (70, 75, 80, 90):
    tau = pcts[p]
    E.TAU = tau
    t0 = time.time()
    E.main(baseline_mode='peer_med', one_sided=False)
    # 找最新 run 目录（main() 以时间戳命名；顺序执行即最新）
    base = os.path.join(ROOT, '04_output', 'experiments', 'E01b')
    dirs = sorted((d for d in os.listdir(base) if d[0].isdigit()),
                  key=lambda d: os.path.getmtime(os.path.join(base, d)))
    run = dirs[-1]
    runs[p] = {'tau': round(tau, 4), 'run_dir': run, 'secs': round(time.time() - t0, 1)}
    print(f'  P{p}: τ={tau:.4f} → {run} ({runs[p]["secs"]}s)')
    time.sleep(1.2)   # 保证时间戳目录名不撞

# 3) 解析各 run：naive 臂 α=0.10 的触发/放行/误报 + α_eff（CRC 有效率，m4 同口径）
def crc_eff(n, alpha=0.10):
    k = max(1, min(n, math.ceil((n + 1) * (1 - alpha))))
    return (n + 1 - k) / (n + 1)

table = []
for p in (70, 75, 80, 90):
    run = runs[p]['run_dir']
    f = os.path.join(base, run, 'alpha_010', 'decision_timing.csv')
    rows = [r for r in csv.DictReader(open(f, encoding='utf-8-sig')) if r['arm'] == 'naive']
    n_valid = sum(1 for r in rows if r['adjust'] != 'None')
    n_rel = sum(1 for r in rows if r['adjust'] == 'True')
    fp = 0
    for r in rows:
        if r['adjust'] != 'True':
            continue
        c, tj = r['country'], int(r['trigger_year'])
        i = countries.index(c)
        d = Y[i, tj - Y0] - Y[i, tj - Y0 - 1]
        med = float(r['med_prev'])
        if not np.isnan(d) and d >= med:
            fp += 1
    # α_eff：触发残差 CRC 有效率（m4 同口径）
    store_rr = {}
    for r in rows:
        if r['arm'] != 'naive':
            continue
        c, tj = r['country'], int(r['trigger_year'])
        ti = tj - Y0
        if ti < 5: continue
        if ti not in store_rr:
            store_rr[ti] = E.rolling_pseudo_residuals_na(Y[:, :ti], 5)[0][:, -1]
        key = (c, tj)
        pass
    # 触发残差：每有效事件的滚动残差
    trig_resid = []
    seen = set()
    for r in rows:
        if r['arm'] != 'naive':
            continue
        c, tj = r['country'], int(r['trigger_year'])
        ti = tj - Y0
        if ti < 5 or (c, tj) in seen: continue
        seen.add((c, tj))
        rv = E.rolling_pseudo_residuals_na(Y[:, :ti], 5)[0][:, -1][countries.index(c)]
        if not np.isnan(rv):
            trig_resid.append(float(rv))
    n_res = len(trig_resid)
    aeff = crc_eff(n_res)
    xs = sorted(trig_resid)
    k = max(1, min(n_res, math.ceil((n_res + 1) * 0.9)))
    # 经验超出率（对各自年份 q 的超出计数 / n_res）：与正文 α_eff=0.106 同口径
    # 简化：用 CRC 有效率（m4 同口径）为主报告值
    table.append({'pct': p, 'tau': runs[p]['tau'], 'n_valid': n_valid, 'n_rel': n_rel,
                  'n_fp': fp, 'fp_rate': round(fp / n_rel, 3) if n_rel else None,
                  'n_res': n_res, 'alpha_eff_crc': round(aeff, 4)})
    print(f"  P{p} (τ={runs[p]['tau']:.3f}): 有效 {n_valid} | 放行 {n_rel} | 误报 {fp} "
          f"| 放行条件误报率 {fp/max(1,n_rel):.3f} | α_eff(CRC) {aeff:.4f}")

# 三臂 α=0.10 汇总（每 τ：三臂放行/误报梯度）
arm_grid = []
for p in (70, 75, 80, 90):
    run = runs[p]['run_dir']
    f = os.path.join(base, run, 'alpha_010', 'decision_timing.csv')
    rows = [r for r in csv.DictReader(open(f, encoding='utf-8-sig')) if r['arm'] != 'naive' or True]
    line = {'pct': p}
    for arm in ('naive', 'time_block', 'country_block'):
        rs = [r for r in rows if r['arm'] == arm]
        rel = sum(1 for r in rs if r['adjust'] == 'True')
        fp = 0
        for r in rs:
            if r['adjust'] != 'True':
                continue
            c, tj = r['country'], int(r['trigger_year'])
            i = countries.index(c)
            d = Y[i, tj - Y0] - Y[i, tj - Y0 - 1]
            med = float(r['med_prev'])
            if not np.isnan(d) and d >= med:
                fp += 1
        line[arm] = f'{rel}/{fp}'
    arm_grid.append(line)
    print(f"  P{p} 三臂(放行/误报): naive {line['naive']} | time {line['time_block']} | country {line['country_block']}")

json.dump({'percentiles': pcts, 'runs': runs, 'table': table, 'arm_grid': arm_grid,
           'seed_note': 'deterministic pipeline, no randomness'},
          open(os.path.join(ROOT, '04_output', 'tables', 'tau_sensitivity_20261011.json'), 'w',
               encoding='utf-8'), ensure_ascii=False, indent=1)
print('归档 tau_sensitivity_20261011.json ✓')
