#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figure_style_upgrade.py -- 六图顶刊样式改造（灰阶/衬线/去图内 caption/黑字标签）"""
import re

P = r'D:\Paper\policy-panel-conformal\03_program\figures\05_paper_figures.py'
s = open(P, encoding='utf-8').read()
LOG = []


def rep_re(pat, new, name, flags=re.S):
    global s
    m = list(re.finditer(pat, s, flags))
    assert len(m) == 1, (name, len(m))
    s = s[:m[0].start()] + new + s[m[0].end():]
    LOG.append(name + ': applied (regex)')


def rep(old, new, name):
    global s
    assert s.count(old) == 1, (name, s.count(old))
    s = s.replace(old, new)
    LOG.append(name)


# ===== 1) 灰阶配色字典 =====
rep('''C = {"naive": "#1f77b4", "time_block": "#2ca02c", "country_block": "#d62728",
     "fp": "#d62728", "rel": "#ffffff", "trig": "#bbbbbb", "ok": "#2ca02c"}''',
    '''C = {"naive": "#1a1a1a", "time_block": "#777777", "country_block": "#bbbbbb",
     "fp": "#1a1a1a", "rel": "#ffffff", "trig": "#999999", "ok": "#777777"}''',
    '灰阶配色字典')

# ===== 2) 衬线字体 + 字号（顶刊正文匹配）=====
rep('''plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 300})''',
    '''plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 300,
                     "font.family": "serif",
                     "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False})''',
    '衬线字体与去顶右边框')

# ===== 3) 十处 set_title → 面板字母（caption 内容归正文）=====
rep('ax[0].set_title("(a) three arms at the baseline regime ($s$ = 2)")',
    'ax[0].set_title("(a)")', 'fig1a 标题')
rep('ax[1].set_title("(b) time-regime gap $\\\\varepsilon^{time}(s)$: monotone, saturating")',
    'ax[1].set_title("(b)")', 'fig1b 标题')
rep('ax[0].set_title("(a) release decisions: $\\\\alpha$-monotone, arm-ordered")',
    'ax[0].set_title("(a)")', 'fig2a 标题')
rep('ax[1].set_title("(b) width premium by arm ($\\\\alpha$ = 0.10)")',
    'ax[1].set_title("(b)")', 'fig2b 标题')
rep_re(r'ax\.set_title\("\(a\) 135 trigger events under the naive arm \([^)]*\): "\s*\n\s*"[^"]*"\)',
       'ax.set_title("(a)")', 'fig3a 标题')
rep('ax2.set_title("(b) benchmark choice reallocates false releases across two channels (Section 5.3)")',
    'ax2.set_title("(b)")', 'fig3b 标题')
rep('ax[0].set_title("(a) simulation scale: additive equality rejected\\n($s=4$ corner: measured 0.242 vs additive 0.453)")',
    'ax[0].set_title("(a)")', 'fig4a 标题')
rep('ax[1].set_title("(b) empirical trend scale (three seeds): negative\\nwhere informative \u2014 additive form is conservative")',
    'ax[1].set_title("(b)")', 'fig4b 标题')
rep('ax[0].set_title("(a) per-event coverage bounds, 2006\u20132020")',
    'ax[0].set_title("(a)")', 'fig6a 标题')
rep('ax[1].set_title("(b) time-regime coordinate: 7 of 15 years clamped\\n(75 of 135 events at the event level)")',
    'ax[1].set_title("(b)")', 'fig6b 标题')

# ===== 4) 白字标签 → 黑字（柱上方）=====
rep('ax[0].text(i, v - 0.045, f"{v:.3f}", ha="center", color="w", fontsize=8.5)',
    'ax[0].text(i, v + 0.012, f"{v:.3f}", ha="center", color="k", fontsize=8.5)',
    'fig1a 柱标签')

# ===== 5) fig6 彩色柱 → 灰阶 =====
rep('color="#8c564b", width=0.62, alpha=0.75, label="$\\\\varepsilon^{grp}$ (guarantee version)")',
    'color="#cccccc", width=0.62, label="$\\\\varepsilon^{grp}$ (guarantee version)")',
    'fig6 eps_grp 柱')
rep('color="#ff7f0e", width=0.62, label="$\\\\varepsilon^{time}(w^{(t)}_{time})$")',
    'color="#777777", width=0.62, label="$\\\\varepsilon^{time}(w^{(t)}_{time})$")',
    'fig6 eps_time 柱')

open(P, 'w', encoding='utf-8', newline='').write(s)
print('=== 样式改造完成 ===')
for x in LOG: print('  -', x)
