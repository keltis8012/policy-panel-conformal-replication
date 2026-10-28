#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e01b_figures.py —— E01b 主图绘制（R16c 执行顺序 #5）
版本：v1.0 ｜ 2026-08-28

图 1：α-时点决策曲线（档案 §3.3 骨架 #2 主图）——x=α，y=放行率，三臂三线
图 2：边界带与正负分层补充（决策敏感度）

用法：python e01b_figures.py <run_dir>
输出：04_output/figures/e01b_<run_dir>/{fig1_alpha_timing.png, fig2_boundary_sign.png}
"""
import csv, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def load_alpha_report(base, alpha):
    return json.load(open(os.path.join(base, f"alpha_{int(alpha*100):03d}", "e01b_report.json"), encoding="utf-8"))

def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else None
    assert run_dir, "用法: python e01b_figures.py <run_dir>"
    base = os.path.join(r"04_output/experiments/E01b", run_dir)
    post = json.load(open(os.path.join(base, "post_analysis.json"), encoding="utf-8"))
    alphas = [0.05, 0.10, 0.20]
    arms = ["naive", "time_block", "country_block"]
    en = "--en" in sys.argv
    if en:
        arms_l = {"naive": "Naive conformal", "time_block": "Time block", "country_block": "Country block"}
        xlab, ylab, t1, t2, t3 = ("Risk level $\\alpha$", "Policy release rate (adjust / valid triggers)",
                                  "Policy timing under risk level (E01b, N=39, T=21; 135 fixed triggers; $\\alpha$ enters interval modulation only)",
                                  "Trigger sign decomposition ($\\alpha$=0.10)", "Decision sensitivity: boundary-band events")
        bd_lab, up_lab, down_lab, hold_lab, bd_ev = ("Boundary events", "Up-shift released", "Down-shift released",
                                                     "Held", "Boundary events ($|e_{lo}| <$ inter-arm $q$ gap)")
        colors = {"naive": "#0072B2", "time_block": "#E69F00", "country_block": "#009E73"}  # Okabe-Ito 色盲安全
        suffix = "_en"
        ylab2 = "Trigger events ($\\alpha$=0.10)"
    else:
        arms_l = {"naive": "朴素保形", "time_block": "时间块", "country_block": "国家块"}
        xlab, ylab, t1, t2, t3 = ("风险水平 α", "政策放行率（调整事件 / 有效触发）",
                                  "α-时点决策曲线（E01b，N=39, T=21；触发 135 事件恒定，α 只进区间调制）",
                                  "正/负 ΔEPS 触发分层", "决策敏感度：边界带事件")
        bd_lab, up_lab, down_lab, hold_lab, bd_ev = ("边界事件", "上调触发·放行", "下调触发·放行",
                                                     "持有", "边界事件数（|e_lo| < 臂间 q 差）")
        colors = {"naive": "#111111", "time_block": "#666666", "country_block": "#BBBBBB"}
        suffix = ""
        ylab2 = "触发事件数（α=0.10）"
    markers = {"naive": "o", "time_block": "s", "country_block": "^"}
    outdir = os.path.join(r"04_output/figures", f"e01b_{run_dir}")
    os.makedirs(outdir, exist_ok=True)

    # ============ 图 1：α-时点决策曲线 ============
    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    for a in arms:
        rates = [load_alpha_report(base, al)["decision"][a]["release_rate"] for al in alphas]
        ax1.plot(alphas, rates, marker=markers[a], linewidth=2, color=colors[a], label=arms_l[a])
    ax1.set_xlabel(xlab, fontsize=12)
    ax1.set_ylabel(ylab, fontsize=12)
    ax1.set_xticks(alphas)
    ax1.set_ylim(0, 0.16)
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.legend(fontsize=10, frameon=False)
    ax1.set_title(t1, fontsize=11)
    # 附注边界事件
    bd = [post[f"alpha_{int(al*100):03d}"]["boundary_events_count"] for al in alphas]
    for i, al in enumerate(alphas):
        ax1.annotate(f"{bd_lab} {bd[i]}", xy=(al, 0.145), ha="center", fontsize=9, color="#444444")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig1_alpha_timing{suffix}.png"), dpi=200)
    plt.close(fig)

    # ============ 图 2：正负分层 + 边界带 ============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    # 左：正/负触发分层（α=0.10 主设定）
    rep = load_alpha_report(base, 0.10)
    up = post["alpha_010"]["sign_layers"]["up"]
    down = post["alpha_010"]["sign_layers"]["down"]
    x = range(len(arms))
    w = 0.22
    ax1.bar([i - w for i in x], [up[a]["True"] for a in arms], w, color=colors["naive"], label=up_lab)
    ax1.bar([i for i in x], [down[a]["True"] for a in arms], w, color=colors["time_block"], label=down_lab)
    ax1.bar([i + w for i in x], [up[a]["False"] + down[a]["False"] for a in arms], w, color=colors["country_block"], label=hold_lab)
    ax1.set_xticks(list(x)); ax1.set_xticklabels([arms_l[a] for a in arms])
    ax1.set_ylabel(ylab2)
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_title(t2, fontsize=11)
    # 右：边界带（决策敏感度，三 α 对比）
    alphas_l = [0.05, 0.10, 0.20]
    bds = [post[f"alpha_{int(al*100):03d}"]["boundary_events_count"] for al in alphas_l]
    ax2.bar([str(a) for a in alphas_l], bds, color=["#111111", "#666666", "#BBBBBB"])
    ax2.set_xlabel("$\\alpha$" if en else "α")
    ax2.set_ylabel(bd_ev)
    ax2.set_title(t3, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig2_boundary_sign{suffix}.png"), dpi=200)
    plt.close(fig)
    print(f"[fig] 图1: {os.path.join(outdir, f'fig1_alpha_timing{suffix}.png')}（{'EN' if en else 'ZH'}）")
    print(f"[fig] 图2: {os.path.join(outdir, f'fig2_boundary_sign{suffix}.png')}（{'EN' if en else 'ZH'}）")

if __name__ == "__main__":
    main()
