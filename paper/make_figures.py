"""
AISO Paper — Figure Generator
Produces Fig 1, 2, 3, 5 for the Memetic Computing submission.
Run: python make_figures.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = r'c:\Users\kevin\OneDrive\Desktop\aiso-paper\aiso-paper\paper\figures'
os.makedirs(OUT, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
C_BLUE   = '#2C6FAC'
C_GREEN  = '#2E8B57'
C_ORANGE = '#E07B39'
C_GRAY   = '#888888'
C_RED    = '#C0392B'
C_LIGHT  = '#EAF3FB'
C_LGRAY  = '#F5F5F5'


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — AISO Mechanism Diagram
# ══════════════════════════════════════════════════════════════════════════════

def make_fig1():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    def box(cx, cy, w, h, label, sublabel='', color=C_LIGHT, ec=C_BLUE, lw=1.5, fs=9):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle='round,pad=0.08', facecolor=color, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(rect)
        if sublabel:
            ax.text(cx, cy + 0.18, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color='#1a1a1a')
            ax.text(cx, cy - 0.22, sublabel, ha='center', va='center',
                    fontsize=fs - 1.5, color='#444444', style='italic')
        else:
            ax.text(cx, cy, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color='#1a1a1a')

    def arrow(x1, y1, x2, y2, label='', color=C_BLUE):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.18, label, ha='center', va='bottom',
                    fontsize=7.5, color=color)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(5, 5.2, 'AISO: Per-Iteration Update Cycle',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # ── Top row: Agent State ──────────────────────────────────────────────────
    box(1.4, 3.9, 2.4, 1.0,
        r'Agent State',
        r'$X_i \in \mathbb{R}^d$,  $W_i \in \Delta^{K-1}$,  $s_i$',
        color='#FFF8E7', ec=C_ORANGE)

    box(4.2, 3.9, 2.2, 1.0,
        r'Compatibility',
        r'$c_{ij} = W_i^\top M W_j \neq c_{ji}$',
        color=C_LIGHT, ec=C_BLUE)

    box(7.2, 3.9, 2.2, 1.0,
        r'Partner Selection',
        r'$j^* = \arg\max_j\; c_{ij} \cdot s_j/s_{\max}$',
        color=C_LIGHT, ec=C_BLUE)

    # ── Bottom row ────────────────────────────────────────────────────────────
    box(1.4, 2.1, 2.4, 1.0,
        r'Adaptive Repulsion',
        r'$M^{\rm eff}_{ij} \propto (1+3e^{-\delta/0.12})$ if $M_{ij}<0$',
        color='#F0FFF0', ec=C_GREEN)

    box(4.2, 2.1, 2.2, 1.0,
        r'Type Assimilation',
        r'$W_i \leftarrow (1-\beta)W_i + \beta W_{j^*}$ (if improved)',
        color='#F0FFF0', ec=C_GREEN)

    box(7.2, 2.1, 2.2, 1.0,
        r'Position Update',
        r'$X_i \leftarrow X_i + \alpha\, c_{i,j^*}(X_{j^*}-X_i)$',
        color='#F0FFF0', ec=C_GREEN)

    # ── Shared M box ─────────────────────────────────────────────────────────
    box(5.0, 0.7, 2.6, 0.8,
        r'Asymmetric $M \in \mathbb{R}^{K \times K}$  (shared, zero diag)',
        color='#F9ECF9', ec='#8B4F9E')

    # ── Arrows top row ────────────────────────────────────────────────────────
    arrow(2.6, 3.9, 3.1, 3.9)
    arrow(5.3, 3.9, 6.1, 3.9)

    # ── Arrows top → bottom ───────────────────────────────────────────────────
    arrow(7.2, 3.4, 7.2, 2.6)          # partner → position update
    arrow(6.1, 2.1, 5.3, 2.1)          # position → type assim
    arrow(3.1, 2.1, 2.6, 2.1)          # type assim → adaptive repulsion
    arrow(1.4, 2.6, 1.4, 3.4)          # adaptive repulsion → agent state (loop)

    # ── M connections ─────────────────────────────────────────────────────────
    arrow(5.0, 1.1, 4.2, 1.7, color='#8B4F9E')   # M → type assim region
    arrow(5.0, 1.1, 4.2, 3.4, color='#8B4F9E')   # M → compatibility

    # ── Phase annotations ─────────────────────────────────────────────────────
    ax.text(0.15, 1.45, 'Phase 1\n(0–0.7T)\nGlobal',
            ha='left', va='center', fontsize=7.5, color=C_BLUE,
            bbox=dict(facecolor='white', edgecolor=C_BLUE, boxstyle='round,pad=0.2', lw=0.8))
    ax.text(0.15, 0.55, 'Phase 2\n(0.7T–T)\nGaussian walk',
            ha='left', va='center', fontsize=7.5, color=C_GREEN,
            bbox=dict(facecolor='white', edgecolor=C_GREEN, boxstyle='round,pad=0.2', lw=0.8))

    plt.tight_layout()
    path = os.path.join(OUT, 'fig1_aiso_mechanism.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f'Saved {path}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Two-Stage Architecture
# ══════════════════════════════════════════════════════════════════════════════

def make_fig2():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    def box(cx, cy, w, h, lines, color=C_LIGHT, ec=C_BLUE, lw=1.5, fs=8.5):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle='round,pad=0.1', facecolor=color, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(rect)
        step = h / (len(lines) + 1)
        for k, line in enumerate(lines):
            bold = (k == 0)
            ax.text(cx, cy + h/2 - step*(k+1), line,
                    ha='center', va='center', fontsize=fs,
                    fontweight='bold' if bold else 'normal',
                    color='#1a1a1a')

    def arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label,
                    ha='center', va='bottom', fontsize=7.5, color='#333333')

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(5.5, 4.25, 'Two-Stage AISO Pipeline for GNN Fraud Detection',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # ── Input ─────────────────────────────────────────────────────────────────
    box(0.85, 2.2, 1.4, 2.0,
        ['Input', r'Graph $G=(V,E)$', r'Features $X$', r'Labels $y$'],
        color='#FFF8E7', ec=C_ORANGE)

    arrow(1.55, 2.2, 2.0, 2.2)

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    # Stage 1 background
    stage1_rect = mpatches.FancyBboxPatch(
        (2.0, 0.5), 3.8, 3.4,
        boxstyle='round,pad=0.1', facecolor='#EAF3FB', edgecolor=C_BLUE, linewidth=2,
        linestyle='--'
    )
    ax.add_patch(stage1_rect)
    ax.text(3.9, 3.75, 'Stage 1 — Feature Selection', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_BLUE)

    box(3.0, 2.8, 1.6, 0.9,
        ['Correlation', 'clustering', r'$(K_1=15)$'],
        color='white', ec=C_BLUE, lw=1.0)

    box(4.9, 2.8, 1.6, 0.9,
        ['Smart $M^{(1)}$', r'MI gradient +', 'corr. repulsion'],
        color='white', ec=C_BLUE, lw=1.0)

    box(3.0, 1.5, 1.6, 0.9,
        ['AISO', r'$N=20,\;T=60$', 'global + refine'],
        color='#D0E8FF', ec=C_BLUE, lw=1.5)

    box(4.9, 1.5, 1.6, 0.9,
        [r'Top-$B_1$ features', r'$\mathcal{F}^*$', '(budget-constrained)'],
        color='#D0FFD8', ec=C_GREEN, lw=1.5)

    arrow(2.1, 2.8, 2.2, 2.8)
    arrow(3.8, 2.8, 4.1, 2.8)
    arrow(4.9, 2.35, 4.9, 1.95)
    arrow(4.1, 1.5, 3.8, 1.5)
    arrow(2.2, 1.5, 2.1, 1.5)

    arrow(5.8, 1.5, 6.2, 1.5, label=r'$\mathcal{F}^*$')

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    stage2_rect = mpatches.FancyBboxPatch(
        (6.2, 0.5), 3.8, 3.4,
        boxstyle='round,pad=0.1', facecolor='#F0FFF0', edgecolor=C_GREEN, linewidth=2,
        linestyle='--'
    )
    ax.add_patch(stage2_rect)
    ax.text(8.1, 3.75, 'Stage 2 — Node Selection', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=C_GREEN)

    box(7.1, 2.8, 1.6, 0.9,
        ['Node clustering', r'within $\mathcal{F}^*$', r'$(K_2=20)$'],
        color='white', ec=C_GREEN, lw=1.0)

    box(9.0, 2.8, 1.6, 0.9,
        ['Smart $M^{(2)}$', 'recomputed in', r'$\mathcal{F}^*$ subspace'],
        color='white', ec=C_GREEN, lw=1.0)

    box(7.1, 1.5, 1.6, 0.9,
        ['AISO', r'$N=20,\;T=60$', 'node-level'],
        color='#C8F5D0', ec=C_GREEN, lw=1.5)

    box(9.0, 1.5, 1.6, 0.9,
        [r'Training subgraph', r'$\mathcal{S} \subseteq V$',
         r'$B_2$ fraud nodes'],
        color='#D0FFD8', ec=C_GREEN, lw=1.5)

    arrow(6.3, 2.8, 6.3, 2.8)
    arrow(7.9, 2.8, 8.2, 2.8)
    arrow(9.0, 2.35, 9.0, 1.95)
    arrow(8.2, 1.5, 7.9, 1.5)
    arrow(7.3, 1.5, 6.3, 1.5)

    arrow(9.8, 1.5, 10.2, 1.5)

    # ── GNN Output ────────────────────────────────────────────────────────────
    box(10.6, 1.5, 0.7, 1.1,
        ['GNN', 'train'],
        color='#F9ECF9', ec='#8B4F9E', lw=1.5)

    # ── Key label ─────────────────────────────────────────────────────────────
    ax.text(5.5, 0.2,
            r'$M^{(2)}$ recomputed within $\mathcal{F}^*$: cluster structure shifts after projection '
            r'$\;\Rightarrow\;$ independent diversity pressure at each stage',
            ha='center', va='center', fontsize=7.5, color='#555555', style='italic')

    plt.tight_layout()
    path = os.path.join(OUT, 'fig2_two_stage.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f'Saved {path}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — CEC Ablation Bar Chart
# ══════════════════════════════════════════════════════════════════════════════

def make_fig3():
    # Data from Table 1 + Table 2 in the draft
    variants = [
        'AISO baseline',
        '+ Type-Position\nCoupling',
        '+ Adaptive Anchor',
        '+ Fitness Gating',
        '+ W Diversity\nPenalty',
        '+ Smart W Init',
        '+ W Sparsity',
        '+ Anti-Assimilation',
        '+ Hebbian M',
        '+ Sparse M',
        '+ Cyclic M Init',
        '+ Fitness-wtd β',
        '+ Asym Assimilation',
        '+ Niche Detection',
        '+ Surrogate (RF)',
        '+ Memetic LS',
        '+ Phased Refinement\n(AISO + Refine)',
    ]
    pr = [0.445, 0.420, 0.905, 0.905, 0.908, 0.905, 0.899,
          0.911, 0.912, 0.912, 0.909, 0.911, 0.909, 0.908, 0.911, 0.914, 0.911]

    colors = []
    for i, v in enumerate(variants):
        if 'Phased Refinement' in v:
            colors.append(C_GREEN)
        elif i == 0:
            colors.append(C_GRAY)
        elif pr[i] < 0.445:
            colors.append(C_RED)
        else:
            colors.append(C_BLUE)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={'width_ratios': [2.5, 1]})

    # ── Left: ablation ────────────────────────────────────────────────────────
    ax = axes[0]
    y = np.arange(len(variants))
    bars = ax.barh(y, pr, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(variants, fontsize=8.5)
    ax.set_xlabel('Average Peak Ratio (CEC2013 F1–F8)', fontsize=10)
    ax.set_title('(a) Ablation of 15+ Mechanisms', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(0.35, 0.97)
    ax.axvline(0.911, color=C_GREEN, linestyle='--', lw=1.2, alpha=0.7)

    # value labels
    for bar, val in zip(bars, pr):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=7.5, color='#222222')

    # legend
    handles = [
        mpatches.Patch(color=C_GRAY,  label='Baseline (0.445)'),
        mpatches.Patch(color=C_BLUE,  label='Mechanism variant (n.s., p≥0.25)'),
        mpatches.Patch(color=C_GREEN, label='Phased Refinement (+0.466)'),
        mpatches.Patch(color=C_RED,   label='Harmful variant'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=7.5, framealpha=0.9)

    # ── Right: key comparison ─────────────────────────────────────────────────
    ax2 = axes[1]
    methods = ['PSO + LS\n(T=200)', 'PSO + LS\n(T=400)', 'PSO + LS\n(T=800)',
               'Random\n+ Refine', 'CrowdingDE', 'AISO\n+ Refine']
    vals   = [0.460, 0.427, 0.416, 0.906, 0.952, 0.911]
    cols   = [C_RED, C_RED, C_RED, C_BLUE, C_ORANGE, C_GREEN]

    x = np.arange(len(methods))
    bars2 = ax2.bar(x, vals, color=cols, edgecolor='white', linewidth=0.5, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylabel('Average Peak Ratio', fontsize=10)
    ax2.set_title('(b) Method Comparison', fontsize=11, fontweight='bold', pad=8)
    ax2.set_ylim(0.35, 1.02)
    ax2.axhline(0.911, color=C_GREEN, linestyle='--', lw=1.0, alpha=0.6)

    for bar, val in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                 fontweight='bold' if val == 0.911 else 'normal')

    plt.tight_layout(w_pad=2.0)
    path = os.path.join(OUT, 'fig3_cec_ablation.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f'Saved {path}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Jaccard Diversity + 2×2 Stage Ablation
# ══════════════════════════════════════════════════════════════════════════════

def make_fig5():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── Left: Jaccard bar ─────────────────────────────────────────────────────
    ax = axes[0]
    labels  = ['Asymmetric $M$\n(Smart M)', 'Symmetric $M_\\mathrm{sym}$']
    jaccard = [0.136, 1.000]
    errs    = [0.027, 0.000]
    colors  = [C_GREEN, C_RED]

    bars = ax.bar([0, 1], jaccard, yerr=errs, color=colors,
                  edgecolor='white', linewidth=0.5, width=0.5,
                  capsize=5, error_kw=dict(ecolor='#444', elinewidth=1.5))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Mean Inter-Agent Jaccard Similarity', fontsize=10)
    ax.set_title('(a) Diversity Collapse Under Symmetric $M$',
                 fontsize=11, fontweight='bold', pad=8)
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color=C_RED, linestyle=':', lw=1.0, alpha=0.5)

    for bar, val in zip(bars, jaccard):
        label = f'{val:.3f}' if val < 1 else '1.000\n(full collapse)'
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.04,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.text(0.5, 0.55,
            'Jaccard = 1.000:\nall agents identical',
            ha='center', va='center', fontsize=8, color=C_RED,
            transform=ax.get_xaxis_transform())

    # ── Right: 2×2 heatmap ────────────────────────────────────────────────────
    ax2 = axes[1]
    # Rows: Stage-1 (Rand M, Smart M); Cols: Stage-2 (Rand M, Smart M)
    data = np.array([[0.5502, 0.5872],
                     [0.6510, 0.6644]])
    im = ax2.imshow(data, cmap='Blues', vmin=0.50, vmax=0.70)

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Stage-2\nRand $M$', 'Stage-2\nSmart $M$'], fontsize=9.5)
    ax2.set_yticklabels(['Stage-1\nRand $M$', 'Stage-1\nSmart $M$'], fontsize=9.5)
    ax2.set_title('(b) 2×2 Stage Decomposition (PR-AUC)', fontsize=11,
                  fontweight='bold', pad=8)

    for i in range(2):
        for j in range(2):
            val = data[i, j]
            weight = 'bold' if (i == 1 and j == 1) else 'normal'
            star = ' ★' if (i == 1 and j == 1) else ''
            ax2.text(j, i, f'{val:.4f}{star}', ha='center', va='center',
                     fontsize=11, fontweight=weight,
                     color='white' if val > 0.62 else '#1a1a1a')

    # Gain annotations
    ax2.annotate('', xy=(1.55, 0), xytext=(1.55, 1),
                 xycoords='data', textcoords='data',
                 arrowprops=dict(arrowstyle='<->', color=C_BLUE, lw=1.5))
    ax2.text(1.75, 0.5, '+0.101\n(Stage-1)', ha='left', va='center',
             fontsize=8, color=C_BLUE, transform=ax2.transData)

    ax2.annotate('', xy=(0, -0.55), xytext=(1, -0.55),
                 xycoords='data', textcoords='data',
                 arrowprops=dict(arrowstyle='<->', color=C_GREEN, lw=1.5))
    ax2.text(0.5, -0.85, '+0.013 (Stage-2)', ha='center', va='top',
             fontsize=8, color=C_GREEN, transform=ax2.transData)

    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='PR-AUC')

    plt.tight_layout(w_pad=2.5)
    path = os.path.join(OUT, 'fig5_jaccard_diversity.pdf')
    plt.savefig(path)
    plt.savefig(path.replace('.pdf', '.png'))
    print(f'Saved {path}')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Generating AISO paper figures...')
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig5()
    print('\nDone. Files saved to:')
    print(f'  {OUT}')
