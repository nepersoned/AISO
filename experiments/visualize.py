"""Generate paper figures from results_final.pkl"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

ROOT = os.path.join(os.path.dirname(__file__), '..')
results = pickle.load(open(os.path.join(ROOT, 'results', 'results_final.pkl'), 'rb'))

algs = ['AISO_v4', 'AISO_v3', 'AISO_base', 'SPSO', 'CrowdingDE', 'LIPS']
algs_short = ['AISO-v4', 'AISO-v3', 'AISO', 'SPSO', 'CDE', 'LIPS']
colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#2ecc71', '#1abc9c']

# Fig 1
fig, ax = plt.subplots(figsize=(11, 5))
benches = list(results.keys())
x = np.arange(len(benches))
w = 0.13
for i, (alg, short) in enumerate(zip(algs, algs_short)):
    vals = [results[b][alg] for b in benches]
    ax.bar(x + i*w, vals, w, label=short, color=colors[i], alpha=0.85)
ax.set_xticks(x + w*2.5)
ax.set_xticklabels([b.replace('_', '\n') for b in benches], fontsize=9)
ax.set_ylabel('Peak Ratio')
ax.set_title('CEC2013 Niching Benchmark Comparison (5 seeds avg)')
ax.legend(ncol=6, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12))
ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'results', 'fig1_comparison.png'), dpi=140, bbox_inches='tight')
plt.close()

# Fig 2
fig, ax = plt.subplots(figsize=(9, 5))
ablation = ['AISO_base', 'AISO_v3', 'AISO_v4']
ablation_short = ['Baseline\nAISO', '+Coupling\n(v3)', '+Refinement\n(v4)']
avg_vals = [np.mean([results[b][a] for b in results]) for a in ablation]
colors_ab = ['#95a5a6', '#f39c12', '#e74c3c']
bars = ax.bar(ablation_short, avg_vals, color=colors_ab, alpha=0.88, width=0.55)
for bar, v in zip(bars, avg_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Peak Ratio', fontsize=11)
ax.set_title('AISO Ablation — Component Contribution', fontsize=12)
ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'results', 'fig2_ablation.png'), dpi=140, bbox_inches='tight')
plt.close()

# Fig 3
fig, ax = plt.subplots(figsize=(9, 5))
overall = [(s, np.mean([results[b][a] for b in results])) for a, s in zip(algs, algs_short)]
overall.sort(key=lambda x: -x[1])
names = [x[0] for x in overall]; vals = [x[1] for x in overall]
colors_rank = [colors[algs_short.index(n)] for n in names]
bars = ax.barh(names[::-1], vals[::-1], color=colors_rank[::-1], alpha=0.85)
for bar, v in zip(bars, vals[::-1]):
    ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=11)
ax.set_xlabel('Average Peak Ratio', fontsize=11)
ax.set_title('Overall Ranking on CEC2013 Niching', fontsize=12)
ax.set_xlim(0, 1); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'results', 'fig3_ranking.png'), dpi=140, bbox_inches='tight')
plt.close()

print("Figures saved to results/")
