"""
W14: PSO+LS convergence check at larger iteration budgets.

Tests whether PSO+LS improves with more iterations (T=400, 800)
vs AISO fixed at T=200. If PSO still underperforms at 4x budget,
the W4 comparison is robust to the "not enough iterations" critique.
"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from benchmarks import BENCHMARKS, peak_ratio

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_AGENTS = 80
SEEDS    = list(range(30))
ACCURACY = 0.01


class PSO_Refine:
    def __init__(self, n_agents, dim, bounds, refine_after=0.7, seed=42):
        self.n_agents     = n_agents
        self.dim          = dim
        self.bounds       = bounds
        self.refine_after = refine_after
        self.rng          = np.random.RandomState(seed)
        lo, hi = bounds
        self.X      = self.rng.uniform(lo, hi, (n_agents, dim))
        self.V      = self.rng.uniform(-(hi-lo)*0.1, (hi-lo)*0.1, (n_agents, dim))
        self.pBest  = self.X.copy()
        self.pBestF = np.full(n_agents, -np.inf)
        self.w, self.c1, self.c2 = 0.7, 1.5, 1.5

    def run(self, fn, n_iter):
        lo, hi = self.bounds
        span   = hi - lo
        for t in range(n_iter):
            fits = fn(self.X)
            imp  = fits > self.pBestF
            self.pBest[imp]  = self.X[imp]
            self.pBestF[imp] = fits[imp]
            phase = t / n_iter
            if phase < self.refine_after:
                gBest = self.pBest[np.argmax(self.pBestF)]
                r1 = self.rng.rand(self.n_agents, self.dim)
                r2 = self.rng.rand(self.n_agents, self.dim)
                self.V  = (self.w * self.V
                           + self.c1 * r1 * (self.pBest - self.X)
                           + self.c2 * r2 * (gBest - self.X))
                self.V  = np.clip(self.V, -0.2*span, 0.2*span)
                self.X  = np.clip(self.X + self.V, lo, hi)
            else:
                step  = 0.05 * span * (1-phase) / (1-self.refine_after+1e-9)
                pert  = self.rng.randn(self.n_agents, self.dim) * step
                new_X = np.clip(self.X + pert, lo, hi)
                imp2  = fn(new_X) > fn(self.X)
                self.X[imp2] = new_X[imp2]
        return self.X


configs = [
    ('AISO+Refine T=200',  200, 'aiso'),
    ('PSO+LS T=200',       200, 'pso'),
    ('PSO+LS T=400',       400, 'pso'),
    ('PSO+LS T=800',       800, 'pso'),
]

def run_one(bench, seed, n_iter, mode):
    if mode == 'pso':
        alg = PSO_Refine(N_AGENTS, bench.dim, bench.bounds, seed=seed)
        X   = alg.run(bench.fn, n_iter)
    else:
        from aiso_v7 import AISO_v7
        K   = max(4, bench.n_peaks)
        alg = AISO_v7(N_AGENTS, bench.dim, K, bench.bounds,
                      seed=seed, refinement=True, refine_after=0.7)
        alg.run(bench.fn, n_iter)
        X = alg.X
    return peak_ratio(X, bench.optima, ACCURACY, bench.bounds)


def main():
    data  = {label: {b.name: [] for b in BENCHMARKS} for label, _, _ in configs}
    total = len(configs) * len(BENCHMARKS) * len(SEEDS)
    done  = 0

    for label, n_iter, mode in configs:
        for bench in BENCHMARKS:
            for seed in SEEDS:
                pr = run_one(bench, seed, n_iter, mode)
                data[label][bench.name].append(pr)
                done += 1
        avg = float(np.mean([np.mean(data[label][b.name]) for b in BENCHMARKS]))
        print(f"  {label:<25}  avg PR={avg:.3f}  [{done}/{total}]", flush=True)

    print("\nW14 Results:")
    print(f"{'Config':<25} {'Avg PR':>8}")
    print("-" * 35)
    avgs = {}
    for label, _, _ in configs:
        avg = float(np.mean([np.mean(data[label][b.name]) for b in BENCHMARKS]))
        avgs[label] = avg
        marker = " ← AISO reference" if 'AISO' in label else ""
        print(f"  {label:<23}  {avg:>8.3f}{marker}")

    out = RESULTS_DIR / 'w14_pso_budget.json'
    with open(out, 'w') as f:
        json.dump({'averages': avgs,
                   'per_bench': {l: {b.name: data[l][b.name]
                                     for b in BENCHMARKS}
                                 for l, _, _ in configs}}, f, indent=2)
    print(f"\nSaved -> {out}")
    visualize(avgs)


def visualize(avgs):
    labels = [l for l, _, _ in configs]
    vals   = [avgs[l] for l in labels]
    colors = ['#2196F3'] + ['#FF5722']*3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.axhline(avgs['AISO+Refine T=200'], color='#2196F3',
               linestyle='--', alpha=0.6, label='AISO T=200')
    ax.set_ylabel('Avg Peak Ratio'); ax.set_ylim(0.3, 1.05)
    ax.set_title('W14: PSO+LS at Higher Budgets vs AISO\n(CEC F1-F8, 30 seeds)')
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(); ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    out = RESULTS_DIR / 'w14_pso_budget.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure -> {out}")
    plt.show()


if __name__ == '__main__':
    main()
