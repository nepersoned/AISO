"""
W4: Memetic baseline comparison on CEC F1-F8.

Adds PSO + Gaussian Local Search as a memetic baseline.
PSO+LS = standard memetic algorithm: global PSO phase (phase 1) +
         per-particle Gaussian refinement (phase 2), same schedule as AISO.

Compares:
  (A) AISO + Refine       — our method (reference: 0.911)
  (B) PSO  + Refine       — memetic baseline (same phase split)
  (C) Random + Refine     — refinement-only lower bound (from W1: 0.906)

Same protocol: N=80, T=200, 30 seeds, CEC F1-F8, accuracy=0.01.
Output: results/w4_pso_results.json + .png
"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

from aiso_v7 import AISO_v7
from benchmarks import BENCHMARKS, peak_ratio

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

N_AGENTS    = 80
N_ITER      = 200
SEEDS       = list(range(30))
ACCURACY    = 0.01
REFINE_AFTER = 0.7


class PSO_Refine:
    """Standard PSO with phased Gaussian local refinement (memetic baseline).
    Phase 1 (0–0.7T): inertia-weight PSO (w=0.7, c1=c2=1.5).
    Phase 2 (0.7T–T): per-particle Gaussian random walk (same as AISO).
    """
    def __init__(self, n_agents, dim, bounds, refine_after=0.7, seed=42):
        self.n_agents    = n_agents
        self.dim         = dim
        self.bounds      = bounds
        self.refine_after = refine_after
        self.rng         = np.random.RandomState(seed)

        lo, hi = bounds
        self.X  = self.rng.uniform(lo, hi, (n_agents, dim))
        self.V  = self.rng.uniform(-(hi-lo)*0.1, (hi-lo)*0.1, (n_agents, dim))
        self.pBest  = self.X.copy()
        self.pBestF = np.full(n_agents, -np.inf)

        self.w  = 0.7    # inertia
        self.c1 = 1.5    # cognitive
        self.c2 = 1.5    # social

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
                self.V  = np.clip(self.V, -0.2 * span, 0.2 * span)
                self.X  = np.clip(self.X + self.V, lo, hi)
            else:
                step  = (0.05 * span * (1 - phase) / (1 - self.refine_after + 1e-9))
                pert  = self.rng.randn(self.n_agents, self.dim) * step
                new_X = np.clip(self.X + pert, lo, hi)
                imp2  = fn(new_X) > fn(self.X)
                self.X[imp2] = new_X[imp2]

        return self.X


def run_aiso(bench, seed):
    K   = max(4, bench.n_peaks)
    alg = AISO_v7(N_AGENTS, bench.dim, K, bench.bounds,
                  seed=seed, refinement=True, refine_after=REFINE_AFTER)
    alg.run(bench.fn, N_ITER)
    return peak_ratio(alg.X, bench.optima, ACCURACY, bench.bounds)


def run_pso(bench, seed):
    alg = PSO_Refine(N_AGENTS, bench.dim, bench.bounds,
                     refine_after=REFINE_AFTER, seed=seed)
    X   = alg.run(bench.fn, N_ITER)
    return peak_ratio(X, bench.optima, ACCURACY, bench.bounds)


def run_random_refine(bench, seed):
    K   = max(4, bench.n_peaks)
    alg = AISO_v7(N_AGENTS, bench.dim, K, bench.bounds,
                  seed=seed, refinement=True, refine_after=REFINE_AFTER,
                  alpha=0.0)
    alg.run(bench.fn, N_ITER)
    return peak_ratio(alg.X, bench.optima, ACCURACY, bench.bounds)


def main():
    data = {cfg: {b.name: [] for b in BENCHMARKS}
            for cfg in ['AISO+Refine', 'PSO+Refine', 'Random+Refine']}

    total = 3 * len(BENCHMARKS) * len(SEEDS)
    done  = 0

    print(f"Running {total} experiments...")
    for bench in BENCHMARKS:
        for seed in SEEDS:
            data['AISO+Refine'][bench.name].append(run_aiso(bench, seed))
            data['PSO+Refine'][bench.name].append(run_pso(bench, seed))
            data['Random+Refine'][bench.name].append(run_random_refine(bench, seed))
            done += 3
            if seed % 10 == 0:
                print(f"  [{done:4d}/{total}] {bench.name:30s} seed={seed:2d}", flush=True)

    print("\nW4 Results:")
    print(f"{'Benchmark':<30} {'AISO+R':>8} {'PSO+R':>8} {'Rand+R':>8}")
    print("-" * 58)

    all_a, all_p, all_r = [], [], []
    for bench in BENCHMARKS:
        a = float(np.mean(data['AISO+Refine'][bench.name]))
        p = float(np.mean(data['PSO+Refine'][bench.name]))
        r = float(np.mean(data['Random+Refine'][bench.name]))
        all_a.extend(data['AISO+Refine'][bench.name])
        all_p.extend(data['PSO+Refine'][bench.name])
        all_r.extend(data['Random+Refine'][bench.name])
        print(f"  {bench.name:<28} {a:>8.3f} {p:>8.3f} {r:>8.3f}")

    avg_a = float(np.mean(all_a))
    avg_p = float(np.mean(all_p))
    avg_r = float(np.mean(all_r))
    print("-" * 58)
    print(f"  {'AVERAGE':<28} {avg_a:>8.3f} {avg_p:>8.3f} {avg_r:>8.3f}")

    _, p_ap = mannwhitneyu(all_a, all_p, alternative='two-sided')
    _, p_ar = mannwhitneyu(all_a, all_r, alternative='two-sided')
    print(f"\n  AISO vs PSO:    p={p_ap:.4f}  (Δ={avg_a - avg_p:+.3f})")
    print(f"  AISO vs Random: p={p_ar:.4f}  (Δ={avg_a - avg_r:+.3f})")

    results = {
        'averages': {'AISO+Refine': avg_a, 'PSO+Refine': avg_p, 'Random+Refine': avg_r},
        'per_bench': {b.name: {
            'AISO+Refine':   data['AISO+Refine'][b.name],
            'PSO+Refine':    data['PSO+Refine'][b.name],
            'Random+Refine': data['Random+Refine'][b.name],
        } for b in BENCHMARKS},
        'p_aiso_vs_pso':    float(p_ap),
        'p_aiso_vs_random': float(p_ar),
    }

    out = RESULTS_DIR / 'w4_pso_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")

    visualize(results, data)


def visualize(results, data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bench_names = [b.name for b in BENCHMARKS]
    colors = {'AISO+Refine': '#2196F3', 'PSO+Refine': '#FF5722', 'Random+Refine': '#9E9E9E'}

    ax = axes[0]
    x, w = np.arange(len(bench_names)), 0.25
    for idx, cfg in enumerate(['AISO+Refine', 'PSO+Refine', 'Random+Refine']):
        means = [np.mean(data[cfg][bn]) for bn in bench_names]
        stds  = [np.std(data[cfg][bn])  for bn in bench_names]
        ax.bar(x + idx*w, means, w, yerr=stds, label=cfg,
               color=colors[cfg], alpha=0.85, capsize=3)
    ax.set_xticks(x + w)
    ax.set_xticklabels([bn.split('_')[0] for bn in bench_names], rotation=30, ha='right')
    ax.set_ylabel('Peak Ratio'); ax.set_ylim(0, 1.15)
    ax.set_title('W4: AISO vs PSO+LS vs Random\n(CEC F1-F8, 30 seeds)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis='y')

    ax2 = axes[1]
    cfgs = ['AISO+Refine', 'PSO+Refine', 'Random+Refine']
    avgs = [results['averages'][c] for c in cfgs]
    bars = ax2.bar(cfgs, avgs, color=[colors[c] for c in cfgs], alpha=0.85, width=0.5)
    for bar, avg in zip(bars, avgs):
        ax2.text(bar.get_x() + bar.get_width()/2, avg+0.005,
                 f'{avg:.3f}', ha='center', fontsize=10, fontweight='bold')
    p_ap = results['p_aiso_vs_pso']
    ax2.set_title(f'W4: Average PR\nAISO vs PSO p={p_ap:.4f}')
    ax2.set_xticklabels(['AISO\n+Refine', 'PSO\n+Refine', 'Random\n+Refine'])
    ax2.set_ylim(0.7, 1.05); ax2.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    out = RESULTS_DIR / 'w4_pso_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure → {out}")
    plt.show()


if __name__ == '__main__':
    main()
