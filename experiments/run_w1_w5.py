"""
W1 + W5 experiment.

W1: Isolate AISO mechanism contribution vs. local refinement alone.
    Compares three configurations on CEC F1-F8, 30 seeds:
      (A) AISO + Refine  — full validated system (baseline = 0.911)
      (B) Random init + Refine only  — AISO mechanism completely removed
      (C) Symmetric M + Refine  — asymmetric interaction removed, refinement kept

    If (B) << (A): AISO global phase contributes to basin distribution.
    If (C) < (A): asymmetric M contributes beyond symmetric interaction.

W5: Phase ratio sensitivity.
    Sweeps refine_after (rho) in {0.3, 0.5, 0.7, 0.9} for config (A).

Output: results/w1_w5_results.json + w1_w5_results.png
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

N_AGENTS  = 80
N_ITER    = 200
SEEDS     = list(range(30))
ACCURACY  = 0.01
RHO_SWEEP = [0.3, 0.5, 0.7, 0.9]  # W5: phase ratio sweep


# ─────────────────────────────────────────────────────────────────────
# Helper: build symmetric M from random M (for config C)
# ─────────────────────────────────────────────────────────────────────
def make_sym_M(K, seed):
    rng = np.random.RandomState(seed)
    M = rng.uniform(-1, 1, (K, K))
    np.fill_diagonal(M, 0)
    M_abs = np.abs(M)
    M_sym = 0.5 * (M_abs + M_abs.T)
    np.fill_diagonal(M_sym, 0)
    return M_sym


# ─────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────
def run_aiso_refine(bench, seed, refine_after=0.7):
    """Config A: standard AISO + phased refinement."""
    K   = max(4, bench.n_peaks)
    alg = AISO_v7(N_AGENTS, bench.dim, K, bench.bounds,
                  seed=seed, refinement=True, refine_after=refine_after)
    alg.run(bench.fn, N_ITER)
    return peak_ratio(alg.X, bench.optima, ACCURACY, bench.bounds)


def run_random_refine(bench, seed, refine_after=0.7):
    """Config B: skip all AISO interaction, only Phase-2 Gaussian refinement.
    Implementation: run the AISO loop but with alpha=0 (no movement in phase 1)
    so agents stay at random init, then Phase 2 refines in place.
    """
    K   = max(4, bench.n_peaks)
    alg = AISO_v7(N_AGENTS, bench.dim, K, bench.bounds,
                  seed=seed, refinement=True, refine_after=refine_after,
                  alpha=0.0)   # zero step → no global movement
    alg.run(bench.fn, N_ITER)
    return peak_ratio(alg.X, bench.optima, ACCURACY, bench.bounds)


def run_symM_refine(bench, seed, refine_after=0.7):
    """Config C: symmetric M + phased refinement (asymmetric interaction removed)."""
    K    = max(4, bench.n_peaks)
    Msym = make_sym_M(K, seed)
    alg  = AISO_v7(N_AGENTS, bench.dim, K, bench.bounds,
                   M=Msym, seed=seed, refinement=True, refine_after=refine_after)
    alg.run(bench.fn, N_ITER)
    return peak_ratio(alg.X, bench.optima, ACCURACY, bench.bounds)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    results = {
        'w1': {},   # per-benchmark PR for A, B, C
        'w5': {},   # per-rho avg PR
    }

    # ── W1 ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("W1: AISO+Refine vs Random+Refine vs SymM+Refine")
    print("=" * 60)

    w1_data = {cfg: {b.name: [] for b in BENCHMARKS}
               for cfg in ['AISO+Refine', 'Random+Refine', 'SymM+Refine']}

    total = 3 * len(BENCHMARKS) * len(SEEDS)
    done  = 0

    for bench in BENCHMARKS:
        for seed in SEEDS:
            pr_a = run_aiso_refine(bench, seed)
            pr_b = run_random_refine(bench, seed)
            pr_c = run_symM_refine(bench, seed)

            w1_data['AISO+Refine'][bench.name].append(pr_a)
            w1_data['Random+Refine'][bench.name].append(pr_b)
            w1_data['SymM+Refine'][bench.name].append(pr_c)

            done += 3
            if seed % 10 == 0:
                print(f"  [{done:4d}/{total}] {bench.name:30s} seed={seed:2d} "
                      f"A={pr_a:.3f} B={pr_b:.3f} C={pr_c:.3f}", flush=True)

    # W1 summary
    print("\nW1 Results:")
    print(f"{'Benchmark':<30} {'AISO+R':>8} {'Rand+R':>8} {'SymM+R':>8}")
    print("-" * 58)

    all_a, all_b, all_c = [], [], []
    for bench in BENCHMARKS:
        a = float(np.mean(w1_data['AISO+Refine'][bench.name]))
        b = float(np.mean(w1_data['Random+Refine'][bench.name]))
        c = float(np.mean(w1_data['SymM+Refine'][bench.name]))
        all_a.extend(w1_data['AISO+Refine'][bench.name])
        all_b.extend(w1_data['Random+Refine'][bench.name])
        all_c.extend(w1_data['SymM+Refine'][bench.name])
        print(f"  {bench.name:<28} {a:>8.3f} {b:>8.3f} {c:>8.3f}")

    avg_a = float(np.mean(all_a))
    avg_b = float(np.mean(all_b))
    avg_c = float(np.mean(all_c))
    print("-" * 58)
    print(f"  {'AVERAGE':<28} {avg_a:>8.3f} {avg_b:>8.3f} {avg_c:>8.3f}")

    # Statistical tests
    stat_ab, p_ab = mannwhitneyu(all_a, all_b, alternative='greater')
    stat_ac, p_ac = mannwhitneyu(all_a, all_c, alternative='greater')
    stat_bc, p_bc = mannwhitneyu(all_b, all_c, alternative='two-sided')
    print(f"\n  AISO+R > Rand+R:  p={p_ab:.4f}  (Δ={avg_a - avg_b:+.3f})")
    print(f"  AISO+R > SymM+R:  p={p_ac:.4f}  (Δ={avg_a - avg_c:+.3f})")
    print(f"  Rand+R vs SymM+R: p={p_bc:.4f}  (Δ={avg_b - avg_c:+.3f})")

    results['w1'] = {
        'per_bench': {b.name: {
            'AISO+Refine':   w1_data['AISO+Refine'][b.name],
            'Random+Refine': w1_data['Random+Refine'][b.name],
            'SymM+Refine':   w1_data['SymM+Refine'][b.name],
        } for b in BENCHMARKS},
        'averages': {'AISO+Refine': avg_a, 'Random+Refine': avg_b, 'SymM+Refine': avg_c},
        'p_aiso_vs_rand': float(p_ab),
        'p_aiso_vs_sym':  float(p_ac),
        'p_rand_vs_sym':  float(p_bc),
    }

    # ── W5 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("W5: Phase ratio (rho) sensitivity")
    print("=" * 60)

    w5_data = {rho: {b.name: [] for b in BENCHMARKS} for rho in RHO_SWEEP}

    total = len(RHO_SWEEP) * len(BENCHMARKS) * len(SEEDS)
    done  = 0

    for rho in RHO_SWEEP:
        for bench in BENCHMARKS:
            for seed in SEEDS:
                pr = run_aiso_refine(bench, seed, refine_after=rho)
                w5_data[rho][bench.name].append(pr)
                done += 1
        avg_rho = float(np.mean([w5_data[rho][b.name] for b in BENCHMARKS]))
        print(f"  rho={rho:.1f}  avg PR={avg_rho:.3f}  [{done}/{total}]", flush=True)

    results['w5'] = {
        'rho_sweep': RHO_SWEEP,
        'averages':  {str(rho): float(np.mean([w5_data[rho][b.name]
                                                for b in BENCHMARKS]))
                      for rho in RHO_SWEEP},
        'per_bench': {str(rho): {b.name: w5_data[rho][b.name]
                                  for b in BENCHMARKS}
                      for rho in RHO_SWEEP},
    }

    print("\nW5 Results:")
    for rho in RHO_SWEEP:
        avg = results['w5']['averages'][str(rho)]
        marker = " ← current" if rho == 0.7 else ""
        print(f"  rho={rho:.1f}  avg PR={avg:.3f}{marker}")

    # Save JSON
    out_json = RESULTS_DIR / 'w1_w5_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_json}")

    visualize(results, w1_data, w5_data)


def visualize(results, w1_data, w5_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    bench_names = [b.name for b in BENCHMARKS]
    colors = {'AISO+Refine': '#2196F3', 'Random+Refine': '#FF5722', 'SymM+Refine': '#9C27B0'}

    # ── Panel 1: W1 per-benchmark bar chart ──────────────────────────
    ax = axes[0]
    x   = np.arange(len(bench_names))
    w   = 0.25
    for idx, cfg in enumerate(['AISO+Refine', 'Random+Refine', 'SymM+Refine']):
        means = [np.mean(w1_data[cfg][bn]) for bn in bench_names]
        stds  = [np.std(w1_data[cfg][bn])  for bn in bench_names]
        ax.bar(x + idx * w, means, w, yerr=stds, label=cfg,
               color=colors[cfg], alpha=0.85, capsize=3)
    ax.set_xticks(x + w)
    ax.set_xticklabels([bn.split(' ')[0] for bn in bench_names], rotation=30, ha='right')
    ax.set_ylabel('Peak Ratio', fontsize=11)
    ax.set_title('W1: Mechanism Contribution\n(per benchmark, 30 seeds)', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25, axis='y')
    ax.set_ylim(0, 1.1)

    # ── Panel 2: W1 overall summary ──────────────────────────────────
    ax2 = axes[1]
    cfgs  = ['AISO+Refine', 'Random+Refine', 'SymM+Refine']
    avgs  = [results['w1']['averages'][c] for c in cfgs]
    cols  = [colors[c] for c in cfgs]
    bars  = ax2.bar(cfgs, avgs, color=cols, alpha=0.85, width=0.5)
    for bar, avg in zip(bars, avgs):
        ax2.text(bar.get_x() + bar.get_width() / 2, avg + 0.01,
                 f'{avg:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    p_ab = results['w1']['p_aiso_vs_rand']
    p_ac = results['w1']['p_aiso_vs_sym']
    ax2.set_title(f'W1: Average PR\nA>B p={p_ab:.4f}  A>C p={p_ac:.4f}', fontsize=11)
    ax2.set_ylabel('Average Peak Ratio', fontsize=11)
    ax2.set_xticklabels(['AISO\n+Refine', 'Random\n+Refine', 'SymM\n+Refine'])
    ax2.set_ylim(0, 1.1); ax2.grid(True, alpha=0.25, axis='y')

    # ── Panel 3: W5 rho sensitivity ──────────────────────────────────
    ax3 = axes[2]
    rhos = RHO_SWEEP
    avgs_rho = [results['w5']['averages'][str(r)] for r in rhos]
    ax3.plot(rhos, avgs_rho, 'o-', color='#4CAF50', linewidth=2, markersize=8)
    for r, a in zip(rhos, avgs_rho):
        ax3.annotate(f'{a:.3f}', (r, a), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=9)
    ax3.axvline(0.7, color='red', linestyle='--', alpha=0.7, label='current (ρ=0.7)')
    ax3.set_xlabel('Phase-1 fraction (ρ)', fontsize=11)
    ax3.set_ylabel('Average Peak Ratio', fontsize=11)
    ax3.set_title('W5: Phase Ratio Sensitivity', fontsize=11)
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.25)
    ax3.set_ylim(0.7, 1.0)

    plt.tight_layout()
    out_png = RESULTS_DIR / 'w1_w5_results.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Figure saved → {out_png}")
    plt.show()


if __name__ == '__main__':
    main()
