"""
AISO v6 W-reinforcement ablation experiment.

Reuses baselines from results_v5.pkl.
Runs 6 AISO-v6 variants + lambda_div sensitivity grid.

Run from repo root: python experiments/run_v6.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pickle
from scipy import stats

from aiso_v6 import AISO_v6
from benchmarks import BENCHMARKS, peak_ratio

N_AGENTS = 80
N_ITER   = 200
SEEDS    = list(range(30))
ACCURACY = 0.01

V6_CONFIGS = [
    ('AISO_refine_only', dict()),
    ('AISO_v6_Div',      dict(lambda_div=0.1)),
    ('AISO_v6_Smart',    dict(smart_init=True, smart_init_alpha=0.7)),
    ('AISO_v6_Sparse',   dict(sparsity_gamma=1.5)),
    ('AISO_v6_Anti',     dict(anti_assim_mu=0.05)),
    ('AISO_v6_Full',     dict(lambda_div=0.1, smart_init=True, smart_init_alpha=0.7,
                              sparsity_gamma=1.5, anti_assim_mu=0.05)),
]

LAMBDA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2]


def run_one_v6(bench, seed, **kw):
    K = max(4, bench.n_peaks)
    alg = AISO_v6(N_AGENTS, bench.dim, K, bench.bounds, seed=seed, **kw)
    alg.run(bench.fn, N_ITER)
    pr = peak_ratio(alg.X, bench.optima, ACCURACY, bench.bounds)
    ent = alg.type_entropy()
    return pr, ent


if __name__ == '__main__':
    res_dir = os.path.join(os.path.dirname(__file__), '..', 'results')

    # Load baseline results from v5
    v5 = pickle.load(open(os.path.join(res_dir, 'results_v5.pkl'), 'rb'))['all_results']
    BASELINE_ALGS = ['SPSO', 'CrowdingDE', 'LIPS', 'RingPSO']

    # ---- Run v6 variants ----
    v6_names = [n for n, _ in V6_CONFIGS]
    print(f"Running {len(BENCHMARKS)} benchmarks × {len(SEEDS)} seeds × {len(V6_CONFIGS)} v6 variants")

    v6_raw     = {b.name: {n: [] for n in v6_names} for b in BENCHMARKS}
    v6_entropy = {b.name: {n: [] for n in v6_names} for b in BENCHMARKS}

    for bench in BENCHMARKS:
        for s in SEEDS:
            for name, kw in V6_CONFIGS:
                pr, ent = run_one_v6(bench, s, **kw)
                v6_raw[bench.name][name].append(pr)
                v6_entropy[bench.name][name].append(ent)

    # Build unified all_results
    all_results = {}
    for bench in BENCHMARKS:
        merged = {}
        for n in v6_names:
            merged[n] = {
                'mean': float(np.mean(v6_raw[bench.name][n])),
                'std':  float(np.std(v6_raw[bench.name][n])),
                'raw':  v6_raw[bench.name][n],
                'entropy_mean': float(np.mean(v6_entropy[bench.name][n])),
            }
        for a in BASELINE_ALGS:
            merged[a] = v5[bench.name]['mean'][a]  # scalar only for display
        all_results[bench.name] = merged

    ALL_DISPLAY = v6_names + BASELINE_ALGS

    # ---- Main results table ----
    col_w = 16
    header = f"{'Benchmark':<24} " + " ".join(f"{a:>{col_w}}" for a in ALL_DISPLAY)
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for bench in BENCHMARKS:
        def _m(a):
            v = all_results[bench.name][a]
            return v['mean'] if isinstance(v, dict) else v
        print(f"{bench.name:<24} " + " ".join(f"{_m(a):>{col_w}.3f}" for a in ALL_DISPLAY))

    def avg(a):
        vals = []
        for bench in BENCHMARKS:
            v = all_results[bench.name][a]
            vals.append(v['mean'] if isinstance(v, dict) else v)
        return float(np.mean(vals))

    print(sep)
    print(f"{'AVERAGE':<24} " + " ".join(f"{avg(a):>{col_w}.3f}" for a in ALL_DISPLAY))

    # ---- Type entropy table ----
    print()
    print("Mean final type entropy (higher = more diverse):")
    ent_header = f"{'Benchmark':<24} " + " ".join(f"{n:>{col_w}}" for n in v6_names)
    print(ent_header)
    print("-" * len(ent_header))
    for bench in BENCHMARKS:
        row = " ".join(
            f"{all_results[bench.name][n]['entropy_mean']:>{col_w}.3f}"
            for n in v6_names
        )
        print(f"{bench.name:<24} {row}")
    avg_ent = {n: float(np.mean([all_results[b.name][n]['entropy_mean'] for b in BENCHMARKS]))
               for n in v6_names}
    print("-" * len(ent_header))
    print(f"{'AVERAGE':<24} " + " ".join(f"{avg_ent[n]:>{col_w}.3f}" for n in v6_names))

    # ---- Lambda_div sensitivity ----
    print()
    print(f"lambda_div sensitivity (F1-F8 average PR, {len(SEEDS)} seeds each):")
    lam_raw = {lam: {b.name: [] for b in BENCHMARKS} for lam in LAMBDA_GRID}
    print(f"  Running {len(LAMBDA_GRID)} × {len(BENCHMARKS)} × {len(SEEDS)} = "
          f"{len(LAMBDA_GRID)*len(BENCHMARKS)*len(SEEDS)} runs...")
    for lam in LAMBDA_GRID:
        for bench in BENCHMARKS:
            for s in SEEDS:
                pr, _ = run_one_v6(bench, s, lambda_div=lam)
                lam_raw[lam][bench.name].append(pr)

    print(f"  {'lambda_div':<12} " + " ".join(f"{b.name:>22}" for b in BENCHMARKS) + f"  {'AVG':>8}")
    for lam in LAMBDA_GRID:
        bench_avgs = [float(np.mean(lam_raw[lam][b.name])) for b in BENCHMARKS]
        overall = float(np.mean(bench_avgs))
        print(f"  {lam:<12.2f} " + " ".join(f"{v:>22.3f}" for v in bench_avgs) + f"  {overall:>8.3f}")

    # ---- Wilcoxon: best v6 vs refine-only and CrowdingDE ----
    best_v6 = max(v6_names, key=lambda n: avg(n))
    print()
    print(f"Best v6 variant: {best_v6} (avg PR = {avg(best_v6):.3f})")
    print(f"Wilcoxon signed-rank test ({best_v6} > comparison, one-sided):")
    best_scores = [float(np.mean(v6_raw[b.name][best_v6])) for b in BENCHMARKS]
    for cmp in ['AISO_refine_only', 'CrowdingDE', 'SPSO']:
        if cmp in v6_names:
            cmp_scores = [float(np.mean(v6_raw[b.name][cmp])) for b in BENCHMARKS]
        else:
            cmp_scores = [v5[b.name]['mean'][cmp] for b in BENCHMARKS]
        diff = np.array(best_scores) - np.array(cmp_scores)
        if (diff == 0).all():
            print(f"  vs {cmp:<20}: no difference")
        else:
            try:
                stat, p = stats.wilcoxon(best_scores, cmp_scores, alternative='greater')
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                print(f"  vs {cmp:<20}: W={stat:.0f}, p={p:.4f} {sig}")
            except Exception as e:
                print(f"  vs {cmp:<20}: {e}")

    # ---- Save ----
    out_path = os.path.join(res_dir, 'results_v6.pkl')
    pickle.dump({
        'all_results': all_results,
        'lambda_sensitivity': lam_raw,
        'entropy': v6_entropy,
        'seeds': SEEDS,
    }, open(out_path, 'wb'))
    print(f"\nSaved to {out_path}")
