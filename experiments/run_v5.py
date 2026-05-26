"""
AISO v5 experiment: 기존 결과 재활용 + 신규 3개 변형만 추가 실행.

SOURCE:
  F1-F7  (old 7 algs) : results_final.pkl
  F8     (old 7 algs) : results_f68_fixed.pkl  (corrected optima)
  NEW 3 variants       : run fresh on all 8 benchmarks

NEW (runs from scratch):
  AISO_refine_only  coupling=F, refinement=T
  AISO_v5_A         coupling=T, adaptive_anchor=T, refinement=T
  AISO_v5_AC        coupling=T, adaptive_anchor=T, fitness_gated=T, refinement=T

Run from repo root: python experiments/run_v5.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pickle
from scipy import stats

from aiso_v5 import AISO_v5
from benchmarks import BENCHMARKS, peak_ratio

N_AGENTS = 80
N_ITER   = 200
SEEDS    = list(range(30))
ACCURACY = 0.01

NEW_CONFIGS = [
    ('AISO_refine_only', dict(coupling=False, refinement=True)),
    ('AISO_v5_A',        dict(coupling=True,  refinement=True, adaptive_anchor=True)),
    ('AISO_v5_AC',       dict(coupling=True,  refinement=True, adaptive_anchor=True,
                              fitness_gated=True)),
]

AISO_ORDER = ['AISO_base', 'AISO_v3', 'AISO_refine_only', 'AISO_v4', 'AISO_v5_A', 'AISO_v5_AC']
BASELINE_ORDER = ['SPSO', 'CrowdingDE', 'LIPS', 'RingPSO']
ALL_ALGS = AISO_ORDER + BASELINE_ORDER


def run_new(bench, seed):
    results = {}
    bounds, dim = bench.bounds, bench.dim
    K = max(4, bench.n_peaks)
    for name, kw in NEW_CONFIGS:
        alg = AISO_v5(N_AGENTS, dim, K, bounds, seed=seed, **kw)
        alg.run(bench.fn, N_ITER)
        results[name] = peak_ratio(alg.X, bench.optima, ACCURACY, bounds)
    return results


if __name__ == '__main__':
    # ---- Load existing results ----
    res_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    existing = pickle.load(open(os.path.join(res_dir, 'results_final.pkl'), 'rb'))['all_results']
    f68_fixed = pickle.load(open(os.path.join(res_dir, 'results_f68_fixed.pkl'), 'rb'))
    # Overwrite F8 with corrected-optima results (key name differs)
    existing['F8_ModifiedRastrigin'] = f68_fixed['F8_ModRastrigin_fixed']

    # ---- Run new variants ----
    new_names = [n for n, _ in NEW_CONFIGS]
    print(f"Running {len(BENCHMARKS)} benchmarks × {len(SEEDS)} seeds × {len(new_names)} new variants")

    new_raw = {b.name: {n: [] for n in new_names} for b in BENCHMARKS}
    for bench in BENCHMARKS:
        for s in SEEDS:
            r = run_new(bench, s)
            for n in new_names:
                new_raw[bench.name][n].append(r[n])

    # ---- Merge into unified all_results ----
    all_results = {}
    for bench in BENCHMARKS:
        merged_raw  = {a: existing[bench.name]['raw'][a] for a in existing[bench.name]['raw']}
        merged_mean = {a: existing[bench.name]['mean'][a] for a in existing[bench.name]['mean']}
        merged_std  = {a: existing[bench.name]['std'][a]  for a in existing[bench.name]['std']}
        for n in new_names:
            merged_raw[n]  = new_raw[bench.name][n]
            merged_mean[n] = float(np.mean(new_raw[bench.name][n]))
            merged_std[n]  = float(np.std(new_raw[bench.name][n]))
        all_results[bench.name] = {'mean': merged_mean, 'std': merged_std, 'raw': merged_raw}

    # ---- Main results table ----
    col_w = 13
    header = f"{'Benchmark':<24} " + " ".join(f"{a:>{col_w}}" for a in ALL_ALGS)
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for bench in BENCHMARKS:
        means = all_results[bench.name]['mean']
        print(f"{bench.name:<24} " + " ".join(f"{means[a]:>{col_w}.3f}" for a in ALL_ALGS))
    avgs = {a: float(np.mean([all_results[b.name]['mean'][a] for b in BENCHMARKS]))
            for a in ALL_ALGS}
    print(sep)
    print(f"{'AVERAGE':<24} " + " ".join(f"{avgs[a]:>{col_w}.3f}" for a in ALL_ALGS))

    # ---- Ablation table ----
    print()
    print("=" * 55)
    print("Ablation (AISO variants only)")
    print("=" * 55)
    abl_header = f"{'Benchmark':<24} " + " ".join(f"{a:>16}" for a in AISO_ORDER)
    print(abl_header)
    print("-" * len(abl_header))
    for bench in BENCHMARKS:
        means = all_results[bench.name]['mean']
        print(f"{bench.name:<24} " + " ".join(f"{means[a]:>16.3f}" for a in AISO_ORDER))
    abl_avgs = {a: float(np.mean([all_results[b.name]['mean'][a] for b in BENCHMARKS]))
                for a in AISO_ORDER}
    print("-" * len(abl_header))
    print(f"{'AVERAGE':<24} " + " ".join(f"{abl_avgs[a]:>16.3f}" for a in AISO_ORDER))

    # ---- Wilcoxon (AISO_v5_AC > each, one-sided) ----
    print()
    print("Wilcoxon signed-rank test (AISO_v5_AC > comparison, one-sided):")
    v5ac = [all_results[b.name]['mean']['AISO_v5_AC'] for b in BENCHMARKS]
    for cmp in BASELINE_ORDER + ['AISO_v4']:
        base = [all_results[b.name]['mean'][cmp] for b in BENCHMARKS]
        diff = np.array(v5ac) - np.array(base)
        if (diff == 0).all():
            print(f"  vs {cmp:<16}: no difference")
        else:
            try:
                stat, p = stats.wilcoxon(v5ac, base, alternative='greater')
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                print(f"  vs {cmp:<16}: W={stat:.0f}, p={p:.4f} {sig}")
            except Exception as e:
                print(f"  vs {cmp:<16}: {e}")

    # ---- Save ----
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'results_v5.pkl')
    pickle.dump({'all_results': all_results, 'seeds': SEEDS}, open(out_path, 'wb'))
    print(f"\nSaved to {out_path}")
