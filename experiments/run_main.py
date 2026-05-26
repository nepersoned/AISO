"""
Main experiment: AISO-v4 vs baselines on CEC2013 niching benchmarks.
Run from repo root: python experiments/run_main.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pickle
from scipy import stats

from aiso_v3 import AISO_v3
from aiso_v4 import AISO_v4
from baselines import SPSO, CrowdingDE, LIPS, RingPSO
from benchmarks import BENCHMARKS, peak_ratio

N_AGENTS = 80
N_ITER   = 200
SEEDS    = list(range(30))
ACCURACY = 0.01

ALGS = ['AISO_v4', 'AISO_v3', 'AISO_base', 'SPSO', 'CrowdingDE', 'LIPS', 'RingPSO']


def run_one(bench, seed):
    results = {}
    bounds, dim, K = bench.bounds, bench.dim, max(4, bench.n_peaks)

    aiso_v4 = AISO_v4(N_AGENTS, dim, K, bounds, coupling=True, refinement=True, seed=seed)
    aiso_v4.run(bench.fn, N_ITER)
    results['AISO_v4'] = peak_ratio(aiso_v4.X, bench.optima, ACCURACY, bounds)

    aiso_v3 = AISO_v3(N_AGENTS, dim, K, bounds, coupling=True, seed=seed)
    for _ in range(N_ITER): aiso_v3.update(bench.fn)
    results['AISO_v3'] = peak_ratio(aiso_v3.X, bench.optima, ACCURACY, bounds)

    aiso_base = AISO_v3(N_AGENTS, dim, K, bounds, coupling=False, seed=seed)
    for _ in range(N_ITER): aiso_base.update(bench.fn)
    results['AISO_base'] = peak_ratio(aiso_base.X, bench.optima, ACCURACY, bounds)

    spso = SPSO(N_AGENTS, dim, bounds, r_species=0.1, seed=seed)
    for _ in range(N_ITER): spso.update(bench.fn)
    results['SPSO'] = peak_ratio(spso.pos, bench.optima, ACCURACY, bounds)

    cde = CrowdingDE(N_AGENTS, dim, bounds, seed=seed)
    for _ in range(N_ITER): cde.update(bench.fn)
    results['CrowdingDE'] = peak_ratio(cde.pop, bench.optima, ACCURACY, bounds)

    lips = LIPS(N_AGENTS, dim, bounds, seed=seed)
    for _ in range(N_ITER): lips.update(bench.fn)
    results['LIPS'] = peak_ratio(lips.pos, bench.optima, ACCURACY, bounds)

    ring = RingPSO(N_AGENTS, dim, bounds, seed=seed)
    for _ in range(N_ITER): ring.update(bench.fn)
    results['RingPSO'] = peak_ratio(ring.pos, bench.optima, ACCURACY, bounds)

    return results


def _wilcoxon_row(aiso_scores, base_scores, name):
    """Wilcoxon signed-rank test: AISO_v4 > baseline (one-sided)."""
    diff = np.array(aiso_scores) - np.array(base_scores)
    if diff.sum() == 0:
        return f"  vs {name:<12}: no difference"
    try:
        stat, p = stats.wilcoxon(aiso_scores, base_scores, alternative='greater')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        return f"  vs {name:<12}: W={stat:.0f}, p={p:.4f} {sig}"
    except Exception as e:
        return f"  vs {name:<12}: {e}"


if __name__ == '__main__':
    print(f"Running {len(BENCHMARKS)} benchmarks × {len(SEEDS)} seeds × {len(ALGS)} algorithms")
    print(f"{'Benchmark':<24} " + " ".join(f"{a:>10}" for a in ALGS))
    print("-" * (26 + 11 * len(ALGS)))

    # all_results[bench_name][alg] = list of per-seed peak ratios
    all_results = {}
    for bench in BENCHMARKS:
        sr = {a: [] for a in ALGS}
        for s in SEEDS:
            r = run_one(bench, s)
            for k, v in r.items():
                sr[k].append(v)
        means = {k: float(np.mean(v)) for k, v in sr.items()}
        stds  = {k: float(np.std(v))  for k, v in sr.items()}
        all_results[bench.name] = {'mean': means, 'std': stds, 'raw': sr}
        print(f"{bench.name:<24} " + " ".join(f"{means[a]:>10.3f}" for a in ALGS))

    print("-" * (26 + 11 * len(ALGS)))
    overall_mean = {a: float(np.mean([all_results[b]['mean'][a] for b in all_results]))
                    for a in ALGS}
    overall_std  = {a: float(np.std([all_results[b]['mean'][a] for b in all_results]))
                    for a in ALGS}
    print(f"{'AVERAGE':<24} " + " ".join(f"{overall_mean[a]:>10.3f}" for a in ALGS))

    # Wilcoxon signed-rank tests (per-benchmark means as observations)
    print("\nWilcoxon signed-rank test (AISO_v4 > baseline, one-sided):")
    aiso_scores = [all_results[b]['mean']['AISO_v4'] for b in all_results]
    for baseline in ['SPSO', 'CrowdingDE', 'LIPS', 'RingPSO', 'AISO_v3', 'AISO_base']:
        base_scores = [all_results[b]['mean'][baseline] for b in all_results]
        print(_wilcoxon_row(aiso_scores, base_scores, baseline))

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'results_final.pkl')
    pickle.dump({'all_results': all_results, 'overall_mean': overall_mean,
                 'overall_std': overall_std, 'seeds': SEEDS}, open(out, 'wb'))
    print(f"\nSaved to {out}")
