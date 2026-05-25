"""
Main experiment: AISO-v4 vs baselines on CEC2013 niching benchmarks.
Run from repo root: python experiments/run_main.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pickle
from aiso_v3 import AISO_v3
from aiso_v4 import AISO_v4
from baselines import SPSO, CrowdingDE, LIPS
from benchmarks import BENCHMARKS, peak_ratio

N_AGENTS = 80
N_ITER = 200
SEEDS = [42, 7, 77, 123, 0]
ACCURACY = 0.01


def run_b(bench, seed):
    results = {}
    bounds, dim, K = bench.bounds, bench.dim, max(4, bench.n_peaks)

    aiso_v4 = AISO_v4(N_AGENTS, dim, K, bounds, coupling=True, refinement=True, seed=seed)
    aiso_v4.run(bench.fn, N_ITER)
    results['AISO_v4'] = peak_ratio(aiso_v4.X, bench.optima, ACCURACY, bounds)

    aiso_couple = AISO_v3(N_AGENTS, dim, K, bounds, coupling=True, seed=seed)
    for _ in range(N_ITER): aiso_couple.update(bench.fn)
    results['AISO_v3'] = peak_ratio(aiso_couple.X, bench.optima, ACCURACY, bounds)

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

    return results


if __name__ == '__main__':
    algs = ['AISO_v4', 'AISO_v3', 'AISO_base', 'SPSO', 'CrowdingDE', 'LIPS']
    print(f"{'Benchmark':<22} " + " ".join(f"{a:>10}" for a in algs))
    print("-" * (24 + 11*len(algs)))

    all_results = {}
    for bench in BENCHMARKS:
        sr = {a: [] for a in algs}
        for s in SEEDS:
            r = run_b(bench, s)
            for k, v in r.items(): sr[k].append(v)
        means = {k: np.mean(v) for k, v in sr.items()}
        all_results[bench.name] = means
        print(f"{bench.name:<22} " + " ".join(f"{means[a]:>10.3f}" for a in algs))

    print("-" * (24 + 11*len(algs)))
    overall = {a: np.mean([all_results[b][a] for b in all_results]) for a in algs}
    print(f"{'AVERAGE':<22} " + " ".join(f"{overall[a]:>10.3f}" for a in algs))

    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'results'), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), '..', 'results', 'results_final.pkl')
    pickle.dump(all_results, open(out, 'wb'))
    print(f"\nSaved to {out}")
