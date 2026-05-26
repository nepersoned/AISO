"""
F6 (Shubert) + F8 (Modified Rastrigin) 수정 버전 단독 실험.

수정 사항:
- F6: Shubert optima grid 해상도 2000으로 향상 (기존 800)
- F8: optima 공식 수정 (i/3,j/4) → ((2i-1)/6,(2j-1)/8)
      기존 optima는 전부 global minimum(f=-38)이었음 → global max(f=-2)로 교정

Run from repo root: python experiments/run_f68_fixed.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pickle
from scipy import stats

from aiso_v3 import AISO_v3
from aiso_v4 import AISO_v4
from baselines import SPSO, CrowdingDE, LIPS, RingPSO
from benchmarks import (Benchmark, shubert, modified_rastrigin,
                        _find_shubert_optima, peak_ratio)

N_AGENTS = 80
N_ITER   = 200
SEEDS    = list(range(30))
ACCURACY = 0.01

ALGS = ['AISO_v4', 'AISO_v3', 'AISO_base', 'SPSO', 'CrowdingDE', 'LIPS', 'RingPSO']


# ---------------------------------------------------------------------------
# 수정된 벤치마크 정의
# ---------------------------------------------------------------------------

def _find_shubert_optima_hires(n=2000):
    """Shubert optima: grid 해상도 2000으로 향상."""
    x = np.linspace(-10, 10, n)
    XX, YY = np.meshgrid(x, x)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    f = shubert(pts)
    f_max = f.max()

    mask = f >= f_max - 1.0   # 더 타이트한 threshold
    candidates, f_cands = pts[mask], f[mask]

    optima = []
    remaining = np.ones(len(candidates), dtype=bool)
    for i in np.argsort(-f_cands):
        if remaining[i]:
            optima.append(candidates[i].copy())
            remaining[np.linalg.norm(candidates - candidates[i], axis=1) < 0.5] = False

    return [np.array(o) for o in optima]


F6_FIXED = Benchmark(
    'F6_Shubert_fixed', shubert, (-10, 10),
    _find_shubert_optima_hires(), 2, 18
)

F8_FIXED = Benchmark(
    'F8_ModRastrigin_fixed', modified_rastrigin, (0, 1),
    [np.array([(2*i-1)/6, (2*j-1)/8])
     for i in range(1, 4) for j in range(1, 5)],
    2, 12
)

FIXED_BENCHMARKS = [F6_FIXED, F8_FIXED]


# ---------------------------------------------------------------------------
# 실험 루틴
# ---------------------------------------------------------------------------

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


if __name__ == '__main__':
    print(f"F6 optima count: {len(F6_FIXED.optima)}")
    # F8 optima 값 확인
    opt_vals = [modified_rastrigin(np.array([o]))[0] for o in F8_FIXED.optima]
    print(f"F8 optima f-values: {set(round(v,1) for v in opt_vals)} (should be {{-2.0}})")
    print()
    print(f"Running 2 benchmarks × {len(SEEDS)} seeds × {len(ALGS)} algorithms")
    print(f"{'Benchmark':<26} " + " ".join(f"{a:>10}" for a in ALGS))
    print("-" * (28 + 11 * len(ALGS)))

    all_results = {}
    for bench in FIXED_BENCHMARKS:
        sr = {a: [] for a in ALGS}
        for s in SEEDS:
            r = run_one(bench, s)
            for k, v in r.items():
                sr[k].append(v)
        means = {k: float(np.mean(v)) for k, v in sr.items()}
        stds  = {k: float(np.std(v))  for k, v in sr.items()}
        all_results[bench.name] = {'mean': means, 'std': stds, 'raw': sr}
        print(f"{bench.name:<26} " + " ".join(f"{means[a]:>10.3f}" for a in ALGS))

    # Wilcoxon
    print("\nWilcoxon signed-rank (AISO_v4 > baseline, one-sided, 2 benchmarks):")
    aiso_scores = [all_results[b]['mean']['AISO_v4'] for b in all_results]
    for baseline in ['SPSO', 'CrowdingDE', 'LIPS', 'RingPSO']:
        base_scores = [all_results[b]['mean'][baseline] for b in all_results]
        diff = np.array(aiso_scores) - np.array(base_scores)
        if diff.sum() == 0:
            print(f"  vs {baseline:<12}: no difference")
        else:
            try:
                stat, p = stats.wilcoxon(aiso_scores, base_scores, alternative='greater')
                print(f"  vs {baseline:<12}: W={stat:.0f}, p={p:.4f}")
            except Exception as e:
                print(f"  vs {baseline:<12}: {e}")

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'results_f68_fixed.pkl')
    pickle.dump(all_results, open(out, 'wb'))
    print(f"\nSaved to {out}")
