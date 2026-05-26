"""
AISO v7 ablation: 9 mechanisms vs refine-only baseline.
Each variant adds exactly one mechanism to refine-only.

Run from repo root: python experiments/run_v7_ablation.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pickle
from scipy import stats

from aiso_v7 import AISO_v7
from benchmarks import BENCHMARKS, peak_ratio

N_AGENTS = 80
N_ITER   = 200
SEEDS    = list(range(30))
ACCURACY = 0.01

VARIANTS = [
    ('Baseline',   dict()),
    ('M1_Hebb',    dict(hebbian_M=True)),
    ('M3_Spar',    dict(sparse_M=True)),
    ('M5_Cyc',     dict(cyclic_init_M=True)),
    ('W1_FW',      dict(fitness_weighted_beta=True)),
    ('W7_Asym',    dict(asymmetric_assim=True)),
    ('WM1_Joint',  dict(joint_hebbian=True)),
    ('S4_Niche',   dict(niche_detect=True)),
    ('S5_Surr',    dict(surrogate=True)),
    ('S7_Mem',     dict(memetic=True)),
]

VAR_NAMES = [n for n, _ in VARIANTS]
M_VARIANTS = {'M1_Hebb', 'M3_Spar', 'M5_Cyc', 'WM1_Joint'}


def run_one(bench, seed, **kw):
    K   = max(4, bench.n_peaks)
    alg = AISO_v7(N_AGENTS, bench.dim, K, bench.bounds, seed=seed, **kw)
    M0  = alg.M_norm()
    alg.run(bench.fn, N_ITER)
    pr  = peak_ratio(alg.X, bench.optima, ACCURACY, bench.bounds)
    ent = alg.type_entropy()
    Mf  = alg.M_norm()
    return pr, ent, M0, Mf


if __name__ == '__main__':
    n_runs = len(VARIANTS) * len(BENCHMARKS) * len(SEEDS)
    print(f"Running {len(VARIANTS)} variants × {len(BENCHMARKS)} benchmarks × {len(SEEDS)} seeds = {n_runs} total")

    raw     = {n: {b.name: [] for b in BENCHMARKS} for n in VAR_NAMES}
    entropy = {n: {b.name: [] for b in BENCHMARKS} for n in VAR_NAMES}
    M_init  = {n: {b.name: [] for b in BENCHMARKS} for n in VAR_NAMES}
    M_final = {n: {b.name: [] for b in BENCHMARKS} for n in VAR_NAMES}

    for bench in BENCHMARKS:
        for s in SEEDS:
            for name, kw in VARIANTS:
                pr, ent, m0, mf = run_one(bench, s, **kw)
                raw[name][bench.name].append(pr)
                entropy[name][bench.name].append(ent)
                M_init[name][bench.name].append(m0)
                M_final[name][bench.name].append(mf)

    def mean(name, bname): return float(np.mean(raw[name][bname]))
    def std_(name, bname): return float(np.std(raw[name][bname]))
    def avg(name): return float(np.mean([mean(name, b.name) for b in BENCHMARKS]))

    # ---- PR table ----
    cw = 10
    header = f"{'Benchmark':<24} " + " ".join(f"{n:>{cw}}" for n in VAR_NAMES)
    sep    = "-" * len(header)
    print(); print(header); print(sep)
    for bench in BENCHMARKS:
        print(f"{bench.name:<24} " + " ".join(f"{mean(n, bench.name):>{cw}.3f}" for n in VAR_NAMES))
    print(sep)
    print(f"{'AVERAGE':<24} " + " ".join(f"{avg(n):>{cw}.3f}" for n in VAR_NAMES))

    # ---- Delta + Wilcoxon table ----
    print()
    print(f"{'Variant':<12} {'Delta PR':>10} {'Wilcoxon p':>12} {'Sig':>5} {'Avg STD':>9} {'Unstable':>9}")
    print("-" * 62)
    base_per_bench = [float(np.mean(raw['Baseline'][b.name])) for b in BENCHMARKS]
    for name, _ in VARIANTS[1:]:
        var_per_bench = [float(np.mean(raw[name][b.name])) for b in BENCHMARKS]
        delta = avg(name) - avg('Baseline')
        avg_std = float(np.mean([std_(name, b.name) for b in BENCHMARKS]))
        unstable = "YES" if avg_std > 0.2 else ""
        diff = np.array(var_per_bench) - np.array(base_per_bench)
        if (diff == 0).all():
            pstr, sig = "N/A", ""
        else:
            try:
                stat, p = stats.wilcoxon(var_per_bench, base_per_bench, alternative='greater')
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                pstr = f"{p:.4f}"
            except Exception as e:
                pstr, sig = str(e)[:10], ""
        print(f"{name:<12} {delta:>+10.3f} {pstr:>12} {sig:>5} {avg_std:>9.3f} {unstable:>9}")

    # ---- Type entropy table ----
    print()
    print("Mean final type entropy:")
    print(f"{'Benchmark':<24} " + " ".join(f"{n:>{cw}}" for n in VAR_NAMES))
    print("-" * len(header))
    for bench in BENCHMARKS:
        row = " ".join(
            f"{float(np.mean(entropy[n][bench.name])):>{cw}.3f}" for n in VAR_NAMES
        )
        print(f"{bench.name:<24} {row}")
    avg_ent = {n: float(np.mean([np.mean(entropy[n][b.name]) for b in BENCHMARKS]))
               for n in VAR_NAMES}
    print("-" * len(header))
    print(f"{'AVERAGE':<24} " + " ".join(f"{avg_ent[n]:>{cw}.3f}" for n in VAR_NAMES))

    # ---- M Frobenius norm table ----
    print()
    print("M Frobenius norm (M-modifying variants):")
    print(f"{'Variant':<14} {'|M|_F initial':>16} {'|M|_F final':>14}")
    print("-" * 46)
    for name in M_VARIANTS:
        m0 = float(np.mean([np.mean(M_init[name][b.name]) for b in BENCHMARKS]))
        mf = float(np.mean([np.mean(M_final[name][b.name]) for b in BENCHMARKS]))
        print(f"{name:<14} {m0:>16.3f} {mf:>14.3f}")
    # Baseline for reference
    m0b = float(np.mean([np.mean(M_init['Baseline'][b.name]) for b in BENCHMARKS]))
    mfb = float(np.mean([np.mean(M_final['Baseline'][b.name]) for b in BENCHMARKS]))
    print(f"{'Baseline':<14} {m0b:>16.3f} {mfb:>14.3f}")

    # ---- Save ----
    res_dir  = os.path.join(os.path.dirname(__file__), '..', 'results')
    out_path = os.path.join(res_dir, 'results_v7.pkl')
    pickle.dump({'raw': raw, 'entropy': entropy, 'M_init': M_init,
                 'M_final': M_final, 'seeds': SEEDS}, open(out_path, 'wb'))
    print(f"\nSaved to {out_path}")
