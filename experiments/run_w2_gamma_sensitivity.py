"""
W2: γ sensitivity for Smart M construction.

Sweeps γ ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 2.0} on a synthetic dataset
matching Elliptic's profile (CV(μ)≈0.667, K=15, n=1500).

Uses MI-weighted proxy fitness (same as governing condition experiment)
to avoid full GNN training overhead. Evaluates AISO(Smart M(γ)) vs random.

Output: results/w2_gamma_results.json + .png
"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

GAMMA_VALUES  = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
N_SEEDS       = 10
K             = 15       # matching Elliptic N_TYPES=15
TARGET_CV     = 0.667    # matching Elliptic CV(μ)


def generate_elliptic_like(K, target_cv, n_samples=1500, n_per_cluster=11,
                            imbalance=0.1, seed=0):
    """Synthetic dataset matching Elliptic profile."""
    rng = np.random.RandomState(seed)
    mu_base = 0.25
    shape   = max(1.0 / target_cv**2, 0.05)
    mu_vals = rng.gamma(shape, mu_base / shape, K)
    mu_vals = np.clip(mu_vals, 0.005, 0.80)

    y = (rng.rand(n_samples) < imbalance).astype(int)
    X_parts = []
    for k in range(K):
        r    = float(np.sqrt(np.clip(mu_vals[k], 0, 0.95)))
        base = r * (2 * y.astype(float) - 1)
        feats = []
        for j in range(n_per_cluster):
            noise = rng.randn(n_samples)
            f     = base + np.sqrt(max(1 - r**2, 1e-4)) * noise
            if j > 0:
                f = 0.6 * feats[0] + 0.4 * f
            feats.append(f)
        X_parts.append(np.column_stack(feats))

    X = StandardScaler().fit_transform(np.hstack(X_parts))
    return X, y


def build_smart_M(X, y, K, gamma):
    C_abs = np.abs(np.corrcoef(X.T))
    dist  = 1.0 - C_abs
    np.fill_diagonal(dist, 0.0)

    labels = AgglomerativeClustering(
        n_clusters=K, metric='precomputed', linkage='average'
    ).fit_predict(dist)

    MI      = mutual_info_classif(X, y, random_state=0)
    mean_MI = np.array([
        MI[labels == k].mean() if (labels == k).any() else 0.0
        for k in range(K)
    ])
    rng_mi  = mean_MI.max() - mean_MI.min()
    MI_norm = (mean_MI - mean_MI.min()) / rng_mi if rng_mi > 1e-9 else np.ones(K) * 0.5

    M = np.zeros((K, K))
    for i in range(K):
        fi = np.where(labels == i)[0]
        for j in range(K):
            fj = np.where(labels == j)[0]
            if i == j:
                M[i, j] = -1.0
            else:
                M[i, j] = -np.mean(C_abs[np.ix_(fi, fj)]) + gamma * (MI_norm[j] - MI_norm[i])

    return M, labels, mean_MI


def make_proxy_fitness(mean_MI, X, labels, K):
    C_abs = np.abs(np.corrcoef(X.T))
    cc    = np.zeros((K, K))
    for i in range(K):
        fi = np.where(labels == i)[0]
        for j in range(K):
            fj = np.where(labels == j)[0]
            if i != j and len(fi) > 0 and len(fj) > 0:
                cc[i, j] = np.mean(C_abs[np.ix_(fi, fj)])

    mu_norm = mean_MI / (mean_MI.sum() + 1e-9)
    lam     = 0.5

    def fitness(positions):
        w        = np.clip(positions, 0, 1)
        w        = w / (w.sum(axis=1, keepdims=True) + 1e-9)
        mi_score = w @ mu_norm
        red      = np.array([
            lam * sum(w[n,i]*w[n,j]*cc[i,j]
                      for i in range(K) for j in range(i+1, K))
            for n in range(len(w))
        ])
        return mi_score - red
    return fitness


def run_aiso(fitness_fn, M, K, n_agents=20, n_iter=60,
             alpha=0.4, beta=0.1, seed=0):
    rng = np.random.RandomState(seed)
    X   = rng.dirichlet(np.ones(K), n_agents)
    W   = rng.dirichlet(np.ones(K), n_agents)

    for t in range(n_iter):
        fits  = fitness_fn(X)
        max_f = fits.max() + 1e-9
        div   = np.mean(np.linalg.norm(X[:, None] - X[None, :], axis=-1))
        w_r   = 1 + 3 * np.exp(-div / 0.12)
        M_eff = np.where(M < 0, M * w_r, M)

        C = (W @ M_eff) @ W.T * (fits / max_f)[None, :]
        np.fill_diagonal(C, -1e18)
        jstar = np.argmax(C, axis=1)
        c_ij  = np.einsum('nk,kj,nj->n', W, M_eff, W[jstar])

        phase = t / n_iter
        if phase < 0.7:
            new_X    = np.clip(X + alpha * c_ij[:, None] * (X[jstar] - X), 0, 1)
            new_fits = fitness_fn(new_X)
            imp      = new_fits > fits
            X[imp]   = new_X[imp]
            if imp.any():
                idx   = np.where(imp)[0]
                new_W = (1 - beta) * W[idx] + beta * W[jstar[idx]]
                W[idx] = new_W / new_W.sum(axis=1, keepdims=True)
        else:
            step  = 0.05 * (1 - phase) / 0.3
            pert  = rng.randn(n_agents, K) * step
            new_X = np.clip(X + pert, 0, 1)
            imp   = fitness_fn(new_X) > fitness_fn(X)
            X[imp] = new_X[imp]

    return float(fitness_fn(X).max())


def run_random(fitness_fn, K, n_trials=50, seed=0):
    rng       = np.random.RandomState(seed)
    positions = rng.dirichlet(np.ones(K), n_trials)
    return float(fitness_fn(positions).max())


def main():
    results = {g: [] for g in GAMMA_VALUES}
    total   = len(GAMMA_VALUES) * N_SEEDS
    done    = 0

    print(f"W2: γ sensitivity sweep ({total} runs)")
    for gamma in GAMMA_VALUES:
        for seed in range(N_SEEDS):
            X, y      = generate_elliptic_like(K, TARGET_CV, seed=seed)
            M, labels, mean_MI = build_smart_M(X, y, K, gamma)
            fn        = make_proxy_fitness(mean_MI, X, labels, K)
            aiso_s    = run_aiso(fn, M, K, seed=seed)
            rand_s    = run_random(fn, K, seed=seed)
            delta     = aiso_s - rand_s
            results[gamma].append({'aiso': aiso_s, 'rand': rand_s, 'delta': delta})
            done += 1
        avg_d = float(np.mean([r['delta'] for r in results[gamma]]))
        avg_a = float(np.mean([r['aiso']  for r in results[gamma]]))
        marker = " ← current" if gamma == 0.5 else ""
        print(f"  γ={gamma:.1f}  avg AISO={avg_a:.4f}  avg Δ={avg_d:+.4f}{marker}",
              flush=True)

    # Summary
    print("\nW2 Summary:")
    print(f"{'γ':>6} {'avg AISO':>10} {'avg Δ':>10} {'std Δ':>10}")
    print("-" * 40)
    for gamma in GAMMA_VALUES:
        deltas = [r['delta'] for r in results[gamma]]
        aisos  = [r['aiso']  for r in results[gamma]]
        marker = " ←" if gamma == 0.5 else ""
        print(f"  {gamma:>4.1f}  {np.mean(aisos):>10.4f}  "
              f"{np.mean(deltas):>+10.4f}  {np.std(deltas):>10.4f}{marker}")

    out = RESULTS_DIR / 'w2_gamma_results.json'
    with open(out, 'w') as f:
        json.dump({str(g): results[g] for g in GAMMA_VALUES}, f, indent=2)
    print(f"\nSaved → {out}")

    visualize(results)


def visualize(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    gammas   = GAMMA_VALUES
    avg_aiso = [np.mean([r['aiso']  for r in results[g]]) for g in gammas]
    avg_delta= [np.mean([r['delta'] for r in results[g]]) for g in gammas]
    std_delta= [np.std( [r['delta'] for r in results[g]]) for g in gammas]

    ax = axes[0]
    ax.plot(gammas, avg_aiso, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='current (γ=0.5)')
    for g, a in zip(gammas, avg_aiso):
        ax.annotate(f'{a:.3f}', (g, a), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xlabel('γ (information gradient weight)', fontsize=11)
    ax.set_ylabel('avg AISO score', fontsize=11)
    ax.set_title('W2: γ Sensitivity — AISO Score', fontsize=11)
    ax.legend(); ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    ax2.errorbar(gammas, avg_delta, yerr=std_delta, fmt='o-',
                 color='#FF5722', linewidth=2, markersize=8, capsize=4)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='current (γ=0.5)')
    ax2.set_xlabel('γ (information gradient weight)', fontsize=11)
    ax2.set_ylabel('Δ (AISO − Random)', fontsize=11)
    ax2.set_title('W2: γ Sensitivity — AISO Advantage', fontsize=11)
    ax2.legend(); ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out = RESULTS_DIR / 'w2_gamma_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure → {out}")
    plt.show()


if __name__ == '__main__':
    main()
