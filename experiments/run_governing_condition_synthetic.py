"""
Synthetic experiment: Governing Condition validation (Section 6).

Tests whether CV(μ) predicts AISO(Smart M) advantage over random baseline.
Uses MI-weighted proxy fitness (fast) instead of full SGD training.

Design mirrors expA_sym_run.py:
  - AgglomerativeClustering on correlation distance
  - Smart M from MI + correlation structure
  - AISO feature-weight optimization

Sweep: CV(μ) ∈ [0.2, 2.0] × K ∈ {4, 8, 12} × 5 seeds → n=135 data points.
"""
import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import json
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. Synthetic dataset with controlled CV(μ)
# ─────────────────────────────────────────────────────────────
def generate_dataset(K, target_cv, n_samples=1500, n_per_cluster=10,
                     imbalance=0.1, seed=0):
    rng = np.random.RandomState(seed)

    # Target μ values via Gamma: CV(Gamma) = 1/sqrt(shape)
    mu_base = 0.25
    if target_cv < 0.05:
        mu_vals = np.full(K, mu_base)
    else:
        shape = max(1.0 / target_cv**2, 0.05)
        mu_vals = rng.gamma(shape, mu_base / shape, K)
        mu_vals = np.clip(mu_vals, 0.005, 0.80)

    actual_cv = mu_vals.std() / (mu_vals.mean() + 1e-9)

    y = (rng.rand(n_samples) < imbalance).astype(int)

    X_parts = []
    for k in range(K):
        r = float(np.sqrt(np.clip(mu_vals[k], 0, 0.95)))
        base = r * (2 * y.astype(float) - 1)
        feats = []
        for j in range(n_per_cluster):
            noise = rng.randn(n_samples)
            f = base + np.sqrt(max(1 - r**2, 1e-4)) * noise
            if j > 0:
                f = 0.6 * feats[0] + 0.4 * f
            feats.append(f)
        X_parts.append(np.column_stack(feats))

    X = StandardScaler().fit_transform(np.hstack(X_parts))
    return X, y, mu_vals, actual_cv


# ─────────────────────────────────────────────────────────────
# 2. Smart M (identical to expA_sym_run.py)
# ─────────────────────────────────────────────────────────────
def build_smart_M(X, y, K, gamma=0.5):
    C_abs = np.abs(np.corrcoef(X.T))
    dist  = 1.0 - C_abs
    np.fill_diagonal(dist, 0.0)

    labels = AgglomerativeClustering(
        n_clusters=K, metric='precomputed', linkage='average'
    ).fit_predict(dist)

    MI = mutual_info_classif(X, y, random_state=0)
    mean_MI = np.array([
        MI[labels == k].mean() if (labels == k).any() else 0.0
        for k in range(K)
    ])
    cv_mu = mean_MI.std() / (mean_MI.mean() + 1e-9)

    rng_mi = mean_MI.max() - mean_MI.min()
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

    return M, labels, mean_MI, cv_mu


# ─────────────────────────────────────────────────────────────
# 3. Fast proxy fitness: diversity-adjusted MI score
#    Rewards selecting high-MI clusters without redundancy
# ─────────────────────────────────────────────────────────────
def make_proxy_fitness(mean_MI, cross_corr, K):
    """
    fitness(w) = sum_k w_k * mu_k  -  lambda * sum_{i<j} w_i*w_j*corr_ij
    Captures the governing condition directly:
    - When CV(mu) is high: one cluster dominates → greedy wins, diversity useless
    - When CV(mu) is moderate: diverse selection beats greedy
    """
    mu_norm = mean_MI / (mean_MI.sum() + 1e-9)
    lam = 0.5

    def fitness(positions):
        # positions: (N, K)
        w = np.clip(positions, 0, 1)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-9)
        mi_score  = w @ mu_norm                          # (N,)
        red_score = np.array([
            lam * np.sum([w[n,i]*w[n,j]*cross_corr[i,j]
                          for i in range(K) for j in range(i+1, K)])
            for n in range(len(w))
        ])
        return mi_score - red_score

    return fitness


def compute_cross_corr(X, labels, K):
    C_abs = np.abs(np.corrcoef(X.T))
    cc = np.zeros((K, K))
    for i in range(K):
        fi = np.where(labels == i)[0]
        for j in range(K):
            fj = np.where(labels == j)[0]
            if i != j and len(fi) > 0 and len(fj) > 0:
                cc[i, j] = np.mean(C_abs[np.ix_(fi, fj)])
    return cc


# ─────────────────────────────────────────────────────────────
# 4. AISO optimizer (lightweight, no aiso_v7 dependency)
# ─────────────────────────────────────────────────────────────
def run_aiso(fitness_fn, M, K, n_agents=20, n_iter=60,
             alpha=0.4, beta=0.1, seed=0):
    rng = np.random.RandomState(seed)
    X   = rng.dirichlet(np.ones(K), n_agents)   # positions on simplex
    W   = rng.dirichlet(np.ones(K), n_agents)   # type vectors

    best_score = -np.inf

    for t in range(n_iter):
        fits   = fitness_fn(X)
        max_f  = fits.max() + 1e-9

        # Adaptive repulsion
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
            # Refinement phase
            step  = 0.05 * (1 - phase) / 0.3
            pert  = rng.randn(n_agents, K) * step
            new_X = np.clip(X + pert, 0, 1)
            imp   = fitness_fn(new_X) > fitness_fn(X)
            X[imp] = new_X[imp]

        best_score = max(best_score, fitness_fn(X).max())

    return float(fitness_fn(X).max())


def run_random(fitness_fn, K, n_trials=50, seed=0):
    rng = np.random.RandomState(seed)
    positions = rng.dirichlet(np.ones(K), n_trials)
    return float(fitness_fn(positions).max())


# ─────────────────────────────────────────────────────────────
# 5. Main sweep
# ─────────────────────────────────────────────────────────────
def main():
    cv_targets = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    K_values   = [4, 8, 12]
    n_seeds    = 5

    results = []
    total   = len(cv_targets) * len(K_values) * n_seeds
    done    = 0

    for K in K_values:
        for cv_target in cv_targets:
            for seed in range(n_seeds):
                X, y, true_mu, actual_cv = generate_dataset(
                    K=K, target_cv=cv_target, seed=seed
                )

                M_smart, labels, est_mu, cv_est = build_smart_M(X, y, K)
                cc = compute_cross_corr(X, labels, K)
                fn = make_proxy_fitness(est_mu, cc, K)

                aiso_score = run_aiso(fn, M_smart, K, seed=seed)
                rand_score = run_random(fn, K, seed=seed)
                delta      = aiso_score - rand_score

                results.append({
                    'K': K, 'cv_target': cv_target,
                    'cv_actual': float(actual_cv),
                    'cv_estimated': float(cv_est),
                    'aiso': float(aiso_score),
                    'rand': float(rand_score),
                    'delta': float(delta),
                    'seed': seed
                })
                done += 1
                print(f"[{done:3d}/{total}] K={K:2d} CV={cv_target:.1f}"
                      f" (act={actual_cv:.3f}) AISO={aiso_score:.4f}"
                      f" Rand={rand_score:.4f} Δ={delta:+.4f}", flush=True)

    # Save
    out = RESULTS_DIR / 'governing_condition_synthetic.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")

    # Stats
    cv_vals    = [r['cv_actual'] for r in results]
    delta_vals = [r['delta'] for r in results]
    rho, pval  = spearmanr(cv_vals, delta_vals)
    n = len(results)
    low  = [r['delta'] for r in results if r['cv_actual'] < 1.0]
    high = [r['delta'] for r in results if r['cv_actual'] >= 1.0]
    print(f"\nSpearman ρ = {rho:.3f}  p = {pval:.4f}  n = {n}")
    print(f"CV<1.0  mean Δ = {np.mean(low):+.4f}  (n={len(low)})")
    print(f"CV≥1.0  mean Δ = {np.mean(high):+.4f}  (n={len(high)})")

    visualize(results)


def visualize(results):
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu, spearmanr

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {4: '#2196F3', 8: '#FF5722', 12: '#4CAF50'}

    ax = axes[0]
    for K in [4, 8, 12]:
        sub = [r for r in results if r['K'] == K]
        cv  = np.array([r['cv_actual'] for r in sub])
        d   = np.array([r['delta'] for r in sub])
        ax.scatter(cv, d, alpha=0.35, color=colors[K], s=25, label=f'K={K}')
        for cv_t in sorted(set(r['cv_target'] for r in sub)):
            g  = [r for r in sub if r['cv_target'] == cv_t]
            xm = np.mean([r['cv_actual'] for r in g])
            ym = np.mean([r['delta'] for r in g])
            ys = np.std([r['delta'] for r in g])
            ax.errorbar(xm, ym, yerr=ys, fmt='o-', color=colors[K],
                        capsize=3, markersize=4, linewidth=1.5)

    # Real datasets overlay
    for name, cv, delta in [('Elliptic', 0.667, 0.111),
                              ('Amazon',   1.601, 0.020),
                              ('YelpChi',  1.733, -0.014)]:
        ax.scatter(cv, delta, marker='*', s=220, zorder=6,
                   color='gold', edgecolors='black', linewidth=1)
        ax.annotate(name, (cv, delta), textcoords='offset points',
                    xytext=(6, 4), fontsize=9)

    all_cv = [r['cv_actual'] for r in results]
    all_d  = [r['delta'] for r in results]
    rho, pval = spearmanr(all_cv, all_d)

    ax.axhline(0,   color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(1.0, color='red',   linestyle='--', linewidth=1.5,
               label='CV(μ)=1.0', alpha=0.8)
    ax.set_xlabel('CV(μ)', fontsize=12)
    ax.set_ylabel('Δ Score  (AISO Smart M − Random)', fontsize=12)
    ax.set_title(f'AISO Advantage vs CV(μ)  [ρ={rho:.3f}, p={pval:.3f}, n={len(results)}]',
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    low  = [r['delta'] for r in results if r['cv_actual'] < 1.0]
    high = [r['delta'] for r in results if r['cv_actual'] >= 1.0]
    ax2.boxplot([low, high],
                labels=[f'CV(μ)<1.0\n(n={len(low)})', f'CV(μ)≥1.0\n(n={len(high)})'],
                patch_artist=True,
                boxprops=dict(facecolor='#E3F2FD'),
                medianprops=dict(color='navy', linewidth=2))
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    stat, p = mannwhitneyu(low, high, alternative='greater')
    ax2.text(0.5, 0.95, f'Mann-Whitney p={p:.4f}',
             transform=ax2.transAxes, ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax2.set_ylabel('Δ Score', fontsize=12)
    ax2.set_title('AISO Advantage by CV(μ) Threshold', fontsize=12)
    ax2.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    out = RESULTS_DIR / 'governing_condition_synthetic.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
