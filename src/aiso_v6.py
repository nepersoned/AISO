"""
AISO v6: W-Reinforcement Mechanisms
=====================================
Extends AISO refine-only (coupling=OFF, refinement=ON) with four W-space
enhancements designed to prevent type collapse and strengthen specialization.

Mechanisms:
  lambda_div     — diversity penalty: push each W_i away from population mean
                   applied every iteration to ALL agents
  smart_init     — K-means spatial clustering at init; agents in cluster k get
                   W_i = purity*e_k + (1-purity)*Dirichlet(1)
  sparsity_gamma — power-sharpening: W_i ∝ W_i^gamma (gamma>1 → specialist)
                   applied every iteration to ALL agents
  anti_assim_mu  — anti-assimilation: subtract mu*W_enemy from W on improvement
                   (enemy = least compatible agent)
"""
import numpy as np


def _kmeans(X, K, rng, max_iter=30):
    """Numpy-only K-means. Returns cluster label for each row of X."""
    n = len(X)
    centers = X[rng.choice(n, min(K, n), replace=False)].copy()
    if len(centers) < K:
        extra = rng.uniform(X.min(0), X.max(0), (K - len(centers), X.shape[1]))
        centers = np.vstack([centers, extra])
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers[None], axis=-1)  # (N, K)
        new_labels = np.argmin(dists, axis=1)
        if (new_labels == labels).all():
            break
        labels = new_labels
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
    return labels


class AISO_v6:
    def __init__(self, n_agents, dim, K, bounds, M=None,
                 alpha=0.4, beta=0.1, gamma_coup=0.3,
                 refinement=True,
                 lambda_div=0.0,
                 smart_init=False, smart_init_alpha=0.7,
                 sparsity_gamma=1.0,
                 anti_assim_mu=0.0,
                 refine_after=0.7, seed=42):

        self.n_agents = n_agents
        self.dim = dim
        self.K = K
        self.bounds = bounds
        self.alpha = alpha
        self.beta = beta
        self.gamma_coup = gamma_coup
        self.refinement = refinement
        self.lambda_div = lambda_div
        self.smart_init_alpha = smart_init_alpha
        self.sparsity_gamma = sparsity_gamma
        self.anti_assim_mu = anti_assim_mu
        self.refine_after = refine_after
        self.rng = np.random.RandomState(seed)

        if M is None:
            self.M = self.rng.uniform(-1, 1, (K, K))
            np.fill_diagonal(self.M, 0)
        else:
            self.M = M.copy()

        self.X = self.rng.uniform(bounds[0], bounds[1], (n_agents, dim))

        if smart_init:
            labels = _kmeans(self.X, K, self.rng)
            W = np.zeros((n_agents, K))
            for i in range(n_agents):
                e_k = np.zeros(K); e_k[labels[i] % K] = 1.0
                d = self.rng.dirichlet(np.ones(K))
                W[i] = smart_init_alpha * e_k + (1 - smart_init_alpha) * d
            self.W = W / W.sum(axis=1, keepdims=True)
        else:
            self.W = self.rng.dirichlet(np.ones(K), n_agents)

        self.iter_count = 0
        self.total_iters = None

    def type_entropy(self):
        """Mean Shannon entropy of W vectors (higher = more diverse types)."""
        W_safe = np.clip(self.W, 1e-10, None)
        return float(-np.sum(self.W * np.log(W_safe), axis=1).mean())

    def _normalize_W(self, W):
        W = np.clip(W, 0, None)
        s = W.sum(axis=1, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        return W / s

    def update(self, fn):
        if self.total_iters is None:
            raise ValueError("Call run() instead of update().")

        fits = fn(self.X)
        max_f = fits.max() + 1e-9

        # Adaptive repulsion
        sub_idx = np.arange(0, self.n_agents, 5)
        sub = self.X[sub_idx]
        d = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
        div = d[np.triu_indices(len(sub_idx), 1)].mean() / (self.bounds[1] - self.bounds[0])
        w_r = 1 + 3 * np.exp(-div / 0.12)
        M_eff = np.where(self.M < 0, self.M * w_r, self.M)

        # Compatibility matrix and partner selection
        WM = self.W @ M_eff
        C = (WM @ self.W.T) * (fits / max_f)[None, :]
        np.fill_diagonal(C, -1e18)
        jstar = np.argmax(C, axis=1)
        c_ij = np.einsum('nk,kj,nj->n', self.W, M_eff, self.W[jstar])

        phase = self.iter_count / self.total_iters

        if phase < self.refine_after:
            # Phase 1: global search (no coupling)
            new_X = self.X + self.alpha * c_ij[:, None] * (self.X[jstar] - self.X)
            new_X = np.clip(new_X, self.bounds[0], self.bounds[1])
            new_fits = fn(new_X)
            improved = new_fits > fits
            self.X[improved] = new_X[improved]

            if improved.any():
                idx = np.where(improved)[0]

                # Enemy for anti-assimilation
                if self.anti_assim_mu > 0:
                    C_neg = C.copy()
                    np.fill_diagonal(C_neg, 1e18)
                    jenemy = np.argmin(C_neg, axis=1)

                new_W = (1 - self.beta) * self.W[idx] + self.beta * self.W[jstar[idx]]

                if self.anti_assim_mu > 0:
                    new_W -= self.anti_assim_mu * self.W[jenemy[idx]]

                self.W[idx] = self._normalize_W(new_W)

            # Diversity penalty: all agents every iteration
            if self.lambda_div > 0:
                mean_W = self.W.mean(axis=0)
                self.W = self._normalize_W(
                    self.W + self.lambda_div * (self.W - mean_W)
                )

            # Sparsity: all agents every iteration
            if self.sparsity_gamma != 1.0:
                self.W = self._normalize_W(self.W ** self.sparsity_gamma)

        else:
            # Phase 2: local refinement
            if self.refinement:
                step = (0.05 * (self.bounds[1] - self.bounds[0])
                        * (1 - phase) / (1 - self.refine_after + 1e-9))
                perturbation = self.rng.randn(self.n_agents, self.dim) * step
                new_X = np.clip(self.X + perturbation, self.bounds[0], self.bounds[1])
                new_fits = fn(new_X)
                improved = new_fits > fits
                self.X[improved] = new_X[improved]

        self.iter_count += 1

    def run(self, fn, n_iter):
        self.total_iters = n_iter
        for _ in range(n_iter):
            self.update(fn)
        return self.X
