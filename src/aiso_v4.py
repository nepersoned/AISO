"""
AISO v4: Type-Position Coupling + Local Refinement
====================================================
Innovation 1: Type-Position Coupling (anchor-guided exploration)
Innovation 2: Hybrid Local Search (gradient-free refinement near convergence)

Rationale: AISO's strength is diversity preservation across modes;
its weakness is per-mode refinement.
Solution: Decoupled design - global diversity (AISO) + local refinement (Nelder-Mead-lite)
"""
import numpy as np


class AISO_v4:
    def __init__(self, n_agents, dim, K, bounds, M=None,
                 alpha=0.4, beta=0.1, coupling=True, refinement=True,
                 refine_after=0.7, seed=42):
        self.n_agents = n_agents
        self.dim = dim; self.K = K; self.bounds = bounds
        self.alpha = alpha; self.beta = beta
        self.coupling = coupling
        self.refinement = refinement
        self.refine_after = refine_after  # fraction of iters before local search kicks in
        self.rng = np.random.RandomState(seed)

        if M is None:
            self.M = self.rng.uniform(-1, 1, (K, K))
            np.fill_diagonal(self.M, 0)
        else:
            self.M = M.copy()

        self.anchors = self.rng.uniform(bounds[0], bounds[1], (K, dim))
        self.X = self.rng.uniform(bounds[0], bounds[1], (n_agents, dim))
        self.W = self.rng.dirichlet(np.ones(K), n_agents)
        self.iter_count = 0
        self.total_iters = None

    def update(self, fn):
        if self.total_iters is None:
            raise ValueError("Call run() instead of update() to use refinement.")
        fits = fn(self.X)
        max_f = fits.max() + 1e-9

        # Adaptive repulsion
        sub_idx = np.arange(0, self.n_agents, 5)
        sub = self.X[sub_idx]
        d = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
        div = d[np.triu_indices(len(sub_idx), 1)].mean() / (self.bounds[1] - self.bounds[0])
        w_r = 1 + 3 * np.exp(-div / 0.12)
        M_eff = np.where(self.M < 0, self.M * w_r, self.M)

        # Compatibility
        WM = self.W @ M_eff
        C = (WM @ self.W.T) * (fits / max_f)[None, :]
        np.fill_diagonal(C, -1e18)
        jstar = np.argmax(C, axis=1)
        c_ij = np.einsum('nk,kj,nj->n', self.W, M_eff, self.W[jstar])

        # Phase 1: AISO global search (diversity preservation)
        phase = self.iter_count / self.total_iters
        if phase < self.refine_after:
            if self.coupling:
                dom_type = np.argmax(self.W, axis=1)
                anchor_dir = self.anchors[dom_type] - self.X
                partner_dir = self.X[jstar] - self.X
                type_strength = self.W[np.arange(self.n_agents), dom_type]
                new_X = self.X + self.alpha * (
                    c_ij[:, None] * partner_dir +
                    0.3 * type_strength[:, None] * anchor_dir
                )
            else:
                new_X = self.X + self.alpha * c_ij[:, None] * (self.X[jstar] - self.X)
            new_X = np.clip(new_X, self.bounds[0], self.bounds[1])
            new_fits = fn(new_X)
            improved = new_fits > fits
            self.X[improved] = new_X[improved]
            if improved.any():
                idx = np.where(improved)[0]
                new_W = (1 - self.beta) * self.W[idx] + self.beta * self.W[jstar[idx]]
                self.W[idx] = new_W / new_W.sum(axis=1, keepdims=True)
        else:
            # Phase 2: Local refinement (small random walk + greedy)
            if self.refinement:
                # adaptive step: smaller as we get closer to end
                step = 0.05 * (self.bounds[1] - self.bounds[0]) * (1 - phase) / (1 - self.refine_after + 1e-9)
                perturbation = self.rng.randn(self.n_agents, self.dim) * step
                new_X = np.clip(self.X + perturbation, self.bounds[0], self.bounds[1])
                new_fits = fn(new_X)
                improved = new_fits > fits
                self.X[improved] = new_X[improved]

        self.iter_count += 1

    def run(self, fn, n_iter):
        self.total_iters = n_iter
        for _ in range(n_iter): self.update(fn)
        return self.X


if __name__ == '__main__':
    import sys; sys.path.insert(0, '/home/claude/aiso_paper')
    from benchmarks import BENCHMARKS, peak_ratio

    print(f"{'Benchmark':<22} {'PR':>6}")
    for b in BENCHMARKS:
        aiso = AISO_v4(80, b.dim, max(4, b.n_peaks), b.bounds,
                       coupling=True, refinement=True, seed=42)
        aiso.run(b.fn, 200)
        pr = peak_ratio(aiso.X, b.optima, 0.01, b.bounds)
        print(f"{b.name:<22} {pr:>6.3f}")
