"""
AISO v5: Adaptive Anchor + Fitness-Gated Coupling
===================================================
Builds on v4 with two targeted fixes for the coupling mechanism.

Fix A — Adaptive Anchor Repositioning:
  When an agent succeeds (fitness improves), its dominant type's anchor
  is nudged toward the new position at rate eta_A.
  Over time, anchors migrate from random positions toward actual peaks,
  eliminating false-attractor behavior.

Fix C — Fitness-Gated Coupling:
  Coupling strength is multiplied by f(A_k*) / f_max (the fitness at
  the anchor location, normalized).  Anchors in poor regions exert
  near-zero pull; anchors near peaks exert full pull.

Flags (all bool):
  coupling         — enable the anchor-directed exploration term
  adaptive_anchor  — enable Fix A  (requires coupling=True)
  fitness_gated    — enable Fix C  (requires coupling=True)
  refinement       — enable phase-2 local random walk

Typical configurations
  AISO_base        : coupling=F, refinement=F
  AISO_v3          : coupling=T, refinement=F
  AISO_refine_only : coupling=F, refinement=T   (Fix B ablation)
  AISO_v4          : coupling=T, refinement=T
  AISO_v5_A        : coupling=T, adaptive_anchor=T, refinement=T
  AISO_v5_AC       : coupling=T, adaptive_anchor=T, fitness_gated=T, refinement=T
"""
import numpy as np


class AISO_v5:
    def __init__(self, n_agents, dim, K, bounds, M=None,
                 alpha=0.4, beta=0.1, gamma=0.3,
                 coupling=True, refinement=True,
                 adaptive_anchor=False, fitness_gated=False,
                 eta_A=0.1, refine_after=0.7, seed=42):

        self.n_agents = n_agents
        self.dim = dim
        self.K = K
        self.bounds = bounds
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.coupling = coupling
        self.refinement = refinement
        self.adaptive_anchor = adaptive_anchor and coupling
        self.fitness_gated = fitness_gated and coupling
        self.eta_A = eta_A
        self.refine_after = refine_after
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
            raise ValueError("Call run() instead of update().")

        fits = fn(self.X)
        max_f = fits.max() + 1e-9

        # Adaptive repulsion: scale negative M entries by diversity
        sub_idx = np.arange(0, self.n_agents, 5)
        sub = self.X[sub_idx]
        d = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
        div = d[np.triu_indices(len(sub_idx), 1)].mean() / (self.bounds[1] - self.bounds[0])
        w_r = 1 + 3 * np.exp(-div / 0.12)
        M_eff = np.where(self.M < 0, self.M * w_r, self.M)

        # Compatibility-weighted partner selection
        WM = self.W @ M_eff
        C = (WM @ self.W.T) * (fits / max_f)[None, :]
        np.fill_diagonal(C, -1e18)
        jstar = np.argmax(C, axis=1)
        c_ij = np.einsum('nk,kj,nj->n', self.W, M_eff, self.W[jstar])

        phase = self.iter_count / self.total_iters

        if phase < self.refine_after:
            # ----------------------------------------------------------------
            # Phase 1: AISO global search
            # ----------------------------------------------------------------
            if self.coupling:
                dom_type = np.argmax(self.W, axis=1)                          # (N,)
                type_strength = self.W[np.arange(self.n_agents), dom_type]    # (N,)
                anchor_dir = self.anchors[dom_type] - self.X                  # (N, dim)
                partner_dir = self.X[jstar] - self.X                          # (N, dim)

                # Fix C: gate coupling by fitness at each anchor
                if self.fitness_gated:
                    anchor_fits = fn(self.anchors)                             # (K,)
                    gate = np.clip(anchor_fits[dom_type] / max_f, 0.0, 1.0)   # (N,)
                else:
                    gate = np.ones(self.n_agents)

                new_X = self.X + self.alpha * (
                    c_ij[:, None] * partner_dir
                    + self.gamma * type_strength[:, None] * gate[:, None] * anchor_dir
                )
            else:
                new_X = self.X + self.alpha * c_ij[:, None] * (self.X[jstar] - self.X)

            new_X = np.clip(new_X, self.bounds[0], self.bounds[1])
            new_fits = fn(new_X)
            improved = new_fits > fits
            self.X[improved] = new_X[improved]

            if improved.any():
                idx = np.where(improved)[0]

                # Type assimilation
                new_W = (1 - self.beta) * self.W[idx] + self.beta * self.W[jstar[idx]]
                self.W[idx] = new_W / new_W.sum(axis=1, keepdims=True)

                # Fix A: move dominant type's anchor toward new position
                if self.adaptive_anchor:
                    for i in idx:
                        k_star = np.argmax(self.W[i])
                        self.anchors[k_star] = (
                            (1 - self.eta_A) * self.anchors[k_star]
                            + self.eta_A * self.X[i]
                        )
        else:
            # ----------------------------------------------------------------
            # Phase 2: local refinement (greedy random walk)
            # ----------------------------------------------------------------
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


if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from benchmarks import BENCHMARKS, peak_ratio

    configs = [
        ('AISO_base',       dict(coupling=False, refinement=False)),
        ('AISO_refine_only',dict(coupling=False, refinement=True)),
        ('AISO_v4',         dict(coupling=True,  refinement=True)),
        ('AISO_v5_A',       dict(coupling=True,  refinement=True, adaptive_anchor=True)),
        ('AISO_v5_AC',      dict(coupling=True,  refinement=True, adaptive_anchor=True,
                                 fitness_gated=True)),
    ]
    print(f"{'Benchmark':<22} " + " ".join(f"{n:>14}" for n, _ in configs))
    for b in BENCHMARKS:
        row = []
        for name, kw in configs:
            alg = AISO_v5(80, b.dim, max(4, b.n_peaks), b.bounds, seed=42, **kw)
            alg.run(b.fn, 200)
            row.append(peak_ratio(alg.X, b.optima, 0.01, b.bounds))
        print(f"{b.name:<22} " + " ".join(f"{v:>14.3f}" for v in row))
