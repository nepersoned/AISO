"""
AISO v7: 9-mechanism ablation framework
========================================
Base: refine-only (coupling=OFF, refinement=ON, M fixed).

Mechanism flags (all default False):
  hebbian_M          — M1: Hebbian M update on improvement
  sparse_M           — M3: zero small M entries every 50 iters
  cyclic_init_M      — M5: cyclic asymmetric M initialization
  fitness_weighted_beta — W1: scale beta by fitness improvement
  asymmetric_assim   — W7: repel W when c_ij < 0
  joint_hebbian      — WM1: hebbian_M + fitness_weighted_beta simultaneously
  niche_detect       — S4: DBSCAN niche-restricted j* selection
  surrogate          — S5: RF surrogate weighting for j* selection
  memetic            — S7: mini local search every 20 iters (phase 1)
"""
import numpy as np


class AISO_v7:
    def __init__(self, n_agents, dim, K, bounds, M=None,
                 alpha=0.4, beta=0.1,
                 refinement=True,
                 hebbian_M=False, eta_M=0.01,
                 sparse_M=False,
                 cyclic_init_M=False,
                 fitness_weighted_beta=False,
                 asymmetric_assim=False,
                 joint_hebbian=False,
                 niche_detect=False,
                 surrogate=False,
                 memetic=False,
                 refine_after=0.7, seed=42):

        self.n_agents  = n_agents
        self.dim       = dim
        self.K         = K
        self.bounds    = bounds
        self.alpha     = alpha
        self.beta      = beta
        self.refinement = refinement
        self.eta_M     = eta_M
        self.refine_after = refine_after
        self.rng       = np.random.RandomState(seed)

        # Resolve joint_hebbian
        if joint_hebbian:
            hebbian_M = True
            fitness_weighted_beta = True
        self.hebbian_M           = hebbian_M
        self.sparse_M            = sparse_M
        self.cyclic_init_M       = cyclic_init_M
        self.fitness_weighted_beta = fitness_weighted_beta
        self.asymmetric_assim    = asymmetric_assim
        self.niche_detect        = niche_detect
        self.surrogate           = surrogate
        self.memetic             = memetic

        # M initialization
        if M is not None:
            self.M = M.copy()
        elif cyclic_init_M:
            self.M = np.zeros((K, K))
            for i in range(K):
                self.M[i, (i + 1) % K]  =  1.0
                self.M[i, (i + 2) % K]  =  0.5
                self.M[i, (i - 1) % K]  = -0.3
        else:
            self.M = self.rng.uniform(-1, 1, (K, K))
            np.fill_diagonal(self.M, 0)

        self.X = self.rng.uniform(bounds[0], bounds[1], (n_agents, dim))
        self.W = self.rng.dirichlet(np.ones(K), n_agents)

        self.iter_count  = 0
        self.total_iters = None

        # S4: niche labels cache
        self._niche_labels = None

        # S5: surrogate
        if self.surrogate:
            from sklearn.ensemble import RandomForestRegressor
            self._surr_model   = RandomForestRegressor(
                n_estimators=20, max_depth=5, random_state=seed, n_jobs=1)
            self._surr_trained = False
            self._X_hist       = []
            self._f_hist       = []

    # ------------------------------------------------------------------
    def type_entropy(self):
        W_safe = np.clip(self.W, 1e-10, None)
        return float(-np.sum(self.W * np.log(W_safe), axis=1).mean())

    def M_norm(self):
        return float(np.linalg.norm(self.M, 'fro'))

    def _normalize_W(self, W):
        W = np.clip(W, 0, None)
        s = W.sum(axis=1, keepdims=True)
        return W / np.where(s == 0, 1.0, s)

    # ------------------------------------------------------------------
    def update(self, fn):
        if self.total_iters is None:
            raise ValueError("Call run() instead of update().")

        fits = fn(self.X)
        max_f = fits.max() + 1e-9

        # Adaptive repulsion
        sub_idx = np.arange(0, self.n_agents, 5)
        sub     = self.X[sub_idx]
        d       = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
        div     = d[np.triu_indices(len(sub_idx), 1)].mean() / (self.bounds[1] - self.bounds[0])
        w_r     = 1 + 3 * np.exp(-div / 0.12)
        M_eff   = np.where(self.M < 0, self.M * w_r, self.M)

        # S5: collect history + retrain surrogate
        if self.surrogate:
            self._X_hist.append(self.X.copy())
            self._f_hist.append(fits.copy())
            if self.iter_count % 100 == 0 and len(self._X_hist) >= 5:
                X_tr = np.vstack(self._X_hist[-10:])
                f_tr = np.concatenate(self._f_hist[-10:])
                self._surr_model.fit(X_tr, f_tr)
                self._surr_trained = True

        # S4: DBSCAN niche detection every 20 iters
        if self.niche_detect and self.iter_count % 20 == 0:
            from sklearn.cluster import DBSCAN
            eps = 0.1 * (self.bounds[1] - self.bounds[0])
            self._niche_labels = DBSCAN(eps=eps, min_samples=3).fit(self.X).labels_

        # Compatibility matrix
        WM  = self.W @ M_eff
        raw_C = WM @ self.W.T                                          # (N, N)

        if self.surrogate and self._surr_trained:
            surr_j = np.clip(self._surr_model.predict(self.X) / max_f, 0, None)
            C = raw_C * (fits / max_f)[None, :] * surr_j[None, :]
        else:
            C = raw_C * (fits / max_f)[None, :]

        np.fill_diagonal(C, -1e18)

        # S4: restrict j* to same niche
        if self.niche_detect and self._niche_labels is not None:
            labels = self._niche_labels
            same   = (labels[:, None] == labels[None, :])              # (N, N)
            noise  = (labels < 0)[:, None]
            C_sel  = np.where(same | noise, C, -1e18)
            np.fill_diagonal(C_sel, -1e18)
            jstar  = np.argmax(C_sel, axis=1)
        else:
            jstar  = np.argmax(C, axis=1)

        c_ij = np.einsum('nk,kj,nj->n', self.W, M_eff, self.W[jstar])

        phase = self.iter_count / self.total_iters

        if phase < self.refine_after:
            # ---- S7: memetic mini local search (phase 1, every 20 iters) ----
            if self.memetic and self.iter_count % 20 == 0:
                step      = 0.02 * (self.bounds[1] - self.bounds[0])
                curr_fits = fits.copy()
                for _ in range(3):
                    pert     = self.rng.randn(self.n_agents, self.dim) * step
                    X_pert   = np.clip(self.X + pert, self.bounds[0], self.bounds[1])
                    pf       = fn(X_pert)
                    imp      = pf > curr_fits
                    self.X[imp]     = X_pert[imp]
                    curr_fits[imp]  = pf[imp]

            # ---- Phase 1: global search ----
            new_X     = self.X + self.alpha * c_ij[:, None] * (self.X[jstar] - self.X)
            new_X     = np.clip(new_X, self.bounds[0], self.bounds[1])
            new_fits  = fn(new_X)
            improved  = new_fits > fits
            self.X[improved] = new_X[improved]

            if improved.any():
                idx      = np.where(improved)[0]
                delta_f  = (new_fits[idx] - fits[idx]) / max_f         # (|idx|,)

                # Adaptive beta (W1)
                if self.fitness_weighted_beta:
                    ab = np.clip(self.beta * (1 + delta_f), 0.01, 0.5) # (|idx|,)
                else:
                    ab = np.full(len(idx), self.beta)

                new_W = (1 - ab[:, None]) * self.W[idx] + ab[:, None] * self.W[jstar[idx]]

                # Asymmetric assimilation (W7): repel when c_ij < 0
                if self.asymmetric_assim:
                    neg    = c_ij[idx] < 0
                    repel  = np.clip((1 - ab[:, None]) * self.W[idx]
                                     - ab[:, None] * self.W[jstar[idx]], 0, None)
                    new_W  = np.where(neg[:, None], repel, new_W)

                self.W[idx] = self._normalize_W(new_W)

                # Hebbian M update (M1 / WM1)
                if self.hebbian_M:
                    for ii, i in enumerate(idx):
                        dM = self.eta_M * delta_f[ii] * np.outer(self.W[i], self.W[jstar[i]])
                        self.M += dM
                        np.fill_diagonal(self.M, 0)
                    # Decay + norm constraint
                    self.M *= 0.999
                    frob = np.linalg.norm(self.M, 'fro')
                    if frob > 5.0:
                        self.M *= 5.0 / frob
                    np.fill_diagonal(self.M, 0)

            # Sparse M (M3): every 50 iters
            if self.sparse_M and self.iter_count % 50 == 0:
                thresh = 0.1 * np.abs(self.M).mean()
                self.M[np.abs(self.M) < thresh] = 0.0
                np.fill_diagonal(self.M, 0)

        else:
            # ---- Phase 2: local refinement ----
            if self.refinement:
                step  = (0.05 * (self.bounds[1] - self.bounds[0])
                         * (1 - phase) / (1 - self.refine_after + 1e-9))
                pert  = self.rng.randn(self.n_agents, self.dim) * step
                new_X = np.clip(self.X + pert, self.bounds[0], self.bounds[1])
                imp   = fn(new_X) > fn(self.X)
                self.X[imp] = new_X[imp]

        self.iter_count += 1

    def run(self, fn, n_iter):
        self.total_iters = n_iter
        for _ in range(n_iter):
            self.update(fn)
        return self.X
