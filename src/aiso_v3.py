"""
AISO v3 with Type-Position Coupling
====================================
Key innovation: Dominant type explicitly determines exploration direction
"""
import numpy as np

class AISO_v3:
    def __init__(self, n_agents, dim, K, bounds, M=None,
                 alpha=0.4, beta=0.1, coupling=True, seed=42):
        self.n_agents = n_agents
        self.dim = dim
        self.K = K
        self.bounds = bounds  # (lo, hi)
        self.alpha = alpha
        self.beta = beta
        self.coupling = coupling
        self.rng = np.random.RandomState(seed)

        # Asymmetric M (random or provided)
        if M is None:
            self.M = self.rng.uniform(-1, 1, (K, K))
            np.fill_diagonal(self.M, 0)
        else:
            self.M = M.copy()

        # Type-region anchors (for coupling)
        # Each type k has a "preferred direction" in search space
        self.anchors = self.rng.uniform(bounds[0], bounds[1], (K, dim))

        # Initialize
        self.X = self.rng.uniform(bounds[0], bounds[1], (n_agents, dim))
        self.W = self.rng.dirichlet(np.ones(K), n_agents)

    def update(self, fitness_fn):
        fits = fitness_fn(self.X)
        max_f = fits.max() + 1e-9

        # Adaptive repulsion
        sub_idx = np.arange(0, self.n_agents, 5)
        sub = self.X[sub_idx]
        d = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
        div = d[np.triu_indices(len(sub_idx), 1)].mean() / (self.bounds[1] - self.bounds[0])
        w_r = 1 + 3 * np.exp(-div / 0.12)
        M_eff = np.where(self.M < 0, self.M * w_r, self.M)

        # Compatibility matrix: C[i,j] = W_i^T M_eff W_j * fits[j]/max_f
        WM = self.W @ M_eff
        C = (WM @ self.W.T) * (fits / max_f)[None, :]
        np.fill_diagonal(C, -1e18)
        jstar = np.argmax(C, axis=1)

        # c_ij values
        c_ij = np.einsum('nk,kj,nj->n', self.W, M_eff, self.W[jstar])

        # Position update
        if self.coupling:
            # Type-Position Coupling: blend partner direction with anchor direction
            dom_type = np.argmax(self.W, axis=1)
            anchor_dir = self.anchors[dom_type] - self.X  # toward dominant type's anchor
            partner_dir = self.X[jstar] - self.X
            # Coupling weight: stronger when type is more concentrated
            type_strength = self.W[np.arange(self.n_agents), dom_type]  # max W value
            coupling_w = type_strength[:, None]
            new_X = self.X + self.alpha * (
                c_ij[:, None] * partner_dir +
                0.3 * coupling_w * anchor_dir
            )
        else:
            new_X = self.X + self.alpha * c_ij[:, None] * (self.X[jstar] - self.X)

        new_X = np.clip(new_X, self.bounds[0], self.bounds[1])
        new_fits = fitness_fn(new_X)
        improved = new_fits > fits

        # Only update improved agents (paper rule)
        self.X[improved] = new_X[improved]
        # Type assimilation on improvement
        if improved.any():
            idx = np.where(improved)[0]
            new_W = (1 - self.beta) * self.W[idx] + self.beta * self.W[jstar[idx]]
            self.W[idx] = new_W / new_W.sum(axis=1, keepdims=True)

        return self.X, self.W, fitness_fn(self.X)


if __name__ == '__main__':
    # quick smoke test
    def himmelblau(X):
        x, y = X[:, 0], X[:, 1]
        return -((x**2 + y - 11)**2 + (x + y**2 - 7)**2)

    aiso = AISO_v3(n_agents=50, dim=2, K=4, bounds=(-5, 5), coupling=True)
    for _ in range(100):
        X, W, f = aiso.update(himmelblau)
    print(f"AISO v3 (coupling=True): max fitness = {f.max():.4f}")
    print(f"Final positions sample:\n{X[:5]}")
