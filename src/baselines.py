"""
Baseline algorithms for comparison
==================================
1. SPSO   - Species-based PSO (Li 2004)
2. CrowdingDE - Crowding Differential Evolution
3. LIPS   - Locally Informed PSO
"""
import numpy as np


class SPSO:
    def __init__(self, n_particles, dim, bounds, r_species=0.1,
                 w=0.7, c1=1.5, c2=1.5, seed=42):
        self.n = n_particles; self.dim = dim; self.bounds = bounds
        self.r = r_species * (bounds[1] - bounds[0])
        self.w, self.c1, self.c2 = w, c1, c2
        self.rng = np.random.RandomState(seed)
        self.pos = self.rng.uniform(bounds[0], bounds[1], (n_particles, dim))
        self.vel = self.rng.uniform(-0.1, 0.1, (n_particles, dim))
        self.pbest = self.pos.copy()
        self.pbest_fit = None

    def update(self, fn):
        fits = fn(self.pos)
        if self.pbest_fit is None: self.pbest_fit = fits.copy()

        # Species formation
        order = np.argsort(-fits)
        seeds = []
        assigned = np.zeros(self.n, dtype=bool)
        for idx in order:
            if not assigned[idx]:
                seeds.append(idx)
                d = np.linalg.norm(self.pos - self.pos[idx], axis=1)
                assigned[d < self.r] = True

        gbest_of = {}
        for s in seeds:
            d = np.linalg.norm(self.pos - self.pos[s], axis=1)
            for m in np.where(d < self.r)[0]: gbest_of[m] = s

        # Update
        for i in range(self.n):
            g = gbest_of.get(i, seeds[0])
            r1, r2 = self.rng.rand(self.dim), self.rng.rand(self.dim)
            self.vel[i] = (self.w * self.vel[i]
                           + self.c1 * r1 * (self.pbest[i] - self.pos[i])
                           + self.c2 * r2 * (self.pos[g] - self.pos[i]))
            v_max = 0.2 * (self.bounds[1] - self.bounds[0])
            self.vel[i] = np.clip(self.vel[i], -v_max, v_max)
            self.pos[i] = np.clip(self.pos[i] + self.vel[i],
                                  self.bounds[0], self.bounds[1])

        new_fit = fn(self.pos)
        imp = new_fit > self.pbest_fit
        self.pbest[imp] = self.pos[imp]; self.pbest_fit[imp] = new_fit[imp]
        return self.pos


class CrowdingDE:
    def __init__(self, n_pop, dim, bounds, F=0.5, CR=0.9, seed=42):
        self.n = n_pop; self.dim = dim; self.bounds = bounds
        self.F = F; self.CR = CR
        self.rng = np.random.RandomState(seed)
        self.pop = self.rng.uniform(bounds[0], bounds[1], (n_pop, dim))

    def update(self, fn):
        fits = fn(self.pop)
        new_pop = self.pop.copy()
        for i in range(self.n):
            # Mutation: pick 3 random distinct
            idxs = [j for j in range(self.n) if j != i]
            a, b, c = self.rng.choice(idxs, 3, replace=False)
            mutant = self.pop[a] + self.F * (self.pop[b] - self.pop[c])
            mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

            # Crossover
            cross_mask = self.rng.rand(self.dim) < self.CR
            if not cross_mask.any():
                cross_mask[self.rng.randint(self.dim)] = True
            trial = np.where(cross_mask, mutant, self.pop[i])

            # Crowding: replace nearest
            dists = np.linalg.norm(self.pop - trial, axis=1)
            nearest = np.argmin(dists)
            if fn(trial[None, :])[0] > fits[nearest]:
                new_pop[nearest] = trial

        self.pop = new_pop
        return self.pop


class LIPS:
    """Locally Informed PSO — each particle informed by nsize nearest pbests"""
    def __init__(self, n_particles, dim, bounds, nsize=3, seed=42):
        self.n = n_particles; self.dim = dim; self.bounds = bounds
        self.nsize = nsize
        self.rng = np.random.RandomState(seed)
        self.pos = self.rng.uniform(bounds[0], bounds[1], (n_particles, dim))
        self.vel = np.zeros((n_particles, dim))
        self.pbest = self.pos.copy()
        self.pbest_fit = None
        self.chi = 0.7298  # constriction

    def update(self, fn):
        fits = fn(self.pos)
        if self.pbest_fit is None: self.pbest_fit = fits.copy()

        for i in range(self.n):
            # nsize nearest pbests
            d = np.linalg.norm(self.pbest - self.pos[i], axis=1)
            d[i] = np.inf
            nbrs = np.argsort(d)[:self.nsize]

            # Random weights for each neighbor's pbest
            phi = 4.1 / self.nsize
            P = np.zeros(self.dim)
            for nb in nbrs:
                r = self.rng.uniform(0, phi, self.dim)
                P += r * (self.pbest[nb] - self.pos[i])
            P /= self.nsize

            self.vel[i] = self.chi * (self.vel[i] + P)
            v_max = 0.2 * (self.bounds[1] - self.bounds[0])
            self.vel[i] = np.clip(self.vel[i], -v_max, v_max)
            self.pos[i] = np.clip(self.pos[i] + self.vel[i],
                                  self.bounds[0], self.bounds[1])

        new_fit = fn(self.pos)
        imp = new_fit > self.pbest_fit
        self.pbest[imp] = self.pos[imp]; self.pbest_fit[imp] = new_fit[imp]
        return self.pos


if __name__ == '__main__':
    def himmelblau(X):
        x, y = X[:, 0], X[:, 1]
        return 200 - (x**2+y-11)**2 - (x+y**2-7)**2

    for cls in [SPSO, CrowdingDE, LIPS]:
        algo = cls(50, 2, (-6, 6), seed=42)
        for _ in range(50): algo.update(himmelblau)
        pos = algo.pos if hasattr(algo, 'pos') else algo.pop
        print(f"{cls.__name__}: best fitness = {himmelblau(pos).max():.4f}")
