import numpy as np
import warnings
warnings.filterwarnings('ignore')

N_TYPES = 8; N_AGENTS = 20; N_ITER = 100; N_RUNS = 5
K = 20; NOISE_LIST = [0, 0.5, 1.0, 2.0, 5.0, 10.0]

def make_noisy_rastrigin(K, sigma):
    A = 10
    def f(X):
        x = X * 10.24 - 5.12
        return -(A*K + np.sum(x**2 - A*np.cos(2*np.pi*x))) + np.random.normal(0, sigma)
    return f

def run_aiso(f, K, seed, n_iter=N_ITER):
    rng = np.random.RandomState(seed)
    n_types = N_TYPES; n_ag = N_AGENTS
    M = rng.randn(n_types, n_types) * 0.5
    np.fill_diagonal(M, -0.5)
    W = rng.dirichlet(np.ones(n_types), n_ag)
    X = rng.rand(n_ag, K)
    s = np.array([f(X[i]) for i in range(n_ag)])
    best = max(s)
    alpha, beta = 0.2, 0.08; w_repel = 2.0
    for _ in range(n_iter):
        C = W @ M @ W.T
        for i in range(n_ag):
            sc = s / (np.max(np.abs(s)) + 1e-8)
            attract = np.argsort(C[i])[-3:]
            repel   = np.argsort(C[i])[:3]
            div = np.mean([np.linalg.norm(X[i]-X[j]) for j in range(n_ag) if j!=i])
            wr = 1.0 + 3.0 * np.exp(-div / 0.12)
            F = sum(C[i,j]*(X[j]-X[i]) for j in attract) + wr*sum(C[i,j]*(X[j]-X[i]) for j in repel)
            Xn = np.clip(X[i] + alpha*F, 0, 1)
            sn = f(Xn)
            if sn > s[i]:
                X[i] = Xn; s[i] = sn
                j_star = attract[np.argmax([C[i,j]*sc[j] for j in attract])]
                Wn = (1-beta)*W[i] + beta*W[j_star]
                W[i] = Wn / Wn.sum()
        best = max(best, max(s))
    return best

def run_pso(f, K, seed, n_iter=N_ITER):
    rng = np.random.RandomState(seed)
    n = N_AGENTS; w,c1,c2 = 0.729,1.494,1.494
    X = rng.rand(n, K); V = rng.randn(n, K)*0.1
    pbest = X.copy(); pbest_s = np.array([f(X[i]) for i in range(n)])
    gbest = X[np.argmax(pbest_s)].copy(); gbest_s = max(pbest_s)
    for _ in range(n_iter):
        r1,r2 = rng.rand(n,K), rng.rand(n,K)
        V = w*V + c1*r1*(pbest-X) + c2*r2*(gbest-X)
        X = np.clip(X+V, 0, 1)
        s = np.array([f(X[i]) for i in range(n)])
        imp = s > pbest_s; pbest[imp]=X[imp]; pbest_s[imp]=s[imp]
        if max(s) > gbest_s: gbest_s=max(s); gbest=X[np.argmax(s)].copy()
    return gbest_s

print("Running noise experiment...")
results = {}
for sigma in NOISE_LIST:
    f = make_noisy_rastrigin(K, sigma)
    for aname, fn in [('AISO', run_aiso), ('PSO', run_pso)]:
        scores = [fn(f, K, seed) for seed in range(N_RUNS)]
        results[(sigma, aname)] = np.mean(scores)
        print(f"  sigma={sigma}  {aname}  {np.mean(scores):.3f}")

print("\n=== NOISE RESULTS ===")
print(f"{'sigma':<8} {'AISO':>10} {'PSO':>10} {'delta%':>10}")
for sigma in NOISE_LIST:
    a = results[(sigma,'AISO')]; p = results[(sigma,'PSO')]
    delta = (a-p)/abs(p)*100 if p != 0 else 0
    print(f"{sigma:<8} {a:>10.3f} {p:>10.3f} {delta:>+10.1f}%")

# relative to sigma=0
a0 = results[(0,'AISO')]; p0 = results[(0,'PSO')]
print(f"\nAt sigma=10:")
print(f"  AISO change: {(results[(10,'AISO')]-a0)/abs(a0)*100:+.1f}%")
print(f"  PSO  change: {(results[(10,'PSO')]-p0)/abs(p0)*100:+.1f}%")
