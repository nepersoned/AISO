"""
CEC2013 Niching Benchmark Functions (F1-F8)
============================================
F1: Five-Uneven-Peak Trap   (1D, 2 global peaks)
F2: Equal Maxima            (1D, 5 peaks)
F3: Uneven Decreasing Maxima(1D, 1 global peak)
F4: Himmelblau              (2D, 4 peaks)
F5: Six-Hump Camel Back     (2D, 2 peaks)
F6: Shubert                 (2D, 18 peaks)
F7: Vincent                 (2D, 36 peaks)
F8: Modified Rastrigin-All  (2D, 12 peaks)

F9-F20 (composition functions) require rotation matrix data files from the
official CEC2013 technical report and are not included here.
"""
import numpy as np


class Benchmark:
    def __init__(self, name, fn, bounds, optima, dim, n_peaks):
        self.name = name
        self.fn = fn        # X: (N, D) -> (N,)
        self.bounds = bounds  # (lo, hi)
        self.optima = optima  # list of (D,) arrays
        self.dim = dim
        self.n_peaks = n_peaks


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def five_uneven_peak_trap(X):
    """1D, 2 global peaks (at x=0 and x=30)."""
    x = X[:, 0] if X.ndim > 1 else X
    f = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xi = x[i]
        if 0 <= xi < 2.5:    f[i] = 80 * (2.5 - xi)
        elif xi < 5.0:        f[i] = 64 * (xi - 2.5)
        elif xi < 7.5:        f[i] = 64 * (7.5 - xi)
        elif xi < 12.5:       f[i] = 28 * (xi - 7.5)
        elif xi < 17.5:       f[i] = 28 * (17.5 - xi)
        elif xi < 22.5:       f[i] = 32 * (xi - 17.5)
        elif xi < 27.5:       f[i] = 32 * (27.5 - xi)
        elif xi <= 30:        f[i] = 80 * (xi - 27.5)
    return f


def equal_maxima(X):
    """1D, 5 equal peaks at x = 0.1, 0.3, 0.5, 0.7, 0.9."""
    x = X[:, 0] if X.ndim > 1 else X
    return np.sin(5 * np.pi * x) ** 6


def uneven_decreasing_maxima(X):
    """1D, 1 global peak (multimodal landscape)."""
    x = X[:, 0] if X.ndim > 1 else X
    return (np.exp(-2 * np.log(2) * ((x - 0.08) / 0.854) ** 2)
            * np.sin(5 * np.pi * (x ** 0.75 - 0.05)) ** 6)


def himmelblau(X):
    """2D, 4 global peaks."""
    x, y = X[:, 0], X[:, 1]
    return 200 - (x**2 + y - 11)**2 - (x + y**2 - 7)**2


def six_hump_camel(X):
    """2D, 2 global optima."""
    x, y = X[:, 0], X[:, 1]
    return -(4 - 2.1*x**2 + x**4/3) * x**2 - x*y - (-4 + 4*y**2) * y**2


def shubert(X):
    """2D, 18 global peaks."""
    x, y = X[:, 0], X[:, 1]
    s1 = sum(i * np.cos((i+1)*x + i) for i in range(1, 6))
    s2 = sum(i * np.cos((i+1)*y + i) for i in range(1, 6))
    return -s1 * s2


def vincent(X):
    """2D, 36 global peaks. CEC2013 mean-form: (1/D) * sum sin(10 log(x_i))."""
    Xc = np.clip(X, 1e-10, None)
    return np.mean(np.sin(10 * np.log(Xc)), axis=1)


def modified_rastrigin(X):
    """2D, 12 peaks (k1=3, k2=4 per CEC2013 spec)."""
    k = np.array([3, 4])
    return -np.sum(10 + 9 * np.cos(2 * np.pi * k * X), axis=1)


# ---------------------------------------------------------------------------
# Optima computation helpers
# ---------------------------------------------------------------------------

def _find_shubert_optima(n=800):
    """Locate 18 global optima of Shubert via dense grid scan + greedy clustering."""
    x = np.linspace(-10, 10, n)
    XX, YY = np.meshgrid(x, x)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    f = shubert(pts)
    f_max = f.max()

    mask = f >= f_max - 2.0
    candidates, f_cands = pts[mask], f[mask]

    optima = []
    remaining = np.ones(len(candidates), dtype=bool)
    for i in np.argsort(-f_cands):
        if remaining[i]:
            optima.append(candidates[i].copy())
            remaining[np.linalg.norm(candidates - candidates[i], axis=1) < 0.6] = False

    return [np.array(o) for o in optima]


def _vincent_optima_2d():
    """
    36 global optima of Vincent (mean form) in [0.25, 10]^2.
    Peaks where sin(10*log(x_i)) = 1 for each dimension:
      x = exp(pi*(4k+1)/20) for k in {-2, -1, 0, 1, 2, 3}.
    """
    ks = np.arange(-2, 4)
    xs = np.exp(np.pi * (4*ks + 1) / 20.0)
    return [np.array([xi, xj]) for xi in xs for xj in xs]


# ---------------------------------------------------------------------------
# BENCHMARKS list
# ---------------------------------------------------------------------------

BENCHMARKS = [
    Benchmark('F1_FiveUneven', five_uneven_peak_trap, (0, 30),
              [np.array([0.0]), np.array([30.0])], 1, 2),
    Benchmark('F2_EqualMaxima', equal_maxima, (0, 1),
              [np.array([0.1]), np.array([0.3]), np.array([0.5]),
               np.array([0.7]), np.array([0.9])], 1, 5),
    Benchmark('F3_UnevenDecreasing', uneven_decreasing_maxima, (0, 1),
              [np.array([0.08])], 1, 1),
    Benchmark('F4_Himmelblau', himmelblau, (-6, 6),
              [np.array([3.0, 2.0]),
               np.array([-2.805118, 3.131312]),
               np.array([-3.779310, -3.283186]),
               np.array([3.584428, -1.848126])], 2, 4),
    Benchmark('F5_SixHumpCamel', six_hump_camel, (-1.9, 1.9),
              [np.array([0.0898, -0.7126]),
               np.array([-0.0898, 0.7126])], 2, 2),
    Benchmark('F6_Shubert', shubert, (-10, 10),
              _find_shubert_optima(), 2, 18),
    Benchmark('F7_Vincent', vincent, (0.25, 10),
              _vincent_optima_2d(), 2, 36),
    Benchmark('F8_ModifiedRastrigin', modified_rastrigin, (0, 1),
              [np.array([(2*i-1)/6, (2*j-1)/8])
               for i in range(1, 4) for j in range(1, 5)],
              2, 12),
]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def peak_ratio(positions, optima, accuracy=0.01, bounds=None):
    """Fraction of known peaks covered by at least one agent within `accuracy`."""
    scale = (bounds[1] - bounds[0]) if bounds else 1.0
    thresh = accuracy * scale
    covered = sum(
        1 for opt in optima
        if np.linalg.norm(positions - opt, axis=1).min() < thresh
    )
    return covered / len(optima)


def success_rate(positions, optima, accuracy=0.01, bounds=None):
    """1 if ALL peaks covered, else 0."""
    return 1.0 if peak_ratio(positions, optima, accuracy, bounds) == 1.0 else 0.0


if __name__ == '__main__':
    for b in BENCHMARKS:
        X = np.random.uniform(b.bounds[0], b.bounds[1], (50, b.dim))
        f = b.fn(X)
        print(f"{b.name}: dim={b.dim}, peaks={b.n_peaks}, "
              f"n_optima={len(b.optima)}, f=[{f.min():.3f}, {f.max():.3f}]")
