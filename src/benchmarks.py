"""
CEC2013 Niching Benchmark Functions (subset)
=============================================
F1: Five-Uneven-Peak Trap   (1D, 5 peaks)
F2: Equal Maxima            (1D, 5 peaks)
F3: Uneven Decreasing Maxima(1D, 5 peaks)
F4: Himmelblau              (2D, 4 peaks)
F5: Six-Hump Camel Back     (2D, 2 peaks)
F6: Shubert                 (2D, 18 peaks)
F7: Vincent                 (2D, 36 peaks)
F8: Modified Rastrigin      (2D, 12 peaks)

Each returns: fitness function, bounds, known optima list
"""
import numpy as np

class Benchmark:
    def __init__(self, name, fn, bounds, optima, dim, n_peaks):
        self.name = name
        self.fn = fn       # X: (N,D) -> (N,)
        self.bounds = bounds  # (lo, hi)
        self.optima = optima  # list of (D,) arrays
        self.dim = dim
        self.n_peaks = n_peaks


def five_uneven_peak_trap(X):
    """1D, 5 peaks of varying heights"""
    x = X[:, 0] if X.ndim > 1 else X
    f = np.zeros_like(x)
    for i in range(len(x)):
        xi = x[i]
        if 0 <= xi < 2.5:   f[i] = 80 * (2.5 - xi)
        elif xi < 5.0:       f[i] = 64 * (xi - 2.5)
        elif xi < 7.5:       f[i] = 64 * (7.5 - xi)
        elif xi < 12.5:      f[i] = 28 * (xi - 7.5)
        elif xi < 17.5:      f[i] = 28 * (17.5 - xi)
        elif xi < 22.5:      f[i] = 32 * (xi - 17.5)
        elif xi < 27.5:      f[i] = 32 * (27.5 - xi)
        elif xi <= 30:       f[i] = 80 * (xi - 27.5)
    return f


def equal_maxima(X):
    """1D, 5 equal peaks at x = 0.1, 0.3, 0.5, 0.7, 0.9"""
    x = X[:, 0] if X.ndim > 1 else X
    return np.sin(5 * np.pi * x) ** 6


def uneven_decreasing_maxima(X):
    """1D, 5 peaks with decreasing heights"""
    x = X[:, 0] if X.ndim > 1 else X
    return np.exp(-2 * np.log(2) * ((x - 0.08) / 0.854) ** 2) * \
           np.sin(5 * np.pi * (x ** 0.75 - 0.05)) ** 6


def himmelblau(X):
    """2D, 4 peaks"""
    x, y = X[:, 0], X[:, 1]
    return 200 - (x**2 + y - 11)**2 - (x + y**2 - 7)**2


def six_hump_camel(X):
    """2D, 2 global optima"""
    x, y = X[:, 0], X[:, 1]
    return -(4 - 2.1*x**2 + x**4/3) * x**2 - x*y - (-4 + 4*y**2) * y**2


def shubert(X):
    """2D, 18 global peaks"""
    x, y = X[:, 0], X[:, 1]
    s1 = sum(i * np.cos((i+1)*x + i) for i in range(1, 6))
    s2 = sum(i * np.cos((i+1)*y + i) for i in range(1, 6))
    return -s1 * s2


def vincent(X):
    """2D, 36 peaks"""
    return np.prod(np.sin(10 * np.log(X)), axis=1)


def modified_rastrigin(X):
    """2D, 12 peaks"""
    k = np.array([3, 4])
    return -np.sum(10 + 9 * np.cos(2 * np.pi * k * X), axis=1)


# Known optima (approximate, from CEC2013 spec)
BENCHMARKS = [
    Benchmark('F1_FiveUneven', five_uneven_peak_trap, (0, 30),
              [np.array([0]), np.array([5]), np.array([15]),
               np.array([22.5]), np.array([30])], 1, 2),  # 2 global
    Benchmark('F2_EqualMaxima', equal_maxima, (0, 1),
              [np.array([0.1]), np.array([0.3]), np.array([0.5]),
               np.array([0.7]), np.array([0.9])], 1, 5),
    Benchmark('F3_UnevenDecreasing', uneven_decreasing_maxima, (0, 1),
              [np.array([0.08])], 1, 1),  # 1 global, multimodal
    Benchmark('F4_Himmelblau', himmelblau, (-6, 6),
              [np.array([3.0, 2.0]),
               np.array([-2.805118, 3.131312]),
               np.array([-3.779310, -3.283186]),
               np.array([3.584428, -1.848126])], 2, 4),
    Benchmark('F5_SixHumpCamel', six_hump_camel, (-1.9, 1.9),
              [np.array([0.0898, -0.7126]),
               np.array([-0.0898, 0.7126])], 2, 2),
    Benchmark('F8_ModifiedRastrigin', modified_rastrigin, (0, 1),
              [np.array([i/3, j/4]) for i in range(1,4) for j in range(1,5)],
              2, 12),
]


def peak_ratio(positions, optima, accuracy=0.01, bounds=None):
    """How many known peaks are covered by at least one agent within `accuracy`"""
    if bounds:
        scale = bounds[1] - bounds[0]
    else:
        scale = 1.0
    thresh = accuracy * scale
    covered = 0
    for opt in optima:
        dists = np.linalg.norm(positions - opt, axis=1)
        if dists.min() < thresh:
            covered += 1
    return covered / len(optima)


def success_rate(positions, optima, accuracy=0.01, bounds=None):
    """1 if ALL peaks covered, else 0"""
    return 1.0 if peak_ratio(positions, optima, accuracy, bounds) == 1.0 else 0.0


if __name__ == '__main__':
    for b in BENCHMARKS:
        X = np.random.uniform(b.bounds[0], b.bounds[1], (50, b.dim))
        f = b.fn(X)
        print(f"{b.name}: dim={b.dim}, peaks={b.n_peaks}, "
              f"f-range=[{f.min():.3f}, {f.max():.3f}]")
