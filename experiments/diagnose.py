"""
진단: AISO가 F4(Himmelblau, 4 peaks) 같은 간단한 멀티모달에서 왜 못 찾는지
- 에이전트가 한곳으로 수렴하는지
- accuracy=0.01이 너무 빡빡한지
"""
import numpy as np
import sys; sys.path.insert(0, '/home/claude/aiso_paper')
from aiso_v3 import AISO_v3
from benchmarks import himmelblau, peak_ratio

bounds = (-6, 6)
optima = [np.array([3.0, 2.0]), np.array([-2.805, 3.131]),
          np.array([-3.779, -3.283]), np.array([3.584, -1.848])]

for acc in [0.01, 0.02, 0.05, 0.1]:
    aiso = AISO_v3(80, 2, 4, bounds, coupling=True, seed=42)
    for _ in range(200): aiso.update(himmelblau)
    pr = peak_ratio(aiso.X, optima, acc, bounds)
    # 분산 측정
    var = np.var(aiso.X, axis=0).sum()
    print(f"acc={acc}: PR={pr:.3f}  positions variance={var:.3f}")

# 각 peak 근처 에이전트 수
print("\nNearest agent distance to each peak:")
aiso = AISO_v3(80, 2, 4, bounds, coupling=False, seed=42)
for _ in range(200): aiso.update(himmelblau)
for opt in optima:
    d = np.linalg.norm(aiso.X - opt, axis=1)
    print(f"  peak {opt}: min dist={d.min():.4f}, agents within 0.5: {(d<0.5).sum()}")
