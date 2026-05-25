# AISO: Asymmetric Interaction Swarm Optimization

Multimodal optimization with bilinear asymmetric compatibility on simplex-valued type vectors.

This repository accompanies the paper draft in [`paper/draft.md`](paper/draft.md).

---

## Key Idea

Each agent carries a **type vector** $W_i \in \Delta^{K-1}$ on the probability simplex.
Inter-agent interaction is governed by an **asymmetric bilinear compatibility**:

$$c_{ij} = W_i^\top M W_j \neq c_{ji}$$

Agents move toward high-compatibility partners and assimilate their types upon successful moves.

Compared to spatial niching (SPSO, NichePSO), AISO replaces the radius parameter with **type diversity** as the organizing principle.

---

## Repository Structure

```
aiso-paper/
├── README.md
├── requirements.txt
├── src/                       # Core algorithm implementations
│   ├── aiso_v3.py             # AISO with Type-Position Coupling
│   ├── aiso_v4.py             # AISO + Coupling + Phased Refinement (final)
│   ├── benchmarks.py          # CEC2013 niching benchmark subset
│   └── baselines.py           # SPSO, CrowdingDE, LIPS
├── experiments/
│   ├── run_main.py            # Main benchmark comparison (5 seeds)
│   ├── visualize.py           # Generate paper figures
│   └── diagnose.py            # Diagnostic for type-position decoupling
├── results/
│   ├── fig1_comparison.png    # Per-benchmark comparison bar chart
│   ├── fig2_ablation.png      # AISO ablation study
│   ├── fig3_ranking.png       # Overall ranking
│   └── results_final.pkl      # Raw experimental results
└── paper/
    └── draft.md               # Full paper draft
```

---

## Quick Start

### Requirements

- Python 3.8+
- numpy
- matplotlib

```bash
pip install -r requirements.txt
```

### Reproduce Main Results

```bash
# Run full benchmark comparison (5 seeds, ~5 min on CPU)
python experiments/run_main.py

# Generate figures
python experiments/visualize.py
```

### Run AISO on a Custom Problem

```python
from src.aiso_v4 import AISO_v4
import numpy as np

# Define your fitness function (vectorized over rows)
def my_fitness(X):
    return -np.sum(X**2, axis=1)  # maximize, so negate cost

# Initialize AISO
aiso = AISO_v4(
    n_agents=80,
    dim=2,
    K=4,                # number of latent types
    bounds=(-5, 5),
    coupling=True,      # Type-Position Coupling (Innovation 1)
    refinement=True,    # Phased Local Refinement (Innovation 2)
    seed=42,
)

# Run
final_positions = aiso.run(my_fitness, n_iter=200)
```

---

## Main Results

CEC2013 niching benchmark, accuracy threshold $\epsilon = 0.01$, 5 seeds:

| Algorithm | Avg Peak Ratio |
|---|---|
| **AISO-v4 (ours)** | **0.800** |
| CrowdingDE | 0.800 |
| SPSO | 0.767 |
| LIPS | 0.623 |
| AISO baseline | 0.595 |

**Ablation:**

| Variant | Avg PR | $\Delta$ |
|---|---|---|
| AISO baseline | 0.595 | — |
| + Type-Position Coupling | 0.545 | -0.050 |
| + Phased Refinement (full) | **0.800** | +0.255 |

The two innovations are **complementary**: coupling alone disperses agents to distinct modes; refinement alone converges to too few modes. Both together achieve state-of-the-art.

---

## Algorithm Variants

| Variant | Description |
|---|---|
| `AISO_v3` (coupling=False) | Base AISO from original framework |
| `AISO_v3` (coupling=True) | + Type-Position Coupling |
| `AISO_v4` | + Phased Local Refinement (final algorithm) |

---

## Citation

If you use this work, please cite:

```bibtex
@misc{aiso2025,
  title={AISO: Asymmetric Interaction Swarm Optimization with Type-Position Coupling for Multimodal Optimization},
  author={[Author]},
  year={2025},
  note={Draft},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work builds on foundational niching research by Li (2010), Brits et al. (2007), and the broader swarm intelligence community.
