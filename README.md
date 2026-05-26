# AISO: Asymmetric Interaction Swarm Optimization

Population-based metaheuristic with asymmetric bilinear compatibility on simplex-valued type vectors, applied to diversity-aware sub-graph selection in imbalanced fraud detection.

Paper draft: [`paper/unified_draft.md`](paper/unified_draft.md)

---

## Key Idea

Each agent carries a **type vector** $W_i \in \Delta^{K-1}$ on the probability simplex.
Inter-agent interaction is governed by an **asymmetric bilinear compatibility**:

$$c_{ij} = W_i^\top M W_j \neq c_{ji}$$

Because $c_{ij} \neq c_{ji}$, agents can have simultaneously divergent interests — type $k$ attracted to type $l$ while type $l$ is simultaneously repelled — producing persistent specialization without hard-coded spatial partitioning.

---

## Repository Structure

```
aiso-paper/
├── README.md
├── requirements.txt
├── src/                        # Core algorithm
│   ├── aiso_v7.py              # Final AISO implementation
│   ├── benchmarks.py           # CEC2013 niching benchmarks (F1–F8)
│   └── baselines.py            # SPSO, CrowdingDE, PSO+LS
├── experiments/
│   ├── run_main.py             # CEC2013 main benchmark
│   ├── run_v7_ablation.py      # 15+ mechanism ablation
│   ├── run_w4_pso_baseline.py  # PSO+LS memetic baseline
│   ├── run_w14_pso_budget.py   # PSO+LS budget robustness
│   ├── run_governing_condition_synthetic.py  # n=135 synthetic CV sweep
│   └── run_w1_w5.py            # Sensitivity (γ, ρ)
├── gnn/                        # GNN fraud detection experiments
│   └── scripts/                # expA pipeline (Elliptic, Amazon, YelpChi)
├── results/                    # Saved experimental results
└── paper/
    ├── unified_draft.md        # Full paper draft (v2.5, 540 lines)
    ├── make_figures.py         # Figure generation script
    └── figures/                # Generated figures (PDF + PNG)
```

---

## Main Results

### CEC2013 Niching (F1–F8, 30 seeds, avg peak ratio)

| Method | Avg PR | vs AISO |
|---|---|---|
| **AISO + Phased Refinement** | **0.911** | — |
| Random + Refine | 0.906 | −0.005 (p=0.43, n.s.) |
| PSO + LS (T=200) | 0.460 | −0.451 (p<0.0001) |
| PSO + LS (T=400) | 0.427 | structural limit |
| PSO + LS (T=800) | 0.416 | structural limit |
| CrowdingDE | 0.952 | reference |
| SPSO | 0.793 | reference |

Asymmetric $M$ is the structural prerequisite: symmetric $M$ collapses all agents to identical solutions (Jaccard = 1.000 vs 0.136 under asymmetric).

### Elliptic Bitcoin Fraud Detection (PR-AUC, 5 seeds)

| Method | PR-AUC | Rank |
|---|---|---|
| **AISO (Smart→Smart)** | **0.6644 ± 0.020** | **1 / 18** |
| mRMR → SGD | 0.6348 ± 0.006 | 2 |
| Random baseline | ~0.553 | — |
| Full graph (ceiling) | 0.7128 | — |

Two-stage AISO recovers **93.2%** of full-graph performance under a 2% labeling budget.

### Governing Condition

$\mathrm{CV}(\mu) < 1.0$ perfectly rank-orders three real datasets by AISO outcome (Spearman $\rho = -1.0$). Synthetic validation across $n = 135$ conditions confirms AISO robustly outperforms random (mean $\Delta > 0$, $p < 0.0001$).

---

## Quick Start

```bash
pip install -r requirements.txt

# CEC2013 ablation
python experiments/run_v7_ablation.py

# Generate paper figures
python paper/make_figures.py

# GNN fraud detection (requires Elliptic dataset)
# See gnn/scripts/ for pipeline
```

---

## Citation

```bibtex
@article{aiso2025,
  title   = {{AISO}: Asymmetric Interaction Swarm Optimization for
             Diversity-Aware Sub-graph Selection in Imbalanced Fraud Detection},
  author  = {Bae, Jinhyung},
  journal = {Memetic Computing},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
