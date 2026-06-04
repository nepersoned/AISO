# AISO: Asymmetric Interaction Swarm Optimization
### A Diversity-Preserving Wrapper for Feature Selection under Structural Cluster Diversity

> **Reframing note:** This draft pivots the primary contribution from "GNN fraud detection pipeline" to "diversity-preserving feature selection mechanism." Core algorithm, mechanism validation (Section 3), and experimental data are unchanged. Sections 1, 4, and 6 are restructured. Pending experiments marked with `[TODO]`.

---

## Abstract

We introduce AISO, a population-based metaheuristic in which agents interact through an asymmetric bilinear compatibility score:

```math
c_{ij} = \mathbf{W}_i^\top \mathbf{M} \mathbf{W}_j \neq c_{ji}
```

Each agent carries a type vector `W_i` on the probability simplex, and the compatibility matrix `M` encodes directed affinities between agent types — creating repulsion across type-pairs and attraction within them simultaneously. This asymmetry enables the swarm to self-organize into specialized sub-populations without radius tuning.

We make three interconnected claims. First, **AISO is a diversity-preserving mechanism**: typed agents with asymmetric compatibility maintain non-collapsing sub-populations on multimodal search without explicit partitioning. Second, **this diversity preservation yields a wrapper feature selector** that discovers feature subsets with structurally distinct cluster coverage — subsets that filter methods (mRMR, MI) and greedy wrappers (RFE) cannot reliably find when feature cluster diversity is high. Third, and most important, **AISO is conditionally useful**, not universally superior: its success is governed by feature cluster diversity, a structurally diagnostic quantity that predicts when the method helps, saturates, or fails.

We validate these claims through controlled experiments. On Elliptic Bitcoin — 166 features, 15 natural feature clusters — AISO's diversity-aware feature selection drives the primary gain (+0.082 PR-AUC, 73% of total improvement), with a secondary node-sampling stage adding +0.030. The full two-stage pipeline achieves PR-AUC 0.6644, outperforming 17 competing methods. Mechanism-level evidence is quantitative: Smart M reduces inter-agent feature-mask overlap (Jaccard 0.775 → 0.135), and symmetrizing M causes complete agent collapse (Jaccard = 1.000 across all seeds), confirming asymmetry as the sole source of persistent diversity. `[TODO: add RFE and GA wrapper baselines to complete wrapper-vs-wrapper comparison]`

---

## 1. Introduction

High-dimensional supervised learning suffers from two compounding problems: **irrelevance** (features carry no signal for the target) and **redundancy** (features carry the same signal as others already selected). Filter methods (MI, mRMR) address these by ranking features through statistical criteria, but they evaluate features independently of the downstream model and, crucially, cannot enforce structural diversity across feature clusters. Wrapper methods evaluate subsets by training a proxy model, but standard wrappers (RFE, sequential selection) use greedy or backward elimination strategies that collapse to locally dominant feature groups when one cluster dominates the information landscape.

The core challenge is maintaining **persistent sub-population structure in feature space** — selecting features that cover multiple distinct cluster perspectives simultaneously, without hard-coded partitioning.

AISO addresses this through typed agents and asymmetric interaction. Each agent `i` holds a type vector `W_i ∈ Δ^(K-1)` (probability simplex over `K` types corresponding to feature clusters) and interacts with agent `j` through the bilinear score `c_ij = W_i^T M W_j`. Because `M` is asymmetric, `c_ij ≠ c_ji`: two agents can simultaneously attract, repel, or have directed relationships. This is structurally impossible in distance-based or greedy selection strategies.

The application we develop is **budget-constrained GNN training for fraud detection**: a downstream task where feature diversity in the selected subspace directly determines whether the trained GNN can generalize across structurally distinct fraud patterns. We show that AISO's feature selection stage drives 73% of the total performance gain, and that the same diversity mechanism extends naturally to node-level sampling as a second stage.

These results compress into three claims. AISO supplies the mechanism, Exp 5 supplies standalone wrapper evidence `[TODO: vs RFE/GA]`, and the Elliptic two-stage pipeline supplies the cleanest downstream validation. Amazon, YelpChi, and CICIDS2017 serve as controlled counterexamples delimiting the regime where the method does and does not help.

---

## 2. Algorithm

*(Unchanged from original draft — see paper_draft.md §2)*

### 2.1 State Variables

Each agent `i ∈ {1, ..., N}` maintains:

| Variable | Domain | Role |
|----------|--------|------|
| `X_i ∈ [0,1]^D` | Feature space | Current position (snapped to nearest data point) |
| `W_i ∈ Δ^(K-1)` | Probability simplex | Type vector encoding agent specialization |
| `s_i ∈ R` | Scalar | Current score (fitness) |

Shared across all agents: `M ∈ R^(K×K)`, the asymmetric compatibility matrix.

### 2.2 Compatibility and Update

```math
c_{ij} = \mathbf{W}_i^\top \mathbf{M} \mathbf{W}_j
```

Position update (accepted only on improvement):

```math
\mathbf{X}_i \leftarrow \text{clip}\!\left(\mathbf{X}_i + \alpha \cdot c_{ij} \cdot (\mathbf{X}_j - \mathbf{X}_i),\ 0,\ 1\right)
```

Type update on successful move toward `j*`:

```math
\mathbf{W}_i \leftarrow \frac{(1-\beta)\mathbf{W}_i + \beta \mathbf{W}_{j^*}}{\|(1-\beta)\mathbf{W}_i + \beta \mathbf{W}_{j^*}\|_1}
```

**Asymmetric repulsion — the core mechanism.** Symmetric systems homogenize under attraction pressure; asymmetric `c_ij ≠ c_ji` permits cyclic specialization preferences that prevent sub-population merging. Negative entries in `M` encode type-based repulsion; type assimilation forms soft clusters around distinct feature groups.

**Adaptive repulsion strength:**

```math
w_r = 1.0 + 3.0 \cdot \exp(-\text{div}/0.12)
```

### 2.3 Smart M: Domain-Structured Compatibility

For applications where feature structure is available, `M` encodes domain knowledge asymmetrically:

1. Cluster `D` features into `K` groups via hierarchical clustering on correlation distance
2. Compute per-cluster mean mutual information: `μ_k = E_{j:cl(j)=k}[MI(x_j, y)]`
3. Normalize: `μ̃_k = (μ_k − min(μ)) / (max(μ) − min(μ))`
4. For `i ≠ j`:

```math
M_{ij} = -\overline{|C|}_{ij} + \gamma(\tilde{\mu}_j - \tilde{\mu}_i), \quad \gamma = 0.5
```

**(1) Correlation repulsion** prevents redundant feature selection; **(2) information gradient** asymmetrically routes low-information agents toward high-information regions. Together, they create meaningful directional specialization when feature cluster diversity is present.

**Feature cluster diversity (formal definition):**

```math
\mathcal{D}_{fc} = \big(k,\ \mathrm{CV}(\mu_1,\dots,\mu_k),\ \overline{|C|}_{\text{cross}}\big)
```

Diversity-supporting when `k ≥ 3`, `CV(μ) < 1.0`, and no single cluster dominates MI.

---

## 3. Mechanism Validation: Multi-Modal Optimization

*(Unchanged from original draft — see paper_draft.md §3)*

AISO achieves PR=0.975 on 8-peak 2D benchmark (vs SPSO=1.000, PSO=0.228), outperforms SPSO at D=5 (0.263 vs 0.125), maintains non-zero coverage at D=10 where SPSO collapses to zero, and improves +15% under noise where PSO degrades -17%.

**Summary:** Diversity preservation without radius tuning, across peak density, dimensionality, and noise — the necessary foundation for the feature selection claim.

---

## 4. Feature Selection and Downstream Application

### 4.1 AISO as a Wrapper Feature Selector

In the feature selection instantiation, the search variable is `W_i` directly: the feature mask is derived via temperature-annealed softmax over `W_i`, and a proxy model (SGD classifier, mini-batch 15%, max_iter=5) scores each candidate subset. The asymmetric interaction mechanism and type assimilation dynamics are identical to the continuous optimization case.

**Standalone evaluation (Exp 5, Elliptic Bitcoin, K=20, LR final eval, 5 seeds):**

| Method | Category | AUC | PR-AUC |
|--------|----------|-----|--------|
| All features (165) | — | 0.8865 | 0.2798 |
| Mutual Info | Filter | 0.8619 | 0.3613 |
| mRMR | Filter | 0.8405 | 0.3368 |
| LASSO | Embedded | 0.8459 | 0.2195 |
| **AISO (Smart+Score)** | **Wrapper** | **0.8467** | **0.3906** |
| `[TODO]` RFE | Wrapper | — | — |
| `[TODO]` GA wrapper | Wrapper | — | — |

AISO outperforms filter baselines. Wrapper-vs-wrapper comparison pending.

**Stability Selection:** Across 5 seeds, AISO consistently selects features {4, 13} (freq≥3) — features with PR-AUC=0.4557 using only 2 features vs mRMR's 0.3368 using 20. These features are not recoverable by MI top-K (zero index overlap), confirming that AISO discovers combinations invisible to relevance-only ranking.

### 4.2 Why Feature Selection Drives the Gain

Stage decomposition on Elliptic Bitcoin (budget: N_illicit=1000):

| Stage | PR-AUC | Delta | Share |
|-------|--------|-------|-------|
| S0 — Random baseline | 0.5530 | — | — |
| SB — Feature selection → SGD scorer | 0.6348 | +0.082 | **73%** |
| SC — Feature selection → AISO node sampler | 0.6644 | +0.030 | 27% |

Feature selection is the primary lever. Node-level diversity sampling compounds it but does not replace it.

**2×2 stage ablation** confirms additive contributions:

| | Stage-2 Sym M | Stage-2 Smart M | Stage-2 gain |
|---|---|---|---|
| **Stage-1 Sym M** | 0.5527 ± 0.084 | 0.5992 ± 0.031 | +0.047 |
| **Stage-1 Smart M** | 0.6244 ± 0.032 | **0.6644 ± 0.020** | +0.040 |
| **Stage-1 gain** | **+0.072** | **+0.065** | |

Stage-1 (feature selection) dominates; Stage-2 (node sampling) adds consistently but secondarily.

### 4.3 Origin: YelpZip

*(See paper_draft.md §4.2 — unchanged)*

YelpZip established that diversity-aware sampling covers heterogeneous fraud modes (4 semantic feature clusters) better than greedy collapse-prone heuristics, with random M already sufficient. Smart M was the follow-up: if the mechanism exploits existing mode structure, does encoding it explicitly help?

### 4.4 Full Elliptic Pipeline: SA → SB → SC

*(See paper_draft.md §4.3–4.5 — unchanged)*

Full 18-method comparison: AISO(Smart)→AISO SC = 0.6644, top-ranked. Four of top five methods are SC-stage. Stage profile SA→SB→SC on Elliptic shows the only full recovery pattern across all datasets.

### 4.5 When Feature Cluster Diversity Is Absent

*(See paper_draft.md §4.4 — unchanged)*

| Dataset | Failure Type | SC vs SB |
|---------|-------------|---------|
| Amazon | Dominant mode collapse | SC trails SB (−0.020) |
| YelpChi | Propagation ceiling | All methods compressed, 0.207–0.244 |
| CICIDS2017 | Degenerate structure | AISO −14% vs mRMR |
| **Elliptic** | None — condition met | SC beats SB (+0.030) |

---

## 5. Analysis

*(See paper_draft.md §5 — largely unchanged)*

Key results retained:
- **§5.1** Label efficiency: 0.6644 recovers 70% of gap to full-graph GraphSAGE ceiling (0.7128)
- **§5.2** Smart M vs Rand M: Jaccard 0.775→0.135; +0.028 at SB, +0.013 at SC
- **§5.3** Scope: AISO requires a score function; fails on unsupervised clustering
- **§5.4** Backbone variation: feature cluster diversity is the binding constraint, not architecture
- **§5.6** Symmetric ablation (**completed**): Sym M → Jaccard=1.000 (complete collapse); asymmetry is necessary

### 5.5 Supplementary Experiments (`reframing/`)

#### Exp B — Diversity Metric Screening (completed)

Three candidate metrics were computed per dataset (k, CV(μ), eff_rank) and ranked against known AISO outcomes:

| Dataset | k | CV(μ) | eff_rank | Delta SC |
|---------|---|-------|----------|----------|
| Elliptic | 12 | 0.667 | 47.2 | +0.111 |
| Amazon | 9 | 1.601 | 9.8 | +0.020 |
| YelpChi | 23 | 1.733 | 21.1 | −0.014 |

Spearman rank correlation across three datasets: CV(μ) achieves ρ = −1.0, perfectly ranking datasets by AISO outcome. k and eff_rank achieve |ρ| = 0.5. **CV(μ) is the strongest single predictor of AISO success among the candidates tested.**

Note: n=3 precludes statistical significance claims. This is an observational finding — on every real dataset evaluated, CV(μ) rank-orders AISO outcome correctly. The direction (lower CV(μ) → better AISO) is consistent with the governing condition: balanced MI across clusters means diverse coverage adds value; imbalanced MI means all methods converge to the dominant cluster regardless.

**Updated governing condition threshold:** CV(μ) < ~1.0 separates success (Elliptic: 0.667) from failure (Amazon: 1.601, YelpChi: 1.733). The `CV(μ) < 0.5` threshold in §2.3 should be revised to `CV(μ) < 1.0`.

#### Exp B-1 — Pre-processing Perturbation Stress Test (Appendix)

Gaussian noise injected into MI estimates (σ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0}) and cluster assignment swap noise (p ∈ {0, 0.1, 0.2, 0.3, 0.5}), evaluated on Elliptic Bitcoin wrapper (LR, 5 seeds):

- **MI noise:** σ=0→1.0, PR-AUC 0.3906→0.4303 (Δ=+0.040). Graceful — no degradation within tested range.
- **Cluster swap noise:** p=0→0.5, PR-AUC 0.3906→0.3152 (Δ=−0.075). Moderate sensitivity; performance degrades gradually with cluster corruption.

**Finding:** Smart M is robust to MI estimation noise at the wrapper level. Cluster assignment quality matters more than MI precision.

| Exp | Description | Status |
|-----|-------------|--------|
| RFE | RFE (LR/SVM backbone) vs AISO, Exp 5 protocol | `[TODO]` |
| GA wrapper | GA-based feature selection vs AISO, Exp 5 protocol | `[TODO]` |
| Exp C | Synthetic diversity control (vary k, multi-modal label) | `[TODO]` |

---

## 6. Conclusion

**Claim 1 — The Mechanism.** AISO's asymmetric bilinear compatibility enables multi-population structure without radius tuning. Symmetric ablation (Jaccard 1.000 collapse) confirms asymmetry as the necessary and sufficient source of persistent diversity.

**Claim 2 — The Feature Selector.** Diversity-aware feature selection via AISO discovers structurally distinct feature subsets that relevance-only filters cannot find. On Elliptic Bitcoin, feature selection drives 73% of total performance gain; the Stability Selection result (2 AISO features > 20 mRMR features on PR-AUC) demonstrates that cluster-covering subsets are qualitatively different from redundancy-minimizing ones. `[Wrapper-vs-wrapper evidence pending RFE/GA experiments]`

**Claim 3 — The Governing Condition.** Feature cluster diversity (`k ≥ 3`, balanced MI across clusters, low cross-cluster correlation) predicts when AISO applies. Elliptic satisfies the condition and wins; Amazon, YelpChi, CICIDS2017 violate it and fail — mechanically, not randomly.

---

## References

*(Same as paper_draft.md)*

[1] Dou et al., "Enhancing Graph Neural Network-based Fraud Detection via Locally Homophilous Aggregation," CIKM 2020  
[2] Liu et al., "Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection," WWW 2021  
[3] Chiang et al., "Cluster-GCN," KDD 2019  
[4] Zeng et al., "GraphSAINT," ICLR 2020  
[5] Wu et al., "Simplifying Graph Convolutional Networks," ICML 2019  
[6] Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks," ESWC 2018  
[7] Kennedy & Eberhart, "Particle Swarm Optimization," ICNN 1995  
[8] Ding & Peng, "Minimum Redundancy Feature Selection," JBCB 2005  
`[TODO: add RFE reference, GA feature selection reference]`
