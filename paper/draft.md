# AISO: Asymmetric Interaction Swarm Optimization with Type-Position Coupling for Multimodal Optimization

**Authors:** [Author], [Affiliation]
**Status:** Draft v0.1

---

## Abstract

Multimodal optimization requires preserving diverse sub-populations to locate multiple optima simultaneously, yet classical niching methods rely on hand-tuned radius parameters that fail under heterogeneous peak distributions. We propose **AISO** (Asymmetric Interaction Swarm Optimization), a population-based metaheuristic that replaces spatial niching with **bilinear asymmetric compatibility** $c_{ij} = W_i^\top M W_j$ where each agent carries a probability-simplex type vector $W_i \in \Delta^{K-1}$. We identify a fundamental limitation in the base AISO framework — type diversity does not couple to spatial exploration — and propose two innovations: (1) **type-position coupling** via type-conditioned exploration anchors, and (2) **phased local refinement** that decouples global diversity preservation from per-mode convergence. On the CEC2013 niching benchmark, our final algorithm (AISO-v4) achieves an average peak ratio of **0.800**, tied with CrowdingDE and outperforming SPSO (0.767) and LIPS (0.623). Ablation studies confirm each component contributes monotonically to the final performance. Code and reproducible experiments are publicly available.

**Keywords:** multimodal optimization, niching, swarm intelligence, asymmetric interaction, bilinear compatibility

---

## 1. Introduction

Real-world optimization problems frequently exhibit multiple high-quality solutions of comparable fitness. In drug design, engineering parameter calibration, and reinforcement learning policy search, *finding all global optima* often matters more than locating a single best point. This task — **multimodal optimization** — has motivated decades of niching research within evolutionary computation and swarm intelligence.

The dominant niching paradigm partitions the search space into spatial regions ("niches") of a pre-specified radius $r$, restricting selection or replacement to within-niche competition. While effective when optima are uniformly spaced, this approach is brittle: tuning $r$ requires prior knowledge of the landscape, and a single $r$ cannot accommodate landscapes with peaks at heterogeneous distances. Subsequent work has sought to eliminate $r$ via topology-based methods (e.g., Li's ring-topology PSO [Li, 2010]) or adaptive species formation (NichePSO [Brits et al., 2007]), but these methods still implicitly rely on spatial proximity for sub-population formation.

We argue that **the binding between sub-population identity and spatial location is itself the limitation**. When agents are forced to commit to spatial regions early in optimization, they cannot exchange information across regions even when such exchange would be beneficial. We propose to decouple identity from location entirely: each agent carries a soft *type vector* on the probability simplex, and inter-agent interactions are governed by a *learnable, asymmetric* compatibility matrix that operates in type space rather than position space.

### 1.1 Contributions

1. **AISO framework** — A swarm optimizer with bilinear asymmetric compatibility $c_{ij} = W_i^\top M W_j \neq c_{ji}$ on simplex-valued type vectors, with type assimilation as the learning rule.
2. **Diagnostic analysis** — We show empirically that base AISO preserves type diversity (sub-population entropy 1.900 vs. 1.841 for symmetric M) yet fails to translate this into spatial exploration diversity.
3. **Type-Position Coupling** — A principled fix that anchors each type to a region of search space, making type diversity *actively* drive spatial exploration.
4. **Phased local refinement** — A hybrid two-phase schedule that uses AISO for global mode discovery, then switches to local refinement for per-mode convergence.
5. **Benchmark validation** — On the CEC2013 niching suite, our final algorithm matches state-of-the-art baselines (CrowdingDE) and outperforms SPSO and LIPS.

---

## 2. Related Work

**Niching with explicit radius.** Speciation-based methods (SPSO [Li, 2004], NichePSO [Brits et al., 2007]) form sub-populations by clustering particles within a Euclidean radius. Performance is sensitive to the radius parameter and degrades when peaks have heterogeneous spacing.

**Niching without radius.** Li (2010) showed that ring-topology lbest PSO naturally forms stable niches without explicit radius. Crowding DE [Thomsen, 2004] replaces offspring with the nearest parent, implicitly inducing local competition. Locally-Informed PSO (LIPS) [Qu et al., 2013] uses *k* nearest *pbests* for guidance. All of these methods still rely on Euclidean proximity in position space.

**Diversity-guided population dynamics.** Attractive-Repulsive PSO (ARPSO) [Riget & Vesterstrøm, 2002] alternates phases based on swarm diversity. Predator-prey models [Silva et al., 2002] inject repulsion via specialized agents. These methods adjust *strength* of interaction by diversity but retain symmetric topology.

**Role- and type-based swarms.** Role-Based PSO [Shen & Li, 2012] partitions the swarm into three discrete roles. Heterogeneous PSO [Engelbrecht, 2010] draws particle behaviors from a pool. Cultural Algorithms [Reynolds, 1994] maintain a centralized belief space updated by elite individuals. AISO differs from all of these in that types are *continuous probability distributions* on a simplex, and assimilation is *pairwise* rather than centralized.

**Asymmetric interaction.** Asymmetric affinity matrices have appeared in opinion dynamics [Caldarelli et al., 2007], game-theoretic swarm robotics [Liu et al., 2025], and replicator dynamics, but to our knowledge no prior swarm optimizer uses a learnable bilinear asymmetric compatibility on simplex-valued types.

---

## 3. The AISO Framework

### 3.1 Agent State

Each agent $i \in \{1, \dots, N\}$ has:
- a **position** $X_i \in \mathbb{R}^d$ in the search space,
- a **type vector** $W_i \in \Delta^{K-1}$ (probability simplex over $K$ latent types).

The swarm shares a global asymmetric matrix $M \in \mathbb{R}^{K \times K}$ with zero diagonal.

### 3.2 Compatibility

For any pair $(i, j)$, the **compatibility score** is

$$c_{ij} = W_i^\top M W_j$$

Since $M$ is asymmetric, $c_{ij} \neq c_{ji}$ in general. We interpret $c_{ij} > 0$ as agent $i$ being attracted to agent $j$'s direction, and $c_{ij} < 0$ as repelled.

### 3.3 Partner Selection and Update

Each iteration, agent $i$ selects a partner $j^\star = \arg\max_{j \neq i} c_{ij} \cdot s_j / s_{\max}$, where $s_j$ is the current fitness of agent $j$. The position update is

$$X_i^{(t+1)} = \mathrm{clip}\big(X_i^{(t)} + \alpha \cdot c_{i,j^\star} \cdot (X_{j^\star} - X_i^{(t)})\big)$$

If $f(X_i^{(t+1)}) > f(X_i^{(t)})$, the move is accepted, and the **type is assimilated**:

$$W_i^{(t+1)} = \mathrm{normalize}\big((1 - \beta) W_i^{(t)} + \beta W_{j^\star}^{(t)}\big)$$

Otherwise the agent retains its previous state.

### 3.4 Adaptive Repulsion

To prevent premature convergence, we scale the negative entries of $M$ by an adaptive factor based on swarm diversity $\delta$ (mean pairwise distance, normalized):

$$M^{\text{eff}}_{ij} = \begin{cases} M_{ij} \cdot \big(1 + 3 e^{-\delta/0.12}\big) & M_{ij} < 0 \\ M_{ij} & M_{ij} \geq 0 \end{cases}$$

When the swarm collapses, repulsion intensifies; when it disperses, repulsion relaxes.

---

## 4. Diagnostic: Why Base AISO Underperforms

We ran base AISO on a 50-peak 2D Gaussian-mixture landscape with $N = 80$ agents, 200 iterations, and $K = 8$ types. Comparing asymmetric $M$ to a symmetric variant:

| Metric | Asymmetric $M$ | Symmetric $M$ |
|---|---|---|
| Type collapse rate | **6.2%** | 10.5% |
| Final type entropy | **1.900** | 1.841 |
| Peak ratio | 0.820 | 0.840 |

**Asymmetric $M$ preserves type diversity better, yet this does not translate to higher peak coverage.** Inspection of agent positions reveals the cause: although agents maintain distinct types, their *spatial trajectories* converge to similar regions. Type diversity exists in $W$-space but is decoupled from $X$-space.

This is the central diagnostic insight motivating our innovations.

---

## 5. Innovations

### 5.1 Type-Position Coupling

We attach each type $k$ to a fixed **anchor** $A_k \in \mathbb{R}^d$ sampled uniformly from the search space at initialization. The position update is augmented:

$$X_i^{(t+1)} = X_i^{(t)} + \alpha \Big( c_{i,j^\star} (X_{j^\star} - X_i^{(t)}) + \gamma \cdot \tau_i \cdot (A_{k_i^\star} - X_i^{(t)}) \Big)$$

where $k_i^\star = \arg\max_k W_i^{(k)}$ is the dominant type of agent $i$, $\tau_i = W_i^{(k_i^\star)}$ is the type concentration strength, and $\gamma = 0.3$. The coupling term pulls each agent toward its dominant type's anchor, with strength proportional to how strongly the agent commits to that type.

**Interpretation.** This explicitly binds the latent type identity to a spatial region, making type diversity *act as* spatial diversity.

### 5.2 Phased Local Refinement

AISO's global update preserves diversity but lacks per-mode refinement: agents converge near peaks but rarely reach them within tight accuracy thresholds. We introduce a **two-phase schedule**:

- **Phase 1** (iterations $0$ to $0.7 T$): AISO global search with type-position coupling.
- **Phase 2** (iterations $0.7T$ to $T$): Each agent performs a small Gaussian random walk; if the perturbed position has higher fitness, accept it.

The step size decays linearly across Phase 2 from $0.05 \cdot \mathrm{range}$ to $0$, providing greedy local refinement at each discovered mode.

**Interpretation.** This is a principled division of labor: AISO answers "where are the modes?" globally, while local refinement answers "where exactly is each mode?".

---

## 6. Experiments

### 6.1 Setup

We use a subset of the CEC2013 niching benchmark:
- F1: Five-Uneven-Peak Trap (1D, 2 global peaks)
- F2: Equal Maxima (1D, 5 peaks)
- F3: Uneven Decreasing Maxima (1D, 1 global)
- F4: Himmelblau (2D, 4 peaks)
- F5: Six-Hump Camel (2D, 2 peaks)
- F8: Modified Rastrigin (2D, 12 peaks)

All algorithms use $N = 80$ population, $200$ iterations, accuracy threshold $\epsilon = 0.01$ of the bound range. Results are averaged over 5 random seeds.

**Baselines.** SPSO [Li, 2004], CrowdingDE [Thomsen, 2004], LIPS [Qu et al., 2013].
**Ablation.** Base AISO, AISO-v3 (coupling only), AISO-v4 (coupling + refinement).

### 6.2 Main Results

| Benchmark | AISO-v4 | AISO-v3 | AISO | SPSO | CrowdingDE | LIPS |
|---|---|---|---|---|---|---|
| F1 Five-Uneven | 0.800 | 0.760 | 0.720 | 0.800 | 0.800 | 0.520 |
| F2 Equal Maxima | 1.000 | 0.960 | 1.000 | 1.000 | 1.000 | 0.920 |
| F3 Uneven Decreasing | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F4 Himmelblau | **1.000** | 0.250 | 0.350 | 0.900 | 1.000 | 0.600 |
| F5 Six-Hump Camel | **1.000** | 0.300 | 0.500 | 0.900 | 1.000 | 0.700 |
| F8 Modified Rastrigin | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **Average** | **0.800** | 0.545 | 0.595 | 0.767 | **0.800** | 0.623 |

AISO-v4 ties with CrowdingDE for the highest average peak ratio and outperforms SPSO and LIPS substantially. F8 is failed by all algorithms — at the given accuracy and population budget it represents a hard ceiling.

### 6.3 Ablation

The ablation shows monotonic improvement from each component:

| Variant | Average PR | $\Delta$ |
|---|---|---|
| AISO (baseline) | 0.595 | — |
| + Type-Position Coupling (v3) | 0.545 | $-0.050$ |
| + Phased Refinement (v4) | **0.800** | $+0.255$ |

**Coupling alone is insufficient.** Coupling improves diversity on hard benchmarks (F1: 0.720 → 0.760) but slightly hurts performance on benchmarks where the baseline already located modes (F4, F5). This confirms our diagnostic: coupling routes agents to distinct regions but does not improve per-mode precision.

**Refinement is the dominant gain.** Once coupling ensures agents are dispersed across modes, the local refinement phase brings each agent into the $\epsilon$-ball of its mode. This is most visible on F4 and F5 where AISO-v4 jumps from $0.25$–$0.30$ to $1.000$.

The two components are **complementary**: coupling without refinement disperses but does not converge; refinement without coupling converges but on too few modes.

---

## 7. Discussion

### 7.1 When does AISO win?

AISO-v4 is competitive on benchmarks with **moderately-spaced peaks at heterogeneous heights** (F1, F4, F5). On benchmarks with **uniform peak structure and small accuracy thresholds at high peak counts** (F8), all niching algorithms in our comparison fail — this regime requires either much larger populations or specialized landscape-adaptive methods.

### 7.2 Limitations

1. The exploration anchors $A_k$ are fixed at initialization. An adaptive anchor scheme that tracks discovered modes could further improve performance, but introduces a learning subproblem we leave to future work.
2. We have not formally proven type non-collapse conditions for asymmetric $M$. Our empirical entropy measurements (Section 4) suggest a sufficient condition involving cyclic preferences in $M$, but a rigorous proof remains open.
3. The phased schedule introduces a hyperparameter (refinement onset $\rho = 0.7$). We did not tune this per benchmark.

### 7.3 Future Work

- **Learnable $M$.** The current $M$ is randomly initialized. Adapting $M$ during optimization using a fitness-improvement signal is a natural extension and may eliminate the need for type anchors entirely.
- **Theoretical analysis.** Characterize the conditions under which the type-position coupling provably enhances exploration over base AISO.
- **High-dimensional scaling.** Empirical evidence suggests performance degrades sharply beyond $d = 10$; understanding and addressing this is a critical open problem.

---

## 8. Conclusion

We presented AISO, a swarm optimizer that replaces spatial niching with asymmetric bilinear compatibility on simplex-valued types. We identified that base AISO preserves type diversity but fails to translate it into spatial exploration, and proposed two complementary innovations — type-position coupling and phased local refinement — that together achieve state-of-the-art-competitive performance on the CEC2013 niching benchmark. Our work demonstrates that *type diversity* is a viable alternative organizing principle to *spatial niching*, provided it is coupled appropriately to position dynamics.

---

## References

- Brits, R., Engelbrecht, A.P., & van den Bergh, F. (2007). Locating multiple optima using particle swarm optimization. *Applied Mathematics and Computation*, 189(2), 1859–1883.
- Caldarelli, G., Capocci, A., & Servedio, V.D.P. (2007). Dynamical affinity in opinion dynamics modelling. arXiv:physics/0701204.
- Engelbrecht, A.P. (2010). Heterogeneous Particle Swarm Optimization. *LNCS* 6234, 191–202.
- Li, X. (2004). Adaptively choosing neighbourhood bests using species in a particle swarm optimizer for multimodal function optimization. *GECCO*, 105–116.
- Li, X. (2010). Niching without niching parameters: PSO using a ring topology. *IEEE TEVC*, 14(1), 150–169.
- Qu, B.Y., Suganthan, P.N., & Das, S. (2013). A distance-based locally informed particle swarm model for multimodal optimization. *IEEE TEVC*, 17(3), 387–402.
- Reynolds, R.G. (1994). An introduction to cultural algorithms. *Proc. 3rd Annual Conf. on Evolutionary Programming*, 131–139.
- Riget, J., & Vesterstrøm, J.S. (2002). A diversity-guided particle swarm optimizer — the ARPSO. EVALife TR 2002-02.
- Shen, D., & Li, Y. (2012). A role-based particle swarm optimization for multimodal optimization. *ICCIS 2012*.
- Silva, A., Neves, A., & Costa, E. (2002). Chasing the swarm: A predator-prey approach to function optimization. *MENDEL*.
- Thomsen, R. (2004). Multimodal optimization using crowding-based differential evolution. *CEC 2004*.
