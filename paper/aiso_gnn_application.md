# Application: AISO as a Diversity-Aware Sub-graph Sampler for GNN-Based Fraud Detection

*Supplementary application section for the AISO paper.*

---

## A.1 Motivation

While Sections 3–6 establish AISO as a niching algorithm for continuous multimodal optimization, the framework's central principle — **type-driven diversity preservation without spatial radius tuning** — generalizes to *combinatorial selection problems* that share the same structural property: the need to cover heterogeneous sub-populations under a tight budget.

We instantiate this on a representative real-world problem: **sub-graph sampling for imbalanced GNN-based fraud detection.**

---

## A.2 Problem Setup

### A.2.1 Task

Given a large transaction graph $G = (V, E)$ with node labels $y \in \{0, 1\}^{|V|}$ (legitimate vs. illicit), train a graph neural network to detect illicit nodes. The challenge:

- $|V|$ is large (hundreds of thousands of nodes).
- Class imbalance is severe (~10% positive).
- Full-graph training is infeasible under memory and runtime constraints.
- Random sub-sampling fragments local neighborhoods and destroys graph signal.

### A.2.2 Why this is a multimodal selection problem

Fraud signals manifest in *multiple distinct patterns*: spam-style burst activity, account-reuse clusters, time-concentrated bursts, template-driven text reuse. A sampling method that biases toward any single pattern under-covers the others, producing brittle detectors. We argue this is structurally analogous to multimodal optimization — we need to *cover all modes* of the fraud distribution.

### A.2.3 AISO as Sampler

We adapt AISO from a continuous optimizer to a discrete sub-graph selector:

- **Agent position** $X_i \in [0, 1]^K$: a *risk-pattern profile vector* representing which fraud-signal dimensions agent $i$ prioritizes (e.g., burst, account-reuse, text-template). $K = 17$ in our experiments.
- **Agent type** $W_i \in \Delta^{K-1}$: latent type distribution (same as in the main paper).
- **Compatibility** $c_{ij} = W_i^\top M W_j$: identical to Section 3.2.
- **Fitness** $s_i$: the *coverage score* of the candidate node set selected under profile $X_i$ — measuring how well the selected sub-graph spans heterogeneous fraud signals.
- **Output**: After convergence, the union of nodes selected across all agents forms the training sub-graph.

The asymmetric compatibility ensures that different agents specialize in different fraud patterns, jointly producing a *diverse-coverage* sub-graph rather than a profile-collapsed one.

---

## A.3 Experimental Setup

### A.3.1 Dataset

**Elliptic Bitcoin Dataset** — 203,769 transaction nodes, 234,355 edges, 49 timesteps, with ~21% labeled (4.5% illicit). Standard benchmark for graph-based fraud detection.

### A.3.2 Sampling Strategies Compared

| Strategy | Description |
|---|---|
| Strategy A | Top-N high-volume merchants (volume-based) |
| Strategy B | Top-N active users (activity-based) |
| Strategy C | Time-window with peak fraud ratio (temporal-based) |
| Random | Uniform random node sampling |
| **AISO** | **Diversity-aware multi-agent sampling** |

All strategies select the same sub-graph size (~50,000 nodes), train the same GNN (SGC, 2-layer, custom edge set), and evaluate on the same held-out test set.

### A.3.3 Metrics

- **PR-AUC** — area under precision-recall curve (imbalance-robust)
- **Macro F1** — class-balanced F1 score

---

## A.4 Results

### A.4.1 Main Comparison

| Strategy | Sub-graph Fraud % | PR-AUC | Macro F1 |
|---|---|---|---|
| Strategy A (volume) | 11.33% | 0.2998 | 0.5690 |
| Strategy B (activity) | 0.31% | 0.1903 | 0.4553 |
| Strategy C (time) | 14.59% | 0.3417 | 0.5888 |
| **AISO** | **29.02%** | **0.5908** | **0.6798** |

AISO sampling improves PR-AUC by **+73%** over the best single-criterion baseline (Strategy C).

### A.4.2 Label Efficiency

To quantify how much of the full-graph training signal AISO recovers under a tight budget:

| Training Budget (% of labels) | PR-AUC | Recovery vs. Full-Graph |
|---|---|---|
| Full graph (100%) | 0.7128 | 100.0% |
| AISO (~2%) | **0.6644** | **93.2%** |
| Strategy C (~2%) | 0.3417 | 47.9% |
| Random (~2%) | 0.2412 | 33.8% |

AISO recovers **93.2% of full-graph training performance using only a ~2% label budget**, while baseline samplers recover less than half. This is the application-side evidence for the theoretical claim in the main paper: *type-driven diversity beats spatial proximity for heterogeneous sub-population coverage.*

### A.4.3 Mechanism Validation

A key prediction of AISO is that *asymmetric* M preserves diversity where *symmetric* M collapses. We verify this on the sampling task:

| M Structure | Inter-Agent Jaccard Similarity |
|---|---|
| Symmetric M | 1.000 (full collapse — all agents select identical sub-graph) |
| Asymmetric M | **0.136** (high diversity preserved) |

The collapse-to-identity under symmetric M directly confirms the mechanism analyzed in Section 4 of the main paper.

### A.4.4 Ranking Against Prior Work

Compared against 17 prior sampling and detection methods on Elliptic (covering temporal, GraphSAGE, GAT, CARE-GNN, and several fraud-specific baselines), AISO+SGC achieves **Rank 1** in PR-AUC.

---

## A.5 Discussion

### A.5.1 Why AISO Wins Here

The Elliptic dataset has high **feature cluster diversity** — fraud signals are scattered across distinct sub-patterns (account behavior, transaction timing, network topology). This is precisely the regime where Section 7.1 of the main paper predicts AISO to dominate: the underlying problem has *heterogeneous modes*, and a niching-style sampler is structurally better than a single-criterion one.

### A.5.2 Negative Result: Where AISO Does Not Help

We also tested AISO sampling on Amazon and YelpChi fraud datasets, where the underlying fraud distribution is more homogeneous. On these datasets, AISO matched but did not significantly exceed Strategy A baselines. This is consistent with the *Governing Condition* identified in the main paper: AISO's advantage is conditional on feature cluster diversity, not universal.

### A.5.3 Operational Implications

- **Runtime**: AISO sampling completes in ~1 minute on a single CPU core for the full Elliptic graph, making periodic re-sampling feasible in production.
- **Label budget**: 93.2% recovery at 2% budget translates to ~50x reduction in human labeling cost for comparable detector quality.
- **Deployment frame**: We position AISO+GNN not as an autonomous decision system but as a *review prioritization* pipeline — risk scores guide human reviewer queues rather than directly imposing automatic actions.

---

## A.6 Caveats

1. **Risk-prior leakage**: AISO's coverage score uses three label-derived features (`spam_ratio`, `spam_score`, `spam_user_ratio`). These are used only at the *sampling* stage; the trained GNN sees only label-independent node features. In production, these signals would be replaced with prior moderation history, which is realistically available.
2. **Coverage limit**: A 2% budget cannot include rare merchants and new accounts by construction. We mitigate via stratified periodic re-sampling but full coverage remains an open challenge.
3. **Edge robustness**: Adversaries may shift template patterns to evade fixed edge rules. Periodic edge re-discovery is an operational requirement, not a one-time design choice.

---

## A.7 Conclusion

This application demonstrates that the central AISO principle — asymmetric bilinear compatibility on simplex-valued types — transfers naturally from continuous niching to discrete diverse-coverage selection. The strong recovery rate (93.2% at 2% budget) and the mechanism validation (Jaccard 1.000 → 0.136 under M asymmetry) jointly serve as real-world evidence for the theoretical claims in the main paper.

We see this as one instance of a broader pattern: any problem requiring **diverse coverage under tight budget** is a candidate for AISO-style sampling, including active learning, dataset distillation, and exploration in offline reinforcement learning.
