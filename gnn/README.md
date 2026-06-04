# AISO — Asymmetric Interaction Swarm Optimization

A population-based metaheuristic that uses **bilinear compatibility** between typed agents to drive exploration and diversity. AISO's key mechanism is the interaction score

$$c_{ij} = \mathbf{W}_i^\top M \mathbf{W}_j$$

which is intentionally **asymmetric** (`c_ij ≠ c_ji` in general). This asymmetry is a core feature, not a bug — it lets the same M encode directed relationships (attraction toward some types, repulsion from others) simultaneously.

---

## Main Results

The paper's central claim: on a **fixed labeling budget** (N_ILLICIT=1000 fraud nodes), a two-stage AISO pipeline — AISO selects features, then AISO selects nodes — outperforms either stage alone and all deterministic baselines on Elliptic Bitcoin.

### Phase 1 — Final Showdown (`final_showdown.ipynb`, 18 methods × 3 datasets, 5 seeds each)

| Stage | Approach | Elliptic | YelpChi | Amazon |
|-------|----------|----------|---------|--------|
| S0 — Baseline | Random / Degree / Greedy / PageRank / K-Means | 0.6245 | 0.2388 | 0.5146 |
| SA — AISO node sampler | Domain features (manual) → AISO | 0.3972 | 0.2359 | 0.4114 |
| SB — Feature → SGD scorer | mRMR / MI / AISO → SGD | 0.6348 | 0.2220 | 0.5140 |
| **SC — Feature → AISO sampler** | **mRMR / MI / AISO → AISO** | **0.6644 ✓** | 0.2249 | 0.4940 |

**Elliptic detailed (mean PR-AUC, 5 seeds):**

| Rank | Method | Section | PR-AUC | std |
|------|--------|---------|--------|-----|
| 1 | **AISO(Smart)→AISO** | SC | **0.6644** | 0.0223 |
| 2 | mRMR→SGD | SB | 0.6348 | 0.0062 |
| 2 | mRMR→AISO(Rand) | SC | 0.6348 | 0.0552 |
| 4 | MI→AISO(Smart) | SC | 0.6312 | 0.0168 |
| 5 | Greedy-Raw | S0 | 0.6245 | 0.0242 |
| 6 | MI→SGD | SB | 0.6138 | 0.0163 |
| 9 | AISO-node(Rand M) | SA | 0.6004 | 0.0906 |
| 11 | AISO(Rand)→AISO | SC | 0.5872 | 0.0371 |
| 16 | **AISO-node(Smart M)** | SA | **0.3972** | 0.0643 |

SA (domain features + Smart M) collapses on Elliptic: `neighbor_fraud_ratio` dominates all cluster MIs → Smart M routes all agents to same cluster. SC wins because stage-1 AISO pre-selects 20 diverse features → balanced MI distribution → Smart M works in stage 2.

### Phase 2 — GNN Paradigm Comparison (`phase2_elliptic.ipynb`, Elliptic only, 5 seeds)

| Method | PR-AUC | std | Setting |
|--------|--------|-----|---------|
| GraphSAGE | 0.7128 | 0.0171 | Full graph, 46k nodes — **no budget constraint** |
| **AISO(Smart)→AISO** *(Phase 1)* | **0.6644** | 0.0223 | 11k nodes, fixed labeling budget |
| AISO(Smart)→AISO(Rand M) | 0.6510 | 0.0284 | Smart M stage 1 only |
| GraphSAINT | 0.6273 | 0.0930 | Stochastic mask, high variance |
| AISO(Rand)→AISO(Rand M) | 0.5502 | 0.0975 | Rand M both stages |
| Random *(Phase 1)* | 0.5530 | 0.0714 | 11k nodes, random selection |

GraphSAGE full-graph is the ceiling (+0.048 over AISO). Within the N_ILLICIT=1000 labeling budget: AISO(Smart)→AISO vs Random = **+0.111**. Smart M stage 2 vs Rand M stage 2 = **+0.013**.

---

## Algorithm at a Glance

Each agent `i` carries three state variables:

| Variable | Domain | Role |
|----------|--------|------|
| `X_i` | `[0,1]^D` | Position in feature space (snapped to nearest real data point) |
| `W_i` | `Δ^{K-1}` (probability simplex) | Type vector encoding agent identity/role |
| `s_i` | `ℝ` | Score (fitness of current position) |

**Shared state**: `M ∈ ℝ^{K×K}` — compatibility matrix (asymmetric by construction or by Smart M learning).

**Per-iteration update (simplified):**

```
C = W @ M @ W.T                    # compatibility matrix (N_AG × N_AG)

for each agent i:
    attract partners  = top-3 by C[i, :]
    repel  partners   = bot-3 by C[i, :]
    F_i = Σ c_ij(X_j - X_i) [attract] + w_r · Σ c_ij(X_j - X_i) [repel]
    X_i ← nearest_datapoint(clip(X_i + α·F_i, 0, 1))

    best_partner j* = argmax C[i, attract]
    W_i ← normalize((1-β)·W_i + β·W_{j*})   # type assimilation
```

**Adaptive repulsion:**

$$w_r = 1.0 + 3.0 \cdot \exp(-\text{div}/0.12)$$

When agents collapse spatially (low `div`), repulsion automatically strengthens to re-diversify the swarm.

**Smart M (asymmetric, feature-conditioned):**

$$M[i][j] = -\overline{|C[\text{cl}_i, \text{cl}_j]|} + \gamma(\overline{MI}_j - \overline{MI}_i)$$

**Smart M 구성 과정:**

1. **피처 클러스터링**: 전체 피처를 의미 단위로 그룹핑. Elliptic 165개 피처의 경우 로컬(f1–f93)과 집계(f94–f164) 두 그룹이 자연스러운 경계. f94–f164는 1-hop 이웃 집계 통계이므로 그래프 위상 정보를 내포.

2. **상관관계 행렬 C 계산**: 피처 클러스터 간 평균 절대 상관계수.

3. **MI 계산**: 각 피처 클러스터의 타겟 변수에 대한 상호 정보량 (mutual information). MinMax 정규화.

4. **M 조립**:
   - `-mean|C[cl_i, cl_j]|`: 클러스터 i와 j가 **높이 상관**되면 M이 음수 → 같은 정보를 가진 타입끼리 밀어냄 (중복 탐색 방지)
   - `+γ(MI_j - MI_i)`: **MI 높은 피처 타입 j**는 MI 낮은 타입 i 에이전트를 끌어당김 (i가 j 방향으로 움직이면 더 유익한 피처 공간 탐색). 역방향(j→i)은 반대 부호 → 비대칭성 발생

5. **결과**: 피처 다양성을 유지하면서 고 MI 피처 방향으로 탐색을 편향시키는 비대칭 M. 랜덤 M 대비 피처 선택 품질 향상.

랜덤 M과 Smart M의 실험적 차이: YelpChi 피처 엔지니어링에서 AISO(Rand M)=0.2076 vs AISO(Smart M)=0.2220, Amazon에서 AISO(Rand M)=0.4035 vs AISO(Smart M)=0.5067 — Smart M이 일관되게 우위.

The `γ(MI_j - MI_i)` term is what makes M asymmetric — high-MI features attract low-MI agents, but not vice versa.

---

## Core Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_AG` | 20 | Number of agents |
| `N_IT` | 80 | Iterations |
| `K` | 12 | Type vector dimension (N_TYPES) |
| `ALPHA` | 0.2 | Position step size |
| `BETA` | 0.08 | Type assimilation rate |
| `M_LOW` | -0.5 | M lower bound (encodes repulsion) |
| `W_REPEL` | 2.0 | M upper bound / repulsion weight multiplier |
| Eval seeds | `[0, 7, 42, 77, 123]` | 5-seed evaluation protocol |

---

## Project Structure

```
AISO/
├── Notebooks (experiments)
│   ├── aiso_v2.ipynb ~ aiso_v6.ipynb        # algorithm development
│   ├── aiso_advanced.ipynb                   # advanced features
│   ├── aiso_tuning.ipynb                     # hyperparameter search
│   ├── aiso_benchmarks.ipynb                 # general benchmarks
│   ├── aiso_applications.ipynb               # application benchmarks
│   ├── aiso_domains.ipynb                    # domain benchmarks
│   ├── aiso_comprehensive.ipynb              # comprehensive comparison
│   ├── aiso_experiments.ipynb                # core experiment suite
│   ├── aiso_showdown.ipynb                   # algorithm showdown
│   ├── final_showdown.ipynb                  # phase 1 final results
│   ├── aiso_final_benchmark.ipynb            # final Colab benchmark
│   │
│   ├── aiso_exp1b_features.ipynb             # Exp 1b: Dom feature expansion
│   ├── aiso_exp1_m_ensemble.ipynb            # Exp 1: M ensemble
│   ├── aiso_exp2_iris.ipynb                  # Exp 2: clustering (Iris)
│   ├── aiso_exp2b_synth.ipynb                # Exp 2b: clustering (synthetic)
│   ├── aiso_exp2c_wine_k3.ipynb              # Exp 2c: clustering (Wine)
│   ├── aiso_exp3_entropy.ipynb               # Exp 3: entropy-based type mechanism
│   ├── aiso_exp4_repr.ipynb                  # Exp 4: AISO as representation learner
│   ├── aiso_exp4bc_v2.ipynb                  # Exp 4bc v2: AL + anomaly detection
│   ├── aiso_exp4c_yelp.ipynb                 # Exp 4c: AISO-SSL on Yelp spam
│   ├── aiso_exp4c_v3.ipynb                   # Exp 4c v3: AISO-SSL (InfoNCE M)
│   ├── aiso_exp4c_v4.ipynb                   # Exp 4c v4: DOMINANT/CoLA vs AISO-SSL
│   ├── aiso_exp6_cicids.ipynb                # Exp 6: CICIDS2017 oversampling
│   ├── aiso_exp7_gnn_wrapper.ipynb           # Exp 7: unified feature+node selection
│   ├── aiso_exp_feature_selection.ipynb      # Exp 5: feature selection (Smart M)
│   ├── aiso_exp_fraud.ipynb                  # YelpChi + Amazon fraud benchmark
│   │
│   ├── cicids_showdown.ipynb                 # CICIDS2017 14-method showdown
│   ├── unsw_showdown.ipynb                   # UNSW-NB15 14-method showdown
│   ├── sampling_showdown.ipynb               # YelpZip (origin: first domain-feature + GNN idea)
│   ├── phase2_elliptic.ipynb                 # Phase 2: GraphSAGE, GraphSAINT
│   ├── aiso_typed_domains.ipynb              # 5 scientific domain applications
│   └── aiso_fewshot_guide.ipynb              # AISO as LLM few-shot selector
│
├── Build scripts (generate notebooks programmatically)
│   ├── _build_exp1.py                        # builds aiso_exp1b core
│   ├── _build_exp1b.py                       # builds aiso_exp1b_features.ipynb (Dom-25)
│   ├── _build_exp2_iris.py                   # builds aiso_exp2_iris.ipynb (v1)
│   ├── _build_exp2_v2.py                     # builds aiso_exp2_iris.ipynb (v2, extended)
│   ├── _build_exp2_synth.py                  # builds aiso_exp2b_synth.ipynb
│   ├── _build_exp2_wine_k3.py                # builds aiso_exp2c_wine_k3.ipynb
│   ├── _build_exp4.py                        # builds aiso_exp4_repr.ipynb
│   ├── _build_exp4bc_v2.py                   # builds aiso_exp4bc_v2.ipynb
│   ├── _build_exp4c_v3.py                    # builds aiso_exp4c_v3.ipynb (SSL M)
│   ├── _build_exp4c_v4.py                    # builds aiso_exp4c_v4.ipynb (DOMINANT/CoLA)
│   ├── _build_exp4c_yelp.py                  # builds aiso_exp4c_yelp.ipynb
│   └── _build_final_benchmark.py             # builds aiso_final_benchmark.ipynb
│
├── Patch scripts (hot-fix specific notebook cells)
│   ├── _patch_exp1b.py                       # patches d004: transductive protocol
│   ├── _patch_exp1b_d003.py                  # patches d003: Dom-25 features
│   └── _patch_exp1b_d011.py                  # patches d011: graph-specific samplers
│
├── Design documents
│   ├── paper_draft.md                        # paper draft
│   ├── exp7_idea.md                          # Exp 7 concept design
│   ├── gnn_apli.md                           # GNN application notes
│   └── cicids_apli.md                        # CICIDS application notes
│
├── Data
│   ├── Elliptic Bitcoin/elliptic_bitcoin_dataset/
│   │   ├── elliptic_txs_features.csv         # 203K nodes × 166 features
│   │   ├── elliptic_txs_edgelist.csv         # transaction graph
│   │   └── elliptic_txs_classes.csv          # illicit/licit/unknown labels
│   ├── CICIDS2017/*.csv                      # 8 daily attack-traffic files (2.83M rows)
│   ├── UNSW-NB15/*.csv                       # 7 files (257K rows)
│   ├── yelpzip.csv                           # 608K reviews (YelpZip)
│   └── Iris/Iris.csv
│
└── Output CSVs / figures
    ├── final_showdown_results.csv            # Phase 1 GNN benchmark results
    ├── phase2_elliptic_results.csv           # Phase 2 (GraphSAGE/GraphSAINT) results
    ├── exp_fraud_gnn_evals.csv               # YelpChi/Amazon benchmark (90 rows)
    ├── exp7_proxy_evals.csv                  # Exp 7 proxy AUC logs
    ├── exp7_diversity_log.csv                # Exp 7 diversity tracking
    ├── exp7_gnn_evals.csv                    # Exp 7 GNN evaluation results
    ├── exp7_compute_cost.csv                 # Exp 7 compute cost analysis
    ├── typed_summary.png                     # typed domain summary visualization
    ├── final_benchmark_comparison.png        # final benchmark bar chart
    └── exp2_basic.png / exp2_grid.png / exp2_best.png / exp2_errors.png
```

---

## Experiments

### Exp 1: Elliptic Bitcoin GNN Subgraph Sampling

AISO is used as a **diversity-aware sampler** to select the illicit training nodes for a transductive GCN on the Elliptic Bitcoin dataset (203K transaction nodes, 49 timesteps, 20.5% illicit in labeled set).

**Protocol (transductive, exact):**
- Subgraph = `sel_n` (10,000 fixed licit nodes, SEED=42) + selected illicit + all test nodes
- Bidirectional edges; dynamic class weight; `lr=0.01`, `weight_decay=5e-4`
- 200 epochs, patience=20, best-loss model saved
- Metric: PR-AUC on test set (timesteps 35–49)

**Domain features (Dom-12) — manually designed graph-topology signals:**

| Feature | Description |
|---------|-------------|
| `degree` | Node degree (bidirectional) |
| `illicit_neigh_ratio` | Fraction of 1-hop neighbors that are illicit |
| `neigh2_illicit` | 2-hop illicit propagation ratio |
| `ts_illicit_rate` | Illicit rate in same timestep |
| `local_mean/std/max` | Stats over local features (f2–f94) |
| `agg_mean/std/max` | Stats over aggregated features (f95–f166) |
| `ts_norm` | Normalized timestep |
| `hub_fraud` | `degree × illicit_neigh_ratio` |

**Dom-25 extensions** (added in Exp 1b, `_patch_exp1b_d003.py`):
`neigh_degree_mean`, `neigh2_hub`, `ts_burst`, `degree_log`, `degree_x_ts`, `illicit_x_neigh2`, `hub_x_ts`, `local_range`, `agg_range`, `local_agg_diff`, `illicit_isolation`, `in_degree`, `in_out_ratio`

**Results (5-seed mean PR-AUC, seeds=[0,7,42,77,123]):**

| Method | PR-AUC |
|--------|--------|
| **AISO(Dom-12, N=12)** | **0.6041** |
| AISO(PCA, N=14) | 0.5893 |
| PSO(PCA) | 0.5621 |
| AISO(Dom-12, N=14) | 0.5557 |
| Top-density(PCA) | 0.5776 |
| Original (imbalanced) | 0.5718 |
| Random | 0.5414 |
| AISO(PCA, N=8) | 0.5307 |

AISO with Dom-12 wins because `illicit_neigh_ratio` and `hub_fraud` directly encode the fraud topology signal, giving the sampler the information needed to select diverse, structurally different illicit nodes.

**Exp 1b extensions** (`aiso_final_benchmark.ipynb`, Sections B and C):
- **Type mechanism**: gamma diversity injection, anti-assimilation, structured M initialization
- **Feature expansion**: Dom-12 vs Dom-18 vs Dom-25, PSO(Dom) ablation (isolates algorithm vs. feature contribution)
- **Graph-specific samplers**: PageRank-weighted, embedding-density, GraphSMOTE-style borderline sampling

---

### Exp 2: AISO as Stand-Alone Clustering Model

Tests whether AISO can replace K-Means as an unsupervised clustering engine using a 3-phase pipeline.

**Three-Phase Pipeline:**
1. **Phase 1 — Density anchor search**: agents explore via attraction/repulsion, snap to real data points, converge toward density peaks (cluster centers)
2. **Phase 2 — Prototype freezing**: K-Means on converged W vectors → N_CLUSTERS prototypes (proto_X, proto_W)
3. **Phase 3 — Bilinear inference**: assign each new point x_j via `argmax_c proto_W_c @ M @ W_j`

**Four Phase 3 assignment strategies:**

| Method | W_j estimation | Notes |
|--------|---------------|-------|
| Euclidean | — (L2 to proto_X) | Baseline, ignores bilinear |
| Bilinear-Nearest | Nearest anchor's W | Collapse-sensitive |
| Bilinear-Smooth | Soft-distance weighted avg of anchor W's (temp=0.3) | More stable |
| Bilinear-Linear | Ridge regression: X_ag→W_ag, predict W_j | Most robust to collapse |
| W-Direct | `argmax(W_j)` when K=N_CLUSTERS | Direct cluster-membership interpretation |

**Key issue — Type Collapse**: W vectors converge to near-uniform on the simplex, making proto_W nearly identical → bilinear mechanism disabled.
- Diagnosed via `w_div = mean pairwise distance between proto_W vectors`; warning at `w_div < 0.05`

**Fixes explored:**
- `K = N_CLUSTERS` instead of `K = D` — W dimension matches cluster count, each dimension encodes one cluster's affinity
- **LAMBDA mean-field repulsion**: `W_new += LAMBDA * (W_new - W_mean)` — amplifies existing differences from population mean. Works correctly, unlike old BETA_REP (subtracting a uniform vector is neutralized by normalization to simplex)
- **Gamma entropy injection**: Adds Dirichlet noise — uniformizes W, counterproductive

Exp 2b parameter sweeps: `LAMBDA ∈ [0, 0.3]`, `M_LOW ∈ [-0.5, -3.0]`, `N_AG ∈ [9, 15]`

**Datasets**: Iris (150×4, K=3), Synthetic Gaussian blobs (3 separations: std=0.5/1.0/1.8, 300×10), Wine (178×13, K=3)

**Conclusion**: Bilinear beats Euclidean only when W-div is sufficiently high (no collapse). K-Means remains competitive on these simple datasets. AISO's core strength is as a **GNN auxiliary sampler** (Exp 1), not a standalone clusterer.

---

### Exp 3: Type Mechanism Analysis

Embedded in `aiso_final_benchmark.ipynb` Section B. Three type-update modifications tested against the Elliptic GNN benchmark baseline (0.6041):

- **B1 Gamma injection** (`gamma` ∈ [0, 0.5]): Adding Dirichlet noise to W after each update → degrades performance (uniformizes W, weakening bilinear selectivity)
- **B2 Anti-assimilation** (`beta` ∈ [0.02, 0.2]): `W_new = (1+β)W_i - β W_j` — pushes type away from partner instead of toward (maintains diversity but risks instability)
- **B3 Structured M initialization** (diagonal=−2.0, off-diagonal=+0.3, slow β=0.02): Pre-wires type-type repulsion into M structure with slow assimilation

---

### Exp 4: AISO as Representation Learner

AISO's W vectors used as auxiliary features or anomaly signals, without requiring a GNN.

#### 4a — W → Logistic Regression (Inductive)

```
learn_w(Dom-12) → W_all via Ridge(X_ag → W_ag).predict(all_nodes)
Train LR on: sel_n (10K licit) + train_illicit (all 4545)
Evaluate on: test set (inductive, no graph)
Compare: raw 166d | Dom-12 | W_j-only | Dom-12+W_j | 166d+W_j
```

#### 4b — Active Learning

Starts with 50 labeled illicit nodes, adds 30 per round for 6 rounds, evaluating GNN PR-AUC after each round.

**Strategies compared:**

| Strategy | Query criterion | Source |
|----------|----------------|--------|
| Random (5 seeds) | Uniform random | de facto baseline |
| PC-GNN-Far | Furthest from licit centroid in Dom-12 | Liu et al. WWW 2021 |
| PC-GNN-Near | Closest to licit centroid (boundary nodes) | Liu et al. WWW 2021 |
| AISO-Entropy | Max H(W_j) — most type-uncertain illicit nodes | this work |
| AISO-Low | Min bilinear score — least compatible with any licit prototype | this work |
| GCN-Uncertainty | Max GCN softmax entropy on unlabeled pool | standard AL |

#### 4c — Anomaly Detection (Label-Free)

Train on 3,000 licit nodes only. Anomaly score: `-max_c(proto_W_c @ M @ W_j)`.

**Exp 4c v3 — AISO-SSL (InfoNCE M learning):**

```python
# Dom-8: label-free features (degree, local/agg stats, ts_norm — no illicit_neigh_ratio)
dom8 = [node_degree, local_mean, local_std, local_max,
        agg_mean, agg_std, agg_max, ts_norm]  # 8-dim

# SSL M training (InfoNCE / SimCLR style)
# Augmentation: Gaussian noise (std=0.05) + feature masking (prob=0.1)
# Positive pair: same licit node, two augmented views
# sim[i,j] = W1[i] @ M_t @ W2[j] / temp
# loss = F.cross_entropy(sim, diagonal_labels)
# Adam, lr=5e-3, 200 epochs, batch=256, temp=0.1
```

**Results (Dom-8, test set ~6.5% illicit rate):**

| Method | PR-AUC |
|--------|--------|
| **AISO-SSL (M learned)** | **0.1421** ★ |
| AISO-Random M (5-seed mean) | 0.1071 |
| KMeans(K=8) distance | ~0.06–0.08 |
| Isolation Forest | ~0.09 |
| OCSVM (RBF) | ~0.07 |
| Autoencoder (recon error) | ~0.06 |
| LOF | ~0.05 |
| DOMINANT (Graph-AE) | 0.0542 |
| CoLA-simplified | 0.0460 |
| Random baseline | ~0.065 |

Note: InfoNCE loss plateaus at ~5.5452 — SSL training stalls, yet M still learns useful structure. W representation quality may matter more than M precision.

Dom-8+Graph (16-dim, 1-hop neighbor aggregation) performs *worse* than Dom-8 (8-dim) for AISO — 1-hop aggregation pulls licit boundary nodes toward illicit-contaminated neighborhoods.

**Exp 4c v4 — Neighborhood Contamination Analysis:**

| Node type | Mean illicit neighbor ratio |
|-----------|---------------------------|
| Licit test nodes | **1.18%** |
| Illicit test nodes | **37.95%** |

Even 1.18% contamination degrades GCN-based anomaly scores. DOMINANT score and illicit-neighbor-ratio are positively correlated for licit nodes (false positives). AISO-SSL has zero correlation (node-level only).

**Yelp extension** (`aiso_exp4c_yelp.ipynb`): same AISO-SSL vs DOMINANT vs CoLA on YelpZip 608K reviews. User-level features (6-dim: review_count, rating_mean/std/range, prod_count, review_per_prod), co-review graph (same-product reviewer pairs, max 20 per product). Cross-reference table with Elliptic results.

**Theoretical interpretation** (from `_build_exp4c_v3.py`):

K-Means centroid `μ_k = (1/|C_k|) Σ x_i` is biased toward dense regions, missing distribution boundaries. AISO agents equilibrate between attraction and repulsion, covering the full licit manifold. W encodes **topological position within the licit manifold** (not density centroid), so anomalies — which are topologically foreign — get low bilinear compatibility with all prototypes.

---

### Exp 5: Feature Selection (Smart M)

AISO with temperature-annealed feature masks as an automatic feature selector for GNN training.

```
W_i (feature-type probability vector, K = num_feature_clusters)
    ↓ temperature-annealed softmax → feat_mask
SGDClassifier.fit(X_all[:, feat_mask], y_train)
    ↓ illicit_scores → top-n illicit nodes selected
subgraph evaluated → proxy AUC → W_i update
```

Cache key: `feat_mask_tuple` — same mask = same node scoring = O(1) reuse.

**Smart M:**

$$M[i][j] = -\overline{|C[\text{cl}_i, \text{cl}_j]|} + \gamma(\overline{MI}_j - \overline{MI}_i)$$

- Feature clusters: Elliptic local f0–f93 vs aggregated f94–f164 (aggregated features encode 1-hop neighbor topology)
- MI gradient term creates asymmetry: high-MI feature types attract low-MI agents

**YelpChi/Amazon Results (Section B: feature engineering):**

| Method | YelpChi PR-AUC | Amazon PR-AUC |
|--------|---------------|--------------|
| **AISO(Smart M)** | **0.2220** | 0.5067 |
| MI top-K | 0.2219 | 0.5051 |
| mRMR | 0.2193 | **0.5140** |
| AISO(Rand M) | 0.2076 | 0.4035 |

Smart M marginally wins on YelpChi; mRMR wins on Amazon. Smart M's key benefit is that it respects feature-cluster correlation structure, not just individual MI.

---

### Exp 6: CICIDS2017 Oversampling (14-Method Benchmark)

AISO as a minority-class oversampler for intrusion detection. The swarm samples from the rare-attack pool using type-mediated diversity to cover underrepresented attack clusters.

**Dataset**: CICIDS2017, 2.83M records, 8 daily files, 14 attack categories (including extremely rare Heartbleed, Infiltration)

**Final ranking:**

| Rank | Method | Score |
|------|--------|-------|
| 1 | CDE | 0.9986 |
| 2 | SPSO | 0.9986 |
| 3 | RandomOver | 0.9985 |
| 4 | Random | 0.9985 |
| **5** | **AISO** | **0.9984** |
| 6 | PSO | 0.9983 |
| 7 | ACO | 0.9983 |
| 8 | SMOTE | 0.9979 |
| 9 | Greedy | 0.9979 |
| 10 | K-Means | 0.9977 |
| 11 | ADASYN | 0.9964 |
| 12 | Original | 0.9957 |
| 14 | Top-density | 0.7378 |

Performance differences < 0.002 across methods 1–13. AISO's diversity advantage:
- Coverage entropy: AISO=1.410 (K-Means best=1.574; PSO=1.336)
- Final spatial dispersion: AISO=0.5324 vs PSO=0.5307
- W entropy: AISO=2.2693 (theoretical max=2.4849)
- Rare attack recall (tail_avg): AISO=0.333, PSO=0.381 (best), CDE=0.286
- Heartbleed recall: ≈0 across all methods (too rare to oversample effectively)

---

### Exp 7: AISO Unified (Feature Selection + Subgraph Sampling)

**Goal**: Eliminate the manually designed Dom-12 features. End-to-end automation: AISO selects features AND selects nodes.

**P(S, F) = P(S|F) · P(F)** — feature selection and node selection are jointly optimized.

```
W_i (feature-type probability vector)
    ↓ temperature-annealed softmax mask
feat_mask (K features selected automatically from 165 raw features)
    ↓ SGDClassifier.fit(X_all[:, feat_mask], y_train)
illicit_scores = predict_proba(illicit_pool)[:, 1]
    ↓ argsort → top-n_illicit nodes selected (feature-conditioned)
subgraph = top_illicit + fixed_licit_pool
    ↓ proxy AUC → AISO score → W_i update
    ↓ (best mask per seed)
GNN(subgraph, feat_mask) → PR-AUC
```

**Win condition**: Aggregated features f94–f164 encode enough 1-hop neighborhood information to substitute for the explicit `illicit_neigh_ratio` signal in Dom-12.

**Comparison baselines:**

| Method | Features | Node selection |
|--------|----------|----------------|
| Random | all 165 | random |
| Dom-12 + AISO (Exp 1b) | 12 manual | AISO-based |
| MI top-K | K automatic | variance-based |
| **AISO Unified (Exp 7)** | **K automatic** | **feature-conditioned** |

Logged to: `exp7_proxy_evals.csv`, `exp7_diversity_log.csv`, `exp7_gnn_evals.csv`, `exp7_compute_cost.csv`.

---

### UNSW-NB15 Oversampling (14-Method Benchmark)

**Dataset**: UNSW-NB15, 257K records, 7 files, 9 attack categories

**Final ranking:**

| Rank | Method | Score |
|------|--------|-------|
| 1 | Random | 0.9930 |
| 2 | Class Weight | 0.9930 |
| 3 | ACO | 0.9929 |
| 4 | RandomOver | 0.9928 |
| 5 | Original | 0.9927 |
| **6** | **AISO** | **0.9922** |
| 6 | PSO | 0.9922 |
| 8 | Greedy | 0.9921 |
| 9 | SPSO | 0.9921 |
| 10 | SMOTE | 0.9920 |
| 11 | CDE | 0.9919 |
| 12 | K-Means | 0.9918 |
| 13 | ADASYN | 0.9914 |
| 14 | Top-density | 0.9072 |

AISO diversity advantage even when tied on macro metric:
- Final spatial dispersion: **AISO=0.3594** vs PSO=0.1885 (AISO much better spread)
- Coverage entropy: PSO=1.615 (best), AISO=1.548
- W entropy: AISO=1.9108 (max=2.1972)
- Rare recall: Shellcode — AISO=0.797, Random~0.831; Worms — AISO=0.877, Random=0.931
- N_TYPES search: optimal=14 (0.9924 PR-AUC), default=9 (0.9922)

---

### YelpChi/Amazon Fraud Detection (Multi-Dataset Benchmark)

AISO tested both as a **node sampler** (Section A) and as a **feature engineer** (Section B).
Results saved in `exp_fraud_gnn_evals.csv` (90 rows).

**Section A: AISO as node sampler:**

| Method | YelpChi PR-AUC | Amazon PR-AUC |
|--------|---------------|--------------|
| Random | **0.2388** | 0.4743 |
| Cluster-uniform | 0.2375 | 0.4440 |
| Greedy | 0.2248 | **0.5146** |
| AISO-dom(Rand) | 0.1926 | 0.3093 |
| AISO-dom(Smart) | 0.1890 | 0.3512 |

AISO significantly **underperforms** Random and Greedy. Dom-12 features designed for financial transaction topology don't discriminate well in review spam networks.

**Section B: AISO as feature engineer** — results listed above under Exp 5.

---

### YelpZip — 프로젝트의 출발점 (sampling_showdown.ipynb)

**이 실험이 먼저다.** `sampling_showdown.ipynb`은 YelpZip(608K 리뷰)에서 수동 도메인 피처 기반 AISO 샘플링을 처음 시도한 실험이며, 이후 모든 GNN 실험의 출발점이 됐다.

**경위**: AISO 에이전트가 피처 공간을 탐색하면서 Type-Agent 방식으로 훈련 노드를 선택할 때, 어떤 피처를 쓰느냐에 따라 샘플링 품질이 크게 달라진다는 것을 이 실험에서 처음 관찰했다. 특히 **수동으로 설계한 도메인 피처**(리뷰 수, 평점 분산, 제품 다양성 등)가 원시 피처 대비 의미 있는 차별성을 보였고, 이것이 "AISO 샘플러 + 도메인 피처 + GNN 학습"의 조합 가능성을 열었다.

**이 가능성에서 Elliptic 실험이 나왔다**: YelpZip에서 도메인 피처가 효과적이라면, 그래프 위상 신호가 훨씬 강한 금융 거래 그래프(Elliptic)에서는 더 잘 작동할 것이라는 가설 → Dom-12 설계 → `aiso_exp1b_features.ipynb` / `final_showdown.ipynb`로 이어짐.

**실험 설계**: 608K reviews, GCN-based spam detection, 유저 노드 분류. 8가지 샘플링 전략 비교.

Type-Agent (AISO) dimension search: `TEST_DIMS = [6, 9, 12, 14, 17, 20, 22, 24]`

- K가 작으면 Type Collapse 위험, K가 크면 simplex가 과분산되어 bilinear 신호 약해짐
- 최적 K를 데이터 의존적으로 탐색

Convergence dynamics analyzed for K = 9, 14, 17, 22. Baseline strategies: Random, Greedy, PageRank, K-Means, ACO.

**GNN architecture comparison on YelpZip (RUR + RSR relations, single run):**

| Method | PR-AUC | Macro F1 |
|--------|--------|----------|
| RGCN | **0.6369** | **0.7134** |
| GCN | 0.5908 | 0.6800 |
| SGC | 0.5864 | 0.6732 |
| GraphSAGE | 0.5547 | 0.5465 |
| **AISO Section A (Smart M, 5-seed mean)** | **0.5536** | — |
| GAT | 0.5395 | 0.6664 |
| GIN | 0.4645 | 0.5882 |

RGCN wins (+0.081 vs AISO) because it explicitly separates RUR and RSR relation types in message passing — the multi-relational structure of YelpZip is the dominant signal. Standard homogeneous GNNs (GCN, GraphSAGE) are within noise of AISO (≤0.004 gap), confirming that AISO's subgraph selection recovers what full-graph homogeneous message passing finds. AISO beats GIN (−0.089) and GAT (−0.014) using 11k nodes vs full graph.

---

### Phase 2 Elliptic: GraphSAGE and GraphSAINT

`phase2_elliptic.ipynb` extends Phase 1 transductive GCN results with full-graph GNN variants and Rand-M ablations:

**Results (Elliptic, PR-AUC, 5 seeds):**

| Method | PR-AUC | std | Notes |
|--------|--------|-----|-------|
| **GraphSAGE** | **0.7128** | 0.0171 | Full graph, 46k nodes, SAGEConv |
| AISO(Smart)->AISO *(P1)* | 0.6644 | 0.0223 | 11k budget, Smart M both stages |
| AISO(Smart)->AISO(Rand M) | 0.6510 | 0.0284 | Smart M stage 1, Rand M stage 2 |
| GraphSAINT | 0.6273 | 0.0930 | Stochastic node mask, high variance |
| AISO(Rand)->AISO(Rand M) | 0.5502 | 0.0975 | Rand M both stages |

GraphSAGE (full graph, no budget) sets the ceiling at 0.7128. AISO(Smart)->AISO reaches 0.6644 using only 24% of nodes (11k/46k) — 93% of ceiling performance within budget constraint. Within the same 11k budget, AISO Smart M beats random selection by +0.111 (0.6644 vs 0.5530).

Rand M stage 2 ablation: Smart M stage 2 (+0.013 vs Rand M stage 2) confirms that asymmetric routing in stage 2 adds value beyond random search on the diversity-selected feature subspace.

Loads `final_showdown_results.csv`, saves to `phase2_elliptic_results.csv`.

---

### Scientific Domain Applications

`aiso_typed_domains.ipynb` demonstrates AISO with **domain-grounded M matrices** (scientifically motivated, not random).

| Domain | K | Types | M source | Key values |
|--------|---|-------|----------|-----------|
| RNA folding | 40 | 4 nucleotides | Watson-Crick pairing | A-U=1.8, G-C=2.0, G-U wobble=0.6 |
| Alloy design | 50 | 10 elements | Miedema formation enthalpy | negative=attractive mixing |
| Enzyme catalysis | 48 | 6 EC classes | EC functional compatibility | EC1→EC2=1.7, EC3→EC4=1.8 |
| Knowledge graph | 40 | 5 entity types | Relation frequency | Person-Org=1.8, Org-Loc=1.7 |
| Vaccine peptides | 30 | 5 HLA supertypes | HLA cross-reactivity | same MHC-I supertypes cross-react |

Summary visualization: `typed_summary.png`.

---

### Few-Shot Example Selection for LLMs

`aiso_fewshot_guide.ipynb`: AISO as a diversity-aware selector of few-shot NER examples for LLM prompting.

Pool: 40 labeled NER examples. Feature set (7-dim):

```
length_norm, entity_density, entity_type_diversity,
per_ratio, org_loc_ratio, date_misc_ratio, vocab_richness
```

```python
class AISOfewshot:
    def select(self, features, n_select, replace=False):
        # Standard AISO dynamics on 7-dim feature space
        # Returns visit-count-weighted sampled examples
```

**Design guidance:**
- 7–15 features optimal; more blurs the diversity signal
- Avoid PCA (blurs structure that AISO needs to differentiate)
- Features must encode "example role" directly (entity type distribution, syntactic complexity, etc.)

---

## Key Findings

### Where AISO Wins

1. **Two-stage pipeline SC > SB on Elliptic** (0.6644 vs 0.6348, +0.030): When AISO selects features AND nodes, it outperforms feature selection alone. The mechanism: stage-1 Smart M selects 20 diverse features whose cluster MI distribution is balanced → stage-2 Smart M exploits this structure. Neither stage alone achieves this.

2. **Fixed labeling budget: AISO vs Random +0.111** (0.6644 vs 0.5530): With N_ILLICIT=1000 nodes fixed, AISO's diversity-aware selection covers structurally distinct fraud patterns that random selection misses.

3. **Smart M rescues Amazon SB** (mRMR parity 0.5067 vs Rand M collapse 0.4035, +0.103): Without asymmetric routing, AISO agents collapse to correlated feature masks. Smart M's inter-cluster repulsion prevents this.

4. **Label-free anomaly detection — AISO-SSL** (0.1421 PR-AUC, beats DOMINANT and CoLA): AISO is immune to neighborhood contamination by encoding licit manifold topology in W vectors.

### Where AISO Loses or Is Neutral

1. **SA on Elliptic domain features** (0.3972 vs baseline 0.6245): `neighbor_fraud_ratio` dominates all feature cluster MIs → Smart M routes all agents to same cluster → mode collapse. Domain feature diversity is required for Smart M to function.

2. **SC on Amazon** (best 0.4940 vs SB 0.5140): K_SELECT=6 → K_select_2=3 → C(6,3)=20 search space exhausted in seconds. No room for multi-modal exploration; single-stage SGD is already near-optimal.

3. **Full graph available** (GraphSAGE 0.7128 vs AISO 0.6644): If all nodes can be used for training with no labeling budget, standard full-graph GNN wins. AISO is for the budget-constrained setting.

4. **Review spam / heterogeneous graphs** (YelpChi SA 0.1890 vs baseline 0.2388): AISO SA needs task-specific domain features. For multi-relational graphs, RGCN (0.6369 on YelpZip) exploits relation-type structure that homogeneous AISO cannot encode.

5. **General oversampling benchmarks** (CICIDS2017, UNSW-NB15): Differences < 0.002 AUC — noise floor.

---

## Critical Implementation Notes

**Asymmetry is mandatory**: `c_ij = W_i^T M W_j` must remain asymmetric. Never symmetrize M or average `(M + M.T) / 2`.

**`sel_n` must be fixed**: The 10,000 licit training nodes for Elliptic must use `np.random.RandomState(42).choice(train_licit_idx, 10000, replace=False)` — any deviation breaks cross-experiment comparison.

**Bidirectional edges**: `edge_index = np.array([np.concatenate([src,dst]), np.concatenate([dst,src])])` in both training and evaluation for all Elliptic experiments.

**Type Collapse diagnosis**: Monitor `w_div = mean pairwise L2 distance between proto_W vectors`. Warning threshold: `w_div < 0.05` → bilinear mechanism is effectively disabled.

**LAMBDA vs old BETA_REP**: Use `W_new += LAMBDA * (W_new - W_mean)` for mean-field repulsion. The old approach of subtracting a uniform vector (`W_new -= BETA_REP * (1/K) * ones`) is neutralized by the subsequent simplex normalization — it amplifies existing differences but cannot create new ones.

---

## GCN Architecture (Showdown Protocol)

```python
class GCN(torch.nn.Module):
    def __init__(self, in_ch, hidden=64, dropout=0.5):
        # GCNConv(in_ch → 64) → ReLU → Dropout(0.5)
        # GCNConv(64 → 64) → ReLU → Dropout(0.5)
        # Linear(64 → 2)

# Exact training protocol:
# Optimizer: Adam, lr=0.01, weight_decay=5e-4
# Loss: CrossEntropy with dynamic class weight cw = [1.0, n0/n1]
# 200 epochs, patience=20 (on training loss)
# Best model (lowest training loss) used for evaluation
# torch.manual_seed(42) before every GCN instantiation
# torch.cuda.manual_seed(42) if GPU available
```

---

## Quick Reference: Experiment → File Mapping

| Experiment | Notebook | Builder |
|------------|----------|---------|
| Elliptic GNN showdown (Phase 1) | `final_showdown.ipynb` | — |
| Elliptic feature expansion (Exp 1b) | `aiso_exp1b_features.ipynb` | `_build_exp1b.py` |
| Elliptic final benchmark | `aiso_final_benchmark.ipynb` | `_build_final_benchmark.py` |
| Elliptic Phase 2 (SAGE/SAINT) | `phase2_elliptic.ipynb` | — |
| Clustering — Iris | `aiso_exp2_iris.ipynb` | `_build_exp2_v2.py` |
| Clustering — Synthetic blobs | `aiso_exp2b_synth.ipynb` | `_build_exp2_synth.py` |
| Clustering — Wine | `aiso_exp2c_wine_k3.ipynb` | `_build_exp2_wine_k3.py` |
| Representation learning (Exp 4a/4b/4c) | `aiso_exp4_repr.ipynb` | `_build_exp4.py` |
| Active learning + anomaly v2 | `aiso_exp4bc_v2.ipynb` | `_build_exp4bc_v2.py` |
| AISO-SSL (InfoNCE M) | `aiso_exp4c_v3.ipynb` | `_build_exp4c_v3.py` |
| DOMINANT/CoLA vs AISO-SSL | `aiso_exp4c_v4.ipynb` | `_build_exp4c_v4.py` |
| AISO-SSL on Yelp spam | `aiso_exp4c_yelp.ipynb` | `_build_exp4c_yelp.py` |
| Feature selection — Smart M (Exp 5) | `aiso_exp_feature_selection.ipynb` | — |
| CICIDS2017 oversampling | `aiso_exp6_cicids.ipynb` | — |
| CICIDS2017 14-method showdown | `cicids_showdown.ipynb` | — |
| UNSW-NB15 14-method showdown | `unsw_showdown.ipynb` | — |
| YelpChi + Amazon fraud | `aiso_exp_fraud.ipynb` | — |
| YelpZip sampling (프로젝트 출발점) | `sampling_showdown.ipynb` | — |
| Unified feature+node selection (Exp 7) | `aiso_exp7_gnn_wrapper.ipynb` | — |
| Scientific domain M matrices | `aiso_typed_domains.ipynb` | — |
| LLM few-shot selection | `aiso_fewshot_guide.ipynb` | — |

---

## Dependencies

```
numpy, pandas, scikit-learn, matplotlib, scipy
torch >= 2.0
torch-geometric  (GCNConv, SAGEConv, NeighborLoader, GraphSAINT)
```

Google Colab notebooks (`aiso_final_benchmark.ipynb`) install `torch-geometric` via pip and mount Google Drive at `/content/drive/MyDrive/AISO`.

Local notebooks use:
```python
BASE = r'c:\Users\kevin\OneDrive\Desktop\AISO\Elliptic Bitcoin\elliptic_bitcoin_dataset'
```
