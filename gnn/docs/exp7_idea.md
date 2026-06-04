# Experiment 7 — AISO Unified: Feature Selection + Subgraph Sampling

## 핵심 아이디어

기존 한계:
- Exp 1b: 노드 샘플링 됨, but **Dom-12 피처를 사람이 수동 설계**
- Exp 5: 피처 자동 선택 됨, but GNN 연결 없음

통합 제안:

> **AISO가 피처(Column)를 자동 선택하고, 그 피처 기준으로 노드(Row)를 스코어링해서 서브그래프를 뽑는다.**

Dom-12 수동 설계 과정 완전 제거. End-to-End 자동화.

---

## 파이프라인

```
W_i (feature type vector)
    ↓ temperature-annealed mask
feat_mask (K개 피처 선택)
    ↓ SGDClassifier.fit(X_all[:, feat_mask], y_train)
illicit_scores = predict_proba(illicit_pool)[:, 1]
    ↓ argsort → top-n_illicit 노드 선택
subgraph = top_illicit + fixed_licit_pool
    ↓ SGDClassifier proxy 평가 (inner loop)
score → b* 선택 → W_i 업데이트
    ↓ 최종
GNN(subgraph, feat_mask) → PR-AUC
```

**캐시 키**: `feat_mask_tuple` — feat_mask가 같으면 node scoring 결과도 동일 → O(1) 재사용

---

## Exp 5와의 차이

| 항목 | Exp 5 | Exp 7 |
|------|-------|-------|
| W_i 역할 | 피처 선택 확률 | 동일 |
| score 대상 | 피처 조합의 분류 품질 | **피처로 선별된 서브그래프**의 분류 품질 |
| 최종 평가 | LR on selected features | **GNN on selected subgraph + features** |
| Dom 피처 | 불필요 | **불필요** (자동화) |

---

## Smart M

Exp 5와 동일:
$$M[i][j] = -\overline{|C[\text{cl}_i, \text{cl}_j]|} + \gamma(\overline{MI}_j - \overline{MI}_i)$$

피처 클러스터: Elliptic 165개 원시 피처 (local f0~f93, aggregated f94~f164)
- f94~f164: 1-hop 이웃 집계 통계 → 그래프 위상 정보 내포

---

## 비교 대상

| 방법 | 피처 | 노드 선택 |
|------|------|----------|
| Random | 165 all | random |
| Dom-12 + AISO (Exp 1b) | 12 수동 | AISO 기반 |
| MI top-K | K개 자동 | variance-based |
| **AISO Unified (Exp 7)** | **K개 자동** | **feature-conditioned** |

---

## 승패 조건

**이길 조건**: f94~f164 aggregated 피처 안에 이웃 패턴 신호가 충분히 있을 것  
**질 조건**: 원시 피처만으로는 `illicit_neigh_ratio` 같은 명시적 위상 신호를 재현 못할 것

첫 실험은 raw 165 피처로 시작. 부족하면 degree/PageRank 4~5개 추가 (Hybrid Pool).

---

## 구현 순서

1. Elliptic 데이터 로드 + edgelist (GNN용)
2. 피처 클러스터링 + Smart M (Exp 5 코드 재사용)
3. `score_fn(feat_mask)`: SGD node scoring → top-n 선택 → proxy AUC
4. AISO wrapper (Exp 5 구조, score_fn만 교체)
5. 5 seeds × Smart M / Rand M
6. 최종 GNN 평가 (best mask per seed)
7. vs Dom-12 비교
