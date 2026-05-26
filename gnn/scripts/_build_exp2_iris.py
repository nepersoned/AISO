import json

def code(cell_id, source):
    return {
        "id": cell_id,
        "cell_type": "code",
        "source": source,
        "metadata": {},
        "outputs": [],
        "execution_count": None
    }

def md(cell_id, source):
    return {
        "id": cell_id,
        "cell_type": "markdown",
        "source": source,
        "metadata": {}
    }

cells = []

cells.append(md('e001', '''# Experiment 2: AISO-based Unsupervised Clustering Model

AISO를 GNN 보조 도구가 아닌 독립형 클러스터링 엔진으로 재구성.

- **Phase 1** — 밀도 기반 앵커 탐색: 에이전트들이 반발력을 통해 데이터 밀도 피크(클러스터)를 자율 탐색
- **Phase 2** — 프로토타입 프리징: 수렴된 위치(X_i)와 타입 벡터(W_i)를 클러스터 앵커로 고정
- **Phase 3** — 쌍선형 호환성 추론: 새 데이터의 W_j를 추정 후 W_i^T M W_j로 클러스터 할당

검증 데이터셋: Iris (150샘플, 4피처, 3클래스)
'''))

cells.append(code('e002', '''import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings("ignore")

iris = load_iris()
X_raw  = iris.data
y_true = iris.target
names  = iris.target_names

X = MinMaxScaler().fit_transform(X_raw)
N, D = X.shape

print(f"Iris: {N}개 샘플, {D}개 피처, {len(names)}개 클래스 {list(names)}")
print(f"클래스 분포: {np.bincount(y_true)}")
'''))

cells.append(md('e003', '''## Phase 1: 밀도 기반 앵커 탐색

에이전트들이 호환성 행렬 M을 통한 인력/척력으로 데이터 공간 탐색.
가장 가까운 실제 데이터 포인트에 스냅하면서 고밀도 영역(클러스터 중심)에 수렴.
'''))

cells.append(code('e004', '''# ── 하이퍼파라미터 ──
N_AG       = 15    # 에이전트 수 (클러스터당 ~5)
N_IT       = 300   # 이터레이션
ALPHA      = 0.3   # 이동 스텝
BETA       = 0.08  # 타입 믹스율
M_LOW      = -0.5  # M 하한 (척력 인코딩)
W_REPEL    = 2.0   # M 상한
K          = D     # 타입 차원 = 피처 차원
N_CLUSTERS = 3     # 목표 클러스터 수
SEED       = 42

rng = np.random.RandomState(SEED)

# 초기화
init_idx = rng.choice(N, N_AG, replace=False)
X_ag = X[init_idx].copy()                        # 에이전트 위치
W_ag = rng.dirichlet(np.ones(K), N_AG)           # 타입 벡터
M    = rng.uniform(M_LOW, W_REPEL, (K, K))       # 호환성 행렬 (고정)
visit = np.zeros(N)
for i in init_idx: visit[i] += 1

# ── Phase 1 실행 ──
div_log = []

for t in range(N_IT):
    C = W_ag @ M @ W_ag.T
    np.fill_diagonal(C, 0)

    for i in range(N_AG):
        ci = C[i].copy(); ci[i] = 0
        att = np.argsort(ci)[-3:]   # 상위 3 인력 파트너
        rep = np.argsort(ci)[:3]    # 하위 3 척력 파트너

        Fv = (sum(ci[j] * (X_ag[j] - X_ag[i]) for j in att) +
              W_REPEL * sum(ci[j] * (X_ag[j] - X_ag[i]) for j in rep)) / 6.0

        candidate = np.clip(X_ag[i] + ALPHA * Fv, 0, 1)
        nn = np.argmin(np.linalg.norm(X - candidate, axis=1))
        X_ag[i] = X[nn]
        visit[nn] += 1

        bj = att[np.argmax(ci[att])]
        W_ag[i] = (1 - BETA) * W_ag[i] + BETA * W_ag[bj]
        W_ag[i] /= W_ag[i].sum()

    if t % 50 == 0:
        div = np.mean([np.linalg.norm(X_ag[i] - X_ag[j])
                       for i in range(N_AG) for j in range(i+1, N_AG)])
        div_log.append((t, div))
        print(f"  t={t:3d}  agent diversity={div:.4f}")

print("Phase 1 완료")
'''))

cells.append(code('e005', '''# 에이전트 수렴 시각화 (PCA 2D)
pca2 = PCA(n_components=2, random_state=SEED).fit(X)
X_2d  = pca2.transform(X)
Ag_2d = pca2.transform(X_ag)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 왼쪽: 실제 레이블
colors = ['steelblue', 'tomato', 'seagreen']
for c, name in enumerate(names):
    mask = y_true == c
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[c], label=name, alpha=0.5, s=30)
axes[0].scatter(Ag_2d[:, 0], Ag_2d[:, 1], c='black', marker='*', s=200, zorder=5, label='Agents')
axes[0].set_title('True Labels + Agent Positions (PCA 2D)')
axes[0].legend(fontsize=8)

# 오른쪽: 방문 히트맵
sc = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=visit, cmap='YlOrRd', s=30)
axes[1].scatter(Ag_2d[:, 0], Ag_2d[:, 1], c='black', marker='*', s=200, zorder=5)
plt.colorbar(sc, ax=axes[1], label='Visit Count')
axes[1].set_title('Agent Visit Density')

plt.tight_layout()
plt.show()

# diversity 추이
ts, divs = zip(*div_log)
plt.figure(figsize=(6, 3))
plt.plot(ts, divs, 'o-')
plt.xlabel('Iteration'); plt.ylabel('Agent Diversity')
plt.title('Phase 1: Agent Diversity over Time')
plt.tight_layout(); plt.show()
'''))

cells.append(md('e006', '''## Phase 2: 프로토타입 프리징

수렴된 에이전트들의 W 벡터를 K-Means로 그룹핑 → N_CLUSTERS개 프로토타입 생성.
각 프로토타입은 위치(proto_X)와 타입 벡터(proto_W)로 표현됨.
'''))

cells.append(code('e007', '''# ── Phase 2: 프로토타입 프리징 ──
km_type = KMeans(N_CLUSTERS, random_state=SEED, n_init=10).fit(W_ag)
a2c = km_type.labels_  # 에이전트 → 클러스터

proto_X = np.array([X_ag[a2c == c].mean(0) for c in range(N_CLUSTERS)])
proto_W = np.array([W_ag[a2c == c].mean(0) for c in range(N_CLUSTERS)])
proto_W /= proto_W.sum(1, keepdims=True)

print("=== 프로토타입 (Phase 2) ===")
for c in range(N_CLUSTERS):
    members = np.where(a2c == c)[0]
    print(f"  클러스터 {c}: 에이전트 {list(members)}  ({len(members)}개)")
    print(f"    proto_W: [{', '.join(f'{v:.3f}' for v in proto_W[c])}]")

# W 벡터 시각화
fig, ax = plt.subplots(figsize=(7, 4))
for c in range(N_CLUSTERS):
    ax.plot(proto_W[c], 'o-', label=f'Cluster {c}')
ax.set_xticks(range(K))
ax.set_xticklabels([f'f{i+1}' for i in range(K)])
ax.set_title('Frozen Prototype W Vectors')
ax.set_ylabel('Weight'); ax.legend()
plt.tight_layout(); plt.show()
'''))

cells.append(md('e008', '''## Phase 3: 쌍선형 호환성 추론

새 데이터 포인트 x_j에 대해:
1. 가장 가까운 앵커의 W를 W_j로 추정
2. 쌍선형 점수 `proto_W @ M @ W_j` 계산
3. 최고 점수 클러스터에 할당

비교: Euclidean 거리 기반 할당 vs Bilinear 기반 할당 vs K-Means
'''))

cells.append(code('e009', '''# ── Phase 3: 쌍선형 호환성 추론 ──
pred_bi = np.zeros(N, dtype=int)
pred_eu = np.zeros(N, dtype=int)

for j in range(N):
    # Euclidean 할당 (프로토타입까지 거리)
    pred_eu[j] = np.argmin(np.linalg.norm(proto_X - X[j], axis=1))

    # Bilinear 할당
    nn_anchor = np.argmin(np.linalg.norm(X_ag - X[j], axis=1))
    W_j = W_ag[nn_anchor]
    pred_bi[j] = np.argmax(proto_W @ M @ W_j)

# K-Means 비교
km = KMeans(N_CLUSTERS, random_state=SEED, n_init=10).fit(X)

print(f"{'Method':<22} {'ARI':>7} {'NMI':>7}")
print('-' * 38)
results = {}
for name, pred in [('AISO Bilinear', pred_bi),
                    ('AISO Euclidean', pred_eu),
                    ('K-Means', km.labels_)]:
    ari = adjusted_rand_score(y_true, pred)
    nmi = normalized_mutual_info_score(y_true, pred)
    results[name] = (ari, nmi)
    flag = ' ★' if ari == max(adjusted_rand_score(y_true, p)
                               for p in [pred_bi, pred_eu, km.labels_]) else ''
    print(f"  {name:<20} {ari:>7.4f} {nmi:>7.4f}{flag}")
'''))

cells.append(code('e010', '''# 클러스터링 결과 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ['True Labels', 'AISO Bilinear', 'K-Means']
preds  = [y_true, pred_bi, km.labels_]

Proto_2d = pca2.transform(proto_X)

for ax, title, pred in zip(axes, titles, preds):
    for c in range(N_CLUSTERS):
        mask = pred == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], alpha=0.5, s=30, label=f'C{c}')
    if title != 'True Labels':
        ax.scatter(Proto_2d[:, 0], Proto_2d[:, 1], c='black', marker='*', s=200, zorder=5)
    ax.set_title(title); ax.legend(fontsize=8)

plt.tight_layout(); plt.show()

# confusion 매트릭스 스타일 비교
from sklearn.metrics import confusion_matrix
print("\\n=== AISO Bilinear vs True Labels ===")
# best permutation 찾기
from itertools import permutations
best_acc, best_perm = 0, None
for perm in permutations(range(N_CLUSTERS)):
    mapped = np.array([perm[p] for p in pred_bi])
    acc = (mapped == y_true).mean()
    if acc > best_acc:
        best_acc, best_perm = acc, perm
mapped_bi = np.array([best_perm[p] for p in pred_bi])
print(f"  최적 매핑 정확도: {best_acc:.4f}")
print(confusion_matrix(y_true, mapped_bi))
'''))

cells.append(md('e011', '''## 결과 요약 및 해석

### 핵심 질문
1. **Bilinear > Euclidean?** → Phase 3의 쌍선형 추론이 단순 거리보다 나으면 M 행렬이 유의미한 구조를 인코딩한 것
2. **AISO > K-Means?** → GNN 없이 독립형 클러스터러로서 경쟁력 확인

### 의의
- AISO의 반발력 메커니즘이 K-Means의 탐욕적 클러스터 수렴을 방지할 수 있음
- W 벡터가 클러스터 타입 정보를 인코딩하면, 쌍선형 형식이 유클리디안 거리보다 더 적절한 유사도 측정 가능
- Iris에서 검증 후 → 고차원 실제 데이터(Elliptic, CICIDS 등) 적용 방향
'''))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "cells": cells
}

with open('aiso_exp2_iris.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('저장 완료: aiso_exp2_iris.ipynb')
print(f'총 {len(cells)}개 셀 ({sum(1 for c in cells if c["cell_type"]=="code")}개 코드, '
      f'{sum(1 for c in cells if c["cell_type"]=="markdown")}개 마크다운)')
