import json

cells = []

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src, "outputs": [], "execution_count": None}

def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

# ── d001 ──────────────────────────────────────────────────────────────────────
cells.append(md_cell("d001", """\
# Exp 1b: 도메인 피처 확장 + 공정 벤치마크

## 목표
1. **피처 12 → 25개**: 2홉 통계, 교호작용, 시간 버스트 등 추가 → AISO 성능 천장 탐색
2. **PSO(Dom) 비교**: 최적 피처셋에서 PSO도 동일 공간 탐색 → 알고리즘 기여 분리
3. **그래프 특화 baseline**: PageRank 가중 샘플링, 임베딩 기반 밀도 샘플링

## 기준선
AISO(Dom-12, N=12) mean=0.6041 ± 0.0250, 5/5 beats PSO(PCA)"""))

# ── d002 ──────────────────────────────────────────────────────────────────────
cells.append(code_cell("d002", """\
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler as MMS, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE = r'c:\\Users\\kevin\\OneDrive\\Desktop\\AISO\\Elliptic Bitcoin\\elliptic_bitcoin_dataset'

print('데이터 로딩...')
feat_df = pd.read_csv(f'{BASE}/elliptic_txs_features.csv', header=None)
cls_df  = pd.read_csv(f'{BASE}/elliptic_txs_classes.csv')
edge_df = pd.read_csv(f'{BASE}/elliptic_txs_edgelist.csv')
feat_df.columns = ['txId'] + [f'f{i}' for i in range(1, 167)]
feat_df['timestep'] = feat_df['f1'].astype(int)
df = feat_df.merge(cls_df, on='txId')
labeled = df[df['class'] != 'unknown'].copy().reset_index(drop=True)
labeled['y'] = (labeled['class'] == '1').astype(int)
feat_cols = [f'f{i}' for i in range(1, 167)]
X_all = labeled[feat_cols].values.astype(float)
y_all = labeled["y"].values
ts_all = labeled["timestep"].values
N_NODES = len(labeled)
txid_to_idx = {txid: i for i, txid in enumerate(labeled["txId"].values)}

e_s = edge_df['txId1'].map(txid_to_idx)
e_d = edge_df['txId2'].map(txid_to_idx)
valid = e_s.notna() & e_d.notna()
s_arr = e_s[valid].astype(int).values
d_arr = e_d[valid].astype(int).values
mask  = (s_arr < N_NODES) & (d_arr < N_NODES)
ei_np2 = np.stack([s_arr[mask], d_arr[mask]])
# 방향 그래프: ei_np2[0]=src, ei_np2[1]=dst
src_arr, dst_arr = ei_np2[0], ei_np2[1]

SPLIT_T = 34
train_mask = ts_all <= SPLIT_T
test_mask  = ts_all >  SPLIT_T
train_illicit_idx = np.where(train_mask & (y_all == 1))[0]
train_licit_idx   = np.where(train_mask & (y_all == 0))[0]
N_LICIT = 10000; N_SEEN = 1000
X_scaled = StandardScaler().fit_transform(X_all)
print(f'로딩 완료: {N_NODES:,} 노드, illicit_train={len(train_illicit_idx):,}')"""))

# ── d003 ──────────────────────────────────────────────────────────────────────
cells.append(code_cell("d003", """\
# ---- 25차원 도메인 피처 구성 ----
illicit_labels = y_all.astype(float)

# 기본 degree
out_degree = np.bincount(src_arr, minlength=N_NODES).astype(float)
in_degree  = np.bincount(dst_arr, minlength=N_NODES).astype(float)
node_degree = out_degree + in_degree  # total degree

# 1홉 illicit 통계
illicit_neigh_sum = np.zeros(N_NODES)
np.add.at(illicit_neigh_sum, src_arr, illicit_labels[dst_arr])
illicit_neigh_ratio = illicit_neigh_sum / np.maximum(node_degree, 1)

# 1홉 이웃 degree 평균
neigh_degree_sum = np.zeros(N_NODES)
np.add.at(neigh_degree_sum, src_arr, node_degree[dst_arr])
neigh_degree_mean = neigh_degree_sum / np.maximum(node_degree, 1)

# 2홉 illicit 접촉
neigh2_illicit = np.zeros(N_NODES)
np.add.at(neigh2_illicit, src_arr, illicit_neigh_ratio[dst_arr])
neigh2_illicit /= np.maximum(node_degree, 1)

# 2홉 hub_fraud 평균 (hub_fraud 먼저 임시 계산)
hub_fraud_raw = node_degree * illicit_neigh_ratio
neigh2_hub_fraud = np.zeros(N_NODES)
np.add.at(neigh2_hub_fraud, src_arr, hub_fraud_raw[dst_arr])
neigh2_hub_fraud /= np.maximum(node_degree, 1)

# 타임스텝 통계
ts_rate_map = {t: y_all[ts_all==t].mean() for t in np.unique(ts_all)}
node_ts_illicit_rate = np.array([ts_rate_map[t] for t in ts_all])
global_illicit_mean = y_all.mean()
ts_burst = np.abs(node_ts_illicit_rate - global_illicit_mean)
ts_min, ts_max = ts_all.min(), ts_all.max()
ts_norm = (ts_all - ts_min) / max(ts_max - ts_min, 1)

# 고유 타임스텝 수: 노드의 활동 지속성 (Elliptic은 노드당 1 timestep이라 그냥 1)
# local/agg 피처 통계
local_feat = X_scaled[:, 1:94]; agg_feat = X_scaled[:, 94:166]
local_mean = local_feat.mean(1); local_std = local_feat.std(1); local_max = local_feat.max(1)
local_min  = local_feat.min(1)
agg_mean   = agg_feat.mean(1);  agg_std  = agg_feat.std(1);  agg_max  = agg_feat.max(1)
agg_min    = agg_feat.min(1)

hub_fraud = node_degree * illicit_neigh_ratio

# 확장 피처
degree_log       = np.log1p(node_degree)
local_range      = local_max - local_min
agg_range        = agg_max - agg_min
local_agg_diff   = np.abs(local_mean - agg_mean)
illicit_isolation = illicit_neigh_ratio - neigh2_illicit  # 직접 > 2홉: 국소 집중 패턴
degree_x_ts      = node_degree * node_ts_illicit_rate
illicit_x_neigh2 = illicit_neigh_ratio * neigh2_illicit
hub_x_ts         = hub_fraud * node_ts_illicit_rate
in_out_ratio     = in_degree / np.maximum(out_degree, 1)  # 유향 그래프 비율

# 25차원 조립
feat_names_25 = [
    'degree', 'illicit_neigh', 'neigh2_illicit', 'ts_illicit',
    'local_mean', 'local_std', 'local_max',
    'agg_mean', 'agg_std', 'agg_max', 'ts_norm', 'hub_fraud',
    # 확장 13개
    'neigh_degree_mean', 'neigh2_hub_fraud', 'ts_burst',
    'degree_log', 'degree_x_ts', 'illicit_x_neigh2', 'hub_x_ts',
    'local_range', 'agg_range', 'local_agg_diff',
    'illicit_isolation', 'in_degree', 'in_out_ratio',
]

domain_raw_25 = np.stack([
    node_degree, illicit_neigh_ratio, neigh2_illicit, node_ts_illicit_rate,
    local_mean, local_std, local_max,
    agg_mean, agg_std, agg_max, ts_norm, hub_fraud,
    neigh_degree_mean, neigh2_hub_fraud, ts_burst,
    degree_log, degree_x_ts, illicit_x_neigh2, hub_x_ts,
    local_range, agg_range, local_agg_diff,
    illicit_isolation, in_degree, in_out_ratio,
], axis=1)

domain_norm_25 = MMS().fit_transform(domain_raw_25)
X_anom_25 = domain_norm_25[train_illicit_idx]

# 기존 12차원도 보존
domain_raw_12 = domain_raw_25[:, :12]
domain_norm_12 = MMS().fit_transform(domain_raw_12)
X_anom_12 = domain_norm_12[train_illicit_idx]

print(f'피처 25차원: {X_anom_25.shape}')
print(f'피처 12차원: {X_anom_12.shape}')
print(f'피처명: {feat_names_25}')"""))

# ── d004 ──────────────────────────────────────────────────────────────────────
cells.append(code_cell("d004", """\
class GCN(torch.nn.Module):
    def __init__(self, in_ch, hidden=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden); self.conv2 = GCNConv(hidden, hidden)
        self.lin = torch.nn.Linear(hidden, 2); self.drop = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.drop, self.training)
        return self.lin(F.relu(self.conv2(x, edge_index)))

_LOOKUP = np.full(N_NODES, -1, dtype=np.int32)
ei_torch = torch.tensor(ei_np2, dtype=torch.long)

def evaluate_gnn(sampled_anom_idx, seed=42):
    torch.manual_seed(seed)
    licit_idx = np.random.RandomState(seed).choice(train_licit_idx, N_LICIT, replace=False)
    train_idx = np.concatenate([licit_idx, train_illicit_idx[sampled_anom_idx]])
    _LOOKUP[:] = -1
    for loc, glob in enumerate(train_idx): _LOOKUP[glob] = loc
    sub_src = []; sub_dst = []
    for s, d in zip(src_arr, dst_arr):
        if _LOOKUP[s] >= 0 and _LOOKUP[d] >= 0:
            sub_src.append(_LOOKUP[s]); sub_dst.append(_LOOKUP[d])
    if not sub_src: return {'PR-AUC': 0.0}
    ei_sub = torch.tensor([sub_src, sub_dst], dtype=torch.long).to(DEVICE)
    X_tr = torch.tensor(X_scaled[train_idx], dtype=torch.float).to(DEVICE)
    y_tr = torch.tensor(y_all[train_idx], dtype=torch.long).to(DEVICE)
    model = GCN(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    w   = torch.tensor([1.0, 10.0]).to(DEVICE)
    for _ in range(100):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(X_tr, ei_sub), y_tr, weight=w).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        test_idx = np.where(test_mask)[0]
        X_te = torch.tensor(X_scaled[test_idx], dtype=torch.float).to(DEVICE)
        tidx_map = {g: l for l, g in enumerate(test_idx)}
        src_te = []; dst_te = []
        for s, d in zip(src_arr, dst_arr):
            if s in tidx_map and d in tidx_map:
                src_te.append(tidx_map[s]); dst_te.append(tidx_map[d])
        if not src_te: return {'PR-AUC': 0.0}
        ei_te = torch.tensor([src_te, dst_te], dtype=torch.long).to(DEVICE)
        probs = torch.softmax(model(X_te, ei_te), 1)[:, 1].cpu().numpy()
        y_te  = y_all[test_idx]
    return {'PR-AUC': average_precision_score(y_te, probs),
            'F1':     f1_score(y_te, (probs>0.5).astype(int), zero_division=0),
            'AUC':    roc_auc_score(y_te, probs)}

print('GCN 정의 완료')"""))

# ── d005 ──────────────────────────────────────────────────────────────────────
cells.append(code_cell("d005", """\
GNN_N_AG = 20; GNN_N_IT = 80; GNN_ALPHA = 0.2; GNN_BETA = 0.08
GNN_M_LOW = -0.5; GNN_W_REPEL = 2.0
SEEDS_VAL = [0, 7, 42, 77, 123]
BASELINE_12 = 0.6041; BASELINE_STD = 0.0250

def _norm(X):
    mn, mx = X.min(0), X.max(0)
    return (X - mn) / np.where(mx - mn > 1e-8, mx - mn, 1.0)

def run_aiso(X_a, n, seed, n_types=None):
    rng = np.random.RandomState(seed)
    Xn = _norm(X_a); N_a, D = Xn.shape
    if n_types is None: n_types = D
    X = Xn[rng.choice(N_a, GNN_N_AG, replace=True)].copy().astype(float)
    W = rng.dirichlet(np.ones(n_types), GNN_N_AG)
    M = rng.uniform(GNN_M_LOW, GNN_W_REPEL, (n_types, n_types))
    visit = np.zeros(N_a); w_r = GNN_W_REPEL
    for t in range(GNN_N_IT):
        if t % 10 == 0:
            div = np.mean([np.linalg.norm(X[i]-X[j])
                           for i in range(GNN_N_AG) for j in range(i+1, GNN_N_AG)])
            w_r = 1.0 + 3.0 * np.exp(-div / 0.12)
        C = W @ M @ W.T; np.fill_diagonal(C, 0)
        for i in range(GNN_N_AG):
            ci = C[i].copy(); ci[i] = 0
            ta  = np.argsort(ci)[-3:]; tr2 = np.argsort(ci)[:3]
            Fv  = sum(ci[j]*(X[j]-X[i]) for j in ta)
            Fv += w_r * sum(ci[j]*(X[j]-X[i]) for j in tr2)
            Fv /= 6.0
            nn_idx = np.argmin(np.linalg.norm(Xn - np.clip(X[i]+GNN_ALPHA*Fv,0,1), axis=1))
            X[i] = Xn[nn_idx]; visit[nn_idx] += 1.0
            bja = ta[np.argmax(ci[ta])]
            W_new = (1-GNN_BETA)*W[i] + GNN_BETA*W[bja]; W_new /= W_new.sum()
            W[i] = W_new
    probs = visit + 1.0; probs /= probs.sum()
    return rng.choice(N_a, n, replace=True, p=probs)

print('AISO 함수 정의 완료')"""))

# ── d006 ──────────────────────────────────────────────────────────────────────
cells.append(md_cell("d006", """\
## Part 1: 피처 셋 그리드 탐색

| 셋 | 피처 수 | N_TYPES | 내용 |
|----|---------|---------|------|
| Dom-12 | 12 | 12 | 기존 기준선 |
| Dom-25 | 25 | 25 | 전체 확장 |
| Dom-25 N=12 | 25 | 12 | 피처 많되 타입은 12 유지 |
| Dom-18 | 18 | 18 | 중간 (기존 12 + 교호작용 6개) |
| Dom-Top12 | 12 | 12 | 25개 중 상관관계 낮은 12개 선택 |"""))

# ── d007 ──────────────────────────────────────────────────────────────────────
cells.append(code_cell("d007", """\
from sklearn.preprocessing import MinMaxScaler

# Dom-18: 기존 12 + 교호작용/확장 6개
feat_idx_18 = list(range(12)) + [14, 16, 17, 18, 21, 22]  # ts_burst, degree_x_ts, illicit_x_neigh2, hub_x_ts, local_agg_diff, illicit_isolation
X_anom_18 = MMS().fit_transform(domain_raw_25[:, feat_idx_18])[train_illicit_idx]

# Dom-Top12: 25개 중 상호 상관이 가장 낮은 12개 선택 (다양성 극대화)
corr_25 = np.corrcoef(X_anom_25.T)
# 탐욕적 선택: 이미 선택된 피처들과 평균 상관이 가장 낮은 피처를 순서대로 추가
selected = [0]  # degree부터 시작
for _ in range(11):
    best_feat, best_score = -1, np.inf
    for f in range(25):
        if f in selected: continue
        avg_corr = np.mean([abs(corr_25[f, s]) for s in selected])
        if avg_corr < best_score:
            best_score = avg_corr; best_feat = f
    selected.append(best_feat)
print(f'Top12 선택 피처: {[feat_names_25[i] for i in sorted(selected)]}')
X_anom_top12 = MMS().fit_transform(domain_raw_25[:, sorted(selected)])[train_illicit_idx]

feat_configs = [
    ('Dom-12  N=12', X_anom_12,    12),
    ('Dom-18  N=18', X_anom_18,    18),
    ('Dom-25  N=25', X_anom_25,    25),
    ('Dom-25  N=12', X_anom_25,    12),
    ('Dom-Top12 N=12', X_anom_top12, 12),
]

feat_results = {}
print(f'{"피처셋":<18} {"mean":>7} {"std":>7}  vs 기준')
print('-' * 50)
for label, X_a, n_t in feat_configs:
    scores = [evaluate_gnn(run_aiso(X_a, N_SEEN, sd, n_types=n_t))['PR-AUC']
              for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    feat_results[label] = (scores, m, s, X_a, n_t)
    flag = '★' if m > BASELINE_12 else ''
    print(f'  {label:<16} {m:.4f}  {s:.4f}  {m-BASELINE_12:+.4f} {flag}')
print()
print(f'  {"기존 Dom-12 N=12":<16} {BASELINE_12:.4f}  {BASELINE_STD:.4f}  (기준)')"""))

# ── d008 ──────────────────────────────────────────────────────────────────────
cells.append(md_cell("d008", """\
## Part 2: PSO(Dom) — 알고리즘 기여 분리

최적 피처셋에서 PSO도 동일 공간 탐색.
PSO 목적함수: 각 위치에서 k-NN 기반 이상치 밀도 (높을수록 좋음).
방문 빈도 기반 샘플링은 AISO와 동일 방식."""))

# ── d009 ──────────────────────────────────────────────────────────────────────
cells.append(code_cell("d009", """\
def run_pso(X_a, n, seed):
    rng = np.random.RandomState(seed)
    Xn = _norm(X_a); N_a, D = Xn.shape

    # 목적함수: 위치에서 가장 가까운 5개 이상치까지 평균 거리 (음수)
    def density_score(pos_idx):
        dists = np.linalg.norm(Xn - Xn[pos_idx], axis=1)
        dists[pos_idx] = np.inf
        return -np.sort(dists)[:5].mean()

    init_idx = rng.choice(N_a, GNN_N_AG, replace=True)
    X = Xn[init_idx].copy().astype(float)
    V = rng.randn(GNN_N_AG, D) * 0.02
    S = np.array([density_score(i) for i in init_idx])
    pb = X.copy(); ps = S.copy()
    best_i = np.argmax(S)
    gb = X[best_i].copy(); gs = S[best_i]

    visit = np.zeros(N_a)
    for i in init_idx: visit[i] += 1.0

    w_in = 0.729; c1 = 1.494; c2 = 1.494

    for t in range(GNN_N_IT):
        r1 = rng.rand(GNN_N_AG, D); r2 = rng.rand(GNN_N_AG, D)
        V = w_in*V + c1*r1*(pb-X) + c2*r2*(gb-X)
        X_new = np.clip(X + V, 0, 1)
        # 가장 가까운 이상치 노드로 스냅
        for i in range(GNN_N_AG):
            nn_idx = np.argmin(np.linalg.norm(Xn - X_new[i], axis=1))
            X[i] = Xn[nn_idx]; visit[nn_idx] += 1.0
            s_i = density_score(nn_idx)
            if s_i > ps[i]:
                pb[i] = X[i]; ps[i] = s_i
            if s_i > gs:
                gb = X[i].copy(); gs = s_i

    probs = visit + 1.0; probs /= probs.sum()
    return rng.choice(N_a, n, replace=True, p=probs)

# 기존 PSO(PCA) 기준선
BASELINE_PSO_PCA = 0.5621

# 최적 피처셋으로 PSO 실행
best_feat_label = max(feat_results, key=lambda k: feat_results[k][1])
_, best_m, _, X_best, n_t_best = feat_results[best_feat_label]
print(f'최적 피처셋: {best_feat_label}  mean={best_m:.4f}')
print()

pso_configs = [
    ('PSO(Dom-12)',    X_anom_12),
    ('PSO(Dom-25)',    X_anom_25),
    (f'PSO({best_feat_label.strip()})', X_best),
]

pso_results = {}
print(f'{"방법":<22} {"mean":>7} {"std":>7}  vs PSO(PCA) | vs AISO기준')
print('-' * 65)
for label, X_a in pso_configs:
    scores = [evaluate_gnn(run_pso(X_a, N_SEEN, sd))['PR-AUC'] for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    pso_results[label] = scores
    flag = '★' if m > BASELINE_12 else ''
    print(f'  {label:<20} {m:.4f}  {s:.4f}  {m-BASELINE_PSO_PCA:+.4f}     | {m-BASELINE_12:+.4f} {flag}')
print()
print(f'  {"PSO(PCA) 기존":<20} {BASELINE_PSO_PCA:.4f}  0.0732  (기준)')
print(f'  {"AISO(Dom-12)":<20} {BASELINE_12:.4f}  {BASELINE_STD:.4f}  (AISO 기준)')"""))

# ── d010 ──────────────────────────────────────────────────────────────────────
cells.append(md_cell("d010", """\
## Part 3: 그래프 특화 Baseline

### PageRank 가중 샘플링
이상치 노드 서브그래프 내 근사 PageRank 계산 → 높은 PR 노드 위주 샘플링.
"중심적인" 이상치 노드가 GNN 학습에 더 유익하다는 가설.

### 임베딩 밀도 샘플링
GCN 인코더로 노드 임베딩 생성 → 임베딩 공간에서 이상치 클러스터 밀도 계산
→ 밀도 비례 샘플링 (희귀 패턴 보존)."""))

# ── d011 ──────────────────────────────────────────────────────────────────────
cells.append(code_cell("d011", """\
# PageRank 가중 샘플링
def pagerank_sampling(n, seed, n_iter=30, damping=0.85):
    rng = np.random.RandomState(seed)
    # 이상치 노드 서브그래프만
    anom_set = set(train_illicit_idx.tolist())
    anom_idx_map = {g: l for l, g in enumerate(train_illicit_idx)}
    N_a = len(train_illicit_idx)

    # 이상치-이상치 간 엣지만 추출
    sub_src_list = []; sub_dst_list = []
    for s, d in zip(src_arr, dst_arr):
        if s in anom_set and d in anom_set:
            sub_src_list.append(anom_idx_map[s])
            sub_dst_list.append(anom_idx_map[d])

    # PageRank (반복법)
    pr = np.ones(N_a) / N_a
    out_deg = np.bincount(sub_src_list, minlength=N_a).astype(float)
    out_deg = np.maximum(out_deg, 1)

    for _ in range(n_iter):
        new_pr = np.ones(N_a) * (1 - damping) / N_a
        for s, d in zip(sub_src_list, sub_dst_list):
            new_pr[d] += damping * pr[s] / out_deg[s]
        pr = new_pr
        pr /= pr.sum()

    # PR 비례 샘플링
    probs = pr + 1e-6; probs /= probs.sum()
    return rng.choice(N_a, n, replace=True, p=probs)


# 임베딩 밀도 샘플링
def embedding_density_sampling(n, seed, hidden=32, n_iter=50):
    rng_np = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # 간단한 GCN 인코더로 이상치 임베딩 추출
    licit_idx = rng_np.choice(train_licit_idx, 3000, replace=False)
    sub_idx = np.concatenate([licit_idx, train_illicit_idx])
    sub_map = {g: l for l, g in enumerate(sub_idx)}

    sub_s = []; sub_d = []
    for s, d in zip(src_arr, dst_arr):
        if s in sub_map and d in sub_map:
            sub_s.append(sub_map[s]); sub_d.append(sub_map[d])

    if not sub_s:
        probs = np.ones(len(train_illicit_idx)) / len(train_illicit_idx)
        return rng_np.choice(len(train_illicit_idx), n, replace=True, p=probs)

    ei_sub = torch.tensor([sub_s, sub_d], dtype=torch.long).to(DEVICE)
    X_sub  = torch.tensor(X_scaled[sub_idx], dtype=torch.float).to(DEVICE)
    y_sub  = torch.tensor(y_all[sub_idx], dtype=torch.long).to(DEVICE)

    # 인코더 학습
    enc = GCN(X_sub.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    w   = torch.tensor([1.0, 10.0]).to(DEVICE)
    for _ in range(n_iter):
        enc.train(); opt.zero_grad()
        F.cross_entropy(enc(X_sub, ei_sub), y_sub, weight=w).backward(); opt.step()

    enc.eval()
    with torch.no_grad():
        emb = F.relu(enc.conv2(F.relu(enc.conv1(X_sub, ei_sub)), ei_sub)).cpu().numpy()

    # 이상치 임베딩만
    anom_local_idx = [sub_map[g] for g in train_illicit_idx if g in sub_map]
    if not anom_local_idx:
        probs = np.ones(len(train_illicit_idx)) / len(train_illicit_idx)
        return rng_np.choice(len(train_illicit_idx), n, replace=True, p=probs)

    anom_emb = emb[anom_local_idx]

    # k-NN 역밀도: 희귀한 임베딩 패턴에 더 높은 가중치 (다양성 확보)
    k = min(10, len(anom_emb)-1)
    dists_all = np.linalg.norm(anom_emb[:, None] - anom_emb[None, :], axis=2)
    np.fill_diagonal(dists_all, np.inf)
    knn_dist = np.sort(dists_all, axis=1)[:, :k].mean(axis=1)

    # 밀도 역수 가중치 (희귀 노드 우대)
    probs = knn_dist + 1e-6; probs /= probs.sum()
    # anom_local_idx가 train_illicit_idx의 부분집합일 수 있으므로 전체에 매핑
    full_probs = np.ones(len(train_illicit_idx)) * 1e-8
    for loc, emb_i in enumerate(range(len(anom_local_idx))):
        full_probs[loc] = probs[emb_i]
    full_probs /= full_probs.sum()
    return rng_np.choice(len(train_illicit_idx), n, replace=True, p=full_probs)


graph_results = {}
print('그래프 특화 baseline 실행...')
for label, fn in [('PageRank 샘플링', pagerank_sampling),
                   ('임베딩 역밀도', embedding_density_sampling)]:
    scores = [evaluate_gnn(fn(N_SEEN, sd))['PR-AUC'] for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    graph_results[label] = scores
    flag = '★' if m > BASELINE_12 else ''
    print(f'  {label:<20} {m:.4f}  {s:.4f}  {m-BASELINE_12:+.4f} {flag}')"""))

# ── d012 ──────────────────────────────────────────────────────────────────────
cells.append(code_cell("d012", """\
# ---- 최종 종합 비교 ----
all_methods = {}
all_methods['AISO(Dom-12) 기준'] = ([BASELINE_12]*5, BASELINE_12, BASELINE_STD)
for label, (scores, m, s, _, _) in feat_results.items():
    all_methods[f'AISO({label.strip()})'] = (scores, m, s)
for label, scores in pso_results.items():
    all_methods[label] = (scores, np.mean(scores), np.std(scores))
for label, scores in graph_results.items():
    all_methods[label] = (scores, np.mean(scores), np.std(scores))
all_methods['PSO(PCA) 기존'] = ([BASELINE_PSO_PCA]*5, BASELINE_PSO_PCA, 0.0732)

sorted_methods = sorted(all_methods.items(), key=lambda x: x[1][1], reverse=True)

print('=' * 68)
print('최종 종합 비교')
print('=' * 68)
print(f'{"방법":<30} {"mean":>7} {"std":>7}  vs AISO기준')
print('-' * 68)
for label, (scores, m, s) in sorted_methods:
    diff = m - BASELINE_12
    flag = '★' if m > BASELINE_12 and '기준' not in label else ''
    print(f'  {label:<28} {m:.4f}  {s:.4f}  {diff:+.4f} {flag}')
print('=' * 68)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# 왼쪽: 피처셋별 AISO 성능
ax = axes[0]
feat_labels = [k.replace('AISO(','').replace(')','') for k in all_methods if 'AISO' in k]
feat_means  = [all_methods[k][1] for k in all_methods if 'AISO' in k]
feat_stds   = [all_methods[k][2] for k in all_methods if 'AISO' in k]
clrs = ['#C44E52' if '기준' in l else '#4C72B0' for l in feat_labels]
bars = ax.bar(range(len(feat_labels)), feat_means, yerr=feat_stds,
              capsize=5, color=clrs, alpha=0.85, edgecolor='white')
ax.axhline(BASELINE_12, color='#C44E52', ls='--', lw=1.5)
ax.set_xticks(range(len(feat_labels)))
ax.set_xticklabels(feat_labels, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('PR-AUC'); ax.set_title('피처셋별 AISO 성능', fontweight='bold')
ax.grid(alpha=0.3, axis='y')
for bar, m in zip(bars, feat_means):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.003, f'{m:.4f}',
            ha='center', va='bottom', fontsize=8)

# 오른쪽: 전체 방법 비교 (상위 12)
ax = axes[1]
top12 = sorted_methods[:12]
lbls2 = [l for l,_ in top12]
ms2   = [v[1] for _,v in top12]
ss2   = [v[2] for _,v in top12]
clrs2 = ['#C44E52' if '기준' in l else
         '#4C72B0' if 'AISO' in l else
         '#2ca02c' if 'PSO' in l else
         'orange' for l in lbls2]
bars2 = ax.bar(range(len(lbls2)), ms2, yerr=ss2,
               capsize=5, color=clrs2, alpha=0.85, edgecolor='white')
ax.axhline(BASELINE_12, color='#C44E52', ls='--', lw=1.5, label=f'AISO기준={BASELINE_12:.4f}')
ax.set_xticks(range(len(lbls2)))
ax.set_xticklabels(lbls2, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('PR-AUC'); ax.set_title('전체 방법 비교 (상위 12)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
for bar, m in zip(bars2, ms2):
    ax.text(bar.get_x()+bar.get_width()/2, m+0.003, f'{m:.4f}',
            ha='center', va='bottom', fontsize=7)

plt.suptitle('Exp 1b: 피처 확장 + 공정 벤치마크', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('exp1b_final.png', bbox_inches='tight', dpi=120)
plt.show()"""))

# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.12.0"}},
    "cells": cells
}

with open('aiso_exp1b_features.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'aiso_exp1b_features.ipynb 생성 완료 ({len(cells)} 셀)')
