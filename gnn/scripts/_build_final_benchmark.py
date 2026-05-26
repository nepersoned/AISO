"""
Build aiso_final_benchmark.ipynb
- Exact showdown evaluate_gnn protocol (transductive, bidirectional edges)
- Section A: Baseline verification
- Section B: Exp 3 - Type mechanism experiments
- Section C: Exp 1b - Feature expansion experiments
- Section D: Final comparison table
"""

import json, ast, sys

NOTEBOOK_PATH = 'aiso_final_benchmark.ipynb'

def check(code, name):
    try:
        ast.parse(code)
        print(f'{name}: syntax OK')
    except SyntaxError as e:
        print(f'{name} ERROR line {e.lineno}: {e.msg}')
        lines = code.split('\n')
        for i, l in enumerate(lines[max(0, e.lineno-3):e.lineno+2], max(1, e.lineno-2)):
            print(f'  {i}: {repr(l)}')
        sys.exit(1)

def md(text):
    return {'cell_type': 'markdown', 'id': None, 'metadata': {}, 'source': text}

def code(cid, src):
    check(src, cid)
    return {'cell_type': 'code', 'id': cid, 'metadata': {},
            'source': src, 'outputs': [], 'execution_count': None}

cells = []

# ── Cell f00: imports ─────────────────────────────────────────────────────
cells.append(code('f00', r'''import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch-geometric', '-q'], check=False)

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import copy
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
print(f'Device: {DEVICE}  |  SEED={SEED}')
'''))

# ── Cell f01: data loading (exact showdown) ───────────────────────────────
cells.append(code('f01', r'''BASE = '/content/drive/MyDrive/AISO'

# Exact showdown cell02: txId + f1(=timestep) + f2..f166 = 167 columns
feat_df  = pd.read_csv(f'{BASE}/elliptic_txs_features.csv', header=None)
edge_df  = pd.read_csv(f'{BASE}/elliptic_txs_edgelist.csv')
class_df = pd.read_csv(f'{BASE}/elliptic_txs_classes.csv')

feat_df.columns = ['txId'] + [f'f{i}' for i in range(1, 167)]
feat_df['timestep'] = feat_df['f1'].astype(int)

df = feat_df.merge(class_df, on='txId')
print(f'Nodes: {len(df):,}  Edges: {len(edge_df):,}  Features: 166')
print(f'class: {df["class"].value_counts().to_dict()}')
'''))

# ── Cell f02: graph setup + fixed sel_n (exact showdown) ─────────────────
cells.append(code('f02', r'''labeled = df[df['class'] != 'unknown'].copy().reset_index(drop=True)
labeled['y'] = (labeled['class'] == '1').astype(int)

# Exact showdown cell03: feat_cols includes f1(timestep), 166 features total
feat_cols = [f'f{i}' for i in range(1, 167)]
X_all = labeled[feat_cols].values.astype(float)
y_all = labeled['y'].values
ts_all = labeled['timestep'].values
N_NODES = len(labeled)

print(f'Labeled: {N_NODES:,}  (illicit={y_all.sum():,}, licit={(y_all==0).sum():,})')

txid_to_idx = {txid: i for i, txid in enumerate(labeled['txId'].values)}
s_arr = edge_df.iloc[:,0].values
d_arr = edge_df.iloc[:,1].values
s_map = np.array([txid_to_idx.get(t, -1) for t in s_arr])
d_map = np.array([txid_to_idx.get(t, -1) for t in d_arr])
valid = (s_map >= 0) & (d_map >= 0)
s_v, d_v = s_map[valid], d_map[valid]
edge_index = torch.tensor(
    [np.concatenate([s_v, d_v]), np.concatenate([d_v, s_v])],
    dtype=torch.long
)
print(f'Valid edges: {valid.sum():,}  (bidirectional: {edge_index.shape[1]:,})')

train_mask_all = ts_all <= 34
test_mask      = ts_all > 34
train_norm_idx = np.where(train_mask_all & (y_all==0))[0]
train_anom_idx = np.where(train_mask_all & (y_all==1))[0]

N_NORMAL = min(10000, len(train_norm_idx))
N_SEEN   = min(1000,  len(train_anom_idx))
rng = np.random.RandomState(SEED)
sel_n = rng.choice(train_norm_idx, N_NORMAL, replace=False)  # fixed once

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

pca = PCA(n_components=30, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
X_anom_pca = X_pca[train_anom_idx]

print(f'Train licit pool: {N_NORMAL:,} | illicit pool: {len(train_anom_idx):,} -> N_SEEN: {N_SEEN:,}')
print(f'Test: {test_mask.sum():,}  illicit={y_all[test_mask].sum():,}')
print(f'sel_n fixed (SEED=42): {len(sel_n):,} nodes')
'''))

# ── Cell f03: GCN + evaluate_gnn (exact showdown transductive protocol) ──
cells.append(code('f03', r'''ei_np = edge_index.numpy()
_LOOKUP = np.full(N_NODES, -1, dtype=np.int32)

class GCN(torch.nn.Module):
    def __init__(self, in_ch, hidden=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = torch.nn.Linear(hidden, 2)
        self.drop  = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.lin(x)

def evaluate_gnn(sel_pool_idx, label=''):
    """
    Transductive protocol (exact showdown):
      subgraph = sel_n (fixed licit) + selected illicit + ALL test nodes
      bidirectional edges, dynamic class weight, lr=0.01, 200 epochs,
      patience=20, best model, torch.manual_seed(SEED=42)
    """
    uniq_anom = np.unique(sel_pool_idx)
    sel_anom_global = train_anom_idx[uniq_anom]
    test_global     = np.where(test_mask)[0]

    sub_nodes = np.unique(np.concatenate([sel_n, sel_anom_global, test_global]))
    n_sub = len(sub_nodes)

    _LOOKUP[:] = -1
    _LOOKUP[sub_nodes] = np.arange(n_sub)
    src_loc = _LOOKUP[ei_np[0]]
    dst_loc = _LOOKUP[ei_np[1]]
    ok = (src_loc >= 0) & (dst_loc >= 0)
    sub_ei = torch.tensor([src_loc[ok], dst_loc[ok]], dtype=torch.long).to(DEVICE)

    sub_X = torch.from_numpy(X_scaled[sub_nodes]).float().to(DEVICE)
    sub_y = torch.from_numpy(y_all[sub_nodes]).long().to(DEVICE)

    train_set = set(np.concatenate([sel_n, sel_anom_global]).tolist())
    test_set  = set(test_global.tolist())
    tr_mask = torch.tensor([g in train_set for g in sub_nodes], dtype=torch.bool).to(DEVICE)
    te_mask = torch.tensor([g in test_set  for g in sub_nodes], dtype=torch.bool).to(DEVICE)

    n0 = int((y_all[sub_nodes][tr_mask.cpu().numpy()]==0).sum())
    n1 = int((y_all[sub_nodes][tr_mask.cpu().numpy()]==1).sum())
    cw = torch.tensor([1.0, n0/max(n1,1)], dtype=torch.float).to(DEVICE)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    model = GCN(X_scaled.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_loss, best_state, patience = float('inf'), None, 0
    for ep in range(200):
        model.train(); opt.zero_grad()
        out  = model(sub_X, sub_ei)
        loss = F.cross_entropy(out[tr_mask], sub_y[tr_mask], weight=cw)
        loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item(); best_state = copy.deepcopy(model.state_dict()); patience = 0
        else:
            patience += 1
        if patience >= 20: break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        prob = F.softmax(model(sub_X, sub_ei), dim=1)[:,1].cpu().numpy()
    pred = (prob >= 0.5).astype(int)

    te_cpu = te_mask.cpu().numpy()
    res = {
        'PR-AUC': average_precision_score(y_all[sub_nodes][te_cpu], prob[te_cpu]),
        'F1':     f1_score(y_all[sub_nodes][te_cpu], pred[te_cpu], zero_division=0),
        'AUC':    roc_auc_score(y_all[sub_nodes][te_cpu], prob[te_cpu]),
        'n_edges': int(ok.sum()),
    }
    if label:
        print(f'  {label:<28} PR-AUC={res["PR-AUC"]:.4f}  F1={res["F1"]:.4f}  AUC={res["AUC"]:.4f}')
    return res

print('GCN + evaluate_gnn ready (exact showdown transductive protocol)')
'''))

# ── Cell f04: domain features (exact showdown) ────────────────────────────
cells.append(code('f04', r'''ei_np2 = edge_index.numpy()

node_degree = np.bincount(ei_np2[0], minlength=N_NODES).astype(float)  # out-degree only

illicit_neigh_sum = np.zeros(N_NODES)
neigh_count       = np.zeros(N_NODES)
for src, dst in zip(ei_np2[0], ei_np2[1]):
    illicit_neigh_sum[src] += y_all[dst]
    neigh_count[src]       += 1
illicit_neigh_ratio = illicit_neigh_sum / np.maximum(neigh_count, 1)

neigh2_illicit = np.zeros(N_NODES)
for src, dst in zip(ei_np2[0], ei_np2[1]):
    neigh2_illicit[src] += illicit_neigh_ratio[dst]
neigh2_illicit /= np.maximum(neigh_count, 1)

ts_illicit_rate_map = {}
for ts in np.unique(ts_all):
    m = ts_all == ts
    ts_illicit_rate_map[ts] = float(y_all[m].mean()) if m.sum() > 0 else 0.0
node_ts_illicit_rate = np.array([ts_illicit_rate_map[t] for t in ts_all])

local_feat = X_scaled[:, 1:94]
local_mean = local_feat.mean(axis=1); local_std = local_feat.std(axis=1); local_max = local_feat.max(axis=1)

agg_feat = X_scaled[:, 94:166]  # f95-f166 = 72 aggregate features
agg_mean = agg_feat.mean(axis=1); agg_std = agg_feat.std(axis=1); agg_max = agg_feat.max(axis=1)

ts_norm = (ts_all - ts_all.min()) / (ts_all.max() - ts_all.min() + 1e-8)
hub_fraud = node_degree * illicit_neigh_ratio

domain_feat_raw = np.stack([
    node_degree, illicit_neigh_ratio, neigh2_illicit, node_ts_illicit_rate,
    local_mean, local_std, local_max,
    agg_mean, agg_std, agg_max,
    ts_norm, hub_fraud,
], axis=1)

domain_feat_norm = MinMaxScaler().fit_transform(domain_feat_raw)
X_anom_domain    = domain_feat_norm[train_anom_idx]
print(f'Dom-12 features: {domain_feat_raw.shape}  illicit pool: {X_anom_domain.shape}')
'''))

# ── Cell f05: sampler functions ────────────────────────────────────────────
cells.append(code('f05', r'''N_AG    = 20
N_IT    = 80
ALPHA   = 0.2
BETA    = 0.08
W_REPEL = 2.0
M_LOW   = -0.5
SEEDS_VAL = [0, 7, 42, 77, 123]

def _norm(X):
    mn, mx = X.min(0), X.max(0)
    return (X - mn) / np.where(mx - mn > 1e-8, mx - mn, 1.0)

def run_random(X_a, n, seed):
    return np.random.RandomState(seed).choice(len(X_a), n, replace=True)

def run_topdensity(X_a, n, seed, k=5):
    nn = NearestNeighbors(n_neighbors=min(k+1, len(X_a))).fit(X_a)
    d, _ = nn.kneighbors(X_a)
    density = 1.0 / (d[:,1:].mean(1) + 1e-8)
    probs = density / density.sum()
    return np.random.RandomState(seed).choice(len(X_a), n, replace=True, p=probs)

def run_pso(X_a, n, seed):
    rng2 = np.random.RandomState(seed)
    Xn = _norm(X_a); N_a, D = Xn.shape
    X = Xn[rng2.choice(N_a, N_AG, replace=True)].copy().astype(float)
    V = np.zeros_like(X); pX = X.copy()
    pS = np.array([-np.min(np.linalg.norm(Xn-X[i], axis=1)) for i in range(N_AG)])
    gi = np.argmax(pS); gX = X[gi].copy(); visit = np.zeros(N_a)
    for _ in range(N_IT):
        r1, r2 = rng2.rand(N_AG, D), rng2.rand(N_AG, D)
        V = 0.729*V + 1.494*r1*(pX-X) + 1.494*r2*(gX-X)
        X = np.clip(X + ALPHA*V, 0, 1)
        for i in range(N_AG):
            nn = np.argmin(np.linalg.norm(Xn - X[i], axis=1))
            X[i] = Xn[nn]; visit[nn] += 1
            sc = -np.min(np.linalg.norm(Xn - X[i], axis=1))
            if sc > pS[i]: pX[i] = X[i].copy(); pS[i] = sc
        gi = np.argmax(pS); gX = pX[gi].copy()
    probs = visit + 1.0; probs /= probs.sum()
    return rng2.choice(N_a, n, replace=True, p=probs)

def run_aiso_nt(X_a, n, seed, n_types):
    rng2 = np.random.RandomState(seed)
    Xn = _norm(X_a); N_a, D = Xn.shape
    X = Xn[rng2.choice(N_a, N_AG, replace=True)].copy().astype(float)
    W = rng2.dirichlet(np.ones(n_types), N_AG)
    M = rng2.uniform(M_LOW, W_REPEL, (n_types, n_types))
    visit = np.zeros(N_a); w_r = W_REPEL
    for t in range(N_IT):
        if t % 10 == 0:
            div = np.mean([np.linalg.norm(X[i]-X[j])
                           for i in range(N_AG) for j in range(i+1, N_AG)])
            w_r = 1.0 + 3.0 * np.exp(-div / 0.12)
        C = W @ M @ W.T; np.fill_diagonal(C, 0)
        for i in range(N_AG):
            ci = C[i].copy(); ci[i] = 0
            ta = np.argsort(ci)[-3:]; tr2 = np.argsort(ci)[:3]
            Fv  = sum(ci[j] * (X[j]-X[i]) for j in ta)
            Fv += w_r * sum(ci[j] * (X[j]-X[i]) for j in tr2)
            Fv /= 6.0
            nn = np.argmin(np.linalg.norm(Xn - np.clip(X[i]+ALPHA*Fv, 0, 1), axis=1))
            X[i] = Xn[nn]; visit[nn] += 1.0
            bja = ta[np.argmax(ci[ta])]
            W[i] = (1-BETA)*W[i] + BETA*W[bja]; W[i] /= W[i].sum()
    probs = visit + 1.0; probs /= probs.sum()
    return rng2.choice(N_a, n, replace=True, p=probs)

print('Samplers ready: run_random, run_pso, run_aiso_nt, run_topdensity')
print(f'SEEDS_VAL = {SEEDS_VAL}')
'''))

# ── Cell f06: markdown - Section A ────────────────────────────────────────
cells.append(md('''## Section A — Known Baseline Results (from showdown) + Protocol Sanity Check

5-seed results loaded directly from `elliptic_gnn_showdown.ipynb` cells 15-17
(re-running all ~40 GCN trainings is unnecessary — same protocol, same data).

**Sanity check**: run AISO(Dom-12, N=12) with sampler seed=42 and confirm
PR-AUC ≈ 0.5640 (expected from showdown). Tolerance ±0.005.
'''))

# ── Cell f07: Section A — hardcoded showdown results + sanity check ───────
cells.append(code('f07', r'''# Known 5-seed results from elliptic_gnn_showdown.ipynb cells 15-17
# SEEDS_VAL = [0, 7, 42, 77, 123]
SHOWDOWN_RESULTS = {
    'AISO(Dom-12, N=12)': [0.6074, 0.6329, 0.5640, 0.5902, 0.6263],
    'AISO(Dom-12, N=14)': [0.5968, 0.5303, 0.6153, 0.5110, 0.5253],
    'AISO(PCA, N=14)':    [0.6297, 0.6465, 0.6378, 0.5230, 0.5096],
    'AISO(PCA, N=8)':     [0.3883, 0.5376, 0.6002, 0.5694, 0.5579],
    'PSO(PCA)':           [0.6670, 0.5079, 0.6030, 0.4570, 0.5757],
    'Top-density(PCA)':   [0.6293, 0.5500, 0.5701, 0.5965, 0.5420],
    'Random':             [0.5080, 0.4985, 0.5538, 0.6555, 0.4913],
    '원본(불균형)':          [0.5545, 0.6190, 0.5752, 0.4685, 0.6420],
}

baseline_results = {}
print('=== Section A: Known Baseline Results (showdown) ===')
print(f'  Seeds: {SEEDS_VAL}')
print()
print(f'  {"Method":<24} {"mean":>7} {"std":>7}  per-seed')
print('-' * 75)
for name, vals in sorted(SHOWDOWN_RESULTS.items(),
                          key=lambda x: np.mean(x[1]), reverse=True):
    m, s = np.mean(vals), np.std(vals)
    baseline_results[name] = {'mean': m, 'std': s, 'vals': vals}
    print(f'  {name:<24} {m:.4f} {s:.4f}  {[round(v,4) for v in vals]}')

KNOWN_BASELINE = baseline_results['AISO(Dom-12, N=12)']['mean']
print()
print(f'  Reference AISO(Dom-12 N=12) mean = {KNOWN_BASELINE:.4f}')
print()

# ── Protocol sanity check: one re-run to confirm evaluate_gnn matches ────
print('--- Protocol sanity check: AISO(Dom-12 N=12) SEED=42 ---')
EXPECTED = 0.5640
TOL = 0.005
idx_check = run_aiso_nt(X_anom_domain, N_SEEN, 42, 12)
got = evaluate_gnn(idx_check)['PR-AUC']
delta = abs(got - EXPECTED)
status = 'PASS' if delta <= TOL else 'FAIL'
print(f'  Expected: {EXPECTED:.4f}  Got: {got:.4f}  Delta: {delta:.4f}  [{status}]')
if status == 'FAIL':
    print(f'  WARNING: protocol mismatch — check evaluate_gnn vs showdown cell04')
else:
    print(f'  Protocol confirmed: evaluate_gnn matches showdown transductive protocol')
'''))

# ── Cell f08: markdown - Section B ────────────────────────────────────────
cells.append(md('''## Section B — Exp 3: Type Mechanism Experiments

Investigates whether modifying the type-update rule in AISO improves diversity
and coverage on the Elliptic GNN subgraph sampling task.

- **B1**: Gamma diversity injection (Dirichlet noise on W)
- **B2**: Anti-assimilation (push W away from partner's type)
- **B3**: Structured M initialization (one-hot W + repulsion matrix)
'''))

# ── Cell f09: B1 - gamma diversity injection ───────────────────────────────
cells.append(code('f09', r'''def run_aiso_gamma(X_a, n, seed, n_types=12, gamma=0.0, beta=0.08):
    rng2 = np.random.RandomState(seed)
    Xn = _norm(X_a); N_a, D = Xn.shape
    X = Xn[rng2.choice(N_a, N_AG, replace=True)].copy().astype(float)
    W = rng2.dirichlet(np.ones(n_types), N_AG)
    M = rng2.uniform(M_LOW, W_REPEL, (n_types, n_types))
    visit = np.zeros(N_a); w_r = W_REPEL
    for t in range(N_IT):
        if t % 10 == 0:
            div = np.mean([np.linalg.norm(X[i]-X[j])
                           for i in range(N_AG) for j in range(i+1, N_AG)])
            w_r = 1.0 + 3.0 * np.exp(-div / 0.12)
        C = W @ M @ W.T; np.fill_diagonal(C, 0)
        for i in range(N_AG):
            ci = C[i].copy(); ci[i] = 0
            ta = np.argsort(ci)[-3:]; tr2 = np.argsort(ci)[:3]
            Fv  = sum(ci[j] * (X[j]-X[i]) for j in ta)
            Fv += w_r * sum(ci[j] * (X[j]-X[i]) for j in tr2)
            Fv /= 6.0
            nn = np.argmin(np.linalg.norm(Xn - np.clip(X[i]+ALPHA*Fv, 0, 1), axis=1))
            X[i] = Xn[nn]; visit[nn] += 1.0
            bja = ta[np.argmax(ci[ta])]
            W_new = (1 - beta) * W[i] + beta * W[bja]
            if gamma > 0:
                noise = rng2.dirichlet(np.ones(n_types)) * gamma
                W_new = (1 - gamma) * W_new + noise
            W_new /= W_new.sum()
            W[i] = W_new
    probs = visit + 1.0; probs /= probs.sum()
    return rng2.choice(N_a, n, replace=True, p=probs)

gamma_configs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

print('=== Section B1: Gamma Diversity Injection ===')
print(f'  {"gamma":<10} {"mean":>7} {"std":>7}  vs baseline')
print('-' * 50)
b1_results = {}
for g in gamma_configs:
    scores = [evaluate_gnn(run_aiso_gamma(X_anom_domain, N_SEEN, sd, gamma=g))['PR-AUC']
              for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    b1_results[f'gamma={g:.2f}'] = {'mean': m, 'std': s, 'vals': scores}
    flag = ' ★' if m > KNOWN_BASELINE else ''
    print(f'  gamma={g:.2f}     {m:.4f} {s:.4f}  {m-KNOWN_BASELINE:+.4f}{flag}')
print()
print(f'  Baseline AISO(Dom-12 N=12): {KNOWN_BASELINE:.4f}')
'''))

# ── Cell f10: B2 - anti-assimilation ──────────────────────────────────────
cells.append(code('f10', r'''def run_aiso_anti(X_a, n, seed, n_types=12, beta=0.08):
    rng2 = np.random.RandomState(seed)
    Xn = _norm(X_a); N_a, D = Xn.shape
    X = Xn[rng2.choice(N_a, N_AG, replace=True)].copy().astype(float)
    W = rng2.dirichlet(np.ones(n_types), N_AG)
    M = rng2.uniform(M_LOW, W_REPEL, (n_types, n_types))
    visit = np.zeros(N_a); w_r = W_REPEL
    for t in range(N_IT):
        if t % 10 == 0:
            div = np.mean([np.linalg.norm(X[i]-X[j])
                           for i in range(N_AG) for j in range(i+1, N_AG)])
            w_r = 1.0 + 3.0 * np.exp(-div / 0.12)
        C = W @ M @ W.T; np.fill_diagonal(C, 0)
        for i in range(N_AG):
            ci = C[i].copy(); ci[i] = 0
            ta = np.argsort(ci)[-3:]; tr2 = np.argsort(ci)[:3]
            Fv  = sum(ci[j] * (X[j]-X[i]) for j in ta)
            Fv += w_r * sum(ci[j] * (X[j]-X[i]) for j in tr2)
            Fv /= 6.0
            nn = np.argmin(np.linalg.norm(Xn - np.clip(X[i]+ALPHA*Fv, 0, 1), axis=1))
            X[i] = Xn[nn]; visit[nn] += 1.0
            bja = ta[np.argmax(ci[ta])]
            W_new = (1 + beta) * W[i] - beta * W[bja]  # anti-assimilation
            W_new = np.clip(W_new, 0, None)
            s_w = W_new.sum()
            if s_w < 1e-10:
                W_new = np.ones(n_types) / n_types
            else:
                W_new /= s_w
            W[i] = W_new
    probs = visit + 1.0; probs /= probs.sum()
    return rng2.choice(N_a, n, replace=True, p=probs)

anti_betas = [0.02, 0.05, 0.08, 0.12, 0.20]

print('=== Section B2: Anti-Assimilation ===')
print(f'  {"beta":<10} {"mean":>7} {"std":>7}  vs baseline')
print('-' * 50)
b2_results = {}
for bt in anti_betas:
    scores = [evaluate_gnn(run_aiso_anti(X_anom_domain, N_SEEN, sd, beta=bt))['PR-AUC']
              for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    b2_results[f'anti beta={bt:.2f}'] = {'mean': m, 'std': s, 'vals': scores}
    flag = ' ★' if m > KNOWN_BASELINE else ''
    print(f'  beta={bt:.2f}      {m:.4f} {s:.4f}  {m-KNOWN_BASELINE:+.4f}{flag}')
print()
print(f'  Baseline AISO(Dom-12 N=12): {KNOWN_BASELINE:.4f}')
'''))

# ── Cell f11: B3 - structured M initialization ────────────────────────────
cells.append(code('f11', r'''def run_aiso_structured(X_a, n, seed, n_types=12, beta_slow=0.02,
                         m_diag=-2.0, m_offdiag=0.3):
    rng2 = np.random.RandomState(seed)
    Xn = _norm(X_a); N_a, D = Xn.shape

    W = np.zeros((N_AG, n_types))
    for i in range(N_AG):
        W[i, i % n_types] = 1.0
    noise = rng2.dirichlet(np.ones(n_types), N_AG) * 0.05
    W = W + noise; W /= W.sum(1, keepdims=True)

    M = np.full((n_types, n_types), m_offdiag)
    np.fill_diagonal(M, m_diag)

    X = Xn[rng2.choice(N_a, N_AG, replace=True)].copy().astype(float)
    visit = np.zeros(N_a); w_r = W_REPEL

    for t in range(N_IT):
        if t % 10 == 0:
            div = np.mean([np.linalg.norm(X[i]-X[j])
                           for i in range(N_AG) for j in range(i+1, N_AG)])
            w_r = 1.0 + 3.0 * np.exp(-div / 0.12)
        C = W @ M @ W.T; np.fill_diagonal(C, 0)
        for i in range(N_AG):
            ci = C[i].copy(); ci[i] = 0
            ta = np.argsort(ci)[-3:]; tr2 = np.argsort(ci)[:3]
            Fv  = sum(ci[j] * (X[j]-X[i]) for j in ta)
            Fv += w_r * sum(ci[j] * (X[j]-X[i]) for j in tr2)
            Fv /= 6.0
            nn = np.argmin(np.linalg.norm(Xn - np.clip(X[i]+ALPHA*Fv, 0, 1), axis=1))
            X[i] = Xn[nn]; visit[nn] += 1.0
            bja = ta[np.argmax(ci[ta])]
            W_new = (1 - beta_slow) * W[i] + beta_slow * W[bja]
            W_new /= W_new.sum()
            W[i] = W_new
    probs = visit + 1.0; probs /= probs.sum()
    return rng2.choice(N_a, n, replace=True, p=probs)

struct_configs = [
    {'beta_slow': 0.02, 'm_diag': -2.0, 'm_offdiag': 0.3,  'label': 'b=0.02 M(-2,+0.3)'},
    {'beta_slow': 0.05, 'm_diag': -2.0, 'm_offdiag': 0.3,  'label': 'b=0.05 M(-2,+0.3)'},
    {'beta_slow': 0.08, 'm_diag': -2.0, 'm_offdiag': 0.3,  'label': 'b=0.08 M(-2,+0.3)'},
    {'beta_slow': 0.02, 'm_diag': -1.0, 'm_offdiag': 0.5,  'label': 'b=0.02 M(-1,+0.5)'},
    {'beta_slow': 0.02, 'm_diag': -3.0, 'm_offdiag': 0.1,  'label': 'b=0.02 M(-3,+0.1)'},
]

print('=== Section B3: Structured M Initialization ===')
print(f'  {"config":<28} {"mean":>7} {"std":>7}  vs baseline')
print('-' * 60)
b3_results = {}
for cfg in struct_configs:
    scores = [evaluate_gnn(run_aiso_structured(
                  X_anom_domain, N_SEEN, sd,
                  beta_slow=cfg['beta_slow'],
                  m_diag=cfg['m_diag'],
                  m_offdiag=cfg['m_offdiag']))['PR-AUC']
              for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    b3_results[cfg['label']] = {'mean': m, 'std': s, 'vals': scores}
    flag = ' ★' if m > KNOWN_BASELINE else ''
    print(f'  {cfg["label"]:<28} {m:.4f} {s:.4f}  {m-KNOWN_BASELINE:+.4f}{flag}')
print()
print(f'  Baseline AISO(Dom-12 N=12): {KNOWN_BASELINE:.4f}')
'''))

# ── Cell f12: B summary ────────────────────────────────────────────────────
cells.append(code('f12', r'''all_b_results = {}
all_b_results.update(b1_results)
all_b_results.update(b2_results)
all_b_results.update(b3_results)

print('=== Section B Summary: Type Mechanism Experiments ===')
print(f'  {"Method":<32} {"mean":>7} {"std":>7}  vs baseline')
print('-' * 60)
for name, r in sorted(all_b_results.items(), key=lambda x: x[1]['mean'], reverse=True):
    flag = ' ★' if r['mean'] > KNOWN_BASELINE else ''
    print(f'  {name:<32} {r["mean"]:.4f} {r["std"]:.4f}  {r["mean"]-KNOWN_BASELINE:+.4f}{flag}')
print()
print(f'  Baseline AISO(Dom-12 N=12): {KNOWN_BASELINE:.4f}')

best_b = max(all_b_results, key=lambda k: all_b_results[k]['mean'])
print(f'  Best Exp3 variant: {best_b}  mean={all_b_results[best_b]["mean"]:.4f}')
'''))

# ── Cell f13: markdown - Section C ────────────────────────────────────────
cells.append(md('''## Section C — Exp 1b: Feature Expansion Experiments

Investigates whether expanding the domain feature space improves AISO sampling.

- **C1**: Feature subsets (Dom-12 vs Dom-18 vs Dom-25)
- **C2**: PSO(Dom) — isolates algorithm contribution (PSO with domain features)
- **C3**: Graph-specific baselines (PageRank-weighted, embedding density)
'''))

# ── Cell f14: extended domain features ────────────────────────────────────
cells.append(code('f14', r'''print('Building extended domain features...')

in_degree = np.bincount(ei_np2[1], minlength=N_NODES).astype(float)
in_out_ratio = in_degree / np.maximum(node_degree, 1)

neigh_degree_mean = np.zeros(N_NODES)
for src, dst in zip(ei_np2[0], ei_np2[1]):
    neigh_degree_mean[src] += node_degree[dst]
neigh_degree_mean /= np.maximum(neigh_count, 1)

neigh2_hub = np.zeros(N_NODES)
for src, dst in zip(ei_np2[0], ei_np2[1]):
    neigh2_hub[src] += node_degree[dst] * illicit_neigh_ratio[dst]
neigh2_hub /= np.maximum(neigh_count, 1)

ts_burst = np.zeros(N_NODES)
ts_counts = {}
for t in np.unique(ts_all):
    mask = ts_all == t
    ts_counts[t] = mask.sum()
for i, t in enumerate(ts_all):
    ts_burst[i] = ts_counts[t]
ts_burst = (ts_burst - ts_burst.min()) / (ts_burst.max() - ts_burst.min() + 1e-8)

degree_log = np.log1p(node_degree)
degree_log = (degree_log - degree_log.min()) / (degree_log.max() - degree_log.min() + 1e-8)

degree_x_ts = node_degree * ts_norm
degree_x_ts = (degree_x_ts - degree_x_ts.min()) / (degree_x_ts.max() - degree_x_ts.min() + 1e-8)

illicit_x_neigh2 = illicit_neigh_ratio * neigh2_illicit

hub_x_ts = hub_fraud * ts_norm
hub_x_ts = (hub_x_ts - hub_x_ts.min()) / (hub_x_ts.max() - hub_x_ts.min() + 1e-8)

local_range = local_max - local_feat.min(axis=1)
agg_range   = agg_max  - agg_feat.min(axis=1)
local_agg_diff = local_mean - agg_mean

illicit_isolation = illicit_neigh_ratio * (1.0 - illicit_neigh_ratio)

feat_base_12 = [
    node_degree, illicit_neigh_ratio, neigh2_illicit, node_ts_illicit_rate,
    local_mean, local_std, local_max, agg_mean, agg_std, agg_max, ts_norm, hub_fraud,
]
feat_extra_6 = [neigh_degree_mean, neigh2_hub, ts_burst, degree_log, degree_x_ts, illicit_x_neigh2]
feat_extra_7 = [hub_x_ts, local_range, agg_range, local_agg_diff, illicit_isolation, in_degree, in_out_ratio]

def make_feat(parts):
    raw = np.stack(parts, axis=1)
    return MinMaxScaler().fit_transform(raw)

domain_12  = make_feat(feat_base_12)
domain_18  = make_feat(feat_base_12 + feat_extra_6)
domain_25  = make_feat(feat_base_12 + feat_extra_6 + feat_extra_7)

X_anom_12  = domain_12[train_anom_idx]
X_anom_18  = domain_18[train_anom_idx]
X_anom_25  = domain_25[train_anom_idx]

print(f'Dom-12: {domain_12.shape}  Dom-18: {domain_18.shape}  Dom-25: {domain_25.shape}')
print(f'Illicit pools: 12={X_anom_12.shape} 18={X_anom_18.shape} 25={X_anom_25.shape}')
'''))

# ── Cell f15: C1 - feature subset comparison ──────────────────────────────
cells.append(code('f15', r'''print('=== Section C1: Feature Subset Comparison ===')

feat_configs = [
    ('AISO(Dom-12, N=12)', X_anom_12,  12),
    ('AISO(Dom-18, N=12)', X_anom_18,  12),
    ('AISO(Dom-25, N=12)', X_anom_25,  12),
    ('AISO(Dom-12, N=14)', X_anom_12,  14),
    ('AISO(Dom-18, N=14)', X_anom_18,  14),
    ('AISO(Dom-25, N=14)', X_anom_25,  14),
]

print(f'  {"Method":<26} {"mean":>7} {"std":>7}  vs Dom-12')
print('-' * 60)
c1_results = {}
for name, X_feat, nt in feat_configs:
    scores = [evaluate_gnn(run_aiso_nt(X_feat, N_SEEN, sd, nt))['PR-AUC']
              for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    c1_results[name] = {'mean': m, 'std': s, 'vals': scores}
    flag = ' ★' if m > KNOWN_BASELINE else ''
    print(f'  {name:<26} {m:.4f} {s:.4f}  {m-KNOWN_BASELINE:+.4f}{flag}')

print()
print(f'  Baseline AISO(Dom-12 N=12): {KNOWN_BASELINE:.4f}')
best_c1 = max(c1_results, key=lambda k: c1_results[k]['mean'])
print(f'  Best C1: {best_c1}  mean={c1_results[best_c1]["mean"]:.4f}')
'''))

# ── Cell f16: C2 - PSO(Dom) comparison ────────────────────────────────────
cells.append(code('f16', r'''print('=== Section C2: PSO(Dom) — Algorithm Contribution Isolation ===')

pso_dom_configs = [
    ('PSO(Dom-12)', X_anom_12),
    ('PSO(Dom-18)', X_anom_18),
    ('PSO(Dom-25)', X_anom_25),
]

print(f'  {"Method":<22} {"mean":>7} {"std":>7}  vs AISO(Dom-12)')
print('-' * 55)
c2_results = {}
for name, X_feat in pso_dom_configs:
    scores = [evaluate_gnn(run_pso(X_feat, N_SEEN, sd))['PR-AUC']
              for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    c2_results[name] = {'mean': m, 'std': s, 'vals': scores}
    flag = ' ★' if m > KNOWN_BASELINE else ''
    print(f'  {name:<22} {m:.4f} {s:.4f}  {m-KNOWN_BASELINE:+.4f}{flag}')

print()
print(f'  Baseline AISO(Dom-12 N=12): {KNOWN_BASELINE:.4f}')
print()

# AISO vs PSO differential (algorithm contribution)
print('  Algorithm contribution (AISO - PSO, same feature set):')
for feat in ['Dom-12', 'Dom-18', 'Dom-25']:
    aiso_key = next((k for k in c1_results if feat in k and 'N=12' in k), None)
    pso_key  = f'PSO({feat})'
    if aiso_key and pso_key in c2_results:
        diff = c1_results[aiso_key]['mean'] - c2_results[pso_key]['mean']
        print(f'    {feat}: AISO={c1_results[aiso_key]["mean"]:.4f}  PSO={c2_results[pso_key]["mean"]:.4f}  diff={diff:+.4f}')
'''))

# ── Cell f17: C3 - graph-specific baselines ───────────────────────────────
cells.append(code('f17', r'''import scipy.sparse as sp

print('=== Section C3: Graph-Specific Baselines ===')

# PageRank-weighted sampling from illicit pool
def run_pagerank_sampling(n, seed, alpha=0.85, max_iter=50):
    rng2 = np.random.RandomState(seed)
    N = N_NODES
    ei = edge_index.numpy()

    # sparse adjacency (out-normalized)
    row, col = ei[0], ei[1]
    out_deg = np.bincount(row, minlength=N).astype(float)
    out_deg_safe = np.where(out_deg > 0, out_deg, 1.0)
    data = 1.0 / out_deg_safe[row]
    A = sp.csr_matrix((data, (row, col)), shape=(N, N))

    pr = np.ones(N) / N
    for _ in range(max_iter):
        pr = alpha * A.T.dot(pr) + (1 - alpha) / N
        pr /= pr.sum()

    illicit_pr = pr[train_anom_idx]
    illicit_pr = illicit_pr / illicit_pr.sum()
    pool_idx = rng2.choice(len(train_anom_idx), n, replace=True, p=illicit_pr)
    return pool_idx

# Embedding density sampling
def run_embedding_density(X_a, n, seed, k=5):
    rng2 = np.random.RandomState(seed)
    nn_model = NearestNeighbors(n_neighbors=min(k+1, len(X_a))).fit(X_a)
    d, _ = nn_model.kneighbors(X_a)
    density = 1.0 / (d[:,1:].mean(1) + 1e-8)
    probs = density / density.sum()
    return rng2.choice(len(X_a), n, replace=True, p=probs)

c3_configs = [
    ('PageRank(graph)',         lambda sd: run_pagerank_sampling(N_SEEN, sd)),
    ('EmbDensity(Dom-12)',      lambda sd: run_embedding_density(X_anom_12, N_SEEN, sd)),
    ('EmbDensity(Dom-18)',      lambda sd: run_embedding_density(X_anom_18, N_SEEN, sd)),
]

print(f'  {"Method":<26} {"mean":>7} {"std":>7}  vs baseline')
print('-' * 60)
c3_results = {}
for name, fn in c3_configs:
    scores = [evaluate_gnn(fn(sd))['PR-AUC'] for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    c3_results[name] = {'mean': m, 'std': s, 'vals': scores}
    flag = ' ★' if m > KNOWN_BASELINE else ''
    print(f'  {name:<26} {m:.4f} {s:.4f}  {m-KNOWN_BASELINE:+.4f}{flag}')

print()
print(f'  Baseline AISO(Dom-12 N=12): {KNOWN_BASELINE:.4f}')
'''))

# ── Cell f18: markdown - Section D ────────────────────────────────────────
cells.append(md('''## Section D — Final Comprehensive Comparison

All methods ranked by 5-seed mean PR-AUC.
Showdown protocol: transductive GCN, bidirectional edges, fixed sel_n (SEED=42).
'''))

# ── Cell f19: final table + visualization ─────────────────────────────────
cells.append(code('f19', r'''all_results = {}
all_results.update(baseline_results)
all_results.update({f'[Exp3] {k}': v for k, v in all_b_results.items()})
all_results.update({f'[Exp1b] {k}': v for k, v in c1_results.items()})
all_results.update({f'[Exp1b] {k}': v for k, v in c2_results.items()})
all_results.update({f'[Exp1b] {k}': v for k, v in c3_results.items()})

print('=' * 80)
print(f'  FINAL COMPREHENSIVE COMPARISON (5 seeds, showdown transductive protocol)')
print('=' * 80)
print(f'  {"Rank":<5} {"Method":<42} {"mean":>7} {"std":>7}  vs AISO(Dom-12 N=12)')
print('-' * 80)
ranked = sorted(all_results.items(), key=lambda x: x[1]['mean'], reverse=True)
for rank, (name, r) in enumerate(ranked, 1):
    flag = ' ★' if r['mean'] > KNOWN_BASELINE else ''
    print(f'  {rank:<5} {name:<42} {r["mean"]:>7.4f} {r["std"]:>7.4f}  {r["mean"]-KNOWN_BASELINE:+.4f}{flag}')
print('=' * 80)
print(f'  Reference AISO(Dom-12 N=12): {KNOWN_BASELINE:.4f}')

# ── visualization ─────────────────────────────────────────────────────────
top_n = min(20, len(all_results))
top_ranked = ranked[:top_n]
names_r  = [n for n, _ in top_ranked]
means_r  = [r['mean'] for _, r in top_ranked]
stds_r   = [r['std']  for _, r in top_ranked]

def get_color(name):
    if '[Exp3]' in name:   return '#E07B54'
    if '[Exp1b]' in name:
        if 'PSO' in name:  return '#55A868'
        return '#4C72B0'
    if 'AISO' in name:     return '#C44E52'
    if 'PSO'  in name:     return '#55A868'
    return '#aaaaaa'

colors_r = [get_color(n) for n in names_r]

fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.42)))
bars = ax.barh(range(top_n), means_r[::-1], xerr=stds_r[::-1],
               capsize=4, color=colors_r[::-1], alpha=0.85, edgecolor='white')
ax.axvline(KNOWN_BASELINE, color='black', ls='--', lw=1.5,
           label=f'AISO(Dom-12 N=12) mean={KNOWN_BASELINE:.4f}')
ax.set_yticks(range(top_n))
ax.set_yticklabels([n.replace('[Exp3] ', 'E3: ').replace('[Exp1b] ', 'E1b: ')
                    for n in names_r[::-1]], fontsize=8)
ax.set_xlabel('PR-AUC (mean over 5 seeds)')
ax.set_title('Final Benchmark — All Methods (Top 20)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='x')

from matplotlib.patches import Patch as P
ax.legend(handles=[
    P(facecolor='#C44E52', label='Baseline (AISO)'),
    P(facecolor='#E07B54', label='Exp 3: Type mechanism'),
    P(facecolor='#4C72B0', label='Exp 1b: Feature expansion (AISO)'),
    P(facecolor='#55A868', label='Exp 1b / Baseline (PSO)'),
    P(facecolor='#aaaaaa', label='Baseline (other)'),
] + [plt.Line2D([0],[0], color='black', ls='--',
                label=f'AISO(Dom-12 N=12)={KNOWN_BASELINE:.4f}')],
fontsize=8)

plt.tight_layout()
plt.savefig('final_benchmark_comparison.png', bbox_inches='tight', dpi=120)
plt.show()
print('Saved: final_benchmark_comparison.png')
'''))

# ── Assign unique IDs ──────────────────────────────────────────────────────
for i, c in enumerate(cells):
    if c.get('id') is None:
        c['id'] = f'auto_{i:02d}'

nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'},
    },
    'cells': cells,
}

with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

total = len(cells)
code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
md_cells   = total - code_cells
print(f'\n완료: {NOTEBOOK_PATH}')
print(f'  총 {total} 셀 (코드 {code_cells}, 마크다운 {md_cells})')
print()
print('구조:')
print('  f00-f05  Setup (imports, data, GCN, domain feats, samplers)')
print('  f06-f07  Section A: Baseline verification (5 seeds)')
print('  f08-f12  Section B: Exp 3 - Type mechanism (B1 gamma / B2 anti / B3 struct)')
print('  f13-f17  Section C: Exp 1b - Feature expansion (C1 subsets / C2 PSO / C3 graph)')
print('  f18-f19  Section D: Final comprehensive comparison')
