import json

def code_cell(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source}

def md_cell(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source}

cells = []

cells.append(md_cell(
    "# Exp A Extended: 2x2 Ablation (Stage1 M x Stage2 M)\n"
    "| | Stage2 Sym | Stage2 Smart |\n"
    "|---|---|---|\n"
    "| **Stage1 Sym** | Case1: Worst | Case2: Mid-Low |\n"
    "| **Stage1 Smart** | Case3: Mid-High | Case4: 0.6644 (known) |"
))

cells.append(code_cell(
"""import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from torch_geometric.nn import GCNConv
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

BASE      = Path(r'c:\\Users\\kevin\\OneDrive\\Desktop\\AISO\\Elliptic Bitcoin\\elliptic_bitcoin_dataset')
K_SELECT  = 20
N_TYPES   = 15
SEEDS     = [0, 7, 42, 77, 123]
GAMMA     = 0.5
SPLIT_T   = 34
N_ILLICIT = 1000
N_LICIT   = 10000
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')"""
))

cells.append(code_cell(
"""print('Loading data...')
feat_df = pd.read_csv(BASE / 'elliptic_txs_features.csv', header=None)
cls_df  = pd.read_csv(BASE / 'elliptic_txs_classes.csv')
edge_df = pd.read_csv(BASE / 'elliptic_txs_edgelist.csv')
feat_df.columns = ['txId', 'timestep'] + [f'f{i}' for i in range(165)]
cls_df.columns  = ['txId', 'class']
df      = feat_df.merge(cls_df, on='txId')
labeled = df[df['class'] != 'unknown'].copy().reset_index(drop=True)
labeled['label'] = (labeled['class'] == '1').astype(int)
feat_cols = [f'f{i}' for i in range(165)]
X_raw  = labeled[feat_cols].values.astype(float)
y_all  = labeled['label'].values
ts_all = labeled['timestep'].values
N_NODES = len(labeled)
txid_to_idx = {txid: i for i, txid in enumerate(labeled['txId'].values)}

e_s  = edge_df['txId1'].map(txid_to_idx)
e_d  = edge_df['txId2'].map(txid_to_idx)
valid = e_s.notna() & e_d.notna()
s_arr = e_s[valid].astype(int).values
d_arr = e_d[valid].astype(int).values
msk   = (s_arr < N_NODES) & (d_arr < N_NODES)
src_bi = np.concatenate([s_arr[msk], d_arr[msk]])
dst_bi = np.concatenate([d_arr[msk], s_arr[msk]])
edge_index = torch.tensor([src_bi, dst_bi], dtype=torch.long)

train_mask        = ts_all <= SPLIT_T
test_mask         = ts_all >  SPLIT_T
train_illicit_idx = np.where(train_mask & (y_all == 1))[0]
train_licit_idx   = np.where(train_mask & (y_all == 0))[0]

rng_data     = np.random.RandomState(42)
licit_sample = rng_data.choice(train_licit_idx, min(N_LICIT * 3, len(train_licit_idx)), replace=False)
train_idx    = np.concatenate([train_illicit_idx, licit_sample])

scaler         = StandardScaler()
X_scaled       = scaler.fit_transform(X_raw)
X_tr_all       = X_scaled[train_idx]
y_tr_all       = y_all[train_idx]
X_illicit_pool = X_scaled[train_illicit_idx]
print(f'Train: {len(X_tr_all)} | Illicit pool: {len(X_illicit_pool)}')"""
))

cells.append(code_cell(
"""class GCN(torch.nn.Module):
    def __init__(self, in_ch, hidden=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, 2)
        self.drop  = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.conv2(x, edge_index)

def evaluate_gnn(top_n_illicit, seed=42):
    torch.manual_seed(seed)
    licit_idx = np.random.RandomState(seed).choice(
        np.where(train_mask & (y_all == 0))[0], N_LICIT, replace=False)
    sub_idx = np.concatenate([train_illicit_idx[top_n_illicit], licit_idx])
    sub_set = set(sub_idx.tolist())
    sub_map = {v: i for i, v in enumerate(sub_idx)}
    X_sub   = torch.tensor(X_scaled[sub_idx], dtype=torch.float32)
    y_sub   = torch.tensor(y_all[sub_idx], dtype=torch.long)
    ei_sub  = torch.tensor([
        [sub_map[s] for s, d in zip(src_bi, dst_bi) if s in sub_set and d in sub_set],
        [sub_map[d] for s, d in zip(src_bi, dst_bi) if s in sub_set and d in sub_set]
    ], dtype=torch.long) if any(s in sub_set and d in sub_set for s, d in zip(src_bi[:100], dst_bi[:100])) else torch.zeros((2,0), dtype=torch.long)

    test_idx = np.where(test_mask)[0]
    X_test   = torch.tensor(X_scaled[test_idx], dtype=torch.float32)
    y_test   = y_all[test_idx]

    model = GCN(X_sub.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    w     = torch.tensor([1.0, (y_sub==0).sum().item()/(y_sub==1).sum().item()+1e-8], dtype=torch.float32).to(DEVICE)
    best_val, patience, best_state = 0, 0, None
    for ep in range(300):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(X_sub.to(DEVICE), ei_sub.to(DEVICE)), y_sub.to(DEVICE), weight=w)
        loss.backward(); opt.step()
        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                vsc = average_precision_score(y_sub.numpy(),
                      torch.softmax(model(X_sub.to(DEVICE), ei_sub.to(DEVICE)), -1)[:,1].cpu().numpy())
            if vsc > best_val: best_val = vsc; patience = 0; best_state = {k:v.clone() for k,v in model.state_dict().items()}
            else: patience += 1
            if patience >= 15: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    X_all = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        prob_all = torch.softmax(model(X_all.to(DEVICE), edge_index.to(DEVICE)), -1)[:,1].cpu().numpy()
    prob = prob_all[test_idx]
    return {'PR-AUC': average_precision_score(y_test, prob),
            'F1':     f1_score(y_test, (prob>0.5).astype(int), zero_division=0)}
print('GCN defined')"""
))

cells.append(code_cell(
"""print('Building M_smart / M_sym...')
C_abs  = np.abs(np.corrcoef(X_tr_all.T))
dist_m = 1.0 - C_abs; np.fill_diagonal(dist_m, 0.0)
clustering     = AgglomerativeClustering(n_clusters=N_TYPES, metric='precomputed', linkage='average')
cluster_labels = clustering.fit_predict(dist_m)
MI_global = mutual_info_classif(X_tr_all, y_tr_all, random_state=42)
mean_MI   = np.array([MI_global[cluster_labels==k].mean() if (cluster_labels==k).any() else 0.0 for k in range(N_TYPES)])
MI_norm   = (mean_MI - mean_MI.min()) / (mean_MI.max() - mean_MI.min() + 1e-8)

K = N_TYPES
M_smart = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        fi = np.where(cluster_labels==i)[0]; fj = np.where(cluster_labels==j)[0]
        M_smart[i][j] = -1.0 if i==j else -np.mean(C_abs[np.ix_(fi,fj)]) + GAMMA*(MI_norm[j]-MI_norm[i])

M_abs = np.abs(M_smart)
M_sym = 0.5*(M_abs + M_abs.T); np.fill_diagonal(M_sym, -1.0)
print(f'M_smart asymmetric: [{M_smart[0,1]:.3f}, {M_smart[1,0]:.3f}]')
print(f'M_sym   symmetric:  [{M_sym[0,1]:.3f}, {M_sym[1,0]:.3f}]')"""
))

cells.append(code_cell(
"""def run_stage1(M, seed, n_agents=20, n_iter=80, beta=0.15,
               subsample_ratio=0.15, val_ratio=0.2, T_start=1.0, T_end=0.05):
    rng = np.random.RandomState(seed)
    N, D = X_tr_all.shape; K = M.shape[0]; nb_ = min(3, n_agents-1)
    val_idx  = rng.choice(N, int(val_ratio*N), replace=False)
    tr_mask_ = np.ones(N, bool); tr_mask_[val_idx] = False
    X_val, y_val = X_tr_all[val_idx], y_tr_all[val_idx]
    X_t, y_t     = X_tr_all[tr_mask_], y_tr_all[tr_mask_]
    sub_n = max(50, int(subsample_ratio*len(X_t)))
    cache = {}

    def get_mask(Wi, t, explore=True):
        if explore:
            T = T_start*((T_end/T_start)**(t/max(1,n_iter-1)))
            logits = Wi[cluster_labels]/T; logits -= logits.max()
            probs = np.exp(logits); probs /= probs.sum()
            return tuple(rng.choice(D, K_SELECT, replace=False, p=probs))
        return tuple(np.argsort(Wi[cluster_labels])[-K_SELECT:])

    def get_score(mk, X_sub, y_sub):
        if mk not in cache:
            sgd = SGDClassifier(loss='log_loss', max_iter=5, tol=None, class_weight='balanced', random_state=seed)
            try:
                sgd.fit(X_sub[:, list(mk)], y_sub)
                cache[mk] = roc_auc_score(y_val, sgd.predict_proba(X_val[:, list(mk)])[:,1])
            except: cache[mk] = 0.5
        return cache[mk]

    W = rng.dirichlet(np.ones(K), n_agents)
    scores = np.array([get_score(get_mask(W[i],0), X_t[rng.choice(len(X_t),sub_n,False)], y_t[rng.choice(len(X_t),sub_n,False)]) for i in range(n_agents)])

    for t in range(n_iter):
        Cm = W@M@W.T; np.fill_diagonal(Cm,0.0)
        sub = rng.choice(len(X_t), sub_n, replace=False)
        X_sub, y_sub = X_t[sub], y_t[sub]
        for i in range(n_agents):
            att = np.argsort(Cm[i])[-nb_:]
            att_s = np.array([get_score(get_mask(W[j],t), X_sub, y_sub) for j in att])
            bj = att[np.argmax(Cm[i,att]*att_s/(att_s.max()+1e-8))]
            W_new = (1-beta)*W[i]+beta*W[bj]; W[i] = W_new/W_new.sum()
            scores[i] = get_score(get_mask(W[i],t), X_sub, y_sub)
        if (t+1) % 20 == 0:
            print(f'      S1 iter {t+1:3d}/80  cache={len(cache)}  best_proxy={max(cache.values()):.4f}')

    best_mask = get_mask(W[np.argmax(scores)], n_iter, explore=False)
    # force-cache if not seen during exploration
    if best_mask not in cache:
        get_score(best_mask, X_t, y_t)
    return list(best_mask)


def run_stage2(M, feat_mask, seed, n_agents=20, n_iter=80, beta=0.15,
               subsample_ratio=0.15, val_ratio=0.2, T_start=1.0, T_end=0.05):
    X_feat    = X_tr_all[:, feat_mask]
    X_ill_sub = X_illicit_pool[:, feat_mask]
    C_sub = np.abs(np.corrcoef(X_feat.T))
    dist_s = 1.0 - C_sub; np.fill_diagonal(dist_s, 0.0)
    K2 = M.shape[0]
    rng = np.random.RandomState(seed)
    N, D2 = X_feat.shape; nb_ = min(3, n_agents-1)
    val_idx  = rng.choice(N, int(val_ratio*N), replace=False)
    tr_mask_ = np.ones(N, bool); tr_mask_[val_idx] = False
    X_val2, y_val2 = X_feat[val_idx], y_tr_all[val_idx]
    X_t2, y_t2     = X_feat[tr_mask_], y_tr_all[tr_mask_]
    sub_n = max(50, int(subsample_ratio*len(X_t2)))
    cl2 = AgglomerativeClustering(n_clusters=min(K2,D2), metric='precomputed', linkage='average').fit_predict(dist_s)
    cache2 = {}

    def get_mask2(Wi, t, explore=True):
        if explore:
            T = T_start*((T_end/T_start)**(t/max(1,n_iter-1)))
            logits = Wi[cl2]/T; logits -= logits.max()
            probs = np.exp(logits); probs /= probs.sum()
        else:
            logits = Wi[cl2]; logits -= logits.max()
            probs = np.exp(logits); probs /= probs.sum()
        return tuple(sorted(rng.choice(D2, max(5,D2//2), replace=False, p=probs)))

    def get_score2(mk2, X_sub, y_sub):
        if mk2 not in cache2:
            sgd = SGDClassifier(loss='log_loss', max_iter=5, tol=None, class_weight='balanced', random_state=seed)
            try:
                sgd.fit(X_sub[:, list(mk2)], y_sub)
                ill_p = sgd.predict_proba(X_ill_sub[:, list(mk2)])[:,1]
                cache2[mk2] = (roc_auc_score(y_val2, sgd.predict_proba(X_val2[:, list(mk2)])[:,1]), ill_p)
            except: cache2[mk2] = (0.5, rng.rand(len(X_ill_sub)))
        return cache2[mk2][0]

    W = rng.dirichlet(np.ones(K2), n_agents)
    scores = np.array([get_score2(get_mask2(W[i],0), X_t2[rng.choice(len(X_t2),sub_n,False)], y_t2[rng.choice(len(X_t2),sub_n,False)]) for i in range(n_agents)])

    for t in range(n_iter):
        Cm = W@M@W.T; np.fill_diagonal(Cm,0.0)
        sub = rng.choice(len(X_t2), sub_n, replace=False)
        X_sub2, y_sub2 = X_t2[sub], y_t2[sub]
        for i in range(n_agents):
            att = np.argsort(Cm[i])[-nb_:]
            att_s = np.array([get_score2(get_mask2(W[j],t), X_sub2, y_sub2) for j in att])
            bj = att[np.argmax(Cm[i,att]*att_s/(att_s.max()+1e-8))]
            W_new = (1-beta)*W[i]+beta*W[bj]; W[i] = W_new/W_new.sum()
            scores[i] = get_score2(get_mask2(W[i],t), X_sub2, y_sub2)
        if (t+1) % 20 == 0:
            print(f'      S2 iter {t+1:3d}/80  cache={len(cache2)}  best_proxy={max(v[0] for v in cache2.values()):.4f}')

    best_i   = np.argmax(scores)
    best_mk2 = get_mask2(W[best_i], n_iter, explore=False)
    # BUG FIX: force-evaluate final mask if not cached during exploration
    if best_mk2 not in cache2:
        get_score2(best_mk2, X_t2, y_t2)
    _, ill_p = cache2[best_mk2]
    return np.argsort(ill_p)[-N_ILLICIT:]

print('Stage 1 / Stage 2 defined')"""
))

cells.append(code_cell(
"""CASES = [
    ('Case1: Sym->Sym',   M_sym,   'Sym',   M_sym,   'Sym'),
    ('Case2: Sym->Smart', M_sym,   'Sym',   M_smart, 'Smart'),
    ('Case3: Smart->Sym', M_smart, 'Smart', M_sym,   'Sym'),
]

# Stage1 cache: Cases 1&2 share Sym Stage1, Case3 uses Smart Stage1
stage1_cache = {}
def get_stage1(M, m1_label, seed):
    key = (m1_label, seed)
    if key not in stage1_cache:
        print(f'    Stage1 ({m1_label}) running...')
        t = time.time()
        stage1_cache[key] = run_stage1(M, seed)
        print(f'    Stage1 ({m1_label}) done: feat={stage1_cache[key][:5]}...  ({time.time()-t:.1f}s)')
    else:
        print(f'    Stage1 ({m1_label}) [cached]')
    return stage1_cache[key]

results_2x2 = {}

for case_name, M1, m1_label, M2, m2_label in CASES:
    print('\\n' + '='*55)
    print('  ' + case_name)
    print('='*55)
    pr_aucs = []
    t_case = time.time()

    for seed in SEEDS:
        print(f'\\n  [seed={seed}]')
        t0 = time.time()

        feat_mask = get_stage1(M1, m1_label, seed)

        print(f'    Stage2 (node select, M2={m2_label}) ...')
        t2 = time.time()
        top_n = run_stage2(M2, feat_mask, seed)
        print(f'    Stage2 done: {len(top_n)} nodes selected  ({time.time()-t2:.1f}s)')

        print('    GCN eval ...')
        t3 = time.time()
        res = evaluate_gnn(top_n, seed=seed)
        print(f'    GCN done: PR-AUC={res["PR-AUC"]:.4f}  F1={res["F1"]:.4f}  ({time.time()-t3:.1f}s)')

        pr_aucs.append(res['PR-AUC'])
        print(f'  => seed={seed} total: {time.time()-t0:.1f}s')

    results_2x2[case_name] = pr_aucs
    print(f'\\n  {case_name}  MEAN={np.mean(pr_aucs):.4f}  STD={np.std(pr_aucs):.4f}  (case total: {time.time()-t_case:.1f}s)')"""
))

cells.append(code_cell(
"""print('\\n=== 2x2 Ablation Final Table ===')
print(f'{\"Case\":<22} {\"PR-AUC\":>8} {\"Std\":>7}')
print('-' * 42)
for case_name, vals in results_2x2.items():
    print(f'{case_name:<22} {np.mean(vals):>8.4f} {np.std(vals):>7.4f}')
print(f'{\"Case4: Smart->Smart\":<22} {\"0.6644\":>8} {\"0.0200\":>7}  (known)')
print()
c1 = np.mean(results_2x2.get("Case1: Sym->Sym",   [0]))
c3 = np.mean(results_2x2.get("Case3: Smart->Sym", [0]))
print(f'Stage1 Smart M contribution (Case3-Case1): {c3-c1:+.4f}')
print(f'Stage2 Smart M contribution (Case4-Case3): {0.6644-c3:+.4f}')"""
))

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python', 'version': '3.10.0'}},
    'cells': cells
}

with open('aiso_expA_2x2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('aiso_expA_2x2.ipynb 재생성 완료')
