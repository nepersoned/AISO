"""
Exp A (corrected): M_sym = |M_smart| vs Smart M, measured during actual AISO training.
Compares mask_jaccard_mean at end of training (same pipeline as exp7).
GNN eval skipped (budget=0) so this runs fast.
"""
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import time
from pathlib import Path
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

BASE      = Path(r'c:\Users\kevin\OneDrive\Desktop\AISO\Elliptic Bitcoin\elliptic_bitcoin_dataset')
K_SELECT  = 20
N_TYPES   = 15
SEEDS     = [0, 7, 42, 77, 123]
GAMMA     = 0.5
N_ILLICIT = 1000
N_LICIT   = 10000
SPLIT_T   = 34

# ── Data ────────────────────────────────────────────────────────
print('Loading data...')
feat_df = pd.read_csv(BASE / 'elliptic_txs_features.csv', header=None)
cls_df  = pd.read_csv(BASE / 'elliptic_txs_classes.csv')
feat_df.columns = ['txId', 'timestep'] + [f'f{i}' for i in range(165)]
cls_df.columns  = ['txId', 'class']
df      = feat_df.merge(cls_df, on='txId')
labeled = df[df['class'] != 'unknown'].copy().reset_index(drop=True)
labeled['label'] = (labeled['class'] == '1').astype(int)
feat_cols = [f'f{i}' for i in range(165)]
X_raw  = labeled[feat_cols].values.astype(float)
y_all  = labeled['label'].values
ts_all = labeled['timestep'].values

train_mask = ts_all <= SPLIT_T
test_mask  = ts_all >  SPLIT_T
train_illicit_idx = np.where(train_mask & (y_all == 1))[0]
train_licit_idx   = np.where(train_mask & (y_all == 0))[0]

rng_data = np.random.RandomState(42)
licit_sample = rng_data.choice(train_licit_idx, min(N_LICIT * 3, len(train_licit_idx)), replace=False)
train_idx = np.concatenate([train_illicit_idx, licit_sample])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X_tr_all = X_scaled[train_idx]
y_tr_all = y_all[train_idx]

illicit_pool_idx = train_illicit_idx
X_illicit_pool   = X_scaled[illicit_pool_idx]

print(f'Train: {len(X_tr_all)} | Illicit pool: {len(X_illicit_pool)}')

# ── Smart M & Sym M ─────────────────────────────────────────────
print('Building M_smart and M_sym...')
C_full = np.corrcoef(X_tr_all.T)
C_abs  = np.abs(C_full)
dist_matrix = 1.0 - C_abs
np.fill_diagonal(dist_matrix, 0.0)

clustering = AgglomerativeClustering(n_clusters=N_TYPES, metric='precomputed', linkage='average')
cluster_labels = clustering.fit_predict(dist_matrix)

MI_global = mutual_info_classif(X_tr_all, y_tr_all, random_state=42)
mean_MI = np.array([
    MI_global[cluster_labels == k].mean() if (cluster_labels == k).any() else 0.0
    for k in range(N_TYPES)
])
MI_norm = (mean_MI - mean_MI.min()) / (mean_MI.max() - mean_MI.min() + 1e-8)

K = N_TYPES
M_smart = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        fi = np.where(cluster_labels == i)[0]
        fj = np.where(cluster_labels == j)[0]
        if i == j:
            M_smart[i][j] = -1.0
        else:
            corr_pen = -np.mean(C_abs[np.ix_(fi, fj)])
            M_smart[i][j] = corr_pen + GAMMA * (MI_norm[j] - MI_norm[i])

# Sym M: take abs then symmetrize
M_abs = np.abs(M_smart)
M_sym = 0.5 * (M_abs + M_abs.T)
np.fill_diagonal(M_sym, -1.0)

print(f'M_smart asymmetry check: M[0,1]={M_smart[0,1]:.3f}, M[1,0]={M_smart[1,0]:.3f}')
print(f'M_sym   symmetry check:  M[0,1]={M_sym[0,1]:.3f},   M[1,0]={M_sym[1,0]:.3f}')

# ── Diversity utils ─────────────────────────────────────────────
def mask_jaccard(masks):
    if len(masks) < 2:
        return 0.0
    vals = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            si, sj = set(masks[i]), set(masks[j])
            vals.append(len(si & sj) / max(len(si | sj), 1))
    return float(np.mean(vals))

# ── Stripped-down wrapper (no GNN, diversity only) ──────────────
def run_aiso_diversity(M, seed, n_agents=20, n_iter=80, beta=0.15,
                       subsample_ratio=0.15, val_ratio=0.2,
                       T_start=1.0, T_end=0.05, div_interval=20):
    rng = np.random.RandomState(seed)
    N, D = X_tr_all.shape

    val_n   = int(val_ratio * N)
    val_idx = rng.choice(N, val_n, replace=False)
    tr_mask = np.ones(N, bool); tr_mask[val_idx] = False
    X_val, y_val = X_tr_all[val_idx], y_tr_all[val_idx]
    X_t,   y_t   = X_tr_all[tr_mask], y_tr_all[tr_mask]
    sub_n = max(50, int(subsample_ratio * len(X_t)))

    cache = {}

    def get_mask(Wi, t, explore=True):
        if explore:
            ratio  = t / max(1, n_iter - 1)
            T      = T_start * ((T_end / T_start) ** ratio)
            logits = Wi[cluster_labels] / T
            logits -= logits.max()
            probs  = np.exp(logits); probs /= probs.sum()
            return tuple(rng.choice(D, K_SELECT, replace=False, p=probs))
        else:
            return tuple(np.argsort(Wi[cluster_labels])[-K_SELECT:])

    def get_score(mask_key, X_sub, y_sub):
        if mask_key not in cache:
            feat = list(mask_key)
            sgd  = SGDClassifier(loss='log_loss', max_iter=5, tol=None,
                                  class_weight='balanced', random_state=seed)
            try:
                sgd.fit(X_sub[:, feat], y_sub)
                val_proba = sgd.predict_proba(X_val[:, feat])[:, 1]
                proxy_auc = roc_auc_score(y_val, val_proba)
                illicit_s = sgd.predict_proba(X_illicit_pool[:, feat])[:, 1]
            except Exception:
                proxy_auc = 0.5
                illicit_s = rng.rand(len(X_illicit_pool))
            cache[mask_key] = {'proxy_auc': proxy_auc, 'illicit_scores': illicit_s}
        return cache[mask_key]['proxy_auc']

    W      = rng.dirichlet(np.ones(K), n_agents)
    scores = np.zeros(n_agents)
    nb_    = min(3, n_agents - 1)
    for i in range(n_agents):
        sub = rng.choice(len(X_t), sub_n, replace=False)
        mk  = get_mask(W[i], 0)
        scores[i] = get_score(mk, X_t[sub], y_t[sub])

    div_log = []
    for t in range(n_iter):
        Cm = W @ M @ W.T
        np.fill_diagonal(Cm, 0.0)
        sub = rng.choice(len(X_t), sub_n, replace=False)
        X_sub, y_sub = X_t[sub], y_t[sub]

        for i in range(n_agents):
            att     = np.argsort(Cm[i])[-nb_:]
            att_s   = np.array([get_score(get_mask(W[j], t), X_sub, y_sub) for j in att])
            max_s   = att_s.max() + 1e-8
            bj      = att[np.argmax(Cm[i, att] * att_s / max_s)]
            W_new   = (1 - beta) * W[i] + beta * W[bj]
            W[i]    = W_new / W_new.sum()
            scores[i] = get_score(get_mask(W[i], t), X_sub, y_sub)

        if t % div_interval == 0 or t == n_iter - 1:
            cur_masks = [get_mask(W[i], t, explore=False) for i in range(n_agents)]
            mj = mask_jaccard(cur_masks)
            div_log.append({'generation': t, 'mask_jaccard_mean': mj})

    return div_log

# ── Run ─────────────────────────────────────────────────────────
configs = [
    ('AISO (Smart M)', M_smart),
    ('AISO (Sym M)',   M_sym),
]

results = {}
for mname, M_use in configs:
    print(f'\n=== {mname} ===')
    seed_finals = []
    for seed in SEEDS:
        t0 = time.time()
        dlog = run_aiso_diversity(M_use, seed)
        final_j = dlog[-1]['mask_jaccard_mean']
        seed_finals.append(final_j)
        print(f'  seed={seed:3d}  final_jaccard={final_j:.4f}  ({time.time()-t0:.1f}s)')
    results[mname] = seed_finals
    print(f'  MEAN={np.mean(seed_finals):.4f}  STD={np.std(seed_finals):.4f}')

print('\n\n=== FINAL COMPARISON ===')
print(f'{"Method":<22} {"Mean Jaccard":>14} {"Std":>8}')
print('-' * 46)
# Also include existing data from exp7_diversity_log.csv
import pandas as pd
try:
    ex = pd.read_csv('exp7_diversity_log.csv')
    bm = ex.groupby('method')['mask_jaccard_mean'].agg(['mean','std'])
    for m in bm.index:
        print(f'{m:<22} {bm.loc[m,"mean"]:>14.4f} {bm.loc[m,"std"]:>8.4f}  (from exp7 log)')
except Exception as e:
    print(f'(could not load exp7 log: {e})')

for mname, vals in results.items():
    print(f'{mname:<22} {np.mean(vals):>14.4f} {np.std(vals):>8.4f}  (new run)')
