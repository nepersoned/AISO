"""
BURST_WINDOW / USER_CO_REVIEW 엣지 개수 계산
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

SAMPLE_SIZE = 50_000
N_AGENTS = 30; N_ITER = 60; ALPHA_A = 0.3; BETA_A = 0.1; N_TYPES = 17
RANDOM_STATE = 42

# ── 데이터 로드 ──────────────────────────────────────────────
print("데이터 로드 중...")
df = pd.read_pickle(r'c:\Users\kevin\OneDrive\Desktop\AISO\archive\yelpzip_w2v_features.pkl')
df['label'] = df['label'].map({-1: 1, 1: 0})
df['date']  = pd.to_datetime(df['date'])
df['text_len_col'] = df['text'].str.len()

user_review_cnt = df.groupby('user_id').size()
prod_review_cnt = df.groupby('prod_id').size()
user_spam_rate  = df.groupby('user_id')['label'].mean()

# ── AISO 피처 ────────────────────────────────────────────────
g = df.groupby('prod_id')
prod_ids  = np.array(sorted(df['prod_id'].unique()))
prod_spam = g['label'].mean()
prod_deg  = g.size()

feat_dict = {
    'spam_ratio':     g['label'].mean(),
    'spam_score':     g['label'].mean() * np.log1p(g.size()),
    'star1_ratio':    (df['rating']==1).groupby(df['prod_id']).mean(),
    'star5_ratio':    (df['rating']==5).groupby(df['prod_id']).mean(),
    'extreme_ratio':  df['rating'].isin([1,5]).groupby(df['prod_id']).mean(),
    'user_conc':      g.apply(lambda x: (x.groupby('user_id').size()**2).sum()/len(x)**2),
    'log_count':      np.log1p(g.size()),
    'burst':          g.apply(lambda x:
                          x.groupby(x['date'].dt.to_period('M')).size().std() /
                          (x.groupby(x['date'].dt.to_period('M')).size().mean()+1e-6)),
    'span_days':      g['date'].apply(lambda x: (x.max()-x.min()).days),
    'user_diversity': df.groupby('prod_id')['user_id'].nunique()/g.size(),
    'avg_user_rev':   g.apply(lambda x: x.groupby('user_id').size().mean()),
    'spam_user_ratio':g.apply(lambda x:
                          x['user_id'].map(user_spam_rate).fillna(0).gt(0.5).mean()),
    'peak_month':     g.apply(lambda x:
                          x.groupby(x['date'].dt.to_period('M')).size().max()/len(x)),
    'interval_mean':  g['date'].apply(lambda x:
                          x.sort_values().diff().dt.days.dropna().mean() if len(x)>1 else 0),
    'avg_text_len':   g['text_len_col'].mean(),
    'text_len_std':   g['text_len_col'].std().fillna(0),
    'short_ratio':    (df['text_len_col']<50).groupby(df['prod_id']).mean(),
}

prod_feat = MinMaxScaler().fit_transform(
    pd.DataFrame(feat_dict).fillna(0).loc[prod_ids].values
).astype(np.float32)

# ── AISO ────────────────────────────────────────────────────
def _select(X):
    scores = prod_feat @ X
    order  = np.argsort(scores)[::-1]
    sel, total = [], 0
    for i in order:
        p = prod_ids[i]; n = prod_deg.get(p, 0)
        if n == 0: continue
        if total + n > SAMPLE_SIZE: continue
        sel.append(p); total += n
    return sel

def _eval(X):
    sel = _select(X)
    if not sel: return 0.0
    return np.average([prod_spam[p] for p in sel], weights=[prod_deg[p] for p in sel])

np.random.seed(RANDOM_STATE)
M  = np.random.uniform(0, 2, (N_TYPES, N_TYPES)).astype(np.float32)
aX = np.random.rand(N_AGENTS, N_TYPES).astype(np.float32)
aW = np.random.dirichlet(np.ones(N_TYPES), N_AGENTS).astype(np.float32)
aS = np.array([_eval(aX[i]) for i in range(N_AGENTS)])
bX = aX[np.argmax(aS)].copy(); bS = aS.max()

for it in range(N_ITER):
    for i in range(N_AGENTS):
        j = np.random.choice([k for k in range(N_AGENTS) if k != i])
        c = float(aW[i] @ M @ aW[j])
        nX = np.clip(aX[i] + ALPHA_A * c * (aX[j] - aX[i]), 0, 1)
        ns = _eval(nX)
        if ns > aS[i]:
            aX[i] = nX; aS[i] = ns
        aW[i] = (1 - BETA_A) * aW[i] + BETA_A * aW[j]; aW[i] /= aW[i].sum()
    if aS.max() > bS: bS = aS.max(); bX = aX[np.argmax(aS)].copy()
    if (it + 1) % 20 == 0:
        print(f'Iter {it+1:3d} | Best: {aS.max():.4f}')

best_prods = _select(bX)
sub = df[df['prod_id'].isin(best_prods)].copy().reset_index(drop=True)
sub['node_id'] = range(len(sub))
print(f'\n서브그래프 | 노드: {len(sub):,} | 스팸: {sub["label"].mean():.2%}')

# ── BURST_WINDOW 엣지 (동일 사용자, 72시간 이내) ─────────────
print("\nBURST_WINDOW 엣지 계산 중...")
burst_src, burst_dst = [], []
BURST_HOURS = 72
for uid, grp in sub.groupby('user_id'):
    if len(grp) < 2:
        continue
    grp_sorted = grp.sort_values('date')
    nodes = grp_sorted['node_id'].values
    dates = grp_sorted['date'].values
    for idx_i in range(len(nodes)):
        for idx_j in range(idx_i + 1, len(nodes)):
            diff = (dates[idx_j] - dates[idx_i]) / np.timedelta64(1, 'h')
            if diff > BURST_HOURS:
                break  # sorted by date, no need to check further
            burst_src += [nodes[idx_i], nodes[idx_j]]
            burst_dst += [nodes[idx_j], nodes[idx_i]]

print(f'BURST_WINDOW 엣지: {len(burst_src)//2:,}개')

# ── USER_CO_REVIEW 엣지 수 계산 (노드 쌍 개수만, 실제 리스트 미생성) ─────────
print("\nUSER_CO_REVIEW 엣지 계산 중...")

from collections import defaultdict

# 상품별 리뷰어 집합 (상품당 최대 50명 샘플링 — combinations 폭발 방지)
MAX_USERS_PER_PROD = 50
np.random.seed(RANDOM_STATE)
prod_users = {}
for prod, grp in sub.groupby('prod_id')['user_id']:
    users = list(set(grp.tolist()))
    if len(users) > MAX_USERS_PER_PROD:
        users = list(np.random.choice(users, MAX_USERS_PER_PROD, replace=False))
    prod_users[prod] = users

# 공동 출현 카운트: 사용자 쌍 → 함께 리뷰한 상품 수
pair_count = defaultdict(int)
for prod, users_list in prod_users.items():
    users_sorted = sorted(users_list)
    if len(users_sorted) < 2:
        continue
    for u, v in combinations(users_sorted, 2):
        pair_count[(u, v)] += 1

# 2개 이상 상품에서 공동 출현한 사용자 쌍
co_pairs = [(u, v) for (u, v), cnt in pair_count.items() if cnt >= 2]
print(f'공동 출현 사용자 쌍 수: {len(co_pairs):,}')

# 엣지 수 = 각 사용자 쌍의 노드 수 곱의 합
user_node_cnt = sub.groupby('user_id').size()
co_edge_count = sum(
    user_node_cnt.get(u, 0) * user_node_cnt.get(v, 0)
    for u, v in co_pairs
)

print(f'USER_CO_REVIEW 엣지: {co_edge_count:,}개')
print('\n=== 결과 요약 ===')
print(f'BURST_WINDOW  : {len(burst_src)//2:,}개')
print(f'USER_CO_REVIEW: {co_edge_count:,}개')
