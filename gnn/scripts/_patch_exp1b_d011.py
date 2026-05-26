import json, ast, sys

NEW_D011 = r'''# PageRank 기반 샘플링
def pagerank_sampling(n, seed, n_iter=30, damping=0.85):
    rng = np.random.RandomState(seed)
    anom_set = set(train_illicit_idx.tolist())
    anom_idx_map = {g: l for l, g in enumerate(train_illicit_idx)}
    N_a = len(train_illicit_idx)

    sub_src_list = []; sub_dst_list = []
    for s, d in zip(src_arr, dst_arr):
        if s in anom_set and d in anom_set:
            sub_src_list.append(anom_idx_map[s])
            sub_dst_list.append(anom_idx_map[d])

    pr = np.ones(N_a) / N_a
    out_deg = np.bincount(sub_src_list, minlength=N_a).astype(float)
    out_deg = np.maximum(out_deg, 1)
    for _ in range(n_iter):
        new_pr = np.ones(N_a) * (1 - damping) / N_a
        for s, d in zip(sub_src_list, sub_dst_list):
            new_pr[d] += damping * pr[s] / out_deg[s]
        pr = new_pr; pr /= pr.sum()

    probs = pr + 1e-6; probs /= probs.sum()
    return rng.choice(N_a, n, replace=True, p=probs)


# 임베딩 밀도 샘플링
def embedding_density_sampling(n, seed, hidden=32, n_iter=50):
    rng_np = np.random.RandomState(seed)
    torch.manual_seed(seed)

    licit_idx = rng_np.choice(train_licit_idx, 3000, replace=False)
    sub_idx = np.concatenate([licit_idx, train_illicit_idx])
    sub_map = {g: l for l, g in enumerate(sub_idx)}

    sub_s = []; sub_d = []
    for s, d in zip(src_arr, dst_arr):
        if s in sub_map and d in sub_map:
            sub_s.append(sub_map[s]); sub_d.append(sub_map[d])

    if not sub_s:
        return rng_np.choice(len(train_illicit_idx), n, replace=True)

    ei_sub = torch.tensor([sub_s, sub_d], dtype=torch.long).to(DEVICE)
    X_sub  = torch.tensor(X_scaled[sub_idx], dtype=torch.float).to(DEVICE)
    y_sub  = torch.tensor(y_all[sub_idx], dtype=torch.long).to(DEVICE)

    enc = GCN(X_sub.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    w   = torch.tensor([1.0, 10.0]).to(DEVICE)
    for _ in range(n_iter):
        enc.train(); opt.zero_grad()
        F.cross_entropy(enc(X_sub, ei_sub), y_sub, weight=w).backward(); opt.step()

    enc.eval()
    with torch.no_grad():
        emb = F.relu(enc.conv2(F.relu(enc.conv1(X_sub, ei_sub)), ei_sub)).cpu().numpy()

    anom_local = np.array([sub_map[g] for g in train_illicit_idx if g in sub_map])
    if len(anom_local) == 0:
        return rng_np.choice(len(train_illicit_idx), n, replace=True)

    anom_emb = emb[anom_local]
    k = min(10, len(anom_emb) - 1)
    dists_all = np.linalg.norm(anom_emb[:, None] - anom_emb[None, :], axis=2)
    np.fill_diagonal(dists_all, np.inf)
    knn_dist = np.sort(dists_all, axis=1)[:, :k].mean(axis=1)

    full_probs = np.ones(len(train_illicit_idx)) * 1e-8
    full_probs[:len(anom_local)] = knn_dist + 1e-6
    full_probs /= full_probs.sum()
    return rng_np.choice(len(train_illicit_idx), n, replace=True, p=full_probs)


# Borderline 샘플링 (GraphSMOTE-inspired)
def borderline_sampling(n, seed, k=10, hidden=32, n_iter=50):
    rng_np = np.random.RandomState(seed)
    torch.manual_seed(seed)

    licit_sub = rng_np.choice(train_licit_idx, 3000, replace=False)
    sub_idx   = np.concatenate([licit_sub, train_illicit_idx])
    sub_map   = {g: l for l, g in enumerate(sub_idx)}

    sub_s = []; sub_d = []
    for s, d in zip(src_arr, dst_arr):
        if s in sub_map and d in sub_map:
            sub_s.append(sub_map[s]); sub_d.append(sub_map[d])

    if not sub_s:
        return rng_np.choice(len(train_illicit_idx), n, replace=True)

    ei_sub = torch.tensor([sub_s, sub_d], dtype=torch.long).to(DEVICE)
    X_sub  = torch.tensor(X_scaled[sub_idx], dtype=torch.float).to(DEVICE)
    y_sub  = torch.tensor(y_all[sub_idx],    dtype=torch.long).to(DEVICE)

    enc = GCN(X_sub.shape[1], hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    w   = torch.tensor([1.0, 10.0]).to(DEVICE)
    for _ in range(n_iter):
        enc.train(); opt.zero_grad()
        F.cross_entropy(enc(X_sub, ei_sub), y_sub, weight=w).backward(); opt.step()

    enc.eval()
    with torch.no_grad():
        emb = F.relu(enc.conv2(F.relu(enc.conv1(X_sub, ei_sub)), ei_sub)).cpu().numpy()

    anom_local = np.array([sub_map[g] for g in train_illicit_idx if g in sub_map])
    if len(anom_local) == 0:
        return rng_np.choice(len(train_illicit_idx), n, replace=True)

    anom_emb  = emb[anom_local]
    all_label = y_all[sub_idx]

    # k-NN 중 licit 비율 = borderline score
    k = min(k, len(emb) - 1)
    dists = np.linalg.norm(emb[None, :] - anom_emb[:, None], axis=2)
    dists[np.arange(len(anom_local)), anom_local] = np.inf
    knn_idx = np.argpartition(dists, k, axis=1)[:, :k]
    borderline = np.array([(all_label[knn_idx[i]] == 0).mean() for i in range(len(anom_local))])

    # noise(완전 licit 이웃) 제외, borderline 우선, safe 소량 유지
    weights = np.where(borderline >= 1.0, 0.0,
              np.where(borderline > 0.0,  borderline, 0.1))
    weights = weights + 1e-8; weights /= weights.sum()

    full_probs = np.ones(len(train_illicit_idx)) * 1e-8
    full_probs[:len(anom_local)] = weights
    full_probs /= full_probs.sum()
    return rng_np.choice(len(train_illicit_idx), n, replace=True, p=full_probs)


graph_results = {}
print('그래프 전용 baseline 실행...')
print(f'{"방법":<24} {"mean":>7} {"std":>7}  vs AISO기준')
print('-' * 55)
for label, fn in [('PageRank 기반',        pagerank_sampling),
                   ('임베딩 밀도',          embedding_density_sampling),
                   ('Borderline(SMOTE)',    borderline_sampling)]:
    scores = [evaluate_gnn(fn(N_SEEN, sd))['PR-AUC'] for sd in SEEDS_VAL]
    m, s = np.mean(scores), np.std(scores)
    graph_results[label] = scores
    flag = '★' if m > BASELINE_12 else ''
    print(f'  {label:<22} {m:.4f}  {s:.4f}  {m-BASELINE_12:+.4f} {flag}')
'''

try:
    ast.parse(NEW_D011)
    print('syntax OK')
except SyntaxError as e:
    print(f'ERROR line {e.lineno}: {e.msg}')
    lines = NEW_D011.split('\n')
    for i, l in enumerate(lines[max(0, e.lineno-3):e.lineno+2], max(1, e.lineno-2)):
        print(f'  {i}: {repr(l)}')
    sys.exit(1)

with open('aiso_exp1b_features.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c.get('id') == 'd011':
        nb['cells'][i]['source'] = NEW_D011
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        print(f'd011 교체 완료 (셀 인덱스 {i})')
        break

with open('aiso_exp1b_features.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('저장 완료')
