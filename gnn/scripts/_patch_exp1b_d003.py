import json, ast, sys

NEW_D003 = r'''# ---- 25개의 도메인 피처 계산 ----
illicit_labels = y_all.astype(float)

# 양방향 엣지 (showdown 동일: src_bi = [src→dst] + [dst→src])
src_bi = np.concatenate([src_arr, dst_arr])
dst_bi = np.concatenate([dst_arr, src_arr])

# node_degree = total degree (showdown: bincount(ei_np[0]) on bidirectional)
node_degree = np.bincount(src_bi, minlength=N_NODES).astype(float)

# 1홉 illicit 이웃비율 (양방향)
illicit_neigh_sum = np.zeros(N_NODES)
np.add.at(illicit_neigh_sum, src_bi, illicit_labels[dst_bi])
illicit_neigh_ratio = illicit_neigh_sum / np.maximum(node_degree, 1)

# 2홉 illicit 비율 (양방향)
neigh2_illicit = np.zeros(N_NODES)
np.add.at(neigh2_illicit, src_bi, illicit_neigh_ratio[dst_bi])
neigh2_illicit /= np.maximum(node_degree, 1)

hub_fraud = node_degree * illicit_neigh_ratio

# 타임스텝
ts_rate_map = {t: y_all[ts_all==t].mean() for t in np.unique(ts_all)}
node_ts_illicit_rate = np.array([ts_rate_map[t] for t in ts_all])
ts_min, ts_max = ts_all.min(), ts_all.max()
ts_norm = (ts_all - ts_min) / max(ts_max - ts_min, 1)

# local/agg 피처
local_feat = X_scaled[:, 1:94]; agg_feat = X_scaled[:, 94:166]
local_mean = local_feat.mean(1); local_std = local_feat.std(1); local_max = local_feat.max(1)
local_min  = local_feat.min(1)
agg_mean   = agg_feat.mean(1);  agg_std  = agg_feat.std(1);  agg_max  = agg_feat.max(1)
agg_min    = agg_feat.min(1)

# ---- 기본 12개 (양방향 기준 — showdown 동일) ----
domain_raw_12 = np.stack([node_degree, illicit_neigh_ratio, neigh2_illicit,
    node_ts_illicit_rate, local_mean, local_std, local_max,
    agg_mean, agg_std, agg_max, ts_norm, hub_fraud], axis=1)

# ---- 확장 13개 (in/out-degree 등 directed 피처 추가) ----
out_degree = np.bincount(src_arr, minlength=N_NODES).astype(float)
in_degree  = np.bincount(dst_arr, minlength=N_NODES).astype(float)
in_out_ratio = in_degree / np.maximum(out_degree, 1)

neigh_degree_sum = np.zeros(N_NODES)
np.add.at(neigh_degree_sum, src_bi, node_degree[dst_bi])
neigh_degree_mean = neigh_degree_sum / np.maximum(node_degree, 1)

neigh2_hub = np.zeros(N_NODES)
np.add.at(neigh2_hub, src_bi, hub_fraud[dst_bi])
neigh2_hub /= np.maximum(node_degree, 1)

global_illicit_mean = y_all.mean()
ts_burst         = np.abs(node_ts_illicit_rate - global_illicit_mean)
degree_log       = np.log1p(node_degree)
degree_x_ts      = node_degree * node_ts_illicit_rate
illicit_x_neigh2 = illicit_neigh_ratio * neigh2_illicit
hub_x_ts         = hub_fraud * node_ts_illicit_rate
local_range      = local_max - local_min
agg_range        = agg_max - agg_min
local_agg_diff   = np.abs(local_mean - agg_mean)
illicit_isolation = illicit_neigh_ratio - neigh2_illicit

domain_raw_25 = np.concatenate([domain_raw_12, np.stack([
    neigh_degree_mean, neigh2_hub, ts_burst,
    degree_log, degree_x_ts, illicit_x_neigh2, hub_x_ts,
    local_range, agg_range, local_agg_diff,
    illicit_isolation, in_degree, in_out_ratio,
], axis=1)], axis=1)

from sklearn.preprocessing import MinMaxScaler
domain_norm_12 = MinMaxScaler().fit_transform(domain_raw_12)
domain_norm_25 = MinMaxScaler().fit_transform(domain_raw_25)
X_anom_12 = domain_norm_12[train_illicit_idx]
X_anom_25 = domain_norm_25[train_illicit_idx]

feat_names_25 = [
    'degree', 'illicit_neigh', 'neigh2_illicit', 'ts_illicit',
    'local_mean', 'local_std', 'local_max',
    'agg_mean', 'agg_std', 'agg_max', 'ts_norm', 'hub_fraud',
    'neigh_degree_mean', 'neigh2_hub', 'ts_burst',
    'degree_log', 'degree_x_ts', 'illicit_x_neigh2', 'hub_x_ts',
    'local_range', 'agg_range', 'local_agg_diff',
    'illicit_isolation', 'in_degree', 'in_out_ratio',
]
print(f'Dom-12: {X_anom_12.shape}, Dom-25: {X_anom_25.shape}')
print(f'  node_degree 양방향 확인: 예) node0={node_degree[0]:.1f}')
'''

try:
    ast.parse(NEW_D003)
    print('syntax OK')
except SyntaxError as e:
    print(f'ERROR line {e.lineno}: {e.msg}')
    lines = NEW_D003.split('\n')
    for i, l in enumerate(lines[max(0, e.lineno-3):e.lineno+2], max(1, e.lineno-2)):
        print(f'  {i}: {repr(l)}')
    sys.exit(1)

with open('aiso_exp1b_features.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = False
for i, c in enumerate(nb['cells']):
    if c.get('id') == 'd003':
        nb['cells'][i]['source'] = NEW_D003
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        print(f'd003 셀 교체 완료 (셀 인덱스 {i})')
        patched = True
        break

if not patched:
    print('ERROR: d003 셀을 찾지 못했습니다')
    sys.exit(1)

with open('aiso_exp1b_features.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('저장 완료: aiso_exp1b_features.ipynb')
print()
print('다음 단계: 노트북에서 d002→d003→d004→d007 순서로 실행하면')
print('Dom-12 N=12가 0.6041에 근접해야 합니다.')
