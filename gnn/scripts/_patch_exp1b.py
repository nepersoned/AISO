import json, ast, sys

NEW_D004 = r'''SEED = 42

# bidirectional edge_index (showdown 동일)
edge_index = torch.tensor(
    [np.concatenate([src_arr, dst_arr]),
     np.concatenate([dst_arr, src_arr])],
    dtype=torch.long
)
ei_np = edge_index.numpy()

# sel_n 고정 1회 (showdown: SEED=42)
sel_n = np.random.RandomState(SEED).choice(train_licit_idx, N_LICIT, replace=False)

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

import copy

def evaluate_gnn(sel_pool_idx, label=''):
    """Exact showdown transductive protocol"""
    uniq_anom       = np.unique(sel_pool_idx)
    sel_anom_global = train_illicit_idx[uniq_anom]
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

    n0 = int((y_all[sub_nodes][tr_mask.cpu().numpy()] == 0).sum())
    n1 = int((y_all[sub_nodes][tr_mask.cpu().numpy()] == 1).sum())
    cw = torch.tensor([1.0, n0 / max(n1, 1)], dtype=torch.float).to(DEVICE)

    torch.manual_seed(SEED)
    model = GCN(X_scaled.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_loss, best_state, patience = float('inf'), None, 0
    for ep in range(200):
        model.train(); opt.zero_grad()
        out  = model(sub_X, sub_ei)
        loss = F.cross_entropy(out[tr_mask], sub_y[tr_mask], weight=cw)
        loss.backward(); opt.step()
        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = copy.deepcopy(model.state_dict())
            patience   = 0
        else:
            patience += 1
        if patience >= 20:
            break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        prob = F.softmax(model(sub_X, sub_ei), dim=1)[:, 1].cpu().numpy()
    pred   = (prob >= 0.5).astype(int)
    te_cpu = te_mask.cpu().numpy()
    res = {
        'PR-AUC': average_precision_score(y_all[sub_nodes][te_cpu], prob[te_cpu]),
        'F1':     f1_score(y_all[sub_nodes][te_cpu], pred[te_cpu], zero_division=0),
        'AUC':    roc_auc_score(y_all[sub_nodes][te_cpu], prob[te_cpu]),
    }
    if label:
        print(f'  {label:<28} PR-AUC={res["PR-AUC"]:.4f}  F1={res["F1"]:.4f}')
    return res

# sanity check
SEEDS_VAL = [0, 7, 42, 77, 123]
print('evaluate_gnn 준비 완료 (showdown 트랜스덕티브 프로토콜)')
print(f'  sel_n 고정: {len(sel_n):,}개  |  edge_index 양방향: {edge_index.shape[1]:,}개')
print()
print('--- sanity check: AISO(Dom-12 N=12) SEED=42 ---')
'''

try:
    ast.parse(NEW_D004)
    print('syntax OK')
except SyntaxError as e:
    print(f'ERROR line {e.lineno}: {e.msg}')
    lines = NEW_D004.split('\n')
    for i, l in enumerate(lines[max(0, e.lineno-3):e.lineno+2], max(1, e.lineno-2)):
        print(f'  {i}: {repr(l)}')
    sys.exit(1)

with open('aiso_exp1b_features.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# d004 셀 교체
for i, c in enumerate(nb['cells']):
    if c.get('id') == 'd004':
        nb['cells'][i]['source'] = NEW_D004
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        print(f'd004 셀 교체 완료 (셀 인덱스 {i})')
        break

with open('aiso_exp1b_features.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('저장 완료: aiso_exp1b_features.ipynb')
