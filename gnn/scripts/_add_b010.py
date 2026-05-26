import json, ast

with open('aiso_exp3_entropy.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cell_code = r'''# ---- 구조적 타입 초기화 실험 ----
# W를 one-hot 역할 배정, M을 반발/인력 구조화
# -> 에이전트가 처음부터 다른 특성 축을 담당하도록 강제

def run_aiso_structured_gnn(X_a, n, seed, n_types=12,
                             beta_slow=0.02, m_diag=-2.0, m_offdiag=0.3):
    rng2 = np.random.RandomState(seed)
    Xn = _norm_gnn(X_a); N_a, D = Xn.shape

    # one-hot W 초기화 (에이전트별 역할 배정)
    W = np.zeros((GNN_N_AG, n_types))
    for i in range(GNN_N_AG):
        W[i, i % n_types] = 1.0
    noise = rng2.dirichlet(np.ones(n_types), GNN_N_AG) * 0.05
    W = W + noise
    W /= W.sum(1, keepdims=True)

    # 구조적 M (같은 타입 반발, 다른 타입 인력)
    M = np.full((n_types, n_types), m_offdiag)
    np.fill_diagonal(M, m_diag)

    X = Xn[rng2.choice(N_a, GNN_N_AG, replace=True)].copy().astype(float)
    visit = np.zeros(N_a)
    w_r = GNN_W_REPEL

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
            W_new = (1-beta_slow)*W[i] + beta_slow*W[bja]
            W_new /= W_new.sum()
            W[i] = W_new

    probs = visit + 1.0; probs /= probs.sum()
    return rng2.choice(N_a, n, replace=True, p=probs)


configs = [
    {"beta_slow": 0.02, "m_diag": -2.0, "m_offdiag": 0.3,  "label": "onehot b=0.02 M(-2,+0.3)"},
    {"beta_slow": 0.05, "m_diag": -2.0, "m_offdiag": 0.3,  "label": "onehot b=0.05 M(-2,+0.3)"},
    {"beta_slow": 0.08, "m_diag": -2.0, "m_offdiag": 0.3,  "label": "onehot b=0.08 M(-2,+0.3)"},
    {"beta_slow": 0.02, "m_diag": -1.0, "m_offdiag": 0.5,  "label": "onehot b=0.02 M(-1,+0.5)"},
    {"beta_slow": 0.02, "m_diag": -3.0, "m_offdiag": 0.1,  "label": "onehot b=0.02 M(-3,+0.1)"},
]

struct_results = {}
print(f"{'설정':<32} {'mean':>7} {'std':>7}  vs 기존")
print("-" * 60)
for cfg in configs:
    lbl = cfg["label"]
    scores = []
    for sd in SEEDS_VAL:
        idx = run_aiso_structured_gnn(X_anom_domain, N_SEEN, sd,
                                      n_types=N_TYPES_DOM,
                                      beta_slow=cfg["beta_slow"],
                                      m_diag=cfg["m_diag"],
                                      m_offdiag=cfg["m_offdiag"])
        scores.append(evaluate_gnn(idx)["PR-AUC"])
    m, s = np.mean(scores), np.std(scores)
    struct_results[lbl] = scores
    flag = "★" if m > KNOWN_BASELINE else ""
    print(f"  {lbl:<30} {m:.4f} {s:.4f}  {m-KNOWN_BASELINE:+.4f} {flag}")

print()
print(f"  기존 AISO(Dom, N=12)           {KNOWN_BASELINE:.4f}  (기준)")
best_lbl = max(struct_results, key=lambda k: np.mean(struct_results[k]))
best_m   = np.mean(struct_results[best_lbl])
print(f"  최고: {best_lbl}  -> {best_m:.4f}")
'''

try:
    ast.parse(cell_code)
    print('b010: syntax OK')
except SyntaxError as e:
    print(f'b010 ERROR line {e.lineno}: {e.msg}')
    lines = cell_code.split('\n')
    for i, l in enumerate(lines[max(0, e.lineno-3):e.lineno+2], e.lineno-2):
        print(f'  {i}: {repr(l)}')
    exit(1)

new_cell = {
    'cell_type': 'code',
    'id': 'b010',
    'metadata': {},
    'source': cell_code,
    'outputs': [],
    'execution_count': None
}

nb['cells'].append(new_cell)

with open('aiso_exp3_entropy.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'저장 완료 - 총 {len(nb["cells"])} 셀')
