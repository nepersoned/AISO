import json

file_path = 'c:\\Users\\kevin\\OneDrive\\Desktop\\AISO\\yelpzip_sampling_abc_aiso.ipynb'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 앞서 잘못 추가했던 2개의 셀을 제거합니다 (가장 마지막 두 셀)
    # 내용이 방금 추가한 코드인지 간단히 확인
    if len(nb['cells']) >= 2 and '각 엣지가 포함된' in ''.join(nb['cells'][-2]['source']) or 'compare_df' in ''.join(nb['cells'][-2]['source']):
        nb['cells'] = nb['cells'][:-2]
        
    code1_source = [
        "from itertools import combinations\n",
        "import pandas as pd\n",
        "\n",
        "# RUR 엣지를 기본으로 고정하고, 나머지 6개 엣지를 선택지로 두어 64개 조합 스윕을 진행합니다.\n",
        "optional_edges = [\n",
        "    ('RSR', lambda d: pairwise_edges(d, 'prod_rating')),\n",
        "    ('DEV_SELF', pairwise_deviation_self_edges),\n",
        "    ('PROD_MON_PFX', pairwise_prod_month_prefix_edges),\n",
        "    ('PROD_SIG', pairwise_prod_sig_edges),\n",
        "    ('SENT_MISMATCH', pairwise_sent_mismatch_edges),\n",
        "    ('CROWD', pairwise_crowd_disagreement_edges)\n",
        "]\n",
        "\n",
        "rur_edge = [('RUR', lambda d: pairwise_edges(d, 'user_id'))]\n",
        "\n",
        "sweep_rows = []\n",
        "print('RUR 고정 + 6개 엣지 대상 64개 조합 스윕 시작...')\n",
        "for r in range(7):  # 0개 선택부터 6개 선택까지\n",
        "    for combo in combinations(optional_edges, r):\n",
        "        edge_builders = rur_edge + list(combo)\n",
        "        combo_names = [name for name, _ in combo]\n",
        "        \n",
        "        # 모델 학습 및 평가 (실행 시간이 조금 소요될 수 있습니다)\n",
        "        res = run_sgc_with_edge_set(sub_raw, edge_builders)\n",
        "        \n",
        "        row = {\n",
        "            'PR-AUC': res['pr_auc'],\n",
        "            'Macro F1': res['macro_f1']\n",
        "        }\n",
        "        # 각 엣지의 포함 여부 기록\n",
        "        for name, _ in optional_edges:\n",
        "            row[name] = (name in combo_names)\n",
        "            \n",
        "        sweep_rows.append(row)\n",
        "\n",
        "sweep_df = pd.DataFrame(sweep_rows)\n",
        "print('64개 조합 스윕 완료!')\n",
        "\n",
        "# 엣지 과적합 증거를 위해 각 엣지가 포함된 32개 조합의 평균 성능 계산\n",
        "avg_results = []\n",
        "for name, _ in optional_edges:\n",
        "    subset = sweep_df[sweep_df[name] == True]\n",
        "    avg_results.append({\n",
        "        'Edge': name,\n",
        "        'Avg PR-AUC': subset['PR-AUC'].mean(),\n",
        "        'Avg Macro F1': subset['Macro F1'].mean()\n",
        "    })\n",
        "\n",
        "avg_df = pd.DataFrame(avg_results)\n",
        "print(\"\\n=== 각 엣지가 포함된 32개 조합의 평균 성능 ===\")\n",
        "print(avg_df.to_string(index=False))\n"
    ]

    code2_source = [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# 성능을 하락시키는 대규모 엣지(RSR, CROWD)를 빨간색으로 시각화\n",
        "colors = ['#d62728' if edge in ['RSR', 'CROWD'] else '#4C72B0' for edge in avg_df['Edge']]\n",
        "labels = [e for e in avg_df['Edge']]\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# 1. 평균 PR-AUC 차트\n",
        "axes[0].bar(labels, avg_df['Avg PR-AUC'], color=colors, edgecolor='white')\n",
        "for i, v in enumerate(avg_df['Avg PR-AUC']):\n",
        "    axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')\n",
        "axes[0].set_title('각 엣지 포함 시 평균 PR-AUC (과적합 증거)', fontweight='bold')\n",
        "axes[0].set_ylabel('PR-AUC')\n",
        "axes[0].set_ylim(0, avg_df['Avg PR-AUC'].max() * 1.15)\n",
        "axes[0].tick_params(axis='x', rotation=15)\n",
        "\n",
        "# 2. 평균 Macro F1 차트\n",
        "axes[1].bar(labels, avg_df['Avg Macro F1'], color=colors, edgecolor='white')\n",
        "for i, v in enumerate(avg_df['Avg Macro F1']):\n",
        "    axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')\n",
        "axes[1].set_title('각 엣지 포함 시 평균 Macro F1 (과적합 증거)', fontweight='bold')\n",
        "axes[1].set_ylabel('Macro F1')\n",
        "axes[1].set_ylim(0, avg_df['Avg Macro F1'].max() * 1.15)\n",
        "axes[1].tick_params(axis='x', rotation=15)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('edge_oversmoothing_evidence.png', bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print('저장 완료: edge_oversmoothing_evidence.png')\n"
    ]

    cell1 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code1_source
    }
    
    cell2 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code2_source
    }

    nb['cells'].extend([cell1, cell2])
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Cells successfully appended using json.")
except Exception as e:
    print(f"Error: {e}")
