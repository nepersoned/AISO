import json

file_path = 'c:\\Users\\kevin\\OneDrive\\Desktop\\AISO\\yelpzip_sampling_abc_aiso.ipynb'
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code1_source = [
        "import pandas as pd\n",
        "\n",
        "# 64개 조합 스윕 결과가 저장된 데이터프레임의 변수명을 'compare_df'라고 가정합니다.\n",
        "# 만약 64개 조합 결과가 다른 변수에 있다면 아래 compare_df를 변경하세요.\n",
        "custom_edges = ['R-S-R', 'DeviationSelf', 'PROD_MON_PFX', 'PROD_SIG', 'SentimentMismatch', 'CrowdDisagreement']\n",
        "\n",
        "avg_results = []\n",
        "\n",
        "for edge in custom_edges:\n",
        "    if edge in compare_df.columns:\n",
        "        # 데이터가 True/False, 1/0, 'O'/'X' 등 어떤 형태인지에 따라 조건 수정 필요\n",
        "        subset = compare_df[compare_df[edge] == True] \n",
        "        if subset.empty:\n",
        "            subset = compare_df[compare_df[edge] == 1]\n",
        "        if subset.empty:\n",
        "            subset = compare_df[compare_df[edge] == 'O']\n",
        "            \n",
        "        avg_auc = subset['PR-AUC'].mean() if 'PR-AUC' in subset.columns else subset['pr_auc'].mean()\n",
        "        avg_f1 = subset['Macro F1'].mean() if 'Macro F1' in subset.columns else subset['macro_f1'].mean()\n",
        "        \n",
        "        avg_results.append({\n",
        "            'Edge': edge,\n",
        "            'Avg PR-AUC': avg_auc,\n",
        "            'Avg Macro F1': avg_f1\n",
        "        })\n",
        "\n",
        "if avg_results:\n",
        "    avg_df = pd.DataFrame(avg_results)\n",
        "    print(\"=== 각 엣지가 포함된 32개 조합의 평균 성능 ===\")\n",
        "    print(avg_df.to_string(index=False))\n",
        "else:\n",
        "    print(\"compare_df에 커스텀 엣지 컬럼이 존재하지 않거나 조건에 맞는 데이터가 없습니다. 변수명이나 데이터 형식을 확인해 주세요.\")\n"
    ]

    code2_source = [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "if 'avg_df' in locals() and not avg_df.empty:\n",
        "    colors = ['#d62728' if edge in ['R-S-R', 'CrowdDisagreement'] else '#4C72B0' for edge in avg_df['Edge']]\n",
        "    labels = [e.replace('SentimentMismatch', 'SentMis') for e in avg_df['Edge']]\n",
        "\n",
        "    fig, axes = plt.subplots(2, 1, figsize=(10, 10))\n",
        "\n",
        "    # 1. 평균 PR-AUC 차트\n",
        "    axes[0].bar(labels, avg_df['Avg PR-AUC'], color=colors, edgecolor='white')\n",
        "    for i, v in enumerate(avg_df['Avg PR-AUC']):\n",
        "        if pd.notna(v):\n",
        "            axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')\n",
        "    axes[0].set_title('각 엣지 포함 시 평균 PR-AUC (과적합 증거)', fontweight='bold')\n",
        "    axes[0].set_ylabel('PR-AUC')\n",
        "    axes[0].set_ylim(0, avg_df['Avg PR-AUC'].max() * 1.1)\n",
        "\n",
        "    # 2. 평균 Macro F1 차트\n",
        "    axes[1].bar(labels, avg_df['Avg Macro F1'], color=colors, edgecolor='white')\n",
        "    for i, v in enumerate(avg_df['Avg Macro F1']):\n",
        "        if pd.notna(v):\n",
        "            axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')\n",
        "    axes[1].set_title('각 엣지 포함 시 평균 Macro F1 (과적합 증거)', fontweight='bold')\n",
        "    axes[1].set_ylabel('Macro F1')\n",
        "    axes[1].set_ylim(0, avg_df['Avg Macro F1'].max() * 1.1)\n",
        "\n",
        "    plt.tight_layout()\n",
        "    plt.savefig('edge_oversmoothing_evidence.png', bbox_inches='tight')\n",
        "    plt.show()\n",
        "\n",
        "    print('저장 완료: edge_oversmoothing_evidence.png')\n"
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
