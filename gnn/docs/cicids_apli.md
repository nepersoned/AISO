# CICIDS Application Guide — AISO for Network Intrusion Detection

AISO를 **네트워크 침입 탐지(IDS)** 에 적용하는 실전 가이드.  
왜 이 문제가 AISO에 적합한지, 어떻게 파이프라인을 구성하는지,  
결과를 어떻게 해석하는지를 다룬다.

---

## 1. 왜 IDS 문제에 AISO를 쓰는가

### 1.1 구조적 적합성

네트워크 트래픽 이상 탐지는 다음 두 가지 특성을 동시에 가진다:

| 특성 | 설명 |
|------|------|
| **극단적 클래스 불균형** | 정상 트래픽 : 공격 = 수십 만 : 수십 (Heartbleed 11개) |
| **공격 유형의 다중 모달성** | DoS / Probe / R2L / U2R / 웹 공격 등이 특징 공간에서 분리된 클러스터 형성 |

AISO의 **type-mediated 척력** `c_ij = W_i^T M W_j < 0` 은 에이전트를 서로  
다른 클러스터로 밀어낸다 — 즉, 다수 클래스(DoS Hulk)에 수렴하지 않고  
Heartbleed, Infiltration 같은 희귀 공격 클러스터도 탐색하게 된다.

PSO의 global-best 수렴과의 차이:

```
PSO   : 모든 에이전트 → 가장 밀집한 공격 클러스터 (DoS Hulk) 로 수렴
AISO  : 에이전트 그룹별로 다른 공격 유형으로 분산 → 희귀 공격 샘플 방문 유지
```

### 1.2 문제 정의

```
목표: 소수 이상 샘플 집합 X_anom 에서 N개 인덱스를 선택
      → 이것으로 불균형 학습셋을 오버샘플링
      → 훈련된 분류기가 희귀 공격 유형도 탐지하도록 함

평가 기준:
  1. PR-AUC     — 전체 이상 탐지 성능
  2. Rare Recall — holdout 희귀 공격(Heartbleed, Infiltration, SQL Injection) 탐지율
  3. Coverage H  — 오버샘플 집합의 공격 유형 분포 엔트로피
```

---

## 2. 데이터 파이프라인

### 2.1 CICIDS2017 구성

```
원본: 2,830,743 레코드 × 79 피처 × 15 공격 유형
소스: 8개 CSV (요일별 분리)
      Monday-WorkingHours.pcap_ISCX.csv   (정상만)
      Tuesday-WorkingHours.pcap_ISCX.csv  (FTP/SSH Bruteforce)
      Wednesday-workingHours.pcap_ISCX.csv (DoS 계열)
      Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
      Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
      Friday-WorkingHours-Morning.pcap_ISCX.csv (Botnet, PortScan)
      Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
      Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

**인코딩 주의사항**: 라벨 문자열에 비 ASCII 문자 포함

```python
df['Label'] = df['Label'].str.replace('\x96', '-')   # Windows-1252 em-dash
              .str.replace('–', '-')              # Unicode en-dash
              .str.replace('\xa0', ' ')                # non-breaking space
```

### 2.2 전처리 순서

```
1. inf → NaN → 행 제거
2. 상수 컬럼 제거 (nunique ≤ 1)
3. StandardScaler 정규화
4. 희귀 공격 holdout 분리 (샘플 수 < 100 → 테스트 전용)
5. 학습: 50K 정상 + 5K 이상 (10% 불균형)
6. 최적화 샘플러용: PCA(40) 변환
```

### 2.3 실험 시나리오

```
         ┌─────────────────────────────┐
         │  CICIDS2017 전체 (2.83M)    │
         └──────────┬──────────────────┘
                    │ 레이블별 분리
          ┌─────────┴──────────┐
          │ 일반 (≥100개)       │ 희귀 (<100개)
          │                    │ Heartbleed    11
          │ train / test 분리  │ Infiltration  36
          │ (80 / 20)          │ SQL Injection 21
          │                    │    ↓ 전량 test holdout
          │ train subsample    │
          │ 50K normal         │
          │ 5K anomaly   ──────┤
          └─────────┬──────────┘
                    │                    ┌─────────────────┐
            오버샘플러 14종              │  테스트셋        │
                    │                    │ (일반 test +     │
            balanced train  ──────────→ │  희귀 holdout)   │
                    │                    └────────┬────────┘
            GBClassifier                         │
                    │                    PR-AUC, Rare Recall
                    └────────────────────────────┘
```

---

## 3. AISO 설정 가이드

### 3.1 권장 하이퍼파라미터

| 파라미터 | 값 | 이유 |
|----------|----|------|
| `N_AG` | 20 | 79D → PCA-40 공간에서 20개 에이전트로 충분한 커버리지 |
| `N_IT` | 100 | 5K 이상 샘플 탐색에 100 이터레이션이면 수렴 |
| `ALPHA` | 0.2 | 작은 스텝 → snap-to-sample 이후 과잉 이동 방지 |
| `N_TYPES` | 12 | 알려진 공격 유형(약 12종) 과 매핑 |
| `BETA` | 0.08 | 타입 혼합률 낮춤 → 타입 다양성 더 오래 유지 |
| `M_LOW` | −0.5 | 약한 척력 → 분산은 유지하되 탐색 불안정 방지 |
| `W_REPEL` | 2.0 | 인력 상한 |

### 3.2 Adaptive Repulsion 작동 원리

```python
div = mean pairwise distance(agents)
w_r = 1.0 + 3.0 * exp(-div / 0.12)
```

- 에이전트가 몰릴수록(div↓) → w_r↑ → 척력 강화 → 자동 분산
- 이미 분산된 상태(div↑) → w_r ≈ 1.0 → 정상 탐색 모드

### 3.3 PCA 압축의 역할

79D 원본 공간에서 `argmin ||Xn - X[i]||` 를 N_AG × N_IT = 2000번 계산하면 느림.  
PCA(40) 로 압축 후 탐색, 인덱스는 원본 `X_anom` 에 그대로 적용.

```python
# 탐색: PCA 공간
idx = run_aiso(X_anom_pca, N_TARGET, seed)
# 학습 셋 구성: 원본 79D 공간
X_tr, y_tr = build_train(X_norm, X_anom, idx)
```

---

## 4. 결과 해석 가이드

### 4.1 PR-AUC vs Rare Recall 트레이드오프

**PR-AUC** 는 전체 이상 탐지 성능이다.  
데이터의 95%+ 가 DoS 계열이므로, PR-AUC 만 보면 DoS 에 최적화된 방법이 1등.

**Rare Recall** 이 핵심 지표:

```
Heartbleed recall = TP_heartbleed / (TP_heartbleed + FN_heartbleed)
                  = 탐지된 Heartbleed 수 / 전체 Heartbleed 수 (11개)
```

방법별로 이 둘을 함께 봐야 한다:

| 시나리오 | 의미 |
|----------|------|
| PR-AUC ↑ + Rare Recall ↑ | 최선 — 다양한 공격을 고루 탐지 |
| PR-AUC ↑ + Rare Recall ↓ | Mode collapse — DoS 에만 최적화됨 |
| PR-AUC ↓ + Rare Recall ↑ | 희귀 공격 전문, 전체 성능 희생 |

### 4.2 Mode Collapse 진단

`cicids_mode_collapse.png` 에서 확인:

```
Agent Dispersion 곡선이 빠르게 0 에 수렴
    → PSO: global best 로 모든 에이전트 집중 = mode collapse
    → AISO: 척력으로 분산 유지

Visit Count Entropy 곡선이 낮게 유지
    → 방문 샘플이 특정 클러스터에 편중
    → 낮을수록 희귀 공격 방문 확률↓
```

### 4.3 Minority Coverage Entropy 해석

```
H = -Σ p_k · log(p_k)   (k = 공격 유형)

H_max = log(K)   (K개 유형이 완전 균등 분포일 때)
H_min = 0        (단일 유형에 100% 집중)
```

H 가 높을수록 오버샘플된 학습 이상 집합이 다양한 공격 유형을 포함한다.  
H 가 낮으면 학습셋이 DoS Hulk 같은 다수 공격에 편향 → 희귀 공격 미학습.

---

## 5. 다른 네트워크 데이터셋에 적용

### 5.1 NSL-KDD

```
파일: KDDTrain+_20Percent.txt, KDDTest+.txt
특이점: 3개 범주형 피처 (protocol_type, service, flag) → 원-핫 인코딩 필요
       KDDTest+ 에 학습에 없는 novel attack type 포함 → unseen attack 탐지 지표로 활용
```

### 5.2 UNSW-NB15

```
- 9개 공격 유형, 2,540,044 레코드
- NSL-KDD 보다 현대적인 공격 패턴
- 희귀 유형: Worms(44), Shellcode(378)
- 적용 방법: CICIDS 파이프라인과 동일
```

### 5.3 일반화 체크리스트

새 데이터셋에 적용하기 전 확인사항:

- [ ] 공격 유형 분포 확인 → rare_types 기준(100개 미만) 조정
- [ ] 피처 수에 따라 PCA 컴포넌트 수 조정 (권장: 원래 분산의 90% 보존)
- [ ] N_TYPES 를 알려진 공격 유형 수와 맞춤 (범주 수 ± 2 정도)
- [ ] 학습 이상 비율 확인 → N_SEEN / N_NORMAL 이 너무 낮으면 SMOTE 비교군 불안정

---

## 6. 실패 패턴과 대응

### 6.1 희귀 recall 이 전 방법에서 0인 경우

원인: holdout 샘플이 너무 적어서 특징이 학습셋과 너무 다름.

```python
# holdout 기준 완화: <100 → <200
rare_types = set(val_counts[(val_counts < 200) & (val_counts.index != 'BENIGN')].index)
```

### 6.2 AISO 가 PSO 보다 낮은 경우

원인 1: 공격 유형이 1-2개뿐 → 다중 모달성 없음 → AISO 척력이 오히려 방해.  
원인 2: N_TYPES 가 실제 클러스터 수보다 훨씬 큰 경우.

대응:
```python
N_TYPES = max(3, len(atk_imbal_a.unique()))  # 실제 유형 수에 맞춤
```

### 6.3 SMOTE 가 실패하는 경우

원인: 이상 샘플 수가 너무 적어 k-NN 이웃을 못 찾음.

```python
# k_neighbors 줄이기
SMOTE(random_state=SEED, k_neighbors=min(3, N_SEEN-1))
```

---

## 7. 확장: 실시간 IDS 적용 시나리오

AISO 오버샘플링을 실시간 IDS 파이프라인에 통합하는 개념적 흐름:

```
[트래픽 수집] → [피처 추출 (79D)] → [StandardScaler]
      ↓
[이상 탐지 분류기]
      ↓ (이상 판정된 샘플)
[X_anom 업데이트 버퍼]
      ↓ (버퍼가 임계 크기 도달 시)
[AISO 오버샘플링 재실행]
      ↓
[분류기 재학습]
      ↓
[배포] → 다음 사이클
```

**주의**: AISO 는 동적 환경(이동하는 최적점)에 취약하다 (Section 6.3 of README).  
실시간 재학습 주기는 공격 패턴 변화 속도에 맞춰 조정 필요.  
패턴이 빠르게 변하면 SPSO 기반 샘플러를 대신 사용.
