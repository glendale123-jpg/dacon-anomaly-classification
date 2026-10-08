# 이상신호 감지 기반 비정상 작동 진단

2026-1 머신러닝 팀과제 (4인: 권지원 · 이나경 · 이성진 · 이재하) · DACON 대회 데이터

**Private Macro-F1 0.8598 → 0.8744 (+0.0146)** · 4명이 50회 이상 실험

## 문제

- 비식별 센서 변수 52개(`X_01`~`X_52`)로 설비 상태 **21개 클래스**를 분류하는 문제
- 학습 데이터 21,693행, 클래스 균등, 평가지표 Macro-F1
- 출발점: DACON 공개 6등 코드 (TabNet + MLP-ReLU + MLP-GELU + AutoGluon 균등 앙상블 + Hungarian 후처리)

## 베이스라인의 한계 → 개선

| 한계 | 개선 |
|---|---|
| 피처 엔지니어링 없음 (원본 52개만 사용) | **LDA 20차원 추가** (52 → 72차원, fold 안에서만 fit) |
| 전체 데이터로 scaler fit → 검증 누수 | **Fold-safe scaling** (train fold로만 fit) |
| 모델마다 검증 방식·지표가 다름 (1-split·accuracy / 5-fold·F1 / log_loss) | 전 모델 **5-fold StratifiedKFold + Macro-F1** 통일 |
| 앙상블 가중치 0.25 고정, ReLU·GELU가 거의 같은 모델 | **OOF 기반 가중치 탐색**(Nelder-Mead) + ReLU를 **SNN**으로 교체 |
| `best_state`가 참조 복사돼 마지막 epoch 가중치가 저장되는 버그 | 텐서 복사로 수정, EarlyStopping 추가 |

ReLU와 GELU MLP는 활성화 함수만 다른 같은 구조라 예측이 겹쳤고, 가중치 탐색에서 ReLU에 0.14만 배정됐습니다. 정규화 방식 자체가 다른 SNN(SELU + AlphaDropout, BatchNorm 없음)으로 바꾸자 가중치가 고르게 분산되며 점수가 올랐습니다 — 모델을 **추가**한 게 아니라 **치환**해서 다양성을 확보한 경우입니다.

## 최종 파이프라인

```
52개 센서 → LDA 20차원 추가(fold-safe) → TabNet · SNN(3 seed) · MLP-GELU · AutoGluon
  → OOF 가중치 탐색(3모델) × 0.8 + AutoGluon × 0.2 → Hungarian 균등 배분 → 제출
```

코드: [code/최종코드.ipynb](code/최종코드.ipynb) (Google Colab, GPU)

## 내 역할 (이성진)

피처 엔지니어링 탐색과 후처리 검증을 맡았습니다.

| 실험 | 내용 | 결과 |
|---|---|---|
| FE 1차 | 행 통계 7개 + 핵심 센서 조합 17개 | LightGBM OOF F1 +0.002 |
| FE 2차 | 상위 5개 센서 비율·차·곱 45개 | 개선 없음 — feature importance는 분할 빈도일 뿐 성능 기여도가 아님 |
| FE 최종 | FE 1차 + `X_01` 제거 (68개) | OOF F1 +0.003 — 빼는 것도 개선 |
| 후처리 | Argmax vs Hungarian | Argmax 0.8146 / Hungarian 0.8621 → **Hungarian 필수** 근거 |
| 정규화 | Train+Test 통합 정규화 | Private 0.8635 (+0.004) |

이 실험들의 공통 방법론이 팀 최종 코드의 기반이 됐고, 이후 피처를 더 넣는 시도는 모두 기준을 넘지 못해 **"LDA가 이미 잡은 신호와 겹치는 피처는 손해"** 라는 결론을 냈습니다.

## 자료

- [최종 발표 자료 (PDF)](report/DACON_보고서_최종.pdf) — 팀원별 실험 결과와 실패 원인 분석 포함
- 한계: OOF 0.8780 vs Private 0.8744 갭, AutoGluon 블록의 LDA 누수 가능성, 클래스 0·9·15 혼동 미해결

## 실행

DACON 대회 데이터(`train.csv`, `test.csv`)는 포함하지 않았습니다. 노트북의 `DATA_DIR`을 데이터 위치로 바꾸고 Colab(GPU)에서 실행합니다.

```bash
pip install autogluon.tabular pytorch-tabnet torch scikit-learn pandas numpy scipy
```
