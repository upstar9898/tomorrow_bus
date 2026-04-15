# 🚌 Bus Seat Prediction - Experiment Logging Guide

## 📌 개요

본 프로젝트는 버스 잔여좌석 예측 및 혼잡도 분류 모델의 **재현성 확보 및 실험 비교**를 위해
`experiment_logger.py` 기반의 실험 기록 시스템을 사용합니다.

---

## 📂 폴더 구조

```
training/
 ┣ artifacts/
 ┃ ┣ logs/                 # 전체 실험 기록 (experiment_log.csv)
 ┃ ┣ summaries/            # best 결과 저장 (best_results.csv)
 ┃ ┣ reports/              # txt 리포트
 ┃ ┣ confusion_matrices/   # confusion matrix csv
 ┃ ┗ params/               # 모델 파라미터 json
 ┣ models/                 # 학습된 모델 저장
 ┣ utils/
 ┃ ┣ experiment_logger.py
 ┃ ┗ evaluation.py
 ┗ train_*.py              # 학습 코드
```

---

## 📊 실험 결과 저장 방식

### 1. 모든 실행 기록

* `logs/experiment_log.csv`
* 모든 실험 결과 누적 저장

### 2. 최고 성능 결과

* `summaries/best_results.csv`
* 동일 조건 내 best 성능만 유지

### 3. 상세 결과

* txt 리포트
* confusion matrix csv
* hyperparameter json

---

## 🧪 실험 실행 방법

### 1. 모델 학습 후 결과 저장

```python
logger.log_regression_result(...)
logger.log_classification_result(...)
```

---

## 🏷️ 필수 메타 정보

```python
RUNNER = "이름"
DATASET_NAME = "사용 데이터"
DATA_VERSION = "데이터 버전"
SPLIT_VERSION = "데이터 분할 방식"
FEATURE_VERSION = "feature 버전"
MODEL_NAME = "모델명"
MODEL_VERSION = "모델 버전"
```

---

## 📌 저장 예시

```
eunbyeol_lgbm_peak_cls_20260415_173012_classification_report.txt
```

---

## 🎯 목적

* 실험 재현성 확보
* 모델 비교 분석
* 팀원 간 협업 효율 향상

```
```
