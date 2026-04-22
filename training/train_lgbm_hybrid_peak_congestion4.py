# =========================================================
# [통합 최종 실행본]
# LightGBM 회귀 + 출퇴근 시간대 전용 4단계 혼잡도 분류기
# + 만차 여부 이진 분류
# + 저장된 encoder / pattern stats 로드 사용
# + experiment_logger 연동
# =========================================================

# =========================================================
# 1. 표준 라이브러리 import
# =========================================================
import json
import os
import sys
import time

# =========================================================
# 2. 외부 라이브러리 import
# =========================================================
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =========================================================
# 3. 사용자 정의 모듈 import
# =========================================================
from utils.config import (
    CONGESTION_CLASS_LABELS,
    DATASET_NAME,
    DATA_VERSION,
    FEATURE_VERSION,
    FULL_BINARY_LABELS,
    FULL_BINARY_THRESHOLD,
    FULL_MODEL_NAME,
    LABEL_DEFINITION_DETAIL,
    LABEL_DEFINITION_NAME,
    MODEL_VERSION,
    PEAK_CONGESTION_MODEL_NAME,
    PEAK_THRESHOLDS,
    REG_MODEL_NAME,
    RUNNER,
    SPLIT_VERSION,
)
from utils.encoder_utils import load_label_encoders, transform_with_encoders
from utils.experiment_logger import ExperimentLogger
from utils.feature_utils import (
    MAX_SEAT,
    get_feature_cols,
    prepare_training_base_dataframe,
    split_by_date,
)
from utils.inference_utils import run_service_inference
from utils.pattern_stats_utils import load_pattern_stats, merge_pattern_features
from utils.model_save_utils import save_model_artifacts

# =========================================================
# 4. 프로젝트 경로 및 출력 폴더 설정
# =========================================================
# 현재 실행 파일 기준 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 경로 (예: training 상위 폴더)
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# 데이터 폴더 경로
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 결과물 저장 루트
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs_peak_v2")

# 실험 로그 / 통계 산출물 저장 폴더
ARTIFACT_DIR = os.path.join(OUTPUT_ROOT, "artifacts")

# 모델 저장 상위 폴더
MODEL_BASE_DIR = os.path.join(OUTPUT_ROOT, "models")

# 실제 현재 실험 모델 저장 폴더
MODEL_DIR = os.path.join(MODEL_BASE_DIR, "lgbm_hybrid_peak_congestion4")

# 폴더가 없으면 자동 생성
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODEL_BASE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# utils import가 꼬일 수 있는 환경을 대비해 현재 실행 경로를 sys.path에 추가
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# 실험 결과 기록용 로거 객체 생성
logger = ExperimentLogger(artifact_dir=ARTIFACT_DIR)

print("MODEL_DIR real path:", os.path.abspath(MODEL_DIR))
print(
    "stid encoder file exists?:",
    os.path.exists(os.path.join(MODEL_DIR, "stid_encoder.pkl"))
)


# =========================================================
# 5. 학습 데이터 파일 경로 설정
# =========================================================
file_path = os.path.join(DATA_DIR, "bus_all_raw_weather_260421.csv")

# 파일 존재 여부 확인
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"데이터 파일을 찾을 수 없습니다.\n"
        f"확인한 경로: {file_path}\n"
        f"현재 PROJECT_ROOT: {PROJECT_ROOT}\n"
        f"현재 DATA_DIR: {DATA_DIR}"
    )

print(f"[INFO] PROJECT_ROOT : {PROJECT_ROOT}")
print(f"[INFO] DATA_DIR      : {DATA_DIR}")
print(f"[INFO] file_path     : {file_path}")
print(f"[INFO] OUTPUT_ROOT   : {OUTPUT_ROOT}")
print(f"[INFO] MODEL_DIR     : {MODEL_DIR}")


# =========================================================
# 6. 원본 데이터 불러오기
# =========================================================
# ID 계열 컬럼은 문자열로 고정해서 읽는다.
# 그래야 앞자리 0 유실, dtype 혼합 문제를 줄일 수 있다.
df = pd.read_csv(
    file_path,
    dtype={
        "busRouteId": str,
        "stId": str,
        "arsId": str,
    },
    low_memory=False
)

# 시간 컬럼 datetime 변환
df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")


# =========================================================
# 7. 공통 전처리 및 파생변수 생성
# =========================================================
# 이 함수 내부에서 다음이 처리된다고 가정:
# - 기본 결측/형식 정리
# - 날씨 관련 전처리
# - 시간 파생변수 생성
# - 주기형(sin/cos) feature 생성
# - target 및 관련 feature 생성
df = prepare_training_base_dataframe(df)

print("총 데이터 수:", len(df))
print("\n[remaining_seat 기초통계]")
print(df["remaining_seat"].describe())

print("\n[congestion_class 전체 분포 - 4단계]")
print(df["congestion_class"].value_counts(normalize=True).sort_index())

print("\n[is_full_target 전체 분포 - 이진]")
print(df["is_full_target"].value_counts(normalize=True).sort_index())

print("\n[is_rain 분포]")
print(df["is_rain"].value_counts())


# =========================================================
# 8. 날짜 기준 train / valid / test 분할
# =========================================================
# 시간 누수를 막기 위해 날짜 단위로 분리
train_df, valid_df, test_df, split_info = split_by_date(df)

# 서비스 추론용 원본 테스트셋 백업
test_df_raw = test_df.copy()

unique_dates = split_info["unique_dates"]
train_dates = split_info["train_dates"]
valid_dates = split_info["valid_dates"]
test_dates = split_info["test_dates"]

print("\n사용 날짜들:", unique_dates)
print("총 사용 날짜 수:", len(unique_dates))
print("train dates:", train_dates[0], "~", train_dates[-1], f"({len(train_dates)}일)")
print("valid dates:", valid_dates[0], "~", valid_dates[-1], f"({len(valid_dates)}일)")
print("test dates :", test_dates[0], "~", test_dates[-1], f"({len(test_dates)}일)")


# =========================================================
# 9. 범주형 ID 인코딩
# =========================================================
# 학습 때 저장해둔 encoder를 불러온다.
# train은 strict=True로 학습 클래스만 허용,
# valid/test는 strict=False로 unseen 값을 -1 등으로 처리하도록 설계된 것으로 보인다.
encoders = load_label_encoders(MODEL_DIR)

train_df = transform_with_encoders(train_df, encoders, strict=True)
valid_df = transform_with_encoders(valid_df, encoders, strict=False)
test_df = transform_with_encoders(test_df, encoders, strict=False)

print("valid unseen route 개수:", (valid_df["route_enc"] == -1).sum())
print("valid unseen stid 개수 :", (valid_df["stid_enc"] == -1).sum())
print("valid unseen arsid 개수:", (valid_df["arsid_enc"] == -1).sum())

print("test unseen route 개수 :", (test_df["route_enc"] == -1).sum())
print("test unseen stid 개수  :", (test_df["stid_enc"] == -1).sum())
print("test unseen arsid 개수 :", (test_df["arsid_enc"] == -1).sum())


# =========================================================
# 10. 패턴 통계 로드 및 feature merge
# =========================================================
# 기존에 저장해둔 pattern stats를 불러와서
# route / stop / 시간대 기반 통계 feature를 붙인다.
stats_dict, pattern_meta = load_pattern_stats(ARTIFACT_DIR)

train_df = merge_pattern_features(train_df, stats_dict, pattern_meta)
valid_df = merge_pattern_features(valid_df, stats_dict, pattern_meta)
test_df = merge_pattern_features(test_df, stats_dict, pattern_meta)

print("\n[패턴 feature merge 완료]")
print("train_df:", train_df.shape)
print("valid_df:", valid_df.shape)
print("test_df :", test_df.shape)


# =========================================================
# 11. 모델 입력 feature 정의
# =========================================================
FEATURE_COLS = get_feature_cols()
missing_cols = [col for col in FEATURE_COLS if col not in train_df.columns]
print("\n[FEATURE_COLS 누락 확인]")
print(missing_cols)

# =========================================================
# 12. 학습 / 평가용 데이터셋 구성
# =========================================================
# ---------- 회귀 ----------
X_train = train_df[FEATURE_COLS]
X_valid = valid_df[FEATURE_COLS]
X_test = test_df[FEATURE_COLS]

y_train_reg = train_df["remaining_seat"]
y_valid_reg = valid_df["remaining_seat"]
y_test_reg = test_df["remaining_seat"]

# ---------- 만차 여부 이진 분류 ----------
y_train_full = train_df["is_full_target"]
y_valid_full = valid_df["is_full_target"]
y_test_full = test_df["is_full_target"]

# ---------- 출퇴근 시간대 전용 4클래스 혼잡도 분류 ----------
peak_train_df = train_df[train_df["is_peak"] == 1].copy()
peak_valid_df = valid_df[valid_df["is_peak"] == 1].copy()
peak_test_df = test_df[test_df["is_peak"] == 1].copy()

X_train_peak = peak_train_df[FEATURE_COLS]
X_valid_peak = peak_valid_df[FEATURE_COLS]
X_test_peak = peak_test_df[FEATURE_COLS]

y_train_peak_cong = peak_train_df["congestion_class"]
y_valid_peak_cong = peak_valid_df["congestion_class"]
y_test_peak_cong = peak_test_df["congestion_class"]

print("\n[전체 데이터 크기]")
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("X_test :", X_test.shape)

print("\n[출퇴근 전용 분류 데이터 크기]")
print("X_train_peak:", X_train_peak.shape)
print("X_valid_peak:", X_valid_peak.shape)
print("X_test_peak :", X_test_peak.shape)

print("\n[출퇴근 전용 4클래스 분포 - train]")
print(y_train_peak_cong.value_counts(normalize=True).sort_index())

print("\n[출퇴근 전용 4클래스 분포 - valid]")
print(y_valid_peak_cong.value_counts(normalize=True).sort_index())

print("\n[출퇴근 전용 4클래스 분포 - test]")
print(y_test_peak_cong.value_counts(normalize=True).sort_index())


# =========================================================
# 13. 회귀 모델 학습
# =========================================================
# 목적:
# - 실제 남은 좌석 수(remaining_seat)를 직접 예측
lgbm_reg = LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=127,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
)

start = time.time()
lgbm_reg.fit(
    X_train,
    y_train_reg,
    eval_set=[(X_train, y_train_reg), (X_valid, y_valid_reg)],
    eval_metric="l1",
)
reg_train_time = time.time() - start


# =========================================================
# 14. 출퇴근 시간대 전용 4단계 혼잡도 분류 모델 학습
# =========================================================
# 목적:
# - 출퇴근 시간대(is_peak == 1) 데이터만 따로 사용하여
#   4단계 혼잡도 클래스를 분류
lgbm_peak_congestion_cls = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=700,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=127,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

start = time.time()
lgbm_peak_congestion_cls.fit(
    X_train_peak,
    y_train_peak_cong,
    eval_set=[(X_train_peak, y_train_peak_cong), (X_valid_peak, y_valid_peak_cong)],
    eval_metric="multi_logloss",
)
peak_cls_train_time = time.time() - start


# =========================================================
# 15. 만차 여부 이진 분류 모델 학습
# =========================================================
# 목적:
# - 만차 / 비만차 여부를 별도 분류기로 예측
lgbm_full_cls = LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=63,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

start = time.time()
lgbm_full_cls.fit(
    X_train,
    y_train_full,
    eval_set=[(X_train, y_train_full), (X_valid, y_valid_full)],
    eval_metric="binary_logloss",
)
full_cls_train_time = time.time() - start


# =========================================================
# 16. threshold 적용 예측 함수
# =========================================================
def predict_peak_congestion_with_thresholds(proba: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """
    출퇴근 시간대 4클래스 혼잡도 분류 확률값에 대해
    커스텀 threshold를 적용해 최종 클래스를 결정한다.

    Parameters
    ----------
    proba : np.ndarray
        predict_proba 결과. shape = (N, 4)
    thresholds : list[float]
        클래스별 기준 threshold.
        예: [class0_threshold, class1_threshold, class2_threshold]

    Returns
    -------
    np.ndarray
        최종 예측 클래스 배열
    """
    preds = []

    for row in proba:
        p0, p1, p2, p3 = row

        if p0 >= thresholds[0]:
            preds.append(0)
        elif p1 >= thresholds[1]:
            preds.append(1)
        elif p2 >= thresholds[2]:
            preds.append(2)
        else:
            preds.append(3)

    return np.array(preds)


# =========================================================
# 17. 평가 함수 정의
# =========================================================
def evaluate_regression(model, X, y, name="dataset"):
    """
    회귀 모델 평가 함수
    - 예측값은 물리적으로 가능한 좌석 범위(0 ~ MAX_SEAT)로 clip
    - MAE / RMSE / R2 출력
    """
    pred = np.clip(model.predict(X), 0, MAX_SEAT)

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)

    print(f"\n[{name}]")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return pred


def evaluate_multiclass(y_true, y_pred, name="dataset"):
    """
    다중분류 평가 함수
    - ACC
    - Macro F1
    - Weighted F1
    - Confusion Matrix
    - Classification Report
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print(f"\n[{name}]")
    print(f"ACC        : {acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(cm)
    print(report)

    return y_pred


def evaluate_binary(y_true, y_pred, name="dataset"):
    """
    이진분류 평가 함수
    - ACC
    - Macro F1
    - Weighted F1
    - Confusion Matrix
    - Classification Report
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)

    print(f"\n[{name}]")
    print(f"ACC        : {acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(cm)
    print(report)

    return y_pred


# =========================================================
# 18. 모델 평가
# =========================================================
# ---------- 회귀 ----------
valid_pred_reg = evaluate_regression(lgbm_reg, X_valid, y_valid_reg, "VALID REG")
test_pred_reg = evaluate_regression(lgbm_reg, X_test, y_test_reg, "TEST REG")

# ---------- 출퇴근 시간대 4클래스 혼잡도 ----------
valid_peak_proba = lgbm_peak_congestion_cls.predict_proba(X_valid_peak)
test_peak_proba = lgbm_peak_congestion_cls.predict_proba(X_test_peak)

valid_pred_peak_cong = predict_peak_congestion_with_thresholds(valid_peak_proba, PEAK_THRESHOLDS)
test_pred_peak_cong = predict_peak_congestion_with_thresholds(test_peak_proba, PEAK_THRESHOLDS)

valid_pred_peak_cong = evaluate_multiclass(
    y_valid_peak_cong,
    valid_pred_peak_cong,
    "VALID PEAK CONGESTION CLS 4CLASS"
)
test_pred_peak_cong = evaluate_multiclass(
    y_test_peak_cong,
    test_pred_peak_cong,
    "TEST PEAK CONGESTION CLS 4CLASS"
)

# ---------- 만차 여부 이진 분류 ----------
valid_full_prob = lgbm_full_cls.predict_proba(X_valid)[:, 1]
test_full_prob = lgbm_full_cls.predict_proba(X_test)[:, 1]

valid_pred_full = (valid_full_prob >= FULL_BINARY_THRESHOLD).astype(int)
test_pred_full = (test_full_prob >= FULL_BINARY_THRESHOLD).astype(int)

valid_pred_full = evaluate_binary(
    y_valid_full,
    valid_pred_full,
    "VALID FULL BINARY CLS"
)
test_pred_full = evaluate_binary(
    y_test_full,
    test_pred_full,
    "TEST FULL BINARY CLS"
)


# =========================================================
# 19. experiment_logger 결과 저장
# =========================================================
# ---------- 회귀 결과 저장 ----------
logger.log_regression_result(
    y_true=y_valid_reg,
    y_pred=valid_pred_reg,
    runner=RUNNER,
    model_name=REG_MODEL_NAME,
    model_version=MODEL_VERSION,
    dataset_name=DATASET_NAME,
    data_version=DATA_VERSION,
    split_version=SPLIT_VERSION,
    feature_version=FEATURE_VERSION,
    hyperparams=lgbm_reg.get_params(),
    train_time_sec=reg_train_time,
    notes="VALID regression result / clip 0~45 적용"
)

logger.log_regression_result(
    y_true=y_test_reg,
    y_pred=test_pred_reg,
    runner=RUNNER,
    model_name=REG_MODEL_NAME,
    model_version=MODEL_VERSION,
    dataset_name=DATASET_NAME,
    data_version=DATA_VERSION,
    split_version=SPLIT_VERSION,
    feature_version=FEATURE_VERSION,
    hyperparams=lgbm_reg.get_params(),
    train_time_sec=reg_train_time,
    notes="TEST regression result / clip 0~45 적용"
)

# ---------- 출퇴근 시간대 4클래스 혼잡도 분류 결과 저장 ----------
logger.log_classification_result(
    y_true=y_valid_peak_cong,
    y_pred=valid_pred_peak_cong,
    runner=RUNNER,
    model_name=PEAK_CONGESTION_MODEL_NAME,
    model_version=MODEL_VERSION,
    dataset_name=DATASET_NAME,
    data_version=DATA_VERSION,
    split_version=SPLIT_VERSION,
    feature_version=FEATURE_VERSION,
    label_definition_name=LABEL_DEFINITION_NAME,
    label_definition_detail=LABEL_DEFINITION_DETAIL,
    hyperparams=lgbm_peak_congestion_cls.get_params(),
    class_labels=CONGESTION_CLASS_LABELS,
    train_time_sec=peak_cls_train_time,
    notes="VALID peak-only 4class congestion classification / threshold applied"
)

logger.log_classification_result(
    y_true=y_test_peak_cong,
    y_pred=test_pred_peak_cong,
    runner=RUNNER,
    model_name=PEAK_CONGESTION_MODEL_NAME,
    model_version=MODEL_VERSION,
    dataset_name=DATASET_NAME,
    data_version=DATA_VERSION,
    split_version=SPLIT_VERSION,
    feature_version=FEATURE_VERSION,
    label_definition_name=LABEL_DEFINITION_NAME,
    label_definition_detail=LABEL_DEFINITION_DETAIL,
    hyperparams=lgbm_peak_congestion_cls.get_params(),
    class_labels=CONGESTION_CLASS_LABELS,
    train_time_sec=peak_cls_train_time,
    notes="TEST peak-only 4class congestion classification / threshold applied"
)

# ---------- 만차 여부 이진 분류 결과 저장 ----------
logger.log_classification_result(
    y_true=y_valid_full,
    y_pred=valid_pred_full,
    runner=RUNNER,
    model_name=FULL_MODEL_NAME,
    model_version=MODEL_VERSION,
    dataset_name=DATASET_NAME,
    data_version=DATA_VERSION,
    split_version=SPLIT_VERSION,
    feature_version=FEATURE_VERSION,
    label_definition_name="full_binary_0_notfull_1_full",
    label_definition_detail={"0": "여석있음", "1": "만차"},
    hyperparams=lgbm_full_cls.get_params(),
    class_labels=FULL_BINARY_LABELS,
    train_time_sec=full_cls_train_time,
    notes="VALID full/not-full binary classification / threshold applied"
)

logger.log_classification_result(
    y_true=y_test_full,
    y_pred=test_pred_full,
    runner=RUNNER,
    model_name=FULL_MODEL_NAME,
    model_version=MODEL_VERSION,
    dataset_name=DATASET_NAME,
    data_version=DATA_VERSION,
    split_version=SPLIT_VERSION,
    feature_version=FEATURE_VERSION,
    label_definition_name="full_binary_0_notfull_1_full",
    label_definition_detail={"0": "여석있음", "1": "만차"},
    hyperparams=lgbm_full_cls.get_params(),
    class_labels=FULL_BINARY_LABELS,
    train_time_sec=full_cls_train_time,
    notes="TEST full/not-full binary classification / threshold applied"
)

print("\n[INFO] experiment_logger 저장 완료")


# =========================================================
# 20. 모델 아티팩트 저장
# =========================================================
save_model_artifacts(
    model_dir=MODEL_DIR,
    reg_model=lgbm_reg,
    peak_congestion_model=lgbm_peak_congestion_cls,
    full_model=lgbm_full_cls,
    feature_cols=FEATURE_COLS,
    peak_thresholds=PEAK_THRESHOLDS,
    full_binary_threshold=FULL_BINARY_THRESHOLD,
    label_definition_detail=LABEL_DEFINITION_DETAIL,
)

# =========================================================
# 21. 서비스 추론 샘플 실행
# =========================================================
# test_df_raw를 기반으로 실제 서비스 추론 함수가 잘 동작하는지 확인
service_result = run_service_inference(
    prepared_df=test_df_raw,
    model_dir=MODEL_DIR,
    artifact_dir=ARTIFACT_DIR,
)

print("\n[서비스 추론 샘플 결과]")
print(service_result.head())