# =========================================================
# [통합 최종 실행본]
# LightGBM 회귀 + 출퇴근 시간대 전용 4단계 혼잡도 분류기 + 만차 여부 이진 분류
# + 저장된 encoder / pattern stats 로드 사용
# + experiment_logger 연동
# =========================================================

# =========================================================
# 1. 라이브러리 import
# =========================================================
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import time

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from lightgbm import LGBMRegressor, LGBMClassifier

from utils.encoder_utils import load_label_encoders, transform_with_encoders
from utils.pattern_stats_utils import load_pattern_stats, merge_pattern_features
from utils.experiment_logger import ExperimentLogger
from utils.feature_utils import (
    MAX_SEAT,
    prepare_training_base_dataframe,
    split_by_date,
    get_feature_cols,
    seat_to_congestion_4,
    congestion_label_text,
    full_binary_label_text,
)


# =========================================================
# 2. 프로젝트 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs_peak_v2")
ARTIFACT_DIR = os.path.join(OUTPUT_ROOT, "artifacts")
MODEL_BASE_DIR = os.path.join(OUTPUT_ROOT, "models")
MODEL_DIR = os.path.join(MODEL_BASE_DIR, "lgbm_hybrid_peak_congestion4")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MODEL_BASE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

logger = ExperimentLogger(artifact_dir=ARTIFACT_DIR)

print("MODEL_DIR real path:", os.path.abspath(MODEL_DIR))
print("stid encoder file exists?:", os.path.exists(os.path.join(MODEL_DIR, "stid_encoder.pkl")))


# =========================================================
# 3. 실험 메타 정보 설정
# =========================================================
RUNNER = "eunbyeol"
DATASET_NAME = "bus_all_raw_weather"
DATA_VERSION = "20260416"
SPLIT_VERSION = "date_70_15_15"
FEATURE_VERSION = "pattern_weather_peak_congestion4_cleanlog_v1"

REG_MODEL_NAME = "lgbm_reg"
PEAK_CONGESTION_MODEL_NAME = "lgbm_peak_congestion_cls_4class"
FULL_MODEL_NAME = "lgbm_full_binary_cls"

MODEL_VERSION = "v2_peak_0_1to20_21to30_31to45_cleanlog"

LABEL_DEFINITION_NAME = "peak_congestion_4class_0_1to20_21to30_31to45"
LABEL_DEFINITION_DETAIL = {
    "0": "만차(0석)",
    "1": "혼잡(1~20석)",
    "2": "보통(21~30석)",
    "3": "여유(31~45석)"
}

CONGESTION_CLASS_LABELS = ["만차", "혼잡", "보통", "여유"]
FULL_BINARY_LABELS = ["여석있음", "만차"]


# =========================================================
# 4. 데이터 파일 경로 설정
# =========================================================
file_path = os.path.join(DATA_DIR, "bus_all_raw_weather.csv")

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
# 5. 설정값
# =========================================================

PEAK_THRESHOLDS = {
    0: 0.15,   # 만차
    1: 0.30,   # 혼잡
    2: 0.40    # 보통
}

FULL_BINARY_THRESHOLD = 0.50



# =========================================================
# 7. 원본 파일 불러오기
# =========================================================
df = pd.read_csv(
    file_path,
    dtype={
        "busRouteId": str,
        "stId": str,
        "arsId": str
    },
    low_memory=False
)

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")


# =========================================================
# 8. 공통 전처리 + 파생변수 생성
# =========================================================
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
# 9. 날짜 기준 train / valid / test 분할
# =========================================================
train_df, valid_df, test_df, split_info = split_by_date(df)

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
# 17. 범주형 ID 인코딩
# =========================================================
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
# 18. 패턴 통계 로드 + feature merge
# =========================================================
stats_dict, pattern_meta = load_pattern_stats(ARTIFACT_DIR)

train_df = merge_pattern_features(train_df, stats_dict, pattern_meta)
valid_df = merge_pattern_features(valid_df, stats_dict, pattern_meta)
test_df = merge_pattern_features(test_df, stats_dict, pattern_meta)

print("\n[패턴 feature merge 완료]")
print("train_df:", train_df.shape)
print("valid_df:", valid_df.shape)
print("test_df :", test_df.shape)


# =========================================================
# 19. feature 정의
# =========================================================
FEATURE_COLS = get_feature_cols()


# =========================================================
# 20. 학습 데이터 준비
# =========================================================
X_train = train_df[FEATURE_COLS]
X_valid = valid_df[FEATURE_COLS]
X_test = test_df[FEATURE_COLS]

y_train_reg = train_df["remaining_seat"]
y_valid_reg = valid_df["remaining_seat"]
y_test_reg = test_df["remaining_seat"]

y_train_full = train_df["is_full_target"]
y_valid_full = valid_df["is_full_target"]
y_test_full = test_df["is_full_target"]

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
# 21. 회귀 모델 학습
# =========================================================
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
    n_jobs=-1
)

start = time.time()
lgbm_reg.fit(
    X_train,
    y_train_reg,
    eval_set=[(X_train, y_train_reg), (X_valid, y_valid_reg)],
    eval_metric="l1"
)
reg_train_time = time.time() - start


# =========================================================
# 22. 출퇴근 시간대 전용 4단계 혼잡도 분류 모델 학습
# =========================================================
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
    n_jobs=-1
)

start = time.time()
lgbm_peak_congestion_cls.fit(
    X_train_peak,
    y_train_peak_cong,
    eval_set=[(X_train_peak, y_train_peak_cong), (X_valid_peak, y_valid_peak_cong)],
    eval_metric="multi_logloss"
)
peak_cls_train_time = time.time() - start


# =========================================================
# 23. 만차 여부 이진 분류 모델 학습
# =========================================================
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
    n_jobs=-1
)

start = time.time()
lgbm_full_cls.fit(
    X_train,
    y_train_full,
    eval_set=[(X_train, y_train_full), (X_valid, y_valid_full)],
    eval_metric="binary_logloss"
)
full_cls_train_time = time.time() - start


# =========================================================
# 24. threshold 적용 함수
# =========================================================
def predict_peak_congestion_with_thresholds(proba, thresholds):
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
# 25. 평가 함수
# =========================================================
def evaluate_regression(model, X, y, name="dataset"):
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
# 26. 모델 평가
# =========================================================
valid_pred_reg = evaluate_regression(lgbm_reg, X_valid, y_valid_reg, "VALID REG")
test_pred_reg = evaluate_regression(lgbm_reg, X_test, y_test_reg, "TEST REG")

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
# 27. experiment_logger 저장
# =========================================================
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
# 28. 서비스 추론 함수
# =========================================================
def predict_service(row_df):
    result = row_df.copy()

    pred_seat = np.clip(lgbm_reg.predict(result[FEATURE_COLS]), 0, MAX_SEAT)
    result["pred_remaining_seat"] = np.round(pred_seat, 2)
    result["pred_remaining_seat_rounded"] = np.clip(np.round(pred_seat), 0, MAX_SEAT).astype(int)

    result["reg_based_congestion_class"] = result["pred_remaining_seat_rounded"].apply(seat_to_congestion_4)
    result["reg_based_congestion_label"] = result["reg_based_congestion_class"].apply(congestion_label_text)

    peak_mask = result["is_peak"] == 1

    result["pred_peak_congestion_class"] = np.nan
    result["pred_peak_congestion_label"] = None
    result["pred_peak_congestion_prob_0"] = np.nan
    result["pred_peak_congestion_prob_1"] = np.nan
    result["pred_peak_congestion_prob_2"] = np.nan
    result["pred_peak_congestion_prob_3"] = np.nan

    if peak_mask.sum() > 0:
        peak_rows = result.loc[peak_mask, FEATURE_COLS]
        peak_proba = lgbm_peak_congestion_cls.predict_proba(peak_rows)
        peak_pred = predict_peak_congestion_with_thresholds(peak_proba, PEAK_THRESHOLDS)

        result.loc[peak_mask, "pred_peak_congestion_class"] = peak_pred
        result.loc[peak_mask, "pred_peak_congestion_label"] = [congestion_label_text(x) for x in peak_pred]

        for i in range(4):
            result.loc[peak_mask, f"pred_peak_congestion_prob_{i}"] = peak_proba[:, i]

    full_prob = lgbm_full_cls.predict_proba(result[FEATURE_COLS])[:, 1]
    result["pred_full_prob"] = full_prob
    result["pred_is_full"] = (result["pred_full_prob"] >= FULL_BINARY_THRESHOLD).astype(int)
    result["pred_is_full_label"] = result["pred_is_full"].apply(full_binary_label_text)
    result["pred_not_full_prob"] = 1 - result["pred_full_prob"]

    result["final_congestion_class"] = result["reg_based_congestion_class"]
    result["final_congestion_label"] = result["reg_based_congestion_label"]

    if peak_mask.sum() > 0:
        result.loc[peak_mask, "final_congestion_class"] = result.loc[peak_mask, "pred_peak_congestion_class"]
        result.loc[peak_mask, "final_congestion_label"] = result.loc[peak_mask, "pred_peak_congestion_label"]

    full_override_mask = result["pred_full_prob"] >= FULL_BINARY_THRESHOLD
    result.loc[full_override_mask, "final_congestion_class"] = 0
    result.loc[full_override_mask, "final_congestion_label"] = "만차"

    result["ui_expected_remaining_seat"] = result["pred_remaining_seat_rounded"].astype(str) + "석"
    result["ui_congestion_with_full_prob"] = (
        result["final_congestion_label"]
        + " (만차확률 "
        + (result["pred_full_prob"] * 100).round(1).astype(str)
        + "%)"
    )

    return result


# =========================================================
# 29. 테스트셋 전체에 대해 서비스 방식으로 예측
# =========================================================
service_result = predict_service(test_df)

save_cols = [
    "mkTm", "busRouteId", "stId", "arsId", "staOrd",
    "remaining_seat", "is_peak",
    "rainfall", "precipitation", "fog", "temperature",
    "is_rain", "rain_peak",
    "pred_remaining_seat",
    "pred_remaining_seat_rounded",
    "reg_based_congestion_class",
    "reg_based_congestion_label",
    "pred_peak_congestion_class",
    "pred_peak_congestion_label",
    "pred_peak_congestion_prob_0",
    "pred_peak_congestion_prob_1",
    "pred_peak_congestion_prob_2",
    "pred_peak_congestion_prob_3",
    "pred_is_full",
    "pred_is_full_label",
    "pred_not_full_prob",
    "pred_full_prob",
    "final_congestion_class",
    "final_congestion_label",
    "ui_expected_remaining_seat",
    "ui_congestion_with_full_prob"
]

service_result_path = os.path.join(
    ARTIFACT_DIR,
    "test_service_result_hybrid_peak_congestion4.csv"
)

service_result[save_cols].to_csv(
    service_result_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:", service_result_path)
print("\n[test_service_result 샘플]")
print(service_result[save_cols].head(10))


# =========================================================
# 30. 구간 이동시간 통계 생성 및 저장
# =========================================================
if "vehId1" in df.columns:
    travel_df = df.copy()

    travel_df["vehId1"] = travel_df["vehId1"].astype(str).str.strip()
    travel_df = travel_df.dropna(subset=["mkTm", "busRouteId", "vehId1", "staOrd"]).copy()
    travel_df = travel_df.sort_values(["busRouteId", "vehId1", "mkTm"]).reset_index(drop=True)

    travel_df["prev_staOrd"] = travel_df.groupby(["busRouteId", "vehId1"])["staOrd"].shift(1)
    travel_df["prev_mkTm"] = travel_df.groupby(["busRouteId", "vehId1"])["mkTm"].shift(1)

    travel_df["staOrd_gap"] = travel_df["staOrd"] - travel_df["prev_staOrd"]
    travel_df["travel_sec"] = (travel_df["mkTm"] - travel_df["prev_mkTm"]).dt.total_seconds()

    segment_df = travel_df[
        (travel_df["staOrd_gap"] == 1) &
        (travel_df["travel_sec"] > 0) &
        (travel_df["travel_sec"] <= 1800)
    ].copy()

    if len(segment_df) > 0:
        segment_df["from_staOrd"] = segment_df["prev_staOrd"].astype(int)
        segment_df["to_staOrd"] = segment_df["staOrd"].astype(int)

        route_segment_travel_time = (
            segment_df.groupby(["busRouteId", "from_staOrd", "to_staOrd"])
            .agg(
                avg_travel_sec=("travel_sec", "mean"),
                median_travel_sec=("travel_sec", "median"),
                std_travel_sec=("travel_sec", "std"),
                segment_count=("travel_sec", "count"),
            )
            .reset_index()
        )

        route_segment_travel_time["std_travel_sec"] = route_segment_travel_time["std_travel_sec"].fillna(0)

        route_segment_travel_time_path = os.path.join(
            ARTIFACT_DIR,
            "route_segment_travel_time.csv"
        )

        route_segment_travel_time.to_csv(
            route_segment_travel_time_path,
            index=False,
            encoding="utf-8-sig"
        )

        print("\n구간 이동시간 통계 저장 완료:", route_segment_travel_time_path)
        print(route_segment_travel_time.head())
    else:
        print("\n[경고] 구간 이동시간 통계를 만들 수 있는 인접 정류소 데이터가 없습니다.")
else:
    print("\n[경고] vehId1 컬럼이 없어 route_segment_travel_time.csv를 생성할 수 없습니다.")


# =========================================================
# 31. 노선별 정류소 순서 저장
# =========================================================
route_station_order = (
    df[["busRouteId", "stId", "arsId", "staOrd"]]
    .drop_duplicates()
    .sort_values(["busRouteId", "staOrd"])
    .reset_index(drop=True)
)

route_station_order_path = os.path.join(ARTIFACT_DIR, "route_station_order.csv")
route_station_order.to_csv(
    route_station_order_path,
    index=False,
    encoding="utf-8-sig"
)
print("\n정류소 순서 저장 완료:", route_station_order_path)


# =========================================================
# 32. 모델 / threshold / label 정의 저장
# =========================================================
joblib.dump(lgbm_reg, os.path.join(MODEL_DIR, "reg.pkl"))
joblib.dump(lgbm_peak_congestion_cls, os.path.join(MODEL_DIR, "peak_congestion_cls.pkl"))
joblib.dump(lgbm_full_cls, os.path.join(MODEL_DIR, "full_cls.pkl"))

with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w", encoding="utf-8") as f:
    json.dump(FEATURE_COLS, f, ensure_ascii=False, indent=2)

with open(os.path.join(MODEL_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "peak_congestion_thresholds": PEAK_THRESHOLDS,
            "full_binary_threshold": FULL_BINARY_THRESHOLD
        },
        f,
        ensure_ascii=False,
        indent=2
    )

with open(os.path.join(MODEL_DIR, "label_definition.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "peak_congestion_4class": LABEL_DEFINITION_DETAIL,
            "full_binary": {
                "0": "여석있음",
                "1": "만차"
            }
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# =========================================================
# 33. 프론트 전달용 JSON 미리보기
# =========================================================
single_response_json = {
    "success": True,
    "message": "예측이 완료되었습니다.",
    "data": {
        "remaining_seat": int(service_result.iloc[0]["pred_remaining_seat_rounded"]),
        "full_probability": round(float(service_result.iloc[0]["pred_full_prob"]), 4)
    }
}

print("\n[최종 API 응답 형태 JSON]")
print(json.dumps(single_response_json, ensure_ascii=False, indent=2))

print("\n모델 저장 완료")
print(f"- {os.path.join(MODEL_DIR, 'reg.pkl')}")
print(f"- {os.path.join(MODEL_DIR, 'peak_congestion_cls.pkl')}")
print(f"- {os.path.join(MODEL_DIR, 'full_cls.pkl')}")
print(f"- {os.path.join(MODEL_DIR, 'route_encoder.pkl')}")
print(f"- {os.path.join(MODEL_DIR, 'stid_encoder.pkl')}")
print(f"- {os.path.join(MODEL_DIR, 'arsid_encoder.pkl')}")
print(f"- {os.path.join(MODEL_DIR, 'feature_cols.json')}")
print(f"- {os.path.join(MODEL_DIR, 'thresholds.json')}")
print(f"- {os.path.join(MODEL_DIR, 'label_definition.json')}")