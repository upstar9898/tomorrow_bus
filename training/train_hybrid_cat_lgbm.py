# =========================================================
# 하이브리드 모델 실행
# CatBoost: 좌석수 회귀 + 출퇴근 4클래스 혼잡도
# LightGBM: 만차 여부 이진 분류
# =========================================================

# =========================================================
# 1. 라이브러리 import
# =========================================================

import os
import sys
import json
import time

import pandas as pd
import numpy as np
import joblib

from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from utils.experiment_logger import ExperimentLogger


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =========================================================
# 2. 설정값
# =========================================================

RUNNER = "eunbyeol"

# ── 경로 설정 ───────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
ARTIFACT_DIR = os.path.join(MODEL_ROOT, "artifacts")
ENCODER_DIR = os.path.join(MODEL_ROOT, "encoder")
ML_MODEL_DIR = os.path.join(MODEL_ROOT, "ml_models")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)
os.makedirs(ML_MODEL_DIR, exist_ok=True)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

logger = ExperimentLogger(artifact_dir=ARTIFACT_DIR)

# ── 실험 버전 ────────────────────────────────────────────
DATA_VERSION    = "v2"
SPLIT_VERSION   = "date_70_15_15"
FEATURE_VERSION = "pattern_weather_peak_congestion4_v1"
DATASET_NAME    = "bus_prepared_v1"

# ── 모델명 ───────────────────────────────────────────────
REG_MODEL_NAME       = "catboost_reg"
PEAK_CLS_MODEL_NAME  = "catboost_peak_congestion_cls_4class"
FULL_MODEL_NAME      = "lgbm_full_binary_cls"          # ← LightGBM
MODEL_VERSION        = "v1_peak_0_1to20_21to30_31to45"

# ── 레이블 정의 ──────────────────────────────────────────
LABEL_DEFINITION_NAME   = "peak_congestion_4class_0_1to20_21to30_31to45"
LABEL_DEFINITION_DETAIL = {
    "0": "만차(0석)",
    "1": "혼잡(1~20석)",
    "2": "보통(21~30석)",
    "3": "여유(31~45석)",
}
PEAK_CLASS_LABELS  = ["만차", "혼잡", "보통", "여유"]
FULL_BINARY_LABELS = ["여석있음", "만차"]

MAX_SEAT          = 45
LOW_SEAT_THRESHOLD = 10

# LightGBM binary threshold (lgbm 코드의 FULL_BINARY_THRESHOLD 참고)
FULL_BINARY_THRESHOLD = 0.5


# =========================================================
# 3. 데이터 로드
# =========================================================

file_path = os.path.join(DATA_DIR, "bus_all_raw_weather_260428.csv")

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

df = pd.read_csv(
    file_path,
    dtype={"busRouteId": str, "stId": str, "arsId": str},
    low_memory=False,
)

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")


# =========================================================
# 4. 기본 정리
# =========================================================

df = df.dropna(subset=["mkTm", "busRouteId", "staOrd", "remaining_seat"]).copy()

df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
df["staOrd"]         = pd.to_numeric(df["staOrd"],         errors="coerce")

df = df.dropna(subset=["remaining_seat", "staOrd"]).copy()
df = df[df["remaining_seat"] >= 0].copy()

df["busRouteId"] = df["busRouteId"].astype(str).str.strip()
df["date"]       = df["mkTm"].dt.date

print(f"정리 완료: {len(df):,}행")
print("\n[remaining_seat 기초통계]")
print(df["remaining_seat"].describe())


# =========================================================
# 5. 시간 feature 생성
# =========================================================
# 시간/주기성 feature 생성 (sin/cos + 시간 그룹화)

df["peak_level"] = 0
df.loc[(df["hour"] >= 7) & (df["hour"] <= 8), "peak_level"] = 1
df.loc[df["hour"] == 9,                        "peak_level"] = 2
df.loc[(df["hour"] >= 17) & (df["hour"] <= 18), "peak_level"] = 3
df.loc[df["hour"] == 19,                        "peak_level"] = 4

df["hour_group"]       = (df["hour"]   // 2).astype(int)
df["minute_group"]     = (df["minute"] // 10).astype(int)
df["hour_weekday_key"] = df["hour"] * 10 + df["dayofweek"]

df["month_sin"]  = np.sin(2 * np.pi * df["month"]   / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["month"]   / 12)
df["day_sin"]    = np.sin(2 * np.pi * df["day"]     / 31)
df["day_cos"]    = np.cos(2 * np.pi * df["day"]     / 31)
df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]    / 24)
df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]    / 24)
df["minute_sin"] = np.sin(2 * np.pi * df["minute"]  / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"]  / 60)
df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)

df["minute_5"] = (df["minute"] // 5).astype(int)
df["minute_3"] = (df["minute"] // 3).astype(int)

df["is_rain"] = (df["rainfall"] > 0).astype(int)
df["rain_peak"] = df["is_rain"] * df["is_peak"]

df["fog"] = pd.to_numeric(df.get("fog", 0), errors="coerce").fillna(0)
df["temperature"] = pd.to_numeric(df.get("temperature", 20), errors="coerce").fillna(20)
df["rainfall_missing"] = pd.to_numeric(df.get("rainfall_missing", 0), errors="coerce").fillna(0).astype(int)

print("시간 feature 생성 완료")
print(f"현재 컬럼 수: {len(df.columns)}개")


# =========================================================
# 6. 타깃 생성
# =========================================================
# 타깃 생성 (혼잡도 4class + 만차 여부)
df["is_low_seat"] = (df["remaining_seat"] <= LOW_SEAT_THRESHOLD).astype(int)


def seat_to_congestion_4class(seat):
    if seat == 0:       return 0  # 만차
    elif seat <= 20:    return 1  # 혼잡
    elif seat <= 30:    return 2  # 보통
    else:               return 3  # 여유


df["congestion_class"]  = df["remaining_seat"].apply(seat_to_congestion_4class)
df["full_flag_binary"]  = (df["remaining_seat"] == 0).astype(int)

print("\n[congestion_class 분포]")
print(df["congestion_class"].value_counts(normalize=True).sort_index())
print("\n[full_flag_binary 분포]")
print(df["full_flag_binary"].value_counts(normalize=True))


# =========================================================
# 7. 날짜 기준 분할 (70 / 15 / 15)
# =========================================================
# 날짜 기준 데이터 분할 (train/valid/test)
df = df.sort_values("mkTm").reset_index(drop=True)

unique_dates = sorted(df["date"].unique())
n_dates  = len(unique_dates)
train_end = int(n_dates * 0.70)
valid_end = int(n_dates * 0.85)

train_dates = unique_dates[:train_end]
valid_dates = unique_dates[train_end:valid_end]
test_dates  = unique_dates[valid_end:]

train_df = df[df["date"].isin(train_dates)].copy()
valid_df = df[df["date"].isin(valid_dates)].copy()
test_df  = df[df["date"].isin(test_dates)].copy()

print(f"train: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)}일) {len(train_df):,}행")
print(f"valid: {valid_dates[0]} ~ {valid_dates[-1]} ({len(valid_dates)}일) {len(valid_df):,}행")
print(f"test : {test_dates[0]} ~ {test_dates[-1]}  ({len(test_dates)}일) {len(test_df):,}행")


# =========================================================
# 8. 패턴 통계 feature 생성
# =========================================================
# 패턴 통계 feature 생성 (노선/정류소/시간별 평균, 분산, 혼잡 비율)
def add_pattern_features(train_base, target_df):
    result = target_df.copy()

    global_mean      = train_base["remaining_seat"].mean()
    global_low_ratio = train_base["is_low_seat"].mean()

    # 1) 노선 단위
    route_stat = (
        train_base.groupby("busRouteId")
        .agg(
            route_mean_seat=("remaining_seat", "mean"),
            route_std_seat =("remaining_seat", "std"),
            route_low_ratio=("is_low_seat",    "mean"),
        )
        .reset_index()
    )
    result = result.merge(route_stat, on="busRouteId", how="left")

    # 2) 노선 + 정류소
    route_stop_stat = (
        train_base.groupby(["busRouteId", "stId"])
        .agg(
            route_stop_mean_seat=("remaining_seat", "mean"),
            route_stop_std_seat =("remaining_seat", "std"),
            route_stop_low_ratio=("is_low_seat",    "mean"),
        )
        .reset_index()
    )
    result = result.merge(route_stop_stat, on=["busRouteId", "stId"], how="left")

    # 3) 노선 + 정류소 + 요일 + 시간그룹
    route_stop_time_stat = (
        train_base.groupby(["busRouteId", "stId", "dayofweek", "hour", "minute_group"])
        .agg(
            route_stop_time_mean_seat=("remaining_seat", "mean"),
            route_stop_time_std_seat =("remaining_seat", "std"),
            route_stop_time_low_ratio=("is_low_seat",    "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_stop_time_stat,
        on=["busRouteId", "stId", "dayofweek", "hour", "minute_group"],
        how="left",
    )
    route_staord_stat = (
        train_base.groupby(["busRouteId", "staOrd"])
        .agg(
            route_staord_mean_seat=("remaining_seat", "mean"),
            route_staord_std_seat=("remaining_seat", "std"),
            route_staord_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )

    result = result.merge(route_staord_stat, on=["busRouteId", "staOrd"], how="left")

    # 5) 노선 + 요일 + 시간
    route_time_stat = (
        train_base.groupby(["busRouteId", "dayofweek", "hour"])
        .agg(
            route_time_mean_seat=("remaining_seat", "mean"),
            route_time_std_seat =("remaining_seat", "std"),
            route_time_low_ratio=("is_low_seat",    "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_time_stat, on=["busRouteId", "dayofweek", "hour"], how="left"
    )

    # fallback: 통계 없는 조합은 전체 평균으로 채우기
    mean_cols = [
        "route_mean_seat",
        "route_stop_mean_seat",
        "route_stop_time_mean_seat",
        "route_staord_mean_seat",
        "route_time_mean_seat",
    ]

    std_cols = [
        "route_std_seat",
        "route_stop_std_seat",
        "route_stop_time_std_seat",
        "route_staord_std_seat",
        "route_time_std_seat",
    ]

    low_ratio_cols = [
        "route_low_ratio",
        "route_stop_low_ratio",
        "route_stop_time_low_ratio",
        "route_staord_low_ratio",
        "route_time_low_ratio",
    ]

    for col in mean_cols:      result[col] = result[col].fillna(global_mean)
    for col in std_cols:       result[col] = result[col].fillna(0)
    for col in low_ratio_cols: result[col] = result[col].fillna(global_low_ratio)

    return result


print("패턴 통계 feature 생성 중... (시간 좀 걸려요 ☕)")
train_df = add_pattern_features(train_df, train_df)
valid_df = add_pattern_features(train_df, valid_df)
test_df  = add_pattern_features(train_df, test_df)
print("패턴 통계 feature 생성 완료!")
print(f"train 컬럼 수: {len(train_df.columns)}개")


# =========================================================
# 9. feature 정의
# =========================================================
# 최종 입력 feature 정의
FEATURE_COLS = [
    "route_enc",
    "stid_enc",
    "arsid_enc",
    "staOrd",
    "sta_ratio",

    "year",
    "month",
    "day",
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
    "is_holiday",
    "is_peak",
    "peak_level",

    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",

    "hour_group",
    "minute_group",
    "minute_5",
    "minute_3",
    "hour_weekday_key",

    "rainfall",
    "precipitation",
    "fog",
    "temperature",
    "rainfall_missing",
    "is_rain",
    "rain_peak",

    "route_mean_seat",
    "route_stop_mean_seat",
    "route_stop_time_mean_seat",
    "route_staord_mean_seat",
    "route_time_mean_seat",

    "route_std_seat",
    "route_stop_std_seat",
    "route_stop_time_std_seat",
    "route_staord_std_seat",
    "route_time_std_seat",

    "route_low_ratio",
    "route_stop_low_ratio",
    "route_stop_time_low_ratio",
    "route_staord_low_ratio",
    "route_time_low_ratio",
]

REG_TARGET       = "remaining_seat"
PEAK_CLS_TARGET  = "congestion_class"
FULL_CLS_TARGET  = "full_flag_binary"

# LabelEncoder 적용 (route/station/arsId)
route_encoder = LabelEncoder()
stid_encoder = LabelEncoder()
arsid_encoder = LabelEncoder()

train_df["route_enc"] = route_encoder.fit_transform(train_df["busRouteId"].astype(str))
train_df["stid_enc"] = stid_encoder.fit_transform(train_df["stId"].astype(str))
train_df["arsid_enc"] = arsid_encoder.fit_transform(train_df["arsId"].astype(str))


def safe_transform(encoder, series):
    class_map = {cls: idx for idx, cls in enumerate(encoder.classes_)}
    return series.astype(str).map(lambda x: class_map.get(x, -1)).astype(int)


valid_df["route_enc"] = safe_transform(route_encoder, valid_df["busRouteId"])
test_df["route_enc"] = safe_transform(route_encoder, test_df["busRouteId"])

valid_df["stid_enc"] = safe_transform(stid_encoder, valid_df["stId"])
test_df["stid_enc"] = safe_transform(stid_encoder, test_df["stId"])

valid_df["arsid_enc"] = safe_transform(arsid_encoder, valid_df["arsId"])
test_df["arsid_enc"] = safe_transform(arsid_encoder, test_df["arsId"])

route_max_sta = train_df.groupby("busRouteId")["staOrd"].max().reset_index()
route_max_sta.columns = ["busRouteId", "max_staOrd"]

for target_df in [train_df, valid_df, test_df]:
    target_df.merge(route_max_sta, on="busRouteId", how="left")

train_df = train_df.merge(route_max_sta, on="busRouteId", how="left")
valid_df = valid_df.merge(route_max_sta, on="busRouteId", how="left")
test_df = test_df.merge(route_max_sta, on="busRouteId", how="left")

for target_df in [train_df, valid_df, test_df]:
    target_df["sta_ratio"] = target_df["staOrd"] / target_df["max_staOrd"]
    target_df["sta_ratio"] = target_df["sta_ratio"].fillna(0)


# ── 전체 데이터용 ────────────────────────────────────────
X_train = train_df[FEATURE_COLS]
X_valid = valid_df[FEATURE_COLS]
X_test  = test_df[FEATURE_COLS]

y_train_reg   = train_df[REG_TARGET]
y_valid_reg   = valid_df[REG_TARGET]
y_test_reg    = test_df[REG_TARGET]

y_train_full  = train_df[FULL_CLS_TARGET]
y_valid_full  = valid_df[FULL_CLS_TARGET]
y_test_full   = test_df[FULL_CLS_TARGET]

# ── 출퇴근 전용 (is_peak == 1) ───────────────────────────
train_peak_df = train_df[train_df["is_peak"] == 1]
valid_peak_df = valid_df[valid_df["is_peak"] == 1]
test_peak_df  = test_df[test_df["is_peak"] == 1]

X_train_peak = train_peak_df[FEATURE_COLS]
X_valid_peak = valid_peak_df[FEATURE_COLS]
X_test_peak  = test_peak_df[FEATURE_COLS]

y_train_peak  = train_peak_df[PEAK_CLS_TARGET]
y_valid_peak  = valid_peak_df[PEAK_CLS_TARGET]
y_test_peak   = test_peak_df[PEAK_CLS_TARGET]

X_train_lgbm = X_train.copy()
X_valid_lgbm = X_valid.copy()
X_test_lgbm = X_test.copy()

print(f"\n전체 - train: {X_train.shape} / valid: {X_valid.shape} / test: {X_test.shape}")
print(f"피크  - train: {X_train_peak.shape} / valid: {X_valid_peak.shape} / test: {X_test_peak.shape}")
print(f"feature 수: {len(FEATURE_COLS)}개")

# =========================================================
# 10. [CatBoost] 회귀 모델 학습 (좌석수 예측)
# =========================================================

cat_reg = CatBoostRegressor(
    iterations=1200,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    min_data_in_leaf=10,
    subsample=0.8,
    colsample_bylevel=0.8,
    loss_function="RMSE",
    eval_metric="MAE",
    random_seed=42,
    verbose=100,
)

cat_reg.fit(
    X_train, y_train_reg,
    eval_set=(X_valid, y_valid_reg),
    use_best_model=True,
    early_stopping_rounds=50,
)
print("CatBoost 회귀 모델 학습 완료!")


# =========================================================
# 11. [CatBoost] 출퇴근 4클래스 분류 모델 학습 (출퇴근 혼잡도)
# =========================================================

cat_peak_cls = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    min_data_in_leaf=10,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    colsample_bylevel=0.8,
    loss_function="MultiClass",
    eval_metric="TotalF1",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=100,
)

cat_peak_cls.fit(
    X_train_peak, y_train_peak,
    eval_set=(X_valid_peak, y_valid_peak),
    use_best_model=True,
    early_stopping_rounds=50,
)
print("CatBoost 출퇴근 4클래스 분류 모델 학습 완료!")


# =========================================================
# 12. [LightGBM] 만차 이진 분류 모델 학습 (만차 여부)
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
    n_jobs=-1,
)

start = time.time()
lgbm_full_cls.fit(
    X_train_lgbm, y_train_full,
    eval_set=[(X_train_lgbm, y_train_full), (X_valid_lgbm, y_valid_full)],
    eval_metric="binary_logloss",
)
full_cls_train_time = time.time() - start
print("LightGBM 만차 이진 분류 모델 학습 완료!")


# =========================================================
# 13. 평가 함수 정의
# =========================================================

def evaluate_regression(model, X, y, name="dataset"):
    pred = np.clip(model.predict(X), 0, MAX_SEAT)
    mae  = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2   = r2_score(y, pred)
    print(f"\n[{name}]")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    return pred


def evaluate_classification(model, X, y, name="dataset", class_labels=None):
    pred       = model.predict(X).flatten()
    acc        = accuracy_score(y, pred)
    macro_f1   = f1_score(y, pred, average="macro")
    weighted_f1= f1_score(y, pred, average="weighted")
    cm         = confusion_matrix(y, pred)
    print(f"\n[{name}]")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y, pred, target_names=class_labels, digits=4))
    return pred


def evaluate_binary_lgbm(model, X, y, threshold, name="dataset"):
    """LightGBM binary: threshold 적용 후 평가"""
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    acc        = accuracy_score(y, pred)
    macro_f1   = f1_score(y, pred, average="macro")
    weighted_f1= f1_score(y, pred, average="weighted")
    cm         = confusion_matrix(y, pred)
    print(f"\n[{name}]")
    print(f"Threshold   : {threshold}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y, pred, target_names=FULL_BINARY_LABELS, digits=4))
    return pred


# =========================================================
# 14. 모델 평가
# =========================================================

print("\n========== [CatBoost] 회귀 ==========")
valid_pred_reg = evaluate_regression(cat_reg, X_valid, y_valid_reg, "VALID")
test_pred_reg  = evaluate_regression(cat_reg, X_test,  y_test_reg,  "TEST")

print("\n========== [CatBoost] 출퇴근 4클래스 분류 ==========")
valid_pred_peak = evaluate_classification(
    cat_peak_cls, X_valid_peak, y_valid_peak,
    "VALID 출퇴근 4클래스", class_labels=PEAK_CLASS_LABELS,
)
test_pred_peak = evaluate_classification(
    cat_peak_cls, X_test_peak, y_test_peak,
    "TEST 출퇴근 4클래스", class_labels=PEAK_CLASS_LABELS,
)

print("\n========== [LightGBM] 만차 이진 분류 ==========")
valid_pred_full = evaluate_binary_lgbm(
    lgbm_full_cls, X_valid_lgbm, y_valid_full,
    FULL_BINARY_THRESHOLD, "VALID 만차 이진",
)
test_pred_full = evaluate_binary_lgbm(
    lgbm_full_cls, X_test_lgbm, y_test_full,
    FULL_BINARY_THRESHOLD, "TEST 만차 이진",
)


# =========================================================
# 15. 모델 저장
# =========================================================

os.makedirs(ARTIFACT_DIR, exist_ok=True)

joblib.dump(cat_reg, os.path.join(ML_MODEL_DIR, "reg.pkl"))
joblib.dump(cat_peak_cls, os.path.join(ML_MODEL_DIR, "peak_congestion_cls.pkl"))
joblib.dump(lgbm_full_cls, os.path.join(ML_MODEL_DIR, "full_cls.pkl"))

joblib.dump(route_encoder, os.path.join(ENCODER_DIR, "route_encoder.pkl"))
joblib.dump(stid_encoder, os.path.join(ENCODER_DIR, "stid_encoder.pkl"))
joblib.dump(arsid_encoder, os.path.join(ENCODER_DIR, "arsid_encoder.pkl"))

with open(os.path.join(ENCODER_DIR, "feature_cols.json"), "w", encoding="utf-8") as f:
    json.dump(FEATURE_COLS, f, ensure_ascii=False, indent=2)

thresholds = {
    "peak_congestion_thresholds": [0.15, 0.30, 0.40],
    "full_binary_threshold": FULL_BINARY_THRESHOLD,
}

with open(os.path.join(ENCODER_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
    json.dump(thresholds, f, ensure_ascii=False, indent=2)

with open(os.path.join(ENCODER_DIR, "label_definition.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "peak_class":  LABEL_DEFINITION_DETAIL,
            "full_binary": {"0": "여석있음", "1": "만차"},
            "full_binary_threshold": FULL_BINARY_THRESHOLD,
        },
        f, ensure_ascii=False, indent=2,
    )

pattern_meta = {
    "global_mean":      float(train_df["remaining_seat"].mean()),
    "global_low_ratio": float(train_df["is_low_seat"].mean()),
    "max_seat":         int(MAX_SEAT),
}
with open(os.path.join(ARTIFACT_DIR, "pattern_meta.json"), "w", encoding="utf-8") as f:
    json.dump(pattern_meta, f, ensure_ascii=False, indent=2)

# 패턴 통계 CSV 저장
train_df.groupby("busRouteId").agg(
    route_mean_seat=("remaining_seat", "mean"),
    route_std_seat=("remaining_seat", "std"),
    route_low_ratio=("is_low_seat", "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stat.csv"),
    index=False,
    encoding="utf-8-sig",
)

train_df.groupby(["busRouteId", "stId"]).agg(
    route_stop_mean_seat=("remaining_seat", "mean"),
    route_stop_std_seat=("remaining_seat", "std"),
    route_stop_low_ratio=("is_low_seat", "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stop_stat.csv"),
    index=False,
    encoding="utf-8-sig",
)

train_df.groupby(["busRouteId", "stId", "dayofweek", "hour", "minute_group"]).agg(
    route_stop_time_mean_seat=("remaining_seat", "mean"),
    route_stop_time_std_seat=("remaining_seat", "std"),
    route_stop_time_low_ratio=("is_low_seat", "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stop_time_stat.csv"),
    index=False,
    encoding="utf-8-sig",
)

train_df.groupby(["busRouteId", "staOrd"]).agg(
    route_staord_mean_seat=("remaining_seat", "mean"),
    route_staord_std_seat=("remaining_seat", "std"),
    route_staord_low_ratio=("is_low_seat", "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_staord_stat.csv"),
    index=False,
    encoding="utf-8-sig",
)

train_df.groupby(["busRouteId", "dayofweek", "hour"]).agg(
    route_time_mean_seat=("remaining_seat", "mean"),
    route_time_std_seat=("remaining_seat", "std"),
    route_time_low_ratio=("is_low_seat", "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_time_stat.csv"),
    index=False,
    encoding="utf-8-sig",
)

print(f"\n모델 및 메타정보 저장 완료!\n저장 경로: {ML_MODEL_DIR}")


# =========================================================
# 16. 실험 결과 logger 저장
# =========================================================

# ── [CatBoost] 회귀 ──────────────────────────────────────
for stage, y_true, y_pred in [
    ("VALID", y_valid_reg, valid_pred_reg),
    ("TEST",  y_test_reg,  test_pred_reg),
]:
    logger.log_regression_result(
        y_true=y_true, y_pred=y_pred,
        runner=RUNNER, model_name=REG_MODEL_NAME, model_version=MODEL_VERSION,
        dataset_name=DATASET_NAME, data_version=DATA_VERSION,
        split_version=SPLIT_VERSION, feature_version=FEATURE_VERSION,
        hyperparams=cat_reg.get_all_params(),
        notes=f"catboost {stage} regression result / clip 0~45",
    )

# ── [CatBoost] 출퇴근 4클래스 분류 ──────────────────────
for stage, y_true, y_pred in [
    ("VALID", y_valid_peak, valid_pred_peak),
    ("TEST",  y_test_peak,  test_pred_peak),
]:
    logger.log_classification_result(
        y_true=y_true, y_pred=y_pred,
        runner=RUNNER, model_name=PEAK_CLS_MODEL_NAME, model_version=MODEL_VERSION,
        dataset_name=DATASET_NAME, data_version=DATA_VERSION,
        split_version=SPLIT_VERSION, feature_version=FEATURE_VERSION,
        label_definition_name=LABEL_DEFINITION_NAME,
        label_definition_detail=LABEL_DEFINITION_DETAIL,
        hyperparams=cat_peak_cls.get_all_params(),
        class_labels=PEAK_CLASS_LABELS,
        notes=f"catboost {stage} peak-only 4class classification",
    )

# ── [LightGBM] 만차 이진 분류 ───────────────────────────
for stage, y_true, y_pred in [
    ("VALID", y_valid_full, valid_pred_full),
    ("TEST",  y_test_full,  test_pred_full),
]:
    logger.log_classification_result(
        y_true=y_true, y_pred=y_pred,
        runner=RUNNER, model_name=FULL_MODEL_NAME, model_version=MODEL_VERSION,
        dataset_name=DATASET_NAME, data_version=DATA_VERSION,
        split_version=SPLIT_VERSION, feature_version=FEATURE_VERSION,
        label_definition_name="full_binary_0_notfull_1_full",
        label_definition_detail={"0": "여석있음", "1": "만차"},
        hyperparams=lgbm_full_cls.get_params(),
        class_labels=FULL_BINARY_LABELS,
        notes=f"lgbm {stage} full/not-full binary classification / threshold applied",
    )

print("\n[INFO] experiment_logger 저장 완료")