

# =========================================================
# 1. 라이브러리 import
# =========================================================

import os
import sys
import glob
import json

import pandas as pd
import numpy as np
import joblib

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)


# =========================================================
# 2. 설정값
# =========================================================

RUNNER = "yeaeun"

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data", "prepared")
ARTIFACT_DIR = os.path.join(BASE_DIR, "training", "artifacts")

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.experiment_logger import ExperimentLogger

logger = ExperimentLogger(artifact_dir=ARTIFACT_DIR)

DATA_VERSION    = "v1"
SPLIT_VERSION   = "date_70_15_15"
FEATURE_VERSION = "pattern_weather_peak_congestion4_v1"

MODEL_FAMILY        = "catboost"
REG_MODEL_NAME      = "catboost_reg"
PEAK_CLS_MODEL_NAME = "catboost_peak_congestion_cls_4class"
FULL_MODEL_NAME     = "catboost_full_binary_cls"
MODEL_VERSION       = "v1_peak_0_1to20_21to30_31to45"

LABEL_DEFINITION_NAME   = "peak_congestion_4class_0_1to20_21to30_31to45"
LABEL_DEFINITION_DETAIL = {
    "0": "만차(0석)",
    "1": "혼잡(1~20석)",
    "2": "보통(21~30석)",
    "3": "여유(31~45석)"
}
PEAK_CLASS_LABELS  = ["만차", "혼잡", "보통", "여유"]
FULL_BINARY_LABELS = ["여석있음", "만차"]

MAX_SEAT           = 45
LOW_SEAT_THRESHOLD = 10


# =========================================================
# 3. 데이터 로드 (일별 CSV 합치기)
# =========================================================

file_list = sorted(glob.glob(os.path.join(DATA_DIR, "bus_data_*_preprocessed_withweather_foranalysis.csv")))

if len(file_list) == 0:
    raise FileNotFoundError(f"데이터 파일이 없습니다: {DATA_DIR}")

print(f"총 {len(file_list)}개 파일 로드 중...")

df_list = []
for f in file_list:
    tmp = pd.read_csv(f, low_memory=False)
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)

print(f"로드 완료: 총 {len(df):,}행")
print(df.columns.tolist())


# =========================================================
# 4. 기본 정리
# =========================================================

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")

df = df.dropna(subset=["mkTm", "route_name", "staOrd", "remaining_seat"]).copy()

df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
df["staOrd"]         = pd.to_numeric(df["staOrd"], errors="coerce")

df = df.dropna(subset=["remaining_seat", "staOrd"]).copy()

df = df[df["remaining_seat"] >= 0].copy()

df["route_name"] = df["route_name"].astype(str).str.strip()

print(f"정리 완료: {len(df):,}행")
print("\n[remaining_seat 기초통계]")
print(df["remaining_seat"].describe())


# =========================================================
# 5. 시간 feature 생성
# =========================================================

# 출퇴근 세분화
df["peak_level"] = 0
df.loc[(df["hour"] >= 7) & (df["hour"] <= 8),  "peak_level"] = 1
df.loc[df["hour"] == 9,                         "peak_level"] = 2
df.loc[(df["hour"] >= 17) & (df["hour"] <= 18), "peak_level"] = 3
df.loc[df["hour"] == 19,                        "peak_level"] = 4

# 시간대 보조 key
df["hour_group"]       = (df["hour"] // 2).astype(int)
df["minute_group"]     = (df["minute"] // 10).astype(int)
df["hour_weekday_key"] = df["hour"] * 10 + df["dayofweek"]

# sin/cos 주기 인코딩
df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)

df["day_sin"]    = np.sin(2 * np.pi * df["day"] / 31)
df["day_cos"]    = np.cos(2 * np.pi * df["day"] / 31)

df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)

df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)

print("시간 feature 생성 완료")
print(f"현재 컬럼 수: {len(df.columns)}개")


# =========================================================
# 6. 혼잡도 타깃 생성
# =========================================================

# 패턴 통계용 보조 컬럼
df["is_low_seat"] = (df["remaining_seat"] <= LOW_SEAT_THRESHOLD).astype(int)

# 출퇴근 4클래스 분류 타깃
def seat_to_congestion_4class(seat):
    if seat == 0:    return 0  # 만차
    elif seat <= 20: return 1  # 혼잡
    elif seat <= 30: return 2  # 보통
    else:            return 3  # 여유

df["congestion_class"] = df["remaining_seat"].apply(seat_to_congestion_4class)

# 만차 이진 분류 타깃
df["full_flag_binary"] = (df["remaining_seat"] == 0).astype(int)

print("\n[congestion_class 분포]")
print(df["congestion_class"].value_counts(normalize=True).sort_index())

print("\n[full_flag_binary 분포]")
print(df["full_flag_binary"].value_counts(normalize=True))


# =========================================================
# 7. 날짜 기준 분할 (70 / 15 / 15)
# =========================================================

df = df.sort_values("mkTm").reset_index(drop=True)

unique_dates = sorted(df["date"].unique())
print(f"총 사용 날짜 수: {len(unique_dates)}일")
print(f"시작일: {unique_dates[0]} / 종료일: {unique_dates[-1]}")

n_dates   = len(unique_dates)
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
print(f"test : {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)}일) {len(test_df):,}행")


# =========================================================
# 8. 패턴 통계 feature 생성
# =========================================================

def add_pattern_features(train_base, target_df):

    result = target_df.copy()

    global_mean      = train_base["remaining_seat"].mean()
    global_low_ratio = train_base["is_low_seat"].mean()

    # 1) 노선 단위
    route_stat = (
        train_base.groupby("route_name")
        .agg(
            route_mean_seat =("remaining_seat", "mean"),
            route_std_seat  =("remaining_seat", "std"),
            route_low_ratio =("is_low_seat",    "mean"),
        )
        .reset_index()
    )
    result = result.merge(route_stat, on="route_name", how="left")

    # 2) 노선 + 정류소
    route_stop_stat = (
        train_base.groupby(["route_name", "staOrd"])
        .agg(
            route_stop_mean_seat =("remaining_seat", "mean"),
            route_stop_std_seat  =("remaining_seat", "std"),
            route_stop_low_ratio =("is_low_seat",    "mean"),
        )
        .reset_index()
    )
    result = result.merge(route_stop_stat, on=["route_name", "staOrd"], how="left")

    # 3) 노선 + 정류소 + 요일 + 시간그룹 (핵심!)
    route_stop_time_stat = (
        train_base.groupby(["route_name", "staOrd", "dayofweek", "hour_group"])
        .agg(
            route_stop_time_mean_seat =("remaining_seat", "mean"),
            route_stop_time_std_seat  =("remaining_seat", "std"),
            route_stop_time_low_ratio =("is_low_seat",    "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_stop_time_stat,
        on=["route_name", "staOrd", "dayofweek", "hour_group"],
        how="left"
    )

    # 4) 노선 + 정류소 + 출퇴근 여부 (담당자와 차별점!)
    route_stop_peak_stat = (
        train_base.groupby(["route_name", "staOrd", "is_peak"])
        .agg(
            route_stop_peak_mean_seat =("remaining_seat", "mean"),
            route_stop_peak_std_seat  =("remaining_seat", "std"),
            route_stop_peak_low_ratio =("is_low_seat",    "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_stop_peak_stat,
        on=["route_name", "staOrd", "is_peak"],
        how="left"
    )

    # 5) 노선 + 요일 + 시간
    route_time_stat = (
        train_base.groupby(["route_name", "dayofweek", "hour"])
        .agg(
            route_time_mean_seat =("remaining_seat", "mean"),
            route_time_std_seat  =("remaining_seat", "std"),
            route_time_low_ratio =("is_low_seat",    "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_time_stat,
        on=["route_name", "dayofweek", "hour"],
        how="left"
    )

    # fallback: 통계 없는 조합은 전체 평균으로 채우기
    mean_cols = [
        "route_mean_seat",
        "route_stop_mean_seat",
        "route_stop_time_mean_seat",
        "route_stop_peak_mean_seat",
        "route_time_mean_seat",
    ]
    std_cols = [
        "route_std_seat",
        "route_stop_std_seat",
        "route_stop_time_std_seat",
        "route_stop_peak_std_seat",
        "route_time_std_seat",
    ]
    low_ratio_cols = [
        "route_low_ratio",
        "route_stop_low_ratio",
        "route_stop_time_low_ratio",
        "route_stop_peak_low_ratio",
        "route_time_low_ratio",
    ]

    for col in mean_cols:
        result[col] = result[col].fillna(global_mean)
    for col in std_cols:
        result[col] = result[col].fillna(0)
    for col in low_ratio_cols:
        result[col] = result[col].fillna(global_low_ratio)

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

FEATURE_COLS = [
    "route_name",
    "staOrd",
    "year", "month", "day",
    "hour", "minute", "dayofweek",
    "is_weekend", "is_peak", "is_holiday",
    "peak_level",
    "hour_group", "minute_group", "hour_weekday_key",
    "month_sin", "month_cos",
    "day_sin",   "day_cos",
    "hour_sin",  "hour_cos",
    "minute_sin","minute_cos",
    "dow_sin",   "dow_cos",
    "precipitation", "rainfall",
    "route_mean_seat",
    "route_stop_mean_seat",
    "route_stop_time_mean_seat",
    "route_stop_peak_mean_seat",
    "route_time_mean_seat",
    "route_std_seat",
    "route_stop_std_seat",
    "route_stop_time_std_seat",
    "route_stop_peak_std_seat",
    "route_time_std_seat",
    "route_low_ratio",
    "route_stop_low_ratio",
    "route_stop_time_low_ratio",
    "route_stop_peak_low_ratio",
    "route_time_low_ratio",
]

CAT_FEATURES = ["route_name"]

REG_TARGET      = "remaining_seat"
PEAK_CLS_TARGET = "congestion_class"
FULL_CLS_TARGET = "full_flag_binary"

# 회귀용
X_train = train_df[FEATURE_COLS]
X_valid = valid_df[FEATURE_COLS]
X_test  = test_df[FEATURE_COLS]

y_train_reg = train_df[REG_TARGET]
y_valid_reg = valid_df[REG_TARGET]
y_test_reg  = test_df[REG_TARGET]

# 출퇴근 4클래스 분류용 (is_peak==1만!)
train_peak_df = train_df[train_df["is_peak"] == 1]
valid_peak_df = valid_df[valid_df["is_peak"] == 1]
test_peak_df  = test_df[test_df["is_peak"] == 1]

X_train_peak = train_peak_df[FEATURE_COLS]
X_valid_peak = valid_peak_df[FEATURE_COLS]
X_test_peak  = test_peak_df[FEATURE_COLS]

y_train_peak = train_peak_df[PEAK_CLS_TARGET]
y_valid_peak = valid_peak_df[PEAK_CLS_TARGET]
y_test_peak  = test_peak_df[PEAK_CLS_TARGET]

# 만차 이진 분류용 (전체 데이터)
y_train_full = train_df[FULL_CLS_TARGET]
y_valid_full = valid_df[FULL_CLS_TARGET]
y_test_full  = test_df[FULL_CLS_TARGET]

print(f"전체 - train: {X_train.shape} / valid: {X_valid.shape} / test: {X_test.shape}")
print(f"피크 - train: {X_train_peak.shape} / valid: {X_valid_peak.shape} / test: {X_test_peak.shape}")
print(f"feature 수: {len(FEATURE_COLS)}개")
print(f"cat_features: {CAT_FEATURES}")


# =========================================================
# 10. 회귀 모델 학습
# =========================================================

cat_reg = CatBoostRegressor(
    iterations=800,
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
    cat_features=CAT_FEATURES,
)

cat_reg.fit(
    X_train, y_train_reg,
    eval_set=(X_valid, y_valid_reg),
    use_best_model=True,
    early_stopping_rounds=50,
)

print("CatBoost 회귀 모델 학습 완료!")


# =========================================================
# 11. 회귀 평가
# =========================================================

def evaluate_regression(model, X, y, name="dataset"):
    pred = model.predict(X)
    pred = np.clip(pred, 0, MAX_SEAT)

    mae  = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2   = r2_score(y, pred)

    print(f"\n[{name}]")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    print("\n실제값 통계:")
    print(pd.Series(y).describe())
    print("\n예측값 통계:")
    print(pd.Series(pred).describe())

    # 노선별 MAE
    print("\n노선별 MAE:")
    route_mae = {}
    for route in X["route_name"].unique():
        mask = X["route_name"] == route
        if mask.sum() > 0:
            r_pred  = np.clip(model.predict(X[mask]), 0, MAX_SEAT)
            r_mae   = mean_absolute_error(y[mask], r_pred)
            route_mae[route] = round(r_mae, 4)

    for route, mae_val in sorted(route_mae.items()):
        print(f"  {route}: {mae_val}")

    return pred

valid_pred_reg = evaluate_regression(cat_reg, X_valid, y_valid_reg, "VALID")
test_pred_reg  = evaluate_regression(cat_reg, X_test,  y_test_reg,  "TEST")


# =========================================================
# 12. 분류 모델 학습
# =========================================================

# 출퇴근 4클래스 분류
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
    cat_features=CAT_FEATURES,
)

cat_peak_cls.fit(
    X_train_peak, y_train_peak,
    eval_set=(X_valid_peak, y_valid_peak),
    use_best_model=True,
    early_stopping_rounds=50,
)

print("출퇴근 4클래스 분류 모델 학습 완료!")


# 만차 이진 분류
cat_full_cls = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    min_data_in_leaf=10,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    colsample_bylevel=0.8,
    loss_function="Logloss",
    eval_metric="F1",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=100,
    cat_features=CAT_FEATURES,
)

cat_full_cls.fit(
    X_train, y_train_full,
    eval_set=(X_valid, y_valid_full),
    use_best_model=True,
    early_stopping_rounds=50,
)

print("만차 이진 분류 모델 학습 완료!")


# =========================================================
# 13. 분류 평가
# =========================================================

def evaluate_classification(model, X, y, name="dataset", class_labels=None):
    pred       = model.predict(X).flatten()
    pred_proba = model.predict_proba(X)

    acc         = accuracy_score(y, pred)
    macro_f1    = f1_score(y, pred, average="macro")
    weighted_f1 = f1_score(y, pred, average="weighted")
    cm          = confusion_matrix(y, pred)

    print(f"\n[{name}]")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    print("\n실제 class 분포:")
    print(pd.Series(y).value_counts(normalize=True).sort_index())

    print("\n예측 class 분포:")
    print(pd.Series(pred).value_counts(normalize=True).sort_index())

    print("\nConfusion Matrix (행=실제, 열=예측):")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y, pred, target_names=class_labels, digits=4))

    return pred, pred_proba


print("\n========== 출퇴근 4클래스 분류 ==========")
valid_pred_peak, valid_proba_peak = evaluate_classification(
    cat_peak_cls,
    X_valid_peak, y_valid_peak,
    "VALID 출퇴근 4클래스",
    class_labels=PEAK_CLASS_LABELS
)
test_pred_peak, test_proba_peak = evaluate_classification(
    cat_peak_cls,
    X_test_peak, y_test_peak,
    "TEST 출퇴근 4클래스",
    class_labels=PEAK_CLASS_LABELS
)

print("\n========== 만차 이진 분류 ==========")
valid_pred_full, valid_proba_full = evaluate_classification(
    cat_full_cls,
    X_valid, y_valid_full,
    "VALID 만차 이진",
    class_labels=FULL_BINARY_LABELS
)
test_pred_full, test_proba_full = evaluate_classification(
    cat_full_cls,
    X_test, y_test_full,
    "TEST 만차 이진",
    class_labels=FULL_BINARY_LABELS
)


# =========================================================
# 14. 모델 및 메타정보 저장
# =========================================================

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 모델 저장
joblib.dump(cat_reg,
    os.path.join(ARTIFACT_DIR, "catboost_reg.pkl"))
joblib.dump(cat_peak_cls,
    os.path.join(ARTIFACT_DIR, "catboost_peak_congestion_cls.pkl"))
joblib.dump(cat_full_cls,
    os.path.join(ARTIFACT_DIR, "catboost_full_cls.pkl"))

print("모델 저장 완료!")

# feature 목록 저장
with open(os.path.join(ARTIFACT_DIR, "feature_cols.json"), "w", encoding="utf-8") as f:
    json.dump(FEATURE_COLS, f, ensure_ascii=False, indent=2)

# 혼잡도 기준 저장
with open(os.path.join(ARTIFACT_DIR, "label_definition.json"), "w", encoding="utf-8") as f:
    json.dump({
        "peak_class": LABEL_DEFINITION_DETAIL,
        "full_binary": {"0": "여석있음", "1": "만차"}
    }, f, ensure_ascii=False, indent=2)

# 패턴 통계 메타 저장
pattern_meta = {
    "global_mean":      float(train_df["remaining_seat"].mean()),
    "global_low_ratio": float(train_df["is_low_seat"].mean()),
    "max_seat":         int(MAX_SEAT)
}
with open(os.path.join(ARTIFACT_DIR, "pattern_meta.json"), "w", encoding="utf-8") as f:
    json.dump(pattern_meta, f, ensure_ascii=False, indent=2)

# 패턴 통계 CSV 저장
train_df.groupby("route_name").agg(
    route_mean_seat =("remaining_seat", "mean"),
    route_std_seat  =("remaining_seat", "std"),
    route_low_ratio =("is_low_seat",    "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stat.csv"),
    index=False, encoding="utf-8-sig"
)

train_df.groupby(["route_name", "staOrd"]).agg(
    route_stop_mean_seat =("remaining_seat", "mean"),
    route_stop_std_seat  =("remaining_seat", "std"),
    route_stop_low_ratio =("is_low_seat",    "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stop_stat.csv"),
    index=False, encoding="utf-8-sig"
)

train_df.groupby(["route_name", "staOrd", "dayofweek", "hour_group"]).agg(
    route_stop_time_mean_seat =("remaining_seat", "mean"),
    route_stop_time_std_seat  =("remaining_seat", "std"),
    route_stop_time_low_ratio =("is_low_seat",    "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stop_time_stat.csv"),
    index=False, encoding="utf-8-sig"
)

train_df.groupby(["route_name", "staOrd", "is_peak"]).agg(
    route_stop_peak_mean_seat =("remaining_seat", "mean"),
    route_stop_peak_std_seat  =("remaining_seat", "std"),
    route_stop_peak_low_ratio =("is_low_seat",    "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stop_peak_stat.csv"),
    index=False, encoding="utf-8-sig"
)

train_df.groupby(["route_name", "dayofweek", "hour"]).agg(
    route_time_mean_seat =("remaining_seat", "mean"),
    route_time_std_seat  =("remaining_seat", "std"),
    route_time_low_ratio =("is_low_seat",    "mean"),
).reset_index().to_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_time_stat.csv"),
    index=False, encoding="utf-8-sig"
)

print("메타정보 및 패턴 통계 저장 완료!")
print(f"저장 경로: {ARTIFACT_DIR}")

# =========================================================
# 15. 실험 결과 logger 저장
# =========================================================

DATASET_NAME = "bus_prepared_v1"

# 회귀 VALID
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
    hyperparams=cat_reg.get_all_params(),
    notes=f"{MODEL_FAMILY} VALID regression result / clip 0~45"
)

# 회귀 TEST
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
    hyperparams=cat_reg.get_all_params(),
    notes=f"{MODEL_FAMILY} TEST regression result / clip 0~45"
)

# 출퇴근 4클래스 분류 VALID
logger.log_classification_result(
    y_true=y_valid_peak,
    y_pred=valid_pred_peak,
    runner=RUNNER,
    model_name=PEAK_CLS_MODEL_NAME,
    model_version=MODEL_VERSION,
    dataset_name=DATASET_NAME,
    data_version=DATA_VERSION,
    split_version=SPLIT_VERSION,
    feature_version=FEATURE_VERSION,
    label_definition_name=LABEL_DEFINITION_NAME,
    label_definition_detail=LABEL_DEFINITION_DETAIL,
    hyperparams=cat_peak_cls.get_all_params(),
    class_labels=PEAK_CLASS_LABELS,
    notes=f"{MODEL_FAMILY} VALID peak-only 4class classification"
)

# 출퇴근 4클래스 분류 TEST
logger.log_classification_result(
    y_true=y_test_peak,
    y_pred=test_pred_peak,
    runner=RUNNER,
    model_name=PEAK_CLS_MODEL_NAME,
    model_version=MODEL_VERSION,
    dataset_name=DATASET_NAME,
    data_version=DATA_VERSION,
    split_version=SPLIT_VERSION,
    feature_version=FEATURE_VERSION,
    label_definition_name=LABEL_DEFINITION_NAME,
    label_definition_detail=LABEL_DEFINITION_DETAIL,
    hyperparams=cat_peak_cls.get_all_params(),
    class_labels=PEAK_CLASS_LABELS,
    notes=f"{MODEL_FAMILY} TEST peak-only 4class classification"
)

# 만차 이진 분류 VALID
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
    hyperparams=cat_full_cls.get_all_params(),
    class_labels=FULL_BINARY_LABELS,
    notes=f"{MODEL_FAMILY} VALID full binary classification"
)

# 만차 이진 TEST
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
    hyperparams=cat_full_cls.get_all_params(),
    class_labels=FULL_BINARY_LABELS,
    notes=f"{MODEL_FAMILY} TEST full binary classification"
)

print("\n[INFO] logger 저장 완료")