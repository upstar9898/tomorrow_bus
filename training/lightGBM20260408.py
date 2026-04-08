# =========================================================
# [모델 설계 개요 - LightGBM 단독 + Pattern Feature 기반]
# =========================================================
# 목적:
# - 특정 날짜/시간/정류장/노선 조건에서 해당 시점의 좌석 수를 예측
# - "현재 기준 10분 뒤"가 아니라 "그 시점 자체의 좌석 수"를 예측함

# 핵심 전략:
# 1. 미래 타깃 생성(target_time, merge_asof) 제거
# 2. 각 행의 remaining_seat 자체를 타깃(y)으로 사용
# 3. lag 없이 패턴 기반 feature 사용
# 4. train 통계만으로 valid/test 패턴 feature 생성

# 참고:
# - 모델 입력은 FEATURE_COLS에 포함된 컬럼만 사용
# - date는 분할용으로만 사용
# - is_low_seat는 혼잡 비율(route_low_ratio 등) 계산용 보조 컬럼
# - 예측값은 모델 학습 시 제한하지 않고, 평가 및 실사용 시에만 0~45 범위로 clip 처리
# =========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from lightgbm import LGBMRegressor

# =========================================================
# 0. 설정
# =========================================================
file_path = "bus_all_raw.csv"
MAX_SEAT = 45
LOW_SEAT_THRESHOLD = 10   # 혼잡 기준

# =========================================================
# 1. 파일 불러오기
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
# 2. 기본 정리
# =========================================================
df = df.dropna(subset=["mkTm", "busRouteId", "stId", "arsId", "remaining_seat"]).copy()

df["busRouteId"] = df["busRouteId"].astype(str).str.strip()
df["stId"] = df["stId"].astype(str).str.strip()
df["arsId"] = df["arsId"].astype(str).str.strip()

df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")

if "exps1" in df.columns:
    df["exps1"] = pd.to_numeric(df["exps1"], errors="coerce")
else:
    df["exps1"] = np.nan

if "full_flag" in df.columns:
    df["full_flag"] = pd.to_numeric(df["full_flag"], errors="coerce")
else:
    df["full_flag"] = np.nan

df = df.dropna(subset=["remaining_seat", "staOrd"]).copy()
df = df[df["remaining_seat"] >= 0].copy()

print("총 데이터 수:", len(df))
print("\n[remaining_seat 기초통계]")
print(df["remaining_seat"].describe())

# =========================================================
# 3. 시간 feature
# =========================================================
df["year"] = df["mkTm"].dt.year
df["month"] = df["mkTm"].dt.month
df["day"] = df["mkTm"].dt.day
df["hour"] = df["mkTm"].dt.hour
df["minute"] = df["mkTm"].dt.minute
df["dayofweek"] = df["mkTm"].dt.dayofweek
df["date"] = df["mkTm"].dt.date

df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
df["is_peak"] = (
    ((df["hour"] >= 7) & (df["hour"] <= 9)) |
    ((df["hour"] >= 17) & (df["hour"] <= 19))
).astype(int)

# 출퇴근 세분화
df["peak_level"] = 0
df.loc[(df["hour"] >= 7) & (df["hour"] <= 8), "peak_level"] = 1
df.loc[df["hour"] == 9, "peak_level"] = 2
df.loc[(df["hour"] >= 17) & (df["hour"] <= 18), "peak_level"] = 3
df.loc[df["hour"] == 19, "peak_level"] = 4

# 주기 인코딩
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

# 보조 key
df["hour_group"] = (df["hour"] // 2).astype(int)
df["minute_group"] = (df["minute"] // 10).astype(int)
df["hour_weekday_key"] = df["hour"] * 10 + df["dayofweek"]

# =========================================================
# 4. 혼잡 보정용 라벨
# =========================================================
df["is_low_seat"] = (df["remaining_seat"] <= LOW_SEAT_THRESHOLD).astype(int)

# =========================================================
# 5. 인코딩
# =========================================================
route_le = LabelEncoder()
stid_le = LabelEncoder()
arsid_le = LabelEncoder()

df["route_enc"] = route_le.fit_transform(df["busRouteId"])
df["stid_enc"] = stid_le.fit_transform(df["stId"])
df["arsid_enc"] = arsid_le.fit_transform(df["arsId"])

# =========================================================
# 6. 날짜 기준 분할
# =========================================================
df = df.sort_values("mkTm").reset_index(drop=True)

unique_dates = sorted(df["date"].unique())
print("사용 날짜들:", unique_dates)
print("총 사용 날짜 수:", len(unique_dates))

if len(unique_dates) < 10:
    raise ValueError("최소 10일 이상은 있어야 안정적으로 분할 가능합니다.")

n_dates = len(unique_dates)
train_end = int(n_dates * 0.7)
valid_end = int(n_dates * 0.85)

train_dates = unique_dates[:train_end]
valid_dates = unique_dates[train_end:valid_end]
test_dates = unique_dates[valid_end:]

print("train dates:", train_dates[0], "~", train_dates[-1], f"({len(train_dates)}일)")
print("valid dates:", valid_dates[0], "~", valid_dates[-1], f"({len(valid_dates)}일)")
print("test dates :", test_dates[0], "~", test_dates[-1], f"({len(test_dates)}일)")

train_df = df[df["date"].isin(train_dates)].copy()
valid_df = df[df["date"].isin(valid_dates)].copy()
test_df = df[df["date"].isin(test_dates)].copy()

# =========================================================
# 7. 패턴 통계 + 혼잡 비율 feature
# =========================================================
def add_pattern_features(train_base, target_df):
    result = target_df.copy()

    global_mean = train_base["remaining_seat"].mean()
    global_low_ratio = train_base["is_low_seat"].mean()

    # 1) 노선 평균
    route_stat = (
        train_base.groupby("busRouteId")
        .agg(
            route_mean_seat=("remaining_seat", "mean"),
            route_std_seat=("remaining_seat", "std"),
            route_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(route_stat, on="busRouteId", how="left")

    # 2) 노선 + 정류장
    route_stop_stat = (
        train_base.groupby(["busRouteId", "stId"])
        .agg(
            route_stop_mean_seat=("remaining_seat", "mean"),
            route_stop_std_seat=("remaining_seat", "std"),
            route_stop_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(route_stop_stat, on=["busRouteId", "stId"], how="left")

    # 3) 노선 + 정류장 + 요일 + 시간그룹
    route_stop_time_stat = (
        train_base.groupby(["busRouteId", "stId", "dayofweek", "hour_group"])
        .agg(
            route_stop_time_mean_seat=("remaining_seat", "mean"),
            route_stop_time_std_seat=("remaining_seat", "std"),
            route_stop_time_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_stop_time_stat,
        on=["busRouteId", "stId", "dayofweek", "hour_group"],
        how="left"
    )

    # 4) 노선 + staOrd
    route_staord_stat = (
        train_base.groupby(["busRouteId", "staOrd"])
        .agg(
            route_staord_mean_seat=("remaining_seat", "mean"),
            route_staord_std_seat=("remaining_seat", "std"),
            route_staord_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_staord_stat,
        on=["busRouteId", "staOrd"],
        how="left"
    )

    # 5) 노선 + 요일 + 시간
    route_time_stat = (
        train_base.groupby(["busRouteId", "dayofweek", "hour"])
        .agg(
            route_time_mean_seat=("remaining_seat", "mean"),
            route_time_std_seat=("remaining_seat", "std"),
            route_time_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_time_stat,
        on=["busRouteId", "dayofweek", "hour"],
        how="left"
    )

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

    for col in mean_cols:
        result[col] = result[col].fillna(global_mean)
    for col in std_cols:
        result[col] = result[col].fillna(0)
    for col in low_ratio_cols:
        result[col] = result[col].fillna(global_low_ratio)

    return result

train_df = add_pattern_features(train_df, train_df)
valid_df = add_pattern_features(train_df, valid_df)
test_df = add_pattern_features(train_df, test_df)

# =========================================================
# 8. 입력 feature 정의
# =========================================================
FEATURE_COLS = [
    # 인코딩
    "route_enc", "stid_enc", "arsid_enc",

    # 시간
    "year", "month", "day",
    "hour", "minute", "dayofweek",
    "is_weekend", "is_peak", "peak_level",

    # 주기 인코딩
    "month_sin", "month_cos",
    "day_sin", "day_cos",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dow_sin", "dow_cos",

    # 위치/운행
    "staOrd",
    "hour_group",
    "minute_group",
    "hour_weekday_key",

    # 평균 패턴
    "route_mean_seat", "route_std_seat",
    "route_stop_mean_seat", "route_stop_std_seat",
    "route_stop_time_mean_seat", "route_stop_time_std_seat",
    "route_staord_mean_seat", "route_staord_std_seat",
    "route_time_mean_seat", "route_time_std_seat",

    # 혼잡 보정
    "route_low_ratio",
    "route_stop_low_ratio",
    "route_stop_time_low_ratio",
    "route_staord_low_ratio",
    "route_time_low_ratio",
]

TARGET_COL = "remaining_seat"

for df_part in [train_df, valid_df, test_df]:
    df_part["exps1"] = df_part["exps1"].fillna(train_df["exps1"].median())
    df_part["full_flag"] = df_part["full_flag"].fillna(0)

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]

X_valid = valid_df[FEATURE_COLS]
y_valid = valid_df[TARGET_COL]

X_test = test_df[FEATURE_COLS]
y_test = test_df[TARGET_COL]

print("train:", X_train.shape, y_train.shape)
print("valid:", X_valid.shape, y_valid.shape)
print("test :", X_test.shape, y_test.shape)

# =========================================================
# 9. 모델 학습
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

lgbm_reg.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    eval_metric="l1"
)

print("LightGBM 회귀 모델 학습 완료")

# =========================================================
# 10. 평가 함수
# =========================================================
def evaluate_regression(model, X, y, name="dataset"):
    pred = model.predict(X)

    # 평가용 clip
    pred = np.clip(pred, 0, MAX_SEAT)

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)

    print(f"\n[{name}]")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    print("\n실제값 통계:")
    print(pd.Series(y).describe())

    print("\n예측값 통계:")
    print(pd.Series(pred).describe())

    return pred

valid_pred = evaluate_regression(lgbm_reg, X_valid, y_valid, "VALID")
test_pred = evaluate_regression(lgbm_reg, X_test, y_test, "TEST")

# =========================================================
# 11. 출퇴근 / 비출퇴근 성능
# =========================================================
def evaluate_regression_by_peak(model, df_eval, name="TEST"):
    peak_df = df_eval[df_eval["is_peak"] == 1].copy()
    non_peak_df = df_eval[df_eval["is_peak"] == 0].copy()

    print(f"\n===== {name} (출퇴근 vs 비출퇴근 성능) =====")

    if len(peak_df) > 0:
        print("\n[출퇴근 시간대 (is_peak=1)]")
        evaluate_regression(model, peak_df[FEATURE_COLS], peak_df[TARGET_COL], "PEAK")

    if len(non_peak_df) > 0:
        print("\n[비출퇴근 시간대 (is_peak=0)]")
        evaluate_regression(model, non_peak_df[FEATURE_COLS], non_peak_df[TARGET_COL], "NON-PEAK")

evaluate_regression_by_peak(lgbm_reg, valid_df, "VALID")
evaluate_regression_by_peak(lgbm_reg, test_df, "TEST")

# =========================================================
# 12. 참고용 혼잡도 분포 확인
# =========================================================
def seat_to_congestion(seat):
    if seat <= 5:
        return 0
    elif seat <= 15:
        return 1
    else:
        return 2

test_pred_rounded = np.clip(np.round(test_pred), 0, MAX_SEAT)
test_pred_class = pd.Series(test_pred_rounded).apply(seat_to_congestion)
test_true_class = y_test.apply(seat_to_congestion)

print("\n[참고용 TEST class 분포 - 실제]")
print(test_true_class.value_counts(normalize=True).sort_index())

print("\n[참고용 TEST class 분포 - 예측]")
print(test_pred_class.value_counts(normalize=True).sort_index())

# =========================================================
# 13. 모델 저장
# =========================================================
joblib.dump(lgbm_reg, "lgbm_point_seat_regressor_final.pkl")
joblib.dump(route_le, "route_label_encoder.pkl")
joblib.dump(stid_le, "stid_label_encoder.pkl")
joblib.dump(arsid_le, "arsid_label_encoder.pkl")

print("\n모델과 인코더 저장 완료")

# =========================================================
# 14. 실서비스용 예측
# =========================================================
test_pred_raw = lgbm_reg.predict(X_test)
test_pred_service = np.clip(test_pred_raw, 0, MAX_SEAT)