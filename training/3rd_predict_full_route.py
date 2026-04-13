import os
import json
import joblib
import numpy as np
import pandas as pd

# =========================================================
# 0. 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models", "4class_0-5_6-15_16-30_31-45")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")


# =========================================================
# 1. 모델 및 인코더 로드
# =========================================================
lgbm_reg = joblib.load(os.path.join(MODEL_DIR, "reg.pkl"))
lgbm_cls = joblib.load(os.path.join(MODEL_DIR, "cls.pkl"))

route_le = joblib.load(os.path.join(MODEL_DIR, "route_encoder.pkl"))
stid_le = joblib.load(os.path.join(MODEL_DIR, "stid_encoder.pkl"))
arsid_le = joblib.load(os.path.join(MODEL_DIR, "arsid_encoder.pkl"))


# =========================================================
# 2. 필수 데이터 로드 (artifacts 기준)
# =========================================================
travel_time_table = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "travel_time_table.csv"),
    dtype={"busRouteId": str, "prev_stId": str, "stId": str}
)

route_station_order = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "route_station_order.csv"),
    dtype={"busRouteId": str, "stId": str, "arsId": str}
)

pattern_route_stat = pd.read_csv(os.path.join(ARTIFACT_DIR, "pattern_route_stat.csv"))
pattern_route_stop_stat = pd.read_csv(os.path.join(ARTIFACT_DIR, "pattern_route_stop_stat.csv"))
pattern_route_stop_time_stat = pd.read_csv(os.path.join(ARTIFACT_DIR, "pattern_route_stop_time_stat.csv"))
pattern_route_staord_stat = pd.read_csv(os.path.join(ARTIFACT_DIR, "pattern_route_staord_stat.csv"))
pattern_route_time_stat = pd.read_csv(os.path.join(ARTIFACT_DIR, "pattern_route_time_stat.csv"))

with open(os.path.join(ARTIFACT_DIR, "pattern_meta.json"), "r", encoding="utf-8") as f:
    pattern_meta = json.load(f)

MAX_SEAT = pattern_meta["max_seat"]
GLOBAL_MEAN = pattern_meta["global_mean"]
GLOBAL_LOW_RATIO = pattern_meta["global_low_ratio"]


# =========================================================
# 3. Feature 목록 (학습과 반드시 동일)
# =========================================================
FEATURE_COLS = [
    "route_enc", "stid_enc",
    "year", "month", "day",
    "hour", "minute", "dayofweek",
    "is_weekend", "is_peak", "peak_level",
    "month_sin", "month_cos",
    "day_sin", "day_cos",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dow_sin", "dow_cos",
    "staOrd",
    "hour_group",
    "minute_group",
    "hour_weekday_key",
    "route_mean_seat", "route_std_seat",
    "route_stop_mean_seat", "route_stop_std_seat",
    "route_stop_time_mean_seat", "route_stop_time_std_seat",
    "route_staord_mean_seat", "route_staord_std_seat",
    "route_time_mean_seat", "route_time_std_seat",
    "route_low_ratio",
    "route_stop_low_ratio",
    "route_stop_time_low_ratio",
    "route_staord_low_ratio",
    "route_time_low_ratio",
]


# =========================================================
# 4. 인코딩 안전 처리
# =========================================================
def safe_transform(le, value):
    value = str(value).strip()
    if value not in set(le.classes_):
        raise ValueError(f"❌ 인코더에 없는 값: {value}")
    return int(le.transform([value])[0])


# =========================================================
# 5. 시간 feature 생성
# =========================================================
def add_time_features(df, time_col="pred_time"):
    dt = pd.to_datetime(df[time_col])

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["dayofweek"] = dt.dt.dayofweek

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["is_peak"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9)) |
        ((df["hour"] >= 17) & (df["hour"] <= 19))
    ).astype(int)

    df["peak_level"] = 0
    df.loc[(df["hour"] >= 7) & (df["hour"] <= 8), "peak_level"] = 1
    df.loc[df["hour"] == 9, "peak_level"] = 2
    df.loc[(df["hour"] >= 17) & (df["hour"] <= 18), "peak_level"] = 3
    df.loc[df["hour"] == 19, "peak_level"] = 4

    # cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    df["hour_group"] = (df["hour"] // 2).astype(int)
    df["minute_group"] = (df["minute"] // 10).astype(int)
    df["hour_weekday_key"] = df["hour"] * 10 + df["dayofweek"]

    return df


# =========================================================
# 6. 패턴 feature 적용
# =========================================================
def add_pattern_features_service(df):
    df = df.merge(pattern_route_stat, on="busRouteId", how="left")
    df = df.merge(pattern_route_stop_stat, on=["busRouteId", "stId"], how="left")
    df = df.merge(pattern_route_stop_time_stat, on=["busRouteId", "stId", "dayofweek", "hour_group"], how="left")
    df = df.merge(pattern_route_staord_stat, on=["busRouteId", "staOrd"], how="left")
    df = df.merge(pattern_route_time_stat, on=["busRouteId", "dayofweek", "hour"], how="left")

    for col in ["route_mean_seat", "route_stop_mean_seat", "route_time_mean_seat"]:
        df[col] = df[col].fillna(GLOBAL_MEAN)

    return df


# =========================================================
# 7. 전체 노선 예측
# =========================================================
def predict_full_route(route_id, base_stId, base_time):

    pred_df = route_station_order[route_station_order["busRouteId"] == route_id].copy()

    pred_df["pred_time"] = pd.to_datetime(base_time)

    pred_df["route_enc"] = pred_df["busRouteId"].apply(lambda x: safe_transform(route_le, x))
    pred_df["stid_enc"] = pred_df["stId"].apply(lambda x: safe_transform(stid_le, x))

    pred_df = add_time_features(pred_df)
    pred_df = add_pattern_features_service(pred_df)

    X = pred_df[FEATURE_COLS]

    pred_reg = np.clip(lgbm_reg.predict(X), 0, MAX_SEAT)
    pred_cls = lgbm_cls.predict(X)

    pred_df["pred_remaining_seat"] = pred_reg
    pred_df["pred_congestion_class"] = pred_cls

    return pred_df


# =========================================================
# 8. 실행 예시
# =========================================================
if __name__ == "__main__":

    result = predict_full_route(
        route_id="100100389",
        base_stId="206000321",
        base_time="2026-03-10 11:24:56"
    )

    result.to_csv(
        os.path.join(ARTIFACT_DIR, "full_route_prediction_result.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("✅ 저장 완료:", os.path.join(ARTIFACT_DIR, "full_route_prediction_result.csv"))