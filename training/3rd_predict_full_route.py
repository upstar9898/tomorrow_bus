import json
import joblib
import numpy as np
import pandas as pd

# =========================================================
# 0. 파일 로드
# =========================================================
lgbm_reg = joblib.load("lgbm_point_seat_regressor_final.pkl")
lgbm_cls = joblib.load("lgbm_congestion_classifier_final.pkl")

route_le = joblib.load("route_label_encoder.pkl")
stid_le = joblib.load("stid_label_encoder.pkl")
arsid_le = joblib.load("arsid_label_encoder.pkl")

travel_time_table = pd.read_csv(
    "travel_time_table.csv",
    dtype={"busRouteId": str, "prev_stId": str, "stId": str}
)

route_station_order = pd.read_csv(
    "route_station_order.csv",
    dtype={"busRouteId": str, "stId": str, "arsId": str}
)

pattern_route_stat = pd.read_csv("pattern_route_stat.csv", dtype={"busRouteId": str})
pattern_route_stop_stat = pd.read_csv("pattern_route_stop_stat.csv", dtype={"busRouteId": str, "stId": str})
pattern_route_stop_time_stat = pd.read_csv("pattern_route_stop_time_stat.csv", dtype={"busRouteId": str, "stId": str})
pattern_route_staord_stat = pd.read_csv("pattern_route_staord_stat.csv", dtype={"busRouteId": str})
pattern_route_time_stat = pd.read_csv("pattern_route_time_stat.csv", dtype={"busRouteId": str})

with open("pattern_meta.json", "r", encoding="utf-8") as f:
    pattern_meta = json.load(f)

MAX_SEAT = pattern_meta["max_seat"]
GLOBAL_MEAN = pattern_meta["global_mean"]
GLOBAL_LOW_RATIO = pattern_meta["global_low_ratio"]

# =========================================================
# 1. feature 목록 (학습 코드와 동일해야 함)
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
# 2. 유틸
# =========================================================
def safe_transform(le, value):
    value = str(value).strip()
    if value not in set(le.classes_):
        raise ValueError(f"인코더에 없는 값입니다: {value}")
    return int(le.transform([value])[0])

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

    df["hour_group"] = (df["hour"] // 2).astype(int)
    df["minute_group"] = (df["minute"] // 10).astype(int)
    df["hour_weekday_key"] = df["hour"] * 10 + df["dayofweek"]

    return df

def add_pattern_features_service(df):
    df = df.merge(pattern_route_stat, on="busRouteId", how="left")
    df = df.merge(pattern_route_stop_stat, on=["busRouteId", "stId"], how="left")
    df = df.merge(
        pattern_route_stop_time_stat,
        on=["busRouteId", "stId", "dayofweek", "hour_group"],
        how="left"
    )
    df = df.merge(pattern_route_staord_stat, on=["busRouteId", "staOrd"], how="left")
    df = df.merge(pattern_route_time_stat, on=["busRouteId", "dayofweek", "hour"], how="left")

    mean_cols = [
        "route_mean_seat", "route_stop_mean_seat", "route_stop_time_mean_seat",
        "route_staord_mean_seat", "route_time_mean_seat"
    ]
    std_cols = [
        "route_std_seat", "route_stop_std_seat", "route_stop_time_std_seat",
        "route_staord_std_seat", "route_time_std_seat"
    ]
    low_ratio_cols = [
        "route_low_ratio", "route_stop_low_ratio", "route_stop_time_low_ratio",
        "route_staord_low_ratio", "route_time_low_ratio"
    ]

    for col in mean_cols:
        df[col] = df[col].fillna(GLOBAL_MEAN)

    for col in std_cols:
        df[col] = df[col].fillna(0)

    for col in low_ratio_cols:
        df[col] = df[col].fillna(GLOBAL_LOW_RATIO)

    return df

# =========================================================
# 3. 기준 정류소 기준 전체 시각 생성
# =========================================================
def build_route_schedule(route_id, base_stId, base_time):
    route_id = str(route_id).strip()
    base_stId = str(base_stId).strip()
    base_time = pd.to_datetime(base_time)

    route_df = (
        route_station_order[route_station_order["busRouteId"] == route_id]
        .copy()
        .sort_values("staOrd")
        .reset_index(drop=True)
    )

    if route_df.empty:
        raise ValueError("해당 노선이 없습니다.")
    if base_stId not in set(route_df["stId"]):
        raise ValueError("기준 정류소가 해당 노선에 없습니다.")

    base_ord = int(route_df.loc[route_df["stId"] == base_stId, "staOrd"].iloc[0])
    route_df["delta_seconds"] = 0.0

    # 이전 정류소
    prev_part = route_df[route_df["staOrd"] < base_ord].copy().sort_values("staOrd", ascending=False)
    acc = 0.0
    current_st = base_stId

    for idx, row in prev_part.iterrows():
        prev_st = row["stId"]
        match = travel_time_table[
            (travel_time_table["busRouteId"] == route_id) &
            (travel_time_table["prev_stId"] == prev_st) &
            (travel_time_table["stId"] == current_st)
        ]
        sec = float(match["avg_travel_time"].iloc[0]) if len(match) > 0 else 120.0
        acc += sec
        route_df.loc[idx, "delta_seconds"] = -acc
        current_st = prev_st

    # 이후 정류소
    next_part = route_df[route_df["staOrd"] > base_ord].copy().sort_values("staOrd")
    acc = 0.0
    current_st = base_stId

    for idx, row in next_part.iterrows():
        next_st = row["stId"]
        match = travel_time_table[
            (travel_time_table["busRouteId"] == route_id) &
            (travel_time_table["prev_stId"] == current_st) &
            (travel_time_table["stId"] == next_st)
        ]
        sec = float(match["avg_travel_time"].iloc[0]) if len(match) > 0 else 120.0
        acc += sec
        route_df.loc[idx, "delta_seconds"] = acc
        current_st = next_st

    route_df["pred_time"] = base_time + pd.to_timedelta(route_df["delta_seconds"], unit="s")
    return route_df

# =========================================================
# 4. 전체 정류소 예측
# =========================================================
def predict_full_route(route_id, base_stId, base_time):
    pred_df = build_route_schedule(route_id, base_stId, base_time)

    pred_df["route_enc"] = pred_df["busRouteId"].apply(lambda x: safe_transform(route_le, x))
    pred_df["stid_enc"] = pred_df["stId"].apply(lambda x: safe_transform(stid_le, x))

    pred_df = add_time_features(pred_df, time_col="pred_time")
    pred_df = add_pattern_features_service(pred_df)

    X = pred_df[FEATURE_COLS]

    pred_reg = lgbm_reg.predict(X)
    pred_reg = np.clip(pred_reg, 0, MAX_SEAT)

    pred_cls = lgbm_cls.predict(X)
    pred_cls_proba = lgbm_cls.predict_proba(X)

    result = pred_df[["busRouteId", "stId", "arsId", "staOrd", "pred_time"]].copy()
    result["pred_remaining_seat"] = np.round(pred_reg, 2)
    result["pred_remaining_seat_int"] = np.clip(np.round(pred_reg), 0, MAX_SEAT).astype(int)
    result["pred_congestion_class"] = pred_cls
    result["pred_congestion_prob_0"] = pred_cls_proba[:, 0]
    result["pred_congestion_prob_1"] = pred_cls_proba[:, 1]
    result["pred_congestion_prob_2"] = pred_cls_proba[:, 2]

    return result.sort_values("staOrd").reset_index(drop=True)

# =========================================================
# 5. 예시
# =========================================================
if __name__ == "__main__":
    result = predict_full_route(
        route_id="100100389",
        base_stId="206000321",
        base_time="2026-03-10 11:24:56"
    )

    print(result.head(20))
    result.to_csv("full_route_prediction_result.csv", index=False, encoding="utf-8-sig")
    print("저장 완료: full_route_prediction_result.csv")