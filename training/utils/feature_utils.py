# =========================================================
# feature_utils.py
# ---------------------------------------------------------
# 전처리 / 시간 파생변수 / 라벨 생성 / 날짜 분할 / feature 목록 관리
# =========================================================

import pandas as pd
import numpy as np

from utils.config import (
    MAX_SEAT, LOW_SEAT_THRESHOLD
)


def seat_to_congestion_4(seat, max_seat=MAX_SEAT):
    seat = int(np.clip(np.round(seat), 0, max_seat))

    if seat == 0:
        return 0
    elif seat <= 20:
        return 1
    elif seat <= 30:
        return 2
    else:
        return 3


def seat_to_full_binary(seat, max_seat=MAX_SEAT):
    seat = int(np.clip(np.round(seat), 0, max_seat))
    return 1 if seat == 0 else 0


def congestion_label_text(cls):
    mapping = {
        0: "만차",
        1: "혼잡",
        2: "보통",
        3: "여유"
    }
    return mapping.get(int(cls), "알수없음")


def full_binary_label_text(cls):
    mapping = {
        0: "여석있음",
        1: "만차"
    }
    return mapping.get(int(cls), "알수없음")


def make_peak_level(hour):
    if hour == 7:
        return 1
    elif hour == 8:
        return 2
    elif hour == 9:
        return 3
    elif hour == 17:
        return 4
    elif hour == 18:
        return 5
    elif hour == 19:
        return 6
    return 0


def preprocess_base_dataframe(df: pd.DataFrame, max_seat: int = MAX_SEAT) -> pd.DataFrame:
    result = df.copy()

    required_cols = ["mkTm", "busRouteId", "stId", "arsId", "remaining_seat", "staOrd"]
    result = result.dropna(subset=required_cols).copy()

    result["busRouteId"] = result["busRouteId"].astype(str).str.strip()
    result["stId"] = result["stId"].astype(str).str.strip()
    result["arsId"] = result["arsId"].astype(str).str.strip()

    result["remaining_seat"] = pd.to_numeric(result["remaining_seat"], errors="coerce")
    result["staOrd"] = pd.to_numeric(result["staOrd"], errors="coerce")

    optional_numeric_cols = [
        "exps1", "full_flag", "precipitation", "fog",
        "temperature", "rainfall", "rainfall_missing"
    ]

    for col in optional_numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        else:
            result[col] = np.nan

    result = result.dropna(subset=["remaining_seat", "staOrd"]).copy()
    result = result[result["remaining_seat"] >= 0].copy()
    result["remaining_seat"] = result["remaining_seat"].clip(0, max_seat)

    return result


def preprocess_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["rainfall"] = result["rainfall"].fillna(0)
    result["precipitation"] = result["precipitation"].fillna(0)
    result["fog"] = result["fog"].fillna(0)

    result.loc[result["temperature"] <= -90, "temperature"] = np.nan
    result["temperature"] = result["temperature"].fillna(result["temperature"].median())

    result["rainfall_missing"] = result["rainfall_missing"].fillna(0).astype(int)

    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["year"] = result["mkTm"].dt.year
    result["month"] = result["mkTm"].dt.month
    result["day"] = result["mkTm"].dt.day
    result["hour"] = result["mkTm"].dt.hour
    result["minute"] = result["mkTm"].dt.minute
    result["dayofweek"] = result["mkTm"].dt.dayofweek

    result["date"] = result["mkTm"].dt.date
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)

    if "is_holiday" not in result.columns:
        result["is_holiday"] = 0
    else:
        result["is_holiday"] = (
            pd.to_numeric(result["is_holiday"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    result["is_peak"] = (
        ((result["hour"] >= 7) & (result["hour"] <= 9)) |
        ((result["hour"] >= 17) & (result["hour"] <= 19))
    ).astype(int)

    result["peak_level"] = result["hour"].apply(make_peak_level)

    return result


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    result["day_sin"] = np.sin(2 * np.pi * result["day"] / 31)
    result["day_cos"] = np.cos(2 * np.pi * result["day"] / 31)

    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)

    result["minute_sin"] = np.sin(2 * np.pi * result["minute"] / 60)
    result["minute_cos"] = np.cos(2 * np.pi * result["minute"] / 60)

    result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)

    return result


def add_auxiliary_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["hour_group"] = (result["hour"] // 2).astype(int)
    result["minute_group"] = (result["minute"] // 10).astype(int)
    result["minute_5"] = (result["minute"] // 5).astype(int)
    result["minute_3"] = (result["minute"] // 3).astype(int)
    result["hour_weekday_key"] = result["hour"] * 10 + result["dayofweek"]

    return result


def add_route_progress_feature(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    route_max_sta = result.groupby("busRouteId")["staOrd"].transform("max")
    result["sta_ratio"] = np.where(route_max_sta > 0, result["staOrd"] / route_max_sta, 0)

    return result


def add_target_and_weather_features(
    df: pd.DataFrame,
    low_seat_threshold: int = LOW_SEAT_THRESHOLD,
    max_seat: int = MAX_SEAT
) -> pd.DataFrame:
    result = df.copy()

    result["is_low_seat"] = (result["remaining_seat"] <= low_seat_threshold).astype(int)
    result["congestion_class"] = result["remaining_seat"].apply(lambda x: seat_to_congestion_4(x, max_seat))
    result["is_full_target"] = result["remaining_seat"].apply(lambda x: seat_to_full_binary(x, max_seat))

    result["is_rain"] = (result["rainfall"] > 0).astype(int)
    result["rain_peak"] = result["is_rain"] * result["is_peak"]

    return result


def prepare_training_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = preprocess_base_dataframe(result)
    result = preprocess_weather_columns(result)
    result = add_time_features(result)
    result = add_cyclical_features(result)
    result = add_auxiliary_time_keys(result)
    result = add_route_progress_feature(result)
    result = add_target_and_weather_features(result)

    for col in ["travel_time", "travel_arrival_gap_sec"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    if "travel_time_missing" in result.columns:
        result["travel_time_missing"] = (
            pd.to_numeric(result["travel_time_missing"], errors="coerce")
            .fillna(1)
            .astype(int)
        )

    result = result.sort_values("mkTm").reset_index(drop=True)
    return result

def split_by_date(df: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15):
    result = df.copy()

    unique_dates = sorted(result["date"].unique())

    if len(unique_dates) < 10:
        raise ValueError("최소 10일 이상은 있어야 안정적으로 분할 가능합니다.")

    n_dates = len(unique_dates)
    train_end = int(n_dates * train_ratio)
    valid_end = int(n_dates * (train_ratio + valid_ratio))

    train_dates = unique_dates[:train_end]
    valid_dates = unique_dates[train_end:valid_end]
    test_dates = unique_dates[valid_end:]

    train_df = result[result["date"].isin(train_dates)].copy()
    valid_df = result[result["date"].isin(valid_dates)].copy()
    test_df = result[result["date"].isin(test_dates)].copy()

    split_info = {
        "unique_dates": unique_dates,
        "train_dates": train_dates,
        "valid_dates": valid_dates,
        "test_dates": test_dates,
    }

    return train_df, valid_df, test_df, split_info


def get_feature_cols():
    return [
        "route_enc", "stid_enc", "arsid_enc",
        "year", "month", "day",
        "hour", "minute", "dayofweek",
        "is_weekend", "is_holiday", "is_peak", "peak_level",
        "month_sin", "month_cos",
        "day_sin", "day_cos",
        "hour_sin", "hour_cos",
        "minute_sin", "minute_cos",
        "dow_sin", "dow_cos",
        "staOrd", "sta_ratio",
        "hour_group", "minute_group", "minute_5", "minute_3", "hour_weekday_key",
        "route_mean_seat", "route_std_seat",
        "route_stop_mean_seat", "route_stop_std_seat",
        "route_stop_time_mean_seat", "route_stop_time_std_seat",
        "route_staord_mean_seat", "route_staord_std_seat",
        "route_time_mean_seat", "route_time_std_seat",
        "route_low_ratio", "route_stop_low_ratio",
        "route_stop_time_low_ratio", "route_staord_low_ratio",
        "route_time_low_ratio",
        "rainfall", "precipitation", "fog", "temperature",
        "rainfall_missing", "is_rain", "rain_peak",
        "travel_time", "travel_time_missing", "travel_arrival_gap_sec",
    ]