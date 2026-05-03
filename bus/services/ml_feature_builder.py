import numpy as np
import pandas as pd
from django.db.models import Max

from bus.models import Route_station
from .ml_config import (
    route_encoder,
    stid_encoder,
    arsid_encoder,
    feature_cols,
    pattern_route_stat_df,
    pattern_route_stop_stat_df,
    pattern_route_stop_time_stat_df,
    pattern_route_staord_stat_df,
    pattern_route_time_stat_df,
    global_mean,
    global_low_ratio,
)
from .ml_utils import safe_label_encode, make_peak_level


def build_feature_row(
    route_id,
    station_id,
    date_time,
    precipitation=0,
    sta_ord=None,
    ars_id=None,
):
    route_id = str(route_id).strip()
    station_id = str(station_id).strip()
    dt = pd.to_datetime(date_time, errors="coerce")

    if pd.isna(dt):
        raise ValueError("date_time 형식이 올바르지 않습니다. 예: 2026-04-17 08:20:00")

    try:
        precipitation = float(precipitation)
    except Exception:
        precipitation = 0.0

    # sta_ord, ars_id가 없을 때만 DB 조회
    if sta_ord is None or ars_id is None:
        route_station = (
            Route_station.objects
            .filter(route_id=route_id, station_id=station_id)
            .select_related("station")
            .first()
        )

        if not route_station:
            raise ValueError(
                f"route_id={route_id}, station_id={station_id} 조합을 찾을 수 없습니다."
            )

        sta_ord = float(route_station.staOrd)
        ars_id = str(route_station.station.arsId).strip()
    else:
        sta_ord = float(sta_ord)
        ars_id = str(ars_id).strip()

    # 아래부터는 route_station 쓰지 말고 sta_ord, ars_id만 사용
    route_max_sta = (
        Route_station.objects
        .filter(route_id=route_id)
        .aggregate(max_sta=Max("staOrd"))
        .get("max_sta")
    )

    if route_max_sta is None or route_max_sta <= 0:
        sta_ratio = 0.0
    else:
        sta_ratio = float(sta_ord) / float(route_max_sta)

    row_df = pd.DataFrame([{
        "busRouteId": route_id,
        "stId": station_id,
        "arsId": ars_id,
        "staOrd": sta_ord,
        "sta_ratio": sta_ratio,
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "minute": dt.minute,
        "dayofweek": dt.dayofweek,
        "is_weekend": 1 if dt.dayofweek >= 5 else 0,
        "is_holiday": 0,
        "is_peak": 1 if (7 <= dt.hour <= 9 or 17 <= dt.hour <= 19) else 0,
        "peak_level": make_peak_level(dt.hour),
    }])

    row_df["month_sin"] = np.sin(2 * np.pi * row_df["month"] / 12)
    row_df["month_cos"] = np.cos(2 * np.pi * row_df["month"] / 12)
    row_df["day_sin"] = np.sin(2 * np.pi * row_df["day"] / 31)
    row_df["day_cos"] = np.cos(2 * np.pi * row_df["day"] / 31)
    row_df["hour_sin"] = np.sin(2 * np.pi * row_df["hour"] / 24)
    row_df["hour_cos"] = np.cos(2 * np.pi * row_df["hour"] / 24)
    row_df["minute_sin"] = np.sin(2 * np.pi * row_df["minute"] / 60)
    row_df["minute_cos"] = np.cos(2 * np.pi * row_df["minute"] / 60)
    row_df["dow_sin"] = np.sin(2 * np.pi * row_df["dayofweek"] / 7)
    row_df["dow_cos"] = np.cos(2 * np.pi * row_df["dayofweek"] / 7)

    row_df["hour_group"] = (row_df["hour"] // 2).astype(int)
    row_df["minute_group"] = (row_df["minute"] // 10).astype(int)
    row_df["minute_5"] = (row_df["minute"] // 5).astype(int)
    row_df["minute_3"] = (row_df["minute"] // 3).astype(int)
    row_df["hour_weekday_key"] = row_df["hour"] * 10 + row_df["dayofweek"]

    row_df["rainfall"] = precipitation
    row_df["precipitation"] = precipitation
    row_df["fog"] = 0.0
    row_df["temperature"] = 20.0
    row_df["rainfall_missing"] = 0
    row_df["is_rain"] = (row_df["rainfall"] > 0).astype(int)
    row_df["rain_peak"] = row_df["is_rain"] * row_df["is_peak"]

    row_df["route_enc"] = safe_label_encode(route_encoder, route_id, "route_id")
    row_df["stid_enc"] = safe_label_encode(stid_encoder, station_id, "station_id")
    row_df["arsid_enc"] = safe_label_encode(arsid_encoder, ars_id, "ars_id")

    row_df = row_df.merge(pattern_route_stat_df, on="busRouteId", how="left")
    row_df = row_df.merge(pattern_route_stop_stat_df, on=["busRouteId", "stId"], how="left")
    row_df = row_df.merge(
        pattern_route_stop_time_stat_df,
        on=["busRouteId", "stId", "dayofweek", "hour", "minute_group"],
        how="left"
    )
    row_df = row_df.merge(pattern_route_staord_stat_df, on=["busRouteId", "staOrd"], how="left")
    row_df = row_df.merge(pattern_route_time_stat_df, on=["busRouteId", "dayofweek", "hour"], how="left")

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
        if col not in row_df.columns:
            row_df[col] = global_mean
        row_df[col] = row_df[col].fillna(global_mean)

    for col in std_cols:
        if col not in row_df.columns:
            row_df[col] = 0.0
        row_df[col] = row_df[col].fillna(0.0)

    for col in low_ratio_cols:
        if col not in row_df.columns:
            row_df[col] = global_low_ratio
        row_df[col] = row_df[col].fillna(global_low_ratio)

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0

    return row_df[feature_cols].copy()