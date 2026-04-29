# ==============================================
# 예측 전 운행 가능 시간 검증 코드
# ==============================================

import os
from datetime import datetime, timedelta
from functools import lru_cache

from .ml_config import ARTIFACT_DIR

import pandas as pd
from django.conf import settings

from bus.models import Bus_info, Route_station

MAX_GRACE_MINUTES = 30
MIN_GRACE_MINUTES = 15


ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "route_station_travel_time.csv")


@lru_cache(maxsize=1)
def load_route_station_travel_time():
    df = pd.read_csv(
        ARTIFACT_PATH,
        dtype={"busRouteId": str},
        low_memory=False,
    )

    df["from_staOrd"] = pd.to_numeric(df["from_staOrd"], errors="coerce")
    df["to_staOrd"] = pd.to_numeric(df["to_staOrd"], errors="coerce")
    df["median_travel_sec"] = pd.to_numeric(df["median_travel_sec"], errors="coerce")

    df = df.dropna(
        subset=["busRouteId", "from_staOrd", "to_staOrd", "median_travel_sec"]
    ).copy()

    df["from_staOrd"] = df["from_staOrd"].astype(int)
    df["to_staOrd"] = df["to_staOrd"].astype(int)

    return df


def get_day_type(target_datetime):
    weekday = target_datetime.weekday()

    if weekday == 5:
        return "saturday"

    if weekday == 6:
        return "holiday"

    return "weekday"


def get_operation_times(bus_info, day_type):
    if day_type == "saturday":
        return (
            bus_info.saturdayFirstTm,
            bus_info.saturdayLastTm,
            bus_info.saturdaytimeGap,
        )

    if day_type == "holiday":
        return (
            bus_info.holidayFirstTm,
            bus_info.holidayLastTm,
            bus_info.holidaytimeGap,
        )

    return (
        bus_info.firstTm,
        bus_info.lastTm,
        bus_info.timeGap,
    )


def get_station_staord(route_id, station_id):
    route_station = Route_station.objects.get(
        route_id=route_id,
        station_id=station_id,
    )
    return int(route_station.staOrd)


def get_cumulative_travel_sec(route_id, target_staord):
    if target_staord <= 1:
        return 0

    df = load_route_station_travel_time()

    route_df = df[
        (df["busRouteId"] == str(route_id))
        & (df["to_staOrd"] <= target_staord)
        & (df["from_staOrd"] >= 1)
    ].copy()

    # peak/normal 둘 다 있으면 중복 구간이 생기므로 normal 우선 사용
    normal_df = route_df[route_df["time_band"] == "normal"].copy()

    if not normal_df.empty:
        route_df = normal_df

    # 같은 구간이 중복될 가능성 방지
    route_df = route_df.sort_values(
        ["from_staOrd", "to_staOrd", "median_travel_sec"]
    ).drop_duplicates(subset=["from_staOrd", "to_staOrd"], keep="first")

    return float(route_df["median_travel_sec"].sum())


def combine_date_time(base_date, time_value):
    return datetime.combine(base_date, time_value)


def format_datetime_label(dt, base_date):
    if dt.date() > base_date:
        return f"익일 {dt.strftime('%H:%M')}"
    return dt.strftime("%H:%M")


def build_available_window(route_id, station_id, service_date, target_staord):
    bus_info = Bus_info.objects.select_related("route").get(route_id=route_id)

    dummy_dt = datetime.combine(service_date, datetime.min.time())
    day_type = get_day_type(dummy_dt)

    first_time, last_time, time_gap = get_operation_times(bus_info, day_type)

    first_dt = datetime.combine(service_date, first_time)
    last_dt = datetime.combine(service_date, last_time)

    if last_dt <= first_dt:
        last_dt += timedelta(days=1)

    cumulative_sec = get_cumulative_travel_sec(route_id, target_staord)

    available_start = first_dt + timedelta(seconds=cumulative_sec)

    grace_minutes = min(
        max(int(time_gap), MIN_GRACE_MINUTES),
        MAX_GRACE_MINUTES,
    )

    available_end = (
        last_dt + timedelta(seconds=cumulative_sec) + timedelta(minutes=grace_minutes)
    )

    return available_start, available_end, time_gap, first_time, last_time


def validate_operation_time(route_id, station_id, target_datetime):
    try:
        bus_info = Bus_info.objects.select_related("route").get(route_id=route_id)
    except Bus_info.DoesNotExist:
        return {
            "is_available": True,
            "reason": "NO_BUS_INFO",
            "message": "운행시간 정보가 없어 운행시간 검증을 생략합니다.",
            "available_start": None,
            "available_end": None,
        }

    try:
        target_staord = get_station_staord(route_id, station_id)

        service_dates = [
            target_datetime.date(),
            target_datetime.date() - timedelta(days=1),
        ]

        matched_window = None

        for service_date in service_dates:
            available_start, available_end, time_gap, first_time, last_time = (
                build_available_window(
                    route_id=route_id,
                    station_id=station_id,
                    service_date=service_date,
                    target_staord=target_staord,
                )
            )

            if available_start <= target_datetime <= available_end:
                matched_window = {
                    "available_start": available_start,
                    "available_end": available_end,
                    "time_gap": time_gap,
                    "service_date": service_date,
                    "first_time": first_time,
                    "last_time": last_time,
                }
                break

        if matched_window:
            return {
                "is_available": True,
                "reason": "AVAILABLE",
                "message": "운행 가능 시간입니다.",
                "available_start": matched_window["available_start"].isoformat(),
                "available_end": matched_window["available_end"].isoformat(),
                "available_start_label": format_datetime_label(
                    matched_window["available_start"],
                    matched_window["service_date"],
                ),
                "available_end_label": format_datetime_label(
                    matched_window["available_end"],
                    matched_window["service_date"],
                ),
                "time_gap": matched_window["time_gap"],
                "target_staord": target_staord,
                # 추가
                "first_time": matched_window["first_time"],
                "last_time": matched_window["last_time"],
                "interval_min": matched_window["time_gap"],
            }

    except Route_station.DoesNotExist:
        return {
            "is_available": True,
            "reason": "NO_ROUTE_STATION",
            "message": "정류장 순서 정보가 없어 운행시간 검증을 생략합니다.",
            "available_start": None,
            "available_end": None,
        }

    day_type = get_day_type(target_datetime)
    first_time, last_time, time_gap = get_operation_times(bus_info, day_type)

    base_date = target_datetime.date()

    first_dt = combine_date_time(base_date, first_time)
    last_dt = combine_date_time(base_date, last_time)

    # 막차 시간이 00:00처럼 첫차보다 이른 시각이면 익일로 처리
    if last_dt <= first_dt:
        last_dt += timedelta(days=1)

    cumulative_sec = get_cumulative_travel_sec(route_id, target_staord)

    available_start = first_dt + timedelta(seconds=cumulative_sec)
    available_end = last_dt + timedelta(seconds=cumulative_sec)

    if available_start <= target_datetime <= available_end:
        return {
            "is_available": True,
            "reason": "AVAILABLE",
            "message": "운행 가능 시간입니다.",
            "available_start": available_start.isoformat(),
            "available_end": available_end.isoformat(),
            "available_start_label": format_datetime_label(available_start, base_date),
            "available_end_label": format_datetime_label(available_end, base_date),
            "time_gap": time_gap,
            "target_staord": target_staord,
            "first_time": first_time,
            "last_time": last_time,
            "interval_min": time_gap,
        }

    message = (
        "선택한 시간은 해당 정류장의 운행 시간대가 아닙니다. "
        f"예측 가능 시간: {format_datetime_label(available_start, base_date)} "
        f"~ {format_datetime_label(available_end, base_date)}"
    )

    return {
        "is_available": False,
        "reason": "OUT_OF_OPERATION_TIME",
        "message": message,
        "available_start": available_start.isoformat(),
        "available_end": available_end.isoformat(),
        "available_start_label": format_datetime_label(available_start, base_date),
        "available_end_label": format_datetime_label(available_end, base_date),
        "time_gap": time_gap,
        "target_staord": target_staord,
        "first_time": first_time,
        "last_time": last_time,
        "interval_min": time_gap,
    }
