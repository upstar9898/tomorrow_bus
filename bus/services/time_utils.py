# ===============================================
# 실제 배차표상 해당 정류장에 도착 가능한 가장 가까운 미래 도착시각 보정
# ===============================================

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAVEL_TIME_CSV = BASE_DIR / "training" / "outputs_peak_v2" / "artifacts" / "route_station_travel_time.csv"


def get_cumulative_travel_sec(route_id, target_sta_ord, time_band="normal"):
    df = pd.read_csv(TRAVEL_TIME_CSV)

    route_df = df[
        (df["busRouteId"].astype(str) == str(route_id)) &
        (df["time_band"] == time_band) &
        (df["from_staOrd"] < int(target_sta_ord)) &
        (df["to_staOrd"] <= int(target_sta_ord))
    ].copy()

    if route_df.empty:
        return 0

    return float(route_df["median_travel_sec"].sum())


def get_next_arrival_time(
    user_datetime,
    first_time,
    last_time,
    interval_min,
    cumulative_travel_sec
):
    service_date = user_datetime.date()

    first_departure = datetime.combine(service_date, first_time)
    last_departure = datetime.combine(service_date, last_time)

    # 막차가 00:00처럼 첫차보다 이른 시각이면 익일 막차로 처리
    if last_departure <= first_departure:
        last_departure += timedelta(days=1)

    travel_delta = timedelta(seconds=float(cumulative_travel_sec))

    first_arrival = first_departure + travel_delta
    last_arrival = last_departure + travel_delta

    if user_datetime <= first_arrival:
        return first_arrival

    if user_datetime > last_arrival:
        return None

    interval = timedelta(minutes=int(interval_min))

    current_arrival = first_arrival
    while current_arrival < user_datetime:
        current_arrival += interval

    return current_arrival