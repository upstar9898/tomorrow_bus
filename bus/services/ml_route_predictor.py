import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .ml_config import (
    reg_model,
    full_model,
    MAX_SEAT,
    route_travel_time_df,
)
from .ml_feature_builder import build_feature_row
from .ml_config import stid_encoder
from bus.models import Route_station

KNOWN_STATION_IDS = set(stid_encoder.classes_)


def _parse_date_time(date_time_str):
    dt = pd.to_datetime(date_time_str, errors="coerce")
    if pd.isna(dt):
        raise ValueError("date_time 형식이 올바르지 않습니다. 예: 2026-04-17 08:20:00")
    return dt.to_pydatetime()


def _get_route_stops(route_id):
    route_id = str(route_id).strip()

    qs = (
        Route_station.objects
        .filter(route_id=route_id)
        .select_related("station")
        .order_by("staOrd")
    )

    if not qs.exists():
        raise ValueError(f"노선 없음: {route_id}")

    data = []
    for obj in qs:
        data.append({
            "station_id": obj.station.stationId,
            "station_name": obj.station.stationName,
            "ars_id": str(obj.station.arsId),
            "sta_ord": int(obj.staOrd),
        })

    return data


def _get_base_stop_info(route_stops, station_id):
    station_id = str(station_id).strip()

    for row in route_stops:
        if str(row["station_id"]).strip() == station_id:
            return {
                "station_id": str(row["station_id"]).strip(),
                "station_name": row["station_name"],
                "ars_id": str(row["ars_id"]).strip() if row["ars_id"] else "",
                "sta_ord": int(row["sta_ord"]),
            }

    raise ValueError(
        f"입력한 station_id가 해당 route_id에 존재하지 않습니다. station_id={station_id}"
    )


def _build_segment_time_map(route_id):
    route_id = str(route_id).strip()

    df = route_travel_time_df[
        route_travel_time_df["busRouteId"] == route_id
    ].copy()

    if df.empty:
        raise ValueError(f"해당 노선의 구간 이동시간 정보가 없습니다. route_id={route_id}")

    df["from_staOrd"] = pd.to_numeric(df["from_staOrd"], errors="coerce")
    df["to_staOrd"] = pd.to_numeric(df["to_staOrd"], errors="coerce")
    df["median_travel_sec"] = pd.to_numeric(df["median_travel_sec"], errors="coerce")

    df = df.dropna(subset=["from_staOrd", "to_staOrd", "median_travel_sec"])

    segment_map = {}
    for _, row in df.iterrows():
        key = (int(row["from_staOrd"]), int(row["to_staOrd"]))
        segment_map[key] = float(row["median_travel_sec"])

    return segment_map


def _calc_relative_time_sec(base_sta_ord, target_sta_ord, segment_map):
    """
    기준 정류소(base_sta_ord)를 0초로 보고,
    target 정류소까지의 상대 시간을 초 단위로 반환
    - 이전 정류소: 음수
    - 이후 정류소: 양수
    """
    if target_sta_ord == base_sta_ord:
        return 0

    # 기준보다 이후 정류소
    if target_sta_ord > base_sta_ord:
        total_sec = 0.0
        for current_ord in range(base_sta_ord, target_sta_ord):
            key = (current_ord, current_ord + 1)
            if key not in segment_map:
                return None
            total_sec += segment_map[key]
        return int(round(total_sec))

    # 기준보다 이전 정류소
    total_sec = 0.0
    for current_ord in range(target_sta_ord, base_sta_ord):
        key = (current_ord, current_ord + 1)
        if key not in segment_map:
            return None
        total_sec += segment_map[key]
    return -int(round(total_sec))


def _make_relative_time_label(relative_time_sec):
    if relative_time_sec is None:
        return "시간 정보 없음"

    if relative_time_sec == 0:
        return "곧 도착"

    abs_sec = abs(int(relative_time_sec))
    total_minutes = round(abs_sec / 60)

    # 1분 미만 → 곧 도착
    if total_minutes < 1:
        return "곧 도착"

    hours = total_minutes // 60
    minutes = total_minutes % 60

    if hours == 0:
        return f"약 {minutes}분"
    
    if minutes == 0:
        return f"약 {hours}시간"
    
    return f"약 {hours}시간 {minutes}분"

def _make_predicted_time(base_dt, relative_time_sec):
    if relative_time_sec is None:
        return None

    predicted_dt = base_dt + timedelta(seconds=relative_time_sec)
    return predicted_dt.strftime("%Y-%m-%d %H:%M:%S")


def predict_service2_result(route_id, station_id, date_time, precipitation=0):
    route_id = str(route_id).strip()
    station_id = str(station_id).strip()

    base_dt = _parse_date_time(date_time)
    route_stops = _get_route_stops(route_id)
    base_stop = _get_base_stop_info(route_stops, station_id)
    base_sta_ord = base_stop["sta_ord"]

    segment_map = _build_segment_time_map(route_id)

    predictions = []

    for row in route_stops:
        target_station_id = row["station_id"]

        if target_station_id not in KNOWN_STATION_IDS:
            continue
        
        target_ars_id = row["ars_id"]
        target_sta_ord = row["sta_ord"]
        station_name = row["station_name"]

        relative_stop_offset = target_sta_ord - base_sta_ord
        relative_time_sec = _calc_relative_time_sec(
            base_sta_ord=base_sta_ord,
            target_sta_ord=target_sta_ord,
            segment_map=segment_map,
        )
        relative_time_label = _make_relative_time_label(relative_time_sec)
        predicted_time = _make_predicted_time(base_dt, relative_time_sec)

        feature_df = build_feature_row(
            route_id=route_id,
            station_id=target_station_id,
            date_time=predicted_time if predicted_time is not None else date_time,
            precipitation=precipitation,
            sta_ord=target_sta_ord,
            ars_id=target_ars_id,
        )

        pred_seat = np.clip(reg_model.predict(feature_df), 0, MAX_SEAT)
        remaining_seat = int(np.clip(np.round(pred_seat[0]), 0, MAX_SEAT))
        full_prob = float(full_model.predict_proba(feature_df)[:, 1][0])

        predictions.append(
            {
                "route_id": route_id,
                "station_id": target_station_id,
                "station_name": station_name,  # 실제 이름 있으면 여기 교체
                "ars_id": target_ars_id,
                "sta_ord": target_sta_ord,
                "relative_stop_offset": relative_stop_offset,
                "relative_time_sec": relative_time_sec,
                "relative_time_label": relative_time_label,
                "predicted_time": predicted_time,
                "remaining_seat": remaining_seat,
                "full_prob": round(full_prob, 4),
            }
        )

    return {
        "route_id": route_id,
        "base_station_id": station_id,
        "base_date_time": date_time,
        "predictions": predictions,
    }