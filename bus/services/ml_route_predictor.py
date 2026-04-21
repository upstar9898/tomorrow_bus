import numpy as np
import pandas as pd

from .ml_config import (
    reg_model,
    full_model,
    MAX_SEAT,
    route_station_order_df,
)
from .ml_feature_builder import build_feature_row


def _get_route_stops(route_id):
    route_id = str(route_id).strip()

    df = route_station_order_df[
        route_station_order_df["busRouteId"] == route_id
    ].copy()

    if df.empty:
        raise ValueError(f"해당 노선의 정류소 순서 정보가 없습니다. route_id={route_id}")

    df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
    df = df.dropna(subset=["staOrd"]).sort_values("staOrd").reset_index(drop=True)
    df["staOrd"] = df["staOrd"].astype(int)

    return df


def _get_base_sta_ord(route_stops_df, station_id):
    station_id = str(station_id).strip()

    matched = route_stops_df[route_stops_df["stId"] == station_id]
    if matched.empty:
        raise ValueError(
            f"입력한 station_id가 해당 route_id에 존재하지 않습니다. station_id={station_id}"
        )

    return int(matched.iloc[0]["staOrd"])


def _make_relative_stop_label(relative_stop_offset):
    if relative_stop_offset == 0:
        return "기준 정류소"
    if relative_stop_offset > 0:
        return f"+{relative_stop_offset}정류소"
    return f"{relative_stop_offset}정류소"


def predict_service2_result(route_id, station_id, date_time, precipitation=0):
    route_id = str(route_id).strip()
    station_id = str(station_id).strip()

    route_stops_df = _get_route_stops(route_id)
    base_sta_ord = _get_base_sta_ord(route_stops_df, station_id)

    predictions = []

    for _, row in route_stops_df.iterrows():
        target_station_id = str(row["stId"]).strip()
        target_ars_id = str(row["arsId"]).strip() if pd.notna(row["arsId"]) else ""
        target_sta_ord = int(row["staOrd"])

        relative_stop_offset = target_sta_ord - base_sta_ord
        relative_stop_label = _make_relative_stop_label(relative_stop_offset)

        feature_df = build_feature_row(
            route_id=route_id,
            station_id=target_station_id,
            date_time=date_time,
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
                "ars_id": target_ars_id,
                "sta_ord": target_sta_ord,
                "relative_stop_offset": relative_stop_offset,
                "relative_stop_label": relative_stop_label,
                "date_time": date_time,
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