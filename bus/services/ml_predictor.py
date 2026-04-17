import numpy as np
from .ml_config import reg_model, full_model, MAX_SEAT
from .ml_feature_builder import build_feature_row


def predict_service1_result(route_id, station_id, date_time, precipitation=0):
    feature_df = build_feature_row(
        route_id=route_id,
        station_id=station_id,
        date_time=date_time,
        precipitation=precipitation,
    )

    pred_seat = np.clip(reg_model.predict(feature_df), 0, MAX_SEAT)
    remaining_seat = int(np.clip(np.round(pred_seat[0]), 0, MAX_SEAT))
    full_prob = float(full_model.predict_proba(feature_df)[:, 1][0])

    return {
        "remaining_seat": remaining_seat,
        "full_prob": round(full_prob, 4),
        "date_time": date_time,
    }