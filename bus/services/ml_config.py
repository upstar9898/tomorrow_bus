import json
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "outputs_peak_v2",
    "models",
    "lgbm_hybrid_peak_congestion4"
)

ARTIFACT_DIR = os.path.join(
    PROJECT_ROOT,
    "training",
    "outputs_peak_v2",
    "artifacts"
)

MAX_SEAT = 45

reg_model = joblib.load(os.path.join(MODEL_DIR, "reg.pkl"))
full_model = joblib.load(os.path.join(MODEL_DIR, "full_cls.pkl"))

route_encoder = joblib.load(os.path.join(MODEL_DIR, "route_encoder.pkl"))
stid_encoder = joblib.load(os.path.join(MODEL_DIR, "stid_encoder.pkl"))
arsid_encoder = joblib.load(os.path.join(MODEL_DIR, "arsid_encoder.pkl"))

with open(os.path.join(MODEL_DIR, "feature_cols.json"), "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

pattern_route_stat_df = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stat.csv"),
    dtype={"busRouteId": str}
)

pattern_route_stop_stat_df = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stop_stat.csv"),
    dtype={"busRouteId": str, "stId": str}
)

pattern_route_stop_time_stat_df = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_stop_time_stat.csv"),
    dtype={"busRouteId": str, "stId": str}
)

pattern_route_staord_stat_df = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_staord_stat.csv"),
    dtype={"busRouteId": str}
)

pattern_route_time_stat_df = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "pattern_route_time_stat.csv"),
    dtype={"busRouteId": str}
)

with open(os.path.join(ARTIFACT_DIR, "pattern_meta.json"), "r", encoding="utf-8") as f:
    pattern_meta = json.load(f)

global_mean = float(pattern_meta["global_mean"])
global_low_ratio = float(pattern_meta["global_low_ratio"])

route_station_order_df = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "route_station_order.csv"),
    dtype={"busRouteId": str, "stId": str, "arsId": str}
)

route_travel_time_df = pd.read_csv(
    os.path.join(ARTIFACT_DIR, "route_segment_travel_time.csv"),
    dtype={"busRouteId": str}
)