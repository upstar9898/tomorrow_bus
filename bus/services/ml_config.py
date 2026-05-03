import json
import os
import joblib
import pandas as pd



BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ROOT = os.path.join(BASE_DIR, "models")
ARTIFACT_DIR = os.path.join(MODEL_ROOT, "artifacts")
ENCODER_DIR = os.path.join(MODEL_ROOT, "encoder")
ML_MODEL_DIR = os.path.join(MODEL_ROOT, "ml_models")

MAX_SEAT = 45

reg_model = joblib.load(os.path.join(ML_MODEL_DIR, "reg.pkl"))
full_model = joblib.load(os.path.join(ML_MODEL_DIR, "full_cls.pkl"))
peak_model = joblib.load(os.path.join(ML_MODEL_DIR, "peak_congestion_cls.pkl"))

route_encoder = joblib.load(os.path.join(ENCODER_DIR, "route_encoder.pkl"))
stid_encoder = joblib.load(os.path.join(ENCODER_DIR, "stid_encoder.pkl"))
arsid_encoder = joblib.load(os.path.join(ENCODER_DIR, "arsid_encoder.pkl"))

with open(os.path.join(ENCODER_DIR, "feature_cols.json"), "r", encoding="utf-8") as f:
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

with open(os.path.join(ENCODER_DIR, "thresholds.json"), "r", encoding="utf-8") as f:
    thresholds = json.load(f)

peak_thresholds = thresholds.get("peak_congestion_thresholds", [0.15, 0.3, 0.4])
full_binary_threshold = thresholds.get("full_binary_threshold", 0.5)