# =========================================================
# config.py
# ---------------------------------------------------------
# 공통 설정값 관리
# =========================================================

MAX_SEAT = 45
LOW_SEAT_THRESHOLD = 10

PEAK_THRESHOLDS = {
    0: 0.15,   # 만차
    1: 0.30,   # 혼잡
    2: 0.40    # 보통
}

FULL_BINARY_THRESHOLD = 0.50

RUNNER = "eunbyeol"
DATASET_NAME = "bus_all_raw_weather_260428"
DATA_VERSION = "20260428"
SPLIT_VERSION = "date_70_15_15"
FEATURE_VERSION = "pattern_weather_peak_congestion4_cleanlog_v1"

REG_MODEL_NAME = "lgbm_reg"
PEAK_CONGESTION_MODEL_NAME = "lgbm_peak_congestion_cls_4class"
FULL_MODEL_NAME = "lgbm_full_binary_cls"

MODEL_VERSION = "v2_peak_0_1to20_21to30_31to45_cleanlog"

LABEL_DEFINITION_NAME = "peak_congestion_4class_0_1to20_21to30_31to45"
LABEL_DEFINITION_DETAIL = {
    "0": "만차(0석)",
    "1": "혼잡(1~20석)",
    "2": "보통(21~30석)",
    "3": "여유(31~45석)"
}

CONGESTION_CLASS_LABELS = ["만차", "혼잡", "보통", "여유"]
FULL_BINARY_LABELS = ["여석있음", "만차"]