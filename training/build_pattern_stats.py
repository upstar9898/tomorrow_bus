# =========================================================
# build_pattern_stats.py
# ---------------------------------------------------------
# train 데이터 기준 패턴 통계 CSV 생성 / 저장 전용 스크립트
#
# [역할]
# 1. 원본 데이터 로드
# 2. 기본 전처리
# 3. 시간 / 날씨 / 보조 라벨 생성
# 4. 날짜 기준 train / valid / test 분할
# 5. train_df 기준 패턴 통계 생성
# 6. csv / json 저장
# =========================================================

import os
import json
import pandas as pd
import numpy as np

from utils.feature_utils import (
    MAX_SEAT,
    prepare_training_base_dataframe,
    split_by_date,
)

# =========================================================
# 1. 프로젝트 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs_peak_v2")
ARTIFACT_DIR = os.path.join(OUTPUT_ROOT, "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================================================
# 2. 데이터 파일 경로
# =========================================================
file_path = os.path.join(DATA_DIR, "bus_all_raw_weather_260424_rebuilt_v2.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"데이터 파일을 찾을 수 없습니다.\n"
        f"확인한 경로: {file_path}"
    )

print(f"[INFO] file_path   : {file_path}")
print(f"[INFO] ARTIFACT_DIR: {ARTIFACT_DIR}")

# =========================================================
# 3. 설정값
# =========================================================
LOW_SEAT_THRESHOLD = 10

# =========================================================
# 4. 원본 파일 불러오기
# =========================================================
df = pd.read_csv(
    file_path,
    dtype={
        "busRouteId": str,
        "stId": str,
        "arsId": str
    },
    low_memory=False
)

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")

df = prepare_training_base_dataframe(df)
train_df, valid_df, test_df, split_info = split_by_date(df)

train_dates = split_info["train_dates"]
valid_dates = split_info["valid_dates"]
test_dates = split_info["test_dates"]

print("총 데이터 수:", len(df))
print("\n사용 날짜들:", split_info["unique_dates"])
print("총 사용 날짜 수:", len(split_info["unique_dates"]))
print("train dates:", train_dates[0], "~", train_dates[-1], f"({len(train_dates)}일)")
print("valid dates:", valid_dates[0], "~", valid_dates[-1], f"({len(valid_dates)}일)")
print("test dates :", test_dates[0], "~", test_dates[-1], f"({len(test_dates)}일)")

print("\n[분할 결과]")
print("train_df:", train_df.shape)
print("valid_df:", valid_df.shape)
print("test_df :", test_df.shape)


# =========================================================
# 11. train 기준 패턴 통계 생성
# =========================================================
route_stat = (
    train_df.groupby("busRouteId")
    .agg(
        route_mean_seat=("remaining_seat", "mean"),
        route_std_seat=("remaining_seat", "std"),
        route_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

route_stop_stat = (
    train_df.groupby(["busRouteId", "stId"])
    .agg(
        route_stop_mean_seat=("remaining_seat", "mean"),
        route_stop_std_seat=("remaining_seat", "std"),
        route_stop_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

route_stop_time_stat = (
    train_df.groupby(["busRouteId", "stId", "dayofweek", "hour", "minute_group"])
    .agg(
        route_stop_time_mean_seat=("remaining_seat", "mean"),
        route_stop_time_std_seat=("remaining_seat", "std"),
        route_stop_time_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

route_staord_stat = (
    train_df.groupby(["busRouteId", "staOrd"])
    .agg(
        route_staord_mean_seat=("remaining_seat", "mean"),
        route_staord_std_seat=("remaining_seat", "std"),
        route_staord_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

route_time_stat = (
    train_df.groupby(["busRouteId", "dayofweek", "hour"])
    .agg(
        route_time_mean_seat=("remaining_seat", "mean"),
        route_time_std_seat=("remaining_seat", "std"),
        route_time_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

# std 결측 보정
for stat_df in [route_stat, route_stop_stat, route_stop_time_stat, route_staord_stat, route_time_stat]:
    std_cols = [col for col in stat_df.columns if col.endswith("_std_seat")]
    for col in std_cols:
        stat_df[col] = stat_df[col].fillna(0)

# =========================================================
# 12. 패턴 통계 저장
# =========================================================
route_stat_path = os.path.join(ARTIFACT_DIR, "pattern_route_stat.csv")
route_stop_stat_path = os.path.join(ARTIFACT_DIR, "pattern_route_stop_stat.csv")
route_stop_time_stat_path = os.path.join(ARTIFACT_DIR, "pattern_route_stop_time_stat.csv")
route_staord_stat_path = os.path.join(ARTIFACT_DIR, "pattern_route_staord_stat.csv")
route_time_stat_path = os.path.join(ARTIFACT_DIR, "pattern_route_time_stat.csv")

route_stat.to_csv(route_stat_path, index=False, encoding="utf-8-sig")
route_stop_stat.to_csv(route_stop_stat_path, index=False, encoding="utf-8-sig")
route_stop_time_stat.to_csv(route_stop_time_stat_path, index=False, encoding="utf-8-sig")
route_staord_stat.to_csv(route_staord_stat_path, index=False, encoding="utf-8-sig")
route_time_stat.to_csv(route_time_stat_path, index=False, encoding="utf-8-sig")

pattern_meta = {
    "source_file": file_path,
    "train_date_range": {
        "start": str(train_dates[0]),
        "end": str(train_dates[-1]),
        "num_days": len(train_dates),
    },
    "valid_date_range": {
        "start": str(valid_dates[0]),
        "end": str(valid_dates[-1]),
        "num_days": len(valid_dates),
    },
    "test_date_range": {
        "start": str(test_dates[0]),
        "end": str(test_dates[-1]),
        "num_days": len(test_dates),
    },
    "global_mean": float(train_df["remaining_seat"].mean()),
    "global_low_ratio": float(train_df["is_low_seat"].mean()),
    "max_seat": int(MAX_SEAT),
    "saved_files": [
        "pattern_route_stat.csv",
        "pattern_route_stop_stat.csv",
        "pattern_route_stop_time_stat.csv",
        "pattern_route_staord_stat.csv",
        "pattern_route_time_stat.csv",
    ]
}

pattern_meta_path = os.path.join(ARTIFACT_DIR, "pattern_meta.json")

with open(pattern_meta_path, "w", encoding="utf-8") as f:
    json.dump(pattern_meta, f, ensure_ascii=False, indent=2)

print("\n패턴 통계 저장 완료")
print(f"- {route_stat_path}")
print(f"- {route_stop_stat_path}")
print(f"- {route_stop_time_stat_path}")
print(f"- {route_staord_stat_path}")
print(f"- {route_time_stat_path}")
print(f"- {pattern_meta_path}")