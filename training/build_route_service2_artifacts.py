# =========================================================
# build_route_service2_artifacts.py
# ---------------------------------------------------------
# 서비스2용 아티팩트 생성 전용 스크립트
#
# [역할]
# 1. bus_data_날짜별여행시간.csv 로드
# 2. 서비스2용 travel_time 집계 테이블 생성
# 3. 노선별 정류소 순서표 생성
# 4. csv 저장
#
# [시간대 분리 규칙]
# - is_holiday == 0 and is_peak == 1 -> peak
# - 그 외 -> normal
# =========================================================

import os
import json
import numpy as np
import pandas as pd

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
file_path = os.path.join(DATA_DIR, "bus_traveltime_merged_260422.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"데이터 파일을 찾을 수 없습니다.\n"
        f"확인한 경로: {file_path}"
    )

print(f"[INFO] file_path   : {file_path}")
print(f"[INFO] ARTIFACT_DIR: {ARTIFACT_DIR}")

# =========================================================
# 3. 원본 파일 불러오기
# =========================================================
df = pd.read_csv(
    file_path,
    dtype={
        "busRouteId": str,
        "stId": str,
        "arsId": str,
    },
    low_memory=False
)

# =========================================================
# 4. 기본 전처리
# =========================================================
if "mkTm" in df.columns:
    df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")

if "arrival_time" in df.columns:
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")

df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")

if "is_holiday" not in df.columns:
    df["is_holiday"] = 0
if "is_peak" not in df.columns:
    df["is_peak"] = 0

df["is_holiday"] = pd.to_numeric(df["is_holiday"], errors="coerce").fillna(0).astype(int)
df["is_peak"] = pd.to_numeric(df["is_peak"], errors="coerce").fillna(0).astype(int)

df = df.dropna(subset=["busRouteId", "stId", "arsId", "staOrd", "travel_time"]).copy()

# 음수 / 0 제외, 너무 큰 값 제거
df = df[(df["travel_time"] > 0) & (df["travel_time"] <= 1800)].copy()

df["staOrd"] = df["staOrd"].astype(int)

print("총 데이터 수:", len(df))
print("노선 수:", df["busRouteId"].nunique())
print("정류소 수:", df["stId"].nunique())

# =========================================================
# 5. 시간대 분리
# =========================================================
df["time_band"] = np.where(
    (df["is_holiday"] == 0) & (df["is_peak"] == 1),
    "peak",
    "normal"
)

print("\n[time_band 분포]")
print(df["time_band"].value_counts())

# =========================================================
# 6. 서비스2용 travel_time 집계 테이블 생성
# ---------------------------------------------------------
# 해석:
# staOrd = k 의 travel_time은
# (k-1) -> k 구간의 이동시간으로 사용
# =========================================================
route_station_travel_time = (
    df.groupby(["busRouteId", "staOrd", "time_band"])
    .agg(
        avg_travel_sec=("travel_time", "mean"),
        median_travel_sec=("travel_time", "median"),
        std_travel_sec=("travel_time", "std"),
        min_travel_sec=("travel_time", "min"),
        max_travel_sec=("travel_time", "max"),
        sample_count=("travel_time", "count"),
    )
    .reset_index()
)

route_station_travel_time["from_staOrd"] = (
    route_station_travel_time["staOrd"] - 1
).astype(int)

route_station_travel_time["to_staOrd"] = (
    route_station_travel_time["staOrd"]
).astype(int)

route_station_travel_time["std_travel_sec"] = (
    route_station_travel_time["std_travel_sec"].fillna(0)
)

route_station_travel_time = route_station_travel_time[
    [
        "busRouteId",
        "from_staOrd",
        "to_staOrd",
        "time_band",
        "avg_travel_sec",
        "median_travel_sec",
        "std_travel_sec",
        "min_travel_sec",
        "max_travel_sec",
        "sample_count",
    ]
].sort_values(
    ["busRouteId", "time_band", "from_staOrd", "to_staOrd"]
).reset_index(drop=True)

print("\n[route_station_travel_time 생성 완료]")
print(route_station_travel_time.head())

# =========================================================
# 7. 노선별 정류소 순서표 생성
# =========================================================
route_station_order = (
    df[["busRouteId", "stId", "arsId", "staOrd"]]
    .drop_duplicates()
    .sort_values(["busRouteId", "staOrd"])
    .reset_index(drop=True)
)

print("\n[route_station_order 생성 완료]")
print(route_station_order.head())

# =========================================================
# 8. 저장
# =========================================================
route_station_travel_time_path = os.path.join(
    ARTIFACT_DIR,
    "route_station_travel_time.csv"
)
route_station_order_path = os.path.join(
    ARTIFACT_DIR,
    "route_station_order.csv"
)
route_service2_meta_path = os.path.join(
    ARTIFACT_DIR,
    "route_service2_meta.json"
)

route_station_travel_time.to_csv(
    route_station_travel_time_path,
    index=False,
    encoding="utf-8-sig"
)
route_station_order.to_csv(
    route_station_order_path,
    index=False,
    encoding="utf-8-sig"
)

meta = {
    "source_file": file_path,
    "time_band_rule": {
        "peak": "is_holiday == 0 and is_peak == 1",
        "normal": "otherwise"
    },
    "saved_files": [
        "route_station_travel_time.csv",
        "route_station_order.csv"
    ],
    "row_count": int(len(df)),
    "route_count": int(df["busRouteId"].nunique()),
    "station_count": int(df["stId"].nunique()),
}

with open(route_service2_meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n서비스2용 아티팩트 저장 완료")
print(f"- {route_station_travel_time_path}")
print(f"- {route_station_order_path}")
print(f"- {route_service2_meta_path}")