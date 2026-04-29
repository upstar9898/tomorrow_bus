import os
import json
import numpy as np
import pandas as pd

# =========================================================
# 서비스2용 아티팩트 생성 코드
# - 입력: bus_traveltime_260429_segment.csv
# - 출력: route_station_travel_time.csv
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs_peak_v2")
ARTIFACT_DIR = os.path.join(OUTPUT_ROOT, "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

file_path = os.path.join(DATA_DIR, "bus_traveltime_260429_segment.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"데이터 파일을 찾을 수 없습니다.\n확인한 경로: {file_path}"
    )

print(f"[INFO] file_path   : {file_path}")
print(f"[INFO] ARTIFACT_DIR: {ARTIFACT_DIR}")

df = pd.read_csv(
    file_path,
    dtype={
        "busRouteId": str,
        "stId": str,
        "arsId": str,
        "vehId1": str,
    },
    low_memory=False,
)

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")

if "arrival_time" in df.columns:
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")

df["from_staOrd"] = pd.to_numeric(df["from_staOrd"], errors="coerce")
df["to_staOrd"] = pd.to_numeric(df["to_staOrd"], errors="coerce")
df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")

if "is_holiday" not in df.columns:
    df["is_holiday"] = 0

if "is_peak" not in df.columns:
    df["is_peak"] = 0

df["is_holiday"] = (
    pd.to_numeric(df["is_holiday"], errors="coerce")
    .fillna(0)
    .astype(int)
)

df["is_peak"] = (
    pd.to_numeric(df["is_peak"], errors="coerce")
    .fillna(0)
    .astype(int)
)

df = df.dropna(
    subset=[
        "mkTm",
        "busRouteId",
        "stId",
        "arsId",
        "from_staOrd",
        "to_staOrd",
        "travel_time",
    ]
).copy()

df = df[
    (df["from_staOrd"] >= 1)
    & (df["to_staOrd"] >= 2)
    & (df["travel_time"] >= 10)
    & (df["travel_time"] <= 1800)
].copy()

df["from_staOrd"] = df["from_staOrd"].astype(int)
df["to_staOrd"] = df["to_staOrd"].astype(int)

df["base_date"] = df["mkTm"].dt.date

# 모델 훈련 및 서비스2 아티팩트에 2026-04-26까지 반영
train_dates = pd.date_range("2026-03-09", "2026-04-26").date.tolist()

train_df = df[df["base_date"].isin(train_dates)].copy()

if train_df.empty:
    raise ValueError("train_df가 비어 있습니다. 날짜 범위와 원본 데이터를 확인하세요.")

print("\n[train 데이터 정보]")
print("train row 수:", len(train_df))
print("train 날짜 범위:", min(train_df["base_date"]), "~", max(train_df["base_date"]))
print("train 노선 수:", train_df["busRouteId"].nunique())
print("train 정류소 수:", train_df["stId"].nunique())

train_df["time_band"] = np.where(
    (train_df["is_holiday"] == 0) & (train_df["is_peak"] == 1),
    "peak",
    "normal",
)

route_station_travel_time = (
    train_df.groupby(["busRouteId", "from_staOrd", "to_staOrd", "time_band"])
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

route_station_travel_time_path = os.path.join(
    ARTIFACT_DIR,
    "route_station_travel_time.csv",
)

route_service2_meta_path = os.path.join(
    ARTIFACT_DIR,
    "route_service2_meta.json",
)

route_station_travel_time.to_csv(
    route_station_travel_time_path,
    index=False,
    encoding="utf-8-sig",
)

meta = {
    "source_file": file_path,
    "artifact_scope": "train_until_2026_04_26",
    "train_dates": [str(x) for x in sorted(train_dates)],
    "time_band_rule": {
        "peak": "is_holiday == 0 and is_peak == 1",
        "normal": "otherwise",
    },
    "saved_files": [
        "route_station_travel_time.csv",
    ],
    "full_row_count": int(len(df)),
    "train_row_count": int(len(train_df)),
    "route_count": int(train_df["busRouteId"].nunique()),
    "station_count": int(train_df["stId"].nunique()),
}

with open(route_service2_meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n[서비스2용 아티팩트 저장 완료]")
print(f"- {route_station_travel_time_path}")
print(f"- {route_service2_meta_path}")

print("\n[route_station_travel_time 확인]")
print(route_station_travel_time.head())
print("행 수:", len(route_station_travel_time))
print("노선 수:", route_station_travel_time["busRouteId"].nunique())