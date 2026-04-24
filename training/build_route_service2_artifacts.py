# =========================================================
# build_route_service2_artifacts.py
# ---------------------------------------------------------
# 서비스2용 아티팩트 생성 전용 스크립트
#
# [역할]
# 1. travel_time 원본 파일 로드
# 2. train 날짜만 필터링
# 3. 서비스2용 travel_time 집계 테이블 생성
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
file_path = os.path.join(DATA_DIR, "bus_traveltime_merged_260424_rebuilt_v1.csv")

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

# rebuilt segment 파일은 staOrd 대신 to_staOrd를 사용
if "staOrd" not in df.columns and "to_staOrd" in df.columns:
    df["staOrd"] = df["to_staOrd"]

if "from_staOrd" in df.columns:
    df["from_staOrd"] = pd.to_numeric(df["from_staOrd"], errors="coerce")

if "to_staOrd" in df.columns:
    df["to_staOrd"] = pd.to_numeric(df["to_staOrd"], errors="coerce")

if "staOrd" not in df.columns and "to_staOrd" not in df.columns:
    raise KeyError("staOrd 또는 to_staOrd 컬럼이 필요합니다.")

df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
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
    subset=["busRouteId", "stId", "arsId", "from_staOrd", "to_staOrd", "travel_time"]
).copy()

# 음수 / 0 제외, 너무 큰 값 제거
df = df[(df["travel_time"] > 0) & (df["travel_time"] <= 1800)].copy()

df["staOrd"] = df["staOrd"].astype(int)

print("총 데이터 수:", len(df))
print("노선 수:", df["busRouteId"].nunique())
print("정류소 수:", df["stId"].nunique())

# =========================================================
# 5. 날짜 컬럼 생성
# ---------------------------------------------------------
# mkTm 우선 사용, 없으면 arrival_time 사용
# =========================================================
if "mkTm" in df.columns and df["mkTm"].notna().any():
    df["base_date"] = df["mkTm"].dt.date
elif "arrival_time" in df.columns and df["arrival_time"].notna().any():
    df["base_date"] = df["arrival_time"].dt.date
else:
    raise ValueError(
        "train 날짜 분리를 위한 시간 컬럼(mkTm 또는 arrival_time)이 없습니다."
    )

all_dates = sorted(df["base_date"].dropna().unique())
print("\n[전체 날짜 목록]")
print(all_dates)

# =========================================================
# 6. train 날짜 지정
# ---------------------------------------------------------
# 학습 split 기준:
# train: 2026-03-09 ~ 2026-03-27
# valid: 2026-03-28 ~ 2026-03-31
# test : 2026-04-01 ~ 2026-04-05
# =========================================================
train_dates = pd.date_range("2026-03-09", "2026-03-27").date.tolist()

train_df = df[df["base_date"].isin(train_dates)].copy()

if train_df.empty:
    raise ValueError(
        "train_df가 비어 있습니다. train_dates와 원본 날짜 범위를 확인하세요."
    )

print("\n[train 데이터 정보]")
print("train row 수:", len(train_df))
print("train 날짜 개수:", len(train_dates))
print("train 날짜 범위:", min(train_df["base_date"]), "~", max(train_df["base_date"]))
print("train 노선 수:", train_df["busRouteId"].nunique())
print("train 정류소 수:", train_df["stId"].nunique())

# =========================================================
# 7. 시간대 분리
# =========================================================
train_df["time_band"] = np.where(
    (train_df["is_holiday"] == 0) & (train_df["is_peak"] == 1),
    "peak",
    "normal"
)

print("\n[train time_band 분포]")
print(train_df["time_band"].value_counts())

# =========================================================
# 8. 서비스2용 travel_time 집계 테이블 생성
# ---------------------------------------------------------
# 해석:
# staOrd = k 의 travel_time은
# (k-1) -> k 구간의 이동시간으로 사용
# =========================================================
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

print("\n[route_station_travel_time 생성 완료 - train only]")
print(route_station_travel_time.head())

print("\n[route_station_travel_time 기초 확인]")
print("노선 수:", route_station_travel_time["busRouteId"].nunique())
print("행 수:", len(route_station_travel_time))
print("sample_count 요약:")
print(route_station_travel_time["sample_count"].describe())

# =========================================================
# 9. 저장
# ---------------------------------------------------------
# route_station_order.csv는 서비스2에서 더 이상 사용하지 않으므로 저장하지 않음
# =========================================================
route_station_travel_time_path = os.path.join(
    ARTIFACT_DIR,
    "route_station_travel_time.csv"
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

meta = {
    "source_file": file_path,
    "artifact_scope": "train_only",
    "train_dates": [str(x) for x in sorted(train_dates)],
    "time_band_rule": {
        "peak": "is_holiday == 0 and is_peak == 1",
        "normal": "otherwise"
    },
    "saved_files": [
        "route_station_travel_time.csv"
    ],
    "full_row_count": int(len(df)),
    "train_row_count": int(len(train_df)),
    "route_count": int(train_df["busRouteId"].nunique()),
    "station_count": int(train_df["stId"].nunique()),
}

with open(route_service2_meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n서비스2용 아티팩트 저장 완료")
print(f"- {route_station_travel_time_path}")
print(f"- {route_service2_meta_path}")