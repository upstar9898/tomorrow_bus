# =========================================================
# error_station_report.py
# 첫 정류소 오류 의심 케이스 분석
# =========================================================

import os
import glob
import pandas as pd

# =========================================================
# 1. 경로 설정
# =========================================================

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "prepared")
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. 데이터 로드 (원본 그대로)
# =========================================================

file_list = sorted(glob.glob(
    os.path.join(DATA_DIR, "bus_data_*_preprocessed_withweather_foranalysis.csv")
))

print(f"총 {len(file_list)}개 파일 로드 중...")

df_list = []
for f in file_list:
    tmp = pd.read_csv(f, low_memory=False)
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)
df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")

print(f"로드 완료: 총 {len(df):,}행")

# =========================================================
# 3. trip 기준 첫 정류소 찾기
# =========================================================

# trip_id 기준으로 정렬
df = df.sort_values(["trip_id", "staOrd", "mkTm"]).reset_index(drop=True)

# trip별 최솟값 staOrd (첫 정류소)
min_staord = df.groupby("trip_id")["staOrd"].transform("min")

# 첫 정류소 행만 추출
first_stop_df = df[df["staOrd"] == min_staord].copy()

print(f"첫 정류소 행 수: {len(first_stop_df):,}행")

# =========================================================
# 4. 오류 판단
# =========================================================

# trip별 첫 정류소 이후 최대 잔여좌석 (회복 여부 확인용)
def get_max_seat_after_first(group):
    """첫 정류소 이후 행들의 최대 잔여좌석"""
    min_ord = group["staOrd"].min()
    after = group[group["staOrd"] > min_ord]["remaining_seat"]
    if len(after) == 0:
        return None
    return after.max()

print("trip별 이후 최대 잔여좌석 계산 중...")
max_seat_after = df.groupby("trip_id").apply(
    get_max_seat_after_first
).reset_index()
max_seat_after.columns = ["trip_id", "max_seat_after"]

# 첫 정류소 df에 붙이기
first_stop_df = first_stop_df.merge(max_seat_after, on="trip_id", how="left")

# 오류 판단 기준 A: 새벽(5~6시) + 첫 정류소 0석
mask_a = (
    (first_stop_df["remaining_seat"] == 0) &
    (first_stop_df["hour"].isin([5, 6]))
)

# 오류 판단 기준 B: 첫 정류소 0석 → 이후 10석 이상 회복
mask_b = (
    (first_stop_df["remaining_seat"] == 0) &
    (first_stop_df["max_seat_after"] >= 10)
)

# 둘 중 하나라도 해당하면 오류 의심
error_df = first_stop_df[mask_a | mask_b].copy()

# 오류 유형 표시
error_df["error_type"] = ""
error_df.loc[mask_a & ~mask_b, "error_type"] = "A (새벽+만차)"
error_df.loc[~mask_a & mask_b, "error_type"] = "B (만차후회복)"
error_df.loc[mask_a & mask_b,  "error_type"] = "A+B (둘다해당)"

print(f"\n오류 의심 케이스: {len(error_df):,}건")

# =========================================================
# 5. 결과 정리
# =========================================================

# 정류소별 집계
station_summary = (
    error_df.groupby(["route_name", "staOrd", "station_name"])
    .agg(
        발생횟수       =("trip_id",        "count"),
        새벽오류       =("error_type",     lambda x: (x.str.contains("A")).sum()),
        회복오류       =("error_type",     lambda x: (x.str.contains("B")).sum()),
        평균회복좌석   =("max_seat_after", "mean"),
        최대회복좌석   =("max_seat_after", "max"),
        주요발생시간대 =("hour",           lambda x: x.mode()[0] if len(x) > 0 else None),
    )
    .reset_index()
    .sort_values(["route_name", "발생횟수"], ascending=[True, False])
)

# 전체 trip 수 대비 비율 계산
total_trips = df.groupby("route_name")["trip_id"].nunique().reset_index()
total_trips.columns = ["route_name", "전체trip수"]

station_summary = station_summary.merge(total_trips, on="route_name", how="left")
station_summary["발생비율(%)"] = (
    station_summary["발생횟수"] / station_summary["전체trip수"] * 100
).round(2)

# =========================================================
# 6. 출력 및 저장
# =========================================================

print("\n========== 오류 의심 정류소 요약 ==========")
print(station_summary.to_string(index=False))

# 상세 케이스 저장
error_detail = error_df[[
    "trip_id", "route_name", "staOrd", "station_name",
    "mkTm", "hour", "remaining_seat", "max_seat_after", "error_type"
]].copy()

error_detail.to_csv(
    os.path.join(OUTPUT_DIR, "error_station_detail.csv"),
    index=False, encoding="utf-8-sig"
)

station_summary.to_csv(
    os.path.join(OUTPUT_DIR, "error_station_summary.csv"),
    index=False, encoding="utf-8-sig"
)

print(f"\n저장 완료:")
print(f"  상세: eda_output/error_station_detail.csv")
print(f"  요약: eda_output/error_station_summary.csv")