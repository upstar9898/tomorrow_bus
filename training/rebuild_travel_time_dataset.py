# merge_traveltime_files.py

from pathlib import Path
import pandas as pd

# =========================================================
# 1. 경로 설정
# =========================================================
DATA_DIR = Path(r"C:\Users\Administrator\Desktop\langchain_semiproject\tomorrow_bus\data\traveltime_raw")
OUTPUT_PATH = DATA_DIR / "bus_traveltime_merged_260424_rebuilt_v1.csv"

# 파일명 패턴은 실제 파일명에 맞게 수정
file_paths = sorted(DATA_DIR.glob("*traveltime*.csv"))

if not file_paths:
    raise FileNotFoundError("traveltime CSV 파일을 찾지 못했습니다. 파일명 패턴을 확인하세요.")

print("[읽을 파일 목록]")
for p in file_paths:
    print("-", p.name)

# =========================================================
# 2. 날짜별 파일 병합
# =========================================================
dfs = []

for path in file_paths:
    temp = pd.read_csv(
        path,
        dtype={
            "busRouteId": str,
            "stId": str,
            "arsId": str,
            "vehId1": str,
        },
        low_memory=False,
    )
    temp["source_file"] = path.name
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

# =========================================================
# 3. 기본 타입 정리
# =========================================================
df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")

df = df.dropna(
    subset=["mkTm", "arrival_time", "busRouteId", "stId", "arsId", "staOrd", "vehId1"]
).copy()

df["staOrd"] = df["staOrd"].astype(int)

# =========================================================
# 4. 중복 제거
# ---------------------------------------------------------
# 같은 차량이 같은 정류소에 여러 번 잡힌 경우
# 가장 빠른 arrival_time 1개만 남김
# =========================================================
df = (
    df.sort_values(["busRouteId", "vehId1", "arrival_time", "staOrd"])
    .drop_duplicates(
        subset=["busRouteId", "vehId1", "stId", "staOrd", "arrival_time"],
        keep="first"
    )
    .copy()
)

# =========================================================
# 5. 차량별 정렬 후 운행 회차 분리 + travel_time 재계산
# =========================================================
df["operation_date"] = df["arrival_time"].dt.date

df = df.sort_values(
    ["busRouteId", "vehId1", "operation_date", "arrival_time", "staOrd"]
).reset_index(drop=True)

base_group_cols = ["busRouteId", "vehId1", "operation_date"]

# 같은 날짜/차량 안에서 직전 staOrd 확인
df["raw_prev_staOrd"] = df.groupby(base_group_cols)["staOrd"].shift(1)
df["raw_prev_arrival_time"] = df.groupby(base_group_cols)["arrival_time"].shift(1)

# staOrd가 뒤로 크게 돌아가면 새 운행 회차 시작으로 판단
# 예: 65 -> 2
df["is_new_trip"] = (
    df["raw_prev_staOrd"].isna()
    | (df["staOrd"] <= df["raw_prev_staOrd"])
).astype(int)

# trip_id 생성
df["trip_id"] = df.groupby(base_group_cols)["is_new_trip"].cumsum()

group_cols = ["busRouteId", "vehId1", "operation_date", "trip_id"]

df["prev_staOrd"] = df.groupby(group_cols)["staOrd"].shift(1)
df["prev_arrival_time"] = df.groupby(group_cols)["arrival_time"].shift(1)

df["recalc_travel_time"] = (
    df["arrival_time"] - df["prev_arrival_time"]
).dt.total_seconds()

# =========================================================
# 6. gap 보간 방식 적용
# ---------------------------------------------------------
# 예:
# prev=1, curr=5
# total_time = 120초
#
# → 1→2 = 30
# → 2→3 = 30
# → 3→4 = 30
# → 4→5 = 30
# =========================================================

expanded_rows = []

for _, row in df.iterrows():

    prev_sta = row["prev_staOrd"]
    curr_sta = row["staOrd"]
    total_sec = row["recalc_travel_time"]

    if pd.isna(prev_sta) or pd.isna(total_sec):
        continue

    prev_sta = int(prev_sta)
    curr_sta = int(curr_sta)

    gap = curr_sta - prev_sta

    # 역방향 / 이상 데이터 제거
    if gap <= 0:
        continue

    if gap > 8:
        continue

    # 너무 큰 시간 제거
    if total_sec <= 0 or total_sec > 1800:
        continue

    per_segment_sec = total_sec / gap

    if per_segment_sec < 3:
        continue

    for target_sta in range(prev_sta + 1, curr_sta + 1):
        expanded_rows.append({
            "mkTm": row["mkTm"],
            "arrival_time": row["arrival_time"],
            "busRouteId": row["busRouteId"],
            "stId": row["stId"],
            "arsId": row["arsId"],
            "vehId1": row["vehId1"],
            "from_staOrd": target_sta - 1,
            "to_staOrd": target_sta,
            "travel_time": per_segment_sec
        })

expanded_df = pd.DataFrame(expanded_rows)

print("\n[보간 후 travel_time 데이터]")
print(expanded_df.head())

print("총 segment 수:", len(expanded_df))

# =========================================================
# 7. 확인 로그
# =========================================================
print("\n[재계산 후 전체 정보]")
print("총 행 수:", len(df))
print("노선 수:", df["busRouteId"].nunique())
print("정류소 수:", df["stId"].nunique())

print("\n[노선별 staOrd 최소/최대]")
print(
    df.groupby("busRouteId")["staOrd"]
    .agg(["min", "max", "nunique"])
    .sort_index()
)

print("\n[travel_time 있는 행 기준 staOrd 최소/최대]")
valid_tt = df.dropna(subset=["travel_time"]).copy()
print(
    valid_tt.groupby("busRouteId")["staOrd"]
    .agg(["min", "max", "nunique"])
    .sort_index()
)

print("\n[100100389 앞 정류소 확인]")
tmp = df[df["busRouteId"] == "100100389"].copy()
print(
    tmp[tmp["staOrd"] <= 5][
        ["mkTm", "arrival_time", "busRouteId", "stId", "arsId", "staOrd", "vehId1", "prev_staOrd", "travel_time"]
    ].head(30)
)

# =========================================================
# 8. 저장
# =========================================================
# 기존 컬럼 순서 최대한 유지
preferred_cols = [
    "mkTm",
    "arrival_time",
    "busRouteId",
    "stId",
    "arsId",
    "staOrd",
    "vehId1",
    "travel_time",
]

extra_cols = [col for col in df.columns if col not in preferred_cols]
df = df[preferred_cols + extra_cols]

print("expanded_df travel_time NaN 수:", expanded_df["travel_time"].isna().sum())

expanded_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n[저장 완료]")
print(OUTPUT_PATH)