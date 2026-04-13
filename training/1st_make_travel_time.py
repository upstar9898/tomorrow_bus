import pandas as pd

# ==============================================
# 1단계. 이동시간 테이블 만들기
# ==============================================

file_path = "bus_traveltime_merged.csv"

df = pd.read_csv(
    file_path,
    dtype={
        "busRouteId": str,
        "stId": str,
        "arsId": str,
        "vehId1": str
    },
    low_memory=False
)

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")

df = df.dropna(subset=["mkTm", "busRouteId", "stId", "vehId1", "staOrd"]).copy()

for col in ["busRouteId", "stId", "arsId", "vehId1"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df = df.sort_values(["busRouteId", "vehId1", "mkTm", "staOrd"]).reset_index(drop=True)

# 이전 행 정보
df["prev_mkTm"] = df.groupby(["busRouteId", "vehId1"])["mkTm"].shift(1)
df["prev_stId"] = df.groupby(["busRouteId", "vehId1"])["stId"].shift(1)
df["prev_staOrd"] = df.groupby(["busRouteId", "vehId1"])["staOrd"].shift(1)

# 이동시간 계산
df["travel_time"] = (df["mkTm"] - df["prev_mkTm"]).dt.total_seconds()

# 바로 다음 정류장으로 간 경우만 사용
df_tt = df[
    (df["staOrd"] - df["prev_staOrd"] == 1)
].copy()

# 이상치 제거
df_tt = df_tt[df_tt["travel_time"].between(10, 600)].copy()

travel_time_table = (
    df_tt.groupby(["busRouteId", "prev_stId", "stId"])
    .agg(
        avg_travel_time=("travel_time", "mean"),
        median_travel_time=("travel_time", "median"),
        sample_count=("travel_time", "count")
    )
    .reset_index()
)

travel_time_table.to_csv("travel_time_table.csv", index=False, encoding="utf-8-sig")
print("저장 완료: travel_time_table.csv")
print(travel_time_table.head())

# ===============================================================
# 2단계. 노선별 정류소 순서 테이블 저장
# ===============================================================

df = pd.read_csv(
    "bus_all_raw3.csv",
    dtype={"busRouteId": str, "stId": str, "arsId": str},
    low_memory=False
)

df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
df = df.dropna(subset=["busRouteId", "stId", "arsId", "staOrd"]).copy()

for col in ["busRouteId", "stId", "arsId"]:
    df[col] = df[col].astype(str).str.strip()

route_station_order = (
    df[["busRouteId", "stId", "arsId", "staOrd"]]
    .drop_duplicates()
    .sort_values(["busRouteId", "staOrd"])
    .reset_index(drop=True)
)

route_station_order.to_csv("route_station_order.csv", index=False, encoding="utf-8-sig")
print("저장 완료: route_station_order.csv")