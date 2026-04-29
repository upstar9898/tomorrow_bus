from pathlib import Path
import pandas as pd

# =========================================================
# staOrd + travel_time 기반으로 from_staOrd, to_staOrd 생성
# =========================================================

PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\langchain_semiproject\tomorrow_bus")

INPUT_PATH = PROJECT_ROOT / "data" / "bus_traveltime_260429_merged.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "bus_traveltime_260429_segment.csv"

df = pd.read_csv(
    INPUT_PATH,
    dtype={
        "busRouteId": str,
        "stId": str,
        "arsId": str,
        "vehId1": str,
    },
    low_memory=False,
)

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
df["arrival_time"] = pd.to_datetime(df.get("arrival_time"), errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")

df = df.dropna(
    subset=["mkTm", "busRouteId", "stId", "arsId", "staOrd", "travel_time"]
).copy()

# 현재 데이터 구조:
# staOrd = k 의 travel_time은 (k-1) -> k 구간 이동시간으로 해석
df["to_staOrd"] = df["staOrd"]
df["from_staOrd"] = df["staOrd"] - 1

df["from_staOrd"] = pd.to_numeric(df["from_staOrd"], errors="coerce")
df["to_staOrd"] = pd.to_numeric(df["to_staOrd"], errors="coerce")

# 첫 번째 정류장 이전 구간은 존재하지 않으므로 제외
df = df[df["from_staOrd"] >= 1].copy()

# 비정상 travel_time 제거
df = df[(df["travel_time"] > 0) & (df["travel_time"] <= 1800)].copy()

df["from_staOrd"] = df["from_staOrd"].astype(int)
df["to_staOrd"] = df["to_staOrd"].astype(int)
df["staOrd"] = df["staOrd"].astype(int)

df = df.sort_values(
    ["busRouteId", "vehId1", "mkTm", "from_staOrd", "to_staOrd"]
).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n[저장 완료]")
print(OUTPUT_PATH)
print("rows:", len(df))
print("routes:", df["busRouteId"].nunique())
print("stops:", df["stId"].nunique())
print("date range:", df["mkTm"].min(), "~", df["mkTm"].max())

print("\n[컬럼 확인]")
print(df[["busRouteId", "from_staOrd", "to_staOrd", "staOrd", "travel_time"]].head())