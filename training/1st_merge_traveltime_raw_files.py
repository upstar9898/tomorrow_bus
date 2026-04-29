from pathlib import Path
import pandas as pd

# =========================================================
# 날짜별 traveltime csv 단순 병합
# =========================================================

PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\langchain_semiproject\tomorrow_bus")
DATA_DIR = PROJECT_ROOT / "data" / "traveltime_raw"

OUTPUT_PATH = PROJECT_ROOT / "data" / "bus_traveltime_260429_merged.csv"

file_paths = sorted(DATA_DIR.glob("*preprocessed*.csv"))

if not file_paths:
    raise FileNotFoundError(f"병합할 CSV 파일이 없습니다: {DATA_DIR}")

dfs = []

for path in file_paths:
    print(f"[LOAD] {path.name}")

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

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
df["arrival_time"] = pd.to_datetime(df.get("arrival_time"), errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
df["travel_time"] = pd.to_numeric(df.get("travel_time"), errors="coerce")

required_cols = [
    "mkTm",
    "busRouteId",
    "stId",
    "arsId",
    "staOrd",
    "vehId1",
]

df = df.dropna(subset=required_cols).copy()

# 2026-04-26까지 반영
df = df[df["mkTm"].dt.date <= pd.to_datetime("2026-04-26").date()].copy()

df = df.sort_values(
    ["mkTm", "busRouteId", "vehId1", "staOrd"]
).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n[저장 완료]")
print(OUTPUT_PATH)
print("rows:", len(df))
print("routes:", df["busRouteId"].nunique())
print("stops:", df["stId"].nunique())
print("date range:", df["mkTm"].min(), "~", df["mkTm"].max())
print("\n[날짜별 row 수]")
print(df["mkTm"].dt.date.value_counts().sort_index())