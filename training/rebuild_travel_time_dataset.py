from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\Users\Administrator\Desktop\langchain_semiproject\tomorrow_bus\data\traveltime_raw")
OUTPUT_PATH = DATA_DIR / "bus_traveltime_260428_merged.csv"

file_paths = sorted(DATA_DIR.glob("*preprocessed*.csv"))

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

# 타입만 정리
df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
df["travel_time"] = pd.to_numeric(df["travel_time"], errors="coerce")

# travel_time은 절대 dropna 하지 않음
df = df.dropna(
    subset=["mkTm", "arrival_time", "busRouteId", "stId", "arsId", "staOrd", "vehId1"]
).copy()

df = df.sort_values(
    ["mkTm", "busRouteId", "vehId1", "staOrd"]
).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("[저장 완료]")
print(OUTPUT_PATH)
print("rows:", len(df))
print("travel_time 결측:", df["travel_time"].isna().sum())