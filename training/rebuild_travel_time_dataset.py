from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\Users\Administrator\Desktop\langchain_semiproject\tomorrow_bus\data\traveltime_raw")
OUTPUT_PATH = DATA_DIR / "bus_all_raw_weather_260424_rebuilt_v2.csv"

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

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
df["full_flag"] = pd.to_numeric(df["full_flag"], errors="coerce")

df = df.dropna(subset=[
    "mkTm", "busRouteId", "stId", "arsId", "staOrd",
    "vehId1", "remaining_seat", "full_flag"
]).copy()

df = df.sort_values(["mkTm", "busRouteId", "staOrd"]).reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("[저장 완료]")
print(OUTPUT_PATH)
print("rows:", len(df))
print("routes:", df["busRouteId"].nunique())
print("stops:", df["stId"].nunique())