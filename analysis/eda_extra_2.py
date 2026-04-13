import pandas as pd
from pathlib import Path

# -------------------------------
# 1. 경로
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

BUS_DATA_PATH = BASE_DIR / "data" / "combined" / "bus_data_v2.csv"
STATION_INFO_PATH = BASE_DIR / "data"/ "bus_station_coordinate_final.csv"

OUTPUT_PATH = BASE_DIR / "data" / "combined" / "bus_data_with_station.csv"

# -------------------------------
# 2. 데이터 로드
# -------------------------------
df = pd.read_csv(BUS_DATA_PATH)
station_df = pd.read_csv(STATION_INFO_PATH)

# -------------------------------
# 3. 컬럼 설정
# -------------------------------
bus_station_col = "stId"  # 너 데이터 기준
diff_col = "diff"

# -------------------------------
# 4. station 정보 정리
# -------------------------------
station_df = station_df[["stationId", "arsId", "stNm", "위도", "경도"]].copy()

station_df = station_df.rename(columns={
    "stationId": "stationId_station",
    "arsId": "arsId_station",
    "stNm": "station_name",
    "위도": "lat",
    "경도": "lon",
})

# -------------------------------
# 5. 타입 맞추기
# -------------------------------
df[bus_station_col] = df[bus_station_col].astype(str).str.strip()
df["arsId"] = df["arsId"].astype(str).str.strip()

station_df["stationId_station"] = station_df["stationId_station"].astype(str).str.strip()
station_df["arsId_station"] = station_df["arsId_station"].astype(str).str.strip()

# -------------------------------
# 6. 1차 merge (stationId 기준)
# -------------------------------
merged_df = df.merge(
    station_df,
    left_on=bus_station_col,
    right_on="stationId_station",
    how="left"
)

print("1차 매핑 실패:", merged_df["station_name"].isna().sum())

# -------------------------------
# 7. 2차 merge (arsId 보정)
# -------------------------------
missing_mask = merged_df["station_name"].isna()

ars_lookup = station_df[["arsId_station", "station_name", "lat", "lon"]].drop_duplicates()

fill_df = merged_df.loc[missing_mask, ["arsId"]].merge(
    ars_lookup,
    left_on="arsId",
    right_on="arsId_station",
    how="left"
)

merged_df.loc[missing_mask, "station_name"] = fill_df["station_name"].values
merged_df.loc[missing_mask, "lat"] = fill_df["lat"].values
merged_df.loc[missing_mask, "lon"] = fill_df["lon"].values

print("최종 매핑 실패:", merged_df["station_name"].isna().sum())

# -------------------------------
# 8. fallback (그래도 없으면 ID)
# -------------------------------
merged_df["station_name"] = merged_df["station_name"].fillna(
    "UNKNOWN_" + merged_df[bus_station_col].astype(str)
)

# -------------------------------
# 9. 저장
# -------------------------------
merged_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n완료")
print("저장 위치:", OUTPUT_PATH)