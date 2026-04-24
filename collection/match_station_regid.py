import os
import glob
import re
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv


def get_latest_file(data_dir, prefix):
    pattern = os.path.join(data_dir, f"{prefix}_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"{pattern} 파일이 없습니다.")

    def extract_date(file_path):
        filename = os.path.basename(file_path)
        match = re.search(r"_(\d{6})\.csv$", filename)
        return match.group(1) if match else "000000"

    latest_file = max(files, key=extract_date)
    print(f"[선택된 파일] {prefix}: {latest_file}")
    return latest_file


def get_file_date(file_path):
    filename = os.path.basename(file_path)
    match = re.search(r"_(\d{6})\.csv$", filename)
    if match:
        return match.group(1)
    return datetime.now().strftime("%y%m%d")


def main():
    load_dotenv()

    current_file = os.path.abspath(__file__)
    collection_dir = os.path.dirname(current_file)
    base_dir = os.path.dirname(collection_dir)
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    station_path = get_latest_file(data_dir, "bus_station_with_city")
    zone_path = os.path.join(data_dir, "fcst_zone_regid_regname.csv")
    station_id_path = get_latest_file(data_dir, "bus_station_coordinate_stationId")

    file_date = get_file_date(station_path)

    output_path = os.path.join(data_dir, f"bus_station_with_regid_{file_date}.csv")
    unmatched_path = os.path.join(data_dir, f"bus_station_unmatched_{file_date}.csv")

    station_df = pd.read_csv(station_path, encoding="utf-8-sig")
    zone_df = pd.read_csv(zone_path, encoding="utf-8-sig")
    station_id_df = pd.read_csv(station_id_path, encoding="utf-8-sig")

    station_df["city"] = station_df["city"].astype(str).str.strip()
    zone_df["regName"] = zone_df["regName"].astype(str).str.strip()

    station_df["stNm"] = station_df["stNm"].astype(str).str.strip()
    station_id_df["stNm"] = station_id_df["stNm"].astype(str).str.strip()

    station_df["위도"] = pd.to_numeric(station_df["위도"], errors="coerce")
    station_df["경도"] = pd.to_numeric(station_df["경도"], errors="coerce")
    station_id_df["위도"] = pd.to_numeric(station_id_df["위도"], errors="coerce")
    station_id_df["경도"] = pd.to_numeric(station_id_df["경도"], errors="coerce")

    station_df["lat_key"] = station_df["위도"].round(6)
    station_df["lon_key"] = station_df["경도"].round(6)
    station_id_df["lat_key"] = station_id_df["위도"].round(6)
    station_id_df["lon_key"] = station_id_df["경도"].round(6)

    station_id_df = station_id_df[
        ["stationId", "arsId", "stNm", "lat_key", "lon_key"]
    ].drop_duplicates()

    station_df["is_virtual"] = (
        station_df["stNm"]
        .astype(str)
        .str.contains("가상|미정차", na=False)
        .astype(int)
    )

    reg_map = dict(zip(zone_df["regName"], zone_df["regId"]))

    metro_map = {
        "서울": "서울",
        "부산": "부산",
        "대구": "대구",
        "인천": "인천",
        "광주": "광주",
        "대전": "대전",
        "울산": "울산",
        "세종특별자치시": "세종",
        "제주특별자치도": "제주",
    }

    province_set = {"경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남"}

    def extract_fcst_name(city_value):
        if pd.isna(city_value):
            return None

        text = str(city_value).strip()

        if not text or text.upper() == "UNKNOWN":
            return None

        parts = text.split()
        if not parts:
            return None

        first = parts[0]

        if first in metro_map:
            return metro_map[first]

        if first in province_set:
            if len(parts) >= 2:
                second = parts[1]
                second = second.replace("특별자치시", "")
                second = second.replace("특별자치도", "")
                second = second.rstrip("시군")
                return second
            return None

        cleaned = text.replace("특별자치시", "").replace("특별자치도", "")
        return cleaned

    station_df["fcst_name"] = station_df["city"].apply(extract_fcst_name)
    station_df["regId"] = station_df["fcst_name"].map(reg_map)

    station_df = station_df.merge(
        station_id_df,
        how="left",
        on=["stNm", "lat_key", "lon_key"],
    )

    if "stationId_x" in station_df.columns and "stationId_y" in station_df.columns:
        station_df["stationId"] = station_df["stationId_y"].combine_first(
            station_df["stationId_x"]
        )
    elif "stationId_y" in station_df.columns:
        station_df["stationId"] = station_df["stationId_y"]
    elif "stationId_x" in station_df.columns:
        station_df["stationId"] = station_df["stationId_x"]

    if "arsId_x" in station_df.columns and "arsId_y" in station_df.columns:
        station_df["arsId"] = station_df["arsId_y"].combine_first(
            station_df["arsId_x"]
        )
    elif "arsId_y" in station_df.columns:
        station_df["arsId"] = station_df["arsId_y"]
    elif "arsId_x" in station_df.columns:
        station_df["arsId"] = station_df["arsId_x"]

    result_cols = [
        "stationId",
        "stNm",
        "위도",
        "경도",
        "address",
        "is_virtual",
        "regId",
        "arsId",
    ]

    result_df = station_df[result_cols].copy()

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    unmatched_df = result_df[
        result_df["regId"].isna() | result_df["stationId"].isna()
    ].copy()

    unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    print("\n[완료]")
    print(f"전체 행 수: {len(result_df)}")
    print(f"결과 파일: {output_path}")
    print(f"미매칭 파일: {unmatched_path}")


if __name__ == "__main__":
    main()