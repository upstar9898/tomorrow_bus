import os
import glob
from datetime import datetime

import pandas as pd


def get_latest_file(data_dir, prefix):
    pattern = os.path.join(data_dir, f"{prefix}_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(f"{pattern} 파일이 없습니다.")

    latest_file = max(files, key=os.path.getmtime)
    print(f"[선택된 파일] {prefix}: {latest_file}")
    return latest_file


def extract_fcst_name(city_value):
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


def main():
    # 현재 파일: 프로젝트루트/collection/match_station_regid.py
    current_file = os.path.abspath(__file__)
    collection_dir = os.path.dirname(current_file)
    base_dir = os.path.dirname(collection_dir)   # 프로젝트 루트
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    print("현재 파일 위치:", current_file)
    print("collection 폴더:", collection_dir)
    print("프로젝트 루트:", base_dir)
    print("data 폴더:", data_dir)

    # ---------------------------
    # 입력 파일
    # ---------------------------
    station_path = os.path.join(data_dir, "bus_station_with_city.csv")
    zone_path = os.path.join(data_dir, "fcst_zone_regid_regname.csv")
    station_id_path = get_latest_file(data_dir, "bus_station_coordinate_stationId")

    # 입력 파일 존재 확인
    for path in [station_path, zone_path, station_id_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    # 날짜
    today_str = datetime.now().strftime("%y%m%d")

    # ---------------------------
    # 출력 파일
    # ---------------------------
    output_path = os.path.join(data_dir, f"bus_station_with_regid_{today_str}.csv")
    unmatched_path = os.path.join(data_dir, f"bus_station_unmatched_{today_str}.csv")

    print("결과 파일 경로:", output_path)
    print("미매칭 파일 경로:", unmatched_path)

    # ---------------------------
    # 데이터 로드
    # ---------------------------
    station_df = pd.read_csv(station_path, encoding="utf-8-sig")
    zone_df = pd.read_csv(zone_path, encoding="utf-8-sig")
    station_id_df = pd.read_csv(station_id_path, encoding="utf-8-sig")

    print("\n[로드 완료]")
    print("station_df:", station_df.shape)
    print("zone_df:", zone_df.shape)
    print("station_id_df:", station_id_df.shape)

    # ---------------------------
    # 문자열 / 좌표 정리
    # ---------------------------
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

    # 가상 / 미정차 표시
    station_df["is_virtual"] = (
        station_df["stNm"]
        .astype(str)
        .str.contains("가상|미정차", case=False, na=False)
        .astype(int)
    )

    # ---------------------------
    # regId 매핑
    # ---------------------------
    reg_map = dict(zip(zone_df["regName"], zone_df["regId"]))

    station_df["fcst_name"] = station_df["city"].apply(extract_fcst_name)
    station_df["regId"] = station_df["fcst_name"].map(reg_map)

    # ---------------------------
    # stationId 매칭
    # ---------------------------
    station_df = station_df.merge(
        station_id_df,
        how="left",
        on=["stNm", "lat_key", "lon_key"]
    )

    # arsId 정리
    if "arsId_x" in station_df.columns and "arsId_y" in station_df.columns:
        station_df["arsId"] = station_df["arsId_y"].combine_first(station_df["arsId_x"])
        station_df = station_df.drop(columns=["arsId_x", "arsId_y"])
    elif "arsId_y" in station_df.columns:
        station_df = station_df.rename(columns={"arsId_y": "arsId"})
    elif "arsId_x" in station_df.columns:
        station_df = station_df.rename(columns={"arsId_x": "arsId"})

    station_df["arsId"] = pd.to_numeric(station_df["arsId"], errors="coerce").astype("Int64")

    # ---------------------------
    # 컬럼 정리
    # ---------------------------
    drop_cols = ["fcst_name", "city", "lat_key", "lon_key"]
    station_df = station_df.drop(columns=[c for c in drop_cols if c in station_df.columns])

    if "stationId" in station_df.columns:
        cols = ["stationId"] + [c for c in station_df.columns if c != "stationId"]
        station_df = station_df[cols]

    # ---------------------------
    # 저장
    # ---------------------------
    station_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    unmatched_df = station_df[
        station_df["regId"].isna() | station_df["stationId"].isna()
    ].copy()
    unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    # ---------------------------
    # 결과 로그
    # ---------------------------
    total_count = len(station_df)
    unmatched_count = len(unmatched_df)
    matched_count = total_count - unmatched_count

    print("\n[완료]")
    print(f"전체 행 수: {total_count}")
    print(f"매칭 완료: {matched_count}")
    print(f"미매칭 수: {unmatched_count}")
    print(f"결과 파일 저장: {output_path}")
    print(f"미매칭 파일 저장: {unmatched_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]")
        print(type(e).__name__, ":", e)
        raise