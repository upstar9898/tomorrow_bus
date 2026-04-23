import os
import glob
from datetime import datetime

import pandas as pd


def main():
    # 현재 파일 위치: 프로젝트루트/collection/파일명.py
    current_file = os.path.abspath(__file__)
    collection_dir = os.path.dirname(current_file)
    base_dir = os.path.dirname(collection_dir)   # 프로젝트 루트
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    print("현재 파일 위치:", current_file)
    print("collection 폴더:", collection_dir)
    print("프로젝트 루트:", base_dir)
    print("data 폴더:", data_dir)

    pattern = os.path.join(data_dir, "bus_station_with_stn_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError("bus_station_with_stn_*.csv 파일이 없습니다.")

    # 가장 최신 파일 선택
    input_path = max(files, key=os.path.getmtime)
    today_str = datetime.now().strftime("%y%m%d")
    output_path = os.path.join(data_dir, f"bus_station_for_admin_{today_str}.csv")

    print("입력 파일:", input_path)
    print("출력 파일:", output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

    # CSV 읽기
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    required_cols = [
        "stationId",
        "stn",
        "stNm",
        "위도",
        "경도",
        "address",
        "is_virtual",
        "regId",
        "arsId",
        "distance_km",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    # 필요한 컬럼만 선택
    result_df = df[
        [
            "stationId",
            "stn",
            "stNm",
            "위도",
            "경도",
            "address",
            "is_virtual",
            "arsId",
        ]
    ].copy()

    # 문자열 처리
    for col in ["stationId", "stn", "stNm", "address", "is_virtual", "arsId"]:
        result_df[col] = (
            result_df[col]
            .fillna("")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

    # stationId 없는 행 제거
    result_df = result_df[result_df["stationId"] != ""].copy()

    # 위도/경도 숫자 변환
    result_df["위도"] = pd.to_numeric(result_df["위도"], errors="coerce")
    result_df["경도"] = pd.to_numeric(result_df["경도"], errors="coerce")

    # stationId 중복 제거
    result_df = result_df.drop_duplicates(subset=["stationId"])

    # 저장
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n[완료]")
    print("정류소 CSV 변환 완료")
    print(f"입력 파일: {input_path}")
    print(f"출력 파일: {output_path}")
    print(f"저장 건수: {len(result_df)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]")
        print(type(e).__name__, ":", e)
        raise