import os
import time
import pandas as pd
import requests
import glob
import re
from dotenv import load_dotenv

def main():

    load_dotenv()

    # 현재 파일 위치: 프로젝트루트/collection/파일명.py
    current_file = os.path.abspath(__file__)
    collection_dir = os.path.dirname(current_file)
    base_dir = os.path.dirname(collection_dir)   # 프로젝트 루트
    data_dir = os.path.join(base_dir, "data")

    os.makedirs(data_dir, exist_ok=True)

    pattern = os.path.join(data_dir, "bus_station_coordinate*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError("bus_station_coordinate 파일이 없습니다.")

    def extract_date(file_path):
        filename = os.path.basename(file_path)

        # 숫자 6자리 (YYMMDD) 찾기
        match = re.search(r"(\d{6})", filename)
        if match:
            return match.group(1)
        return "000000"  # 날짜 없는 파일은 가장 오래된 걸로 처리

    # 날짜 기준 최신 파일 선택
    input_path = max(files, key=extract_date)
    file_date = extract_date(input_path)

    output_path = os.path.join(
        data_dir,
        f"bus_station_with_city_{file_date}.csv"
    )

    print("현재 파일 위치:", current_file)
    print("프로젝트 루트:", base_dir)
    print("data 폴더:", data_dir)
    print("입력 파일:", input_path)
    print("출력 파일:", output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_path}")

    # API 키
    kakao_api_key = os.environ.get("KAKAO_ADMIN_KEY")
    if not kakao_api_key:
        raise EnvironmentError("환경변수 KAKAO_ADMIN_KEY가 설정되지 않았습니다.")

    headers = {"Authorization": f"KakaoAK {kakao_api_key}"}

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    if "위도" not in df.columns or "경도" not in df.columns:
        raise ValueError("입력 CSV에 '위도', '경도' 컬럼이 필요합니다.")

    city_list = []
    address_list = []

    url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"

    for idx, row in df.iterrows():
        lat = row["위도"]
        lon = row["경도"]

        try:
            params = {"x": lon, "y": lat}
            res = requests.get(url, headers=headers, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()

            if data.get("documents"):
                addr = data["documents"][0]["address"]

                city = addr.get("region_1depth_name", "")
                district = addr.get("region_2depth_name", "")
                city_name = f"{city} {district}".strip()
                address_name = addr.get("address_name", "UNKNOWN")
            else:
                city_name = "UNKNOWN"
                address_name = "UNKNOWN"

        except Exception as e:
            city_name = "ERROR"
            address_name = "ERROR"
            print(f"[에러] idx={idx}, lat={lat}, lon={lon}, error={e}")

        city_list.append(city_name)
        address_list.append(address_name)

        time.sleep(0.1)

        if idx % 50 == 0:
            print(f"{idx} 처리중...")

    df["city"] = city_list
    df["address"] = address_list

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n[완료]")
    print(f"저장 위치: {output_path}")
    print(f"총 처리 건수: {len(df)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]")
        print(type(e).__name__, ":", e)
        raise