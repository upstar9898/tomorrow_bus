import os
import glob
import requests
import pandas as pd
from datetime import datetime


def download_file(file_url, save_path):
    response = requests.get(file_url, timeout=60)

    if response.status_code == 200:
        # 🔥 핵심: 기상청은 EUC-KR
        text = response.content.decode("euc-kr", errors="replace")

        # UTF-8로 저장
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)

        print("다운로드 성공 (EUC-KR → UTF-8 변환 완료)")
    else:
        print("다운로드 실패:", response.status_code)S
        print(response.text)


def get_latest_station_stn_file(base_dir):
    data_dir = os.path.join(base_dir, "data")
    pattern = os.path.join(data_dir, "bus_station_with_stn_*.csv")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError("bus_station_with_stn_*.csv 파일이 없습니다.")

    return max(files, key=os.path.getmtime)


def get_unique_stn_param(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "stn" not in df.columns:
        raise ValueError("CSV에 stn 컬럼이 없습니다.")

    # 숫자로 변환 후 결측 제거
    stn_series = pd.to_numeric(df["stn"], errors="coerce").dropna()

    # 정수형으로 변환 후 중복 제거 + 정렬
    unique_stn = sorted(stn_series.astype(int).unique().tolist())

    if not unique_stn:
        raise ValueError("유효한 stn 값이 없습니다.")

    # API 형식: 108:119:239
    return ":".join(map(str, unique_stn))


def collect_weather():
    # 프로젝트 루트
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    data_dir = os.path.join(base_dir, "data")

    tm1 = "202603090000"
    tm2 = "202604060000"

    # 오늘 날짜 YYMMDD
    today_str = datetime.now().strftime("%y%m%d")

    api_key = os.environ.get("WEATHER_API_KEY")
    if not api_key:
        raise ValueError("환경변수 WEATHER_API_KEY가 설정되지 않았습니다.")

    # 최신 stn 파일 찾기
    station_stn_path = get_latest_station_stn_file(data_dir)
    print("사용할 STN 파일:", station_stn_path)

    # stn 중복 제거 후 : 형태로 만들기
    stn_param = get_unique_stn_param(station_stn_path)
    print("호출할 STN:", stn_param)

    url = (
        "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
        f"?tm1={tm1}&tm2={tm2}&stn={stn_param}&help=1&authKey={api_key}"
    )

    save_path = os.path.join(data_dir, f"weather_raw_{today_str}.txt")

    download_file(url, save_path)
