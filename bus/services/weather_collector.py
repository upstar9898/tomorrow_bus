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
        print("다운로드 실패:", response.status_code)
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
    tm2 = datetime.now().strftime("%Y%m%d") + "0000"

    # 오늘 날짜 YYMMDD
    today_str = datetime.now().strftime("%y%m%d")

    api_key = os.environ.get("WEATHER_API_KEY")
    if not api_key:
        raise ValueError("환경변수 WEATHER_API_KEY가 설정되지 않았습니다.")

    # 최신 stn 파일 찾기
    station_stn_path = get_latest_station_stn_file(base_dir)
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



def get_latest_station_file(base_dir):
    data_dir = os.path.join(base_dir, "data")
    pattern = os.path.join(data_dir, "bus_station_with_stn_*.csv")

    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("bus_station_with_stn_*.csv 파일이 없습니다.")

    return max(files, key=os.path.getmtime)


def get_station_info_by_id(station_id, csv_path):
    """
    stationId 기준으로 위도/경도 및 정보 조회
    """
    df = pd.read_csv(csv_path)

    # 타입 맞추기 (중요)
    df["stationId"] = df["stationId"].astype(str).str.replace(".0", "", regex=False)
    station_id = str(station_id)


    match = df[df["stationId"] == station_id]

    if match.empty:
        raise ValueError(f"stationId '{station_id}'를 찾을 수 없습니다.")

    row = match.iloc[0]

    return {
        "stationId": row["stationId"],
        "stationName": row["stNm"],  # 컬럼명 맞춰줘
        "lat": float(row["위도"]),
        "lon": float(row["경도"]),
        "regId": row.get("regId"),
        "arsId": row.get("arsId"),
    }


def collect_forecast(station_id="", target_dt=None):
    """
    stationId로 위경도 찾고 OpenWeather 호출 후 결과 반환
    """

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    csv_path = get_latest_station_file(base_dir)

    station_info = get_station_info_by_id(station_id, csv_path)
    if not station_info:
        raise ValueError("해당 station_id의 정류소 정보를 찾을 수 없습니다.")

    lat = station_info["lat"]
    lon = station_info["lon"]

    api_key = os.environ.get("OPEN_WEATHER_API_KEY")
    if not api_key:
        raise ValueError("환경변수 OPEN_WEATHER_API_KEY가 설정되지 않았습니다.")

    url = "https://api.openweathermap.org/data/2.5/forecast"

    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "lang": "kr",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if target_dt:
        if isinstance(target_dt, str):
            try:
                target_dt = datetime.strptime(target_dt, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    target_dt = datetime.strptime(target_dt, "%Y-%m-%d %H:%M")
                except ValueError:
                    target_dt = datetime.strptime(target_dt, "%Y-%m-%dT%H:%M")

        closest_item = min(
            data["list"],
            key=lambda item: abs(
                datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S") - target_dt
            ),
        )

        return {
            "stationId": station_info["stationId"],
            "stationName": station_info["stationName"],
            "lat": lat,
            "lon": lon,
            "regId": station_info["regId"],
            "arsId": station_info["arsId"],
            "target_dt": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "forecast": closest_item,
        }

    return {
        "stationId": station_info["stationId"],
        "stationName": station_info["stationName"],
        "lat": lat,
        "lon": lon,
        "regId": station_info["regId"],
        "arsId": station_info["arsId"],
        "forecast_list": data["list"],
    }