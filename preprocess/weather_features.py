import pandas as pd
import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
WEATHER_DATA_DIR = os.path.join(DATA_DIR, "data_for_weather_process")
STATION_PATH = os.path.join(WEATHER_DATA_DIR, "bus_station_for_admin_260424.csv")
WEATHER_PATH = os.path.join(WEATHER_DATA_DIR, "weather_260428_first_processed.csv")


# 날씨 데이터 추가 함수
def add_weather_features(df):
    station_df = pd.read_csv(STATION_PATH, dtype={"stationId": str, "stn": str})

    weather_df = pd.read_csv(WEATHER_PATH, dtype={"STN": str})

    # -----------------------------
    # 1) 정류장별 기상관측소 번호(stn) 붙이기
    # -----------------------------
    df["stId"] = df["stId"].astype(str)

    station_map = (
        station_df[["stationId", "stn"]]
        .dropna(subset=["stationId", "stn"])
        .drop_duplicates(subset=["stationId"])
        .rename(columns={"stationId": "stId"})
    )

    df = df.merge(station_map, on="stId", how="left")

    # -----------------------------
    # 2) mkTm을 가장 가까운 정시로 반올림
    # -----------------------------
    df["weather_time"] = df["mkTm"].dt.round("h")

    df["weather_year"] = df["weather_time"].dt.year
    df["weather_month"] = df["weather_time"].dt.month
    df["weather_day"] = df["weather_time"].dt.day
    df["weather_hour"] = df["weather_time"].dt.hour

    # -----------------------------
    # 3) weather 파일과 merge
    # -----------------------------
    weather_df = weather_df.rename(columns={"STN": "stn"})
    weather_df["stn"] = weather_df["stn"].astype(str)

    weather_use = weather_df[
        [
            "year",
            "month",
            "day",
            "hour",
            "stn",
            "precipitation",
            "TA",
            "RN",
        ]
    ].copy()

    df["stn"] = df["stn"].astype(str)

    weather_use = weather_df.rename(
        columns={
            "year": "w_year",
            "month": "w_month",
            "day": "w_day",
            "hour": "w_hour",
            "STN": "stn",
            "TA": "temperature",
            "RN": "rainfall",
        }
    )

    df = df.merge(
        weather_use,
        left_on=["weather_year", "weather_month", "weather_day", "weather_hour", "stn"],
        right_on=["w_year", "w_month", "w_day", "w_hour", "stn"],
        how="left",
    )

    # merge 후 보조 컬럼 정리
    df = df.drop(
        columns=[
            "weather_time",
            "weather_year",
            "weather_month",
            "weather_day",
            "weather_hour",
        ],
        errors="ignore",
    )

    # 결측치가 있을 경우 그 시간대 평균/최빈값으로 처리

    # 강수 여부는 최빈값 처리
    df["precipitation"] = df["precipitation"].fillna(
        df.groupby(["year", "month", "day", "hour"])["precipitation"].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 0
        )
    )
    df["precipitation"] = df["precipitation"].fillna(0)

    # 안개 여부도 최빈값 처리
    df["fog"] = df["fog"].fillna(
        df.groupby(["year", "month", "day", "hour"])["precipitation"].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 0
        )
    )
    df["fog"] = df["fog"].fillna(0)

    # precipitation == 0 이면 rainfall은 무조건 0
    df.loc[df["precipitation"] == 0, "rainfall"] = 0

    # precipitation == 1 이고 rainfall이 결측인 경우만 처리
    mask = (df["precipitation"] == 1) & (df["rainfall"].isna())

    # 1차: 그 날짜의 그 시간 평균
    rain_date_hour_mean = (
        df.loc[df["precipitation"] == 1]
        .groupby(["year", "month", "day", "hour"])["rainfall"]
        .mean()
    )

    df.loc[mask, "rainfall"] = (
        df.loc[mask, ["year", "month", "day", "hour"]]
        .apply(tuple, axis=1)
        .map(rain_date_hour_mean)
    )

    # 2차: 그래도 못 채운 경우 같은 날짜 평균
    mask = (df["precipitation"] == 1) & (df["rainfall"].isna())

    rain_date_mean = (
        df.loc[df["precipitation"] == 1]
        .groupby(["year", "month", "day"])["rainfall"]
        .mean()
    )

    df.loc[mask, "rainfall"] = (
        df.loc[mask, ["year", "month", "day"]].apply(tuple, axis=1).map(rain_date_mean)
    )

    # 마지막 안전장치
    df["rainfall"] = df["rainfall"].fillna(0)

    # 기온 평균값 대체
    df["temperature"] = df["temperature"].fillna(
        df.groupby(["year", "month", "day", "hour"])["temperature"].transform("mean")
    )

    df["temperature"] = df["temperature"].fillna(
        df.groupby(["year", "month", "day"])["temperature"].transform("mean")
    )

    df["temperature"] = df["temperature"].fillna(df["temperature"].mean())

    return df
