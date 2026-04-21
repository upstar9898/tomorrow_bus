# -*- coding: utf-8 -*-
"""
내일의버스 - 스냅샷 기반 전처리
- 입력: CSV 1개
- 출력: CSV 1개
- 예측 시점에 알 수 없는 정보는 생성하지 않음
"""

import numpy as np
import pandas as pd
import holidays
import os

from utils import (
    to_int,
    to_float,
    cyclical_encode,
    normalize_arrmsg,
    load_csv,
)  # 유틸 함수
from weather_features import add_weather_features
from config.constants import (
    # 총 좌석수 clip, eta 제한
    TOTAL_SEATS,
    MAX_VALID_ETA_SEC,
    # trip의 정의에 관한 상수
    TIME_GAP_MINUTES,
    STAORD_BACKWARD_THRESHOLD,
)

# 필요한 컬럼만 사용
from config.columns import USE_COLS, FINAL_COLS, NUMERIC_COLS

# =========================================================
# 1. 설정
# =========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
BUS_API_DATA_DIR = os.path.join(DATA_DIR, "bus_api_data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed_withweather")
OUTPUT_SUFFIX = "_preprocessed_withweather.csv"

os.makedirs(PREPROCESSED_DIR, exist_ok=True)

MODE = "training"

SKIP_IF_EXIST = False  # 이미 output 파일이 존재할 경우 skip할지 여부


# =========================================================
# 2. 전처리
# =========================================================
def preprocess(df):
    # -----------------------------
    # 타입 정리
    # -----------------------------
    int_cols = ["stId", "arsId", "busRouteId", "staOrd", "vehId1", "reride_Num1"]
    for c in int_cols:
        df[c] = to_int(df[c], fill_value=0)

    df["exps1"] = to_float(df["exps1"])
    df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
    df["arrmsg1"] = df["arrmsg1"].apply(normalize_arrmsg)
    mask1 = df["arrmsg1"].str.contains("곧", na=False)
    mask2 = df["arrmsg1"].str.contains(r"\[0번째 전\]", na=False)

    df = df[mask1 | mask2].copy()

    # -----------------------------
    # 기본 클리닝
    # -----------------------------
    df = df[df["mkTm"].notna()].copy()
    df = df[df["stId"] > 0].copy()
    df = df[df["busRouteId"] > 0].copy()
    df = df[df["staOrd"] > 0].copy()
    df = df[df["vehId1"] != 0].copy()
    df = df[df["rerdie_Div1"] != 0].copy()

    # ETA 범위 제한
    df["exps1"] = df["exps1"].where(
        (df["exps1"] >= 0) & (df["exps1"] <= MAX_VALID_ETA_SEC), np.nan
    )

    # -----------------------------
    # 타깃/입력에 필요한 좌석값 정리
    # -----------------------------
    # reride_Num1을 잔여좌석수로 해석
    df["remaining_seat"] = to_int(df["reride_Num1"], fill_value=np.nan)
    df["remaining_seat"] = df["remaining_seat"].clip(lower=0, upper=TOTAL_SEATS)

    # remaining_seat가 0이면 만차처리
    df["full_flag"] = (df["remaining_seat"] == 0).astype(int)

    # 좌석값 없는 행 제거
    df = df[df["remaining_seat"].notna()].copy()

    # -----------------------------
    # 현재 시점에 알 수 있는 파생 변수만 생성
    # -----------------------------
    df["year"] = df["mkTm"].dt.year
    df["month"] = df["mkTm"].dt.month
    df["day"] = df["mkTm"].dt.day
    df["hour"] = df["mkTm"].dt.hour
    df["minute"] = df["mkTm"].dt.minute
    df["dayofweek"] = df["mkTm"].dt.dayofweek
    df["date"] = df["mkTm"].dt.date

    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    years = sorted(df["year"].dropna().unique().tolist())
    kr_holidays = holidays.KR(years=years)
    df["is_holiday"] = df["date"].apply(lambda d: 1 if d in kr_holidays else 0)

    df["is_peak"] = (
        ((df["hour"] >= 7) & (df["hour"] <= 9))
        | ((df["hour"] >= 17) & (df["hour"] <= 20))
    ).astype(int)

    df["month_sin"], df["month_cos"] = cyclical_encode(df["month"], 12)
    df["day_sin"], df["day_cos"] = cyclical_encode(df["day"], 31)
    df["hour_sin"], df["hour_cos"] = cyclical_encode(df["hour"], 24)
    df["minute_sin"], df["minute_cos"] = cyclical_encode(df["minute"], 60)
    df["dow_sin"], df["dow_cos"] = cyclical_encode(df["dayofweek"], 7)

    # -----------------------------
    # 중복 제거
    # -----------------------------
    # 같은 노선-정류소-시각에 중복이 있으면
    # ETA가 더 작고 좌석정보가 있는 쪽을 우선
    # 그 후 같은 운행 안에서 우선순위를 정해서 하나의 행만 남김

    df = df.sort_values(["busRouteId", "vehId1", "mkTm"]).copy()

    df["time_diff"] = (
        df.groupby(["busRouteId", "vehId1"])["mkTm"].diff().dt.total_seconds().div(60)
    )

    # 같은 노선-차량 내 정류장 순번 차이
    df["staOrd_diff"] = df.groupby(["busRouteId", "vehId1"])["staOrd"].diff()

    df["new_trip_flag"] = (
        df["time_diff"].isna()
        | (df["time_diff"] > TIME_GAP_MINUTES)
        | (df["staOrd_diff"] <= -STAORD_BACKWARD_THRESHOLD)
    ).astype(int)

    df["trip_group"] = df.groupby(["busRouteId", "vehId1"])["new_trip_flag"].cumsum()
    df["arr_priority"] = 99

    df.loc[df["arrmsg1"].str.contains(r"\[0번째 전\]", na=False), "arr_priority"] = 0
    df.loc[df["arrmsg1"].str.contains(r"곧", na=False), "arr_priority"] = 1

    df = df.sort_values(
        ["busRouteId", "stId", "vehId1", "trip_group", "arr_priority", "mkTm"],
        ascending=[True, True, True, True, True, True],
    ).copy()

    df = df.drop_duplicates(
        subset=["busRouteId", "stId", "vehId1", "trip_group"], keep="first"
    ).copy()

    df = df.sort_values(
        ["busRouteId", "vehId1", "trip_group", "staOrd", "mkTm"],
        ascending=[True, True, True, True, True],
    ).copy()

    # 날씨 데이터 추가
    df = add_weather_features(df)

    # -----------------------------
    # 최종 정리
    # -----------------------------
    final_cols = FINAL_COLS[MODE]

    df = df[final_cols].copy()

    numeric_cols = NUMERIC_COLS[MODE]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =========================================================
# 3. 실행
# =========================================================
if __name__ == "__main__":
    # data 폴더 안의 모든 csv 파일 가져오기
    # file_list = [f for f in os.listdir(BUS_API_DATA_DIR) if f.endswith(".csv")]
    file_list = ["bus_data_2026_03_12.csv"]

    use_cols = USE_COLS[MODE]

    for filename in file_list:
        input_path = os.path.join(BUS_API_DATA_DIR, filename)
        output_filename = filename.replace(".csv", OUTPUT_SUFFIX)
        output_path = os.path.join(PREPROCESSED_DIR, output_filename)
        # 이미 존재하면 skip, SKIP_IF_EXIST가 True일 때만
        if SKIP_IF_EXIST:
            if os.path.exists(output_path):
                print(f"[SKIP] 해당 파일이 이미 존재합니다.: {output_filename}")
                continue
        print(f"\n===== 처리 중: {filename} =====")

        raw_df = load_csv(input_path, use_cols)
        processed_df = preprocess(raw_df)

        processed_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"완료 → {output_path}")
