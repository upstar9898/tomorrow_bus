# -*- coding: utf-8 -*-
"""
내일의버스 - 스냅샷 기반 전처리
- 입력: CSV 1개
- 출력: CSV 1개
- 예측 시점에 알 수 없는 정보는 생성하지 않음
"""

import numpy as np
import pandas as pd

# import holidays
import os

from utils import to_int, to_float, normalize_arrmsg, load_csv  # 유틸 함수
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
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed_traveltime")
OUTPUT_SUFFIX = "_preprocessed_traveltime.csv"


os.makedirs(PREPROCESSED_DIR, exist_ok=True)

MODE = "traveltime"

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
    mask2 = df["arrmsg1"].str.contains(r"\[(?:0|1)번째 전\]", na=False)

    df = df[mask1 | mask2].copy()

    # -----------------------------
    # 기본 클리닝
    # -----------------------------
    df = df[df["mkTm"].notna()].copy()
    df = df[df["stId"] > 0].copy()
    df = df[df["busRouteId"] > 0].copy()
    df = df[df["staOrd"] > 0].copy()
    df = df[df["vehId1"] != 0].copy()

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

    # 만차 플래그가 있으면 0석 처리
    df.loc[df["full_flag"] == 1, "remaining_seat"] = 0

    # remaining_seat가 0이면 무조건 만차 처리
    df.loc[df["remaining_seat"] == 0, "full_flag"] = 1

    # 좌석값 없는 행 제거
    df = df[df["remaining_seat"].notna()].copy()

    # 도착 시간 열 추가
    df["arrival_time"] = df["mkTm"] + pd.to_timedelta(df["exps1"], unit="s")

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

    df.loc[df["arrmsg1"].str.contains(r"\[0번째 전\]", na=False), "arr_priority"] = 1
    df.loc[df["arrmsg1"].str.contains(r"곧", na=False), "arr_priority"] = 0
    df.loc[df["arrmsg1"].str.contains(r"\[1번째 전\]", na=False), "arr_priority"] = 2

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

    # -----------------------------
    # 이전 관측 정류장 기준 이동시간(travel_time) 추가
    # -----------------------------
    group_cols = ["busRouteId", "vehId1", "trip_group"]

    df["prev_staOrd"] = df.groupby(group_cols)["staOrd"].shift(1)
    df["prev_arrival_time"] = df.groupby(group_cols)["arrival_time"].shift(1)

    sta_gap = df["staOrd"] - df["prev_staOrd"]
    time_gap = (df["arrival_time"] - df["prev_arrival_time"]).dt.total_seconds()

    df["travel_time"] = time_gap / sta_gap
    df.loc[sta_gap <= 0, "travel_time"] = np.nan
    # travel_time이 음수인 경우 결측치 처리
    df.loc[df["travel_time"] < 0, "travel_time"] = np.nan

    # -----------------------------
    # 최종 정리
    # -----------------------------
    final_cols = FINAL_COLS[MODE]

    df = df[final_cols].copy()

    # feature 컬럼 숫자형 보정
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
    file_list = [f for f in os.listdir(BUS_API_DATA_DIR) if f.endswith(".csv")]
    # file_list = ["bus_data_2026_03_10.csv"]

    for filename in file_list:
        input_path = os.path.join(BUS_API_DATA_DIR, filename)
        output_filename = filename.replace(".csv", OUTPUT_SUFFIX)
        output_path = os.path.join(PREPROCESSED_DIR, output_filename)
        # 이미 존재하면 skip
        if os.path.exists(output_path):
            print(f"[SKIP] 해당 파일이 이미 존재합니다.: {output_filename}")
            continue
        print(f"\n===== 처리 중: {filename} =====")

        raw_df = load_csv(input_path, USE_COLS[MODE])
        processed_df = preprocess(raw_df)

        processed_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"완료 → {output_path}")
