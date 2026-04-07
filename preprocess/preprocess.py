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


# =========================================================
# 1. 설정
# =========================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

TOTAL_SEATS = 45
MAX_VALID_ETA_SEC = 3600

# 필요한 컬럼만 사용
USE_COLS = [
    "stId",
    "arsId",
    "busRouteId",
    "mkTm",
    "staOrd",
    "vehId1",
    "exps1",
    "arrmsg1",
    "reride_Num1",
    "full1",
]

# 최종 학습용 feature 컬럼
FEATURE_COLS = [
    "staOrd",
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
    "is_holiday",
    "is_peak",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
    "exps1",
    "remaining_seat",
]


# =========================================================
# 2. 유틸
# =========================================================
def to_int(series, fill_value=0):
    return pd.to_numeric(series, errors="coerce").fillna(fill_value).astype(int)


def to_float(series, fill_value=np.nan):
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def cyclical_encode(series, max_value):
    angle = 2 * np.pi * series / max_value
    return np.sin(angle), np.cos(angle)


def normalize_arrmsg(text):
    if pd.isna(text):
        return ""
    return str(text).strip()


# =========================================================
# 3. 로드
# =========================================================
def load_csv(path, use_cols):
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)
    print(f"로드 완료: {path}, shape={df.shape}")
    return df


# =========================================================
# 4. 전처리
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

    df["full1"] = df["full1"].astype(str).str.strip()
    df["full_flag"] = df["full1"].isin(["1", "Y", "y", "True", "true"]).astype(int)

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
    df["exps1_for_sort"] = df["exps1"].fillna(999999)
    df = df.sort_values(
        ["busRouteId", "stId", "mkTm", "exps1_for_sort", "vehId1"],
        ascending=[True, True, True, True, True],
    ).copy()

    df = df.drop_duplicates(subset=["busRouteId", "stId", "mkTm"], keep="first").copy()

    # -----------------------------
    # 최종 정리
    # -----------------------------
    final_cols = [
        # 원본/식별
        "mkTm",
        "busRouteId",
        "stId",
        "arsId",
        "staOrd",
        "vehId1",
        # 원본 기반 현재 정보
        "exps1",
        "arrmsg1",
        "remaining_seat",
        "full_flag",
        # 시간 파생
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "dayofweek",
        "is_weekend",
        "is_holiday",
        "is_peak",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "dow_sin",
        "dow_cos",
    ]

    df = df[final_cols].copy()

    # feature 컬럼 숫자형 보정
    numeric_cols = [
        "staOrd",
        "exps1",
        "remaining_seat",
        "full_flag",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "dayofweek",
        "is_weekend",
        "is_holiday",
        "is_peak",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "dow_sin",
        "dow_cos",
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


# =========================================================
# 5. 실행
# =========================================================
if __name__ == "__main__":
    # data 폴더 안의 모든 csv 파일 가져오기
    file_list = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    for filename in file_list:
        input_path = os.path.join(DATA_DIR, filename)
        output_filename = filename.replace(".csv", "_preprocessed.csv")
        output_path = os.path.join(PREPROCESSED_DIR, output_filename)
        # 이미 존재하면 skip
        if os.path.exists(output_path):
            print(f"[SKIP] 해당 파일이 이미 존재합니다.: {output_filename}")
            continue
        print(f"\n===== 처리 중: {filename} =====")

        raw_df = load_csv(input_path, USE_COLS)
        processed_df = preprocess(raw_df)

        processed_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"완료 → {output_path}")