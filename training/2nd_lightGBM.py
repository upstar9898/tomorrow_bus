# =========================================================
# [완전 최종본]
# LightGBM 회귀 + 출퇴근 시간대 전용 4단계 혼잡도 분류기
# + 패턴 통계 저장 + 서비스 결과 저장 + 모델 저장
#
# ---------------------------------------------------------
# 변경사항 요약
# 1. 전체 데이터는 잔여좌석 회귀모델로 학습하도록 유지
# 2. 출퇴근 시간대(is_peak=1) 데이터만 별도로 추출해
#    4단계 혼잡도 분류모델 추가
# 3. 혼잡도 기준을 아래와 같이 4단계로 재정의
#    - 0: 매우 혼잡  (0~5석)
#    - 1: 혼잡      (6~15석)
#    - 2: 보통      (16~30석)
#    - 3: 여유      (31~45석)
# 4. 서비스 예측 시
#    - 비출퇴근: 회귀 기반 혼잡도 사용
#    - 출퇴근: 분류기 결과를 최종 혼잡도로 우선 사용
# 5. train 기준 패턴 통계(feature) 파일과 meta 정보 저장 기능 추가
# 6. classification report txt 저장 기능 추가
# 7. 테스트 서비스 결과 CSV, 노선별 정류소 순서 CSV,
#    모델 및 인코더 저장 기능 포함
#
# ---------------------------------------------------------
# 모델 구조
# 1) 전체 시간대 데이터로 잔여좌석 회귀모델 학습
# 2) 출퇴근 시간대(is_peak=1) 데이터만 따로 뽑아
#    4단계 혼잡도 분류모델 학습
# 3) 서비스 예측 시
#    - 비출퇴근: 회귀 예측값을 혼잡도로 변환
#    - 출퇴근: 분류기 결과를 최종 혼잡도로 사용
#
# ---------------------------------------------------------
# 왜 이렇게 설계했나?
# - 잔여좌석 수 자체는 전체 데이터로 회귀하는 것이 가장 안정적
# - 하지만 혼잡도 분류는 전체 시간대로 하면 "여유" 쪽으로 과하게 쏠릴 수 있음
# - 그래서 실제 혼잡도 판단이 중요한 출퇴근 시간대만 따로 분류
# - 4단계 기준도 단순 21~45 통합이 아니라
#   0~5 / 6~15 / 16~30 / 31~45 로 재설정해
#   중간 구간을 더 잘 구분하도록 설계
# =========================================================


# =========================================================
# 1. 라이브러리 import
# =========================================================

import os

# 표 형태 데이터 처리용
import pandas as pd

# 수치 계산, 결측값 처리, clip, sin/cos 계산 등에 사용
import numpy as np

# 패턴 통계 메타정보(JSON) 저장용
import json

# 모델 및 인코더 저장용
import joblib

# 문자열 ID를 숫자로 인코딩하기 위한 도구
from sklearn.preprocessing import LabelEncoder

# 회귀 / 분류 평가 지표
from sklearn.metrics import (
    mean_absolute_error,      # MAE
    mean_squared_error,       # MSE
    r2_score,                 # R^2
    accuracy_score,           # 분류 정확도
    f1_score,                 # F1 score
    classification_report,    # 분류 상세 리포트
    confusion_matrix          # 혼동행렬
)

# LightGBM 회귀 / 분류 모델
from lightgbm import LGBMRegressor, LGBMClassifier


# =========================================================
# 2. 설정값
# =========================================================

# 현재 실행 중인 파이썬 파일 위치를 기준으로 절대경로 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 학습 결과 산출물(csv, json, txt 등)을 저장할 폴더
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 학습에 사용할 통합 원본 CSV 파일 경로
file_path = os.path.join(BASE_DIR, "..", "data", "bus_all_raw3.csv")

# 광역버스 최대 좌석 수
# 회귀 예측 결과를 최종 서비스 시 0~45 범위로 제한할 때 사용
MAX_SEAT = 45

# "좌석 부족" 상태 판단 기준
# 패턴 통계에서 route_low_ratio 같은 feature를 만들 때 사용
LOW_SEAT_THRESHOLD = 10


# =========================================================
# 3. 혼잡도 기준 함수
# =========================================================
# 출퇴근 시간대 전용 분류기에서 사용할 혼잡도 4단계 기준
#
# 0: 매우 혼잡  (0~5석)
# 1: 혼잡      (6~15석)
# 2: 보통      (16~30석)
# 3: 여유      (31~45석)
#
# 이 함수는 두 곳에서 사용됨:
# 1) 분류 타깃(y_cls) 생성
# 2) 회귀 예측값을 서비스용 혼잡도로 변환할 때
# =========================================================
def seat_to_peak_congestion_4(seat):
    # 실수값일 수 있으므로 반올림 후, 0~45 범위 제한
    seat = int(np.clip(np.round(seat), 0, MAX_SEAT))

    if seat <= 5:
        return 0   # 매우 혼잡
    elif seat <= 15:
        return 1   # 혼잡
    elif seat <= 30:
        return 2   # 보통
    else:
        return 3   # 여유


# 숫자 class를 사람이 읽을 수 있는 한글 라벨로 바꾸는 함수
def congestion_label_text_4(cls):
    mapping = {
        0: "매우 혼잡",
        1: "혼잡",
        2: "보통",
        3: "여유"
    }
    return mapping.get(int(cls), "알수없음")


# =========================================================
# 4. 원본 파일 불러오기
# =========================================================
# 버스 노선ID, 정류장ID, 정류장 번호는 식별자이므로 문자열(str) 유지
# low_memory=False는 큰 CSV 읽을 때 dtype 추론 경고를 줄이는 용도
# =========================================================
df = pd.read_csv(
    file_path,
    dtype={
        "busRouteId": str,
        "stId": str,
        "arsId": str
    },
    low_memory=False
)

# 관측시각을 datetime으로 변환
# 잘못된 값은 NaT로 처리
df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")


# =========================================================
# 5. 기본 정리
# =========================================================
# 학습에 꼭 필요한 핵심 컬럼이 비어 있는 행 제거
# =========================================================
df = df.dropna(subset=["mkTm", "busRouteId", "stId", "arsId", "remaining_seat"]).copy()

# 문자열 컬럼 공백 제거
df["busRouteId"] = df["busRouteId"].astype(str).str.strip()
df["stId"] = df["stId"].astype(str).str.strip()
df["arsId"] = df["arsId"].astype(str).str.strip()

# 숫자형 컬럼 변환
df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")

# exps1이 있으면 숫자형으로 변환, 없으면 빈 컬럼 생성
if "exps1" in df.columns:
    df["exps1"] = pd.to_numeric(df["exps1"], errors="coerce")
else:
    df["exps1"] = np.nan

# full_flag가 있으면 숫자형으로 변환, 없으면 빈 컬럼 생성
if "full_flag" in df.columns:
    df["full_flag"] = pd.to_numeric(df["full_flag"], errors="coerce")
else:
    df["full_flag"] = np.nan

# 숫자 변환 후 NaN이 된 행 제거
df = df.dropna(subset=["remaining_seat", "staOrd"]).copy()

# 음수 좌석 수는 비정상값이므로 제거
df = df[df["remaining_seat"] >= 0].copy()

print("총 데이터 수:", len(df))

print("\n[remaining_seat 기초통계]")
print(df["remaining_seat"].describe())


# =========================================================
# 6. 시간 관련 feature 생성
# =========================================================
# mkTm에서 연/월/일/시/분/요일을 분리해 모델 입력으로 사용
# =========================================================
df["year"] = df["mkTm"].dt.year
df["month"] = df["mkTm"].dt.month
df["day"] = df["mkTm"].dt.day
df["hour"] = df["mkTm"].dt.hour
df["minute"] = df["mkTm"].dt.minute
df["dayofweek"] = df["mkTm"].dt.dayofweek

# 날짜만 따로 저장
# train / valid / test를 날짜 기준으로 분리할 때 사용
df["date"] = df["mkTm"].dt.date

# 주말 여부
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# 출퇴근 시간대 여부
# 오전 7~9시 또는 오후 17~19시
df["is_peak"] = (
    ((df["hour"] >= 7) & (df["hour"] <= 9)) |
    ((df["hour"] >= 17) & (df["hour"] <= 19))
).astype(int)

# 출퇴근 시간대를 조금 더 세분화한 변수
df["peak_level"] = 0
df.loc[(df["hour"] >= 7) & (df["hour"] <= 8), "peak_level"] = 1
df.loc[df["hour"] == 9, "peak_level"] = 2
df.loc[(df["hour"] >= 17) & (df["hour"] <= 18), "peak_level"] = 3
df.loc[df["hour"] == 19, "peak_level"] = 4


# =========================================================
# 7. 주기형(cyclical) feature 생성
# =========================================================
# 시간, 요일, 월 등은 순환 구조를 가지므로 sin/cos 변환 사용
# 예: 23시와 0시는 숫자로는 멀지만 실제로는 가까움
# =========================================================
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)


# =========================================================
# 8. 보조 시간 key 생성
# =========================================================
# 패턴 통계(feature engineering)를 만들 때 활용할 보조 key
# =========================================================

# 2시간 단위 그룹
df["hour_group"] = (df["hour"] // 2).astype(int)

# 10분 단위 그룹
df["minute_group"] = (df["minute"] // 10).astype(int)

# 시간 + 요일 조합 key
df["hour_weekday_key"] = df["hour"] * 10 + df["dayofweek"]


# =========================================================
# 9. 혼잡 비율 계산용 보조 라벨
# =========================================================
# LOW_SEAT_THRESHOLD 이하인 경우 1, 아니면 0
# route_low_ratio, route_stop_low_ratio 같은 feature 계산용
# =========================================================
df["is_low_seat"] = (df["remaining_seat"] <= LOW_SEAT_THRESHOLD).astype(int)


# =========================================================
# 10. 출퇴근 전용 분류 타깃 생성
# =========================================================
# remaining_seat를 4단계 혼잡도로 변환해서 분류 target 생성
# =========================================================
df["peak_congestion_class"] = df["remaining_seat"].apply(seat_to_peak_congestion_4)

print("\n[peak_congestion_class 전체 분포 - 4단계]")
print(df["peak_congestion_class"].value_counts(normalize=True).sort_index())


# =========================================================
# 11. 범주형 ID 인코딩
# =========================================================
# LightGBM 입력용으로 문자열 ID를 정수로 변환
# =========================================================
route_le = LabelEncoder()
stid_le = LabelEncoder()
arsid_le = LabelEncoder()

df["route_enc"] = route_le.fit_transform(df["busRouteId"])
df["stid_enc"] = stid_le.fit_transform(df["stId"])
df["arsid_enc"] = arsid_le.fit_transform(df["arsId"])


# =========================================================
# 12. 날짜 기준 train / valid / test 분할
# =========================================================
# 시계열 성격이 있으므로 랜덤분할이 아니라 날짜 기준 분할 사용
# =========================================================
df = df.sort_values("mkTm").reset_index(drop=True)

unique_dates = sorted(df["date"].unique())
print("\n사용 날짜들:", unique_dates)
print("총 사용 날짜 수:", len(unique_dates))

if len(unique_dates) < 10:
    raise ValueError("최소 10일 이상은 있어야 안정적으로 분할 가능합니다.")

n_dates = len(unique_dates)

# 앞 70% train / 다음 15% valid / 마지막 15% test
train_end = int(n_dates * 0.7)
valid_end = int(n_dates * 0.85)

train_dates = unique_dates[:train_end]
valid_dates = unique_dates[train_end:valid_end]
test_dates = unique_dates[valid_end:]

print("train dates:", train_dates[0], "~", train_dates[-1], f"({len(train_dates)}일)")
print("valid dates:", valid_dates[0], "~", valid_dates[-1], f"({len(valid_dates)}일)")
print("test dates :", test_dates[0], "~", test_dates[-1], f"({len(test_dates)}일)")

train_df = df[df["date"].isin(train_dates)].copy()
valid_df = df[df["date"].isin(valid_dates)].copy()
test_df = df[df["date"].isin(test_dates)].copy()


# =========================================================
# 13. 패턴 통계 feature 생성 함수
# =========================================================
# train_base에서만 통계를 만들고,
# 그 통계를 target_df에 merge해서 feature로 붙임
#
# 왜 train_base만 쓰나?
# -> valid/test 정보를 미리 쓰면 leakage(데이터 누수) 발생
#
# 생성되는 통계 예시:
# - route_mean_seat: 노선 단위 평균 좌석 수
# - route_stop_mean_seat: 노선+정류장 평균 좌석 수
# - route_time_mean_seat: 노선+요일+시간 평균 좌석 수
# - route_low_ratio: 노선 단위 혼잡 비율 등
# =========================================================
def add_pattern_features(train_base, target_df):
    result = target_df.copy()

    # 특정 조합 통계가 없을 때 사용할 fallback 값
    global_mean = train_base["remaining_seat"].mean()
    global_low_ratio = train_base["is_low_seat"].mean()

    # -----------------------------------------------------
    # 1) 노선 단위 통계
    # -----------------------------------------------------
    route_stat = (
        train_base.groupby("busRouteId")
        .agg(
            route_mean_seat=("remaining_seat", "mean"),
            route_std_seat=("remaining_seat", "std"),
            route_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(route_stat, on="busRouteId", how="left")

    # -----------------------------------------------------
    # 2) 노선 + 정류장 통계
    # -----------------------------------------------------
    route_stop_stat = (
        train_base.groupby(["busRouteId", "stId"])
        .agg(
            route_stop_mean_seat=("remaining_seat", "mean"),
            route_stop_std_seat=("remaining_seat", "std"),
            route_stop_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(route_stop_stat, on=["busRouteId", "stId"], how="left")

    # -----------------------------------------------------
    # 3) 노선 + 정류장 + 요일 + 시간그룹 통계
    # -----------------------------------------------------
    route_stop_time_stat = (
        train_base.groupby(["busRouteId", "stId", "dayofweek", "hour_group"])
        .agg(
            route_stop_time_mean_seat=("remaining_seat", "mean"),
            route_stop_time_std_seat=("remaining_seat", "std"),
            route_stop_time_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_stop_time_stat,
        on=["busRouteId", "stId", "dayofweek", "hour_group"],
        how="left"
    )

    # -----------------------------------------------------
    # 4) 노선 + 정류장 순번(staOrd) 통계
    # -----------------------------------------------------
    route_staord_stat = (
        train_base.groupby(["busRouteId", "staOrd"])
        .agg(
            route_staord_mean_seat=("remaining_seat", "mean"),
            route_staord_std_seat=("remaining_seat", "std"),
            route_staord_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_staord_stat,
        on=["busRouteId", "staOrd"],
        how="left"
    )

    # -----------------------------------------------------
    # 5) 노선 + 요일 + 시간 통계
    # -----------------------------------------------------
    route_time_stat = (
        train_base.groupby(["busRouteId", "dayofweek", "hour"])
        .agg(
            route_time_mean_seat=("remaining_seat", "mean"),
            route_time_std_seat=("remaining_seat", "std"),
            route_time_low_ratio=("is_low_seat", "mean"),
        )
        .reset_index()
    )
    result = result.merge(
        route_time_stat,
        on=["busRouteId", "dayofweek", "hour"],
        how="left"
    )

    # 평균 / 표준편차 / 혼잡비율 컬럼 목록
    mean_cols = [
        "route_mean_seat",
        "route_stop_mean_seat",
        "route_stop_time_mean_seat",
        "route_staord_mean_seat",
        "route_time_mean_seat",
    ]

    std_cols = [
        "route_std_seat",
        "route_stop_std_seat",
        "route_stop_time_std_seat",
        "route_staord_std_seat",
        "route_time_std_seat",
    ]

    low_ratio_cols = [
        "route_low_ratio",
        "route_stop_low_ratio",
        "route_stop_time_low_ratio",
        "route_staord_low_ratio",
        "route_time_low_ratio",
    ]

    # 통계가 없는 경우 fallback 값 채우기
    for col in mean_cols:
        result[col] = result[col].fillna(global_mean)

    for col in std_cols:
        result[col] = result[col].fillna(0)

    for col in low_ratio_cols:
        result[col] = result[col].fillna(global_low_ratio)

    return result


# =========================================================
# 14. 패턴 feature 생성
# =========================================================
# train 통계로 train / valid / test에 동일한 방식으로 feature 부착
# =========================================================
train_df = add_pattern_features(train_df, train_df)
valid_df = add_pattern_features(train_df, valid_df)
test_df = add_pattern_features(train_df, test_df)


# =========================================================
# 15. train 기준 패턴 통계 저장
# =========================================================
# 왜 저장하나?
# -> 나중에 실서비스에서 새 입력 1건이 들어왔을 때
#    이 통계값들을 다시 붙여야 모델 입력을 완성할 수 있기 때문
#
# 저장 파일:
# - pattern_route_stat.csv
# - pattern_route_stop_stat.csv
# - pattern_route_stop_time_stat.csv
# - pattern_route_staord_stat.csv
# - pattern_route_time_stat.csv
# - pattern_meta.json
# =========================================================

# 1) 노선 단위 통계
route_stat = (
    train_df.groupby("busRouteId")
    .agg(
        route_mean_seat=("remaining_seat", "mean"),
        route_std_seat=("remaining_seat", "std"),
        route_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

# 2) 노선 + 정류장 통계
route_stop_stat = (
    train_df.groupby(["busRouteId", "stId"])
    .agg(
        route_stop_mean_seat=("remaining_seat", "mean"),
        route_stop_std_seat=("remaining_seat", "std"),
        route_stop_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

# 3) 노선 + 정류장 + 요일 + 시간그룹 통계
route_stop_time_stat = (
    train_df.groupby(["busRouteId", "stId", "dayofweek", "hour_group"])
    .agg(
        route_stop_time_mean_seat=("remaining_seat", "mean"),
        route_stop_time_std_seat=("remaining_seat", "std"),
        route_stop_time_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

# 4) 노선 + 정류장 순번 통계
route_staord_stat = (
    train_df.groupby(["busRouteId", "staOrd"])
    .agg(
        route_staord_mean_seat=("remaining_seat", "mean"),
        route_staord_std_seat=("remaining_seat", "std"),
        route_staord_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

# 5) 노선 + 요일 + 시간 통계
route_time_stat = (
    train_df.groupby(["busRouteId", "dayofweek", "hour"])
    .agg(
        route_time_mean_seat=("remaining_seat", "mean"),
        route_time_std_seat=("remaining_seat", "std"),
        route_time_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

# CSV 저장
route_stat.to_csv(os.path.join(ARTIFACT_DIR, "pattern_route_stat.csv"), index=False, encoding="utf-8-sig")
route_stop_stat.to_csv(os.path.join(ARTIFACT_DIR, "pattern_route_stop_stat.csv"), index=False, encoding="utf-8-sig")
route_stop_time_stat.to_csv(os.path.join(ARTIFACT_DIR, "pattern_route_stop_time_stat.csv"), index=False, encoding="utf-8-sig")
route_staord_stat.to_csv(os.path.join(ARTIFACT_DIR, "pattern_route_staord_stat.csv"), index=False, encoding="utf-8-sig")
route_time_stat.to_csv(os.path.join(ARTIFACT_DIR, "pattern_route_time_stat.csv"), index=False, encoding="utf-8-sig")

# fallback용 메타정보 저장
pattern_meta = {
    "global_mean": float(train_df["remaining_seat"].mean()),
    "global_low_ratio": float(train_df["is_low_seat"].mean()),
    "max_seat": int(MAX_SEAT)
}

with open(os.path.join(ARTIFACT_DIR, "pattern_meta.json"), "w", encoding="utf-8") as f:
    json.dump(pattern_meta, f, ensure_ascii=False, indent=2)

print("\n패턴 통계 저장 완료")


# =========================================================
# 16. 모델 입력 feature 정의
# =========================================================
# 실제 모델이 사용하는 입력 컬럼 목록
# =========================================================
FEATURE_COLS = [
    # 인코딩된 ID
    "route_enc", "stid_enc",

    # 시간 관련 기본 컬럼
    "year", "month", "day",
    "hour", "minute", "dayofweek",
    "is_weekend", "is_peak", "peak_level",

    # 주기형 인코딩
    "month_sin", "month_cos",
    "day_sin", "day_cos",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dow_sin", "dow_cos",

    # 정류장 순서
    "staOrd",

    # 보조 시간 그룹
    "hour_group", "minute_group", "hour_weekday_key",

    # 패턴 통계
    "route_mean_seat", "route_std_seat",
    "route_stop_mean_seat", "route_stop_std_seat",
    "route_stop_time_mean_seat", "route_stop_time_std_seat",
    "route_staord_mean_seat", "route_staord_std_seat",
    "route_time_mean_seat", "route_time_std_seat",

    # 혼잡 관련 통계
    "route_low_ratio", "route_stop_low_ratio",
    "route_stop_time_low_ratio", "route_staord_low_ratio",
    "route_time_low_ratio",
]


# =========================================================
# 17. 전체 회귀 모델용 데이터 준비
# =========================================================
# 회귀모델은 전체 시간대 데이터를 사용
# 타깃은 remaining_seat
# =========================================================
X_train_reg = train_df[FEATURE_COLS]
y_train_reg = train_df["remaining_seat"]

X_valid_reg = valid_df[FEATURE_COLS]
y_valid_reg = valid_df["remaining_seat"]

X_test_reg = test_df[FEATURE_COLS]
y_test_reg = test_df["remaining_seat"]

print("\n회귀용 데이터 크기")
print("train:", X_train_reg.shape, y_train_reg.shape)
print("valid:", X_valid_reg.shape, y_valid_reg.shape)
print("test :", X_test_reg.shape, y_test_reg.shape)


# =========================================================
# 18. 전체 회귀 모델 학습
# =========================================================
# 잔여좌석 수를 직접 예측하는 핵심 모델
# =========================================================
lgbm_reg = LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=127,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

lgbm_reg.fit(
    X_train_reg,
    y_train_reg,
    eval_set=[(X_train_reg, y_train_reg), (X_valid_reg, y_valid_reg)],
    eval_metric="l1"
)


# =========================================================
# 19. 출퇴근 전용 분류 모델용 데이터 준비
# =========================================================
# is_peak == 1 인 데이터만 따로 뽑아서 분류 모델 학습
# 타깃은 peak_congestion_class
# =========================================================
peak_train_df = train_df[train_df["is_peak"] == 1].copy()
peak_valid_df = valid_df[valid_df["is_peak"] == 1].copy()
peak_test_df = test_df[test_df["is_peak"] == 1].copy()

X_train_cls = peak_train_df[FEATURE_COLS]
y_train_cls = peak_train_df["peak_congestion_class"]

X_valid_cls = peak_valid_df[FEATURE_COLS]
y_valid_cls = peak_valid_df["peak_congestion_class"]

X_test_cls = peak_test_df[FEATURE_COLS]
y_test_cls = peak_test_df["peak_congestion_class"]

print("\n출퇴근 전용 분류용 데이터 크기")
print("train:", X_train_cls.shape, y_train_cls.shape)
print("valid:", X_valid_cls.shape, y_valid_cls.shape)
print("test :", X_test_cls.shape, y_test_cls.shape)


# =========================================================
# 20. 출퇴근 전용 분류 모델 학습
# =========================================================
# class_weight='balanced'를 사용해 클래스 불균형을 일부 보정
# =========================================================
lgbm_peak_cls = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=800,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=127,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

lgbm_peak_cls.fit(
    X_train_cls,
    y_train_cls,
    eval_set=[(X_train_cls, y_train_cls), (X_valid_cls, y_valid_cls)],
    eval_metric="multi_logloss"
)


# =========================================================
# 21. 회귀 평가 함수
# =========================================================
# 예측값을 0~45 범위로 제한한 뒤 MAE / RMSE / R2 계산
# =========================================================
def evaluate_regression(model, X, y, name="dataset"):
    pred = np.clip(model.predict(X), 0, MAX_SEAT)

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)

    print(f"\n[{name}]")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return pred


# =========================================================
# 22. 분류 평가 함수
# =========================================================
# accuracy, macro F1, weighted F1, 혼동행렬, 리포트 출력
# + 긴 출력이 잘리는 문제를 막기 위해 txt 파일로도 저장
# =========================================================
def evaluate_classification(model, X, y, name="dataset"):
    pred = model.predict(X)

    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro")
    weighted_f1 = f1_score(y, pred, average="weighted")

    cm = confusion_matrix(y, pred)
    report = classification_report(y, pred, digits=4)

    print(f"\n[{name}]")
    print(f"ACC        : {acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(cm)
    print(report)

    file_name = os.path.join(ARTIFACT_DIR, f"{name.replace(' ', '_').lower()}_report.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(f"[{name}]\n")
        f.write(f"ACC        : {acc:.4f}\n")
        f.write(f"Macro F1   : {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write("Confusion Matrix\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report\n")
        f.write(report)

    print(f"📄 저장 완료: {file_name}")

    return pred


# =========================================================
# 23. 모델 평가
# =========================================================
# 전체 회귀 성능 평가
# 출퇴근 전용 분류 성능 평가
# =========================================================
valid_pred_reg = evaluate_regression(lgbm_reg, X_valid_reg, y_valid_reg, "VALID REG")
test_pred_reg = evaluate_regression(lgbm_reg, X_test_reg, y_test_reg, "TEST REG")

valid_pred_cls = evaluate_classification(
    lgbm_peak_cls, X_valid_cls, y_valid_cls, "VALID PEAK CLS 4CLASS"
)
test_pred_cls = evaluate_classification(
    lgbm_peak_cls, X_test_cls, y_test_cls, "TEST PEAK CLS 4CLASS"
)


# =========================================================
# 24. 서비스 추론 함수
# =========================================================
# 이 함수는 실제 서비스에서 사용할 예측 흐름을 구현한 것
#
# 처리 순서
# 1) 모든 행에 대해 회귀로 좌석 수 예측
# 2) 회귀 예측값을 기준으로 기본 혼잡도 생성
# 3) 출퇴근 시간대(is_peak=1)면 분류기 추가 적용
# 4) 최종 혼잡도는
#    - 출퇴근: 분류기 결과 우선
#    - 비출퇴근: 회귀 후처리 결과 사용
# =========================================================
def predict_service(row_df):
    result = row_df.copy()

    # -----------------------------------------------------
    # 1) 기본 좌석 수 예측
    # -----------------------------------------------------
    pred_seat = np.clip(lgbm_reg.predict(result[FEATURE_COLS]), 0, MAX_SEAT)
    result["pred_remaining_seat"] = np.round(pred_seat, 2)

    # -----------------------------------------------------
    # 2) 회귀 예측값 기반 기본 혼잡도 생성
    # -----------------------------------------------------
    result["pred_congestion_class_from_reg"] = (
        pd.Series(pred_seat).apply(seat_to_peak_congestion_4).values
    )
    result["pred_congestion_label_from_reg"] = (
        result["pred_congestion_class_from_reg"].apply(congestion_label_text_4)
    )

    # -----------------------------------------------------
    # 3) 출퇴근 시간대 분류기 결과 저장용 컬럼 초기화
    # -----------------------------------------------------
    result["pred_peak_cls"] = np.nan
    result["pred_peak_cls_label"] = None

    peak_mask = result["is_peak"] == 1

    # 출퇴근 시간대가 하나라도 있으면 분류기 적용
    if peak_mask.sum() > 0:
        peak_pred = lgbm_peak_cls.predict(result.loc[peak_mask, FEATURE_COLS])
        result.loc[peak_mask, "pred_peak_cls"] = peak_pred
        result.loc[peak_mask, "pred_peak_cls_label"] = (
            pd.Series(peak_pred).apply(congestion_label_text_4).values
        )

    # -----------------------------------------------------
    # 4) 최종 서비스용 혼잡도 결정
    # -----------------------------------------------------
    # 기본값: 회귀 후처리 기반 혼잡도
    result["final_congestion_class"] = result["pred_congestion_class_from_reg"]
    result["final_congestion_label"] = result["pred_congestion_label_from_reg"]

    # 출퇴근 시간대는 분류기 결과를 우선 사용
    result.loc[peak_mask, "final_congestion_class"] = result.loc[peak_mask, "pred_peak_cls"]
    result.loc[peak_mask, "final_congestion_label"] = result.loc[peak_mask, "pred_peak_cls_label"]

    return result


# =========================================================
# 25. 테스트셋 전체에 대해 서비스 방식으로 예측
# =========================================================
# test_df에 대해 실제 서비스 로직 그대로 적용한 결과 저장
# =========================================================
service_result = predict_service(test_df)

save_cols = [
    "mkTm", "busRouteId", "stId", "arsId", "staOrd",
    "remaining_seat", "is_peak",
    "pred_remaining_seat",
    "pred_congestion_class_from_reg", "pred_congestion_label_from_reg",
    "pred_peak_cls", "pred_peak_cls_label",
    "final_congestion_class", "final_congestion_label"
]

service_result[save_cols].to_csv(
    os.path.join(ARTIFACT_DIR, "test_service_result_peak_classifier_4class.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:", os.path.join(ARTIFACT_DIR, "test_service_result_peak_classifier_4class.csv"))
print("\n[test_service_result 샘플]")
print(service_result[save_cols].head(10))


# =========================================================
# 26. 노선별 정류소 순서 저장
# =========================================================
# 이후 full-route 예측 또는 특정 기준 정류소 앞뒤 정류장 탐색에 활용 가능
# =========================================================
route_station_order = (
    df[["busRouteId", "stId", "arsId", "staOrd"]]
    .drop_duplicates()
    .sort_values(["busRouteId", "staOrd"])
    .reset_index(drop=True)
)

route_station_order.to_csv(
    os.path.join(ARTIFACT_DIR, "route_station_order.csv"),
    index=False,
    encoding="utf-8-sig"
)
print("\n정류소 순서 저장 완료:", os.path.join(ARTIFACT_DIR, "route_station_order.csv"))


# =========================================================
# 27. 모델 / 인코더 저장
# =========================================================
# 나중에 실제 서비스 추론 시 재사용할 수 있도록 저장
# 실험 버전별로 구분하기 위해 별도 폴더에 저장
# =========================================================
model_dir = os.path.join(BASE_DIR, "models", "4class_0-5_6-15_16-30_31-45")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(lgbm_reg, os.path.join(model_dir, "reg.pkl"))
joblib.dump(lgbm_peak_cls, os.path.join(model_dir, "cls.pkl"))

joblib.dump(route_le, os.path.join(model_dir, "route_encoder.pkl"))
joblib.dump(stid_le, os.path.join(model_dir, "stid_encoder.pkl"))
joblib.dump(arsid_le, os.path.join(model_dir, "arsid_encoder.pkl"))

print("\n모델 저장 완료")
print(f"- {os.path.join(model_dir, 'reg.pkl')}")
print(f"- {os.path.join(model_dir, 'cls.pkl')}")
print(f"- {os.path.join(model_dir, 'route_encoder.pkl')}")
print(f"- {os.path.join(model_dir, 'stid_encoder.pkl')}")
print(f"- {os.path.join(model_dir, 'arsid_encoder.pkl')}")