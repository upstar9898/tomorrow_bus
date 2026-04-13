# =========================================================
# [모델 설계 개요 - LightGBM 회귀 + 혼잡도 분류 병렬 구조]
# =========================================================
# 목적:
# - 특정 날짜/시간/정류장/노선 조건에서 해당 시점의 잔여 좌석 수를 예측
# - 동시에 해당 시점의 혼잡도 클래스(0/1/2)도 분류
# -> 추후 더 세밀한 단계로 구분
#
# 핵심 전략:
# 1. 미래 타깃 생성(target_time, merge_asof) 제거
#    → "현재 기준 몇 분 뒤" 예측이 아니라, "그 시점 자체의 좌석 수" 예측
# 2. 각 행의 remaining_seat 자체를 회귀 타깃(y_reg)으로 사용
# 3. remaining_seat 기반 congestion_class를 분류 타깃(y_cls)으로 생성
# 4. lag 없이 패턴 기반 feature 사용
#    → 과거 동일 조건(노선/정류장/시간대)의 평균 패턴을 feature로 사용
# 5. train 통계만으로 valid/test 패턴 feature 생성
#    → 데이터 누수(leakage) 방지
#
# 참고:
# - 모델 입력은 FEATURE_COLS에 포함된 컬럼만 사용
# - date는 train/valid/test 날짜 분할용으로만 사용
# - is_low_seat는 혼잡 비율(route_low_ratio 등) 계산용 보조 컬럼
# - 예측값은 학습 시 제한하지 않고, 평가 및 실사용 시에만 0~45 범위로 clip 처리
# - full_flag는 현재 입력 feature에 넣지 않음
# =========================================================


# =========================================================
# 1. 라이브러리 import
# =========================================================

# 데이터프레임 처리용
import pandas as pd

# 수치 계산 및 결측 처리용
import numpy as np

# 패턴 통계 메타정보 저장용(JSON)
import json

# 문자열 ID를 숫자로 바꾸는 인코더
from sklearn.preprocessing import LabelEncoder

# 회귀/분류 성능평가 지표
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,  # 회귀 평가
    accuracy_score, f1_score, classification_report, confusion_matrix  # 분류 평가
)

# 학습된 모델/인코더 저장용
import joblib

# LightGBM 회귀 / 분류 모델
from lightgbm import LGBMRegressor, LGBMClassifier


# =========================================================
# 2. full_flag 관련 설명 (데이터 누수 체크)
# =========================================================
# full_flag: 만차 여부(이진 변수)로, 향후 별도 분류모델의 target으로 사용할 수 있는 컬럼
#
# [현재 상태]
# - 본 코드는 remaining_seat 회귀 + congestion_class 분류 구조
# - full_flag는 FEATURE_COLS에 포함되지 않아 입력 feature로 사용되지 않음
# → 따라서 현재 모델에서는 direct leakage 없음
#
# [주의 사항 - 향후 full_flag 모델 구축 시]
# - full_flag가 remaining_seat 기반으로 생성된 경우
#   remaining_seat 또는 동일 시점 직접 파생 변수들을 input에 넣으면 leakage 발생 가능
#
# [정리]
# - 현재: input 제외 상태 → leakage 없음
# - 향후: target으로 사용할 경우 feature 구성 재검토 필수
# =========================================================


# =========================================================
# 3. 설정값
# =========================================================

# 학습에 사용할 통합 CSV 파일 경로
file_path = "bus_all_raw3.csv"

# 버스 최대 좌석 수
# 예측 결과를 서비스에 사용할 때 0~45 범위로 제한하는 기준
MAX_SEAT = 45

# 잔여좌석이 10석 이하이면 "좌석 부족 상태"로 보는 기준
LOW_SEAT_THRESHOLD = 10

# 혼잡도 클래스 기준
# 0: 매우 혼잡 (<= 5석)
# 1: 보통      (6 ~ 15석)
# 2: 여유      (> 15석)


# =========================================================
# 4. 원본 파일 불러오기
# =========================================================

df = pd.read_csv(
    file_path,
    dtype={
        "busRouteId": str,  # 노선 ID는 숫자가 아니라 식별자이므로 문자열 유지
        "stId": str,        # 정류장 ID도 문자열 유지
        "arsId": str        # 정류소 번호도 문자열 유지
    },
    low_memory=False       # 큰 파일을 읽을 때 dtype 추론 오류를 줄이기 위해 사용
)

# 관측 시각(mkTm)을 datetime으로 변환
# 변환 실패값은 NaT로 처리
df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")


# =========================================================
# 5. 기본 정리
# =========================================================

# 핵심 컬럼이 비어 있는 행은 제거
# - mkTm: 시간 정보
# - busRouteId, stId, arsId: 식별자
# - remaining_seat: 회귀 타깃
df = df.dropna(subset=["mkTm", "busRouteId", "stId", "arsId", "remaining_seat"]).copy()

# 문자열 ID 컬럼은 공백 제거
df["busRouteId"] = df["busRouteId"].astype(str).str.strip()
df["stId"] = df["stId"].astype(str).str.strip()
df["arsId"] = df["arsId"].astype(str).str.strip()

# 잔여좌석 / 정류장 순서를 숫자형으로 변환
df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")

# exps1 컬럼이 존재하면 숫자형으로 변환
# 없으면 빈 컬럼 생성
if "exps1" in df.columns:
    df["exps1"] = pd.to_numeric(df["exps1"], errors="coerce")
else:
    df["exps1"] = np.nan

# full_flag 컬럼이 존재하면 숫자형으로 변환
# 없으면 빈 컬럼 생성
if "full_flag" in df.columns:
    df["full_flag"] = pd.to_numeric(df["full_flag"], errors="coerce")
else:
    df["full_flag"] = np.nan

# 숫자형 변환 후 remaining_seat 또는 staOrd가 NaN이 된 행 제거
df = df.dropna(subset=["remaining_seat", "staOrd"]).copy()

# 잔여 좌석 수가 음수인 행 제거 (비정상값)
df = df[df["remaining_seat"] >= 0].copy()

print("총 데이터 수:", len(df))

print("\n[remaining_seat 기초통계]")
print(df["remaining_seat"].describe())


# =========================================================
# 6. 시간 관련 feature 생성
# =========================================================

# 연 / 월 / 일 / 시 / 분 / 요일 추출
df["year"] = df["mkTm"].dt.year
df["month"] = df["mkTm"].dt.month
df["day"] = df["mkTm"].dt.day
df["hour"] = df["mkTm"].dt.hour
df["minute"] = df["mkTm"].dt.minute
df["dayofweek"] = df["mkTm"].dt.dayofweek

# 날짜만 따로 추출
# → train/valid/test를 날짜 기준으로 분할할 때 사용
df["date"] = df["mkTm"].dt.date

# 주말 여부
# 토요일(5), 일요일(6)이면 1
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# 출퇴근 시간대 여부
# 오전 7~9시 또는 오후 17~19시이면 1
df["is_peak"] = (
    ((df["hour"] >= 7) & (df["hour"] <= 9)) |
    ((df["hour"] >= 17) & (df["hour"] <= 19))
).astype(int)

# 출퇴근 시간대를 조금 더 세분화한 변수
# 0: 비출퇴근
# 1: 오전 7~8시
# 2: 오전 9시
# 3: 오후 17~18시
# 4: 오후 19시
df["peak_level"] = 0
df.loc[(df["hour"] >= 7) & (df["hour"] <= 8), "peak_level"] = 1
df.loc[df["hour"] == 9, "peak_level"] = 2
df.loc[(df["hour"] >= 17) & (df["hour"] <= 18), "peak_level"] = 3
df.loc[df["hour"] == 19, "peak_level"] = 4

# -----------------------------
# 주기형(cyclical) 변수 인코딩
# -----------------------------
# 월/일/시/분/요일은 숫자상 끝과 처음이 실제로 이어져 있으므로
# sin/cos 변환으로 순환 구조를 반영

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

# -----------------------------
# 시간대 보조 key
# -----------------------------
# 2시간 단위 그룹
df["hour_group"] = (df["hour"] // 2).astype(int)

# 10분 단위 그룹
df["minute_group"] = (df["minute"] // 10).astype(int)

# 시간 + 요일 조합 key
# 예: 월요일 8시, 화요일 8시를 구분하기 위한 보조 변수
df["hour_weekday_key"] = df["hour"] * 10 + df["dayofweek"]


# =========================================================
# 7. 혼잡 보정용 라벨 + 분류용 타깃 생성
# =========================================================

# 잔여 좌석이 LOW_SEAT_THRESHOLD 이하인지 여부
# → 혼잡 비율 통계 계산용
df["is_low_seat"] = (df["remaining_seat"] <= LOW_SEAT_THRESHOLD).astype(int)

# 좌석 수를 혼잡도 클래스로 바꾸는 함수
def seat_to_congestion(seat):
    # 매우 혼잡
    if seat <= 5:
        return 0
    # 보통
    elif seat <= 15:
        return 1
    # 여유
    else:
        return 2

# 분류 타깃 생성
df["congestion_class"] = df["remaining_seat"].apply(seat_to_congestion)

print("\n[congestion_class 분포]")
print(df["congestion_class"].value_counts(normalize=True).sort_index())


# =========================================================
# 8. 범주형 ID 인코딩
# =========================================================

# LightGBM 입력을 위해 문자열 ID를 정수로 변환할 인코더 생성
route_le = LabelEncoder()
stid_le = LabelEncoder()
arsid_le = LabelEncoder()

# 노선 / 정류장 ID 인코딩
df["route_enc"] = route_le.fit_transform(df["busRouteId"])
df["stid_enc"] = stid_le.fit_transform(df["stId"])

# 현재 FEATURE_COLS에는 안 쓰지만, 필요 시 활용 가능하도록 인코더는 저장
# 여기서는 실제 입력 feature에 사용하지 않으므로 df 컬럼으로는 만들지 않음


# =========================================================
# 9. 날짜 기준 train / valid / test 분할
# =========================================================

# 시간순 정렬
df = df.sort_values("mkTm").reset_index(drop=True)

# 날짜만 중복 없이 추출
unique_dates = sorted(df["date"].unique())

print("사용 날짜들:", unique_dates)
print("총 사용 날짜 수:", len(unique_dates))

# 날짜 수가 너무 적으면 분할이 불안정하므로 방어 코드
if len(unique_dates) < 10:
    raise ValueError("최소 10일 이상은 있어야 안정적으로 분할 가능합니다.")

n_dates = len(unique_dates)

# 앞 70% = train, 다음 15% = valid, 마지막 15% = test
train_end = int(n_dates * 0.7)
valid_end = int(n_dates * 0.85)

train_dates = unique_dates[:train_end]
valid_dates = unique_dates[train_end:valid_end]
test_dates = unique_dates[valid_end:]

print("train dates:", train_dates[0], "~", train_dates[-1], f"({len(train_dates)}일)")
print("valid dates:", valid_dates[0], "~", valid_dates[-1], f"({len(valid_dates)}일)")
print("test dates :", test_dates[0], "~", test_dates[-1], f"({len(test_dates)}일)")

# 실제 데이터셋 분리
train_df = df[df["date"].isin(train_dates)].copy()
valid_df = df[df["date"].isin(valid_dates)].copy()
test_df = df[df["date"].isin(test_dates)].copy()


# =========================================================
# 10. 패턴 통계 feature 생성 함수
# =========================================================
def add_pattern_features(train_base, target_df):
    """
    train_base에서 통계를 계산하고,
    그 통계를 target_df에 merge해서 패턴 feature를 붙이는 함수

    train_base:
        통계를 계산할 기준 데이터
        → 반드시 train_df만 넣어야 leakage 방지 가능

    target_df:
        통계 feature를 붙일 대상 데이터
        → train_df / valid_df / test_df 모두 가능
    """
    result = target_df.copy()

    # fallback용 전체 평균 / 전체 혼잡비율
    global_mean = train_base["remaining_seat"].mean()
    global_low_ratio = train_base["is_low_seat"].mean()

    # -----------------------------------------
    # 1) 노선 단위 평균 통계
    # -----------------------------------------
    route_stat = (
        train_base.groupby("busRouteId")
        .agg(
            route_mean_seat=("remaining_seat", "mean"),   # 노선 평균 좌석 수
            route_std_seat=("remaining_seat", "std"),     # 노선 좌석 변동성
            route_low_ratio=("is_low_seat", "mean"),      # 노선 혼잡 비율
        )
        .reset_index()
    )
    result = result.merge(route_stat, on="busRouteId", how="left")

    # -----------------------------------------
    # 2) 노선 + 정류장 통계
    # -----------------------------------------
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

    # -----------------------------------------
    # 3) 노선 + 정류장 + 요일 + 시간그룹 통계
    # -----------------------------------------
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

    # -----------------------------------------
    # 4) 노선 + 정류장 순번(staOrd) 통계
    # -----------------------------------------
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

    # -----------------------------------------
    # 5) 노선 + 요일 + 시간 통계
    # -----------------------------------------
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

    # 학습 통계가 없는 조합은 fallback 값으로 채움
    for col in mean_cols:
        result[col] = result[col].fillna(global_mean)

    for col in std_cols:
        result[col] = result[col].fillna(0)

    for col in low_ratio_cols:
        result[col] = result[col].fillna(global_low_ratio)

    return result


# train 통계로 train / valid / test 각각 패턴 feature 생성
train_df = add_pattern_features(train_df, train_df)
valid_df = add_pattern_features(train_df, valid_df)
test_df = add_pattern_features(train_df, test_df)


# =========================================================
# 11. train 기준 패턴 통계 저장
# =========================================================
# 서비스 추론 시에도 같은 통계를 다시 써야 하므로 파일로 저장

route_stat = (
    train_df.groupby("busRouteId")
    .agg(
        route_mean_seat=("remaining_seat", "mean"),
        route_std_seat=("remaining_seat", "std"),
        route_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

route_stop_stat = (
    train_df.groupby(["busRouteId", "stId"])
    .agg(
        route_stop_mean_seat=("remaining_seat", "mean"),
        route_stop_std_seat=("remaining_seat", "std"),
        route_stop_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

route_stop_time_stat = (
    train_df.groupby(["busRouteId", "stId", "dayofweek", "hour_group"])
    .agg(
        route_stop_time_mean_seat=("remaining_seat", "mean"),
        route_stop_time_std_seat=("remaining_seat", "std"),
        route_stop_time_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

route_staord_stat = (
    train_df.groupby(["busRouteId", "staOrd"])
    .agg(
        route_staord_mean_seat=("remaining_seat", "mean"),
        route_staord_std_seat=("remaining_seat", "std"),
        route_staord_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

route_time_stat = (
    train_df.groupby(["busRouteId", "dayofweek", "hour"])
    .agg(
        route_time_mean_seat=("remaining_seat", "mean"),
        route_time_std_seat=("remaining_seat", "std"),
        route_time_low_ratio=("is_low_seat", "mean"),
    )
    .reset_index()
)

# 각 통계 테이블을 CSV로 저장
route_stat.to_csv("pattern_route_stat.csv", index=False, encoding="utf-8-sig")
route_stop_stat.to_csv("pattern_route_stop_stat.csv", index=False, encoding="utf-8-sig")
route_stop_time_stat.to_csv("pattern_route_stop_time_stat.csv", index=False, encoding="utf-8-sig")
route_staord_stat.to_csv("pattern_route_staord_stat.csv", index=False, encoding="utf-8-sig")
route_time_stat.to_csv("pattern_route_time_stat.csv", index=False, encoding="utf-8-sig")

# 전체 평균 / 혼잡비율 / 최대좌석 수 같은 fallback 메타정보 저장
pattern_meta = {
    "global_mean": float(train_df["remaining_seat"].mean()),
    "global_low_ratio": float(train_df["is_low_seat"].mean()),
    "max_seat": int(MAX_SEAT)
}

with open("pattern_meta.json", "w", encoding="utf-8") as f:
    json.dump(pattern_meta, f, ensure_ascii=False, indent=2)

print("패턴 통계 저장 완료")


# =========================================================
# 12. 모델 입력 feature 정의
# =========================================================

FEATURE_COLS = [
    # 인코딩
    "route_enc", "stid_enc",

    # 시간
    "year", "month", "day",
    "hour", "minute", "dayofweek",
    "is_weekend", "is_peak", "peak_level",

    # 주기 인코딩
    "month_sin", "month_cos",
    "day_sin", "day_cos",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dow_sin", "dow_cos",

    # 위치/운행
    "staOrd",
    "hour_group",
    "minute_group",
    "hour_weekday_key",

    # 평균 패턴
    "route_mean_seat", "route_std_seat",
    "route_stop_mean_seat", "route_stop_std_seat",
    "route_stop_time_mean_seat", "route_stop_time_std_seat",
    "route_staord_mean_seat", "route_staord_std_seat",
    "route_time_mean_seat", "route_time_std_seat",

    # 혼잡 보정
    "route_low_ratio",
    "route_stop_low_ratio",
    "route_stop_time_low_ratio",
    "route_staord_low_ratio",
    "route_time_low_ratio",
]

# 회귀 타깃 / 분류 타깃
REG_TARGET_COL = "remaining_seat"
CLS_TARGET_COL = "congestion_class"

# 현재 모델에서는 exps1, full_flag 미사용
# 필요 시 향후 별도 실험 가능
# for df_part in [train_df, valid_df, test_df]:
#     df_part["exps1"] = df_part["exps1"].fillna(train_df["exps1"].median())
#     df_part["full_flag"] = df_part["full_flag"].fillna(0)

# 모델 입력(X) / 정답(y) 분리
X_train = train_df[FEATURE_COLS]
y_train_reg = train_df[REG_TARGET_COL]
y_train_cls = train_df[CLS_TARGET_COL]

X_valid = valid_df[FEATURE_COLS]
y_valid_reg = valid_df[REG_TARGET_COL]
y_valid_cls = valid_df[CLS_TARGET_COL]

X_test = test_df[FEATURE_COLS]
y_test_reg = test_df[REG_TARGET_COL]
y_test_cls = test_df[CLS_TARGET_COL]

print("train:", X_train.shape, y_train_reg.shape, y_train_cls.shape)
print("valid:", X_valid.shape, y_valid_reg.shape, y_valid_cls.shape)
print("test :", X_test.shape, y_test_reg.shape, y_test_cls.shape)


# =========================================================
# 13. 회귀 모델 학습
# =========================================================

lgbm_reg = LGBMRegressor(
    objective="regression",   # 회귀 문제
    n_estimators=800,         # 부스팅 반복 횟수
    learning_rate=0.05,       # 학습률
    max_depth=10,             # 트리 최대 깊이
    num_leaves=127,           # 리프 노드 수
    min_child_samples=10,     # 최소 샘플 수
    subsample=0.8,            # 행 샘플링 비율
    colsample_bytree=0.8,     # 열 샘플링 비율
    reg_alpha=0.1,            # L1 정규화
    reg_lambda=0.1,           # L2 정규화
    random_state=42,          # 재현성 고정
    n_jobs=-1                 # CPU 전체 사용
)

lgbm_reg.fit(
    X_train,
    y_train_reg,
    eval_set=[(X_train, y_train_reg), (X_valid, y_valid_reg)],
    eval_metric="l1"          # MAE 기준 모니터링
)

print("LightGBM 회귀 모델 학습 완료")


# =========================================================
# 14. 회귀 평가 함수
# =========================================================
def evaluate_regression(model, X, y, name="dataset"):
    # 예측 수행
    pred = model.predict(X)

    # 예측 좌석 수를 0 ~ MAX_SEAT 범위로 제한
    pred = np.clip(pred, 0, MAX_SEAT)

    # 성능 지표 계산
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)

    print(f"\n[{name}]")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    print("\n실제값 통계:")
    print(pd.Series(y).describe())

    print("\n예측값 통계:")
    print(pd.Series(pred).describe())

    return pred

# valid / test 회귀 성능 평가
valid_pred_reg = evaluate_regression(lgbm_reg, X_valid, y_valid_reg, "VALID")
test_pred_reg = evaluate_regression(lgbm_reg, X_test, y_test_reg, "TEST")


# =========================================================
# 15. 출퇴근 / 비출퇴근 회귀 성능
# =========================================================
def evaluate_regression_by_peak(model, df_eval, name="TEST"):
    # 출퇴근 / 비출퇴근 나눠서 성능 확인
    peak_df = df_eval[df_eval["is_peak"] == 1].copy()
    non_peak_df = df_eval[df_eval["is_peak"] == 0].copy()

    print(f"\n===== {name} (출퇴근 vs 비출퇴근 성능) =====")

    if len(peak_df) > 0:
        print("\n[출퇴근 시간대 (is_peak=1)]")
        evaluate_regression(model, peak_df[FEATURE_COLS], peak_df[REG_TARGET_COL], "PEAK")

    if len(non_peak_df) > 0:
        print("\n[비출퇴근 시간대 (is_peak=0)]")
        evaluate_regression(model, non_peak_df[FEATURE_COLS], non_peak_df[REG_TARGET_COL], "NON-PEAK")

evaluate_regression_by_peak(lgbm_reg, valid_df, "VALID")
evaluate_regression_by_peak(lgbm_reg, test_df, "TEST")


# =========================================================
# 16. 분류 모델 학습
# =========================================================

lgbm_cls = LGBMClassifier(
    objective="multiclass",   # 다중분류 문제
    num_class=3,              # 혼잡도 클래스 3개
    n_estimators=800,
    learning_rate=0.05,
    max_depth=10,
    num_leaves=127,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",  # 클래스 불균형 보정
    random_state=42,
    n_jobs=-1
)

lgbm_cls.fit(
    X_train,
    y_train_cls,
    eval_set=[(X_train, y_train_cls), (X_valid, y_valid_cls)],
    eval_metric="multi_logloss"
)

print("\nLightGBM 분류 모델 학습 완료")


# =========================================================
# 17. 분류 평가 함수
# =========================================================
def evaluate_classification(model, X, y, name="dataset"):
    # 분류 결과 및 확률 예측
    pred = model.predict(X)
    pred_proba = model.predict_proba(X)

    # 성능 지표 계산
    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro")
    weighted_f1 = f1_score(y, pred, average="weighted")
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])

    print(f"\n[{name}]")
    print(f"ACC        : {acc:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\n실제 class 분포:")
    print(pd.Series(y).value_counts(normalize=True).sort_index())

    print("\n예측 class 분포:")
    print(pd.Series(pred).value_counts(normalize=True).sort_index())

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y, pred, digits=4))

    return pred, pred_proba

# valid / test 분류 성능 평가
valid_pred_cls, valid_pred_proba = evaluate_classification(
    lgbm_cls, X_valid, y_valid_cls, "VALID CLASSIFICATION"
)

test_pred_cls, test_pred_proba = evaluate_classification(
    lgbm_cls, X_test, y_test_cls, "TEST CLASSIFICATION"
)


# =========================================================
# 18. 출퇴근 / 비출퇴근 분류 성능
# =========================================================
def evaluate_classification_by_peak(model, df_eval, name="TEST"):
    # 출퇴근 / 비출퇴근 구간별 분류 성능 확인
    peak_df = df_eval[df_eval["is_peak"] == 1].copy()
    non_peak_df = df_eval[df_eval["is_peak"] == 0].copy()

    print(f"\n===== {name} CLASSIFICATION (출퇴근 vs 비출퇴근 성능) =====")

    if len(peak_df) > 0:
        print("\n[출퇴근 시간대 (is_peak=1)]")
        evaluate_classification(
            model,
            peak_df[FEATURE_COLS],
            peak_df[CLS_TARGET_COL],
            "PEAK CLASSIFICATION"
        )

    if len(non_peak_df) > 0:
        print("\n[비출퇴근 시간대 (is_peak=0)]")
        evaluate_classification(
            model,
            non_peak_df[FEATURE_COLS],
            non_peak_df[CLS_TARGET_COL],
            "NON-PEAK CLASSIFICATION"
        )

evaluate_classification_by_peak(lgbm_cls, valid_df, "VALID")
evaluate_classification_by_peak(lgbm_cls, test_df, "TEST")


# =========================================================
# 19. 회귀 기반 혼잡도 분포 참고용 확인
# =========================================================

# 회귀 예측 좌석수를 반올림해서 혼잡도 클래스로 변환
test_pred_reg_rounded = np.clip(np.round(test_pred_reg), 0, MAX_SEAT)
test_pred_reg_class = pd.Series(test_pred_reg_rounded).apply(seat_to_congestion)

# 실제 test 좌석수도 혼잡도 클래스로 변환
test_true_class = y_test_reg.apply(seat_to_congestion)

print("\n[참고용 TEST class 분포 - 실제]")
print(test_true_class.value_counts(normalize=True).sort_index())

print("\n[참고용 TEST class 분포 - 회귀예측값 기반 변환]")
print(test_pred_reg_class.value_counts(normalize=True).sort_index())

print("\n[직접 분류모델 TEST class 분포 - 예측]")
print(pd.Series(test_pred_cls).value_counts(normalize=True).sort_index())


# =========================================================
# 20. 모델 / 인코더 저장
# =========================================================

# 회귀 / 분류 모델 저장
joblib.dump(lgbm_reg, "lgbm_point_seat_regressor_final.pkl")
joblib.dump(lgbm_cls, "lgbm_congestion_classifier_final.pkl")

# 인코더 저장
joblib.dump(route_le, "route_label_encoder.pkl")
joblib.dump(stid_le, "stid_label_encoder.pkl")
joblib.dump(arsid_le, "arsid_label_encoder.pkl")

print("\n회귀 모델 / 분류 모델 / 인코더 저장 완료")


# =========================================================
# 21. test set 예측 결과 저장
# =========================================================

# 회귀 예측
test_pred_raw_reg = lgbm_reg.predict(X_test)
test_pred_service_reg = np.clip(test_pred_raw_reg, 0, MAX_SEAT)

# 분류 예측
test_pred_service_cls = lgbm_cls.predict(X_test)
test_pred_service_cls_proba = lgbm_cls.predict_proba(X_test)

# 결과를 보기 좋게 정리
service_result = test_df[[
    "mkTm", "busRouteId", "stId", "arsId", "staOrd",
    "remaining_seat", "congestion_class"
]].copy()

service_result["pred_remaining_seat"] = np.round(test_pred_service_reg, 2)
service_result["pred_congestion_class"] = test_pred_service_cls
service_result["pred_congestion_prob_0"] = test_pred_service_cls_proba[:, 0]
service_result["pred_congestion_prob_1"] = test_pred_service_cls_proba[:, 1]
service_result["pred_congestion_prob_2"] = test_pred_service_cls_proba[:, 2]

print("\n[test set 예측 결과 샘플]")
print(service_result.head(10))

# test 예측 결과 CSV 저장
service_result.to_csv("test_service_result_with_reg_and_cls.csv", index=False, encoding="utf-8-sig")
print("\ntest set 예측 결과 저장 완료: test_service_result_with_reg_and_cls.csv")


# =========================================================
# 22. 노선별 정류소 순서 저장
# =========================================================
# 이후 full-route 예측 서비스에서
# 기준 정류소를 중심으로 이전/이후 정류소를 찾기 위해 필요

route_station_order = (
    df[["busRouteId", "stId", "arsId", "staOrd"]]
    .drop_duplicates()
    .sort_values(["busRouteId", "staOrd"])
    .reset_index(drop=True)
)

route_station_order.to_csv("route_station_order.csv", index=False, encoding="utf-8-sig")
print("정류소 순서 저장 완료")