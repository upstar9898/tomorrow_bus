# =========================================================
# [모델 설계 개요 - LightGBM 단독 + Pattern Feature 기반]
# =========================================================
# 목적:
# - 특정 날짜/시간/정류장/노선 조건에서 해당 시점의 좌석 수를 예측
# - "현재 기준 10분 뒤"가 아니라 "그 시점 자체의 좌석 수"를 예측함

# 핵심 전략:
# 1. 미래 타깃 생성(target_time, merge_asof) 제거
# 2. 각 행의 remaining_seat 자체를 타깃(y)으로 사용
# 3. lag 없이 패턴 기반 feature 사용
# 4. train 통계만으로 valid/test 패턴 feature 생성

# 참고:
# - 모델 입력은 FEATURE_COLS에 포함된 컬럼만 사용
# - date는 분할용으로만 사용
# - is_low_seat는 혼잡 비율(route_low_ratio 등) 계산용 보조 컬럼
# - 예측값은 모델 학습 시 제한하지 않고, 평가 및 실사용 시에만 0~45 범위로 clip 처리
# "full_flag" -> 만차 확률 모델 만들 때 target으로 사용할 예정
# =========================================================

# 데이터 분석용 라이브러리
import pandas as pd          # 표 형태 데이터(DataFrame) 처리
import numpy as np           # 수치 계산, 결측값(np.nan) 처리 등에 사용

# 전처리 / 평가 관련 라이브러리
from sklearn.preprocessing import LabelEncoder   # 문자열 범주형 데이터를 숫자로 인코딩
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 회귀모델 성능평가 지표
# - mean_absolute_error: 평균 절대 오차(MAE)
# - mean_squared_error: 평균 제곱 오차(MSE)
# - r2_score: 결정계수(R²)

import joblib                # 학습된 모델/인코더 저장 및 불러오기용
from lightgbm import LGBMRegressor   # LightGBM 회귀 모델

# =========================================================
# 0. 설정
# =========================================================

file_path = "bus_all_raw.csv"
# 사용할 원본 데이터 파일 경로
# 현재 작업 폴더에 bus_all_raw.csv가 있어야 읽을 수 있음

MAX_SEAT = 45
# 버스 최대 좌석 수
# 예측 결과를 나중에 0~45 범위로 제한(clip)할 때 기준으로 사용

LOW_SEAT_THRESHOLD = 10   # 혼잡 기준
# 잔여 좌석이 10석 이하이면 "좌석이 부족한 혼잡 상태"라고 보기 위한 기준값
# 이후 is_low_seat 같은 파생변수 만들 때 사용됨

# =========================================================
# 1. 파일 불러오기
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
# CSV 파일을 DataFrame으로 읽어옴
#
# dtype 지정 이유:
# - busRouteId, stId, arsId 는 숫자처럼 보여도 "식별자(ID)" 역할이라
#   계산용 숫자가 아니라 문자열로 유지하는 게 안전함
# - 예: 앞에 0이 있는 값, 큰 숫자 ID, 혼합형 데이터가 있을 수 있음
#
# low_memory=False 이유:
# - 큰 CSV를 읽을 때 pandas가 열 타입을 중간중간 추측하다가
#   dtype warning이 나는 걸 줄여줌
# - 메모리는 조금 더 쓰지만 타입이 덜 꼬임

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
# mkTm 컬럼을 날짜시간(datetime) 형식으로 변환
#
# 왜 필요하냐:
# - 이후 hour, minute, dayofweek 같은 시간 feature를 만들려면
#   날짜형으로 바꿔야 함
#
# errors="coerce" 의미:
# - 변환이 안 되는 잘못된 값은 에러를 내지 않고 NaT(결측 날짜)로 처리
# - 나중에 dropna로 정리 가능

# =========================================================
# 2. 기본 정리
# =========================================================

df = df.dropna(subset=["mkTm", "busRouteId", "stId", "arsId", "remaining_seat"]).copy()
# 핵심 컬럼에 결측값이 있는 행 제거
# - mkTm: 시간 정보
# - busRouteId: 노선 ID
# - stId: 정류장 ID
# - arsId: 정류소 번호
# - remaining_seat: 예측 타깃(잔여 좌석 수)
# 이 값들이 없으면 이후 feature 생성이나 모델 학습이 어려우므로 미리 제거
# .copy()를 붙여 SettingWithCopyWarning을 방지

df["busRouteId"] = df["busRouteId"].astype(str).str.strip()
# busRouteId를 문자열로 통일하고, 앞뒤 공백 제거
# 예: " 100100118 " -> "100100118"

df["stId"] = df["stId"].astype(str).str.strip()
# stId도 문자열 통일 + 공백 제거

df["arsId"] = df["arsId"].astype(str).str.strip()
# arsId도 문자열 통일 + 공백 제거

df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
# remaining_seat를 숫자형으로 변환
# 숫자로 바꿀 수 없는 값은 NaN 처리
# 예: 문자, 빈값, 이상한 기호가 들어있으면 결측으로 바뀜

df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
# 정류장 순번(staOrd)도 숫자형으로 변환
# 이후 노선 내 정류장 위치를 나타내는 feature로 사용 예정

if "exps1" in df.columns:
    df["exps1"] = pd.to_numeric(df["exps1"], errors="coerce")
# exps1 컬럼이 있으면 숫자형으로 변환
# exps1은 도착예정시간(초) 관련 컬럼으로 보조 feature 후보
# 숫자로 바꿀 수 없는 값은 NaN 처리

else:
    df["exps1"] = np.nan
# exps1 컬럼이 아예 없으면 새로 만들고 NaN으로 채움
# 이렇게 해두면 데이터셋마다 컬럼 유무가 달라도 코드가 안 깨짐

if "full_flag" in df.columns:
    df["full_flag"] = pd.to_numeric(df["full_flag"], errors="coerce")
# full_flag 컬럼이 있으면 숫자형으로 변환
# 만차 여부 같은 보조 정보로 사용할 수 있음

else:
    df["full_flag"] = np.nan
# full_flag 컬럼이 없으면 NaN 컬럼 생성
# 마찬가지로 컬럼 유무 차이에 대응하기 위한 처리

df = df.dropna(subset=["remaining_seat", "staOrd"]).copy()
# 숫자 변환 후 remaining_seat 또는 staOrd가 NaN이 된 행 제거
# 즉, 실제로 숫자형으로 쓸 수 없는 데이터 제거

df = df[df["remaining_seat"] >= 0].copy()
# 잔여 좌석 수가 음수인 비정상 데이터 제거
# 좌석 수는 음수가 될 수 없으므로 이상치로 판단

print("총 데이터 수:", len(df))
# 최종 전처리 후 남은 데이터 개수 확인

print("\n[remaining_seat 기초통계]")
print(df["remaining_seat"].describe())
# remaining_seat의 기초통계 출력
# count, mean, std, min, 25%, 50%, 75%, max 확인 가능
# 데이터 분포가 정상적인지 빠르게 점검하는 용도

# =========================================================
# 3. 시간 feature
# =========================================================

df["year"] = df["mkTm"].dt.year
# 관측 시각에서 연도 추출
# 예: 2026-03-15 08:20:00 -> 2026

df["month"] = df["mkTm"].dt.month
# 월 추출
# 계절성, 월별 이용 패턴 차이를 반영할 수 있음

df["day"] = df["mkTm"].dt.day
# 일(day of month) 추출
# 월초/월말 패턴 차이가 있을 가능성을 반영

df["hour"] = df["mkTm"].dt.hour
# 시(hour) 추출
# 버스 좌석 수는 시간대 영향이 매우 크므로 핵심 feature

df["minute"] = df["mkTm"].dt.minute
# 분(minute) 추출
# 같은 시간대 안에서도 분 단위 변화 반영 가능

df["dayofweek"] = df["mkTm"].dt.dayofweek
# 요일 추출
# 월=0, 화=1, ..., 일=6
# 평일/주말, 요일별 출퇴근 패턴 차이를 반영 가능

df["date"] = df["mkTm"].dt.date
# 날짜만 따로 추출
# 이후 train/valid/test를 날짜 기준으로 나눌 때 사용

df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
# 주말 여부 이진 변수 생성
# 토요일(5), 일요일(6)이면 1 / 아니면 0
# 평일과 주말의 수요 차이를 단순하게 반영

df["is_peak"] = (
    ((df["hour"] >= 7) & (df["hour"] <= 9)) |
    ((df["hour"] >= 17) & (df["hour"] <= 19))
).astype(int)
# 출퇴근 혼잡 시간대 여부 이진 변수
# 오전 7~9시 또는 오후 17~19시면 1, 아니면 0
# 버스 좌석 부족이 심해지는 대표 시간대를 직접 표시

# 출퇴근 세분화
df["peak_level"] = 0
# 기본값 0: 출퇴근 시간대 아님

df.loc[(df["hour"] >= 7) & (df["hour"] <= 8), "peak_level"] = 1
# 오전 출근 피크 초반 (7~8시)

df.loc[df["hour"] == 9, "peak_level"] = 2
# 오전 출근 피크 후반 (9시)

df.loc[(df["hour"] >= 17) & (df["hour"] <= 18), "peak_level"] = 3
# 오후 퇴근 피크 초반 (17~18시)

df.loc[df["hour"] == 19, "peak_level"] = 4
# 오후 퇴근 피크 후반 (19시)

# 주기 인코딩
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
# 월(month)을 sin/cos 형태로 변환
# 12월과 1월처럼 "숫자는 멀지만 실제로는 가까운" 순환 구조를 반영

df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
# 일(day of month)을 sin/cos로 변환
# 월초와 월말도 순환적인 구조로 표현하기 위함

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
# 시간(hour)을 sin/cos로 변환
# 23시와 0시는 숫자 차이는 크지만 실제로는 붙어 있으므로
# 순환 구조를 모델이 더 자연스럽게 이해하게 도와줌

df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
# 분(minute)도 순환형으로 변환
# 59분과 0분의 거리 문제를 완화

df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
# 요일(dayofweek)도 sin/cos 변환
# 일요일(6)과 월요일(0)처럼 순환적 관계 반영

# 보조 key
df["hour_group"] = (df["hour"] // 2).astype(int)
# 시간을 2시간 단위 그룹으로 묶음
# 예: 0~1시=0, 2~3시=1, ..., 22~23시=11
# 너무 세밀한 시간 대신 조금 더 안정적인 시간대 패턴을 보기 위한 그룹

df["minute_group"] = (df["minute"] // 10).astype(int)
# 분을 10분 단위 그룹으로 묶음
# 예: 00~09분=0, 10~19분=1, ..., 50~59분=5
# minute를 그대로 쓰는 것보다 노이즈를 줄일 수 있음

df["hour_weekday_key"] = df["hour"] * 10 + df["dayofweek"]
# 시간과 요일을 조합한 보조 key 생성
# 예: 8시 월요일 -> 80, 8시 화요일 -> 81
# "같은 시간이라도 요일마다 다르다"는 패턴을 한 컬럼에 담기 위한 조합형 변수

# =========================================================
# 4. 혼잡 보정용 라벨
# =========================================================

df["is_low_seat"] = (df["remaining_seat"] <= LOW_SEAT_THRESHOLD).astype(int)
# 잔여 좌석 수가 특정 기준 이하인지 여부를 이진 변수로 생성
#
# 조건:
# remaining_seat <= LOW_SEAT_THRESHOLD (현재 10)
#
# 결과:
# - 좌석 ≤ 10 → 1 (혼잡 상태)
# - 좌석 > 10 → 0 (여유 있음)
#
# 즉, "좌석 부족 여부"를 나타내는 라벨 생성

# =========================================================
# 5. 인코딩
# =========================================================

route_le = LabelEncoder()
stid_le = LabelEncoder()
arsid_le = LabelEncoder()
# 각각 노선, 정류장ID, 정류소번호를 숫자로 변환하기 위한 인코더 생성

df["route_enc"] = route_le.fit_transform(df["busRouteId"])
# busRouteId (문자열) → 숫자 인덱스로 변환
# 예: "100100118" → 0, "100100119" → 1 ...

df["stid_enc"] = stid_le.fit_transform(df["stId"])
# 정류장 ID 인코딩

df["arsid_enc"] = arsid_le.fit_transform(df["arsId"])
# 정류소 번호 인코딩

# =========================================================
# 6. 날짜 기준 분할
# =========================================================

df = df.sort_values("mkTm").reset_index(drop=True)
# 관측 시각(mkTm) 기준으로 전체 데이터를 시간순 정렬
# 시계열 데이터는 섞으면 안 되기 때문에 먼저 시간 순서를 맞춰줌

unique_dates = sorted(df["date"].unique())
# 데이터에 포함된 날짜만 중복 없이 추출한 뒤 오름차순 정렬
# 예: [2026-03-09, 2026-03-10, 2026-03-11, ...]

print("사용 날짜들:", unique_dates)
print("총 사용 날짜 수:", len(unique_dates))
# 실제 학습에 사용할 날짜 목록과 총 일수 확인

if len(unique_dates) < 10:
    raise ValueError("최소 10일 이상은 있어야 안정적으로 분할 가능합니다.")
# 날짜 수가 너무 적으면 train/valid/test 분할이 불안정해지므로 예외 처리
# 최소 10일은 있어야 각 구간이 어느 정도 의미 있게 나뉨

n_dates = len(unique_dates)
# 전체 날짜 수 저장

train_end = int(n_dates * 0.7)
valid_end = int(n_dates * 0.85)
# 날짜 기준 분할 지점 계산
# - train: 앞쪽 70%
# - valid: 그 다음 15%
# - test : 마지막 15%

train_dates = unique_dates[:train_end]
valid_dates = unique_dates[train_end:valid_end]
test_dates = unique_dates[valid_end:]
# 실제 날짜 리스트를 train / valid / test로 나눔
# 중요한 점: 랜덤 분할이 아니라 "과거 → 미래" 순서 유지

print("train dates:", train_dates[0], "~", train_dates[-1], f"({len(train_dates)}일)")
print("valid dates:", valid_dates[0], "~", valid_dates[-1], f"({len(valid_dates)}일)")
print("test dates :", test_dates[0], "~", test_dates[-1], f"({len(test_dates)}일)")
# 각 데이터셋에 포함된 날짜 범위와 일수 확인

train_df = df[df["date"].isin(train_dates)].copy()
valid_df = df[df["date"].isin(valid_dates)].copy()
test_df = df[df["date"].isin(test_dates)].copy()
# 날짜 기준으로 실제 DataFrame 분할
# 같은 날짜의 데이터는 반드시 같은 세트에만 들어가도록 처리

# =========================================================
# 7. 패턴 통계 + 혼잡 비율 feature
# =========================================================
def add_pattern_features(train_base, target_df):
    # train_base:
    #   통계(feature)를 계산할 기준 데이터
    #   반드시 학습용(train) 데이터만 들어가야 데이터 누수를 막을 수 있음
    #
    # target_df:
    #   계산된 통계 feature를 붙일 대상 데이터
    #   train/valid/test 중 어떤 것이든 가능

    result = target_df.copy()
    # 원본 target_df를 직접 수정하지 않기 위해 복사본 생성

    global_mean = train_base["remaining_seat"].mean()
    # train 전체의 평균 잔여좌석 수
    # 세부 그룹 통계를 붙이지 못하는 경우 fallback 값으로 사용

    global_low_ratio = train_base["is_low_seat"].mean()
    # train 전체의 혼잡 비율(잔여좌석이 LOW_SEAT_THRESHOLD 이하인 비율)
    # low_ratio 계열 feature의 결측 대체값으로 사용

    # 1) 노선 평균
    route_stat = (
        train_base.groupby("busRouteId")
        .agg(
            route_mean_seat=("remaining_seat", "mean"),
            # 노선별 평균 좌석 수
            route_std_seat=("remaining_seat", "std"),
            # 노선별 좌석 수 변동성
            route_low_ratio=("is_low_seat", "mean"),
            # 노선별 혼잡 비율
        )
        .reset_index()
    )
    result = result.merge(route_stat, on="busRouteId", how="left")
    # 같은 busRouteId 기준으로 노선 통계 붙이기

    # 2) 노선 + 정류장
    route_stop_stat = (
        train_base.groupby(["busRouteId", "stId"])
        .agg(
            route_stop_mean_seat=("remaining_seat", "mean"),
            # 특정 노선의 특정 정류장에서 평균적으로 좌석이 얼마나 남는지
            route_stop_std_seat=("remaining_seat", "std"),
            # 해당 구간의 좌석 수 변동성
            route_stop_low_ratio=("is_low_seat", "mean"),
            # 해당 노선-정류장 조합의 혼잡 비율
        )
        .reset_index()
    )
    result = result.merge(route_stop_stat, on=["busRouteId", "stId"], how="left")
    # 노선별 전체 평균보다 더 세밀하게 "그 노선의 그 정류장" 패턴을 반영

    # 3) 노선 + 정류장 + 요일 + 시간그룹
    route_stop_time_stat = (
        train_base.groupby(["busRouteId", "stId", "dayofweek", "hour_group"])
        .agg(
            route_stop_time_mean_seat=("remaining_seat", "mean"),
            # 특정 노선-정류장-요일-시간대의 평균 좌석 수
            route_stop_time_std_seat=("remaining_seat", "std"),
            # 해당 조합의 좌석 변동성
            route_stop_time_low_ratio=("is_low_seat", "mean"),
            # 해당 조합의 혼잡 비율
        )
        .reset_index()
    )
    result = result.merge(
        route_stop_time_stat,
        on=["busRouteId", "stId", "dayofweek", "hour_group"],
        how="left"
    )
    # 가장 세밀한 패턴 중 하나
    # 예: "어느 노선의 어느 정류장이 월요일 오전 8~9시대에 보통 얼마나 혼잡한가"

    # 4) 노선 + staOrd
    route_staord_stat = (
        train_base.groupby(["busRouteId", "staOrd"])
        .agg(
            route_staord_mean_seat=("remaining_seat", "mean"),
            # 노선 내 정류장 순번 위치별 평균 좌석 수
            route_staord_std_seat=("remaining_seat", "std"),
            # 위치별 변동성
            route_staord_low_ratio=("is_low_seat", "mean"),
            # 위치별 혼잡 비율
        )
        .reset_index()
    )
    result = result.merge(
        route_staord_stat,
        on=["busRouteId", "staOrd"],
        how="left"
    )
    # 특정 정류장 ID가 아니더라도
    # "노선 초반/중반/후반" 같은 운행 흐름상 위치 정보를 반영할 수 있음

    # 5) 노선 + 요일 + 시간
    route_time_stat = (
        train_base.groupby(["busRouteId", "dayofweek", "hour"])
        .agg(
            route_time_mean_seat=("remaining_seat", "mean"),
            # 노선의 요일/시간대별 평균 좌석 수
            route_time_std_seat=("remaining_seat", "std"),
            # 노선의 요일/시간대별 변동성
            route_time_low_ratio=("is_low_seat", "mean"),
            # 노선의 요일/시간대별 혼잡 비율
        )
        .reset_index()
    )
    result = result.merge(
        route_time_stat,
        on=["busRouteId", "dayofweek", "hour"],
        how="left"
    )
    # 정류장 단위보다 조금 덜 세밀하지만,
    # 노선 자체의 시간대별 흐름을 반영하는 데 유용

    mean_cols = [
        "route_mean_seat",
        "route_stop_mean_seat",
        "route_stop_time_mean_seat",
        "route_staord_mean_seat",
        "route_time_mean_seat",
    ]
    # 평균 좌석 계열 컬럼 목록

    std_cols = [
        "route_std_seat",
        "route_stop_std_seat",
        "route_stop_time_std_seat",
        "route_staord_std_seat",
        "route_time_std_seat",
    ]
    # 표준편차(변동성) 계열 컬럼 목록

    low_ratio_cols = [
        "route_low_ratio",
        "route_stop_low_ratio",
        "route_stop_time_low_ratio",
        "route_staord_low_ratio",
        "route_time_low_ratio",
    ]
    # 혼잡 비율 계열 컬럼 목록

    for col in mean_cols:
        result[col] = result[col].fillna(global_mean)
    # 세부 그룹 통계가 없는 경우 전체 평균 좌석 수로 대체
    # 예: valid/test에 train에서 한 번도 못 본 조합이 등장한 경우

    for col in std_cols:
        result[col] = result[col].fillna(0)
    # 표준편차가 계산되지 않으면 0으로 대체
    # 보통 표본이 1개뿐이거나 해당 그룹이 존재하지 않는 경우

    for col in low_ratio_cols:
        result[col] = result[col].fillna(global_low_ratio)
    # 혼잡 비율 통계가 없으면 전체 train의 평균 혼잡 비율로 대체

    return result
    # 통계 feature가 추가된 DataFrame 반환

train_df = add_pattern_features(train_df, train_df)
# train 데이터 자체에도 train 기준 통계 feature 추가

valid_df = add_pattern_features(train_df, valid_df)
# valid 데이터는 오직 train 통계만 사용해서 feature 생성
# valid 정보를 기준으로 통계 내지 않기 때문에 데이터 누수 방지 가능

test_df = add_pattern_features(train_df, test_df)
# test도 동일하게 train 통계만 사용
# 실제 서비스 상황과 가장 유사한 방식

# =========================================================
# 8. 입력 feature 정의
# =========================================================
FEATURE_COLS = [
    # 인코딩
    "route_enc", "stid_enc",
    # 노선과 정류장 ID를 숫자로 인코딩한 값
    # 모델이 어떤 노선/정류장인지 구분할 수 있게 해줌

    # 시간
    "year", "month", "day",
    "hour", "minute", "dayofweek",
    "is_weekend", "is_peak", "peak_level",
    # 기본 시간 정보 + 주말 여부 + 출퇴근 여부 + 출퇴근 세분화 정보

    # 주기 인코딩
    "month_sin", "month_cos",
    "day_sin", "day_cos",
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dow_sin", "dow_cos",
    # 월/일/시/분/요일의 순환적 성격을 반영한 sin/cos 인코딩

    # 위치/운행
    "staOrd",
    "hour_group",
    "minute_group",
    "hour_weekday_key",
    # 노선 내 정류장 순번, 시간 그룹, 시간-요일 조합 key

    # 평균 패턴
    "route_mean_seat", "route_std_seat",
    "route_stop_mean_seat", "route_stop_std_seat",
    "route_stop_time_mean_seat", "route_stop_time_std_seat",
    "route_staord_mean_seat", "route_staord_std_seat",
    "route_time_mean_seat", "route_time_std_seat",
    # train 기준으로 만든 통계형 패턴 feature
    # 그룹별 평균 좌석 수와 좌석 변동성 반영

    # 혼잡 보정
    "route_low_ratio",
    "route_stop_low_ratio",
    "route_stop_time_low_ratio",
    "route_staord_low_ratio",
    "route_time_low_ratio",
    # 그룹별 혼잡 비율 feature
    # 단순 평균뿐 아니라 "얼마나 자주 좌석 부족이 발생하는지" 반영
]

TARGET_COL = "remaining_seat"
# 예측 대상(target)은 현재 시점의 잔여 좌석 수

for df_part in [train_df, valid_df, test_df]:
    df_part["exps1"] = df_part["exps1"].fillna(train_df["exps1"].median())
    # exps1 결측치는 train 데이터의 중앙값으로 대체
    # 다만 현재 FEATURE_COLS에는 exps1이 포함되어 있지 않아서
    # 이 코드 기준으로는 실제 모델 입력에는 사용되지 않음

    df_part["full_flag"] = df_part["full_flag"].fillna(0)
    # full_flag 결측치는 0으로 대체
    # 이것도 현재 FEATURE_COLS에는 포함되어 있지 않아서
    # 지금 회귀 모델 입력에는 직접 사용되지 않음

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET_COL]
# train 입력(X)과 정답(y) 분리

X_valid = valid_df[FEATURE_COLS]
y_valid = valid_df[TARGET_COL]
# valid 입력(X)과 정답(y) 분리

X_test = test_df[FEATURE_COLS]
y_test = test_df[TARGET_COL]
# test 입력(X)과 정답(y) 분리

print("train:", X_train.shape, y_train.shape)
print("valid:", X_valid.shape, y_valid.shape)
print("test :", X_test.shape, y_test.shape)
# 각 데이터셋의 행 수와 feature 수 확인
# 예: X_train.shape = (샘플 수, feature 수)

# =========================================================
# 9. 모델 학습
# =========================================================
lgbm_reg = LGBMRegressor(
    objective="regression",
    # 회귀 문제로 설정
    # 현재 목표는 좌석 수(remaining_seat) 자체를 예측하는 것이므로 regression 사용

    n_estimators=800,
    # 생성할 트리 개수
    # boosting을 800번 반복하며 점진적으로 오차를 줄여감

    learning_rate=0.05,
    # 한 번 학습할 때 반영하는 정도(학습 속도)
    # 작을수록 천천히 학습하지만 보통 더 안정적

    max_depth=10,
    # 각 트리의 최대 깊이
    # 너무 깊으면 과적합, 너무 얕으면 복잡한 패턴을 못 잡을 수 있음

    num_leaves=127,
    # 하나의 트리가 가질 수 있는 최대 리프 노드 수
    # LightGBM에서 모델 복잡도를 조절하는 핵심 하이퍼파라미터 중 하나

    min_child_samples=10,
    # 리프 노드가 되기 위한 최소 샘플 수
    # 너무 적은 샘플에 맞춰 과적합되는 것을 방지

    subsample=0.8,
    # 각 트리 학습 시 전체 데이터의 80%만 샘플링해서 사용
    # 과적합 완화와 일반화 성능 향상에 도움

    colsample_bytree=0.8,
    # 각 트리 학습 시 전체 feature 중 80%만 사용
    # 특정 feature에 과도하게 의존하는 것을 줄이는 역할

    reg_alpha=0.1,
    # L1 정규화 강도
    # 복잡한 모델을 억제해 과적합을 완화

    reg_lambda=0.1,
    # L2 정규화 강도
    # 가중치가 지나치게 커지는 것을 방지하여 안정화

    random_state=42,
    # 실험 재현성을 위한 시드 고정

    n_jobs=-1
    # 사용 가능한 CPU 코어를 모두 사용하여 학습 속도 향상
)

lgbm_reg.fit(
    X_train,
    y_train,
    # 학습 데이터

    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    # 학습 중 성능을 확인할 평가 데이터셋
    # train 성능과 valid 성능을 함께 모니터링할 수 있음

    eval_metric="l1"
    # 평가 지표로 MAE(L1 loss) 사용
    # 예측 좌석 수와 실제 좌석 수의 절대 오차 기준으로 성능 확인
)

print("LightGBM 회귀 모델 학습 완료")
# 모델 학습이 끝났는지 확인용 출력

# =========================================================
# 10. 평가 함수
# =========================================================
def evaluate_regression(model, X, y, name="dataset"):
    pred = model.predict(X)
    # 입력 X에 대해 모델이 예측한 잔여 좌석 수

    # 평가용 clip
    pred = np.clip(pred, 0, MAX_SEAT)
    # 예측값을 0 ~ MAX_SEAT 범위로 제한
    # 실제 좌석 수는 음수도, 최대 좌석 수 초과도 불가능하기 때문

    mae = mean_absolute_error(y, pred)
    # MAE (평균 절대 오차)
    # 실제값과 예측값 차이의 절대값 평균

    rmse = np.sqrt(mean_squared_error(y, pred))
    # RMSE (평균 제곱근 오차)
    # 큰 오차에 더 큰 패널티를 주는 지표

    r2 = r2_score(y, pred)
    # R² (설명력)
    # 모델이 데이터 변동성을 얼마나 잘 설명하는지

    print(f"\n[{name}]")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    # 핵심 성능 지표 출력

    print("\n실제값 통계:")
    print(pd.Series(y).describe())
    # 실제 좌석 수 분포 (평균, 최소, 최대 등)

    print("\n예측값 통계:")
    print(pd.Series(pred).describe())
    # 예측값 분포
    # 실제값과 비교해서 편향 여부 확인 가능

    return pred
    # 예측값 반환 (후속 분석이나 시각화에 활용 가능)

valid_pred = evaluate_regression(lgbm_reg, X_valid, y_valid, "VALID")
# validation 데이터 성능 평가

test_pred = evaluate_regression(lgbm_reg, X_test, y_test, "TEST")
# test 데이터 성능 평가 (최종 성능)

# =========================================================
# 11. 출퇴근 / 비출퇴근 성능
# =========================================================
def evaluate_regression_by_peak(model, df_eval, name="TEST"):
    peak_df = df_eval[df_eval["is_peak"] == 1].copy()
    # 출퇴근 시간대 데이터만 추출
    # is_peak=1 이면 오전 7~9시 또는 오후 17~19시

    non_peak_df = df_eval[df_eval["is_peak"] == 0].copy()
    # 비출퇴근 시간대 데이터만 추출

    print(f"\n===== {name} (출퇴근 vs 비출퇴근 성능) =====")
    # 현재 평가 중인 데이터셋(VALID 또는 TEST)에 대해
    # 출퇴근/비출퇴근 성능 비교 구간 시작 표시

    if len(peak_df) > 0:
        print("\n[출퇴근 시간대 (is_peak=1)]")
        evaluate_regression(model, peak_df[FEATURE_COLS], peak_df[TARGET_COL], "PEAK")
        # 출퇴근 시간대 subset에 대해서만 회귀 성능 평가
        # MAE, RMSE, R², 실제값/예측값 통계 출력

    if len(non_peak_df) > 0:
        print("\n[비출퇴근 시간대 (is_peak=0)]")
        evaluate_regression(model, non_peak_df[FEATURE_COLS], non_peak_df[TARGET_COL], "NON-PEAK")
        # 비출퇴근 시간대 subset에 대해서만 회귀 성능 평가

evaluate_regression_by_peak(lgbm_reg, valid_df, "VALID")
# validation 데이터에서 출퇴근 / 비출퇴근 성능 비교

evaluate_regression_by_peak(lgbm_reg, test_df, "TEST")
# test 데이터에서 출퇴근 / 비출퇴근 성능 비교

# =========================================================
# 12. 참고용 혼잡도 분포 확인
# =========================================================
def seat_to_congestion(seat):
    if seat <= 5:
        return 0
    elif seat <= 15:
        return 1
    else:
        return 2
    # 좌석 수 → 혼잡도 클래스 변환 함수
    # 0: 매우 혼잡 (좌석 거의 없음)
    # 1: 보통
    # 2: 여유 있음

test_pred_rounded = np.clip(np.round(test_pred), 0, MAX_SEAT)
# 예측값을 반올림해서 정수 좌석으로 만들고
# 0 ~ MAX_SEAT 범위로 제한

test_pred_class = pd.Series(test_pred_rounded).apply(seat_to_congestion)
# 예측 좌석 수 → 혼잡도 클래스 변환

test_true_class = y_test.apply(seat_to_congestion)
# 실제 좌석 수 → 혼잡도 클래스 변환

print("\n[참고용 TEST class 분포 - 실제]")
print(test_true_class.value_counts(normalize=True).sort_index())
# 실제 데이터에서 각 혼잡도 클래스 비율

print("\n[참고용 TEST class 분포 - 예측]")
print(test_pred_class.value_counts(normalize=True).sort_index())
# 모델 예측 결과에서의 클래스 비율

# =========================================================
# 13. 모델 저장
# =========================================================
joblib.dump(lgbm_reg, "lgbm_point_seat_regressor_final.pkl")
# 학습된 LightGBM 회귀 모델 저장
# → 나중에 다시 학습하지 않고 바로 불러서 사용할 수 있음

joblib.dump(route_le, "route_label_encoder.pkl")
joblib.dump(stid_le, "stid_label_encoder.pkl")
joblib.dump(arsid_le, "arsid_label_encoder.pkl")
# LabelEncoder들도 함께 저장
# → 실제 서비스에서는 새로운 데이터를 같은 방식으로 인코딩해야 하기 때문

print("\n모델과 인코더 저장 완료")

# =========================================================
# 14. 실서비스용 예측
# =========================================================
test_pred_raw = lgbm_reg.predict(X_test)
# 모델이 예측한 raw 값 (이상치 포함 가능)

test_pred_service = np.clip(test_pred_raw, 0, MAX_SEAT)
# 실제 서비스에서 사용할 값
# 0 ~ 최대 좌석 수 범위로 제한