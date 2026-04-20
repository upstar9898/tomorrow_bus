import pandas as pd
from .test_ml_model_load import ml_model_1, ml_model_2

FEATURE_COLS = [
    # 인코딩
    "route_enc",
    "stid_enc",
    # 시간
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
    "is_peak",
    "peak_level",
    # 주기 인코딩
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
    # 위치/운행
    "staOrd",
    "hour_group",
    "minute_group",
    "hour_weekday_key",
    # 평균 패턴
    "route_mean_seat",
    "route_std_seat",
    "route_stop_mean_seat",
    "route_stop_std_seat",
    "route_stop_time_mean_seat",
    "route_stop_time_std_seat",
    "route_staord_mean_seat",
    "route_staord_std_seat",
    "route_time_mean_seat",
    "route_time_std_seat",
    # 혼잡 보정
    "route_low_ratio",
    "route_stop_low_ratio",
    "route_stop_time_low_ratio",
    "route_staord_low_ratio",
    "route_time_low_ratio",
]


# routeId, stationId, dateTime이 들어가있는 json을 받아서 실제 머신러닝 모델에 들어갈 52개 컬럼으로 변환하는 preprocess 작업을 수행해야 함
def preprocess(input_json):
    # 이 부분에 preprocess 작업 추가 필요
    # 평균 등의 통계를 이용한 값의 경우, 모델 자체에 저장되는 값이라 그 값을 불러올 필요가 있음
    # 아마 ml_model1(lgbm_reg로 예상)만 불러와도 될 것이라 생각되나, 모델 구조가 변경될 수 있음
    # stops가 있는 경우, 전 노선 처리로 생각하여 전 노선에 해당하는 X 값
    # 없는 경우, 한 정류장 처리로 생각하여 한 정류장에 해당하는 X 값
    # 더미
    stops = input_json.get("stops")

    if not stops:
        df = pd.DataFrame(0, index=[0], columns=[FEATURE_COLS])
    else:
        df = pd.DataFrame(0, index=range(len(stops)), columns=[FEATURE_COLS])
    return df
