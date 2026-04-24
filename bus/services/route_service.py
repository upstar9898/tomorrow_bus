import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bus.models import Bus_station, Route_station

# =========================================================
# 경로 설정
# =========================================================
# bus/services/route_service.py 기준
# 프로젝트 루트 = 상위 2단계
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
ENCODER_DIR = MODEL_DIR / "encoder"
ML_MODEL_DIR = MODEL_DIR / "ml_models"

# encoder 직접 로드
def _load_encoders_local():
    return {
        "route_encoder": joblib.load(ENCODER_DIR / "route_encoder.pkl"),
        "stid_encoder": joblib.load(ENCODER_DIR / "stid_encoder.pkl"),
        "arsid_encoder": joblib.load(ENCODER_DIR / "arsid_encoder.pkl"),
    }

# unseen 허용 인코딩
def _transform_with_encoders_local(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    result = df.copy()

    def safe_transform(value, encoder):
        value = str(value).strip()
        classes = set(encoder.classes_)
        if value not in classes:
            return -1
        return int(encoder.transform([value])[0])

    result["route_enc"] = result["busRouteId"].astype(str).apply(
        lambda x: safe_transform(x, encoders["route_encoder"])
    )
    result["stid_enc"] = result["stId"].astype(str).apply(
        lambda x: safe_transform(x, encoders["stid_encoder"])
    )
    result["arsid_enc"] = result["arsId"].astype(str).apply(
        lambda x: safe_transform(x, encoders["arsid_encoder"])
    )

    return result

# pattern csv 직접 로드
def _load_pattern_stats_local():
    stats_dict = {
        "route_stat": pd.read_csv(
            ARTIFACT_DIR / "pattern_route_stat.csv",
            dtype={"busRouteId": str}
        ),
        "route_stop_stat": pd.read_csv(
            ARTIFACT_DIR / "pattern_route_stop_stat.csv",
            dtype={"busRouteId": str, "stId": str}
        ),
        "route_stop_time_stat": pd.read_csv(
            ARTIFACT_DIR / "pattern_route_stop_time_stat.csv",
            dtype={"busRouteId": str, "stId": str}
        ),
        "route_staord_stat": pd.read_csv(
            ARTIFACT_DIR / "pattern_route_staord_stat.csv",
            dtype={"busRouteId": str}
        ),
        "route_time_stat": pd.read_csv(
            ARTIFACT_DIR / "pattern_route_time_stat.csv",
            dtype={"busRouteId": str}
        ),
    }

    with open(ARTIFACT_DIR / "pattern_meta.json", "r", encoding="utf-8") as f:
        pattern_meta = json.load(f)

    return stats_dict, pattern_meta

# pattern feature 직접 merge
def _merge_pattern_features_local(df: pd.DataFrame, stats_dict: dict, pattern_meta: dict) -> pd.DataFrame:
    result = df.copy()

    global_mean = float(pattern_meta["global_mean"])
    global_low_ratio = float(pattern_meta["global_low_ratio"])

    result["minute_group"] = (result["minute"] // 10) * 10

    result = result.merge(
        stats_dict["route_stat"],
        on="busRouteId",
        how="left"
    )

    result = result.merge(
        stats_dict["route_stop_stat"],
        on=["busRouteId", "stId"],
        how="left"
    )

    result = result.merge(
        stats_dict["route_stop_time_stat"],
        on=["busRouteId", "stId", "dayofweek", "hour", "minute_group"],
        how="left"
    )

    result = result.merge(
        stats_dict["route_staord_stat"],
        on=["busRouteId", "staOrd"],
        how="left"
    )

    result = result.merge(
        stats_dict["route_time_stat"],
        on=["busRouteId", "dayofweek", "hour"],
        how="left"
    )

    fill_mean_cols = [
        "route_mean_seat",
        "route_stop_mean_seat",
        "route_stop_time_mean_seat",
        "route_staord_mean_seat",
        "route_time_mean_seat",
    ]

    fill_std_cols = [
        "route_std_seat",
        "route_stop_std_seat",
        "route_stop_time_std_seat",
        "route_staord_std_seat",
        "route_time_std_seat",
    ]

    fill_ratio_cols = [
        "route_low_ratio",
        "route_stop_low_ratio",
        "route_stop_time_low_ratio",
        "route_staord_low_ratio",
        "route_time_low_ratio",
    ]

    for col in fill_mean_cols:
        if col in result.columns:
            result[col] = result[col].fillna(global_mean)

    for col in fill_std_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    for col in fill_ratio_cols:
        if col in result.columns:
            result[col] = result[col].fillna(global_low_ratio)

    return result

# 정류소명 매핑 함수
def _load_station_name_map() -> dict:
    station_rows = Bus_station.objects.all().values("stationId", "stationName")
    return {
        str(row["stationId"]): row["stationName"]
        for row in station_rows
    }

def _load_route_station_order_from_db(route_id: str | None = None) -> pd.DataFrame:
    """
    DB의 route_station + bus_station 정보를 사용해서
    서비스2용 노선-정류소 순서표 DataFrame 생성
    컬럼:
    - busRouteId
    - stId
    - arsId
    - staOrd
    """
    qs = Route_station.objects.all().values("routeId", "stationId", "staOrd")

    if route_id is not None:
        qs = qs.filter(routeId=str(route_id))

    route_df = pd.DataFrame(list(qs))
    if route_df.empty:
        return pd.DataFrame(columns=["busRouteId", "stId", "arsId", "staOrd"])

    route_df = route_df.rename(
        columns={
            "routeId": "busRouteId",
            "stationId": "stId",
        }
    )

    route_df["busRouteId"] = route_df["busRouteId"].astype(str)
    route_df["stId"] = route_df["stId"].astype(str)
    route_df["staOrd"] = pd.to_numeric(route_df["staOrd"], errors="coerce").astype("Int64")

    # Bus_station에서 arsId 매핑
    station_rows = Bus_station.objects.all().values("stationId", "arsId")
    station_df = pd.DataFrame(list(station_rows))

    if station_df.empty:
        route_df["arsId"] = ""
    else:
        station_df = station_df.rename(columns={"stationId": "stId"})
        station_df["stId"] = station_df["stId"].astype(str)
        station_df["arsId"] = station_df["arsId"].astype(str)

        route_df = route_df.merge(station_df, on="stId", how="left")

    route_df["arsId"] = route_df["arsId"].fillna("").astype(str)
    route_df = route_df.sort_values(["busRouteId", "staOrd"]).reset_index(drop=True)

    return route_df

# MAX_SEAT은 훈련 코드 기준 45로 사용
MAX_SEAT = 45


# =========================================================
# 한국 공휴일 계산 (holidays 라이브러리 있으면 사용)
# 없으면 주말 외 공휴일은 False 처리
# =========================================================
try:
    import holidays  # pip install holidays
except ImportError:
    holidays = None


# =========================================================
# 기본 유틸
# =========================================================
def _safe_read_json(path: Path, default: dict | list | None = None):
    if not path.exists():
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_datetime(dt_value) -> pd.Timestamp:
    dt = pd.to_datetime(dt_value, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"target_datetime 파싱 실패: {dt_value}")
    return dt


def _is_holiday_kr(dt: pd.Timestamp) -> int:
    """
    한국 공휴일 여부 반환
    - holidays 라이브러리 설치되어 있으면 공휴일 + 주말 반영
    - 없으면 주말만 반영
    """
    if dt.weekday() >= 5:
        return 1

    if holidays is None:
        return 0

    kr_holidays = holidays.KR(years=[dt.year])
    return 1 if dt.date() in kr_holidays else 0


def _is_peak(dt: pd.Timestamp, is_holiday: int) -> int:
    """
    출퇴근 시간대:
    - 공휴일이 아니고
    - 07~09시 또는 17~19시이면 1
    """
    if is_holiday == 1:
        return 0

    hour = dt.hour
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return 1
    return 0


def _make_time_features(df: pd.DataFrame, dt_col: str = "mkTm") -> pd.DataFrame:
    """
    훈련 데이터와 맞추기 위한 시간 파생변수 생성
    """
    result = df.copy()
    result[dt_col] = pd.to_datetime(result[dt_col], errors="coerce")

    result["year"] = result[dt_col].dt.year
    result["month"] = result[dt_col].dt.month
    result["day"] = result[dt_col].dt.day
    result["hour"] = result[dt_col].dt.hour
    result["minute"] = result[dt_col].dt.minute
    result["dayofweek"] = result[dt_col].dt.dayofweek
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)

    # is_holiday / is_peak는 행별로 다시 계산
    result["is_holiday"] = result[dt_col].apply(_is_holiday_kr).astype(int)
    result["is_peak"] = result.apply(
        lambda row: _is_peak(row[dt_col], int(row["is_holiday"])),
        axis=1
    ).astype(int)

    # 주기형 인코딩
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    # day는 31일 기준
    result["day_sin"] = np.sin(2 * np.pi * result["day"] / 31)
    result["day_cos"] = np.cos(2 * np.pi * result["day"] / 31)

    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)

    result["minute_sin"] = np.sin(2 * np.pi * result["minute"] / 60)
    result["minute_cos"] = np.cos(2 * np.pi * result["minute"] / 60)

    result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)

    return result

def _normalize_peak_thresholds(thresholds):
    """
    thresholds.json 에 저장된 peak threshold 형식을
    서비스 코드에서 항상 [t0, t1, t2] 리스트로 맞춘다.

    허용 형식:
    - [0.15, 0.3, 0.4]
    - {"0": 0.15, "1": 0.3, "2": 0.4}
    - {0: 0.15, 1: 0.3, 2: 0.4}
    """
    if isinstance(thresholds, list):
        return [float(x) for x in thresholds]

    if isinstance(thresholds, dict):
        return [
            float(thresholds.get(0, thresholds.get("0", 0.25))),
            float(thresholds.get(1, thresholds.get("1", 0.25))),
            float(thresholds.get(2, thresholds.get("2", 0.25))),
        ]

    return [0.25, 0.25, 0.25]

def _predict_peak_congestion_with_thresholds(
    proba: np.ndarray,
    thresholds
) -> np.ndarray:
    """
    출퇴근 시간대 4클래스 threshold 적용
    """
    thresholds = _normalize_peak_thresholds(thresholds)

    preds = []

    for row in proba:
        p0, p1, p2, p3 = row

        if p0 >= thresholds[0]:
            preds.append(0)
        elif p1 >= thresholds[1]:
            preds.append(1)
        elif p2 >= thresholds[2]:
            preds.append(2)
        else:
            preds.append(3)

    return np.array(preds)

def _fallback_congestion_from_seat(remaining_seat: int) -> tuple[int, str]:
    """
    출퇴근 시간대가 아닐 때 사용할 좌석수 기반 혼잡도 추정 규칙
    """
    if remaining_seat <= 5:
        return 3, "매우 혼잡"
    elif remaining_seat <= 15:
        return 2, "혼잡"
    elif remaining_seat <= 30:
        return 1, "보통"
    else:
        return 0, "여유"


def _peak_class_to_label(cls_value: int) -> str:
    label_map = {
        0: "여유",
        1: "보통",
        2: "혼잡",
        3: "매우 혼잡",
    }
    return label_map.get(int(cls_value), "알 수 없음")


def _match_station_row(route_station_order: pd.DataFrame, route_id: str, station_id: str) -> pd.Series:
    """
    station_id는 stId 또는 arsId 둘 다 받을 수 있게 처리
    """
    route_df = route_station_order[route_station_order["busRouteId"] == str(route_id)].copy()

    if route_df.empty:
        raise ValueError(f"해당 노선의 정류소 순서 정보가 없습니다. route_id={route_id}")

    # stId 우선
    matched = route_df[route_df["stId"].astype(str) == str(station_id)]
    if len(matched) > 0:
        return matched.iloc[0]

    # arsId fallback
    matched = route_df[route_df["arsId"].astype(str) == str(station_id)]
    if len(matched) > 0:
        return matched.iloc[0]

    raise ValueError(
        f"해당 노선에서 station_id와 일치하는 정류소를 찾지 못했습니다. "
        f"route_id={route_id}, station_id={station_id}"
    )


# =========================================================
# 아티팩트 로딩
# =========================================================
@lru_cache(maxsize=1)
def _load_artifacts():
    # 모델
    reg_model = joblib.load(ML_MODEL_DIR / "reg.pkl")
    peak_model = joblib.load(ML_MODEL_DIR / "peak_congestion_cls.pkl")
    full_model = joblib.load(ML_MODEL_DIR / "full_cls.pkl")

    # feature / threshold
    feature_cols = _safe_read_json(ENCODER_DIR / "feature_cols.json", default=[])
    thresholds = _safe_read_json(ENCODER_DIR / "thresholds.json", default={})

    peak_thresholds_raw = thresholds.get("peak_congestion_thresholds", [0.25, 0.25, 0.25])

    if isinstance(peak_thresholds_raw, dict):
        peak_thresholds = [
            float(peak_thresholds_raw.get(0, peak_thresholds_raw.get("0", 0.25))),
            float(peak_thresholds_raw.get(1, peak_thresholds_raw.get("1", 0.25))),
            float(peak_thresholds_raw.get(2, peak_thresholds_raw.get("2", 0.25))),
        ]
    else:
        peak_thresholds = list(peak_thresholds_raw)
    full_binary_threshold = thresholds.get("full_binary_threshold", 0.5)

    # encoder
    encoders = _load_encoders_local()

    # pattern stats
    stats_dict, pattern_meta = _load_pattern_stats_local()

    route_station_travel_time = pd.read_csv(
        ARTIFACT_DIR / "route_station_travel_time.csv",
        dtype={"busRouteId": str}
    )
    route_station_travel_time["from_staOrd"] = pd.to_numeric(
        route_station_travel_time["from_staOrd"], errors="coerce"
    ).astype("Int64")
    route_station_travel_time["to_staOrd"] = pd.to_numeric(
        route_station_travel_time["to_staOrd"], errors="coerce"
    ).astype("Int64")
    route_station_travel_time["avg_travel_sec"] = pd.to_numeric(
        route_station_travel_time["avg_travel_sec"], errors="coerce"
    )

    print("peak_thresholds raw =", peak_thresholds)
    print("peak_thresholds normalized =", _normalize_peak_thresholds(peak_thresholds))

    return {
        "reg_model": reg_model,
        "peak_model": peak_model,
        "full_model": full_model,
        "feature_cols": feature_cols,
        "peak_thresholds": peak_thresholds,
        "full_binary_threshold": full_binary_threshold,
        "encoders": encoders,
        "stats_dict": stats_dict,
        "pattern_meta": pattern_meta,
        "route_station_travel_time": route_station_travel_time,
    }


# =========================================================
# 서비스2 ETA 계산
# =========================================================
def _get_segment_seconds(
    travel_df: pd.DataFrame,
    route_id: str,
    from_staOrd: int,
    to_staOrd: int,
    time_band: str,
) -> float:
    """
    구간 이동시간 조회
    우선순위:
    1) 같은 노선 + 같은 구간 + 같은 time_band
    2) 같은 노선 + 같은 구간 + normal
    3) 같은 노선 전체 평균
    4) 전체 평균
    """
    route_id = str(route_id)

    exact = travel_df[
        (travel_df["busRouteId"] == route_id)
        & (travel_df["from_staOrd"] == from_staOrd)
        & (travel_df["to_staOrd"] == to_staOrd)
        & (travel_df["time_band"] == time_band)
    ]
    if len(exact) > 0:
        return float(exact.iloc[0]["avg_travel_sec"])

    fallback_normal = travel_df[
        (travel_df["busRouteId"] == route_id)
        & (travel_df["from_staOrd"] == from_staOrd)
        & (travel_df["to_staOrd"] == to_staOrd)
        & (travel_df["time_band"] == "normal")
    ]
    if len(fallback_normal) > 0:
        return float(fallback_normal.iloc[0]["avg_travel_sec"])

    route_avg = travel_df[travel_df["busRouteId"] == route_id]["avg_travel_sec"].mean()
    if not np.isnan(route_avg):
        return float(route_avg)

    global_avg = travel_df["avg_travel_sec"].mean()
    if not np.isnan(global_avg):
        return float(global_avg)

    return 60.0  # 최후 fallback


def _build_route_eta_table(
    route_id: str,
    station_id: str,
    target_datetime: str,
    route_station_order: pd.DataFrame,
    route_station_travel_time: pd.DataFrame,
) -> pd.DataFrame:
    """
    기준 정류소 / 기준 시각을 중심으로
    전체 노선 정류소의 예상 시각을 생성
    """
    target_dt = _to_datetime(target_datetime)
    target_holiday = _is_holiday_kr(target_dt)
    target_peak = _is_peak(target_dt, target_holiday)
    target_time_band = "peak" if (target_holiday == 0 and target_peak == 1) else "normal"

    route_df = (
        route_station_order[route_station_order["busRouteId"] == str(route_id)]
        .copy()
        .sort_values("staOrd")
        .reset_index(drop=True)
    )

    if route_df.empty:
        raise ValueError(f"DB route_station에서 노선 정보가 없습니다. route_id={route_id}")

    base_station_row = _match_station_row(route_df, route_id, station_id)
    base_staOrd = int(base_station_row["staOrd"])

    eta_map = {base_staOrd: target_dt}

    sta_list = route_df["staOrd"].dropna().astype(int).tolist()

    # 기준 정류소 index 찾기
    base_idx_list = route_df.index[route_df["staOrd"].astype(int) == base_staOrd].tolist()
    if not base_idx_list:
        raise ValueError(f"기준 정류소의 staOrd를 route_df에서 찾지 못했습니다. route_id={route_id}, station_id={station_id}")

    base_idx = base_idx_list[0]

    # 기준 정류소 이후 정류소
    for i in range(base_idx + 1, len(route_df)):
        prev_sta = int(route_df.loc[i - 1, "staOrd"])
        curr_sta = int(route_df.loc[i, "staOrd"])
        prev_dt = eta_map.get(prev_sta)

        if prev_dt is None:
            continue

        seg_sec = _get_segment_seconds(
            travel_df=route_station_travel_time,
            route_id=str(route_id),
            from_staOrd=prev_sta,
            to_staOrd=curr_sta,
            time_band=target_time_band,
        )
        eta_map[curr_sta] = prev_dt + pd.to_timedelta(seg_sec, unit="s")

    # 기준 정류소 이전 정류소
    for i in range(base_idx - 1, -1, -1):
        curr_sta = int(route_df.loc[i, "staOrd"])
        next_sta = int(route_df.loc[i + 1, "staOrd"])
        next_dt = eta_map.get(next_sta)

        if next_dt is None:
            continue

        seg_sec = _get_segment_seconds(
            travel_df=route_station_travel_time,
            route_id=str(route_id),
            from_staOrd=curr_sta,
            to_staOrd=next_sta,
            time_band=target_time_band,
        )
        eta_map[curr_sta] = next_dt - pd.to_timedelta(seg_sec, unit="s")

    route_df["eta_dt"] = route_df["staOrd"].astype(int).map(eta_map)
    route_df["eta_dt"] = pd.to_datetime(route_df["eta_dt"], errors="coerce")

    return route_df


# =========================================================
# 모델 입력용 feature dataframe 생성
# =========================================================
def _build_feature_dataframe(route_eta_df: pd.DataFrame) -> pd.DataFrame:
    """
    서비스 입력을 모델 입력 형태로 변환
    """
    df = route_eta_df.copy()

    df["mkTm"] = pd.to_datetime(df["eta_dt"], errors="coerce")
    df["busRouteId"] = df["busRouteId"].astype(str)
    df["stId"] = df["stId"].astype(str)
    df["arsId"] = df["arsId"].astype(str)
    df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")

    # travel_time은 "직전 정류소 -> 현재 정류소" 이동시간 개념이라
    # route_eta_df에 별도 값이 없으므로 기본값 0으로 두고,
    # feature_cols에 필요 시 나중에 넣도록 처리
    if "travel_time" not in df.columns:
        df["travel_time"] = 0.0

    # 서비스 시점에는 아래 컬럼은 보통 없으므로 placeholder
    placeholder_defaults = {
        "vehId1": "0",
        "exps1": 0,
        "arrmsg1": "",
        "remaining_seat": 0,
        "full_flag": 0,
        "precipitation": 0.0,
        "fog": 0.0,
        "temperature": 0.0,
        "rainfall": 0.0,
        "rainfall_missing": 1,
        "source_file": "service2_generated",
    }

    for col, default_value in placeholder_defaults.items():
        if col not in df.columns:
            df[col] = default_value

    df = _make_time_features(df, dt_col="mkTm")

    return df

def _make_relative_time_label(relative_time_sec):
    if relative_time_sec is None:
        return "시간 정보 없음"

    if relative_time_sec == 0:
        return "곧 도착"

    abs_sec = abs(int(relative_time_sec))
    total_minutes = round(abs_sec / 60)

    if total_minutes < 1:
        return "곧 도착"

    hours = total_minutes // 60
    minutes = total_minutes % 60

    if hours == 0:
        text = f"{minutes}분"
    elif minutes == 0:
        text = f"{hours}시간"
    else:
        text = f"{hours}시간 {minutes}분"

    return f"{text} 후" if relative_time_sec > 0 else f"{text} 전"

# =========================================================
# 추론 실행
# =========================================================
def predict_route_service(route_id: str, station_id: str, target_datetime: str) -> list[dict]:
    """
    서비스2 메인 함수

    Parameters
    ----------
    route_id : str
        노선 ID
    station_id : str
        기준 정류소 ID (stId 또는 arsId 둘 다 허용)
    target_datetime : str
        기준 정류소 기준 시각
        예: "2026-04-22 08:10:00"

    Returns
    -------
    list[dict]
        전체 정류소 결과 리스트
    """
    artifacts = _load_artifacts()

    reg_model = artifacts["reg_model"]
    peak_model = artifacts["peak_model"]
    full_model = artifacts["full_model"]
    feature_cols = artifacts["feature_cols"]
    peak_thresholds = artifacts["peak_thresholds"]
    full_binary_threshold = artifacts["full_binary_threshold"]
    encoders = artifacts["encoders"]
    stats_dict = artifacts["stats_dict"]
    pattern_meta = artifacts["pattern_meta"]
    route_station_order = _load_route_station_order_from_db(str(route_id))
    route_station_travel_time = artifacts["route_station_travel_time"]

    # -----------------------------------------------------
    # 1) 전체 정류소 ETA 계산
    # -----------------------------------------------------
    route_eta_df = _build_route_eta_table(
        route_id=str(route_id),
        station_id=str(station_id),
        target_datetime=target_datetime,
        route_station_order=route_station_order,
        route_station_travel_time=route_station_travel_time,
    )

    # -----------------------------------------------------
    # 2) feature dataframe 생성
    # -----------------------------------------------------
    infer_df = _build_feature_dataframe(route_eta_df)

    # -----------------------------------------------------
    # 3) encoder 적용
    # -----------------------------------------------------
    infer_df = _transform_with_encoders_local(infer_df, encoders)

    # -----------------------------------------------------
    # 4) pattern feature merge
    # -----------------------------------------------------
    infer_df = _merge_pattern_features_local(infer_df, stats_dict, pattern_meta)

    # -----------------------------------------------------
    # 5) 누락된 컬럼 보정
    # -----------------------------------------------------
    for col in feature_cols:
        if col not in infer_df.columns:
            infer_df[col] = 0

    X = infer_df[feature_cols].copy()

    # -----------------------------------------------------
    # 6) 회귀 / 만차 이진분류
    # -----------------------------------------------------
    pred_remaining_seat = np.clip(reg_model.predict(X), 0, MAX_SEAT)
    pred_remaining_seat_rounded = np.rint(pred_remaining_seat).astype(int)

    pred_full_prob = full_model.predict_proba(X)[:, 1]
    pred_is_full = (pred_full_prob >= full_binary_threshold).astype(int)

    infer_df["pred_remaining_seat"] = pred_remaining_seat
    infer_df["pred_remaining_seat_rounded"] = pred_remaining_seat_rounded
    infer_df["pred_full_prob"] = pred_full_prob
    infer_df["pred_is_full"] = pred_is_full

    # -----------------------------------------------------
    # 7) 출퇴근 시간대 4클래스 혼잡도
    # -----------------------------------------------------
    infer_df["pred_congestion_class"] = None
    infer_df["pred_congestion_label"] = None
    infer_df["congestion_source"] = None

    peak_mask = (infer_df["is_holiday"] == 0) & (infer_df["is_peak"] == 1)

    if peak_mask.sum() > 0:
        peak_X = infer_df.loc[peak_mask, feature_cols]
        peak_proba = peak_model.predict_proba(peak_X)
        peak_pred = _predict_peak_congestion_with_thresholds(peak_proba, peak_thresholds)

        infer_df.loc[peak_mask, "pred_congestion_class"] = peak_pred
        infer_df.loc[peak_mask, "pred_congestion_label"] = [
            _peak_class_to_label(x) for x in peak_pred
        ]
        infer_df.loc[peak_mask, "congestion_source"] = "peak_classifier"

    # 출퇴근 외 시간대는 좌석수 기반 fallback
    non_peak_mask = ~peak_mask
    if non_peak_mask.sum() > 0:
        fallback_values = infer_df.loc[non_peak_mask, "pred_remaining_seat_rounded"].apply(
            _fallback_congestion_from_seat
        )
        infer_df.loc[non_peak_mask, "pred_congestion_class"] = fallback_values.apply(lambda x: x[0])
        infer_df.loc[non_peak_mask, "pred_congestion_label"] = fallback_values.apply(lambda x: x[1])
        infer_df.loc[non_peak_mask, "congestion_source"] = "seat_heuristic"

    # -----------------------------------------------------
    # 8) 응답 정리
    # -----------------------------------------------------
    # 기준 시각
    target_dt = _to_datetime(target_datetime)

    station_name_map = _load_station_name_map()
    infer_df["station_name"] = infer_df["stId"].astype(str).map(station_name_map)
    infer_df["station_name"] = infer_df["station_name"].fillna(infer_df["stId"].astype(str))

    result_df = infer_df[
        [
            "busRouteId",
            "stId",
            "arsId",
            "station_name",
            "staOrd",
            "mkTm",
            "is_holiday",
            "is_peak",
            "pred_remaining_seat_rounded",
            "pred_full_prob",
            "pred_is_full",
            "pred_congestion_class",
            "pred_congestion_label",
            "congestion_source",
        ]
    ].copy()

    result_df = result_df.sort_values("staOrd").reset_index(drop=True)

    response = []
    for _, row in result_df.iterrows():
        predicted_dt = pd.to_datetime(row["mkTm"]) if pd.notna(row["mkTm"]) else None

        if predicted_dt is not None:
            relative_time_sec = int((predicted_dt - target_dt).total_seconds())
            relative_time_label = _make_relative_time_label(relative_time_sec)
            predicted_arrival_time = predicted_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            relative_time_sec = None
            relative_time_label = None
            predicted_arrival_time = None

        response.append(
            {
                "route_id": str(row["busRouteId"]),
                "station_id": str(row["stId"]),
                "station_name": row.get("station_name", ""),
                "ars_id": str(row["arsId"]),
                "staOrd": int(row["staOrd"]),
                "predicted_arrival_time": predicted_arrival_time,
                "relative_time_sec": relative_time_sec,
                "relative_time_label": relative_time_label,
                "is_holiday": int(row["is_holiday"]),
                "is_peak": int(row["is_peak"]),
                "remaining_seat": int(row["pred_remaining_seat_rounded"]),
                "full_probability": round(float(row["pred_full_prob"]), 4),
                "is_full": int(row["pred_is_full"]),
                "congestion_class": int(row["pred_congestion_class"])
                if row["pred_congestion_class"] is not None else None,
                "congestion_label": row["pred_congestion_label"],
                "congestion_source": row["congestion_source"],
            }
        )

    return response