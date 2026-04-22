# =========================================================
# inference_utils.py
# ---------------------------------------------------------
# 서비스 추론용 유틸
#
# [역할]
# 1. 학습된 모델 로드
# 2. encoder 로드
# 3. pattern stats 로드
# 4. 입력 데이터에 encoder / pattern feature 적용
# 5. 회귀 + 4클래스 혼잡도 + 만차 이진분류 예측
# 6. 서비스 응답용 컬럼 생성
# =========================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from utils.encoder_utils import load_label_encoders, transform_with_encoders
from utils.pattern_stats_utils import load_pattern_stats, merge_pattern_features
from utils.feature_utils import (
    MAX_SEAT,
    seat_to_congestion_4,
    congestion_label_text,
    full_binary_label_text,
)


# =========================================================
# 1. threshold 적용 함수
# =========================================================
def predict_peak_congestion_with_thresholds(proba, thresholds):
    """
    4클래스 확률값에 threshold 규칙을 적용하여 최종 클래스를 반환한다.

    Parameters
    ----------
    proba : np.ndarray
        shape = (n_samples, 4)
    thresholds : dict
        예: {0: 0.15, 1: 0.30, 2: 0.40}

    Returns
    -------
    np.ndarray
    """
    preds = []

    for row in proba:
        p0, p1, p2, p3 = row

        if p0 >= thresholds[0]:
            preds.append(0)  # 만차
        elif p1 >= thresholds[1]:
            preds.append(1)  # 혼잡
        elif p2 >= thresholds[2]:
            preds.append(2)  # 보통
        else:
            preds.append(3)  # 여유

    return np.array(preds)


# =========================================================
# 2. 모델 / 설정 로드
# =========================================================
def load_inference_artifacts(model_dir: str, artifact_dir: str):
    """
    서비스 추론에 필요한 아티팩트를 한 번에 로드한다.

    Returns
    -------
    dict
    """
    reg_model_path = os.path.join(model_dir, "reg.pkl")
    peak_model_path = os.path.join(model_dir, "peak_congestion_cls.pkl")
    full_model_path = os.path.join(model_dir, "full_cls.pkl")
    feature_cols_path = os.path.join(model_dir, "feature_cols.json")
    thresholds_path = os.path.join(model_dir, "thresholds.json")

    required_files = [
        reg_model_path,
        peak_model_path,
        full_model_path,
        feature_cols_path,
        thresholds_path,
    ]

    missing_files = [p for p in required_files if not os.path.exists(p)]
    if missing_files:
        raise FileNotFoundError(
            "서비스 추론용 파일이 없습니다.\n" + "\n".join(missing_files)
        )

    reg_model = joblib.load(reg_model_path)
    peak_model = joblib.load(peak_model_path)
    full_model = joblib.load(full_model_path)

    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds_info = json.load(f)

    encoders = load_label_encoders(model_dir)
    stats_dict, pattern_meta = load_pattern_stats(artifact_dir)

    return {
        "reg_model": reg_model,
        "peak_model": peak_model,
        "full_model": full_model,
        "feature_cols": feature_cols,
        "peak_thresholds": thresholds_info["peak_congestion_thresholds"],
        "full_binary_threshold": thresholds_info["full_binary_threshold"],
        "encoders": encoders,
        "stats_dict": stats_dict,
        "pattern_meta": pattern_meta,
    }


# =========================================================
# 3. 입력 데이터 전처리 (inference용)
# =========================================================
def prepare_inference_features(
    input_df: pd.DataFrame,
    encoders: dict,
    stats_dict: dict,
    pattern_meta: dict,
) -> pd.DataFrame:
    """
    이미 기본 feature가 생성되어 있는 입력 df에 대해
    encoder 적용 + pattern feature merge를 수행한다.

    주의:
    - 이 함수는 input_df에 최소한 아래 컬럼들이 이미 있다고 가정한다.
      busRouteId, stId, arsId, dayofweek, hour, minute_group, staOrd, ...
    - 즉 feature_utils 기반 기본 전처리/파생변수 생성 이후에 쓰는 함수다.
    """
    result = input_df.copy()

    # 서비스에서는 unseen 허용
    result = transform_with_encoders(result, encoders, strict=False)

    # 패턴 feature 붙이기
    result = merge_pattern_features(result, stats_dict, pattern_meta)

    return result


# =========================================================
# 4. 서비스 예측
# =========================================================
def predict_service(
    row_df: pd.DataFrame,
    reg_model,
    peak_model,
    full_model,
    feature_cols,
    peak_thresholds,
    full_binary_threshold,
) -> pd.DataFrame:
    """
    서비스용 최종 예측 수행

    Returns
    -------
    pd.DataFrame
        예측 결과 컬럼이 추가된 데이터프레임
    """
    result = row_df.copy()

    # 1) 회귀 예측
    pred_seat = np.clip(reg_model.predict(result[feature_cols]), 0, MAX_SEAT)
    result["pred_remaining_seat"] = np.round(pred_seat, 2)
    result["pred_remaining_seat_rounded"] = np.clip(
        np.round(pred_seat), 0, MAX_SEAT
    ).astype(int)

    # 2) 회귀 기반 혼잡도
    result["reg_based_congestion_class"] = result["pred_remaining_seat_rounded"].apply(seat_to_congestion_4)
    result["reg_based_congestion_label"] = result["reg_based_congestion_class"].apply(congestion_label_text)

    # 3) 출퇴근 시간대 전용 4클래스 분류
    peak_mask = result["is_peak"] == 1

    result["pred_peak_congestion_class"] = np.nan
    result["pred_peak_congestion_label"] = None
    result["pred_peak_congestion_prob_0"] = np.nan
    result["pred_peak_congestion_prob_1"] = np.nan
    result["pred_peak_congestion_prob_2"] = np.nan
    result["pred_peak_congestion_prob_3"] = np.nan

    if peak_mask.sum() > 0:
        peak_rows = result.loc[peak_mask, feature_cols]
        peak_proba = peak_model.predict_proba(peak_rows)
        peak_pred = predict_peak_congestion_with_thresholds(peak_proba, peak_thresholds)

        result.loc[peak_mask, "pred_peak_congestion_class"] = peak_pred
        result.loc[peak_mask, "pred_peak_congestion_label"] = [congestion_label_text(x) for x in peak_pred]

        for i in range(4):
            result.loc[peak_mask, f"pred_peak_congestion_prob_{i}"] = peak_proba[:, i]

    # 4) 만차 여부 이진 분류
    full_prob = full_model.predict_proba(result[feature_cols])[:, 1]
    result["pred_full_prob"] = full_prob
    result["pred_is_full"] = (result["pred_full_prob"] >= full_binary_threshold).astype(int)
    result["pred_is_full_label"] = result["pred_is_full"].apply(full_binary_label_text)
    result["pred_not_full_prob"] = 1 - result["pred_full_prob"]

    # 5) 최종 혼잡도
    result["final_congestion_class"] = result["reg_based_congestion_class"]
    result["final_congestion_label"] = result["reg_based_congestion_label"]

    if peak_mask.sum() > 0:
        result.loc[peak_mask, "final_congestion_class"] = result.loc[peak_mask, "pred_peak_congestion_class"]
        result.loc[peak_mask, "final_congestion_label"] = result.loc[peak_mask, "pred_peak_congestion_label"]

    full_override_mask = result["pred_full_prob"] >= full_binary_threshold
    result.loc[full_override_mask, "final_congestion_class"] = 0
    result.loc[full_override_mask, "final_congestion_label"] = "만차"

    # 6) UI용 표시 컬럼
    result["ui_expected_remaining_seat"] = result["pred_remaining_seat_rounded"].astype(str) + "석"
    result["ui_congestion_with_full_prob"] = (
        result["final_congestion_label"]
        + " (만차확률 "
        + (result["pred_full_prob"] * 100).round(1).astype(str)
        + "%)"
    )

    return result


# =========================================================
# 5. 한 번에 서비스 추론 수행
# =========================================================
def run_service_inference(
    prepared_df: pd.DataFrame,
    model_dir: str,
    artifact_dir: str,
) -> pd.DataFrame:
    """
    기본 파생변수까지 준비된 df를 받아
    encoder + pattern + 예측까지 한 번에 수행한다.

    Parameters
    ----------
    prepared_df : pd.DataFrame
        feature_utils 기반 기본 전처리/파생변수 생성이 완료된 df
    model_dir : str
    artifact_dir : str

    Returns
    -------
    pd.DataFrame
    """
    artifacts = load_inference_artifacts(model_dir, artifact_dir)

    feature_ready_df = prepare_inference_features(
        input_df=prepared_df,
        encoders=artifacts["encoders"],
        stats_dict=artifacts["stats_dict"],
        pattern_meta=artifacts["pattern_meta"],
    )

    result = predict_service(
        row_df=feature_ready_df,
        reg_model=artifacts["reg_model"],
        peak_model=artifacts["peak_model"],
        full_model=artifacts["full_model"],
        feature_cols=artifacts["feature_cols"],
        peak_thresholds=artifacts["peak_thresholds"],
        full_binary_threshold=artifacts["full_binary_threshold"],
    )

    return result