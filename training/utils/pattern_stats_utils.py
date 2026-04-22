# =========================================================
# pattern_stats_utils.py
# ---------------------------------------------------------
# 저장된 패턴 통계 CSV 로드 + target_df에 merge하는 유틸
#
# [역할]
# 1. pattern_*.csv 파일 로드
# 2. pattern_meta.json 로드
# 3. target_df에 패턴 통계 feature merge
# 4. 결측은 global_mean / global_low_ratio / 0 으로 보정
# =========================================================

import os
import json
import pandas as pd


# =========================================================
# 1. 컬럼 정의
# =========================================================
MEAN_COLS = [
    "route_mean_seat",
    "route_stop_mean_seat",
    "route_stop_time_mean_seat",
    "route_staord_mean_seat",
    "route_time_mean_seat",
]

STD_COLS = [
    "route_std_seat",
    "route_stop_std_seat",
    "route_stop_time_std_seat",
    "route_staord_std_seat",
    "route_time_std_seat",
]

LOW_RATIO_COLS = [
    "route_low_ratio",
    "route_stop_low_ratio",
    "route_stop_time_low_ratio",
    "route_staord_low_ratio",
    "route_time_low_ratio",
]

PATTERN_COLS = MEAN_COLS + STD_COLS + LOW_RATIO_COLS


# =========================================================
# 2. 패턴 통계 로드
# =========================================================
def load_pattern_stats(artifact_dir: str):
    """
    build_pattern_stats.py에서 저장한 패턴 통계 파일들을 불러온다.

    Returns
    -------
    stats_dict : dict
        각 통계 DataFrame
    pattern_meta : dict
        global_mean, global_low_ratio 등이 포함된 meta 정보
    """
    route_stat_path = os.path.join(artifact_dir, "pattern_route_stat.csv")
    route_stop_stat_path = os.path.join(artifact_dir, "pattern_route_stop_stat.csv")
    route_stop_time_stat_path = os.path.join(artifact_dir, "pattern_route_stop_time_stat.csv")
    route_staord_stat_path = os.path.join(artifact_dir, "pattern_route_staord_stat.csv")
    route_time_stat_path = os.path.join(artifact_dir, "pattern_route_time_stat.csv")
    pattern_meta_path = os.path.join(artifact_dir, "pattern_meta.json")

    required_files = [
        route_stat_path,
        route_stop_stat_path,
        route_stop_time_stat_path,
        route_staord_stat_path,
        route_time_stat_path,
        pattern_meta_path,
    ]

    missing_files = [path for path in required_files if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            "패턴 통계 파일이 없습니다.\n"
            + "\n".join(missing_files)
        )

    stats_dict = {
        "route_stat": pd.read_csv(route_stat_path, dtype={"busRouteId": str}),
        "route_stop_stat": pd.read_csv(route_stop_stat_path, dtype={"busRouteId": str, "stId": str}),
        "route_stop_time_stat": pd.read_csv(
            route_stop_time_stat_path,
            dtype={"busRouteId": str, "stId": str}
        ),
        "route_staord_stat": pd.read_csv(route_staord_stat_path, dtype={"busRouteId": str}),
        "route_time_stat": pd.read_csv(route_time_stat_path, dtype={"busRouteId": str}),
    }

    with open(pattern_meta_path, "r", encoding="utf-8") as f:
        pattern_meta = json.load(f)

    return stats_dict, pattern_meta


# =========================================================
# 3. 패턴 feature merge
# =========================================================
def merge_pattern_features(target_df: pd.DataFrame, stats_dict: dict, pattern_meta: dict) -> pd.DataFrame:
    """
    저장된 패턴 통계를 target_df에 merge하여 패턴 feature를 붙인다.

    Parameters
    ----------
    target_df : pd.DataFrame
        패턴 feature를 붙일 데이터프레임
    stats_dict : dict
        load_pattern_stats() 결과
    pattern_meta : dict
        load_pattern_stats() 결과

    Returns
    -------
    pd.DataFrame
    """
    result = target_df.copy()

    global_mean = float(pattern_meta["global_mean"])
    global_low_ratio = float(pattern_meta["global_low_ratio"])

    # 혹시 기존 동일 컬럼이 있으면 제거해서 _x/_y 충돌 방지
    drop_cols = [col for col in PATTERN_COLS if col in result.columns]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    result["busRouteId"] = result["busRouteId"].astype(str).str.strip()
    result["stId"] = result["stId"].astype(str).str.strip()

    # route_stat
    result = result.merge(
        stats_dict["route_stat"],
        on="busRouteId",
        how="left"
    )

    # route_stop_stat
    result = result.merge(
        stats_dict["route_stop_stat"],
        on=["busRouteId", "stId"],
        how="left"
    )

    # route_stop_time_stat
    result = result.merge(
        stats_dict["route_stop_time_stat"],
        on=["busRouteId", "stId", "dayofweek", "hour", "minute_group"],
        how="left"
    )

    # route_staord_stat
    result = result.merge(
        stats_dict["route_staord_stat"],
        on=["busRouteId", "staOrd"],
        how="left"
    )

    # route_time_stat
    result = result.merge(
        stats_dict["route_time_stat"],
        on=["busRouteId", "dayofweek", "hour"],
        how="left"
    )

    # 결측 보정
    for col in MEAN_COLS:
        if col not in result.columns:
            result[col] = global_mean
        else:
            result[col] = result[col].fillna(global_mean)

    for col in STD_COLS:
        if col not in result.columns:
            result[col] = 0
        else:
            result[col] = result[col].fillna(0)

    for col in LOW_RATIO_COLS:
        if col not in result.columns:
            result[col] = global_low_ratio
        else:
            result[col] = result[col].fillna(global_low_ratio)

    return result