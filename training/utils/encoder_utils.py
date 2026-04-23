# =========================================================
# encoder_utils.py
# ---------------------------------------------------------
# 범주형 ID 인코더 생성 / 저장 / 로드 / 변환 전용 유틸
#
# [기본 대상 컬럼]
# - busRouteId -> route_enc
# - stId       -> stid_enc
# - arsId      -> arsid_enc
#
# [권장 사용 흐름]
# 1. train 데이터로 fit_label_encoders(df_train)
# 2. save_label_encoders(encoders, save_dir)
# 3. 학습/추론 시 load_label_encoders(load_dir)
# 4. transform_with_encoders(df, encoders)
# =========================================================

import os
import joblib
import pandas as pd

from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder


# =========================================================
# 1. 기본 설정
# =========================================================
ENCODER_SPECS = {
    "busRouteId": {
        "encoder_name": "route_encoder",
        "output_col": "route_enc",
        "filename": "route_encoder.pkl",
    },
    "stId": {
        "encoder_name": "stid_encoder",
        "output_col": "stid_enc",
        "filename": "stid_encoder.pkl",
    },
    "arsId": {
        "encoder_name": "arsid_encoder",
        "output_col": "arsid_enc",
        "filename": "arsid_encoder.pkl",
    },
}


# =========================================================
# 2. 공통 보조 함수
# =========================================================
def _normalize_categorical_value(value: Any) -> str:
    """
    인코딩 전 공통 문자열 정리 함수

    - 문자열로 변환
    - 양쪽 공백 제거
    - None / NaN 은 빈 문자열로 처리
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def validate_required_encoder_columns(df: pd.DataFrame) -> None:
    """
    인코더 학습/변환에 필요한 원본 컬럼이 모두 있는지 검사한다.
    """
    required_cols = list(ENCODER_SPECS.keys())
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"인코더 처리에 필요한 컬럼이 없습니다: {missing_cols}\n"
            f"현재 컬럼: {list(df.columns)}"
        )


# =========================================================
# 3. 인코더 학습
# =========================================================
def fit_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """
    train 데이터 기준으로 LabelEncoder를 학습한다.

    Parameters
    ----------
    df : pd.DataFrame
        train 데이터프레임

    Returns
    -------
    Dict[str, LabelEncoder]
        학습된 encoder dict
    """
    validate_required_encoder_columns(df)

    encoders = {}

    for raw_col, spec in ENCODER_SPECS.items():
        le = LabelEncoder()

        values = df[raw_col].map(_normalize_categorical_value)

        if (values == "").all():
            raise ValueError(f"{raw_col} 컬럼이 전부 비어 있어 encoder를 학습할 수 없습니다.")

        le.fit(values)
        encoders[spec["encoder_name"]] = le

    return encoders


# =========================================================
# 4. 인코더 저장
# =========================================================
def save_label_encoders(encoders: Dict[str, LabelEncoder], save_dir: str) -> None:
    """
    학습된 encoder들을 joblib 파일로 저장한다.

    Parameters
    ----------
    encoders : Dict[str, LabelEncoder]
        fit 완료된 encoder dict
    save_dir : str
        저장 폴더 경로
    """
    os.makedirs(save_dir, exist_ok=True)

    for raw_col, spec in ENCODER_SPECS.items():
        encoder_name = spec["encoder_name"]
        filename = spec["filename"]

        if encoder_name not in encoders:
            raise ValueError(f"encoders dict에 '{encoder_name}' 가 없습니다.")

        save_path = os.path.join(save_dir, filename)
        joblib.dump(encoders[encoder_name], save_path)


# =========================================================
# 5. 인코더 로드
# =========================================================
def load_label_encoders(load_dir: str) -> Dict[str, LabelEncoder]:
    """
    저장된 encoder들을 불러온다.

    Parameters
    ----------
    load_dir : str
        encoder가 저장된 폴더

    Returns
    -------
    Dict[str, LabelEncoder]
        로드된 encoder dict
    """
    encoders = {}

    for raw_col, spec in ENCODER_SPECS.items():
        encoder_name = spec["encoder_name"]
        filename = spec["filename"]
        load_path = os.path.join(load_dir, filename)

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"encoder 파일이 없습니다: {load_path}"
            )

        encoders[encoder_name] = joblib.load(load_path)

    return encoders


# =========================================================
# 6. 단일 값 인코딩
# =========================================================
def safe_label_encode_value(
    encoder: LabelEncoder,
    value: Any,
    field_name: str,
    strict: bool = True,
) -> int:
    """
    단일 값을 안전하게 인코딩한다.
    서비스 추론 시 route/station 단건 변환용으로 사용 가능.

    Parameters
    ----------
    encoder : LabelEncoder
    value : Any
        변환할 값
    field_name : str
        에러 메시지용 필드명
    strict : bool
        True  -> 학습에 없는 값이면 에러 발생
        False -> 학습에 없는 값이면 -1 반환

    Returns
    -------
    int
    """
    norm_value = _normalize_categorical_value(value)

    if norm_value not in encoder.classes_:
        if strict:
            raise ValueError(f"{field_name} 값이 학습 데이터에 없습니다: {norm_value}")
        return -1

    return int(encoder.transform([norm_value])[0])


# =========================================================
# 7. 데이터프레임 전체 변환
# =========================================================
def transform_with_encoders(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    strict: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """
    저장된 encoder로 데이터프레임 전체를 변환한다.

    Parameters
    ----------
    df : pd.DataFrame
        변환 대상 데이터프레임
    encoders : Dict[str, LabelEncoder]
        load_label_encoders() 결과
    strict : bool
        True  -> 학습에 없는 값이 하나라도 있으면 에러 발생
        False -> 학습에 없는 값은 -1로 처리
    copy : bool
        True이면 원본 복사 후 반환

    Returns
    -------
    pd.DataFrame
        인코딩 컬럼(route_enc, stid_enc, arsid_enc)이 추가된 데이터프레임
    """
    validate_required_encoder_columns(df)

    result = df.copy() if copy else df

    for raw_col, spec in ENCODER_SPECS.items():
        encoder_name = spec["encoder_name"]
        output_col = spec["output_col"]

        if encoder_name not in encoders:
            raise ValueError(f"encoders dict에 '{encoder_name}' 가 없습니다.")

        le = encoders[encoder_name]
        values = result[raw_col].map(_normalize_categorical_value)

        if strict:
            unknown_values = sorted(set(values) - set(le.classes_))
            if unknown_values:
                preview = unknown_values[:10]
                raise ValueError(
                    f"{raw_col} 컬럼에 학습 데이터에 없는 값이 있습니다.\n"
                    f"예시: {preview}\n"
                    f"총 {len(unknown_values)}개"
                )

            result[output_col] = le.transform(values).astype(int)

        else:
            class_to_index = {cls: idx for idx, cls in enumerate(le.classes_)}
            result[output_col] = values.map(lambda x: class_to_index.get(x, -1)).astype(int)

    return result


# =========================================================
# 8. 현재 encoder 클래스 정보 확인용
# =========================================================
def get_encoder_class_summary(encoders: Dict[str, LabelEncoder]) -> Dict[str, int]:
    """
    각 encoder가 몇 개의 클래스를 갖고 있는지 요약해서 반환한다.
    """
    summary = {}

    for raw_col, spec in ENCODER_SPECS.items():
        encoder_name = spec["encoder_name"]

        if encoder_name not in encoders:
            summary[encoder_name] = 0
        else:
            summary[encoder_name] = len(encoders[encoder_name].classes_)

    return summary