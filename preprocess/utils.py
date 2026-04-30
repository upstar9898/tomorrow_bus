import numpy as np
import pandas as pd


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


def load_csv(path, use_cols):
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)
    print(f"로드 완료: {path}, shape={df.shape}")
    return df
