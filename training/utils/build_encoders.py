# =========================================================
# build_encoders.py
# ---------------------------------------------------------
# train 데이터 기준으로 LabelEncoder를 생성하고 저장하는 전용 스크립트
#
# [역할]
# 1. 원본 데이터 로드
# 2. 기본 전처리
# 3. 시간 파생변수 생성
# 4. 날짜 기준 train / valid / test 분할
# 5. train_df 기준 encoder 학습
# 6. encoder 저장
# =========================================================

import os
import json
import pandas as pd
import numpy as np

from utils.encoder_utils import (
    fit_label_encoders,
    save_label_encoders,
    get_encoder_class_summary,
)

# =========================================================
# 1. 프로젝트 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs_peak_v2")
MODEL_BASE_DIR = os.path.join(OUTPUT_ROOT, "models")
MODEL_DIR = os.path.join(MODEL_BASE_DIR, "lgbm_hybrid_peak_congestion4")

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# 2. 데이터 파일 경로
# =========================================================
file_path = os.path.join(DATA_DIR, "bus_all_raw_weather.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"데이터 파일을 찾을 수 없습니다.\n"
        f"확인한 경로: {file_path}"
    )

print(f"[INFO] file_path : {file_path}")
print(f"[INFO] MODEL_DIR : {MODEL_DIR}")

# =========================================================
# 3. 설정값
# =========================================================
MAX_SEAT = 45

# =========================================================
# 4. 원본 파일 불러오기
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

df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")

# =========================================================
# 5. 기본 정리
# =========================================================
required_cols = ["mkTm", "busRouteId", "stId", "arsId", "remaining_seat", "staOrd"]
df = df.dropna(subset=required_cols).copy()

df["busRouteId"] = df["busRouteId"].astype(str).str.strip()
df["stId"] = df["stId"].astype(str).str.strip()
df["arsId"] = df["arsId"].astype(str).str.strip()

df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")

df = df.dropna(subset=["remaining_seat", "staOrd"]).copy()
df = df[df["remaining_seat"] >= 0].copy()
df["remaining_seat"] = df["remaining_seat"].clip(0, MAX_SEAT)

print("총 데이터 수:", len(df))

# =========================================================
# 6. 날짜 관련 컬럼 생성
# =========================================================
df = df.sort_values("mkTm").reset_index(drop=True)

df["date"] = df["mkTm"].dt.date

unique_dates = sorted(df["date"].unique())
print("\n사용 날짜들:", unique_dates)
print("총 사용 날짜 수:", len(unique_dates))

if len(unique_dates) < 10:
    raise ValueError("최소 10일 이상은 있어야 안정적으로 분할 가능합니다.")

n_dates = len(unique_dates)
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

print("\n[분할 결과]")
print("train_df:", train_df.shape)
print("valid_df:", valid_df.shape)
print("test_df :", test_df.shape)

# =========================================================
# 7. train 기준 encoder 학습
# =========================================================
encoders = fit_label_encoders(train_df)

summary = get_encoder_class_summary(encoders)

print("\n[encoder class summary]")
for k, v in summary.items():
    print(f"{k}: {v}")

# =========================================================
# 8. encoder 저장
# =========================================================
save_label_encoders(encoders, save_dir=MODEL_DIR)

# =========================================================
# 9. encoder 메타 저장
# =========================================================
encoder_meta = {
    "source_file": file_path,
    "train_date_range": {
        "start": str(train_dates[0]),
        "end": str(train_dates[-1]),
        "num_days": len(train_dates),
    },
    "valid_date_range": {
        "start": str(valid_dates[0]),
        "end": str(valid_dates[-1]),
        "num_days": len(valid_dates),
    },
    "test_date_range": {
        "start": str(test_dates[0]),
        "end": str(test_dates[-1]),
        "num_days": len(test_dates),
    },
    "encoder_class_summary": summary,
    "saved_files": [
        "route_encoder.pkl",
        "stid_encoder.pkl",
        "arsid_encoder.pkl",
    ]
}

meta_path = os.path.join(MODEL_DIR, "encoder_meta.json")

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(encoder_meta, f, ensure_ascii=False, indent=2)

print("\nencoder 저장 완료")
print(f"- {os.path.join(MODEL_DIR, 'route_encoder.pkl')}")
print(f"- {os.path.join(MODEL_DIR, 'stid_encoder.pkl')}")
print(f"- {os.path.join(MODEL_DIR, 'arsid_encoder.pkl')}")
print(f"- {meta_path}")