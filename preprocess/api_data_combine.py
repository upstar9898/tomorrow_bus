import pandas as pd
import os
import numpy as np
import glob

# =========================================================
# 1. 설정
# =========================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
WITHWEATHER_DATA_DIR = os.path.join(DATA_DIR, "preprocessed_withweather")

SEAT_COL = "remaining_seat"
TIME_COL = "mkTm"

OUTPUT_FILE = os.path.join(WITHWEATHER_DATA_DIR, "bus_all_raw_weather.csv")

# =========================================================
# 2. 파일 탐색 (날씨 포함 파일)
# =========================================================
file_list = sorted(glob.glob(os.path.join(WITHWEATHER_DATA_DIR, "*_withweather.csv")))

if not file_list:
    raise FileNotFoundError("withweather CSV 파일이 없습니다.")

print(f"[총 파일 개수] {len(file_list)}")

dfs = []

for file_path in file_list:
    
    file_name = os.path.basename(file_path)
    
    print(f"[불러오는 중] {file_name}")

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="cp949", low_memory=False)

    # 파일별 이상 컬럼 체크
    tab_cols = [c for c in df.columns if "\t" in str(c)]
    if tab_cols:
        print(f"[주의] 탭이 포함된 이상 컬럼 발견: {file_name}")
        print(tab_cols)

    df["source_file"] = file_name
    dfs.append(df)

# =========================================================
# 3. 하나로 합치기
# =========================================================
df_all = pd.concat(dfs, ignore_index=True)
print(f"\n[원본 데이터 크기] {df_all.shape}")

# =========================================================
# 4. 기본 전처리
# =========================================================
df_all[TIME_COL] = pd.to_datetime(df_all[TIME_COL], errors="coerce")

required_cols = [
    TIME_COL, SEAT_COL,
    "busRouteId", "stId", "arsId"
]

for col in required_cols:
    if col not in df_all.columns:
        raise KeyError(f"필수 컬럼 없음: {col}")

df_all[SEAT_COL] = pd.to_numeric(df_all[SEAT_COL], errors="coerce")

# =========================================================
# 5. 결측 및 이상값 제거
# =========================================================
before = len(df_all)
df_all = df_all.dropna(subset=required_cols)
print(f"[결측 제거] {before - len(df_all)}행 제거")

before = len(df_all)
df_all = df_all[df_all[SEAT_COL] >= 0]
print(f"[음수 제거] {before - len(df_all)}행 제거")

# =========================================================
# 6. 문자열 정리
# =========================================================
df_all["busRouteId"] = df_all["busRouteId"].astype(str).str.strip()
df_all["stId"] = df_all["stId"].astype(str).str.strip()
df_all["arsId"] = df_all["arsId"].astype(str).str.strip()

# =========================================================
# 7. 날씨 컬럼 처리
# =========================================================
weather_cols = ["temperature", "rainfall", "precipitation", "fog"]

for col in weather_cols:
    if col not in df_all.columns:
        print(f"[경고] {col} 컬럼이 없습니다.")
        df_all[col] = np.nan

df_all["temperature"] = pd.to_numeric(df_all["temperature"], errors="coerce")
df_all["temperature"] = df_all["temperature"].fillna(df_all["temperature"].median())

df_all["rainfall"] = pd.to_numeric(df_all["rainfall"], errors="coerce")
df_all["rainfall_missing"] = df_all["rainfall"].isna().astype(int)
df_all["rainfall"] = df_all["rainfall"].fillna(0)

precip_raw = df_all["precipitation"].copy()
df_all["precipitation"] = pd.to_numeric(df_all["precipitation"], errors="coerce")
if df_all["precipitation"].isna().all():
    print("[주의] precipitation이 숫자형이 아닐 가능성이 큽니다. 원본 값 확인 필요")
    print(precip_raw.astype(str).value_counts().head(20))
df_all["precipitation"] = df_all["precipitation"].fillna(0)

fog_raw = df_all["fog"].copy()
df_all["fog"] = pd.to_numeric(df_all["fog"], errors="coerce")
if df_all["fog"].isna().all():
    print("[주의] fog가 숫자형이 아닐 가능성이 큽니다. 원본 값 확인 필요")
    print(fog_raw.astype(str).value_counts().head(20))
df_all["fog"] = df_all["fog"].fillna(0)

# =========================================================
# 8. 이상 컬럼 제거
# =========================================================
bad_cols = [col for col in df_all.columns if "\t" in str(col)]
if bad_cols:
    print("\n[이상 컬럼 삭제]")
    print(bad_cols)
    df_all = df_all.drop(columns=bad_cols, errors="ignore")

# =========================================================
# 9. 정렬 및 확인
# =========================================================
df_all = df_all.sort_values(by=["busRouteId", "stId", "mkTm"]).reset_index(drop=True)

print("\n[remaining_seat 기초통계]")
print(df_all[SEAT_COL].describe())

print("\n[상위 분위수]")
print(df_all[SEAT_COL].quantile([0.90, 0.95, 0.99, 0.999, 1.0]))

print("\n[최종 데이터 크기]")
print(df_all.shape)

print("\n[컬럼 목록]")
print(df_all.columns.tolist())

print("\n[rainfall 요약]")
print(df_all["rainfall"].describe())
print(df_all["rainfall"].value_counts().head(20))
print("[rainfall > 0 비율]", (df_all["rainfall"] > 0).mean())

print("\n[precipitation 요약]")
print(df_all["precipitation"].describe())
print(df_all["precipitation"].value_counts(dropna=False).head(20))

print("\n[fog 요약]")
print(df_all["fog"].describe())
print(df_all["fog"].value_counts(dropna=False).head(20))

# =========================================================
# 10. 저장
# =========================================================
df_all.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n[저장 완료] {OUTPUT_FILE}")