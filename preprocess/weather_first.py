import os
import pandas as pd

# 프로젝트 루트
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# data 폴더
DATA_DIR = os.path.join(BASE_DIR, "data")

# 파일명
input_filename = "weather_raw_260408.csv"

# 전체 경로
input_filepath = os.path.join(DATA_DIR, input_filename)

# 데이터 타입 확실하게 지정
dtype = {"TM": str, "STN": str, "WW": str, "TA": float, "RN": float}

# 읽기
df = pd.read_csv(input_filepath, dtype=dtype)

# datetime 변환
date_time = pd.to_datetime(df["TM"], format="%Y%m%d%H%M")


# WW 변환
def process_ww(x):
    if pd.isna(x) or x == "-":
        return ""

    x = x.strip()

    # 홀수 길이면 앞에 0 추가
    if len(x) % 2 == 1:
        x = "0" + x

    # 2자리씩 분리
    codes = [x[i : i + 2] for i in range(0, len(x), 2)]

    # 문자열로 변환 (쉼표 기준)
    return ",".join(codes)


df["WW_processed"] = df["WW"].apply(process_ww)

# 컬럼 분리
df["year"] = date_time.dt.year
df["month"] = date_time.dt.month
df["day"] = date_time.dt.day
df["hour"] = date_time.dt.hour

# 문자열 → 리스트
df["WW_list"] = (
    df["WW_processed"].fillna("").apply(lambda x: x.split(",") if x != "" else [])
)

# 강수로 지정한 코드
precipitation_code = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
]

# 안개로 지정한 코드
fog_code = ["16", "17", "18", "19"]


def check_codes(x, target_codes):
    if pd.isna(x) or x == "":
        return 0
    codes = x.split(",")
    return int(any(code in target_codes for code in codes))


# feature 생성
df["precipitation"] = df["WW_processed"].apply(
    lambda x: check_codes(x, precipitation_code)
)
df["fog"] = df["WW_processed"].apply(lambda x: check_codes(x, fog_code))

# 필요한 컬럼만 선택
selected_cols = [
    "year",
    "month",
    "day",
    "hour",
    "STN",
    "WW_processed",
    "TA",
    "RN",
    "precipitation",
    "fog",
]  # 원하는 컬럼 넣기
df_selected = df[selected_cols]


# CSV 저장
output_filename = input_filename.replace("_raw","").replace(".csv", "_first_processed.csv")
output_filepath = os.path.join(DATA_DIR, output_filename)
df_selected.to_csv(output_filepath, index=False)
