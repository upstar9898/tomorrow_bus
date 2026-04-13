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
]  # 원하는 컬럼 넣기
df_selected = df[selected_cols]

# 테스트용으로 조금만 잘라내기
# df_selected = df_selected.head(50)

# 중복 제거
unique_values = df["WW"].dropna().unique()

for v in unique_values:
    print(v)


target = "1"

rows = df[df["WW"] == target].index

print(rows.tolist())


# 문자열 → 리스트
df["WW_list"] = (
    df["WW_processed"].fillna("").apply(lambda x: x.split(",") if x != "" else [])
)
# explode (행 늘리기)
df_exploded = df.explode("WW_list")
# 빈 값 제거
df_exploded = df_exploded[df_exploded["WW_list"] != ""]
# 컬럼 이름 정리
df_exploded = df_exploded.rename(columns={"WW_list": "WW_code"})

unique_values = df_exploded["WW_code"].dropna().unique()

unique_values_list = []
for v in unique_values:
    unique_values_list.append(v)

for v in sorted(unique_values_list):
    print(v, end=" ")

target = "05"

rows = df_exploded[df_exploded["WW_code"] == target]
print()

print(
    rows["month"].values,
    rows["day"].values,
    rows["hour"].values,
    rows["STN"].values,
    end="\n",
)
