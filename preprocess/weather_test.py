import os
import pandas as pd

# 프로젝트 루트
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# data 폴더
DATA_DIR = os.path.join(BASE_DIR, "data")

# 파일명
file_name = "weather_raw_260408.csv"

# 전체 경로
file_path = os.path.join(DATA_DIR, file_name)

# 읽기
df = pd.read_csv(file_path)

# 중복 제거
unique_values = df["WW"].dropna().unique()

for v in unique_values:
    print(v)
    

target = "1"

rows = df[df["WW"] == target].index

print(rows.tolist())