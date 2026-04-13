from pathlib import Path
import pandas as pd

# =========================
# 경로 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "preprocessed_v2"

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", DATA_DIR)
print("폴더 존재 여부:", DATA_DIR.exists())

# =========================
# CSV 파일 불러오기
# =========================
csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"파일 수: {len(csv_files)}")
for file in csv_files[:10]:
    print(file.name)

if not csv_files:
    raise FileNotFoundError("CSV 파일을 찾지 못했습니다. data/preprocessed_v2 경로를 다시 확인하세요.")

df_list = []

for file in csv_files:
    df = pd.read_csv(file)
    print(f"{file.name} 로드 완료, shape={df.shape}")
    df_list.append(df)

# =========================
# concat
# =========================
df_all = pd.concat(df_list, ignore_index=True)

print("\n=== concat 완료 ===")
print("전체 shape:", df_all.shape)
print(df_all.head())

# =========================
# 기본 정보 확인
# =========================
print("\n=== 기본 정보 ===")
print(df_all.shape)
print(df_all.columns.tolist())

print("\n=== prev_seat / diff 존재 여부 ===")
print('prev_seat' in df_all.columns, 'diff' in df_all.columns)

print("\n=== info ===")
df_all.info()

print("\n=== 결측치 상위 20개 ===")
print(df_all.isna().sum().sort_values(ascending=False).head(20))

print("\n=== 중복 행 수 ===")
print(df_all.duplicated().sum())

# =========================
# 시간 처리
# =========================
df_all['mkTm'] = pd.to_datetime(df_all['mkTm'])

df_all = df_all.sort_values(by=['busRouteId', 'stId', 'mkTm'])

print(df_all[['busRouteId', 'stId', 'mkTm']].head(10))
print(df_all['mkTm'].min(), df_all['mkTm'].max())

# =========================
# 시간 간격 분석
# =========================
df_all['time_diff'] = df_all.groupby(['busRouteId', 'stId'])['mkTm'].diff().dt.total_seconds()
df_all['time_diff_min'] = df_all['time_diff'] / 60

print(df_all['time_diff_min'].describe())
print(df_all['time_diff_min'].value_counts().head(10))

print("\n=== 10분 이상 끊긴 데이터 ===")
print(df_all[df_all['time_diff_min'] > 10].head(20))

# =========================
# rolling feature (유지)
# =========================
df_all['date'] = df_all['mkTm'].dt.date

df_all['seat_roll_mean_3'] = df_all.groupby(
    ['busRouteId', 'stId', 'date']
)['remaining_seat'].transform(lambda x: x.rolling(3).mean())

df_all['seat_roll_std_3'] = df_all.groupby(
    ['busRouteId', 'stId', 'date']
)['remaining_seat'].transform(lambda x: x.rolling(3).std())

print(df_all[['remaining_seat', 'prev_seat', 'diff', 'seat_roll_mean_3', 'seat_roll_std_3']].head(10))

# =========================
# 저장
# =========================
output_path = BASE_DIR / "data" / "combined" / "bus_data_v2.csv"

output_path.parent.mkdir(parents=True, exist_ok=True)

df_all.to_csv(output_path, index=False)

print("저장 완료:", output_path)