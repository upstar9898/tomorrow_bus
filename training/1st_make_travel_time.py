import os
import pandas as pd

# ==============================================
# 0단계. 경로 설정
# ==============================================
# 현재 파이썬 파일 위치를 기준으로 경로를 잡는다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 생성 파일(csv 등)을 저장할 폴더
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ==============================================
# 1단계. 이동시간 테이블 만들기
# ==============================================
# 버스 이동시간 원본 파일 경로
travel_raw_path = os.path.join(BASE_DIR, "bus_traveltime_merged.csv")

df = pd.read_csv(
    travel_raw_path,
    dtype={
        "busRouteId": str,
        "stId": str,
        "arsId": str,
        "vehId1": str
    },
    low_memory=False
)

# 시간 / 정류장 순번 컬럼 형 변환
df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")

# 필수값 없는 행 제거
df = df.dropna(subset=["mkTm", "busRouteId", "stId", "vehId1", "staOrd"]).copy()

# 문자열 컬럼 공백 제거
for col in ["busRouteId", "stId", "arsId", "vehId1"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# 같은 노선 / 같은 차량 기준으로 시간순 정렬
df = df.sort_values(["busRouteId", "vehId1", "mkTm", "staOrd"]).reset_index(drop=True)

# 이전 행 정보 만들기
df["prev_mkTm"] = df.groupby(["busRouteId", "vehId1"])["mkTm"].shift(1)
df["prev_stId"] = df.groupby(["busRouteId", "vehId1"])["stId"].shift(1)
df["prev_staOrd"] = df.groupby(["busRouteId", "vehId1"])["staOrd"].shift(1)

# 현재 정류장과 이전 정류장 사이 이동시간(초) 계산
df["travel_time"] = (df["mkTm"] - df["prev_mkTm"]).dt.total_seconds()

# 바로 다음 정류장으로 이동한 경우만 사용
# 예: 5 -> 6 은 사용, 5 -> 7 은 제외
df_tt = df[(df["staOrd"] - df["prev_staOrd"] == 1)].copy()

# 이상치 제거
# 너무 짧거나 너무 긴 이동시간은 제거
df_tt = df_tt[df_tt["travel_time"].between(10, 600)].copy()

# 노선 + 이전정류장 + 현재정류장 기준 평균/중앙값 이동시간 계산
travel_time_table = (
    df_tt.groupby(["busRouteId", "prev_stId", "stId"])
    .agg(
        avg_travel_time=("travel_time", "mean"),
        median_travel_time=("travel_time", "median"),
        sample_count=("travel_time", "count")
    )
    .reset_index()
)

# artifacts 폴더에 저장
travel_time_output_path = os.path.join(ARTIFACT_DIR, "travel_time_table.csv")
travel_time_table.to_csv(travel_time_output_path, index=False, encoding="utf-8-sig")

print("저장 완료:", travel_time_output_path)
print(travel_time_table.head())


# ===============================================================
# 2단계. 노선별 정류소 순서 테이블 저장
# ===============================================================
# 전체 원본 데이터에서 노선별 정류소 순서를 추출
raw_bus_path = os.path.join(BASE_DIR, "bus_all_raw3.csv")

df_route = pd.read_csv(
    raw_bus_path,
    dtype={"busRouteId": str, "stId": str, "arsId": str},
    low_memory=False
)

df_route["staOrd"] = pd.to_numeric(df_route["staOrd"], errors="coerce")
df_route = df_route.dropna(subset=["busRouteId", "stId", "arsId", "staOrd"]).copy()

for col in ["busRouteId", "stId", "arsId"]:
    df_route[col] = df_route[col].astype(str).str.strip()

# 노선별 정류소 순번 테이블 생성
route_station_order = (
    df_route[["busRouteId", "stId", "arsId", "staOrd"]]
    .drop_duplicates()
    .sort_values(["busRouteId", "staOrd"])
    .reset_index(drop=True)
)

# artifacts 폴더에 저장
route_station_output_path = os.path.join(ARTIFACT_DIR, "route_station_order.csv")
route_station_order.to_csv(route_station_output_path, index=False, encoding="utf-8-sig")

print("저장 완료:", route_station_output_path)
print(route_station_order.head())