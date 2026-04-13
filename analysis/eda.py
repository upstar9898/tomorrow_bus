import warnings
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings("ignore")


# =========================================================
# 0. 프로젝트 경로 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV_PATH = BASE_DIR / "data" / "combined" / "bus_data_v2.csv"

OUTPUT_DIR = BASE_DIR / "data" / "eda_output"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. 시각화 스타일
# =========================================================
def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#222222",
        "grid.color": "#DDDDDD",
        "grid.linestyle": "-",
        "grid.alpha": 0.7,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "axes.unicode_minus": False,
    })

    sns.set_palette([
        "#2F2F2F",
        "#6B6B6B",
        "#A6A6A6",
        "#4A4A4A",
    ])


# =========================================================
# 2. 한글 폰트 설정
# =========================================================
def set_korean_font():
    font_candidates = [
        "Apple SD Gothic Neo",   # mac
        "AppleGothic",           # mac
        "NanumGothic",           # 설치형
        "Malgun Gothic",         # windows
        "DejaVu Sans",           # fallback
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    selected_font = None
    for font_name in font_candidates:
        if font_name in available_fonts:
            selected_font = font_name
            break

    if selected_font is None:
        print("[경고] 사용 가능한 한글 폰트를 찾지 못했습니다.")
        print("[안내] NanumGothic 설치 또는 Apple SD Gothic Neo 사용 가능 여부를 확인하세요.")
    else:
        print(f"[폰트 설정] {selected_font}")
        plt.rcParams["font.family"] = selected_font
        mpl.rcParams["font.family"] = selected_font
        sns.set(font=selected_font)

    plt.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.unicode_minus"] = False


set_plot_style()
set_korean_font()


# =========================================================
# 3. 공통 함수
# =========================================================
def save_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_table(df, filename):
    df.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")


def find_column(df, candidates, required=True):
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"필수 컬럼을 찾지 못했습니다. 후보: {candidates}")
    return None


def detect_columns(df):
    col_info = {
        "timestamp": find_column(
            df,
            ["mkTm", "collect_time", "collected_at", "timestamp", "datetime", "date_time", "created_at"]
        ),
        "veh_id": find_column(
            df,
            ["vehId1", "vehid1", "vehicle_id", "vehicleId", "veh_id"]
        ),
        "route_id": find_column(
            df,
            ["busRouteId", "route_id", "routeId", "bus_route_id"]
        ),
        "route_name": find_column(
            df,
            ["rtNm", "routeNm", "route_name", "busRouteNm"],
            required=False
        ),
        "station_id": find_column(
            df,
            ["stationId", "stId", "arsId", "station_id"]
        ),
        "station_name": find_column(
            df,
            ["stationNm", "stNm", "arsNm", "station_name"],
            required=False
        ),
        "remaining_seat": find_column(
            df,
            ["remaining_seat", "remain_seat", "reride_Num1", "reride_num1", "rerideNum1", "seat_left"]
        ),
        "full_flag": find_column(
            df,
            ["full1", "is_full", "full_flag"],
            required=False
        ),
    }
    return col_info


def preprocess(df, col):
    work = df.copy()

    work[col["timestamp"]] = pd.to_datetime(work[col["timestamp"]], errors="coerce")
    work = work.dropna(subset=[col["timestamp"]])

    work[col["remaining_seat"]] = pd.to_numeric(work[col["remaining_seat"]], errors="coerce")
    work = work.dropna(subset=[col["remaining_seat"], col["veh_id"], col["route_id"], col["station_id"]])

    if col["full_flag"] is not None:
        work[col["full_flag"]] = pd.to_numeric(work[col["full_flag"]], errors="coerce").fillna(0)
        work["is_full_vehicle"] = (work[col["full_flag"]] == 1).astype(int)
    else:
        work["is_full_vehicle"] = (work[col["remaining_seat"]] <= 0).astype(int)

    work["date"] = work[col["timestamp"]].dt.date
    work["hour"] = work[col["timestamp"]].dt.hour
    work["weekday_num"] = work[col["timestamp"]].dt.weekday
    work["weekday"] = work[col["timestamp"]].dt.day_name()

    weekday_map = {
        "Monday": "월",
        "Tuesday": "화",
        "Wednesday": "수",
        "Thursday": "목",
        "Friday": "금",
        "Saturday": "토",
        "Sunday": "일",
    }
    work["weekday_kr"] = work["weekday"].map(weekday_map)

    work = work.sort_values([col["veh_id"], col["timestamp"]]).reset_index(drop=True)
    work["seat_diff"] = work.groupby(col["veh_id"])[col["remaining_seat"]].diff()

    if col["station_name"] is None:
        work["station_name_for_analysis"] = work[col["station_id"]].astype(str)
    else:
        work["station_name_for_analysis"] = work[col["station_name"]].astype(str)

    if col["route_name"] is None:
        work["route_name_for_analysis"] = work[col["route_id"]].astype(str)
    else:
        work["route_name_for_analysis"] = work[col["route_name"]].astype(str)

    return work


# =========================================================
# 4. 요약 저장
# =========================================================
def make_summary_table(df, col):
    summary = pd.DataFrame({
        "item": [
            "전체 행 수",
            "전체 차량 수",
            "전체 노선 수",
            "전체 정류소 수",
            "잔여 좌석 최소값",
            "잔여 좌석 최대값",
            "만차 행 비율",
            "seat_diff 존재 행 수",
        ],
        "value": [
            len(df),
            df[col["veh_id"]].nunique(),
            df[col["route_id"]].nunique(),
            df[col["station_id"]].nunique(),
            df[col["remaining_seat"]].min(),
            df[col["remaining_seat"]].max(),
            round(df["is_full_vehicle"].mean(), 4),
            df["seat_diff"].notna().sum(),
        ]
    })

    save_table(summary, "0_summary.csv")


# =========================================================
# 5. 4-1. 시간 기반 패턴 분석
# =========================================================
def analyze_time_patterns(df, col):
    hourly_vehicle_status = (
        df.groupby(["hour", col["veh_id"]])
        .agg(
            min_remaining_seat=(col["remaining_seat"], "min"),
            full_count=("is_full_vehicle", "sum"),
        )
        .reset_index()
    )

    hourly_vehicle_status["is_congested_vehicle"] = (
        (hourly_vehicle_status["min_remaining_seat"] <= 3) |
        (hourly_vehicle_status["full_count"] > 0)
    ).astype(int)

    hourly_congestion = (
        hourly_vehicle_status.groupby("hour")
        .agg(
            total_vehicle_count=(col["veh_id"], "nunique"),
            congested_vehicle_count=("is_congested_vehicle", "sum"),
            full_vehicle_count=("full_count", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .sort_values("hour")
    )

    hourly_congestion["congested_vehicle_ratio"] = (
        hourly_congestion["congested_vehicle_count"] / hourly_congestion["total_vehicle_count"]
    )
    hourly_congestion["full_vehicle_ratio"] = (
        hourly_congestion["full_vehicle_count"] / hourly_congestion["total_vehicle_count"]
    )

    save_table(hourly_congestion, "4-1-1_hourly_vehicle_congestion.csv")

    plt.figure(figsize=(12, 6), dpi=120)
    plt.bar(
        hourly_congestion["hour"],
        hourly_congestion["congested_vehicle_ratio"],
        color="#4A4A4A",
        edgecolor="#2F2F2F",
    )
    plt.title("시간대별 혼잡 차량 비율")
    plt.xlabel("시간")
    plt.ylabel("혼잡 차량 비율")
    plt.xticks(hourly_congestion["hour"])
    save_plot("4-1-1_hourly_vehicle_congestion_ratio.png")

    plt.figure(figsize=(12, 6), dpi=120)
    plt.bar(
        hourly_congestion["hour"],
        hourly_congestion["full_vehicle_ratio"],
        color="#6B6B6B",
        edgecolor="#2F2F2F",
    )
    plt.title("시간대별 만차 차량 비율")
    plt.xlabel("시간")
    plt.ylabel("만차 차량 비율")
    plt.xticks(hourly_congestion["hour"])
    save_plot("4-1-1_hourly_full_vehicle_ratio.png")

    veh_hour_matrix = (
        hourly_vehicle_status
        .pivot_table(
            index=col["veh_id"],
            columns="hour",
            values="min_remaining_seat",
            aggfunc="min",
        )
    )

    save_table(
        veh_hour_matrix.reset_index(),
        "4-1-2_vehicle_hour_min_remaining_seat_matrix.csv",
    )

    sample_matrix = veh_hour_matrix.copy()
    if len(sample_matrix) > 100:
        sample_matrix = sample_matrix.head(100)

    plt.figure(figsize=(16, 12), dpi=120)
    sns.heatmap(sample_matrix, cmap="Greys", linewidths=0.2)
    plt.title("차량별 시간대 최소 잔여 좌석 분포 (상위 100개 차량)")
    plt.xlabel("시간")
    plt.ylabel("차량(vehId1)")
    save_plot("4-1-2_vehicle_hour_min_remaining_seat_heatmap.png")

    veh_congestion_freq = (
        hourly_vehicle_status.groupby(col["veh_id"])
        .agg(
            congested_hour_count=("is_congested_vehicle", "sum"),
            observed_hour_count=("hour", "nunique"),
            min_remaining_seat_overall=("min_remaining_seat", "min"),
        )
        .reset_index()
    )

    veh_congestion_freq["congested_hour_ratio"] = (
        veh_congestion_freq["congested_hour_count"] / veh_congestion_freq["observed_hour_count"]
    )

    veh_congestion_freq = veh_congestion_freq.sort_values(
        ["congested_hour_count", "min_remaining_seat_overall"],
        ascending=[False, True]
    )

    save_table(veh_congestion_freq, "4-1-2_vehicle_congestion_frequency.csv")

    top_20 = veh_congestion_freq.head(20)

    plt.figure(figsize=(12, 8), dpi=120)
    plt.barh(
        top_20[col["veh_id"]].astype(str),
        top_20["congested_hour_count"],
        color="#4A4A4A",
        edgecolor="#2F2F2F",
    )
    plt.title("혼잡 시간대가 많은 차량 TOP 20")
    plt.xlabel("혼잡 시간대 수")
    plt.ylabel("차량(vehId1)")
    plt.gca().invert_yaxis()
    save_plot("4-1-2_top20_vehicle_congested_hours.png")

    daily_vehicle_status = (
        df.groupby(["weekday_num", "weekday_kr", col["veh_id"]])
        .agg(
            min_remaining_seat=(col["remaining_seat"], "min"),
            full_count=("is_full_vehicle", "sum"),
        )
        .reset_index()
    )

    daily_vehicle_status["is_congested_vehicle"] = (
        (daily_vehicle_status["min_remaining_seat"] <= 3) |
        (daily_vehicle_status["full_count"] > 0)
    ).astype(int)

    weekday_congestion = (
        daily_vehicle_status.groupby(["weekday_num", "weekday_kr"])
        .agg(
            total_vehicle_count=(col["veh_id"], "nunique"),
            congested_vehicle_count=("is_congested_vehicle", "sum"),
            full_vehicle_count=("full_count", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .sort_values("weekday_num")
    )

    weekday_congestion["congested_vehicle_ratio"] = (
        weekday_congestion["congested_vehicle_count"] / weekday_congestion["total_vehicle_count"]
    )
    weekday_congestion["full_vehicle_ratio"] = (
        weekday_congestion["full_vehicle_count"] / weekday_congestion["total_vehicle_count"]
    )

    save_table(weekday_congestion, "4-1-3_weekday_vehicle_congestion.csv")

    plt.figure(figsize=(10, 6), dpi=120)
    plt.bar(
        weekday_congestion["weekday_kr"],
        weekday_congestion["congested_vehicle_ratio"],
        color="#4A4A4A",
        edgecolor="#2F2F2F",
    )
    plt.title("요일별 혼잡 차량 비율")
    plt.xlabel("요일")
    plt.ylabel("혼잡 차량 비율")
    save_plot("4-1-3_weekday_vehicle_congestion_ratio.png")


# =========================================================
# 6. 4-2. 좌석 변화 분석
# =========================================================
def analyze_diff_patterns(df, col):
    diff_df = df.dropna(subset=["seat_diff"]).copy()

    diff_df["seat_diff_negative"] = diff_df["seat_diff"].where(diff_df["seat_diff"] < 0, 0)
    diff_df["seat_drop_abs"] = diff_df["seat_diff_negative"].abs()

    hourly_diff = (
        diff_df.groupby("hour")
        .agg(
            total_drop=("seat_drop_abs", "sum"),
            drop_event_count=("seat_drop_abs", lambda x: (x > 0).sum()),
            max_drop=("seat_diff_negative", "min"),
        )
        .reset_index()
        .sort_values("hour")
    )

    save_table(hourly_diff, "4-2-1_hourly_diff_summary.csv")

    plt.figure(figsize=(12, 6), dpi=120)
    plt.bar(
        hourly_diff["hour"],
        hourly_diff["total_drop"],
        color="#4A4A4A",
        edgecolor="#2F2F2F",
    )
    plt.title("시간대별 총 좌석 감소량")
    plt.xlabel("시간")
    plt.ylabel("총 좌석 감소량")
    plt.xticks(hourly_diff["hour"])
    save_plot("4-2-1_hourly_total_seat_drop.png")

    plt.figure(figsize=(12, 6), dpi=120)
    plt.bar(
        hourly_diff["hour"],
        hourly_diff["drop_event_count"],
        color="#6B6B6B",
        edgecolor="#2F2F2F",
    )
    plt.title("시간대별 좌석 감소 발생 건수")
    plt.xlabel("시간")
    plt.ylabel("감소 발생 건수")
    plt.xticks(hourly_diff["hour"])
    save_plot("4-2-1_hourly_drop_event_count.png")

    vehicle_diff = (
        diff_df.groupby(col["veh_id"])
        .agg(
            total_drop=("seat_drop_abs", "sum"),
            drop_event_count=("seat_drop_abs", lambda x: (x > 0).sum()),
            max_single_drop=("seat_diff_negative", "min"),
        )
        .reset_index()
        .sort_values(["total_drop", "drop_event_count"], ascending=[False, False])
    )

    save_table(vehicle_diff, "4-2-2_vehicle_diff_summary.csv")

    top_drop_20 = vehicle_diff.head(20)

    plt.figure(figsize=(12, 8), dpi=120)
    plt.barh(
        top_drop_20[col["veh_id"]].astype(str),
        top_drop_20["total_drop"],
        color="#4A4A4A",
        edgecolor="#2F2F2F",
    )
    plt.title("좌석 감소량이 큰 차량 TOP 20")
    plt.xlabel("총 좌석 감소량")
    plt.ylabel("차량(vehId1)")
    plt.gca().invert_yaxis()
    save_plot("4-2-2_top20_vehicle_total_drop.png")

    rapid_drop_events = diff_df[diff_df["seat_diff"] <= -5].copy()

    rapid_drop_vehicle = (
        rapid_drop_events.groupby(col["veh_id"])
        .agg(
            rapid_drop_count=("seat_diff", "size"),
            worst_drop=("seat_diff", "min"),
        )
        .reset_index()
        .sort_values(["rapid_drop_count", "worst_drop"], ascending=[False, True])
    )

    save_table(rapid_drop_vehicle, "4-2-2_rapid_drop_vehicle_summary.csv")

    top_rapid_20 = rapid_drop_vehicle.head(20)

    if len(top_rapid_20) > 0:
        plt.figure(figsize=(12, 8), dpi=120)
        plt.barh(
            top_rapid_20[col["veh_id"]].astype(str),
            top_rapid_20["rapid_drop_count"],
            color="#6B6B6B",
            edgecolor="#2F2F2F",
        )
        plt.title("급격한 좌석 감소 발생 차량 TOP 20 (diff <= -5)")
        plt.xlabel("급격한 감소 발생 횟수")
        plt.ylabel("차량(vehId1)")
        plt.gca().invert_yaxis()
        save_plot("4-2-2_top20_rapid_drop_vehicle.png")


# =========================================================
# 7. 4-3. 공간 기반 분석
# =========================================================
def analyze_spatial_patterns(df, col):
    route_vehicle_status = (
        df.groupby([col["route_id"], "route_name_for_analysis", col["veh_id"]])
        .agg(
            min_remaining_seat=(col["remaining_seat"], "min"),
            full_count=("is_full_vehicle", "sum"),
        )
        .reset_index()
    )

    route_vehicle_status["is_congested_vehicle"] = (
        (route_vehicle_status["min_remaining_seat"] <= 3) |
        (route_vehicle_status["full_count"] > 0)
    ).astype(int)

    route_congestion = (
        route_vehicle_status.groupby([col["route_id"], "route_name_for_analysis"])
        .agg(
            total_vehicle_count=(col["veh_id"], "nunique"),
            congested_vehicle_count=("is_congested_vehicle", "sum"),
            full_vehicle_count=("full_count", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    route_congestion["congested_vehicle_ratio"] = (
        route_congestion["congested_vehicle_count"] / route_congestion["total_vehicle_count"]
    )
    route_congestion["full_vehicle_ratio"] = (
        route_congestion["full_vehicle_count"] / route_congestion["total_vehicle_count"]
    )

    route_congestion = route_congestion.sort_values(
        ["full_vehicle_ratio", "congested_vehicle_ratio"],
        ascending=[False, False]
    )

    save_table(route_congestion, "4-3-1_route_vehicle_congestion.csv")

    top_route_20 = route_congestion.head(20).copy()
    top_route_20["route_label"] = top_route_20["route_name_for_analysis"].astype(str)

    plt.figure(figsize=(12, 8), dpi=120)
    plt.barh(
        top_route_20["route_label"],
        top_route_20["full_vehicle_ratio"],
        color="#4A4A4A",
        edgecolor="#2F2F2F",
    )
    plt.title("노선별 만차 차량 비율 TOP 20")
    plt.xlabel("만차 차량 비율")
    plt.ylabel("노선")
    plt.gca().invert_yaxis()
    save_plot("4-3-1_top20_route_full_vehicle_ratio.png")

    diff_df = df.dropna(subset=["seat_diff"]).copy()
    boarding_df = diff_df[diff_df["seat_diff"] < 0].copy()
    boarding_df["boarding_count_est"] = boarding_df["seat_diff"].abs()

    station_inflow = (
        boarding_df.groupby([col["station_id"], "station_name_for_analysis"])
        .agg(
            total_estimated_boarding=("boarding_count_est", "sum"),
            drop_event_count=("boarding_count_est", "size"),
            max_single_boarding=("boarding_count_est", "max"),
        )
        .reset_index()
        .sort_values(["total_estimated_boarding", "drop_event_count"], ascending=[False, False])
    )

    save_table(station_inflow, "4-3-2_station_inflow_top.csv")

    top_station_20 = station_inflow.head(20)

    plt.figure(figsize=(12, 8), dpi=120)
    plt.barh(
        top_station_20["station_name_for_analysis"].astype(str),
        top_station_20["total_estimated_boarding"],
        color="#4A4A4A",
        edgecolor="#2F2F2F",
    )
    plt.title("정류소별 추정 탑승 유입량 TOP 20")
    plt.xlabel("추정 탑승 유입량")
    plt.ylabel("정류소")
    plt.gca().invert_yaxis()
    save_plot("4-3-2_top20_station_estimated_boarding.png")

    plt.figure(figsize=(12, 8), dpi=120)
    plt.barh(
        top_station_20["station_name_for_analysis"].astype(str),
        top_station_20["drop_event_count"],
        color="#6B6B6B",
        edgecolor="#2F2F2F",
    )
    plt.title("정류소별 좌석 감소 발생 횟수 TOP 20")
    plt.xlabel("좌석 감소 발생 횟수")
    plt.ylabel("정류소")
    plt.gca().invert_yaxis()
    save_plot("4-3-2_top20_station_drop_events.png")


# =========================================================
# 8. 메인 실행
# =========================================================
def main():
    print(f"[1] 데이터 불러오는 중: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    print("[2] 컬럼 탐지 중")
    col = detect_columns(df)
    print("탐지된 컬럼:", col)

    print("[3] 전처리 중")
    work = preprocess(df, col)
    print("전처리 후 shape:", work.shape)

    print("[4] 요약 테이블 저장")
    make_summary_table(work, col)

    print("[5] 4-1 시간 기반 패턴 분석")
    analyze_time_patterns(work, col)

    print("[6] 4-2 좌석 변화 분석")
    analyze_diff_patterns(work, col)

    print("[7] 4-3 공간 기반 분석")
    analyze_spatial_patterns(work, col)

    print("\n완료")
    print(f"- 그래프 저장 폴더: {PLOT_DIR}")
    print(f"- 표 저장 폴더: {TABLE_DIR}")


if __name__ == "__main__":
    main()