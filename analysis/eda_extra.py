import warnings
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings("ignore")


# =========================================================
# 0. 경로 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV_PATH = BASE_DIR / "data" / "combined" / "bus_data_v2.csv"
ROUTE_INFO_PATH = BASE_DIR / "data" / "bus_route.csv"

OUTPUT_DIR = BASE_DIR / "data" / "eda_extra_output"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. 스타일 / 폰트
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


def set_korean_font():
    font_candidates = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "DejaVu Sans",
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}
    selected_font = None

    for font_name in font_candidates:
        if font_name in available_fonts:
            selected_font = font_name
            break

    if selected_font is None:
        print("[경고] 사용 가능한 한글 폰트를 찾지 못했습니다.")
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
# 2. 공통 함수
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


def clean_id_series(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def detect_columns(df):
    return {
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


# =========================================================
# 3. 노선 정보 불러오기
# =========================================================
def load_route_info():
    if not ROUTE_INFO_PATH.exists():
        raise FileNotFoundError(f"bus_route.csv 파일이 없습니다: {ROUTE_INFO_PATH}")

    route_df = pd.read_csv(ROUTE_INFO_PATH)

    # 너 파일 기준 컬럼
    route_df = route_df.rename(columns={
        "routeId": "busRouteId",
        "routeName": "busRouteNm"
    })

    # 타입 맞추기 (중요)
    route_df["busRouteId"] = (
        route_df["busRouteId"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    route_df["busRouteNm"] = route_df["busRouteNm"].astype(str).str.strip()

    return route_df[["busRouteId", "busRouteNm"]]
# =========================================================
# 4. 전처리
# =========================================================
def preprocess(df, col, route_info):
    work = df.copy()

    work[col["timestamp"]] = pd.to_datetime(work[col["timestamp"]], errors="coerce")
    work = work.dropna(subset=[col["timestamp"]])

    work[col["remaining_seat"]] = pd.to_numeric(work[col["remaining_seat"]], errors="coerce")
    work = work.dropna(
        subset=[col["remaining_seat"], col["veh_id"], col["route_id"], col["station_id"]]
    )

    # 타입 정리
    work[col["route_id"]] = clean_id_series(work[col["route_id"]])
    work[col["veh_id"]] = clean_id_series(work[col["veh_id"]])
    work[col["station_id"]] = clean_id_series(work[col["station_id"]])

    # full flag
    if col["full_flag"] is not None:
        work[col["full_flag"]] = pd.to_numeric(work[col["full_flag"]], errors="coerce").fillna(0)
        work["is_full_vehicle"] = (work[col["full_flag"]] == 1).astype(int)
    else:
        work["is_full_vehicle"] = (work[col["remaining_seat"]] <= 0).astype(int)

    work["hour"] = work[col["timestamp"]].dt.hour
    work["date"] = work[col["timestamp"]].dt.date

    # merge
    work = work.merge(
        route_info,
        left_on=col["route_id"],
        right_on="busRouteId",
        how="left"
    )

    # merge 확인용
    unmatched_count = work["busRouteNm"].isna().sum()
    print(f"[merge 확인] busRouteNm 매핑 실패 행 수: {unmatched_count}")

    # route label
    work["route_nm_for_analysis"] = work["busRouteNm"].fillna(work[col["route_id"]].astype(str))
    work["route_label"] = work["route_nm_for_analysis"].astype(str) + "번"

    # station label
    if col["station_name"] is None:
        work["station_name_for_analysis"] = work[col["station_id"]].astype(str)
    else:
        work["station_name_for_analysis"] = work[col["station_name"]].astype(str)

    # 차량-노선 조합
    work["veh_route"] = (
        work[col["veh_id"]].astype(str) + " | " + work["route_label"].astype(str)
    )

    # 차량-노선 diff
    work = work.sort_values(["veh_route", col["timestamp"]]).reset_index(drop=True)
    work["seat_diff_veh_route"] = work.groupby("veh_route")[col["remaining_seat"]].diff()

    return work


# =========================================================
# 5. 요약
# =========================================================
def make_summary(work, col):
    summary = pd.DataFrame({
        "item": [
            "전체 행 수",
            "전체 차량 수",
            "전체 노선 수",
            "전체 차량-노선 조합 수",
            "전체 정류소 수",
            "잔여 좌석 최소값",
            "잔여 좌석 최대값",
        ],
        "value": [
            len(work),
            work[col["veh_id"]].nunique(),
            work[col["route_id"]].nunique(),
            work["veh_route"].nunique(),
            work[col["station_id"]].nunique(),
            work[col["remaining_seat"]].min(),
            work[col["remaining_seat"]].max(),
        ]
    })
    save_table(summary, "0_extra_summary.csv")


# =========================================================
# 6. 차량-노선 시간대 혼잡 분석
# =========================================================
def analyze_veh_route_time_pattern(work, col):
    veh_route_hour = (
        work.groupby(["veh_route", "hour"])
        .agg(
            min_remaining_seat=(col["remaining_seat"], "min"),
            full_count=("is_full_vehicle", "sum"),
        )
        .reset_index()
    )

    veh_route_hour["is_congested"] = (
        (veh_route_hour["min_remaining_seat"] <= 3) |
        (veh_route_hour["full_count"] > 0)
    ).astype(int)

    save_table(veh_route_hour, "1_veh_route_hour_summary.csv")

    veh_route_congestion = (
        veh_route_hour.groupby("veh_route")
        .agg(
            congested_hour_count=("is_congested", "sum"),
            observed_hour_count=("hour", "nunique"),
            min_remaining_seat_overall=("min_remaining_seat", "min"),
        )
        .reset_index()
    )

    veh_route_congestion["congested_hour_ratio"] = (
        veh_route_congestion["congested_hour_count"] / veh_route_congestion["observed_hour_count"]
    )

    veh_route_congestion = veh_route_congestion.sort_values(
        ["congested_hour_count", "congested_hour_ratio", "min_remaining_seat_overall"],
        ascending=[False, False, True]
    )

    save_table(veh_route_congestion, "2_veh_route_congestion_rank.csv")

    top_veh_route = veh_route_congestion.head(80)["veh_route"].tolist()

    heatmap_df = veh_route_hour[veh_route_hour["veh_route"].isin(top_veh_route)].copy()
    heatmap_matrix = heatmap_df.pivot_table(
        index="veh_route",
        columns="hour",
        values="min_remaining_seat",
        aggfunc="min"
    )

    save_table(heatmap_matrix.reset_index(), "3_veh_route_heatmap_matrix.csv")

    plt.figure(figsize=(16, 14), dpi=120)
    sns.heatmap(heatmap_matrix, cmap="Greys", linewidths=0.2)
    plt.title("차량-노선별 시간대 최소 잔여 좌석 분포 (상위 80개)")
    plt.xlabel("시간")
    plt.ylabel("차량 | 노선")
    save_plot("1_veh_route_hour_heatmap.png")

    top20 = veh_route_congestion.head(20).copy()

    plt.figure(figsize=(12, 8), dpi=120)
    plt.barh(
        top20["veh_route"],
        top20["congested_hour_count"],
        color="#4A4A4A",
        edgecolor="#2F2F2F"
    )
    plt.title("혼잡 시간대가 많은 차량-노선 TOP 20")
    plt.xlabel("혼잡 시간대 수")
    plt.ylabel("차량 | 노선")
    plt.gca().invert_yaxis()
    save_plot("2_top20_veh_route_congested_hours.png")


# =========================================================
# 7. 노선 단위 혼잡도
# =========================================================
def analyze_route_level_pattern(work, col):
    route_hour = (
        work.groupby([col["route_id"], "route_label", "hour"])
        .agg(
            min_remaining_seat=(col["remaining_seat"], "min"),
            full_count=("is_full_vehicle", "sum"),
            vehicle_count=(col["veh_id"], "nunique"),
        )
        .reset_index()
    )

    route_hour["is_congested_hour"] = (
        (route_hour["min_remaining_seat"] <= 3) |
        (route_hour["full_count"] > 0)
    ).astype(int)

    save_table(route_hour, "4_route_hour_summary.csv")

    route_congestion = (
        route_hour.groupby([col["route_id"], "route_label"])
        .agg(
            congested_hour_count=("is_congested_hour", "sum"),
            observed_hour_count=("hour", "nunique"),
            min_remaining_seat_overall=("min_remaining_seat", "min"),
        )
        .reset_index()
    )

    route_congestion["congested_hour_ratio"] = (
        route_congestion["congested_hour_count"] / route_congestion["observed_hour_count"]
    )

    route_congestion = route_congestion.sort_values(
        ["congested_hour_ratio", "congested_hour_count", "min_remaining_seat_overall"],
        ascending=[False, False, True]
    )

    save_table(route_congestion, "5_route_congestion_rank.csv")

    plt.figure(figsize=(12, 8), dpi=120)
    plt.barh(
        route_congestion["route_label"],
        route_congestion["congested_hour_ratio"],
        color="#6B6B6B",
        edgecolor="#2F2F2F"
    )
    plt.title("노선별 혼잡 시간 비율")
    plt.xlabel("혼잡 시간 비율")
    plt.ylabel("노선")
    plt.gca().invert_yaxis()
    save_plot("3_route_congested_hour_ratio.png")


# =========================================================
# 8. 차량-노선 diff 분석
# =========================================================
def analyze_veh_route_diff(work, col):
    diff_df = work.dropna(subset=["seat_diff_veh_route"]).copy()
    diff_df["seat_diff_negative"] = diff_df["seat_diff_veh_route"].where(
        diff_df["seat_diff_veh_route"] < 0, 0
    )
    diff_df["seat_drop_abs"] = diff_df["seat_diff_negative"].abs()

    veh_route_diff = (
        diff_df.groupby("veh_route")
        .agg(
            total_drop=("seat_drop_abs", "sum"),
            drop_event_count=("seat_drop_abs", lambda x: (x > 0).sum()),
            max_single_drop=("seat_diff_negative", "min"),
        )
        .reset_index()
        .sort_values(["total_drop", "drop_event_count"], ascending=[False, False])
    )

    save_table(veh_route_diff, "6_veh_route_diff_rank.csv")

    top20_total_drop = veh_route_diff.head(20).copy()

    plt.figure(figsize=(12, 8), dpi=120)
    plt.barh(
        top20_total_drop["veh_route"],
        top20_total_drop["total_drop"],
        color="#4A4A4A",
        edgecolor="#2F2F2F"
    )
    plt.title("좌석 감소량이 큰 차량-노선 TOP 20")
    plt.xlabel("총 좌석 감소량")
    plt.ylabel("차량 | 노선")
    plt.gca().invert_yaxis()
    save_plot("4_top20_veh_route_total_drop.png")

    rapid_drop = veh_route_diff[veh_route_diff["max_single_drop"] <= -5].copy()
    rapid_drop = rapid_drop.sort_values(
        ["drop_event_count", "max_single_drop"],
        ascending=[False, True]
    )

    save_table(rapid_drop, "7_veh_route_rapid_drop_rank.csv")

    rapid_top20 = rapid_drop.head(20).copy()

    if len(rapid_top20) > 0:
        plt.figure(figsize=(12, 8), dpi=120)
        plt.barh(
            rapid_top20["veh_route"],
            rapid_top20["drop_event_count"],
            color="#6B6B6B",
            edgecolor="#2F2F2F"
        )
        plt.title("급격한 좌석 감소가 많은 차량-노선 TOP 20")
        plt.xlabel("급격한 감소 발생 횟수")
        plt.ylabel("차량 | 노선")
        plt.gca().invert_yaxis()
        save_plot("5_top20_veh_route_rapid_drop.png")


# =========================================================
# 9. 메인
# =========================================================
def main():
    print(f"[1] 데이터 불러오는 중: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    print("[2] 컬럼 탐지 중")
    col = detect_columns(df)
    print("탐지된 컬럼:", col)

    print("[3] 노선 정보 불러오는 중")
    route_info = load_route_info()
    print("노선 정보 shape:", route_info.shape)
    print(route_info.head())

    print("[4] 전처리 중")
    work = preprocess(df, col, route_info=route_info)
    print("전처리 후 shape:", work.shape)

    print("[5] 요약 저장")
    make_summary(work, col)

    print("[6] 차량-노선 시간대 혼잡 분석")
    analyze_veh_route_time_pattern(work, col)

    print("[7] 노선 단위 혼잡 분석")
    analyze_route_level_pattern(work, col)

    print("[8] 차량-노선 diff 분석")
    analyze_veh_route_diff(work, col)

    print("\n완료")
    print(f"- 그래프 저장: {PLOT_DIR}")
    print(f"- 표 저장: {TABLE_DIR}")


if __name__ == "__main__":
    main()