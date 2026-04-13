import warnings
from pathlib import Path

import numpy as np
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

BUS_DATA_PATH = BASE_DIR / "data" / "combined"/ "bus_data_v2.csv"
BUS_ROUTE_PATH = BASE_DIR / "data" / "bus_route.csv"
STATION_MASTER_PATH = BASE_DIR / "data" / "bus_station_coordinate_final.csv"

OUTPUT_DIR = BASE_DIR / "data" / "eda_redesign_output_v2"
DATA_DIR = OUTPUT_DIR / "data"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

for d in [OUTPUT_DIR, DATA_DIR, TABLE_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. 설정값
# =========================================================
CONGESTED_SEAT_THRESHOLD = 3
FULL_SEAT_THRESHOLD = 0
SHARP_DROP_THRESHOLD = -5
ROLLING_WINDOW = 3

TOP_N_STATION = 20
TOP_N_ROUTE = 20
TOP_N_VEHICLE = 20
TOP_N_VEH_ROUTE = 25
TOP_N_ROUTE_STATION = 20


# =========================================================
# 2. 시각화 설정
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
        "figure.figsize": (12, 6),
    })
    sns.set_palette(["#4C72B0"])


def set_korean_font():
    font_candidates = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "DejaVu Sans",
    ]
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font_name in font_candidates:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            mpl.rcParams["font.family"] = font_name
            sns.set(font=font_name)
            print(f"[폰트 설정] {font_name}")
            break

    plt.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.unicode_minus"] = False


set_plot_style()
set_korean_font()


# =========================================================
# 3. 유틸
# =========================================================
def save_plot(filename: str):
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_table(df: pd.DataFrame, filename: str):
    df.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")


def normalize_str(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )


def find_col(df: pd.DataFrame, candidates, required=True):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"필수 컬럼을 찾지 못했습니다. 후보={candidates}")
    return None


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =========================================================
# 4. 컬럼 탐지
# =========================================================
def detect_bus_columns(df: pd.DataFrame):
    return {
        "timestamp": find_col(df, ["mkTm", "timestamp", "datetime"]),
        "route_id": find_col(df, ["busRouteId", "route_id", "routeId"]),
        "veh_id": find_col(df, ["vehId1", "veh_id", "vehicle_id"]),
        "station_id": find_col(df, ["stId", "stationId", "station_id"]),
        "ars_id": find_col(df, ["arsId", "ars_id"], required=False),
        "remaining_seat": find_col(df, ["remaining_seat", "reride_Num1", "reride_num1", "rerideNum1"]),
        "full_flag": find_col(df, ["full1", "full_flag"], required=False),
        "staOrd": find_col(df, ["staOrd", "station_order", "stop_order"], required=False),
    }


# =========================================================
# 5. 로드
# =========================================================
def load_data():
    print_section("1. 데이터 로드")

    bus_df = pd.read_csv(BUS_DATA_PATH)
    route_df = pd.read_csv(BUS_ROUTE_PATH)
    station_df = pd.read_csv(STATION_MASTER_PATH)

    print(f"[bus_df] shape={bus_df.shape}")
    print(f"[route_df] shape={route_df.shape}")
    print(f"[station_df] shape={station_df.shape}")

    return bus_df, route_df, station_df


# =========================================================
# 6. bus_route 정리
# =========================================================
def prepare_route_master(route_df: pd.DataFrame):
    print_section("2. 노선 마스터 정리")

    route_id_col = find_col(route_df, ["routeId", "route_id", "busRouteId"])
    route_name_col = find_col(route_df, ["routeName", "route_name", "rtNm"])

    work = route_df[[route_id_col, route_name_col]].copy()
    work.columns = ["route_id", "route_name"]

    work["route_id"] = normalize_str(work["route_id"])
    work["route_name"] = normalize_str(work["route_name"])

    work = work.drop_duplicates(subset=["route_id"])

    print(work.head())
    return work



# =========================================================
# 7. 정류소 마스터 정리
# =========================================================
def explode_station_ids(df: pd.DataFrame, col_name: str):
    """
    stationId에 '277104177|277104178' 같이 들어간 경우 분리
    """
    if col_name not in df.columns:
        return df.copy()

    work = df.copy()
    work[col_name] = work[col_name].astype(str).str.split("|")
    work = work.explode(col_name)
    work[col_name] = normalize_str(work[col_name])

    return work


def prepare_station_master(station_df: pd.DataFrame):
    print_section("3. 정류소 마스터 정리")

    station_id_col = find_col(station_df, ["stationId", "stId", "station_id"])
    ars_id_col = find_col(station_df, ["arsId", "ars_id"], required=False)
    station_name_col = find_col(station_df, ["stNm", "station_name", "stationNm"])
    lat_col = find_col(station_df, ["위도", "lat", "latitude"], required=False)
    lon_col = find_col(station_df, ["경도", "lon", "longitude"], required=False)

    use_cols = [station_id_col, station_name_col]
    if ars_id_col:
        use_cols.append(ars_id_col)
    if lat_col:
        use_cols.append(lat_col)
    if lon_col:
        use_cols.append(lon_col)

    work = station_df[use_cols].copy()
    work = explode_station_ids(work, station_id_col)

    rename_map = {
        station_id_col: "station_id",
        station_name_col: "station_name",
    }
    if ars_id_col:
        rename_map[ars_id_col] = "ars_id"
    if lat_col:
        rename_map[lat_col] = "lat"
    if lon_col:
        rename_map[lon_col] = "lon"

    work = work.rename(columns=rename_map)

    work["station_id"] = normalize_str(work["station_id"])
    work["station_name"] = normalize_str(work["station_name"])

    if "ars_id" in work.columns:
        work["ars_id"] = normalize_str(work["ars_id"])

    work = work.drop_duplicates(subset=["station_id"])

    print(work.head())
    return work


# =========================================================
# 8. 버스 데이터 정리
# =========================================================
def prepare_bus_data(bus_df: pd.DataFrame):
    print_section("4. 버스 데이터 정리")

    cols = detect_bus_columns(bus_df)

    work = bus_df.copy()
    rename_map = {
        cols["timestamp"]: "timestamp",
        cols["route_id"]: "route_id",
        cols["veh_id"]: "veh_id",
        cols["station_id"]: "station_id",
        cols["remaining_seat"]: "remaining_seat",
    }

    if cols["ars_id"]:
        rename_map[cols["ars_id"]] = "ars_id"
    if cols["full_flag"]:
        rename_map[cols["full_flag"]] = "full_flag"
    if cols["staOrd"]:
        rename_map[cols["staOrd"]] = "staOrd"

    work = work.rename(columns=rename_map)

    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["route_id"] = normalize_str(work["route_id"])
    work["veh_id"] = normalize_str(work["veh_id"])
    work["station_id"] = normalize_str(work["station_id"])
    work["remaining_seat"] = pd.to_numeric(work["remaining_seat"], errors="coerce")

    if "ars_id" in work.columns:
        work["ars_id"] = normalize_str(work["ars_id"])
    else:
        work["ars_id"] = np.nan

    if "full_flag" in work.columns:
        work["full_flag"] = pd.to_numeric(work["full_flag"], errors="coerce")
    else:
        work["full_flag"] = np.nan

    if "staOrd" in work.columns:
        work["staOrd"] = pd.to_numeric(work["staOrd"], errors="coerce")
    else:
        work["staOrd"] = np.nan

    before_rows = len(work)
    work = work.dropna(subset=["timestamp", "route_id", "veh_id", "station_id", "remaining_seat"]).copy()

    print(f"[정리 전] {before_rows:,} rows")
    print(f"[정리 후] {len(work):,} rows")

    return work


# =========================================================
# 9. 노선/정류소 매핑
# =========================================================
def map_route_and_station(bus_df: pd.DataFrame, route_master: pd.DataFrame, station_master: pd.DataFrame):
    print_section("5. 노선/정류소 매핑")

    work = bus_df.copy()

    # route 매핑
    work = work.merge(route_master, on="route_id", how="left")
    work["route_name"] = work["route_name"].fillna("UNKNOWN_ROUTE_" + work["route_id"])

    # station_id 기준 매핑
    station_lookup = station_master.copy()
    work = work.merge(
        station_lookup[["station_id", "station_name"] + [c for c in ["ars_id", "lat", "lon"] if c in station_lookup.columns]],
        on="station_id",
        how="left",
        suffixes=("", "_master")
    )

    missing_station = work["station_name"].isna().sum()
    print(f"[1차 station_id 기준 미매핑] {missing_station:,}")

    # ars_id 기준 보정
    if "ars_id" in station_lookup.columns:
        ars_lookup = (
            station_lookup[["ars_id", "station_name"]]
            .dropna(subset=["ars_id"])
            .drop_duplicates(subset=["ars_id"])
            .rename(columns={"station_name": "station_name_by_ars"})
        )

        missing_mask = work["station_name"].isna()
        fill_df = work.loc[missing_mask, ["ars_id"]].merge(
            ars_lookup,
            on="ars_id",
            how="left"
        )
        work.loc[missing_mask, "station_name"] = fill_df["station_name_by_ars"].values

    missing_station = work["station_name"].isna().sum()
    print(f"[2차 ars_id 기준 미매핑] {missing_station:,}")

    work["station_name"] = work["station_name"].fillna("UNKNOWN_" + work["station_id"])

    work["route_display"] = work["route_name"]
    work["station_display"] = work["station_name"] + " (" + work["station_id"] + ")"
    work["veh_route_key"] = work["route_name"] + " | 차량 " + work["veh_id"]

    unknown_route_cnt = work["route_name"].str.startswith("UNKNOWN_ROUTE_", na=False).sum()
    unknown_station_cnt = work["station_name"].str.startswith("UNKNOWN_", na=False).sum()

    print(f"[최종 UNKNOWN route 개수] {unknown_route_cnt:,}")
    print(f"[최종 UNKNOWN station 개수] {unknown_station_cnt:,}")

    return work


# =========================================================
# 10. feature 생성
# =========================================================
def create_features(df: pd.DataFrame):
    print_section("6. feature 생성")

    work = df.copy()

    work["date"] = work["timestamp"].dt.date
    work["hour"] = work["timestamp"].dt.hour
    work["weekday_num"] = work["timestamp"].dt.weekday
    work["weekday"] = work["timestamp"].dt.day_name()

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
    work["is_weekend"] = work["weekday_num"].isin([5, 6]).astype(int)
    work["is_commute_hour"] = work["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # route 내 station별 staOrd 대표값 보정
    if "staOrd" in work.columns:
        mode_sta = (
            work.dropna(subset=["staOrd"])
            .groupby(["route_id", "station_id"])["staOrd"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            .reset_index()
            .rename(columns={"staOrd": "staOrd_mode"})
        )
        work = work.merge(mode_sta, on=["route_id", "station_id"], how="left")
        work["staOrd"] = work["staOrd"].fillna(work["staOrd_mode"])
        work.drop(columns=["staOrd_mode"], inplace=True)

    route_stop_cnt = (
        work.groupby("route_id")["staOrd"]
        .max()
        .reset_index()
        .rename(columns={"staOrd": "route_total_stops"})
    )
    work = work.merge(route_stop_cnt, on="route_id", how="left")

    work["progress_ratio"] = work["staOrd"] / work["route_total_stops"]
    work["is_first_stop"] = (work["staOrd"] == 1).astype(int)
    work["is_last_stop"] = (work["staOrd"] == work["route_total_stops"]).astype(int)

    sort_cols = ["route_id", "veh_id", "timestamp", "staOrd"]
    work = work.sort_values(sort_cols).reset_index(drop=True)

    group_cols = ["route_id", "veh_id"]

    work["remaining_seat_lag1"] = work.groupby(group_cols)["remaining_seat"].shift(1)
    work["remaining_seat_lag2"] = work.groupby(group_cols)["remaining_seat"].shift(2)
    work["remaining_seat_lag3"] = work.groupby(group_cols)["remaining_seat"].shift(3)

    work["seat_diff"] = work["remaining_seat"] - work["remaining_seat_lag1"]

    work["remaining_seat_roll3"] = (
        work.groupby(group_cols)["remaining_seat"]
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    work["seat_diff_roll3"] = (
        work.groupby(group_cols)["seat_diff"]
        .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )

    work["is_congested"] = (work["remaining_seat"] <= CONGESTED_SEAT_THRESHOLD).astype(int)
    work["is_full"] = (work["remaining_seat"] <= FULL_SEAT_THRESHOLD).astype(int)
    work["boarding_est"] = np.where(work["seat_diff"] < 0, -work["seat_diff"], 0)
    work["alighting_est"] = np.where(work["seat_diff"] > 0, work["seat_diff"], 0)
    work["is_sharp_drop"] = (work["seat_diff"] <= SHARP_DROP_THRESHOLD).astype(int)

    work.to_csv(DATA_DIR / "bus_data_structured_for_eda_v2.csv", index=False, encoding="utf-8-sig")
    print(work.head())

    return work


# =========================================================
# 11. 품질 점검
# =========================================================
def make_quality_table(df: pd.DataFrame):
    print_section("7. 데이터 품질 점검")

    quality = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [df[c].isna().sum() for c in df.columns],
        "missing_ratio": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
        "nunique": [df[c].nunique(dropna=True) for c in df.columns],
    }).sort_values(["missing_ratio", "nunique"], ascending=[False, False])

    save_table(quality, "data_quality_summary.csv")

    summary = pd.DataFrame({
        "item": [
            "전체 행 수",
            "전체 노선 수",
            "전체 차량 수",
            "전체 정류소 수",
            "UNKNOWN 노선 수",
            "UNKNOWN 정류소 수",
            "혼잡 비율",
            "만차 비율",
            "급격한 감소 비율",
        ],
        "value": [
            f"{len(df):,}",
            df["route_id"].nunique(),
            df["veh_id"].nunique(),
            df["station_id"].nunique(),
            df["route_name"].str.startswith("UNKNOWN_ROUTE_", na=False).sum(),
            df["station_name"].str.startswith("UNKNOWN_", na=False).sum(),
            round(df["is_congested"].mean(), 4),
            round(df["is_full"].mean(), 4),
            round(df["is_sharp_drop"].mean(), 4),
        ]
    })

    print(summary)
    save_table(summary, "overall_summary.csv")


# =========================================================
# 12. 시간 기반
# =========================================================
def eda_time_based(df: pd.DataFrame):
    print_section("8. EDA - 시간 기반")

    hour_summary = (
        df.groupby("hour")
        .agg(
            avg_remaining_seat=("remaining_seat", "mean"),
            congested_ratio=("is_congested", "mean"),
            full_ratio=("is_full", "mean"),
            boarding_est_sum=("boarding_est", "sum"),
            sharp_drop_count=("is_sharp_drop", "sum"),
        )
        .reset_index()
    )
    save_table(hour_summary, "eda_hour_summary.csv")

    plt.figure(figsize=(13, 6))
    sns.barplot(data=hour_summary, x="hour", y="avg_remaining_seat")
    plt.title("시간대별 평균 잔여 좌석")
    plt.xlabel("hour")
    plt.ylabel("평균 잔여 좌석")
    save_plot("time_avg_remaining_seat.png")

    plt.figure(figsize=(13, 6))
    sns.barplot(data=hour_summary, x="hour", y="boarding_est_sum")
    plt.title("시간대별 추정 탑승 인원 합계")
    plt.xlabel("hour")
    plt.ylabel("추정 탑승 인원")
    save_plot("time_boarding_est_sum.png")

    plt.figure(figsize=(13, 6))
    sns.barplot(data=hour_summary, x="hour", y="congested_ratio")
    plt.title("시간대별 혼잡 비율")
    plt.xlabel("hour")
    plt.ylabel("혼잡 비율")
    save_plot("time_congested_ratio.png")

    plt.figure(figsize=(13, 6))
    sns.barplot(data=hour_summary, x="hour", y="sharp_drop_count")
    plt.title("시간대별 급격한 좌석 감소 발생 횟수")
    plt.xlabel("hour")
    plt.ylabel("건수")
    save_plot("time_sharp_drop_count.png")



def eda_weekday_based(df: pd.DataFrame):
    print_section("9. EDA - 요일 기반")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_label_map = {
        "Monday": "월",
        "Tuesday": "화",
        "Wednesday": "수",
        "Thursday": "목",
        "Friday": "금",
        "Saturday": "토",
        "Sunday": "일",
    }

    weekday_summary = (
        df.groupby("weekday")
        .agg(
            avg_remaining_seat=("remaining_seat", "mean"),
            congested_ratio=("is_congested", "mean"),
            full_ratio=("is_full", "mean"),
            boarding_est_sum=("boarding_est", "sum"),
            sharp_drop_count=("is_sharp_drop", "sum"),
        )
        .reindex(weekday_order)
        .reset_index()
    )

    weekday_summary["weekday_kr"] = weekday_summary["weekday"].map(weekday_label_map)
    save_table(weekday_summary, "eda_weekday_summary.csv")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=weekday_summary, x="weekday_kr", y="congested_ratio")
    plt.title("요일별 혼잡 비율")
    plt.xlabel("요일")
    plt.ylabel("혼잡 비율")
    save_plot("weekday_congested_ratio.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=weekday_summary, x="weekday_kr", y="avg_remaining_seat")
    plt.title("요일별 평균 잔여 좌석")
    plt.xlabel("요일")
    plt.ylabel("평균 잔여 좌석")
    save_plot("weekday_avg_remaining_seat.png")

def eda_weekday_commute_based(df: pd.DataFrame):
    print_section("9. EDA - 요일/출퇴근 시간 기반")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    commute_df = df.copy()

    commute_df["commute_type"] = np.select(
        [
            commute_df["hour"].isin([7, 8, 9]),
            commute_df["hour"].isin([17, 18, 19]),
        ],
        [
            "출근시간대",
            "퇴근시간대",
        ],
        default="기타"
    )

    commute_df = commute_df[commute_df["commute_type"] != "기타"].copy()

    weekday_commute_summary = (
        commute_df.groupby(["weekday", "commute_type"])
        .agg(
            congested_ratio=("is_congested", "mean"),
            avg_remaining_seat=("remaining_seat", "mean"),
            boarding_est_sum=("boarding_est", "sum"),
        )
        .reset_index()
    )

    weekday_commute_summary["weekday"] = pd.Categorical(
        weekday_commute_summary["weekday"],
        categories=weekday_order,
        ordered=True
    )
    weekday_commute_summary = weekday_commute_summary.sort_values(["weekday", "commute_type"])

    save_table(weekday_commute_summary, "eda_weekday_commute_summary.csv")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=weekday_commute_summary,
        x="weekday",
        y="congested_ratio",
        hue="commute_type"
    )
    plt.title("요일별 출퇴근 시간대 혼잡 비율")
    plt.xlabel("요일")
    plt.ylabel("혼잡 비율")
    save_plot("weekday_commute_congested_ratio.png")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=weekday_commute_summary,
        x="weekday",
        y="avg_remaining_seat",
        hue="commute_type"
    )
    plt.title("요일별 출퇴근 시간대 평균 잔여 좌석")
    plt.xlabel("요일")
    plt.ylabel("평균 잔여 좌석")
    save_plot("weekday_commute_avg_remaining_seat.png")


# =========================================================
# 시간대별 좌석 변화 (diff)
# =========================================================
def eda_diff_time_based(df: pd.DataFrame):
    print_section("10. EDA - 시간대별 좌석 변화 (diff)")

    diff_time = (
        df.groupby("hour")
        .agg(
            avg_diff=("seat_diff", "mean"),
            sharp_drop_ratio=("is_sharp_drop", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=diff_time, x="hour", y="avg_diff", marker="o")
    plt.title("시간대별 평균 좌석 변화량 (diff)")
    plt.xlabel("hour")
    plt.ylabel("평균 diff")
    save_plot("time_avg_diff.png")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=diff_time, x="hour", y="sharp_drop_ratio")
    plt.title("시간대별 급격한 좌석 감소 비율")
    plt.xlabel("hour")
    plt.ylabel("급감 비율")
    save_plot("time_sharp_drop_ratio.png")

    save_table(diff_time, "eda_time_diff_summary.csv")


# =========================================================
# 차량별 좌석 급감 패턴
# =========================================================
def eda_diff_vehicle_based(df: pd.DataFrame):
    print_section("11. EDA - 차량별 좌석 급감 패턴")

    veh_diff = (
        df.groupby("veh_id")
        .agg(
            sharp_drop_ratio=("is_sharp_drop", "mean"),
            avg_diff=("seat_diff", "mean"),
        )
        .reset_index()
        .sort_values("sharp_drop_ratio", ascending=False)
    )

    top_veh = veh_diff.head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_veh, y="veh_id", x="sharp_drop_ratio")
    plt.title("차량별 급격한 좌석 감소 비율 TOP 20")
    plt.xlabel("급감 비율")
    plt.ylabel("차량 ID")
    save_plot("vehicle_top_sharp_drop_ratio.png")

    save_table(veh_diff, "eda_vehicle_diff_summary.csv")


# =========================================================
# 13. 차량 기반
# =========================================================
def eda_vehicle_based(df: pd.DataFrame):
    print_section("9. EDA - 차량 기반")

    vehicle_summary = (
        df.groupby("veh_id")
        .agg(
            obs_count=("veh_id", "size"),
            route_count=("route_id", "nunique"),
            avg_remaining_seat=("remaining_seat", "mean"),
            congested_ratio=("is_congested", "mean"),
            boarding_est_sum=("boarding_est", "sum"),
            sharp_drop_count=("is_sharp_drop", "sum"),
        )
        .reset_index()
        .sort_values(["congested_ratio", "boarding_est_sum"], ascending=[False, False])
    )
    save_table(vehicle_summary, "eda_vehicle_summary.csv")

    top_vehicle = vehicle_summary.head(TOP_N_VEHICLE).copy()

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_vehicle, y="veh_id", x="congested_ratio")
    plt.title(f"차량별 혼잡 비율 TOP {TOP_N_VEHICLE}")
    plt.xlabel("혼잡 비율")
    plt.ylabel("veh_id")
    save_plot("vehicle_top_congested_ratio.png")


# =========================================================
# 14. 노선 기반
# =========================================================
def eda_route_based(df: pd.DataFrame):
    print_section("10. EDA - 노선 기반")

    route_summary = (
        df.groupby(["route_id", "route_name"])
        .agg(
            vehicle_count=("veh_id", "nunique"),
            station_count=("station_id", "nunique"),
            obs_count=("route_id", "size"),
            avg_remaining_seat=("remaining_seat", "mean"),
            congested_ratio=("is_congested", "mean"),
            full_ratio=("is_full", "mean"),
            boarding_est_sum=("boarding_est", "sum"),
            sharp_drop_count=("is_sharp_drop", "sum"),
        )
        .reset_index()
        .sort_values(["congested_ratio", "boarding_est_sum"], ascending=[False, False])
    )
    save_table(route_summary, "eda_route_summary.csv")

    top_route_congest = route_summary.head(TOP_N_ROUTE).copy()
    top_route_boarding = route_summary.sort_values("boarding_est_sum", ascending=False).head(TOP_N_ROUTE).copy()

    plt.figure(figsize=(13, 8))
    sns.barplot(data=top_route_congest, y="route_name", x="congested_ratio")
    plt.title(f"노선별 혼잡 비율 TOP {TOP_N_ROUTE}")
    plt.xlabel("혼잡 비율")
    plt.ylabel("노선번호")
    save_plot("route_top_congested_ratio.png")

    plt.figure(figsize=(13, 8))
    sns.barplot(data=top_route_boarding, y="route_name", x="boarding_est_sum")
    plt.title(f"노선별 추정 탑승 인원 TOP {TOP_N_ROUTE}")
    plt.xlabel("추정 탑승 인원")
    plt.ylabel("노선번호")
    save_plot("route_top_boarding_est_sum.png")


# =========================================================
# 15. 정류소 기반
# =========================================================
def eda_station_based(df: pd.DataFrame):
    print_section("11. EDA - 정류소 기반")

    station_boarding = (
        df.groupby(["station_id", "station_display"])
        .agg(
            boarding_est_sum=("boarding_est", "sum"),
            sharp_drop_count=("is_sharp_drop", "sum"),
            congested_ratio=("is_congested", "mean"),
            route_count=("route_id", "nunique"),
            obs_count=("station_id", "size"),
        )
        .reset_index()
        .sort_values(["boarding_est_sum", "sharp_drop_count"], ascending=[False, False])
    )
    save_table(station_boarding, "eda_station_boarding_summary.csv")

    top_station_boarding = station_boarding.head(TOP_N_STATION).copy()

    plt.figure(figsize=(13, 8))
    sns.barplot(data=top_station_boarding, y="station_display", x="boarding_est_sum")
    plt.title(f"많이 타는 정류소 TOP {TOP_N_STATION}")
    plt.xlabel("추정 탑승 인원 합계")
    plt.ylabel("정류소")
    save_plot("station_top_boarding.png")

    station_sharp_drop = (
        df.groupby(["station_id", "station_display"])
        .agg(
            sharp_drop_count=("is_sharp_drop", "sum"),
            boarding_est_sum=("boarding_est", "sum"),
            avg_diff=("seat_diff", "mean"),
            min_diff=("seat_diff", "min"),
            obs_count=("station_id", "size"),
        )
        .reset_index()
        .sort_values(["sharp_drop_count", "boarding_est_sum"], ascending=[False, False])
    )
    save_table(station_sharp_drop, "eda_station_sharp_drop_summary.csv")

    top_station_sharp_drop = station_sharp_drop.head(TOP_N_STATION).copy()

    plt.figure(figsize=(13, 8))
    sns.barplot(data=top_station_sharp_drop, y="station_display", x="sharp_drop_count")
    plt.title(f"급격히 감소하는 정류소 TOP {TOP_N_STATION}")
    plt.xlabel("급격한 감소 발생 횟수")
    plt.ylabel("정류소")
    save_plot("station_top_sharp_drop.png")

    # 노선별 혼잡 정류소: route_name 기준으로 표시
    route_station = (
        df.groupby(["route_name", "station_display"])
        .agg(
            congested_ratio=("is_congested", "mean"),
            boarding_est_sum=("boarding_est", "sum"),
            sharp_drop_count=("is_sharp_drop", "sum"),
            obs_count=("station_id", "size"),
        )
        .reset_index()
    )

    route_station = route_station[route_station["obs_count"] >= 5].copy()
    route_station = route_station.sort_values(["congested_ratio", "boarding_est_sum"], ascending=[False, False])
    save_table(route_station, "eda_route_station_congestion_summary.csv")

    top_route_station = route_station.head(TOP_N_ROUTE_STATION).copy()
    top_route_station["route_station_label"] = top_route_station["route_name"] + " | " + top_route_station["station_display"]

    plt.figure(figsize=(14, 9))
    sns.barplot(data=top_route_station, y="route_station_label", x="congested_ratio")
    plt.title(f"노선별 혼잡 정류소 TOP {TOP_N_ROUTE_STATION}")
    plt.xlabel("혼잡 비율")
    plt.ylabel("노선 | 정류소")
    save_plot("route_station_top_congestion.png")


# =========================================================
# 16. 차량-노선 반복 혼잡 패턴
# =========================================================
def eda_vehicle_route_pattern(df: pd.DataFrame):
    print_section("12. EDA - 차량-노선 반복 혼잡 패턴")

    veh_route_summary = (
        df.groupby(["veh_route_key", "route_name", "veh_id"])
        .agg(
            congested_ratio=("is_congested", "mean"),
            full_ratio=("is_full", "mean"),
            boarding_est_sum=("boarding_est", "sum"),
            sharp_drop_count=("is_sharp_drop", "sum"),
            obs_count=("veh_id", "size"),
        )
        .reset_index()
        .sort_values(["congested_ratio", "boarding_est_sum"], ascending=[False, False])
    )
    save_table(veh_route_summary, "eda_vehicle_route_summary.csv")

    top_veh_route = veh_route_summary.head(TOP_N_VEH_ROUTE).copy()

    plt.figure(figsize=(14, 9))
    sns.barplot(data=top_veh_route, y="veh_route_key", x="congested_ratio")
    plt.title(f"차량-노선별 반복 혼잡 비율 TOP {TOP_N_VEH_ROUTE}")
    plt.xlabel("혼잡 비율")
    plt.ylabel("차량-노선")
    save_plot("vehicle_route_top_congested_ratio.png")


    heatmap_df = (
    df.groupby(["veh_route_key", "hour"])
    .agg(avg_remaining_seat=("remaining_seat", "mean"))
    .reset_index()
    )

    rank_key = top_veh_route["veh_route_key"].tolist()
    heatmap_df = heatmap_df[heatmap_df["veh_route_key"].isin(rank_key)].copy()

    pivot_df = heatmap_df.pivot(index="veh_route_key", columns="hour", values="avg_remaining_seat")
    pivot_df = pivot_df.reindex(rank_key)

    plt.figure(figsize=(16, max(8, len(pivot_df) * 0.35)))
    sns.heatmap(pivot_df, cmap="Greys_r", linewidths=0.3, linecolor="white")
    plt.title("차량-노선별 시간대 평균 잔여 좌석 heatmap")
    plt.xlabel("hour")
    plt.ylabel("차량-노선")
    save_plot("vehicle_route_hour_avg_remaining_heatmap.png")

# =========================================================
# 17. 인사이트 자동 생성
# =========================================================
def generate_insight_text(df: pd.DataFrame):
    print_section("13. 인사이트 자동 생성")

    hour_summary = (
        df.groupby("hour")
        .agg(
            congested_ratio=("is_congested", "mean"),
            boarding_est_sum=("boarding_est", "sum"),
        )
        .reset_index()
    )

    peak_congest_hour = hour_summary.sort_values("congested_ratio", ascending=False).iloc[0]
    peak_boarding_hour = hour_summary.sort_values("boarding_est_sum", ascending=False).iloc[0]

    top_station = (
        df.groupby("station_display")["boarding_est"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
        .index[0]
    )

    top_sharp_station = (
        df.groupby("station_display")["is_sharp_drop"]
        .sum()
        .sort_values(ascending=False)
        .head(1)
        .index[0]
    )

    top_route = (
        df.groupby("route_name")["is_congested"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
        .index[0]
    )

    top_veh_route = (
        df.groupby("veh_route_key")["is_congested"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
        .index[0]
    )

    insight_lines = [
        "[시간 기반]",
        f"- 혼잡 비율이 가장 높은 시간대는 {int(peak_congest_hour['hour'])}시이다.",
        f"- 추정 탑승 인원이 가장 많은 시간대는 {int(peak_boarding_hour['hour'])}시이다.",
        "",
        "[정류소 기반]",
        f"- 많이 타는 정류소 1위는 {top_station} 이다.",
        f"- 급격한 좌석 감소가 가장 많이 발생한 정류소는 {top_sharp_station} 이다.",
        "",
        "[노선 기반]",
        f"- 혼잡 비율이 가장 높은 노선은 {top_route} 이다.",
        "",
        "[차량-노선 기반]",
        f"- 반복 혼잡이 가장 강한 차량-노선 조합은 {top_veh_route} 이다.",
    ]

    insight_text = "\n".join(insight_lines)

    with open(TABLE_DIR / "insight_summary.txt", "w", encoding="utf-8") as f:
        f.write(insight_text)

    print(insight_text)


# =========================================================
# 18. 메인
# =========================================================
def main():
    bus_df, route_df, station_df = load_data()

    route_master = prepare_route_master(route_df)
    station_master = prepare_station_master(station_df)
    bus_df = prepare_bus_data(bus_df)

    mapped_df = map_route_and_station(
        bus_df=bus_df,
        route_master=route_master,
        station_master=station_master
    )

    structured_df = create_features(mapped_df)

    make_quality_table(structured_df)

    eda_time_based(structured_df)
    eda_weekday_based(structured_df)
    eda_weekday_commute_based(structured_df)
    eda_diff_time_based(structured_df)
    eda_vehicle_based(structured_df)
    eda_route_based(structured_df)
    eda_station_based(structured_df)
    eda_vehicle_route_pattern(structured_df)
    
    generate_insight_text(structured_df)

    print_section("완료")
    print(f"저장 경로: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

