"""
block3_station.py
=================
정류소별 잔여좌석 순감소량 패턴 분석

[하는 일]
  3-1. 노선별 정류소 잔여좌석 순감소량 TOP 20
  3-2. 시간대별 정류소 유입 히트맵 (주요 정류소 상위 10개)
  3-3. 노선별 혼잡 정류소 순위
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.utils import (
    load_data, set_plot_style, save_plot, save_table,
    ROUTE_ORDER, CONGESTION_COLORS, top_n_title
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

set_plot_style()


# ================================================================
# 3-1. 노선별 정류소 잔여좌석 순감소량 TOP 20
# ================================================================
def plot_boarding_top20(df: pd.DataFrame):
    print("\n[3-1] 노선별 정류소 잔여좌석 순감소량 TOP 20")

    # seat_diff < 0 인 경우만 (잔여좌석 순감소)
    boarding = df[df["seat_diff"] < 0].copy()
    boarding["seat_decrease"] = boarding["seat_diff"].abs()

    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    axes = axes.flatten()

    all_tables = []

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        rdf = boarding[boarding["route_name"] == route]

        if len(rdf) == 0:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{route}번", fontsize=13)
            continue

        station_boarding = (
            rdf.groupby("station_name")["seat_decrease"]
            .sum()
            .reset_index()
            .sort_values("seat_decrease", ascending=False)
        )

        total_stations = station_boarding.shape[0]
        top20 = station_boarding.head(20)

        ax.barh(
            top20["station_name"],
            top20["seat_decrease"],
            color="#2196F3", edgecolor="white"
        )
        ax.invert_yaxis()
        ax.set_title(
            top_n_title(f"{route}번 잔여좌석 순감소량 많은 정류소", 20, total_stations, "개"),
            fontsize=10
        )
        ax.set_xlabel("잔여좌석 순감소량 합계")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.3, axis="x")

        top20["route_name"] = route
        all_tables.append(top20)

    axes[-1].axis("off")
    fig.suptitle("노선별 잔여좌석 순감소량 많은 정류소 TOP 20",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block3_1_boarding_top20.png")

    if all_tables:
        save_table(pd.concat(all_tables), "block3_1_boarding_top20.csv")


# ================================================================
# 3-2. 시간대별 정류소 유입 히트맵 (노선별 상위 10개 정류소)
# ================================================================
def plot_station_hour_heatmap(df: pd.DataFrame):
    print("\n[3-2] 시간대별 정류소 유입 히트맵")

    boarding = df[df["seat_diff"] < 0].copy()
    boarding["seat_decrease"] = boarding["seat_diff"].abs()

    hours = list(range(5, 24))

    fig, axes = plt.subplots(2, 4, figsize=(24, 16))
    axes = axes.flatten()

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        rdf = boarding[boarding["route_name"] == route]

        if len(rdf) == 0:
            ax.axis("off")
            continue

        # 잔여좌석 순감소량 상위 10개 정류소
        top_stations = (
            rdf.groupby("station_name")["seat_decrease"]
            .sum()
            .nlargest(10)
            .index.tolist()
        )

        pivot = (
            rdf[rdf["station_name"].isin(top_stations)]
            .groupby(["station_name", "hour"])["seat_decrease"]
            .sum()
            .unstack(fill_value=0)
            .reindex(columns=hours, fill_value=0)
        )
        # 잔여좌석 순감소량 많은 순으로 정렬
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0
        )

        ax.set_title(f"{route}번 (상위 10개 정류소)", fontsize=11)
        ax.set_xticks(range(len(hours)))
        ax.set_xticklabels(hours, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_xlabel("시간")
        plt.colorbar(im, ax=ax, shrink=0.8, label="잔여좌석 순감소량")

    axes[-1].axis("off")
    fig.suptitle("시간대별 정류소 잔여좌석 순감소량 히트맵 (노선별 상위 10개 정류소)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block3_2_station_hour_heatmap.png")


# ================================================================
# 3-3. 노선별 혼잡 정류소 순위
# ================================================================
def plot_congested_station(df: pd.DataFrame):
    print("\n[3-3] 노선별 혼잡 정류소 순위")

    df = df.copy()
    df["is_congested"] = df["congestion_level"].isin(["혼잡", "만차"]).astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    axes = axes.flatten()

    all_tables = []

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        rdf = df[df["route_name"] == route]

        station_cong = (
            rdf.groupby("station_name")
            .agg(
                congested_ratio=("is_congested", "mean"),
                obs_count=("is_congested", "size")
            )
            .reset_index()
        )

        # 관측 수 30건 이상인 정류소만 (신뢰도 확보)
        station_cong = station_cong[station_cong["obs_count"] >= 30]
        total_stations = len(station_cong)
        top20 = station_cong.nlargest(20, "congested_ratio")

        if len(top20) == 0:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{route}번", fontsize=13)
            continue

        colors = [
            CONGESTION_COLORS["만차"] if r >= 0.3
            else CONGESTION_COLORS["혼잡"] if r >= 0.1
            else CONGESTION_COLORS["보통"] if r >= 0.05
            else CONGESTION_COLORS["여유"]
            for r in top20["congested_ratio"]
        ]

        ax.barh(
            top20["station_name"],
            top20["congested_ratio"],
            color=colors, edgecolor="white"
        )
        ax.invert_yaxis()
        ax.set_title(
            top_n_title(f"{route}번 혼잡 정류소", 20, total_stations, "개"),
            fontsize=10
        )
        ax.set_xlabel("혼잡 비율")
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.3, axis="x")

        top20["route_name"] = route
        all_tables.append(top20)

    axes[-1].axis("off")
    fig.suptitle("노선별 혼잡 비율 높은 정류소 TOP 20\n(관측 30건 이상 정류소 기준)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block3_3_congested_station.png")

    if all_tables:
        save_table(pd.concat(all_tables), "block3_3_congested_station.csv")


# ================================================================
# MAIN
# ================================================================
def main():
    df = load_data(version="v1")

    plot_boarding_top20(df)
    plot_station_hour_heatmap(df)
    plot_congested_station(df)

    print("\n✅ Block 3 완료")
    print("결과 저장 위치: data/eda_output/")


if __name__ == "__main__":
    main()