"""
block4_trip.py
==============
Trip 단위 패턴 분석

[하는 일]
  4-1. Trip별 총 잔여좌석 순감소량 분포 (노선별)
  4-2. Trip 초반 잔여좌석 순감소 속도
  (ACF 분석 제거됨)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.utils import (
    load_data, set_plot_style, save_plot, save_table, ROUTE_ORDER
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

set_plot_style()


# ================================================================
# 4-1. Trip별 총 잔여좌석 순감소량 분포
# ================================================================
def plot_trip_total_boarding(df: pd.DataFrame):
    print("\n[4-1] Trip별 총 잔여좌석 순감소량 분포")

    boarding = df[df["seat_diff"] < 0].copy()
    boarding["seat_decrease"] = boarding["seat_diff"].abs()

    trip_boarding = (
        boarding.groupby(["route_name", "trip_id"])["seat_decrease"]
        .sum()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = trip_boarding[trip_boarding["route_name"] == route]["seat_decrease"]

        if len(data) == 0:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{route}번", fontsize=13)
            continue

        ax.hist(data, bins=30, color="#2196F3", edgecolor="white", linewidth=0.5)
        ax.axvline(x=data.median(), color="#F44336", linestyle="--",
                   linewidth=1.5, label=f"중간값: {data.median():.0f}석")
        ax.axvline(x=data.mean(), color="#FF9800", linestyle="--",
                   linewidth=1.5, label=f"평균: {data.mean():.1f}석")
        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xlabel("Trip당 잔여좌석 순감소량 (석)")
        ax.set_ylabel("Trip 수")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    axes[-1].axis("off")
    fig.suptitle("Trip별 총 잔여좌석 순감소량 분포 (노선별)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block4_1_trip_total_boarding.png")
    save_table(trip_boarding, "block4_1_trip_total_boarding.csv")


# ================================================================
# 4-2. Trip 초반 잔여좌석 순감소 속도
# ================================================================
def plot_seat_drain_speed(df: pd.DataFrame):
    print("\n[4-2] Trip 초반 잔여좌석 순감소 속도")

    # trip 내 처음 5개 관측에서 감소한 좌석 수
    df_sorted = df.sort_values(["trip_id", "staOrd", "mkTm"]).copy()
    df_sorted["obs_rank"] = df_sorted.groupby("trip_id").cumcount()

    first5 = df_sorted[df_sorted["obs_rank"] < 5].copy()
    first5 = first5[first5["seat_diff"] < 0].copy()
    first5["seat_decrease"] = first5["seat_diff"].abs()

    drain_speed = (
        first5.groupby(["route_name", "trip_id"])["seat_decrease"]
        .sum()
        .reset_index()
        .rename(columns={"seat_decrease": "early_seat_decrease"})
    )

    # 시간대별 소진 속도 비교
    trip_hour = (
        df_sorted[df_sorted["obs_rank"] == 0]
        [["trip_id", "hour"]]
        .rename(columns={"hour": "start_hour"})
    )
    drain_speed = drain_speed.merge(trip_hour, on="trip_id", how="left")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = drain_speed[drain_speed["route_name"] == route]

        if len(data) == 0:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{route}번", fontsize=13)
            continue

        hourly_drain = (
            data.groupby("start_hour")["early_seat_decrease"]
            .mean()
            .reset_index()
        )

        ax.bar(hourly_drain["start_hour"], hourly_drain["early_seat_decrease"],
               color="#FF9800", edgecolor="white", width=0.8)
        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xlabel("출발 시간대")
        ax.set_ylabel("초반 5개 정류소\n평균 잔여좌석 순감소량 (석)")
        ax.set_xticks(range(5, 24))
        ax.grid(True, alpha=0.4, axis="y")

    axes[-1].axis("off")
    fig.suptitle("시간대별 Trip 초반(첫 5개 정류소) 평균 잔여좌석 순감소량",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block4_2_seat_drain_speed.png")
    save_table(drain_speed, "block4_2_seat_drain_speed.csv")



# ================================================================
# MAIN
# ================================================================
def main():
    df = load_data(version="all")

    plot_trip_total_boarding(df)
    plot_seat_drain_speed(df)
    print("\n✅ Block 4 완료")
    print("결과 저장 위치: data/eda_output/")


if __name__ == "__main__":
    main()