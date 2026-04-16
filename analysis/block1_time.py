"""
block1_time.py
==============
시간 기반 패턴 분석

[하는 일]
  1-1. 시간대별 평균 잔여좌석 (노선별)
  1-2. 시간대별 혼잡 단계 분포 (스택 바차트)
  1-3. 요일 × 시간대 혼잡 히트맵 (노선별)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.utils import (
    load_data, set_plot_style, save_plot, save_table,
    ROUTE_ORDER, CONGESTION_COLORS, CONGESTION_ORDER
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

set_plot_style()

WEEKDAY_MAP = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
WEEKDAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]


# ================================================================
# 1-1. 시간대별 평균 잔여좌석 (노선별)
# ================================================================
def plot_hourly_remaining_seat(df: pd.DataFrame):
    print("\n[1-1] 시간대별 평균 잔여좌석 (노선별)")

    hourly = (
        df.groupby(["route_name", "hour"])["remaining_seat"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)
    axes = axes.flatten()

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = hourly[hourly["route_name"] == route]
        ax.plot(data["hour"], data["remaining_seat"],
                marker="o", linewidth=2, color="#2196F3", markersize=4)
        ax.fill_between(data["hour"], data["remaining_seat"], alpha=0.15, color="#2196F3")
        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xlabel("시간")
        ax.set_ylabel("평균 잔여좌석")
        ax.set_xticks(range(5, 24))
        ax.axvspan(6, 10, alpha=0.07, color="red", label="출근")
        ax.axvspan(17, 21, alpha=0.07, color="orange", label="퇴근")
        ax.grid(True, alpha=0.4)

    # 마지막 칸에 범례
    axes[-1].axis("off")
    axes[-1].plot([], [], color="red", alpha=0.4, linewidth=8, label="출근(06~10시)")
    axes[-1].plot([], [], color="orange", alpha=0.4, linewidth=8, label="퇴근(17~21시)")
    axes[-1].legend(loc="center", fontsize=11)

    fig.suptitle("시간대별 평균 잔여좌석 (노선별)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block1_1_hourly_remaining_seat.png")

    save_table(hourly, "block1_1_hourly_remaining_seat.csv")


# ================================================================
# 1-2. 시간대별 혼잡 단계 분포 (노선별 스택 바차트)
# ================================================================
def plot_hourly_congestion(df: pd.DataFrame):
    print("\n[1-2] 시간대별 혼잡 단계 분포")

    hourly_cong = (
        df.groupby(["route_name", "hour", "congestion_level"])
        .size()
        .reset_index(name="count")
    )
    hourly_total = df.groupby(["route_name", "hour"]).size().reset_index(name="total")
    hourly_cong = hourly_cong.merge(hourly_total, on=["route_name", "hour"])
    hourly_cong["ratio"] = hourly_cong["count"] / hourly_cong["total"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)
    axes = axes.flatten()

    hours = sorted(df["hour"].unique())

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = hourly_cong[hourly_cong["route_name"] == route]
        bottom = np.zeros(len(hours))

        for level in CONGESTION_ORDER:
            vals = []
            for h in hours:
                row = data[(data["hour"] == h) & (data["congestion_level"] == level)]
                vals.append(row["ratio"].values[0] if len(row) > 0 else 0)

            ax.bar(hours, vals, bottom=bottom,
                   color=CONGESTION_COLORS[level], label=level, width=0.8)
            bottom += np.array(vals)

        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xlabel("시간")
        ax.set_ylabel("비율")
        ax.set_xticks(range(5, 24, 2))
        ax.set_ylim(0, 1.05)
        ax.axvline(x=6, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axvline(x=10, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axvline(x=17, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axvline(x=21, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    axes[-1].axis("off")
    from matplotlib.patches import Patch
    legend_elements = [
    Patch(facecolor=CONGESTION_COLORS[level], label=level)
    for level in CONGESTION_ORDER
    ]
    axes[-1].legend(
    handles=legend_elements,
    title="혼잡 단계",
    loc="center",
    fontsize=11
    )




    fig.suptitle("시간대별 혼잡 단계 분포 (노선별)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block1_2_hourly_congestion.png")

    save_table(hourly_cong, "block1_2_hourly_congestion.csv")


# ================================================================
# 1-3. 요일 × 시간대 혼잡 히트맵 (노선별)
# ================================================================
def plot_weekday_hour_heatmap(df: pd.DataFrame):
    print("\n[1-3] 요일 × 시간대 혼잡 히트맵")

    df = df.copy()
    df["weekday_kr"] = df["dayofweek"].map(WEEKDAY_MAP)

    # 혼잡+만차 비율
    df["is_congested"] = (df["congestion_level"].isin(["매우혼잡", "혼잡", "만차"])).astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    hours = list(range(5, 24))

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        rdf = df[df["route_name"] == route]

        pivot = (
            rdf.groupby(["weekday_kr", "hour"])["is_congested"]
            .mean()
            .unstack(fill_value=0)
        )
        pivot = pivot.reindex(index=WEEKDAY_ORDER, columns=hours, fill_value=0)

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0, vmax=pivot.values.max() if pivot.values.max() > 0 else 0.1
        )

        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xticks(range(len(hours)))
        ax.set_xticklabels(hours, fontsize=7)
        ax.set_yticks(range(len(WEEKDAY_ORDER)))
        ax.set_yticklabels(WEEKDAY_ORDER, fontsize=9)
        ax.set_xlabel("시간")
        plt.colorbar(im, ax=ax, shrink=0.8)

    axes[-1].axis("off")
    fig.suptitle("요일 × 시간대 혼잡 비율 히트맵 (노선별)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block1_3_weekday_hour_heatmap.png")


# ================================================================
# MAIN
# ================================================================
def main():
    df = load_data(version="v1")

    plot_hourly_remaining_seat(df)
    plot_hourly_congestion(df)
    plot_weekday_hour_heatmap(df)

    print("\n✅ Block 1 완료")
    print("결과 저장 위치: data/eda_output/")


if __name__ == "__main__":
    main()