"""
block2_staord.py
================
정류소 순서(staOrd) × 시간대 패턴 분석 ⭐ 핵심 블록

[하는 일]
  2-1. 노선별 staOrd × 평균 잔여좌석 커브
  2-2. staOrd × 시간대 2D 히트맵 (노선별)
  2-3. 처음 혼잡이 되는 정류소 분포
  2-4. 출발 시 잔여좌석 분포 (staOrd 최솟값 기준)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.utils import (
    load_data, set_plot_style, save_plot, save_table,
    ROUTE_ORDER, CONGESTION_COLORS
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

set_plot_style()


# ================================================================
# 2-1. 노선별 staOrd × 평균 잔여좌석 커브
# ================================================================
def plot_staord_curve(df: pd.DataFrame):
    print("\n[2-1] 노선별 staOrd × 평균 잔여좌석 커브")

    staord_avg = (
        df.groupby(["route_name", "staOrd"])["remaining_seat"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=False)
    axes = axes.flatten()

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = staord_avg[staord_avg["route_name"] == route].sort_values("staOrd")

        ax.plot(data["staOrd"], data["remaining_seat"],
                linewidth=2, color="#2196F3")
        ax.fill_between(data["staOrd"], data["remaining_seat"], alpha=0.15, color="#2196F3")

        # 혼잡 기준선 (새 4단계 기준)
        ax.axhline(y=30, color="#64B5F6", linestyle="--", linewidth=1, alpha=0.8, label="보통 기준 (30석)")
        ax.axhline(y=20, color="#FF9800", linestyle="--", linewidth=1, alpha=0.8, label="혼잡 기준 (20석)")
        ax.axhline(y=0,  color="#F44336", linestyle="--", linewidth=1, alpha=0.8, label="만석 기준 (0석)")

        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xlabel("정류소 순번 (staOrd)")
        ax.set_ylabel("평균 잔여좌석")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=7)

    axes[-1].axis("off")
    fig.suptitle("노선별 정류소 순번 × 평균 잔여좌석 커브", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block2_1_staord_curve.png")
    save_table(staord_avg, "block2_1_staord_avg.csv")


# ================================================================
# 2-2. staOrd × 시간대 2D 히트맵 (노선별)
# ================================================================
def plot_staord_hour_heatmap(df: pd.DataFrame):
    print("\n[2-2] staOrd × 시간대 2D 히트맵 (노선별)")

    hours = list(range(5, 24))

    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    axes = axes.flatten()

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        rdf = df[df["route_name"] == route]

        pivot = (
            rdf.groupby(["staOrd", "hour"])["remaining_seat"]
            .mean()
            .unstack(fill_value=np.nan)
        )
        pivot = pivot.reindex(columns=hours)

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=45,
            origin="lower"
        )

        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xlabel("시간")
        ax.set_ylabel("정류소 순번 (staOrd)")

        # x축: 시간
        ax.set_xticks(range(len(hours)))
        ax.set_xticklabels(hours, fontsize=7)

        # y축: staOrd 일부만 표시
        staords = sorted(pivot.index.tolist())
        step = max(1, len(staords) // 10)
        tick_positions = list(range(0, len(staords), step))
        tick_labels = [staords[p] for p in tick_positions]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)

        plt.colorbar(im, ax=ax, shrink=0.8, label="평균 잔여좌석")

    axes[-1].axis("off")
    fig.suptitle("노선별 정류소 순번 × 시간대 평균 잔여좌석 히트맵\n(초록=여유, 빨강=만차)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block2_2_staord_hour_heatmap.png")


# ================================================================
# 2-3. 처음 혼잡이 되는 정류소 분포
# ================================================================
def plot_first_congested_staord(df: pd.DataFrame):
    print("\n[2-3] 처음 혼잡이 되는 정류소 분포")

    # trip 내에서 처음으로 혼잡(혼잡+만차) 상태가 되는 staOrd 추출
    # 잔여좌석=0 오류값은 utils.py load_data()에서 이미 제거됨
    congested = df[df["congestion_level"].isin(["혼잡", "만차"])].copy()
    first_cong = (
        congested.groupby(["route_name", "trip_id"])["staOrd"]
        .min()
        .reset_index()
        .rename(columns={"staOrd": "first_congested_staord"})
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = first_cong[first_cong["route_name"] == route]["first_congested_staord"]

        if len(data) == 0:
            ax.text(0.5, 0.5, "혼잡 없음", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{route}번", fontsize=13)
            continue

        ax.hist(data, bins=20, color="#FF9800", edgecolor="white", linewidth=0.5)
        ax.axvline(x=data.median(), color="#F44336", linestyle="--",
                   linewidth=1.5, label=f"중간값: {data.median():.0f}번")
        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xlabel("첫 혼잡 정류소 순번")
        ax.set_ylabel("Trip 수")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    axes[-1].axis("off")
    fig.suptitle("Trip별 처음 혼잡이 되는 정류소 순번 분포", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block2_3_first_congested_staord.png")
    save_table(first_cong, "block2_3_first_congested_staord.csv")


# ================================================================
# 2-4. 출발 시 잔여좌석 분포
# ================================================================
def plot_departure_seat(df: pd.DataFrame):
    print("\n[2-4] 출발 시 잔여좌석 분포")

    # trip별 첫 번째 정류소 (staOrd 최솟값) 기준
    departure = (
    df.sort_values(["trip_id", "staOrd"])
    .groupby(["route_name", "trip_id"])
    .first()
    .reset_index()
    [["route_name", "trip_id", "staOrd", "remaining_seat", "hour"]]
    )
    # 잔여좌석=0 오류값은 utils.py load_data()에서 이미 제거됨

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = departure[departure["route_name"] == route]["remaining_seat"]

        ax.hist(data, bins=20, color="#2196F3", edgecolor="white", linewidth=0.5)
        ax.axvline(x=data.median(), color="#F44336", linestyle="--",
                   linewidth=1.5, label=f"중간값: {data.median():.0f}석")
        ax.axvline(x=data.mean(), color="#FF9800", linestyle="--",
                   linewidth=1.5, label=f"평균: {data.mean():.1f}석")
        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xlabel("출발 시 잔여좌석")
        ax.set_ylabel("Trip 수")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    axes[-1].axis("off")
    fig.suptitle("Trip 출발 시 잔여좌석 분포 (staOrd 최솟값 기준, 잔여좌석=0 오류값 제외)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block2_4_departure_seat.png")
    save_table(departure, "block2_4_departure_seat.csv")


# ================================================================
# MAIN
# ================================================================
def main():
    df = load_data(version="v1")

    plot_staord_curve(df)
    plot_staord_hour_heatmap(df)
    plot_first_congested_staord(df)
    plot_departure_seat(df)

    print("\n✅ Block 2 완료")
    print("결과 저장 위치: data/eda_output/")


if __name__ == "__main__":
    main()