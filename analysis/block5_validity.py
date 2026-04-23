"""
block5_validity.py
==================
모델 연결 유효성 검증 분석

[하는 일]
  5-1. 노선별 만차(full_flag) 비율 분석
       → 모델 출력 중 "만차 확률" 예측의 근거
  5-2. (노선, 정류소, 시간대) 조합별 잔여좌석 변동성 분석
       → 예측 난이도 파악 / 모델이 얼마나 어려운 문제를 푸는지 확인
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
# 5-1. 노선별 만차 비율 분석
# ================================================================
def plot_fullness_rate(df: pd.DataFrame):
    print("\n[5-1] 노선별 만차 비율 분석")

    # 전체 만차 비율
    total = len(df)
    full_count = (df["full_flag"] == 1).sum()
    print(f"  전체 만차 비율: {full_count:,}건 / {total:,}건 ({full_count/total*100:.1f}%)")

    # 노선별 만차 비율
    route_full = (
        df.groupby("route_name")["full_flag"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "full_ratio", "sum": "full_count", "count": "total"})
    )
    route_full = route_full.set_index("route_name").reindex(ROUTE_ORDER).reset_index()
    print(f"\n  노선별 만차 비율:")
    print(route_full[["route_name", "full_ratio", "full_count", "total"]].to_string(index=False))

    # 시간대별 × 노선별 만차 비율 히트맵
    hourly_full = (
        df.groupby(["route_name", "hour"])["full_flag"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 왼쪽: 노선별 만차 비율 바차트
    ax = axes[0]
    colors = [
        CONGESTION_COLORS["만차"] if r >= 0.05
        else CONGESTION_COLORS["혼잡"] if r >= 0.01
        else CONGESTION_COLORS["보통"] if r >= 0.005
        else CONGESTION_COLORS["여유"]
        for r in route_full["full_ratio"]
    ]
    bars = ax.bar(route_full["route_name"], route_full["full_ratio"],
                  color=colors, edgecolor="white", width=0.6)
    ax.set_title("노선별 만차(full_flag=1) 비율", fontsize=13)
    ax.set_xlabel("노선")
    ax.set_ylabel("만차 비율")
    ax.set_ylim(0, max(route_full["full_ratio"].max() * 1.3, 0.1))

    for bar, val in zip(bars, route_full["full_ratio"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.grid(True, alpha=0.3, axis="y")

    # 오른쪽: 시간대별 × 노선별 만차 히트맵
    ax2 = axes[1]
    pivot = (
        hourly_full.pivot(index="route_name", columns="hour", values="full_flag")
        .reindex(ROUTE_ORDER)
    )
    hours = list(range(5, 24))
    pivot = pivot.reindex(columns=hours, fill_value=0)

    im = ax2.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax2.set_title("시간대별 만차 비율 히트맵 (노선별)", fontsize=13)
    ax2.set_xticks(range(len(hours)))
    ax2.set_xticklabels(hours, fontsize=8)
    ax2.set_yticks(range(len(ROUTE_ORDER)))
    ax2.set_yticklabels(ROUTE_ORDER)
    ax2.set_xlabel("시간")
    plt.colorbar(im, ax=ax2, shrink=0.8, label="만차 비율")

    plt.tight_layout()
    save_plot("block5_1_fullness_rate.png")
    save_table(route_full, "block5_1_fullness_rate.csv")


# ================================================================
# 5-2. (노선, 정류소, 시간대) 조합별 잔여좌석 변동성
# ================================================================
def plot_seat_variability(df: pd.DataFrame):
    print("\n[5-2] (노선, 정류소, 시간대) 조합별 잔여좌석 변동성")

    # 조합별 통계
    combo_stats = (
        df.groupby(["route_name", "staOrd", "hour"])["remaining_seat"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_seat", "std": "std_seat", "count": "obs"})
    )

    # 관측 수 10건 이상인 조합만 신뢰
    combo_stats = combo_stats[combo_stats["obs"] >= 10].copy()
    combo_stats["cv"] = combo_stats["std_seat"] / (combo_stats["avg_seat"] + 1)  # 변동계수

    print(f"  유효 조합 수: {len(combo_stats):,}개 (관측 10건 이상)")

    # 노선별 변동성 분포
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    route_variability = []

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = combo_stats[combo_stats["route_name"] == route]["std_seat"].dropna()

        if len(data) == 0:
            ax.axis("off")
            continue

        ax.hist(data, bins=30, color="#9C27B0", edgecolor="white", linewidth=0.5)
        ax.axvline(x=data.median(), color="#F44336", linestyle="--",
                   linewidth=1.5, label=f"중간값: {data.median():.1f}석")
        ax.axvline(x=data.mean(), color="#FF9800", linestyle="--",
                   linewidth=1.5, label=f"평균: {data.mean():.1f}석")
        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xlabel("잔여좌석 표준편차 (석)")
        ax.set_ylabel("조합 수")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

        route_variability.append({
            "route_name": route,
            "avg_std": round(data.mean(), 2),
            "median_std": round(data.median(), 2),
            "max_std": round(data.max(), 2),
            "combo_count": len(data)
        })

    axes[-1].axis("off")
    fig.suptitle("(노선, 정류소, 시간대) 조합별 잔여좌석 표준편차 분포\n"
                 "(표준편차가 클수록 예측이 어려운 조합)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block5_2_seat_variability.png")

    variability_df = pd.DataFrame(route_variability)
    print(f"\n  노선별 평균 표준편차:")
    print(variability_df.to_string(index=False))
    save_table(variability_df, "block5_2_variability_summary.csv")
    save_table(combo_stats, "block5_2_combo_stats.csv")

    # 예측 어려운 조합 TOP 20 (표준편차 기준)
    hard_combos = (
        combo_stats.nlargest(20, "std_seat")
        [["route_name", "staOrd", "hour", "avg_seat", "std_seat", "obs"]]
        .round(2)
    )
    print(f"\n  예측 어려운 조합 TOP 20 (표준편차 높은 순):")
    print(hard_combos.to_string(index=False))
    save_table(hard_combos, "block5_2_hard_combos.csv")


# ================================================================
# MAIN
# ================================================================
def main():
    df = load_data(version="all")

    plot_fullness_rate(df)
    plot_seat_variability(df)

    print("\n✅ Block 5 완료")
    print("결과 저장 위치: data/eda_output/")


if __name__ == "__main__":
    main()