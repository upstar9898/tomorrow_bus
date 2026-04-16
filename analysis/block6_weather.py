"""
block6_weather.py
=================
날씨와 잔여좌석/혼잡도의 관계 분석

[하는 일]
  6-1. 날씨 변수 기본 현황 확인
  6-2. 강수 여부별 혼잡 비율 비교 (비 오는 날 vs 맑은 날)
  6-3. 기온 × 잔여좌석 상관관계
  6-4. 강수량 구간별 잔여좌석 분포
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
# 6-1. 날씨 변수 기본 현황
# ================================================================
def overview_weather(df: pd.DataFrame):
    print("\n[6-1] 날씨 변수 기본 현황")

    weather_cols = ["temperature", "precipitation", "rainfall", "fog"]
    available = [c for c in weather_cols if c in df.columns]

    if not available:
        print("  날씨 컬럼이 없습니다.")
        return

    # 날짜별 날씨 (중복 제거 — 하루 하나의 날씨값)
    daily_weather = (
        df.groupby("date")[available]
        .first()
        .reset_index()
    )

    print(f"  날씨 데이터 기간: {daily_weather['date'].min()} ~ {daily_weather['date'].max()}")
    print(f"  총 일수: {len(daily_weather)}일")
    print(f"\n  날씨 변수 기초 통계:")
    print(daily_weather[available].describe().round(2))

    # 강수 발생일 수
    if "rainfall" in daily_weather.columns:
        rain_days = (daily_weather["rainfall"] > 0).sum()
        print(f"\n  강수 발생일: {rain_days}일 / 전체 {len(daily_weather)}일 ({rain_days/len(daily_weather)*100:.1f}%)")

    save_table(daily_weather, "block6_1_daily_weather.csv")


# ================================================================
# 6-2. 강수 여부별 혼잡 비율 비교
# ================================================================
def plot_rain_congestion(df: pd.DataFrame):
    print("\n[6-2] 강수 여부별 혼잡 비율 비교")

    if "rainfall" not in df.columns:
        print("  rainfall 컬럼 없음")
        return

    df = df.copy()
    df["is_rainy"] = (df["rainfall"] > 0).astype(int)
    df["is_congested"] = df["congestion_level"].isin(["혼잡", "만차"]).astype(int)

    # 노선별 강수/비강수 혼잡 비율 비교
    rain_cong = (
        df.groupby(["route_name", "is_rainy"])["is_congested"]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    results = []

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = rain_cong[rain_cong["route_name"] == route]

        dry = data[data["is_rainy"] == 0]["is_congested"].values
        wet = data[data["is_rainy"] == 1]["is_congested"].values

        dry_val = dry[0] if len(dry) > 0 else 0
        wet_val = wet[0] if len(wet) > 0 else 0

        bars = ax.bar(["맑은 날", "비 오는 날"], [dry_val, wet_val],
                      color=["#64B5F6", "#78909C"], edgecolor="white", width=0.5)

        for bar, val in zip(bars, [dry_val, wet_val]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val*100:.1f}%", ha="center", va="bottom", fontsize=10)

        ax.set_title(f"{route}번", fontsize=13)
        ax.set_ylabel("혼잡 비율")
        ax.set_ylim(0, max(dry_val, wet_val) * 1.5 + 0.01)
        ax.grid(True, alpha=0.3, axis="y")

        diff = wet_val - dry_val
        results.append({
            "route_name": route,
            "dry_congestion": round(dry_val, 4),
            "wet_congestion": round(wet_val, 4),
            "diff": round(diff, 4),
            "diff_pct": f"{diff*100:+.1f}%p"
        })

    axes[-1].axis("off")
    legend_elements = [
        mpatches.Patch(facecolor="#64B5F6", label="맑은 날"),
        mpatches.Patch(facecolor="#78909C", label="비 오는 날"),
    ]
    axes[-1].legend(handles=legend_elements, loc="center", fontsize=11)

    fig.suptitle("강수 여부별 혼잡 비율 비교 (노선별)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block6_2_rain_congestion.png")

    results_df = pd.DataFrame(results)
    print(f"\n  강수 여부별 혼잡 비율 차이:")
    print(results_df.to_string(index=False))
    save_table(results_df, "block6_2_rain_congestion.csv")


# ================================================================
# 6-3. 기온 × 잔여좌석 상관관계
# ================================================================
def plot_temp_correlation(df: pd.DataFrame):
    print("\n[6-3] 기온 × 잔여좌석 상관관계")

    if "temperature" not in df.columns:
        print("  temperature 컬럼 없음")
        return

    # 출퇴근 시간대만 (패턴이 뚜렷한 구간)
    peak = df[df["hour"].isin(range(6, 10))].copy()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    corr_results = []

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        data = peak[peak["route_name"] == route][["temperature", "remaining_seat"]].dropna()

        if len(data) < 10:
            ax.axis("off")
            continue

        corr = data["temperature"].corr(data["remaining_seat"])
        corr_results.append({"route_name": route, "correlation": round(corr, 4)})

        ax.scatter(data["temperature"], data["remaining_seat"],
                   alpha=0.1, s=5, color="#2196F3")

        # 추세선
        z = np.polyfit(data["temperature"], data["remaining_seat"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(data["temperature"].min(), data["temperature"].max(), 100)
        ax.plot(x_range, p(x_range), color="#F44336", linewidth=2)

        ax.set_title(f"{route}번  (r={corr:.3f})", fontsize=12)
        ax.set_xlabel("기온 (°C)")
        ax.set_ylabel("잔여좌석")
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("기온 × 잔여좌석 상관관계 (출근시간 6~10시 기준)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block6_3_temp_correlation.png")

    corr_df = pd.DataFrame(corr_results)
    print(f"\n  기온-잔여좌석 상관계수:")
    print(corr_df.to_string(index=False))
    save_table(corr_df, "block6_3_temp_correlation.csv")


# ================================================================
# 6-4. 강수량 구간별 잔여좌석 분포
# ================================================================
def plot_rainfall_seat(df: pd.DataFrame):
    print("\n[6-4] 강수량 구간별 잔여좌석 분포")

    if "rainfall" not in df.columns:
        print("  rainfall 컬럼 없음")
        return

    df = df.copy()

    # 강수량 구간 분류
    df["rain_level"] = pd.cut(
        df["rainfall"],
        bins=[-0.1, 0, 1, 5, 100],
        labels=["없음", "약한 비\n(0~1mm)", "보통 비\n(1~5mm)", "강한 비\n(5mm+)"]
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    rain_labels = ["없음", "약한 비\n(0~1mm)", "보통 비\n(1~5mm)", "강한 비\n(5mm+)"]
    colors = ["#64B5F6", "#4FC3F7", "#0288D1", "#01579B"]

    for i, route in enumerate(ROUTE_ORDER):
        ax = axes[i]
        rdf = df[df["route_name"] == route]

        means = []
        stds = []
        counts = []

        for label in rain_labels:
            group = rdf[rdf["rain_level"] == label]["remaining_seat"]
            means.append(group.mean() if len(group) > 0 else 0)
            stds.append(group.std() if len(group) > 0 else 0)
            counts.append(len(group))

        x = range(len(rain_labels))
        bars = ax.bar(x, means, color=colors, edgecolor="white",
                      yerr=stds, capsize=4, width=0.6)

        ax.set_title(f"{route}번", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(rain_labels, fontsize=8)
        ax.set_ylabel("평균 잔여좌석")
        ax.grid(True, alpha=0.3, axis="y")

        for j, (bar, cnt) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width()/2,
                    0.5, f"n={cnt:,}", ha="center", va="bottom",
                    fontsize=7, color="white", fontweight="bold")

    axes[-1].axis("off")
    fig.suptitle("강수량 구간별 평균 잔여좌석 (오차막대: 표준편차)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_plot("block6_4_rainfall_seat.png")


# ================================================================
# MAIN
# ================================================================
def main():
    df = load_data(version="v1")

    overview_weather(df)
    plot_rain_congestion(df)
    plot_temp_correlation(df)
    plot_rainfall_seat(df)

    print("\n✅ Block 6 완료")
    print("결과 저장 위치: data/eda_output/")


if __name__ == "__main__":
    main()