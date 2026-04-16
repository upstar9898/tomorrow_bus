"""
block0_overview.py
==================
데이터 개요 확인

[하는 일]
  1. 전체 데이터 기본 현황 출력
  2. 노선별 현황 테이블 저장
  3. 날짜별 수집량 확인 (누락일 체크)
  4. 혼잡 단계 분포 확인
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.utils import (
    load_data, set_plot_style, save_plot, save_table, ROUTE_ORDER, CONGESTION_COLORS
)
import matplotlib.pyplot as plt
import pandas as pd

set_plot_style()


def overview_summary(df: pd.DataFrame):
    """전체 기본 현황 출력"""
    print("\n" + "="*50)
    print("1. 전체 기본 현황")
    print("="*50)

    summary = pd.DataFrame({
        "항목": [
            "전체 행 수",
            "분석 기간",
            "노선 수",
            "전체 차량 수",
            "전체 Trip 수",
            "전체 정류소 수",
        ],
        "값": [
            f"{len(df):,}건",
            f"{df['mkTm'].min().date()} ~ {df['mkTm'].max().date()}",
            f"{df['route_name'].nunique()}개",
            f"{df['vehId1'].nunique()}대",
            f"{df['trip_id'].nunique():,}개",
            f"{df['stId'].nunique():,}개",
        ]
    })

    print(summary.to_string(index=False))
    save_table(summary, "block0_summary.csv")


def overview_by_route(df: pd.DataFrame):
    """노선별 현황"""
    print("\n" + "="*50)
    print("2. 노선별 현황")
    print("="*50)

    route_summary = (
        df.groupby("route_name")
        .agg(
            행수=("route_name", "size"),
            차량수=("vehId1", "nunique"),
            Trip수=("trip_id", "nunique"),
            차량당_일평균_Trip수=("trip_id", lambda x: round(
                x.nunique() / df.loc[x.index, "vehId1"].nunique() / df["date"].nunique(), 1
            )),
            평균_잔여좌석=("remaining_seat", lambda x: round(x.mean(), 1)),
        )
        .reindex(ROUTE_ORDER)
        .reset_index()
    )

    print(route_summary.to_string(index=False))
    save_table(route_summary, "block0_route_summary.csv")


def overview_congestion(df: pd.DataFrame):
    """혼잡 단계 분포"""
    print("\n" + "="*50)
    print("3. 혼잡 단계 분포")
    print("="*50)

    total = len(df)
    dist = df["congestion_level"].value_counts()
    for level in ["여유", "보통", "혼잡", "만차"]:
        cnt = dist.get(level, 0)
        print(f"  {level}: {cnt:,}건 ({cnt/total*100:.1f}%)")

    # 노선별 혼잡 단계 분포 바차트
    route_cong = (
        df.groupby(["route_name", "congestion_level"])
        .size()
        .reset_index(name="count")
    )
    route_total = df.groupby("route_name").size().reset_index(name="total")
    route_cong = route_cong.merge(route_total, on="route_name")
    route_cong["ratio"] = route_cong["count"] / route_cong["total"]

    fig, ax = plt.subplots(figsize=(12, 5))

    bottom = {r: 0 for r in ROUTE_ORDER}
    for level in ["여유", "보통", "혼잡", "만차"]:
        vals = []
        for route in ROUTE_ORDER:
            row = route_cong[
                (route_cong["route_name"] == route) &
                (route_cong["congestion_level"] == level)
            ]
            vals.append(row["ratio"].values[0] if len(row) > 0 else 0)

        bars = ax.bar(
            ROUTE_ORDER, vals,
            bottom=[bottom[r] for r in ROUTE_ORDER],
            color=CONGESTION_COLORS[level],
            label=level, width=0.6
        )
        for i, route in enumerate(ROUTE_ORDER):
            bottom[route] += vals[i]

    ax.set_title("노선별 혼잡 단계 분포")
    ax.set_xlabel("노선")
    ax.set_ylabel("비율")
    ax.set_ylim(0, 1.05)
    ax.legend(title="혼잡 단계", bbox_to_anchor=(1.01, 1), loc="upper left")
    save_plot("block0_congestion_by_route.png")


def overview_daily(df: pd.DataFrame):
    """날짜별 수집량 확인"""
    print("\n" + "="*50)
    print("4. 날짜별 수집량")
    print("="*50)

    daily = df.groupby("date").size().reset_index(name="count")
    print(daily.to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(daily["date"], daily["count"], color="#2196F3", width=0.7)
    ax.set_title("날짜별 데이터 수집량")
    ax.set_xlabel("날짜")
    ax.set_ylabel("행 수")
    plt.xticks(rotation=45, ha="right")
    save_plot("block0_daily_count.png")


def main():
    df = load_data(version="v1")

    overview_summary(df)
    overview_by_route(df)
    overview_congestion(df)
    overview_daily(df)

    print("\n✅ Block 0 완료")
    print(f"결과 저장 위치: data/eda_output/")


if __name__ == "__main__":
    main()