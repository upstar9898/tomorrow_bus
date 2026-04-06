import os
import glob
import math
import pandas as pd
from django.core.management.base import BaseCommand
from datetime import datetime


class Command(BaseCommand):
    help = "bus_station_with_regid 최신 파일과 weather_regioncode.csv를 매칭해 regId/FCT_ID 범위 내에서 가장 가까운 STN을 stn 컬럼에 저장한다."

    def handle(self, *args, **kwargs):
        # 프로젝트 루트
        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        )

        # 1) 최신 정류소 파일 찾기
        station_pattern = os.path.join(base_dir, "bus_station_with_regid_*.csv")
        station_files = glob.glob(station_pattern)

        if not station_files:
            self.stdout.write(
                self.style.ERROR("bus_station_with_regid_*.csv 파일이 프로젝트 루트에 없습니다.")
            )
            return

        station_path = max(station_files, key=os.path.getmtime)

        # 2) weather csv 파일
        weather_path = os.path.join(base_dir, "weather_regioncode.csv")
        if not os.path.exists(weather_path):
            self.stdout.write(
                self.style.ERROR(f"weather_regioncode.csv 파일이 없습니다: {weather_path}")
            )
            return

        today_str = datetime.now().strftime("%y%m%d")
        output_path = os.path.join(base_dir, f"bus_station_with_stn_{today_str}.csv")
        unmatched_path = os.path.join(base_dir, f"bus_station_stn_unmatched_{today_str}.csv")

        self.stdout.write(f"사용할 정류소 파일: {station_path}")
        self.stdout.write(f"사용할 weather 파일: {weather_path}")

        # 3) 버스 정류소 읽기
        station_df = pd.read_csv(station_path, encoding="utf-8-sig")

        station_df["위도"] = pd.to_numeric(station_df["위도"], errors="coerce")
        station_df["경도"] = pd.to_numeric(station_df["경도"], errors="coerce")

        if "regId" in station_df.columns:
            station_df["regId"] = station_df["regId"].astype(str).str.strip()
        else:
            station_df["regId"] = None

        # 4) weather csv 읽기
        weather_df = pd.read_csv(weather_path, encoding="utf-8-sig")

        required_cols = ["STN", "LON", "LAT", "FCT_ID"]
        for col in required_cols:
            if col not in weather_df.columns:
                self.stdout.write(
                    self.style.ERROR(f"weather_regioncode.csv에 {col} 컬럼이 없습니다.")
                )
                return

        weather_df["STN"] = pd.to_numeric(weather_df["STN"], errors="coerce")
        weather_df["LON"] = pd.to_numeric(weather_df["LON"], errors="coerce")
        weather_df["LAT"] = pd.to_numeric(weather_df["LAT"], errors="coerce")
        weather_df["FCT_ID"] = weather_df["FCT_ID"].astype(str).str.strip()

        weather_df = weather_df.dropna(subset=["STN", "LON", "LAT"]).copy()

        if weather_df.empty:
            self.stdout.write(
                self.style.ERROR("weather_regioncode.csv에서 사용할 좌표 데이터가 없습니다.")
            )
            return

        weather_df["STN"] = weather_df["STN"].astype(int)

        # 5) 거리 계산 함수
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0

            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)

            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            return R * c

        # 6) FCT_ID별 후보 묶기
        weather_by_fct = {
            fct_id: g[["STN", "LON", "LAT"]].to_dict("records")
            for fct_id, g in weather_df.groupby("FCT_ID")
        }

        all_weather_records = weather_df[["STN", "LON", "LAT"]].to_dict("records")

        # 7) 가장 가까운 STN 찾기
        stn_list = []
        distance_km_list = []

        for _, row in station_df.iterrows():
            bus_lat = row["위도"]
            bus_lon = row["경도"]
            reg_id = str(row["regId"]).strip() if pd.notna(row["regId"]) else ""

            if pd.isna(bus_lat) or pd.isna(bus_lon) or (bus_lat == 0 and bus_lon == 0):
                stn_list.append(None)
                distance_km_list.append(None)
                continue

            # 1차: regId와 같은 FCT_ID만 후보
            candidate_records = weather_by_fct.get(reg_id, [])

            # 2차 fallback: 같은 regId 후보가 없으면 전국 전체
            if not candidate_records:
                candidate_records = all_weather_records

            best_stn = None
            best_dist = None

            for w in candidate_records:
                dist = haversine(bus_lat, bus_lon, w["LAT"], w["LON"])
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_stn = w["STN"]

            stn_list.append(best_stn)
            distance_km_list.append(round(best_dist, 4) if best_dist is not None else None)

        # 8) 컬럼 추가
        station_df["stn"] = stn_list
        station_df["distance_km"] = distance_km_list

        # stationId, stn 앞으로
        front_cols = []
        if "stationId" in station_df.columns:
            front_cols.append("stationId")
        if "stn" in station_df.columns:
            front_cols.append("stn")

        remain_cols = [c for c in station_df.columns if c not in front_cols]
        station_df = station_df[front_cols + remain_cols]

        # 9) 저장용 DataFrame 생성 (.0 방지)
        save_df = station_df.copy()

        def int_str_or_blank(v):
            if pd.isna(v):
                return ""
            return str(int(v))

        for col in ["stationId", "stn", "arsId"]:
            if col in save_df.columns:
                save_df[col] = pd.to_numeric(save_df[col], errors="coerce")
                save_df[col] = save_df[col].apply(int_str_or_blank)

        # 10) 저장
        save_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        unmatched_df = save_df[save_df["stn"] == ""].copy()
        unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

        matched_count = (save_df["stn"] != "").sum()
        total_count = len(save_df)

        self.stdout.write(
            self.style.SUCCESS(
                f"전체 행 수: {total_count}\n"
                f"stn 매칭: {matched_count}/{total_count}\n"
                f"결과 파일: {output_path}\n"
                f"미매칭 파일: {unmatched_path}"
            )
        )