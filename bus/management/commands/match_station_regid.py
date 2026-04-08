import os
import pandas as pd
from django.core.management.base import BaseCommand
from datetime import datetime


class Command(BaseCommand):
    help = "정류장 city 컬럼에서 예보구역용 도시명을 추출해 regId와 매칭하고 stationId도 붙인다."

    def handle(self, *args, **kwargs):
        # 프로젝트 루트
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        # ✅ data 폴더
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # ---------------------------
        # 입력 파일 (전부 data 폴더)
        # ---------------------------
        station_path = os.path.join(data_dir, "bus_station_with_city.csv")
        zone_path = os.path.join(data_dir, "fcst_zone_regid_regname.csv")
        station_id_path = os.path.join(
            data_dir, "bus_station_coordinate_stationId_260406.csv"
        )

        # 날짜
        today_str = datetime.now().strftime("%y%m%d")

        # ---------------------------
        # 출력 파일 (전부 data 폴더)
        # ---------------------------
        output_path = os.path.join(data_dir, f"bus_station_with_regid_{today_str}.csv")

        unmatched_path = os.path.join(
            data_dir, f"bus_station_unmatched_{today_str}.csv"
        )

        # ---------------------------
        # 데이터 로드
        # ---------------------------
        station_df = pd.read_csv(station_path, encoding="utf-8-sig")
        zone_df = pd.read_csv(zone_path, encoding="utf-8-sig")
        station_id_df = pd.read_csv(station_id_path, encoding="utf-8-sig")

        # ---------------------------
        # 문자열/좌표 정리
        # ---------------------------
        station_df["city"] = station_df["city"].astype(str).str.strip()
        zone_df["regName"] = zone_df["regName"].astype(str).str.strip()

        station_df["stNm"] = station_df["stNm"].astype(str).str.strip()
        station_id_df["stNm"] = station_id_df["stNm"].astype(str).str.strip()

        station_df["위도"] = pd.to_numeric(station_df["위도"], errors="coerce")
        station_df["경도"] = pd.to_numeric(station_df["경도"], errors="coerce")
        station_id_df["위도"] = pd.to_numeric(station_id_df["위도"], errors="coerce")
        station_id_df["경도"] = pd.to_numeric(station_id_df["경도"], errors="coerce")

        # 좌표 매칭 키
        station_df["lat_key"] = station_df["위도"].round(6)
        station_df["lon_key"] = station_df["경도"].round(6)
        station_id_df["lat_key"] = station_id_df["위도"].round(6)
        station_id_df["lon_key"] = station_id_df["경도"].round(6)

        station_id_df = station_id_df[
            ["stationId", "arsId", "stNm", "lat_key", "lon_key"]
        ].drop_duplicates()

        station_df["is_virtual"] = (
            station_df["stNm"].astype(str).str.contains("가상", na=False).astype(int)
        )

        # ---------------------------
        # regId 매핑
        # ---------------------------
        reg_map = dict(zip(zone_df["regName"], zone_df["regId"]))

        metro_map = {
            "서울": "서울",
            "부산": "부산",
            "대구": "대구",
            "인천": "인천",
            "광주": "광주",
            "대전": "대전",
            "울산": "울산",
            "세종특별자치시": "세종",
            "제주특별자치도": "제주",
        }

        province_set = {"경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남"}

        def extract_fcst_name(city_value):
            if pd.isna(city_value):
                return None

            text = str(city_value).strip()

            if not text or text.upper() == "UNKNOWN":
                return None

            parts = text.split()
            if not parts:
                return None

            first = parts[0]

            if first in metro_map:
                return metro_map[first]

            if first in province_set:
                if len(parts) >= 2:
                    second = parts[1]
                    second = second.replace("특별자치시", "")
                    second = second.replace("특별자치도", "")
                    second = second.rstrip("시군")
                    return second
                return None

            cleaned = text.replace("특별자치시", "").replace("특별자치도", "")
            return cleaned

        station_df["fcst_name"] = station_df["city"].apply(extract_fcst_name)
        station_df["regId"] = station_df["fcst_name"].map(reg_map)

        # ---------------------------
        # stationId 매칭
        # ---------------------------
        station_df = station_df.merge(
            station_id_df, how="left", on=["stNm", "lat_key", "lon_key"]
        )

        # arsId 정리
        if "arsId_x" in station_df.columns and "arsId_y" in station_df.columns:
            station_df["arsId"] = station_df["arsId_y"].combine_first(
                station_df["arsId_x"]
            )
            station_df = station_df.drop(columns=["arsId_x", "arsId_y"])
        elif "arsId_y" in station_df.columns:
            station_df = station_df.rename(columns={"arsId_y": "arsId"})
        elif "arsId_x" in station_df.columns:
            station_df = station_df.rename(columns={"arsId_x": "arsId"})

        station_df["arsId"] = pd.to_numeric(
            station_df["arsId"], errors="coerce"
        ).astype("Int64")

        # ---------------------------
        # 컬럼 정리
        # ---------------------------
        drop_cols = ["fcst_name", "city", "lat_key", "lon_key"]
        station_df = station_df.drop(
            columns=[c for c in drop_cols if c in station_df.columns]
        )

        # stationId 맨 앞으로
        if "stationId" in station_df.columns:
            cols = ["stationId"] + [c for c in station_df.columns if c != "stationId"]
            station_df = station_df[cols]

        # ---------------------------
        # 저장
        # ---------------------------
        station_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        unmatched_df = station_df[
            station_df["regId"].isna() | station_df["stationId"].isna()
        ].copy()

        unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

        # 로그
        total_count = len(station_df)

        self.stdout.write(
            self.style.SUCCESS(
                f"전체 행 수: {total_count}\n"
                f"결과 파일: {output_path}\n"
                f"미매칭 파일: {unmatched_path}"
            )
        )
