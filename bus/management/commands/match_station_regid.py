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

        station_path = os.path.join(base_dir, "bus_station_with_city.csv")
        zone_path = os.path.join(base_dir, "fcst_zone_regid_regname.csv")
        station_id_path = os.path.join(
            base_dir, "bus_station_coordinate_stationId_260406.csv"
        )

        today_str = datetime.now().strftime("%y%m%d")

        output_path = os.path.join(base_dir, f"bus_station_with_regid_{today_str}.csv")
        unmatched_path = os.path.join(
            base_dir, f"bus_station_unmatched_{today_str}.csv"
        )

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

        # 좌표 컬럼 숫자형 변환
        station_df["위도"] = pd.to_numeric(station_df["위도"], errors="coerce")
        station_df["경도"] = pd.to_numeric(station_df["경도"], errors="coerce")
        station_id_df["위도"] = pd.to_numeric(station_id_df["위도"], errors="coerce")
        station_id_df["경도"] = pd.to_numeric(station_id_df["경도"], errors="coerce")

        # 소수점 오차 방지용 반올림 키
        station_df["lat_key"] = station_df["위도"].round(6)
        station_df["lon_key"] = station_df["경도"].round(6)
        station_id_df["lat_key"] = station_id_df["위도"].round(6)
        station_id_df["lon_key"] = station_id_df["경도"].round(6)

        # stationId용 중복 제거
        station_id_df = station_id_df[
            ["stationId", "arsId", "stNm", "lat_key", "lon_key"]
        ].drop_duplicates()

        station_df["is_virtual"] = (
            station_df["stNm"].astype(str).str.contains("가상", na=False).astype(int)
        )

        # ---------------------------
        # regName -> regId 매핑
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

            # 특별시/광역시/특별자치시/도
            if first in metro_map:
                return metro_map[first]

            # 도 단위면 두 번째 단어 사용
            if first in province_set:
                if len(parts) >= 2:
                    second = parts[1]
                    second = second.replace("특별자치시", "")
                    second = second.replace("특별자치도", "")
                    second = second.rstrip("시군")
                    return second
                return None

            # 이미 "성남", "서울" 형태면 그대로 사용
            cleaned = text.replace("특별자치시", "").replace("특별자치도", "")
            return cleaned

        station_df["fcst_name"] = station_df["city"].apply(extract_fcst_name)
        station_df["regId"] = station_df["fcst_name"].map(reg_map)

        # ---------------------------
        # stationId 붙이기
        # stNm + 위도 + 경도 기준 매칭
        # ---------------------------
        station_df = station_df.merge(
            station_id_df, how="left", on=["stNm", "lat_key", "lon_key"]
        )

        # arsId 통일 (station_id 기준 우선)
        if "arsId_x" in station_df.columns and "arsId_y" in station_df.columns:
            station_df["arsId"] = station_df["arsId_y"].combine_first(
                station_df["arsId_x"]
            )
            station_df = station_df.drop(columns=["arsId_x", "arsId_y"])

        # 혹시 하나만 있을 경우 대비
        elif "arsId_y" in station_df.columns:
            station_df = station_df.rename(columns={"arsId_y": "arsId"})
        elif "arsId_x" in station_df.columns:
            station_df = station_df.rename(columns={"arsId_x": "arsId"})

        station_df["arsId"] = pd.to_numeric(
            station_df["arsId"], errors="coerce"
        ).astype("Int64")

        # ---------------------------
        # 정리
        # ---------------------------
        drop_cols = ["fcst_name", "city", "lat_key", "lon_key"]
        station_df = station_df.drop(
            columns=[c for c in drop_cols if c in station_df.columns]
        )

        # stationId를 맨 앞으로 이동
        if "stationId" in station_df.columns:
            cols = ["stationId"] + [c for c in station_df.columns if c != "stationId"]
            station_df = station_df[cols]
        # 결과 저장
        station_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # regId 또는 stationId 둘 중 하나라도 없는 것 따로 저장
        unmatched_df = station_df[
            station_df["regId"].isna() | station_df["stationId"].isna()
        ].copy()
        unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

        matched_reg_count = station_df["regId"].notna().sum()
        matched_stationid_count = station_df["stationId"].notna().sum()
        total_count = len(station_df)

        self.stdout.write(
            self.style.SUCCESS(
                f"전체 행 수: {total_count}\n"
                f"regId 매칭: {matched_reg_count}/{total_count}\n"
                f"stationId 매칭: {matched_stationid_count}/{total_count}\n"
                f"결과 파일: {output_path}\n"
                f"미매칭 파일: {unmatched_path}"
            )
        )
