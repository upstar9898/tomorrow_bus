import os
import pandas as pd
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "정류장 city 컬럼에서 예보구역용 도시명을 추출해 regId와 매칭"

    def handle(self, *args, **kwargs):
        # 프로젝트 루트
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        station_path = os.path.join(base_dir, "bus_station_with_city.csv")
        zone_path = os.path.join(base_dir, "fcst_zone_regid_regname.csv")
        output_path = os.path.join(base_dir, "bus_station_with_regid.csv")
        unmatched_path = os.path.join(base_dir, "bus_station_unmatched.csv")

        station_df = pd.read_csv(station_path, encoding="utf-8-sig")
        zone_df = pd.read_csv(zone_path, encoding="utf-8-sig")

        station_df["city"] = station_df["city"].astype(str).str.strip()
        zone_df["regName"] = zone_df["regName"].astype(str).str.strip()

        # regName 매핑용 dict
        reg_map = dict(zip(zone_df["regName"], zone_df["regId"]))

        # 광역/특별시 계열은 첫 단어 자체를 사용
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

        # 도 단위는 두 번째 단어(시/군)를 사용
        province_set = {"경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남"}

        def extract_fcst_name(city_value):
            """
            bus_station_with_city.csv 의 city 문자열에서
            fcst_zone_regid_regname.csv 의 regName 과 맞는 이름만 뽑는다.
            """
            if pd.isna(city_value):
                return None

            text = str(city_value).strip()

            if not text or text.upper() == "UNKNOWN":
                return None

            parts = text.split()

            if not parts:
                return None

            first = parts[0]

            # 서울, 대구, 대전, 울산, 세종특별자치시 등
            if first in metro_map:
                return metro_map[first]

            # 경기 성남시 수정구 -> 성남
            # 경북 김천시 -> 김천
            # 경기 파주시 -> 파주
            if first in province_set:
                if len(parts) >= 2:
                    second = parts[1]
                    second = second.replace("특별자치시", "")
                    second = second.replace("특별자치도", "")
                    second = second.rstrip("시군")
                    return second
                return None

            # 혹시 city 컬럼이 이미 "성남", "서울" 같은 형태면 그대로 사용
            cleaned = text.replace("특별자치시", "").replace("특별자치도", "")
            return cleaned

        # 예보구역용 도시명 추출
        station_df["fcst_name"] = station_df["city"].apply(extract_fcst_name)

        # regId / regName 붙이기
        station_df["regId"] = station_df["fcst_name"].map(reg_map)
        station_df["regName"] = station_df["fcst_name"]
        station_df["address"] = station_df["address"]

        station_df = station_df.drop(columns=["fcst_name"])
        station_df = station_df.drop(columns=["regName"])
        station_df = station_df.drop(columns=["city"])

        # 결과 저장
        station_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # 미매칭 따로 저장
        unmatched_df = station_df[station_df["regId"].isna()].copy()
        unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

        matched_count = station_df["regId"].notna().sum()
        total_count = len(station_df)

        self.stdout.write(
            self.style.SUCCESS(
                f"매칭 완료: {matched_count}/{total_count}\n"
                f"결과 파일: {output_path}\n"
                f"미매칭 파일: {unmatched_path}"
            )
        )
