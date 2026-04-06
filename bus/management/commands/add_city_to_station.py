import os
import time
import pandas as pd
import requests

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "위도/경도로 도시명 + 전체 주소 추가 후 CSV 저장"

    def handle(self, *args, **kwargs):
        # 프로젝트 루트 경로
        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        )

        input_path = os.path.join(base_dir, "bus_station_coordinate.csv")
        output_path = os.path.join(base_dir, "bus_station_with_city.csv")

        # 👉 카카오 API 키
        KAKAO_API_KEY = os.environ.get("KAKAO_ADMIN_KEY")

        headers = {
            "Authorization": f"KakaoAK {KAKAO_API_KEY}"
        }

        # CSV 읽기
        df = pd.read_csv(input_path)

        city_list = []
        address_list = []   # ✅ 추가

        for idx, row in df.iterrows():
            lat = row["위도"]
            lon = row["경도"]

            url = "https://dapi.kakao.com/v2/local/geo/coord2address.json"

            params = {
                "x": lon,
                "y": lat
            }

            try:
                res = requests.get(url, headers=headers, params=params, timeout=10)
                data = res.json()

                if data["documents"]:
                    addr = data["documents"][0]["address"]

                    # 👉 도시 (시/도)
                    city = addr["region_1depth_name"]

                    # 👉 구
                    district = addr["region_2depth_name"]

                    city_name = f"{city} {district}"

                    # ✅ 전체 주소 (핵심 추가)
                    address_name = addr["address_name"]

                else:
                    city_name = "UNKNOWN"
                    address_name = "UNKNOWN"

            except Exception as e:
                city_name = "ERROR"
                address_name = "ERROR"
                print(f"에러: {e}")

            city_list.append(city_name)
            address_list.append(address_name)  # ✅ 추가

            # 👉 API 제한 대비
            time.sleep(0.1)

            if idx % 50 == 0:
                print(f"{idx} 처리중...")

        # 컬럼 추가
        df["city"] = city_list
        df["address"] = address_list   # ✅ 추가

        # 저장
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        self.stdout.write(
            self.style.SUCCESS(f"완료! 저장 위치: {output_path}")
        )