import os
import csv
import requests
import xml.etree.ElementTree as ET

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "txt의 REG_ID 목록으로 예보구역코드 API를 호출해 regId, regName CSV 저장"

    def handle(self, *args, **kwargs):
        # commands 폴더 기준 -> 프로젝트 루트 경로
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        # data 폴더
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        input_txt_path = os.path.join(data_dir, "reg_id_list.txt")
        output_csv_path = os.path.join(data_dir, "fcst_zone_regid_regname.csv")

        service_key = os.environ.get("WEATHER_ZONE_API_KEY")
        base_url = "https://apis.data.go.kr/1360000/FcstZoneInfoService/getFcstZoneCd"

        # txt에서 REG_ID 읽기
        with open(input_txt_path, "r", encoding="utf-8") as f:
            reg_ids = [line.strip() for line in f if line.strip()]

        result_rows = []

        for reg_id in reg_ids:
            params = {
                "serviceKey": service_key,
                "pageNo": 1,
                "numOfRows": 10,
                "dataType": "XML",
                "regId": reg_id,
            }

            try:
                response = requests.get(base_url, params=params, timeout=20)
                response.raise_for_status()

                root = ET.fromstring(response.text)

                result_code = root.findtext(".//header/resultCode")
                result_msg = root.findtext(".//header/resultMsg")

                if result_code != "00":
                    self.stdout.write(
                        self.style.WARNING(
                            f"[건너뜀] regId={reg_id}, resultCode={result_code}, resultMsg={result_msg}"
                        )
                    )
                    continue

                items = root.findall(".//items/item")

                if not items:
                    self.stdout.write(
                        self.style.WARNING(f"[데이터 없음] regId={reg_id}")
                    )
                    continue

                for item in items:
                    api_reg_id = item.findtext("regId", default="").strip()
                    reg_name = item.findtext("regName", default="").strip()

                    result_rows.append(
                        {
                            "regId": api_reg_id,
                            "regName": reg_name,
                        }
                    )

                self.stdout.write(self.style.SUCCESS(f"[완료] regId={reg_id}"))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"[오류] regId={reg_id}, error={e}"))

        # CSV 저장
        with open(output_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["regId", "regName"])
            writer.writeheader()
            writer.writerows(result_rows)

        self.stdout.write(
            self.style.SUCCESS(
                f"\nCSV 저장 완료: {output_csv_path}\n총 {len(result_rows)}건"
            )
        )
