from django.core.management.base import BaseCommand
import pandas as pd
import re


class Command(BaseCommand):
    help = "weather txt 파일을 CSV로 변환"

    def handle(self, *args, **kwargs):
        input_path = "/bus/tomorrow_bus/output_file.txt"
        output_path = "weather.csv"

        data = []
        columns = None

        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                # 빈줄 skip
                if not line:
                    continue

                # 여러 공백 → 하나 기준 split
                parts = re.split(r'\s+', line)

                # 첫 줄 → 컬럼
                if columns is None:
                    columns = parts
                    continue

                # 데이터 길이 안 맞으면 skip (깨진 데이터 방지)
                if len(parts) != len(columns):
                    continue

                data.append(parts)

        # DataFrame 생성
        df = pd.DataFrame(data, columns=columns)

        # CSV 저장
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        self.stdout.write(self.style.SUCCESS(f"CSV 변환 완료: {output_path}"))
        self.stdout.write(f"총 데이터 수: {len(df)}")