import os
import pandas as pd
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "weather_regioncode.txt를 csv로 변환"

    def handle(self, *args, **kwargs):
        # 프로젝트 루트
        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        )

        input_path = os.path.join(base_dir, "weather_regioncode.txt")
        output_path = os.path.join(base_dir, "weather_regioncode.csv")

        rows = []

        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip()

                # 주석 제거
                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                # 최소 길이 체크
                if len(parts) < 13:
                    continue

                try:
                    row = {
                        "STN": int(parts[0]),
                        "LON": float(parts[1]),
                        "LAT": float(parts[2]),
                        "STN_KO": parts[10],
                        "STN_EN": parts[11],
                        "FCT_ID": parts[12],
                        "LAW_ID": parts[13] if len(parts) > 13 else None,
                        "LAW_ADDR": " ".join(parts[15:]) if len(parts) > 15 else None,
                    }
                    rows.append(row)

                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        self.stdout.write(self.style.SUCCESS(f"CSV 변환 완료: {output_path}"))