import os
import glob
import pandas as pd
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "기상청 txt 파일을 csv로 변환"

    FILE_NAME_FORMAT = "weather_raw_*.txt"

    def handle(self, *args, **kwargs):
        # 프로젝트 루트
        base_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )
        )

        # 최신 txt 파일 찾기
        pattern = os.path.join(base_dir, self.FILE_NAME_FORMAT)
        files = glob.glob(pattern)

        if not files:
            self.stdout.write(self.style.ERROR("txt 파일이 없습니다."))
            return

        txt_path = max(files, key=os.path.getmtime)
        self.stdout.write(f"사용할 파일: {txt_path}")

        csv_path = txt_path.replace(".txt", ".csv")

        header = None
        data_lines = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # 헤더 찾기
                if line.startswith("# YYMMDDHHMI"):
                    header = line.lstrip("# ").split()

                # 데이터 라인
                elif line and line[0].isdigit():
                    data_lines.append(line.split())

        if header is None:
            self.stdout.write(self.style.ERROR("헤더를 찾지 못했습니다."))
            return

        # DataFrame 생성
        df = pd.DataFrame(data_lines, columns=header)

        # 숫자 변환 (선택)
        df = df.apply(pd.to_numeric, errors="coerce")

        # csv 저장
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        self.stdout.write(self.style.SUCCESS(f"CSV 변환 완료: {csv_path}"))

        