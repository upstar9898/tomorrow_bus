import os
import glob
import pandas as pd
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "기상청 txt 파일을 csv로 변환"

    FILE_NAME_FORMAT = "weather_raw_*.txt"

    # 강제 고정 컬럼
    COLS = [
        "TM",
        "STN",
        "WD",
        "WS",
        "GST_WD",
        "GST_WS",
        "GST_TM",
        "PA",
        "PS",
        "PT",
        "PR",
        "TA",
        "TD",
        "HM",
        "PV",
        "RN",
        "RN_DAY",
        "RN_JUN",
        "RN_INT",
        "SD_HR3",
        "SD_DAY",
        "SD_TOT",
        "WC",
        "WP",
        "WW",
        "CA_TOT",
        "CA_MID",
        "CH_MIN",
        "CT",
        "CT_TOP",
        "CT_MID",
        "CT_LOW",
        "VS",
        "SS",
        "SI",
        "ST_GD",
        "TS",
        "TE_005",
        "TE_01",
        "TE_02",
        "TE_03",
        "ST_SEA",
        "WH",
        "BF",
        "IR",
        "IX",
    ]

    def handle(self, *args, **kwargs):
        # 프로젝트 루트
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # 최신 txt 파일 찾기
        pattern = os.path.join(data_dir, self.FILE_NAME_FORMAT)
        files = glob.glob(pattern)

        if not files:
            self.stdout.write(self.style.ERROR(f"{pattern} 파일이 없습니다."))
            return

        txt_path = max(files, key=os.path.getmtime)
        self.stdout.write(f"사용할 파일: {txt_path}")

        csv_path = txt_path.replace(".txt", ".csv")

        data_lines = []

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # 빈 줄 제외
                if not line:
                    continue

                # 주석/설명 줄 제외
                if line.startswith("#"):
                    continue

                # 데이터 라인만 수집
                if line[0].isdigit():
                    row = line.split()

                    # 컬럼 수 맞추기
                    if len(row) < len(self.COLS):
                        row += [None] * (len(self.COLS) - len(row))
                    elif len(row) > len(self.COLS):
                        row = row[: len(self.COLS)]

                    data_lines.append(row)

        if not data_lines:
            self.stdout.write(self.style.ERROR("데이터 라인을 찾지 못했습니다."))
            return

        # 강제 컬럼으로 DataFrame 생성
        df = pd.DataFrame(data_lines, columns=self.COLS)

        # 숫자형 변환
        # TM은 시간값이므로 문자열 유지 가능
        numeric_cols = [col for col in self.COLS if col != "TM"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # csv 저장
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        self.stdout.write(
            self.style.SUCCESS(
                f"CSV 변환 완료: {csv_path} / 행 수: {len(df)} / 컬럼 수: {len(df.columns)}"
            )
        )
