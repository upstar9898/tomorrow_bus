import os
import pandas as pd
import glob
from datetime import datetime
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "정류소 원본 CSV를 admin 업로드용 CSV 형태로 변환"

    def handle(self, *args, **options):
        # 프로젝트 루트
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )

        data_dir = os.path.join(base_dir, "data")
        pattern = os.path.join(data_dir, "bus_station_with_stn_*.csv")
        files = glob.glob(pattern)

        if not files:
            self.stdout.write(
                self.style.ERROR("bus_station_with_stn_*.csv 파일이 없습니다.")
            )
            return

        # 가장 최신 파일 선택
        input_path = max(files, key=os.path.getmtime)
        today_str = datetime.now().strftime("%y%m%d")  # YYmmDD 형식

        output_path = os.path.join(data_dir, f"bus_station_for_admin_{today_str}.csv")

        # 입력 파일 존재 여부 확인
        if not os.path.exists(input_path):
            self.stdout.write(self.style.ERROR(f"입력 파일이 없습니다: {input_path}"))
            return

        # CSV 읽기
        try:
            df = pd.read_csv(input_path, encoding="utf-8-sig")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"CSV 읽기 실패: {str(e)}"))
            return

        # 원본 필수 컬럼
        required_cols = [
            "stationId",
            "stn",
            "stNm",
            "위도",
            "경도",
            "address",
            "is_virtual",
            "regId",
            "arsId",
            "distance_km",
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.stdout.write(self.style.ERROR(f"필수 컬럼이 없습니다: {missing_cols}"))
            return

        # 필요한 컬럼만 선택 (regId, distance_km 제거)
        result_df = df[
            [
                "stationId",
                "stn",
                "stNm",
                "위도",
                "경도",
                "address",
                "is_virtual",
                "arsId",
            ]
        ].copy()

        # 문자열 처리
        for col in ["stationId", "stn", "stNm", "address", "is_virtual", "arsId"]:
            result_df[col] = (
                result_df[col]
                .fillna("")
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)  # 🔥 핵심
                .str.strip()
            )

        # stationId 없는 행 제거
        result_df = result_df[result_df["stationId"] != ""].copy()

        # 위도/경도 숫자 변환
        result_df["위도"] = pd.to_numeric(result_df["위도"], errors="coerce")
        result_df["경도"] = pd.to_numeric(result_df["경도"], errors="coerce")

        # stationId 중복 제거
        result_df = result_df.drop_duplicates(subset=["stationId"])

        # 저장
        try:
            result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"CSV 저장 실패: {str(e)}"))
            return

        self.stdout.write(self.style.SUCCESS("정류소 CSV 변환 완료"))
        self.stdout.write(f"입력 파일: {input_path}")
        self.stdout.write(f"출력 파일: {output_path}")
        self.stdout.write(f"저장 건수: {len(result_df)}")
