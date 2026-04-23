import os
import glob
import pandas as pd

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from bus.models import Bus_arrival_info, Bus_route, Bus_station


class Command(BaseCommand):
    help = "전처리된 CSV를 Bus_arrival_info 테이블에 누적 insert 합니다."

    REQUIRED_COLUMNS = [
        "mkTm",
        "busRouteId",
        "stId",
        "staOrd",
        "remaining_seat",
        "full_flag",
    ]

    def add_arguments(self, parser):
        parser.add_argument(
            "path",
            type=str,
            help="CSV 파일 경로 또는 CSV 폴더 경로",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1000,
            help="bulk_create 배치 크기 (기본값: 1000)",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="폴더 경로일 때 하위 폴더까지 포함해 모든 CSV를 insert",
        )

    def handle(self, *args, **options):
        path = options["path"]
        batch_size = options["batch_size"]
        import_all = options["all"]

        if not os.path.exists(path):
            raise CommandError(f"경로가 존재하지 않습니다: {path}")

        csv_files = self.get_csv_files(path, import_all)

        if not csv_files:
            raise CommandError("처리할 CSV 파일이 없습니다.")

        self.stdout.write(self.style.SUCCESS(f"총 {len(csv_files)}개 CSV 파일 처리 시작"))

        total_file_count = 0
        total_csv_rows = 0
        total_inserted = 0
        total_skipped_missing_fk = 0
        total_duplicate_or_conflict = 0

        for csv_file in csv_files:
            self.stdout.write(f"\n[처리 중] {csv_file}")

            result = self.import_one_file(csv_file, batch_size)

            total_file_count += 1
            total_csv_rows += result["csv_rows"]
            total_inserted += result["inserted"]
            total_skipped_missing_fk += result["skipped_missing_fk"]
            total_duplicate_or_conflict += result["duplicate_or_conflict"]

        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("전체 CSV import 완료"))
        self.stdout.write(f"처리한 파일 수: {total_file_count}")
        self.stdout.write(f"총 CSV 행 수: {total_csv_rows}")
        self.stdout.write(f"실제 insert 수: {total_inserted}")
        self.stdout.write(f"FK 없음으로 건너뜀: {total_skipped_missing_fk}")
        self.stdout.write(f"중복 등으로 무시된 수: {total_duplicate_or_conflict}")
        self.stdout.write("=" * 60)

    def get_csv_files(self, path, import_all=False):
        """
        path가 파일이면 그 파일만 반환
        path가 폴더이면:
        - --all 없으면 해당 폴더 바로 아래 CSV만 반환
        - --all 있으면 하위 폴더 포함 모든 CSV 반환
        """
        if os.path.isfile(path):
            if not path.lower().endswith(".csv"):
                raise CommandError("지정한 파일이 CSV가 아닙니다.")
            return [path]

        if os.path.isdir(path):
            if import_all:
                pattern = os.path.join(path, "**", "*.csv")
                files = sorted(glob.glob(pattern, recursive=True))
            else:
                pattern = os.path.join(path, "*.csv")
                files = sorted(glob.glob(pattern))

            return files

        raise CommandError(f"유효하지 않은 경로입니다: {path}")

    def import_one_file(self, csv_path, batch_size):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"CSV 읽기 실패: {csv_path} / {e}"))
            return {
                "csv_rows": 0,
                "inserted": 0,
                "skipped_missing_fk": 0,
                "duplicate_or_conflict": 0,
            }

        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            self.stdout.write(
                self.style.WARNING(f"필수 컬럼이 없어 건너뜁니다: {missing}")
            )
            return {
                "csv_rows": 0,
                "inserted": 0,
                "skipped_missing_fk": 0,
                "duplicate_or_conflict": 0,
            }

        # 필요한 컬럼만 사용
        df = df[self.REQUIRED_COLUMNS].copy()

        # 필수값 없는 행 제거
        df = df.dropna(subset=["mkTm", "busRouteId", "stId", "staOrd", "remaining_seat"])

        if df.empty:
            self.stdout.write(self.style.WARNING("유효한 데이터가 없어 건너뜁니다."))
            return {
                "csv_rows": 0,
                "inserted": 0,
                "skipped_missing_fk": 0,
                "duplicate_or_conflict": 0,
            }

        # 타입 정리
        df["mkTm"] = pd.to_datetime(df["mkTm"], errors="coerce")
        df["busRouteId"] = df["busRouteId"].astype(str).str.strip()
        df["stId"] = df["stId"].astype(str).str.strip()
        df["staOrd"] = pd.to_numeric(df["staOrd"], errors="coerce")
        df["remaining_seat"] = pd.to_numeric(df["remaining_seat"], errors="coerce")
        df["full_flag"] = df["full_flag"].apply(self.parse_bool)

        # 파싱 실패 제거
        df = df.dropna(subset=["mkTm", "staOrd", "remaining_seat"])

        if df.empty:
            self.stdout.write(self.style.WARNING("파싱 후 남은 데이터가 없어 건너뜁니다."))
            return {
                "csv_rows": 0,
                "inserted": 0,
                "skipped_missing_fk": 0,
                "duplicate_or_conflict": 0,
            }

        # 최종 형변환
        df["staOrd"] = df["staOrd"].astype(int)
        df["remaining_seat"] = df["remaining_seat"].astype(int)

        # 같은 CSV 내부 중복 제거
        df = df.drop_duplicates(subset=["mkTm", "busRouteId", "stId"])

        csv_rows = len(df)

        # FK 미리 조회
        route_ids = df["busRouteId"].unique().tolist()
        station_ids = df["stId"].unique().tolist()

        route_map = {
            obj.routeId: obj
            for obj in Bus_route.objects.filter(routeId__in=route_ids)
        }
        station_map = {
            obj.stationId: obj
            for obj in Bus_station.objects.filter(stationId__in=station_ids)
        }

        missing_routes = sorted(set(route_ids) - set(route_map.keys()))
        missing_stations = sorted(set(station_ids) - set(station_map.keys()))

        if missing_routes:
            self.stdout.write(
                self.style.WARNING(
                    f"존재하지 않는 노선 ID {len(missing_routes)}개 발견 (해당 행 건너뜀)"
                )
            )
            self.stdout.write(", ".join(missing_routes[:20]))

        if missing_stations:
            self.stdout.write(
                self.style.WARNING(
                    f"존재하지 않는 정류소 ID {len(missing_stations)}개 발견 (해당 행 건너뜀)"
                )
            )
            self.stdout.write(", ".join(missing_stations[:20]))

        objects = []
        skipped_missing_fk = 0

        for row in df.itertuples(index=False):
            route_obj = route_map.get(row.busRouteId)
            station_obj = station_map.get(row.stId)

            if not route_obj or not station_obj:
                skipped_missing_fk += 1
                continue

            objects.append(
                Bus_arrival_info(
                    mkTm=row.mkTm.to_pydatetime(),
                    route=route_obj,
                    station=station_obj,
                    staOrd=row.staOrd,
                    remaining_seat=row.remaining_seat,
                    full_flag=row.full_flag,
                )
            )

        if not objects:
            self.stdout.write(self.style.WARNING("저장할 데이터가 없습니다."))
            return {
                "csv_rows": csv_rows,
                "inserted": 0,
                "skipped_missing_fk": skipped_missing_fk,
                "duplicate_or_conflict": 0,
            }

        before_count = Bus_arrival_info.objects.count()

        with transaction.atomic():
            Bus_arrival_info.objects.bulk_create(
                objects,
                batch_size=batch_size,
                ignore_conflicts=True,
            )

        after_count = Bus_arrival_info.objects.count()
        inserted_count = after_count - before_count
        duplicate_or_conflict = len(objects) - inserted_count

        self.stdout.write(f"CSV 행 수: {csv_rows}")
        self.stdout.write(f"FK 없음으로 건너뜀: {skipped_missing_fk}")
        self.stdout.write(f"실제 insert 수: {inserted_count}")
        self.stdout.write(f"중복 등으로 무시된 수: {duplicate_or_conflict}")

        return {
            "csv_rows": csv_rows,
            "inserted": inserted_count,
            "skipped_missing_fk": skipped_missing_fk,
            "duplicate_or_conflict": duplicate_or_conflict,
        }

    def parse_bool(self, value):
        """
        full_flag가 0/1, True/False, Y/N 형태여도 처리
        """
        if pd.isna(value):
            return False

        value = str(value).strip().lower()
        return value in ["1", "true", "t", "y", "yes"]