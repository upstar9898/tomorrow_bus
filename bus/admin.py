import csv
import io

from django.contrib import admin, messages
from django.db import transaction
from django.shortcuts import redirect, render
from django.urls import path

from .forms import CsvImportForm
from .models import Bus_route, Bus_station, Route_station


class CsvUploadAdminMixin:
    """
    관리자 목록 화면에 CSV 업로드 버튼과 업로드 기능을 붙이는 공통 mixin
    """
    change_list_template = "admin/csv_changelist.html"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "upload-csv/",
                self.admin_site.admin_view(self.upload_csv),
                name=f"{self.model._meta.app_label}_{self.model._meta.model_name}_upload_csv",
            ),
        ]
        return custom_urls + urls

    def changelist_view(self, request, extra_context=None):
        extra_context = extra_context or {}
        extra_context["csv_upload_url"] = "upload-csv/"
        return super().changelist_view(request, extra_context=extra_context)

    def upload_csv(self, request):
        if request.method == "POST":
            form = CsvImportForm(request.POST, request.FILES)

            if form.is_valid():
                csv_file = form.cleaned_data["csv_file"]

                if not csv_file.name.lower().endswith(".csv"):
                    self.message_user(
                        request,
                        "CSV 파일만 업로드할 수 있습니다.",
                        level=messages.ERROR,
                    )
                    return redirect(request.path)

                try:
                    decoded_file = csv_file.read().decode("utf-8-sig")
                    io_string = io.StringIO(decoded_file)
                    reader = csv.DictReader(io_string)

                    if not reader.fieldnames:
                        self.message_user(
                            request,
                            "CSV 헤더를 읽을 수 없습니다.",
                            level=messages.ERROR,
                        )
                        return redirect(request.path)

                    count = self.process_csv(reader)

                    self.message_user(
                        request,
                        f"CSV 업로드 성공: {count}건 저장되었습니다.",
                        level=messages.SUCCESS,
                    )
                    return redirect("../")

                except Exception as e:
                    self.message_user(
                        request,
                        f"업로드 중 오류 발생: {str(e)}",
                        level=messages.ERROR,
                    )
                    return redirect(request.path)

        else:
            form = CsvImportForm()

        context = {
            "form": form,
            "title": f"{self.model._meta.verbose_name} CSV 업로드",
            "opts": self.model._meta,
        }
        return render(request, "admin/csv_upload.html", context)

    def process_csv(self, reader):
        raise NotImplementedError("각 Admin 클래스에서 process_csv를 구현해야 합니다.")


@admin.register(Bus_route)
class BusRouteAdmin(CsvUploadAdminMixin, admin.ModelAdmin):
    list_display = ("routeId", "routeName")
    search_fields = ("routeId", "routeName")
    ordering = ("routeId",)

    @transaction.atomic
    def process_csv(self, reader):
        required_headers = ["routeId", "routeName"]
        uploaded_headers = reader.fieldnames

        missing_headers = [h for h in required_headers if h not in uploaded_headers]
        if missing_headers:
            raise ValueError(
                f"헤더가 맞지 않습니다. 필요한 헤더: {required_headers}"
            )

        route_list = []
        seen_route_ids = set()

        for row_num, row in enumerate(reader, start=2):
            route_id = (row.get("routeId") or "").strip()
            route_name = (row.get("routeName") or "").strip()

            if not route_id or not route_name:
                raise ValueError(f"{row_num}행: routeId 또는 routeName이 비어 있습니다.")

            if route_id in seen_route_ids:
                raise ValueError(f"{row_num}행: 중복 routeId({route_id})가 있습니다.")

            seen_route_ids.add(route_id)

            route_list.append(
                Bus_route(
                    routeId=route_id,
                    routeName=route_name,
                )
            )

        Bus_route.objects.all().delete()
        Bus_route.objects.bulk_create(route_list)

        return len(route_list)


@admin.register(Bus_station)
class BusStationAdmin(CsvUploadAdminMixin, admin.ModelAdmin):
    list_display = (
        "stationId",
        "stn",
        "stationName",
        "locationX",
        "locationY",
        "arsId",
        "isVirtual",
    )
    search_fields = ("stationId", "stationName", "arsId", "address")
    ordering = ("stationId",)

    @transaction.atomic
    def process_csv(self, reader):
        required_headers = [
            "stationId",
            "stn",
            "stNm",
            "위도",
            "경도",
            "address",
            "is_virtual",
            "arsId",
        ]
        uploaded_headers = reader.fieldnames

        missing_headers = [h for h in required_headers if h not in uploaded_headers]
        if missing_headers:
            raise ValueError(
                f"헤더가 맞지 않습니다. 필요한 헤더: {required_headers}"
            )

        station_list = []
        seen_station_ids = set()

        for row_num, row in enumerate(reader, start=2):
            station_id = (row.get("stationId") or "").strip()
            stn = (row.get("stn") or "").strip()
            station_name = (row.get("stNm") or "").strip()
            latitude = (row.get("위도") or "").strip()
            longitude = (row.get("경도") or "").strip()
            address = (row.get("address") or "").strip()
            is_virtual = (row.get("is_virtual") or "").strip()
            ars_id = (row.get("arsId") or "").strip()

            if not station_id or not station_name or not latitude or not longitude:
                raise ValueError(f"{row_num}행: 필수값이 비어 있습니다.")

            if station_id in seen_station_ids:
                raise ValueError(f"{row_num}행: 중복 stationId({station_id})가 있습니다.")
            seen_station_ids.add(station_id)

            try:
                latitude = float(latitude)
                longitude = float(longitude)
            except ValueError:
                raise ValueError(f"{row_num}행: 위도/경도 값이 숫자가 아닙니다.")

            station_list.append(
                Bus_station(
                    stationId=station_id,
                    stn=stn,
                    stationName=station_name,
                    locationX=longitude,   # 경도 -> X
                    locationY=latitude,    # 위도 -> Y
                    address=address,
                    isVirtual=is_virtual,
                    arsId=ars_id,
                )
            )

        Bus_station.objects.all().delete()
        Bus_station.objects.bulk_create(station_list)

        return len(station_list)


@admin.register(Route_station)
class RouteStationAdmin(CsvUploadAdminMixin, admin.ModelAdmin):
    list_display = ("id", "route", "station", "staOrd")
    search_fields = (
        "route__routeId",
        "route__routeName",
        "station__stationId",
        "station__stationName",
    )
    list_filter = ("route", "station")
    ordering = ("route", "staOrd")

    @transaction.atomic
    def process_csv(self, reader):
        required_headers = ["routeId", "stationId", "staOrd"]
        uploaded_headers = reader.fieldnames

        missing_headers = [h for h in required_headers if h not in uploaded_headers]
        if missing_headers:
            raise ValueError(
                f"헤더가 맞지 않습니다. 필요한 헤더: {required_headers}"
            )

        route_map = {obj.routeId: obj for obj in Bus_route.objects.all()}
        station_map = {obj.stationId: obj for obj in Bus_station.objects.all()}

        route_station_list = []
        seen_pairs = set()

        for row_num, row in enumerate(reader, start=2):
            route_id = (row.get("routeId") or "").strip()
            station_id = (row.get("stationId") or "").strip()
            sta_ord = (row.get("staOrd") or "").strip()

            if not route_id or not station_id or not sta_ord:
                raise ValueError(f"{row_num}행: routeId, stationId, staOrd 중 빈 값이 있습니다.")

            if route_id not in route_map:
                raise ValueError(f"{row_num}행: 존재하지 않는 routeId({route_id})입니다.")

            if station_id not in station_map:
                raise ValueError(f"{row_num}행: 존재하지 않는 stationId({station_id})입니다.")

            try:
                sta_ord = int(sta_ord)
            except ValueError:
                raise ValueError(f"{row_num}행: staOrd는 정수여야 합니다.")

            pair = (route_id, station_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"{row_num}행: 중복 route-station 조합({route_id}, {station_id})이 있습니다."
                )
            seen_pairs.add(pair)

            route_station_list.append(
                Route_station(
                    route=route_map[route_id],
                    station=station_map[station_id],
                    staOrd=sta_ord,
                )
            )

        Route_station.objects.all().delete()
        Route_station.objects.bulk_create(route_station_list)

        return len(route_station_list)