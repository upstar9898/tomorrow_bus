import csv
import io

from django.contrib import admin, messages
from django.db import transaction
from django.shortcuts import redirect, render
from django.urls import path

from .forms import CsvImportForm
from .models import Bus_route, Bus_station, Route_station, Weather_station, Bus_info


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
                        request, f"업로드 중 오류 발생: {str(e)}", level=messages.ERROR
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
        missing_headers = [h for h in required_headers if h not in reader.fieldnames]
        if missing_headers:
            raise ValueError(f"헤더가 맞지 않습니다. 필요한 헤더: {required_headers}")

        route_list = []
        seen_route_ids = set()

        for row_num, row in enumerate(reader, start=2):
            route_id = (row.get("routeId") or "").strip()
            route_name = (row.get("routeName") or "").strip()

            if not route_id or not route_name:
                raise ValueError(
                    f"{row_num}행: routeId 또는 routeName이 비어 있습니다."
                )

            if route_id in seen_route_ids:
                raise ValueError(f"{row_num}행: 중복 routeId({route_id})가 있습니다.")
            seen_route_ids.add(route_id)

            route_list.append(Bus_route(routeId=route_id, routeName=route_name))

        Bus_route.objects.all().delete()
        Bus_route.objects.bulk_create(route_list)
        return len(route_list)


@admin.register(Bus_station)
class BusStationAdmin(CsvUploadAdminMixin, admin.ModelAdmin):
    list_display = (
        "stationId",
        "stationName",
        "stn",
        "locationX",
        "locationY",
        "address",
        "isVirtual",
        "arsId",
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
        missing_headers = [h for h in required_headers if h not in reader.fieldnames]
        if missing_headers:
            raise ValueError(f"헤더가 맞지 않습니다. 필요한 헤더: {required_headers}")

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

            if stn == "":
                stn = None

            if not station_id or not station_name or not latitude or not longitude:
                raise ValueError(f"{row_num}행: 필수값이 비어 있습니다.")

            if station_id in seen_station_ids:
                raise ValueError(
                    f"{row_num}행: 중복 stationId({station_id})가 있습니다."
                )
            seen_station_ids.add(station_id)

            try:
                latitude = float(latitude)
                longitude = float(longitude)
            except ValueError:
                raise ValueError(f"{row_num}행: 위도/경도 값이 숫자가 아닙니다.")

            station_list.append(
                Bus_station(
                    stationId=station_id,
                    stn_id=stn,
                    stationName=station_name,
                    locationX=longitude,
                    locationY=latitude,
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
        missing_headers = [h for h in required_headers if h not in reader.fieldnames]
        if missing_headers:
            raise ValueError(f"헤더가 맞지 않습니다. 필요한 헤더: {required_headers}")

        route_map = {obj.routeId: obj for obj in Bus_route.objects.all()}
        station_map = {obj.stationId: obj for obj in Bus_station.objects.all()}

        route_station_list = []
        seen_pairs = set()

        for row_num, row in enumerate(reader, start=2):
            route_id = (row.get("routeId") or "").strip()
            station_id = (row.get("stationId") or "").strip()
            sta_ord = (row.get("staOrd") or "").strip()

            if not route_id or not station_id or not sta_ord:
                raise ValueError(f"{row_num}행: 필수값이 비어 있습니다.")

            if route_id not in route_map:
                raise ValueError(
                    f"{row_num}행: 존재하지 않는 routeId({route_id})입니다."
                )

            if station_id not in station_map:
                raise ValueError(
                    f"{row_num}행: 존재하지 않는 stationId({station_id})입니다."
                )

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


@admin.register(Weather_station)
class WeatherStationAdmin(CsvUploadAdminMixin, admin.ModelAdmin):
    list_display = ("stnId", "stnName", "locationX", "locationY")
    search_fields = ("stnId", "stnName")
    ordering = ("stnId",)

    @transaction.atomic
    def process_csv(self, reader):
        required_headers = ["STN_ID", "LON", "LAT", "STN_KO"]
        missing_headers = [h for h in required_headers if h not in reader.fieldnames]
        if missing_headers:
            raise ValueError(f"헤더가 맞지 않습니다. 필요한 헤더: {required_headers}")

        weather_station_list = []
        seen_stn_ids = set()

        for row_num, row in enumerate(reader, start=2):
            stn_id = (row.get("STN_ID") or "").strip()
            stn_name = (row.get("STN_KO") or "").strip()
            location_x = (row.get("LON") or "").strip()  # 경도
            location_y = (row.get("LAT") or "").strip()  # 위도

            if not stn_id or not stn_name or not location_x or not location_y:
                raise ValueError(f"{row_num}행: 필수값이 비어 있습니다.")

            if stn_id in seen_stn_ids:
                raise ValueError(f"{row_num}행: 중복 STN_ID({stn_id})가 있습니다.")
            seen_stn_ids.add(stn_id)

            try:
                location_x = float(location_x)
                location_y = float(location_y)
            except ValueError:
                raise ValueError(f"{row_num}행: LON/LAT 값이 숫자가 아닙니다.")

            weather_station_list.append(
                Weather_station(
                    stnId=stn_id,
                    stnName=stn_name,
                    locationX=location_x,
                    locationY=location_y,
                )
            )

        Weather_station.objects.all().delete()
        Weather_station.objects.bulk_create(weather_station_list)
        return len(weather_station_list)


@admin.register(Bus_info)
class BusInfoAdmin(CsvUploadAdminMixin, admin.ModelAdmin):
    list_display = (
        "route",
        "firstTm",
        "lastTm",
        "timeGap",
        "saturdayFirstTm",
        "saturdayLastTm",
        "saturdaytimeGap",
        "holidayFirstTm",
        "holidayLastTm",
        "holidaytimeGap",
    )
    search_fields = ("route__routeId", "route__routeName")
    ordering = ("route",)

    @transaction.atomic
    def process_csv(self, reader):
        required_headers = [
            "busRouteId",
            "firstTm",
            "lastTm",
            "timeGap",
            "saturdayFirstTm",
            "saturdayLastTm",
            "saturdaytimeGap",
            "holidayFirstTm",
            "holidayLastTm",
            "holidaytimeGap",
        ]

        missing_headers = [h for h in required_headers if h not in reader.fieldnames]
        if missing_headers:
            raise ValueError(f"헤더가 맞지 않습니다. 필요한 헤더: {required_headers}")

        bus_info_list = []
        seen_route_ids = set()

        for row_num, row in enumerate(reader, start=2):
            route_id = (row.get("busRouteId") or "").strip()

            first_tm = (row.get("firstTm") or "").strip()
            last_tm = (row.get("lastTm") or "").strip()
            time_gap = (row.get("timeGap") or "").strip()

            saturday_first_tm = (row.get("saturdayFirstTm") or "").strip()
            saturday_last_tm = (row.get("saturdayLastTm") or "").strip()
            saturday_time_gap = (row.get("saturdaytimeGap") or "").strip()

            holiday_first_tm = (row.get("holidayFirstTm") or "").strip()
            holiday_last_tm = (row.get("holidayLastTm") or "").strip()
            holiday_time_gap = (row.get("holidaytimeGap") or "").strip()

            if not route_id:
                raise ValueError(f"{row_num}행: busRouteId가 비어 있습니다.")

            if route_id in seen_route_ids:
                raise ValueError(
                    f"{row_num}행: 중복 busRouteId({route_id})가 있습니다."
                )
            seen_route_ids.add(route_id)

            try:
                route = Bus_route.objects.get(routeId=route_id)
            except Bus_route.DoesNotExist:
                raise ValueError(
                    f"{row_num}행: Bus_route에 없는 routeId({route_id})입니다."
                )

            try:
                time_gap = int(time_gap)
                saturday_time_gap = int(saturday_time_gap)
                holiday_time_gap = int(holiday_time_gap)
            except ValueError:
                raise ValueError(f"{row_num}행: 배차간격 값이 숫자가 아닙니다.")

            bus_info_list.append(
                Bus_info(
                    route=route,
                    firstTm=first_tm,
                    lastTm=last_tm,
                    timeGap=time_gap,
                    saturdayFirstTm=saturday_first_tm,
                    saturdayLastTm=saturday_last_tm,
                    saturdaytimeGap=saturday_time_gap,
                    holidayFirstTm=holiday_first_tm,
                    holidayLastTm=holiday_last_tm,
                    holidaytimeGap=holiday_time_gap,
                )
            )

        Bus_info.objects.all().delete()
        Bus_info.objects.bulk_create(bus_info_list)

        return len(bus_info_list)
