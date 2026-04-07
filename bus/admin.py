from django.contrib import admin, messages
from django.urls import path
from django.shortcuts import render, redirect
from django import forms
from django.db import transaction
import csv
import io

from .models import Bus_route, Bus_station, Route_station


@admin.register(Bus_route)
class BusRouteAdmin(admin.ModelAdmin):
    list_display = ("routeId", "routeName")
    search_fields = ("routeId", "routeName")
    ordering = ("routeId",)


class CsvImportForm(forms.Form):
    csv_file = forms.FileField(label="CSV 파일")


@admin.register(Bus_station)
class BusStationAdmin(admin.ModelAdmin):
    list_display = ("stationId", "stationName", "locationX", "locationY")
    search_fields = ("stationId", "stationName")
    ordering = ("stationId",)
    change_list_template = "admin/bus_station_changelist.html"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "upload-csv/",
                self.admin_site.admin_view(self.upload_csv),
                name="bus_station_upload_csv",
            ),
        ]
        return custom_urls + urls

    def upload_csv(self, request):
        if request.method == "POST":
            form = CsvImportForm(request.POST, request.FILES)

            if form.is_valid():
                csv_file = form.cleaned_data["csv_file"]

                if not csv_file.name.endswith(".csv"):
                    self.message_user(
                        request,
                        "CSV 파일만 업로드할 수 있습니다.",
                        level=messages.ERROR,
                    )
                    return redirect("..")

                try:
                    decoded_file = csv_file.read().decode("utf-8-sig")
                    io_string = io.StringIO(decoded_file)
                    reader = csv.DictReader(io_string)

                    required_headers = ["arsId", "stNm", "위도", "경도"]
                    uploaded_headers = reader.fieldnames

                    if uploaded_headers is None:
                        self.message_user(
                            request,
                            "CSV 헤더를 읽을 수 없습니다.",
                            level=messages.ERROR,
                        )
                        return redirect("..")

                    missing_headers = [
                        header
                        for header in required_headers
                        if header not in uploaded_headers
                    ]

                    if missing_headers:
                        self.message_user(
                            request,
                            f"헤더가 맞지 않습니다. 필요한 헤더: {required_headers}",
                            level=messages.ERROR,
                        )
                        return redirect("..")

                    station_list = []

                    for row_num, row in enumerate(reader, start=2):
                        ars_id = (row.get("arsId") or "").strip()
                        st_nm = (row.get("stNm") or "").strip()
                        lat = (row.get("위도") or "").strip()
                        lng = (row.get("경도") or "").strip()

                        if not ars_id or not st_nm or not lat or not lng:
                            raise ValueError(f"{row_num}행: 필수값이 비어 있습니다.")

                        try:
                            lat = float(lat)
                            lng = float(lng)
                        except ValueError:
                            raise ValueError(
                                f"{row_num}행: 위도/경도 값이 숫자가 아닙니다."
                            )

                        station_list.append(
                            Bus_station(
                                stationId=ars_id,
                                stationName=st_nm,
                                locationX=lng,
                                locationY=lat,
                            )
                        )

                    with transaction.atomic():
                        Bus_station.objects.all().delete()
                        Bus_station.objects.bulk_create(station_list)

                    self.message_user(
                        request,
                        f"CSV 업로드 성공: {len(station_list)}건이 새로 저장되었습니다.",
                        level=messages.SUCCESS,
                    )
                    return redirect("..")

                except Exception as e:
                    self.message_user(
                        request,
                        f"업로드 중 오류 발생: {str(e)}",
                        level=messages.ERROR,
                    )
                    return redirect("..")

        else:
            form = CsvImportForm()

        context = {
            "form": form,
            "title": "정류소 CSV 업로드",
        }
        return render(request, "admin/csv_upload.html", context)


@admin.register(Route_station)
class RouteStationAdmin(admin.ModelAdmin):
    list_display = ("route", "station", "staOrd")
    search_fields = (
        "route__routeId",
        "route__routeName",
        "station__stationId",
        "station__stationName",
    )
    list_filter = ("route", "station")
    ordering = ("route", "staOrd")
