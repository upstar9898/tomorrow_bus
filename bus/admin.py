from django.contrib import admin, messages
from django.urls import path
from django.shortcuts import render, redirect
from django import forms
from django.db import transaction
import csv
import io

from .models import Bus_route, Bus_station, Route_station


# ============================================================
# 공통 CSV 업로드 폼
# ------------------------------------------------------------
# 역할:
# - admin 업로드 화면에서 사용할 파일 업로드 폼
# - 사용자가 선택한 CSV 파일은 csv_file 필드로 받음
# ============================================================
class CsvImportForm(forms.Form):
    csv_file = forms.FileField(label="CSV 파일")


# ============================================================
# bus_route 관리자
# ------------------------------------------------------------
# 역할:
# 1) 노선 목록 표시
# 2) 노선 검색 지원
# 3) 노선 CSV 업로드 URL 연결
# 4) 업로드한 CSV를 bus_route 테이블에 저장
# ============================================================
@admin.register(Bus_route)
class BusRouteAdmin(admin.ModelAdmin):
    # 관리자 목록 페이지에서 보여줄 컬럼
    list_display = ("routeId", "routeName")

    # 관리자 검색창에서 검색할 대상 컬럼
    search_fields = ("routeId", "routeName")

    # 기본 정렬 기준
    ordering = ("routeId",)

    # 관리자 목록 페이지 템플릿 변경
    # 이 템플릿 안에 CSV 업로드 버튼이 들어감
    change_list_template = "admin/bus_route_changelist.html"

    # --------------------------------------------------------
    # 관리자 페이지 전용 URL 추가
    # --------------------------------------------------------
    # 기본 admin URL에 upload-csv/ 주소를 추가함
    # 최종적으로 이 주소로 들어오면 upload_csv() 함수가 실행됨
    # --------------------------------------------------------
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "upload-csv/",
                self.admin_site.admin_view(self.upload_csv),
                name="bus_route_upload_csv",
            ),
        ]
        return custom_urls + urls

    # --------------------------------------------------------
    # 노선 CSV 업로드 처리
    # --------------------------------------------------------
    # 흐름:
    # 1) admin 화면에서 CSV 업로드
    # 2) request.FILES 로 파일 받기
    # 3) csv.DictReader 로 한 줄씩 읽기
    # 4) Bus_route 객체 리스트 만들기
    # 5) 기존 데이터 삭제
    # 6) 새 CSV 데이터 저장
    # --------------------------------------------------------
    def upload_csv(self, request):
        # POST 요청이면 실제 업로드 처리
        if request.method == "POST":
            # 업로드한 파일 정보를 폼에 담아서 검사
            form = CsvImportForm(request.POST, request.FILES)

            if form.is_valid():
                # 업로드한 파일 꺼내기
                csv_file = form.cleaned_data["csv_file"]

                # CSV 확장자 검사
                if not csv_file.name.endswith(".csv"):
                    self.message_user(
                        request,
                        "CSV 파일만 업로드할 수 있습니다.",
                        level=messages.ERROR,
                    )
                    return redirect("..")

                try:
                    # ------------------------------------------------
                    # 파일 읽기
                    # ------------------------------------------------
                    # 업로드한 파일은 바이트 형태이므로 문자열로 변환
                    # utf-8-sig 는 BOM 있는 CSV도 깨지지 않게 처리
                    # ------------------------------------------------
                    decoded_file = csv_file.read().decode("utf-8-sig")

                    # 문자열을 파일처럼 다루기 위한 객체 생성
                    io_string = io.StringIO(decoded_file)

                    # CSV를 헤더 기준 딕셔너리 형태로 읽기
                    # 예:
                    # {"routeId": "100100118", "routeName": "753"}
                    reader = csv.DictReader(io_string)

                    # CSV에 반드시 있어야 하는 헤더
                    required_headers = ["routeId", "routeName"]

                    # 실제 업로드한 CSV의 헤더 목록
                    uploaded_headers = reader.fieldnames

                    # 헤더를 못 읽은 경우
                    if uploaded_headers is None:
                        self.message_user(
                            request,
                            "CSV 헤더를 읽을 수 없습니다.",
                            level=messages.ERROR,
                        )
                        return redirect("..")

                    # 필요한 헤더가 빠졌는지 검사
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

                    # DB에 넣기 전에 Bus_route 객체들을 담아둘 리스트
                    route_list = []

                    # ------------------------------------------------
                    # CSV를 한 줄씩 읽어서 모델 객체 생성
                    # ------------------------------------------------
                    # start=2 인 이유:
                    # 1행은 헤더이고 실제 데이터는 2행부터 시작하므로
                    # 에러 메시지 행 번호를 실제 CSV와 맞추기 위해서
                    # ------------------------------------------------
                    for row_num, row in enumerate(reader, start=2):
                        route_id = (row.get("routeId") or "").strip()
                        route_name = (row.get("routeName") or "").strip()

                        # 필수값 검사
                        if not route_id or not route_name:
                            raise ValueError(f"{row_num}행: 필수값이 비어 있습니다.")

                        # Django 모델 객체 생성
                        route_list.append(
                            Bus_route(
                                routeId=route_id,
                                routeName=route_name,
                            )
                        )

                    # ------------------------------------------------
                    # DB 저장
                    # ------------------------------------------------
                    # transaction.atomic():
                    # 중간에 하나라도 에러 나면 전체 저장 취소
                    # ------------------------------------------------
                    with transaction.atomic():
                        # 기존 노선 데이터 전체 삭제
                        Bus_route.objects.all().delete()

                        # 새 CSV 데이터 한번에 저장
                        Bus_route.objects.bulk_create(route_list)

                    # 성공 메시지 출력
                    self.message_user(
                        request,
                        f"노선 CSV 업로드 성공: {len(route_list)}건이 새로 저장되었습니다.",
                        level=messages.SUCCESS,
                    )
                    return redirect("..")

                except Exception as e:
                    # 업로드 중 예외가 발생하면 에러 메시지 출력
                    self.message_user(
                        request,
                        f"업로드 중 오류 발생: {str(e)}",
                        level=messages.ERROR,
                    )
                    return redirect("..")

        else:
            # GET 요청이면 빈 업로드 폼 화면 보여주기
            form = CsvImportForm()

        # 업로드 화면으로 전달할 데이터
        context = {
            "form": form,
            "title": "노선 CSV 업로드",
        }
        return render(request, "admin/csv_upload.html", context)


# ============================================================
# bus_station 관리자
# ------------------------------------------------------------
# 역할:
# 1) 정류소 목록 표시
# 2) 정류소 검색 지원
# 3) 정류소 CSV 업로드 URL 연결
# 4) 업로드한 CSV를 bus_station 테이블에 저장
# ============================================================
@admin.register(Bus_station)
class BusStationAdmin(admin.ModelAdmin):
    # 관리자 목록 페이지에서 보여줄 컬럼
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

    # 관리자 검색창에서 검색할 대상 컬럼
    search_fields = ("stationId", "stationName", "arsId")

    # 기본 정렬 기준
    ordering = ("stationId",)

    # 관리자 목록 페이지 템플릿 변경
    # 이 템플릿 안에 CSV 업로드 버튼이 들어감
    change_list_template = "admin/bus_station_changelist.html"

    # --------------------------------------------------------
    # 관리자 페이지 전용 URL 추가
    # --------------------------------------------------------
    # /upload-csv/ 주소로 들어오면 upload_csv() 함수가 실행됨
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 정류소 CSV 업로드 처리
    # --------------------------------------------------------
    # CSV 예시 헤더:
    # arsId, stNm, 위도, 경도
    #
    # 저장 매핑:
    # arsId -> stationId
    # stNm -> stationName
    # 경도 -> locationX
    # 위도 -> locationY
    # --------------------------------------------------------
    def upload_csv(self, request):
        # POST 요청이면 실제 업로드 처리
        if request.method == "POST":
            # 업로드한 파일 정보를 폼에 담아서 검사
            form = CsvImportForm(request.POST, request.FILES)

            if form.is_valid():
                # 업로드한 파일 꺼내기
                csv_file = form.cleaned_data["csv_file"]

                # CSV 확장자 검사
                if not csv_file.name.endswith(".csv"):
                    self.message_user(
                        request,
                        "CSV 파일만 업로드할 수 있습니다.",
                        level=messages.ERROR,
                    )
                    return redirect("..")

                try:
                    # 업로드 파일 읽기
                    decoded_file = csv_file.read().decode("utf-8-sig")
                    io_string = io.StringIO(decoded_file)

                    # CSV를 헤더 기준 딕셔너리 형태로 읽기
                    reader = csv.DictReader(io_string)

                    # CSV에 반드시 있어야 하는 헤더
                    required_headers = [
                        "stationId",
                        "stNm",
                        "stn",
                        "위도",
                        "경도",
                        "address",
                        "isVirtual",
                        "arsId",
                    ]

                    # 실제 업로드한 CSV의 헤더 목록
                    uploaded_headers = reader.fieldnames

                    # 헤더를 못 읽은 경우
                    if uploaded_headers is None:
                        self.message_user(
                            request,
                            "CSV 헤더를 읽을 수 없습니다.",
                            level=messages.ERROR,
                        )
                        return redirect("..")

                    # 필요한 헤더가 빠졌는지 검사
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

                    # DB에 넣기 전에 Bus_station 객체들을 담아둘 리스트
                    station_list = []

                    # CSV를 한 줄씩 읽어서 모델 객체 생성
                    for row_num, row in enumerate(reader, start=2):
                        # CSV 컬럼값 꺼내기
                        station_id = (row.get("stationId") or "").strip()
                        st_nm = (row.get("stNm") or "").strip()
                        stn = (row.get("stn") or "").strip()
                        lat = (row.get("위도") or "").strip()
                        lng = (row.get("경도") or "").strip()
                        address = (row.get("address") or "").strip()
                        is_virtual = (row.get("isVirtual") or "").strip()
                        ars_id = (row.get("arsId") or "").strip()

                        # 필수값 검사
                        if (
                            not station_id
                            or not st_nm
                            or not stn
                            or not lat
                            or not lng
                            or not address
                            or not is_virtual
                            or not ars_id
                        ):
                            raise ValueError(f"{row_num}행: 필수값이 비어 있습니다.")

                        # 위도/경도를 float로 변환 가능한지 검사
                        try:
                            lat = float(lat)
                            lng = float(lng)
                        except ValueError:
                            raise ValueError(
                                f"{row_num}행: 위도/경도 값이 숫자가 아닙니다."
                            )

                        # Django 모델 객체 생성
                        station_list.append(
                            Bus_station(
                                stationId=station_id,
                                stationName=st_nm,
                                stn=stn,
                                locationX=lng,  # 경도
                                locationY=lat,  # 위도
                                address=address,
                                isVirtual=is_virtual,
                                arsId=ars_id,
                            )
                        )

                    # DB 저장
                    with transaction.atomic():
                        # 기존 정류소 데이터 전체 삭제
                        Bus_station.objects.all().delete()

                        # 새 CSV 데이터 한번에 저장
                        Bus_station.objects.bulk_create(station_list)

                    # 성공 메시지 출력
                    self.message_user(
                        request,
                        f"정류소 CSV 업로드 성공: {len(station_list)}건이 새로 저장되었습니다.",
                        level=messages.SUCCESS,
                    )
                    return redirect("..")

                except Exception as e:
                    # 업로드 중 예외가 발생하면 에러 메시지 출력
                    self.message_user(
                        request,
                        f"업로드 중 오류 발생: {str(e)}",
                        level=messages.ERROR,
                    )
                    return redirect("..")

        else:
            # GET 요청이면 빈 업로드 폼 화면 보여주기
            form = CsvImportForm()

        # 업로드 화면으로 전달할 데이터
        context = {
            "form": form,
            "title": "정류소 CSV 업로드",
        }
        return render(request, "admin/csv_upload.html", context)


# ============================================================
# route_station 관리자
# ------------------------------------------------------------
# 역할:
# 1) 노선-정류소 매핑 목록 표시
# 2) 노선/정류소 검색 지원
# 3) 노선-정류소 매핑 CSV 업로드 URL 연결
# 4) 업로드한 CSV를 route_station 테이블에 저장
#
# 중요:
# route_station은 외래키(FK) 테이블이므로
# CSV에 들어있는 routeId, stationId 값이
# 실제 bus_route, bus_station 테이블에 먼저 존재해야 함
# ============================================================
@admin.register(Route_station)
class RouteStationAdmin(admin.ModelAdmin):
    # 관리자 목록 페이지에서 보여줄 컬럼
    list_display = ("id", "route", "station", "staOrd")

    # 관리자 검색창에서 검색할 대상 컬럼
    search_fields = (
        "route__routeId",
        "route__routeName",
        "station__stationId",
        "station__stationName",
    )

    # 오른쪽 필터 영역
    list_filter = ("route", "station")

    # 기본 정렬 기준
    ordering = ("route", "staOrd")

    # 관리자 목록 페이지 템플릿 변경
    # 이 템플릿 안에 CSV 업로드 버튼이 들어감
    change_list_template = "admin/route_station_changelist.html"

    # --------------------------------------------------------
    # 관리자 페이지 전용 URL 추가
    # --------------------------------------------------------
    # /upload-csv/ 주소로 들어오면 upload_csv() 함수가 실행됨
    # --------------------------------------------------------
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                "upload-csv/",
                self.admin_site.admin_view(self.upload_csv),
                name="route_station_upload_csv",
            ),
        ]
        return custom_urls + urls

    # --------------------------------------------------------
    # 노선-정류소 매핑 CSV 업로드 처리
    # --------------------------------------------------------
    # CSV 예시 헤더:
    # routeId,stationId,staOrd
    #
    # 주의:
    # - routeId 는 bus_route 테이블에 있어야 함
    # - stationId 는 bus_station 테이블에 있어야 함
    # --------------------------------------------------------
    def upload_csv(self, request):
        # POST 요청이면 실제 업로드 처리
        if request.method == "POST":
            # 업로드한 파일 정보를 폼에 담아서 검사
            form = CsvImportForm(request.POST, request.FILES)

            if form.is_valid():
                # 업로드한 파일 꺼내기
                csv_file = form.cleaned_data["csv_file"]

                # CSV 확장자 검사
                if not csv_file.name.endswith(".csv"):
                    self.message_user(
                        request,
                        "CSV 파일만 업로드할 수 있습니다.",
                        level=messages.ERROR,
                    )
                    return redirect("..")

                try:
                    # 업로드 파일 읽기
                    decoded_file = csv_file.read().decode("utf-8-sig")
                    io_string = io.StringIO(decoded_file)

                    # CSV를 헤더 기준 딕셔너리 형태로 읽기
                    reader = csv.DictReader(io_string)

                    # CSV에 반드시 있어야 하는 헤더
                    required_headers = [
                        "routeId",
                        "stationId",
                        "staOrd",
                    ]

                    # 실제 업로드한 CSV의 헤더 목록
                    uploaded_headers = reader.fieldnames

                    # 헤더를 못 읽은 경우
                    if uploaded_headers is None:
                        self.message_user(
                            request,
                            "CSV 헤더를 읽을 수 없습니다.",
                            level=messages.ERROR,
                        )
                        return redirect("..")

                    # 필요한 헤더가 빠졌는지 검사
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

                    # ------------------------------------------------
                    # FK 조회용 딕셔너리 미리 생성
                    # ------------------------------------------------
                    # 이유:
                    # CSV 한 줄 읽을 때마다 DB 조회를 반복하면 느려질 수 있어서
                    # 미리 전체를 딕셔너리로 만들어서 빠르게 찾음
                    # ------------------------------------------------
                    route_map = {
                        route.routeId: route for route in Bus_route.objects.all()
                    }

                    station_map = {
                        station.stationId: station
                        for station in Bus_station.objects.all()
                    }

                    # DB에 넣기 전에 Route_station 객체들을 담아둘 리스트
                    route_station_list = []

                    # CSV를 한 줄씩 읽어서 모델 객체 생성
                    for row_num, row in enumerate(reader, start=2):
                        
                        route_id = (row.get("routeId") or "").strip()
                        station_id = (row.get("stationId") or "").strip()
                        sta_ord = (row.get("staOrd") or "").strip()

                        # 필수값 검사
                        if not route_id or not station_id or not sta_ord:
                            raise ValueError(f"{row_num}행: 필수값이 비어 있습니다.")

                        # staOrd 숫자 변환 검사
                        try:
                            sta_ord = int(sta_ord)
                        except ValueError:
                            raise ValueError(
                                f"{row_num}행: staOrd 값이 숫자가 아닙니다."
                            )

                        # routeId로 실제 Bus_route 객체 찾기
                        route_obj = route_map.get(route_id)
                        if route_obj is None:
                            raise ValueError(
                                f"{row_num}행: bus_route에 없는 routeId입니다. ({route_id})"
                            )

                        # stationId로 실제 Bus_station 객체 찾기
                        station_obj = station_map.get(station_id)
                        if station_obj is None:
                            raise ValueError(
                                f"{row_num}행: bus_station에 없는 stationId입니다. ({station_id})"
                            )

                        # Django 모델 객체 생성
                        route_station_list.append(
                            Route_station(
                                
                                route=route_obj,
                                station=station_obj,
                                staOrd=sta_ord,
                            )
                        )

                    # DB 저장
                    with transaction.atomic():
                        # 기존 노선-정류소 매핑 데이터 전체 삭제
                        Route_station.objects.all().delete()

                        # 새 CSV 데이터 한번에 저장
                        Route_station.objects.bulk_create(route_station_list)

                    # 성공 메시지 출력
                    self.message_user(
                        request,
                        f"노선-정류소 매핑 CSV 업로드 성공: {len(route_station_list)}건이 새로 저장되었습니다.",
                        level=messages.SUCCESS,
                    )
                    return redirect("..")

                except Exception as e:
                    # 업로드 중 예외가 발생하면 에러 메시지 출력
                    self.message_user(
                        request,
                        f"업로드 중 오류 발생: {str(e)}",
                        level=messages.ERROR,
                    )
                    return redirect("..")

        else:
            # GET 요청이면 빈 업로드 폼 화면 보여주기
            form = CsvImportForm()

        # 업로드 화면으로 전달할 데이터
        context = {
            "form": form,
            "title": "노선-정류소 매핑 CSV 업로드",
        }
        return render(request, "admin/csv_upload.html", context)
