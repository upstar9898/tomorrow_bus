from django.shortcuts import render

from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST


import json
from datetime import datetime, timedelta

from .models import Bus_route, Route_station, Bus_arrival_info, Weather_station
from .services.ml_predictor import predict_service1_result
from bus.services.route_service import predict_route_service
from .services import weather_collector
from django.db.models import Max
from django.db.models.functions import Abs
from django.conf import settings
from django.utils import timezone

def index(request):
    return render(request, "index.html")


def service1(request):
    routes = Bus_route.objects.all().order_by("routeName")
    return render(request, "service1.html", {"routes": routes})


def service2(request):
    routes = Bus_route.objects.all().order_by("routeName")
    return render(
        request,
        "service2.html",
        {"routes": routes, "KAKAO_JS_KEY": settings.KAKAO_JS_KEY},
    )


def favorites(request):
    return render(request, "favorites.html")


@require_GET
def get_stations_by_route(request):
    route_id = request.GET.get("route_id")

    if not route_id:
        return JsonResponse(
            {"success": False, "error": "route_id가 필요합니다."},
            status=400,
        )

    route_stations = (
        Route_station.objects
        .filter(route_id=route_id)
        .select_related("station")
        .order_by("staOrd")
    )

    stations = [
        {
            "station_id": rs.station.stationId,
            "station_name": rs.station.stationName,
            "ars_id": rs.station.arsId,
            "staOrd": rs.staOrd,   # 여기 중요
        }
        for rs in route_stations
    ]

    return JsonResponse(
        {
            "success": True,
            "data": {
                "stations": stations,
            },
        }
    )


@require_POST
def predict_service1(request):
    try:
        data = json.loads(request.body)

        route_id = data.get("route_id")
        station_id = data.get("station_id")
        date_time = data.get("date_time")

        if not route_id or not station_id or not date_time:
            return JsonResponse(
                {
                    "success": False,
                    "error": "route_id, station_id, date_time은 필수입니다.",
                },
                status=400,
            )

        precipitation = 0

        weather_used = False      # 모델에 반영 여부
        weather_fetched = False   # ⭐ 실제 API 성공 여부
        weather_info = None

        # 5일 이내일 때만 날씨 API 호출
        if is_within_5_days(date_time):
            try:
                forecast_data = weather_collector.collect_forecast(
                    station_id=station_id,
                    target_dt=date_time,
                )

                forecast = forecast_data["forecast"]
                weather_main = forecast["weather"][0]["main"]

                precipitation = 1 if weather_main in ["Rain", "Snow", "Drizzle"] else 0

                weather_used = True
                weather_fetched = True   # ⭐ 여기 핵심

                # print("[날씨 API 성공]", weather_fetched)

                weather_info = {
                    "main": weather_main,
                    "description": forecast["weather"][0].get("description"),
                    "dt_txt": forecast.get("dt_txt"),
                }

            except Exception as e:
                # print("[날씨 API 실패]", e)

                precipitation = 0
                weather_used = False
                weather_fetched = False  # ⭐ 실패

                weather_info = {
                    "reason": "API 호출 실패"
                }

        else:
            weather_fetched = False  # ⭐ 호출 자체 안함

            # print("[날씨 API 안가져옴]", weather_fetched)

            weather_info = {
                "reason": "5일 이후 예보 범위 초과"
            }

        result = predict_service1_result(
            route_id=route_id,
            station_id=station_id,
            date_time=date_time,
            precipitation=precipitation,
        )

        result = convert_numpy(result)

        result["weather_used"] = weather_used
        result["weather_fetched"] = weather_fetched  # ⭐ 추가
        result["weather_info"] = weather_info
        result["precipitation"] = precipitation

        return JsonResponse(
            {
                "success": True,
                "data": result,
            }
        )

    except json.JSONDecodeError:
        return JsonResponse(
            {"success": False, "error": "잘못된 JSON 요청입니다."},
            status=400,
        )
    except Exception as e:
        return JsonResponse(
            {"success": False, "error": str(e)},
            status=500,
        )
    

# 미사용 함수, 추후 서비스2에 사용 가능성 있음
@require_GET
def get_route_seat_chart(request):
    route_id = request.GET.get("route_id")
    target_date = request.GET.get("date")  # 예: 2026-04-01
    target_time = request.GET.get("time")  # 예: 06:33

    if not route_id or not target_date or not target_time:
        return JsonResponse(
            {"success": False, "error": "route_id, date, time은 필수입니다."},
            status=400,
        )

    try:
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        target_time_obj = datetime.strptime(target_time, "%H:%M").time()
        target_seconds = target_time_obj.hour * 3600 + target_time_obj.minute * 60
    except ValueError:
        return JsonResponse(
            {"success": False, "error": "date 또는 time 형식이 올바르지 않습니다."},
            status=400,
        )

    # 월~금: 평일, 토~일: 주말
    is_weekend = target_date_obj.weekday() >= 5

    qs = Bus_arrival_info.objects.filter(route_id=route_id)

    # Django week_day: 일=1, 월=2, ..., 토=7
    if is_weekend:
        qs = qs.filter(mkTm__week_day__in=[1, 7])
    else:
        qs = qs.exclude(mkTm__week_day__in=[1, 7])

    if not qs.exists():
        return JsonResponse(
            {"success": False, "error": "해당 조건의 데이터가 없습니다."},
            status=404,
        )

    candidate_mktms = qs.values_list("mkTm", flat=True).distinct()

    nearest_mkTm = None
    min_diff = None

    for mk in candidate_mktms:
        mk_seconds = mk.hour * 3600 + mk.minute * 60 + mk.second
        diff = abs(mk_seconds - target_seconds)

        if min_diff is None or diff < min_diff:
            min_diff = diff
            nearest_mkTm = mk

    if nearest_mkTm is None:
        return JsonResponse(
            {"success": False, "error": "가장 가까운 시각 데이터를 찾지 못했습니다."},
            status=404,
        )

    snapshot_qs = (
        Bus_arrival_info.objects.filter(route_id=route_id, mkTm=nearest_mkTm)
        .select_related("station", "route")
        .order_by("staOrd")
    )

    chart_data = [
        {
            "sta_ord": row.staOrd,
            "station_id": row.station.stationId,
            "station_name": row.station.stationName,
            "remaining_seat": row.remaining_seat,
            "full_flag": row.full_flag,
        }
        for row in snapshot_qs
    ]

    return JsonResponse(
        {
            "success": True,
            "data" : {
            "route_id": route_id,
            "route_name": snapshot_qs.first().route.routeName
            if snapshot_qs.exists()
            else "",
            "requested_date": target_date,
            "requested_time": target_time,
            "day_type": "weekend" if is_weekend else "weekday",
            "nearest_mkTm": nearest_mkTm.strftime("%Y-%m-%d %H:%M:%S"),
            "stations": chart_data,
            }
        }
    )


@require_GET
def get_station_week_chart(request):
    route_id = request.GET.get("route_id")
    station_id = request.GET.get("station_id")
    target_date = request.GET.get("date")  # 예: 2026-04-15
    target_time = request.GET.get("time")  # 예: 07:30

    if not route_id or not station_id or not target_date or not target_time:
        return JsonResponse(
            {
                "success": False,
                "error": "route_id, station_id, date, time은 모두 필요합니다.",
            },
            status=400,
        )

    try:
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        target_time_obj = datetime.strptime(target_time, "%H:%M").time()
    except ValueError:
        return JsonResponse(
            {
                "success": False,
                "error": "date 또는 time 형식이 올바르지 않습니다.",
            },
            status=400,
        )

    target_seconds = (
        target_time_obj.hour * 3600
        + target_time_obj.minute * 60
        + target_time_obj.second
    )

    # Python weekday: 월=0 ... 일=6
    is_weekend = target_date_obj.weekday() >= 5

    if is_weekend:
        weekday_groups = [
            ("토", 7),  # Django week_day 기준
            ("일", 1),
        ]
        day_type = "weekend"
    else:
        weekday_groups = [
            ("월", 2),
            ("화", 3),
            ("수", 4),
            ("목", 5),
            ("금", 6),
        ]
        day_type = "weekday"

    base_qs = Bus_arrival_info.objects.filter(
        route_id=route_id,
        station_id=station_id,
    ).select_related("route", "station")

    if not base_qs.exists():
        return JsonResponse(
            {
                "success": False,
                "error": "해당 노선/정류소 데이터가 없습니다.",
            },
            status=404,
        )

    first_row = base_qs.first()
    route_name = first_row.route.routeName
    station_name = first_row.station.stationName

    chart_data = []

    SAMPLE_SIZE_PER_DAY = 10

    for label, week_day_value in weekday_groups:
        day_rows = base_qs.filter(mkTm__week_day=week_day_value)

        # (시간차이, row) 형태로 저장
        diff_rows = []

        for row in day_rows:
            row_seconds = row.mkTm.hour * 3600 + row.mkTm.minute * 60 + row.mkTm.second
            diff = abs(row_seconds - target_seconds)
            diff_rows.append((diff, row))

        # 요청 시간과 가까운 순 정렬
        diff_rows.sort(key=lambda x: x[0])

        # 가장 가까운 10개 (SAMPLE_SIZE_PER_DAY)만 선택
        nearest_rows = [row for diff, row in diff_rows[:SAMPLE_SIZE_PER_DAY]]

        if nearest_rows:
            avg_remaining_seat = round(
                sum(row.remaining_seat for row in nearest_rows) / len(nearest_rows), 1
            )

            chart_data.append(
                {
                    "day_label": label,
                    "remaining_seat": avg_remaining_seat,
                    "sample_count": len(nearest_rows),
                }
            )
        else:
            chart_data.append(
                {
                    "day_label": label,
                    "remaining_seat": 0,
                    "sample_count": 0,
                }
            )

    return JsonResponse(
        {
            "success": True,
            "data": {
                "route_id": route_id,
                "route_name": route_name,
                "station_id": station_id,
                "station_name": station_name,
                "requested_date": target_date,
                "requested_time": target_time,
                "day_type": day_type,
                "sample_size_per_day": SAMPLE_SIZE_PER_DAY,
                "bars": chart_data,
            },
        }
    )


@require_POST
def predict_service2(request):
    try:
        data = json.loads(request.body)

        route_id = data.get("route_id")
        station_id = data.get("station_id")
        date_time = data.get("date_time")

        if not route_id or not station_id or not date_time:
            return JsonResponse(
                {
                    "success": False,
                    "error": "route_id, station_id, date_time은 필수입니다.",
                },
                status=400,
            )

        stops = predict_route_service(
            route_id=route_id,
            station_id=station_id,
            target_datetime=date_time,
        )

        def normalize_stn(stn):
            if stn is None:
                return None

            if hasattr(stn, "pk"):
                stn = stn.pk

            stn = str(stn).strip()

            if stn in ["", "None", "NULL", "nan"]:
                return None

            if stn.endswith(".0"):
                stn = stn[:-2]

            return stn

        weather_fetched = False   # 실제 날씨 API 성공 여부
        weather_used = False      # 예측/정류소 데이터에 날씨 반영 여부
        weather_info = {
            "requested_stn_count": 0,
            "success_stn_count": 0,
            "failed_stn_count": 0,
            "reason": None,
        }
        
        # 1. stn 수집
        unique_stn_ids = {
            normalize_stn(stop.get("stn"))
            for stop in stops
            if normalize_stn(stop.get("stn")) is not None
        }

        weather_info["requested_stn_count"] = len(unique_stn_ids)  # ⭐ 추가

        # 2. API 호출
        forecast_map = {}

        if is_within_5_days(date_time):
            for stn_id in unique_stn_ids:
                try:
                    forecast_map[stn_id] = weather_collector.collect_forecast_by_stn(
                        stn_id=stn_id,
                        target_dt=date_time,
                    )
                except Exception as e:
                    print(f"[날씨 API 실패] stn={stn_id}, 오류={e}")

            weather_info["success_stn_count"] = len(forecast_map)
            weather_info["failed_stn_count"] = len(unique_stn_ids) - len(forecast_map)

            if forecast_map:
                weather_fetched = True
                weather_used = True
                weather_info["reason"] = "날씨 API 일부 또는 전체 성공"
            else:
                weather_info["reason"] = "날씨 API 호출 실패 또는 사용 가능한 날씨 데이터 없음"

        else:
            print("[날씨 API 안가져옴]")
            weather_info["failed_stn_count"] = len(unique_stn_ids)
            weather_info["reason"] = "5일 이후 예보 범위 초과"

        # 3. 매핑 (정리된 버전)
        for stop in stops:
            stn = normalize_stn(stop.get("stn"))

            if stn is None:
                stop["precipitation"] = 0
                continue

            forecast_data = forecast_map.get(stn)

            if not forecast_data:
                stop["precipitation"] = 0
                continue

            weather_main = forecast_data["forecast"]["weather"][0]["main"]

            stop["precipitation"] = 1 if weather_main in ["Rain", "Snow", "Drizzle"] else 0

        result = {}
        result["stops"] = stops

        # 전체 결과 기준 날씨 상태
        result["weather_fetched"] = weather_fetched
        result["weather_used"] = weather_used
        result["weather_info"] = weather_info
        
        return JsonResponse(
            {
                "success": True,
                "data": result,
            },
            status=200,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse(
            {"success": False, "error": str(e)},
            status=500,
        )
    
@require_GET
def get_route_map_data(request):
    route_id = request.GET.get("route_id")

    if not route_id:
        return JsonResponse(
            {"success": False, "error": "route_id가 필요합니다."}, status=400
        )

    route_stations = (
        Route_station.objects.filter(route_id=route_id)
        .select_related("station")
        .order_by("staOrd")
    )

    stations = []
    for rs in route_stations:
        st = rs.station
        if st.locationY is None or st.locationX is None:
            continue

        stations.append(
            {
                "station_id": st.stationId,
                "station_name": st.stationName,
                "ars_id": st.arsId,
                "latitude": st.locationY,
                "longitude": st.locationX,
                "is_virtual": st.isVirtual,
                "staOrd": rs.staOrd,
            }
        )

    return JsonResponse({
        "success": True,
        "data" : {
            "stations": stations,
        }   
    })

@require_GET
def get_route_name(request):
    route_id = request.GET.get("routeId")

    if not route_id:
        return JsonResponse(
            {"success": False, "error": "routeId가 필요합니다."},
            status=400,
        )

    try:
        route = Bus_route.objects.get(routeId=route_id)
    except Bus_route.DoesNotExist:
        return JsonResponse(
            {"success": False, "error": "해당 노선을 찾을 수 없습니다."},
            status=404,
        )

    return JsonResponse(
        {
            "success": True,
            "data": {
                "route_id": route.routeId,
                "route_name": route.routeName,
            },
        }
    )

def convert_numpy(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj
    
def is_within_5_days(date_time_str):
    """
    OpenWeather 5 day / 3 hour forecast 사용 가능 범위 체크
    """
    target_dt = datetime.fromisoformat(date_time_str)

    if timezone.is_naive(target_dt):
        target_dt = timezone.make_aware(target_dt)

    now = timezone.now()

    return now <= target_dt <= now + timedelta(days=5)