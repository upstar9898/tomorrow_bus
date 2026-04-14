from django.shortcuts import render

from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST

import json

from service_test.backend_test import dummy_service1  # 실제 서비스로 변경 필요

from .models import Bus_route, Route_station, Bus_arrival_info

from datetime import datetime
from django.db.models import F
from django.db.models.functions import Abs


def index(request):
    return render(request, "index.html")


def service1(request):
    routes = Bus_route.objects.all().order_by("routeName")
    return render(request, "service1.html", {"routes": routes})


def service2(request):
    return render(request, "service2.html")


def favorites(request):
    return render(request, "favorites.html")


@require_GET
def get_stations_by_route(request):
    route_id = request.GET.get("routeId")

    if not route_id:
        return JsonResponse(
            {"success": False, "error": "routeId가 필요합니다."},
            status=400,
        )

    route_stations = (
        Route_station.objects.filter(route_id=route_id)
        .select_related("station")
        .order_by("staOrd")
    )

    stations = [
        {
            "stationId": rs.station.stationId,
            "stationName": rs.station.stationName,
            "arsId": rs.station.arsId,
            "staOrd": rs.staOrd,
        }
        for rs in route_stations
    ]

    return JsonResponse(
        {
            "success": True,
            "stations": stations,
        }
    )


@require_POST
def predict_service1(request):
    try:
        data = json.loads(request.body)

        route_id = data.get("routeId")
        station_id = data.get("stationId")
        date_time = data.get("date_time")

        if not route_id or not station_id or not date_time:
            return JsonResponse(
                {
                    "success": False,
                    "error": "routeId, stationId, date_time은 필수입니다.",
                },
                status=400,
            )

        result = dummy_service1(
            route_id, station_id, date_time
        )  # 실제 서비스로 변경 필요

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


@require_GET
def get_route_seat_chart(request):
    route_id = request.GET.get("routeId")
    target_date = request.GET.get("date")   # 예: 2026-04-01
    target_time = request.GET.get("time")   # 예: 06:33

    if not route_id or not target_date or not target_time:
        return JsonResponse(
            {"success": False, "error": "routeId, date, time은 필수입니다."},
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
        Bus_arrival_info.objects
        .filter(route_id=route_id, mkTm=nearest_mkTm)
        .select_related("station", "route")
        .order_by("staOrd")
    )

    chart_data = [
        {
            "staOrd": row.staOrd,
            "stationId": row.station.stationId,
            "stationName": row.station.stationName,
            "remaining_seat": row.remaining_seat,
            "full_flag": row.full_flag,
        }
        for row in snapshot_qs
    ]

    return JsonResponse(
        {
            "success": True,
            "routeId": route_id,
            "routeName": snapshot_qs.first().route.routeName if snapshot_qs.exists() else "",
            "requestedDate": target_date,
            "requestedTime": target_time,
            "dayType": "weekend" if is_weekend else "weekday",
            "nearestMkTm": nearest_mkTm.strftime("%Y-%m-%d %H:%M:%S"),
            "stations": chart_data,
        }
    )