from django.shortcuts import render

from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST

import json

from service_test.backend_test import dummy_service1, dummy_service2 # 실제 서비스로 변경 필요

from .models import Bus_route, Route_station

def index(request):
    return render(request, "index.html")


def service1(request):
    routes = Bus_route.objects.all().order_by("routeName")
    return render(request, "service1.html", {"routes": routes})


def service2(request):
    routes = Bus_route.objects.all().order_by("routeName")
    return render(request, "service2.html", {"routes": routes})


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
        Route_station.objects
        .filter(route_id=route_id)
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

    return JsonResponse({
        "success": True,
        "stations": stations,
    })


@require_POST
def predict_service1(request):
    try:
        data = json.loads(request.body)

        route_id = data.get("routeId")
        station_id = data.get("stationId")
        date_time = data.get("date_time")

        if not route_id or not station_id or not date_time:
            return JsonResponse(
                {"success": False, "error": "routeId, stationId, date_time은 필수입니다."},
                status=400,
            )

        result = dummy_service1(route_id, station_id, date_time) # 실제 서비스로 변경 필요

        return JsonResponse({
            "success": True,
            "data": result,
        })

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
        
@require_POST
def predict_service2(request):
    try:
        data = json.loads(request.body)

        route_id = data.get("routeId")
        station_id = data.get("stationId")
        date_time = data.get("date_time")

        if not route_id or not station_id or not date_time:
            return JsonResponse(
                {"success": False, "error": "routeId, stationId, date_time은 필수입니다."},
                status=400,
            )

        result = dummy_service2(route_id, station_id, date_time)  # 실제 서비스로 변경 필요

        return JsonResponse({
            "success": True,
            "data": result,
        })

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