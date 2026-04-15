import random

from bus.models import Route_station



def dummy_prob(remaining_seat):
    if remaining_seat >= 10:
        return round(random.uniform(0.01, 0.05), 4)
    elif remaining_seat > 2:
        return round(random.uniform(0.5, 0.8), 4)
    else:
        return round(random.uniform(0.95, 0.99), 4)

def dummy_service1(routeId, stationId, date_time):
    remaining_seat = random.randint(0, 45)
    full_prob = dummy_prob(remaining_seat)
    return {
        "routeId": routeId,
        "stationId": stationId,
        "date_time": date_time,
        "remaining_seat": remaining_seat,
        "full_prob": full_prob,
    }

def dummy_service2(route_id, station_id, date_time):
    route_stations = (
        Route_station.objects
        .filter(route_id=route_id)
        .select_related("station")
        .order_by("staOrd")
    )

    stops = []
    selected_station_name = None

    for idx, rs in enumerate(route_stations, start=1):
        is_selected = str(rs.station.stationId) == str(station_id)

        if is_selected:
            selected_station_name = rs.station.stationName

        is_virtual = getattr(rs, "is_virtual", 0)

        if is_virtual == 1:
            remaining_seat = None
            full_prob = None
            predicted_time = ""
        else:
            remaining_seat = random.randint(0, 45)
            full_prob = dummy_prob(remaining_seat)
            predicted_time = ""

        stops.append({
            "station_id": rs.station.stationId,
            "station_name": rs.station.stationName,
            "ars_id": rs.station.arsId,
            "predicted_time": predicted_time,
            "remaining_seat": remaining_seat,
            "full_prob": full_prob,
            "is_selected": is_selected,
            "is_virtual": is_virtual,
        })

    return {
        "route_id": route_id,
        "selected_station_id": station_id,
        "selected_station_name": selected_station_name,
        "date_time": date_time,
        "stops": stops,
    }