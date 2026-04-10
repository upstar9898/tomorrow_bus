import random


def dummy_service1(routeId, stationId, date_time):
    remaining_seat = random.randint(0, 45)
    full_prob = round(random.random(), 4)
    return {
        "routeId": routeId,
        "stationId": stationId,
        "date_time": date_time,
        "remaining_seat": remaining_seat,
        "full_prob": full_prob,
    }
