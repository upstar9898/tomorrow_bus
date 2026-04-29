import os
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "route_station_mapping.csv"
OUTPUT_DIR = BASE_DIR / "data" / "route_path_geometries"

API_URL = "http://ws.bus.go.kr/api/rest/busRouteInfo/getRoutePath"
# 문서/호출 결과에 따라 getRoutePathList가 맞으면 아래로 바꿔봐
# API_URL = "http://ws.bus.go.kr/api/rest/busRouteInfo/getRoutePathList"

load_dotenv()
SERVICE_KEY = os.getenv("SEOUL_BUS_API_KEY")


def normalize_id(value):
    if pd.isna(value):
        return None
    return str(int(float(value)))


def load_route_ids():
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", dtype=str, low_memory=False)

    df["route_id"] = df["ROUTE_ID"].apply(normalize_id)
    df["route_name"] = df["노선명"].astype(str)

    route_df = (
        df[["route_id", "route_name"]]
        .dropna()
        .drop_duplicates()
        .sort_values("route_id")
    )

    return route_df.to_dict("records")


def parse_items_xml(xml_text):
    root = ET.fromstring(xml_text)

    header_cd = root.findtext(".//headerCd")
    header_msg = root.findtext(".//headerMsg")

    if header_cd not in [None, "0"]:
        print(f"[API ERROR] {header_cd} / {header_msg}")
        return []

    items = []

    for item in root.findall(".//itemList"):
        gps_x = item.findtext("gpsX")
        gps_y = item.findtext("gpsY")
        no = item.findtext("no") or item.findtext("seq")

        if gps_x is None or gps_y is None:
            continue

        items.append({
            "seq": int(no) if no and no.isdigit() else len(items) + 1,
            "lng": float(gps_x),
            "lat": float(gps_y),
        })

    items.sort(key=lambda x: x["seq"])
    return items


def fetch_route_path(route_id):
    params = {
        "serviceKey": SERVICE_KEY,
        "busRouteId": route_id,
    }

    response = requests.get(API_URL, params=params, timeout=10)
    response.raise_for_status()

    return parse_items_xml(response.text)


def save_route_path(route_id, route_name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path_items = fetch_route_path(route_id)

    if not path_items:
        print(f"[SKIP] path 없음: {route_id} / {route_name}")
        return

    route_coords = [
        [item["lat"], item["lng"]]
        for item in path_items
    ]

    result = {
        "route_id": route_id,
        "route_name": route_name,
        "source": "seoul_bus_getRoutePath",
        "route_coords": route_coords,
        "path_items": path_items,
    }

    output_path = OUTPUT_DIR / f"{route_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {output_path} / coords={len(route_coords)}")


if __name__ == "__main__":
    if not SERVICE_KEY:
        raise ValueError(".env에 SEOUL_BUS_API_KEY가 없습니다.")

    routes = load_route_ids()
    print(f"총 노선 수: {len(routes)}")

    for idx, route in enumerate(routes, start=1):
        route_id = route["route_id"]
        route_name = route["route_name"]

        print("=" * 60)
        print(f"[{idx}/{len(routes)}] {route_name} / {route_id}")

        try:
            save_route_path(route_id, route_name)
            time.sleep(0.2)
        except Exception as e:
            print(f"[FAIL] {route_id} / {route_name}: {e}")