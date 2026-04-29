import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 공식 노선 path 우선
ROUTE_PATH_DIR = BASE_DIR / "data" / "route_path_geometries"

# 기존 OSRM 캐시 fallback
OSRM_GEOMETRY_DIR = BASE_DIR / "data" / "route_geometries"


def load_json(path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_route_map_data(route_id):
    route_id = str(route_id).strip()

    official_path = ROUTE_PATH_DIR / f"{route_id}.json"
    osrm_path = OSRM_GEOMETRY_DIR / f"{route_id}.json"

    data = load_json(official_path)
    source = "official_route_path"

    if data is None:
        data = load_json(osrm_path)
        source = "osrm_cached"

    if data is None:
        return {
            "stops": [],
            "route_coords": [],
            "source": "not_found",
        }

    return {
        "stops": data.get("stops", []),
        "route_coords": data.get("route_coords", []),
        "source": source,
    }