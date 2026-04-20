import { showToast } from "./utils.js";

export function getFavorites() {
    const data = localStorage.getItem("favorites");

    if (!data) {
        return { buses: [], stations: [] };
    }

    try {
        const parsed = JSON.parse(data);

        return {
            buses: Array.isArray(parsed.buses) ? parsed.buses : [],
            stations: Array.isArray(parsed.stations) ? parsed.stations : [],
        };
    } catch (error) {
        console.error("즐겨찾기 파싱 실패:", error);
        return { buses: [], stations: [] };
    }
}

export function saveFavorites(favorites) {
    localStorage.setItem("favorites", JSON.stringify(favorites));
}

export function addFavorite(type, payload) {
    const favorites = getFavorites();
    const limit = 5;

    if (type === "bus") {
        const routeId = String(payload);

        if (favorites.buses.includes(routeId)) {
            showToast("이미 즐겨찾기에 있는 버스예요.");
            return;
        }

        if (favorites.buses.length >= limit) {
            showToast("버스 즐겨찾기는 최대 5개까지 저장할 수 있어요.");
            return;
        }

        favorites.buses.push(routeId);
        saveFavorites(favorites);
        showToast("버스 즐겨찾기에 추가됐어요.");
        return;
    }

    if (type === "station") {
        const stationItem = payload;

        if (
            !stationItem ||
            !stationItem.stationName ||
            !stationItem.arsId ||
            !stationItem.routeId ||
            !stationItem.routeName
        ) {
            showToast("정류장 즐겨찾기 정보가 올바르지 않아요.");
            return;
        }

        const arsId = String(stationItem.arsId);
        const routeId = String(stationItem.routeId);

        const alreadyExists = favorites.stations.some(
            (item) =>
                String(item.arsId) === arsId &&
                String(item.routeId) === routeId
        );

        if (alreadyExists) {
            showToast("이미 즐겨찾기에 있는 정류장이에요.");
            return;
        }

        if (favorites.stations.length >= limit) {
            showToast("정류장 즐겨찾기는 최대 5개까지 저장할 수 있어요.");
            return;
        }

        favorites.stations.push({
            stationName: stationItem.stationName,
            arsId,
            routeId,
            routeName: stationItem.routeName,
        });

        saveFavorites(favorites);
        showToast("정류장 즐겨찾기에 추가됐어요.");
    }
}