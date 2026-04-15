export function getFavorites() {
    const data = localStorage.getItem("favorites");
    return data ? JSON.parse(data) : { buses: [], stations: [] };
}

export function saveFavorites(favorites) {
    localStorage.setItem("favorites", JSON.stringify(favorites));
}

export function addFavorite(type, name) {
    const favorites = getFavorites();
    const limit = 5;

    if (type === "bus") {
        if (favorites.buses.includes(name)) {
            showToast("이미 즐겨찾기에 있는 버스예요.");
            return;
        }

        if (favorites.buses.length >= limit) {
            showToast("버스 즐겨찾기는 최대 5개까지 저장할 수 있어요.");
            return;
        }

        favorites.buses.push(name);
        saveFavorites(favorites);
        showToast("버스 즐겨찾기에 추가됐어요.");
        return;
    }

    if (type === "station") {
        if (favorites.stations.includes(name)) {
            showToast("이미 즐겨찾기에 있는 정류장이에요.");
            return;
        }

        if (favorites.stations.length >= limit) {
            showToast("정류장 즐겨찾기는 최대 5개까지 저장할 수 있어요.");
            return;
        }

        favorites.stations.push(name);
        saveFavorites(favorites);
        showToast("정류장 즐겨찾기에 추가됐어요.");
        return;
    }
}
