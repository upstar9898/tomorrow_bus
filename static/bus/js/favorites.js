import { getFavorites, saveFavorites } from "./favorite.js";

function removeFavorite(type, value) {
    const favorites = getFavorites();

    if (type === "bus") {
        favorites.buses = favorites.buses.filter(
            (item) => String(item) !== String(value)
        );
    }

    if (type === "station") {
        favorites.stations = favorites.stations.filter(
            (item) =>
                !(
                    String(item.arsId) === String(value.arsId) &&
                    String(item.routeId) === String(value.routeId)
                )
        );
    }

    saveFavorites(favorites);
    renderFavorites();
}

function clearAllFavorites() {
    localStorage.removeItem("favorites");
    renderFavorites();
}

async function fetchRouteName(routeId) {
    try {
        const response = await fetch(
            `/favorite/route-name/?routeId=${encodeURIComponent(routeId)}`
        );
        const data = await response.json();

        if (!response.ok || !data.success) {
            return routeId;
        }

        return data.data.route_name || routeId;
    } catch (error) {
        console.error("노선명 조회 실패:", error);
        return routeId;
    }
}

function createRouteBadge(routeName) {
    const badge = document.createElement("span");
    badge.className = "favorite-badge";
    badge.textContent = routeName;
    return badge;
}

async function renderFavorites() {
    const favorites = getFavorites();

    const busList = document.getElementById("busList");
    const stationList = document.getElementById("stationList");
    const busEmpty = document.getElementById("busEmpty");
    const stationEmpty = document.getElementById("stationEmpty");

    busList.innerHTML = "";
    stationList.innerHTML = "";

    if (favorites.buses.length === 0) {
        busEmpty.style.display = "block";
    } else {
        busEmpty.style.display = "none";

        for (const bus of favorites.buses) {
            const routeName = await fetchRouteName(bus);

            const li = document.createElement("li");
            const itemWrap = document.createElement("div");
            itemWrap.className = "favorite-item-wrap";

            const nameSpan = document.createElement("span");
            nameSpan.className = "favorite-item-name";
            nameSpan.textContent = routeName;

            itemWrap.appendChild(nameSpan);

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "btn btn-delete";
            deleteBtn.textContent = "삭제";
            deleteBtn.addEventListener("click", () => {
                removeFavorite("bus", bus);
            });

            li.appendChild(itemWrap);
            li.appendChild(deleteBtn);
            busList.appendChild(li);
        }
    }

    if (favorites.stations.length === 0) {
        stationEmpty.style.display = "block";
    } else {
        stationEmpty.style.display = "none";

        for (const station of favorites.stations) {
            const li = document.createElement("li");
            const itemWrap = document.createElement("div");
            itemWrap.className = "favorite-item-wrap";

            const nameSpan = document.createElement("span");
            nameSpan.className = "favorite-item-name";
            nameSpan.textContent = station.stationName || "정류소명 없음";
            itemWrap.appendChild(nameSpan);

            if (station.routeName) {
                const badgeWrap = document.createElement("div");
                badgeWrap.style.display = "flex";
                badgeWrap.style.flexWrap = "wrap";
                badgeWrap.style.gap = "8px";
                badgeWrap.style.marginTop = "6px";

                badgeWrap.appendChild(createRouteBadge(station.routeName));
                itemWrap.appendChild(badgeWrap);
            } else {
                const subText = document.createElement("span");
                subText.className = "favorite-sub-text";
                subText.textContent = "노선 정보가 없습니다.";
                itemWrap.appendChild(subText);
            }

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "btn btn-delete";
            deleteBtn.textContent = "삭제";
            deleteBtn.addEventListener("click", () => {
                removeFavorite("station", {
                    arsId: station.arsId,
                    routeId: station.routeId,
                });
            });

            li.appendChild(itemWrap);
            li.appendChild(deleteBtn);
            stationList.appendChild(li);
        }
    }
}

document.addEventListener("DOMContentLoaded", renderFavorites);
window.clearAllFavorites = clearAllFavorites;