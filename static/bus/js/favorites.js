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

function setFavoriteDataset(element, { type, routeId = "", stationId = "", arsId = "" }) {
    element.dataset.type = type;

    if (routeId) {
        element.dataset.routeId = String(routeId);
    }

    if (stationId) {
        element.dataset.stationId = String(stationId);
    }

    if (arsId) {
        element.dataset.arsId = String(arsId);
    }
}

function moveToPredictPageByBus(routeId) {
    const url = new URL("/service1/", window.location.origin);
    url.searchParams.set("route_id", routeId);
    window.location.href = url.toString();
}

function moveToPredictPageByStation({ stationId, routeId, arsId }) {
    const url = new URL("/service1/", window.location.origin);

    if (routeId) url.searchParams.set("route_id", routeId);
    if (stationId) url.searchParams.set("station_id", stationId);
    if (arsId) url.searchParams.set("ars_id", arsId);

    window.location.href = url.toString();
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
            itemWrap.style.cursor = "pointer";

            setFavoriteDataset(itemWrap, {
                type: "bus",
                routeId: bus,
            });

            const nameSpan = document.createElement("span");
            nameSpan.className = "favorite-item-name";
            nameSpan.textContent = routeName;

            itemWrap.appendChild(nameSpan);

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "btn btn-delete";
            deleteBtn.textContent = "삭제";
            deleteBtn.addEventListener("click", (event) => {
                event.stopPropagation();
                removeFavorite("bus", bus);
            });

            
            const btnWrap = document.createElement("div");
            btnWrap.style.display = "flex";
            btnWrap.style.gap = "8px";

            // 서비스1 버튼
            const service1Btn = document.createElement("button");
            service1Btn.className = "btn btn-secondary";
            service1Btn.textContent = "서비스1";
            service1Btn.addEventListener("click", (event) => {
                event.stopPropagation();
                moveToPredictPageByBus(bus);
            });

            // 서비스2 버튼
            const service2Btn = document.createElement("button");
            service2Btn.className = "btn btn-secondary";
            service2Btn.textContent = "서비스2";
            service2Btn.addEventListener("click", (event) => {
                event.stopPropagation();
                moveToPredictPageByBus2(bus);
            });

            btnWrap.appendChild(service1Btn);
            btnWrap.appendChild(service2Btn);

            const actionWrap = document.createElement("div");
            actionWrap.className = "favorite-action-wrap";

            actionWrap.appendChild(service1Btn);
            actionWrap.appendChild(service2Btn);
            actionWrap.appendChild(deleteBtn);

            li.appendChild(itemWrap);
            li.appendChild(actionWrap);
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
            itemWrap.style.cursor = "pointer";

            setFavoriteDataset(itemWrap, {
                type: "station",
                routeId: station.routeId,
                stationId: station.stationId,
                arsId: station.arsId,
            });

            itemWrap.addEventListener("click", () => {
                moveToPredictPageByStation({
                    routeId: itemWrap.dataset.routeId,
                    stationId: itemWrap.dataset.stationId,
                    arsId: itemWrap.dataset.arsId,
                });
            });

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
            deleteBtn.addEventListener("click", (event) => {
                event.stopPropagation();
                removeFavorite("station", {
                    arsId: station.arsId,
                    routeId: station.routeId,
                });
            });

            const btnWrap = document.createElement("div");
            btnWrap.style.display = "flex";
            btnWrap.style.gap = "8px";

            // 서비스1 버튼
            const service1Btn = document.createElement("button");
            service1Btn.className = "btn btn-secondary";
            service1Btn.textContent = "서비스1";
            service1Btn.addEventListener("click", (event) => {
                event.stopPropagation();
                // 정류소 서비스1
                moveToPredictPageByStation({
                    routeId: station.routeId,
                    stationId: station.stationId,
                    arsId: station.arsId,
                });
            });

            // 서비스2 버튼
            const service2Btn = document.createElement("button");
            service2Btn.className = "btn btn-secondary";
            service2Btn.textContent = "서비스2";
            service2Btn.addEventListener("click", (event) => {
                event.stopPropagation();
                // 정류소 서비스2
                moveToPredictPageByStation2({
                    routeId: station.routeId,
                    stationId: station.stationId,
                    arsId: station.arsId,
                });
            });

            btnWrap.appendChild(service1Btn);
            btnWrap.appendChild(service2Btn);

            const actionWrap = document.createElement("div");
            actionWrap.className = "favorite-action-wrap";

            actionWrap.appendChild(service1Btn);
            actionWrap.appendChild(service2Btn);
            actionWrap.appendChild(deleteBtn);

            li.appendChild(itemWrap);
            li.appendChild(actionWrap);
            stationList.appendChild(li);
        }
    }
}

document.addEventListener("DOMContentLoaded", renderFavorites);
window.clearAllFavorites = clearAllFavorites;

function moveToPredictPageByBus2(routeId) {
    const url = new URL("/service2/", window.location.origin);
    url.searchParams.set("route_id", routeId);
    window.location.href = url.toString();
}

function moveToPredictPageByStation2({ stationId, routeId, arsId }) {
    const url = new URL("/service2/", window.location.origin);

    if (routeId) url.searchParams.set("route_id", routeId);
    if (stationId) url.searchParams.set("station_id", stationId);
    if (arsId) url.searchParams.set("ars_id", arsId);

    window.location.href = url.toString();
}