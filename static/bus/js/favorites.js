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
        const response = await fetch(`/favorite/route-name/?routeId=${encodeURIComponent(routeId)}`);
        const data = await response.json();
        if (!response.ok || !data.success) return routeId;
        return data.data.route_name || routeId;
    } catch (error) {
        return routeId;
    }
}

function goToService1ByBus(routeId) {
    const url = new URL("/service1/", window.location.origin);
    url.searchParams.set("route_id", routeId);
    window.location.href = url.toString();
}

function goToService2ByBus(routeId) {
    const url = new URL("/service2/", window.location.origin);
    url.searchParams.set("route_id", routeId);
    window.location.href = url.toString();
}

function goToService1ByStation({ stationId, routeId, arsId }) {
    const url = new URL("/service1/", window.location.origin);
    if (routeId) url.searchParams.set("route_id", routeId);
    if (stationId) url.searchParams.set("station_id", stationId);
    if (arsId) url.searchParams.set("ars_id", arsId);
    window.location.href = url.toString();
}

function goToService2ByStation({ stationId, routeId, arsId }) {
    const url = new URL("/service2/", window.location.origin);
    if (routeId) url.searchParams.set("route_id", routeId);
    if (stationId) url.searchParams.set("station_id", stationId);
    if (arsId) url.searchParams.set("ars_id", arsId);
    window.location.href = url.toString();
}

function createBtn(label, type, onClick) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `fav-action-btn fav-action-btn--${type}`;
    btn.textContent = label;
    btn.addEventListener("click", (e) => { e.stopPropagation(); onClick(); });
    return btn;
}

async function renderFavorites() {
    const favorites = getFavorites();
    const busList = document.getElementById("busList");
    const stationList = document.getElementById("stationList");
    const busEmpty = document.getElementById("busEmpty");
    const stationEmpty = document.getElementById("stationEmpty");

    busList.innerHTML = "";
    stationList.innerHTML = "";

    // ── 버스 즐겨찾기 ──
    if (favorites.buses.length === 0) {
        busEmpty.style.display = "block";
    } else {
        busEmpty.style.display = "none";
        for (const bus of favorites.buses) {
            const routeName = await fetchRouteName(bus);
            const li = document.createElement("li");

            const iconWrap = document.createElement("div");
            iconWrap.className = "fav-item-icon";
            iconWrap.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="13" rx="2"/><path d="M2 11h20"/><path d="M7 19h10"/><path d="M5 16v3"/><path d="M19 16v3"/><circle cx="7.5" cy="18.5" r="1.5"/><circle cx="16.5" cy="18.5" r="1.5"/></svg>`;

            const itemBody = document.createElement("div");
            itemBody.className = "fav-item-body";

            const nameSpan = document.createElement("span");
            nameSpan.className = "fav-item-name";
            nameSpan.textContent = routeName;
            itemBody.appendChild(nameSpan);

            const actionWrap = document.createElement("div");
            actionWrap.className = "fav-action-wrap";

            actionWrap.appendChild(createBtn("정류장 단위 예측", "service", () => goToService1ByBus(bus)));
            actionWrap.appendChild(createBtn("노선 전체 예측", "service", () => goToService2ByBus(bus)));
            actionWrap.appendChild(createBtn("삭제", "delete", () => removeFavorite("bus", bus)));

            li.appendChild(iconWrap);
            li.appendChild(itemBody);
            li.appendChild(actionWrap);
            busList.appendChild(li);
        }
    }

    // ── 정류장 즐겨찾기 ──
    if (favorites.stations.length === 0) {
        stationEmpty.style.display = "block";
    } else {
        stationEmpty.style.display = "none";
        for (const station of favorites.stations) {
            const li = document.createElement("li");

            const iconWrap = document.createElement("div");
            iconWrap.className = "fav-item-icon";
            iconWrap.style.background = "#f0f4ff";
            iconWrap.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4f46e5" stroke-width="2.5" stroke-linecap="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>`;

            const itemBody = document.createElement("div");
            itemBody.className = "fav-item-body";

            const nameSpan = document.createElement("span");
            nameSpan.className = "fav-item-name";
            nameSpan.textContent = station.stationName || "정류소명 없음";

            const subSpan = document.createElement("span");
            subSpan.className = "fav-item-sub";
            subSpan.textContent = station.routeName || "";

            itemBody.appendChild(nameSpan);
            itemBody.appendChild(subSpan);

            const actionWrap = document.createElement("div");
            actionWrap.className = "fav-action-wrap";

            actionWrap.appendChild(createBtn("정류장 단위 예측", "service", () => goToService1ByStation({
                routeId: station.routeId,
                stationId: station.stationId,
                arsId: station.arsId,
            })));
            actionWrap.appendChild(createBtn("노선 전체 예측", "service", () => goToService2ByStation({
                routeId: station.routeId,
                stationId: station.stationId,
                arsId: station.arsId,
            })));
            actionWrap.appendChild(createBtn("삭제", "delete", () => removeFavorite("station", {
                arsId: station.arsId,
                routeId: station.routeId,
            })));

            li.appendChild(iconWrap);
            li.appendChild(itemBody);
            li.appendChild(actionWrap);
            stationList.appendChild(li);
        }
    }
}

document.addEventListener("DOMContentLoaded", renderFavorites);
window.clearAllFavorites = clearAllFavorites;
