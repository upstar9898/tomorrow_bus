function getFavorites() {
    const data = localStorage.getItem("favorites");
    return data ? JSON.parse(data) : { buses: [], stations: [] };
}

function saveFavorites(favorites) {
    localStorage.setItem("favorites", JSON.stringify(favorites));
}

function removeFavorite(type, name) {
    const favorites = getFavorites();

    if (type === "bus") {
        favorites.buses = favorites.buses.filter(item => item !== name);
    }

    if (type === "station") {
        favorites.stations = favorites.stations.filter(item => item !== name);
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

        return data.routeName || routeId;
    } catch (error) {
        console.error("노선명 조회 실패:", error);
        return routeId;
    }
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

            const nameSpan = document.createElement("span");
            nameSpan.className = "favorite-item-name";
            nameSpan.textContent = routeName;

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "btn btn-delete";
            deleteBtn.textContent = "삭제";
            deleteBtn.addEventListener("click", () => {
                removeFavorite("bus", bus);
            });

            li.appendChild(nameSpan);
            li.appendChild(deleteBtn);
            busList.appendChild(li);
        }
    }

    if (favorites.stations.length === 0) {
        stationEmpty.style.display = "block";
    } else {
        stationEmpty.style.display = "none";

        favorites.stations.forEach(station => {
            const li = document.createElement("li");
            li.innerHTML = `
                <span class="favorite-item-name">${station}</span>
                <button class="btn btn-delete">삭제</button>
            `;

            li.querySelector("button").addEventListener("click", () => {
                removeFavorite("station", station);
            });

            stationList.appendChild(li);
        });
    }
}

document.addEventListener("DOMContentLoaded", renderFavorites);

window.clearAllFavorites = clearAllFavorites;