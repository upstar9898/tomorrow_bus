    function getFavorites() {
        const data = localStorage.getItem('favorites');
        return data ? JSON.parse(data) : { buses: [], stations: [] };
    }

    function saveFavorites(favorites) {
        localStorage.setItem('favorites', JSON.stringify(favorites));
    }

    function removeFavorite(type, name) {
        const favorites = getFavorites();

        if (type === 'bus') {
            favorites.buses = favorites.buses.filter(item => item !== name);
        }

        if (type === 'station') {
            favorites.stations = favorites.stations.filter(item => item !== name);
        }

        saveFavorites(favorites);
        renderFavorites();
    }

    function clearAllFavorites() {
        localStorage.removeItem('favorites');
        renderFavorites();
    }

    function renderFavorites() {
        const favorites = getFavorites();

        const busList = document.getElementById('busList');
        const stationList = document.getElementById('stationList');
        const busEmpty = document.getElementById('busEmpty');
        const stationEmpty = document.getElementById('stationEmpty');

        busList.innerHTML = '';
        stationList.innerHTML = '';

        if (favorites.buses.length === 0) {
            busEmpty.style.display = 'block';
        } else {
            busEmpty.style.display = 'none';
            favorites.buses.forEach(bus => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span class="favorite-item-name">${bus}</span>
                    <button class="btn btn-delete" onclick="removeFavorite('bus', '${bus}')">삭제</button>
                `;
                busList.appendChild(li);
            });
        }

        if (favorites.stations.length === 0) {
            stationEmpty.style.display = 'block';
        } else {
            stationEmpty.style.display = 'none';
            favorites.stations.forEach(station => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span class="favorite-item-name">${station}</span>
                    <button class="btn btn-delete" onclick="removeFavorite('station', '${station}')">삭제</button>
                `;
                stationList.appendChild(li);
            });
        }
    }

    document.addEventListener('DOMContentLoaded', renderFavorites);