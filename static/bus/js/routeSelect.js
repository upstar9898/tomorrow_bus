async function loadStationsByRoute(routeSelect, stationSelect, selectedStationId = "", selectedArsId = "") {
    const routeId = routeSelect.value;

    stationSelect.innerHTML = `<option value="">정류장을 불러오는 중...</option>`;
    stationSelect.disabled = true;

    if (!routeId) {
        stationSelect.innerHTML = `<option value="">먼저 노선을 선택하세요</option>`;
        return;
    }

    try {
        const response = await fetch(
            `/ajax/stations/?route_id=${encodeURIComponent(routeId)}`
        );
        const result = await response.json();

        if (!response.ok || !result.success) {
            stationSelect.innerHTML = `<option value="">정류장 조회 실패</option>`;
            return;
        }

        const resultData = result.data;
        const stations = resultData.stations;

        if (!stations || stations.length === 0) {
            stationSelect.innerHTML = `<option value="">정류장이 없습니다</option>`;
            return;
        }

        const staOrdList = stations
            .map((station) => Number(station.staOrd))
            .filter((staOrd) => !Number.isNaN(staOrd));

        const minStaOrd = staOrdList.length ? Math.min(...staOrdList) : null;
        const maxStaOrd = staOrdList.length ? Math.max(...staOrdList) : null;

        let options = `<option value="">정류장을 선택하세요</option>`;

        for (const station of stations) {
            const staOrd = Number(station.staOrd);

            // 첫 정류소, 마지막 정류소는 select option 에서 제외
            if (
                !Number.isNaN(staOrd) &&
                (staOrd === minStaOrd || staOrd === maxStaOrd)
            ) {
                continue;
            }

            const label = station.ars_id
                ? `${station.station_name} (${station.ars_id})`
                : station.station_name;

            options += `
                <option 
                    value="${station.station_id}" 
                    data-ars-id="${station.ars_id || ""}"
                >
                    ${label}
                </option>
            `;
        }

        stationSelect.innerHTML = options;
        stationSelect.disabled = false;

        // 1순위: station_id
        if (selectedStationId) {
            stationSelect.value = String(selectedStationId);
        }

        // 2순위: ars_id
        if (!stationSelect.value && selectedArsId) {
            const matchedOption = Array.from(stationSelect.options).find(
                (option) => String(option.dataset.arsId) === String(selectedArsId)
            );

            if (matchedOption) {
                stationSelect.value = matchedOption.value;
            }
        }
    } catch (error) {
        console.error("정류장 조회 실패:", error);
        stationSelect.innerHTML = `<option value="">정류장 조회 실패</option>`;
        stationSelect.disabled = true;
    }
}

function routeSelectChangeEvent(routeSelect, stationSelect) {
    routeSelect.addEventListener("change", async function () {
        await loadStationsByRoute(routeSelect, stationSelect);
    });
}

export { routeSelectChangeEvent, loadStationsByRoute };