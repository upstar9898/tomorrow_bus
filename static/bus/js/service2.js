import { getCookie, getNowForDateTimeLocal, formatDateTime } from "./utils.js";

const routeSelect = document.getElementById("routeSelect");
const stationSelect = document.getElementById("stationSelect");
const predictForm = document.getElementById("predictForm");
const rideDateTime = document.getElementById("rideDateTime");

const resultSummary = document.getElementById("resultSummary");
const routeList = document.getElementById("routeList");

const selectedStopName = document.getElementById("selectedStopName");
const selectedDateTime = document.getElementById("selectedDateTime");
const selectedSeatPrediction = document.getElementById(
    "selectedSeatPrediction",
);
const summaryTotalStops = document.getElementById("summaryTotalStops");
const summaryBusyStops = document.getElementById("summaryBusyStops");

rideDateTime.min = getNowForDateTimeLocal();
rideDateTime.value = getNowForDateTimeLocal();

routeSelect.addEventListener("change", async function () {
    const routeId = this.value;

    stationSelect.innerHTML = `<option value="">정류장을 불러오는 중...</option>`;
    stationSelect.disabled = true;

    if (!routeId) {
        stationSelect.innerHTML = `<option value="">먼저 노선을 선택하세요</option>`;
        return;
    }

    try {
        const response = await fetch(
            `/ajax/stations/?routeId=${encodeURIComponent(routeId)}`,
        );
        const result = await response.json();

        if (!response.ok || !result.success) {
            stationSelect.innerHTML = `<option value="">정류장 조회 실패</option>`;
            return;
        }

        const stations = result.stations;

        if (stations.length === 0) {
            stationSelect.innerHTML = `<option value="">정류장이 없습니다</option>`;
            return;
        }

        let options = `<option value="">정류장을 선택하세요</option>`;
        for (const station of stations) {
            const label = station.arsId
                ? `${station.stationName} (${station.arsId})`
                : station.stationName;

            options += `<option value="${station.stationId}">${label}</option>`;
        }

        stationSelect.innerHTML = options;
        stationSelect.disabled = false;
    } catch (error) {
        stationSelect.innerHTML = `<option value="">정류장 조회 실패</option>`;
    }
});

predictForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    const routeId = routeSelect.value;
    const stationId = stationSelect.value;
    const dateTime = rideDateTime.value;

    if (!routeId || !stationId || !dateTime) {
        alert("노선, 정류장, 승차 일시를 모두 선택하세요.");
        return;
    }

    const routeName = routeSelect.options[routeSelect.selectedIndex].text;
    const stationName = stationSelect.options[stationSelect.selectedIndex].text;

    resultSummary.textContent = "예측 중...";
    routeList.innerHTML = `
            <li class="text-center py-4 soft-note">전체 노선 예측 결과를 불러오는 중...</li>
        `;

        try {
    const predictFetch = fetch("/ajax/predict/service2/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify({
            routeId: routeId,
            stationId: stationId,
            date_time: dateTime
        })
    });

    const mapFetch = fetch(
        `/ajax/route-map-data/?routeId=${encodeURIComponent(routeId)}`
    );

    const [predictResponse, mapResponse] = await Promise.all([predictFetch, mapFetch]);

    const predictResult = await predictResponse.json();
    const mapResult = await mapResponse.json();

    if (!predictResponse.ok || !predictResult.success) {
        alert(predictResult.error || "예측 요청에 실패했습니다.");
        return;
    }

    renderRouteResult(routeName, stationName, predictResult.data);

    if (!mapResponse.ok || !mapResult.success) {
        document.getElementById("mapSummary").textContent = "지도 데이터를 불러오지 못했습니다.";
        return;
    }

    drawRouteMap(mapResult.stations, stationId);

} catch (error) {
    console.error(error);
    alert("서버 요청 중 오류가 발생했습니다.");
}});

function getSeatState(stop) {
    if (stop.is_virtual === 1) {
        return {
            text: "가상 정류소",
            dotClass: "status-gray",
            badgeClass: "state-gray",
        };
    }

    const seat = stop.remaining_seat;

    if (seat <= 2) {
        return {
            text: "만차임박",
            dotClass: "status-red",
            badgeClass: "state-red",
        };
    }
    if (seat <= 12) {
        return {
            text: "혼잡",
            dotClass: "status-yellow",
            badgeClass: "state-yellow",
        };
    }
    return {
        text: "여유",
        dotClass: "status-green",
        badgeClass: "state-green",
    };
}

function renderRouteResult(routeName, stationName, data) {
    // 상단 요약
    const formattedDate = formatDateTime(data.date_time);
    resultSummary.textContent = `${routeName} · ${stationName} · ${formattedDate}`;

    if (selectedStopName) {
        selectedStopName.textContent =
            data.selected_station_name || stationName;
    }

    if (selectedDateTime) {
        selectedDateTime.textContent = formattedDate;
    }

    if (summaryTotalStops) {
        summaryTotalStops.textContent = data.stops.length;
    }

    // 기준 정류소 찾기
    const selectedStop = data.stops.find((stop) => stop.is_selected);

    if (selectedSeatPrediction) {
        if (!selectedStop || selectedStop.is_virtual === 1) {
            selectedSeatPrediction.textContent = "-";
        } else {
            selectedSeatPrediction.textContent = `${selectedStop.remaining_seat}석`;
        }
    }

    // 혼잡 정류소 수
    if (summaryBusyStops) {
        const busyCount = data.stops.filter(
            (stop) => stop.is_virtual !== 1 && stop.remaining_seat <= 12,
        ).length;
        summaryBusyStops.textContent = busyCount;
    }

    // 가상 정류소 수
    const summaryVirtualStops = document.getElementById("summaryVirtualStops");
    if (summaryVirtualStops) {
        const virtualCount = data.stops.filter(
            (stop) => stop.is_virtual === 1,
        ).length;
        summaryVirtualStops.textContent = virtualCount;
    }

    // 리스트 초기화
    routeList.classList.remove("route-list-empty");
    routeList.innerHTML = "";

    // 상태 판단 함수
    function getSeatState(stop) {
        if (stop.is_virtual === 1) {
            return {
                text: "가상 정류소",
                dotClass: "status-gray",
                badgeClass: "state-gray",
            };
        }

        const seat = stop.remaining_seat;

        if (seat <= 2) {
            return {
                text: "만차임박",
                dotClass: "status-red",
                badgeClass: "state-red",
            };
        }
        if (seat <= 12) {
            return {
                text: "혼잡",
                dotClass: "status-yellow",
                badgeClass: "state-yellow",
            };
        }
        return {
            text: "여유",
            dotClass: "status-green",
            badgeClass: "state-green",
        };
    }

    // 리스트 렌더링
    for (const stop of data.stops) {
        const state = getSeatState(stop);
        const isVirtual = stop.is_virtual === 1;

        const li = document.createElement("li");
        console.log(stop)
        li.className = "stop-item";

        li.innerHTML = `
            <div class="stop-marker">
                <span class="stop-dot ${state.dotClass}"></span>
            </div>

            <div class="stop-card 
                ${stop.is_selected ? "selected" : ""} 
                ${isVirtual ? "virtual" : ""}">
                
                <div class="stop-top">
                    <div>
                        <div class="stop-name">${stop.station_name}</div>
                        <div class="stop-meta">
                            ${stop.ars_id ? `${stop.ars_id}` : ""}
                            ${stop.predicted_time ? ` · 예측 도착 ${stop.predicted_time}` : ""}
                        </div>
                    </div>

                    <div class="stop-badges">
                        ${stop.is_selected ? `<span class="selected-badge">기준 정류소</span>` : ""}

                        ${
                            isVirtual
                                ? ``
                                : `<span class="seat-badge">${stop.remaining_seat}석</span>`
                        }

                        <span class="state-badge ${state.badgeClass}">
                            ${state.text}
                        </span>
                    </div>
                </div>

                <div class="stop-bottom">
                    ${
                        isVirtual
                            ? `<span>예측 대상이 아닌 가상 정류소입니다.</span>`
                            : `<span>예상 만차확률 ${(stop.full_prob * 100).toFixed(1)}%</span>`
                    }
                </div>
            </div>
        `;

        routeList.appendChild(li);
    }

    // 기준 정류소로 스크롤
    const selectedCard = routeList.querySelector(".stop-card.selected");
    if (selectedCard) {
        selectedCard.scrollIntoView({
            behavior: "smooth",
            block: "center",
        });
    }
}

    let kakaoMap = null;
    let mapMarkers = [];
    let mapPolylines = [];

function clearRouteMap() {
    for (const marker of mapMarkers) {
        marker.setMap(null);
    }
    mapMarkers = [];

    for (const polyline of mapPolylines) {
        polyline.setMap(null);
    }
    mapPolylines = [];
}

function makeMarkerImage(color = "#2563eb") {
    const svg = `
        <svg xmlns="http://www.w3.org/2000/svg" width="36" height="48" viewBox="0 0 36 48">
            <path d="M18 2C10.268 2 4 8.268 4 16c0 10.2 14 28 14 28s14-17.8 14-28C32 8.268 25.732 2 18 2z"
                fill="${color}" stroke="#ffffff" stroke-width="2"/>
            <circle cx="18" cy="16" r="5" fill="#ffffff"/>
        </svg>
    `;

    const encoded = encodeURIComponent(svg)
        .replace(/'/g, "%27")
        .replace(/"/g, "%22");

    return new kakao.maps.MarkerImage(
        `data:image/svg+xml;charset=UTF-8,${encoded}`,
        new kakao.maps.Size(36, 48),
        {
            offset: new kakao.maps.Point(18, 48),
        }
    );
}

function drawRouteMap(stations, selectedStationId) {
    const mapContainer = document.getElementById("routeMap");
    const mapSummary = document.getElementById("mapSummary");

    if (!mapContainer || !window.kakao || !window.kakao.maps) {
        return;
    }

    const validStations = stations.filter(
        (st) =>
            st.latitude != null &&
            st.longitude != null &&
            !Number.isNaN(Number(st.latitude)) &&
            !Number.isNaN(Number(st.longitude)) &&
            Number(st.latitude) !== 0 &&
            Number(st.longitude) !== 0,
    );

    if (validStations.length === 0) {
        mapSummary.textContent = "표시할 정류소 좌표가 없습니다.";
        clearRouteMap();
        return;
    }

    mapSummary.textContent = `정류소 ${validStations.length}개를 지도에 표시했습니다.`;

    const first = validStations[0];
    const center = new kakao.maps.LatLng(
        Number(first.latitude),
        Number(first.longitude),
    );

    if (!kakaoMap) {
        kakaoMap = new kakao.maps.Map(mapContainer, {
            center: center,
            level: 7,
        });
    }

    clearRouteMap();

    const bounds = new kakao.maps.LatLngBounds();

    // 선을 여러 구간으로 나눠서 그리기
    let currentPath = [];
    let prevStaOrd = null;

    for (const st of validStations) {
        const latlng = new kakao.maps.LatLng(
            Number(st.latitude),
            Number(st.longitude),
        );

        bounds.extend(latlng);

        const isSelected =
            String(st.station_id) === String(selectedStationId);

        const marker = new kakao.maps.Marker({
            map: kakaoMap,
            position: latlng,
            title: st.station_name,
            image: isSelected
                ? makeMarkerImage("#facc15")
                : makeMarkerImage("#2563eb"),
        });

        mapMarkers.push(marker);

        const infoWindow = new kakao.maps.InfoWindow({
            removable: true,
            content: `
                <div style="
                    padding:10px 12px;
                    font-size:13px;
                    line-height:1.5;
                    border-radius:12px;
                    background:#fff;
                    border:1px solid #dbe4f0;
                    min-width:170px;
                ">
                    <strong>${st.station_name}</strong><br>
                    ${st.ars_id ? `정류소 코드: ${st.ars_id}<br>` : ""}
                    ${st.is_virtual === 1 ? "가상 정류소" : "일반 정류소"}
                    ${
                        isSelected
                            ? `<br><span style="color:#ca8a04;font-weight:700;">기준 정류소</span>`
                            : ""
                    }
                </div>
            `,
        });

        kakao.maps.event.addListener(marker, "click", function () {
            infoWindow.open(kakaoMap, marker);
        });

        // ===== 여기 핵심: staOrd가 끊기면 선도 끊기 =====
        const currentStaOrd = Number(st.staOrd);

        if (
            prevStaOrd !== null &&
            !Number.isNaN(currentStaOrd) &&
            !Number.isNaN(prevStaOrd) &&
            currentStaOrd - prevStaOrd > 1
        ) {
            if (currentPath.length >= 2) {
                const polyline = new kakao.maps.Polyline({
                    map: kakaoMap,
                    path: currentPath,
                    strokeWeight: 5,
                    strokeColor: "#2563eb",
                    strokeOpacity: 0.85,
                    strokeStyle: "solid",
                });
                mapPolylines.push(polyline);
            }

            currentPath = [];
        }

        currentPath.push(latlng);
        prevStaOrd = currentStaOrd;
    }

    // 마지막 구간 polyline 추가
    if (currentPath.length >= 2) {
        const polyline = new kakao.maps.Polyline({
            map: kakaoMap,
            path: currentPath,
            strokeWeight: 5,
            strokeColor: "#2563eb",
            strokeOpacity: 0.85,
            strokeStyle: "solid",
        });
        mapPolylines.push(polyline);
    }

    kakaoMap.setBounds(bounds, 80, 80, 80, 80);
}

kakao.maps.load(function () {
    const mapContainer = document.getElementById("routeMap");
    if (!mapContainer) return;

    kakaoMap = new kakao.maps.Map(mapContainer, {
        center: new kakao.maps.LatLng(37.5665, 126.9780),
        level: 8
    });
});
