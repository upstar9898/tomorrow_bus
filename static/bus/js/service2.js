import { getCookie, getNowForDateTimeLocal, formatDateTime } from "./utils.js";
import { routeSelectChangeEvent } from "./routeSelect.js";

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

// 노선을 선택하면 정류장 목록을 불러오는 이벤트
routeSelectChangeEvent(routeSelect, stationSelect);

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
                route_id: routeId,
                station_id: stationId,
                date_time: dateTime,
            }),
        });

        const mapFetch = fetch(
            `/ajax/route-map-data/?route_id=${encodeURIComponent(routeId)}`,
        );

        const [predictResponse, mapResponse] = await Promise.all([
            predictFetch,
            mapFetch,
        ]);

        const predictResult = await predictResponse.json();
        const mapResult = await mapResponse.json();

        if (!predictResponse.ok || !predictResult.success) {
            alert(predictResult.error || "예측 요청에 실패했습니다.");
            return;
        }

        renderRouteResult(routeName, stationName, predictResult.data);

        if (!mapResponse.ok || !mapResult.success) {
            document.getElementById("mapSummary").textContent =
                "지도 데이터를 불러오지 못했습니다.";
            return;
        }

        const mapResultData = mapResult.data;
        const predictedStops = predictResult.data.stops || [];

        drawRouteMap(mapResultData.stations, stationId, predictedStops);
    } catch (error) {
        console.error(error);
        alert("서버 요청 중 오류가 발생했습니다.");
    }
});

function getBoundaryStaOrd(stops = []) {
    const staOrdList = stops
        .map((stop) => Number(stop.staOrd))
        .filter((v) => !Number.isNaN(v));

    if (staOrdList.length === 0) {
        return { minStaOrd: null, maxStaOrd: null };
    }

    return {
        minStaOrd: Math.min(...staOrdList),
        maxStaOrd: Math.max(...staOrdList),
    };
}

function isBoundaryStop(stop, minStaOrd, maxStaOrd) {
    const staOrd = Number(stop.staOrd);
    if (Number.isNaN(staOrd)) return false;
    return staOrd === minStaOrd || staOrd === maxStaOrd;
}

function getSeatState(stop, minStaOrd = null, maxStaOrd = null) {
    const isBoundary = isBoundaryStop(stop, minStaOrd, maxStaOrd);

    if (isBoundary) {
        return {
            text:
                Number(stop.staOrd) === Number(minStaOrd)
                    ? "첫 정류장"
                    : "마지막 정류장",
            dotClass: "status-gray",
            badgeClass: "state-gray",
        };
    }

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

    if (seat <= 10) {
        return {
            text: "혼잡",
            dotClass: "status-orange",
            badgeClass: "state-orange",
        };
    }

    if (seat <= 20) {
        return {
            text: "보통",
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

// result가 주어졌을 때, result를 바탕으로 예측 결과를 표시해주는 함수
function renderRouteResult(routeName, stationName, data) {
    const formattedDate = formatDateTime(rideDateTime.value);
    // const predictions = Array.isArray(data) ? data : [];

    resultSummary.textContent = `${routeName} · ${stationName} · ${formattedDate}`;

    if (selectedStopName) {
        selectedStopName.textContent = stationName;
    }

    if (selectedDateTime) {
        selectedDateTime.textContent = formattedDate;
    }

    const { minStaOrd, maxStaOrd } = getBoundaryStaOrd(data.stops);

    if (summaryTotalStops) {
        summaryTotalStops.textContent = predictions.length;
    }

    const selectedStop = data.stops.find((stop) => stop.is_selected);
    const selectedIsBoundary =
        selectedStop && isBoundaryStop(selectedStop, minStaOrd, maxStaOrd);

    if (selectedSeatPrediction) {
        if (
            !selectedStop ||
            selectedStop.is_virtual === 1 ||
            selectedIsBoundary
        ) {
            selectedSeatPrediction.textContent = "-";
        } else {
            selectedSeatPrediction.textContent = `${selectedStop.remaining_seat}석`;
        }
    }

    if (summaryBusyStops) {
        const busyCount = data.stops.filter((stop) => {
            const boundary = isBoundaryStop(stop, minStaOrd, maxStaOrd);
            return (
                !boundary && stop.is_virtual !== 1 && stop.remaining_seat <= 10
            );
        }).length;

        summaryBusyStops.textContent = busyCount;
    }

    const summaryVirtualStops = document.getElementById("summaryVirtualStops");
    if (summaryVirtualStops) {
        const virtualCount = data.stops.filter(
            (stop) => stop.is_virtual === 1,
        ).length;
        summaryVirtualStops.textContent = virtualCount;
    }

    routeList.classList.remove("route-list-empty");
    routeList.innerHTML = "";

    if (data.length === 0) {
        routeList.innerHTML = `
            <li class="text-center py-4 soft-note">표시할 예측 결과가 없습니다.</li>
        `;
        return;
    }

    for (const stop of data.stops) {
        const boundary = isBoundaryStop(stop, minStaOrd, maxStaOrd);
        const state = getSeatState(stop, minStaOrd, maxStaOrd);
        const isVirtual = stop.is_virtual === 1;
        const isSelected =
            String(stop.station_id) === String(stationSelect.value);
        const predictedTimeText = stop.predicted_arrival_time
            ? stop.predicted_arrival_time.slice(11, 16)
            : "";
        const relativeTimeText = stop.relative_time_label || "";

        const li = document.createElement("li");
        li.className = "stop-item";

        li.innerHTML = `
            <div class="stop-marker">
                <span class="stop-dot ${state.dotClass}"></span>
            </div>

            <div class="stop-card ${isSelected ? "selected" : ""}">
                <div class="stop-top">
                    <div>
                        <div class="stop-name">${stop.station_name || stop.station_id}</div>

                        <div class="stop-meta">
                            ${stop.ars_id ? `${stop.ars_id}` : ""}
                            ${predictedTimeText ? ` · 도착 예정 약 ${predictedTimeText}` : ""}
                            ${relativeTimeText ? ` (${relativeTimeText})` : ""}
                        </div>
                    </div>

                    <div class="stop-badges">
                        ${isSelected ? `<span class="selected-badge">기준 정류소</span>` : ""}

                        ${
                            !isVirtual && !boundary
                                ? `<span class="seat-badge">${stop.remaining_seat}석</span>`
                                : ``
                        }

                        <span class="state-badge ${state.badgeClass}">
                            ${state.text}
                        </span>
                    </div>
                </div>

                <div class="stop-bottom">
                    ${
                        boundary
                            ? `<span>기점/종점 정류장은 예측값을 표시하지 않습니다.</span>`
                            : isVirtual
                              ? `<span>예측 대상이 아닌 가상 정류소입니다.</span>`
                              : `<span>예상 만차확률 ${(stop.full_probability * 100).toFixed(1)}%</span>`
                    }
                </div>
            </div>
        `;

        routeList.appendChild(li);
    }

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
let mapInfoWindows = [];
let mapSeatOverlays = [];

const MARKER_VISIBLE_MAX_LEVEL = 5;
const DEFAULT_FOCUS_LEVEL = 4;
const DEFAULT_MAP_LEVEL = 7;

function isVirtualStop(stop) {
    const name = stop.station_name || "";
    return name.includes("가상") || name.includes("미정차");
}

function clearRouteMap() {
    for (const marker of mapMarkers) {
        marker.setMap(null);
    }
    mapMarkers = [];

    for (const polyline of mapPolylines) {
        polyline.setMap(null);
    }
    mapPolylines = [];

    for (const infoWindow of mapInfoWindows) {
        infoWindow.close();
    }
    mapInfoWindows = [];

    for (const overlay of mapSeatOverlays) {
        overlay.setMap(null);
    }
    mapSeatOverlays = [];
}

function makeMarkerImage(color = "#2563eb") {
    if (!window.kakao || !window.kakao.maps) return null;

    const svg = `
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="28" viewBox="0 0 20 28">
            <path d="M10 1C5.582 1 2 4.582 2 9c0 5.9 8 17 8 17s8-11.1 8-17c0-4.418-3.582-8-8-8z"
                fill="${color}" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="10" cy="9" r="2.8" fill="#ffffff"/>
        </svg>
    `;

    const encoded = encodeURIComponent(svg)
        .replace(/'/g, "%27")
        .replace(/"/g, "%22");

    return new window.kakao.maps.MarkerImage(
        `data:image/svg+xml;charset=UTF-8,${encoded}`,
        new kakao.maps.Size(20, 28),
        {
            offset: new kakao.maps.Point(10, 28),
        },
    );
}

function updateMarkerVisibilityByLevel() {
    if (!kakaoMap) return;

    const currentLevel = kakaoMap.getLevel();
    const shouldShowMarkers = currentLevel <= MARKER_VISIBLE_MAX_LEVEL;

    for (const marker of mapMarkers) {
        marker.setMap(shouldShowMarkers ? kakaoMap : null);
    }

    for (const overlay of mapSeatOverlays) {
        overlay.setMap(shouldShowMarkers ? kakaoMap : null);
    }
}

function drawRouteMap(stations, selectedStationId, predictedStops = []) {
    const predictedStopMap = new Map(
        predictedStops.map((stop) => [String(stop.station_id), stop]),
    );

    const { minStaOrd, maxStaOrd } = getBoundaryStaOrd(
        predictedStops.length > 0 ? predictedStops : stations,
    );

    const mapContainer = document.getElementById("routeMap");
    const mapSummary = document.getElementById("mapSummary");

    if (!mapContainer || !window.kakao || !window.kakao.maps) {
        if (mapSummary) {
            mapSummary.textContent = "카카오맵을 불러오지 못했습니다.";
        }
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

    const selectedStation = validStations.find(
        (st) => String(st.station_id) === String(selectedStationId),
    );

    const focusStation = selectedStation || validStations[0];

    const focusLatLng = new kakao.maps.LatLng(
        Number(focusStation.latitude),
        Number(focusStation.longitude),
    );

    if (!kakaoMap) {
        kakaoMap = new kakao.maps.Map(mapContainer, {
            center: focusLatLng,
            level: DEFAULT_MAP_LEVEL,
        });

        kakao.maps.event.addListener(kakaoMap, "zoom_changed", function () {
            updateMarkerVisibilityByLevel();
        });
    }

    clearRouteMap();

    let currentPath = [];
    let prevStaOrd = null;

    for (const st of validStations) {
        const latlng = new window.kakao.maps.LatLng(
            Number(st.latitude),
            Number(st.longitude),
        );

        const currentStaOrd = Number(st.staOrd);

        if (
            prevStaOrd !== null &&
            !Number.isNaN(currentStaOrd) &&
            !Number.isNaN(prevStaOrd) &&
            currentStaOrd - prevStaOrd > 1
        ) {
            if (currentPath.length >= 2) {
                const polyline = new window.kakao.maps.Polyline({
                    map: kakaoMap,
                    path: currentPath,
                    strokeWeight: 4,
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

        if (isVirtualStop(st)) {
            continue;
        }

        const boundary = isBoundaryStop(st, minStaOrd, maxStaOrd);
        const isSelected = String(st.station_id) === String(selectedStationId);

        const marker = new window.kakao.maps.Marker({
            map: kakaoMap,
            position: latlng,
            title: st.station_name || st.station_id,
            image: isSelected
                ? makeMarkerImage("#facc15")
                : makeMarkerImage("#2563eb"),
        });

        mapMarkers.push(marker);

        const predicted = predictedStopMap.get(String(st.station_id));
        const remainingSeat = predicted ? predicted.remaining_seat : null;
        const isVirtual = predicted ? predicted.is_virtual === 1 : false;

        // 첫/마지막 정류장 및 가상 정류장은 숫자 오버레이 표시 안 함
        if (!boundary && !isVirtual && remainingSeat != null) {
            const seatOverlay = createSeatOverlay(
                latlng,
                `${remainingSeat}석`,
                isSelected,
            );
            seatOverlay.setMap(kakaoMap);
            mapSeatOverlays.push(seatOverlay);
        }

        const boundaryText =
            Number(st.staOrd) === Number(minStaOrd)
                ? "첫 정류장"
                : Number(st.staOrd) === Number(maxStaOrd)
                  ? "마지막 정류장"
                  : "";

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
                    <strong>${st.station_name || st.station_id}</strong><br>
                    ${st.ars_id ? `정류소 코드: ${st.ars_id}<br>` : ""}
                    ${
                        boundary
                            ? `<span style="color:#475569;font-weight:700;">${boundaryText}</span>`
                            : st.is_virtual === 1
                              ? "가상 정류소"
                              : "일반 정류소"
                    }
                    ${
                        isSelected
                            ? `<br><span style="color:#ca8a04;font-weight:700;">기준 정류소</span>`
                            : ""
                    }
                </div>
            `,
        });

        mapInfoWindows.push(infoWindow);

        kakao.maps.event.addListener(marker, "click", function () {
            infoWindow.open(kakaoMap, marker);
        });
    }

    if (currentPath.length >= 2) {
        const polyline = new window.kakao.maps.Polyline({
            map: kakaoMap,
            path: currentPath,
            strokeWeight: 4,
            strokeColor: "#2563eb",
            strokeOpacity: 0.85,
            strokeStyle: "solid",
        });
        mapPolylines.push(polyline);
    }

    kakaoMap.setCenter(focusLatLng);
    kakaoMap.setLevel(DEFAULT_FOCUS_LEVEL);

    mapSummary.textContent = selectedStation
        ? `기준 정류소 중심으로 지도를 표시했습니다.`
        : `선택한 정류소를 찾지 못해 노선 시작 지점을 기준으로 표시했습니다.`;

    updateMarkerVisibilityByLevel();
}

// 카카오맵 초기화 -> 아래 변경 예정
if (window.kakao && window.kakao.maps) {
    window.kakao.maps.load(function () {
        const mapContainer = document.getElementById("routeMap");
        if (!mapContainer) return;

        kakaoMap = new kakao.maps.Map(mapContainer, {
            center: new kakao.maps.LatLng(37.5665, 126.978),
            level: DEFAULT_MAP_LEVEL,
        });

        kakao.maps.event.addListener(kakaoMap, "zoom_changed", function () {
            updateMarkerVisibilityByLevel();
        });
    });

    function createSeatOverlay(latlng, seatText, isSelected = false) {
        const content = document.createElement("div");
        content.style.position = "relative";
        content.style.transform = "translateY(-38px)";
        content.style.padding = "2px 6px";
        content.style.borderRadius = "999px";
        content.style.background = isSelected ? "#facc15" : "#ffffff";
        content.style.color = isSelected ? "#1f2937" : "#111827";
        content.style.border = isSelected
            ? "1px solid #eab308"
            : "1px solid #cbd5e1";
        content.style.fontSize = "11px";
        content.style.fontWeight = "700";
        content.style.lineHeight = "1.2";
        content.style.boxShadow = "0 1px 4px rgba(0,0,0,0.15)";
        content.style.whiteSpace = "nowrap";
        content.textContent = seatText;

        return new kakao.maps.CustomOverlay({
            position: latlng,
            content: content,
            yAnchor: 1,
            zIndex: isSelected ? 4 : 3,
        });
    }
}
