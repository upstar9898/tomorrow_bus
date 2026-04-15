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
        const response = await fetch("/ajax/predict/service2/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken"),
            },
            body: JSON.stringify({
                routeId: routeId,
                stationId: stationId,
                date_time: dateTime,
            }),
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            alert(result.error || "예측 요청에 실패했습니다.");
            return;
        }

        renderRouteResult(routeName, stationName, result.data);
    } catch (error) {
        alert("서버 요청 중 오류가 발생했습니다.");
    }
});

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
