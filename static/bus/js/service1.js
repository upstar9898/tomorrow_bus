import {
    getCookie,
    getNowForDateTimeLocal,
    formatDateTime,
    showToast,
} from "./utils.js";
import { addFavorite } from "./favorite.js";
import { routeSelectChangeEvent, loadStationsByRoute } from "./routeSelect.js";

// input form에서 element 가져오기
const routeSelect = document.getElementById("routeSelect");
const stationSelect = document.getElementById("stationSelect");
const predictForm = document.getElementById("predictForm");
const rideDateTime = document.getElementById("rideDateTime");

// 즐겨찾기 관련 element 가져오기
const favoriteRouteBtn = document.getElementById("favoriteRouteBtn");
const favoriteStationBtn = document.getElementById("favoriteStationBtn");

// 승차일시 초기값 설정
rideDateTime.min = getNowForDateTimeLocal();
rideDateTime.value = getNowForDateTimeLocal();

// 제출 버튼
let currentPrediction = {
    routeId: "",
    routeName: "",
    stationId: "",
    stationName: "",
    arsId: "",
};

// 차트 출력을 위해 필요한 변수
let latestDayType = "";
let latestWeekBars = [];

// 노선을 선택하면 정류장 목록을 불러오는 이벤트
routeSelectChangeEvent(routeSelect, stationSelect);

// 쿼리스트링 기반 초기값 세팅
async function applyQueryStringToForm() {
    const params = new URLSearchParams(window.location.search);

    const routeId = params.get("route_id");
    const stationId = params.get("station_id");
    const arsId = params.get("ars_id");
    const dateTime = params.get("date_time");

    if (!routeId) {
        return;
    }

    routeSelect.value = String(routeId);

    await loadStationsByRoute(routeSelect, stationSelect, stationId, arsId);

    if (dateTime) {
        rideDateTime.value = dateTime;
    }
}

document.addEventListener("DOMContentLoaded", async function () {
    await applyQueryStringToForm();
});

// 예측 버튼을 눌렀을 때
predictForm.addEventListener("submit", async function (e) {
    e.preventDefault();

    const routeId = routeSelect.value;
    const stationId = stationSelect.value;
    const dateTime = rideDateTime.value;
    const nowValue = getNowForDateTimeLocal();

    if (!routeId || !stationId || !dateTime) {
        alert("노선, 정류장, 승차 일시를 모두 선택하세요.");
        return;
    }

    if (dateTime < nowValue) {
        alert("현재 시각 이후만 선택할 수 있습니다.");
        rideDateTime.focus();
        return;
    }

    const routeName = routeSelect.options[routeSelect.selectedIndex].text;
    const stationLabel = stationSelect.options[stationSelect.selectedIndex].text;

    const stationMatch = stationLabel.match(/^(.*?)(?:\s*\(([^)]+)\))?$/);
    const stationName = stationMatch ? stationMatch[1].trim() : stationLabel;
    const arsId = stationMatch && stationMatch[2] ? stationMatch[2].trim() : "";

    const [date, timeRaw] = dateTime.split("T");
    const time = timeRaw ? timeRaw.slice(0, 5) : "";

    if (!date || !time) {
        alert("날짜 또는 시간 값을 읽을 수 없습니다.");
        return;
    }

    document.getElementById("resultSummary").textContent = "예측 중...";
    document.getElementById("seatPrediction").textContent = "-";
    document.getElementById("fullProb").textContent = "-";
    document.getElementById("chartInfo").innerHTML = "차트 불러오는 중...";

    try {
        const predictPromise = fetch("/ajax/predict/service1/", {
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

        const chartPromise = fetch(
            `/ajax/station-week-chart/?route_id=${encodeURIComponent(routeId)}&station_id=${encodeURIComponent(stationId)}&date=${encodeURIComponent(date)}&time=${encodeURIComponent(time)}`,
        );

        const [predictResponse, chartResponse] = await Promise.all([
            predictPromise,
            chartPromise,
        ]);

        const predictResult = await predictResponse.json();
        const chartResult = await chartResponse.json();

        if (!predictResult.success && predictResult.reason === "OUT_OF_OPERATION_TIME") {
            document.getElementById("resultSummary").textContent =
                "운행 시간 외 예측 불가";

            document.getElementById("seatPrediction").textContent = "-";
            document.getElementById("fullProb").textContent = "-";

            document.getElementById("chartInfo").innerHTML = `
                <div class="text-danger fw-bold mb-1">
                    운행 시간 외 예측 불가
                </div>
                <div>
                    ${predictResult.message}
                </div>
            `;
            return;
        }

        if (!predictResponse.ok || !predictResult.success) {
            alert(predictResult.error || "예측 요청에 실패했습니다.");
            return;
        }

        renderResult(routeName, stationName, arsId, predictResult);

        if (!chartResponse.ok || !chartResult.success) {
            document.getElementById("chartInfo").innerHTML =
                "차트 데이터를 불러오지 못했습니다.";
            return;
        }

        const chartResultData = chartResult.data;
        latestWeekBars = chartResultData.bars;
        latestDayType = chartResultData.day_type;

        document.getElementById("chartInfo").innerHTML = `
            <strong>${chartResultData.route_name}</strong><br>
            정류소: ${chartResultData.station_name}<br>
            기준 시간: ${chartResultData.requested_time}<br>
            구분: ${chartResultData.day_type === "weekday" ? "평일(월~금)" : "주말(토~일)"}
        `;

        drawWeekChart(latestWeekBars, latestDayType);
    } catch (error) {
        console.error(error);
        alert("서버 요청 중 오류가 발생했습니다.");
    }
});

// 윈도우 사이즈 변경 시 차트 다시 그리기
window.addEventListener("resize", function () {
    if (latestWeekBars.length > 0) {
        drawWeekChart(latestWeekBars, latestDayType);
    }
});

// 즐겨찾기 노선 추가
favoriteRouteBtn.addEventListener("click", function () {
    if (!currentPrediction.routeId) {
        showToast("먼저 예측을 실행하세요.");
        return;
    }
    addFavorite("bus", currentPrediction.routeId);
});

// 즐겨찾기 정류장 추가
favoriteStationBtn.addEventListener("click", function () {
    if (
        !currentPrediction.stationName ||
        !currentPrediction.arsId ||
        !currentPrediction.routeId ||
        !currentPrediction.routeName
    ) {
        showToast("먼저 예측을 실행하세요.");
        return;
    }

    addFavorite("station", {
        stationName: currentPrediction.stationName,
        arsId: currentPrediction.arsId,
        routeId: currentPrediction.routeId,
        routeName: currentPrediction.routeName,
        stationId: currentPrediction.stationId,
    });
});

// 예측 결과 렌더링
function renderResult(routeName, stationName, arsId, result) {
    const data = result.data;

    const inputDateTime = data.input_datetime || data.date_time;
    const scheduledDateTime = data.scheduled_arrival_time || data.date_time;

    const formattedInputDate = formatDateTime(inputDateTime);
    const formattedScheduledDate = formatDateTime(scheduledDateTime);

    document.getElementById("resultSummary").innerHTML =
        `${routeName} · ${stationName}<br>
        입력 시각: ${formattedInputDate}<br>
        예측 기준 도착 예정 시각: ${formattedScheduledDate}`;

    document.getElementById("seatPrediction").textContent = data.remaining_seat;

    if (data.full_prob < 0.001) {
        document.getElementById("fullProb").textContent = "0.1% 미만";
    } else if (data.full_prob >= 0.999) {
        document.getElementById("fullProb").textContent = "99.9% 이상";
    } else {
        document.getElementById("fullProb").textContent =
            `${(data.full_prob * 100).toFixed(1)}%`;
    }

    const weatherBadge = document.getElementById("weatherUsedBadge");
    if (weatherBadge) {
        weatherBadge.classList.remove("d-none");
        weatherBadge.classList.remove("bg-success", "bg-secondary");
        if (data.weather_fetched) {
            weatherBadge.textContent = "날씨데이터 사용";
            weatherBadge.className = "badge bg-success";
        } else {
            weatherBadge.textContent = "날씨데이터 미사용";
            weatherBadge.className = "badge bg-secondary";
        }
    }

    currentPrediction.routeId = routeSelect.value;
    currentPrediction.routeName = routeName;
    currentPrediction.stationId = stationSelect.value;
    currentPrediction.stationName = stationName;
    currentPrediction.arsId = arsId;

    favoriteRouteBtn.disabled = false;
    favoriteStationBtn.disabled = false;

    const historyTable = document.getElementById("historyTable");
    const row = document.createElement("tr");
    row.innerHTML = `
        <td><span style="font-weight:800;color:#2563eb;">${routeName}</span></td>
        <td>${stationName}</td>
        <td>
            입력 시각: ${formattedInputDate}<br>
            기준 시각: ${formattedScheduledDate}
        </td>
        <td>${data.remaining_seat}석</td>
        <td>${(data.full_prob * 100).toFixed(1)}%</td>
    `;
    historyTable.prepend(row);

    while (historyTable.children.length > 6) {
        historyTable.removeChild(historyTable.lastElementChild);
    }
}

/*
 * 혼잡도 4단계 기준 (프로젝트 확정 기준)
 * 만차: 0석       → 빨강  #ef4444
 * 혼잡: 1~20석    → 핑크  #f472b6
 * 보통: 21~30석   → 노랑  #facc15
 * 여유: 31석 이상 → 초록  #34d399
 */
function getColor(seats) {
    if (seats === 0)    return "#ef4444";
    if (seats <= 20)    return "#f472b6";
    if (seats <= 30)    return "#facc15";
    return "#34d399";
}

function getState(seats) {
    if (seats === 0)    return "만차";
    if (seats <= 20)    return "혼잡";
    if (seats <= 30)    return "보통";
    return "여유";
}

function drawWeekChart(bars, dayType = "") {
    const chartEl = document.getElementById("seatChart");
    if (!chartEl) return;

    const seats = bars.map(b => b.remaining_seat);
    const labels = bars.map(b => b.day_label);

    const totalSeats = seats.some(s => s > 41) ? 45 : 41;

    const colors = seats.map(s => getColor(s));
    const alphaColors = colors.map(c => c + "cc");

    if (chartEl._chartInstance) {
        chartEl._chartInstance.destroy();
    }
    chartEl.innerHTML = "";

    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "요일별 잔여좌석 차트");
    chartEl.appendChild(canvas);

    const isMobile = window.innerWidth <= 768;

    const instance = new Chart(canvas, {
        type: "bar",
        data: {
            labels,
            datasets: [{
                data: seats,
                backgroundColor: alphaColors,
                borderColor: colors,
                borderWidth: 1.5,
                borderRadius: 10,
                borderSkipped: false,
                barThickness: 28,
                maxBarThickness: 36,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const s = ctx.raw;
                            return `${s}석 (${getState(s)})`;
                        }
                    }
                }
            },
            layout: {
                padding: { left: 8, right: 8, top: 8, bottom: 0 }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        color: "#94a3b8",
                        font: { size: isMobile ? 11 : 12 }
                    },
                    border: { display: false }
                },
                y: {
                    min: 0,
                    max: totalSeats,
                    grid: {
                        color: "rgba(226,232,244,0.8)",
                    },
                    ticks: {
                        color: "#94a3b8",
                        font: { size: 11 },
                        stepSize: 10,
                    },
                    border: { display: false }
                }
            }
        }
    });

    chartEl._chartInstance = instance;
}
