import {
    getCookie,
    getNowForDateTimeLocal,
    formatDateTime,
    showToast,
} from "./utils.js";
import { addFavorite } from "./favorite.js";

// 제출 버튼
let currentPrediction = {
    routeId: "",
    routeName: "",
    stationId: "",
    stationName: "",
};

let latestDayType = "";

const routeSelect = document.getElementById("routeSelect");
const stationSelect = document.getElementById("stationSelect");
const predictForm = document.getElementById("predictForm");
const rideDateTime = document.getElementById("rideDateTime");

const favoriteRouteBtn = document.getElementById("favoriteRouteBtn");
const favoriteStationBtn = document.getElementById("favoriteStationBtn");

rideDateTime.min = getNowForDateTimeLocal();
rideDateTime.value = getNowForDateTimeLocal();

function renderResult(routeName, stationName, result) {
    const data = result.data;

    const formattedDate = formatDateTime(data.date_time);

    document.getElementById("resultSummary").textContent =
        `${routeName} · ${stationName} · ${formattedDate}`;

    document.getElementById("seatPrediction").textContent = data.remaining_seat;

    document.getElementById("fullProb").textContent =
        `${(data.full_prob * 100).toFixed(1)}%`;

    currentPrediction.routeId = routeSelect.value;
    currentPrediction.routeName = routeName;
    currentPrediction.stationId = stationSelect.value;
    currentPrediction.stationName = stationName;

    favoriteRouteBtn.disabled = false;
    favoriteStationBtn.disabled = false;

    const historyTable = document.getElementById("historyTable");
    const row = document.createElement("tr");
    row.innerHTML = `
    <td>${routeName}</td>
    <td>${stationName}</td>
    <td>${formattedDate}</td>
    <td>${data.remaining_seat}석</td>
    <td>${(data.full_prob * 100).toFixed(1)}%</td>
`;
    historyTable.prepend(row);

    while (historyTable.children.length > 6) {
        historyTable.removeChild(historyTable.lastElementChild);
    }
}

function drawWeekChart(bars, dayType = "") {
    const chartEl = document.getElementById("seatChart");
    if (!chartEl) return;

    const data = new google.visualization.DataTable();
    data.addColumn("string", "요일");
    data.addColumn("number", "잔여좌석");

    bars.forEach((item) => {
        data.addRow([item.dayLabel, item.remaining_seat]);
    });

    const isMobile = window.innerWidth <= 768;

    const options = {
        title:
            dayType === "weekend"
                ? "주말 요일별 잔여좌석"
                : "평일 요일별 잔여좌석",
        legend: { position: "none" },
        width: "100%",
        height: isMobile ? 320 : 420,
        chartArea: {
            left: isMobile ? 50 : 70,
            top: 50,
            width: isMobile ? "75%" : "85%",
            height: isMobile ? "60%" : "72%",
        },
        hAxis: {
            title: "요일",
            textStyle: {
                fontSize: isMobile ? 11 : 13,
            },
        },
        vAxis: {
            title: "잔여좌석",
            minValue: 0,
        },
    };

    const chart = new google.visualization.ColumnChart(chartEl);
    chart.draw(data, options);
}

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
    const stationName = stationSelect.options[stationSelect.selectedIndex].text;

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
        const predictPromise = await fetch("/ajax/predict/service1/", {
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

        const chartPromise = fetch(
            `/ajax/station-week-chart/?routeId=${encodeURIComponent(routeId)}&stationId=${encodeURIComponent(stationId)}&date=${encodeURIComponent(date)}&time=${encodeURIComponent(time)}`,
        );

        const [predictResponse, chartResponse] = await Promise.all([
            predictPromise,
            chartPromise,
        ]);

        const predictResult = await predictResponse.json();
        const chartResult = await chartResponse.json();

        if (!predictResponse.ok || !predictResult.success) {
            alert(predictResult.error || "예측 요청에 실패했습니다.");
            return;
        }

        renderResult(routeName, stationName, predictResult);

        if (!chartResponse.ok || !chartResult.success) {
            document.getElementById("chartInfo").innerHTML =
                "차트 데이터를 불러오지 못했습니다.";
            return;
        }

        latestWeekBars = chartResult.bars;
        latestDayType = chartResult.dayType;

        document.getElementById("chartInfo").innerHTML = `
                <strong>${chartResult.routeName}</strong><br>
                정류소: ${chartResult.stationName}<br>
                기준 시간: ${chartResult.requestedTime}<br>
                구분: ${chartResult.dayType === "weekday" ? "평일(월~금)" : "주말(토~일)"}
            `;

        drawWeekChart(latestWeekBars, latestDayType);
    } catch (error) {
        console.error(error);
        alert("서버 요청 중 오류가 발생했습니다.");
    }
});

window.addEventListener("resize", function () {
    if (latestWeekBars.length > 0) {
        drawWeekChart(latestWeekBars, latestDayType);
    }
});

favoriteRouteBtn.addEventListener("click", function () {
    if (!currentPrediction.routeId) {
        showToast("먼저 예측을 실행하세요.");
        return;
    }
    addFavorite("bus", currentPrediction.routeId);
});

favoriteStationBtn.addEventListener("click", function () {
    if (!currentPrediction.stationName) {
        showToast("먼저 예측을 실행하세요.");
        return;
    }
    addFavorite("station", currentPrediction.stationName);
});

google.charts.load("current", { packages: ["corechart"] });

let latestWeekBars = [];


