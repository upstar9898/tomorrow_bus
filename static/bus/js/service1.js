import {
    getCookie,
    getNowForDateTimeLocal,
    formatDateTime,
    showToast,
} from "./utils.js";
import { addFavorite } from "./favorite.js";
import { routeSelectChangeEvent, loadStationsByRoute } from "./routeSelect.js";

// 구글 차트 라이브러리 로드
google.charts.load("current", { packages: ["corechart"] });

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

// 예측 버튼을 눌렀을 때, 예외처리를 포함하여 차트 및 예측 결과를 표시하는 이벤트
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
    const stationLabel =
        stationSelect.options[stationSelect.selectedIndex].text;

    // "정든마을.우성아파트 (47043)" -> stationName, arsId 분리
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
        const predictPromise = await fetch("/ajax/predict/service1/", {
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

// 윈도우 사이즈 변경이 있을 경우 차트를 다시 그리는 이벤트
window.addEventListener("resize", function () {
    if (latestWeekBars.length > 0) {
        drawWeekChart(latestWeekBars, latestDayType);
    }
});

// 즐겨찾기 노선 추가 이벤트
favoriteRouteBtn.addEventListener("click", function () {
    if (!currentPrediction.routeId) {
        showToast("먼저 예측을 실행하세요.");
        return;
    }
    addFavorite("bus", currentPrediction.routeId);
});

// 즐겨찾기 정류장 추가 이벤트
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

// result가 주어졌을 때, result를 바탕으로 예측 결과를 표시해주는 함수
function renderResult(routeName, stationName, arsId, result) {
    const data = result.data;

    const formattedDate = formatDateTime(data.date_time);

    document.getElementById("resultSummary").textContent =
        `${routeName} · ${stationName} · ${formattedDate}`;

    document.getElementById("seatPrediction").textContent = data.remaining_seat;

    if (data.full_prob < 0.001) {
        document.getElementById("fullProb").textContent = "0.1% 미만";
    } else if (data.full_prob >= 0.999) {
        document.getElementById("fullProb").textContent = "99.9% 이상";
    } else {
        document.getElementById("fullProb").textContent =
            `${(data.full_prob * 100).toFixed(1)}%`;
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

// 요일별 차트를 그리는 함수
function drawWeekChart(bars, dayType = "") {
    const chartEl = document.getElementById("seatChart");
    if (!chartEl) return;

    const data = new google.visualization.DataTable();
    data.addColumn("string", "요일");
    data.addColumn("number", "잔여좌석");

    bars.forEach((item) => {
        data.addRow([item.day_label, item.remaining_seat]);
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
