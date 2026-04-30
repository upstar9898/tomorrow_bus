/**
 * TB DateTimePicker v2
 * - 클래스 쿼리 기반 (id 충돌 없음)
 * - 달력(왼쪽) + 시/분/오전오후 스크롤(오른쪽) 가로 배치
 * - 선택값 → 원본 datetime-local input에 자동 세팅
 */

class TBDateTimePicker {
    constructor(originalInput) {
        this.inp = originalInput;
        this.selDate = null;
        this.selH = 8;      // 1~12
        this.selM = 0;
        this.selAP = "오전";
        this.viewY = null;
        this.viewM = null;
        this._init();
    }

    _init() {
        const now = new Date();

        // 기존 input 값 파싱
        if (this.inp.value) {
            const d = new Date(this.inp.value);
            if (!isNaN(d)) {
                this.selDate = new Date(d.getFullYear(), d.getMonth(), d.getDate());
                const h = d.getHours();
                this.selAP = h >= 12 ? "오후" : "오전";
                this.selH = h % 12 === 0 ? 12 : h % 12;
                this.selM = d.getMinutes();
            }
        }
        if (!this.selDate) {
            this.selDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            const h = now.getHours();
            this.selAP = h >= 12 ? "오후" : "오전";
            this.selH = h % 12 === 0 ? 12 : h % 12;
            this.selM = Math.round(now.getMinutes() / 10) * 10 % 60;
        }

        this.viewY = this.selDate.getFullYear();
        this.viewM = this.selDate.getMonth();

        this._buildDOM();
        this._renderCal();
        this._renderHours();
        this._renderMins();
        this._renderAmPm();
        this._updateTrigger();
        this._updatePreview();
        this._syncInput();
        this._bindEvents();
    }

    _buildDOM() {
        // wrapper
        const wrap = document.createElement("div");
        wrap.className = "tb-dtp";
        this.inp.parentNode.insertBefore(wrap, this.inp);
        wrap.appendChild(this.inp);
        this.inp.style.display = "none";
        this.wrap = wrap;

        // trigger button
        const trigger = document.createElement("button");
        trigger.type = "button";
        trigger.className = "tb-dtp-trigger";
        trigger.innerHTML = `
            <span class="tb-dtp-trigger-text">날짜와 시간을 선택하세요</span>
            <svg class="tb-dtp-trigger-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
                <rect x="3" y="4" width="14" height="13" rx="2"/>
                <path d="M3 8h14M7 2v4M13 2v4"/>
            </svg>
        `;
        wrap.appendChild(trigger);
        this.trigger = trigger;
        this.triggerText = trigger.querySelector(".tb-dtp-trigger-text");

        // panel
        const panel = document.createElement("div");
        panel.className = "tb-dtp-panel";
        panel.innerHTML = `
            <div class="tb-dtp-panel-top">
                <div class="tb-dtp-cal">
                    <div class="tb-dtp-cal-head">
                        <span class="tb-dtp-cal-title"></span>
                        <div class="tb-dtp-cal-navs">
                            <button type="button" class="tb-dtp-nav tb-dtp-prev">&#8249;</button>
                            <button type="button" class="tb-dtp-nav tb-dtp-next">&#8250;</button>
                        </div>
                    </div>
                    <table class="tb-dtp-table">
                        <thead><tr>
                            <th>일</th><th>월</th><th>화</th><th>수</th><th>목</th><th>금</th><th>토</th>
                        </tr></thead>
                        <tbody class="tb-dtp-cal-body"></tbody>
                    </table>
                </div>
                <div class="tb-dtp-time">
                    <div class="tb-dtp-tcol">
                        <div class="tb-dtp-tcol-hd">시</div>
                        <div class="tb-dtp-tscroll tb-dtp-hour-scroll"></div>
                    </div>
                    <div class="tb-dtp-tcol">
                        <div class="tb-dtp-tcol-hd">분</div>
                        <div class="tb-dtp-tscroll tb-dtp-min-scroll"></div>
                    </div>
                    <div class="tb-dtp-tcol">
                        <div class="tb-dtp-tcol-hd">오전/오후</div>
                        <div class="tb-dtp-tscroll tb-dtp-ap-scroll"></div>
                    </div>
                </div>
            </div>
            <div class="tb-dtp-footer">
                <button type="button" class="tb-dtp-btn-today">오늘</button>
                <span class="tb-dtp-preview"></span>
                <button type="button" class="tb-dtp-btn-ok">확인</button>
            </div>
        `;
        wrap.appendChild(panel);
        this.panel = panel;

        // refs
        this.calTitle = panel.querySelector(".tb-dtp-cal-title");
        this.calBody  = panel.querySelector(".tb-dtp-cal-body");
        this.hourScr  = panel.querySelector(".tb-dtp-hour-scroll");
        this.minScr   = panel.querySelector(".tb-dtp-min-scroll");
        this.apScr    = panel.querySelector(".tb-dtp-ap-scroll");
        this.preview  = panel.querySelector(".tb-dtp-preview");
    }

    _renderCal() {
        const y = this.viewY, m = this.viewM;
        this.calTitle.textContent = `${y}년 ${m + 1}월`;

        const today = new Date();
        const todayKey = `${today.getFullYear()}-${today.getMonth()}-${today.getDate()}`;
        const selKey   = `${this.selDate.getFullYear()}-${this.selDate.getMonth()}-${this.selDate.getDate()}`;

        const firstWD  = new Date(y, m, 1).getDay();
        const lastD    = new Date(y, m + 1, 0).getDate();
        const prevLast = new Date(y, m, 0).getDate();
        const total    = Math.ceil((firstWD + lastD) / 7) * 7;

        let html = "";
        let cur = 1, nxt = 1;

        for (let i = 0; i < total; i++) {
            if (i % 7 === 0) html += "<tr>";

            if (i < firstWD) {
                const d = prevLast - firstWD + i + 1;
                html += `<td><button type="button" class="tb-dtp-day is-other" data-y="${y}" data-m="${m-1}" data-d="${d}">${d}</button></td>`;
            } else if (cur > lastD) {
                html += `<td><button type="button" class="tb-dtp-day is-other" data-y="${y}" data-m="${m+1}" data-d="${nxt}">${nxt}</button></td>`;
                nxt++;
            } else {
                const key = `${y}-${m}-${cur}`;
                let cls = "tb-dtp-day";
                if (key === todayKey) cls += " is-today";
                if (key === selKey)   cls += " is-sel";
                html += `<td><button type="button" class="${cls}" data-y="${y}" data-m="${m}" data-d="${cur}">${cur}</button></td>`;
                cur++;
            }

            if (i % 7 === 6) html += "</tr>";
        }

        this.calBody.innerHTML = html;

        this.calBody.querySelectorAll(".tb-dtp-day").forEach(btn => {
            btn.addEventListener("click", () => {
                this.selDate = new Date(+btn.dataset.y, +btn.dataset.m, +btn.dataset.d);
                this.viewY = this.selDate.getFullYear();
                this.viewM = this.selDate.getMonth();
                this._renderCal();
                this._updatePreview();
                this._syncInput();
            });
        });
    }

    _renderHours() {
        this.hourScr.innerHTML = "";
        for (let h = 1; h <= 12; h++) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "tb-dtp-titem" + (h === this.selH ? " is-sel" : "");
            btn.textContent = String(h).padStart(2, "0");
            btn.addEventListener("click", () => {
                this.selH = h;
                this._renderHours();
                this._updatePreview();
                this._syncInput();
            });
            this.hourScr.appendChild(btn);
        }
        const sel = this.hourScr.querySelector(".is-sel");
        if (sel) setTimeout(() => sel.scrollIntoView({ block: "center" }), 30);
    }

    _renderMins() {
        this.minScr.innerHTML = "";
        for (let mi = 0; mi <= 59; mi++) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "tb-dtp-titem" + (mi === this.selM ? " is-sel" : "");
            btn.textContent = String(mi).padStart(2, "0");
            btn.addEventListener("click", () => {
                this.selM = mi;
                this._renderMins();
                this._updatePreview();
                this._syncInput();
            });
            this.minScr.appendChild(btn);
        }
        const sel = this.minScr.querySelector(".is-sel");
        if (sel) setTimeout(() => sel.scrollIntoView({ block: "center" }), 30);
    }

    _renderAmPm() {
        this.apScr.innerHTML = "";
        ["오전", "오후"].forEach(ap => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "tb-dtp-titem" + (ap === this.selAP ? " is-sel" : "");
            btn.textContent = ap;
            btn.addEventListener("click", () => {
                this.selAP = ap;
                this._renderAmPm();
                this._updatePreview();
                this._syncInput();
            });
            this.apScr.appendChild(btn);
        });
    }

    _updatePreview() {
        const d   = this.selDate;
        const dow = ["일","월","화","수","목","금","토"][d.getDay()];
        const h   = String(this.selH).padStart(2,"0");
        const mi  = String(this.selM).padStart(2,"0");
        this.preview.textContent =
            `${d.getFullYear()}.${d.getMonth()+1}.${d.getDate()} (${dow}) ${this.selAP} ${h}:${mi}`;
    }

    _updateTrigger() {
        const d   = this.selDate;
        const dow = ["일","월","화","수","목","금","토"][d.getDay()];
        const h   = String(this.selH).padStart(2,"0");
        const mi  = String(this.selM).padStart(2,"0");
        this.triggerText.textContent =
            `${d.getFullYear()}.${d.getMonth()+1}.${d.getDate()} (${dow}) ${this.selAP} ${h}:${mi}`;
    }

    _syncInput() {
        const d = this.selDate;
        let h24 = this.selH;
        if (this.selAP === "오전") { if (h24 === 12) h24 = 0; }
        else                        { if (h24 !== 12) h24 += 12; }
        const yyyy = d.getFullYear();
        const mm   = String(d.getMonth()+1).padStart(2,"0");
        const dd   = String(d.getDate()).padStart(2,"0");
        const hh   = String(h24).padStart(2,"0");
        const mi   = String(this.selM).padStart(2,"0");
        this.inp.value = `${yyyy}-${mm}-${dd}T${hh}:${mi}`;
        // change 이벤트 발생 (service1.js의 min 체크 등 연동)
        this.inp.dispatchEvent(new Event("change", { bubbles: true }));
    }

    _bindEvents() {
        // 트리거 클릭
        this.trigger.addEventListener("click", e => {
            e.stopPropagation();
            this._toggle();
        });

        // 이전/다음 달
        this.panel.querySelector(".tb-dtp-prev").addEventListener("click", () => {
            this.viewM--;
            if (this.viewM < 0) { this.viewM = 11; this.viewY--; }
            this._renderCal();
        });
        this.panel.querySelector(".tb-dtp-next").addEventListener("click", () => {
            this.viewM++;
            if (this.viewM > 11) { this.viewM = 0; this.viewY++; }
            this._renderCal();
        });

        // 오늘
        this.panel.querySelector(".tb-dtp-btn-today").addEventListener("click", () => {
            const now = new Date();
            this.selDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            this.viewY = this.selDate.getFullYear();
            this.viewM = this.selDate.getMonth();
            this._renderCal();
            this._updatePreview();
            this._syncInput();
        });

        // 확인
        this.panel.querySelector(".tb-dtp-btn-ok").addEventListener("click", () => {
            this._updateTrigger();
            this._close();
        });

        // 바깥 클릭
        document.addEventListener("click", e => {
            if (!this.wrap.contains(e.target)) this._close();
        });

        // 패널 내부 클릭은 닫기 방지
        this.panel.addEventListener("click", e => e.stopPropagation());
    }

    _toggle() {
        const isOpen = this.panel.classList.contains("is-open");
        // 다른 피커 닫기
        document.querySelectorAll(".tb-dtp-panel.is-open").forEach(p => p.classList.remove("is-open"));
        document.querySelectorAll(".tb-dtp-trigger.is-open").forEach(t => t.classList.remove("is-open"));

        if (!isOpen) {
            this.panel.classList.add("is-open");
            this.trigger.classList.add("is-open");
            // 스크롤 복원
            setTimeout(() => {
                const sh = this.hourScr.querySelector(".is-sel");
                const sm = this.minScr.querySelector(".is-sel");
                if (sh) sh.scrollIntoView({ block: "center" });
                if (sm) sm.scrollIntoView({ block: "center" });
            }, 60);
        }
    }

    _close() {
        this.panel.classList.remove("is-open");
        this.trigger.classList.remove("is-open");
    }
}

// DOMContentLoaded 자동 초기화
document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("input[data-tb-dtp]").forEach(inp => {
        new TBDateTimePicker(inp);
    });
});
