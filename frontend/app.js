// ============================================================
// CONFIGURATION
// ============================================================

function getWebSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    let host = window.location.hostname;
    if (!host || host === "") {
        host = "127.0.0.1";
    }
    return `${protocol}//${host}:8000/safety/ws/camera`;
}

function getApiBase() {
    const proto = window.location.protocol;
    if (proto === "http:" || proto === "https:") {
        return window.location.origin;
    }
    return "http://127.0.0.1:8000";
}

const API = getApiBase();

function apiUrl(path) {
    return `${API}${path}`;
}

function evidenceUrl(relPath) {
    if (!relPath) return null;
    return apiUrl(`/evidence/${relPath}`);
}


// ============================================================
// DOM ELEMENTS
// ============================================================

const video = document.getElementById("camera");
const canvas = document.getElementById("resultCanvas");
const ctx = canvas.getContext("2d");

const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");

const connectionStatus = document.getElementById("connectionStatus");
const cameraMessage = document.getElementById("cameraMessage");
const riskBadge = document.getElementById("riskBadge");

const ppeStatus = document.getElementById("ppeStatus");
const fireStatus = document.getElementById("fireStatus");
const smokeStatus = document.getElementById("smokeStatus");
const personStatus = document.getElementById("personStatus");
const detectionList = document.getElementById("detectionList");

const alertBadge = document.getElementById("alertBadge");
const alertBadgeCount = document.getElementById("alertBadgeCount");
const navAlertBadge = document.getElementById("navAlertBadge");

const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const pageTitle = document.getElementById("pageTitle");
const pageSubtitle = document.getElementById("pageSubtitle");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebarOverlay");
const mobileMenuBtn = document.getElementById("mobileMenuBtn");


// ============================================================
// VARIABLES
// ============================================================

let cameraStream = null;
let socket = null;
let sendingFrames = false;
let awaitingResponse = false;
let activeTab = "dashboard";
let pollTimer = null;

const PAGE_TITLES = {
    dashboard: ["Dashboard", "Real-time safety monitoring and detection"],
    alerts: ["Alerts", "Safety alert feed and notifications"],
    history: ["History", "Complete log of safety events"],
    statistics: ["Statistics", "Aggregated safety metrics and trends"],
    evidence: ["Evidence", "Captured snapshots and video clips"],
    reports: ["Reports", "Export and generate safety reports"],
};


// ============================================================
// SHARED HELPERS
// ============================================================

const RISK_COLORS = {
    SAFE: "#16a34a",
    LOW: "#ca8a04",
    MEDIUM: "#ea580c",
    HIGH: "#dc2626",
    CRITICAL: "#b91c1c",
};

function fmtTime(ts) {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    return d.toLocaleString();
}

function escapeHtml(value) {
    return String(value == null ? "" : value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function riskPill(level) {
    const safe = escapeHtml(level || "SAFE");
    return `<span class="risk-pill risk-${safe}">${safe}</span>`;
}

function violationSummary(counts) {
    if (!counts || typeof counts !== "object") return "—";
    const parts = Object.entries(counts)
        .filter(([, v]) => v > 0)
        .map(([k, v]) => `${escapeHtml(k)}×${v}`);
    return parts.length ? parts.join(", ") : "—";
}

async function getJson(path) {
    const res = await fetch(apiUrl(path), { headers: { "Accept": "application/json" } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}


// ============================================================
// UI UPDATE HELPERS (live camera)
// ============================================================

function updateDetectionUI(summary, detections) {
    if (cameraPlaceholder) {
        cameraPlaceholder.style.display = "none";
    }

    if (summary) {
        if (riskBadge) {
            const risk = summary.risk_level || "SAFE";
            const valueEl = riskBadge.querySelector(".risk-value");
            if (valueEl) {
                valueEl.textContent = risk;
            } else {
                riskBadge.textContent = risk;
            }
            riskBadge.style.color = RISK_COLORS[risk] || "#6b7280";
        }

        if (summary.ppe_violations > 0) {
            ppeStatus.textContent = `${summary.ppe_violations} Violation(s)`;
            ppeStatus.style.color = "#dc2626";
        } else {
            ppeStatus.textContent = "Compliant";
            ppeStatus.style.color = "#16a34a";
        }

        if (summary.fire_count > 0) {
            fireStatus.textContent = `DETECTED (${summary.fire_count})`;
            fireStatus.style.color = "#dc2626";
        } else {
            fireStatus.textContent = "Clear";
            fireStatus.style.color = "#16a34a";
        }

        if (summary.smoke_count > 0) {
            smokeStatus.textContent = `DETECTED (${summary.smoke_count})`;
            smokeStatus.style.color = "#d97706";
        } else {
            smokeStatus.textContent = "Clear";
            smokeStatus.style.color = "#16a34a";
        }

        if (summary.person_count > 0) {
            personStatus.textContent = `${summary.person_count} Present`;
            personStatus.style.color = "#2563eb";
        } else {
            personStatus.textContent = "None";
            personStatus.style.color = "#6b7280";
        }
    }

    if (Array.isArray(detections)) {
        if (detections.length === 0) {
            detectionList.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                    <p>No detections in current frame</p>
                </div>`;
        } else {
            detectionList.innerHTML = detections.map(d => `
                <div class="detection">
                    <strong>${escapeHtml(d.class)}</strong> (${(d.confidence * 100).toFixed(1)}%)
                </div>
            `).join("");
        }
    }
}


// ============================================================
// START CAMERA
// ============================================================

async function startCamera() {
    try {
        cameraMessage.textContent = "Requesting camera permission...";

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false,
        });

        video.srcObject = cameraStream;
        await video.play();

        connectWebSocket();

        startButton.disabled = true;
        stopButton.disabled = false;

    } catch (error) {
        console.error("Camera error:", error);
        cameraMessage.textContent = "Could not access the camera.";
        alert("Camera access was denied or unavailable. Please check your camera permissions.");
    }
}


// ============================================================
// CONNECT WEBSOCKET
// ============================================================

function setConnectionState(state) {
    const dot = connectionStatus.querySelector(".status-dot");
    const text = connectionStatus.querySelector(".status-text");
    connectionStatus.classList.remove("connected", "disconnected");
    connectionStatus.classList.add(state);
    if (text) {
        text.textContent = state === "connected" ? "Connected" : "Disconnected";
    }
}

function connectWebSocket() {
    const wsUrl = getWebSocketUrl();
    console.log("Connecting to WebSocket URL:", wsUrl);

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        console.log("WebSocket connected successfully.");
        setConnectionState("connected");
        cameraMessage.textContent = "Camera is running. AI detection is active.";
        sendingFrames = true;
        sendFrame();
    };

    socket.onmessage = (event) => {
        if (event.data instanceof Blob) {
            const image = new Image();
            image.onload = () => {
                canvas.width = image.width;
                canvas.height = image.height;
                ctx.drawImage(image, 0, 0);
                URL.revokeObjectURL(image.src);
            };
            image.src = URL.createObjectURL(event.data);
        } else if (typeof event.data === "string") {
            try {
                const payload = JSON.parse(event.data);

                if (payload.summary || payload.detections) {
                    updateDetectionUI(payload.summary, payload.detections);
                }

                if (payload.frame_done) {
                    awaitingResponse = false;
                    setTimeout(sendFrame, 30);
                }
            } catch (err) {
                console.error("Error parsing WebSocket JSON message:", err);
                awaitingResponse = false;
            }
        }
    };

    socket.onerror = (error) => {
        console.error("WebSocket error details:", error);
        cameraMessage.textContent = "WebSocket connection error! Make sure the server is running on port 8000.";
        setConnectionState("disconnected");
        awaitingResponse = false;
    };

    socket.onclose = (event) => {
        console.log("WebSocket disconnected.", event);
        sendingFrames = false;
        awaitingResponse = false;
        setConnectionState("disconnected");
    };
}


// ============================================================
// SEND CAMERA FRAME
// ============================================================

function sendFrame() {
    if (!sendingFrames || awaitingResponse) return;

    if (!socket || socket.readyState !== WebSocket.OPEN) {
        if (socket && socket.readyState === WebSocket.CONNECTING) {
            setTimeout(sendFrame, 200);
        }
        return;
    }

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        setTimeout(sendFrame, 100);
        return;
    }

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 640;
    tempCanvas.height = 480;

    const tempContext = tempCanvas.getContext("2d");
    tempContext.drawImage(video, 0, 0, 640, 480);

    tempCanvas.toBlob(
        (blob) => {
            if (blob && socket && socket.readyState === WebSocket.OPEN) {
                awaitingResponse = true;
                socket.send(blob);
            }
        },
        "image/jpeg",
        0.5
    );
}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {
    console.log("Stopping camera...");

    sendingFrames = false;
    awaitingResponse = false;

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    if (socket) {
        socket.close();
        socket = null;
    }

    video.srcObject = null;

    startButton.disabled = false;
    stopButton.disabled = true;

    cameraMessage.textContent = "Camera stopped.";

    ppeStatus.textContent = "Waiting...";
    ppeStatus.style.color = "";
    fireStatus.textContent = "Waiting...";
    fireStatus.style.color = "";
    smokeStatus.textContent = "Waiting...";
    smokeStatus.style.color = "";
    personStatus.textContent = "Waiting...";
    personStatus.style.color = "";

    if (riskBadge) {
        const valueEl = riskBadge.querySelector(".risk-value");
        if (valueEl) valueEl.textContent = "—";
        riskBadge.style.color = "";
    }

    if (cameraPlaceholder) {
        cameraPlaceholder.style.display = "";
    }

    detectionList.innerHTML = `
        <div class="empty-state">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <p>No detections yet</p>
            <span>Start the camera to see live detections</span>
        </div>`;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
}


// ============================================================
// TABS / NAVIGATION
// ============================================================

function switchTab(name) {
    activeTab = name;

    document.querySelectorAll(".nav-item").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === name);
    });
    document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.toggle("active", panel.id === `panel-${name}`);
    });

    const titles = PAGE_TITLES[name] || ["", ""];
    if (pageTitle) pageTitle.textContent = titles[0];
    if (pageSubtitle) pageSubtitle.textContent = titles[1];

    closeMobileMenu();
    refreshActiveTab();
}

function refreshActiveTab() {
    switch (activeTab) {
        case "alerts": return loadAlerts();
        case "history": return loadHistory();
        case "statistics": return loadStatistics();
        case "evidence": return loadEvidence();
        case "reports": return loadReport();
        default: return Promise.resolve();
    }
}


// ============================================================
// MOBILE MENU
// ============================================================

function closeMobileMenu() {
    sidebar.classList.remove("open");
    sidebarOverlay.classList.remove("active");
}

if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener("click", () => {
        sidebar.classList.toggle("open");
        sidebarOverlay.classList.toggle("active");
    });
}

if (sidebarOverlay) {
    sidebarOverlay.addEventListener("click", closeMobileMenu);
}


// ============================================================
// ALERT BADGE (global)
// ============================================================

async function loadAlertBadge() {
    try {
        const data = await getJson("/api/alerts?limit=1&status=new");
        const n = data.unacknowledged || 0;
        const display = n > 99 ? "99+" : String(n);

        if (alertBadgeCount) alertBadgeCount.textContent = display;
        if (navAlertBadge) navAlertBadge.textContent = display;

        if (n > 0) {
            alertBadge.classList.remove("hidden");
            if (navAlertBadge) navAlertBadge.classList.remove("hidden");
        } else {
            alertBadge.classList.add("hidden");
            if (navAlertBadge) navAlertBadge.classList.add("hidden");
        }
    } catch (err) {
        console.debug("Alert badge poll failed:", err);
    }
}


// ============================================================
// ALERTS TAB
// ============================================================

async function loadAlerts() {
    const list = document.getElementById("alertsList");
    const status = document.getElementById("alertStatusFilter").value;
    const qs = status ? `?status=${encodeURIComponent(status)}&limit=100` : "?limit=100";

    try {
        const data = await getJson(`/api/alerts${qs}`);
        const alerts = data.alerts || [];

        if (alerts.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
                        <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
                    </svg>
                    <p>No alerts in this view</p>
                </div>`;
            return;
        }

        list.innerHTML = alerts.map(a => {
            const acked = a.status === "acknowledged";
            const ackControl = acked
                ? '<span class="alert-ack">Acknowledged</span>'
                : `<button class="btn btn-primary" data-ack="${a.id}">Acknowledge</button>`;
            return `
                <div class="alert-item level-${escapeHtml(a.level)}">
                    <div class="alert-main">
                        <div class="alert-title">${escapeHtml(a.title)}</div>
                        <div class="alert-meta">
                            ${riskPill(a.level)} · ${fmtTime(a.ts)} · ${escapeHtml(a.channel)}${a.notified ? " · sent" : ""}
                        </div>
                        <div class="alert-message">${escapeHtml(a.message)}</div>
                    </div>
                    <div>${ackControl}</div>
                </div>
            `;
        }).join("");

        list.querySelectorAll("button[data-ack]").forEach(btn => {
            btn.addEventListener("click", () => ackAlert(parseInt(btn.dataset.ack, 10)));
        });

    } catch (err) {
        list.innerHTML = `<div class="empty">Could not load alerts (${escapeHtml(err.message)}).</div>`;
    }
}

async function ackAlert(id) {
    try {
        await fetch(apiUrl(`/api/alerts/${id}/ack`), { method: "POST" });
        await Promise.all([loadAlerts(), loadAlertBadge()]);
    } catch (err) {
        console.error("Ack failed:", err);
    }
}


// ============================================================
// HISTORY TAB
// ============================================================

async function loadHistory() {
    const body = document.getElementById("historyBody");
    const risk = document.getElementById("historyRiskFilter").value;
    const source = document.getElementById("historySourceFilter").value;

    const params = new URLSearchParams({ limit: "200" });
    if (risk) params.set("risk", risk);
    if (source) params.set("source", source);

    try {
        const data = await getJson(`/api/events?${params.toString()}`);
        const events = data.events || [];

        if (events.length === 0) {
            body.innerHTML = '<tr><td colspan="8"><div class="empty">No events recorded.</div></td></tr>';
            return;
        }

        body.innerHTML = events.map(e => {
            const img = e.evidence_image ? `<a class="thumb-link" href="${evidenceUrl(e.evidence_image)}" target="_blank" rel="noopener">img</a>` : "";
            const clip = e.evidence_clip ? `<a class="thumb-link" href="${evidenceUrl(e.evidence_clip)}" target="_blank" rel="noopener">clip</a>` : "";
            const evidence = [img, clip].filter(Boolean).join(" · ") || "—";
            return `
                <tr>
                    <td>${fmtTime(e.ts)}</td>
                    <td>${escapeHtml(e.source)}</td>
                    <td>${riskPill(e.risk_level)}</td>
                    <td>${escapeHtml(violationSummary(e.violation_counts))}</td>
                    <td>${e.persons ?? 0}</td>
                    <td>${e.fire ?? 0}</td>
                    <td>${e.smoke ?? 0}</td>
                    <td>${evidence}</td>
                </tr>
            `;
        }).join("");

    } catch (err) {
        body.innerHTML = `<tr><td colspan="8"><div class="empty">Could not load history (${escapeHtml(err.message)}).</div></td></tr>`;
    }
}


// ============================================================
// STATISTICS TAB
// ============================================================

function barChart(container, entries, colorFn) {
    const el = document.getElementById(container);
    const items = entries.filter(([, v]) => v > 0);

    if (items.length === 0) {
        el.innerHTML = '<div class="empty">No data in this range.</div>';
        return;
    }

    const max = Math.max(...items.map(([, v]) => v));

    el.innerHTML = `<div class="bar-chart">${items.map(([label, value]) => {
        const pct = max > 0 ? (value / max) * 100 : 0;
        const color = colorFn ? colorFn(label) : "#2563eb";
        return `
            <div class="bar-row">
                <div class="bar-label" title="${escapeHtml(label)}">${escapeHtml(label)}</div>
                <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
                <div class="bar-value">${value}</div>
            </div>
        `;
    }).join("")}</div>`;
}

async function loadStatistics() {
    const cards = document.getElementById("statsCards");
    const range = document.getElementById("statsRange").value;

    try {
        const r = await getJson(`/api/statistics?range=${encodeURIComponent(range)}`);
        const t = r.totals || {};

        cards.innerHTML = [
            ["Events", t.events ?? 0],
            ["Alerts", t.alerts ?? 0],
            ["Unacked alerts", t.alerts_unacknowledged ?? 0],
            ["Frames processed", t.frames_processed ?? 0],
            ["With snapshot", t.events_with_image ?? 0],
            ["With clip", t.events_with_clip ?? 0],
        ].map(([label, value]) => `
            <div class="stat-card">
                <div class="stat-info">
                    <span class="stat-value">${value}</span>
                    <span class="stat-label">${escapeHtml(label)}</span>
                </div>
            </div>
        `).join("");

        const riskOrder = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"];
        barChart("riskChart", riskOrder.map(k => [k, (r.by_risk || {})[k] || 0]),
            label => RISK_COLORS[label] || "#2563eb");

        barChart("violationChart", Object.entries(r.by_violation_type || {}).slice(0, 10));
        barChart("dayChart", Object.entries(r.by_day || {}));

    } catch (err) {
        cards.innerHTML = `<div class="empty">Could not load statistics (${escapeHtml(err.message)}).</div>`;
    }
}


// ============================================================
// EVIDENCE TAB
// ============================================================

async function loadEvidence() {
    const gallery = document.getElementById("evidenceGallery");

    try {
        const data = await getJson("/api/evidence?limit=60");
        const items = data.evidence || [];

        if (items.length === 0) {
            gallery.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                        <circle cx="8.5" cy="8.5" r="1.5"/>
                        <polyline points="21 15 16 10 5 21"/>
                    </svg>
                    <p>No evidence captured yet</p>
                    <span>Evidence is saved automatically on HIGH/CRITICAL events</span>
                </div>`;
            return;
        }

        gallery.innerHTML = items.map(it => {
            let media;
            if (it.clip) {
                media = `<video class="evidence-media" src="${evidenceUrl(it.clip)}" controls muted></video>`;
            } else if (it.image) {
                media = `<img class="evidence-media" src="${evidenceUrl(it.image)}" alt="event ${it.event_id}" loading="lazy">`;
            } else {
                media = "";
            }
            return `
                <div class="evidence-card">
                    ${media}
                    <div class="evidence-body">
                        <div class="ev-head">
                            ${riskPill(it.risk_level)}
                            <span>${escapeHtml(it.source)}</span>
                        </div>
                        <div>#${it.event_id} · ${fmtTime(it.ts)}</div>
                        <div>${escapeHtml(violationSummary(it.violation_counts))}</div>
                    </div>
                </div>
            `;
        }).join("");

    } catch (err) {
        gallery.innerHTML = `<div class="empty">Could not load evidence (${escapeHtml(err.message)}).</div>`;
    }
}


// ============================================================
// REPORTS TAB
// ============================================================

function kv(label, value) {
    return `<div class="kv"><div class="k">${escapeHtml(label)}</div><div class="v">${value}</div></div>`;
}

async function loadReport() {
    const summary = document.getElementById("reportSummary");
    const range = document.getElementById("reportRange").value;

    document.getElementById("downloadJson").href =
        apiUrl(`/api/reports/download?range=${encodeURIComponent(range)}&format=json`);
    document.getElementById("downloadCsv").href =
        apiUrl(`/api/reports/download?range=${encodeURIComponent(range)}&format=csv`);

    try {
        const r = await getJson(`/api/reports?range=${encodeURIComponent(range)}`);
        const t = r.totals || {};

        const topRows = (r.top_events || []).map(e => `
            <tr>
                <td>${fmtTime(e.ts)}</td>
                <td>${escapeHtml(e.source)}</td>
                <td>${riskPill(e.risk_level)}</td>
                <td>${e.violation_count ?? 0}</td>
                <td>${escapeHtml(violationSummary(e.violation_counts))}</td>
            </tr>
        `).join("");

        summary.innerHTML = `
            <div class="report-block">
                <h3>Totals (${escapeHtml(r.range)})</h3>
                <div class="kv-grid">
                    ${kv("Events", t.events ?? 0)}
                    ${kv("Alerts", t.alerts ?? 0)}
                    ${kv("Acknowledged", t.alerts_acknowledged ?? 0)}
                    ${kv("Unacknowledged", t.alerts_unacknowledged ?? 0)}
                    ${kv("Frames", t.frames_processed ?? 0)}
                    ${kv("Snapshots", t.events_with_image ?? 0)}
                    ${kv("Clips", t.events_with_clip ?? 0)}
                </div>
            </div>

            <div class="report-block">
                <h3>By source</h3>
                <div class="kv-grid">
                    ${Object.entries(r.by_source || {}).map(([k, v]) => kv(k, v)).join("") || '<div class="empty">No data.</div>'}
                </div>
            </div>

            <div class="report-block">
                <h3>Most severe events</h3>
                <div class="table-container">
                    <table class="data-table">
                        <thead>
                            <tr><th>Time</th><th>Source</th><th>Risk</th><th>Violations</th><th>Detail</th></tr>
                        </thead>
                        <tbody>${topRows || '<tr><td colspan="5"><div class="empty">No events.</div></td></tr>'}</tbody>
                    </table>
                </div>
            </div>
        `;

    } catch (err) {
        summary.innerHTML = `<div class="empty">Could not generate report (${escapeHtml(err.message)}).</div>`;
    }
}


// ============================================================
// POLLING
// ============================================================

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => {
        loadAlertBadge();
        if (activeTab === "alerts") loadAlerts();
        else if (activeTab === "history") loadHistory();
        else if (activeTab === "statistics") loadStatistics();
    }, 10000);
}


// ============================================================
// EVENT WIRING
// ============================================================

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

document.querySelector(".sidebar-nav").addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-item");
    if (btn) switchTab(btn.dataset.tab);
});

document.getElementById("refreshAlerts").addEventListener("click", loadAlerts);
document.getElementById("alertStatusFilter").addEventListener("change", loadAlerts);

document.getElementById("refreshHistory").addEventListener("click", loadHistory);
document.getElementById("historyRiskFilter").addEventListener("change", loadHistory);
document.getElementById("historySourceFilter").addEventListener("change", loadHistory);

document.getElementById("refreshStats").addEventListener("click", loadStatistics);
document.getElementById("statsRange").addEventListener("change", loadStatistics);

document.getElementById("refreshEvidence").addEventListener("click", loadEvidence);

document.getElementById("generateReport").addEventListener("click", loadReport);
document.getElementById("reportRange").addEventListener("change", loadReport);


// ============================================================
// INIT
// ============================================================

loadAlertBadge();
startPolling();
