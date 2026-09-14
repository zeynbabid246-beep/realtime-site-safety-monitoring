// ============================================================
// CONFIGURATION & API HELPERS
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
    let host = window.location.hostname;
    if (!host || host === "") {
        host = "127.0.0.1";
    }
    return `${proto}//${host}:8000`;
}

const API = getApiBase();

function apiUrl(path) {
    return `${API}${path}`;
}

function outputUrl(path) {
    if (!path) return null;
    const filename = path.split(/[\\/]/).pop();
    return apiUrl(`/output/${filename}`);
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
const snapshotBtn = document.getElementById("snapshotBtn");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const cameraViewport = document.getElementById("cameraViewport");

const fpsCounter = document.getElementById("fpsCounter");
const resolutionBadge = document.getElementById("resolutionBadge");
const liveRecordingDot = document.getElementById("liveRecordingDot");

const connectionStatus = document.getElementById("connectionStatus");
const cameraMessage = document.getElementById("cameraMessage");
const riskBadge = document.getElementById("riskBadge");

const riskBanner = document.getElementById("riskBanner");
const bannerRiskValue = document.getElementById("bannerRiskValue");
const bannerViolations = document.getElementById("bannerViolations");
const bannerWorkers = document.getElementById("bannerWorkers");
const bannerProximity = document.getElementById("bannerProximity");
const bannerZones = document.getElementById("bannerZones");
const bannerMachinery = document.getElementById("bannerMachinery");

const ppeStatus = document.getElementById("ppeStatus");
const fireStatus = document.getElementById("fireStatus");
const smokeStatus = document.getElementById("smokeStatus");
const personStatus = document.getElementById("personStatus");

const ppeSubtext = document.getElementById("ppeSubtext");
const fireSubtext = document.getElementById("fireSubtext");
const smokeSubtext = document.getElementById("smokeSubtext");
const personSubtext = document.getElementById("personSubtext");

const proximityList = document.getElementById("proximityList");
const zoneList = document.getElementById("zoneList");
const detectionList = document.getElementById("detectionList");
const detectionFilterGroup = document.getElementById("detectionFilterGroup");

const alertBadge = document.getElementById("alertBadge");
const alertBadgeCount = document.getElementById("alertBadgeCount");
const navAlertBadge = document.getElementById("navAlertBadge");

const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const pageTitle = document.getElementById("pageTitle");
const pageSubtitle = document.getElementById("pageSubtitle");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebarOverlay");
const mobileMenuBtn = document.getElementById("mobileMenuBtn");

const audioToggleBtn = document.getElementById("audioToggleBtn");
const audioIconOn = document.getElementById("audioIconOn");
const audioIconOff = document.getElementById("audioIconOff");
const audioStatusText = document.getElementById("audioStatusText");

const toastContainer = document.getElementById("toastContainer");
const lightboxModal = document.getElementById("lightboxModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const modalBody = document.getElementById("modalBody");

const riskLegendModal = document.getElementById("riskLegendModal");
const riskLegendTriggerBtn = document.getElementById("riskLegendTriggerBtn");
const closeRiskLegendBtn = document.getElementById("closeRiskLegendBtn");


// ============================================================
// STATE VARIABLES
// ============================================================

let cameraStream = null;
let socket = null;
let sendingFrames = false;
let awaitingResponse = false;
let activeTab = "dashboard";
let pollTimer = null;
let audioEnabled = true;

let currentFrameDetections = [];
let currentDetectionFilter = "all";
let currentEvidenceFilter = "all";

// FPS Calculation
let frameCount = 0;
let lastFpsTime = performance.now();

// Audio Synthesizer Context
let audioCtx = null;

const PAGE_TITLES = {
    dashboard: ["Live Dashboard", "Real-time safety monitoring, tracking, and hazard detection"],
    detection: ["Media Analysis", "Inspect site photos and process full video streams for safety violations"],
    alerts: ["Alert Feed", "High and critical site safety alerts requiring acknowledgment"],
    history: ["Event History Log", "Complete safety event audit log with risk ratings and evidence"],
    statistics: ["Analytics & Trends", "Aggregated safety metrics, frame counts, and violation distributions"],
    evidence: ["Evidence Gallery", "Captured event JPEG snapshots and rolling video clip recordings"],
    reports: ["Safety Audit Reports", "Generate and export regulatory site compliance reports"],
};

const RISK_COLORS = {
    SAFE: "#16a34a",
    LOW: "#ca8a04",
    MEDIUM: "#ea580c",
    HIGH: "#dc2626",
    CRITICAL: "#991b1b",
};


// ============================================================
// UTILITY & FORMATTING HELPERS
// ============================================================

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
    if (!counts || typeof counts !== "object") return "None";
    const parts = Object.entries(counts)
        .filter(([, v]) => v > 0)
        .map(([k, v]) => `${escapeHtml(k)}×${v}`);
    return parts.length ? parts.join(", ") : "None";
}

async function getJson(path) {
    const res = await fetch(apiUrl(path), { headers: { "Accept": "application/json" } });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}


// ============================================================
// AUDIO SYNTHESIZER (Web Audio API)
// ============================================================

function playAlertSound(freq = 880, type = "sine", duration = 0.25) {
    if (!audioEnabled) return;
    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === "suspended") {
            audioCtx.resume();
        }
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.18, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        console.debug("Audio synthesis skipped:", e);
    }
}


// ============================================================
// TOAST NOTIFICATIONS
// ============================================================

function showToast(title, message, level = "HIGH") {
    if (!toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast toast-${level}`;
    toast.innerHTML = `
        <div>
            <div class="toast-title">${escapeHtml(title)}</div>
            <div class="toast-body">${escapeHtml(message)}</div>
        </div>
        <button class="toast-close" aria-label="Close notification">&times;</button>
    `;
    toast.querySelector(".toast-close").addEventListener("click", () => toast.remove());
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 6000);

    if (level === "CRITICAL") {
        playAlertSound(987, "sawtooth", 0.35);
    } else if (level === "HIGH") {
        playAlertSound(784, "sine", 0.25);
    }
}


// ============================================================
// MODALS (Lightbox & Risk Guide)
// ============================================================

function openLightbox(title, subtitle, mediaHtml, detailHtml = "") {
    document.getElementById("modalTitle").textContent = title;
    document.getElementById("modalSubtitle").textContent = subtitle;
    modalBody.innerHTML = `
        <div style="margin-bottom: 16px;">${mediaHtml}</div>
        <div>${detailHtml}</div>
    `;
    lightboxModal.classList.remove("hidden");
}

function closeLightbox() {
    lightboxModal.classList.add("hidden");
    modalBody.innerHTML = "";
}

if (closeModalBtn) closeModalBtn.addEventListener("click", closeLightbox);
if (lightboxModal) {
    lightboxModal.addEventListener("click", (e) => {
        if (e.target === lightboxModal) closeLightbox();
    });
}

function openRiskLegend() {
    if (riskLegendModal) riskLegendModal.classList.remove("hidden");
}

function closeRiskLegend() {
    if (riskLegendModal) riskLegendModal.classList.add("hidden");
}

if (riskLegendTriggerBtn) riskLegendTriggerBtn.addEventListener("click", openRiskLegend);
if (closeRiskLegendBtn) closeRiskLegendBtn.addEventListener("click", closeRiskLegend);
if (riskLegendModal) {
    riskLegendModal.addEventListener("click", (e) => {
        if (e.target === riskLegendModal) closeRiskLegend();
    });
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        closeLightbox();
        closeRiskLegend();
        closeMobileMenu();
    }
});


// ============================================================
// REAL-TIME UI UPDATE LOGIC (Dashboard HUD)
// ============================================================

function updateDetectionUI(summary, detections) {
    if (cameraPlaceholder) {
        cameraPlaceholder.style.display = "none";
    }

    // Calculate FPS
    frameCount++;
    const now = performance.now();
    if (now - lastFpsTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastFpsTime));
        if (fpsCounter) {
            fpsCounter.textContent = `${fps} FPS`;
            fpsCounter.classList.remove("hidden");
        }
        frameCount = 0;
        lastFpsTime = now;
    }

    if (summary) {
        const risk = summary.risk_level || "SAFE";

        // Update HUD Risk Banner
        if (riskBanner) {
            riskBanner.className = `risk-banner banner-${risk}`;
            if (bannerRiskValue) bannerRiskValue.textContent = risk;
            if (bannerViolations) bannerViolations.textContent = summary.violation_count || 0;
            if (bannerWorkers) bannerWorkers.textContent = summary.person_count || 0;
            if (bannerProximity) bannerProximity.textContent = (summary.proximity_alerts || []).length;
            if (bannerZones) bannerZones.textContent = (summary.danger_zones || []).length;
            if (bannerMachinery) bannerMachinery.textContent = summary.machine_count || 0;
        }

        // Camera Footer Badge
        if (riskBadge) {
            const valueEl = riskBadge.querySelector(".risk-value");
            if (valueEl) valueEl.textContent = risk;
            riskBadge.style.color = RISK_COLORS[risk] || "#64748b";
        }

        // PPE Compliance Card
        if (ppeStatus) {
            if (summary.ppe_violations > 0) {
                ppeStatus.textContent = `${summary.ppe_violations} Violation(s)`;
                ppeStatus.style.color = "#dc2626";
                if (ppeSubtext) ppeSubtext.textContent = "Non-compliant gear detected";
            } else {
                ppeStatus.textContent = "Compliant";
                ppeStatus.style.color = "#16a34a";
                if (ppeSubtext) ppeSubtext.textContent = "Hardhats & Vests Verified";
            }
        }

        // Fire Hazard Card
        if (fireStatus) {
            if (summary.fire_count > 0) {
                fireStatus.textContent = `DETECTED (${summary.fire_count})`;
                fireStatus.style.color = "#dc2626";
                if (fireSubtext) fireSubtext.textContent = "Active combustion hazard!";
            } else {
                fireStatus.textContent = "Clear";
                fireStatus.style.color = "#16a34a";
                if (fireSubtext) fireSubtext.textContent = "Combustion Debounced";
            }
        }

        // Smoke Hazard Card
        if (smokeStatus) {
            if (summary.smoke_count > 0) {
                smokeStatus.textContent = `DETECTED (${summary.smoke_count})`;
                smokeStatus.style.color = "#ca8a04";
                if (smokeSubtext) smokeSubtext.textContent = "Plume hazard detected";
            } else {
                smokeStatus.textContent = "Clear";
                smokeStatus.style.color = "#16a34a";
                if (smokeSubtext) smokeSubtext.textContent = "Thermal / Plume Filter";
            }
        }

        // Worker & Machinery Card
        if (personStatus) {
            if (summary.person_count > 0) {
                personStatus.textContent = `${summary.person_count} Worker(s)`;
                personStatus.style.color = "#0284c7";
                if (personSubtext) personSubtext.textContent = `${summary.machine_count || 0} Heavy Equipment Active`;
            } else {
                personStatus.textContent = "None";
                personStatus.style.color = "#64748b";
                if (personSubtext) personSubtext.textContent = "ByteTrack Persistent IDs";
            }
        }

        // Proximity Breaches List
        if (proximityList) {
            const proxAlerts = summary.proximity_alerts || [];
            if (proxAlerts.length === 0) {
                proximityList.innerHTML = `
                    <div class="empty-state">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                        <p>No proximity hazards detected</p>
                        <span>Safe distance maintained between personnel and heavy equipment</span>
                    </div>`;
            } else {
                proximityList.innerHTML = proxAlerts.map(p => `
                    <div class="item-badge danger">
                        <span>⚠️ Worker ID:${p.person_id ?? 'Unassigned'} ↔ ${escapeHtml(p.machine_type || 'Machinery')}</span>
                        <strong>${Math.round(p.distance_px)} px ground vector</strong>
                    </div>
                `).join("");
            }
        }

        // Active Danger Zones
        if (zoneList) {
            const zones = summary.danger_zones || [];
            if (zones.length === 0) {
                zoneList.innerHTML = `
                    <div class="empty-state">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><polygon points="12 2 2 22 22 22 12 2"/></svg>
                        <p>No active zone incursions</p>
                        <span>All workers operating outside defined cone boundary clusters</span>
                    </div>`;
            } else {
                zoneList.innerHTML = zones.map(z => `
                    <div class="item-badge danger">
                        <span>🚨 Cone Danger Zone #${z.zone_id || 'Cluster'} (${z.worker_count || 1} worker inside)</span>
                        <strong>ACTIVE INVASION</strong>
                    </div>
                `).join("");
            }
        }

        // Trigger toast on high or critical
        if ((risk === "HIGH" || risk === "CRITICAL") && summary.violation_count > 0) {
            showToast(`${risk} Safety Alert Triggered`, `${summary.violation_count} active safety violation(s) on job site`, risk);
        }
    }

    // Current Frame Detections List
    if (Array.isArray(detections)) {
        currentFrameDetections = detections;
        renderDetectionList();
    }
}

function renderDetectionList() {
    if (!detectionList) return;

    let filtered = currentFrameDetections;
    if (currentDetectionFilter === "worker") {
        filtered = filtered.filter(d => d.class === "Person");
    } else if (currentDetectionFilter === "ppe") {
        filtered = filtered.filter(d => ["Hardhat", "Safety Vest", "Mask", "NO-Hardhat", "NO-Safety Vest", "NO-Mask"].includes(d.class));
    } else if (currentDetectionFilter === "machinery") {
        filtered = filtered.filter(d => ["machinery", "vehicle", "utility pole"].includes(d.class));
    } else if (currentDetectionFilter === "hazard") {
        filtered = filtered.filter(d => ["Fire", "Smoke", "NO-Hardhat", "NO-Safety Vest", "NO-Mask"].includes(d.class));
    }

    if (filtered.length === 0) {
        detectionList.innerHTML = `
            <div class="empty-state">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                <p>No ${currentDetectionFilter !== "all" ? currentDetectionFilter : ""} detections in current buffer</p>
            </div>`;
        return;
    }

    detectionList.innerHTML = filtered.map(d => {
        const trackStr = d.track_id != null ? `[ID:${d.track_id}]` : "";
        const confPct = (d.confidence * 100).toFixed(1);
        return `
            <div class="detection">
                <strong>${escapeHtml(d.class)} ${trackStr}</strong> (${confPct}%)
            </div>
        `;
    }).join("");
}

if (detectionFilterGroup) {
    detectionFilterGroup.addEventListener("click", (e) => {
        const btn = e.target.closest(".filter-pill");
        if (!btn) return;
        detectionFilterGroup.querySelectorAll(".filter-pill").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentDetectionFilter = btn.dataset.filter || "all";
        renderDetectionList();
    });
}


// ============================================================
// CAMERA STREAM & WEBSOCKET MANAGEMENT
// ============================================================

async function startCamera() {
    try {
        cameraMessage.textContent = "Requesting site camera permissions...";

        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 },
            audio: false,
        });

        video.srcObject = cameraStream;
        await video.play();

        if (cameraPlaceholder) cameraPlaceholder.style.display = "none";
        if (liveRecordingDot) liveRecordingDot.classList.remove("hidden");
        if (resolutionBadge) {
            resolutionBadge.textContent = `${video.videoWidth || 640}×${video.videoHeight || 480}`;
            resolutionBadge.classList.remove("hidden");
        }

        connectWebSocket();

        startButton.disabled = true;
        stopButton.disabled = false;
        snapshotBtn.disabled = false;

    } catch (error) {
        console.error("Camera error:", error);
        cameraMessage.textContent = "Could not access site camera.";
        showToast("Camera Error", "Camera access denied or device unavailable", "HIGH");
    }
}

function setConnectionState(state) {
    if (!connectionStatus) return;
    const text = connectionStatus.querySelector(".status-text");
    connectionStatus.classList.remove("connected", "disconnected");
    connectionStatus.classList.add(state);
    if (text) {
        text.textContent = state === "connected" ? "Stream Active" : "Disconnected";
    }
}

function connectWebSocket() {
    const wsUrl = getWebSocketUrl();
    console.log("Connecting to WebSocket:", wsUrl);

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
        setConnectionState("connected");
        cameraMessage.textContent = "Live camera stream connected. Safety AI multi-model pipeline running.";
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
                console.error("WebSocket JSON parse error:", err);
                awaitingResponse = false;
            }
        }
    };

    socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        cameraMessage.textContent = "Connection error. Verify backend server status.";
        setConnectionState("disconnected");
        awaitingResponse = false;
    };

    socket.onclose = () => {
        sendingFrames = false;
        awaitingResponse = false;
        setConnectionState("disconnected");
    };
}

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

function stopCamera() {
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

    if (cameraPlaceholder) cameraPlaceholder.style.display = "";
    if (liveRecordingDot) liveRecordingDot.classList.add("hidden");
    if (resolutionBadge) resolutionBadge.classList.add("hidden");

    startButton.disabled = false;
    stopButton.disabled = true;
    snapshotBtn.disabled = true;

    if (fpsCounter) fpsCounter.classList.add("hidden");
    cameraMessage.textContent = "Camera stream stopped.";

    ppeStatus.textContent = "Compliant";
    ppeStatus.style.color = "";
    fireStatus.textContent = "Clear";
    fireStatus.style.color = "";
    smokeStatus.textContent = "Clear";
    smokeStatus.style.color = "";
    personStatus.textContent = "None";
    personStatus.style.color = "";

    if (riskBadge) {
        const valueEl = riskBadge.querySelector(".risk-value");
        if (valueEl) valueEl.textContent = "—";
        riskBadge.style.color = "";
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// SNAPSHOT & FULLSCREEN UTILITIES
function captureSnapshot() {
    if (!canvas || canvas.width === 0) return;
    const dataUrl = canvas.toDataURL("image/jpeg");
    const link = document.createElement("a");
    link.href = dataUrl;
    link.download = `site_snapshot_${Date.now()}.jpg`;
    link.click();
    showToast("Snapshot Saved", "Live camera frame captured to downloads", "HIGH");
}

function toggleFullscreen() {
    if (!cameraViewport) return;
    if (!document.fullscreenElement) {
        cameraViewport.requestFullscreen().catch(err => console.error("Fullscreen failed:", err));
    } else {
        document.exitFullscreen();
    }
}

if (snapshotBtn) snapshotBtn.addEventListener("click", captureSnapshot);
if (fullscreenBtn) fullscreenBtn.addEventListener("click", toggleFullscreen);


// ============================================================
// MEDIA ANALYSIS (Image & Video File Uploads)
// ============================================================

const mediaTabBtns = document.querySelectorAll(".media-tab-btn");
mediaTabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        mediaTabBtns.forEach(b => {
            b.classList.remove("active");
            b.setAttribute("aria-selected", "false");
        });
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        const type = btn.dataset.mediatab;
        document.getElementById("media-view-image").classList.toggle("hidden", type !== "image");
        document.getElementById("media-view-video").classList.toggle("hidden", type !== "video");
    });
});

// IMAGE UPLOAD HANDLING
const imageDropzone = document.getElementById("imageDropzone");
const imageFileInput = document.getElementById("imageFileInput");
const imageProcessingState = document.getElementById("imageProcessingState");
const imageResultContainer = document.getElementById("imageResultContainer");
const imageOriginalPreview = document.getElementById("imageOriginalPreview");
const imageAnnotatedPreview = document.getElementById("imageAnnotatedPreview");
const imageSummaryCard = document.getElementById("imageSummaryCard");

if (imageDropzone && imageFileInput) {
    imageDropzone.addEventListener("click", () => imageFileInput.click());
    imageDropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            imageFileInput.click();
        }
    });
    imageDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        imageDropzone.classList.add("dragover");
    });
    imageDropzone.addEventListener("dragleave", () => imageDropzone.classList.remove("dragover"));
    imageDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        imageDropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });
    imageFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleImageUpload(e.target.files[0]);
        }
    });
}

async function handleImageUpload(file) {
    if (!file.type.startsWith("image/")) {
        showToast("Invalid File", "Please upload a valid image file (PNG, JPG, WEBP)", "HIGH");
        return;
    }

    imageDropzone.classList.add("hidden");
    imageProcessingState.classList.remove("hidden");
    imageResultContainer.classList.add("hidden");

    imageOriginalPreview.src = URL.createObjectURL(file);

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(apiUrl("/safety/detect/image"), {
            method: "POST",
            body: formData,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        if (!data.success) {
            throw new Error(data.error || "Image processing failed");
        }

        const summary = data.result.summary || {};

        imageAnnotatedPreview.src = outputUrl(data.annotated_image_path);

        imageSummaryCard.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <h4 style="font-size: 16px; font-weight: 800; color: var(--slate-900);">Safety AI Image Inspection Result</h4>
                ${riskPill(summary.risk_level)}
            </div>
            <div class="kv-grid" style="margin-bottom: 16px;">
                <div class="kv"><div class="k">Active Violations</div><div class="v">${summary.violation_count || 0}</div></div>
                <div class="kv"><div class="k">PPE Violations</div><div class="v">${summary.ppe_violations || 0}</div></div>
                <div class="kv"><div class="k">Tracked Workers</div><div class="v">${summary.person_count || 0}</div></div>
                <div class="kv"><div class="k">Heavy Machinery</div><div class="v">${summary.machine_count || 0}</div></div>
            </div>
            <div>
                <strong>Violation Breakdown:</strong> ${escapeHtml(violationSummary(summary.violation_counts))}
            </div>
        `;

        imageProcessingState.classList.add("hidden");
        imageResultContainer.classList.remove("hidden");
        imageDropzone.classList.remove("hidden");

        showToast("Image Analyzed", `Detection completed. Risk Level: ${summary.risk_level || 'SAFE'}`, summary.risk_level === "CRITICAL" ? "CRITICAL" : "HIGH");

    } catch (err) {
        console.error("Image upload error:", err);
        imageProcessingState.classList.add("hidden");
        imageDropzone.classList.remove("hidden");
        showToast("Upload Error", err.message || "Failed to process image", "HIGH");
    }
}

// VIDEO UPLOAD HANDLING
const videoDropzone = document.getElementById("videoDropzone");
const videoFileInput = document.getElementById("videoFileInput");
const videoProcessingState = document.getElementById("videoProcessingState");
const videoResultContainer = document.getElementById("videoResultContainer");
const videoResultPlayer = document.getElementById("videoResultPlayer");
const videoDownloadBtn = document.getElementById("videoDownloadBtn");
const videoSummaryCard = document.getElementById("videoSummaryCard");

if (videoDropzone && videoFileInput) {
    videoDropzone.addEventListener("click", () => videoFileInput.click());
    videoDropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            videoFileInput.click();
        }
    });
    videoDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        videoDropzone.classList.add("dragover");
    });
    videoDropzone.addEventListener("dragleave", () => videoDropzone.classList.remove("dragover"));
    videoDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        videoDropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleVideoUpload(e.dataTransfer.files[0]);
        }
    });
    videoFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleVideoUpload(e.target.files[0]);
        }
    });
}

async function handleVideoUpload(file) {
    if (!file.type.startsWith("video/")) {
        showToast("Invalid File", "Please upload a valid video file (MP4, AVI, MOV)", "HIGH");
        return;
    }

    videoDropzone.classList.add("hidden");
    videoProcessingState.classList.remove("hidden");
    videoResultContainer.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(apiUrl("/safety/detect/video"), {
            method: "POST",
            body: formData,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const rawBlob = await res.blob();
        const videoBlob = new Blob([rawBlob], { type: "video/mp4" });
        const videoUrl = URL.createObjectURL(videoBlob);

        const framesProcessed = res.headers.get("X-Frames-Processed") || "—";
        const maxRisk = res.headers.get("X-Max-Risk-Level") || "SAFE";

        videoResultPlayer.src = videoUrl;
        videoResultPlayer.load();
        videoDownloadBtn.href = videoUrl;

        videoSummaryCard.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <h4 style="font-size: 16px; font-weight: 800; color: var(--slate-900);">Offline Video Pass Completed</h4>
                ${riskPill(maxRisk)}
            </div>
            <div class="kv-grid">
                <div class="kv"><div class="k">Frames Analyzed</div><div class="v">${framesProcessed}</div></div>
                <div class="kv"><div class="k">Peak Risk Index</div><div class="v">${escapeHtml(maxRisk)}</div></div>
            </div>
        `;

        videoProcessingState.classList.add("hidden");
        videoResultContainer.classList.remove("hidden");
        videoDropzone.classList.remove("hidden");

        showToast("Video Processing Complete", `Analyzed ${framesProcessed} video frames`, "HIGH");

    } catch (err) {
        console.error("Video processing error:", err);
        videoProcessingState.classList.add("hidden");
        videoDropzone.classList.remove("hidden");
        showToast("Video Error", err.message || "Failed to analyze video", "HIGH");
    }
}


// ============================================================
// NAVIGATION & TAB SWITCHING
// ============================================================

function switchTab(name) {
    activeTab = name;

    document.querySelectorAll(".nav-item").forEach(btn => {
        const isActive = btn.dataset.tab === name;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-selected", isActive ? "true" : "false");
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

function closeMobileMenu() {
    if (sidebar) sidebar.classList.remove("open");
    if (sidebarOverlay) sidebarOverlay.classList.remove("active");
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
// AUDIO TOGGLE
// ============================================================

if (audioToggleBtn) {
    audioToggleBtn.addEventListener("click", () => {
        audioEnabled = !audioEnabled;
        audioIconOn.classList.toggle("hidden", !audioEnabled);
        audioIconOff.classList.toggle("hidden", audioEnabled);
        audioStatusText.textContent = audioEnabled ? "Audio Alerts On" : "Audio Alerts Off";
        if (audioEnabled) playAlertSound(600, "sine", 0.15);
    });
}


// ============================================================
// ALERT FEED & BADGES
// ============================================================

async function loadAlertBadge() {
    try {
        const data = await getJson("/api/alerts?limit=1&status=new");
        const n = data.unacknowledged || 0;
        const display = n > 99 ? "99+" : String(n);

        if (alertBadgeCount) {
            alertBadgeCount.textContent = display;
            alertBadgeCount.classList.toggle("hidden", n === 0);
        }
        if (navAlertBadge) {
            navAlertBadge.textContent = display;
            navAlertBadge.classList.toggle("hidden", n === 0);
        }
    } catch (err) {
        console.debug("Alert badge check failed:", err);
    }
}

async function loadAlerts() {
    const list = document.getElementById("alertsList");
    const status = document.getElementById("alertStatusFilter").value;
    const level = document.getElementById("alertLevelFilter").value;

    const params = new URLSearchParams({ limit: "100" });
    if (status) params.set("status", status);
    if (level) params.set("level", level);

    try {
        const data = await getJson(`/api/alerts?${params.toString()}`);
        const alerts = data.alerts || [];

        if (alerts.length === 0) {
            list.innerHTML = `
                <div class="empty-state">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
                    <p>No safety alerts matching current criteria</p>
                </div>`;
            return;
        }

        list.innerHTML = alerts.map(a => {
            const acked = a.status === "acknowledged";
            const ackControl = acked
                ? '<span style="color: var(--green-700); font-size: 12px; font-weight: 700;">✓ Acknowledged</span>'
                : `<button class="btn btn-primary btn-sm" data-ack="${a.id}">Acknowledge</button>`;
            return `
                <div class="alert-item level-${escapeHtml(a.level)}">
                    <div class="alert-main">
                        <div class="alert-title">${escapeHtml(a.title)}</div>
                        <div class="alert-meta">
                            ${riskPill(a.level)} · ${fmtTime(a.ts)} · Source: ${escapeHtml(a.channel)}${a.notified ? " · Telegram Notified" : ""}
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
        list.innerHTML = `<div class="empty-state"><p>Could not load alerts (${escapeHtml(err.message)})</p></div>`;
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

async function ackAllAlerts() {
    try {
        const data = await getJson("/api/alerts?limit=100&status=new");
        const newAlerts = data.alerts || [];
        for (const a of newAlerts) {
            await fetch(apiUrl(`/api/alerts/${a.id}/ack`), { method: "POST" });
        }
        showToast("Alerts Acknowledged", `Acknowledged ${newAlerts.length} active alerts`, "HIGH");
        await Promise.all([loadAlerts(), loadAlertBadge()]);
    } catch (err) {
        console.error("Ack all failed:", err);
    }
}

const ackAllAlertsBtn = document.getElementById("ackAllAlertsBtn");
if (ackAllAlertsBtn) ackAllAlertsBtn.addEventListener("click", ackAllAlerts);


// ============================================================
// EVENT HISTORY LOG
// ============================================================

async function loadHistory() {
    const body = document.getElementById("historyBody");
    const risk = document.getElementById("historyRiskFilter").value;
    const source = document.getElementById("historySourceFilter").value;
    const search = (document.getElementById("historySearchInput")?.value || "").toLowerCase().trim();

    const params = new URLSearchParams({ limit: "200" });
    if (risk) params.set("risk", risk);
    if (source) params.set("source", source);

    try {
        const data = await getJson(`/api/events?${params.toString()}`);
        let events = data.events || [];

        if (search) {
            events = events.filter(e =>
                String(e.id).includes(search) ||
                JSON.stringify(e.violation_counts || {}).toLowerCase().includes(search)
            );
        }

        if (events.length === 0) {
            body.innerHTML = '<tr><td colspan="8"><div class="empty-state"><p>No recorded safety events matching filters.</p></div></td></tr>';
            return;
        }

        body.innerHTML = events.map(e => {
            const img = e.evidence_image ? `<a href="#" class="thumb-link" data-img="${evidenceUrl(e.evidence_image)}" data-id="${e.id}">Snapshot</a>` : "";
            const clip = e.evidence_clip ? `<a href="#" class="thumb-link" data-clip="${evidenceUrl(e.evidence_clip)}" data-id="${e.id}">Video Clip</a>` : "";
            const evidence = [img, clip].filter(Boolean).join(" · ") || "—";
            return `
                <tr>
                    <td><strong>#${e.id}</strong></td>
                    <td>${fmtTime(e.ts)}</td>
                    <td><strong>${escapeHtml(e.source)}</strong></td>
                    <td>${riskPill(e.risk_level)}</td>
                    <td>${escapeHtml(violationSummary(e.violation_counts))}</td>
                    <td>${e.persons ?? 0}</td>
                    <td>${(e.fire || e.smoke) ? "DETECTED" : "Clear"}</td>
                    <td>${evidence}</td>
                </tr>
            `;
        }).join("");

        body.querySelectorAll("a[data-img], a[data-clip]").forEach(link => {
            link.addEventListener("click", (e) => {
                e.preventDefault();
                const img = link.dataset.img;
                const clip = link.dataset.clip;
                const id = link.dataset.id;
                let mediaHtml = "";
                if (clip) {
                    mediaHtml = `<video class="modal-media" src="${clip}" controls autoplay></video>`;
                } else if (img) {
                    mediaHtml = `<img class="modal-media" src="${img}" alt="Event #${id}">`;
                }
                openLightbox(`Event #${id} Evidence Inspector`, `Recorded snapshot/video clip`, mediaHtml);
            });
        });

    } catch (err) {
        body.innerHTML = `<tr><td colspan="8"><div class="empty-state"><p>Could not load event history (${escapeHtml(err.message)})</p></div></td></tr>`;
    }
}

const historySearchInput = document.getElementById("historySearchInput");
if (historySearchInput) {
    historySearchInput.addEventListener("input", loadHistory);
}


// ============================================================
// ANALYTICS & CHARTS
// ============================================================

function barChart(container, entries, colorFn) {
    const el = document.getElementById(container);
    const items = entries.filter(([, v]) => v > 0);

    if (items.length === 0) {
        el.innerHTML = '<div class="empty-state"><p>No data available for selected time window.</p></div>';
        return;
    }

    const max = Math.max(...items.map(([, v]) => v));

    el.innerHTML = `<div class="bar-chart">${items.map(([label, value]) => {
        const pct = max > 0 ? (value / max) * 100 : 0;
        const color = colorFn ? colorFn(label) : "#0284c7";
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
            ["Recorded Events", t.events ?? 0],
            ["Generated Alerts", t.alerts ?? 0],
            ["Unacknowledged Alerts", t.alerts_unacknowledged ?? 0],
            ["Frames Monitored", t.frames_processed ?? 0],
            ["Captured Snapshots", t.events_with_image ?? 0],
            ["Video Recordings", t.events_with_clip ?? 0],
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
            label => RISK_COLORS[label] || "#0284c7");

        barChart("violationChart", Object.entries(r.by_violation_type || {}).slice(0, 10));
        barChart("dayChart", Object.entries(r.by_day || {}));

    } catch (err) {
        cards.innerHTML = `<div class="empty-state"><p>Could not load analytics (${escapeHtml(err.message)})</p></div>`;
    }
}


// ============================================================
// EVIDENCE GALLERY
// ============================================================

async function loadEvidence() {
    const gallery = document.getElementById("evidenceGallery");

    try {
        const data = await getJson("/api/evidence?limit=60");
        let items = data.evidence || [];

        if (currentEvidenceFilter === "clips") {
            items = items.filter(it => it.clip);
        } else if (currentEvidenceFilter === "snapshots") {
            items = items.filter(it => it.image && !it.clip);
        }

        if (items.length === 0) {
            gallery.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                    <p>No evidence artifacts captured yet</p>
                    <span>High and critical risk safety violations automatically capture annotated snapshots & clips</span>
                </div>`;
            return;
        }

        gallery.innerHTML = items.map(it => {
            let media;
            if (it.clip) {
                media = `<video class="evidence-media" src="${evidenceUrl(it.clip)}" muted></video>`;
            } else if (it.image) {
                media = `<img class="evidence-media" src="${evidenceUrl(it.image)}" alt="Event #${it.event_id}" loading="lazy">`;
            } else {
                media = "";
            }
            return `
                <div class="evidence-card" data-id="${it.event_id}" data-img="${evidenceUrl(it.image)}" data-clip="${evidenceUrl(it.clip)}">
                    ${media}
                    <div class="evidence-body">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            ${riskPill(it.risk_level)}
                            <strong>#${it.event_id}</strong>
                        </div>
                        <div>Source: ${escapeHtml(it.source)} · ${fmtTime(it.ts)}</div>
                        <div style="margin-top: 4px; font-weight: 600; color: var(--slate-700);">${escapeHtml(violationSummary(it.violation_counts))}</div>
                    </div>
                </div>
            `;
        }).join("");

        gallery.querySelectorAll(".evidence-card").forEach(card => {
            card.addEventListener("click", () => {
                const img = card.dataset.img;
                const clip = card.dataset.clip;
                const id = card.dataset.id;
                let mediaHtml = "";
                if (clip && clip !== "null") {
                    mediaHtml = `<video class="modal-media" src="${clip}" controls autoplay></video>`;
                } else if (img && img !== "null") {
                    mediaHtml = `<img class="modal-media" src="${img}" alt="Event #${id}">`;
                }
                openLightbox(`Evidence Inspector - Event #${id}`, `High-resolution incident recording`, mediaHtml);
            });
        });

    } catch (err) {
        gallery.innerHTML = `<div class="empty-state"><p>Could not load evidence gallery (${escapeHtml(err.message)})</p></div>`;
    }
}

const evidenceFilterGroup = document.getElementById("evidenceFilterGroup");
if (evidenceFilterGroup) {
    evidenceFilterGroup.addEventListener("click", (e) => {
        const btn = e.target.closest(".filter-pill");
        if (!btn) return;
        evidenceFilterGroup.querySelectorAll(".filter-pill").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentEvidenceFilter = btn.dataset.evidenceFilter || "all";
        loadEvidence();
    });
}


// ============================================================
// SAFETY AUDIT REPORTS GENERATOR
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
                <td><strong>#${e.id}</strong></td>
                <td>${fmtTime(e.ts)}</td>
                <td>${escapeHtml(e.source)}</td>
                <td>${riskPill(e.risk_level)}</td>
                <td>${e.violation_count ?? 0}</td>
                <td>${escapeHtml(violationSummary(e.violation_counts))}</td>
            </tr>
        `).join("");

        summary.innerHTML = `
            <div style="margin-bottom: 24px;">
                <h3 style="font-size: 16px; font-weight: 800; color: var(--slate-900); margin-bottom: 12px;">Site Compliance Overview (${escapeHtml(r.range)})</h3>
                <div class="kv-grid">
                    ${kv("Total Incidents", t.events ?? 0)}
                    ${kv("Alerts Generated", t.alerts ?? 0)}
                    ${kv("Acknowledged Alerts", t.alerts_acknowledged ?? 0)}
                    ${kv("Unacknowledged Alerts", t.alerts_unacknowledged ?? 0)}
                    ${kv("Frames Monitored", t.frames_processed ?? 0)}
                    ${kv("Snapshots Saved", t.events_with_image ?? 0)}
                    ${kv("Video Clips Saved", t.events_with_clip ?? 0)}
                </div>
            </div>

            <div style="margin-bottom: 24px;">
                <h3 style="font-size: 16px; font-weight: 800; color: var(--slate-900); margin-bottom: 12px;">Events by Input Source</h3>
                <div class="kv-grid">
                    ${Object.entries(r.by_source || {}).map(([k, v]) => kv(k, v)).join("") || '<div class="empty-state"><p>No source data recorded.</p></div>'}
                </div>
            </div>

            <div>
                <h3 style="font-size: 16px; font-weight: 800; color: var(--slate-900); margin-bottom: 12px;">Most Severe Site Incidents</h3>
                <div class="table-container" style="border: 1px solid var(--slate-200); border-radius: var(--radius-md);">
                    <table class="data-table">
                        <thead>
                            <tr><th>Event ID</th><th>Timestamp</th><th>Source</th><th>Risk Level</th><th>Violations</th><th>Detail</th></tr>
                        </thead>
                        <tbody>${topRows || '<tr><td colspan="6"><div class="empty-state"><p>No severe incidents recorded in window.</p></div></td></tr>'}</tbody>
                    </table>
                </div>
            </div>
        `;

    } catch (err) {
        summary.innerHTML = `<div class="empty-state"><p>Could not generate audit report (${escapeHtml(err.message)})</p></div>`;
    }
}


// ============================================================
// POLLING & EVENT LISTENERS
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


// EVENT BINDINGS
startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);

if (alertBadge) {
    alertBadge.addEventListener("click", () => switchTab("alerts"));
}

document.querySelector(".sidebar-nav").addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-item");
    if (btn) switchTab(btn.dataset.tab);
});

document.getElementById("refreshAlerts").addEventListener("click", loadAlerts);
document.getElementById("alertStatusFilter").addEventListener("change", loadAlerts);
document.getElementById("alertLevelFilter").addEventListener("change", loadAlerts);

document.getElementById("refreshHistory").addEventListener("click", loadHistory);
document.getElementById("historyRiskFilter").addEventListener("change", loadHistory);
document.getElementById("historySourceFilter").addEventListener("change", loadHistory);

document.getElementById("refreshStats").addEventListener("click", loadStatistics);
document.getElementById("statsRange").addEventListener("change", loadStatistics);

document.getElementById("refreshEvidence").addEventListener("click", loadEvidence);

document.getElementById("generateReport").addEventListener("click", loadReport);
document.getElementById("reportRange").addEventListener("change", loadReport);


// INITIALIZE
loadAlertBadge();
startPolling();
