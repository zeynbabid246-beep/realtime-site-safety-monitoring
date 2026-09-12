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
const captureSnapshotBtn = document.getElementById("captureSnapshotBtn");
const fpsCounter = document.getElementById("fpsCounter");

const toggleBboxes = document.getElementById("toggleBboxes");
const toggleTrackIds = document.getElementById("toggleTrackIds");
const toggleDangerZones = document.getElementById("toggleDangerZones");
const toggleProximity = document.getElementById("toggleProximity");

const connectionStatus = document.getElementById("connectionStatus");
const cameraMessage = document.getElementById("cameraMessage");
const riskBadge = document.getElementById("riskBadge");

const riskBanner = document.getElementById("riskBanner");
const bannerRiskValue = document.getElementById("bannerRiskValue");
const bannerViolations = document.getElementById("bannerViolations");
const bannerWorkers = document.getElementById("bannerWorkers");
const bannerProximity = document.getElementById("bannerProximity");
const bannerZones = document.getElementById("bannerZones");

const ppeStatus = document.getElementById("ppeStatus");
const fireStatus = document.getElementById("fireStatus");
const smokeStatus = document.getElementById("smokeStatus");
const personStatus = document.getElementById("personStatus");
const machineryStatus = document.getElementById("machineryStatus");
const zoneCountStatus = document.getElementById("zoneCountStatus");

const proximityList = document.getElementById("proximityList");
const zoneList = document.getElementById("zoneList");
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

const audioToggleBtn = document.getElementById("audioToggleBtn");
const audioIconOn = document.getElementById("audioIconOn");
const audioIconOff = document.getElementById("audioIconOff");
const audioStatusText = document.getElementById("audioStatusText");

const toastContainer = document.getElementById("toastContainer");
const lightboxModal = document.getElementById("lightboxModal");
const closeModalBtn = document.getElementById("closeModalBtn");
const modalBody = document.getElementById("modalBody");


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

// FPS & Frame Tracking
let frameCount = 0;
let lastFpsTime = performance.now();
let lastDetections = [];
let lastSummary = null;

// Audio Synthesizer Context
let audioCtx = null;

const PAGE_TITLES = {
    dashboard: ["Live Dashboard", "Real-time site safety surveillance, ByteTrack IDs, danger zones, and proximity monitoring"],
    detection: ["Media Analysis", "Inspect job site photos and process full video streams for safety compliance"],
    alerts: ["Real-Time Alert Feed", "HIGH and CRITICAL site safety alerts requiring officer acknowledgment"],
    history: ["Audit History & Safety Log", "Comprehensive audit log of recorded safety events and risk ratings"],
    statistics: ["Statistics & Site KPIs", "Aggregated safety metrics, incident trends, and site compliance scores"],
    evidence: ["Evidence Gallery", "Captured high-resolution snapshots and video recordings"],
    reports: ["Safety Audit Reports", "Generate and export regulatory site compliance reports in CSV/JSON format"],
};

const RISK_COLORS = {
    SAFE: "#16a34a",
    LOW: "#d97706",
    MEDIUM: "#ea580c",
    HIGH: "#dc2626",
    CRITICAL: "#b91c1c",
};


// ============================================================
// HELPERS & FORMATTERS
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
// AUDIO ALERTS
// ============================================================

function playAlertSound(freq = 880, type = "sine", duration = 0.2) {
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
        gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        console.debug("Audio play failed:", e);
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
        playAlertSound(784, "sine", 0.2);
    }
}


// ============================================================
// LIGHTBOX INSPECTOR MODAL
// ============================================================

function openLightbox(title, subtitle, mediaHtml, detailHtml = "") {
    document.getElementById("modalTitle").textContent = title;
    document.getElementById("modalSubtitle").textContent = subtitle;
    modalBody.innerHTML = `
        <div style="margin-bottom: 20px;">${mediaHtml}</div>
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
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightboxModal.classList.contains("hidden")) {
        closeLightbox();
    }
});


// ============================================================
// UI UPDATE HELPERS (Live HUD & Metrics Cards)
// ============================================================

function updateDetectionUI(summary, detections) {
    if (cameraPlaceholder) {
        cameraPlaceholder.style.display = "none";
    }

    lastDetections = detections || [];
    lastSummary = summary || null;

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

        // Update HUD Banner
        if (riskBanner) {
            riskBanner.className = `risk-banner banner-${risk}`;
            if (bannerRiskValue) bannerRiskValue.textContent = risk;
            if (bannerViolations) bannerViolations.textContent = summary.violation_count || 0;
            if (bannerWorkers) bannerWorkers.textContent = summary.person_count || 0;
            if (bannerProximity) bannerProximity.textContent = (summary.proximity_alerts || []).length;
            if (bannerZones) bannerZones.textContent = (summary.danger_zones || []).length;
        }

        // Camera Footer Risk Badge
        if (riskBadge) {
            const valueEl = riskBadge.querySelector(".risk-value");
            if (valueEl) valueEl.textContent = risk;
            riskBadge.style.color = RISK_COLORS[risk] || "#64748b";
        }

        // PPE Status Card
        if (ppeStatus) {
            if (summary.ppe_violations > 0) {
                ppeStatus.textContent = `${summary.ppe_violations} Violation(s)`;
                ppeStatus.style.color = "#dc2626";
            } else {
                ppeStatus.textContent = "Compliant";
                ppeStatus.style.color = "#16a34a";
            }
        }

        // Fire Card
        if (fireStatus) {
            if (summary.fire_count > 0) {
                fireStatus.textContent = `DETECTED (${summary.fire_count})`;
                fireStatus.style.color = "#dc2626";
            } else {
                fireStatus.textContent = "Clear";
                fireStatus.style.color = "#16a34a";
            }
        }

        // Smoke Card
        if (smokeStatus) {
            if (summary.smoke_count > 0) {
                smokeStatus.textContent = `DETECTED (${summary.smoke_count})`;
                smokeStatus.style.color = "#d97706";
            } else {
                smokeStatus.textContent = "Clear";
                smokeStatus.style.color = "#16a34a";
            }
        }

        // Worker Count Card
        if (personStatus) {
            if (summary.person_count > 0) {
                personStatus.textContent = `${summary.person_count} Tracked Worker(s)`;
                personStatus.style.color = "#2563eb";
            } else {
                personStatus.textContent = "None";
                personStatus.style.color = "#64748b";
            }
        }

        // Machinery Card
        if (machineryStatus) {
            const machineCount = summary.machine_count || 0;
            machineryStatus.textContent = `${machineCount} Unit(s)`;
            machineryStatus.style.color = machineCount > 0 ? "#ea580c" : "#64748b";
        }

        // Danger Zone Card
        if (zoneCountStatus) {
            const zCount = (summary.danger_zones || []).length;
            zoneCountStatus.textContent = `${zCount} Active`;
            zoneCountStatus.style.color = zCount > 0 ? "#dc2626" : "#64748b";
        }

        // Proximity Breaches List
        if (proximityList) {
            const proxAlerts = summary.proximity_alerts || [];
            if (proxAlerts.length === 0) {
                proximityList.innerHTML = '<div class="empty-state"><p>No proximity hazards detected</p></div>';
            } else {
                proximityList.innerHTML = proxAlerts.map(p => `
                    <div class="item-badge danger">
                        <span>⚠️ Worker #${p.person_id ?? 'Unassigned'} ↔ ${escapeHtml(p.machine_type || 'Machinery')}</span>
                        <strong>${Math.round(p.distance_px)}px Distance</strong>
                    </div>
                `).join("");
            }
        }

        // Danger Zones Incursion List
        if (zoneList) {
            const zones = summary.danger_zones || [];
            if (zones.length === 0) {
                zoneList.innerHTML = '<div class="empty-state"><p>No danger zone incursions</p></div>';
            } else {
                zoneList.innerHTML = zones.map(z => `
                    <div class="item-badge danger">
                        <span>🚨 Zone #${z.zone_id || 'Cone Cluster'} (${z.worker_count || 0} worker inside perimeter)</span>
                    </div>
                `).join("");
            }
        }

        // Trigger Toast for High / Critical Events
        if ((risk === "HIGH" || risk === "CRITICAL") && summary.violation_count > 0) {
            showToast(`${risk} Risk Event Detected`, `${summary.violation_count} active safety violation(s) on job site`, risk);
        }
    }

    // Live Detections List & Tracking Table
    if (Array.isArray(detections) && detectionList) {
        if (detections.length === 0) {
            detectionList.innerHTML = `
                <div class="empty-state">
                    <p>No objects detected in current frame</p>
                </div>`;
        } else {
            detectionList.innerHTML = detections.map(d => {
                const trackStr = d.track_id != null ? `[ID:${d.track_id}]` : "";
                return `
                    <div class="detection">
                        <strong>${escapeHtml(d.class)} ${trackStr}</strong> (${(d.confidence * 100).toFixed(1)}%)
                    </div>
                `;
            }).join("");
        }
    }
}


// ============================================================
// CAMERA & WEBSOCKET STREAMING
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

        connectWebSocket();

        startButton.disabled = true;
        stopButton.disabled = false;
        if (captureSnapshotBtn) captureSnapshotBtn.disabled = false;

    } catch (error) {
        console.error("Camera error:", error);
        cameraMessage.textContent = "Could not access job site camera.";
        showToast("Camera Error", "Camera access denied or unavailable", "HIGH");
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
        cameraMessage.textContent = "Live camera stream active. AI Safety Pipeline operational.";
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
        cameraMessage.textContent = "Connection error. Verify backend FastAPI server is running.";
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
        0.55
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

    startButton.disabled = false;
    stopButton.disabled = true;
    if (captureSnapshotBtn) captureSnapshotBtn.disabled = true;

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
    if (machineryStatus) { machineryStatus.textContent = "0 Units"; machineryStatus.style.color = ""; }
    if (zoneCountStatus) { zoneCountStatus.textContent = "0 Active"; zoneCountStatus.style.color = ""; }

    if (riskBadge) {
        const valueEl = riskBadge.querySelector(".risk-value");
        if (valueEl) valueEl.textContent = "—";
        riskBadge.style.color = "";
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// CAPTURE FRAME SNAPSHOT
if (captureSnapshotBtn) {
    captureSnapshotBtn.addEventListener("click", () => {
        if (!canvas) return;
        const dataUrl = canvas.toDataURL("image/jpeg");
        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = `site_snapshot_${Date.now()}.jpg`;
        a.click();
        showToast("Snapshot Captured", "Frame saved to downloads folder", "LOW");
    });
}


// ============================================================
// MEDIA ANALYSIS (Image & Video Uploads)
// ============================================================

// Segmented Media Analysis Tabs
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
const resetImageBtn = document.getElementById("resetImageBtn");

if (imageDropzone && imageFileInput) {
    imageDropzone.addEventListener("click", () => imageFileInput.click());
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

if (resetImageBtn) {
    resetImageBtn.addEventListener("click", () => {
        imageResultContainer.classList.add("hidden");
        imageDropzone.classList.remove("hidden");
        imageFileInput.value = "";
    });
}

async function handleImageUpload(file) {
    if (!file.type.startsWith("image/")) {
        showToast("Invalid File Format", "Please upload a valid image (PNG, JPG, WEBP, BMP)", "HIGH");
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
        const detections = data.result.detections || [];

        imageAnnotatedPreview.src = outputUrl(data.annotated_image_path);

        imageSummaryCard.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <h4 style="font-size: 16px; font-weight: 800; color: var(--slate-900);">Safety Analysis Summary</h4>
                ${riskPill(summary.risk_level)}
            </div>
            <div class="kv-grid" style="margin-bottom: 18px;">
                <div class="kv"><div class="k">Active Violations</div><div class="v">${summary.violation_count || 0}</div></div>
                <div class="kv"><div class="k">PPE Violations</div><div class="v">${summary.ppe_violations || 0}</div></div>
                <div class="kv"><div class="k">Tracked Workers</div><div class="v">${summary.person_count || 0}</div></div>
                <div class="kv"><div class="k">Machinery Detected</div><div class="v">${summary.machine_count || 0}</div></div>
            </div>
            <div style="font-size: 13px; color: var(--slate-700);">
                <strong>Violation Types Detected:</strong> ${escapeHtml(violationSummary(summary.violation_counts))}
            </div>
        `;

        imageProcessingState.classList.add("hidden");
        imageResultContainer.classList.remove("hidden");

        showToast("Image Processed", `Detection completed. Risk Level: ${summary.risk_level || 'SAFE'}`, summary.risk_level === "CRITICAL" ? "CRITICAL" : "HIGH");

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
const resetVideoBtn = document.getElementById("resetVideoBtn");

if (videoDropzone && videoFileInput) {
    videoDropzone.addEventListener("click", () => videoFileInput.click());
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

if (resetVideoBtn) {
    resetVideoBtn.addEventListener("click", () => {
        videoResultContainer.classList.add("hidden");
        videoDropzone.classList.remove("hidden");
        videoFileInput.value = "";
    });
}

async function handleVideoUpload(file) {
    if (!file.type.startsWith("video/")) {
        showToast("Invalid File Format", "Please upload a valid video (MP4, AVI, MOV, MKV)", "HIGH");
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
        videoDownloadBtn.download = "safety_analysis.mp4";

        videoSummaryCard.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <h4 style="font-size: 16px; font-weight: 800; color: var(--slate-900);">Video Safety Analysis Result</h4>
                ${riskPill(maxRisk)}
            </div>
            <div class="kv-grid">
                <div class="kv"><div class="k">Frames Analyzed</div><div class="v">${framesProcessed}</div></div>
                <div class="kv"><div class="k">Peak Risk Rating</div><div class="v">${escapeHtml(maxRisk)}</div></div>
            </div>
        `;

        videoProcessingState.classList.add("hidden");
        videoResultContainer.classList.remove("hidden");

        showToast("Video Processing Complete", `Analyzed ${framesProcessed} video frames successfully`, "HIGH");

    } catch (err) {
        console.error("Video processing error:", err);
        videoProcessingState.classList.add("hidden");
        videoDropzone.classList.remove("hidden");
        showToast("Video Processing Error", err.message || "Failed to analyze video", "HIGH");
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
        if (audioEnabled) playAlertSound(600, "sine", 0.1);
    });
}


// ============================================================
// ALERTS FEED & ACKNOWLEDGEMENT
// ============================================================

async function loadAlertBadge() {
    try {
        const data = await getJson("/api/alerts?limit=1&status=new");
        const n = data.unacknowledged || 0;
        const display = n > 99 ? "99+" : String(n);

        if (alertBadgeCount) alertBadgeCount.textContent = display;
        if (navAlertBadge) navAlertBadge.textContent = display;

        if (n > 0) {
            if (alertBadge) alertBadge.classList.remove("hidden");
            if (navAlertBadge) navAlertBadge.classList.remove("hidden");
        } else {
            if (alertBadge) alertBadge.classList.add("hidden");
            if (navAlertBadge) navAlertBadge.classList.add("hidden");
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
                    <p>No active alerts matching filter criteria</p>
                </div>`;
            return;
        }

        list.innerHTML = alerts.map(a => {
            const acked = a.status === "acknowledged";
            const ackControl = acked
                ? '<span style="font-size:12px; font-weight:700; color:var(--green-600);">✓ Acknowledged</span>'
                : `<button class="btn btn-primary btn-sm" data-ack="${a.id}">Acknowledge</button>`;
            return `
                <div class="alert-item level-${escapeHtml(a.level)}">
                    <div class="alert-main">
                        <div class="alert-title">${escapeHtml(a.title)}</div>
                        <div class="alert-meta">
                            ${riskPill(a.level)} · ${fmtTime(a.ts)} · ${escapeHtml(a.channel)}${a.notified ? " · Telegram Notified" : ""}
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

const ackAllAlertsBtn = document.getElementById("ackAllAlertsBtn");
if (ackAllAlertsBtn) {
    ackAllAlertsBtn.addEventListener("click", async () => {
        try {
            const data = await getJson("/api/alerts?status=new&limit=100");
            const alerts = data.alerts || [];
            await Promise.all(alerts.map(a => fetch(apiUrl(`/api/alerts/${a.id}/ack`), { method: "POST" })));
            showToast("Alerts Acknowledged", "All pending alerts have been acknowledged", "LOW");
            await Promise.all([loadAlerts(), loadAlertBadge()]);
        } catch (err) {
            console.error("Ack all failed:", err);
        }
    });
}


// ============================================================
// AUDIT HISTORY PANEL
// ============================================================

let rawHistoryEvents = [];

async function loadHistory() {
    const body = document.getElementById("historyBody");
    const risk = document.getElementById("historyRiskFilter").value;
    const source = document.getElementById("historySourceFilter").value;
    const search = (document.getElementById("historySearchInput").value || "").toLowerCase().trim();

    const params = new URLSearchParams({ limit: "250" });
    if (risk) params.set("risk", risk);
    if (source) params.set("source", source);

    try {
        const data = await getJson(`/api/events?${params.toString()}`);
        rawHistoryEvents = data.events || [];

        let filtered = rawHistoryEvents;
        if (search) {
            filtered = filtered.filter(e => {
                const str = `${e.source} ${e.risk_level} ${JSON.stringify(e.violation_counts || {})}`.toLowerCase();
                return str.includes(search);
            });
        }

        if (filtered.length === 0) {
            body.innerHTML = '<tr><td colspan="8"><div class="empty-state"><p>No recorded safety events matching filters</p></div></td></tr>';
            return;
        }

        body.innerHTML = filtered.map(e => {
            const img = e.evidence_image ? `<a href="#" class="thumb-link" data-img="${evidenceUrl(e.evidence_image)}" data-id="${e.id}">Snapshot</a>` : "";
            const clip = e.evidence_clip ? `<a href="#" class="thumb-link" data-clip="${evidenceUrl(e.evidence_clip)}" data-id="${e.id}">Video Clip</a>` : "";
            const evidence = [img, clip].filter(Boolean).join(" · ") || "—";
            return `
                <tr>
                    <td>${fmtTime(e.ts)}</td>
                    <td><strong>${escapeHtml(e.source)}</strong></td>
                    <td>${riskPill(e.risk_level)}</td>
                    <td>${escapeHtml(violationSummary(e.violation_counts))}</td>
                    <td>${e.persons ?? 0}</td>
                    <td>${e.fire ?? 0}</td>
                    <td>${e.smoke ?? 0}</td>
                    <td>${evidence}</td>
                </tr>
            `;
        }).join("");

        body.querySelectorAll("a[data-img], a[data-clip]").forEach(link => {
            link.addEventListener("click", (evt) => {
                evt.preventDefault();
                const img = link.dataset.img;
                const clip = link.dataset.clip;
                const id = link.dataset.id;
                let mediaHtml = "";
                if (clip) {
                    mediaHtml = `<video class="modal-media" src="${clip}" controls autoplay></video>`;
                } else if (img) {
                    mediaHtml = `<img class="modal-media" src="${img}" alt="Event #${id}">`;
                }
                openLightbox(`Event #${id} Evidence Detail`, `Captured snapshot/recording snapshot`, mediaHtml);
            });
        });

    } catch (err) {
        body.innerHTML = `<tr><td colspan="8"><div class="empty-state"><p>Could not load history (${escapeHtml(err.message)})</p></div></td></tr>`;
    }
}

// Export History Table as CSV
const exportHistoryCsvBtn = document.getElementById("exportHistoryCsvBtn");
if (exportHistoryCsvBtn) {
    exportHistoryCsvBtn.addEventListener("click", () => {
        if (!rawHistoryEvents.length) {
            showToast("Export Failed", "No event history data available to export", "HIGH");
            return;
        }
        const headers = ["ID", "Timestamp", "Source", "RiskLevel", "Persons", "Fire", "Smoke", "Violations"];
        const rows = rawHistoryEvents.map(e => [
            e.id,
            fmtTime(e.ts),
            e.source,
            e.risk_level,
            e.persons || 0,
            e.fire || 0,
            e.smoke || 0,
            `"${violationSummary(e.violation_counts)}"`
        ]);
        const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `safety_history_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
}


// ============================================================
// STATISTICS & KPIS
// ============================================================

function barChart(container, entries, colorFn) {
    const el = document.getElementById(container);
    const items = entries.filter(([, v]) => v > 0);

    if (items.length === 0) {
        el.innerHTML = '<div class="empty-state"><p>No data available for this range</p></div>';
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
    const kpiVal = document.getElementById("kpiScoreValue");

    try {
        const r = await getJson(`/api/statistics?range=${encodeURIComponent(range)}`);
        const t = r.totals || {};

        // Calculate Site Compliance Score KPI
        const totalEv = t.events || 0;
        const criticalHighEv = ((r.by_risk || {}).HIGH || 0) + ((r.by_risk || {}).CRITICAL || 0);
        const score = totalEv > 0 ? Math.max(0, (100 - (criticalHighEv / totalEv) * 100)).toFixed(1) : "100.0";
        if (kpiVal) kpiVal.textContent = `${score}%`;

        cards.innerHTML = [
            ["Recorded Safety Events", t.events ?? 0],
            ["Generated System Alerts", t.alerts ?? 0],
            ["Unacknowledged Alerts", t.alerts_unacknowledged ?? 0],
            ["Total Frames Processed", t.frames_processed ?? 0],
            ["Captured Snapshots", t.events_with_image ?? 0],
            ["Video Recordings Saved", t.events_with_clip ?? 0],
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
        cards.innerHTML = `<div class="empty-state"><p>Could not load statistics (${escapeHtml(err.message)})</p></div>`;
    }
}


// ============================================================
// EVIDENCE GALLERY
// ============================================================

async function loadEvidence() {
    const gallery = document.getElementById("evidenceGallery");
    const typeFilter = document.getElementById("evidenceFilterType").value;

    try {
        const data = await getJson("/api/evidence?limit=60");
        let items = data.evidence || [];

        if (typeFilter === "image") items = items.filter(it => it.image);
        else if (typeFilter === "clip") items = items.filter(it => it.clip);

        if (items.length === 0) {
            gallery.innerHTML = `
                <div class="empty-state" style="grid-column: 1/-1">
                    <p>No evidence items matching filter</p>
                    <span>High and critical risk safety violations automatically record annotated evidence</span>
                </div>`;
            return;
        }

        gallery.innerHTML = items.map(it => {
            let media;
            if (it.clip) {
                media = `<video class="evidence-media" src="${evidenceUrl(it.clip)}" muted></video>`;
            } else if (it.image) {
                media = `<img class="evidence-media" src="${evidenceUrl(it.image)}" alt="event #${it.event_id}" loading="lazy">`;
            } else {
                media = "";
            }
            return `
                <div class="evidence-card" data-id="${it.event_id}" data-img="${evidenceUrl(it.image)}" data-clip="${evidenceUrl(it.clip)}">
                    ${media}
                    <div class="evidence-body">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            ${riskPill(it.risk_level)}
                            <strong>Incident #${it.event_id}</strong>
                        </div>
                        <div>Source: ${escapeHtml(it.source)} · ${fmtTime(it.ts)}</div>
                        <div style="margin-top: 6px; font-weight: 700; color: var(--slate-800);">${escapeHtml(violationSummary(it.violation_counts))}</div>
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
                openLightbox(`Evidence Inspector - Incident #${id}`, `High-resolution safety recording detail`, mediaHtml);
            });
        });

    } catch (err) {
        gallery.innerHTML = `<div class="empty-state"><p>Could not load evidence gallery (${escapeHtml(err.message)})</p></div>`;
    }
}


// ============================================================
// REPORTS GENERATOR
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
            <div style="margin-bottom: 24px;">
                <h3 style="font-size: 16px; font-weight: 800; color: var(--slate-900); margin-bottom: 14px;">Site Compliance Totals (${escapeHtml(r.range)})</h3>
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
                <h3 style="font-size: 16px; font-weight: 800; color: var(--slate-900); margin-bottom: 14px;">Events by Input Source</h3>
                <div class="kv-grid">
                    ${Object.entries(r.by_source || {}).map(([k, v]) => kv(k, v)).join("") || '<div class="empty-state"><p>No source data available</p></div>'}
                </div>
            </div>

            <div>
                <h3 style="font-size: 16px; font-weight: 800; color: var(--slate-900); margin-bottom: 14px;">Most Severe Site Incidents</h3>
                <div class="table-container" style="border: 1px solid var(--slate-200); border-radius: var(--radius-lg);">
                    <table class="data-table">
                        <thead>
                            <tr><th>Timestamp</th><th>Source</th><th>Risk Level</th><th>Violations</th><th>Detail</th></tr>
                        </thead>
                        <tbody>${topRows || '<tr><td colspan="5"><div class="empty-state"><p>No severe incidents recorded</p></div></td></tr>'}</tbody>
                    </table>
                </div>
            </div>
        `;

    } catch (err) {
        summary.innerHTML = `<div class="empty-state"><p>Could not generate report (${escapeHtml(err.message)})</p></div>`;
    }
}


// ============================================================
// POLLING & INITIALIZATION
// ============================================================

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => {
        loadAlertBadge();
        if (activeTab === "alerts") loadAlerts();
        else if (activeTab === "history") loadHistory();
        else if (activeTab === "statistics") loadStatistics();
    }, 8000);
}


// EVENT LISTENERS
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
document.getElementById("historySearchInput").addEventListener("input", loadHistory);

document.getElementById("refreshStats").addEventListener("click", loadStatistics);
document.getElementById("statsRange").addEventListener("change", loadStatistics);

document.getElementById("refreshEvidence").addEventListener("click", loadEvidence);
document.getElementById("evidenceFilterType").addEventListener("change", loadEvidence);

document.getElementById("generateReport").addEventListener("click", loadReport);
document.getElementById("reportRange").addEventListener("change", loadReport);


// INITIAL STARTUP
loadAlertBadge();
startPolling();
