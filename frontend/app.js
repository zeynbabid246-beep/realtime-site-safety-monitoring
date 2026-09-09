// ============================================================
// CONFIGURATION
// ============================================================

function getWebSocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    let host = window.location.hostname;
    if (!host || host === "") {
        host = "127.0.0.1";
    }
    // Connect to backend server on port 8000
    return `${protocol}//${host}:8000/ws/camera`;
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

const ppeStatus = document.getElementById("ppeStatus");
const fireStatus = document.getElementById("fireStatus");
const smokeStatus = document.getElementById("smokeStatus");
const personStatus = document.getElementById("personStatus");
const detectionList = document.getElementById("detectionList");


// ============================================================
// VARIABLES
// ============================================================

let cameraStream = null;
let socket = null;
let sendingFrames = false;


// ============================================================
// UI UPDATE HELPERS
// ============================================================

function updateDetectionUI(summary, detections) {
    if (summary) {
        if (summary.ppe_violations > 0) {
            ppeStatus.textContent = `⚠️ ${summary.ppe_violations} Violation(s)`;
            ppeStatus.style.color = "#dc2626";
        } else {
            ppeStatus.textContent = "✅ Compliant";
            ppeStatus.style.color = "#16a34a";
        }

        if (summary.fire_count > 0) {
            fireStatus.textContent = `🔥 DETECTED (${summary.fire_count})`;
            fireStatus.style.color = "#dc2626";
        } else {
            fireStatus.textContent = "Clear";
            fireStatus.style.color = "#16a34a";
        }

        if (summary.smoke_count > 0) {
            smokeStatus.textContent = `💨 DETECTED (${summary.smoke_count})`;
            smokeStatus.style.color = "#d97706";
        } else {
            smokeStatus.textContent = "Clear";
            smokeStatus.style.color = "#16a34a";
        }

        if (summary.person_count > 0) {
            personStatus.textContent = `👷 ${summary.person_count} Present`;
            personStatus.style.color = "#2563eb";
        } else {
            personStatus.textContent = "None";
            personStatus.style.color = "#6b7280";
        }
    }

    if (Array.isArray(detections)) {
        if (detections.length === 0) {
            detectionList.innerHTML = '<p class="empty">No detections in current frame.</p>';
        } else {
            detectionList.innerHTML = detections.map(d => `
                <div class="detection">
                    <strong>${d.class}</strong> (${(d.confidence * 100).toFixed(1)}%)
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

        // Get webcam stream
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480
            },
            audio: false
        });

        video.srcObject = cameraStream;
        await video.play();

        // Connect WebSocket
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

function connectWebSocket() {

    const wsUrl = getWebSocketUrl();
    console.log("Connecting to WebSocket URL:", wsUrl);

    socket = new WebSocket(wsUrl);

    // Connected
    socket.onopen = () => {

        console.log("WebSocket connected successfully.");

        connectionStatus.textContent = "● Connected";
        connectionStatus.classList.remove("disconnected");
        connectionStatus.classList.add("connected");

        cameraMessage.textContent = "Camera is running. AI detection is active.";

        sendingFrames = true;
        sendFrame();
    };

    // Receive message
    socket.onmessage = (event) => {

        if (typeof event.data === "string") {
            try {
                const payload = JSON.parse(event.data);

                if (payload.image) {
                    const image = new Image();
                    image.onload = () => {
                        canvas.width = image.width;
                        canvas.height = image.height;
                        ctx.drawImage(image, 0, 0);
                    };
                    image.src = payload.image;
                }

                if (payload.summary || payload.detections) {
                    updateDetectionUI(payload.summary, payload.detections);
                }

            } catch (err) {
                console.error("Error parsing WebSocket JSON message:", err);
            }
        } else if (event.data instanceof Blob) {

            const image = new Image();
            image.onload = () => {
                canvas.width = image.width;
                canvas.height = image.height;
                ctx.drawImage(image, 0, 0);
                URL.revokeObjectURL(image.src);
            };

            image.src = URL.createObjectURL(event.data);
        }
    };

    // Error
    socket.onerror = (error) => {

        console.error("WebSocket error details:", error);

        cameraMessage.textContent = "WebSocket connection error! Please make sure the FastAPI server is running on http://127.0.0.1:8000.";

        connectionStatus.textContent = "● Error";
        connectionStatus.classList.remove("connected");
        connectionStatus.classList.add("disconnected");
    };

    // Closed
    socket.onclose = (event) => {

        console.log("WebSocket disconnected.", event);

        sendingFrames = false;

        connectionStatus.textContent = "● Disconnected";
        connectionStatus.classList.remove("connected");
        connectionStatus.classList.add("disconnected");
    };
}


// ============================================================
// SEND CAMERA FRAME
// ============================================================

function sendFrame() {

    if (!sendingFrames) {
        return;
    }

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

    // Capture frame on temporary canvas
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 640;
    tempCanvas.height = 480;

    const tempContext = tempCanvas.getContext("2d");
    tempContext.drawImage(video, 0, 0, 640, 480);

    // Convert frame to JPEG blob
    tempCanvas.toBlob(
        (blob) => {

            if (blob && socket && socket.readyState === WebSocket.OPEN) {
                socket.send(blob);
            }

            // Schedule next frame (~10 FPS)
            if (sendingFrames) {
                setTimeout(sendFrame, 100);
            }
        },
        "image/jpeg",
        0.7
    );
}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

    console.log("Stopping camera...");

    sendingFrames = false;

    // Stop webcam tracks
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    // Close WebSocket
    if (socket) {
        socket.close();
        socket = null;
    }

    // Clear video element
    video.srcObject = null;

    // Reset UI
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

    detectionList.innerHTML = '<p class="empty">No detections yet.</p>';

    ctx.clearRect(0, 0, canvas.width, canvas.height);
}


// ============================================================
// BUTTON EVENTS
// ============================================================

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
stopButton.addEventListener("click", stopCamera);