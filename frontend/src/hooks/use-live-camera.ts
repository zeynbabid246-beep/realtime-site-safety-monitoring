import { useCallback, useEffect, useRef, useState } from "react";
import { getWebSocketUrl } from "@/lib/api";
import type { SafetyPayload } from "@/lib/types";

export type ConnectionState = "idle" | "connecting" | "connected" | "error" | "offline";

interface UseLiveCameraOptions {
  /** JPEG quality for frames sent to the backend (0..1). */
  quality?: number;
  /** Capture resolution. */
  width?: number;
  height?: number;
}

interface UseLiveCameraResult {
  /** Attach to a <video> element to render the raw webcam feed. */
  attachVideo: (element: HTMLVideoElement | null) => void;
  /** Attach to an <img> element to render backend-annotated frames. */
  attachCanvasImage: (element: HTMLImageElement | null) => void;
  connection: ConnectionState;
  start: () => Promise<void>;
  stop: () => void;
  latestPayload: SafetyPayload | null;
  fps: number;
  error: string | null;
}

/**
 * Live camera loop:
 *   getUserMedia -> <video> -> canvas.capture -> JPEG blob
 *     -> WebSocket /safety/ws/camera (binary frame)
 *     <- binary annotated JPEG + JSON SafetyPayload text frames.
 *
 * Backpressure mirrors the original dashboard: exactly one frame is in
 * flight; the next frame is sent after the backend replies with
 * {frame_done: true} (+ a small delay).
 */
export function useLiveCamera(options: UseLiveCameraOptions = {}): UseLiveCameraResult {
  const { quality = 0.55, width = 640, height = 480 } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sendingRef = useRef(false);
  const runningRef = useRef(false);
  const frameCountRef = useRef(0);
  const fpsWindowRef = useRef<number>(0);

  const [connection, setConnection] = useState<ConnectionState>("idle");
  const [latestPayload, setLatestPayload] = useState<SafetyPayload | null>(null);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const sendNextFrame = useCallback(() => {
    const socket = socketRef.current;
    const video = videoRef.current;
    if (!runningRef.current || !socket || socket.readyState !== WebSocket.OPEN) return;
    if (sendingRef.current) return;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      window.setTimeout(sendNextFrame, 120);
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      (blob) => {
        if (blob && socketRef.current?.readyState === WebSocket.OPEN && runningRef.current) {
          sendingRef.current = true;
          socketRef.current.send(blob);
        }
      },
      "image/jpeg",
      quality,
    );
  }, [height, quality, width]);

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      if (event.data instanceof Blob) {
        // Annotated frame -> render into the attached <img>.
        const image = imageRef.current;
        if (image) {
          const url = URL.createObjectURL(event.data);
          image.onload = () => URL.revokeObjectURL(url);
          image.src = url;
        }
        frameCountRef.current += 1;
        const now = performance.now();
        if (now - fpsWindowRef.current >= 1000) {
          setFps(Math.round((frameCountRef.current * 1000) / (now - fpsWindowRef.current)));
          frameCountRef.current = 0;
          fpsWindowRef.current = now;
        }
        return;
      }

      if (typeof event.data === "string") {
        try {
          const payload = JSON.parse(event.data) as SafetyPayload & { frame_done?: boolean };
          if (payload.summary || payload.identities || payload.risk_level) {
            setLatestPayload(payload);
          }
          if (payload.frame_done) {
            sendingRef.current = false;
            window.setTimeout(sendNextFrame, 30);
          }
        } catch {
          // Malformed JSON - ignore and continue.
          sendingRef.current = false;
        }
      }
    },
    [sendNextFrame],
  );

  const stop = useCallback(() => {
    runningRef.current = false;
    sendingRef.current = false;
    socketRef.current?.close();
    socketRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setConnection("idle");
    setFps(0);
  }, []);

  const start = useCallback(async () => {
    if (runningRef.current) return;
    setError(null);
    setConnection("connecting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;

      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play().catch(() => undefined);
      }

      const socket = new WebSocket(getWebSocketUrl());
      socket.binaryType = "blob";
      socketRef.current = socket;
      runningRef.current = true;

      socket.onopen = () => {
        setConnection("connected");
        sendingRef.current = false;
        fpsWindowRef.current = performance.now();
        sendNextFrame();
      };
      socket.onmessage = handleMessage;
      socket.onerror = () => {
        setConnection("error");
        setError("WebSocket error — ensure the backend server is running on port 8000.");
      };
      socket.onclose = () => {
        if (runningRef.current) {
          setConnection("offline");
          setError("Connection closed by the backend.");
        }
        sendingRef.current = false;
      };
    } catch (err) {
      runningRef.current = false;
      setConnection("error");
      setError(
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Camera permission denied. Grant access and try again."
          : "Could not access the camera. Check that no other app is using it.",
      );
    }
  }, [handleMessage, sendNextFrame]);

  // Cleanup on unmount.
  useEffect(() => () => stop(), [stop]);

  const attachVideo = useCallback((element: HTMLVideoElement | null) => {
    videoRef.current = element;
    if (element && streamRef.current && element.srcObject !== streamRef.current) {
      element.srcObject = streamRef.current;
      void element.play().catch(() => undefined);
    }
  }, []);

  const attachCanvasImage = useCallback((element: HTMLImageElement | null) => {
    imageRef.current = element;
  }, []);

  return { attachVideo, attachCanvasImage, connection, start, stop, latestPayload, fps, error };
}
