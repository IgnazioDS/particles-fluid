/**
 * MediaPipe hand tracking → ForceSource[] conversion.
 *
 * Uses @mediapipe/tasks-vision HandLandmarker to detect up to 2 hands
 * from a webcam video element, classify gestures, and produce force
 * sources compatible with the existing SPH solver interface.
 */

import { CFG } from "./config";
import type { ForceSource } from "./sph";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface TrackedHand {
  palm: { x: number; y: number };
  gesture: "open" | "fist" | "point" | "pinch" | "unknown";
  confidence: number;
  landmarks: Array<{ x: number; y: number; z: number }>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lazy-loaded MediaPipe (heavy WASM, only import when needed)
// ─────────────────────────────────────────────────────────────────────────────

let HandLandmarkerClass: any = null;
let FilesetResolverClass: any = null;

async function loadMediaPipe() {
  if (HandLandmarkerClass) return;
  const mod = await import("@mediapipe/tasks-vision");
  HandLandmarkerClass = mod.HandLandmarker;
  FilesetResolverClass = mod.FilesetResolver;
}

// ─────────────────────────────────────────────────────────────────────────────
// HandTracker
// ─────────────────────────────────────────────────────────────────────────────

export class HandTracker {
  private landmarker: any = null;
  private video: HTMLVideoElement | null = null;
  hands: TrackedHand[] = [];
  active = false;
  loading = false;
  private frameSkip = 0;

  async init(video: HTMLVideoElement): Promise<void> {
    this.loading = true;
    try {
      await loadMediaPipe();
      const vision = await FilesetResolverClass.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
      );
      this.landmarker = await HandLandmarkerClass.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU",
        },
        numHands: 2,
        runningMode: "VIDEO",
        minHandDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      this.video = video;
      this.active = true;
    } catch (e) {
      this.active = false;
      throw e;
    } finally {
      this.loading = false;
    }
  }

  /** Call each frame. Internally skips every other frame for perf. */
  detect(): TrackedHand[] {
    if (!this.landmarker || !this.video || !this.active) return [];
    if (!this.video.videoWidth) return [];

    // Run detection at 30fps (skip every other frame)
    this.frameSkip++;
    if (this.frameSkip % 2 !== 0) return this.hands;

    const result = this.landmarker.detectForVideo(this.video, performance.now());
    if (!result?.landmarks?.length) {
      this.hands = [];
      return this.hands;
    }

    this.hands = result.landmarks.map((lm: any[], idx: number) => {
      const landmarks = lm.map((p: any) => ({ x: p.x, y: p.y, z: p.z }));
      const palm = computePalm(landmarks);
      const gesture = classifyGesture(landmarks);
      const confidence = result.handedness?.[idx]?.[0]?.score ?? 0.5;
      return { palm, gesture, confidence, landmarks };
    });

    return this.hands;
  }

  /** Convert detected hands to SPH force sources. */
  toForceSources(W: number, H: number): ForceSource[] {
    const sources: ForceSource[] = [];

    for (const hand of this.hands) {
      // Mirror X (webcam is mirrored)
      const x = (1 - hand.palm.x) * W;
      const y = hand.palm.y * H;

      switch (hand.gesture) {
        case "open":
          sources.push({ x, y, strength: CFG.attractStr, radius: CFG.handRadius });
          break;
        case "fist":
          sources.push({ x, y, strength: CFG.repelStr, radius: CFG.handRadius * 1.35 });
          break;
        case "point": {
          // Directional push: wrist → index tip
          const wrist = hand.landmarks[0];
          const tip = hand.landmarks[8];
          const dx = tip.x - wrist.x;
          const dy = tip.y - wrist.y;
          const mag = Math.sqrt(dx * dx + dy * dy) || 1;
          const tipX = (1 - tip.x) * W;
          const tipY = tip.y * H;
          sources.push({ x: tipX, y: tipY, strength: CFG.attractStr * 0.7, radius: CFG.handRadius * 0.7 });
          break;
        }
        case "pinch":
          sources.push({
            x, y,
            strength: CFG.attractStr * 0.25,
            radius: CFG.handRadius * 1.2,
            vortex: CFG.vortexStr,
          });
          break;
        default:
          // Gentle attract for unrecognized gestures
          sources.push({ x, y, strength: CFG.attractStr * 0.2, radius: CFG.handRadius * 0.6 });
      }
    }

    return sources;
  }

  dispose(): void {
    this.active = false;
    this.landmarker?.close();
    this.landmarker = null;
    this.video = null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gesture helpers
// ─────────────────────────────────────────────────────────────────────────────

function computePalm(lm: Array<{ x: number; y: number; z: number }>): { x: number; y: number } {
  // Average of wrist(0) + MCP joints (5, 9, 13, 17)
  const indices = [0, 5, 9, 13, 17];
  let sx = 0, sy = 0;
  for (const i of indices) { sx += lm[i].x; sy += lm[i].y; }
  return { x: sx / 5, y: sy / 5 };
}

function classifyGesture(
  lm: Array<{ x: number; y: number; z: number }>,
): "open" | "fist" | "point" | "pinch" | "unknown" {
  // Finger extension: tip.y < PIP.y means extended (lower y = higher in image)
  const idxExt = lm[8].y < lm[6].y;
  const midExt = lm[12].y < lm[10].y;
  const ringExt = lm[16].y < lm[14].y;
  const pinkExt = lm[20].y < lm[18].y;
  const nExt = [idxExt, midExt, ringExt, pinkExt].filter(Boolean).length;

  // Pinch: thumb tip ↔ index tip close
  const pinchD = Math.sqrt(
    (lm[4].x - lm[8].x) ** 2 + (lm[4].y - lm[8].y) ** 2,
  );
  if (pinchD < 0.05) return "pinch";

  if (nExt >= 3) return "open";
  if (nExt === 0) return "fist";
  if (idxExt && !midExt && !ringExt && !pinkExt) return "point";
  return "unknown";
}
