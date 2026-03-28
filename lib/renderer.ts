/**
 * Canvas 2D renderer — dramatically upgraded.
 *
 * Visual pipeline:
 *  1. Fade trail canvas (motion trails)
 *  2. Draw particles in 3 layers via color-bucketed batching:
 *       bloom (r=14, alpha=0.03) → glow (r=7, alpha=0.08) → core (r=2.5, alpha=0.85)
 *  3. Compose main canvas: bg + blurred glow + blurred bloom + sharp trails
 *  4. Cinematic vignette (radial gradient edge darkening)
 *  5. Cursor indicator (mode-aware)
 *  6. Formation overlay (ghost lattice + formula text)
 *
 * Color-bucket batching: particles grouped into 32 temperature buckets,
 * each drawn as a single Path2D → 96 draw calls instead of 3×n.
 */

import { CFG } from "./config";
import type { Theme } from "./themes";

const TAU = Math.PI * 2;
const NUM_BUCKETS = 32;

export interface FormationOverlay {
  shapeName: string;
  formulaTex: string;
  formulaPos: [number, number];
  alpha: number;
  latticePts: Float32Array;
  nPts: number;
  outlinePts: Float32Array;
  outlineN: number;
  formState: "forming" | "holding" | "dissolving";
  stateProgress: number;
}

export class Renderer {
  private W: number;
  private H: number;
  private mainCtx: CanvasRenderingContext2D;
  private trailCanvas: OffscreenCanvas;
  private trailCtx: OffscreenCanvasRenderingContext2D;
  private glowCanvas: OffscreenCanvas;
  private glowCtx: OffscreenCanvasRenderingContext2D;
  private vignetteGrad: CanvasGradient | null = null;

  // Bucket storage (pre-allocated)
  private bucketIdx: Int32Array[]; // particle indices per bucket
  private bucketN: Int32Array;     // count per bucket

  constructor(canvas: HTMLCanvasElement, w: number, h: number) {
    this.W = w;
    this.H = h;
    canvas.width = w;
    canvas.height = h;
    this.mainCtx = canvas.getContext("2d", { alpha: false })!;

    this.trailCanvas = new OffscreenCanvas(w, h);
    this.trailCtx = this.trailCanvas.getContext("2d", { alpha: false })!;
    this.glowCanvas = new OffscreenCanvas(w, h);
    this.glowCtx = this.glowCanvas.getContext("2d")!;

    this.bucketIdx = Array.from({ length: NUM_BUCKETS }, () => new Int32Array(4096));
    this.bucketN = new Int32Array(NUM_BUCKETS);

    this.rebuildVignette();
    this.clearTrail("rgb(6,4,10)");
  }

  resize(w: number, h: number): void {
    this.W = w;
    this.H = h;
    this.mainCtx.canvas.width = w;
    this.mainCtx.canvas.height = h;
    this.trailCanvas = new OffscreenCanvas(w, h);
    this.trailCtx = this.trailCanvas.getContext("2d", { alpha: false })!;
    this.glowCanvas = new OffscreenCanvas(w, h);
    this.glowCtx = this.glowCanvas.getContext("2d")!;
    this.rebuildVignette();
    this.clearTrail("rgb(6,4,10)");
  }

  reset(bg: string): void {
    this.clearTrail(bg);
  }

  // ── Main render ──────────────────────────────────────────────────────────

  render(
    px: Float32Array,
    py: Float32Array,
    temp: Float32Array,
    n: number,
    theme: Theme,
    overlay: FormationOverlay | null,
    cursor: { x: number; y: number; active: boolean; mode: string },
    audioEnergy: number,
  ): void {
    const { W, H } = this;
    const tCtx = this.trailCtx;
    const ctx = this.mainCtx;

    // ── 1. Fade trails ─────────────────────────────────────────────────────
    tCtx.globalCompositeOperation = "source-over";
    tCtx.globalAlpha = 1;
    const [br, bg, bb] = theme.bgRGB;
    tCtx.fillStyle = `rgba(${br},${bg},${bb},${CFG.trailFade})`;
    tCtx.fillRect(0, 0, W, H);

    // ── 2. Bucket particles by temperature ─────────────────────────────────
    this.bucketN.fill(0);
    for (let i = 0; i < n; i++) {
      const bi = Math.min(NUM_BUCKETS - 1, Math.max(0, (temp[i] * NUM_BUCKETS) | 0));
      const idx = this.bucketN[bi]++;
      if (idx < this.bucketIdx[bi].length) {
        this.bucketIdx[bi][idx] = i;
      }
    }

    // ── 3. Draw particle layers (additive blend) ───────────────────────────
    tCtx.globalCompositeOperation = "lighter";
    const aScale = 1 + audioEnergy * 0.8;
    const bR = CFG.bloomRadius * aScale;
    const gR = CFG.glowRadius * aScale;
    const cR = CFG.particleRadius * (1 + audioEnergy * 0.3);

    // Pass 1: bloom halos
    tCtx.globalAlpha = CFG.bloomAlpha;
    this.drawBucketedCircles(tCtx, px, py, bR, theme);

    // Pass 2: glow rings
    tCtx.globalAlpha = CFG.glowAlpha;
    this.drawBucketedCircles(tCtx, px, py, gR, theme);

    // Pass 3: bright cores
    tCtx.globalAlpha = CFG.coreAlpha;
    this.drawBucketedCircles(tCtx, px, py, cR, theme);

    // ── 4. Compose on main canvas ──────────────────────────────────────────
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, W, H);

    ctx.globalCompositeOperation = "lighter";

    // Post-process glow
    const gCtx = this.glowCtx;
    gCtx.clearRect(0, 0, W, H);
    gCtx.filter = `blur(${CFG.postGlowSigma}px)`;
    gCtx.drawImage(this.trailCanvas, 0, 0);
    gCtx.filter = "none";
    ctx.globalAlpha = CFG.postGlowAlpha;
    ctx.drawImage(this.glowCanvas, 0, 0);

    // Post-process bloom
    gCtx.clearRect(0, 0, W, H);
    gCtx.filter = `blur(${CFG.postBloomSigma}px)`;
    gCtx.drawImage(this.trailCanvas, 0, 0);
    gCtx.filter = "none";
    ctx.globalAlpha = CFG.postBloomAlpha;
    ctx.drawImage(this.glowCanvas, 0, 0);

    // Sharp layer
    ctx.globalAlpha = 1;
    ctx.drawImage(this.trailCanvas, 0, 0);

    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;

    // ── 5. Vignette ────────────────────────────────────────────────────────
    if (this.vignetteGrad) {
      ctx.fillStyle = this.vignetteGrad;
      ctx.fillRect(0, 0, W, H);
    }

    // ── 6. Cursor indicator ────────────────────────────────────────────────
    if (cursor.x >= 0 && cursor.y >= 0) {
      this.drawCursor(ctx, cursor, theme);
    }

    // ── 7. Formation overlay ───────────────────────────────────────────────
    if (overlay && overlay.alpha > 0.01) {
      this.drawFormation(ctx, overlay, theme);
    }
  }

  // ── Bucketed circle drawing ──────────────────────────────────────────────

  private drawBucketedCircles(
    ctx: OffscreenCanvasRenderingContext2D,
    px: Float32Array,
    py: Float32Array,
    radius: number,
    theme: Theme,
  ): void {
    for (let b = 0; b < NUM_BUCKETS; b++) {
      const count = this.bucketN[b];
      if (count === 0) continue;
      const ci = Math.round((b / (NUM_BUCKETS - 1)) * 255);
      ctx.fillStyle = theme.lut[ci];
      ctx.beginPath();
      for (let j = 0; j < count; j++) {
        const i = this.bucketIdx[b][j];
        ctx.moveTo(px[i] + radius, py[i]);
        ctx.arc(px[i], py[i], radius, 0, TAU);
      }
      ctx.fill();
    }
  }

  // ── Cursor ───────────────────────────────────────────────────────────────

  private drawCursor(
    ctx: CanvasRenderingContext2D,
    cursor: { x: number; y: number; active: boolean; mode: string },
    theme: Theme,
  ): void {
    const { x, y, active, mode } = cursor;
    ctx.save();

    const colors: Record<string, string> = {
      attract: "rgba(80, 220, 100, 0.5)",
      repel: "rgba(220, 60, 80, 0.5)",
      vortex: "rgba(100, 140, 255, 0.5)",
      gravity: "rgba(255, 200, 60, 0.5)",
    };
    const color = colors[mode] || colors.attract;
    const rad = mode === "gravity" ? CFG.handRadius * 2 : CFG.handRadius;

    ctx.globalAlpha = active ? 0.35 : 0.12;
    ctx.strokeStyle = color;
    ctx.lineWidth = active ? 2 : 1;
    ctx.beginPath();
    ctx.arc(x, y, rad, 0, TAU);
    ctx.stroke();

    // Vortex: draw rotation indicator
    if (mode === "vortex" && active) {
      ctx.globalAlpha = 0.2;
      const t = performance.now() / 600;
      for (let a = 0; a < 3; a++) {
        const angle = t + (TAU * a) / 3;
        const r2 = rad * 0.7;
        ctx.beginPath();
        ctx.arc(x + Math.cos(angle) * r2, y + Math.sin(angle) * r2, 3, 0, TAU);
        ctx.fillStyle = color;
        ctx.fill();
      }
    }

    // Centre dot
    ctx.globalAlpha = active ? 0.6 : 0.2;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, active ? 4 : 2, 0, TAU);
    ctx.fill();

    ctx.restore();
  }

  // ── Formation overlay ────────────────────────────────────────────────────

  private drawFormation(
    ctx: CanvasRenderingContext2D,
    ov: FormationOverlay,
    theme: Theme,
  ): void {
    const alpha = ov.alpha;

    // ── Shape outline (neon glow style) ────────────────────────────────────
    if (ov.outlineN > 1) {
      // Outline appears slightly ahead of particles (anticipation)
      const outlineAlpha = ov.formState === "forming"
        ? Math.min(ov.stateProgress / 0.3, 1) * alpha
        : alpha;

      ctx.save();

      // Glow pass (wide, dim, blurred)
      ctx.globalAlpha = outlineAlpha * 0.25;
      ctx.strokeStyle = theme.accent;
      ctx.lineWidth = 5;
      ctx.filter = "blur(4px)";
      ctx.beginPath();
      ctx.moveTo(ov.outlinePts[0], ov.outlinePts[1]);
      for (let i = 1; i < ov.outlineN; i++) {
        ctx.lineTo(ov.outlinePts[i * 2], ov.outlinePts[i * 2 + 1]);
      }
      ctx.stroke();
      ctx.filter = "none";

      // Crisp pass (thin, bright)
      ctx.globalAlpha = outlineAlpha * 0.5;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ov.outlinePts[0], ov.outlinePts[1]);
      for (let i = 1; i < ov.outlineN; i++) {
        ctx.lineTo(ov.outlinePts[i * 2], ov.outlinePts[i * 2 + 1]);
      }
      ctx.stroke();

      ctx.restore();
    }

    // ── Ghost lattice dots (dimmer now that outline provides structure) ─────
    if (ov.nPts > 0 && ov.formState === "forming") {
      ctx.save();
      ctx.globalAlpha = 0.08 * alpha;
      ctx.fillStyle = "rgba(200, 200, 240, 1)";
      ctx.beginPath();
      for (let i = 0; i < ov.nPts; i++) {
        const x = ov.latticePts[i * 2];
        const y = ov.latticePts[i * 2 + 1];
        ctx.moveTo(x + 2, y);
        ctx.arc(x, y, 2, 0, TAU);
      }
      ctx.fill();
      ctx.restore();
    }

    // ── Formula text ───────────────────────────────────────────────────────
    if (ov.formulaTex) {
      // Formula fades in during forming, full during holding, fades during dissolving
      const formulaAlpha =
        ov.formState === "forming"
          ? Math.max(0, (ov.stateProgress - 0.5) / 0.5) * alpha
          : alpha;
      if (formulaAlpha < 0.01) { return; }

      ctx.save();
      const fontSize = Math.max(16, Math.min(28, this.W * 0.019));
      ctx.font = `${fontSize}px "SF Mono", "Fira Code", "JetBrains Mono", monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const [fx, fy] = ov.formulaPos;

      // Glow backdrop
      ctx.globalAlpha = formulaAlpha * 0.2;
      ctx.filter = "blur(6px)";
      ctx.fillStyle = theme.accent;
      ctx.fillText(ov.formulaTex, fx, fy);
      ctx.filter = "none";

      // Shadow
      ctx.globalAlpha = formulaAlpha * 0.6;
      ctx.fillStyle = "rgba(0, 0, 0, 1)";
      ctx.fillText(ov.formulaTex, fx + 1, fy + 1);

      // Main text
      ctx.globalAlpha = formulaAlpha;
      ctx.fillStyle = "rgba(245, 245, 245, 1)";
      ctx.fillText(ov.formulaTex, fx, fy);

      // Shape label above
      ctx.globalAlpha = formulaAlpha * 0.55;
      ctx.font = `700 ${fontSize * 0.55}px "SF Mono", "Fira Code", monospace`;
      ctx.fillStyle = theme.accent;
      const label = ov.shapeName.toUpperCase();
      ctx.fillText(label, fx, fy - fontSize * 1.6);

      ctx.restore();
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  private rebuildVignette(): void {
    const ctx = this.mainCtx;
    const cx = this.W / 2;
    const cy = this.H / 2;
    const maxR = Math.sqrt(cx * cx + cy * cy);
    this.vignetteGrad = ctx.createRadialGradient(cx, cy, maxR * 0.35, cx, cy, maxR);
    this.vignetteGrad.addColorStop(0, "rgba(0,0,0,0)");
    this.vignetteGrad.addColorStop(1, "rgba(0,0,0,0.45)");
  }

  private clearTrail(bg: string): void {
    this.trailCtx.fillStyle = bg;
    this.trailCtx.fillRect(0, 0, this.W, this.H);
  }
}
