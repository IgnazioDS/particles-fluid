/**
 * Canvas 2D renderer with glow + bloom + trail effects.
 *
 * Pipeline:
 *  1. Fade trail canvas (semi-transparent bg fill)
 *  2. Draw particles (heat-mapped colours, additive blend)
 *  3. Compose main canvas: bg + glow (blur 7px) + bloom (blur 22px) + sharp
 *  4. Draw formation overlay (ghost dots + formula text)
 */

import { CFG, HEAT_LUT, HEAT_RGB } from "./config";

const TAU = Math.PI * 2;

export interface FormationOverlay {
  shapeName: string;
  formulaTex: string;
  formulaPos: [number, number];
  alpha: number;
  latticePts: Float32Array; // interleaved [x0,y0, x1,y1, ...]
  nPts: number;
}

export class Renderer {
  private W: number;
  private H: number;

  // Canvases
  private mainCanvas: HTMLCanvasElement;
  private mainCtx: CanvasRenderingContext2D;
  private trailCanvas: OffscreenCanvas;
  private trailCtx: OffscreenCanvasRenderingContext2D;
  private glowCanvas: OffscreenCanvas;
  private glowCtx: OffscreenCanvasRenderingContext2D;

  private pRad: number;
  private fadeBg: string;

  constructor(canvas: HTMLCanvasElement, w: number, h: number) {
    this.W = w;
    this.H = h;
    this.mainCanvas = canvas;
    canvas.width = w;
    canvas.height = h;
    this.mainCtx = canvas.getContext("2d", { alpha: false })!;
    this.pRad = CFG.particleRadius;

    const [r, g, b] = CFG.bgRGB;
    this.fadeBg = `rgba(${r},${g},${b},${CFG.trailFade})`;

    this.trailCanvas = new OffscreenCanvas(w, h);
    this.trailCtx = this.trailCanvas.getContext("2d", { alpha: false })!;
    this.trailCtx.fillStyle = CFG.bgColor;
    this.trailCtx.fillRect(0, 0, w, h);

    this.glowCanvas = new OffscreenCanvas(w, h);
    this.glowCtx = this.glowCanvas.getContext("2d")!;
  }

  resize(w: number, h: number): void {
    this.W = w;
    this.H = h;
    this.mainCanvas.width = w;
    this.mainCanvas.height = h;
    this.trailCanvas = new OffscreenCanvas(w, h);
    this.trailCtx = this.trailCanvas.getContext("2d", { alpha: false })!;
    this.trailCtx.fillStyle = CFG.bgColor;
    this.trailCtx.fillRect(0, 0, w, h);
    this.glowCanvas = new OffscreenCanvas(w, h);
    this.glowCtx = this.glowCanvas.getContext("2d")!;
  }

  reset(): void {
    this.trailCtx.fillStyle = CFG.bgColor;
    this.trailCtx.fillRect(0, 0, this.W, this.H);
  }

  render(
    px: Float32Array,
    py: Float32Array,
    temp: Float32Array,
    n: number,
    overlay: FormationOverlay | null,
    mouseX: number,
    mouseY: number,
    mouseDown: boolean,
    mouseRight: boolean,
  ): void {
    const W = this.W;
    const H = this.H;
    const tCtx = this.trailCtx;
    const ctx = this.mainCtx;

    // ── 1. Fade trails ─────────────────────────────────────────────────────
    tCtx.globalCompositeOperation = "source-over";
    tCtx.fillStyle = this.fadeBg;
    tCtx.fillRect(0, 0, W, H);

    // ── 2. Draw particles ──────────────────────────────────────────────────
    tCtx.globalCompositeOperation = "lighter";
    const rad = this.pRad;
    for (let i = 0; i < n; i++) {
      const t = Math.min(255, Math.max(0, (temp[i] * 255) | 0));
      tCtx.fillStyle = HEAT_LUT[t];
      tCtx.beginPath();
      tCtx.arc(px[i], py[i], rad, 0, TAU);
      tCtx.fill();
    }

    // ── 3. Compose main canvas ─────────────────────────────────────────────
    // Background
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = CFG.bgColor;
    ctx.fillRect(0, 0, W, H);

    // Glow pass (blurred trails)
    ctx.globalCompositeOperation = "lighter";

    const gCtx = this.glowCtx;
    gCtx.clearRect(0, 0, W, H);
    gCtx.filter = `blur(${CFG.glowSigma}px)`;
    gCtx.drawImage(this.trailCanvas, 0, 0);
    gCtx.filter = "none";
    ctx.globalAlpha = CFG.glowAlpha;
    ctx.drawImage(this.glowCanvas, 0, 0);

    // Bloom pass (wide soft halo)
    gCtx.clearRect(0, 0, W, H);
    gCtx.filter = `blur(${CFG.bloomSigma}px)`;
    gCtx.drawImage(this.trailCanvas, 0, 0);
    gCtx.filter = "none";
    ctx.globalAlpha = CFG.bloomAlpha;
    ctx.drawImage(this.glowCanvas, 0, 0);

    // Sharp particles on top
    ctx.globalAlpha = 1.0;
    ctx.drawImage(this.trailCanvas, 0, 0);

    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1.0;

    // ── 4. Mouse indicator ─────────────────────────────────────────────────
    if (mouseX >= 0 && mouseY >= 0) {
      ctx.save();
      ctx.globalAlpha = 0.15;
      ctx.strokeStyle = mouseRight
        ? "rgba(200, 60, 80, 0.6)"
        : mouseDown
          ? "rgba(80, 200, 90, 0.6)"
          : "rgba(180, 180, 220, 0.25)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(mouseX, mouseY, CFG.handRadius, 0, TAU);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(mouseX, mouseY, 3, 0, TAU);
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fill();
      ctx.restore();
    }

    // ── 5. Formation overlay ───────────────────────────────────────────────
    if (overlay && overlay.alpha > 0.01) {
      this.drawFormationOverlay(overlay);
    }
  }

  private drawFormationOverlay(ov: FormationOverlay): void {
    const ctx = this.mainCtx;
    const alpha = ov.alpha;

    // Ghost lattice dots
    if (ov.nPts > 0) {
      ctx.save();
      ctx.globalAlpha = 0.25 * alpha;
      ctx.fillStyle = "rgba(180, 180, 220, 1)";
      for (let i = 0; i < ov.nPts; i++) {
        const x = ov.latticePts[i * 2];
        const y = ov.latticePts[i * 2 + 1];
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, TAU);
        ctx.fill();
      }
      ctx.restore();
    }

    // Formula text
    if (ov.formulaTex) {
      ctx.save();
      const fontSize = Math.max(14, Math.min(24, this.W * 0.016));
      ctx.font = `${fontSize}px "SF Mono", "Fira Code", monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const [fx, fy] = ov.formulaPos;

      // Shadow
      ctx.globalAlpha = alpha * 0.8;
      ctx.fillStyle = "rgba(20, 20, 20, 1)";
      ctx.fillText(ov.formulaTex, fx + 1, fy + 1);

      // Main text
      ctx.globalAlpha = alpha;
      ctx.fillStyle = "rgba(240, 240, 240, 1)";
      ctx.fillText(ov.formulaTex, fx, fy);

      // Shape label (smaller, above formula)
      ctx.globalAlpha = alpha * 0.5;
      ctx.font = `${fontSize * 0.7}px "SF Mono", "Fira Code", monospace`;
      ctx.fillStyle = "rgba(255, 136, 68, 1)";
      ctx.fillText(ov.shapeName.toUpperCase(), fx, fy - fontSize * 1.4);

      ctx.restore();
    }
  }
}
