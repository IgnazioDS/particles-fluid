/**
 * Shape lattice generation + formation state machine.
 *
 * v2 — Direct position blending for pixel-perfect shapes.
 *
 * Instead of injecting spring forces (which fight SPH pressure and never
 * converge precisely), the formation manager directly overrides particle
 * positions AFTER the physics step via eased lerp:
 *
 *   FORMING:    pos = lerp(physics, target, easeInOut(t))
 *   HOLDING:    pos = target  (locked, velocity zeroed)
 *   DISSOLVING: pos = lerp(physics, target, easeInOut(1-t)) + burst kick
 *
 * This guarantees pixel-perfect shapes during HOLDING — critical for
 * video content quality.
 */

import { CFG } from "./config";
import type { ForceSource } from "./sph";

const PI = Math.PI;
const TAU = 2 * PI;

// ─────────────────────────────────────────────────────────────────────────────
// Formula strings
// ─────────────────────────────────────────────────────────────────────────────

export const FORMULAS: Record<string, string> = {
  circle: "x\u00b2 + y\u00b2 = r\u00b2",
  ellipse: "x\u00b2/a\u00b2 + y\u00b2/b\u00b2 = 1",
  triangle: "Pk = r(cos 2\u03c0k/3, sin 2\u03c0k/3)",
  square: "||x||\u221e \u2264 r",
  pentagon: "Pk = r(cos 2\u03c0k/5, sin 2\u03c0k/5)",
  hexagon: "Pk = r(cos \u03c0k/3,  sin \u03c0k/3)",
  rose: "r(\u03b8) = a\u00b7cos(n\u03b8)",
  spiral: "r = a\u00b7\u03b8  (Archimedean)",
  lissajous: "x = A sin(3t+\u03c0/4),  y = B sin(2t)",
};

export const SHAPE_NAMES = [
  "circle", "ellipse", "triangle", "square", "pentagon",
  "hexagon", "rose", "spiral", "lissajous",
] as const;

export type ShapeName = (typeof SHAPE_NAMES)[number];

// ─────────────────────────────────────────────────────────────────────────────
// Easing
// ─────────────────────────────────────────────────────────────────────────────

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

// ─────────────────────────────────────────────────────────────────────────────
// Lattice generators  (fill points — interleaved Float32Array)
// ─────────────────────────────────────────────────────────────────────────────

export function generateLattice(
  shape: ShapeName, cx: number, cy: number, radius: number, n: number,
  opts: { axisRatio?: number; nPetals?: number } = {},
): Float32Array {
  return GENERATORS[shape](cx, cy, radius, n, opts);
}

type GenFn = (cx: number, cy: number, r: number, n: number, opts: { axisRatio?: number; nPetals?: number }) => Float32Array;

function sunflower(cx: number, cy: number, r: number, n: number): Float32Array {
  const out = new Float32Array(n * 2);
  const golden = 2.399963;
  for (let i = 0; i < n; i++) {
    const theta = i * golden;
    const ri = r * Math.sqrt(i / n);
    out[i * 2] = ri * Math.cos(theta) + cx;
    out[i * 2 + 1] = ri * Math.sin(theta) + cy;
  }
  return out;
}

const genCircle: GenFn = (cx, cy, r, n) => sunflower(cx, cy, r, n);

const genEllipse: GenFn = (cx, cy, r, n, { axisRatio = 0.55 }) => {
  const out = new Float32Array(n * 2);
  const a = r, b = r * axisRatio;
  const golden = 2.399963;
  for (let i = 0; i < n; i++) {
    const theta = i * golden;
    const rf = Math.sqrt(i / n);
    out[i * 2] = a * rf * Math.cos(theta) + cx;
    out[i * 2 + 1] = b * rf * Math.sin(theta) + cy;
  }
  return out;
};

const genTriangle: GenFn = (cx, cy, r, n) => {
  const vx: number[] = [], vy: number[] = [];
  for (let k = 0; k < 3; k++) {
    vx.push(r * Math.cos(TAU * k / 3 - PI / 2));
    vy.push(r * Math.sin(TAU * k / 3 - PI / 2));
  }
  const side = Math.ceil(Math.sqrt(n * 2.5)) + 1;
  const pts: number[] = [];
  for (let i = 0; i <= side; i++) {
    for (let j = 0; j <= side - i; j++) {
      const k = side - i - j;
      const u = i / side, v = j / side, w = k / side;
      pts.push(u * vx[0] + v * vx[1] + w * vx[2]);
      pts.push(u * vy[0] + v * vy[1] + w * vy[2]);
    }
  }
  return resample(pts, n, cx, cy);
};

const genSquare: GenFn = (cx, cy, r, n) => {
  const side = Math.max(2, Math.floor(Math.sqrt(n)));
  const pts: number[] = [];
  for (let i = 0; i < side; i++)
    for (let j = 0; j < side; j++) {
      pts.push(-r + (2 * r * i) / (side - 1));
      pts.push(-r + (2 * r * j) / (side - 1));
    }
  return resample(pts, n, cx, cy);
};

function genNgon(sides: number): GenFn {
  return (cx, cy, r, n) => {
    const angles = Array.from({ length: sides }, (_, k) => TAU * k / sides);
    const vxA = angles.map(a => r * Math.cos(a));
    const vyA = angles.map(a => r * Math.sin(a));
    const gridN = Math.ceil(Math.sqrt(n * 2.0)) + 2;
    const pts: number[] = [];
    for (let i = 0; i < gridN; i++)
      for (let j = 0; j < gridN; j++) {
        const px = -r + (2 * r * i) / (gridN - 1);
        const py = -r + (2 * r * j) / (gridN - 1);
        if (pointInPoly(px, py, vxA, vyA)) { pts.push(px); pts.push(py); }
      }
    return resample(pts, n, cx, cy);
  };
}

const genRose: GenFn = (cx, cy, r, n, { nPetals = 3 }) => {
  const np = Math.max(1, nPetals);
  const over = n * 8;
  const pts: number[] = [];
  for (let i = 0; i < over; i++) {
    const theta = (TAU * i) / over;
    const ri = r * Math.abs(Math.cos(np * theta));
    if (ri < r * 0.04) continue;
    pts.push(ri * Math.cos(theta)); pts.push(ri * Math.sin(theta));
  }
  return resample(pts, n, cx, cy);
};

const genSpiral: GenFn = (cx, cy, r, n) => {
  const turns = 3, over = n * 20;
  const a = r / (TAU * turns);
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < over; i++) {
    const theta = (TAU * turns * i) / (over - 1);
    xs.push(a * theta * Math.cos(theta));
    ys.push(a * theta * Math.sin(theta));
  }
  const s = [0];
  for (let i = 1; i < over; i++) {
    const dx = xs[i] - xs[i - 1], dy = ys[i] - ys[i - 1];
    s.push(s[i - 1] + Math.sqrt(dx * dx + dy * dy));
  }
  const total = s[over - 1];
  const out = new Float32Array(n * 2);
  let si = 0;
  for (let i = 0; i < n; i++) {
    const target = (total * i) / (n - 1);
    while (si < over - 2 && s[si + 1] < target) si++;
    const t = (target - s[si]) / (s[si + 1] - s[si] + 1e-12);
    out[i * 2] = xs[si] + t * (xs[si + 1] - xs[si]) + cx;
    out[i * 2 + 1] = ys[si] + t * (ys[si + 1] - ys[si]) + cy;
  }
  return out;
};

const genLissajous: GenFn = (cx, cy, r, n) => {
  const A = r * 0.85;
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const t = (TAU * i) / n;
    out[i * 2] = A * Math.sin(3 * t + PI / 4) + cx;
    out[i * 2 + 1] = A * Math.sin(2 * t) + cy;
  }
  return out;
};

const GENERATORS: Record<ShapeName, GenFn> = {
  circle: genCircle, ellipse: genEllipse, triangle: genTriangle,
  square: genSquare, pentagon: genNgon(5), hexagon: genNgon(6),
  rose: genRose, spiral: genSpiral, lissajous: genLissajous,
};

// ─── helpers ────────────────────────────────────────────────────────────────

function resample(pts: number[], n: number, cx: number, cy: number): Float32Array {
  const m = pts.length / 2;
  const out = new Float32Array(n * 2);
  if (m === 0) return out;
  if (m >= n) {
    for (let i = 0; i < n; i++) {
      const idx = Math.round((i * (m - 1)) / (n - 1));
      out[i * 2] = pts[idx * 2] + cx;
      out[i * 2 + 1] = pts[idx * 2 + 1] + cy;
    }
  } else {
    for (let i = 0; i < m; i++) {
      out[i * 2] = pts[i * 2] + cx;
      out[i * 2 + 1] = pts[i * 2 + 1] + cy;
    }
    for (let i = m; i < n; i++) {
      const src = i % m;
      out[i * 2] = pts[src * 2] + cx + (Math.random() - 0.5);
      out[i * 2 + 1] = pts[src * 2 + 1] + cy + (Math.random() - 0.5);
    }
  }
  return out;
}

function pointInPoly(px: number, py: number, vx: number[], vy: number[]): boolean {
  let inside = false;
  for (let i = 0, j = vx.length - 1; i < vx.length; j = i++) {
    if ((vy[i] > py) !== (vy[j] > py) &&
        px < ((vx[j] - vx[i]) * (py - vy[i])) / (vy[j] - vy[i] + 1e-12) + vx[i]) {
      inside = !inside;
    }
  }
  return inside;
}

// ─────────────────────────────────────────────────────────────────────────────
// Outline generators  (for drawing the mathematical shape as a thin line)
// ─────────────────────────────────────────────────────────────────────────────

export function generateOutline(
  shape: ShapeName, cx: number, cy: number, r: number,
  opts: { axisRatio?: number; nPetals?: number } = {},
): { pts: Float32Array; n: number } {
  const p: number[] = [];
  const N = 150;

  switch (shape) {
    case "circle":
      for (let i = 0; i <= N; i++) { const θ = TAU * i / N; p.push(cx + r * Math.cos(θ), cy + r * Math.sin(θ)); }
      break;
    case "ellipse": {
      const a = r, b = r * (opts.axisRatio || 0.55);
      for (let i = 0; i <= N; i++) { const θ = TAU * i / N; p.push(cx + a * Math.cos(θ), cy + b * Math.sin(θ)); }
      break;
    }
    case "triangle": case "pentagon": case "hexagon": case "square": {
      const sides = shape === "triangle" ? 3 : shape === "square" ? 4 : shape === "pentagon" ? 5 : 6;
      const off = shape === "triangle" ? -PI / 2 : 0;
      for (let k = 0; k <= sides; k++) {
        const a = off + TAU * (k % sides) / sides;
        p.push(cx + r * Math.cos(a), cy + r * Math.sin(a));
      }
      break;
    }
    case "rose": {
      const np = Math.max(1, opts.nPetals || 3);
      for (let i = 0; i <= N * 2; i++) { const θ = TAU * i / (N * 2); const ri = r * Math.abs(Math.cos(np * θ)); p.push(cx + ri * Math.cos(θ), cy + ri * Math.sin(θ)); }
      break;
    }
    case "spiral": {
      const turns = 3, a = r / (TAU * turns);
      for (let i = 0; i <= N * 2; i++) { const θ = TAU * turns * i / (N * 2); const ri = a * θ; p.push(cx + ri * Math.cos(θ), cy + ri * Math.sin(θ)); }
      break;
    }
    case "lissajous": {
      const A = r * 0.85;
      for (let i = 0; i <= N; i++) { const t = TAU * i / N; p.push(cx + A * Math.sin(3 * t + PI / 4), cy + A * Math.sin(2 * t)); }
      break;
    }
  }
  return { pts: new Float32Array(p), n: p.length / 2 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Formation state machine  (v2 — direct position blending)
// ─────────────────────────────────────────────────────────────────────────────

const enum State { IDLE, FORMING, HOLDING, DISSOLVING }

const FORM_DUR = 1.5;   // seconds to form
const HOLD_DUR = 3.2;   // seconds to hold
const DISS_DUR = 1.5;   // seconds to dissolve
const CYCLE_PAUSE = 1.2; // pause between auto-cycle shapes

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

export class FormationManager {
  private W: number;
  private H: number;
  private state: State = State.IDLE;
  private stateTimer = 0;
  private enabled = true;

  private shapeName: ShapeName | null = null;
  private shapeOpts: { axisRatio?: number; nPetals?: number } = {};

  private lattice: Float32Array | null = null;
  private outline: { pts: Float32Array; n: number } | null = null;
  private nPts = 0;
  private slots: Int32Array | null = null; // particle indices assigned to slots
  private blend = 0;
  private dissolveKicked = false;

  /** Forces for unassigned particles — read by main loop before solver.step(). */
  cachedForces: ForceSource[] = [];

  // Auto-cycle
  private autoCycleOn = false;
  private autoCycleIdx = 0;
  private autoCyclePause = 0;

  constructor(w: number, h: number) {
    this.W = w;
    this.H = h;
  }

  get isEnabled(): boolean { return this.enabled; }
  get isAutoCycling(): boolean { return this.autoCycleOn; }

  toggle(): boolean {
    this.enabled = !this.enabled;
    if (!this.enabled) this.resetToIdle();
    return this.enabled;
  }

  toggleAutoCycle(): boolean {
    this.autoCycleOn = !this.autoCycleOn;
    if (this.autoCycleOn && this.state === State.IDLE) {
      this.autoCycleIdx = 0;
      this.triggerNextCycleShape();
    }
    return this.autoCycleOn;
  }

  forceShape(name: ShapeName, opts: { axisRatio?: number; nPetals?: number } = {}): void {
    this.shapeName = name;
    this.shapeOpts = opts;
    this.slots = null;
    this.enterForming();
  }

  /**
   * Main per-frame update.  DIRECTLY MODIFIES particle arrays (px, py, vx, vy, temp).
   * Call this AFTER solver.step() so physics runs first, then we override.
   *
   * Also updates cachedForces for the next frame (mild centre attractor for free particles).
   */
  update(
    px: Float32Array, py: Float32Array,
    vx: Float32Array, vy: Float32Array,
    temp: Float32Array,
    n: number, dt: number,
  ): void {
    if (!this.enabled) { this.cachedForces = []; return; }

    this.stateTimer += dt;

    switch (this.state) {
      case State.IDLE:
        this.cachedForces = [];
        // Auto-cycle pause countdown
        if (this.autoCycleOn) {
          this.autoCyclePause -= dt;
          if (this.autoCyclePause <= 0) this.triggerNextCycleShape();
        }
        break;

      case State.FORMING: {
        const t = Math.min(this.stateTimer / FORM_DUR, 1);
        this.blend = easeInOutCubic(t);
        this.applyBlend(px, py, vx, vy, temp, n);
        this.setCentreAttractor();
        if (t >= 1) {
          this.state = State.HOLDING;
          this.stateTimer = 0;
        }
        break;
      }

      case State.HOLDING: {
        this.blend = 1;
        this.applyBlend(px, py, vx, vy, temp, n);
        this.setCentreAttractor();
        // Gentle pulse on held particles
        if (this.slots && this.lattice) {
          const pulse = 0.42 + 0.12 * Math.sin(this.stateTimer * 3.5);
          for (let s = 0; s < this.nPts; s++) {
            const idx = this.slots[s];
            if (idx >= 0 && idx < n) temp[idx] = Math.max(temp[idx], pulse);
          }
        }
        if (this.stateTimer >= HOLD_DUR) {
          this.enterDissolving(px, py, vx, vy, temp, n);
        }
        break;
      }

      case State.DISSOLVING: {
        const t = Math.min(this.stateTimer / DISS_DUR, 1);
        this.blend = easeInOutCubic(1 - t);
        if (this.blend > 0.01) {
          this.applyBlend(px, py, vx, vy, temp, n);
        }
        this.cachedForces = [];
        if (t >= 1) {
          this.resetToIdle();
          if (this.autoCycleOn) {
            this.autoCyclePause = CYCLE_PAUSE;
          }
        }
        break;
      }
    }
  }

  getOverlay(): FormationOverlay | null {
    if (this.state === State.IDLE || !this.lattice || !this.shapeName) return null;

    let alpha = 1;
    let formState: "forming" | "holding" | "dissolving" = "holding";
    let progress = 0;

    if (this.state === State.FORMING) {
      alpha = Math.min(this.stateTimer / (FORM_DUR * 0.5), 1);
      formState = "forming";
      progress = Math.min(this.stateTimer / FORM_DUR, 1);
    } else if (this.state === State.HOLDING) {
      formState = "holding";
      progress = Math.min(this.stateTimer / HOLD_DUR, 1);
    } else if (this.state === State.DISSOLVING) {
      alpha = Math.max(1 - this.stateTimer / DISS_DUR, 0);
      formState = "dissolving";
      progress = Math.min(this.stateTimer / DISS_DUR, 1);
    }

    return {
      shapeName: this.shapeName,
      formulaTex: FORMULAS[this.shapeName] || "",
      formulaPos: [this.W / 2, this.H / 2 + CFG.formationRadius + 45],
      alpha,
      latticePts: this.lattice,
      nPts: this.nPts,
      outlinePts: this.outline?.pts ?? new Float32Array(0),
      outlineN: this.outline?.n ?? 0,
      formState,
      stateProgress: progress,
    };
  }

  // ── internal ──────────────────────────────────────────────────────────────

  private resetToIdle(): void {
    this.state = State.IDLE;
    this.stateTimer = 0;
    this.shapeName = null;
    this.lattice = null;
    this.outline = null;
    this.slots = null;
    this.blend = 0;
    this.dissolveKicked = false;
    this.cachedForces = [];
  }

  private enterForming(): void {
    if (!this.shapeName) return;
    const cx = this.W / 2, cy = this.H / 2;
    this.nPts = CFG.formationNPoints;
    this.lattice = generateLattice(this.shapeName, cx, cy, CFG.formationRadius, this.nPts, this.shapeOpts);
    this.outline = generateOutline(this.shapeName, cx, cy, CFG.formationRadius, this.shapeOpts);
    this.state = State.FORMING;
    this.stateTimer = 0;
    this.blend = 0;
    this.dissolveKicked = false;
    // Defer slot assignment to first applyBlend call (when we have particle positions)
  }

  private enterDissolving(
    px: Float32Array, py: Float32Array,
    vx: Float32Array, vy: Float32Array,
    temp: Float32Array, n: number,
  ): void {
    this.state = State.DISSOLVING;
    this.stateTimer = 0;

    // Burst kick: radial outward velocity + temp flash
    if (this.slots && this.lattice) {
      const cx = this.W / 2, cy = this.H / 2;
      for (let s = 0; s < this.nPts; s++) {
        const idx = this.slots[s];
        if (idx < 0 || idx >= n) continue;
        const dx = px[idx] - cx;
        const dy = py[idx] - cy;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const kick = 120 + Math.random() * 100;
        vx[idx] = (dx / d) * kick + (Math.random() - 0.5) * 60;
        vy[idx] = (dy / d) * kick + (Math.random() - 0.5) * 60;
        temp[idx] = 0.85; // bright flash
      }
    }
  }

  private assignSlots(px: Float32Array, py: Float32Array, n: number): void {
    if (!this.lattice) return;
    const nSlots = this.nPts;
    const assigned = new Int32Array(nSlots);
    const used = new Set<number>();

    for (let s = 0; s < nSlots; s++) {
      const tx = this.lattice[s * 2];
      const ty = this.lattice[s * 2 + 1];
      let bestD = Infinity, bestIdx = 0;
      for (let p = 0; p < n; p++) {
        if (used.has(p)) continue;
        const dx = tx - px[p], dy = ty - py[p];
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; bestIdx = p; }
      }
      assigned[s] = bestIdx;
      used.add(bestIdx);
    }
    this.slots = assigned;
  }

  /** Lerp assigned particles toward lattice targets with current blend. */
  private applyBlend(
    px: Float32Array, py: Float32Array,
    vx: Float32Array, vy: Float32Array,
    temp: Float32Array, n: number,
  ): void {
    if (!this.lattice) return;
    if (!this.slots) this.assignSlots(px, py, n);
    const slots = this.slots!;
    const lattice = this.lattice;
    const b = this.blend;

    for (let s = 0; s < this.nPts; s++) {
      const idx = slots[s];
      if (idx < 0 || idx >= n) continue;
      const tx = lattice[s * 2];
      const ty = lattice[s * 2 + 1];

      // Lerp position
      px[idx] = px[idx] * (1 - b) + tx * b;
      py[idx] = py[idx] * (1 - b) + ty * b;

      // Dampen velocity (fully zeroed when locked)
      vx[idx] *= (1 - b * 0.92);
      vy[idx] *= (1 - b * 0.92);

      // Temperature boost — glow on arrival
      const dx = px[idx] - tx, dy = py[idx] - ty;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const arrival = 1 - Math.min(dist / 40, 1);
      temp[idx] = Math.max(temp[idx], b * arrival * 0.55);
    }
  }

  private setCentreAttractor(): void {
    this.cachedForces = [{
      x: this.W / 2,
      y: this.H / 2,
      strength: 60,
      radius: CFG.formationRadius * 2.5,
    }];
  }

  private triggerNextCycleShape(): void {
    const name = SHAPE_NAMES[this.autoCycleIdx % SHAPE_NAMES.length];
    const opts: { axisRatio?: number; nPetals?: number } = {};
    if (name === "ellipse") opts.axisRatio = 1.5;
    if (name === "rose") opts.nPetals = 3;
    this.shapeName = name;
    this.shapeOpts = opts;
    this.slots = null;
    this.enterForming();
    this.autoCycleIdx++;
  }
}
