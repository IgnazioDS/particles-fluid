/**
 * Shape lattice generation + formation state machine.
 *
 * Lattice: pure-math generators producing (n, 2) target positions.
 * Formation: 5-state machine driving particles toward target shapes.
 */

import { CFG } from "./config";

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
// Shape lattice generators
// ─────────────────────────────────────────────────────────────────────────────

/** Generate shape as interleaved Float32Array [x0,y0, x1,y1, ...]. */
export function generateLattice(
  shape: ShapeName,
  cx: number,
  cy: number,
  radius: number,
  n: number,
  opts: { axisRatio?: number; nPetals?: number } = {},
): Float32Array {
  const gen = GENERATORS[shape];
  return gen(cx, cy, radius, n, opts);
}

type GenFn = (
  cx: number, cy: number, r: number, n: number,
  opts: { axisRatio?: number; nPetals?: number },
) => Float32Array;

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
  const a = r;
  const b = r * axisRatio;
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
  // Vertices
  const vx: number[] = [], vy: number[] = [];
  for (let k = 0; k < 3; k++) {
    vx.push(r * Math.cos(TAU * k / 3 - PI / 2));
    vy.push(r * Math.sin(TAU * k / 3 - PI / 2));
  }
  // Barycentric grid
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
  for (let i = 0; i < side; i++) {
    for (let j = 0; j < side; j++) {
      pts.push(-r + (2 * r * i) / (side - 1));
      pts.push(-r + (2 * r * j) / (side - 1));
    }
  }
  return resample(pts, n, cx, cy);
};

function genNgon(sides: number): GenFn {
  return (cx, cy, r, n) => {
    const angles = Array.from({ length: sides }, (_, k) => TAU * k / sides);
    const vxArr = angles.map((a) => r * Math.cos(a));
    const vyArr = angles.map((a) => r * Math.sin(a));
    const gridN = Math.ceil(Math.sqrt(n * 2.0)) + 2;
    const pts: number[] = [];
    for (let i = 0; i < gridN; i++) {
      for (let j = 0; j < gridN; j++) {
        const px = -r + (2 * r * i) / (gridN - 1);
        const py = -r + (2 * r * j) / (gridN - 1);
        if (pointInPolygon(px, py, vxArr, vyArr)) {
          pts.push(px);
          pts.push(py);
        }
      }
    }
    return resample(pts, n, cx, cy);
  };
}

const genRose: GenFn = (cx, cy, r, n, { nPetals = 3 }) => {
  const np = Math.max(1, nPetals);
  const over = n * 6;
  const pts: number[] = [];
  for (let i = 0; i < over; i++) {
    const theta = (TAU * i) / over;
    const ri = r * Math.abs(Math.cos(np * theta));
    if (ri < r * 0.05) continue;
    pts.push(ri * Math.cos(theta));
    pts.push(ri * Math.sin(theta));
  }
  return resample(pts, n, cx, cy);
};

const genSpiral: GenFn = (cx, cy, r, n) => {
  const turns = 3;
  const over = n * 20;
  const a = r / (TAU * turns);
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < over; i++) {
    const theta = (TAU * turns * i) / (over - 1);
    const ri = a * theta;
    xs.push(ri * Math.cos(theta));
    ys.push(ri * Math.sin(theta));
  }
  // Arc-length re-parameterisation
  const s = [0];
  for (let i = 1; i < over; i++) {
    const dx = xs[i] - xs[i - 1];
    const dy = ys[i] - ys[i - 1];
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
  const B = r * 0.85;
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const t = (TAU * i) / n;
    out[i * 2] = A * Math.sin(3 * t + PI / 4) + cx;
    out[i * 2 + 1] = B * Math.sin(2 * t) + cy;
  }
  return out;
};

const GENERATORS: Record<ShapeName, GenFn> = {
  circle: genCircle,
  ellipse: genEllipse,
  triangle: genTriangle,
  square: genSquare,
  pentagon: genNgon(5),
  hexagon: genNgon(6),
  rose: genRose,
  spiral: genSpiral,
  lissajous: genLissajous,
};

// ─── helpers ────────────────────────────────────────────────────────────────

function resample(pts: number[], n: number, cx: number, cy: number): Float32Array {
  const m = pts.length / 2;
  const out = new Float32Array(n * 2);
  if (m === 0) { out.fill(0); return out; }
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

function pointInPolygon(px: number, py: number, vx: number[], vy: number[]): boolean {
  let inside = false;
  const nv = vx.length;
  for (let i = 0, j = nv - 1; i < nv; j = i++) {
    if ((vy[i] > py) !== (vy[j] > py) &&
        px < ((vx[j] - vx[i]) * (py - vy[i])) / (vy[j] - vy[i] + 1e-12) + vx[i]) {
      inside = !inside;
    }
  }
  return inside;
}

// ─────────────────────────────────────────────────────────────────────────────
// Formation state machine
// ─────────────────────────────────────────────────────────────────────────────

enum State {
  IDLE,
  FORMING,
  HOLDING,
  DISSOLVING,
}

export class FormationManager {
  private W: number;
  private H: number;
  private state = State.IDLE;
  private enabled = true;

  // Timers
  private formTimer = 0;
  private holdTimer = 0;
  private dissolveTimer = 0;

  // Current shape
  private shapeName: ShapeName | null = null;
  private shapeOpts: { axisRatio?: number; nPetals?: number } = {};

  // Lattice + assignment
  private lattice: Float32Array | null = null;
  private nPts = 0;
  private slotIndices: Int32Array | null = null;

  // Spring
  private springK = 0;

  constructor(w: number, h: number) {
    this.W = w;
    this.H = h;
  }

  get isEnabled(): boolean { return this.enabled; }

  toggle(): boolean {
    this.enabled = !this.enabled;
    if (!this.enabled) this.resetToIdle();
    return this.enabled;
  }

  /** Trigger a specific shape immediately (keyboard debug). */
  forceShape(name: ShapeName, opts: { axisRatio?: number; nPetals?: number } = {}): void {
    this.shapeName = name;
    this.shapeOpts = opts;
    this.enterForming();
  }

  /**
   * Per-frame update.  Returns extra force sources for the SPH solver.
   * px/py are current particle positions.
   */
  update(
    px: Float32Array,
    py: Float32Array,
    nParticles: number,
    dt: number,
  ): Array<{ x: number; y: number; strength: number; radius: number }> {
    if (!this.enabled) return [];
    const sources: Array<{ x: number; y: number; strength: number; radius: number }> = [];

    switch (this.state) {
      case State.IDLE:
        break;

      case State.FORMING: {
        this.formTimer += dt;
        const ramp = Math.min(this.formTimer / 0.5, 1.0);
        this.springK = CFG.formationSpringK * ramp;
        this.computeSprings(px, py, nParticles, sources);
        if (this.checkConvergence(px, py, nParticles, 8, 0.85) || this.formTimer > 6) {
          this.state = State.HOLDING;
          this.holdTimer = 0;
        }
        break;
      }

      case State.HOLDING: {
        this.holdTimer += dt;
        this.springK = CFG.formationSpringK;
        this.computeSprings(px, py, nParticles, sources);
        if (this.holdTimer >= CFG.formationHoldS) {
          this.state = State.DISSOLVING;
          this.dissolveTimer = 0;
        }
        break;
      }

      case State.DISSOLVING: {
        this.dissolveTimer += dt;
        const ramp = Math.max(1 - this.dissolveTimer / 1.2, 0);
        this.springK = CFG.formationSpringK * ramp;
        this.computeSprings(px, py, nParticles, sources);
        if (this.dissolveTimer >= 1.2) this.resetToIdle();
        break;
      }
    }
    return sources;
  }

  /** Get render overlay data, or null when idle. */
  getOverlay(): {
    shapeName: string;
    formulaTex: string;
    formulaPos: [number, number];
    alpha: number;
    latticePts: Float32Array;
    nPts: number;
  } | null {
    if (this.state === State.IDLE || !this.lattice || !this.shapeName) return null;

    let alpha = 1;
    if (this.state === State.FORMING) alpha = Math.min(this.formTimer / 0.5, 1);
    else if (this.state === State.DISSOLVING) alpha = Math.max(1 - this.dissolveTimer / 1.2, 0);

    return {
      shapeName: this.shapeName,
      formulaTex: FORMULAS[this.shapeName] || "",
      formulaPos: [this.W / 2, this.H / 2 + CFG.formationRadius + 40],
      alpha,
      latticePts: this.lattice,
      nPts: this.nPts,
    };
  }

  // ── internal ──────────────────────────────────────────────────────────────

  private resetToIdle(): void {
    this.state = State.IDLE;
    this.shapeName = null;
    this.lattice = null;
    this.slotIndices = null;
    this.springK = 0;
  }

  private enterForming(): void {
    if (!this.shapeName) return;
    const cx = this.W / 2;
    const cy = this.H / 2;
    this.nPts = CFG.formationNPoints;
    this.lattice = generateLattice(
      this.shapeName, cx, cy,
      CFG.formationRadius, this.nPts, this.shapeOpts,
    );
    this.slotIndices = null; // will assign on first spring compute
    this.state = State.FORMING;
    this.formTimer = 0;
  }

  private assignSlots(px: Float32Array, py: Float32Array, nParticles: number): void {
    if (!this.lattice) return;
    const nSlots = this.nPts;
    const assigned = new Int32Array(nSlots);
    const used = new Set<number>();

    // Greedy nearest-neighbour assignment
    for (let s = 0; s < nSlots; s++) {
      const tx = this.lattice[s * 2];
      const ty = this.lattice[s * 2 + 1];
      let bestD = Infinity;
      let bestIdx = 0;
      for (let p = 0; p < nParticles; p++) {
        if (used.has(p)) continue;
        const dx = tx - px[p];
        const dy = ty - py[p];
        const d = dx * dx + dy * dy;
        if (d < bestD) { bestD = d; bestIdx = p; }
      }
      assigned[s] = bestIdx;
      used.add(bestIdx);
    }
    this.slotIndices = assigned;
  }

  private computeSprings(
    px: Float32Array,
    py: Float32Array,
    nParticles: number,
    sources: Array<{ x: number; y: number; strength: number; radius: number }>,
  ): void {
    if (!this.lattice || this.springK < 0.1) return;
    if (!this.slotIndices) this.assignSlots(px, py, nParticles);
    const slots = this.slotIndices!;
    const lattice = this.lattice;
    const k = this.springK;
    const maxF = CFG.formationMaxForce;

    for (let s = 0; s < this.nPts; s++) {
      const pidx = slots[s];
      if (pidx < 0 || pidx >= nParticles) continue;
      const tx = lattice[s * 2];
      const ty = lattice[s * 2 + 1];
      const dx = tx - px[pidx];
      const dy = ty - py[pidx];
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 1) continue;
      let fMag = k * dist;
      if (fMag > maxF) fMag = maxF;
      sources.push({
        x: tx, y: ty,
        strength: Math.min(fMag, maxF),
        radius: Math.max(dist + 5, 10),
      });
    }

    // Centre attractor for free particles
    sources.push({
      x: this.W / 2,
      y: this.H / 2,
      strength: 60,
      radius: CFG.formationRadius * 2.5,
    });
  }

  private checkConvergence(
    px: Float32Array, py: Float32Array,
    nParticles: number,
    threshold: number, fraction: number,
  ): boolean {
    if (!this.lattice || !this.slotIndices) return false;
    const th2 = threshold * threshold;
    let close = 0;
    for (let s = 0; s < this.nPts; s++) {
      const pidx = this.slotIndices[s];
      if (pidx < 0 || pidx >= nParticles) continue;
      const dx = this.lattice[s * 2] - px[pidx];
      const dy = this.lattice[s * 2 + 1] - py[pidx];
      if (dx * dx + dy * dy < th2) close++;
    }
    return close >= fraction * this.nPts;
  }
}
