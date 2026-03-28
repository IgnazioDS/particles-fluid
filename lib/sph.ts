/**
 * 2-D SPH fluid solver (TypeScript).
 *
 * Spatial hashing for O(n * avg_neighbours) neighbour lookup.
 * Supports vortex forces (perpendicular to radial direction).
 */

const PI = Math.PI;

export interface ForceSource {
  x: number;
  y: number;
  strength: number;
  radius: number;
  vortex?: number; // perpendicular force component (spin)
}

export class SPHSolver {
  readonly n: number;
  readonly W: number;
  readonly H: number;

  // SoA particle state
  px: Float32Array;
  py: Float32Array;
  vx: Float32Array;
  vy: Float32Array;
  rho: Float32Array;
  pres: Float32Array;
  temp: Float32Array;

  // Spatial hash grid
  private cellSize: number;
  private gridW: number;
  private gridH: number;
  private nCells: number;
  private cellCount: Int32Array;
  private cellStart: Int32Array;
  private order: Int32Array;
  private writePos: Int32Array;

  // Physics
  private h: number;
  private h2: number;
  private rho0: number;
  private k: number;
  mu: number;
  private dt: number;
  private mass: number;
  private damp: number;
  private cPoly6: number;
  private cSpiky: number;
  private cVisc: number;

  gravX = 0;
  gravY: number;
  sources: ForceSource[] = [];

  constructor(n: number, width: number, height: number, h = 26, rho0 = 280, k = 200, mu = 0.55, gravity = 200, dt = 0.008, mass = 1.0, damp = 0.4) {
    this.n = n;
    this.W = width;
    this.H = height;
    this.h = h;
    this.h2 = h * h;
    this.rho0 = rho0;
    this.k = k;
    this.mu = mu;
    this.dt = dt;
    this.mass = mass;
    this.damp = damp;
    this.gravY = gravity;
    this.cPoly6 = 4.0 / (PI * h ** 8);
    this.cSpiky = -30.0 / (PI * h ** 5);
    this.cVisc = 40.0 / (PI * h ** 5);

    this.px = new Float32Array(n);
    this.py = new Float32Array(n);
    this.vx = new Float32Array(n);
    this.vy = new Float32Array(n);
    this.rho = new Float32Array(n);
    this.pres = new Float32Array(n);
    this.temp = new Float32Array(n);

    this.cellSize = h;
    this.gridW = Math.ceil(width / h) + 1;
    this.gridH = Math.ceil(height / h) + 1;
    this.nCells = this.gridW * this.gridH;
    this.cellCount = new Int32Array(this.nCells);
    this.cellStart = new Int32Array(this.nCells);
    this.order = new Int32Array(n);
    this.writePos = new Int32Array(this.nCells);
  }

  initialize(): void {
    for (let i = 0; i < this.n; i++) {
      this.px[i] = Math.random() * this.W;
      this.py[i] = Math.random() * this.H * 0.8 + this.H * 0.1;
      this.vx[i] = 0;
      this.vy[i] = 0;
      this.temp[i] = Math.random() * 0.15;
    }
    this.gravX = 0;
    this.gravY = 200;
  }

  setGravityTilt(angle: number): void {
    const g = Math.sqrt(this.gravX ** 2 + this.gravY ** 2) || 200;
    this.gravX = Math.sin(angle) * g;
    this.gravY = Math.cos(angle) * g;
  }

  // ── Spatial hash ─────────────────────────────────────────────────────────

  private buildGrid(): void {
    this.cellCount.fill(0);
    const cs = this.cellSize;
    const gw = this.gridW;
    const gh = this.gridH;
    for (let i = 0; i < this.n; i++) {
      const ci = Math.min(Math.max(0, (this.px[i] / cs) | 0), gw - 1);
      const cj = Math.min(Math.max(0, (this.py[i] / cs) | 0), gh - 1);
      this.cellCount[ci + cj * gw]++;
    }
    let sum = 0;
    for (let c = 0; c < this.nCells; c++) {
      this.cellStart[c] = sum;
      sum += this.cellCount[c];
    }
    this.writePos.set(this.cellStart);
    for (let i = 0; i < this.n; i++) {
      const ci = Math.min(Math.max(0, (this.px[i] / cs) | 0), gw - 1);
      const cj = Math.min(Math.max(0, (this.py[i] / cs) | 0), gh - 1);
      const idx = ci + cj * gw;
      this.order[this.writePos[idx]++] = i;
    }
  }

  // ── SPH kernels ──────────────────────────────────────────────────────────

  private computeDensityPressure(): void {
    const { n, h2, mass, rho0, k: kk, cPoly6 } = this;
    const cs = this.cellSize;
    const gw = this.gridW;
    const gh = this.gridH;
    const { px, py, rho, pres, cellStart, cellCount, order } = this;

    for (let i = 0; i < n; i++) {
      let rho_i = 0;
      const xi = px[i];
      const yi = py[i];
      const ci = Math.min(Math.max(0, (xi / cs) | 0), gw - 1);
      const cj = Math.min(Math.max(0, (yi / cs) | 0), gh - 1);
      for (let di = -1; di <= 1; di++) {
        const ni = ci + di;
        if (ni < 0 || ni >= gw) continue;
        for (let dj = -1; dj <= 1; dj++) {
          const nj = cj + dj;
          if (nj < 0 || nj >= gh) continue;
          const cell = ni + nj * gw;
          const start = cellStart[cell];
          const count = cellCount[cell];
          for (let idx = start; idx < start + count; idx++) {
            const j = order[idx];
            const dx = xi - px[j];
            const dy = yi - py[j];
            const r2 = dx * dx + dy * dy;
            if (r2 < h2) {
              const d = h2 - r2;
              rho_i += mass * cPoly6 * d * d * d;
            }
          }
        }
      }
      rho[i] = rho_i;
      pres[i] = Math.max(kk * (rho_i - rho0), 0);
    }
  }

  private computeForcesIntegrate(): void {
    const { n, h, h2, mass, mu, dt, damp, cSpiky, cVisc, px, py, vx, vy, rho, pres, sources } = this;
    const cs = this.cellSize;
    const gw = this.gridW;
    const gh = this.gridH;
    const { cellStart, cellCount, order } = this;
    const W = this.W;
    const H = this.H;
    const gravX = this.gravX;
    const gravY = this.gravY;

    for (let i = 0; i < n; i++) {
      let fpx = 0, fpy = 0, fvx = 0, fvy = 0;
      const xi = px[i], yi = py[i], vxi = vx[i], vyi = vy[i];
      const rho_i = Math.max(rho[i], 1e-3);
      const pres_i = pres[i];
      const ci = Math.min(Math.max(0, (xi / cs) | 0), gw - 1);
      const cj = Math.min(Math.max(0, (yi / cs) | 0), gh - 1);

      for (let di = -1; di <= 1; di++) {
        const ni = ci + di;
        if (ni < 0 || ni >= gw) continue;
        for (let dj = -1; dj <= 1; dj++) {
          const nj = cj + dj;
          if (nj < 0 || nj >= gh) continue;
          const cell = ni + nj * gw;
          const start = cellStart[cell];
          const count = cellCount[cell];
          for (let idx = start; idx < start + count; idx++) {
            const j = order[idx];
            if (i === j) continue;
            const dx = xi - px[j];
            const dy = yi - py[j];
            const r2 = dx * dx + dy * dy;
            if (r2 >= h2 || r2 < 1e-8) continue;
            const r = Math.sqrt(r2);
            const rho_j = Math.max(rho[j], 1e-3);
            const d = h - r;
            const scale = -mass * (pres_i + pres[j]) / (2 * rho_j) * cSpiky * d * d;
            const invR = 1 / r;
            fpx += scale * dx * invR;
            fpy += scale * dy * invR;
            const viscScale = mass * mu / rho_j * cVisc * (h - r);
            fvx += viscScale * (vx[j] - vxi);
            fvy += viscScale * (vy[j] - vyi);
          }
        }
      }

      const fgx = gravX * rho_i;
      const fgy = gravY * rho_i;

      // External forces (with vortex support)
      let fex = 0, fey = 0, heatDelta = 0;
      for (let s = 0; s < sources.length; s++) {
        const src = sources[s];
        const sdx = src.x - xi;
        const sdy = src.y - yi;
        const sd = Math.sqrt(sdx * sdx + sdy * sdy);
        if (sd < src.radius && sd > 1e-4) {
          const w = (src.radius - sd) / src.radius;
          const invD = 1 / sd;
          // Radial force
          fex += src.strength * w * sdx * invD * rho_i;
          fey += src.strength * w * sdy * invD * rho_i;
          // Vortex (perpendicular) force
          if (src.vortex) {
            fex += src.vortex * w * (-sdy * invD) * rho_i;
            fey += src.vortex * w * (sdx * invD) * rho_i;
          }
          heatDelta = Math.max(heatDelta, w * 0.12);
        }
      }

      const invRho = 1 / rho_i;
      const ax = (fpx + fvx + fgx + fex) * invRho;
      const ay = (fpy + fvy + fgy + fey) * invRho;
      let nvx = vxi + ax * dt;
      let nvy = vyi + ay * dt;
      let npx = xi + nvx * dt;
      let npy = yi + nvy * dt;

      const br = 2;
      if (npx < br) { npx = br; nvx = Math.abs(nvx) * damp; }
      if (npx > W - br) { npx = W - br; nvx = -Math.abs(nvx) * damp; }
      if (npy < br) { npy = br; nvy = Math.abs(nvy) * damp; }
      if (npy > H - br) { npy = H - br; nvy = -Math.abs(nvy) * damp; }

      vx[i] = nvx;
      vy[i] = nvy;
      px[i] = npx;
      py[i] = npy;

      const speed = Math.sqrt(nvx * nvx + nvy * nvy);
      const target = Math.min(speed * 0.004 + heatDelta, 1.0);
      this.temp[i] = this.temp[i] * 0.94 + target * 0.06;
    }
  }

  step(substeps = 3): void {
    for (let s = 0; s < substeps; s++) {
      this.buildGrid();
      this.computeDensityPressure();
      this.computeForcesIntegrate();
    }
  }
}
