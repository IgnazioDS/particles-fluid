/** Simulation & rendering configuration. */

export const CFG = {
  // Particle count — tuned for 60 fps in-browser
  numParticles: 800,

  // SPH physics
  smoothingH: 26,
  restDensity: 280,
  pressureK: 200,
  viscosity: 0.55,
  gravity: 200,
  dt: 0.008,
  substeps: 3,
  particleMass: 1.0,
  boundaryDamp: 0.4,

  // Interaction
  attractStr: 900,
  repelStr: -1200,
  handRadius: 90,

  // Rendering
  particleRadius: 3,
  glowSigma: 7,
  bloomSigma: 22,
  glowAlpha: 0.55,
  bloomAlpha: 0.25,
  trailFade: 0.07, // alpha of bg fill per frame (lower = longer trails)
  bgColor: "#060410",
  bgRGB: [6, 4, 10] as const,

  // Shape formation
  formationRadius: 200,
  formationNPoints: 280,
  formationHoldS: 3.0,
  formationSpringK: 280,
  formationMaxForce: 600,
} as const;

// Pre-build 256-entry heat colour LUT (RGB strings for Canvas fillStyle)
export const HEAT_LUT: string[] = [];
export const HEAT_RGB: [number, number, number][] = [];

for (let i = 0; i < 256; i++) {
  const t = i / 255;
  let r: number, g: number, b: number;
  if (t < 0.2) {
    const s = t / 0.2;
    r = s * 150;
    g = s * 4;
    b = s * 18;
  } else if (t < 0.45) {
    const s = (t - 0.2) / 0.25;
    r = 150 + s * 90;
    g = 4 + s * 65;
    b = 18 * (1 - s);
  } else if (t < 0.72) {
    const s = (t - 0.45) / 0.27;
    r = 240 + s * 10;
    g = 69 + s * 141;
    b = 0;
  } else {
    const s = (t - 0.72) / 0.28;
    r = 250 + s * 5;
    g = 210 + s * 45;
    b = s * 120;
  }
  r = Math.min(255, Math.max(0, Math.round(r)));
  g = Math.min(255, Math.max(0, Math.round(g)));
  b = Math.min(255, Math.max(0, Math.round(b)));
  HEAT_LUT.push(`rgb(${r},${g},${b})`);
  HEAT_RGB.push([r, g, b]);
}
