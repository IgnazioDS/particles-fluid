/** Simulation & rendering configuration. */

export const CFG = {
  // Particle count (default — user can adjust via slider)
  numParticles: 1000,

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

  // Interaction strengths
  attractStr: 900,
  repelStr: -1200,
  vortexStr: 800,
  gravityWellStr: 1400,
  handRadius: 90,

  // Rendering — multi-layer particle drawing
  particleRadius: 2.5,   // core circle
  glowRadius: 7,         // medium glow
  bloomRadius: 14,        // wide bloom
  coreAlpha: 0.85,
  glowAlpha: 0.08,
  bloomAlpha: 0.03,

  // Post-processing (canvas filter blur)
  postGlowSigma: 6,
  postBloomSigma: 18,
  postGlowAlpha: 0.5,
  postBloomAlpha: 0.2,
  trailFade: 0.06, // alpha of bg fill per frame (lower = longer trails)

  // Shape formation
  formationRadius: 200,
  formationNPoints: 380,
  formationHoldS: 3.2,
  formationSpringK: 280,
  formationMaxForce: 600,
} as const;
