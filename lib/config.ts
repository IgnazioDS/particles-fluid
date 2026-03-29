/** Simulation & rendering configuration. */

export const CFG = {
  // Particle count (default — user can adjust via slider)
  numParticles: 800,

  // SPH physics
  smoothingH: 26,
  restDensity: 280,
  pressureK: 200,
  viscosity: 0.55,
  gravity: 200,
  dt: 0.008,
  substeps: 2,
  particleMass: 1.0,
  boundaryDamp: 0.4,

  // Interaction strengths
  attractStr: 900,
  repelStr: -1200,
  vortexStr: 800,
  gravityWellStr: 1400,
  handRadius: 90,

  // WebGL rendering
  pointSize: 18.0,      // base gl_PointSize for particles
  trailFade: 0.06,      // trail persistence (lower = longer trails)

  // Post-processing (UnrealBloom)
  bloomStrength: 0.8,
  bloomRadius: 0.5,
  bloomThreshold: 0.15,

  // Shape formation
  formationRadius: 200,
  formationNPoints: 380,
  formationHoldS: 3.2,
  formationSpringK: 280,
  formationMaxForce: 600,
} as const;
