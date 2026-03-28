"""
src/simulation/sph.py
=====================
2-D Smoothed Particle Hydrodynamics (SPH) solver using Taichi.

Mathematical kernels (Müller et al. 2003):
  • Poly6    — density estimation       W(r,h)
  • Spiky    — pressure gradient        ∇W(r,h)  (no vanishing centre gradient)
  • Viscosity Laplacian — vel diffusion ∇²W(r,h)

The outer particle loop is parallelised by Taichi (CPU SIMD or GPU).
The inner neighbour loop is sequential — O(n²) total but fast in practice
because ~99 % of pairs are culled by the r < h distance check.
"""

import math
import taichi as ti
import numpy as np


@ti.data_oriented
class SPHSolver:
    """
    GPU/CPU-parallel 2-D SPH fluid solver.

    Key public methods
    ------------------
    initialize()             — scatter particles, reset velocities
    step(substeps)           — advance simulation by `substeps` time steps
    set_force_sources(list)  — update (x,y,strength,radius) external forces
    set_gravity_tilt(angle)  — tilt gravity direction (head pose)
    get_render_data()        — return (positions, temperatures) as numpy arrays
    """

    def __init__(
        self,
        n:        int,
        width:    float,
        height:   float,
        h:        float  = 26.0,
        rho0:     float  = 280.0,
        k:        float  = 200.0,
        mu:       float  = 0.55,
        gravity:  float  = 200.0,
        dt:       float  = 0.008,
        mass:     float  = 1.0,
        damp:     float  = 0.4,
        max_src:  int    = 32,
    ):
        self.n      = n
        self.W      = width
        self.H      = height
        self.h      = h
        self.h2     = h * h
        self.rho0   = rho0
        self.k      = k
        self.mu     = mu
        self._gravity = gravity   # base magnitude
        self.dt     = dt
        self.mass   = mass
        self.damp   = damp
        self.max_src = max_src

        # Kernel normalisation constants (2-D)
        pi = math.pi
        self.c_poly6 =  4.0  / (pi * h ** 8)
        self.c_spiky = -30.0 / (pi * h ** 5)
        self.c_visc  =  40.0 / (pi * h ** 5)

        # ── Particle state ───────────────────────────────────────────────────
        self.pos  = ti.Vector.field(2, ti.f32, n)
        self.vel  = ti.Vector.field(2, ti.f32, n)
        self.rho  = ti.field(ti.f32, n)
        self.pres = ti.field(ti.f32, n)
        self.temp = ti.field(ti.f32, n)   # visual heat [0, 1]

        # ── External force sources ───────────────────────────────────────────
        self.n_src   = ti.field(ti.i32, shape=())
        self.src_pos = ti.Vector.field(2, ti.f32, max_src)
        self.src_str = ti.field(ti.f32, max_src)   # +attract / −repel
        self.src_rad = ti.field(ti.f32, max_src)

        # ── Gravity (can be tilted via head pose) ────────────────────────────
        self.grav_x = ti.field(ti.f32, shape=())
        self.grav_y = ti.field(ti.f32, shape=())

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def initialize(self):
        """Reset simulation: scatter particles uniformly, zero velocities."""
        self.grav_x[None] = 0.0
        self.grav_y[None] = float(self._gravity)
        self.n_src[None]  = 0
        self._init_kernel()

    @ti.kernel
    def _init_kernel(self):
        for i in self.pos:
            self.pos[i]  = ti.Vector([
                ti.random() * self.W,
                ti.random() * self.H * 0.8 + self.H * 0.1,
            ])
            self.vel[i]  = ti.Vector([0.0, 0.0])
            self.temp[i] = ti.random() * 0.15

    # ─────────────────────────────────────────────────────────────────────────
    # Kernel functions  (called inside @ti.kernel, inline-expanded by Taichi)
    # ─────────────────────────────────────────────────────────────────────────

    @ti.func
    def _poly6(self, r2: ti.f32) -> ti.f32:
        v = 0.0
        if r2 < self.h2:
            d = self.h2 - r2
            v = self.c_poly6 * d * d * d
        return v

    @ti.func
    def _spiky_grad(self, r_vec: ti.template()) -> ti.template():
        r = r_vec.norm()
        g = ti.Vector([0.0, 0.0])
        if r > 1e-4 and r < self.h:
            d = self.h - r
            g = self.c_spiky * d * d * (r_vec / r)
        return g

    @ti.func
    def _visc_lap(self, r: ti.f32) -> ti.f32:
        v = 0.0
        if r < self.h:
            v = self.c_visc * (self.h - r)
        return v

    # ─────────────────────────────────────────────────────────────────────────
    # Physics kernels
    # ─────────────────────────────────────────────────────────────────────────

    @ti.kernel
    def _compute_density_pressure(self):
        for i in range(self.n):
            rho_i = 0.0
            for j in range(self.n):
                r_vec = self.pos[i] - self.pos[j]
                rho_i += self.mass * self._poly6(r_vec.dot(r_vec))
            self.rho[i]  = rho_i
            # Tait equation of state (clamped to avoid tensile instability)
            self.pres[i] = ti.max(self.k * (rho_i - self.rho0), 0.0)

    @ti.kernel
    def _compute_forces_integrate(self):
        n_s = self.n_src[None]

        for i in range(self.n):
            f_pres = ti.Vector([0.0, 0.0])
            f_visc = ti.Vector([0.0, 0.0])
            rho_i  = ti.max(self.rho[i], 1e-3)
            pres_i = self.pres[i]
            vel_i  = self.vel[i]
            pos_i  = self.pos[i]

            for j in range(self.n):
                if i == j:
                    continue
                r_vec  = pos_i - self.pos[j]
                r      = r_vec.norm()
                rho_j  = ti.max(self.rho[j], 1e-3)

                if r < self.h and r > 1e-4:
                    # Symmetric pressure (conserves momentum exactly)
                    f_pres += (
                        -self.mass * (pres_i + self.pres[j]) / (2.0 * rho_j)
                        * self._spiky_grad(r_vec)
                    )
                    # Viscosity
                    f_visc += (
                        self.mass * self.mu * (self.vel[j] - vel_i)
                        / rho_j * self._visc_lap(r)
                    )

            # Gravity (direction may be tilted by head pose)
            f_grav = ti.Vector([self.grav_x[None], self.grav_y[None]]) * rho_i

            # External force sources (hands / face)
            f_ext      = ti.Vector([0.0, 0.0])
            heat_delta = 0.0
            for s in range(n_s):
                d_vec  = self.src_pos[s] - pos_i
                d      = d_vec.norm()
                r_src  = self.src_rad[s]
                if d < r_src and d > 1e-4:
                    w       = (r_src - d) / r_src          # 1 at source, 0 at edge
                    f_ext  += self.src_str[s] * w * (d_vec / d) * rho_i
                    heat_delta = ti.max(heat_delta, w * 0.12)

            # ── Semi-implicit Euler integration ───────────────────────────────
            acc     = (f_pres + f_visc + f_grav + f_ext) / rho_i
            new_vel = vel_i + acc * self.dt
            new_pos = pos_i + new_vel * self.dt

            # ── Boundary conditions ───────────────────────────────────────────
            br = 2.0   # particle boundary offset
            if new_pos[0] < br:
                new_pos[0] = br
                new_vel[0] = ti.abs(new_vel[0]) * self.damp
            if new_pos[0] > self.W - br:
                new_pos[0] = self.W - br
                new_vel[0] = -ti.abs(new_vel[0]) * self.damp
            if new_pos[1] < br:
                new_pos[1] = br
                new_vel[1] = ti.abs(new_vel[1]) * self.damp
            if new_pos[1] > self.H - br:
                new_pos[1] = self.H - br
                new_vel[1] = -ti.abs(new_vel[1]) * self.damp

            self.vel[i] = new_vel
            self.pos[i] = new_pos

            # ── Visual temperature (drives particle colour) ───────────────────
            speed  = new_vel.norm()
            target = ti.min(speed * 0.004 + heat_delta, 1.0)
            self.temp[i] = self.temp[i] * 0.94 + target * 0.06

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, substeps: int = 3):
        """Advance simulation by `substeps` time steps."""
        for _ in range(substeps):
            self._compute_density_pressure()
            self._compute_forces_integrate()

    def set_force_sources(self, sources: list):
        """
        sources : list of (x, y, strength, radius) tuples.
          strength > 0  → attract particles toward source
          strength < 0  → repel particles away from source
        """
        n = min(len(sources), self.max_src)
        self.n_src[None] = n
        for idx, (x, y, s, rad) in enumerate(sources[:n]):
            self.src_pos[idx] = [float(x), float(y)]
            self.src_str[idx] = float(s)
            self.src_rad[idx] = float(rad)

    def set_gravity_tilt(self, angle_rad: float):
        """Tilt gravity direction by `angle_rad` radians from vertical."""
        g = float(self._gravity)
        self.grav_x[None] = math.sin(angle_rad) * g
        self.grav_y[None] = math.cos(angle_rad) * g

    def disable_gravity(self):
        self.grav_x[None] = 0.0
        self.grav_y[None] = 0.0

    def enable_gravity(self):
        self.grav_x[None] = 0.0
        self.grav_y[None] = float(self._gravity)

    def get_render_data(self):
        """Return (positions, temperatures) as numpy float32 arrays."""
        return self.pos.to_numpy(), self.temp.to_numpy()
