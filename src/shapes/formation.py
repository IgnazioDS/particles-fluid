"""
src/shapes/formation.py
=======================
State machine that drives SPH particles toward geometric shape formations.

States:  IDLE -> DETECTING -> FORMING -> HOLDING -> DISSOLVING -> IDLE

The manager never touches SPH internals directly.  It returns extra force
sources (x, y, strength, radius) that the main loop injects via the existing
solver.set_force_sources() API.
"""

import math
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .lattice import ShapeLattice, FORMULA_STRINGS


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (injected from Config at construction time)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FormationConfig:
    radius:       float = 200.0
    n_points:     int   = 320
    hold_s:       float = 3.0
    spring_k:     float = 280.0
    max_force:    float = 600.0
    font_scale:   float = 0.90
    font_color:   tuple = (240, 240, 240)
    shadow_color: tuple = (20, 20, 20)


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE       = auto()
    DETECTING  = auto()
    FORMING    = auto()
    HOLDING    = auto()
    DISSOLVING = auto()


# Gesture → shape priority (highest first)
_SHAPE_PRIORITY = [
    "lissajous", "spiral", "rose", "hexagon", "pentagon",
    "square", "ellipse", "triangle", "circle",
]


def _classify_gesture(face) -> Optional[Tuple[str, dict]]:
    """
    Map face gesture state to (shape_name, extra_kwargs).
    Returns None if no shape should trigger.

    Priority: lissajous > spiral > rose > hexagon > pentagon >
              square > ellipse > triangle > circle
    """
    if face is None:
        return None

    mouth_active = face.mouth_open and face.mouth_openness > 0.35
    brow_active = face.brow_raised
    tilt = face.head_tilt
    head_nod = getattr(face, "head_nod", False)

    # Check combined gestures first (highest priority)
    if mouth_active and brow_active:
        return ("lissajous", {})

    if brow_active:
        return ("spiral", {})

    if mouth_active:
        n_petals = max(1, round(face.mouth_openness * 8))
        return ("rose", {"n_petals": n_petals})

    if head_nod:
        return ("hexagon", {})

    # Head tilt zones
    if tilt > 0.22:
        return ("pentagon", {})
    if tilt < -0.22:
        return ("square", {})
    if 0.08 <= tilt <= 0.22:
        axis_ratio = 1.0 + abs(tilt) * 4
        return ("ellipse", {"axis_ratio": axis_ratio})
    if -0.22 <= tilt <= -0.08:
        return ("circle", {})

    # Upright (|tilt| < 0.08)
    if abs(tilt) < 0.08:
        return ("triangle", {})

    return None


class FormationManager:
    """
    Per-frame controller for the geometric shape formation system.

    Call `update()` every frame with the current face gesture and delta time.
    It returns a list of extra (x, y, strength, radius) force sources to
    append to the SPH solver's force list.

    Call `get_render_overlay()` to get data for the renderer (or None when idle).
    """

    def __init__(
        self,
        sim_w: int,
        sim_h: int,
        cfg: FormationConfig,
        get_positions_fn=None,
    ):
        self.sim_w = sim_w
        self.sim_h = sim_h
        self.cfg = cfg

        # Callable that returns (n,2) current particle positions.
        # Set by main.py after solver is created.
        self._get_positions = get_positions_fn

        self._state = State.IDLE
        self._enabled = True

        # Timing accumulators
        self._detect_timer = 0.0     # time gesture has been stable
        self._form_timer = 0.0       # time spent in FORMING
        self._hold_timer = 0.0       # time spent in HOLDING
        self._dissolve_timer = 0.0   # time spent in DISSOLVING

        # Current gesture tracking
        self._current_shape: Optional[str] = None
        self._current_kwargs: dict = {}
        self._prev_tilt: Optional[float] = None

        # Lattice & assignment
        self._lattice_pts: Optional[np.ndarray] = None
        self._slot_indices: Optional[np.ndarray] = None  # particle indices assigned to slots

        # Spring ramp
        self._spring_k = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self):
        self._enabled = not self._enabled
        if not self._enabled:
            self._reset_to_idle()
        return self._enabled

    # ─────────────────────────────────────────────────────────────────────────
    # Main per-frame update
    # ─────────────────────────────────────────────────────────────────────────

    def update(
        self,
        face_gesture,
        dt: float,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Returns extra force sources for the SPH solver.
        Each element is (x, y, strength, radius).
        """
        if not self._enabled:
            return []

        gesture_result = _classify_gesture(face_gesture)
        shape_name = gesture_result[0] if gesture_result else None
        shape_kwargs = gesture_result[1] if gesture_result else {}

        # Check tilt stability (reset if tilt changes > 0.04 rad)
        current_tilt = face_gesture.head_tilt if face_gesture else 0.0
        tilt_stable = True
        if self._prev_tilt is not None:
            if abs(current_tilt - self._prev_tilt) > 0.04:
                tilt_stable = False
        self._prev_tilt = current_tilt

        sources: List[Tuple[float, float, float, float]] = []

        if self._state == State.IDLE:
            if shape_name and tilt_stable:
                self._state = State.DETECTING
                self._detect_timer = 0.0
                self._current_shape = shape_name
                self._current_kwargs = shape_kwargs

        elif self._state == State.DETECTING:
            gesture_changed = shape_name != self._current_shape
            if gesture_changed or not tilt_stable:
                # Gesture changed or unstable — restart detection
                if shape_name and tilt_stable:
                    self._detect_timer = 0.0
                    self._current_shape = shape_name
                    self._current_kwargs = shape_kwargs
                else:
                    self._reset_to_idle()
                return sources

            self._detect_timer += dt
            # 0.4s detection + 0.6s confirmation = 1.0s total stable
            if self._detect_timer >= 1.0:
                self._enter_forming()

        elif self._state == State.FORMING:
            self._form_timer += dt
            # Ramp spring_k from 0 -> cfg.spring_k over 0.5s
            ramp = min(self._form_timer / 0.5, 1.0)
            self._spring_k = self.cfg.spring_k * ramp

            sources = self._compute_spring_forces()

            # Check if 85% of particles are within 8px of targets
            if self._check_convergence(threshold_px=8.0, fraction=0.85):
                self._state = State.HOLDING
                self._hold_timer = 0.0

            # Also transition if we've been forming for too long (timeout 6s)
            if self._form_timer > 6.0:
                self._state = State.HOLDING
                self._hold_timer = 0.0

        elif self._state == State.HOLDING:
            self._hold_timer += dt
            self._spring_k = self.cfg.spring_k
            sources = self._compute_spring_forces()

            gesture_changed = shape_name != self._current_shape
            if self._hold_timer >= self.cfg.hold_s or gesture_changed:
                self._state = State.DISSOLVING
                self._dissolve_timer = 0.0

        elif self._state == State.DISSOLVING:
            self._dissolve_timer += dt
            # Ramp spring_k from cfg.spring_k -> 0 over 1.2s
            ramp = max(1.0 - self._dissolve_timer / 1.2, 0.0)
            self._spring_k = self.cfg.spring_k * ramp
            sources = self._compute_spring_forces()

            if self._dissolve_timer >= 1.2:
                self._reset_to_idle()

        return sources

    # ─────────────────────────────────────────────────────────────────────────
    # Render overlay
    # ─────────────────────────────────────────────────────────────────────────

    def get_render_overlay(self) -> Optional[dict]:
        """Return overlay data for the renderer, or None when idle."""
        if self._state == State.IDLE or self._state == State.DETECTING:
            return None
        if self._lattice_pts is None or self._current_shape is None:
            return None

        # Compute alpha for fade in/out
        if self._state == State.FORMING:
            alpha = min(self._form_timer / 0.5, 1.0)
        elif self._state == State.HOLDING:
            alpha = 1.0
        elif self._state == State.DISSOLVING:
            alpha = max(1.0 - self._dissolve_timer / 1.2, 0.0)
        else:
            alpha = 0.0

        formula = FORMULA_STRINGS.get(self._current_shape, "")
        cx = self.sim_w / 2
        cy = self.sim_h / 2 + self.cfg.radius + 40

        return {
            "shape_name":  self._current_shape,
            "formula_tex": formula,
            "formula_pos": (int(cx), int(cy)),
            "alpha":       float(alpha),
            "lattice_pts": self._lattice_pts,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Force a specific shape (for debug keyboard triggers)
    # ─────────────────────────────────────────────────────────────────────────

    def force_shape(self, shape_name: str, **kwargs):
        """Immediately start forming the given shape (debug/keyboard trigger)."""
        self._current_shape = shape_name
        self._current_kwargs = kwargs
        self._enter_forming()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_to_idle(self):
        self._state = State.IDLE
        self._detect_timer = 0.0
        self._form_timer = 0.0
        self._hold_timer = 0.0
        self._dissolve_timer = 0.0
        self._current_shape = None
        self._current_kwargs = {}
        self._lattice_pts = None
        self._slot_indices = None
        self._spring_k = 0.0

    def _enter_forming(self):
        """Generate lattice and assign particles to slots."""
        cx = self.sim_w / 2
        cy = self.sim_h / 2
        lattice = ShapeLattice(
            self._current_shape, cx, cy,
            self.cfg.radius, self.cfg.n_points,
            **self._current_kwargs,
        )
        self._lattice_pts = lattice.generate()
        self._assign_slots()
        self._state = State.FORMING
        self._form_timer = 0.0

    def _assign_slots(self):
        """
        Assign each lattice target to the nearest available particle.
        Uses cKDTree for O(n log n) approximate Hungarian assignment.
        """
        if self._get_positions is None or self._lattice_pts is None:
            self._slot_indices = np.arange(self.cfg.n_points)
            return

        positions = self._get_positions()
        n_particles = len(positions)
        n_slots = len(self._lattice_pts)

        if n_particles == 0:
            self._slot_indices = np.zeros(n_slots, dtype=int)
            return

        tree = cKDTree(positions)
        assigned = np.full(n_slots, -1, dtype=int)
        used = set()

        # For each lattice point, find nearest unassigned particle
        # Query k nearest to have fallbacks if closest is taken
        k = min(10, n_particles)
        dists, indices = tree.query(self._lattice_pts, k=k)

        for slot_i in range(n_slots):
            if k == 1:
                candidates = [int(indices[slot_i])]
            else:
                candidates = indices[slot_i].tolist()
            for pidx in candidates:
                if pidx not in used and 0 <= pidx < n_particles:
                    assigned[slot_i] = pidx
                    used.add(pidx)
                    break
            # If all k neighbours were taken, just pick any unused
            if assigned[slot_i] == -1:
                for p in range(n_particles):
                    if p not in used:
                        assigned[slot_i] = p
                        used.add(p)
                        break
                # If truly exhausted, wrap around
                if assigned[slot_i] == -1:
                    assigned[slot_i] = slot_i % n_particles

        self._slot_indices = assigned

    def _compute_spring_forces(
        self,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Generate per-slot spring forces pulling assigned particles to targets.
        Also add a mild centre attractor for free (unassigned) particles.
        """
        sources: List[Tuple[float, float, float, float]] = []

        if self._lattice_pts is None or self._slot_indices is None:
            return sources
        if self._get_positions is None:
            return sources

        positions = self._get_positions()
        n_particles = len(positions)
        k = self._spring_k
        max_f = self.cfg.max_force

        if k < 1e-3:
            return sources

        assigned_set = set(self._slot_indices.tolist())

        # Per-slot spring: pull each assigned particle toward its target
        for slot_i in range(len(self._lattice_pts)):
            pidx = int(self._slot_indices[slot_i])
            if pidx < 0 or pidx >= n_particles:
                continue
            target = self._lattice_pts[slot_i]
            current = positions[pidx]
            dx = target[0] - current[0]
            dy = target[1] - current[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1.0:
                continue
            # F = k * (target - current), clamped to max_force
            fx = k * dx
            fy = k * dy
            f_mag = math.sqrt(fx * fx + fy * fy)
            if f_mag > max_f:
                scale = max_f / f_mag
                fx *= scale
                fy *= scale
            # Express as a force source at the target position attracting
            # with strength proportional to the spring force, radius = dist + 5
            strength = min(f_mag, max_f)
            radius = max(dist + 5.0, 10.0)
            sources.append((
                float(target[0]), float(target[1]),
                float(strength), float(radius),
            ))

        # Mild centre attractor for free particles (strength=60)
        cx = self.sim_w / 2.0
        cy = self.sim_h / 2.0
        sources.append((cx, cy, 60.0, float(self.cfg.radius * 2.5)))

        return sources

    def _check_convergence(self, threshold_px: float, fraction: float) -> bool:
        """Check if enough assigned particles are close to their targets."""
        if self._get_positions is None or self._lattice_pts is None:
            return False
        if self._slot_indices is None:
            return False

        positions = self._get_positions()
        n_particles = len(positions)
        n_close = 0
        n_slots = len(self._lattice_pts)

        for slot_i in range(n_slots):
            pidx = int(self._slot_indices[slot_i])
            if pidx < 0 or pidx >= n_particles:
                continue
            dx = self._lattice_pts[slot_i, 0] - positions[pidx, 0]
            dy = self._lattice_pts[slot_i, 1] - positions[pidx, 1]
            if dx * dx + dy * dy < threshold_px * threshold_px:
                n_close += 1

        return n_close >= fraction * n_slots
