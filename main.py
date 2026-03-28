#!/usr/bin/env python3
"""
main.py — Particles Fluid
=========================
Interactive 2-D SPH fluid simulation driven by hand and face gestures.

Keyboard shortcuts
------------------
  Q / ESC  quit
  R        reset simulation
  G        toggle gravity on/off
  W        toggle webcam overlay
  D        toggle debug overlay

Gesture → force mapping
-----------------------
  open hand   → attract particles to palm
  fist        → explosive repel from palm
  point       → directional push along finger axis
  pinch       → gravity-well vortex at pinch point
  peace (✌)  → dual attract at index + middle tips
  spider      → split attract at index + pinky tips
  mouth open  → blows particles outward (scales with openness)
  head tilt   → tilts gravity axis left / right
"""

import sys
import time
import argparse

import cv2
import numpy as np
import taichi as ti

from config import Config
from src.simulation.sph import SPHSolver
from src.tracking.tracker import Tracker
from src.rendering.renderer import Renderer
from src.shapes.formation import FormationManager, FormationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Gesture → force-source mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_force_sources(tracker: Tracker, cfg: Config) -> list:
    """
    Translate tracked hand/face gestures into a list of
    (x, y, strength, radius) tuples for the SPH solver.

    strength > 0  →  attract
    strength < 0  →  repel
    """
    sources = []

    for hand in tracker.hand_states:
        px, py = tracker.norm_to_sim(*hand.palm_center)

        if hand.name == "open":
            # Full palm attracting everything within hand_radius
            sources.append((px, py, cfg.hand_attract_str, cfg.hand_radius))

        elif hand.name == "fist":
            # Explosive repel — large radius, high negative strength
            sources.append((px, py, cfg.hand_repel_str, cfg.hand_radius * 1.35))

        elif hand.name == "point":
            # Attract to fingertip + repel from wrist → net directional push
            tip_x, tip_y = tracker.norm_to_sim(*hand.finger_tips[1])
            wx,    wy    = tracker.norm_to_sim(*hand.wrist)
            sources.append((tip_x, tip_y,  cfg.point_ray_str,          cfg.hand_radius * 0.70))
            sources.append((wx,    wy,     cfg.hand_repel_str * 0.35,  cfg.hand_radius * 0.50))

        elif hand.name == "pinch":
            # Gravity-well at midpoint between thumb and index tips
            tx, ty = tracker.norm_to_sim(*hand.finger_tips[0])
            ix, iy = tracker.norm_to_sim(*hand.finger_tips[1])
            sources.append(((tx+ix)/2, (ty+iy)/2,
                             cfg.hand_vortex_str, cfg.hand_radius * 0.60))

        elif hand.name == "peace":
            # Gentle attract to both extended tips independently
            for tip in hand.finger_tips[1:3]:   # index, middle
                tx, ty = tracker.norm_to_sim(*tip)
                sources.append((tx, ty,
                                 cfg.hand_attract_str * 0.55, cfg.hand_radius * 0.60))

        elif hand.name == "spider":
            # Two separate strong attractors (index + pinky spread)
            for tip in [hand.finger_tips[1], hand.finger_tips[4]]:
                tx, ty = tracker.norm_to_sim(*tip)
                sources.append((tx, ty,
                                 cfg.hand_attract_str * 0.75, cfg.hand_radius * 0.65))

        elif hand.name == "partial":
            # Partial fold → gentle palm influence
            sources.append((px, py,
                             cfg.hand_attract_str * 0.30, cfg.hand_radius * 0.80))

    # ── Face gestures ─────────────────────────────────────────────────────────
    if tracker.face_state is not None:
        face = tracker.face_state
        if face.mouth_open:
            mx, my = tracker.norm_to_sim(*face.mouth_center)
            # Scale repel strength linearly with mouth openness
            sources.append((mx, my,
                             cfg.face_blow_str * face.mouth_openness,
                             cfg.face_blow_radius))

    return sources


# ─────────────────────────────────────────────────────────────────────────────
# Debug info builder
# ─────────────────────────────────────────────────────────────────────────────

def build_debug_info(
    tracker:     Tracker,
    fps:         float,
    n_particles: int,
    sources:     list,
) -> dict:
    info: dict = {
        "fps":      fps,
        "n":        n_particles,
        "gestures": [],
        "rings":    [],
    }

    for hand in tracker.hand_states:
        px, py = tracker.norm_to_sim(*hand.palm_center)
        info["gestures"].append((
            f"[{hand.handedness[0]}] {hand.name}",
            (px, py - 20),
        ))

    if tracker.face_state is not None and tracker.face_state.mouth_open:
        mx, my = tracker.norm_to_sim(*tracker.face_state.mouth_center)
        info["gestures"].append((
            f"mouth {tracker.face_state.mouth_openness:.2f}",
            (mx, my + 30),
        ))

    for (x, y, s, r) in sources:
        info["rings"].append((x, y, r, s < 0))

    return info


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Particles Fluid — Interactive SPH")
    parser.add_argument("--no-webcam",  action="store_true",
                        help="Disable webcam; use mouse for interaction")
    parser.add_argument("--particles",  type=int, default=None,
                        help="Override particle count from config")
    parser.add_argument("--gpu",        action="store_true",
                        help="Use Taichi GPU backend (requires CUDA/Metal)")
    args = parser.parse_args()

    # ── Taichi init  (must happen before ANY ti.field is created) ─────────────
    if args.gpu:
        ti.init(arch=ti.gpu, device_memory_GB=1.0)
        print("[taichi] GPU backend")
    else:
        ti.init(arch=ti.cpu)
        print("[taichi] CPU backend")

    cfg = Config()
    if args.particles:
        cfg.num_particles = args.particles

    # ── Build components ──────────────────────────────────────────────────────
    solver = SPHSolver(
        n       = cfg.num_particles,
        width   = float(cfg.width),
        height  = float(cfg.height),
        h       = cfg.smoothing_h,
        rho0    = cfg.rest_density,
        k       = cfg.pressure_k,
        mu      = cfg.viscosity,
        gravity = cfg.gravity,
        dt      = cfg.dt,
        mass    = cfg.particle_mass,
        damp    = cfg.boundary_damp,
        max_src = cfg.max_force_sources,
    )
    solver.initialize()

    use_webcam = not args.no_webcam
    tracker    = Tracker(
        webcam_index = cfg.webcam_index,
        sim_w        = cfg.width,
        sim_h        = cfg.height,
    )
    if use_webcam and not tracker.available:
        print("[WARNING] Webcam not found — running in mouse-demo mode")
        use_webcam = False

    renderer = Renderer(
        width           = cfg.width,
        height          = cfg.height,
        particle_radius = cfg.particle_radius,
        glow_sigma      = cfg.glow_sigma,
        bloom_sigma     = cfg.bloom_sigma,
        glow_weight     = cfg.glow_weight,
        bloom_weight    = cfg.bloom_weight,
        trail_fade      = cfg.trail_fade,
        bg_color        = cfg.bg_color,
    )

    # ── Formation system ──────────────────────────────────────────────────────
    formation_cfg = FormationConfig(
        radius=cfg.formation_radius,
        n_points=cfg.formation_n_points,
        hold_s=cfg.formation_hold_s,
        spring_k=cfg.formation_spring_k,
        max_force=cfg.formation_max_force,
        font_scale=cfg.formula_font_scale,
        font_color=cfg.formula_color,
        shadow_color=cfg.formula_shadow,
    )
    formation = FormationManager(
        sim_w=cfg.width,
        sim_h=cfg.height,
        cfg=formation_cfg,
        get_positions_fn=lambda: solver.get_render_data()[0],
    )

    # ── OpenCV window ─────────────────────────────────────────────────────────
    cv2.namedWindow(cfg.window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window_title, cfg.width, cfg.height)

    # Mouse fallback (when no webcam)
    mouse_x, mouse_y   = cfg.width // 2, cfg.height // 2
    mouse_left_down    = False
    mouse_right_down   = False

    def on_mouse(event, x, y, flags, _param):
        nonlocal mouse_x, mouse_y, mouse_left_down, mouse_right_down
        mouse_x, mouse_y  = x, y
        mouse_left_down   = bool(flags & cv2.EVENT_FLAG_LBUTTON)
        mouse_right_down  = bool(flags & cv2.EVENT_FLAG_RBUTTON)

    cv2.setMouseCallback(cfg.window_title, on_mouse)

    # ── Runtime state ─────────────────────────────────────────────────────────
    show_webcam  = cfg.show_webcam and use_webcam
    show_debug   = cfg.show_debug
    gravity_on   = True
    sources: list= []
    fps_count    = 0
    fps_t0       = time.perf_counter()
    fps_display  = 0.0

    print(f"\n{'─'*55}")
    print(f"  Particles Fluid  —  {cfg.num_particles} particles  {cfg.width}×{cfg.height}")
    print(f"  Webcam: {'ON' if use_webcam else 'OFF (mouse demo mode)'}")
    print(f"  Q/ESC=quit  R=reset  G=gravity  W=webcam  D=debug  F=formation")
    print(f"{'─'*55}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    while True:
        t_frame = time.perf_counter()

        # ── Tracking ──────────────────────────────────────────────────────────
        webcam_frame = None
        if use_webcam:
            tracker.read(draw_landmarks=False)
            webcam_frame = tracker.frame
            sources      = build_force_sources(tracker, cfg)

            if gravity_on and tracker.face_state is not None:
                tilt = max(-cfg.head_tilt_max_angle,
                           min( cfg.head_tilt_max_angle,
                                tracker.face_state.head_tilt))
                solver.set_gravity_tilt(tilt)

        else:
            # Mouse fallback
            if mouse_left_down:
                sources = [(mouse_x, mouse_y, cfg.hand_attract_str, cfg.hand_radius)]
            elif mouse_right_down:
                sources = [(mouse_x, mouse_y, cfg.hand_repel_str, cfg.hand_radius)]
            else:
                sources = [(mouse_x, mouse_y, cfg.hand_attract_str * 0.15, cfg.hand_radius * 0.6)]

        # ── Formation forces ─────────────────────────────────────────────────
        dt = 1.0 / cfg.fps_target
        face_for_formation = tracker.face_state if use_webcam else None
        extra_sources = formation.update(face_for_formation, dt)
        all_sources = sources + extra_sources

        solver.set_force_sources(all_sources)

        # ── Physics ───────────────────────────────────────────────────────────
        solver.step(substeps=cfg.substeps)

        # ── Render ────────────────────────────────────────────────────────────
        positions, temps = solver.get_render_data()

        debug_info = None
        if show_debug:
            if use_webcam:
                debug_info = build_debug_info(tracker, fps_display,
                                              cfg.num_particles, all_sources)
            else:
                debug_info = {
                    "fps":      fps_display,
                    "n":        cfg.num_particles,
                    "gestures": [("mouse", (mouse_x, mouse_y - 20))],
                    "rings":    [(mouse_x, mouse_y,
                                  cfg.hand_radius,
                                  mouse_right_down)],
                }

        frame = renderer.render(
            positions,
            temps,
            webcam            = webcam_frame if show_webcam else None,
            webcam_alpha      = cfg.webcam_alpha,
            debug_info        = debug_info,
            formation_overlay = formation.get_render_overlay(),
        )

        cv2.imshow(cfg.window_title, frame)

        # ── FPS counter ───────────────────────────────────────────────────────
        fps_count += 1
        now = time.perf_counter()
        if now - fps_t0 >= 1.0:
            fps_display = fps_count / (now - fps_t0)
            fps_count   = 0
            fps_t0      = now

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            solver.initialize()
            renderer.reset()
            print("[reset]")
        elif key == ord('g'):
            gravity_on = not gravity_on
            if gravity_on:
                solver.enable_gravity()
            else:
                solver.disable_gravity()
            print(f"[gravity] {'ON' if gravity_on else 'OFF'}")
        elif key == ord('w'):
            if use_webcam:
                show_webcam = not show_webcam
                print(f"[webcam overlay] {'ON' if show_webcam else 'OFF'}")
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('f'):
            on = formation.toggle()
            print(f"[formation] {'ON' if on else 'OFF'}")

        # #DEBUG — press 1-9 to directly trigger each shape for testing
        _DEBUG_SHAPES = {
            ord('1'): "circle",
            ord('2'): "ellipse",
            ord('3'): "triangle",
            ord('4'): "square",
            ord('5'): "pentagon",
            ord('6'): "hexagon",
            ord('7'): "rose",
            ord('8'): "spiral",
            ord('9'): "lissajous",
        }
        if key in _DEBUG_SHAPES:
            shape = _DEBUG_SHAPES[key]
            kwargs = {}
            if shape == "rose":
                kwargs["n_petals"] = 3
            elif shape == "ellipse":
                kwargs["axis_ratio"] = 1.5
            formation.force_shape(shape, **kwargs)
            print(f"[formation] forced → {shape}")
        # #END DEBUG

        # ── Frame-rate cap ────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t_frame
        target  = 1.0 / cfg.fps_target
        if elapsed < target:
            time.sleep(target - elapsed)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if use_webcam:
        tracker.close()
    cv2.destroyAllWindows()
    print("Goodbye! 👋")


if __name__ == "__main__":
    main()
