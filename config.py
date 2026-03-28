"""
config.py — Global configuration for Particles Fluid.
Tweak these values to customise behaviour and performance.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ── Window / Display ─────────────────────────────────────────────────────
    width:            int   = 1280
    height:           int   = 720
    fps_target:       int   = 60
    window_title:     str   = "Particles Fluid"
    show_webcam:      bool  = True    # overlay webcam feed behind particles
    webcam_alpha:     float = 0.35    # webcam visibility  [0=hidden, 1=full]
    show_debug:       bool  = True    # FPS, gesture labels, force rings

    # ── Webcam ────────────────────────────────────────────────────────────────
    webcam_index:     int   = 0       # change if wrong camera opens

    # ── Simulation ────────────────────────────────────────────────────────────
    num_particles:    int   = 1500
    smoothing_h:      float = 26.0    # SPH smoothing radius (pixels)
    rest_density:     float = 280.0   # target rest density  ρ₀
    pressure_k:       float = 200.0   # pressure stiffness constant
    viscosity:        float = 0.55    # viscosity coefficient μ
    gravity:          float = 200.0   # gravitational acceleration (px/s²)
    dt:               float = 0.008   # time step (seconds)
    substeps:         int   = 3       # physics substeps per frame
    particle_mass:    float = 1.0
    boundary_damp:    float = 0.4     # velocity retention on wall bounce

    # ── Interaction / Force Sources ───────────────────────────────────────────
    max_force_sources:    int   = 32
    hand_attract_str:     float = 900.0
    hand_repel_str:       float = -1200.0
    hand_vortex_str:      float = 700.0    # pinch gesture
    hand_radius:          float = 90.0
    face_blow_str:        float = -800.0   # repel from open mouth
    face_blow_radius:     float = 130.0
    point_ray_str:        float = 600.0
    head_tilt_max_angle:  float = 0.45     # max gravity tilt from head (radians)

    # ── Rendering ─────────────────────────────────────────────────────────────
    particle_radius:  int   = 4
    glow_sigma:       float = 7.0
    bloom_sigma:      float = 22.0
    glow_weight:      float = 1.3
    bloom_weight:     float = 0.45
    trail_fade:       int   = 18     # background fade per frame (0-255)
    bg_color:         tuple = (6, 4, 10)  # BGR, used when webcam is off

    # ── Shape Formation ─────────────────────────────────────────────────────
    formation_radius:     float = 200.0   # shape size in pixels
    formation_n_points:   int   = 320     # particles recruited per shape
    formation_hold_s:     float = 3.0     # seconds to hold before dissolve
    formation_spring_k:   float = 280.0
    formation_max_force:  float = 600.0
    formula_font_scale:   float = 0.90    # cv2 font scale for formula text
    formula_color:        tuple = (240, 240, 240)   # BGR near-white
    formula_shadow:       tuple = (20, 20, 20)       # thin dark shadow 1 px offset
