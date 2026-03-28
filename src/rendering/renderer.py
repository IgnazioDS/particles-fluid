"""
src/rendering/renderer.py
=========================
OpenCV-based renderer for the fluid particle simulation.

Visual pipeline
---------------
1. Fade particle buffer  → creates motion trails
2. Draw particles        → cv2.circle per particle
3. Glow pass             → GaussianBlur + additive overlay
4. Bloom pass            → wider GaussianBlur for wide glow halo
5. Composite background  → webcam (dimmed) or solid colour
6. Debug overlay         → FPS, gesture labels, force-source rings

Colour ramp
-----------
particle.temp [0,1] → BGR via a pre-built 256-entry LUT.
Ramp:  black → deep crimson → orange → bright orange → near-white yellow
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Pre-build heat colour LUT  (computed once at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _build_heat_lut() -> np.ndarray:
    """Return a (256, 3) uint8 array mapping heat [0,255] → BGR colour."""
    lut = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0
        if t < 0.20:                          # black → deep crimson
            s = t / 0.20
            lut[i] = [s * 18, s * 4, s * 150]
        elif t < 0.45:                        # crimson → orange-red
            s = (t - 0.20) / 0.25
            lut[i] = [18*(1-s), 4 + s*65, 150 + s*90]
        elif t < 0.72:                        # orange-red → bright orange
            s = (t - 0.45) / 0.27
            lut[i] = [0, 69 + s*141, 240 + s*10]
        else:                                 # bright orange → near-white yellow
            s = (t - 0.72) / 0.28
            lut[i] = [s * 120, 210 + s*45, 250 + s*5]
    return lut.clip(0, 255).astype(np.uint8)


_HEAT_LUT: np.ndarray = _build_heat_lut()   # shape (256, 3), dtype uint8


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────

class Renderer:

    def __init__(
        self,
        width:          int   = 1280,
        height:         int   = 720,
        particle_radius: int  = 4,
        glow_sigma:     float = 7.0,
        bloom_sigma:    float = 22.0,
        glow_weight:    float = 1.3,
        bloom_weight:   float = 0.45,
        trail_fade:     int   = 18,          # fade amount per frame [0-255]
        bg_color:       tuple = (6, 4, 10),  # BGR when no webcam
    ):
        self.W             = width
        self.H             = height
        self.p_rad         = particle_radius
        self.glow_sigma    = glow_sigma
        self.bloom_sigma   = bloom_sigma
        self.glow_weight   = glow_weight
        self.bloom_weight  = bloom_weight
        self.fade_factor   = 1.0 - trail_fade / 255.0
        self.bg_color      = bg_color

        # Persistent float32 buffer — carries trail information between frames
        self._pbuf = np.zeros((height, width, 3), dtype=np.float32)

        # Pre-compute odd kernel sizes from sigma
        self._gkw = self._odd_ksize(glow_sigma * 3)
        self._bkw = self._odd_ksize(bloom_sigma * 3)

    @staticmethod
    def _odd_ksize(v: float) -> tuple:
        k = max(3, int(v) | 1)
        return (k, k)

    # ─────────────────────────────────────────────────────────────────────────

    def render(
        self,
        positions:    np.ndarray,            # (n, 2) float32
        temps:        np.ndarray,            # (n,)   float32  [0,1]
        webcam:       Optional[np.ndarray] = None,
        webcam_alpha: float                = 0.35,
        debug_info:   Optional[dict]       = None,
        formation_overlay: Optional[dict]  = None,
    ) -> np.ndarray:
        """
        Compose and return a full BGR frame (H×W×3, uint8).

        Parameters
        ----------
        positions    : particle positions in simulation pixels
        temps        : particle heat values, drives colour mapping
        webcam       : optional BGR webcam frame (must be exactly H×W)
        webcam_alpha : opacity of the webcam layer [0, 1]
        debug_info   : dict with optional keys:
                         fps, n, gestures [(label, (x,y))], rings [(x,y,r,is_repel)]
        """
        # ── 1. Fade trail buffer ──────────────────────────────────────────────
        self._pbuf *= self.fade_factor

        # ── 2. Draw particles ─────────────────────────────────────────────────
        n = len(positions)
        if n > 0:
            # Map temperature → colour index via LUT
            t_idx  = np.clip((temps * 255).astype(np.int32), 0, 255)
            colors = _HEAT_LUT[t_idx]              # (n, 3) uint8

            x_px = np.clip(positions[:, 0].astype(np.int32), 0, self.W - 1)
            y_px = np.clip(positions[:, 1].astype(np.int32), 0, self.H - 1)

            draw_r = max(1, self.p_rad - 1)
            for i in range(n):
                c = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
                cv2.circle(self._pbuf, (int(x_px[i]), int(y_px[i])), draw_r,
                           c, -1, lineType=cv2.LINE_AA)

        # ── 3. Glow pass ──────────────────────────────────────────────────────
        glow = cv2.GaussianBlur(self._pbuf, self._gkw, self.glow_sigma)
        lit  = self._pbuf + glow * self.glow_weight

        # ── 4. Bloom pass (wide, soft halo) ───────────────────────────────────
        bloom = cv2.GaussianBlur(self._pbuf, self._bkw, self.bloom_sigma)
        lit   = lit + bloom * self.bloom_weight

        lit = np.clip(lit, 0, 255).astype(np.uint8)

        # ── 5. Background composition ─────────────────────────────────────────
        if webcam is not None and webcam.shape[:2] == (self.H, self.W):
            bg = (webcam.astype(np.float32) * webcam_alpha).clip(0, 255).astype(np.uint8)
        else:
            bg = np.full((self.H, self.W, 3), self.bg_color, dtype=np.uint8)

        # Additive blend: particles light up whatever is behind them
        result = cv2.add(bg, lit)

        # ── 6. Formation overlay (lattice ghosts + formula text) ────────────
        if formation_overlay is not None:
            self._draw_formation(result, formation_overlay)

        # ── 7. Debug overlay ──────────────────────────────────────────────────
        if debug_info:
            self._draw_debug(result, debug_info)

        return result

    # ─────────────────────────────────────────────────────────────────────────

    def _draw_formation(self, frame: np.ndarray, overlay: dict):
        """Draw lattice ghost dots and formula text for the formation system."""
        alpha = overlay.get("alpha", 0.0)
        if alpha < 0.01:
            return

        lattice_pts = overlay.get("lattice_pts")
        if lattice_pts is not None and len(lattice_pts) > 0:
            # Translucent circles at target positions
            dot_overlay = frame.copy()
            dot_alpha = 0.25 * alpha
            for pt in lattice_pts:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < self.W and 0 <= y < self.H:
                    cv2.circle(dot_overlay, (x, y), 3,
                               (180, 180, 220), -1, cv2.LINE_AA)
            cv2.addWeighted(dot_overlay, dot_alpha, frame, 1.0 - dot_alpha, 0, frame)

        # Formula text
        formula = overlay.get("formula_tex", "")
        if formula:
            fx, fy = overlay.get("formula_pos", (self.W // 2, self.H // 2))
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = overlay.get("font_scale", 0.90)
            # Measure text to centre it
            (tw, th), _ = cv2.getTextSize(formula, font, font_scale, 1)
            tx = fx - tw // 2
            ty = fy + th // 2

            # Fade colour channels by alpha
            shadow = tuple(int(c * alpha) for c in (20, 20, 20))
            color = tuple(int(c * alpha) for c in (240, 240, 240))

            # Shadow pass (1px offset)
            cv2.putText(frame, formula, (tx + 1, ty + 1),
                        font, font_scale, shadow, 1, cv2.LINE_AA)
            # Main pass
            cv2.putText(frame, formula, (tx, ty),
                        font, font_scale, color, 1, cv2.LINE_AA)

    def _draw_debug(self, frame: np.ndarray, info: dict):
        """Overlay FPS counter, gesture labels, and force-source rings."""
        font   = cv2.FONT_HERSHEY_SIMPLEX
        silver = (200, 200, 200)
        orange = (80, 119, 217)    # BGR ≈ Anthropic orange

        y = 30
        if "fps" in info:
            cv2.putText(frame, f"FPS  {info['fps']:.0f}", (14, y),
                        font, 0.70, silver, 1, cv2.LINE_AA)
            y += 26

        if "n" in info:
            cv2.putText(frame, f"Particles  {info['n']}", (14, y),
                        font, 0.60, silver, 1, cv2.LINE_AA)
            y += 22

        # Gesture labels  (shown near the relevant landmark)
        for (label, pos) in info.get("gestures", []):
            gx, gy = int(pos[0]), int(pos[1])
            tw, th = 160, 28
            # Semi-transparent background pill
            overlay = frame.copy()
            cv2.rectangle(overlay,
                          (gx - tw//2, gy - th - 4),
                          (gx + tw//2, gy - 4), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
            cv2.putText(frame, label, (gx - tw//2 + 8, gy - 10),
                        font, 0.58, orange, 1, cv2.LINE_AA)

        # Force-source influence rings
        for (rx, ry, rad, is_repel) in info.get("rings", []):
            ring_color = (80, 60, 200) if is_repel else (80, 200, 90)
            cv2.circle(frame, (int(rx), int(ry)), int(rad),
                       ring_color, 1, cv2.LINE_AA)
            cv2.circle(frame, (int(rx), int(ry)), 4,
                       ring_color, -1, cv2.LINE_AA)

        # Keyboard shortcuts footer
        hints = [
            "Q=quit  R=reset  G=gravity  W=webcam  D=debug",
            "GESTURES: open=attract  fist=repel  point=push  pinch=vortex  mouth=blow",
        ]
        for i, line in enumerate(hints):
            yh = frame.shape[0] - 14 - i * 22
            cv2.putText(frame, line, (10, yh), font, 0.48,
                        (90, 90, 90), 1, cv2.LINE_AA)

    def reset(self):
        """Clear the particle trail buffer."""
        self._pbuf[:] = 0.0
