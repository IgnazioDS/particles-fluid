"""
src/tracking/gesture.py
=======================
Pure-Python gesture classification from MediaPipe landmark lists.
All input coordinates are normalised [0, 1].

Hand gestures detected
----------------------
  open    — all 4 fingers extended  → attract to palm
  fist    — all fingers folded      → explosive repel
  point   — only index extended     → directional push
  pinch   — thumb+index close       → vortex / gravity well
  peace   — index + middle extended → dual attract tips
  spider  — index + pinky extended  → split attract

Face gestures detected
----------------------
  mouth_open    — vertical lip gap relative to face height
  head_tilt     — angle of eye-to-eye line (tilts gravity axis)
  brow_raised   — forehead/brow distance increase (colour burst trigger)
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HandGesture:
    name:        str   = "none"
    palm_center: Tuple = (0.0, 0.0)
    finger_tips: List  = field(default_factory=list)
    wrist:       Tuple = (0.0, 0.0)
    point_dir:   Tuple = (0.0, 1.0)    # unit vector for 'point' gesture
    handedness:  str   = "Right"


@dataclass
class FaceGesture:
    mouth_open:     bool  = False
    mouth_openness: float = 0.0        # 0 = closed, 1 = fully open
    mouth_center:   Tuple = (0.5, 0.5)
    head_tilt:      float = 0.0        # radians; + = right lean, − = left lean
    brow_raised:    bool  = False
    head_nod:       bool  = False      # True when avg eye Y normalised > 0.56


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dist(a: Tuple, b: Tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ─────────────────────────────────────────────────────────────────────────────
# Hand classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_hand(landmarks: List[Tuple], handedness: str) -> HandGesture:
    """
    Classify a single hand into one of the gesture categories.

    MediaPipe landmark indices used
    --------------------------------
    0  = wrist
    1-4  = thumb  (CMC, MCP, IP, TIP)
    5-8  = index  (MCP, PIP, DIP, TIP)
    9-12 = middle (MCP, PIP, DIP, TIP)
    13-16= ring   (MCP, PIP, DIP, TIP)
    17-20= pinky  (MCP, PIP, DIP, TIP)
    """
    if len(landmarks) < 21:
        return HandGesture(handedness=handedness)

    lm = landmarks

    # ── Palm centre (centroid of wrist + 4 finger bases) ─────────────────────
    palm_pts = [lm[0], lm[5], lm[9], lm[13], lm[17]]
    pcx = sum(p[0] for p in palm_pts) / 5
    pcy = sum(p[1] for p in palm_pts) / 5
    palm_c = (pcx, pcy)

    # ── Finger tips ───────────────────────────────────────────────────────────
    tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]

    # ── Extension checks ──────────────────────────────────────────────────────
    # Non-thumb: TIP.y < PIP.y  (smaller y = higher in image = extended)
    def ext(tip_i: int, pip_i: int) -> bool:
        return lm[tip_i][1] < lm[pip_i][1]

    idx_ext   = ext(8,  7)
    mid_ext   = ext(12, 11)
    ring_ext  = ext(16, 15)
    pink_ext  = ext(20, 19)
    n_ext     = sum([idx_ext, mid_ext, ring_ext, pink_ext])

    # Thumb: tip further from palm centre than IP joint
    thumb_ext = _dist(lm[4], lm[9]) > 0.065

    # ── Pinch: thumb tip ↔ index tip close together ───────────────────────────
    pinch_d = _dist(lm[4], lm[8])

    # ── Pointing direction (index tip − wrist, normalised) ───────────────────
    dx = lm[8][0] - lm[0][0]
    dy = lm[8][1] - lm[0][1]
    mag = math.hypot(dx, dy) or 1e-9
    point_dir = (dx / mag, dy / mag)

    # ── Classification ────────────────────────────────────────────────────────
    if pinch_d < 0.045:
        name = "pinch"
    elif n_ext == 0 and not thumb_ext:
        name = "fist"
    elif n_ext == 4:
        name = "open"
    elif idx_ext and not mid_ext and not ring_ext and not pink_ext:
        name = "point"
    elif idx_ext and mid_ext and not ring_ext and not pink_ext:
        name = "peace"
    elif idx_ext and pink_ext and not mid_ext and not ring_ext:
        name = "spider"
    else:
        name = "partial"

    return HandGesture(
        name=name,
        palm_center=palm_c,
        finger_tips=tips,
        wrist=lm[0],
        point_dir=point_dir,
        handedness=handedness,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Face classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_face(landmarks: List[Tuple]) -> FaceGesture:
    """
    Classify face state from MediaPipe FaceMesh landmarks (468 points).

    Key landmark indices
    --------------------
    10  = forehead top
    152 = chin bottom
    13  = upper-lip inner
    14  = lower-lip inner
    33  = left  eye outer corner
    263 = right eye outer corner
    70  = left  brow (upper arc)
    300 = right brow (upper arc)
    """
    if len(landmarks) < 468:
        return FaceGesture()

    lm = landmarks

    # ── Face height ───────────────────────────────────────────────────────────
    face_h = _dist(lm[10], lm[152]) or 0.01

    # ── Mouth openness ────────────────────────────────────────────────────────
    gap        = abs(lm[13][1] - lm[14][1])
    openness   = min(gap / (face_h * 0.25), 1.0)
    mouth_open = openness > 0.35
    mouth_c    = ((lm[13][0] + lm[14][0]) / 2,
                  (lm[13][1] + lm[14][1]) / 2)

    # ── Head tilt  (angle of eye-to-eye axis) ─────────────────────────────────
    dx   = lm[263][0] - lm[33][0]
    dy   = lm[263][1] - lm[33][1]
    tilt = math.atan2(dy, dx)   # ≈ 0 when head is upright

    # ── Brow raise ────────────────────────────────────────────────────────────
    brow_delta = (
        (lm[33][1] - lm[70][1]) + (lm[263][1] - lm[300][1])
    ) / 2
    brow_raised = brow_delta > 0.055 * face_h

    # ── Head nod (chin down — eyes move lower in frame) ────────────────────
    eye_y_mean = (lm[33][1] + lm[263][1]) / 2
    head_nod = eye_y_mean > 0.56

    return FaceGesture(
        mouth_open=mouth_open,
        mouth_openness=openness,
        mouth_center=mouth_c,
        head_tilt=tilt,
        brow_raised=brow_raised,
        head_nod=head_nod,
    )
