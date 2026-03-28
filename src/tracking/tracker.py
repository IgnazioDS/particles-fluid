"""
src/tracking/tracker.py
=======================
MediaPipe Hands + FaceMesh wrapper.

Reads one webcam frame per call to `read()`, runs both detectors,
and exposes `hand_states` and `face_state` for the main loop to consume.

Landmark coordinates from MediaPipe are normalised [0, 1].
Use `norm_to_sim(nx, ny)` to convert to simulation pixel space.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple

from .gesture import HandGesture, FaceGesture, classify_hand, classify_face


class Tracker:
    """
    Wraps MediaPipe Hands + FaceMesh on a single webcam stream.

    Attributes
    ----------
    available     : bool             — False if webcam failed to open
    frame         : np.ndarray|None  — latest BGR frame (sim-space size)
    hand_states   : List[HandGesture]
    face_state    : FaceGesture|None
    """

    def __init__(
        self,
        webcam_index: int = 0,
        sim_w:        int = 1280,
        sim_h:        int = 720,
    ):
        self.sim_w = sim_w
        self.sim_h = sim_h

        # ── MediaPipe detectors ───────────────────────────────────────────────
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
        )
        self._face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.55,
        )

        # ── Webcam ────────────────────────────────────────────────────────────
        self._cap = cv2.VideoCapture(webcam_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self.available = self._cap.isOpened()

        # ── Public state ──────────────────────────────────────────────────────
        self.frame:       Optional[np.ndarray]   = None
        self.hand_states: List[HandGesture]      = []
        self.face_state:  Optional[FaceGesture]  = None

        # Landmark draw util (optional overlay)
        self._mp_draw   = mp.solutions.drawing_utils
        self._hand_conn = mp.solutions.hands.HAND_CONNECTIONS

    # ─────────────────────────────────────────────────────────────────────────

    def read(self, draw_landmarks: bool = False) -> bool:
        """
        Capture and process one frame.
        Returns True if the frame was successfully captured.
        """
        if not self.available:
            return False

        ret, raw = self._cap.read()
        if not ret or raw is None:
            return False

        # Mirror (natural selfie view) and resize to simulation resolution
        raw   = cv2.flip(raw, 1)
        frame = cv2.resize(raw, (self.sim_w, self.sim_h))
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # slight speedup for MediaPipe

        # ── Hand tracking ─────────────────────────────────────────────────────
        self.hand_states = []
        hand_res = self._hands.process(rgb)
        if hand_res.multi_hand_landmarks:
            for i, hlm in enumerate(hand_res.multi_hand_landmarks):
                label = "Right"
                if hand_res.multi_handedness and i < len(hand_res.multi_handedness):
                    label = hand_res.multi_handedness[i].classification[0].label

                lm_list = [(lm.x, lm.y) for lm in hlm.landmark]
                self.hand_states.append(classify_hand(lm_list, label))

                if draw_landmarks:
                    rgb.flags.writeable = True
                    self._mp_draw.draw_landmarks(
                        frame, hlm, self._hand_conn,
                        self._mp_draw.DrawingSpec(color=(0, 220, 100), thickness=1, circle_radius=2),
                        self._mp_draw.DrawingSpec(color=(0, 150, 255), thickness=1),
                    )

        # ── Face tracking ─────────────────────────────────────────────────────
        self.face_state = None
        face_res = self._face.process(rgb)
        if face_res.multi_face_landmarks:
            lm_list = [(lm.x, lm.y) for lm in face_res.multi_face_landmarks[0].landmark]
            self.face_state = classify_face(lm_list)

        self.frame = frame
        return True

    def norm_to_sim(self, nx: float, ny: float) -> Tuple[float, float]:
        """Convert MediaPipe normalised [0,1] coords → simulation pixels."""
        return nx * self.sim_w, ny * self.sim_h

    def close(self):
        self._cap.release()
        self._hands.close()
        self._face.close()
