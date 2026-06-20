"""
hand_tracking.py
----------------
Wraps MediaPipe Hands to provide clean, structured hand data
each frame: landmarks, finger counts, pinch distance, palm centre,
and two-hand spread distance.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


# ── Finger tip / pip landmark indices ────────────────────────────────────────
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS  # type: ignore


@dataclass
class HandData:
    """Processed data for a single detected hand."""
    landmarks: list          # raw NormalizedLandmark list (21 points)
    finger_count: int = 0
    pinch_dist: float = 1.0  # normalised 0-1
    palm_center: tuple[float, float] = (0.5, 0.5)
    index_tip: tuple[float, float] = (0.5, 0.5)
    is_pinching: bool = False


@dataclass
class TrackingResult:
    """Aggregated result for one video frame."""
    hands: list[HandData] = field(default_factory=list)
    two_hand_spread: Optional[float] = None   # None when < 2 hands
    total_fingers: int = 0
    raw_result: object = None                 # mediapipe Results object


class HandTracker:
    """
    Initialises MediaPipe Hands and processes each frame.

    Parameters
    ----------
    max_hands : int
        Maximum number of hands to detect (1 or 2).
    detection_confidence : float
    tracking_confidence : float
    pinch_threshold : float
        Normalised distance below which a pinch is registered.
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.6,
        pinch_threshold: float = 0.06,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.pinch_threshold = pinch_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> TrackingResult:
        """
        Process one BGR frame and return a TrackingResult.

        The frame is NOT modified; a flipped RGB copy is sent to MediaPipe.
        """
        rgb = cv2.cvtColor(cv2.flip(frame_bgr, 1), cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_result = self._hands.process(rgb)
        rgb.flags.writeable = True

        result = TrackingResult(raw_result=mp_result)
        if not mp_result.multi_hand_landmarks:
            return result

        for lm_list in mp_result.multi_hand_landmarks:
            lms = lm_list.landmark
            hand = self._build_hand_data(lms)
            result.hands.append(hand)
            result.total_fingers += hand.finger_count

        if len(result.hands) >= 2:
            c1 = result.hands[0].palm_center
            c2 = result.hands[1].palm_center
            result.two_hand_spread = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

        return result

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        result: TrackingResult,
        color: tuple[int, int, int] = (180, 180, 140),
        tip_color: tuple[int, int, int] = (80, 160, 220),
    ) -> np.ndarray:
        """
        Draw skeleton and landmark dots onto *a copy* of frame_bgr.
        The original frame is not mutated.
        """
        if result.raw_result is None or not result.raw_result.multi_hand_landmarks:
            return frame_bgr

        out = frame_bgr.copy()
        h, w = out.shape[:2]
        mp_draw = mp.solutions.drawing_utils
        draw_spec_lm = mp_draw.DrawingSpec(color=color, thickness=1, circle_radius=2)
        draw_spec_cn = mp_draw.DrawingSpec(color=(60, 60, 60), thickness=1)

        for lm_list in result.raw_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(out, lm_list, HAND_CONNECTIONS, draw_spec_lm, draw_spec_cn)
            # Highlight fingertips
            for idx in FINGER_TIPS:
                lm = lm_list.landmark[idx]
                cx, cy = int((1 - lm.x) * w), int(lm.y * h)
                cv2.circle(out, (cx, cy), 5, tip_color, -1)

        return out

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_hand_data(self, lms) -> HandData:
        palm_x = (lms[0].x + lms[9].x) / 2
        palm_y = (lms[0].y + lms[9].y) / 2
        index_tip = (lms[8].x, lms[8].y)
        pinch_dist = math.hypot(lms[4].x - lms[8].x, lms[4].y - lms[8].y)
        finger_count = self._count_fingers(lms)
        return HandData(
            landmarks=lms,
            finger_count=finger_count,
            pinch_dist=pinch_dist,
            palm_center=(palm_x, palm_y),
            index_tip=index_tip,
            is_pinching=pinch_dist < self.pinch_threshold,
        )

    @staticmethod
    def _count_fingers(lms) -> int:
        count = 0
        # Four fingers: tip above pip (lower y = higher on screen)
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            if lms[tip].y < lms[pip].y:
                count += 1
        # Thumb: check horizontal spread from base
        if abs(lms[4].x - lms[2].x) > 0.06:
            count += 1
        return count
