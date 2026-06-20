"""
ui.py
-----
UIState  – pure-Python state machine that translates each frame's
           TrackingResult into sphere control parameters.

No rendering code lives here; this module is the "controller" layer
between hand_tracking.py (input) and effects.py (output).
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional

from hand_tracking import TrackingResult


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Smoothing factors (0 = no smoothing, 1 = frozen)
    lerp_offset: float   = 0.10
    lerp_radius: float   = 0.07
    lerp_rot:    float   = 0.06
    lerp_hue:    float   = 0.05

    # Sphere limits
    radius_min: float = 0.25
    radius_max: float = 2.60

    # Auto-rotation speed (rad/frame) when no hand present
    auto_rot_speed: float = 0.004

    # Pinch
    pinch_charge_time: float = 1.8   # seconds to fully charge
    pinch_min_charge:  float = 0.30  # minimum charge to trigger explosion

    # Explosion
    explode_duration:  float = 2.2   # seconds

    # Two-hand spread sensitivity
    spread_sensitivity: float = 3.2


# ── UIState ───────────────────────────────────────────────────────────────────

@dataclass
class UIState:
    """
    Maintains all animated state for the sphere.

    Call update(result) once per frame; then read the public properties
    to drive the renderer.
    """

    cfg: Config = field(default_factory=Config)

    # ── Animated sphere parameters (smoothed) ────────────────────────────────
    radius: float       = 1.0
    offset_x: float     = 0.0
    offset_y: float     = 0.0
    rot_x: float        = 0.0
    rot_y: float        = 0.0
    hue: float          = 0.55

    # ── Targets (snapped each frame) ─────────────────────────────────────────
    _target_radius:  float = 1.0
    _target_offset_x: float = 0.0
    _target_offset_y: float = 0.0
    _target_rot_x:   float = 0.0
    _target_rot_y:   float = 0.0
    _target_hue:     float = 0.55

    # ── Pinch / explosion state ───────────────────────────────────────────────
    pinch_charge:   float = 0.0
    is_pinching:    bool  = False
    _pinch_start:   float = 0.0

    exploding:      bool  = False
    explode_factor: float = 0.0
    explode_t:      float = 0.0
    _explode_start: float = 0.0
    particles_reset_needed: bool = False   # main loop should reset particles

    # ── Two-hand spread tracking ──────────────────────────────────────────────
    _last_spread: Optional[float] = None

    # ── HUD info ─────────────────────────────────────────────────────────────
    hand_info: str = "—"
    finger_count: int = 0

    # ── Index-tip position for charge-ring placement ─────────────────────────
    index_tip: tuple[float, float] = (0.5, 0.5)

    # ── Overlay toggles ───────────────────────────────────────────────────────
    show_webcam_bg:  bool = False
    show_landmarks:  bool = False

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, result: TrackingResult) -> None:
        """Main per-frame update.  Call with the latest TrackingResult."""
        self._advance_explosion()

        if not result.hands:
            self._no_hand_update()
            return

        if len(result.hands) == 1:
            self._single_hand_update(result)
        else:
            self._two_hand_update(result)

        self._smooth()

    # ── Single-hand logic ─────────────────────────────────────────────────────

    def _single_hand_update(self, result: TrackingResult) -> None:
        hand = result.hands[0]
        px, py = hand.palm_center

        # Move sphere to palm
        self._target_offset_x = px - 0.5
        self._target_offset_y = -(py - 0.5)

        # Bias rotation toward palm position
        self._target_rot_y += (px - 0.5) * 0.05
        self._target_rot_x += (py - 0.5) * 0.05

        # Finger colour
        self.finger_count = hand.finger_count
        self._target_hue = hand.finger_count / 10.0

        # Store index-tip for charge ring
        self.index_tip = hand.index_tip

        # Pinch logic
        if hand.is_pinching:
            if not self.is_pinching:
                self.is_pinching = True
                self._pinch_start = time.time()
            elapsed = time.time() - self._pinch_start
            self.pinch_charge = min(1.0, elapsed / self.cfg.pinch_charge_time)
        else:
            if self.is_pinching and self.pinch_charge >= self.cfg.pinch_min_charge:
                self._trigger_explosion()
            self.is_pinching = False
            self.pinch_charge = 0.0

        self._last_spread = None
        self.hand_info = f"1 hand  |  {hand.finger_count} finger{'s' if hand.finger_count != 1 else ''}"

    # ── Two-hand logic ────────────────────────────────────────────────────────

    def _two_hand_update(self, result: TrackingResult) -> None:
        h1, h2 = result.hands[0], result.hands[1]

        # Spread → radius
        spread = result.two_hand_spread
        if self._last_spread is not None and spread is not None:
            delta = spread - self._last_spread
            self._target_radius = float(
                max(self.cfg.radius_min,
                    min(self.cfg.radius_max,
                        self._target_radius + delta * self.cfg.spread_sensitivity))
            )
        self._last_spread = spread

        # Move to centroid
        cx = (h1.palm_center[0] + h2.palm_center[0]) / 2
        cy = (h1.palm_center[1] + h2.palm_center[1]) / 2
        self._target_offset_x = cx - 0.5
        self._target_offset_y = -(cy - 0.5)

        # Cancel pinch on two-hand
        self.is_pinching = False
        self.pinch_charge = 0.0

        total_fingers = h1.finger_count + h2.finger_count
        self.finger_count = total_fingers
        self._target_hue = total_fingers / 20.0
        self.hand_info = f"2 hands  |  {total_fingers} fingers"

    # ── No-hand logic ─────────────────────────────────────────────────────────

    def _no_hand_update(self) -> None:
        # Release pinch if hand disappears with enough charge
        if self.is_pinching and self.pinch_charge >= self.cfg.pinch_min_charge:
            self._trigger_explosion()
        self.is_pinching = False
        self.pinch_charge = 0.0
        self._last_spread = None

        # Drift back to centre
        self._target_offset_x = 0.0
        self._target_offset_y = 0.0

        # Auto-rotate
        self._target_rot_y += self.cfg.auto_rot_speed

        self.hand_info = "—"
        self.finger_count = 0

    # ── Explosion ─────────────────────────────────────────────────────────────

    def _trigger_explosion(self) -> None:
        self.exploding = True
        self._explode_start = time.time()
        self.is_pinching = False
        self.pinch_charge = 0.0

    def _advance_explosion(self) -> None:
        if not self.exploding:
            self.explode_factor = 0.0
            self.explode_t = 0.0
            self.particles_reset_needed = False
            return

        elapsed = time.time() - self._explode_start
        dur = self.cfg.explode_duration
        if elapsed >= dur:
            self.exploding = False
            self.explode_factor = 0.0
            self.explode_t = 0.0
            self.particles_reset_needed = True
        else:
            p = elapsed / dur
            import math
            self.explode_factor = math.sin(p * math.pi)
            self.explode_t = p * 1.4

    # ── Smoothing ─────────────────────────────────────────────────────────────

    def _smooth(self) -> None:
        c = self.cfg
        self.radius   = _lerp(self.radius,   self._target_radius,   c.lerp_radius)
        self.offset_x = _lerp(self.offset_x, self._target_offset_x, c.lerp_offset)
        self.offset_y = _lerp(self.offset_y, self._target_offset_y, c.lerp_offset)
        self.rot_x    = _lerp(self.rot_x,    self._target_rot_x,    c.lerp_rot)
        self.rot_y    = _lerp(self.rot_y,    self._target_rot_y,    c.lerp_rot)
        self.hue      = _lerp(self.hue,      self._target_hue,      c.lerp_hue)

    # ── Keyboard handlers ─────────────────────────────────────────────────────

    def handle_key(self, key: int) -> bool:
        """
        Handle a keypress.  Returns True if the app should quit.
        Keys: Q=quit, W=toggle webcam bg, L=toggle landmarks.
        """
        if key in (ord('q'), ord('Q'), 27):   # Q or Esc
            return True
        if key in (ord('w'), ord('W')):
            self.show_webcam_bg = not self.show_webcam_bg
        if key in (ord('l'), ord('L')):
            self.show_landmarks = not self.show_landmarks
        return False


# ── Utility ───────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
