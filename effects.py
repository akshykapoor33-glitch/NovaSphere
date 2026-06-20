"""
effects.py
----------
CPU-side particle sphere renderer using NumPy + OpenCV.

Classes
-------
ParticleSphere   – manages 3-D particle positions, rotation, colour, explosion.
ChargeRing       – draws an SVG-style arc progress ring on the OpenCV canvas.
OverlayRenderer  – composites all visual layers onto the final frame.
"""

from __future__ import annotations
import math
import time
from typing import Optional

import cv2
import numpy as np


TAU = math.pi * 2


# ── Colour helpers ────────────────────────────────────────────────────────────

def hsl_to_bgr(h: float, s: float = 0.75, l: float = 0.72) -> tuple[int, int, int]:
    """Convert HSL (0-1 each) to BGR uint8 tuple."""
    h = h % 1.0
    a = s * min(l, 1 - l)
    def f(n: float) -> float:
        k = (n + h * 12) % 12
        return l - a * max(min(k - 3, 9 - k, 1), -1)
    r, g, b = f(0), f(8), f(4)
    return (int(b * 255), int(g * 255), int(r * 255))


# ── Rotation matrices ─────────────────────────────────────────────────────────

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


# ── Particle Sphere ───────────────────────────────────────────────────────────

class ParticleSphere:
    """
    Manages N particles distributed on a unit sphere.

    State attributes (set externally by main loop)
    -----------------------------------------------
    radius          – current display radius (world units, mapped to pixels)
    offset_x/y      – sphere centre offset in [-1, 1] NDC
    rot_x / rot_y   – current rotation angles (radians)
    hue             – particle hue 0-1
    explode_factor  – 0 = normal, 1 = fully exploded
    """

    def __init__(self, n: int = 3000) -> None:
        self.n = n
        # Sphere surface positions (unit sphere)
        self._base = self._random_sphere(n)       # (N, 3)
        self._phases = np.random.uniform(0, TAU, n).astype(np.float32)
        self._explode_vel = (np.random.rand(n, 3).astype(np.float32) - 0.5) * 5.0

        # Public state
        self.radius: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.rot_x_angle: float = 0.0
        self.rot_y_angle: float = 0.0
        self.hue: float = 0.55
        self.explode_factor: float = 0.0
        self.explode_t: float = 0.0

        self._start = time.time()

    # ── Regenerate base positions (called after explosion settles) ────────────

    def reset_particles(self) -> None:
        self._base = self._random_sphere(self.n)
        self._phases = np.random.uniform(0, TAU, self.n).astype(np.float32)
        self._explode_vel = (np.random.rand(self.n, 3).astype(np.float32) - 0.5) * 5.0

    # ── Render onto canvas ────────────────────────────────────────────────────

    def draw(self, canvas: np.ndarray) -> None:
        """Draw all particles onto *canvas* in-place."""
        h, w = canvas.shape[:2]
        t = time.time() - self._start

        # Wave displacement
        wave = (np.sin(t * 1.2 + self._phases) * 0.012
                + np.cos(t * 0.8 + self._phases * 1.3) * 0.008)  # (N,)

        radius_eff = self.radius + wave                            # (N,)
        pts = self._base * radius_eff[:, None]                    # (N, 3)

        # Explosion blend
        if self.explode_factor > 0:
            exploded = self._base + self._explode_vel * (self.explode_t ** 2)
            pts = pts * (1 - self.explode_factor) + exploded * self.explode_factor

        # Rotate
        pts = (pts @ rot_x(self.rot_x_angle).T) @ rot_y(self.rot_y_angle).T

        # Project to screen (simple perspective)
        aspect = w / h
        scale = min(w, h) * 0.28
        cx = w / 2 + self.offset_x * w * 0.4
        cy = h / 2 - self.offset_y * h * 0.4

        z = pts[:, 2]
        depth = (z + 3.0) / 6.0                                   # 0-1
        depth = np.clip(depth, 0, 1)

        sx = (pts[:, 0] / aspect) * scale + cx
        sy = -pts[:, 1] * scale + cy

        # Sort back-to-front for additive look (painters algo approximation)
        order = np.argsort(z)
        sx, sy, depth = sx[order], sy[order], depth[order]

        # Clip to canvas bounds
        mask = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        sx, sy, depth = sx[mask].astype(int), sy[mask].astype(int), depth[mask]

        # Build colour
        bgr = hsl_to_bgr(self.hue)
        alpha = np.clip(0.25 + 0.75 * depth, 0, 1) * (1 - self.explode_factor * 0.4)

        # Draw as glow dots: small bright centre + soft halo
        for i in range(len(sx)):
            a = float(alpha[i])
            if a < 0.05:
                continue
            px, py = sx[i], sy[i]
            r_dot = max(1, int(1.5 + depth[i] * 2.5))
            # Halo (soft, 30% alpha of particle)
            halo_r = r_dot + 2
            color_halo = tuple(int(c * a * 0.3) for c in bgr)
            cv2.circle(canvas, (px, py), halo_r, color_halo, -1, cv2.LINE_AA)
            # Core
            color_core = tuple(min(255, int(c * a)) for c in bgr)
            cv2.circle(canvas, (px, py), r_dot, color_core, -1, cv2.LINE_AA)

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _random_sphere(n: int) -> np.ndarray:
        u = np.random.rand(n).astype(np.float32)
        v = np.random.rand(n).astype(np.float32)
        theta = TAU * u
        phi = np.arccos(2 * v - 1)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.stack([x, y, z], axis=1)


# ── Charge Ring ───────────────────────────────────────────────────────────────

class ChargeRing:
    """
    Draws an arc-progress ring at a given screen position.

    Parameters
    ----------
    radius : int     outer ring radius in pixels
    thickness : int  arc stroke width
    """

    def __init__(self, radius: int = 50, thickness: int = 2) -> None:
        self.radius = radius
        self.thickness = thickness

    def draw(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        charge: float,           # 0.0 → 1.0
    ) -> None:
        """Draw the charge ring onto *canvas* in-place."""
        if charge <= 0:
            return

        # Background ghost ring
        cv2.circle(canvas, (cx, cy), self.radius,
                   (40, 40, 50), self.thickness, cv2.LINE_AA)

        # Arc (drawn as a series of short line segments)
        span = int(360 * charge)
        color_t = charge
        b = int(100 * (1 - color_t) + 60 * color_t)
        g = int(180 * (1 - color_t) + 130 * color_t)
        r = int(220 * color_t + 120 * (1 - color_t))
        arc_color = (b, g, r)

        start_angle = -90  # top
        end_angle = start_angle + span
        cv2.ellipse(
            canvas, (cx, cy),
            (self.radius, self.radius),
            0, start_angle, end_angle,
            arc_color, self.thickness, cv2.LINE_AA,
        )

        # Dot at tip
        tip_rad = math.radians(end_angle)
        tx = int(cx + self.radius * math.cos(tip_rad))
        ty = int(cy + self.radius * math.sin(tip_rad))
        cv2.circle(canvas, (tx, ty), self.thickness + 1, arc_color, -1, cv2.LINE_AA)


# ── Overlay Renderer ──────────────────────────────────────────────────────────

class OverlayRenderer:
    """
    Composites all HUD elements (title, status dots, hints, finger count)
    onto the final display frame.
    """

    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_MONO = cv2.FONT_HERSHEY_PLAIN

    DIM   = (180, 180, 160)    # dim text
    BRIGHT= (220, 215, 200)    # bright text
    GREEN = (100, 160, 80)     # status on
    DARK  = (55, 55, 70)       # status off

    def draw_title(self, frame: np.ndarray) -> None:
        cv2.putText(frame, "Particle Sphere", (36, 52),
                    self.FONT, 0.75, self.BRIGHT, 1, cv2.LINE_AA)
        cv2.putText(frame, "hand-reactive  |  3000 particles", (36, 72),
                    self.FONT_MONO, 1.0, self.DIM, 1, cv2.LINE_AA)

    def draw_status(
        self,
        frame: np.ndarray,
        cam_ok: bool,
        hand_detected: bool,
        hand_info: str = "—",
    ) -> None:
        h, w = frame.shape[:2]
        lines = [
            ("webcam",       cam_ok),
            ("hand tracking", hand_detected),
        ]
        y = 52
        for label, state in lines:
            col = self.GREEN if state else self.DARK
            cv2.circle(frame, (w - 120, y - 6), 4, col, -1, cv2.LINE_AA)
            cv2.putText(frame, label, (w - 110, y),
                        self.FONT_MONO, 1.0, self.DIM, 1, cv2.LINE_AA)
            y += 22
        cv2.putText(frame, hand_info, (w - 160, y + 4),
                    self.FONT_MONO, 0.9, self.DIM, 1, cv2.LINE_AA)

    def draw_hints(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        hints = [
            "single hand -> move & rotate",
            "pinch & hold -> charge  |  release -> explode",
            "two hands -> spread/pinch to resize",
            "finger count -> colour shift",
            "Q -> quit   W -> toggle webcam bg   L -> landmarks",
        ]
        y = h - len(hints) * 18 - 10
        for line in hints:
            cv2.putText(frame, line, (w - 430, y),
                        self.FONT_MONO, 0.85, self.DIM, 1, cv2.LINE_AA)
            y += 18

    def draw_finger_count(self, frame: np.ndarray, count: int) -> None:
        if count == 0:
            return
        h, w = frame.shape[:2]
        label = str(count)
        sz = cv2.getTextSize(label, self.FONT, 4.0, 1)[0]
        ox = (w - sz[0]) // 2
        oy = (h + sz[1]) // 2
        cv2.putText(frame, label, (ox, oy),
                    self.FONT, 4.0, (30, 30, 38), 1, cv2.LINE_AA)

    def draw_fps(self, frame: np.ndarray, fps: float) -> None:
        cv2.putText(frame, f"{fps:.0f} fps", (36, frame.shape[0] - 36),
                    self.FONT_MONO, 1.0, self.DIM, 1, cv2.LINE_AA)
