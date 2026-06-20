"""
main.py
-------
Entry point for Particle Sphere.

Run:
    python main.py [--camera INDEX] [--width W] [--height H] [--particles N]

The loop:
  1. Grab a webcam frame.
  2. Send it to HandTracker → TrackingResult.
  3. Feed result into UIState.update() → sphere parameters.
  4. Render: black canvas → ParticleSphere.draw() → overlays → show.
"""

from __future__ import annotations
import argparse
import sys
import time

import cv2
import numpy as np

from hand_tracking import HandTracker
from effects import ParticleSphere, ChargeRing, OverlayRenderer
from ui import UIState, Config


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Particle Sphere — hand-reactive visualiser")
    p.add_argument("--camera",    type=int,   default=0,    help="Webcam device index (default 0)")
    p.add_argument("--width",     type=int,   default=1280, help="Window width  (default 1280)")
    p.add_argument("--height",    type=int,   default=720,  help="Window height (default 720)")
    p.add_argument("--particles", type=int,   default=3000, help="Particle count (default 3000)")
    p.add_argument("--fullscreen", action="store_true",     help="Start fullscreen")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    W, H = args.width, args.height

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cam_ok = True
    print("[INFO] Camera opened.")

    # ── Window ────────────────────────────────────────────────────────────────
    win = "Particle Sphere"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    if args.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ── Objects ───────────────────────────────────────────────────────────────
    tracker  = HandTracker(max_hands=2)
    sphere   = ParticleSphere(n=args.particles)
    ring     = ChargeRing(radius=52, thickness=2)
    overlay  = OverlayRenderer()
    state    = UIState(cfg=Config())

    fps_timer = time.time()
    fps_count = 0
    fps_display = 0.0

    print("[INFO] Starting — press Q or Esc to quit.")
    print("       W = toggle webcam background   L = toggle landmarks")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            cam_ok = False
            raw_frame = np.zeros((480, 640, 3), np.uint8)

        # Resize webcam frame to display size
        cam_frame = cv2.resize(raw_frame, (W, H))

        # Hand tracking
        result = tracker.process(raw_frame)

        # Update state machine
        state.update(result)

        # Reset particles after explosion settled
        if state.particles_reset_needed:
            sphere.reset_particles()
            state.particles_reset_needed = False

        # ── Build canvas ──────────────────────────────────────────────────────
        if state.show_webcam_bg:
            # Dark-tinted webcam as background
            canvas = (cam_frame.astype(np.float32) * 0.18).astype(np.uint8)
        else:
            canvas = np.zeros((H, W, 3), np.uint8)

        # Push sphere parameters
        sphere.radius       = state.radius
        sphere.offset_x     = state.offset_x
        sphere.offset_y     = state.offset_y
        sphere.rot_x_angle  = state.rot_x
        sphere.rot_y_angle  = state.rot_y
        sphere.hue          = state.hue
        sphere.explode_factor = state.explode_factor
        sphere.explode_t      = state.explode_t

        # Draw sphere
        sphere.draw(canvas)

        # Landmarks overlay
        if state.show_landmarks and result.hands:
            canvas = tracker.draw_landmarks(canvas, result)

        # Charge ring
        if state.is_pinching and state.pinch_charge > 0:
            ix, iy = state.index_tip
            # Flip x because frame is already mirrored in tracker
            rx = int((1 - ix) * W)
            ry = int(iy * H)
            ring.draw(canvas, rx, ry, state.pinch_charge)

        # HUD overlays
        overlay.draw_title(canvas)
        overlay.draw_status(
            canvas,
            cam_ok=cam_ok,
            hand_detected=len(result.hands) > 0,
            hand_info=state.hand_info,
        )
        overlay.draw_hints(canvas)
        overlay.draw_finger_count(canvas, state.finger_count)
        overlay.draw_fps(canvas, fps_display)

        # ── Show ──────────────────────────────────────────────────────────────
        cv2.imshow(win, canvas)

        # FPS counter
        fps_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps_display = fps_count / (now - fps_timer)
            fps_count = 0
            fps_timer = now

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if state.handle_key(key):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
