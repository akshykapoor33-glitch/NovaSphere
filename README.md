# Particle Sphere

A real-time, hand-reactive 3-D particle visualiser built with Python, OpenCV, MediaPipe Hands, and NumPy.
3 000 particles form a breathing sphere that responds to your hand gestures live from the webcam.

---

## Features

| Gesture | Effect |
|---|---|
| **Single hand** | Moves the sphere; biases rotation toward palm position |
| **Pinch & hold** | Fills a charge ring; release → particles explode across screen, then reform |
| **Two hands spread** | Grows the sphere radius |
| **Two hands pinch** | Shrinks the sphere radius |
| **Finger count** | Shifts particle colour through the full hue spectrum |
| **W key** | Toggles the live webcam feed as a dim background |
| **L key** | Toggles hand-landmark skeleton overlay |
| **Q / Esc** | Quit |

---

## Project Structure

```
particle_sphere/
├── main.py            # Entry point — webcam loop, compositing
├── hand_tracking.py   # MediaPipe wrapper → TrackingResult dataclass
├── effects.py         # ParticleSphere, ChargeRing, OverlayRenderer
├── ui.py              # UIState state machine — translates hand data to sphere params
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

### Module responsibilities

```
webcam frame
     │
     ▼
hand_tracking.py   ← raw frame in, TrackingResult out
     │
     ▼
ui.py              ← TrackingResult in, animated sphere parameters out (UIState)
     │
     ▼
effects.py         ← parameters in, draws onto numpy canvas
     │
     ▼
main.py            ← orchestrates everything, shows window
```

---

## Requirements

- Python 3.9 – 3.11 (MediaPipe does not yet support 3.12+ on all platforms)
- Webcam

### Python packages

```
opencv-python >= 4.8
mediapipe     >= 0.10
numpy         >= 1.24
```

---

## Installation

```bash
# 1. Clone / download this folder
cd particle_sphere

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **macOS Apple Silicon** — MediaPipe wheels are available for arm64:
> `pip install mediapipe` works as-is on M1/M2/M3.

> **Windows** — make sure you have the Visual C++ Redistributable installed
> (usually already present if you have Python from python.org).

---

## Running

```bash
python main.py
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--camera INDEX` | `0` | Webcam device index |
| `--width W` | `1280` | Window width in pixels |
| `--height H` | `720` | Window height in pixels |
| `--particles N` | `3000` | Number of particles |
| `--fullscreen` | off | Start in fullscreen mode |

**Examples**

```bash
# Use a secondary webcam
python main.py --camera 1

# Lighter load — 1 500 particles
python main.py --particles 1500

# Fullscreen at 1920×1080
python main.py --width 1920 --height 1080 --fullscreen
```

---

## How it works

### hand_tracking.py — `HandTracker`

Wraps `mediapipe.solutions.hands`. Each call to `tracker.process(frame)` returns a
`TrackingResult` containing:

- `hands` — list of `HandData` objects (up to 2)
- Each `HandData` has: `landmarks`, `finger_count`, `pinch_dist`,
  `palm_center`, `index_tip`, `is_pinching`
- `two_hand_spread` — normalised distance between palm centres when 2 hands detected

The frame is internally flipped horizontally so left/right feel natural (mirror mode).

### effects.py — `ParticleSphere`

Particles are placed on the surface of a unit sphere using the Fibonacci / uniform
random method. Each frame:

1. A sinusoidal wave displacement is added per-particle (breathing effect).
2. Explosion blend: positions lerp from sphere surface to random velocity-driven
   trajectories controlled by `explode_factor` and `explode_t`.
3. Rotation matrices (rx, ry) are applied in NumPy.
4. Simple perspective projection maps 3-D → 2-D screen pixels.
5. Particles are drawn back-to-front with a soft glow halo + core dot using
   `cv2.circle` with `LINE_AA`.

Colour is driven by `hue` via an HSL → BGR conversion so finger count maps
smoothly to colour.

### ui.py — `UIState`

A pure state machine (no rendering). Receives a `TrackingResult` and updates
smoothed targets for radius, offset, rotation, and hue using linear interpolation
(configurable `lerp_*` factors in `Config`).

Pinch charging: tracks elapsed time since pinch started. If the hand releases
(or disappears) with `pinch_charge ≥ 0.30`, triggers the explosion.

### main.py

Orchestrates the loop:

```
read frame → track hands → update UIState → build canvas → draw sphere
→ draw charge ring → draw HUD overlays → imshow → handle keys
```

---

## Performance tips

- If the frame rate drops, reduce `--particles` (try 1 500 or 1 000).
- The CPU renderer is the bottleneck for large particle counts. A GPU port using
  PyOpenGL / ModernGL would scale to 100 000+ particles.
- Lower webcam resolution via `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)` in
  `main.py` if tracking latency is high.

---

## Extending the project

| Idea | Where to edit |
|---|---|
| Add a new gesture | `ui.py → UIState._single_hand_update()` |
| Change explosion physics | `effects.py → ParticleSphere.draw()` |
| Add audio reactivity | `effects.py` — replace wave formula with FFT magnitude |
| Add more colour modes | `effects.py → hsl_to_bgr()` |
| GPU rendering | Replace `ParticleSphere.draw()` with a ModernGL / PyOpenGL shader |
| Record output | Add `cv2.VideoWriter` in `main.py` after `cv2.imshow` |

---

## Troubleshooting

**Camera not found**
Run `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"` — try index `1` or `2`.

**MediaPipe import error**
Ensure you installed inside the virtual environment: `pip install mediapipe`.

**Laggy / low FPS**
Reduce particle count: `python main.py --particles 1000`.

**Hand not detected**
Ensure good lighting on your hand; avoid cluttered backgrounds.
Try lowering `detection_confidence` in `HandTracker.__init__()` (e.g. `0.5`).

---

## License

MIT — do whatever you like with this code.
