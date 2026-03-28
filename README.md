# Particles Fluid 🌊🔥

> Real-time 2-D SPH fluid simulation driven by hand gestures and facial expressions.  
> Built with **Taichi** (GPU-parallel physics), **MediaPipe** (ML tracking), and **OpenCV** (rendering).

![demo](https://placehold.co/900x400/0d0d0c/d97757?text=Particles+Fluid+Demo)

---

## Features

| Gesture | Effect |
|---------|--------|
| ✋ Open palm | Attract nearby particles to your palm |
| ✊ Fist | Explosive repel burst |
| ☝️ Point | Directional push along finger axis |
| 🤏 Pinch | Gravity-well vortex at pinch point |
| ✌️ Peace | Dual attract from index + middle tips |
| 🤟 Spider | Split attract from index + pinky |
| 👄 Mouth open | Blow particles outward (scales with openness) |
| 😮 Head tilt | Tilt the gravity axis left/right |

---

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Run (webcam + CPU)
make run

# 3. Or use GPU backend (CUDA / Apple Metal)
make run-gpu

# 4. No webcam? Mouse-demo mode
make run-demo
```

### Manual run

```bash
python main.py
python main.py --no-webcam        # mouse demo
python main.py --gpu              # GPU backend
python main.py --particles 2500   # more particles
```

---

## Requirements

- Python 3.10+
- Webcam (optional — mouse fallback available)
- For GPU: CUDA 11+ (NVIDIA) or Metal (Apple Silicon)

```
taichi>=1.7.0
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
```

---

## Architecture

```
particles-fluid/
├── main.py                   ← entry point, main loop, gesture→force mapping
├── config.py                 ← all tunable parameters in one place
└── src/
    ├── simulation/
    │   └── sph.py            ← Taichi SPH solver (density, pressure, viscosity)
    ├── tracking/
    │   ├── tracker.py        ← MediaPipe Hands + FaceMesh wrapper
    │   └── gesture.py        ← gesture classification from landmarks
    └── rendering/
        └── renderer.py       ← OpenCV renderer (glow, bloom, trails, debug HUD)
```

### Physics — Smoothed Particle Hydrodynamics (SPH)

The fluid is modelled as 1 500 particles, each carrying density ρ, pressure p, and velocity v.
Every frame, three Taichi kernels run:

1. **Density** — sum `Poly6(r)` contributions from all neighbours within radius `h`
2. **Pressure** — Tait equation of state: `p = k (ρ − ρ₀)`, clamped ≥ 0
3. **Forces + integration** — `Spiky` pressure gradient + viscosity Laplacian + gravity + external hand/face forces → semi-implicit Euler step

All kernels are parallelised across CPU cores (or GPU threads with `--gpu`).

### Rendering Pipeline

```
particle buffer  →  fade (motion trail)
                 →  draw particles (cv2.circle per particle)
                 →  Gaussian glow (σ = 7 px)
                 →  Gaussian bloom (σ = 22 px)
                 →  additive blend with webcam feed
                 →  HUD overlay
```

Particle colour is driven by a **visual temperature** field that heats up near force sources and cools over time, producing the lava/plasma aesthetic.

---

## Configuration

All parameters live in `config.py`.  Key knobs:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `num_particles` | 1500 | More = denser fluid, slower |
| `smoothing_h` | 26 px | SPH interaction radius |
| `pressure_k` | 200 | How strongly fluid resists compression |
| `viscosity` | 0.55 | How quickly velocity differences smooth out |
| `gravity` | 200 px/s² | Downward pull |
| `glow_sigma` | 7.0 | Tight glow radius |
| `bloom_sigma` | 22.0 | Wide bloom radius |
| `webcam_alpha` | 0.35 | Webcam feed visibility |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `R` | Reset simulation |
| `G` | Toggle gravity on/off |
| `W` | Toggle webcam overlay |
| `D` | Toggle debug HUD |

---

## Performance Tips

| Scenario | Command |
|----------|---------|
| Slow machine | `make run-fast` (800 particles) |
| Fast machine | `make run-hd` (2500 particles) |
| NVIDIA GPU | `make run-gpu` |
| No webcam | `make run-demo` |

---

## Roadmap (Phase 2)

- [ ] Surface tension forces (Müller 2003 cohesion term)
- [ ] Multiple fluid colours per hand (left/right tinted differently)
- [ ] Vorticity confinement for more turbulent swirling
- [ ] Record / export particle frames as GIF or MP4
- [ ] OSC / MIDI parameter control for live performance

---

## License

MIT
