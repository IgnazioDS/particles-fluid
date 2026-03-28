"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { SPHSolver, type ForceSource } from "@/lib/sph";
import { Renderer } from "@/lib/renderer";
import { FormationManager, SHAPE_NAMES, type ShapeName } from "@/lib/shapes";
import { THEMES } from "@/lib/themes";
import { AudioReactor } from "@/lib/audio";
import { CFG } from "@/lib/config";

// ─────────────────────────────────────────────────────────────────────────────
// Types & constants
// ─────────────────────────────────────────────────────────────────────────────

type Mode = "attract" | "repel" | "vortex" | "gravity";

const MODES: { id: Mode; label: string; icon: string }[] = [
  { id: "attract", label: "Attract", icon: "\u25C9" },
  { id: "repel", label: "Repel", icon: "\u25CE" },
  { id: "vortex", label: "Vortex", icon: "\u25CC" },
  { id: "gravity", label: "Well", icon: "\u229B" },
];

const KEY_SHAPES: Record<string, { name: ShapeName; opts?: Record<string, number> }> = {
  "1": { name: "circle" },
  "2": { name: "ellipse", opts: { axisRatio: 1.5 } },
  "3": { name: "triangle" },
  "4": { name: "square" },
  "5": { name: "pentagon" },
  "6": { name: "hexagon" },
  "7": { name: "rose", opts: { nPetals: 3 } },
  "8": { name: "spiral" },
  "9": { name: "lissajous" },
};

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export default function Page() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // UI state
  const [fps, setFps] = useState(0);
  const [showPanel, setShowPanel] = useState(true);
  const [themeIdx, setThemeIdx] = useState(0);
  const [mode, setMode] = useState<Mode>("attract");
  const [particles, setParticles] = useState<number>(CFG.numParticles);
  const [viscosity, setViscosity] = useState(55); // 0-100 slider
  const [gravityPct, setGravityPct] = useState(50); // 0-100 slider
  const [audioOn, setAudioOn] = useState(false);
  const [activeShape, setActiveShape] = useState<string | null>(null);
  const [autoCycle, setAutoCycle] = useState(false);
  const [presentationMode, setPresentationMode] = useState(false);

  // Refs for animation loop (avoid stale closures)
  const paramsRef = useRef({
    themeIdx: 0,
    mode: "attract" as Mode,
    viscosity: 0.55,
    gravity: 200,
  });
  const mouseRef = useRef({ x: -1, y: -1, left: false, right: false });
  const audioRef = useRef<AudioReactor | null>(null);
  const formationRef = useRef<FormationManager | null>(null);

  // Sync state → refs
  useEffect(() => { paramsRef.current.themeIdx = themeIdx; }, [themeIdx]);
  useEffect(() => { paramsRef.current.mode = mode; }, [mode]);
  useEffect(() => { paramsRef.current.viscosity = viscosity / 100; }, [viscosity]);
  useEffect(() => { paramsRef.current.gravity = (gravityPct / 50) * 200; }, [gravityPct]);

  // Audio toggle
  const toggleAudio = useCallback(async () => {
    if (!audioRef.current) audioRef.current = new AudioReactor();
    const ar = audioRef.current;
    if (ar.active) {
      ar.stop();
      setAudioOn(false);
    } else {
      const ok = await ar.start();
      setAudioOn(ok);
    }
  }, []);

  // Screenshot
  const screenshot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `particles-fluid-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }, []);

  // ── Main simulation effect ─────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let W = window.innerWidth;
    let H = window.innerHeight;

    const solver = new SPHSolver(particles, W, H);
    solver.initialize();
    const renderer = new Renderer(canvas, W, H);
    const formation = new FormationManager(W, H);
    formationRef.current = formation;

    let running = true;
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsT0 = performance.now();

    // ── Mouse handlers ───────────────────────────────────────────────────
    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.x = (e.clientX - rect.left) * (W / rect.width);
      mouseRef.current.y = (e.clientY - rect.top) * (H / rect.height);
    };
    const onMouseDown = (e: MouseEvent) => {
      if (e.button === 0) mouseRef.current.left = true;
      if (e.button === 2) mouseRef.current.right = true;
    };
    const onMouseUp = (e: MouseEvent) => {
      if (e.button === 0) mouseRef.current.left = false;
      if (e.button === 2) mouseRef.current.right = false;
    };
    const onContext = (e: Event) => e.preventDefault();
    const onMouseLeave = () => {
      mouseRef.current.x = -1;
      mouseRef.current.y = -1;
      mouseRef.current.left = false;
      mouseRef.current.right = false;
    };

    // Touch
    const onTouchStart = (e: TouchEvent) => {
      e.preventDefault();
      const t = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.x = (t.clientX - rect.left) * (W / rect.width);
      mouseRef.current.y = (t.clientY - rect.top) * (H / rect.height);
      mouseRef.current.left = true;
    };
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      const t = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.x = (t.clientX - rect.left) * (W / rect.width);
      mouseRef.current.y = (t.clientY - rect.top) * (H / rect.height);
    };
    const onTouchEnd = () => { mouseRef.current.left = false; };

    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("mouseleave", onMouseLeave);
    canvas.addEventListener("contextmenu", onContext);
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd);

    // ── Keyboard ─────────────────────────────────────────────────────────
    const onKey = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key === "r") {
        solver.initialize();
        renderer.reset(THEMES[paramsRef.current.themeIdx].bg);
      } else if (key === "h") {
        setShowPanel((p) => !p);
      } else if (key === "p") {
        setPresentationMode((p) => !p);
      } else if (key === "a") {
        const on = formation.toggleAutoCycle();
        setAutoCycle(on);
      } else if (key in KEY_SHAPES) {
        const { name, opts } = KEY_SHAPES[key];
        formation.forceShape(name, opts);
        setActiveShape(name);
        setTimeout(() => setActiveShape(null), 4500);
      }
    };
    window.addEventListener("keydown", onKey);

    // ── Resize ───────────────────────────────────────────────────────────
    const onResize = () => {
      W = window.innerWidth;
      H = window.innerHeight;
      renderer.resize(W, H);
    };
    window.addEventListener("resize", onResize);

    // ── Animation loop ───────────────────────────────────────────────────
    function frame() {
      if (!running) return;
      const now = performance.now();
      const dt = Math.min((now - lastTime) / 1000, 0.05);
      lastTime = now;

      const params = paramsRef.current;
      const theme = THEMES[params.themeIdx];
      const m = mouseRef.current;

      // Update solver params from sliders
      solver.mu = params.viscosity;
      solver.gravY = params.gravity;

      // Build force sources from interaction mode
      const sources: ForceSource[] = [];
      if (m.x >= 0 && m.y >= 0) {
        if (m.right) {
          // Right-click always repels
          sources.push({ x: m.x, y: m.y, strength: CFG.repelStr, radius: CFG.handRadius * 1.35 });
        } else if (m.left) {
          switch (params.mode) {
            case "attract":
              sources.push({ x: m.x, y: m.y, strength: CFG.attractStr, radius: CFG.handRadius });
              break;
            case "repel":
              sources.push({ x: m.x, y: m.y, strength: CFG.repelStr, radius: CFG.handRadius * 1.35 });
              break;
            case "vortex":
              sources.push({
                x: m.x, y: m.y,
                strength: CFG.attractStr * 0.25,
                radius: CFG.handRadius * 1.2,
                vortex: CFG.vortexStr,
              });
              break;
            case "gravity":
              sources.push({ x: m.x, y: m.y, strength: CFG.gravityWellStr, radius: CFG.handRadius * 2.2 });
              break;
          }
        } else {
          // Hover — gentle attract
          sources.push({ x: m.x, y: m.y, strength: CFG.attractStr * 0.12, radius: CFG.handRadius * 0.5 });
        }
      }

      // Formation: set cached forces BEFORE physics (centre attractor for free particles)
      solver.sources = [...sources, ...formation.cachedForces];

      // Audio modulation
      const ar = audioRef.current;
      let audioEnergy = 0;
      if (ar && ar.active) {
        const e = ar.getEnergy();
        audioEnergy = e.overall;
        // Bass → pulse force at centre
        if (e.bass > 0.25) {
          solver.sources.push({
            x: W / 2, y: H / 2,
            strength: e.bass * 500 * (Math.random() > 0.5 ? 1 : -1),
            radius: 250,
          });
        }
        // Mid → gravity wobble
        solver.gravX = Math.sin(now * 0.002) * e.mid * 80;
      }

      // Physics
      solver.step(CFG.substeps);

      // Formation: apply position overrides AFTER physics (direct blending)
      formation.update(
        solver.px, solver.py, solver.vx, solver.vy, solver.temp,
        solver.n, dt,
      );

      // Render
      const fOverlay = formation.getOverlay();

      renderer.render(
        solver.px, solver.py, solver.temp, solver.n,
        theme,
        fOverlay,
        { x: m.x, y: m.y, active: m.left || m.right, mode: m.right ? "repel" : params.mode },
        audioEnergy,
      );

      // FPS
      frameCount++;
      if (now - fpsT0 >= 1000) {
        setFps(Math.round(frameCount / ((now - fpsT0) / 1000)));
        frameCount = 0;
        fpsT0 = now;
      }

      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    return () => {
      running = false;
      canvas.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("mouseup", onMouseUp);
      canvas.removeEventListener("mouseleave", onMouseLeave);
      canvas.removeEventListener("contextmenu", onContext);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
    };
  }, [particles]);

  const theme = THEMES[themeIdx];

  return (
    <div
      className="relative w-screen h-screen overflow-hidden"
      style={{ background: theme.bg, ["--accent" as string]: theme.accent }}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ touchAction: "none" }}
      />

      {/* FPS */}
      {!presentationMode && (
        <div className="fps">
          {fps} FPS &middot; {particles}p
        </div>
      )}

      {/* Control panel */}
      {showPanel && !presentationMode && (
        <div className="panel">
          <h2>Particles Fluid</h2>
          <div className="subtitle">
            Interactive SPH simulation &middot; {particles} particles
          </div>

          <div className="sep" />

          {/* Interaction mode */}
          <div className="section-label">Interaction</div>
          <div className="mode-grid">
            {MODES.map((m) => (
              <button
                key={m.id}
                className={`mode-btn ${mode === m.id ? "active" : ""}`}
                onClick={() => setMode(m.id)}
              >
                <span className="mode-icon">{m.icon}</span>
                {m.label}
              </button>
            ))}
          </div>

          <div className="sep" />

          {/* Theme picker */}
          <div className="section-label">Theme</div>
          <div className="theme-row">
            {THEMES.map((t, i) => (
              <div
                key={t.name}
                className={`theme-dot ${themeIdx === i ? "active" : ""}`}
                style={{ background: t.lut[180] }}
                title={t.name}
                onClick={() => setThemeIdx(i)}
              />
            ))}
          </div>

          <div className="sep" />

          {/* Sliders */}
          <div className="section-label">Physics</div>
          <div className="slider-row">
            <span className="slider-label">Particles</span>
            <input
              type="range"
              min={200}
              max={2000}
              step={100}
              value={particles}
              onChange={(e) => setParticles(Number(e.target.value))}
            />
            <span className="slider-val">{particles}</span>
          </div>
          <div className="slider-row">
            <span className="slider-label">Viscosity</span>
            <input
              type="range"
              min={5}
              max={100}
              value={viscosity}
              onChange={(e) => setViscosity(Number(e.target.value))}
            />
            <span className="slider-val">{(viscosity / 100).toFixed(2)}</span>
          </div>
          <div className="slider-row">
            <span className="slider-label">Gravity</span>
            <input
              type="range"
              min={0}
              max={100}
              value={gravityPct}
              onChange={(e) => setGravityPct(Number(e.target.value))}
            />
            <span className="slider-val">{Math.round((gravityPct / 50) * 200)}</span>
          </div>

          <div className="sep" />

          {/* Actions */}
          <div className="action-row">
            <button
              className={`action-btn ${autoCycle ? "on" : ""}`}
              onClick={() => {
                const fm = formationRef.current;
                if (fm) {
                  const on = fm.toggleAutoCycle();
                  setAutoCycle(on);
                }
              }}
            >
              {autoCycle ? "\u25B6 Cycling" : "\u25B6 Auto"}
            </button>
            <button className={`action-btn ${audioOn ? "on" : ""}`} onClick={toggleAudio}>
              {audioOn ? "\uD83C\uDFA4 ON" : "\uD83C\uDFA4 Mic"}
            </button>
            <button className="action-btn" onClick={screenshot}>
              \uD83D\uDCF7 Snap
            </button>
          </div>
          <div className="action-row" style={{ marginTop: 4 }}>
            <button
              className={`action-btn ${presentationMode ? "on" : ""}`}
              onClick={() => setPresentationMode((p) => !p)}
            >
              \uD83C\uDFAC Present
            </button>
            <button
              className="action-btn"
              onClick={() => {
                const canvas = canvasRef.current;
                if (canvas) canvas.requestFullscreen?.();
              }}
            >
              \u26F6 Full
            </button>
          </div>

          <div className="sep" />

          <div style={{ fontSize: 10, color: "rgba(255,255,255,0.2)" }}>
            <kbd>1</kbd>-<kbd>9</kbd> shapes &middot; <kbd>R</kbd> reset
            &middot; <kbd>H</kbd> hide &middot; <kbd>P</kbd> present
            &middot; <kbd>A</kbd> auto-cycle
          </div>
        </div>
      )}

      {/* Active shape toast */}
      {activeShape && (
        <div className="toast">
          Forming: <strong>{activeShape}</strong>
        </div>
      )}

      {/* Bottom hint */}
      {!presentationMode && (
        <div className="hint-bar">
          Click to interact &middot; <b>H</b> panel &middot; <b>P</b> present
          &middot; <b>A</b> auto-cycle &middot; <b>1-9</b> shapes
        </div>
      )}
    </div>
  );
}
