"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { SPHSolver, type ForceSource } from "@/lib/sph";
import { FormationManager, SHAPE_NAMES, type ShapeName } from "@/lib/shapes";
import { THEMES } from "@/lib/themes";
import { AudioReactor } from "@/lib/audio";
import { WebcamManager } from "@/lib/webcam";
import { HandTracker } from "@/lib/tracker-web";
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
  const [viscosity, setViscosity] = useState(55);
  const [gravityPct, setGravityPct] = useState(50);
  const [audioOn, setAudioOn] = useState(false);
  const [webcamOn, setWebcamOn] = useState(false);
  const [trackingOn, setTrackingOn] = useState(false);
  const [trackingLoading, setTrackingLoading] = useState(false);
  const [handsDetected, setHandsDetected] = useState(0);
  const [activeShape, setActiveShape] = useState<string | null>(null);
  const [autoCycle, setAutoCycle] = useState(false);
  const [presentationMode, setPresentationMode] = useState(false);

  // Formation overlay (rendered as HTML for crisp text)
  const [overlayData, setOverlayData] = useState<{
    shapeName: string;
    formulaTex: string;
    formulaPos: [number, number];
    alpha: number;
    formState: string;
  } | null>(null);

  // Refs
  const paramsRef = useRef({
    themeIdx: 0,
    mode: "attract" as Mode,
    viscosity: 0.55,
    gravity: 200,
  });
  const mouseRef = useRef({ x: -1, y: -1, left: false, right: false });
  const audioRef = useRef<AudioReactor | null>(null);
  const webcamRef = useRef<WebcamManager | null>(null);
  const trackerRef = useRef<HandTracker | null>(null);
  const formationRef = useRef<FormationManager | null>(null);
  const rendererRef = useRef<any>(null);

  // Sync state → refs
  useEffect(() => { paramsRef.current.themeIdx = themeIdx; }, [themeIdx]);
  useEffect(() => { paramsRef.current.mode = mode; }, [mode]);
  useEffect(() => { paramsRef.current.viscosity = viscosity / 100; }, [viscosity]);
  useEffect(() => { paramsRef.current.gravity = (gravityPct / 50) * 200; }, [gravityPct]);

  // Audio toggle
  const toggleAudio = useCallback(async () => {
    if (!audioRef.current) audioRef.current = new AudioReactor();
    const ar = audioRef.current;
    if (ar.active) { ar.stop(); setAudioOn(false); }
    else { setAudioOn(await ar.start()); }
  }, []);

  // Webcam toggle
  const toggleWebcam = useCallback(async () => {
    if (!webcamRef.current) webcamRef.current = new WebcamManager();
    const wm = webcamRef.current;
    if (wm.active) {
      // Also stop tracking if active
      if (trackerRef.current?.active) {
        trackerRef.current.dispose();
        setTrackingOn(false);
        setHandsDetected(0);
      }
      rendererRef.current?.clearBackgroundVideo();
      wm.stop();
      setWebcamOn(false);
    } else {
      const video = await wm.start();
      if (video) {
        rendererRef.current?.setBackgroundVideo(video);
        setWebcamOn(true);
      }
    }
  }, []);

  // Hand tracking toggle
  const toggleTracking = useCallback(async () => {
    if (!webcamRef.current?.active) return; // need webcam first
    if (!trackerRef.current) trackerRef.current = new HandTracker();
    const tr = trackerRef.current;
    if (tr.active) {
      tr.dispose();
      setTrackingOn(false);
      setHandsDetected(0);
    } else {
      setTrackingLoading(true);
      try {
        await tr.init(webcamRef.current.video!);
        setTrackingOn(true);
      } catch {
        setTrackingOn(false);
      }
      setTrackingLoading(false);
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

    // Lazy-import WebGLRenderer (Three.js is client-only)
    let renderer: any = null;
    let solver: SPHSolver | null = null;
    let formation: FormationManager | null = null;
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
      mouseRef.current.x = -1; mouseRef.current.y = -1;
      mouseRef.current.left = false; mouseRef.current.right = false;
    };
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
      if (key === "r" && solver && renderer) {
        solver.initialize();
        renderer.reset(THEMES[paramsRef.current.themeIdx]);
      } else if (key === "h") {
        setShowPanel((p) => !p);
      } else if (key === "p") {
        setPresentationMode((p) => !p);
      } else if (key === "a" && formation) {
        setAutoCycle(formation.toggleAutoCycle());
      } else if (key in KEY_SHAPES && formation) {
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
      renderer?.resize(W, H);
    };
    window.addEventListener("resize", onResize);

    // ── Async init (Three.js is heavy, import dynamically) ───────────────
    (async () => {
      const { WebGLRenderer } = await import("@/lib/renderer-webgl");

      if (!running) return;

      solver = new SPHSolver(particles, W, H);
      solver.initialize();
      renderer = new WebGLRenderer(canvas, W, H, Math.max(particles, 3000));
      rendererRef.current = renderer;
      formation = new FormationManager(W, H);
      formationRef.current = formation;

      // ── Animation loop ─────────────────────────────────────────────────
      function frame() {
        if (!running || !solver || !renderer || !formation) return;
        const now = performance.now();
        const dt = Math.min((now - lastTime) / 1000, 0.05);
        lastTime = now;

        const params = paramsRef.current;
        const theme = THEMES[params.themeIdx];
        const m = mouseRef.current;

        solver.mu = params.viscosity;
        solver.gravY = params.gravity;

        // ── Force sources ────────────────────────────────────────────────
        const sources: ForceSource[] = [];

        // Mouse / touch
        if (m.x >= 0 && m.y >= 0) {
          if (m.right) {
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
                sources.push({ x: m.x, y: m.y, strength: CFG.attractStr * 0.25, radius: CFG.handRadius * 1.2, vortex: CFG.vortexStr });
                break;
              case "gravity":
                sources.push({ x: m.x, y: m.y, strength: CFG.gravityWellStr, radius: CFG.handRadius * 2.2 });
                break;
            }
          } else {
            sources.push({ x: m.x, y: m.y, strength: CFG.attractStr * 0.12, radius: CFG.handRadius * 0.5 });
          }
        }

        // Hand tracking
        const tr = trackerRef.current;
        if (tr?.active) {
          tr.detect();
          const handForces = tr.toForceSources(W, H);
          sources.push(...handForces);
          setHandsDetected(tr.hands.length);
        }

        // Formation cached forces (centre attractor)
        solver.sources = [...sources, ...formation.cachedForces];

        // Audio modulation
        const ar = audioRef.current;
        let audioEnergy = 0;
        if (ar?.active) {
          const e = ar.getEnergy();
          audioEnergy = e.overall;
          if (e.bass > 0.25) {
            solver.sources.push({
              x: W / 2, y: H / 2,
              strength: e.bass * 500 * (Math.random() > 0.5 ? 1 : -1),
              radius: 250,
            });
          }
          solver.gravX = Math.sin(now * 0.002) * e.mid * 80;
        }

        // ── Physics ──────────────────────────────────────────────────────
        solver.step(CFG.substeps);

        // ── Formation (post-physics position blending) ───────────────────
        formation.update(
          solver.px, solver.py, solver.vx, solver.vy, solver.temp,
          solver.n, dt,
        );

        // ── Render ───────────────────────────────────────────────────────
        const overlay = formation.getOverlay();

        // Update HTML overlay state for formula text
        if (overlay && overlay.alpha > 0.01) {
          setOverlayData({
            shapeName: overlay.shapeName,
            formulaTex: overlay.formulaTex,
            formulaPos: overlay.formulaPos,
            alpha: overlay.alpha,
            formState: overlay.formState,
          });
        } else {
          setOverlayData(null);
        }

        renderer.render(
          solver.px, solver.py, solver.temp, solver.n,
          theme, overlay,
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
    })();

    return () => {
      running = false;
      renderer?.dispose();
      rendererRef.current = null;
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
  // eslint-disable-next-line react-hooks/exhaustive-deps
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

      {/* ── HTML formula overlay (crisp subpixel text) ──────────────────── */}
      {overlayData && overlayData.alpha > 0.01 && (
        <div
          className="formula-overlay"
          style={{
            left: overlayData.formulaPos[0],
            top: overlayData.formulaPos[1],
            opacity: overlayData.alpha,
          }}
        >
          <div className="formula-label" style={{ color: theme.accent }}>
            {overlayData.shapeName.toUpperCase()}
          </div>
          <div className="formula-text">{overlayData.formulaTex}</div>
        </div>
      )}

      {/* FPS */}
      {!presentationMode && (
        <div className="fps">
          {fps} FPS &middot; {particles}p
          {handsDetected > 0 && ` \u00b7 ${handsDetected} hand${handsDetected > 1 ? "s" : ""}`}
        </div>
      )}

      {/* Control panel */}
      {showPanel && !presentationMode && (
        <div className="panel">
          <h2>Particles Fluid</h2>
          <div className="subtitle">
            WebGL SPH simulation &middot; {particles} particles
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

          {/* Camera section */}
          <div className="section-label">Camera</div>
          <div className="action-row">
            <button className={`action-btn ${webcamOn ? "on" : ""}`} onClick={toggleWebcam}>
              {webcamOn ? "\uD83D\uDCF7 Cam ON" : "\uD83D\uDCF7 Webcam"}
            </button>
            <button
              className={`action-btn ${trackingOn ? "on" : ""}`}
              onClick={toggleTracking}
              disabled={!webcamOn}
              style={{ opacity: webcamOn ? 1 : 0.4 }}
            >
              {trackingLoading ? "\u231B Loading..." : trackingOn ? "\u270B Tracking" : "\u270B Hands"}
            </button>
          </div>
          {trackingOn && handsDetected > 0 && (
            <div style={{ fontSize: 10, color: theme.accent, marginTop: 4 }}>
              {handsDetected} hand{handsDetected > 1 ? "s" : ""} detected
            </div>
          )}

          <div className="sep" />

          {/* Sliders */}
          <div className="section-label">Physics</div>
          <div className="slider-row">
            <span className="slider-label">Particles</span>
            <input type="range" min={300} max={3000} step={100} value={particles}
              onChange={(e) => setParticles(Number(e.target.value))} />
            <span className="slider-val">{particles}</span>
          </div>
          <div className="slider-row">
            <span className="slider-label">Viscosity</span>
            <input type="range" min={5} max={100} value={viscosity}
              onChange={(e) => setViscosity(Number(e.target.value))} />
            <span className="slider-val">{(viscosity / 100).toFixed(2)}</span>
          </div>
          <div className="slider-row">
            <span className="slider-label">Gravity</span>
            <input type="range" min={0} max={100} value={gravityPct}
              onChange={(e) => setGravityPct(Number(e.target.value))} />
            <span className="slider-val">{Math.round((gravityPct / 50) * 200)}</span>
          </div>

          <div className="sep" />

          {/* Actions */}
          <div className="action-row">
            <button
              className={`action-btn ${autoCycle ? "on" : ""}`}
              onClick={() => {
                const fm = formationRef.current;
                if (fm) setAutoCycle(fm.toggleAutoCycle());
              }}
            >
              {autoCycle ? "\u25B6 Cycling" : "\u25B6 Auto"}
            </button>
            <button className={`action-btn ${audioOn ? "on" : ""}`} onClick={toggleAudio}>
              {audioOn ? "\uD83C\uDFA4 ON" : "\uD83C\uDFA4 Mic"}
            </button>
            <button className="action-btn" onClick={screenshot}>
              \uD83D\uDCF8 Snap
            </button>
          </div>
          <div className="action-row" style={{ marginTop: 4 }}>
            <button
              className={`action-btn ${presentationMode ? "on" : ""}`}
              onClick={() => setPresentationMode((p) => !p)}
            >
              \uD83C\uDFAC Present
            </button>
            <button className="action-btn"
              onClick={() => canvasRef.current?.requestFullscreen?.()}>
              \u26F6 Full
            </button>
          </div>

          <div className="sep" />
          <div style={{ fontSize: 10, color: "rgba(255,255,255,0.2)" }}>
            <kbd>1</kbd>-<kbd>9</kbd> shapes &middot; <kbd>R</kbd> reset
            &middot; <kbd>W</kbd> cam &middot; <kbd>T</kbd> track
            &middot; <kbd>P</kbd> present &middot; <kbd>A</kbd> auto
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
          Click to interact &middot; <b>W</b> webcam &middot; <b>T</b> hands
          &middot; <b>P</b> present &middot; <b>A</b> auto &middot; <b>1-9</b> shapes
        </div>
      )}
    </div>
  );
}
