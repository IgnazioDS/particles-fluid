"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { SPHSolver, type ForceSource } from "@/lib/sph";
import { Renderer, type FormationOverlay } from "@/lib/renderer";
import { FormationManager, SHAPE_NAMES, type ShapeName } from "@/lib/shapes";
import { CFG } from "@/lib/config";

// ─────────────────────────────────────────────────────────────────────────────
// Debug shape key map (1-9)
// ─────────────────────────────────────────────────────────────────────────────
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

export default function Page() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fps, setFps] = useState(0);
  const [showPanel, setShowPanel] = useState(true);
  const [gravityOn, setGravityOn] = useState(true);
  const [formationOn, setFormationOn] = useState(true);
  const [activeShape, setActiveShape] = useState<string | null>(null);
  const [particleCount, setParticleCount] = useState(CFG.numParticles);

  // Refs for mutable state inside animation loop
  const mouseRef = useRef({ x: -1, y: -1, left: false, right: false });
  const stateRef = useRef({
    gravityOn: true,
    formationOn: true,
  });

  // Keep stateRef in sync
  useEffect(() => { stateRef.current.gravityOn = gravityOn; }, [gravityOn]);
  useEffect(() => { stateRef.current.formationOn = formationOn; }, [formationOn]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Size to viewport
    let W = window.innerWidth;
    let H = window.innerHeight;

    const solver = new SPHSolver(particleCount, W, H);
    solver.initialize();
    const renderer = new Renderer(canvas, W, H);
    const formation = new FormationManager(W, H);

    // FPS tracking
    let frameCount = 0;
    let fpsT0 = performance.now();
    let running = true;
    let lastTime = performance.now();

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

    // Touch support
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      mouseRef.current.x = (touch.clientX - rect.left) * (W / rect.width);
      mouseRef.current.y = (touch.clientY - rect.top) * (H / rect.height);
      mouseRef.current.left = true;
    };
    const onTouchStart = (e: TouchEvent) => {
      e.preventDefault();
      mouseRef.current.left = true;
      onTouchMove(e);
    };
    const onTouchEnd = () => {
      mouseRef.current.left = false;
    };

    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("contextmenu", onContext);
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd);

    // ── Keyboard ─────────────────────────────────────────────────────────
    const onKey = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();

      if (key === "r") {
        solver.initialize();
        renderer.reset();
      } else if (key === "g") {
        setGravityOn((prev) => {
          const next = !prev;
          if (!next) {
            solver.gravX = 0;
            solver.gravY = 0;
          } else {
            solver.gravX = 0;
            solver.gravY = 200;
          }
          return next;
        });
      } else if (key === "f") {
        setFormationOn((prev) => {
          const next = !prev;
          formation.toggle();
          return next;
        });
      } else if (key === "h") {
        setShowPanel((p) => !p);
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

      // Build force sources from mouse
      const m = mouseRef.current;
      const sources: ForceSource[] = [];
      if (m.x >= 0 && m.y >= 0) {
        if (m.right) {
          sources.push({ x: m.x, y: m.y, strength: CFG.repelStr, radius: CFG.handRadius * 1.35 });
        } else if (m.left) {
          sources.push({ x: m.x, y: m.y, strength: CFG.attractStr, radius: CFG.handRadius });
        } else {
          sources.push({ x: m.x, y: m.y, strength: CFG.attractStr * 0.15, radius: CFG.handRadius * 0.6 });
        }
      }

      // Formation forces
      const extraSources = formation.update(solver.px, solver.py, solver.n, dt);
      solver.sources = [...sources, ...extraSources];

      // Physics
      solver.step(CFG.substeps);

      // Render
      const overlay = formation.getOverlay();
      let fOverlay: FormationOverlay | null = null;
      if (overlay) {
        fOverlay = {
          shapeName: overlay.shapeName,
          formulaTex: overlay.formulaTex,
          formulaPos: overlay.formulaPos,
          alpha: overlay.alpha,
          latticePts: overlay.latticePts,
          nPts: overlay.nPts,
        };
      }

      renderer.render(
        solver.px, solver.py, solver.temp, solver.n,
        fOverlay,
        m.x, m.y, m.left, m.right,
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
      canvas.removeEventListener("contextmenu", onContext);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onResize);
    };
  }, [particleCount]);

  return (
    <div className="relative w-screen h-screen overflow-hidden" style={{ background: CFG.bgColor }}>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ touchAction: "none" }}
      />

      {/* FPS */}
      <div className="fps">FPS {fps}</div>

      {/* Control panel */}
      {showPanel && (
        <div className="panel">
          <h2>Particles Fluid</h2>
          <div className="muted" style={{ marginBottom: 4 }}>
            {particleCount} particles &middot; SPH simulation
          </div>
          <div className="sep" />
          <div><kbd>Click</kbd> attract &nbsp; <kbd>Right-click</kbd> repel</div>
          <div><kbd>1</kbd>-<kbd>9</kbd> trigger shapes</div>
          <div className="sep" />
          <div>
            <kbd>R</kbd> reset &nbsp;
            <kbd>G</kbd> gravity{" "}
            <span className={gravityOn ? "accent" : "muted"}>
              {gravityOn ? "ON" : "OFF"}
            </span>
          </div>
          <div>
            <kbd>F</kbd> formation{" "}
            <span className={formationOn ? "accent" : "muted"}>
              {formationOn ? "ON" : "OFF"}
            </span>
            &nbsp; <kbd>H</kbd> hide panel
          </div>
          <div className="sep" />
          <div className="muted" style={{ fontSize: 10 }}>
            Shapes: circle, ellipse, triangle, square,
            pentagon, hexagon, rose, spiral, lissajous
          </div>
        </div>
      )}

      {/* Active shape toast */}
      {activeShape && (
        <div
          style={{
            position: "absolute",
            top: 16,
            left: "50%",
            transform: "translateX(-50%)",
            padding: "6px 18px",
            background: "rgba(255, 136, 68, 0.15)",
            border: "1px solid rgba(255, 136, 68, 0.3)",
            borderRadius: 8,
            fontSize: 13,
            color: "#ff8844",
            zIndex: 20,
            pointerEvents: "none",
          }}
        >
          Forming: <strong>{activeShape}</strong>
        </div>
      )}

      {/* Bottom hint */}
      <div className="hint-bar">
        Touch / click to interact &middot; Press <b>H</b> to toggle panel &middot; Keys <b>1-9</b> for shapes
      </div>
    </div>
  );
}
