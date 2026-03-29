/**
 * WebGL renderer using Three.js — replaces Canvas 2D renderer.
 *
 * Pipeline:
 *  1. Trail FBO: fade previous frame + draw particles (ping-pong)
 *  2. Post-processing: UnrealBloom + vignette via EffectComposer
 *  3. Overlays: formation outline (THREE.Line), cursor ring
 *  4. Background: solid color or webcam VideoTexture
 *
 * Single draw call for all particles via THREE.Points + custom ShaderMaterial.
 * Theme LUT uploaded as 256×1 DataTexture.
 */

import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import type { Theme } from "./themes";
import type { FormationOverlay } from "./shapes";
import { CFG } from "./config";

// ─────────────────────────────────────────────────────────────────────────────
// Vignette shader
// ─────────────────────────────────────────────────────────────────────────────

const VignetteShader = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uDarkness: { value: 0.45 },
    uOffset: { value: 0.9 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uDarkness;
    uniform float uOffset;
    varying vec2 vUv;
    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      float dist = distance(vUv, vec2(0.5));
      float vig = smoothstep(uOffset, uOffset - 0.4, dist);
      color.rgb *= mix(1.0 - uDarkness, 1.0, vig);
      gl_FragColor = color;
    }
  `,
};

// ─────────────────────────────────────────────────────────────────────────────
// Trail fade shader (fullscreen quad that fades previous FBO toward black)
// For additive compositing: dark = transparent, so fade toward 0
// ─────────────────────────────────────────────────────────────────────────────

const TrailFadeShader = {
  uniforms: {
    tPrev: { value: null as THREE.Texture | null },
    uFade: { value: 0.94 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tPrev;
    uniform float uFade;
    varying vec2 vUv;
    void main() {
      vec4 prev = texture2D(tPrev, vUv);
      gl_FragColor = vec4(prev.rgb * uFade, 1.0);
    }
  `,
};

// ─────────────────────────────────────────────────────────────────────────────
// Hand skeleton connections (MediaPipe 21-landmark model)
// ─────────────────────────────────────────────────────────────────────────────

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],          // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],           // index
  [0, 9], [9, 10], [10, 11], [11, 12],      // middle
  [0, 13], [13, 14], [14, 15], [15, 16],    // ring
  [0, 17], [17, 18], [18, 19], [19, 20],    // pinky
  [5, 9], [9, 13], [13, 17],                // palm knuckle row
];
// 23 connections × 2 pts × 2 hands = 92 max vertices
const SKELETON_MAX_VERTS = 96;

// ─────────────────────────────────────────────────────────────────────────────
// Particle shaders
// ─────────────────────────────────────────────────────────────────────────────

const particleVertexShader = /* glsl */ `
  attribute float temperature;
  uniform float uPointSize;
  uniform float uAudioScale;
  varying float vTemp;
  void main() {
    vTemp = temperature;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uPointSize * (1.0 + temperature * 0.6) * uAudioScale;
  }
`;

const particleFragmentShader = /* glsl */ `
  uniform sampler2D uLut;
  varying float vTemp;
  void main() {
    float dist = length(gl_PointCoord - 0.5) * 2.0;
    if (dist > 1.0) discard;
    // Multi-layer glow in single shader
    float core = smoothstep(0.35, 0.0, dist);
    float glow = smoothstep(1.0, 0.0, dist);
    float intensity = core * 2.0 + glow * 0.6;
    vec3 color = texture2D(uLut, vec2(vTemp, 0.5)).rgb;
    color = max(color, vec3(0.15, 0.03, 0.08));
    gl_FragColor = vec4(color * intensity, max(glow, core));
  }
`;

// ─────────────────────────────────────────────────────────────────────────────
// WebGL Renderer
// ─────────────────────────────────────────────────────────────────────────────

export class WebGLRenderer {
  private W: number;
  private H: number;

  // Three.js core
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;

  // Particle system
  private points: THREE.Points;
  private geometry: THREE.BufferGeometry;
  private material: THREE.ShaderMaterial;
  private posAttr: THREE.BufferAttribute;
  private tempAttr: THREE.BufferAttribute;
  private posArray: Float32Array;
  private tempArray: Float32Array;
  private lutTexture: THREE.DataTexture;

  // Trail system (ping-pong FBOs)
  private trailFBOs: [THREE.WebGLRenderTarget, THREE.WebGLRenderTarget];
  private trailIdx = 0;
  private trailScene: THREE.Scene;
  private trailCamera: THREE.OrthographicCamera;
  private trailQuad: THREE.Mesh;
  private trailMaterial: THREE.ShaderMaterial;
  private trailDisplayMesh: THREE.Mesh;    // shows trail FBO in main scene (additive)
  private trailDisplayMat: THREE.MeshBasicMaterial;

  // Hand skeleton
  private skeletonGeo: THREE.BufferGeometry;
  private skeletonMat: THREE.LineBasicMaterial;
  private skeletonLines: THREE.LineSegments;
  private skeletonPosAttr: THREE.BufferAttribute;

  // Post-processing
  private composer: EffectComposer;
  private bloomPass: UnrealBloomPass;

  // Formation outline
  private outlineLine: THREE.Line | null = null;
  private outlineMat: THREE.LineBasicMaterial;

  // Background
  private bgMesh: THREE.Mesh;
  private bgMaterial: THREE.MeshBasicMaterial;
  private videoTexture: THREE.VideoTexture | null = null;

  // Cursor
  private cursorRing: THREE.Line;
  private cursorMat: THREE.LineBasicMaterial;
  private cursorDot: THREE.Mesh;
  private cursorDotMat: THREE.MeshBasicMaterial;

  private maxParticles: number;

  constructor(canvas: HTMLCanvasElement, w: number, h: number, maxParticles = 3000) {
    this.W = w;
    this.H = h;
    this.maxParticles = maxParticles;

    // ── Three.js setup ───────────────────────────────────────────────────
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
    this.renderer.setSize(w, h, false);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.autoClear = false;

    this.scene = new THREE.Scene();
    // Orthographic: particle px/py map directly to screen coords (Y flipped)
    // OrthographicCamera(left, right, top, bottom, near, far)
    // top=h, bottom=0 so Y increases upward (GL convention)
    this.camera = new THREE.OrthographicCamera(0, w, h, 0, -10, 10);
    this.camera.position.z = 1;

    // ── Background quad ──────────────────────────────────────────────────
    this.bgMaterial = new THREE.MeshBasicMaterial({ color: 0x060410 });
    const bgGeo = new THREE.PlaneGeometry(w, h);
    this.bgMesh = new THREE.Mesh(bgGeo, this.bgMaterial);
    this.bgMesh.position.set(w / 2, h / 2, -5);
    this.scene.add(this.bgMesh);

    // ── LUT texture ──────────────────────────────────────────────────────
    const lutData = new Uint8Array(256 * 4);
    this.lutTexture = new THREE.DataTexture(lutData, 256, 1, THREE.RGBAFormat);
    this.lutTexture.minFilter = THREE.LinearFilter;
    this.lutTexture.magFilter = THREE.LinearFilter;
    this.lutTexture.needsUpdate = true;

    // ── Particle geometry + material ─────────────────────────────────────
    this.geometry = new THREE.BufferGeometry();
    this.posArray = new Float32Array(maxParticles * 3);
    this.tempArray = new Float32Array(maxParticles);
    this.posAttr = new THREE.BufferAttribute(this.posArray, 3);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.tempAttr = new THREE.BufferAttribute(this.tempArray, 1);
    this.tempAttr.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute("position", this.posAttr);
    this.geometry.setAttribute("temperature", this.tempAttr);
    this.geometry.setDrawRange(0, 0);

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uLut: { value: this.lutTexture },
        uPointSize: { value: CFG.pointSize },
        uAudioScale: { value: 1.0 },
      },
      vertexShader: particleVertexShader,
      fragmentShader: particleFragmentShader,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false,
    });

    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    // ── Trail FBOs (ping-pong) ───────────────────────────────────────────
    const fboOpts: THREE.RenderTargetOptions = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
    };
    this.trailFBOs = [
      new THREE.WebGLRenderTarget(w, h, fboOpts),
      new THREE.WebGLRenderTarget(w, h, fboOpts),
    ];
    this.trailScene = new THREE.Scene();
    this.trailCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.trailMaterial = new THREE.ShaderMaterial({
      uniforms: TrailFadeShader.uniforms,
      vertexShader: TrailFadeShader.vertexShader,
      fragmentShader: TrailFadeShader.fragmentShader,
    });
    this.trailQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.trailMaterial);
    this.trailScene.add(this.trailQuad);

    // Trail display mesh: fullscreen plane that shows the trail FBO via additive blending.
    // Additive blending means dark pixels add nothing, so webcam shows through.
    this.trailDisplayMat = new THREE.MeshBasicMaterial({
      map: null,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const trailDisplayGeo = new THREE.PlaneGeometry(w, h);
    this.trailDisplayMesh = new THREE.Mesh(trailDisplayGeo, this.trailDisplayMat);
    this.trailDisplayMesh.position.set(w / 2, h / 2, -4.9);
    this.trailDisplayMesh.frustumCulled = false;
    this.scene.add(this.trailDisplayMesh);

    // ── Hand skeleton ────────────────────────────────────────────────────
    const skeletonPosArray = new Float32Array(SKELETON_MAX_VERTS * 3);
    this.skeletonPosAttr = new THREE.BufferAttribute(skeletonPosArray, 3);
    this.skeletonPosAttr.setUsage(THREE.DynamicDrawUsage);
    this.skeletonGeo = new THREE.BufferGeometry();
    this.skeletonGeo.setAttribute("position", this.skeletonPosAttr);
    this.skeletonGeo.setDrawRange(0, 0);
    this.skeletonMat = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.4,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.skeletonLines = new THREE.LineSegments(this.skeletonGeo, this.skeletonMat);
    this.skeletonLines.frustumCulled = false;
    this.skeletonLines.visible = false;
    this.scene.add(this.skeletonLines);

    // ── Post-processing ──────────────────────────────────────────────────
    this.composer = new EffectComposer(this.renderer);
    const renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    // Half-resolution bloom for better performance
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(Math.floor(w / 2), Math.floor(h / 2)),
      CFG.bloomStrength,
      CFG.bloomRadius,
      CFG.bloomThreshold,
    );
    this.composer.addPass(this.bloomPass);

    const vignettePass = new ShaderPass(VignetteShader);
    this.composer.addPass(vignettePass);

    // ── Formation outline ────────────────────────────────────────────────
    this.outlineMat = new THREE.LineBasicMaterial({
      color: 0xff8844,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
    });

    // ── Cursor ───────────────────────────────────────────────────────────
    const ringGeo = new THREE.BufferGeometry();
    const ringPts: number[] = [];
    for (let i = 0; i <= 64; i++) {
      const a = (Math.PI * 2 * i) / 64;
      ringPts.push(Math.cos(a), Math.sin(a), 0);
    }
    ringGeo.setAttribute("position", new THREE.Float32BufferAttribute(ringPts, 3));
    this.cursorMat = new THREE.LineBasicMaterial({ color: 0x50dc64, transparent: true, opacity: 0.25 });
    this.cursorRing = new THREE.Line(ringGeo, this.cursorMat);
    this.cursorRing.visible = false;
    this.scene.add(this.cursorRing);

    const dotGeo = new THREE.CircleGeometry(3, 16);
    this.cursorDotMat = new THREE.MeshBasicMaterial({ color: 0x50dc64, transparent: true, opacity: 0.3 });
    this.cursorDot = new THREE.Mesh(dotGeo, this.cursorDotMat);
    this.cursorDot.visible = false;
    this.scene.add(this.cursorDot);
  }

  // ── Theme LUT upload ─────────────────────────────────────────────────────

  private uploadLut(theme: Theme): void {
    const data = this.lutTexture.image.data as Uint8Array;
    for (let i = 0; i < 256; i++) {
      const [r, g, b] = theme.rgb[i];
      data[i * 4] = r;
      data[i * 4 + 1] = g;
      data[i * 4 + 2] = b;
      data[i * 4 + 3] = 255;
    }
    this.lutTexture.needsUpdate = true;
  }

  // ── Webcam background ────────────────────────────────────────────────────

  setBackgroundVideo(video: HTMLVideoElement): void {
    this.videoTexture = new THREE.VideoTexture(video);
    this.videoTexture.minFilter = THREE.LinearFilter;
    this.videoTexture.magFilter = THREE.LinearFilter;
    this.bgMaterial.map = this.videoTexture;
    this.bgMaterial.color.set(0xffffff);
    this.bgMaterial.needsUpdate = true;
    // Dim webcam feed
    this.bgMaterial.opacity = 0.35;
    this.bgMaterial.transparent = true;
  }

  clearBackgroundVideo(): void {
    this.videoTexture?.dispose();
    this.videoTexture = null;
    this.bgMaterial.map = null;
    this.bgMaterial.color.set(0x060410);
    this.bgMaterial.opacity = 1;
    this.bgMaterial.transparent = false;
    this.bgMaterial.needsUpdate = true;
  }

  // ── Resize ───────────────────────────────────────────────────────────────

  resize(w: number, h: number): void {
    this.W = w;
    this.H = h;
    this.renderer.setSize(w, h, false);
    this.camera.right = w;
    this.camera.top = h;
    this.camera.bottom = 0;
    this.camera.updateProjectionMatrix();
    this.composer.setSize(w, h);
    this.bloomPass.resolution.set(w, h);
    this.trailFBOs[0].setSize(w, h);
    this.trailFBOs[1].setSize(w, h);
    // Clear FBOs after resize to avoid stretched trail artifacts
    this.clearTrailFBOs();
    this.bgMesh.geometry.dispose();
    this.bgMesh.geometry = new THREE.PlaneGeometry(w, h);
    this.bgMesh.position.set(w / 2, h / 2, -5);
    this.trailDisplayMesh.geometry.dispose();
    this.trailDisplayMesh.geometry = new THREE.PlaneGeometry(w, h);
    this.trailDisplayMesh.position.set(w / 2, h / 2, -4.9);
  }

  private clearTrailFBOs(): void {
    for (const fbo of this.trailFBOs) {
      this.renderer.setRenderTarget(fbo);
      this.renderer.clear();
    }
    this.renderer.setRenderTarget(null);
  }

  reset(theme: Theme): void {
    this.uploadLut(theme);
    this.setBgColor(theme);
    this.clearTrailFBOs();
  }

  // ── Main render ──────────────────────────────────────────────────────────

  render(
    px: Float32Array,
    py: Float32Array,
    temp: Float32Array,
    n: number,
    theme: Theme,
    overlay: FormationOverlay | null,
    cursor: { x: number; y: number; active: boolean; mode: string },
    audioEnergy: number,
    hands?: Array<{ x: number; y: number; z: number }[]>,
  ): void {
    // Upload theme LUT
    this.uploadLut(theme);
    this.setBgColor(theme);

    // Update particle buffers
    const drawCount = Math.min(n, this.maxParticles);
    for (let i = 0; i < drawCount; i++) {
      this.posArray[i * 3] = px[i];
      this.posArray[i * 3 + 1] = this.H - py[i]; // flip Y for GL
      this.posArray[i * 3 + 2] = 0;
      this.tempArray[i] = temp[i];
    }
    this.posAttr.needsUpdate = true;
    this.tempAttr.needsUpdate = true;
    this.geometry.setDrawRange(0, drawCount);

    // Audio scale
    this.material.uniforms.uAudioScale.value = 1 + audioEnergy * 0.6;

    // ── Formation outline ────────────────────────────────────────────────
    this.updateOutline(overlay, theme);

    // ── Cursor ───────────────────────────────────────────────────────────
    this.updateCursor(cursor);

    // ── Hand skeleton ────────────────────────────────────────────────────
    this.updateHandSkeleton(hands);

    // ── Trail FBO pass ───────────────────────────────────────────────────
    // Pass 1: fade previous trail → write FBO
    const read = this.trailFBOs[this.trailIdx];
    const write = this.trailFBOs[1 - this.trailIdx];
    this.trailMaterial.uniforms.tPrev.value = read.texture;
    this.trailMaterial.uniforms.uFade.value = 1 - CFG.trailFade;

    this.renderer.setRenderTarget(write);
    this.renderer.autoClear = true;
    this.renderer.clear();
    this.renderer.render(this.trailScene, this.trailCamera);

    // Pass 2: render particles + outline into same write FBO (no clear)
    this.renderer.autoClear = false;
    this.bgMesh.visible = false;        // bg already handled by fade
    this.trailDisplayMesh.visible = false;
    this.skeletonLines.visible = false; // skeleton rendered to screen separately
    this.points.visible = true;
    this.cursorRing.visible = cursor.x >= 0;
    this.cursorDot.visible = cursor.x >= 0;
    this.renderer.render(this.scene, this.camera);
    this.renderer.autoClear = true;

    this.trailIdx ^= 1;

    // ── Composite pass: bgMesh + trailDisplayMesh → bloom + vignette ─────
    this.trailDisplayMat.map = write.texture;
    this.trailDisplayMat.needsUpdate = true;

    this.bgMesh.visible = true;
    this.trailDisplayMesh.visible = true;
    this.points.visible = false;        // already in trail FBO
    this.cursorRing.visible = false;    // already in trail FBO
    this.cursorDot.visible = false;     // already in trail FBO

    // Hand skeleton on top of bloom (crisp, not bloomed)
    if (hands && hands.length > 0) {
      this.skeletonLines.visible = true;
    }

    this.renderer.setRenderTarget(null);
    this.composer.render();
  }

  // ── Hand skeleton ────────────────────────────────────────────────────────

  private updateHandSkeleton(
    hands?: Array<{ x: number; y: number; z: number }[]>,
  ): void {
    if (!hands || hands.length === 0) {
      this.skeletonLines.visible = false;
      this.skeletonGeo.setDrawRange(0, 0);
      return;
    }

    const pos = this.skeletonPosAttr.array as Float32Array;
    let vtxCount = 0;

    for (const lm of hands) {
      if (lm.length < 21) continue;
      for (const [a, b] of HAND_CONNECTIONS) {
        if (vtxCount + 2 > SKELETON_MAX_VERTS) break;
        // Mirror X, flip Y to match GL coords
        pos[vtxCount * 3]     = (1 - lm[a].x) * this.W;
        pos[vtxCount * 3 + 1] = this.H - lm[a].y * this.H;
        pos[vtxCount * 3 + 2] = 0.9;
        vtxCount++;
        pos[vtxCount * 3]     = (1 - lm[b].x) * this.W;
        pos[vtxCount * 3 + 1] = this.H - lm[b].y * this.H;
        pos[vtxCount * 3 + 2] = 0.9;
        vtxCount++;
      }
    }

    this.skeletonGeo.setDrawRange(0, vtxCount);
    this.skeletonPosAttr.needsUpdate = true;
  }

  // ── Outline ──────────────────────────────────────────────────────────────

  private updateOutline(overlay: FormationOverlay | null, theme: Theme): void {
    if (this.outlineLine) {
      this.scene.remove(this.outlineLine);
      this.outlineLine.geometry.dispose();
      this.outlineLine = null;
    }

    if (!overlay || overlay.outlineN < 2 || overlay.alpha < 0.01) return;

    const pts: number[] = [];
    for (let i = 0; i < overlay.outlineN; i++) {
      pts.push(
        overlay.outlinePts[i * 2],
        this.H - overlay.outlinePts[i * 2 + 1], // flip Y
        0.5,
      );
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    this.outlineMat.color.set(theme.accent);
    this.outlineMat.opacity = overlay.alpha * 0.6;
    this.outlineLine = new THREE.Line(geo, this.outlineMat);
    this.outlineLine.frustumCulled = false;
    this.scene.add(this.outlineLine);
  }

  // ── Cursor ───────────────────────────────────────────────────────────────

  private updateCursor(cursor: { x: number; y: number; active: boolean; mode: string }): void {
    if (cursor.x < 0 || cursor.y < 0) {
      this.cursorRing.visible = false;
      this.cursorDot.visible = false;
      return;
    }

    const colors: Record<string, number> = {
      attract: 0x50dc64,
      repel: 0xdc3c50,
      vortex: 0x648cff,
      gravity: 0xffc83c,
    };
    const color = colors[cursor.mode] ?? 0x50dc64;
    const rad = cursor.mode === "gravity" ? CFG.handRadius * 2 : CFG.handRadius;
    const y = this.H - cursor.y;

    this.cursorRing.position.set(cursor.x, y, 0.8);
    this.cursorRing.scale.set(rad, rad, 1);
    this.cursorMat.color.setHex(color);
    this.cursorMat.opacity = cursor.active ? 0.35 : 0.12;
    this.cursorRing.visible = true;

    this.cursorDot.position.set(cursor.x, y, 0.8);
    this.cursorDotMat.color.setHex(color);
    this.cursorDotMat.opacity = cursor.active ? 0.5 : 0.15;
    this.cursorDot.visible = true;
  }

  // ── Helpers ──────────────────────────────────────────────────────────────

  private setBgColor(theme: Theme): void {
    if (!this.videoTexture) {
      const [r, g, b] = theme.bgRGB;
      this.bgMaterial.color.setRGB(r / 255, g / 255, b / 255);
    }
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.lutTexture.dispose();
    this.trailFBOs[0].dispose();
    this.trailFBOs[1].dispose();
    this.trailMaterial.dispose();
    this.trailDisplayMat.dispose();
    this.skeletonGeo.dispose();
    this.skeletonMat.dispose();
    this.outlineMat.dispose();
    this.renderer.dispose();
    this.videoTexture?.dispose();
  }
}
