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
// Trail fade shader (fullscreen quad that blends previous FBO with fade)
// ─────────────────────────────────────────────────────────────────────────────

const TrailFadeShader = {
  uniforms: {
    tPrev: { value: null as THREE.Texture | null },
    uFade: { value: 0.94 },
    uBgColor: { value: new THREE.Color(0x060410) },
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
    uniform vec3 uBgColor;
    varying vec2 vUv;
    void main() {
      vec4 prev = texture2D(tPrev, vUv);
      gl_FragColor = vec4(mix(uBgColor, prev.rgb, uFade), 1.0);
    }
  `,
};

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
    float core = smoothstep(0.45, 0.0, dist);
    float glow = smoothstep(1.0, 0.0, dist);
    float intensity = core * 0.8 + glow * 0.2;
    vec3 color = texture2D(uLut, vec2(vTemp, 0.5)).rgb;
    gl_FragColor = vec4(color * intensity, glow);
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
    this.camera = new THREE.OrthographicCamera(0, w, 0, h, -10, 10);
    this.camera.position.z = 1;

    // ── Background quad ──────────────────────────────────────────────────
    this.bgMaterial = new THREE.MeshBasicMaterial({ color: 0x060410 });
    const bgGeo = new THREE.PlaneGeometry(w, h);
    this.bgMesh = new THREE.Mesh(bgGeo, this.bgMaterial);
    this.bgMesh.position.set(w / 2, h / 2, -5);
    this.scene.add(this.bgMesh);

    // ── LUT texture ──────────────────────────────────────────────────────
    const lutData = new Uint8Array(256 * 3);
    this.lutTexture = new THREE.DataTexture(lutData, 256, 1, THREE.RGBFormat);
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

    // ── Post-processing ──────────────────────────────────────────────────
    this.composer = new EffectComposer(this.renderer);
    const renderPass = new RenderPass(this.scene, this.camera);
    this.composer.addPass(renderPass);

    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(w, h),
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
      data[i * 3] = r;
      data[i * 3 + 1] = g;
      data[i * 3 + 2] = b;
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
    this.camera.updateProjectionMatrix();
    this.composer.setSize(w, h);
    this.bloomPass.resolution.set(w, h);
    this.trailFBOs[0].setSize(w, h);
    this.trailFBOs[1].setSize(w, h);
    this.bgMesh.geometry.dispose();
    this.bgMesh.geometry = new THREE.PlaneGeometry(w, h);
    this.bgMesh.position.set(w / 2, h / 2, -5);
  }

  reset(theme: Theme): void {
    this.uploadLut(theme);
    this.setBgColor(theme);
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

    // ── Trail pass (ping-pong) ───────────────────────────────────────────
    const prevFBO = this.trailFBOs[this.trailIdx];
    const nextFBO = this.trailFBOs[1 - this.trailIdx];

    // Fade previous trail
    this.trailMaterial.uniforms.tPrev.value = prevFBO.texture;
    this.trailMaterial.uniforms.uFade.value = 1 - CFG.trailFade;
    const [br, bg, bb] = theme.bgRGB;
    this.trailMaterial.uniforms.uBgColor.value.setRGB(br / 255, bg / 255, bb / 255);

    this.renderer.setRenderTarget(nextFBO);
    this.renderer.clear();
    this.renderer.render(this.trailScene, this.trailCamera);

    // Draw particles on top of faded trail
    this.bgMesh.visible = false;
    this.cursorRing.visible = false;
    this.cursorDot.visible = false;
    if (this.outlineLine) this.outlineLine.visible = false;
    this.renderer.render(this.scene, this.camera);
    this.bgMesh.visible = true;

    this.trailIdx = 1 - this.trailIdx;

    // Use trail as background texture for main scene
    this.bgMesh.visible = true;
    // We'll composite the trail in the main pass

    // ── Formation outline ────────────────────────────────────────────────
    this.updateOutline(overlay, theme);

    // ── Cursor ───────────────────────────────────────────────────────────
    this.updateCursor(cursor);

    // ── Main render via composer (bloom + vignette) ──────────────────────
    this.renderer.setRenderTarget(null);
    this.renderer.clear();

    // Draw background
    this.points.visible = false;
    if (this.outlineLine) this.outlineLine.visible = false;
    this.renderer.render(this.scene, this.camera);

    // Draw trail FBO as fullscreen quad (additive)
    this.trailMaterial.uniforms.tPrev.value = this.trailFBOs[this.trailIdx].texture;
    this.trailMaterial.uniforms.uFade.value = 1.0; // no fade, just display
    this.renderer.render(this.trailScene, this.trailCamera);

    // Draw particles + outline on top
    this.points.visible = true;
    if (this.outlineLine) this.outlineLine.visible = true;
    this.cursorRing.visible = cursor.x >= 0;
    this.cursorDot.visible = cursor.x >= 0;

    // Use composer for bloom + vignette
    this.composer.render();
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
    this.outlineMat.dispose();
    this.renderer.dispose();
    this.videoTexture?.dispose();
  }
}
