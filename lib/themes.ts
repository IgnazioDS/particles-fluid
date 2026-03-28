/**
 * Color theme system — 8 palettes, each a 256-entry LUT
 * built by smooth interpolation between color stops.
 */

export interface Theme {
  name: string;
  bg: string;
  bgRGB: readonly [number, number, number];
  accent: string;
  lut: string[];
  rgb: ReadonlyArray<readonly [number, number, number]>;
}

type Stop = readonly [number, number, number, number]; // [t, r, g, b]

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp(v: number): number {
  return Math.round(Math.min(255, Math.max(0, v)));
}

function buildLut(stops: Stop[]): { lut: string[]; rgb: Array<readonly [number, number, number]> } {
  const lut: string[] = [];
  const rgb: Array<readonly [number, number, number]> = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let s0 = stops[0];
    let s1 = stops[stops.length - 1];
    for (let j = 0; j < stops.length - 1; j++) {
      if (t >= stops[j][0] && t <= stops[j + 1][0]) {
        s0 = stops[j];
        s1 = stops[j + 1];
        break;
      }
    }
    const f = (t - s0[0]) / (s1[0] - s0[0] + 1e-9);
    const r = clamp(lerp(s0[1], s1[1], f));
    const g = clamp(lerp(s0[2], s1[2], f));
    const b = clamp(lerp(s0[3], s1[3], f));
    lut.push(`rgb(${r},${g},${b})`);
    rgb.push([r, g, b] as const);
  }
  return { lut, rgb };
}

function make(
  name: string,
  bg: readonly [number, number, number],
  accent: string,
  stops: Stop[],
): Theme {
  const { lut, rgb } = buildLut(stops);
  return {
    name,
    bg: `rgb(${bg[0]},${bg[1]},${bg[2]})`,
    bgRGB: bg,
    accent,
    lut,
    rgb,
  };
}

export const THEMES: Theme[] = [
  make("Inferno", [6, 4, 10], "#ff8844", [
    [0, 10, 0, 8],
    [0.2, 150, 4, 42],
    [0.45, 240, 69, 0],
    [0.72, 250, 210, 10],
    [1, 255, 250, 120],
  ]),
  make("Ocean", [2, 6, 22], "#40c8e0", [
    [0, 2, 8, 35],
    [0.25, 10, 48, 130],
    [0.5, 24, 140, 200],
    [0.75, 64, 210, 220],
    [1, 200, 248, 255],
  ]),
  make("Aurora", [4, 8, 14], "#40e080", [
    [0, 4, 14, 18],
    [0.2, 10, 80, 44],
    [0.4, 30, 200, 80],
    [0.6, 40, 210, 230],
    [0.82, 180, 60, 220],
    [1, 255, 215, 255],
  ]),
  make("Neon", [10, 0, 18], "#ff20c0", [
    [0, 14, 0, 24],
    [0.25, 180, 12, 160],
    [0.5, 255, 34, 148],
    [0.75, 34, 220, 255],
    [1, 255, 255, 255],
  ]),
  make("Ember", [10, 2, 0], "#ff6020", [
    [0, 14, 2, 0],
    [0.3, 140, 18, 0],
    [0.55, 224, 44, 0],
    [0.8, 255, 160, 32],
    [1, 255, 242, 140],
  ]),
  make("Ice", [0, 4, 20], "#80b0ff", [
    [0, 0, 8, 34],
    [0.25, 16, 64, 170],
    [0.5, 50, 132, 232],
    [0.75, 132, 198, 255],
    [1, 242, 250, 255],
  ]),
  make("Toxic", [4, 8, 0], "#80ff20", [
    [0, 4, 12, 0],
    [0.25, 18, 72, 0],
    [0.5, 72, 200, 0],
    [0.75, 192, 255, 32],
    [1, 255, 255, 144],
  ]),
  make("Sunset", [16, 2, 16], "#ff6060", [
    [0, 18, 0, 18],
    [0.25, 104, 18, 100],
    [0.5, 214, 50, 92],
    [0.75, 255, 130, 50],
    [1, 255, 214, 124],
  ]),
];
