"""
src/shapes/lattice.py
=====================
Pure-math module that generates target particle positions for geometric shapes.
All computation is NumPy-only (no Taichi).

Each shape is produced as an (n_points, 2) float32 array in simulation pixel space.
"""

import math
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Formula strings (Unicode maths, used by renderer overlay)
# ─────────────────────────────────────────────────────────────────────────────

FORMULA_STRINGS: dict = {
    "circle":    "x\u00b2 + y\u00b2 = r\u00b2",
    "ellipse":   "x\u00b2/a\u00b2 + y\u00b2/b\u00b2 = 1",
    "triangle":  "Pk = r(cos 2\u03c0k/3, sin 2\u03c0k/3)",
    "square":    "||x||\u221e \u2264 r",
    "pentagon":  "Pk = r(cos 2\u03c0k/5, sin 2\u03c0k/5)",
    "hexagon":   "Pk = r(cos \u03c0k/3,  sin \u03c0k/3)",
    "rose":      "r(\u03b8) = a\u00b7cos(n\u03b8)",
    "spiral":    "r = a\u00b7\u03b8  (Archimedean)",
    "lissajous": "x = A sin(3t+\u03c0/4),  y = B sin(2t)",
}


class ShapeLattice:
    """
    Generate target particle positions for a named geometric shape.

    Parameters
    ----------
    shape_name : one of circle, ellipse, triangle, square, pentagon,
                 hexagon, rose, spiral, lissajous
    cx, cy     : centre of the shape in simulation pixel space
    radius     : bounding radius in pixels
    n_points   : number of particles to place
    **kwargs   : shape-specific params (axis_ratio, n_petals, etc.)
    """

    _GENERATORS = {}   # filled by _register below

    def __init__(
        self,
        shape_name: str,
        cx: float,
        cy: float,
        radius: float,
        n_points: int,
        **kwargs,
    ):
        self.shape_name = shape_name
        self.cx = float(cx)
        self.cy = float(cy)
        self.radius = float(radius)
        self.n_points = int(n_points)
        self.kwargs = kwargs

    def generate(self) -> np.ndarray:
        """Return (n_points, 2) float32 positions in sim-pixel space."""
        gen = self._GENERATORS.get(self.shape_name)
        if gen is None:
            raise ValueError(f"Unknown shape: {self.shape_name}")
        pts = gen(self)
        return pts.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Shape generators  (each returns exactly self.n_points rows)
    # ─────────────────────────────────────────────────────────────────────────

    def _circle(self) -> np.ndarray:
        """Sunflower seed packing (golden-angle spiral) fills a disk evenly."""
        n = self.n_points
        indices = np.arange(n, dtype=np.float64)
        golden_angle = 2.399963  # radians ≈ 137.508°
        theta = indices * golden_angle
        r = self.radius * np.sqrt(indices / n)
        pts = np.column_stack([
            r * np.cos(theta) + self.cx,
            r * np.sin(theta) + self.cy,
        ])
        return pts

    def _ellipse(self) -> np.ndarray:
        """Points along ellipse contour + fill via scaled sunflower."""
        axis_ratio = self.kwargs.get("axis_ratio", 0.55)
        a = self.radius
        b = self.radius * axis_ratio
        n = self.n_points
        indices = np.arange(n, dtype=np.float64)
        golden_angle = 2.399963
        theta = indices * golden_angle
        r_frac = np.sqrt(indices / n)
        pts = np.column_stack([
            a * r_frac * np.cos(theta) + self.cx,
            b * r_frac * np.sin(theta) + self.cy,
        ])
        return pts

    def _triangle(self) -> np.ndarray:
        """Equilateral triangle filled via barycentric sampling on a grid."""
        n = self.n_points
        r = self.radius
        # Vertices (point-up)
        verts = np.array([
            [r * math.cos(2 * math.pi * k / 3 - math.pi / 2),
             r * math.sin(2 * math.pi * k / 3 - math.pi / 2)]
            for k in range(3)
        ])
        # Generate more candidates than needed on a barycentric grid, then trim
        side = int(math.ceil(math.sqrt(n * 2.5))) + 1
        pts = []
        for i in range(side + 1):
            for j in range(side + 1 - i):
                k = side - i - j
                u, v, w = i / side, j / side, k / side
                pt = u * verts[0] + v * verts[1] + w * verts[2]
                pts.append(pt)
        pts = np.array(pts)
        # Sub-sample or pad to exactly n_points
        pts = _resample(pts, n)
        pts[:, 0] += self.cx
        pts[:, 1] += self.cy
        return pts

    def _square(self) -> np.ndarray:
        """Uniform grid inside [-radius, +radius]^2."""
        n = self.n_points
        side = max(2, int(math.floor(math.sqrt(n))))
        coords = np.linspace(-self.radius, self.radius, side)
        gx, gy = np.meshgrid(coords, coords)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        pts = _resample(pts, n)
        pts[:, 0] += self.cx
        pts[:, 1] += self.cy
        return pts

    def _ngon(self, sides: int) -> np.ndarray:
        """Regular n-gon filled via scanline-style interior grid."""
        n = self.n_points
        r = self.radius
        # Vertex angles
        angles = np.array([2 * math.pi * k / sides for k in range(sides)])
        vx = r * np.cos(angles)
        vy = r * np.sin(angles)
        # Generate candidate grid, reject points outside polygon
        grid_n = int(math.ceil(math.sqrt(n * 2.0))) + 2
        coords = np.linspace(-r, r, grid_n)
        gx, gy = np.meshgrid(coords, coords)
        candidates = np.column_stack([gx.ravel(), gy.ravel()])
        # Point-in-polygon via cross-product winding
        inside = _points_in_polygon(candidates, vx, vy)
        pts = candidates[inside]
        pts = _resample(pts, n)
        pts[:, 0] += self.cx
        pts[:, 1] += self.cy
        return pts

    def _pentagon(self) -> np.ndarray:
        return self._ngon(5)

    def _hexagon(self) -> np.ndarray:
        return self._ngon(6)

    def _rose(self) -> np.ndarray:
        """Rose curve r(theta) = radius * |cos(n_petals * theta)|."""
        n_petals = self.kwargs.get("n_petals", 3)
        n_petals = max(1, int(n_petals))
        n = self.n_points
        # Over-sample, then sub-sample to avoid clumping at petal bases
        oversample = n * 6
        theta = np.linspace(0, 2 * math.pi, oversample, endpoint=False)
        r = self.radius * np.abs(np.cos(n_petals * theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        candidates = np.column_stack([x, y])
        # Rejection: remove points too close to origin (thin petal bases)
        dist = np.sqrt(x**2 + y**2)
        keep = dist > self.radius * 0.05
        candidates = candidates[keep]
        pts = _resample(candidates, n)
        pts[:, 0] += self.cx
        pts[:, 1] += self.cy
        return pts

    def _spiral(self) -> np.ndarray:
        """Archimedean spiral r = a*theta with uniform arc-length spacing."""
        turns = 3
        n = self.n_points
        # Over-sample for arc-length re-parameterisation
        oversample = n * 20
        theta = np.linspace(0, 2 * math.pi * turns, oversample)
        a = self.radius / (2 * math.pi * turns)
        r = a * theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        # Compute cumulative arc length
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        total_len = s[-1]
        # Re-sample at equal arc-length intervals
        target_s = np.linspace(0, total_len, n)
        xi = np.interp(target_s, s, x)
        yi = np.interp(target_s, s, y)
        pts = np.column_stack([xi + self.cx, yi + self.cy])
        return pts

    def _lissajous(self) -> np.ndarray:
        """Lissajous figure x = A sin(3t + pi/4), y = B sin(2t)."""
        n = self.n_points
        A = B = self.radius * 0.85
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        x = A * np.sin(3 * t + math.pi / 4) + self.cx
        y = B * np.sin(2 * t) + self.cy
        return np.column_stack([x, y])


# Register generators
ShapeLattice._GENERATORS = {
    "circle":    ShapeLattice._circle,
    "ellipse":   ShapeLattice._ellipse,
    "triangle":  ShapeLattice._triangle,
    "square":    ShapeLattice._square,
    "pentagon":  ShapeLattice._pentagon,
    "hexagon":   ShapeLattice._hexagon,
    "rose":      ShapeLattice._rose,
    "spiral":    ShapeLattice._spiral,
    "lissajous": ShapeLattice._lissajous,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample points to exactly n: sub-sample if too many, duplicate if too few."""
    m = len(pts)
    if m == 0:
        return np.zeros((n, 2), dtype=np.float64)
    if m == n:
        return pts
    if m > n:
        indices = np.linspace(0, m - 1, n, dtype=int)
        return pts[indices]
    # Pad by repeating with small jitter
    rng = np.random.default_rng(42)
    extra = n - m
    idx = rng.choice(m, size=extra, replace=True)
    jitter = rng.normal(scale=0.5, size=(extra, 2))
    return np.vstack([pts, pts[idx] + jitter])


def _points_in_polygon(
    points: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> np.ndarray:
    """Ray-casting point-in-polygon test. Returns boolean mask."""
    n_verts = len(vx)
    n_pts = len(points)
    inside = np.zeros(n_pts, dtype=bool)
    px, py = points[:, 0], points[:, 1]
    j = n_verts - 1
    for i in range(n_verts):
        cond = (vy[i] > py) != (vy[j] > py)
        slope = (vx[j] - vx[i]) * (py - vy[i]) / (vy[j] - vy[i] + 1e-12) + vx[i]
        cross = cond & (px < slope)
        inside ^= cross
        j = i
    return inside
