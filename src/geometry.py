"""Retinal cap geometry: a spherical-cap mesh of the macular area.

The posterior eye is modelled as a sphere of radius ``radius_mm`` (the globe
radius, ~12 mm). The macula is a small cap of that sphere centred on the fovea.
We build the cap as a polar ``(r, theta)`` mesh in *retinal coordinates*, where
``r`` is the geodesic (arc-length) distance from the fovea along the curved
retina and ``theta`` is the angle around it.

Geometry
--------
The fovea is the cap pole, placed at the **origin**; the globe centre sits at
``(0, 0, radius_mm)`` so the cap rises in +z. A point at geodesic distance ``r``
subtends sphere polar angle ``phi = r / R`` from the pole, giving::

    rho = R * sin(phi)          # distance off the z-axis
    z   = R * (1 - cos(phi))    # height above the fovea
    x, y = rho * cos(theta), rho * sin(theta)

so every vertex lies exactly on the sphere of radius ``R`` centred at
``(0, 0, R)``. The cap spans ``cap_diameter_mm`` of geodesic diameter, i.e.
``r`` runs from 0 to ``cap_diameter_mm / 2``.

Topology
--------
Vertex 0 is the fovea (``r = 0``) at the origin. It is followed by ``n_radial``
rings (``r_i = r_max * i / n_radial`` for ``i = 1..n_radial``), each holding
``n_angular`` vertices. Total vertices = ``1 + n_radial * n_angular``. The fovea
fans to the first ring; consecutive rings are joined by quad strips (two
triangles each). The result is edge-manifold with a single boundary (the rim).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "blender" / "retinal_cap.obj"


def build_retinal_cap_mesh(
    radius_mm: float = 12.0,
    cap_diameter_mm: float = 6.0,
    n_radial: int = 200,
    n_angular: int = 200,
) -> trimesh.Trimesh:
    """Build a spherical-cap mesh of the macular retina centred at the fovea.

    Parameters
    ----------
    radius_mm : globe (sphere) radius the cap is sampled from.
    cap_diameter_mm : geodesic diameter of the cap (retinal coordinate r runs
        0..cap_diameter_mm/2).
    n_radial : number of concentric rings (excluding the central fovea vertex).
    n_angular : number of vertices per ring.

    Returns
    -------
    trimesh.Trimesh with ``1 + n_radial * n_angular`` vertices; vertex 0 is the
    fovea at the origin. Edge-manifold with one boundary loop (the rim).
    """
    if n_radial < 1 or n_angular < 3:
        raise ValueError("Need n_radial >= 1 and n_angular >= 3.")
    R = float(radius_mm)
    r_max = 0.5 * float(cap_diameter_mm)
    if not (0 < r_max <= np.pi * R):
        raise ValueError("cap_diameter_mm must be positive and fit on the sphere.")

    # --- retinal polar grid -> vertex positions on the sphere -----------------
    r = np.linspace(0.0, r_max, n_radial + 1)[1:]      # ring radii (drop r=0)
    theta = np.linspace(0.0, 2 * np.pi, n_angular, endpoint=False)
    phi = r / R                                         # sphere polar angle/ring

    rho = R * np.sin(phi)                               # (n_radial,)
    z = R * (1.0 - np.cos(phi))                         # (n_radial,)
    X = rho[:, None] * np.cos(theta)[None, :]           # (n_radial, n_angular)
    Y = rho[:, None] * np.sin(theta)[None, :]
    Z = np.broadcast_to(z[:, None], X.shape)
    ring_v = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    vertices = np.vstack([[0.0, 0.0, 0.0], ring_v])     # vertex 0 = fovea

    # --- faces (consistent CCW winding in (theta, r) param space) -------------
    j = np.arange(n_angular)
    jn = (j + 1) % n_angular

    def ring_idx(i, jj):  # ring i in 1..n_radial -> global vertex indices
        return 1 + (i - 1) * n_angular + jj

    # fovea fan to the first ring (wound to match the strip orientation below:
    # the shared (ring1_j, ring1_jn) edge must be traversed oppositely by the
    # fan and the first strip for consistent winding)
    fan = np.stack([np.zeros(n_angular, np.int64), ring_idx(1, jn), ring_idx(1, j)], axis=1)

    # quad strips between consecutive rings
    strips = []
    for i in range(1, n_radial):
        a, b = ring_idx(i, j), ring_idx(i, jn)
        c, d = ring_idx(i + 1, j), ring_idx(i + 1, jn)
        strips.append(np.stack([a, b, d], axis=1))
        strips.append(np.stack([a, d, c], axis=1))
    faces = np.vstack([fan, *strips]) if strips else fan

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# Subfield index -> name. 0 = fovea; 1-4 inner ring; 5-8 outer ring; each ring
# ordered superior, nasal, inferior, temporal.
ETDRS_SUBFIELD_NAMES = (
    "central",
    "inner_superior", "inner_nasal", "inner_inferior", "inner_temporal",
    "outer_superior", "outer_nasal", "outer_inferior", "outer_temporal",
)


def assign_etdrs_subfield(
    vertices: np.ndarray,
    fovea_xy: tuple[float, float] = (0.0, 0.0),
    ring_radii_mm: tuple[float, float, float] = (0.5, 1.5, 3.0),
    *,
    laterality: str = "OD",
) -> np.ndarray:
    """Label each vertex with its ETDRS subfield index (0-8).

    The subfield is decided from each vertex's *en-face* (x, y) position relative
    to the fovea: the in-plane distance picks the ring and the angle picks the
    quadrant. (z / cap depth is ignored, matching how the clinical grid is
    projected onto the retina.)

    Parameters
    ----------
    vertices : (N, >=2) array of vertex coordinates in mm; columns 0,1 are x,y.
    fovea_xy : (x, y) of the foveal centre in the same units.
    ring_radii_mm : (central, inner, outer) ring *radii* -- the central circle,
        and the outer radii of the inner and outer rings (ETDRS default
        0.5/1.5/3.0 mm, i.e. 1/3/6 mm diameters).
    laterality : 'OD' (nasal = +x) or 'OS' (nasal = -x). For 'OS' the x axis is
        mirrored internally, which swaps the nasal/temporal labels and leaves
        central and superior/inferior unchanged. Matches ``laterality`` in
        ``src.etdrs.compute_etdrs_subfields``.

    Returns
    -------
    (N,) int array. 0 = central foveal subfield; 1-4 = inner ring
    (superior, nasal, inferior, temporal); 5-8 = outer ring (same order).
    Vertices beyond the outer radius are labelled -1 (outside the grid).

    Notes
    -----
    Quadrants split on the 45 deg diagonals with superior = +y, inferior = -y.
    Horizontal sides follow ``laterality``: nasal = +x for OD, nasal = -x for OS.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] < 2:
        raise ValueError("vertices must be a (N, >=2) array.")
    lat = laterality.upper()
    if lat not in ("OD", "OS"):
        raise ValueError("laterality must be 'OD' or 'OS'.")
    r_c, r_in, r_out = (float(x) for x in ring_radii_mm)

    dx = v[:, 0] - float(fovea_xy[0])
    dy = v[:, 1] - float(fovea_xy[1])
    if lat == "OS":           # mirror x: nasal/temporal swap, sup/inf unchanged
        dx = -dx
    rho = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))   # 0 = +x (nasal), 90 = +y (superior)

    # quadrant offset within a ring: superior=0, nasal=1, inferior=2, temporal=3
    quad = np.zeros(v.shape[0], dtype=int)
    quad[(ang >= 45) & (ang < 135)] = 0    # superior (+y)
    quad[(ang >= -45) & (ang < 45)] = 1    # nasal    (+x)
    quad[(ang >= -135) & (ang < -45)] = 2  # inferior (-y)
    quad[(ang >= 135) | (ang < -135)] = 3  # temporal (-x)

    out = np.full(v.shape[0], -1, dtype=int)
    out[rho < r_c] = 0
    inner = (rho >= r_c) & (rho < r_in)
    out[inner] = 1 + quad[inner]
    outer = (rho >= r_in) & (rho <= r_out)
    out[outer] = 5 + quad[outer]
    return out


def _print_stats(mesh: trimesh.Trimesh) -> None:
    """Print a compact summary of a cap mesh."""
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    z = mesh.vertices[:, 2]
    print("=== retinal cap mesh statistics ===")
    print(f"vertices            : {len(mesh.vertices)}")
    print(f"faces               : {len(mesh.faces)}")
    print(f"unique edges        : {len(counts)}")
    print(f"max faces per edge  : {counts.max()}  (<=2 => edge-manifold)")
    print(f"boundary edges (rim): {(counts == 1).sum()}")
    print(f"winding consistent  : {mesh.is_winding_consistent}")
    print(f"watertight          : {mesh.is_watertight}  (False: cap has a rim)")
    print(f"euler number        : {mesh.euler_number}  (1 => topological disk)")
    print(f"central vertex [0]  : {np.round(mesh.vertices[0], 6).tolist()}")
    print(f"surface area (mm^2) : {mesh.area:.4f}")
    ext = mesh.bounds[1] - mesh.bounds[0]
    print(f"bbox extents (mm)   : {np.round(ext, 4).tolist()}")
    print(f"cap sag / height(mm): {z.max() - z.min():.4f}")
    print(f"rim radius (mm)     : {np.hypot(mesh.vertices[1:, 0], mesh.vertices[1:, 1]).max():.4f}")


if __name__ == "__main__":
    mesh = build_retinal_cap_mesh()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(DEFAULT_OUTPUT)
    print(f"Wrote {DEFAULT_OUTPUT}")
    _print_stats(mesh)
