"""ETDRS macular grid: 9-subfield averaging and a bullseye renderer.

The Early Treatment Diabetic Retinopathy Study (ETDRS) grid is three concentric
circles (default diameters 1, 3, 6 mm) centred on the fovea. The central circle
is one subfield; the inner and outer rings are each split into four quadrants by
the 45 deg diagonals, giving nine subfields total:

    central,
    inner_superior, inner_nasal, inner_inferior, inner_temporal,
    outer_superior, outer_nasal, outer_inferior, outer_temporal

This module locates the fovea (``find_fovea``), averages an en-face thickness
map over those subfields (``compute_etdrs_subfields``) and renders a bullseye
(``plot_etdrs_bullseye``). The bullseye geometry follows matplotlib's "left
ventricle bullseye" gallery example (polar ``pcolormesh`` wedges + segment
outlines), re-laid-out for the 3-ring ETDRS grid instead of the AHA 17-segment
heart diagram.

Conventions
-----------
* ``thickness_map`` is a 2D array indexed ``[row, col]``; rows run top->bottom
  (smaller row = superior), cols run left->right.
* ``fovea_center`` is ``(row, col)`` in pixel coordinates of that map.
* ``pixel_size_mm`` is the *en-face* pixel pitch used to size the rings: a scalar
  (isotropic) or ``(row_mm, col_mm)`` for anisotropic volumes (e.g. Duke, where
  B-scan spacing != A-scan spacing). It does **not** rescale the thickness values
  themselves -- pass a map already in your preferred unit (e.g. microns).
* ``laterality`` ('OD'/'OS') only decides which horizontal quadrant is called
  nasal vs temporal; superior/inferior are unaffected.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

# Display angle (degrees, 0 deg = +x / east, CCW) of each quadrant's centre in
# the bullseye. Superior=top, inferior=bottom, nasal=right, temporal=left.
_QUADRANT_ANGLE_DEG = {"superior": 90.0, "nasal": 0.0, "inferior": 270.0, "temporal": 180.0}
SUBFIELD_ORDER = (
    "central",
    "inner_superior", "inner_nasal", "inner_inferior", "inner_temporal",
    "outer_superior", "outer_nasal", "outer_inferior", "outer_temporal",
)


def _as_pitch(pixel_size_mm) -> tuple[float, float]:
    """Normalise pixel_size_mm to (row_mm, col_mm)."""
    arr = np.atleast_1d(np.asarray(pixel_size_mm, dtype=float))
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    if arr.size == 2:
        return float(arr[0]), float(arr[1])
    raise ValueError("pixel_size_mm must be a scalar or a (row_mm, col_mm) pair.")


# --- fovea detection ----------------------------------------------------------
def _nan_gaussian(z: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaNs (normalised convolution)."""
    mask = np.isfinite(z).astype(float)
    filled = np.where(mask > 0, z, 0.0).astype(float)
    num = gaussian_filter(filled, sigma=sigma, mode="nearest")
    den = gaussian_filter(mask, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def _parabolic_residual(surface: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """surface minus its best-fit 2D parabola z = a + b r + c x + d r^2 + e x^2 + f r x.

    Returns the residual over the whole array (NaN where invalid). Positive
    residual means the surface lies *below* (deeper than) the smooth parabolic
    trend -- i.e. inside the foveal pit, since row/depth increases downward.
    Coordinates are mean-centred for numerical conditioning.
    """
    rows, cols = np.indices(surface.shape)
    rc = rows[mask].astype(float)
    cc = cols[mask].astype(float)
    z = surface[mask].astype(float)
    r_mu, c_mu = rc.mean(), cc.mean()
    rr, xx = rc - r_mu, cc - c_mu
    A = np.column_stack([np.ones_like(rr), rr, xx, rr * rr, xx * xx, rr * xx])
    beta, *_ = np.linalg.lstsq(A, z, rcond=None)
    R = rows.astype(float) - r_mu
    X = cols.astype(float) - c_mu
    fit = (beta[0] + beta[1] * R + beta[2] * X
           + beta[3] * R * R + beta[4] * X * X + beta[5] * R * X)
    resid = surface - fit
    resid[~mask] = np.nan
    return resid


def find_fovea(
    ilm_surface: np.ndarray,
    method: str = "ilm_curvature",
    *,
    thickness_map: np.ndarray | None = None,
    sigma: float = 5.0,
    window: int = 100,
    central_fraction: float = 0.5,
) -> tuple[int, int]:
    """Locate the foveal centre and return its ``(row, col)`` pixel position.

    Parameters
    ----------
    ilm_surface : 2D array of the ILM axial row per (B-scan, A-scan). Larger
        values are deeper, so the foveal pit appears as a local *maximum*.
    method : 'ilm_curvature' (default) or 'thinnest_point'.
    thickness_map : required for 'thinnest_point'; ignored otherwise.
    sigma : Gaussian smoothing width in pixels.
    window : side length (pixels) of the local parabolic-fit window
        ('ilm_curvature'); clipped to the array.
    central_fraction : central fraction of the field searched by
        'thinnest_point' (guards against peripheral artifacts).

    Methods
    -------
    ilm_curvature  : Smooth the ILM, then measure the depth of the foveal pit as
        the residual of the ILM below a 2D parabolic fit of the local macular
        curvature, and take the deepest point. Because it keys on the ILM's
        geometric pit -- which survives even when the retina is thickened or
        distorted -- it is **robust to AMD, DME and most pathology**. Default.
    thinnest_point : Smooth a retinal-thickness map and take the thinnest point
        near the field centre. **Faster, but only safe for healthy controls** --
        edema, drusen or fluid can make the thinnest point fall outside the
        fovea. Provided as a fallback; requires ``thickness_map``.

    Returns
    -------
    (row, col) integer pixel coordinates of the foveal centre.
    """
    if method == "ilm_curvature":
        ilm = np.asarray(ilm_surface, dtype=float)
        smoothed = _nan_gaussian(ilm, sigma)
        valid = np.isfinite(smoothed)
        if not valid.any():
            raise ValueError("ilm_surface has no finite values.")

        # Stage 1: parabolic residual over the central region gives a coarse pit
        # location. Restricting to the centre keeps the broad parabola from
        # mis-fitting the field edges (a macular scan is fovea-centred, so the
        # fovea is never at the extreme periphery) and places the local window.
        H, W = smoothed.shape
        central = np.zeros_like(valid)
        rr0, rr1 = int(H * (1 - central_fraction) / 2), int(H * (1 + central_fraction) / 2)
        cc0, cc1 = int(W * (1 - central_fraction) / 2), int(W * (1 + central_fraction) / 2)
        central[rr0:rr1, cc0:cc1] = True
        coarse_resid = _parabolic_residual(smoothed, valid & central)
        cr, cc = np.unravel_index(np.nanargmax(coarse_resid), coarse_resid.shape)

        # Stage 2: refit the parabola over a `window`x`window` patch around the
        # coarse estimate (local curvature -> cleaner pit depth) and take the max.
        half = window // 2
        r0, r1 = max(0, cr - half), min(H, cr + half)
        c0, c1 = max(0, cc - half), min(W, cc + half)
        patch = smoothed[r0:r1, c0:c1]
        pmask = np.isfinite(patch)
        resid = _parabolic_residual(patch, pmask)
        pr, pc = np.unravel_index(np.nanargmax(resid), resid.shape)
        return int(pr + r0), int(pc + c0)

    if method == "thinnest_point":
        if thickness_map is None:
            raise ValueError("method='thinnest_point' requires thickness_map=...")
        t = _nan_gaussian(np.asarray(thickness_map, dtype=float), sigma)
        H, W = t.shape
        r0, r1 = int(H * (1 - central_fraction) / 2), int(H * (1 + central_fraction) / 2)
        c0, c1 = int(W * (1 - central_fraction) / 2), int(W * (1 + central_fraction) / 2)
        sub = t[r0:r1, c0:c1]
        if not np.isfinite(sub).any():
            raise ValueError("thickness_map has no finite values in the central region.")
        fr, fc = np.unravel_index(np.nanargmin(sub), sub.shape)
        return int(fr + r0), int(fc + c0)

    raise ValueError(f"Unknown method {method!r}; use 'ilm_curvature' or 'thinnest_point'.")


def compute_etdrs_subfields(
    thickness_map: np.ndarray,
    fovea_center: Sequence[float],
    ring_diameters_mm: Sequence[float] = (1.0, 3.0, 6.0),
    *,
    pixel_size_mm,
    laterality: str = "OD",
) -> dict[str, float]:
    """Mean thickness within each of the 9 ETDRS subfields.

    Parameters
    ----------
    thickness_map : 2D array of per-pixel thickness (NaNs are ignored in means).
    fovea_center  : (row, col) pixel location of the foveal centre.
    ring_diameters_mm : (central, inner, outer) circle diameters in mm.
    pixel_size_mm : en-face pixel pitch, scalar or (row_mm, col_mm).
    laterality    : 'OD' (nasal = right) or 'OS' (nasal = left).

    Returns
    -------
    dict of the 9 subfield names -> mean thickness (np.nan if a subfield is empty).
    """
    tmap = np.asarray(thickness_map, dtype=float)
    if tmap.ndim != 2:
        raise ValueError("thickness_map must be 2D.")
    if len(ring_diameters_mm) != 3:
        raise ValueError("ring_diameters_mm must have 3 values (central, inner, outer).")

    row_mm, col_mm = _as_pitch(pixel_size_mm)
    r0, c0 = float(fovea_center[0]), float(fovea_center[1])
    radii_mm = np.asarray(ring_diameters_mm, dtype=float) / 2.0  # -> (0.5, 1.5, 3.0)

    rows, cols = np.indices(tmap.shape)
    dy = (rows - r0) * row_mm          # mm, downward positive
    dx = (cols - c0) * col_mm          # mm, rightward positive
    radius = np.hypot(dx, dy)
    # angle with superior (up) = +90 deg, east = 0, west = +/-180, down = -90
    ang = np.degrees(np.arctan2(-dy, dx))

    is_superior = (ang >= 45) & (ang < 135)
    is_inferior = (ang >= -135) & (ang < -45)
    is_right = (ang >= -45) & (ang < 45)
    is_left = (ang >= 135) | (ang < -135)
    if laterality.upper() == "OD":
        is_nasal, is_temporal = is_right, is_left
    elif laterality.upper() == "OS":
        is_nasal, is_temporal = is_left, is_right
    else:
        raise ValueError("laterality must be 'OD' or 'OS'.")
    quad = {"superior": is_superior, "inferior": is_inferior,
            "nasal": is_nasal, "temporal": is_temporal}

    in_central = radius < radii_mm[0]
    in_inner = (radius >= radii_mm[0]) & (radius < radii_mm[1])
    in_outer = (radius >= radii_mm[1]) & (radius < radii_mm[2])

    def _mean(mask: np.ndarray) -> float:
        vals = tmap[mask]
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    out: dict[str, float] = {"central": _mean(in_central)}
    for ring_name, ring_mask in (("inner", in_inner), ("outer", in_outer)):
        for qname, qmask in quad.items():
            out[f"{ring_name}_{qname}"] = _mean(ring_mask & qmask)
    return out


def plot_etdrs_bullseye(
    values_dict: dict[str, float],
    layer_name: str,
    vmin: float,
    vmax: float,
    cmap: str = "viridis",
    output_path=None,
    ring_diameters_mm: Sequence[float] = (1.0, 3.0, 6.0),
    units: str = "µm",
):
    """Render the 9 ETDRS subfields as a bullseye, coloured by value.

    Modelled on matplotlib's left-ventricle bullseye example: a polar axes filled
    with ``pcolormesh`` wedges, segment outlines, per-segment value annotations
    and a colorbar. Saves to ``output_path`` if given; returns the Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    radii = np.asarray(ring_diameters_mm, dtype=float) / 2.0  # ring boundaries (mm)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), subplot_kw=dict(projection="polar"))

    def _fill(r_in: float, r_out: float, th0_deg: float, th1_deg: float, value: float):
        th = np.radians(np.linspace(th0_deg, th1_deg, 60))
        r = np.array([r_in, r_out])
        TH, R = np.meshgrid(th, r)
        if np.isfinite(value):
            C = np.full((1, th.size - 1), value)
            ax.pcolormesh(TH, R, C, cmap=cmap_obj, norm=norm, shading="flat")
        else:  # missing subfield -> hatched grey
            ax.pcolormesh(TH, R, np.full((1, th.size - 1), np.nan),
                          cmap=cmap_obj, norm=norm, shading="flat")
            ax.fill_between(th, r_in, r_out, color="0.8", hatch="//", edgecolor="0.6")

    def _annotate(r: float, th_deg: float, value: float):
        txt = "n/a" if not np.isfinite(value) else f"{value:.0f}"
        tcol = "black" if (np.isfinite(value) and norm(value) > 0.55) else "white"
        if not np.isfinite(value):
            tcol = "0.3"
        ax.text(np.radians(th_deg), r, txt, ha="center", va="center",
                fontsize=11, fontweight="bold", color=tcol)

    # central disc
    _fill(0.0, radii[0], 0.0, 360.0, values_dict["central"])
    _annotate(0.0, 0.0, values_dict["central"])

    # inner & outer rings: 4 quadrants each, split on the 45 deg diagonals
    quad_bounds = {"nasal": (-45, 45), "superior": (45, 135),
                   "temporal": (135, 225), "inferior": (225, 315)}
    for ring_name, (r_in, r_out) in (("inner", (radii[0], radii[1])),
                                     ("outer", (radii[1], radii[2]))):
        r_mid = 0.5 * (r_in + r_out)
        for qname, (t0, t1) in quad_bounds.items():
            value = values_dict[f"{ring_name}_{qname}"]
            _fill(r_in, r_out, t0, t1, value)
            _annotate(r_mid, _QUADRANT_ANGLE_DEG[qname], value)

    # segment outlines: ring circles + diagonal radial dividers (rings only)
    circ = np.radians(np.linspace(0, 360, 361))
    for r in radii:
        ax.plot(circ, np.full_like(circ, r), color="white", lw=1.5)
    for diag in (45, 135, 225, 315):
        ax.plot(np.radians([diag, diag]), [radii[0], radii[2]], color="white", lw=1.5)

    # direction labels just outside the grid
    for lbl, deg in (("S", 90), ("I", 270), ("N", 0), ("T", 180)):
        ax.text(np.radians(deg), radii[2] * 1.12, lbl, ha="center", va="center",
                fontsize=12, fontweight="bold", color="0.25")

    ax.set_ylim(0, radii[2] * 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title(f"ETDRS subfields — {layer_name}", pad=18, fontsize=13)

    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax,
                      fraction=0.046, pad=0.10)
    cb.set_label(f"thickness ({units})")

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        print(f"Wrote bullseye -> {output_path}")
    return fig
