#!/usr/bin/env python3
"""Stylized OCT B-scan cross-section through the fovea.

Takes the layered cap meshes (the base ``retinal_cap.obj`` = ILM plus the
``layer_*.obj`` offset surfaces from ``src.geometry.build_layered_mesh``), slices
them along the **y = 0 meridian through the fovea**, and renders the stacked
layer bands as a stylized, side-view orthographic OCT B-scan: dark background,
each layer a coloured band (inner -> outer retina, top -> bottom), coloured by
its ETDRS value along the nasal-temporal meridian using the same per-layer
colormaps as the 3-D layered render, with layer labels in the right margin.

This is a standalone renderer (numpy + matplotlib + trimesh) -- it does *not*
need Blender; the slice is computed analytically and rasterised, which gives
exact pixel dimensions and clean labels.

    python blender/render_cross_section.py \
        --csv blender/layered_values.csv --output blender/cross_section.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent

SUBFIELDS = (
    "central",
    "inner_superior", "inner_nasal", "inner_inferior", "inner_temporal",
    "outer_superior", "outer_nasal", "outer_inferior", "outer_temporal",
)
# canonical-index of the subfield seen on the y=0 meridian, by ring & side
CENTRAL, INNER_NAS, INNER_TEM, OUTER_NAS, OUTER_TEM = 0, 2, 4, 6, 8
LAYER_CMAPS = ["Blues", "Greens", "Oranges", "Reds", "Purples", "YlOrBr", "PuBuGn", "RdPu"]


def read_table(csv_path):
    """Ordered dict {layer: [9 subfield values, canonical order]}."""
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    header = [h.strip().lower() for h in rows[0][1:]]
    named = all(name in header for name in SUBFIELDS)
    table = {}
    for r in rows[1:]:
        vals = [float(x) for x in r[1:]]
        table[r[0].strip()] = ([vals[header.index(n)] for n in SUBFIELDS]
                               if named else vals[:9])
    return table


def load_mesh(path):
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def fovea_slice_indices(vertices, eps=1e-6):
    """Indices of the y~0 meridian vertices, sorted by x."""
    sl = np.where(np.abs(vertices[:, 1]) < eps)[0]
    return sl[np.argsort(vertices[sl, 0])]


def value_index_along_x(x, laterality):
    """Which canonical subfield a point at (x, 0) falls in (nasal-temporal)."""
    r = abs(x)
    if r < 0.5:
        return CENTRAL
    nasal = (x >= 0) if laterality.upper() == "OD" else (x < 0)
    if r < 1.5:
        return INNER_NAS if nasal else INNER_TEM
    return OUTER_NAS if nasal else OUTER_TEM


def render(csv_path, mesh_dir, output, laterality="OD",
           width=1920, height=600, raster_w=1600, raster_h=560):
    table = read_table(csv_path)
    layer_names = list(table)

    base = load_mesh(mesh_dir / "retinal_cap.obj")               # ILM
    bv = np.asarray(base.vertices)
    idx = fovea_slice_indices(bv)
    xs = bv[idx, 0]

    # boundary z-profiles along the slice: [ILM, layer1, ..., layerN]
    boundaries = [bv[idx, 2]]
    used_names = []
    for name in layer_names:
        p = mesh_dir / f"layer_{name.replace('/', '_')}.obj"
        if not p.exists():
            print(f"skip {name!r}: missing {p.name}")
            continue
        lv = np.asarray(load_mesh(p).vertices)
        if len(lv) != len(bv):
            print(f"skip {name!r}: vertex count {len(lv)} != base {len(bv)}")
            continue
        boundaries.append(lv[idx, 2])
        used_names.append(name)
    if not used_names:
        raise RuntimeError("No layer surfaces matched the base mesh.")

    # interpolate boundaries onto a regular x grid
    xg = np.linspace(xs.min(), xs.max(), raster_w)
    zb = np.array([np.interp(xg, xs, b) for b in boundaries])     # (L+1, raster_w)

    lo, hi = zb.min(), zb.max()
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    # per-layer colormap + normalisation (matches the 3-D layered render)
    cmaps = [plt.get_cmap(LAYER_CMAPS[i % len(LAYER_CMAPS)]) for i in range(len(used_names))]
    norms = [matplotlib.colors.Normalize(min(table[n]), max(table[n])) for n in used_names]

    # rasterise the bands (row 0 = top = inner/ILM; depth increases downward)
    raster = np.full((raster_h, raster_w, 3), 0.045)

    def z2row(z):
        return np.clip(((z - lo) / (hi - lo) * (raster_h - 1)).astype(int), 0, raster_h - 1)

    for c in range(raster_w):
        rows = z2row(zb[:, c])
        vidx = value_index_along_x(xg[c], laterality)
        for i in range(len(used_names)):
            t = 0.25 + 0.70 * norms[i](table[used_names[i]][vidx])   # vivid mid-range
            raster[rows[i]:rows[i + 1], c] = cmaps[i](float(np.clip(t, 0, 1)))[:3]
    # thin boundary lines
    for b in zb:
        raster[z2row(b), np.arange(raster_w)] *= 0.45

    # --- figure (exact width x height px) -------------------------------------
    bg = "#0a0a0a"
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=bg)
    ax = fig.add_axes([0.045, 0.15, 0.80, 0.78])
    ax.set_facecolor(bg)
    ax.imshow(raster, extent=[xg.min(), xg.max(), hi, lo], origin="upper", aspect="auto")
    ax.set_ylim(hi, lo)               # depth increases downward
    ax.set_yticks([])
    ax.axvline(0, color="white", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("position through fovea (mm)      temporal  ←  •  →  nasal",
                  color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=10)
    for s in ax.spines.values():
        s.set_color("0.4")
    ax.set_title("Stylized OCT B-scan — cross-section through the fovea (nasal–temporal)",
                 color="white", fontsize=13, pad=8)

    # right-margin labels, de-overlapped, with leader lines
    xmax = xg.max()
    label_x = xmax + 0.07 * (xmax - xg.min())
    centers = [(0.5 * (zb[i, -1] + zb[i + 1, -1]), used_names[i], cmaps[i](0.82))
               for i in range(len(used_names))]
    centers.sort(key=lambda t: t[0])
    gap = (hi - lo) * 0.052
    y_prev = lo
    for zc, name, col in centers:
        y = max(zc, y_prev + gap)
        y_prev = y
        ax.annotate(name, xy=(xmax, zc), xytext=(label_x, y), textcoords="data",
                    va="center", ha="left", color=col, fontsize=11, fontweight="bold",
                    annotation_clip=False, clip_on=False,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7, shrinkA=0, shrinkB=2))

    fig.savefig(output, dpi=100, facecolor=bg)
    plt.close(fig)
    print(f"Rendered {len(used_names)} layers ({', '.join(used_names)})")
    print(f"Saved -> {output}  ({width}x{height})")


def main():
    p = argparse.ArgumentParser(description="Stylized OCT B-scan cross-section of the layered cap.")
    p.add_argument("--csv", default=str(SCRIPT_DIR / "layered_values.csv"),
                   help="ETDRS values CSV (rows=layers, cols=9 subfields)")
    p.add_argument("--mesh-dir", default=str(SCRIPT_DIR),
                   help="directory holding retinal_cap.obj and layer_*.obj")
    p.add_argument("--output", default=str(SCRIPT_DIR / "cross_section.png"))
    p.add_argument("--laterality", default="OD", choices=["OD", "OS"])
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=600)
    args = p.parse_args()
    render(Path(args.csv), Path(args.mesh_dir), args.output,
           laterality=args.laterality, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
