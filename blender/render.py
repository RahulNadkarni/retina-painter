"""Render the retinal cap coloured by an ETDRS subfield CSV, with a colorbar.

Run via Blender:

    blender --background --python render.py -- \
        --input values.csv --layer RNFL --output out.png --colormap viridis

Pipeline:
  1. Parse CLI args (after the ``--`` separator Blender uses for script args).
  2. Open ``retinal_template.blend`` (camera / lights / Cycles set up already).
  3. Read the CSV (rows = layer names, columns = the 9 ETDRS subfields).
  4. For the requested layer, label every vertex by ETDRS subfield, look up the
     subfield value, and bake it to a vertex-colour attribute via `--colormap`
     normalised over the layer's own (vmin, vmax). An emission material + the
     Standard view transform make the rendered colours match the colormap.
  5. Render with Cycles, then composite a matplotlib colorbar over the image.
  6. Save the final PNG to ``--output``.

matplotlib must be installed in Blender's bundled Python (numpy already is).
"""
import argparse
import csv
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

# matplotlib gets pip-installed into the *user* site, which Blender's embedded
# Python disables -- append it to sys.path so `import matplotlib` works here.
# (Appended, not prepended, so Blender's bundled numpy keeps priority.)
import site as _site
_user_sites = []
if hasattr(_site, "getusersitepackages"):
    _user_sites.append(_site.getusersitepackages())
_user_sites.append(str(Path.home() / f".local/lib/python3.{sys.version_info.minor}/site-packages"))
for _p in _user_sites:
    if _p and Path(_p).is_dir() and _p not in sys.path:
        sys.path.append(_p)

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE = SCRIPT_DIR / "retinal_template.blend"

# Canonical ETDRS subfield order (== src.geometry.ETDRS_SUBFIELD_NAMES).
SUBFIELDS = (
    "central",
    "inner_superior", "inner_nasal", "inner_inferior", "inner_temporal",
    "outer_superior", "outer_nasal", "outer_inferior", "outer_temporal",
)
COLOR_ATTR = "etdrs_value"


# --- CLI ----------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Render the retinal cap coloured by an ETDRS CSV.")
    p.add_argument("--input", required=True, help="CSV: rows=layers, cols=9 subfields")
    p.add_argument("--layer", required=True, help="layer (CSV row) to render")
    p.add_argument("--output", required=True, help="output PNG path")
    p.add_argument("--colormap", default="viridis", help="matplotlib colormap name")
    p.add_argument("--laterality", default="OD", choices=["OD", "OS"])
    return p.parse_args(argv)


# --- CSV -----------------------------------------------------------------------
def read_layer_values(csv_path, layer):
    """Return the layer's 9 subfield values in canonical order."""
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    header = [h.strip().lower() for h in rows[0][1:]]
    table = {r[0].strip(): [float(x) for x in r[1:]] for r in rows[1:]}
    if layer not in table:
        raise ValueError(f"Layer {layer!r} not in CSV. Available: {list(table)}")
    vals = table[layer]
    # If the header names the subfields, reorder to canonical; else assume order.
    if all(name in header for name in SUBFIELDS):
        return [vals[header.index(name)] for name in SUBFIELDS]
    if len(vals) < 9:
        raise ValueError(f"Layer {layer!r} has {len(vals)} columns; need 9 subfields.")
    return vals[:9]


# --- geometry: ETDRS subfield per vertex (matches src.geometry) ---------------
def vertex_subfields(x, y, laterality="OD", ring_radii=(0.5, 1.5, 3.0)):
    r_c, r_in, r_out = ring_radii
    dx = -x if laterality.upper() == "OS" else x   # OS mirrors x (nasal/temporal swap)
    dy = y
    rho = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx))            # 0=+x nasal(OD), 90=+y superior
    quad = np.zeros(x.shape, dtype=int)
    quad[(ang >= 45) & (ang < 135)] = 0   # superior
    quad[(ang >= -45) & (ang < 45)] = 1   # nasal
    quad[(ang >= -135) & (ang < -45)] = 2 # inferior
    quad[(ang >= 135) | (ang < -135)] = 3 # temporal
    sub = np.full(x.shape, -1, dtype=int)
    sub[rho < r_c] = 0
    inner = (rho >= r_c) & (rho < r_in); sub[inner] = 1 + quad[inner]
    outer = (rho >= r_in) & (rho <= r_out); sub[outer] = 5 + quad[outer]
    return sub


def srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# --- scene setup --------------------------------------------------------------
def get_cap_object():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh object in the template scene.")
    for o in meshes:
        if "retinal_cap" in o.name.lower():
            return o
    return meshes[0]


def bake_vertex_colors(obj, values, cmap_name, laterality):
    """Colour each vertex by its subfield value; returns (vmin, vmax)."""
    import matplotlib

    mesh = obj.data
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    # to world space (use x, y for the en-face subfield assignment)
    mw = np.array(obj.matrix_world)
    world = np.column_stack([co, np.ones(n)]) @ mw.T
    sub = vertex_subfields(world[:, 0], world[:, 1], laterality=laterality)

    values = np.asarray(values, dtype=float)
    vmin, vmax = float(values.min()), float(values.max())
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]

    per_vertex_val = np.where(sub >= 0, values[np.clip(sub, 0, 8)], np.nan)
    rgba = cmap(norm(per_vertex_val))             # (n, 4) sRGB-ish
    rgba[sub < 0] = (0.5, 0.5, 0.5, 1.0)          # outside grid -> neutral grey
    rgba[:, :3] = srgb_to_linear(rgba[:, :3])     # store linear for faithful output

    # write a per-point FLOAT_COLOR attribute
    if COLOR_ATTR in mesh.color_attributes:
        mesh.color_attributes.remove(mesh.color_attributes[COLOR_ATTR])
    attr = mesh.color_attributes.new(name=COLOR_ATTR, type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", rgba.astype(np.float32).ravel())
    mesh.update()
    return vmin, vmax


def make_emission_material(obj):
    mat = bpy.data.materials.new("etdrs_emission")
    if mat.node_tree is None:   # use_nodes is deprecated to touch in Blender 5+
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emis = nt.nodes.new("ShaderNodeEmission")
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = COLOR_ATTR
    nt.links.new(attr.outputs["Color"], emis.inputs["Color"])
    nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def render_to(path):
    scene = bpy.context.scene
    # Standard view transform so emission colours match the colormap exactly.
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def composite_colorbar(render_png, out_png, cmap_name, vmin, vmax, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    img = plt.imread(str(render_png))
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")

    cax = fig.add_axes([0.905, 0.12, 0.018, 0.76])
    sm = ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap_name)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, color="white", fontsize=13)
    cb.outline.set_edgecolor("white")
    cax.tick_params(color="white", labelcolor="white", labelsize=11)

    fig.savefig(str(out_png), dpi=100)
    plt.close(fig)


def main():
    args = parse_args()
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}. Run setup_scene.py first.")
    bpy.ops.wm.open_mainfile(filepath=str(TEMPLATE))

    values = read_layer_values(args.input, args.layer)
    print(f"Layer {args.layer!r} subfield values (canonical order):")
    for name, v in zip(SUBFIELDS, values):
        print(f"  {name:16s} {v:7.2f}")

    obj = get_cap_object()
    vmin, vmax = bake_vertex_colors(obj, values, args.colormap, args.laterality)
    make_emission_material(obj)
    print(f"Coloured {len(obj.data.vertices)} vertices with '{args.colormap}' over "
          f"vmin={vmin:.2f}, vmax={vmax:.2f}")

    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        render_png = Path(tmp) / "render.png"
        render_to(render_png)
        composite_colorbar(render_png, out_png, args.colormap, vmin, vmax,
                           label=f"{args.layer} thickness (µm)")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
