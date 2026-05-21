"""Render the retinal cap coloured by ETDRS values, with a colorbar.

Run via Blender:

    # single layer painted on the base cap
    blender --background --python render.py -- \
        --mode single --input values.csv --layer RNFL --output out.png --colormap viridis

    # all layer_*.obj surfaces stacked, each its own colormap, semi-transparent
    blender --background --python render.py -- \
        --mode layered --input values.csv --output out.png --alpha 0.5

Pipeline (both modes):
  1. Parse CLI args (after Blender's ``--`` separator).
  2. Open ``retinal_template.blend`` (camera / lights / Cycles set up already).
  3. Read the CSV (rows = layer names, columns = the 9 ETDRS subfields).
  4. Colour vertices by subfield value through a colormap, normalised over a
     (vmin, vmax) auto-computed from each layer. Colours are baked to a
     vertex-colour attribute; an emission shader + the Standard view transform
     make rendered colours match the colormap.
       - single  : paint one CSV layer onto the base cap.
       - layered : import every ``blender/layer_*.obj`` offset surface, give each
                   its own colormap, and a Transparent-BSDF mix (per-layer alpha)
                   so the underlying layers show through.
  5. Render with Cycles (256 samples in layered mode), then composite a
     matplotlib colorbar (one per layer in layered mode).
  6. Save the final PNG to ``--output``.

matplotlib must be importable in Blender's bundled Python.
"""
import argparse
import csv
import glob
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

# Distinct sequential colormaps assigned to layers in CSV order (layered mode).
LAYER_CMAPS = ["Blues", "Greens", "Oranges", "Reds", "Purples", "YlOrBr", "PuBuGn", "RdPu"]


# --- CLI ----------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Render the retinal cap coloured by an ETDRS CSV.")
    p.add_argument("--mode", choices=["single", "layered"], default="single")
    p.add_argument("--input", required=True, help="CSV: rows=layers, cols=9 subfields")
    p.add_argument("--layer", default=None, help="layer to render (required for --mode single)")
    p.add_argument("--output", required=True, help="output PNG path")
    p.add_argument("--colormap", default="viridis", help="matplotlib colormap (single mode)")
    p.add_argument("--alpha", type=float, default=0.5, help="per-layer opacity (layered mode)")
    p.add_argument("--samples", type=int, default=0, help="Cycles samples (0 = mode default)")
    p.add_argument("--laterality", default="OD", choices=["OD", "OS"])
    return p.parse_args(argv)


# --- CSV -----------------------------------------------------------------------
def read_table(csv_path):
    """Return an ordered dict {layer: [9 subfield values in canonical order]}."""
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    header = [h.strip().lower() for h in rows[0][1:]]
    named = all(name in header for name in SUBFIELDS)
    table = {}
    for r in rows[1:]:
        vals = [float(x) for x in r[1:]]
        if named:
            table[r[0].strip()] = [vals[header.index(name)] for name in SUBFIELDS]
        else:
            if len(vals) < 9:
                raise ValueError(f"Layer {r[0]!r} has {len(vals)} columns; need 9.")
            table[r[0].strip()] = vals[:9]
    return table


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


# --- mesh helpers -------------------------------------------------------------
def import_obj(path):
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(path), up_axis="Z", forward_axis="Y")
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new:
        raise RuntimeError(f"No mesh imported from {path}")
    return new[0]


def get_cap_object():
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh object in the template scene.")
    for o in meshes:
        if "retinal_cap" in o.name.lower():
            return o
    return meshes[0]


def remove_meshes():
    for o in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
        bpy.data.objects.remove(o, do_unlink=True)


def bake_vertex_colors(obj, values, cmap_name, laterality):
    """Colour each vertex by its subfield value; returns (vmin, vmax)."""
    import matplotlib

    mesh = obj.data
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    mw = np.array(obj.matrix_world)
    world = np.column_stack([co, np.ones(n)]) @ mw.T
    sub = vertex_subfields(world[:, 0], world[:, 1], laterality=laterality)

    values = np.asarray(values, dtype=float)
    vmin, vmax = float(values.min()), float(values.max())
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps[cmap_name]

    per_vertex_val = np.where(sub >= 0, values[np.clip(sub, 0, 8)], np.nan)
    rgba = cmap(norm(per_vertex_val))
    rgba[sub < 0] = (0.5, 0.5, 0.5, 1.0)
    rgba[:, :3] = srgb_to_linear(rgba[:, :3])

    if COLOR_ATTR in mesh.color_attributes:
        mesh.color_attributes.remove(mesh.color_attributes[COLOR_ATTR])
    attr = mesh.color_attributes.new(name=COLOR_ATTR, type="FLOAT_COLOR", domain="POINT")
    attr.data.foreach_set("color", rgba.astype(np.float32).ravel())
    mesh.update()
    return vmin, vmax


def _color_attr_node(nt):
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = "GEOMETRY"
    attr.attribute_name = COLOR_ATTR
    return attr


def make_emission_material(obj):
    """Opaque emission of the vertex colour (single mode)."""
    mat = bpy.data.materials.new("etdrs_emission")
    if mat.node_tree is None:   # use_nodes is deprecated to touch in Blender 5+
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emis = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(_color_attr_node(nt).outputs["Color"], emis.inputs["Color"])
    nt.links.new(emis.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def make_transparent_material(obj, alpha, name):
    """Mix(Transparent BSDF, Emission) so layers below show through (layered mode)."""
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    mix = nt.nodes.new("ShaderNodeMixShader")
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    emis = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(_color_attr_node(nt).outputs["Color"], emis.inputs["Color"])
    mix.inputs["Fac"].default_value = float(alpha)   # 0 -> transparent, 1 -> emission
    nt.links.new(transp.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emis.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def render_to(path, samples=0):
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"   # faithful colormap colours
    if samples > 0:
        scene.cycles.samples = samples
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


# --- colorbar compositing -----------------------------------------------------
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
    cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap_name), cax=cax)
    cb.set_label(label, color="white", fontsize=13)
    cb.outline.set_edgecolor("white")
    cax.tick_params(color="white", labelcolor="white", labelsize=11)
    fig.savefig(str(out_png), dpi=100); plt.close(fig)


def composite_multi_colorbar(render_png, out_png, specs):
    """specs: list of (layer_name, cmap_name, vmin, vmax) -> stacked mini colorbars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    img = plt.imread(str(render_png))
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")

    n = len(specs)
    top, bottom = 0.94, 0.06
    slot = (top - bottom) / max(n, 1)
    barx, barw = 0.845, 0.13
    for k, (name, cmap_name, vmin, vmax) in enumerate(specs):
        by = top - (k + 1) * slot + slot * 0.40
        cax = fig.add_axes([barx, by, barw, slot * 0.20])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap_name),
                          cax=cax, orientation="horizontal")
        cb.set_ticks([vmin, vmax])
        cb.outline.set_edgecolor("white")
        cax.tick_params(color="white", labelcolor="white", labelsize=7, pad=1)
        fig.text(barx, by + slot * 0.26, f"{name}  (µm)", color="white",
                 fontsize=9, fontweight="bold", va="bottom")
    fig.savefig(str(out_png), dpi=100); plt.close(fig)


# --- modes --------------------------------------------------------------------
def run_single(args):
    table = read_table(args.input)
    if args.layer is None:
        raise ValueError("--layer is required for --mode single.")
    if args.layer not in table:
        raise ValueError(f"Layer {args.layer!r} not in CSV. Available: {list(table)}")
    values = table[args.layer]
    obj = get_cap_object()
    vmin, vmax = bake_vertex_colors(obj, values, args.colormap, args.laterality)
    make_emission_material(obj)
    print(f"[single] {args.layer}: {len(obj.data.vertices)} verts, '{args.colormap}', "
          f"vmin={vmin:.2f}, vmax={vmax:.2f}")

    with tempfile.TemporaryDirectory() as tmp:
        render_png = Path(tmp) / "render.png"
        render_to(render_png, samples=args.samples)
        composite_colorbar(render_png, args.output, args.colormap, vmin, vmax,
                            label=f"{args.layer} thickness (µm)")


def run_layered(args):
    table = read_table(args.input)
    remove_meshes()   # drop the template's base cap; we stack the layer surfaces

    specs = []
    for i, (layer, values) in enumerate(table.items()):
        obj_path = SCRIPT_DIR / f"layer_{layer.replace('/', '_')}.obj"
        if not obj_path.exists():
            print(f"[layered] skip {layer!r}: missing {obj_path.name}")
            continue
        obj = import_obj(obj_path)
        obj.name = f"layer_{layer.replace('/', '_')}"
        cmap = LAYER_CMAPS[i % len(LAYER_CMAPS)]
        vmin, vmax = bake_vertex_colors(obj, values, cmap, args.laterality)
        make_transparent_material(obj, args.alpha, name=f"mat_{obj.name}")
        specs.append((layer, cmap, vmin, vmax))
        print(f"[layered] {layer:9s} <- {obj_path.name}  cmap={cmap:8s} "
              f"range=[{vmin:.0f},{vmax:.0f}]  alpha={args.alpha}")
    if not specs:
        raise RuntimeError("No layer_*.obj surfaces matched the CSV rows.")

    samples = args.samples if args.samples > 0 else 256
    with tempfile.TemporaryDirectory() as tmp:
        render_png = Path(tmp) / "render.png"
        render_to(render_png, samples=samples)
        composite_multi_colorbar(render_png, args.output, specs)
    print(f"[layered] {len(specs)} layers, {samples} samples")


def main():
    args = parse_args()
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}. Run setup_scene.py first.")
    bpy.ops.wm.open_mainfile(filepath=str(TEMPLATE))

    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(out_png)

    if args.mode == "layered":
        run_layered(args)
    else:
        run_single(args)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
