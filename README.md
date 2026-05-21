# RetinaPainter

A pipeline that turns OCT retinal-layer measurements into 3D, ETDRS-aware
visualizations. It segments retinal layers from OCT volumes, summarizes them on
the clinical ETDRS macular grid, and renders the macula as a colored spherical
cap in Blender.

## Components

| Module | What it does |
| --- | --- |
| [`src/segment_layers.py`](src/segment_layers.py) | Retinal layer-boundary segmentation of an OCT volume with ReLayNet → per-layer surfaces (dict / HDF5 / CLI). |
| [`src/etdrs.py`](src/etdrs.py) | Fovea detection, ETDRS 9-subfield averaging of a thickness map, and a bullseye renderer. |
| [`src/geometry.py`](src/geometry.py) | Spherical-cap macular mesh (`trimesh`) and per-vertex ETDRS subfield assignment. |
| [`blender/setup_scene.py`](blender/setup_scene.py) | Builds the Blender render template (3-point lights, 30° camera, Cycles, grey world). |
| [`blender/render.py`](blender/render.py) | Renders the cap colored by an ETDRS CSV, with a matplotlib colorbar composited on. |
| [`scripts/render_retinapainter.py`](scripts/render_retinapainter.py) | Typer CLI wrapping the Blender render step. |

## Requirements

- A Python env with `numpy`, `torch`, `eyepy`, `h5py`, `trimesh`, `matplotlib`,
  `scipy`, `typer` (this repo is developed against the conda env `oct-analysis`).
- **Blender 5.1+** for the 3D pipeline. Point the CLI at it with `--blender` or
  the `$BLENDER` environment variable (default:
  `/Applications/Blender.app/Contents/MacOS/Blender`). `render.py` needs
  `matplotlib` in Blender's bundled Python:
  `"$BLENDER_PY" -m pip install matplotlib`.

## Usage

### Render a layer's ETDRS map onto the 3D cap (Day 2 pipeline)

The render reads a CSV where **each row is a layer** and the **columns are the 9
ETDRS subfields**:

```csv
layer,central,inner_superior,inner_nasal,inner_inferior,inner_temporal,outer_superior,outer_nasal,outer_inferior,outer_temporal
RNFL,80,103,108,101,95,118,128,115,105
```

Build the Blender template once (imports the cap mesh, lighting, camera, Cycles):

```bash
# Generate the cap mesh, then the render template
python -m src.geometry                     # writes blender/retinal_cap.obj
"$BLENDER" --background --python blender/setup_scene.py   # writes blender/retinal_template.blend
```

Then render a layer through the Typer CLI:

```bash
python scripts/render_retinapainter.py \
    --input blender/sample_values.csv \
    --layer RNFL \
    --output blender/rnfl_render.png \
    --colormap viridis
```

This colors each vertex by its subfield's value (auto-normalized over the
layer's min/max), renders with Cycles, and composites a colorbar. Use
`--laterality OS` to mirror nasal/temporal, and `--colormap <name>` for any
matplotlib colormap.

### Earlier stages

```bash
# Segment retinal layers from an OCT volume (TIFF folder or .vol)
python -m src.segment_layers --volume data/duke/NORMAL1/TIFFs/8bitTIFFs --output layers.h5
```

`src.etdrs` (subfield averaging, fovea detection, bullseye) and `src.geometry`
(cap mesh, subfield labels) are importable libraries; see their docstrings.

## Tests

```bash
python -m pytest tests/
```
