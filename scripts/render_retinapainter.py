#!/usr/bin/env python3
"""RetinaPainter render CLI.

A thin, friendly Typer front-end over ``blender/render.py``. It locates the
Blender executable, validates inputs, and launches Blender in background mode so
you don't have to remember the ``blender --background --python … --`` incantation.

Example
-------
    python scripts/render_retinapainter.py \
        --input blender/sample_values.csv --layer RNFL \
        --output blender/rnfl_render.png --colormap viridis
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDER_SCRIPT = REPO_ROOT / "blender" / "render.py"
DEFAULT_BLENDER = os.environ.get(
    "BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender"
)


def main(
    input: Path = typer.Option(
        ..., "--input", "-i", exists=True, dir_okay=False, readable=True,
        help="CSV of ETDRS values (rows = layers, columns = the 9 subfields).",
    ),
    layer: str = typer.Option(
        ..., "--layer", "-l", help="Layer (CSV row) to render, e.g. 'RNFL'.",
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output PNG path.",
    ),
    colormap: str = typer.Option(
        "viridis", "--colormap", "-c", help="matplotlib colormap name.",
    ),
    laterality: str = typer.Option(
        "OD", "--laterality", help="Eye laterality: OD or OS (swaps nasal/temporal).",
    ),
    blender: str = typer.Option(
        DEFAULT_BLENDER, "--blender",
        help="Path to the Blender executable (or set the $BLENDER env var).",
    ),
) -> None:
    """Render the retinal cap mesh coloured by an ETDRS subfield CSV.

    Colours are mapped through COLORMAP over a (vmin, vmax) auto-computed from the
    chosen layer, and a matplotlib colorbar is composited onto the final image.
    """
    if laterality.upper() not in ("OD", "OS"):
        typer.secho("laterality must be 'OD' or 'OS'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    blender_exe = blender if Path(blender).exists() else shutil.which(blender)
    if not blender_exe:
        typer.secho(
            f"Blender not found at {blender!r}. Pass --blender or set $BLENDER.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(2)
    if not RENDER_SCRIPT.exists():
        typer.secho(f"Missing render script: {RENDER_SCRIPT}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(blender_exe), "--background", "--python", str(RENDER_SCRIPT), "--",
        "--input", str(input.resolve()),
        "--layer", layer,
        "--output", str(output),
        "--colormap", colormap,
        "--laterality", laterality.upper(),
    ]
    typer.secho(f"$ {' '.join(cmd)}", fg=typer.colors.BLUE)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        typer.secho(f"Blender exited with code {result.returncode}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(result.returncode)
    if not output.exists():
        typer.secho("Blender succeeded but no output file was produced.",
                    fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(1)

    typer.secho(f"✓ Rendered {layer} → {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    typer.run(main)
