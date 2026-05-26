"""Compose the three RetinaPainter views into one publication "hero" figure.

Given three pre-rendered panel images -- a 3D angled render, an en-face ETDRS
bullseye, and a cross-section (stylized OCT B-scan) -- lay them out with
matplotlib ``gridspec`` as a single 3840x1080 figure with a shared title bar
across the top and one unified colorbar on the right.

    from src.compose_views import compose_views
    compose_views("render.png", "bullseye.png", "xsection.png", "hero.png",
                  title="RetinaPainter", colormap="viridis", vmin=80, vmax=128,
                  cbar_label="RNFL thickness (µm)")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

DEFAULT_PANEL_TITLES = (
    "A · 3D angled render",
    "B · En-face ETDRS bullseye",
    "C · Cross-section (OCT B-scan)",
)
TITLE_BAR_COLOR = "#10243f"   # deep navy header bar
FIG_BG = "white"


def _panel(ax, image_path, title):
    """Draw one image panel with a light frame and a sub-title."""
    img = mpimg.imread(str(image_path))
    ax.imshow(img, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.7")
        s.set_linewidth(1.0)
    ax.set_title(title, fontsize=17, fontweight="bold", color="#10243f", pad=8)


def compose_views(
    render_path,
    bullseye_path,
    cross_section_path,
    output_path,
    *,
    title: str = "RetinaPainter",
    subtitle: str | None = None,
    colormap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = "thickness (µm)",
    panel_titles=DEFAULT_PANEL_TITLES,
    width: int = 3840,
    height: int = 1080,
):
    """Compose the three views into a single ``width`` x ``height`` PNG.

    Parameters mirror the inputs/labels; ``colormap``/``vmin``/``vmax`` define the
    shared (unified) colorbar drawn on the right. Returns the saved path.
    """
    paths = [Path(render_path), Path(bullseye_path), Path(cross_section_path)]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing panel image: {p}")

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=FIG_BG)

    # gridspec tuned to the panels' aspect ratios: a title bar spanning the top,
    # the 3D render and (square) bullseye side-by-side, and the wide cross-section
    # as a full-width banner beneath. Right margin reserved for the colorbar.
    gs = fig.add_gridspec(
        3, 2, height_ratios=[0.13, 0.52, 0.35], width_ratios=[1.3, 1.0],
        left=0.015, right=0.925, top=0.98, bottom=0.04, hspace=0.22, wspace=0.05,
    )

    # --- shared title bar ----------------------------------------------------
    tax = fig.add_subplot(gs[0, :])
    tax.set_xticks([]); tax.set_yticks([])
    tax.set_facecolor(TITLE_BAR_COLOR)
    for s in tax.spines.values():
        s.set_visible(False)
    ty = 0.5 if subtitle is None else 0.64
    tax.text(0.5, ty, title, ha="center", va="center", color="white",
             fontsize=34, fontweight="bold", transform=tax.transAxes)
    if subtitle:
        tax.text(0.5, 0.22, subtitle, ha="center", va="center", color="#cfe0f5",
                 fontsize=18, transform=tax.transAxes)

    # --- three panels --------------------------------------------------------
    _panel(fig.add_subplot(gs[1, 0]), paths[0], panel_titles[0])   # 3D render
    _panel(fig.add_subplot(gs[1, 1]), paths[1], panel_titles[1])   # en-face bullseye
    _panel(fig.add_subplot(gs[2, :]), paths[2], panel_titles[2])   # cross-section banner

    # --- unified colorbar (right margin, aligned with the top panels) --------
    cax = fig.add_axes([0.94, 0.45, 0.011, 0.40])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax),
                                     cmap=plt.get_cmap(colormap)), cax=cax)
    cb.set_label(cbar_label, fontsize=16)
    cb.ax.tick_params(labelsize=12)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=FIG_BG)
    plt.close(fig)
    print(f"Saved hero figure -> {out}  ({width}x{height})")
    return out


def main():
    import argparse

    p = argparse.ArgumentParser(description="Compose the 3 RetinaPainter views into a hero figure.")
    p.add_argument("--render", required=True, help="3D angled render PNG")
    p.add_argument("--bullseye", required=True, help="en-face ETDRS bullseye PNG")
    p.add_argument("--cross-section", required=True, help="cross-section PNG")
    p.add_argument("--output", required=True, help="output composite PNG")
    p.add_argument("--title", default="RetinaPainter")
    p.add_argument("--subtitle", default=None)
    p.add_argument("--colormap", default="viridis")
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--cbar-label", default="thickness (µm)")
    p.add_argument("--width", type=int, default=3840)
    p.add_argument("--height", type=int, default=1080)
    a = p.parse_args()
    compose_views(a.render, a.bullseye, a.cross_section, a.output,
                  title=a.title, subtitle=a.subtitle, colormap=a.colormap,
                  vmin=a.vmin, vmax=a.vmax, cbar_label=a.cbar_label,
                  width=a.width, height=a.height)


if __name__ == "__main__":
    main()
