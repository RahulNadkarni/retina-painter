"""Retinal layer boundary segmentation for OCT volumes.

Given an OCT volume, this returns, for every requested retinal layer, a 2D array
of layer-boundary y-coordinates indexed by ``[B-scan, A-scan]`` (the axial pixel
row of that boundary in each column of each B-scan).

Model
-----
OCTolyzer itself does **not** ship a retinal-layer segmentation model -- its
README states layers are read from the device's built-in ``.vol`` metadata, and
its only pretrained OCT models (Choroidalyzer / DeepGPET) segment the choroid.
The Duke SD-OCT data here is a folder of plain TIFFs with no embedded
segmentation, so we use **ReLayNet** (Roy et al., *Biomed. Opt. Express* 2017),
the fully-convolutional network introduced and trained on exactly this Duke
SD-OCT benchmark.

The published, properly-trained weights are distributed as MatConvNet models in
https://github.com/ai-med/ReLayNet (``TrainedModels/NetFold*.mat``, an 8-fold
cross-validation). We cache one fold locally on first use and port its weights
into an equivalent PyTorch model. (The separate ``relaynet_pytorch`` repo ships
only an undertrained demo checkpoint that does not segment usefully, so it is
not used.)

ReLayNet predicts 10 classes; indices 0-8 are ordered top -> bottom of the
image and index 9 is fluid:

    0  RaR      region above the retina
    1  ILM      inner limiting membrane band
    2  NFL-IPL  nerve fibre + ganglion cell + inner plexiform (MERGED)
    3  INL      inner nuclear layer
    4  OPL      outer plexiform layer
    5  ONL-ISM  outer nuclear -> inner-segment myeloid
    6  ISE      inner-segment ellipsoid
    7  OS-RPE   outer segment -> retinal pigment epithelium
    8  RbR      region below the RPE
    9  fluid    intraretinal fluid (absent in healthy eyes)

The 8 interfaces between regions 0-8 are the layer surfaces we return. Each
requested layer maps to the *inner (top) surface* of that layer, with ``BM`` the
retina's outer surface:

    requested layer   ReLayNet surface (class boundary)
    ---------------    --------------------------------
    ILM                top of class 1   (retina inner surface)
    RNFL               top of class 2   (inner surface of the NFL-IPL complex)
    GCL+IPL            NOT RESOLVED      -- ReLayNet merges NFL+GCL+IPL into one
                                            class, so GCL/IPL has no surface
                                            distinct from RNFL; returned as NaN
    INL                top of class 3
    OPL                top of class 4
    ONL                top of class 5   (top of ONL-ISM)
    IS/OS              top of class 6   (inner-segment ellipsoid / IS-OS line)
    RPE                top of class 7   (top of OS-RPE band)
    BM                 top of class 8   (Bruch's membrane / retina outer surface)

CLI
---
    python -m src.segment_layers --volume PATH --output OUT.h5

`--volume` may be a folder of B-scan TIFFs (e.g. a Duke subject's
``TIFFs/8bitTIFFs`` directory) or a Heidelberg ``.vol`` file.
"""
from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --- configuration ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# The user asked to cache the checkpoint here. Despite the "octolyzer" name we
# cache the ReLayNet checkpoint (OCTolyzer has no retinal-layer model).
DEFAULT_CACHE_DIR = REPO_ROOT / "models" / "octolyzer_cache"
CHECKPOINT_NAME = "relaynet_NetFold1.mat"
CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/ai-med/ReLayNet/master/TrainedModels/NetFold1.mat"
)

# ReLayNet class indices (0-8 ordered top -> bottom; 9 is fluid).
CLASS_NAMES = [
    "RaR", "ILM", "NFL-IPL", "INL", "OPL", "ONL-ISM", "ISE", "OS-RPE", "RbR", "fluid",
]
NUM_CLASSES = len(CLASS_NAMES)
FLUID_CLASS = 9

# Requested layer -> (class threshold k, note). The surface is the first row
# whose class index is in [k, 8] (fluid excluded). k=None means unresolvable.
LAYER_TO_THRESHOLD = {
    "ILM":     (1, "retina inner surface (top of class 1)"),
    "RNFL":    (2, "inner surface of NFL-IPL complex (top of class 2)"),
    "GCL+IPL": (None, "merged into NFL-IPL by ReLayNet; not separable -> NaN"),
    "INL":     (3, "top of INL (class 3)"),
    "OPL":     (4, "top of OPL (class 4)"),
    "ONL":     (5, "top of ONL-ISM (class 5)"),
    "IS/OS":   (6, "inner-segment ellipsoid / IS-OS line (top of class 6)"),
    "RPE":     (7, "top of OS-RPE band (class 7)"),
    "BM":      (8, "Bruch's membrane / retina outer surface (top of class 8)"),
}
LAYER_ORDER = list(LAYER_TO_THRESHOLD.keys())


# --- model --------------------------------------------------------------------
class ReLayNet(nn.Module):
    """ReLayNet (encoder / bottleneck / decoder) matching ai-med/ReLayNet.

    3 encoder blocks (conv 7x3 -> BN -> ReLU -> max-pool-with-indices), a
    bottleneck conv block, and 3 decoder blocks (max-unpool -> concat skip ->
    conv -> BN -> ReLU), then a 1x1 classifier to ``num_classes``.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        k = dict(kernel_size=(7, 3), padding=(3, 1))
        self.conv1 = nn.Conv2d(1, 64, **k);   self.bn1 = nn.BatchNorm2d(64, eps=1e-10)
        self.conv2 = nn.Conv2d(64, 64, **k);  self.bn2 = nn.BatchNorm2d(64, eps=1e-10)
        self.conv3 = nn.Conv2d(64, 64, **k);  self.bn3 = nn.BatchNorm2d(64, eps=1e-10)
        self.conv4 = nn.Conv2d(64, 64, **k);  self.bn4 = nn.BatchNorm2d(64, eps=1e-10)
        self.deconv3 = nn.Conv2d(128, 64, **k); self.bn3x = nn.BatchNorm2d(64, eps=1e-10)
        self.deconv2 = nn.Conv2d(128, 64, **k); self.bn2x = nn.BatchNorm2d(64, eps=1e-10)
        self.deconv1 = nn.Conv2d(128, 64, **k); self.bn1x = nn.BatchNorm2d(64, eps=1e-10)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        c1 = torch.relu(self.bn1(self.conv1(x)));  p1, i1 = self.pool(c1)
        c2 = torch.relu(self.bn2(self.conv2(p1))); p2, i2 = self.pool(c2)
        c3 = torch.relu(self.bn3(self.conv3(p2))); p3, i3 = self.pool(c3)
        c4 = torch.relu(self.bn4(self.conv4(p3)))  # bottleneck
        u3 = self.unpool(c4, i3, output_size=c3.shape)
        d3 = torch.relu(self.bn3x(self.deconv3(torch.cat([u3, c3], dim=1))))
        u2 = self.unpool(d3, i2, output_size=c2.shape)
        d2 = torch.relu(self.bn2x(self.deconv2(torch.cat([u2, c2], dim=1))))
        u1 = self.unpool(d2, i1, output_size=c1.shape)
        d1 = torch.relu(self.bn1x(self.deconv1(torch.cat([u1, c1], dim=1))))
        return self.classifier(d1)


def ensure_checkpoint(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return the local checkpoint path, downloading it to `cache_dir` once."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cache_dir / CHECKPOINT_NAME
    if not ckpt.exists() or ckpt.stat().st_size == 0:
        print(f"Downloading ReLayNet checkpoint -> {ckpt}")
        urllib.request.urlretrieve(CHECKPOINT_URL, ckpt)
    return ckpt


def _matconvnet_to_torch(model: ReLayNet, mat_path: Path) -> ReLayNet:
    """Copy MatConvNet (dagnn) parameters from `mat_path` into `model`.

    MatConvNet stores conv filters as (H, W, in, out) and batch-norm moments as
    an (N, 2) array of [running_mean, running_sigma]. We permute filters to
    PyTorch's (out, in, H, W) and set running_var = sigma**2 (eps ~ 0).
    """
    import scipy.io as sio

    m = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    params = {p.name: np.array(p.value) for p in m["net"].params}

    def load_conv(layer: nn.Conv2d, fkey: str, bkey: str):
        w = params[fkey]
        if w.ndim == 3:      # (H, W, out) -- input dim of 1 was squeezed away
            w = w[:, :, None, :]
        elif w.ndim == 2:    # (in, out) for the 1x1 classifier
            w = w[None, None, :, :]
        layer.weight.data = torch.from_numpy(
            np.ascontiguousarray(w.transpose(3, 2, 0, 1))
        ).float()
        layer.bias.data = torch.from_numpy(params[bkey].ravel()).float()

    def load_bn(bn: nn.BatchNorm2d, fkey: str, bkey: str, mkey: str):
        bn.weight.data = torch.from_numpy(params[fkey].ravel()).float()
        bn.bias.data = torch.from_numpy(params[bkey].ravel()).float()
        mom = params[mkey]
        bn.running_mean.data = torch.from_numpy(mom[:, 0]).float()
        bn.running_var.data = torch.from_numpy(mom[:, 1] ** 2).float()

    load_conv(model.conv1, "conv1f", "conv1b");  load_bn(model.bn1, "bn1f", "bn1b", "bn1m")
    load_conv(model.conv2, "conv2f", "conv2b");  load_bn(model.bn2, "bn2f", "bn2b", "bn2m")
    load_conv(model.conv3, "conv3f", "conv3b");  load_bn(model.bn3, "bn3f", "bn3b", "bn3m")
    load_conv(model.conv4, "conv4f", "conv4b");  load_bn(model.bn4, "bn4f", "bn4b", "bn4m")
    load_conv(model.deconv3, "deconv3fx", "deconv3bx"); load_bn(model.bn3x, "bn3fx", "bn3bx", "bn3mx")
    load_conv(model.deconv2, "deconv2fx", "deconv2bx"); load_bn(model.bn2x, "bn2fx", "bn2bx", "bn2mx")
    load_conv(model.deconv1, "deconv1fx", "deconv1bx"); load_bn(model.bn1x, "bn1fx", "bn1bx", "bn1mx")
    load_conv(model.classifier, "classf", "classb")
    return model


def load_model(device: str = "cpu", cache_dir: Path = DEFAULT_CACHE_DIR) -> ReLayNet:
    """Build ReLayNet and load the cached MatConvNet weights onto `device`."""
    ckpt = ensure_checkpoint(cache_dir)
    model = ReLayNet()
    _matconvnet_to_torch(model, ckpt)
    return model.to(device).eval()


# --- volume loading -----------------------------------------------------------
def load_volume(volume_path: str | os.PathLike):
    """Load an OCT volume as a uint8 array of shape (n_bscans, height, width).

    Accepts a folder of B-scan TIFFs (Duke layout) or a Heidelberg ``.vol``.
    Returns ``(data, meta)`` where meta carries voxel spacing if available.
    """
    import eyepy

    path = Path(volume_path)
    if path.is_dir():
        volume = eyepy.import_bscan_folder(path)
    elif path.suffix.lower() == ".vol":
        volume = eyepy.import_heyex_vol(str(path))
    else:
        raise ValueError(
            f"Unsupported volume path: {path}. Provide a B-scan TIFF folder or a .vol file."
        )
    data = np.asarray(volume.data)
    meta = {k: volume.meta[k] for k in volume.meta.keys()} if hasattr(volume, "meta") else {}
    return data, meta


# --- inference ----------------------------------------------------------------
def _preprocess(bscan: np.ndarray) -> np.ndarray:
    """Single B-scan -> float32 in [0, 1] (ReLayNet's expected input range)."""
    img = bscan.astype(np.float32)
    if bscan.dtype == np.uint8:
        img /= 255.0
    else:
        m = img.max()
        if m > 0:
            img /= m
    return img


def segment_volume(data: np.ndarray, model: ReLayNet, device: str = "cpu",
                   batch_size: int = 4) -> np.ndarray:
    """Run ReLayNet over the volume, returning an int label map (n, H, W).

    Requires H and W divisible by 8 (the encoder pools by 8x); Duke B-scans
    (496x512) already satisfy this.
    """
    n, H, W = data.shape
    if H % 8 or W % 8:
        raise ValueError(
            f"B-scan size {H}x{W} must be divisible by 8 for ReLayNet's pooling."
        )
    labels = np.empty((n, H, W), dtype=np.uint8)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            batch = np.stack([_preprocess(data[i]) for i in range(start, stop)])
            x = torch.from_numpy(batch).unsqueeze(1).to(device)  # (b,1,H,W)
            pred = model(x).argmax(dim=1).to("cpu").numpy().astype(np.uint8)
            labels[start:stop] = pred
    return labels


def _surface_from_labels(labels_2d: np.ndarray, k: int) -> np.ndarray:
    """Per-column y of the first row whose class is in [k, 8] (NaN if absent).

    Fluid (class 9) is excluded so it never acts as a boundary.
    """
    mask = (labels_2d >= k) & (labels_2d <= 8)   # (H, W)
    present = mask.any(axis=0)                    # (W,)
    y = mask.argmax(axis=0).astype(np.float32)
    y[~present] = np.nan
    return y


def labels_to_boundaries(label_volume: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a (n, H, W) label volume to {layer: (n, W) y-coordinate array}."""
    n, H, W = label_volume.shape
    boundaries: dict[str, np.ndarray] = {}
    for layer, (k, _note) in LAYER_TO_THRESHOLD.items():
        if k is None:
            boundaries[layer] = np.full((n, W), np.nan, dtype=np.float32)
            continue
        out = np.empty((n, W), dtype=np.float32)
        for i in range(n):
            out[i] = _surface_from_labels(label_volume[i], k)
        boundaries[layer] = out
    return boundaries


# --- public API ---------------------------------------------------------------
def segment_layers(
    volume_path: str | os.PathLike,
    device: str = "cpu",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model: ReLayNet | None = None,
    return_labels: bool = False,
    batch_size: int = 4,
):
    """Segment retinal layers in an OCT volume.

    Parameters
    ----------
    volume_path : path to a B-scan TIFF folder or a Heidelberg ``.vol`` file.
    device      : "cpu", "mps", or "cuda" (cpu is the safe default; MPS may lack
                  MaxUnpool2d support).
    cache_dir   : where the ReLayNet checkpoint is cached.
    model       : a preloaded model (optional; otherwise loaded/cached).
    return_labels : if True, also return the raw (n, H, W) class label volume.

    Returns
    -------
    dict mapping each layer name (ILM, RNFL, GCL+IPL, INL, OPL, ONL, IS/OS, RPE,
    BM) to a float32 array of shape (n_bscans, n_ascans) giving the boundary's
    axial pixel row per A-scan. GCL+IPL is all-NaN (see module docstring).
    If ``return_labels`` is True, returns ``(boundaries, label_volume)``.
    """
    data, _meta = load_volume(volume_path)
    if model is None:
        model = load_model(device=device, cache_dir=cache_dir)
    labels = segment_volume(data, model, device=device, batch_size=batch_size)
    boundaries = labels_to_boundaries(labels)
    if return_labels:
        return boundaries, labels
    return boundaries


# --- HDF5 output --------------------------------------------------------------
def save_h5(
    boundaries: dict[str, np.ndarray],
    out_path: str | os.PathLike,
    volume_path: str | os.PathLike | None = None,
    label_volume: np.ndarray | None = None,
) -> None:
    """Write boundaries (and optional label volume) to an HDF5 file.

    Layer arrays live under the ``/boundaries`` group. Dataset names sanitise
    ``/`` (an HDF5 path separator) to ``_``; each dataset keeps its canonical
    name in a ``layer`` attribute.
    """
    import h5py

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["model"] = "ReLayNet (ai-med/ReLayNet NetFold1.mat, ported to PyTorch)"
        f.attrs["checkpoint_url"] = CHECKPOINT_URL
        f.attrs["layer_order"] = np.array(LAYER_ORDER, dtype=h5py.string_dtype())
        f.attrs["class_names"] = np.array(CLASS_NAMES, dtype=h5py.string_dtype())
        f.attrs["axis_meaning"] = "boundary[b, a] = axial pixel row (y) of layer in B-scan b, A-scan a"
        if volume_path is not None:
            f.attrs["volume_path"] = str(volume_path)

        grp = f.create_group("boundaries")
        for layer in LAYER_ORDER:
            ds = grp.create_dataset(
                layer.replace("/", "_"), data=boundaries[layer], compression="gzip",
            )
            ds.attrs["layer"] = layer
            ds.attrs["mapping"] = LAYER_TO_THRESHOLD[layer][1]

        if label_volume is not None:
            ds = f.create_dataset("label_volume", data=label_volume, compression="gzip")
            ds.attrs["description"] = "ReLayNet per-pixel class indices (see class_names)"


# --- optional demo overlay ----------------------------------------------------
def plot_overlay(volume_path, boundaries, bscan_index, layers, out_png):
    """Save a B-scan with one or more layer boundaries overlaid (for inspection)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, _ = load_volume(volume_path)
    bscan = data[bscan_index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(bscan, cmap="gray", aspect="auto")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for c, layer in enumerate(layers):
        y = boundaries[layer][bscan_index]
        x = np.arange(y.shape[0])
        ax.plot(x, y, lw=1.4, color=colors[c % 10], label=layer)
    ax.set_title(f"ReLayNet layer boundary over B-scan #{bscan_index}")
    ax.set_xlabel("A-scan (lateral, px)")
    ax.set_ylabel("Depth (axial, px)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Wrote overlay -> {out_png}")


# --- CLI ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Segment retinal layer boundaries in an OCT volume with ReLayNet.",
    )
    p.add_argument("--volume", required=True, help="B-scan TIFF folder or .vol file")
    p.add_argument("--output", required=True, help="Output .h5 path")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                   help="Inference device (default: cpu; MPS may lack MaxUnpool2d)")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                   help="Directory to cache the model checkpoint")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--no-labels", action="store_true",
                   help="Do not store the raw class-label volume in the .h5")
    p.add_argument("--plot", default=None, help="Optional PNG overlay output path")
    p.add_argument("--plot-bscan", type=int, default=None,
                   help="B-scan index for --plot (default: central)")
    p.add_argument("--plot-layers", default="ILM,IS/OS,BM",
                   help="Comma-separated layer names for --plot")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    print(f"Loading volume: {args.volume}")
    data, _meta = load_volume(args.volume)
    print(f"Volume shape (n_bscans, H, W): {data.shape}")

    model = load_model(device=args.device, cache_dir=Path(args.cache_dir))
    print(f"Running ReLayNet on device='{args.device}' ...")
    labels = segment_volume(data, model, device=args.device, batch_size=args.batch_size)
    boundaries = labels_to_boundaries(labels)

    save_h5(
        boundaries, args.output, volume_path=args.volume,
        label_volume=None if args.no_labels else labels,
    )
    print(f"Wrote boundaries for {len(boundaries)} layers -> {args.output}")
    for layer in LAYER_ORDER:
        arr = boundaries[layer]
        frac = np.isfinite(arr).mean()
        print(f"  {layer:8s} resolved on {frac*100:5.1f}% of A-scans")

    if args.plot:
        idx = args.plot_bscan if args.plot_bscan is not None else data.shape[0] // 2
        layers = [s.strip() for s in args.plot_layers.split(",") if s.strip()]
        plot_overlay(args.volume, boundaries, idx, layers, args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
