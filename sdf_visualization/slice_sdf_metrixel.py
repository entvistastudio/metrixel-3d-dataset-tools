# *******************************************************************************
#  *                                                                             *
#  *  Metrixel - Cross-platform Application                                      *
#  *                                                                             *
#  *  Copyright © 2026 EntertainmentVista Studio Pte. Ltd.                       *
#  *  All rights reserved.                                                       *
#  *                                                                             *
# *******************************************************************************

import argparse
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_sdf_bin, normalize_frame_name, resolve_sdf_path
from config import get_config


def slice_sdf_metrixel(
    sequence_id: str,
    frame_name: str,
    sdf_dataset_root: str,
    resolution: int,
    slice_z,
    slice_y,
    slice_x,
):
    """Show three orthogonal slices through the SDF volume as heatmaps.

    Diverging colormap: blue = inside (negative SDF), red = outside (positive),
    white = surface (zero). The zero contour is overlaid in black so the surface
    cross-section is unambiguous even at low resolution.
    """
    config = get_config(sdf_dataset_root)
    frame_name = normalize_frame_name(frame_name)
    sdf_path = resolve_sdf_path(config["SDF_DATASET_PATH"], sequence_id, frame_name)

    print(f"Loading {sdf_path} (R={resolution})")
    volume = load_sdf_bin(sdf_path, resolution)

    if slice_z is None:
        slice_z = resolution // 2
    if slice_y is None:
        slice_y = resolution // 2
    if slice_x is None:
        slice_x = resolution // 2

    for name, idx in (("z", slice_z), ("y", slice_y), ("x", slice_x)):
        if not (0 <= idx < resolution):
            raise ValueError(f"--{name}={idx} out of range [0, {resolution - 1}]")

    vmax = float(np.max(np.abs(volume)))
    if vmax <= 0.0:
        vmax = 1.0
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    img_xy = volume[slice_z, :, :]
    im0 = axes[0].imshow(img_xy, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
    if img_xy.min() < 0.0 < img_xy.max():
        axes[0].contour(img_xy, levels=[0.0], colors="black", linewidths=1)
    axes[0].set_title(f"XY slice (z={slice_z})")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im0, ax=axes[0], label="signed distance")

    img_xz = volume[:, slice_y, :]
    im1 = axes[1].imshow(img_xz, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
    if img_xz.min() < 0.0 < img_xz.max():
        axes[1].contour(img_xz, levels=[0.0], colors="black", linewidths=1)
    axes[1].set_title(f"XZ slice (y={slice_y})")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    plt.colorbar(im1, ax=axes[1], label="signed distance")

    img_yz = volume[:, :, slice_x]
    im2 = axes[2].imshow(img_yz, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax)
    if img_yz.min() < 0.0 < img_yz.max():
        axes[2].contour(img_yz, levels=[0.0], colors="black", linewidths=1)
    axes[2].set_title(f"YZ slice (x={slice_x})")
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")
    plt.colorbar(im2, ax=axes[2], label="signed distance")

    fig.suptitle(
        f"Metrixel SDF slices (R={resolution}): {sequence_id}/{frame_name}\n"
        f"blue = inside (d<0), red = outside (d>0), black contour = surface (d=0)"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # python slice_sdf_metrixel.py --sdf_dataset_root /path/to/assets \
    #     --seq 000001 --frame 000_000_000_0000 --sdf_resolution 16
    parser = argparse.ArgumentParser(
        description=(
            "Slice through a Metrixel SDF .bin volume — XY, XZ, YZ heatmaps with "
            "the zero contour overlaid. Useful when marching cubes underwhelms "
            "(low R) or when you want to inspect interior structure."
        )
    )
    parser.add_argument(
        "--sdf_dataset_root",
        type=str,
        required=True,
        help="Folder containing sequence IDs (e.g. .../assets)",
    )
    parser.add_argument("--seq", type=str, required=True, help="Sequence ID (e.g. 000001)")
    parser.add_argument(
        "--frame",
        type=str,
        required=True,
        help="Frame suffix matching the mesh/image filenames (e.g. 000_000_000_0000)",
    )
    parser.add_argument(
        "--sdf_resolution",
        type=int,
        required=True,
        help="Cubic grid resolution R; the .bin must contain R**3 float32 values",
    )
    parser.add_argument(
        "--z", type=int, default=None, help="Z slice index (default: R/2)"
    )
    parser.add_argument(
        "--y", type=int, default=None, help="Y slice index (default: R/2)"
    )
    parser.add_argument(
        "--x", type=int, default=None, help="X slice index (default: R/2)"
    )

    args = parser.parse_args()
    slice_sdf_metrixel(
        args.seq,
        args.frame,
        args.sdf_dataset_root,
        args.sdf_resolution,
        args.z,
        args.y,
        args.x,
    )
