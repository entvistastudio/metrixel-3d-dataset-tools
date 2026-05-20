# *******************************************************************************
#  *                                                                             *
#  *  Metrixel - Cross-platform Application                                      *
#  *                                                                             *
#  *  Copyright © 2025 EntertainmentVista Studio Pte. Ltd.                       *
#  *  All rights reserved.                                                       *
#  *                                                                             *
# *******************************************************************************

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from data_loader import load_sdf_bin, normalize_frame_name, resolve_sdf_path
from config import get_config


def visualize_sdf_metrixel(
    sequence_id: str,
    frame_name: str,
    sdf_dataset_root: str,
    resolution: int,
    iso_level: float,
):
    """Extract the iso-surface from an SDF .bin via marching cubes and render it.

    Coordinate convention matches visualize_mesh_metrixel.py — Metrixel world
    axes (X=Right, Y=Up, Z=Toward) are mapped to matplotlib display by swapping
    Y and Z so the plot reads the same way for mesh and SDF side-by-side.
    """
    config = get_config(sdf_dataset_root)
    frame_name = normalize_frame_name(frame_name)
    sdf_path = resolve_sdf_path(config["SDF_DATASET_PATH"], sequence_id, frame_name)

    print(f"Loading {sdf_path} (R={resolution})")
    volume = load_sdf_bin(sdf_path, resolution)

    inside = int(np.sum(volume < 0.0))
    print(f"  voxels: {volume.size}, inside: {inside} ({100.0 * inside / volume.size:.1f}%)")
    print(f"  range: [{volume.min():.3f}, {volume.max():.3f}]")

    if not (volume.min() < iso_level < volume.max()):
        raise ValueError(
            f"--iso={iso_level} is outside the volume value range "
            f"[{volume.min():.3f}, {volume.max():.3f}]; no surface to extract. "
            f"Pick an iso level inside the range, or check that --sdf_resolution matches the file."
        )

    # marching_cubes operates on volume[z, y, x] and returns verts as (z, y, x).
    # Reverse to (x, y, z) so the rest of the pipeline matches the mesh viewer.
    verts, faces, _, _ = marching_cubes(volume, level=iso_level)
    verts = verts[:, ::-1]

    # Metrixel viz: swap Y and Z so Metrixel's "Up" axis aligns with matplotlib's
    # "Up" (matplotlib 3D draws Z vertical by default).
    verts_t = verts.copy()
    verts_t[:, [1, 2]] = verts[:, [2, 1]]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    mesh_collection = Poly3DCollection(verts_t[faces], alpha=0.7)
    mesh_collection.set_edgecolor("k")
    mesh_collection.set_facecolor("lightblue")
    ax.add_collection3d(mesh_collection)

    mn = np.min(verts_t, axis=0)
    mx = np.max(verts_t, axis=0)
    center = (mn + mx) / 2.0
    span = float(np.max(mx - mn))
    if span <= 0.0:
        span = 1.0
    half = span / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.set_xlabel("X (Right)")
    ax.set_ylabel("Y (Up)")
    ax.set_zlabel("Z (Toward)")
    ax.view_init(elev=0, azim=90)

    plt.title(
        f"Metrixel SDF (R={resolution}, iso={iso_level}): {sequence_id}/{frame_name}\n"
        f"verts: {len(verts)}, faces: {len(faces)}"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # python visualize_sdf_metrixel.py --sdf_dataset_root /path/to/assets \
    #     --seq 000001 --frame 000_000_000_0000 --sdf_resolution 16
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a Metrixel SDF .bin volume by extracting its zero-iso "
            "surface via marching cubes."
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
        "--iso",
        type=float,
        default=0.0,
        help="Iso level to extract (default 0.0 = surface). Negative values pull "
             "the surface inward; positive values push it outward.",
    )

    args = parser.parse_args()
    visualize_sdf_metrixel(
        args.seq, args.frame, args.sdf_dataset_root, args.sdf_resolution, args.iso
    )
