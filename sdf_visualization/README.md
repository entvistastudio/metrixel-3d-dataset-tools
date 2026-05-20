# Metrixel SDF Visualization Tools

Two viewers for the per-iteration `.bin` signed-distance volumes that Metrixel writes alongside meshes and images when `--gen_sdf=true`.

## Overview

Metrixel's SDF generator writes a raw float32 volume per iteration to:

```
<output>/assets/<seq>/sdf/sdf_<viewX>_<viewY>_<viewZ>_<frame>.bin
```

Each file is `R**3` float32 values, indexed as `volume[z, y, x]` where
`idx = (z * R + y) * R + x`. Negative = inside, positive = outside, zero = surface.

These tools give you two complementary views:

| Tool | What it shows | Best for |
|------|---------------|----------|
| `visualize_sdf_metrixel.py` | Marching-cubes iso-surface, rendered as a 3D mesh | "Did the SDF bake the right shape?" — direct comparison with the source mesh |
| `slice_sdf_metrixel.py` | Three orthogonal heatmap slices (XY/XZ/YZ) with the zero contour | Inspecting interior structure, debugging holes/leaks, sanity-checking at low R |

## Requirements

```bash
pip install -r requirements.txt
```

Pulls `numpy`, `matplotlib`, `scikit-image` (for `marching_cubes`).

## Quick Start

```bash
# 3D iso-surface (marching cubes)
python visualize_sdf_metrixel.py \
    --sdf_dataset_root /path/to/assets \
    --seq 000001 \
    --frame 000_000_000_0000 \
    --sdf_resolution 16

# Three orthogonal slices through the volume center
python slice_sdf_metrixel.py \
    --sdf_dataset_root /path/to/assets \
    --seq 000001 \
    --frame 000_000_000_0000 \
    --sdf_resolution 16

# Slice at a non-default index
python slice_sdf_metrixel.py \
    --sdf_dataset_root /path/to/assets \
    --seq 000001 \
    --frame 000_000_000_0000 \
    --sdf_resolution 16 \
    --z 4 --y 8 --x 12
```

## Path Resolution

Both tools look for the `.bin` file at:

1. `<sdf_dataset_root>/<seq>/sdf/sdf_<frame>.bin` (Metrixel native layout)
2. `<sdf_dataset_root>/<seq>/sdf_<frame>.bin` (flattened layout)

Whichever exists wins. For Metrixel-native runs, point `--sdf_dataset_root` at the `assets/` folder.

## Coordinate Convention

Matches `tools/mesh_visualization/visualize_mesh_metrixel.py` so SDF and mesh views render identically:

- **X → Right**
- **Y → Up**
- **Z → Toward** the camera

Internally, `marching_cubes` returns vertices in `(z, y, x)` order; the tool reverses to `(x, y, z)` before display, then swaps `Y` and `Z` for matplotlib's vertical axis.

## Resolution Tips

| R   | Voxels  | Looks like                                    |
|-----|---------|-----------------------------------------------|
| 8   | 512     | Smoke test only — silhouette barely visible   |
| 16  | 4 K     | Chunky, recognisable shape                    |
| 32  | 32 K    | Smooth surface for simple meshes              |
| 64  | 262 K   | Production quality                            |
| 128 | 2 M     | High detail; CPU baseline gets slow (minutes per asset) |

For production training data the recommended floor is `--sdf_resolution=32`. Lower values are useful for quick smoke tests.

## Iso-level

`--iso 0.0` (default) extracts the surface itself. Negative values pull the iso-surface inward (eroded mesh); positive values push it outward (dilated mesh). Useful for visualising distance-band thickness when debugging.
