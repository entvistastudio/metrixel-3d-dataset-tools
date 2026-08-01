# Mesh Validation (`mesh_validation/`)

Validate Metrixel `.pt` mesh exports — **structure, not appearance**.

Metrixel writes vertices, normals, UVs and face indices as raw tensors with a small self-describing
header. This tool checks that what landed on disk is loadable and internally consistent, so a bad
export is caught before it reaches a dataloader instead of after a training run.

## Why this is separate from `mesh_visualization/`

`mesh_visualization` needs **torch** (to return tensors) and **matplotlib** (to draw), and drawing
needs a display. This tool needs **numpy and nothing else**.

That is the point. Exports go wrong on CI runners, inside containers, and on SSH-only training hosts
— exactly the machines with no display and no reason to have a ~2 GB torch install. On a headless
box `matplotlib`'s `plt.show()` selects the `agg` backend and returns **silently**: no error, no
window, nothing rendered. A validator that prints findings and sets an exit code works there.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python validate_mesh_metrixel.py <path>          # a meshes/ dir, or any dir above it
python validate_mesh_metrixel.py <path> --json   # machine-readable, for CI
python validate_mesh_metrixel.py <path> --quiet  # exit code only
```

`<path>` is searched recursively, so pointing it at a whole output root works:

```bash
python validate_mesh_metrixel.py /data/out
```

## What it checks

| Check | Catches |
|---|---|
| Header well-formed (rank, shape, dtype flag) | Not a Metrixel export; corrupt header |
| `filesize == header + prod(shape) × itemsize` | **Truncated writes** — a short file still has a valid header, and numpy would reshape it into silent garbage |
| `vert` is `(N, 3)` float32, all finite, non-degenerate bbox | NaN/Inf; collapsed geometry |
| `face` is `(M, 3)` int64 with every index in `[0, N)` | **Out-of-range indices** — a crash or silent garbage in any downstream loader |
| `norm` / `uv` row counts match `vert` | Tensors within one frame disagreeing |
| All four kinds present per frame | A partially written frame |
| Vertex count across frames | Reported, not failed — varying topology is legitimate for animated exports, but breaks a loader that assumes it is fixed |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All frames valid |
| `1` | At least one problem found |
| `2` | Nothing found, or usage error |

## Expected file layout

```
<output>/assets/<id>/meshes/vert_<viewX>_<viewY>_<viewZ>_<frame>.pt
                            norm_...   uv_...   face_...
```

e.g. `vert_000_000_000_0000.pt` — the three groups are the camera view angles, the last is the frame.

## Example

```
  ok    meshes/000001/frame 0000  (34858 verts)
  FAIL  meshes/000001/frame 0001
          vert_000_000_000_0001.pt: size mismatch: header declares (34858, 3) float32
          = 418324 bytes, file is 413324 — truncated or trailing data
  FAIL  meshes/000001/frame 0002
          vert: 1 non-finite values (NaN/Inf)
          face: index range [0, 999999] outside the 34858 vertices present

4 frame(s) checked, 3 with problems
```
