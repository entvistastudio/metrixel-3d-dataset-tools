# *******************************************************************************
#  *                                                                             *
#  *  Metrixel - Cross-platform Application                                      *
#  *                                                                             *
#  *  Copyright © 2026 EntertainmentVista Studio Pte. Ltd.                       *
#  *  All rights reserved.                                                       *
#  *                                                                             *
#  *  Metrixel is a custom-built C/C++ application designed to read 3D           *
#  *  models in FBX format, with optional support for USD files. The application *
#  *  processes these inputs to generate a "Unified Data Representation" which   *
#  *  includes rendered images, textures, animations, and metadata for use in    *
#  *  training and evaluation.                                                   *
#  *                                                                             *
#  *  The above copyright notice and this permission notice shall be included in *
#  *  all copies or substantial portions of the Software.                        *
#  *                                                                             *
#  *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
#  *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
#  *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
#  *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
#  *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
#  *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
#  *  DEALINGS IN THE SOFTWARE.                                                  *
#  *                                                                             *
# *******************************************************************************


"""Validate Metrixel ``.pt`` mesh exports — structure, not appearance.

Metrixel writes vertices, normals, UVs and face indices as raw tensors with a small self-describing
header. This tool checks that what landed on disk is actually loadable and internally consistent, so a
bad export is caught before it reaches a dataloader rather than after a training run.

**numpy only.** No torch, no matplotlib, no display. That is deliberate: this is the one tool in the
set that has to run on a CI runner, inside a container, or over SSH on a headless training box — the
places where an export is most likely to go wrong and least likely to be looked at.

File layout it expects (written by Metrixel):

    <output>/assets/<id>/meshes/vert_<viewX>_<viewY>_<viewZ>_<frame>.pt
                               norm_...  uv_...  face_...

Usage::

    python validate_mesh_metrixel.py <path>            # a meshes/ dir, or any dir above it
    python validate_mesh_metrixel.py <path> --json     # machine-readable, for CI
    python validate_mesh_metrixel.py <path> --quiet    # exit code only

Exit codes: 0 = all frames valid · 1 = at least one problem · 2 = nothing found / usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# Header layout written by Metrixel's SaveTensorRaw:
#   int64  rank
#   int64  shape[rank]
#   int32  dtype flag   (1 = float32, 2 = int64)
#   raw payload
DTYPE_BY_FLAG = {1: np.float32, 2: np.int64}
KINDS = ("vert", "norm", "uv", "face")
# vert_000_000_000_0000.pt -> kind, view triplet, frame
NAME_RE = re.compile(r"^(vert|norm|uv|face)_(\d+)_(\d+)_(\d+)_(\d+)\.pt$")


class Problem(Exception):
    """A validation failure with a human-readable reason."""


def read_header(path):
    """Return (shape, dtype, header_bytes) without loading the payload."""
    with open(path, "rb") as f:
        raw_rank = f.read(8)
        if len(raw_rank) < 8:
            raise Problem("file is shorter than the 8-byte rank field — truncated or not a .pt export")
        rank = int(np.frombuffer(raw_rank, dtype=np.int64, count=1)[0])
        if not 1 <= rank <= 4:
            raise Problem(f"implausible tensor rank {rank} — header is not a Metrixel raw tensor")
        raw_shape = f.read(8 * rank)
        if len(raw_shape) < 8 * rank:
            raise Problem("truncated inside the shape field")
        shape = tuple(int(v) for v in np.frombuffer(raw_shape, dtype=np.int64, count=rank))
        if any(d < 0 for d in shape):
            raise Problem(f"negative dimension in shape {shape}")
        raw_flag = f.read(4)
        if len(raw_flag) < 4:
            raise Problem("truncated before the dtype flag")
        flag = int(np.frombuffer(raw_flag, dtype=np.int32, count=1)[0])
        if flag not in DTYPE_BY_FLAG:
            raise Problem(f"unknown dtype flag {flag} (expected 1=float32 or 2=int64)")
        return shape, DTYPE_BY_FLAG[flag], 8 + 8 * rank + 4


def load(path):
    """Load one tensor, verifying the declared size matches the bytes on disk.

    The size check is the one that matters most: a truncated write still has a valid header, and
    numpy would happily reshape a short buffer into silent garbage.
    """
    shape, dtype, header_bytes = read_header(path)
    expected = header_bytes + int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual = os.path.getsize(path)
    if actual != expected:
        raise Problem(
            f"size mismatch: header declares {shape} {np.dtype(dtype).name} "
            f"= {expected} bytes, file is {actual} — truncated or trailing data")
    with open(path, "rb") as f:
        f.seek(header_bytes)
        return np.frombuffer(f.read(), dtype=dtype).reshape(shape)


def check_frame(files):
    """Validate one frame's set of tensors. Returns a list of problem strings."""
    problems, data = [], {}
    for kind, path in sorted(files.items()):
        try:
            data[kind] = load(path)
        except Problem as exc:
            problems.append(f"{os.path.basename(path)}: {exc}")
        except OSError as exc:
            problems.append(f"{os.path.basename(path)}: unreadable ({exc})")

    for kind in KINDS:
        if kind not in files:
            problems.append(f"missing {kind}_*.pt for this frame")

    verts = data.get("vert")
    if verts is not None:
        if verts.ndim != 2 or verts.shape[1] != 3:
            problems.append(f"vert: expected (N, 3), got {verts.shape}")
        elif verts.shape[0] == 0:
            problems.append("vert: zero vertices")
        else:
            if not np.isfinite(verts).all():
                problems.append(f"vert: {int((~np.isfinite(verts)).sum())} non-finite values (NaN/Inf)")
            else:
                extent = verts.max(axis=0) - verts.min(axis=0)
                if not (extent > 0).any():
                    problems.append("vert: all vertices coincide — degenerate geometry")

    faces = data.get("face")
    if faces is not None:
        if faces.ndim != 2 or faces.shape[1] != 3:
            problems.append(f"face: expected (M, 3), got {faces.shape}")
        elif faces.shape[0] == 0:
            problems.append("face: zero faces")
        elif verts is not None and verts.ndim == 2:
            # The check that actually catches corruption: an index outside the vertex
            # array is a crash or silent garbage in any downstream loader.
            lo, hi = int(faces.min()), int(faces.max())
            if lo < 0 or hi >= verts.shape[0]:
                problems.append(
                    f"face: index range [{lo}, {hi}] outside the {verts.shape[0]} vertices present")

    # norm and uv are per-vertex; a mismatch means the frame's tensors disagree with each other.
    if verts is not None and verts.ndim == 2:
        n = verts.shape[0]
        for kind, cols in (("norm", 3), ("uv", 2)):
            arr = data.get(kind)
            if arr is None:
                continue
            if arr.ndim != 2 or arr.shape[0] != n:
                problems.append(f"{kind}: expected {n} rows to match vert, got {arr.shape}")
            elif arr.shape[1] != cols:
                problems.append(f"{kind}: expected {cols} columns, got {arr.shape[1]}")
            elif not np.isfinite(arr).all():
                problems.append(f"{kind}: non-finite values")

    return problems, (verts.shape[0] if verts is not None and verts.ndim == 2 else None)


def collect(root):
    """Group every .pt under @p root into {(sequence_dir, view, frame): {kind: path}}."""
    frames = defaultdict(dict)
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            m = NAME_RE.match(name)
            if m:
                kind, vx, vy, vz, frame = m.groups()
                frames[(dirpath, f"{vx}_{vy}_{vz}", frame)][kind] = os.path.join(dirpath, name)
    return frames


def main():
    ap = argparse.ArgumentParser(
        description="Validate Metrixel .pt mesh exports (numpy only — safe on headless hosts).")
    ap.add_argument("path", help="a meshes/ directory, or any directory above one")
    ap.add_argument("--json", action="store_true", help="machine-readable output for CI")
    ap.add_argument("--quiet", "-q", action="store_true", help="no output; rely on the exit code")
    args = ap.parse_args()

    if not os.path.isdir(args.path):
        print(f"not a directory: {args.path}", file=sys.stderr)
        return 2

    frames = collect(args.path)
    if not frames:
        print(f"no Metrixel .pt mesh files found under {args.path}\n"
              f"expected names like vert_000_000_000_0000.pt", file=sys.stderr)
        return 2

    results, bad = [], 0
    for (dirpath, view, frame), files in sorted(frames.items()):
        problems, vert_count = check_frame(files)
        if problems:
            bad += 1
        results.append({
            "directory": os.path.relpath(dirpath, args.path),
            "view": view,
            "frame": frame,
            "vertices": vert_count,
            "ok": not problems,
            "problems": problems,
        })

    # A dataset whose frames disagree on vertex count is not necessarily wrong — Metrixel exports
    # per frame — but it is worth surfacing, because a dataloader that assumes a fixed topology
    # will fail on the frame that differs rather than at load time.
    counts = {r["vertices"] for r in results if r["vertices"] is not None}
    inconsistent = len(counts) > 1

    if args.json:
        print(json.dumps({
            "root": args.path,
            "frames": len(results),
            "failed": bad,
            "vertex_counts": sorted(counts),
            "topology_varies_between_frames": inconsistent,
            "results": results,
        }, indent=2))
    elif not args.quiet:
        for r in results:
            if r["ok"]:
                print(f"  ok    {r['directory']}/frame {r['frame']}  ({r['vertices']} verts)")
            else:
                print(f"  FAIL  {r['directory']}/frame {r['frame']}")
                for p in r["problems"]:
                    print(f"          {p}")
        print(f"\n{len(results)} frame(s) checked, {bad} with problems")
        if inconsistent:
            print(f"note: vertex count varies between frames {sorted(counts)} — expected for animated "
                  f"exports with changing topology, a problem if your loader assumes it is fixed")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
