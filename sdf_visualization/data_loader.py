# *******************************************************************************
#  *                                                                             *
#  *  Metrixel - Cross-platform Application                                      *
#  *                                                                             *
#  *  Copyright © 2025 EntertainmentVista Studio Pte. Ltd.                       *
#  *  All rights reserved.                                                       *
#  *                                                                             *
# *******************************************************************************

import os
import numpy as np


def load_sdf_bin(path: str, resolution: int) -> np.ndarray:
    """Load a Metrixel SDF .bin file.

    File format: raw float32, R**3 values, written in (z, y, x) row-major order
    by SdfGenerator (idx = (z * R + y) * R + x). Negative inside, positive
    outside, zero on the surface.

    Returns a 3D ndarray indexed as volume[z, y, x].
    """
    raw = np.fromfile(path, dtype=np.float32)
    expected = resolution ** 3
    if raw.size != expected:
        raise ValueError(
            f"SDF file {path} has {raw.size} float32 values, expected {expected} "
            f"(R={resolution}). Pass --sdf_resolution matching what Metrixel "
            f"used to generate this volume."
        )
    return raw.reshape(resolution, resolution, resolution)


def normalize_frame_name(frame_name: str) -> str:
    """Strip optional 'sdf_' prefix and '.bin' suffix so paths resolve correctly."""
    s = frame_name.strip()
    if s.endswith(".bin"):
        s = s[:-4]
    if s.startswith("sdf_"):
        s = s[len("sdf_"):]
    return s


def resolve_sdf_path(root: str, seq: str, frame: str) -> str:
    """Find the SDF .bin under <root>/<seq>/.

    Tries Metrixel's native nested layout first ({root}/{seq}/sdf/sdf_{frame}.bin),
    then falls back to a flattened layout ({root}/{seq}/sdf_{frame}.bin) for
    users who post-process or copy files around.
    """
    candidates = [
        os.path.join(root, seq, "sdf", f"sdf_{frame}.bin"),
        os.path.join(root, seq, f"sdf_{frame}.bin"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "SDF file not found. Tried:\n  " + "\n  ".join(candidates)
    )
