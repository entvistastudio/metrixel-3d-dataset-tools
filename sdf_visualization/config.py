# *******************************************************************************
#  *                                                                             *
#  *  Metrixel - Cross-platform Application                                      *
#  *                                                                             *
#  *  Copyright © 2026 EntertainmentVista Studio Pte. Ltd.                       *
#  *  All rights reserved.                                                       *
#  *                                                                             *
# *******************************************************************************

import os


def get_config(sdf_dataset_root: str) -> dict:
    """Validate and normalise the SDF dataset root path."""
    if sdf_dataset_root is None or not sdf_dataset_root.strip():
        raise ValueError("--sdf_dataset_root cannot be empty")
    sdf_dataset_path = os.path.normpath(sdf_dataset_root)
    print(f"SDF dataset root: {sdf_dataset_path}")
    return {"SDF_DATASET_PATH": sdf_dataset_path}
