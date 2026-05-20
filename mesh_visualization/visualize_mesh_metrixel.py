# *******************************************************************************
#  *                                                                             *
#  *  Metrixel - Cross-platform Application                                      *
#  *                                                                             *
#  *  Copyright © 2025 EntertainmentVista Studio Pte. Ltd.                       *
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

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from data_loader import load_mesh_from_filename
from config import get_config


def normalize_frame_name(frame_name: str) -> str:
    """Strip optional 'vert_'/'face_' prefix and '.pt' suffix so paths are correct."""
    s = frame_name.strip()
    if s.endswith(".pt"):
        s = s[:-3]
    for prefix in ("vert_", "face_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return s


def visualize_mesh_metrixel(sequence_id: str, frame_name: str, mesh_dataset_root: str):
    """Visualize mesh using Metrixel coordinate system:
    X -> Right, Y -> Up, Z -> Toward (camera/viewer)
    """
    config = get_config(mesh_dataset_root)
    mesh_dataset_path = config["MESH_DATASET_PATH"]

    frame_name = normalize_frame_name(frame_name)
    mesh_dir = os.path.join(mesh_dataset_path, sequence_id)
    vert_file = os.path.join(mesh_dir, f"vert_{frame_name}.pt")
    face_file = os.path.join(mesh_dir, f"face_{frame_name}.pt")

    # Load mesh
    vertices, faces = load_mesh_from_filename(vert_file, face_file)

    # Convert to numpy
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()

    # Transform coordinates to Metrixel convention:
    # Original matplotlib: X=horizontal, Y=diagonal, Z=vertical
    # Metrixel: X=right, Y=up, Z=toward
    # So we map: X->X, Y->Z, Z->Y
    vertices_transformed = vertices_np.copy()
    vertices_transformed[:, [1, 2]] = vertices_np[:, [2, 1]]  # Swap Y and Z

    # Plot mesh
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(vertices_transformed[faces_np], alpha=0.7)
    mesh.set_edgecolor("k")
    mesh.set_facecolor("lightblue")
    ax.add_collection3d(mesh)

    # Same unit for X, Y, Z (match animate_mesh_metrixel.py): use np on a numpy array to avoid env issues
    v = np.asarray(vertices_transformed)
    min_c = np.min(v, axis=0)
    max_c = np.max(v, axis=0)
    center = (min_c + max_c) / 2
    max_range = float(np.max(max_c - min_c))
    if max_range <= 0:
        max_range = 1.0
    half = max_range / 2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # Set labels according to Metrixel convention
    ax.set_xlabel("X (Right)")
    ax.set_ylabel("Y (Up)")
    ax.set_zlabel("Z (Toward)")

    # Set viewing angle to better show the Metrixel orientation
    ax.view_init(elev=0, azim=90)  # Front view: looking along Z-axis toward the mesh

    plt.title(f"Metrixel Mesh Visualization: {sequence_id}/{frame_name}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # python visualize_mesh_metrixel.py --mesh_dataset_root /path/to/meshes --seq 000001 --frame 000_000_000_0000
    parser = argparse.ArgumentParser(
        description="Visualize 3D mesh using Metrixel coordinate system (X=Right, Y=Up, Z=Toward)."
    )
    parser.add_argument(
        "--mesh_dataset_root",
        type=str,
        required=True,
        help="Path to mesh dataset root (folder containing sequence IDs, e.g. .../generated/meshes)",
    )
    parser.add_argument(
        "--seq", type=str, required=True, help="Sequence ID (e.g., 000001)"
    )
    parser.add_argument(
        "--frame", type=str, required=True, help="Frame name (e.g., 000_000_000_0000)"
    )

    args = parser.parse_args()
    visualize_mesh_metrixel(args.seq, args.frame, args.mesh_dataset_root)
