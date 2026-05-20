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
from matplotlib.animation import FuncAnimation
from data_loader import load_mesh_from_filename
from config import get_config


def normalize_frame_name(frame_str: str) -> str:
    """Strip optional 'vert_'/'face_' prefix and '.pt' suffix."""
    s = frame_str.strip()
    if s.endswith(".pt"):
        s = s[:-3]
    for prefix in ("vert_", "face_"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return s


def parse_frame_number(frame_str):
    """Parse frame string like '000_000_000_0000' to integer"""
    frame_str = normalize_frame_name(frame_str)
    parts = frame_str.split("_")
    if len(parts) == 4:
        return int(parts[3])  # Last part is the frame number
    return int(frame_str)


def format_frame_number(frame_num, padding=4):
    """Format frame number back to string like '000_000_000_0000'"""
    return f"000_000_000_{frame_num:0{padding}d}"


def animate_mesh_metrixel(
    sequence_id: str,
    frame_begin: str,
    frame_end: str,
    mesh_dataset_root: str,
    fps: int = 30,
):
    """Animate mesh sequence using Metrixel coordinate system:
    X -> Right, Y -> Up, Z -> Toward (camera/viewer)
    """
    config = get_config(mesh_dataset_root)
    mesh_dataset_path = config["MESH_DATASET_PATH"]

    mesh_dir = os.path.join(mesh_dataset_path, sequence_id)

    # Parse frame numbers
    begin_num = parse_frame_number(frame_begin)
    end_num = parse_frame_number(frame_end)

    print(f"🎬 Creating animation:")
    print(f"   Sequence: {sequence_id}")
    print(f"   Frames: {frame_begin} to {frame_end} ({end_num - begin_num + 1} frames)")
    print(f"   FPS: {fps}")
    print(f"   Duration: {(end_num - begin_num + 1) / fps:.2f} seconds")

    # Load all meshes
    meshes_data = []
    valid_frames = []

    for frame_num in range(begin_num, end_num + 1):
        frame_name = format_frame_number(frame_num)
        vert_file = os.path.join(mesh_dir, f"vert_{frame_name}.pt")
        face_file = os.path.join(mesh_dir, f"face_{frame_name}.pt")

        if os.path.exists(vert_file) and os.path.exists(face_file):
            try:
                vertices, faces = load_mesh_from_filename(vert_file, face_file)
                vertices_np = vertices.numpy()
                faces_np = faces.numpy()

                # Transform to Metrixel coordinates
                vertices_transformed = vertices_np.copy()
                vertices_transformed[:, [1, 2]] = vertices_np[:, [2, 1]]  # Swap Y and Z

                meshes_data.append((vertices_transformed, faces_np))
                valid_frames.append(frame_name)
                print(f"   ✅ Loaded frame {frame_name}")
            except Exception as e:
                print(f"   ❌ Failed to load frame {frame_name}: {e}")
        else:
            print(f"   ⚠️  Missing files for frame {frame_name}")

    if not meshes_data:
        print("❌ No valid mesh frames found!")
        return

    print(f"📊 Loaded {len(meshes_data)} frames successfully")

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Initialize empty mesh collection
    mesh_collection = Poly3DCollection([], alpha=0.7)
    mesh_collection.set_edgecolor("k")
    mesh_collection.set_facecolor("lightblue")
    ax.add_collection3d(mesh_collection)

    # Set up the plot
    ax.set_xlabel("X (Right)")
    ax.set_ylabel("Y (Up)")
    ax.set_zlabel("Z (Toward)")
    ax.view_init(elev=0, azim=90)  # Front view: looking along Z-axis toward the mesh

    # Same unit for X, Y, Z: use the same axis range so 1 unit has the same length on all axes
    all_vertices = np.concatenate([mesh[0] for mesh in meshes_data])
    min_coords = all_vertices.min(axis=0)
    max_coords = all_vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    max_range = float((max_coords - min_coords).max())
    if max_range <= 0:
        max_range = 1.0
    half = max_range / 2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # Set initial title so tight_layout() reserves space (same as visualize_mesh_metrixel.py)
    plt.title(
        f"Metrixel Animation: {sequence_id}/{valid_frames[0]} (1/{len(meshes_data)})"
    )

    # Animation function
    def animate(frame_idx):
        if frame_idx < len(meshes_data):
            vertices, faces = meshes_data[frame_idx]
            frame_name = valid_frames[frame_idx]

            # Update mesh data
            mesh_collection.set_verts(vertices[faces])

            # Update title
            plt.title(
                f"Metrixel Animation: {sequence_id}/{frame_name} ({frame_idx+1}/{len(meshes_data)})"
            )

            print(f"🎬 Frame {frame_idx+1}/{len(meshes_data)}: {frame_name}")

        return [mesh_collection]

    # Create animation
    print("🎬 Creating animation...")
    anim = FuncAnimation(
        fig,
        animate,
        frames=len(meshes_data),
        interval=1000 / fps,  # Convert FPS to milliseconds
        blit=False,
        repeat=True,
    )

    # Same layout as visualize_mesh_metrixel.py so title has the same vertical position
    plt.tight_layout()
    print("🎬 Animation ready! Close the window to stop.")
    plt.show()

    return anim


if __name__ == "__main__":
    # python animate_mesh_metrixel.py --mesh_dataset_root /path/to/meshes --seq 000001 --frame_begin 000_000_000_0000 --frame_end 000_000_000_0134 --fps 30
    parser = argparse.ArgumentParser(
        description="Animate 3D mesh sequence using Metrixel coordinate system."
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
        "--frame_begin",
        type=str,
        required=True,
        help="Starting frame (e.g., 000_000_000_0000)",
    )
    parser.add_argument(
        "--frame_end",
        type=str,
        required=True,
        help="Ending frame (e.g., 000_000_000_0134)",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )

    args = parser.parse_args()
    animate_mesh_metrixel(
        args.seq, args.frame_begin, args.frame_end, args.mesh_dataset_root, args.fps
    )
