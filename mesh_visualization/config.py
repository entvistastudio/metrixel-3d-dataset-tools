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


def get_config(mesh_dataset_root):
    """
    Get configuration with the specified mesh dataset root directory.

    Args:
        mesh_dataset_root (str): Path to the mesh dataset root (contains sequence
            folders; typically .../generated/meshes under a dataset root).

    Returns:
        dict: Configuration dictionary with MESH_DATASET_PATH.
    """
    if mesh_dataset_root is None or not mesh_dataset_root.strip():
        raise ValueError("⚠️ Mesh dataset root parameter cannot be empty!")

    mesh_dataset_path = os.path.normpath(mesh_dataset_root)

    config = {"MESH_DATASET_PATH": mesh_dataset_path}

    print(f"✅ Configuration loaded. Mesh dataset root: {mesh_dataset_path}")

    return config
