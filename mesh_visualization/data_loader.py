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
import torch
import numpy as np


def load_tensor_raw(filename):
    """Load tensor data from raw binary file format"""
    with open(filename, "rb") as f:
        # Read the rank (number of dimensions)
        num_dims = np.fromfile(f, dtype=np.int64, count=1)[0]

        # Read the shape
        shape = tuple(np.fromfile(f, dtype=np.int64, count=num_dims))

        # Read dtype flag
        dtype_flag = np.fromfile(f, dtype=np.int32, count=1)[0]

        # Determine dtype
        if dtype_flag == 1:
            dtype = np.float32
        elif dtype_flag == 2:
            dtype = np.int64
        else:
            raise ValueError("Unknown dtype flag in file:", dtype_flag)

        # Read tensor data
        tensor_data = np.fromfile(f, dtype=dtype).reshape(shape)

        return torch.tensor(tensor_data)


def load_mesh_from_filename(vert_file, face_file):
    """Load mesh from given vertex and face file paths"""
    if not os.path.exists(vert_file) or not os.path.exists(face_file):
        raise FileNotFoundError(f"❌ Missing files: {face_file} or {vert_file}")

    vertices = load_tensor_raw(vert_file)
    faces = load_tensor_raw(face_file)

    return vertices, faces
