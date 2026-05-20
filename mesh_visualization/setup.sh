#!/bin/bash
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
#
# Metrixel Mesh Visualization Tools - Setup Script
# This script installs the required Python packages for mesh visualization

echo "🎬 Metrixel Mesh Visualization Tools Setup"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Install requirements
echo "📦 Installing required packages..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Installation completed successfully!"
    echo ""
    echo "🚀 Quick Start:"
    echo "   # View single frame:"
    echo "   python3 visualize_mesh_metrixel.py --seq 000001 --frame 000_000_000_0000"
    echo ""
    echo "   # Animate sequence:"
    echo "   python3 animate_mesh_metrixel.py --seq 000001 --frame_begin 000_000_000_0000 --frame_end 000_000_000_0013 --fps 30"
    echo ""
    echo "📖 See README.md for detailed documentation"
else
    echo "❌ Installation failed. Please check the error messages above."
    exit 1
fi
