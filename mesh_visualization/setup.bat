@echo off
REM *******************************************************************************
REM  *                                                                             *
REM  *  Metrixel - Cross-platform Application                                      *
REM  *                                                                             *
REM  *  Copyright (c) 2025 EntertainmentVista Studio Pte. Ltd.                     *
REM  *  All rights reserved.                                                       *
REM  *                                                                             *
REM  *  Metrixel is a custom-built C/C++ application designed to read 3D           *
REM  *  models in FBX format, with optional support for USD files. The application *
REM  *  processes these inputs to generate a "Unified Data Representation" which   *
REM  *  includes rendered images, textures, animations, and metadata for use in    *
REM  *  training and evaluation.                                                   *
REM  *                                                                             *
REM  *  The above copyright notice and this permission notice shall be included in *
REM  *  all copies or substantial portions of the Software.                        *
REM  *                                                                             *
REM  *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
REM  *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
REM  *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
REM  *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
REM  *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
REM  *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
REM  *  DEALINGS IN THE SOFTWARE.                                                  *
REM  *                                                                             *
REM *******************************************************************************
REM
REM Metrixel Mesh Visualization Tools - Setup Script (Windows)
REM This script installs the required Python packages for mesh visualization

echo 🎬 Metrixel Mesh Visualization Tools Setup
echo ==============================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not installed. Please install pip first.
    pause
    exit /b 1
)

echo ✅ pip found
pip --version

REM Install requirements
echo 📦 Installing required packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Installation failed. Please check the error messages above.
    pause
    exit /b 1
) else (
    echo ✅ Installation completed successfully!
    echo.
    echo 🚀 Quick Start:
    echo    # View single frame:
    echo    python visualize_mesh_metrixel.py --seq 000001 --frame 000_000_000_0000
    echo.
    echo    # Animate sequence:
    echo    python animate_mesh_metrixel.py --seq 000001 --frame_begin 000_000_000_0000 --frame_end 000_000_000_0013 --fps 30
    echo.
    echo 📖 See README.md for detailed documentation
    pause
)
