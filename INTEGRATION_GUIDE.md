# Mesh Visualization Integration Guide

This document explains how the mesh visualization tools integrate with Metrixel and how they can be bundled or used independently.

---

## Directory Structure

metrixel/
└── tools/
    └── mesh_visualization/
        ├── visualize_mesh_metrixel.py
        ├── animate_mesh_metrixel.py
        ├── data_loader.py
        ├── config.py
        ├── requirements.txt
        ├── setup.sh
        ├── setup.bat
        ├── README.md
        └── INTEGRATION_GUIDE.md

---

## Data Format

Each frame contains:

• vert_*.pt → vertex tensor  
• face_*.pt → face indices  

Example:

generated/
└── meshes/
    └── 000001/
        ├── vert_000_000_000_0000.pt
        ├── face_000_000_000_0000.pt

---

## Command Line Usage

Single frame viewer:

python visualize_mesh_metrixel.py --mesh_dataset_root /path/to/generated/meshes --seq 000001 --frame 000_000_000_0000

Animation viewer:

python animate_mesh_metrixel.py --mesh_dataset_root /path/to/generated/meshes --seq 000001 --frame_begin 000_000_000_0000 --frame_end 000_000_000_0134 --fps 30

---

## Dependencies

Required Python packages:

• torch  
• matplotlib  
• numpy  

Install:

pip install -r requirements.txt

---

## Integration Options

1. Bundled with Metrixel desktop application  
2. Standalone CLI utilities for researchers  
3. Future cloud visualization option

---

## Testing

Recommended tests:

• Validate `.pt` file loading  
• Test animation playback  
• Test large mesh sequences  
• Cross‑platform testing (Windows / macOS / Linux)
