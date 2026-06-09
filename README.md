# LiveMesh

**Autonomous in-situ bioprinting: from camera frame to deposited biomaterial.**

LiveMesh is a unified pipeline that sees a wound, reconstructs its 3D surface, computes geodesic deposition paths, and drives a robot to print biomaterial directly on curved, living tissue. It replaces the planar-slicing assumption in conventional bioprinting with geometry-aware conformal toolpaths.

Developed as part of the doctoral thesis *"CNN-Based ML for 3D Motion Planning and Control in In-Situ Robotic Bioprinters"* at [Tecnologico de Monterrey](https://tec.mx), in collaboration with the [Geometric Data Processing Group](https://geometry.csail.mit.edu/) at MIT CSAIL (Prof. Justin Solomon).

---

## Pipeline

```
RGB camera                     Depth camera
    |                               |
    v                               v
[PERCEIVE]                    [RECONSTRUCT]
CNN-Transformer               Point cloud -->
+ PolarDecoder                 Poisson / DeepCurrents
--> wound boundary             --> smooth mesh
                    \         /
                     v       v
                    [PLAN]
                    Geodesic toolpaths on mesh
                    (heat method, curvature-adaptive)
                    + OT-based coverage validation
                         |
                         v
                    [EXECUTE]
                    Conformal UV-to-XYZ mapping
                    --> G-code / ROS2 / 8-DOF IK
                         |
                         v
                    [FEEDBACK]
                    CNN visual controller
                    --> mesh update --> re-plan
```

## Modules

| Module | What it does | Origin |
|--------|-------------|--------|
| `livemesh.perception` | CNN-Transformer encoder (ResNet-50 + 6-layer Transformer) with 3 decoder heads: **Polar** (thesis contribution), DETR, Autoregressive | Thesis |
| `livemesh.segmentation` | U-Net wound segmentation + mask-to-boundary extraction | WoundPath-AI |
| `livemesh.reconstruction` | Poisson surface reconstruction from point clouds, Hausdorff/normal benchmarks | MIT collaboration |
| `livemesh.toolpath` | Geodesic paths (heat method), planar slicer baseline, honeycomb generator, TSP path optimization, conformal UV-to-XYZ mapping, OT-based coverage metrics, G-code/ROS2 export | MIT + Thesis |
| `livemesh.robot` | 8-DOF robot model (2 prismatic gantry + 6R myCobot), 3-phase IK solver (L-BFGS-B + null-space + APF+Super-Twisting), STL scaffold analysis | Thesis |
| `livemesh.training` | Training loop, evaluation, 3-decoder ablation study | Thesis |
| `livemesh.data` | FUSeg wound dataset, synthetic 2D wounds, synthetic 3D surfaces (sphere, saddle, crater, cylinder), multi-view volumetric dataset, polar conversion | Both |
| `livemesh.pipeline` | End-to-end orchestrator with per-stage timing and benchmarking | New |

## Quick start

```bash
git clone https://github.com/dianisay/livemesh.git
cd livemesh
pip install -e ".[dev]"
pytest
```

### Run the full pipeline on synthetic data

```python
from livemesh.pipeline import LiveMeshPipeline, PipelineConfig
from livemesh.data.synthetic import sphere_cap, add_noise

# Configure
config = PipelineConfig(
    perception_model="unet",
    reconstruction_method="poisson",
    toolpath_method="geodesic",
    robot_type="gcode",
)
pipeline = LiveMeshPipeline(config)

# Generate test surface
gt_mesh = sphere_cap(radius=50.0, cap_angle_deg=45.0)
depth_points = add_noise(gt_mesh, sigma=0.5)

# Process (skip perception for synthetic test)
result = pipeline.process_frame(depth_points=depth_points)
print(f"Reconstruction: {result['reconstruction']['elapsed_ms']:.1f} ms")
print(f"Toolpath: {result['toolpath']['num_paths']} geodesic paths")
print(f"G-code: {result['robot_commands']['num_lines']} lines")
```

### Train the polar decoder (ablation study)

```bash
python -m livemesh.training.ablation \
    --fuseg-dir data/fuseg \
    --synthetic-dir data/synthetic \
    --epochs 100
```

### Benchmark geodesic vs planar toolpaths

```python
from livemesh.data.synthetic import sphere_cap
from livemesh.toolpath import geodesic_toolpaths, planar_slice, coverage_uniformity

mesh = sphere_cap(radius=50.0, cap_angle_deg=60.0, resolution=64)

geo = geodesic_toolpaths(mesh, spacing_mm=2.0)
pla = planar_slice(mesh, line_spacing_mm=2.0)

geo_cov = coverage_uniformity(mesh, geo.waypoints, nozzle_width_mm=1.5)
pla_cov = coverage_uniformity(mesh, pla.waypoints, nozzle_width_mm=1.5)

print(f"Geodesic coverage: {geo_cov.coverage_fraction:.1%}")
print(f"Planar coverage:   {pla_cov.coverage_fraction:.1%}")
```

## Project structure

```
livemesh/
├── src/livemesh/
│   ├── perception/          # CNN-Transformer + decoders
│   ├── segmentation/        # U-Net wound segmentation
│   ├── reconstruction/      # Point cloud --> mesh
│   ├── toolpath/            # Geodesic, planar, honeycomb, conformal, coverage, G-code
│   ├── robot/               # 8-DOF model, IK solver, STL analysis
│   ├── training/            # Train, evaluate, ablation
│   ├── data/                # Datasets + synthetic generators
│   ├── pipeline/            # End-to-end orchestrator
│   ├── utils/               # Metrics, visualization
│   └── visualization/       # 3D rendering
├── tests/
├── notebooks/               # Ablation studies, demos
├── configs/
├── assets/meshes/
├── docs/
└── paper/
```

## Key results

| Decoder | Chamfer (mm) | Hausdorff (mm) | IoU | Closure (mm) | Ordering |
|---------|-------------|----------------|-----|-------------|----------|
| DETR | 4.72 | 12.41 | 0.71 | 8.34 | 23.1% |
| Autoregressive | 3.18 | 8.67 | 0.79 | 3.52 | 81.4% |
| **Polar (ours)** | **2.31** | **5.14** | **0.91** | **0.00** | **100%** |

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA-capable GPU (training)
- Full list in `pyproject.toml`

## Related repositories

- [WoundPath-AI](https://github.com/dianisay/WoundPath-AI-UNet-for-Multi-Skin-Tone-Wound-Segmentation-and-GCode-Generation) (MATLAB, wound segmentation origin)
- [Conformal-Trajectory](https://github.com/dianisay/Conformal-Trajectory) (MATLAB, conformal mapping origin)

## License

MIT
