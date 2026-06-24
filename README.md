# LiveMesh

**Autonomous in-situ bioprinting: from camera frame to deposited biomaterial on living, curved tissue.**

LiveMesh is a unified Python pipeline that sees a wound, reconstructs its 3D surface in real time, computes geodesic deposition paths that follow the surface curvature, and drives a robot to print biomaterial directly on the patient. It replaces the flat-layer assumption in conventional bioprinting with geometry-aware conformal toolpaths, enabling printing on anatomy that is curved, irregular, and moving with respiration.

Developed as part of the doctoral thesis *"CNN-Based ML for 3D Motion Planning and Control in In-Situ Robotic Bioprinters"* at [Tecnologico de Monterrey](https://tec.mx), in collaboration with the [Geometric Data Processing Group](https://geometry.csail.mit.edu/) at MIT CSAIL (Prof. Justin Solomon).

> **Evolution:** LiveMesh extends [WoundBioprinter](https://github.com/dianisay/diana-bioprinting-pipeline) (Phase 1, Tec de Monterrey) with geodesic toolpath planning via [geodesic-currents](https://github.com/dianisay/geodesic-currents), Poisson surface reconstruction, and optimal-transport coverage metrics developed during the MIT collaboration.

---

## What's inside

```
Camera frame
     |
     v
 PERCEIVE ──────────── CNN-Transformer + PolarDecoder → wound boundary
     |                  U-Net → wound mask
     v
 RECONSTRUCT ────────── Depth point cloud → Poisson surface → smooth mesh
     |
     v
 PLAN ──────────────── Geodesic toolpaths on the mesh (heat method)
     |                  or honeycomb infill (hex grid + TSP)
     |                  coverage validation (optimal transport)
     v
 EXECUTE ───────────── G-code / ROS 2 trajectory / 8-DOF inverse kinematics
     |
     v
 FEEDBACK ──────────── CNN visual controller → mesh update → re-plan
     |
     └──── loop ────┘
```

---

## Installation

```bash
git clone https://github.com/dianisay/livemesh.git
cd livemesh
pip install -e ".[dev]"
```

For robot simulation (optional):
```bash
pip install -e ".[dev,robot]"
```

---

## Quick start: 30-second demo

```python
from livemesh.utils.logging_config import setup_logging
from livemesh.data.synthetic import sphere_cap, add_noise
from livemesh.reconstruction import poisson_reconstruct, benchmark_reconstruction
from livemesh.toolpath import geodesic_toolpaths, planar_slice, coverage_uniformity

setup_logging("INFO")  # see everything LiveMesh does

# 1. Create a curved wound surface (ground truth)
gt_mesh = sphere_cap(radius=50.0, cap_angle_deg=45.0, resolution=64)

# 2. Simulate what a depth camera would see (noisy point cloud)
points = add_noise(gt_mesh, sigma=0.5)

# 3. Reconstruct the surface from the noisy points
rec = poisson_reconstruct(points, depth=8)

# 4. Benchmark reconstruction quality against ground truth
bench = benchmark_reconstruction(rec.mesh, gt_mesh, elapsed_ms=rec.elapsed_ms)

# 5. Generate geodesic toolpaths on the reconstructed surface
geo = geodesic_toolpaths(rec.mesh, spacing_mm=2.0)

# 6. Compare coverage: geodesic vs planar (the old way)
pla = planar_slice(rec.mesh, line_spacing_mm=2.0)

geo_cov = coverage_uniformity(rec.mesh, geo.waypoints, nozzle_width_mm=1.5)
pla_cov = coverage_uniformity(rec.mesh, pla.waypoints, nozzle_width_mm=1.5)

print(f"\nReconstruction: {bench.hausdorff_mm:.2f} mm Hausdorff, {rec.elapsed_ms:.0f} ms")
print(f"Geodesic: {geo.num_paths} paths, {geo_cov.coverage_fraction:.0%} coverage")
print(f"Planar:   {pla.num_layers} layers, {pla_cov.coverage_fraction:.0%} coverage")
```

---

## Full pipeline: one call

```python
from livemesh.pipeline import LiveMeshPipeline, PipelineConfig
from livemesh.utils.logging_config import setup_logging

setup_logging("INFO")

config = PipelineConfig(
    perception_model="unet",       # or "polar"
    reconstruction_method="poisson",
    toolpath_method="geodesic",    # or "planar" or "honeycomb"
    robot_type="gcode",            # or "ros2" or "mycobot"
)

pipeline = LiveMeshPipeline(config)
# pipeline.load_models()  # uncomment when you have trained weights

# Process a frame (depth points only for now, add rgb_image when models are trained)
from livemesh.data.synthetic import sphere_cap, add_noise
depth_points = add_noise(sphere_cap(radius=50.0), sigma=0.5)

result = pipeline.process_frame(depth_points=depth_points)
print(result["reconstruction"]["elapsed_ms"], "ms reconstruction")
print(result["toolpath"]["num_paths"], "geodesic paths")
print(result["robot_commands"]["num_lines"], "G-code lines")
print(pipeline.timing_summary())
```

---

## Module-by-module guide

### `livemesh.perception` — See the wound

The CNN-Transformer encoder extracts 64 spatial tokens from a 256x256 RGB image. Three decoder heads compete in an ablation study:

| Decoder | How it works | Closure | Ordering |
|---------|-------------|---------|----------|
| **PolarDecoder** (ours) | Predicts centroid + 64 radii. Closed loop guaranteed by construction. | 0.00 mm | 100% |
| DETRDecoder | 64 learned queries, Hungarian matching. Unordered. | 8.34 mm | 23.1% |
| AutoregressiveDecoder | Predicts points one by one with causal masking. | 3.52 mm | 81.4% |

```python
from livemesh.perception import CNNTransformerEncoder, PolarDecoder

encoder = CNNTransformerEncoder(d_model=256, num_heads=8, num_layers=6, pretrained=True)
decoder = PolarDecoder(d_model=256, num_radii=64)

import torch
x = torch.randn(1, 3, 256, 256)  # RGB image
features = encoder(x)              # (1, 64, 256) spatial tokens
prediction = decoder(features)     # {centroid: (1,2), radii: (1,64), points: (1,64,2)}
```

### `livemesh.segmentation` — U-Net wound mask

Ported from MATLAB. U-Net with depth 4, 32 base filters, trained on FUSeg wound data.

```python
from livemesh.segmentation import UNet, segment_wound

model = UNet(in_channels=3, num_classes=2, base_filters=32, depth=4)
# model.load_state_dict(torch.load("checkpoint.pth"))

import numpy as np
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
result = segment_wound(image, model, device="cpu")
print(f"Wound area: {result.area_mm2:.1f} mm², boundary: {len(result.boundary)} points")
```

### `livemesh.reconstruction` — Build the mesh

Takes a noisy point cloud from a depth camera and builds a smooth, boundary-aware mesh.

```python
from livemesh.reconstruction import poisson_reconstruct, benchmark_reconstruction
from livemesh.data.synthetic import sphere_cap, add_noise

gt = sphere_cap(radius=50.0, resolution=64)
points = add_noise(gt, sigma=0.5)

rec = poisson_reconstruct(points, depth=8)
bench = benchmark_reconstruction(rec.mesh, gt, elapsed_ms=rec.elapsed_ms)

print(f"Vertices: {rec.num_output_vertices}")
print(f"Hausdorff: {bench.hausdorff_mm:.2f} mm")
print(f"Mean error: {bench.mean_distance_mm:.3f} mm")
print(f"Normal deviation: {bench.normal_deviation_deg:.1f} deg")
print(f"Time: {rec.elapsed_ms:.0f} ms")
```

### `livemesh.toolpath` — Plan the paths

Three toolpath strategies, all comparable through the same coverage metrics:

#### Geodesic (the new way, for Solomon's project)

```python
from livemesh.toolpath import geodesic_toolpaths, coverage_uniformity

result = geodesic_toolpaths(
    mesh,
    spacing_mm=2.0,           # distance between adjacent paths
    adaptive_curvature=True,  # tighter spacing on high-curvature regions
    curvature_factor=0.5,     # 0=uniform, 1=fully adaptive
)
print(f"{result.num_paths} paths, {result.total_length_mm:.0f} mm total, {result.elapsed_ms:.0f} ms")

cov = coverage_uniformity(mesh, result.waypoints, nozzle_width_mm=1.5)
print(f"Coverage: {cov.coverage_fraction:.0%}, uniformity: {cov.uniformity_score:.2f}")
print(f"Max gap: {cov.max_gap_mm:.1f} mm, Wasserstein: {cov.wasserstein_distance:.3f}")
```

#### Planar (the old way, baseline for comparison)

```python
from livemesh.toolpath import planar_slice

result = planar_slice(mesh, layer_height_mm=0.4, line_spacing_mm=1.5, direction="zigzag")
print(f"{result.num_layers} layers, {result.total_length_mm:.0f} mm")
```

#### Honeycomb (for scaffold printing)

```python
from livemesh.toolpath.honeycomb import compute_grid_params, create_hex_grid, hexagon_perimeter

nx, ny, hex_side = compute_grid_params(void_width=42.5, void_length=40.0)
grid = create_hex_grid(nx, ny, hex_side)
perimeter = hexagon_perimeter(center=grid[0, 0], hex_side=hex_side)
```

#### Coverage comparison

```python
from livemesh.toolpath.coverage import coverage_comparison_table

results = coverage_comparison_table(mesh, {
    "geodesic": geodesic_waypoints,
    "planar": planar_waypoints,
})
for name, cov in results.items():
    print(f"{name}: {cov.coverage_fraction:.0%} coverage, {cov.uniformity_score:.2f} uniformity")
```

#### Conformal mapping (UV to XYZ)

```python
from livemesh.toolpath.conformal_map import cylinder_conformal_map, general_conformal_map

# For cylindrical surfaces (your existing scaffold geometry)
mapped = cylinder_conformal_map(traj_uv, cyl_radius=48.6)
# mapped.xyz_mm, mapped.normals, mapped.tool_frames

# For arbitrary wound meshes (Solomon's extension)
mapped = general_conformal_map(traj_uv, mesh_vertices, mesh_faces, mesh_normals, uv_coords)
```

#### Export to robot

```python
from livemesh.toolpath.path_to_robot import toolpath_to_gcode, toolpath_to_ros2_trajectory

# G-code for CNC-style bioprinter
gcode = toolpath_to_gcode(
    waypoints, is_deposition=is_dep_mask,
    safe_z_mm=5.0, feed_rate=1500.0,
    output_path="output/wound_print.gcode",
)

# ROS 2 trajectory for robotic arm
traj = toolpath_to_ros2_trajectory(waypoints, normals, tool_frames, velocity_mm_s=25.0)
```

### `livemesh.robot` — Move the arm

8-DOF robot: 2 prismatic (XY gantry) + 6 revolute (myCobot 280).

```python
from livemesh.robot import forward_kinematics_8dof, InverseKinematicsSolver
import numpy as np

# Forward kinematics
q = np.zeros(8)  # home position
T = forward_kinematics_8dof(q)  # 4x4 homogeneous transform
print(f"TCP position: {T[:3,3]*1000} mm")

# Inverse kinematics (full trajectory)
solver = InverseKinematicsSolver()
result = solver.solve_trajectory(trajectory_m, R_targets)
print(f"Solved {len(result['q_solutions'])} poses, mean error: {np.mean(result['errors'])*1000:.2f} mm")
```

#### STL scaffold analysis

```python
from livemesh.robot.stl_analysis import analyze_scaffold

scaffold = analyze_scaffold("assets/meshes/scaffold_curved_void.stl")
print(f"Cylinder R={scaffold['cylinder']['radius']:.1f} mm")
print(f"Void: {scaffold['void_bounds']['void_width']:.1f} x {scaffold['void_bounds']['void_length']:.1f} mm")
```

### `livemesh.training` — Train and evaluate

```bash
# Train polar decoder
python -m livemesh.training.train --decoder polar --fuseg-dir data/fuseg --synthetic-dir data/synthetic --epochs 100

# Run full ablation (polar vs DETR vs autoregressive)
python -m livemesh.training.ablation --fuseg-dir data/fuseg --synthetic-dir data/synthetic --epochs 100

# Evaluate a checkpoint
python -m livemesh.training.evaluate --checkpoint results/polar/best.pth --fuseg-dir data/fuseg
```

From Python:

```python
from livemesh.training.train import Trainer
from livemesh.data.dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    fuseg_dir="data/fuseg", synthetic_dir="data/synthetic", batch_size=8,
)

trainer = Trainer(decoder_type="polar", max_epochs=100, patience=10)
history = trainer.train(train_loader, val_loader)
```

### `livemesh.data` — Datasets and synthetic generators

#### 3D surfaces (for reconstruction + geodesic testing)

```python
from livemesh.data.synthetic import sphere_cap, saddle_surface, wound_crater, cylinder_patch, flat_plane, add_noise, add_occlusion

mesh = wound_crater(outer_radius=40.0, depth=8.0, resolution=64)
noisy_points = add_noise(mesh, sigma=0.5)
occluded_points = add_occlusion(noisy_points, fraction=0.2)  # simulate nozzle blocking view
```

| Surface | What it simulates | Key parameter |
|---------|------------------|---------------|
| `sphere_cap` | Convex anatomy (scalp, shoulder) | `radius`, `cap_angle_deg` |
| `saddle_surface` | Concave-convex transition | `curvature` |
| `wound_crater` | Deep wound with raised edges | `depth` |
| `cylinder_patch` | Your scaffold geometry (R=48.6 mm) | `radius`, `arc_angle_deg` |
| `flat_plane` | Baseline (trivial case) | `size` |

#### 2D wound images (for CNN training)

```python
from livemesh.data.synthetic_2d import generate_star_convex_wound, generate_dataset

# Single sample
sample = generate_star_convex_wound(image_size=256, num_radii=64)
# sample["image"], sample["mask"], sample["centroid"], sample["radii"]

# Full dataset
generate_dataset("data/synthetic", num_samples=2000, seed=42)
```

#### FUSeg + synthetic dataloaders

```python
from livemesh.data.dataset import create_dataloaders

train, val, test = create_dataloaders(
    fuseg_dir="data/fuseg",
    synthetic_dir="data/synthetic",
    batch_size=8,
    image_size=256,
    num_radii=64,
)
# Each batch: {image: (B,3,256,256), centroid: (B,2), radii: (B,64), points: (B,64,2)}
```

#### Polar conversion utilities

```python
from livemesh.data.polar_conversion import mask_to_polar, polar_to_cartesian, polar_to_mask

polar = mask_to_polar(binary_mask, num_radii=64)
# polar["centroid"], polar["radii"], polar["points"], polar["valid"]

points = polar_to_cartesian(centroid, radii)          # (64, 2) XY points
reconstructed_mask = polar_to_mask(centroid, radii)    # (256, 256) uint8
```

---

## Logging

LiveMesh logs everything. Set up once, see the full pipeline trace:

```python
from livemesh.utils.logging_config import setup_logging

setup_logging("INFO")                              # console only
setup_logging("INFO", log_file="logs/run.log")     # console + file
setup_logging("DEBUG")                             # detailed diagnostics
```

Example INFO output:

```
14:32:01 | livemesh.data.synthetic              | INFO    | Generated sphere_cap: R=50.0 mm, angle=45.0 deg, 4096 vertices
14:32:01 | livemesh.data.synthetic              | INFO    | Added noise: sigma=0.50 mm to 4096 points
14:32:01 | livemesh.reconstruction.poisson      | INFO    | Poisson reconstruction: 4096 points, depth=8, scale=1.1
14:32:01 | livemesh.reconstruction.poisson      | INFO    | Reconstructed mesh: 3842 vertices, 7680 faces in 47.3 ms
14:32:01 | livemesh.reconstruction.benchmarks   | INFO    | Benchmark: Hausdorff=1.24 mm, mean=0.089 mm, normals=2.3 deg
14:32:01 | livemesh.toolpath.geodesic           | INFO    | Source vertex: 0, computing geodesic distances...
14:32:02 | livemesh.toolpath.geodesic           | INFO    | Extracted 18 contour levels, spacing=2.0 mm
14:32:02 | livemesh.toolpath.geodesic           | INFO    | Generated 18 paths, 3604 waypoints, 287.4 mm total in 892.1 ms
14:32:02 | livemesh.toolpath.coverage           | INFO    | Coverage: 94.2%, uniformity: 0.87, max gap: 2.1 mm
14:32:02 | livemesh.toolpath.path_to_robot      | INFO    | Generated 3621 G-code lines
```

---

## Project structure

```
livemesh/
├── src/livemesh/
│   ├── perception/              # CNN-Transformer + 3 decoders
│   │   ├── encoder.py              ResNet-50 + 6-layer Transformer
│   │   ├── polar_decoder.py        Centroid + 64 radii (thesis contribution)
│   │   ├── detr_decoder.py         Hungarian matching baseline
│   │   ├── autoregressive_decoder.py  Sequential baseline
│   │   ├── volumetric_encoder.py   8-view 3D encoder (experimental)
│   │   └── volumetric_decoder.py   Depth + layer decoder (experimental)
│   │
│   ├── segmentation/            # U-Net
│   │   ├── unet.py                 Depth 4, 32 filters, 2-class
│   │   └── wound_pipeline.py       Image → mask → boundary → mm
│   │
│   ├── reconstruction/          # Point cloud → mesh
│   │   ├── poisson.py              Poisson surface reconstruction
│   │   └── benchmarks.py           Hausdorff, normal deviation, timing
│   │
│   ├── toolpath/                # Path planning (8 modules)
│   │   ├── geodesic.py             Heat method, curvature-adaptive
│   │   ├── planar_slicer.py        Flat-layer baseline
│   │   ├── honeycomb.py            Hex grid + fill patterns
│   │   ├── conformal_map.py        UV→XYZ (cylinder + general mesh)
│   │   ├── coverage.py             OT-based uniformity metrics
│   │   ├── trajectory_planner.py   Full honeycomb pipeline
│   │   ├── tsp_solver.py           MILP path optimization
│   │   └── path_to_robot.py        G-code / ROS 2 export
│   │
│   ├── robot/                   # Hardware interface
│   │   ├── robot_model.py          8-DOF FK + Jacobian
│   │   ├── inverse_kinematics.py   3-phase IK (L-BFGS-B + APF + STW)
│   │   └── stl_analysis.py         Scaffold void detection
│   │
│   ├── training/                # ML training
│   │   ├── train.py                Trainer + checkpointing
│   │   ├── evaluate.py             Test set metrics
│   │   └── ablation.py             3-decoder comparison
│   │
│   ├── data/                    # Data
│   │   ├── synthetic.py            3D surfaces (sphere, saddle, crater...)
│   │   ├── synthetic_2d.py         2D wound images
│   │   ├── dataset.py              FUSeg + synthetic PyTorch dataset
│   │   ├── multiview_dataset.py    8-view volumetric dataset
│   │   └── polar_conversion.py     Mask ↔ polar ↔ Cartesian
│   │
│   ├── pipeline/                # Orchestrator
│   │   └── orchestrator.py         PERCEIVE → RECONSTRUCT → PLAN → EXECUTE
│   │
│   ├── utils/
│   │   ├── logging_config.py       setup_logging() for the whole package
│   │   ├── metrics.py              Chamfer, Hausdorff, IoU, closure, ordering
│   │   └── visualization.py        Thesis-style matplotlib figures
│   │
│   └── visualization/
│       └── visualization_3d.py     Plotly/Polyscope 3D rendering
│
├── tests/                       # pytest suite
├── notebooks/                   # Ablation studies, demos
├── configs/                     # YAML configuration
├── assets/meshes/               # STL scaffold geometry
├── docs/                        # Action plan, knowledge base
└── paper/                       # Manuscript LaTeX
```

---

## Configuration

All parameters in one place: `configs/default.yaml`

```yaml
reconstruction:
  method: "poisson"
  depth_camera: "realsense"
  target_latency_ms: 100

toolpath:
  method: "geodesic"
  spacing_mm: 1.5
  adaptive_curvature: true

segmentation:
  input_size: [512, 512]
  num_classes: 2
  encoder_depth: 4
  base_filters: 32

robot:
  type: "gcode"
  safe_z_mm: 5.0
  feed_rate_mm_min: 1500
```

Or configure via Python:

```python
from livemesh.pipeline import PipelineConfig

config = PipelineConfig(
    toolpath_method="geodesic",
    spacing_mm=2.0,
    adaptive_curvature=True,
    robot_type="mycobot",
)
```

---

## Tests

```bash
pytest                          # run all tests
pytest tests/test_synthetic.py  # just synthetic surfaces
pytest tests/test_geodesic.py   # just toolpath generation
pytest -v                       # verbose output
```

---

## Requirements

| Package | Version | Why |
|---------|---------|-----|
| PyTorch | >= 2.0 | CNN-Transformer, U-Net |
| torchvision | >= 0.15 | ResNet-50 pretrained weights |
| Open3D | >= 0.17 | Poisson surface reconstruction |
| trimesh | >= 4.0 | Mesh operations, sampling, slicing |
| potpourri3d | >= 0.0.8 | Heat method geodesics (geometry-central) |
| scipy | >= 1.10 | IK optimization, spatial KDTree |
| scikit-image | >= 0.21 | Morphological operations, contour extraction |
| opencv-python | >= 4.8 | Image processing, resizing |
| PuLP | >= 2.7 | TSP MILP solver |
| numpy-stl | >= 3.0 | STL file loading |
| polyscope | >= 2.1 | 3D visualization |
| matplotlib | >= 3.7 | Publication figures |

Full list in `pyproject.toml`.

---

## Closed-Loop Controller (from WoundBioprinter)

LiveMesh inherits the closed-loop depth feedback system from WoundBioprinter:

```python
from livemesh.controller import PrintingLoopController, DepthSensorModel

sensor = DepthSensorModel()  # simulated RealSense D405
controller = PrintingLoopController(sensor=sensor, num_layers=4)

# Run scan-deposit-verify-correct loop
result = controller.run(wound_depth_profile, planned_trajectory)
print(f"Fill accuracy: {result.fill_fraction:.0%}")
```

The `controller` module provides:
- `DepthSensorModel` / `RealSenseD405` — simulated and hardware depth sensing
- `fuse_depth` — confidence-weighted blending of predicted + measured depth
- `PrintingLoopController` — full closed-loop printing orchestration
- `bridge_decoder_to_planner` — connects perception output to trajectory planning

---

## Related work

- [WoundPath-AI](https://github.com/dianisay/WoundPath-AI-UNet-for-Multi-Skin-Tone-Wound-Segmentation-and-GCode-Generation) — Original MATLAB wound segmentation + G-code
- [Conformal-Trajectory](https://github.com/dianisay/Conformal-Trajectory) — Original MATLAB conformal mapping + 8-DOF IK

## License

MIT
